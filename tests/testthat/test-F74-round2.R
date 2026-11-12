# F74 round 2: the CPU route is decided before the observed summarise runs
# (one set of warnings, no second run of a lazy plan), a broken fixed-keys
# invariant is never silently lost, and a factor-key table's timestamp
# min()/max() is NA for an empty level (B1 of round 1).

f74r_df <- data.frame(
  g = c("a", "a", "b", "c", "d"),
  s = c(NA, NA, "x", "y", "z"),
  d = as.Date("2020-01-01") + 0:4,
  x = c(NA, NA, 3, 4, 5),
  stringsAsFactors = FALSE
)
f74r_y <- data.frame(g = c("a", "c"), stringsAsFactors = FALSE)

f74r_warnings <- function(f) {
  msgs <- character()
  value <- withCallingHandlers(f(), warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = msgs)
}

test_that("a CPU-routed aggregate warns once, like dplyr (no GPU warning first)", {
  skip_if_no_gpu()
  pipeline <- function(d, y) {
    semi_join(group_by(d, g, .drop = FALSE), y, by = "g") |>
      summarise(lo = min(s, na.rm = TRUE), hx = min(x, na.rm = TRUE), n = n())
  }
  expected <- f74r_warnings(function() pipeline(tibble::as_tibble(f74r_df), f74r_y))
  expect_length(expected$warnings, 1L)
  for (lazy in c(FALSE, TRUE)) {
    got <- f74r_warnings(function() {
      collect(pipeline(tbl_gpu(f74r_df, lazy = lazy), tbl_gpu(f74r_y)))
    })
    # (group "a" is all NA: the observed GPU summarise would warn for hx
    # before the CPU route took over)
    expect_length(got$warnings, length(expected$warnings))
    expect_equal(got$value, expected$value, ignore_attr = TRUE)
  }
})

test_that("a CPU-routed aggregate with no empty group stays native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(f74r_df, lazy = lazy) |>
      group_by(g, .drop = FALSE) |>
      filter(g != "zzz") |>
      summarise(lo = min(s), n = n()) |>
      collect()
    expect_equal(got$n, c(2L, 1L, 1L, 1L))
  }
})

test_that("timestamp aggregates other than min()/max() are native on empty groups", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- suppressWarnings(
      semi_join(group_by(tbl_gpu(f74r_df, lazy = lazy), g, .drop = FALSE),
                tbl_gpu(f74r_y), by = "g") |>
        summarise(m = mean(d), md = median(d), lo = min(d)) |>
        collect()
    )
    expect_identical(is.na(got$m), c(FALSE, TRUE, FALSE, TRUE))
    expect_identical(is.na(got$md), c(FALSE, TRUE, FALSE, TRUE))
    expect_identical(is.na(got$lo), c(FALSE, TRUE, FALSE, TRUE))
  }
})

test_that("keys that do not hold the observed groups are never silently lost", {
  skip_if_no_gpu()
  base <- tbl_gpu(f74r_df, lazy = FALSE) |> group_by(g, .drop = FALSE)
  with_keys <- function(keys) cuplyr:::set_tbl_gpu_field(base, "fixed_keys", keys)
  # more observed groups than keys
  broken <- with_keys(tibble::tibble(g = c("b", "c")))
  expect_error(withr::with_options(list(cuplyr.fallback = "error"),
                                   summarise(broken, n = n())),
               "fell back to CPU")
  # an observed key missing from a longer key set
  broken <- with_keys(tibble::tibble(g = c("b", "c", "e", "f", "h")))
  expect_error(withr::with_options(list(cuplyr.fallback = "error"),
                                   summarise(broken, n = n(), m = max(d))),
               "fell back to CPU")
  # the invariant holding keeps it native
  ok <- with_keys(tibble::tibble(g = c("a", "b", "c", "d", "e")))
  got <- withr::with_options(list(cuplyr.fallback = "error"),
                             collect(summarise(ok, n = n())))
  expect_identical(got$g, c("a", "b", "c", "d", "e"))
  expect_identical(got$n, c(2L, 1L, 1L, 1L, 0L))
})

test_that("a factor-key empty level has NA for Date/POSIXct min()/max(), eager and lazy", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("a", "a", "c"), levels = c("a", "b", "c")),
    d = as.Date(c("2020-01-01", "2020-01-05", "2021-03-03")),
    t = as.POSIXct(c("2020-01-01 10:00:00", "2020-01-02 10:00:00", "2020-01-03 10:00:00"),
                   tz = "UTC")
  )
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    cap <- f74r_warnings(function() {
      tbl_gpu(df, lazy = lazy) |> group_by(f, .drop = FALSE) |>
        summarise(lo = min(d), hi = max(t)) |> collect()
    })
    expect_identical(is.na(unclass(cap$value$lo)), c(FALSE, TRUE, FALSE))
    expect_identical(is.na(unclass(cap$value$hi)), c(FALSE, TRUE, FALSE))
    expect_equal(cap$value$lo[c(1, 3)], as.Date(c("2020-01-01", "2021-03-03")))
    expect_true(any(grepl("no non-missing arguments", cap$warnings)))
  }
})

test_that("a Date/POSIXct non-finite value stays NA through filter and arrange", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = 1:2)
  df$d <- structure(c(Inf, 18262), class = "Date")
  expect_warning(got <- tbl_gpu(df) |> arrange(d) |> collect(),
                 class = "cuplyr_warning_infinite_timestamp")
  expect_identical(got$g, c("b", "a"))
  expect_identical(is.na(unclass(got$d)), c(FALSE, TRUE))
  expect_warning(kept <- tbl_gpu(df) |> filter(d < as.Date("2000-01-01") | is.na(d)) |> collect(),
                 class = "cuplyr_warning_infinite_timestamp")
  expect_identical(kept$g, "a")
})

# F74 round 3: the pre-check's IR branch, the single plan run of the lazy CPU
# route, and the factor `.drop = FALSE` path's keys-only expansion.

f74r3_df <- data.frame(
  g = c("a", "a", "b", "c", "d", "a"),
  s = c("x", "y", "z", "w", "v", "u"),
  l = c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE),
  xn = c(NA, NA, 3, 4, 5, NA),
  stringsAsFactors = FALSE
)

test_that("the pre-check types an expression argument of a character min()/max() through the IR", {
  schema <- list(names = c("s", "l", "x"), types = c("STRING", "BOOL8", "FLOAT64"))
  expect_true(cuplyr:::summarise_empty_value_on_cpu(rlang::quos(lo = min(toupper(s))), schema))
  expect_true(cuplyr:::summarise_empty_value_on_cpu(
    rlang::quos(hi = max(if_else(l, s, "zz"), na.rm = TRUE)), schema
  ))
  expect_false(cuplyr:::summarise_empty_value_on_cpu(rlang::quos(lo = min(x * 2)), schema))
})

test_that("a character min() of an expression routes to the CPU before the observed summarise", {
  skip_if_no_gpu()
  # character min()/max() are native only under C collation; elsewhere the
  # whole call is CPU before the empty-group path and this pins nothing
  withr::local_collate("C")
  pipeline <- function(d, y) {
    semi_join(group_by(d, g, .drop = FALSE), y, by = "g") |>
      summarise(lo = min(toupper(s)), xx = min(xn, na.rm = TRUE), n = n())
  }
  expected <- f74r_warnings(function() pipeline(tibble::as_tibble(f74r3_df), f74r_y))
  for (lazy in c(FALSE, TRUE)) {
    got <- f74r_warnings(function() {
      collect(pipeline(tbl_gpu(f74r3_df, lazy = lazy), tbl_gpu(f74r_y)))
    })
    # (group "a" has no non-missing xn: an observed GPU summarise would warn
    # for xx before the CPU route took over)
    expect_length(got$warnings, length(expected$warnings))
    expect_equal(got$value, expected$value, ignore_attr = TRUE)
  }
  # with no empty group the same call stays native
  withr::local_options(cuplyr.fallback = "error")
  y_all <- data.frame(g = c("a", "b", "c", "d"), stringsAsFactors = FALSE)
  got <- suppressWarnings(collect(pipeline(tbl_gpu(f74r3_df, lazy = TRUE), tbl_gpu(y_all))))
  expect_identical(got$lo, c("U", "Z", "W", "V"))
})

test_that("the lazy CPU route runs the pending plan once", {
  skip_if_no_gpu()
  withr::local_collate("C")
  n_opt <- 0L
  real_optimize <- cuplyr:::optimize_ast
  local_mocked_bindings(optimize_ast = function(ast) {
    n_opt <<- n_opt + 1L
    real_optimize(ast)
  })
  lz <- tbl_gpu(f74r3_df, lazy = TRUE) |>
    group_by(g, .drop = FALSE) |>
    semi_join(tbl_gpu(f74r_y), by = "g") |>
    mutate(x2 = xn * 2)
  n_opt <- 0L
  res <- suppressWarnings(summarise(lz, lo = min(s), n = n()))
  expect_identical(n_opt, 1L)
  expect_true(is_lazy(res))
  expected <- suppressWarnings(
    semi_join(group_by(tibble::as_tibble(f74r3_df), g, .drop = FALSE), f74r_y, by = "g") |>
      summarise(lo = min(s), n = n())
  )
  expect_equal(collect(res), expected, ignore_attr = TRUE)
  # the native route as well
  n_opt <- 0L
  res <- summarise(lz, n = n(), s = sum(x2, na.rm = TRUE))
  expect_identical(n_opt, 1L)
})

test_that("factor .drop = FALSE expansion: group order, NA keys and multiple keys", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("b", NA, "b", "d", NA, "b"), levels = c("a", "b", "c", "d")),
    h = factor(c("y", "x", NA, "x", "x", "y"), levels = c("x", "y", "z")),
    k = c(2L, 1L, 1L, NA, 3L, 2L),
    x = c(1, 2, 3, 4, 5, 6)
  )
  withr::local_options(cuplyr.fallback = "error")
  both <- function(d, pipeline) {
    expect_same_as_dplyr(d, pipeline)
    expect_same_as_dplyr_lazy(d, pipeline)
  }
  both(df, function(d) {
    group_by(d, f, .drop = FALSE) |> summarise(n = n(), s = sum(x))
  })
  both(df, function(d) {
    group_by(d, f, h, .drop = FALSE) |> summarise(n = n(), m = mean(x))
  })
  both(df, function(d) {
    group_by(d, k, f, .drop = FALSE) |> summarise(n = n(), s = sum(x))
  })
  both(df, function(d) {
    group_by(d, f, k, .drop = FALSE) |> filter(x > 1) |> summarise(n = n(), s = sum(x))
  })
  # every level observed: the plain summarise
  full <- data.frame(f = factor(c("a", "b"), levels = c("a", "b")), x = 1:2)
  both(full, function(d) {
    group_by(d, f, .drop = FALSE) |> summarise(n = n())
  })
  # zero rows: every level is an empty group
  both(df[0, ], function(d) {
    group_by(d, f, .drop = FALSE) |> summarise(n = n(), s = sum(x))
  })
})
