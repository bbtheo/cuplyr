# summarise() with arithmetic around reducers (#22): cases beyond the acceptance
# file test-parity-22-post-agg.R. They use f124_check() (helper-parity-F124.R):
# values, types, warnings, group vars and no CPU fallback, eager and lazy.

test_that("a column named like the split's temp aggregate is not overwritten", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), .post_agg1 = c(1, 2, 3), x = c(1, 2, 3))
  f124_check(df, function(d) summarise(group_by(d, g), m = mean(.post_agg1) + 1, s = sum(x) / n()))
})

test_that("each min()/max() of a dot warns once, as dplyr evaluates each call", {
  skip_if_no_gpu()
  fdf <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c", "d")),
                        x = c(1, 2, 3))
  f124_check(fdf, function(d) {
    summarise(group_by(d, f, .drop = FALSE), r = max(x) - min(x), t = max(x) * 2)
  })
})

test_that("the same reducer in two dots, and a post dot over a plain one", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 5), y = c(2, 4, 6, 8))
  f124_check(df, function(d) {
    summarise(group_by(d, g), a = mean(x) + 1, b = mean(x) * 2, m = mean(y), r = m / mean(x))
  })
})

test_that("a composite summarise followed by select() keeps what the plan needs", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 5), y = c(2, 4, 6, 8))
  f124_check(df, function(d) {
    d |> group_by(g) |> summarise(a = mean(x) + 1, b = sum(y) / n()) |> select(b, g)
  })
  f124_check(df, function(d) {
    d |> summarise(a = mean(x) + 1, b = sum(y) / n(), .by = g) |> filter(b > 4)
  })
})

test_that("a split call that dplyr runs is not given a post step on the CPU result", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  # a column beside an aggregate: dplyr's error, on every route
  withr::local_options(cuplyr.fallback = "silent")
  expect_error(collect(summarise(group_by(tbl_gpu(df), g), m = mean(x) + x)), "must be size 1")
  expect_error(collect(summarise(group_by(tbl_gpu(df, lazy = TRUE), g), m = mean(x) + x)), "must be size 1")
})

# The notices a split call raises: the conditions `expr` signals, warnings muffled
sp_notices <- function(expr, class) {
  got <- list()
  withCallingHandlers(
    expr,
    condition = function(c) {
      if (inherits(c, class)) got[[length(got) + 1L]] <<- c
    },
    warning = function(w) invokeRestart("muffleWarning")
  )
  got
}

test_that("the plan-break notice of a split call names the user's dot, not a temp aggregate", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1L, 2L, 3L))
  got <- sp_notices(
    tbl_gpu(df, lazy = TRUE) |> filter(x > 0L) |> group_by(g) |> summarise(s = sum(x) + 1L) |> collect(),
    "cuplyr_plan_break"
  )
  expect_length(got, 1L)
  expect_match(got[[1L]]$reason, "(result type depends on the data): s", fixed = TRUE)
  expect_no_match(got[[1L]]$reason, ".post_agg", fixed = TRUE)
})

test_that("a declined post step claims the dot it stopped at, not an earlier native one", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3), y = c(4, 5, 6))
  for (lazy in c(FALSE, TRUE)) {
    got <- sp_notices(
      summarise(tbl_gpu(df, lazy = lazy), a = mean(x) + 1, b = dplyr::lag(mean(y))) |> collect(),
      "cuplyr_cpu_fallback"
    )
    expect_length(got, 1L)
    expect_identical(got[[1L]]$expr, "dplyr::lag(mean(y))")
    expect_match(got[[1L]]$reason, "function of the group's rows around a reducer", fixed = TRUE)
  }
})

test_that("a quosure embedded in a dot keeps its own environment for what is around its reducer", {
  skip_if_no_gpu()
  k <- 2
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  f <- function(d, stat) {
    k <- 100
    summarise(group_by(d, g), m = {{ stat }} * k, s = sum(x) / n())
  }
  f124_check(df, function(d) f(d, mean(x) + k))
  # the same reducer twice, once in each environment, is two leaves
  f2 <- function(d, stat) {
    w <- 5
    summarise(d, m = {{ stat }} + mean(x * w))
  }
  w <- 3
  f124_check(df, function(d) f2(d, mean(x * w)))
})

test_that("many repeated reducers in one dot share one leaf each", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3), y = c(2, 4, 9))
  f124_check(df, function(d) {
    summarise(group_by(d, g), r = (sum(x) + sum(y)) / (sum(x) - sum(y)) + sum(x) * sum(y))
  })
})

# The fallback notice of a call, eager or lazy (NULL when it ran natively)
sp_fallback <- function(f, df, lazy) {
  got <- sp_notices(collect(f(tbl_gpu(df, lazy = lazy))), "cuplyr_cpu_fallback")
  if (length(got) > 0L) got[[1L]]
}

test_that("a power around a reducer runs on the GPU only with a constant exponent R and the GPU agree on", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L, 3L), y = c(NA, 1, 2, -1))
  k <- 0
  # (R: NA^0, 1^NA and NA^FALSE are 1; (-1)^Inf is NaN. The GPU engine's pow gives NA and 1.)
  declined <- list(
    list(f = function(d) summarise(group_by(d, g), p = mean(y)^0), expr = "mean(y)^0"),
    list(f = function(d) summarise(group_by(d, g), p = mean(y)^k), expr = "mean(y)^k"),
    list(f = function(d) summarise(group_by(d, g), p = mean(y)^FALSE), expr = "mean(y)^FALSE"),
    list(f = function(d) summarise(group_by(d, g), p = 1^mean(y)), expr = "1^mean(y)"),
    list(f = function(d) summarise(group_by(d, g), p = 2^n()), expr = "2^n()"),
    list(f = function(d) summarise(group_by(d, g), p = mean(y)^Inf), expr = "mean(y)^Inf"),
    list(f = function(d) summarise(group_by(d, g), p = mean(y)^NA_real_), expr = "mean(y)^NA_real_"),
    list(f = function(d) summarise(group_by(d, g), s = sum(y), p = (s + 1)^0), expr = "(s + 1)^0")
  )
  for (cs in declined) {
    f124_check(df, cs$f, native = FALSE)
    for (lazy in c(FALSE, TRUE)) {
      got <- sp_fallback(cs$f, df, lazy)
      expect_identical(got$expr, cs$expr)
      expect_match(got$reason, "`^` around a reducer runs on the GPU only with a constant exponent", fixed = TRUE)
    }
  }
  f124_check(df, function(d) summarise(group_by(d, g), a = mean(y)^-2, b = n()^3L, c = sd(y)^TRUE))
})

test_that("a power around a reducer that may be -Inf or -0 runs on the GPU only when R and the GPU agree", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  # group 2: max(y, na.rm = TRUE) is -Inf; group 3: -mean(y) is -0
  df <- tibble::tibble(g = c(1L, 1L, 2L, 3L, 3L), y = c(4, 9, NA, -1, 1))
  declined <- list(
    list(f = function(d) summarise(group_by(d, g), p = max(y, na.rm = TRUE)^0.5),
         expr = "max(y, na.rm = TRUE)^0.5"),
    list(f = function(d) summarise(group_by(d, g), p = (-mean(y))^-1), expr = "(-mean(y))^(-1)"),
    list(f = function(d) summarise(group_by(d, g), p = mean(y)^TRUE), expr = "mean(y)^TRUE"),
    list(f = function(d) summarise(group_by(d, g), p = (-sd(y))^3), expr = "(-sd(y))^3"),
    list(f = function(d) summarise(group_by(d, g), m = max(y, na.rm = TRUE), p = m^-3), expr = "m^(-3)")
  )
  for (cs in declined) {
    f124_check(df, cs$f, native = FALSE)
    for (lazy in c(FALSE, TRUE)) {
      got <- sp_fallback(cs$f, df, lazy)
      expect_identical(got$expr, cs$expr)
      expect_match(got$reason, "with an even exponent, or over `n()`", fixed = TRUE)
    }
  }
  # an even exponent for any base; any exponent for a count's or a spread's base, an
  # earlier dot's too
  f124_check(df, function(d) {
    summarise(group_by(d, g), a = max(y, na.rm = TRUE)^-2, b = (-mean(y))^4, c = n()^-0.5,
              d = n_distinct(y)^1.5, e = var(y)^3, v = sd(y), w = v^(1 / 3), .groups = "drop")
  })
})

test_that("`&` and `|` of a string reducer give R's error, with a true reason", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), s = c("a", "b", "c"))
  f <- function(d) summarise(group_by(d, g), r = dplyr::first(s) & TRUE)
  for (lazy in c(FALSE, TRUE)) {
    notes <- list()
    err <- tryCatch(
      withCallingHandlers(collect(f(tbl_gpu(df, lazy = lazy))), cuplyr_cpu_fallback = function(c) {
        notes[[length(notes) + 1L]] <<- c
      }),
      error = function(e) e
    )
    expect_s3_class(err, "error")
    expect_match(conditionMessage(err), "operations are possible only for numeric, logical or complex types")
    expect_length(notes, 1L)
    expect_identical(notes[[1L]]$expr, "dplyr::first(s) & TRUE")
    expect_match(notes[[1L]]$reason, "only on logical or integer values", fixed = TRUE)
  }
})

test_that("leaves whose argument is not a name share by the call, in a bucket of their own", {
  dots <- rlang::quos(m = sum(x > 1) / n() + sum(x > 1), k = mean(x > 2) - sum(x > 1) + mean(x > 2))
  schema <- list(names = c("g", "x"), types = c("INT32", "FLOAT64"))
  split <- summarise_post_split(dots, dots, schema, "g")
  expect_identical(vapply(split$agg_dots, rlang::quo_text, character(1), USE.NAMES = FALSE),
                   c("sum(x > 1)", "n()", "mean(x > 2)"))
})

test_that("the window-sum compute notice of a split call shows the user's dot", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1L, 2L, 3L))
  got <- sp_notices(
    tbl_gpu(df, lazy = TRUE) |> filter(x > 0L) |> group_by(g) |>
      summarise(s = mean(x / sum(x)) + 1) |> collect(),
    "cuplyr_plan_break"
  )
  expect_length(got, 1L)
  expect_identical(got[[1L]]$reason, "a window call inside an aggregation needs its input computed")
  expect_identical(got[[1L]]$expr, "mean(x / sum(x)) + 1")
})

test_that("`&`, `|` and `!` of a double around a reducer read NaN as dplyr does", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  # group 2: mean(y, na.rm = TRUE) is NaN, which R's `|`, `&` and `!` read as NA
  df <- tibble::tibble(g = c(1L, 1L, 2L, 3L), y = c(NA, 1, NA, 0), b = c(TRUE, NA, FALSE, TRUE))
  declined <- list(
    list(f = function(d) summarise(group_by(d, g), r = mean(y, na.rm = TRUE) | FALSE),
         expr = "mean(y, na.rm = TRUE) | FALSE"),
    list(f = function(d) summarise(group_by(d, g), r = any(b) & mean(y, na.rm = TRUE)),
         expr = "any(b) & mean(y, na.rm = TRUE)"),
    list(f = function(d) summarise(group_by(d, g), r = !mean(y, na.rm = TRUE)), expr = "!mean(y, na.rm = TRUE)")
  )
  for (cs in declined) {
    f124_check(df, cs$f, native = FALSE)
    for (lazy in c(FALSE, TRUE)) {
      got <- sp_fallback(cs$f, df, lazy)
      expect_identical(got$expr, cs$expr)
      expect_match(got$reason, "of a double", fixed = TRUE)
    }
  }
  # logical and integer operands keep R's NA rules on the GPU
  f124_check(df, function(d) {
    summarise(group_by(d, g), a = any(b) & all(b), o = sum(b) > 0 | NA, n = !n(), m = mean(y) > 0 & FALSE)
  })
})

test_that("if_else() and case_when() with a double condition around a reducer give dplyr's error", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  withr::local_options(cuplyr.fallback = "silent")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(summarise(group_by(tbl_gpu(df, lazy = lazy), g), m = dplyr::if_else(mean(x), 1, 0))),
                 "`condition` must be a logical vector")
    expect_error(collect(summarise(group_by(tbl_gpu(df, lazy = lazy), g),
                                   m = dplyr::case_when(mean(x) ~ 1, .default = 0))),
                 "must be a logical vector")
  }
  # a bare NA condition is dplyr's NA (the GPU engine would take it for a double)
  f124_check(df, function(d) summarise(group_by(d, g), m = dplyr::if_else(NA, 1, mean(x))), native = FALSE)
})

test_that("a variable named like a temp aggregate is read from the environment", {
  skip_if_no_gpu()
  .post_agg2 <- 100
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3), y = c(5, 6, 7))
  f124_check(df, function(d) summarise(group_by(d, g), m = mean(x) + .post_agg2, s = sum(y) + 1))
})

test_that("a literal NULL argument around a reducer is kept", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  withr::local_options(cuplyr.fallback = "silent")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(summarise(group_by(tbl_gpu(df, lazy = lazy), g), m = round(mean(x), NULL))),
                 "invalid second argument of length 0")
  }
})

test_that("a large constant inside a reducer splits and shares its leaf", {
  # (a key deparsed from the whole call broke R's 10000-byte name limit here, and the
  # split declined)
  v <- stats::runif(2e5)
  q <- rlang::quo(sum(x %in% !!v) + mean(x %in% !!v) + sum(x %in% !!v))
  dots <- rlang::quos(m = !!q)
  schema <- list(names = c("g", "x"), types = c("INT32", "FLOAT64"))
  split <- summarise_post_split(dots, dots, schema, "g")
  expect_length(split$agg_dots, 2L)
})

test_that("a post step that fails on the real result reruns on the CPU with a true reason", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  # (no shape fails only there today: the parse on the real result is made to fail)
  local_mocked_bindings(make_mutate_expr = function(...) stop("no GPU form"))
  for (lazy in c(FALSE, TRUE)) {
    got <- sp_fallback(function(d) summarise(group_by(d, g), m = mean(x) + 1), df, lazy)
    expect_false(is.null(got))
    expect_no_match(got$reason, "integer sum()", fixed = TRUE)
    expect_match(got$reason, "arithmetic around the reducers", fixed = TRUE)
  }
  expect_equal(collect(summarise(group_by(tbl_gpu(df), g), m = mean(x) + 1)),
               summarise(group_by(df, g), m = mean(x) + 1))
})

test_that("the plan-break notice of a split call shows the user's dot", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1L, 2L, 3L))
  for (cs in list(list(f = function(t) summarise(t, m = mean(x), s = sum(x * 2L) + 1L), expr = "sum(x * 2L) + 1L"),
                  list(f = function(t) summarise(t, m = mean(x), s = sum(x) + 1L), expr = "sum(x) + 1L"))) {
    got <- sp_notices(
      tbl_gpu(df, lazy = TRUE) |> filter(x > 0L) |> group_by(g) |> cs$f() |> collect(),
      "cuplyr_plan_break"
    )
    expect_length(got, 1L)
    expect_identical(got[[1L]]$expr, cs$expr)
  }
})

test_that("a timestamp sum around a reducer is not blamed on mutate()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), t = as.POSIXct("2024-01-02 10:00:00", tz = "UTC") + c(0, 60, 120))
  got <- sp_fallback(function(d) summarise(group_by(d, g), m = max(t) + 3600), df, FALSE)
  expect_identical(got$expr, "max(t) + 3600")
  expect_no_match(got$reason, "mutate()", fixed = TRUE)
})
