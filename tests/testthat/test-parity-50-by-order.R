# #50: summarise(.by =) computes dplyr's first-appearance group order without a
# row-number column, and an ungrouped bare row_number() is a plain sequence.
#
# dplyr's contract (dplyr 1.2.1, R/by.R::compute_by_groups(): vctrs'
# vec_group_loc() on the `.by` columns): one row per distinct key combination,
# in order of first appearance (not sorted, unlike group_by()); NA is a group,
# NaN is a group of its own, -0 and 0 are one group; factor keys follow first
# appearance, not level order; the result is ungrouped.
#
# These are result-parity tests. The memory and speed contract is in
# test-parity-50-memory.R.

by50_df <- function(n = 240L, seed = 50L) {
  set.seed(seed)
  pick <- function(v) sample(v, n, replace = TRUE)
  data.frame(
    g = pick(c("m", "c", "x", "a", NA)),
    k = pick(c(7L, -2L, 3L, NA, 11L)),
    d = pick(c(2.5, NaN, NA, -0, 0, -1.25)),
    f = factor(pick(c("lo", "hi", "mid", NA)), levels = c("mid", "unused", "hi", "lo")),
    dt = as.Date("2024-03-01") + pick(c(9L, 0L, NA, 3L)),
    ts = as.POSIXct("2024-03-01 12:00:00", tz = "UTC") + pick(c(3600, 0, NA, 60)),
    b = pick(c(TRUE, FALSE, NA)),
    x = ifelse(runif(n) < 0.1, NA, round(rnorm(n), 3)),
    y = sample.int(50L, n, replace = TRUE),
    w = runif(n),
    stringsAsFactors = FALSE
  )
}

expect_by_parity <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# -----------------------------------------------------------------------------
# Group order and key types
# -----------------------------------------------------------------------------

test_that("summarise(.by) keeps first-appearance order for every key type", {
  skip_if_no_gpu()
  df <- by50_df()
  for (key in c("g", "k", "d", "f", "dt", "ts", "b")) {
    pipeline <- function(d) {
      dplyr::summarise(d, n = dplyr::n(), s = sum(y), m = mean(x, na.rm = TRUE),
                       .by = dplyr::all_of(key))
    }
    expect_by_parity(df, pipeline)
  }
})

test_that("summarise(.by) with several key columns keeps first-appearance order", {
  skip_if_no_gpu()
  df <- by50_df()
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), w = sum(w), .by = c(g, b)))
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(f, k, d)))
  expect_by_parity(df, function(d) dplyr::summarise(d, m = max(w), .by = dplyr::starts_with("d")))
})

test_that("summarise(.by) puts NA, NaN and the first -0/0 group where they first appear", {
  skip_if_no_gpu()
  df <- data.frame(
    d = c(2, NaN, NA, -0, 0, 2, NaN, NA, 5),
    s = c("q", NA, "q", "a", NA, "z", "a", "q", NA),
    x = 1:9
  )
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = d))
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = s))
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(s, d)))
})

test_that("summarise(.by) on a factor follows first appearance, not level order", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("b", "a", "b", NA, "c", "a"), levels = c("c", "b", "a", "z")),
    x = c(1, 2, 3, 4, 5, 6)
  )
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = f))
})

test_that("summarise(.by) matches dplyr on many shuffled groups", {
  skip_if_no_gpu()
  set.seed(5050)
  n <- 30000L
  df <- data.frame(
    k = sample.int(4000L, n, replace = TRUE),
    s = sprintf("s%03d", sample.int(700L, n, replace = TRUE)),
    x = rnorm(n),
    y = sample.int(100L, n, replace = TRUE)
  )
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = k))
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(y), .by = c(s, k)))
})

# -----------------------------------------------------------------------------
# Reducers, across(), n(), key redefinition, CPU fallback
# -----------------------------------------------------------------------------

test_that("summarise(.by) with the GPU reducers keeps first-appearance order", {
  skip_if_no_gpu()
  df <- by50_df()
  df$x2 <- ifelse(is.na(df$x), 0, df$x)
  expect_by_parity(df, function(d) {
    dplyr::summarise(
      d,
      n = dplyr::n(),
      mn = min(x2), mx = max(x2), sd = sd(x2), vr = var(x2),
      md = median(w),
      nd = dplyr::n_distinct(y),
      fi = dplyr::first(w), la = dplyr::last(w), nt = dplyr::nth(w, 2),
      an = any(b, na.rm = TRUE), al = all(b, na.rm = TRUE),
      .by = g
    )
  })
})

test_that("summarise(.by) with integer min()/max()/sum() keeps first-appearance order", {
  skip_if_no_gpu()
  df <- by50_df()
  # (an INT32 min()/max(), and a lazy INT32 sum(), run at the verb: F07/F80)
  expect_by_parity(df, function(d) {
    dplyr::summarise(d, lo = min(y), hi = max(y), s = sum(y), .by = k)
  })
  expect_by_parity(df, function(d) dplyr::summarise(d, s = sum(y), .by = c(f, b)))
})

test_that("summarise(.by) with across() keeps first-appearance order", {
  skip_if_no_gpu()
  df <- by50_df()
  expect_by_parity(df, function(d) {
    dplyr::summarise(d, dplyr::across(c(w, y), list(mean = mean, sd = sd)), .by = g)
  })
  expect_by_parity(df, function(d) {
    dplyr::summarise(d, dplyr::across(c(x, w), ~ sum(.x, na.rm = TRUE)), n = dplyr::n(), .by = c(dt, b))
  })
})

test_that("summarise(.by) with n() only, expressions and a redefined key", {
  skip_if_no_gpu()
  df <- by50_df()
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = f))
  expect_by_parity(df, function(d) {
    dplyr::summarise(d, pos = sum(w > 0.5), m = mean(w * 2), .by = g)
  })
  # an aggregate named like the key replaces it
  expect_by_parity(df, function(d) dplyr::summarise(d, g = dplyr::n(), m = mean(w), .by = g))
  # zero dots: the distinct keys, first appearance
  expect_by_parity(df, function(d) dplyr::summarise(d, .by = c(g, k)))
})

test_that("summarise(.by) with reducers that run in dplyr keeps dplyr's order", {
  skip_if_no_gpu()
  df <- by50_df()
  pipeline <- function(d) {
    dplyr::summarise(d, r = stats::cor(w, y), rng = max(w) - min(w), n = dplyr::n(), .by = g)
  }
  suppressMessages(expect_by_parity(df, pipeline))
})

# -----------------------------------------------------------------------------
# Zero rows, lazy pipelines, result grouping
# -----------------------------------------------------------------------------

test_that("summarise(.by) on zero rows returns a typed zero-row result", {
  skip_if_no_gpu()
  df <- by50_df()[0, ]
  expect_by_parity(df, function(d) {
    dplyr::summarise(d, n = dplyr::n(), m = mean(w), s = sum(y), .by = g)
  })
  expect_by_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(f, dt)))
})

test_that("summarise(.by) after pending steps and before later verbs keeps the order", {
  skip_if_no_gpu()
  df <- by50_df()
  expect_by_parity(df, function(d) {
    d |>
      dplyr::filter(w > 0.2) |>
      dplyr::mutate(z = w * 10) |>
      dplyr::summarise(n = dplyr::n(), z = sum(z), .by = c(g, b)) |>
      dplyr::mutate(per = z / n) |>
      dplyr::filter(n > 1L)
  })
  # arrange() is stable: ties keep the first-appearance order
  expect_by_parity(df, function(d) {
    d |>
      dplyr::summarise(n = dplyr::n(), .by = k) |>
      dplyr::mutate(big = n > 40L) |>
      dplyr::arrange(big)
  })
  # head() of the summary
  expect_by_parity(df, function(d) {
    d |> dplyr::summarise(m = mean(w), .by = d) |> head(3)
  })
})

test_that("summarise(.by) returns an ungrouped table in eager and lazy mode", {
  skip_if_no_gpu()
  df <- by50_df()
  for (lazy in c(FALSE, TRUE)) {
    res <- tbl_gpu(df, lazy = lazy) |> dplyr::summarise(n = dplyr::n(), .by = c(g, k))
    expect_identical(dplyr::group_vars(res), character())
    expect_equal(
      tibble::as_tibble(collect(res)),
      tibble::as_tibble(dplyr::summarise(df, n = dplyr::n(), .by = c(g, k)))
    )
  }
})

test_that("summarise(.by) errors like dplyr", {
  skip_if_no_gpu()
  df <- by50_df()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_error(
      dplyr::summarise(dplyr::group_by(gt, g), n = dplyr::n(), .by = k),
      "Can't supply `.by` when `.data` is a grouped data frame",
      class = "rlang_error"
    )
    expect_error(
      dplyr::summarise(gt, n = dplyr::n(), .by = nope),
      class = "vctrs_error_subscript_oob"
    )
    expect_error(
      dplyr::summarise(gt, n = dplyr::n(), .by = g, .groups = "drop"),
      "Can't supply both `.by` and `.groups`"
    )
  }
})

# -----------------------------------------------------------------------------
# Ungrouped row_number()
# -----------------------------------------------------------------------------

test_that("ungrouped row_number() matches dplyr, alone and next to other window calls", {
  skip_if_no_gpu()
  df <- by50_df()
  expect_by_parity(df, function(d) dplyr::mutate(d, r = dplyr::row_number()))
  expect_by_parity(df, function(d) dplyr::mutate(d, r = dplyr::row_number(), r2 = dplyr::row_number() * 2L))
  expect_by_parity(df, function(d) {
    dplyr::mutate(d, r = dplyr::row_number(), cs = cumsum(y), l = dplyr::lag(w), m = mean(w))
  })
  expect_by_parity(df, function(d) dplyr::transmute(d, g, r = dplyr::row_number()))
  # ranked and grouped forms are unchanged
  expect_by_parity(df, function(d) dplyr::mutate(d, r = dplyr::row_number(w), q = dplyr::row_number()))
  expect_by_parity(df, function(d) dplyr::mutate(d, r = dplyr::row_number(), .by = g))
  expect_by_parity(df, function(d) {
    d |> dplyr::group_by(b) |> dplyr::mutate(r = dplyr::row_number()) |> dplyr::ungroup()
  })
})

test_that("ungrouped row_number() numbers the rows that reach it", {
  skip_if_no_gpu()
  df <- by50_df()
  expect_by_parity(df, function(d) {
    d |> dplyr::filter(y > 20L) |> dplyr::mutate(r = dplyr::row_number()) |> dplyr::filter(r %% 3L == 0L)
  })
  expect_by_parity(df, function(d) {
    d |> dplyr::arrange(w) |> dplyr::mutate(r = dplyr::row_number()) |> dplyr::select(r, w)
  })
  expect_by_parity(df[0, ], function(d) dplyr::mutate(d, r = dplyr::row_number()))
  expect_by_parity(df[1, ], function(d) dplyr::mutate(d, r = dplyr::row_number()))
})
