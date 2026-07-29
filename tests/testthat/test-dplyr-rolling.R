# Mirrored oracle tests for rolling-window functions (Theo-requested
# benchmark task, scratchpad/todo.md "Rolling-window benchmark target").
#
# Oracle choice: data.table::froll*() directly, NOT expect_same_as_dplyr().
# dplyr itself has no rolling-window functions at all, so running the SAME
# pipeline (using roll_mean()/etc.) through expect_same_as_dplyr()'s
# tibble-side "oracle" would just re-invoke cuplyr's OWN R/roll.R
# implementation -- not an independent check. These tests instead compare
# the GPU result directly against data.table::froll*(), a battle-tested,
# independently-implemented reference (same idiom as
# benchmark/benchmark_rolling.R's CPU reference section).
#
# Scope covered:
#   - roll_mean()/roll_sum()/roll_min()/roll_max()/roll_sd() are GPU-native
#     (src/ops_window.cpp's "rolling" window family, via
#     cudf::rolling_window()/grouped_rolling_window()). roll_median() is
#     NOT GPU-native (cudf has no rolling MEDIAN aggregation in this
#     environment, see R/ir.R's design comment) and falls back to CPU
#     (R/roll.R's roll_median(), via mutate_window_fallback()) -- verified
#     to still produce correct, oracle-matching results.
#   - data.table::frollmean()/frollsum()/frollmin()/frollmax()/frollsd()/
#     frollmedian() are accepted as GPU-native aliases (align="right",
#     fill=NA, adaptive=FALSE only -- see R/ir.R's ir_parse_froll_call()).
#   - na.rm=FALSE (NA-poisoning: any NA in a window -> NA) and na.rm=TRUE
#     (leading incomplete windows still NA; internal NAs skipped) --
#     verified against data.table's OWN na.rm=TRUE semantics (not assumed).
#   - grouped rolling (group_by() -> per-group windows, original row order
#     preserved, cross-group boundary NOT crossed).
#   - eager and lazy exec modes.
#   - edge cases: 0-row input, window width > nrow, window width == 1.
#   - align="left" (or any other non-default align=/fill=/adaptive=) is
#     rejected (mutate() hard error, no CPU fallback for mutate()'s own
#     unrecognized-shape contract).
#   - fallback-notification bookkeeping: roll_mean()/etc. never trigger
#     options(cuplyr.fallback=) notifications (fully GPU-native);
#     roll_median() always does (CPU fallback).

rolling_fns <- c("mean", "sum", "min", "max", "sd", "median")

skip_if_no_data_table <- function() {
  testthat::skip_if_not_installed("data.table")
}

froll_fn_for <- function(fn) {
  get(paste0("froll", fn), envir = asNamespace("data.table"))
}

make_rolling_fixture <- function(n = 10000L, seed = 20260729, na_frac = 0.02) {
  set.seed(seed)
  x <- stats::rnorm(n)
  n_na <- floor(n * na_frac)
  if (n_na > 0) {
    x[sample.int(n, n_na)] <- NA_real_
  }
  data.frame(x = x)
}

# =============================================================================
# Ungrouped correctness: all six functions x windows 3/11/101 x na.rm both
# ways, eager AND lazy, oracle = data.table::froll*()
# =============================================================================

test_that("roll_*() native results match data.table::froll*() oracle, ungrouped, eager", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- make_rolling_fixture()
  gpu_df <- tbl_gpu(df, lazy = FALSE)

  for (fn in rolling_fns) {
    cuplyr_fn <- paste0("roll_", fn)
    dt_fn <- froll_fn_for(fn)
    for (w in c(3L, 11L, 101L)) {
      for (na_rm in c(FALSE, TRUE)) {
        expected <- dt_fn(df$x, w, align = "right", na.rm = na_rm)
        call_expr <- rlang::call2(cuplyr_fn, quote(x), w, na.rm = na_rm)

        got <- gpu_df |>
          dplyr::mutate(y = !!call_expr) |>
          collect() |>
          dplyr::pull(y)

        expect_equal(
          got, expected,
          tolerance = 1e-6,
          info = sprintf("%s(x, %d, na.rm=%s), eager", cuplyr_fn, w, na_rm)
        )
      }
    }
  }
})

test_that("roll_*() native results match data.table::froll*() oracle, ungrouped, lazy", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- make_rolling_fixture()
  gpu_df <- tbl_gpu(df, lazy = TRUE)

  for (fn in rolling_fns) {
    cuplyr_fn <- paste0("roll_", fn)
    dt_fn <- froll_fn_for(fn)
    for (w in c(3L, 11L, 101L)) {
      for (na_rm in c(FALSE, TRUE)) {
        expected <- dt_fn(df$x, w, align = "right", na.rm = na_rm)
        call_expr <- rlang::call2(cuplyr_fn, quote(x), w, na.rm = na_rm)

        got <- gpu_df |>
          dplyr::mutate(y = !!call_expr) |>
          collect() |>
          dplyr::pull(y)

        expect_equal(
          got, expected,
          tolerance = 1e-6,
          info = sprintf("%s(x, %d, na.rm=%s), lazy", cuplyr_fn, w, na_rm)
        )
      }
    }
  }
})

# =============================================================================
# Grouped rolling: per-group windows, original row order preserved. Oracle:
# split-by-group + data.table::froll*() per group (mirrors dplyr's own
# group_by()+mutate() contract: per-group computation, broadcast back into
# ORIGINAL row order, never re-sorted).
# =============================================================================

grouped_froll_oracle <- function(x, g, n, dt_fn, na_rm) {
  out <- rep(NA_real_, length(x))
  for (idx in split(seq_along(x), g)) {
    out[idx] <- dt_fn(x[idx], n, align = "right", na.rm = na_rm)
  }
  out
}

test_that("roll_*() native results match grouped data.table::froll*() oracle, eager+lazy", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  set.seed(1)
  n <- 3000L
  df <- data.frame(
    g = sample(c("a", "b", "c"), n, replace = TRUE),
    x = stats::rnorm(n)
  )
  df$x[sample.int(n, 60)] <- NA_real_

  w <- 5L
  for (fn in rolling_fns) {
    cuplyr_fn <- paste0("roll_", fn)
    dt_fn <- froll_fn_for(fn)
    for (na_rm in c(FALSE, TRUE)) {
      expected <- grouped_froll_oracle(df$x, df$g, w, dt_fn, na_rm)
      call_expr <- rlang::call2(cuplyr_fn, quote(x), w, na.rm = na_rm)

      got_eager <- tbl_gpu(df, lazy = FALSE) |>
        dplyr::group_by(g) |>
        dplyr::mutate(y = !!call_expr) |>
        dplyr::ungroup() |>
        collect() |>
        dplyr::pull(y)
      expect_equal(
        got_eager, expected,
        tolerance = 1e-6,
        info = sprintf("grouped eager %s(na.rm=%s)", cuplyr_fn, na_rm)
      )

      got_lazy <- tbl_gpu(df, lazy = TRUE) |>
        dplyr::group_by(g) |>
        dplyr::mutate(y = !!call_expr) |>
        dplyr::ungroup() |>
        collect() |>
        dplyr::pull(y)
      expect_equal(
        got_lazy, expected,
        tolerance = 1e-6,
        info = sprintf("grouped lazy %s(na.rm=%s)", cuplyr_fn, na_rm)
      )
    }
  }
})

# =============================================================================
# data.table::froll*() alias surface
# =============================================================================

test_that("data.table::froll*() alias matches roll_*() and the data.table oracle", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- make_rolling_fixture(n = 2000L)
  gpu_df <- tbl_gpu(df, lazy = FALSE)

  for (fn in rolling_fns) {
    dt_name <- paste0("froll", fn)
    dt_fn <- froll_fn_for(fn)
    w <- 7L
    expected <- dt_fn(df$x, w, align = "right", na.rm = FALSE)
    call_expr <- rlang::call2(dt_name, quote(x), w, .ns = "data.table")

    got <- gpu_df |>
      dplyr::mutate(y = !!call_expr) |>
      collect() |>
      dplyr::pull(y)

    expect_equal(got, expected, tolerance = 1e-6, info = dt_name)
  }
})

test_that("data.table::froll*() na.rm=TRUE alias matches the oracle", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- make_rolling_fixture(n = 2000L)
  gpu_df <- tbl_gpu(df, lazy = FALSE)
  w <- 7L

  expected <- data.table::frollmean(df$x, w, align = "right", na.rm = TRUE)
  got <- gpu_df |>
    dplyr::mutate(y = data.table::frollmean(x, w, na.rm = TRUE)) |>
    collect() |>
    dplyr::pull(y)
  expect_equal(got, expected, tolerance = 1e-6)
})

# =============================================================================
# Edge cases: 0-row, window > nrow, window == 1
# =============================================================================

test_that("roll_mean() handles a 0-row input", {
  skip_if_no_gpu()
  df0 <- data.frame(x = numeric(0))
  g0 <- tbl_gpu(df0, lazy = FALSE)
  got <- g0 |> dplyr::mutate(y = roll_mean(x, 3)) |> collect()
  expect_equal(nrow(got), 0L)
  expect_true("y" %in% names(got))
})

test_that("roll_*() window wider than the table is all-NA, matching data.table", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- data.frame(x = c(1, 2, 3, 4, 5))
  g <- tbl_gpu(df, lazy = FALSE)

  for (fn in rolling_fns) {
    cuplyr_fn <- paste0("roll_", fn)
    dt_fn <- froll_fn_for(fn)
    expected <- dt_fn(df$x, 100L, align = "right", na.rm = FALSE)
    call_expr <- rlang::call2(cuplyr_fn, quote(x), 100L)

    got <- g |> dplyr::mutate(y = !!call_expr) |> collect() |> dplyr::pull(y)
    expect_equal(got, expected, info = sprintf("%s, w > nrow", cuplyr_fn))
    expect_true(all(is.na(got)))
  }
})

test_that("roll_*() with window width 1 is the identity (modulo NA passthrough)", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- data.frame(x = c(1, 2, NA, 4, 5))
  g <- tbl_gpu(df, lazy = FALSE)

  for (fn in rolling_fns) {
    cuplyr_fn <- paste0("roll_", fn)
    dt_fn <- froll_fn_for(fn)
    expected <- dt_fn(df$x, 1L, align = "right", na.rm = FALSE)
    call_expr <- rlang::call2(cuplyr_fn, quote(x), 1L)

    got <- g |> dplyr::mutate(y = !!call_expr) |> collect() |> dplyr::pull(y)
    expect_equal(got, expected, info = sprintf("%s, w = 1", cuplyr_fn))
  }
})

# =============================================================================
# roll_median(): not GPU-native (cudf has no rolling MEDIAN aggregation),
# CPU-fallback path -- verify it's still CORRECT, and that the fallback
# notification actually fires (unlike its five GPU-native siblings).
# =============================================================================

test_that("roll_median() falls back to CPU and still matches the data.table oracle", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- make_rolling_fixture(n = 2000L)
  gpu_df <- tbl_gpu(df, lazy = FALSE)

  for (w in c(3L, 11L, 101L)) {
    for (na_rm in c(FALSE, TRUE)) {
      expected <- data.table::frollmedian(df$x, w, align = "right", na.rm = na_rm)
      got <- gpu_df |>
        dplyr::mutate(y = roll_median(x, w, na.rm = na_rm)) |>
        collect() |>
        dplyr::pull(y)
      expect_equal(
        got, expected,
        tolerance = 1e-8,
        info = sprintf("roll_median(w=%d, na.rm=%s)", w, na_rm)
      )
    }
  }
})

test_that("roll_median()/roll_mean() fallback-notification bookkeeping is correct", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- data.frame(x = c(1, 2, 3, 4, 5, 6, 7, 8))
  g <- tbl_gpu(df, lazy = FALSE)

  # GPU-native: no fallback notification at all.
  expect_no_warning(g |> dplyr::mutate(y = roll_mean(x, 3)) |> collect())
  expect_no_warning(g |> dplyr::mutate(y = roll_sum(x, 3)) |> collect())
  expect_no_warning(g |> dplyr::mutate(y = roll_min(x, 3)) |> collect())
  expect_no_warning(g |> dplyr::mutate(y = roll_max(x, 3)) |> collect())
  expect_no_warning(g |> dplyr::mutate(y = roll_sd(x, 3)) |> collect())

  # roll_median(): CPU fallback, notification fires.
  expect_warning(g |> dplyr::mutate(y = roll_median(x, 3)) |> collect())
})

# =============================================================================
# Rejected shapes: align != "right" (or any other non-default fill=/
# adaptive=/partial=/give.names=) is a hard mutate() error, not a silent
# fallback -- matching mutate()'s existing "no CPU fallback for an
# unrecognized shape" contract (this task's explicit "right-aligned only"
# scope cut).
# =============================================================================

test_that("data.table::froll*(align != 'right') is rejected (mutate() hard error)", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- data.frame(x = c(1, 2, 3, 4, 5))
  g <- tbl_gpu(df, lazy = FALSE)

  expect_error(
    g |> dplyr::mutate(y = data.table::frollmean(x, 3, align = "left")) |> collect()
  )
  expect_error(
    g |> dplyr::mutate(y = data.table::frollmean(x, 3, align = "center")) |> collect()
  )
  expect_error(
    g |> dplyr::mutate(y = data.table::frollmean(x, 3, fill = 0)) |> collect()
  )
  expect_error(
    g |> dplyr::mutate(y = data.table::frollmean(x, 3, adaptive = TRUE)) |> collect()
  )
  expect_error(
    g |> dplyr::mutate(y = data.table::frollmean(x, 3, give.names = TRUE)) |> collect()
  )
})

test_that("roll_mean() with a non-constant/invalid window width is rejected", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3, 4, 5), w = c(1, 2, 3, 4, 5))
  g <- tbl_gpu(df, lazy = FALSE)

  expect_error(g |> dplyr::mutate(y = roll_mean(x, w)) |> collect())  # column, not constant
  expect_error(g |> dplyr::mutate(y = roll_mean(x, 0)) |> collect())  # non-positive
  expect_error(g |> dplyr::mutate(y = roll_mean(x, -1)) |> collect())  # negative
  expect_error(g |> dplyr::mutate(y = roll_mean(x, 1.5)) |> collect())  # non-integer
})

# =============================================================================
# Bonus: the shared window-decomposition machinery also wires roll_*() into
# group-aware filter() for free (Phase 5 W5's design, unchanged by this
# task) -- one smoke test, not the primary scope of this file.
# =============================================================================

test_that("roll_mean() works inside filter() (shared window machinery)", {
  skip_if_no_gpu()
  skip_if_no_data_table()

  df <- make_rolling_fixture(n = 500L)
  expected_roll <- data.table::frollmean(df$x, 5L, align = "right", na.rm = FALSE)
  expected_rows <- which(!is.na(expected_roll) & expected_roll > 0)

  got <- tbl_gpu(df, lazy = FALSE) |>
    dplyr::mutate(rn = dplyr::row_number()) |>
    dplyr::filter(roll_mean(x, 5) > 0) |>
    collect() |>
    dplyr::pull(rn)

  expect_equal(got, expected_rows)
})
