# Mirrored dplyr-oracle tests for the rank family inside mutate() (Phase 5,
# task W4): min_rank()/dense_rank()/percent_rank()/cume_dist()/row_number(x).
#
# All five are lowered by src/ops_window.cpp's rank-kind branch (groupby
# SCAN + cudf::make_rank_aggregation()), sharing the exact "sort by [group
# cols, ranked col] -> scan -> scatter back" architecture W2 built for the
# scan/shift families. R/window.R's extract_one_window_call() enforces the
# section 2.2 constraint that a rank spec's value column IS the frame's
# sole order column; src/ops_window.cpp checks it again defensively.
#
# Semantics verified empirically against dplyr 1.2.1 (recorded here as E3,
# per the W4 task's empirical-check requirement -- see also the file-level
# comment in src/ops_window.cpp):
#   - row_number(x): ties = "sequential" (FIRST method, broken by original
#     row position).
#   - min_rank(x)/dense_rank(x): MIN/DENSE methods; a NA input value gets a
#     NA rank (dplyr's `incomplete = "na"` default).
#   - percent_rank(x) == (min_rank(x) - 1) / (n_non_NA - 1); a group with
#     exactly ONE non-NA value gives NaN (0/0), NOT NA or 0 -- cudf's own
#     ONE_NORMALIZED rank aggregation returns 0 for this case, so
#     src/ops_window.cpp patches it (a real, documented divergence between
#     raw cudf and dplyr, fixed via a small side gb.aggregate() + repeat()
#     call restricted to percent_rank() specs).
#   - cume_dist(x) == rank(x, ties = "max") / n_non_NA; no NaN edge case
#     (a singleton non-NA group gives cume_dist = 1, since 1/1 is well
#     defined, unlike percent_rank's 0/0).
#   - An all-NA group produces NA for every one of these (rank itself is
#     NA, so there's nothing to normalize).
#
# Column TYPES match dplyr exactly for every function here (INT32 for
# min_rank/dense_rank/row_number(x), FLOAT64 for percent_rank/cume_dist) --
# confirmed empirically, so these tests use the default `ignore_col_types =
# FALSE`.
#
# Usage inside filter() predicates is explicitly NOT covered here: filter()
# does not yet check ir_has_window() at all (that's Phase 5 task W5's
# "group-aware filter()" work), so `filter(x > mean(x))`-style predicates
# involving ANY window function (rank family included) currently reach
# ir_bind()'s window guard and surface as a confusing wrapped GPU error
# instead of a clean CPU fallback -- a pre-existing gap (present since the
# W1 registry entries landed), not something this task's scope covers.

# =============================================================================
# Shared fixtures
# =============================================================================

# Ties (two `1`s) + NAs, grouped, deliberately NOT sorted by y so the
# perm/gather/scan/scatter round trip is genuinely exercised.
oracle_rank_grouped_df <- function() {
  data.frame(
    id = 1:9,
    g = c("a", "b", "a", "b", "a", "b", "a", "b", "a"),
    y = c(3, 1, 1, NA, 5, 2, NA, 6, 2),
    stringsAsFactors = FALSE
  )
}

oracle_rank_ungrouped_df <- function() {
  data.frame(y = c(3, 1, NA, 1, 5, NA, 2), stringsAsFactors = FALSE)
}

# =============================================================================
# min_rank() / dense_rank() / row_number(x) -- ties + NAs
# =============================================================================

test_that("min_rank()/dense_rank()/row_number(x) with ties + NAs match dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- oracle_rank_ungrouped_df()
  pipeline <- function(d) {
    dplyr::mutate(d, mr = dplyr::min_rank(y), dr = dplyr::dense_rank(y), rn = dplyr::row_number(y))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("min_rank()/dense_rank()/row_number(x) with ties + NAs match dplyr, grouped", {
  skip_if_no_gpu()
  df <- oracle_rank_grouped_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(mr = dplyr::min_rank(y), dr = dplyr::dense_rank(y), rn = dplyr::row_number(y))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# percent_rank() / cume_dist() -- ties + NAs, incl. the singleton-non-NA-
# group NaN edge case
# =============================================================================

test_that("percent_rank()/cume_dist() with ties + NAs match dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- oracle_rank_ungrouped_df()
  pipeline <- function(d) dplyr::mutate(d, pr = dplyr::percent_rank(y), cd = dplyr::cume_dist(y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("percent_rank()/cume_dist() with ties + NAs match dplyr, grouped", {
  skip_if_no_gpu()
  df <- oracle_rank_grouped_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(pr = dplyr::percent_rank(y), cd = dplyr::cume_dist(y))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("percent_rank() is NaN (not NA/0) for a group with exactly one non-NA value", {
  skip_if_no_gpu()
  df <- data.frame(g = c("x", "y", "y"), y = c(5, 1, NA))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(pr = dplyr::percent_rank(y))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.nan(result$gpu$pr[result$gpu$g == "x"]))
  expect_true(is.nan(result$gpu$pr[result$gpu$g == "y" & !is.na(result$gpu$y)]))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("cume_dist() is 1 (not NaN) for a singleton non-NA group", {
  skip_if_no_gpu()
  df <- data.frame(g = c("x", "y", "y"), y = c(5, 1, NA))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(cd = dplyr::cume_dist(y))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$cd[result$gpu$g == "x"], 1)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("an all-NA group gives NA for every rank-family function", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b"), y = c(NA_real_, NA_real_, NA_real_, 1))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      mr = dplyr::min_rank(y), dr = dplyr::dense_rank(y), rn = dplyr::row_number(y),
      pr = dplyr::percent_rank(y), cd = dplyr::cume_dist(y)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Shuffled rows -- row-order preservation
# =============================================================================

test_that("rank family preserves original row order on shuffled grouped data", {
  skip_if_no_gpu()
  df <- data.frame(
    id = 1:9,
    g = c("b", "a", "c", "a", "b", "c", "a", "b", "c"),
    y = c(10, 1, 100, 2, 20, 200, 3, 30, 300)
  )
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      mr = dplyr::min_rank(y), dr = dplyr::dense_rank(y), rn = dplyr::row_number(y),
      pr = dplyr::percent_rank(y), cd = dplyr::cume_dist(y)
    )
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$id, df$id)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Combined with arithmetic / other window functions in one mutate() call
# =============================================================================

test_that("rank family combined with cumsum()/lag() in one mutate() call matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b", "b"), x = c(3, 1, 2, 6, 4, 5))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      mr = dplyr::min_rank(x),
      cs = cumsum(x),
      lg = dplyr::lag(x),
      delta = mr - dplyr::min_rank(dplyr::lag(x, default = 0))
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Mixed grouped table sizes: single-row groups, 0-row tables
# =============================================================================

test_that("rank family on a grouped table with single-row groups matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b", "b", "c", "c", "c"), x = c(1, 2, 3, 4, 5, 6))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      mr = dplyr::min_rank(x), pr = dplyr::percent_rank(x), cd = dplyr::cume_dist(x)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rank family on a 0-row table matches dplyr, ungrouped and grouped", {
  skip_if_no_gpu()
  df_ungrouped <- data.frame(x = numeric(0))
  pipeline <- function(d) {
    dplyr::mutate(d, mr = dplyr::min_rank(x), pr = dplyr::percent_rank(x), cd = dplyr::cume_dist(x))
  }
  expect_same_as_dplyr(df_ungrouped, pipeline)
  expect_same_as_dplyr_lazy(df_ungrouped, pipeline)

  df_grouped <- data.frame(g = character(0), x = numeric(0))
  grouped_pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(mr = dplyr::min_rank(x), pr = dplyr::percent_rank(x), cd = dplyr::cume_dist(x))
  }
  expect_same_as_dplyr(df_grouped, grouped_pipeline)
  expect_same_as_dplyr_lazy(df_grouped, grouped_pipeline)
})
