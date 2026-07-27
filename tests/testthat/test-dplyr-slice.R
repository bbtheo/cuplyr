# Mirrored dplyr-oracle tests for the slice() family (Phase 3, task 2)
#
# Every test compares a dplyr pipeline run against a plain tibble (the
# oracle) with the same pipeline run against a `tbl_gpu`, in both eager and
# lazy exec modes, via `expect_same_as_dplyr()`/`expect_same_as_dplyr_lazy()`
# (see helper-oracle.R).
#
# GPU-native (ungrouped, no `by=`/`.by=`): slice() with a plain
# data-independent index expression, slice_head()/slice_tail(),
# slice_min()/slice_max() with a plain order_by (R/slice.R, R/ast.R::
# ast_slice(), R/lower.R::lower_slice(), src/ops_slice.cpp). Grouped input,
# on-the-fly `by=`/`.by=` grouping, slice_sample(), and any slice() index
# expression that needs a data mask (references a column or `n()`) stay on
# the CPU fallback (R/fallback.R) -- see the dispatch section at the bottom
# of this file.
#
# Semantics below were verified empirically against dplyr 1.2.1 (and
# cross-checked against dplyr's own `get_slice_size()`/`slice_rank_idx()`
# source) before implementation -- see R/slice.R's module doc for the
# full write-up.

oracle_slice_df <- function() {
  tibble::tibble(x = c(10, 20, 30, 40, 50))
}

oracle_slice_rank_df <- function() {
  tibble::tibble(
    x = c(1, 2, 2, 3, NA, NA, 1),
    id = 1:7
  )
}

# =============================================================================
# slice(): positive indices (duplicates + reordering), negative (drop),
# zero/out-of-range dropped, mixed sign errors, no dots, 0-row input
# =============================================================================

test_that("slice(1:3) keeps the first three rows in order", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, 1:3)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice(c(3,1,1)) allows duplicates and reordering", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, c(3, 1, 1))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice(-(1:2)) drops the first two rows, keeps the rest in order", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, -(1:2))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice(0) returns an empty table", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, 0)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice() silently drops out-of-range positive indices", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, c(1, 100))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice() drops NA indices", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, c(1, NA, 2))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice() with no dots returns 0 rows, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice() with negative indices beyond nrow drops everything", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice(d, -(1:10))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice() on a 0-row table returns 0 rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()[0, ]
  pipeline <- function(d) dplyr::slice(d, 1:3)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice() errors when mixing positive and negative indices", {
  skip_if_no_gpu()
  gt <- tbl_gpu(oracle_slice_df())
  expect_error(collect(dplyr::slice(gt, c(1, -2))))
})

test_that("slice() errors on fractional indices", {
  skip_if_no_gpu()
  gt <- tbl_gpu(oracle_slice_df())
  expect_error(collect(dplyr::slice(gt, 1.5)))
})

test_that("slice() composes correctly with a preceding mutate (lazy barrier)", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::mutate(d, y = x * 2) |> dplyr::slice(c(2, 1))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("a filter() after slice() only sees the sliced rows (barrier correctness)", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, n = 3) |> dplyr::filter(x > 15)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# slice_head() / slice_tail(): n=/prop=, defaults, negative n, clamping
# =============================================================================

test_that("slice_head() default (no n/prop) keeps 1 row", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head(n=2) keeps the first two rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, n = 2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head(prop=) floors the row count", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, prop = 0.41)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head(n=) larger than nrow clamps to all rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, n = 100)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head(n=negative) keeps nrow - abs(n) rows from the front", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, n = -2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head(n=very negative) clamps to 0 rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, n = -100)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head(n=0) returns 0 rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_head(d, n = 0)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_tail(n=2) keeps the last two rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_tail(d, n = 2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_tail(n=negative) keeps nrow - abs(n) rows from the back", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  pipeline <- function(d) dplyr::slice_tail(d, n = -2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_head()/slice_tail() on a 0-row table return 0 rows", {
  skip_if_no_gpu()
  df <- oracle_slice_df()[0, ]
  expect_same_as_dplyr(df, function(d) dplyr::slice_head(d, n = 2))
  expect_same_as_dplyr(df, function(d) dplyr::slice_tail(d, n = 2))
})

test_that("head()/tail() base generics match slice_head()/slice_tail()", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  gt <- tbl_gpu(df)

  expect_equal(tibble::as_tibble(collect(head(gt))), head(df))
  expect_equal(tibble::as_tibble(collect(head(gt, 2))), head(df, 2))
  expect_equal(tibble::as_tibble(collect(tail(gt, 2))), tail(df, 2))
  expect_equal(tibble::as_tibble(collect(head(gt, -2))), head(df, -2))
  expect_equal(tibble::as_tibble(collect(tail(gt, -2))), tail(df, -2))
})

# =============================================================================
# slice_min() / slice_max(): with_ties, na_rm, prop, computed order_by,
# result row order follows order_by (not original row order)
# =============================================================================

test_that("slice_min() default (with_ties=TRUE) can return more than n rows on a tie", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_min(d, x, n = 2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min(with_ties=FALSE) returns exactly n rows, tie-broken by position", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_min(d, x, n = 2, with_ties = FALSE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min(na_rm=TRUE) excludes NA order_by rows", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_min(d, x, n = 2, na_rm = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min(na_rm=FALSE) includes NA rows once n exceeds the non-NA count", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_min(d, x, n = 6, na_rm = FALSE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min(prop=) floors the row count like slice_head()", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_min(d, x, prop = 0.3)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min(n=negative) is accepted, same clamp rule as slice_head()", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_min(d, x, n = -1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_max() with na_rm=FALSE never includes NA before the non-NA rows are exhausted", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_max(d, x, n = 2, na_rm = FALSE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_max(prop=) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  pipeline <- function(d) dplyr::slice_max(d, x, prop = 0.3)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min() with a computed order_by does not add it to the output", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y = c(3, 2, 1))
  pipeline <- function(d) dplyr::slice_min(d, x + y, n = 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min()/slice_max() on a 0-row table return 0 rows", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()[0, ]
  expect_same_as_dplyr(df, function(d) dplyr::slice_min(d, x, n = 2))
  expect_same_as_dplyr(df, function(d) dplyr::slice_max(d, x, n = 2))
})

test_that("slice_min() result order follows order_by, not original row order", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_df()
  gt <- tbl_gpu(df)

  result <- collect(dplyr::slice_min(gt, x, n = 3))
  oracle <- dplyr::slice_min(df, x, n = 3)

  expect_equal(result$id, oracle$id)
})

# =============================================================================
# Native-vs-fallback dispatch: grouped input, on-the-fly by=/.by=, and
# data-dependent slice() indices all fall back to CPU (and notify); the
# ungrouped GPU-native path never notifies.
# =============================================================================

test_that("ungrouped slice() family never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(oracle_slice_df())

  expect_no_error(gt |> dplyr::slice(1:2))
  expect_no_error(gt |> dplyr::slice_head(n = 2))
  expect_no_error(gt |> dplyr::slice_tail(n = 2))
  expect_no_error(gt |> dplyr::slice_min(x, n = 1))
  expect_no_error(gt |> dplyr::slice_max(x, n = 1))
})

test_that("grouped slice_head()/slice_tail() fall back to CPU and notify", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice_head(gt, n = 1)),
    "slice_head.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice_head(dplyr::group_by(df, g), n = 1)
  expect_equal(tibble::as_tibble(result), tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("grouped slice()/slice_min()/slice_max() fall back to CPU and notify", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(collect(dplyr::slice(gt, 1)), "slice.*fell back to CPU evaluation")
  expect_warning(collect(dplyr::slice_min(gt, x, n = 1)), "slice_min.*fell back to CPU evaluation")
  expect_warning(collect(dplyr::slice_max(gt, x, n = 1)), "slice_max.*fell back to CPU evaluation")
})

test_that("on-the-fly by=/.by= forces the CPU fallback even for ungrouped input", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice_head(gt, n = 1, by = g)),
    "fell back to CPU evaluation"
  )
  expect_warning(collect(dplyr::slice(gt, 1, .by = g)), "fell back to CPU evaluation")

  oracle <- dplyr::slice_head(df, n = 1, by = g)
  expect_equal(
    dplyr::arrange(tibble::as_tibble(result), g),
    dplyr::arrange(oracle, g)
  )
})

test_that("slice() indices referencing a column fall back to CPU and still match dplyr", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice(gt, which(x > 20))),
    "slice.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice(df, which(x > 20))
  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("slice() indices referencing n() fall back to CPU and still match dplyr", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice(gt, 1:(dplyr::n() - 1))),
    "slice.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice(df, 1:(dplyr::n() - 1))
  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("slice_min() with an order_by expression the IR can't parse falls back", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(3, 1, 2), s = c("c", "a", "b"))
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice_min(gt, toupper(s), n = 1)),
    "slice_min.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice_min(df, toupper(s), n = 1)
  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("slice_sample() still always falls back, ungrouped or not", {
  skip_if_no_gpu()
  df <- oracle_slice_df()
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(collect(dplyr::slice_sample(gt, n = 2)), "fell back to CPU evaluation")
})
