# Mirrored dplyr-oracle tests for the slice() family (Phase 3 task 2:
# ungrouped; Phase 5 task W7: grouped/`by=`/`.by=`)
#
# Every test compares a dplyr pipeline run against a plain tibble (the
# oracle) with the same pipeline run against a `tbl_gpu`, in both eager and
# lazy exec modes, via `expect_same_as_dplyr()`/`expect_same_as_dplyr_lazy()`
# (see helper-oracle.R). `expect_same_as_dplyr()` also compares `group_vars()`
# of both sides BEFORE collecting, so a grouped-slice test that asserts
# nothing else about grouping is still checking `group_by()`-grouped stays
# grouped and `by=`/`.by=` comes back ungrouped.
#
# GPU-native: slice()/slice_head()/slice_tail()/slice_min()/slice_max(), for
# ungrouped input (Phase 3.2, dedicated C++ kernels, R/ast.R::ast_slice(),
# R/lower.R::lower_slice(), src/ops_slice.cpp), `group_by()`-grouped input,
# and on-the-fly `by=`/`.by=` grouping (Phase 5 W7, a pure-R window/mutate/
# filter/arrange/select desugar, R/slice.R::slice_grouped_native() -- ZERO
# new GPU primitives). Still-fallback shapes: a `slice()` index expression
# needing a data mask (references a column or `n()`); a grouped/`by=`
# `slice()` index vector with duplicates/reordering (the grouped desugar's
# `rn %in% i` predicate can only express a subset-membership test, not an
# arbitrary per-group reordering/duplication); `slice_min()`/`slice_max()`
# with an `order_by` expression the IR can't parse. `slice_sample()` (Phase
# 5 W8) is GPU-native for ungrouped/group_by()/by= (CPU-side RNG index
# generation + GPU gather, R/slice.R::slice_sample_native()), falling back
# only for `weight_by=` -- see test-dplyr-slice-sample.R for its dedicated
# coverage.
#
# Semantics below were verified empirically against dplyr 1.2.1 (and
# cross-checked against dplyr's own `get_slice_size()`/`slice_rank_idx()`
# source) before implementation -- see R/slice.R's module doc for the
# full write-up, including the grouped-vs-`by=` group-ordering rules (Phase
# 5 W7): `group_by()`-grouped output is reordered into ASCENDING group-key
# order (NA-key group last); `by=`/`.by=` output is ordered by each group's
# FIRST-APPEARANCE position in the input (a correction to this task's own
# design doc, which assumed `by=` simply preserves literal original row
# order -- verified empirically NOT to be the case, see the dedicated tests
# below), for every slice_* variant including slice_min()/slice_max(), not
# just the positional ones.

oracle_slice_df <- function() {
  tibble::tibble(x = c(10, 20, 30, 40, 50))
}

oracle_slice_rank_df <- function() {
  tibble::tibble(
    x = c(1, 2, 2, 3, NA, NA, 1),
    id = 1:7
  )
}

oracle_slice_grouped_df <- function() {
  # Unsorted/interleaved groups (including an NA group key), so a test
  # comparing against this fixture actually exercises the group-key
  # reordering rule rather than accidentally passing because the groups
  # were already contiguous/sorted.
  tibble::tibble(
    g = c(3, 1, NA, 2, 1, 3, NA, 2),
    x = c(10, 20, 30, 40, 50, 60, 70, 80)
  )
}

oracle_slice_rank_grouped_df <- function() {
  tibble::tibble(
    g = c(1, 1, 1, 1, 2, 2, 2),
    x = c(5, 3, 3, 1, 2, 2, 9),
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
# Grouped (group_by()) slice() family: GPU-native (Phase 5, task W7)
#
# Output is reordered into ASCENDING group-key order (NA-key group last);
# each group's own surviving rows keep their original relative order (or,
# for slice_min()/slice_max(), order_by order, NA always last regardless of
# direction). Verified empirically against dplyr 1.2.1 -- see R/slice.R's
# module docs.
# =============================================================================

test_that("grouped slice_head(n=1) reorders into ascending group-key order, NA-key group last", {
  skip_if_no_gpu()
  df <- oracle_slice_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_head(n = 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_head(n=2) preserves each group's own original row order", {
  skip_if_no_gpu()
  df <- oracle_slice_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_head(n = 2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_tail(n=1) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_slice_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_tail(n = 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_head(n=negative) drops the last abs(n) rows PER GROUP", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_head(n = -1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_tail(n=negative) drops the first abs(n) rows PER GROUP", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_tail(n = -1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_head(prop=) floors PER-GROUP row count", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_head(prop = 0.5)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_head(n=) larger than a group's size clamps to that group's own size", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_head(n = 10)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice(1:2) is per-group positional and still reorders into group-key order", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 2, 1, 1, 1), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice(1:2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice(-1) drops the first row of each group", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 2, 1, 1, 1), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice(-1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(with_ties=TRUE) orders within-group by x ascending, ties by original position", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = 2, with_ties = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_max(with_ties=TRUE) orders within-group by x descending, ties by original position", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_max(x, n = 2, with_ties = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(with_ties=FALSE) returns exactly n rows per group, tie-broken by position", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = 2, with_ties = FALSE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(na_rm=FALSE, with_ties=TRUE) includes every NA row once n reaches the non-NA count", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 1, 1), x = c(3, NA, 1, NA, 2))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = 4, with_ties = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(na_rm=FALSE, with_ties=FALSE) keeps exactly n rows, only the first-position NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 1, 1), x = c(3, NA, 1, NA, 2))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = 4, with_ties = FALSE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(na_rm=TRUE) excludes NA order_by rows", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 1, 1), x = c(3, NA, 1, NA, 2))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = 4, na_rm = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_max(na_rm=FALSE) never includes NA before the non-NA rows are exhausted", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 1, 1), x = c(3, NA, 1, NA, 2))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_max(x, n = 4, with_ties = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(n=negative) applies the same per-group clamp rule as slice_head()", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = -1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min(prop=) floors PER-GROUP row count", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_grouped_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, prop = 0.5)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min() with a computed order_by does not add it to the output", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(1, 2, 3, 4), y = c(4, 3, 2, 1))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x + y, n = 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grouped slice_min()/slice_max() on a 0-row table return 0 rows", {
  skip_if_no_gpu()
  df <- oracle_slice_rank_grouped_df()[0, ]
  expect_same_as_dplyr(df, function(d) dplyr::group_by(d, g) |> dplyr::slice_min(x, n = 2))
  expect_same_as_dplyr(df, function(d) dplyr::group_by(d, g) |> dplyr::slice_max(x, n = 2))
})

# =============================================================================
# by=/.by= slice() family: GPU-native (Phase 5, task W7)
#
# Ordered by each group's FIRST-APPEARANCE position in the input -- NOT
# ascending group-key order (that's group_by()'s rule, above), and NOT the
# literal original row order of the surviving rows either (see the
# dedicated tests below, which construct cases where those three orderings
# all disagree). Always ungrouped.
# =============================================================================

test_that("by= slice_head(n=1) orders by group FIRST-APPEARANCE position, not ascending group-key", {
  skip_if_no_gpu()
  df <- oracle_slice_grouped_df()
  pipeline <- function(d) dplyr::slice_head(d, n = 1, by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that(".by= slice(1:2) orders by group first-appearance position", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 2, 1, 1, 1), x = 1:5)
  pipeline <- function(d) dplyr::slice(d, 1:2, .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("by= slice_tail(n=1) orders by group first-appearance, not the surviving row's own physical position", {
  skip_if_no_gpu()
  # g's first-appearance order is 3, 1, 2 (rows 1, 2, 3); slice_tail(n=1)'s
  # surviving rows are at physical positions 4 (g=1), 5 (g=3), 6 (g=2) --
  # a plain "keep original row order of survivors" rule would give g order
  # 1, 3, 2, which does NOT match dplyr's actual by= output (verified
  # empirically): it's ordered by group first-appearance (3, 1, 2) instead.
  df <- tibble::tibble(g = c(3, 1, 2, 1, 3, 2), x = 1:6)
  pipeline <- function(d) dplyr::slice_tail(d, n = 1, by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("by= slice_min() orders by group first-appearance, not the surviving (min) row's own position", {
  skip_if_no_gpu()
  # g=2's first row (x=100) is NOT its minimum (x=1, at physical position 3);
  # g=2 still comes first in the output because g=2's GROUP first appears
  # before g=1's group does (row 1 vs row 2) -- confirmed empirically to be
  # dplyr's actual rule, not "original position of the surviving row" (that
  # would put g=1's row, at position 2, ahead of g=2's, at position 3).
  df <- tibble::tibble(g = c(2, 1, 2, 1), x = c(100, 5, 1, 50))
  pipeline <- function(d) dplyr::slice_min(d, x, n = 1, by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("by= slice_max() also uses group first-appearance order", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 1, 2, 1), x = c(100, 5, 1, 50))
  pipeline <- function(d) dplyr::slice_max(d, x, n = 1, by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("by= slice_head()/slice_min() results are always ungrouped", {
  skip_if_no_gpu()
  df <- oracle_slice_grouped_df()
  gt <- tbl_gpu(df)

  r1 <- dplyr::slice_head(gt, n = 1, by = g)
  expect_identical(r1$groups, character())

  r2 <- dplyr::slice_min(gt, x, n = 1, by = g)
  expect_identical(r2$groups, character())
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

test_that("grouped slice_head()/slice_tail()/slice()/slice_min()/slice_max() never trigger a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  expect_no_error(collect(dplyr::slice_head(gt, n = 1)))
  expect_no_error(collect(dplyr::slice_tail(gt, n = 1)))
  expect_no_error(collect(dplyr::slice(gt, 1)))
  expect_no_error(collect(dplyr::slice_min(gt, x, n = 1)))
  expect_no_error(collect(dplyr::slice_max(gt, x, n = 1)))
})

test_that("on-the-fly by=/.by= grouping never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df)

  expect_no_error(collect(dplyr::slice_head(gt, n = 1, by = g)))
  expect_no_error(collect(dplyr::slice(gt, 1, .by = g)))
  expect_no_error(collect(dplyr::slice_min(gt, x, n = 1, by = g)))
})

test_that("grouped slice() with a duplicated/reordered index vector still falls back to CPU and notifies", {
  skip_if_no_gpu()
  # The grouped desugar's `rn %in% i` predicate can only express a
  # per-group subset-membership test, never dplyr's own arbitrary
  # reordering/duplication of the kept rows -- unlike the ungrouped path,
  # which handles this shape natively (see the top-of-file "slice(c(3,1,1))"
  # tests). Still falls back correctly and matches dplyr.
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5)
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice(gt, c(2, 1))),
    "slice.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice(dplyr::group_by(df, g), c(2, 1))
  expect_equal(tibble::as_tibble(result), tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that(".by= slice() with a duplicated/reordered index vector still falls back to CPU and notifies", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5)
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice(gt, c(2, 1), .by = g)),
    "slice.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice(df, c(2, 1), .by = g)
  expect_equal(tibble::as_tibble(result), oracle)
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
  # toupper() is natively registered as of Phase 11 task L5 -- sprintf() is
  # not, and is still a genuinely unparseable order_by expression for this
  # test's purpose (see test-dplyr-strings.R for toupper()'s own coverage).
  df <- tibble::tibble(x = c(3, 1, 2), s = c("c", "a", "b"))
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice_min(gt, sprintf("%s", s), n = 1)),
    "slice_min.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice_min(df, sprintf("%s", s), n = 1)
  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("grouped slice_min() with an order_by expression the IR can't parse falls back", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(3, 1, 2, 5), s = c("c", "a", "b", "d"))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- collect(dplyr::slice_min(gt, sprintf("%s", s), n = 1)),
    "slice_min.*fell back to CPU evaluation"
  )
  oracle <- dplyr::slice_min(dplyr::group_by(df, g), sprintf("%s", s), n = 1)
  expect_equal(tibble::as_tibble(result), tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("slice_min() with a NATIVE string order_by expression (toupper()) works and never falls back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(x = c(3, 1, 2), s = c("c", "a", "b"))
  gt <- tbl_gpu(df)

  expect_no_error(result <- collect(dplyr::slice_min(gt, toupper(s), n = 1)))
  oracle <- dplyr::slice_min(df, toupper(s), n = 1)
  expect_equal(tibble::as_tibble(result), oracle)
})

# slice_sample() is GPU-native (ungrouped, group_by()-grouped, and by=) as
# of Phase 5 task W8 -- see test-dplyr-slice-sample.R for its full
# parity/dispatch coverage (including the `weight_by=` fallback case,
# which is the only shape that still falls back to CPU).
