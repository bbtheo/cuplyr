# Mirrored dplyr-oracle tests for distinct() semantics (Phase 3, task 1)
#
# Every test compares a `dplyr::distinct()` pipeline run against a plain
# tibble (the oracle) with the same pipeline run against a `tbl_gpu`, in
# both eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# distinct() is GPU-native (`cudf::stable_distinct()`, see R/distinct.R,
# R/ast.R::ast_distinct(), R/lower.R::lower_distinct(),
# src/ops_distinct.cpp::gpu_distinct()) -- semantics below were verified
# empirically against dplyr 1.2.1 before implementation (row order
# preservation, column order for `.keep_all = FALSE`, group-column
# prepending rules, NA/NaN equality) rather than assumed.
#
# Factor keys: distinct() never talks to any C++ primitive by TYPE STRING
# (lower_distinct(), R/lower.R, only ever passes column INDICES to
# gpu_distinct()/gpu_select()) -- the DICTIONARY32-vs-physical-INT32 split
# formalized by gpu_physical_type() (Phase 11 L2) never mattered here, and
# distinct() reaches its schema through the generic push_op()/
# propagate_factor_levels() path (R/execute.R), not build_join_schema()
# (the thing Bug 1 actually fixed). See the "Factor keys" section below.

oracle_distinct_df <- function() {
  tibble::tibble(
    x = c(1, 2, 1, 3, 2, 1),
    y = c("a", "b", "a", "c", "b", "a"),
    z = c(10, 20, 30, 40, 50, 60)
  )
}

# =============================================================================
# No arguments: all columns as key, original row order preserved
# =============================================================================

test_that("distinct() with no args uses all columns as key and preserves row order", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() with no args + .keep_all is a no-op (already equivalent)", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, .keep_all = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Key columns only: result contains ONLY the requested columns, in the
# order given
# =============================================================================

test_that("distinct(x, y) keeps only key columns, first occurrence, row order preserved", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, x, y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct(y, x) returns columns in the order given (y, x), not schema order", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, y, x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct(x) on a single column", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .keep_all = TRUE: key determines dedup, but every column is kept
# =============================================================================

test_that("distinct(x, .keep_all = TRUE) keeps all columns, first row per key", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, x, .keep_all = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Expressions: distinct(z = x + 1) computes via a mutate step first
# =============================================================================

test_that("distinct() on a computed expression matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, w = x + 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() on a computed expression + .keep_all appends the new column", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, w = x + 1, .keep_all = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() on an expression that replaces an existing column in-place", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, x = y, .keep_all = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() unnamed computed expression uses deparsed text as column name", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, x + 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() does not warn for an unnamed computed expression", {
  skip_if_no_gpu()
  gt <- tbl_gpu(oracle_distinct_df())
  expect_no_warning(gt |> dplyr::distinct(x + 1))
})

# =============================================================================
# Grouped tables: group columns are always in the key and the result
# =============================================================================

test_that("distinct() on a grouped table prepends a missing group column", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) d |> dplyr::group_by(y) |> dplyr::distinct(x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() on a grouped table keeps the given position when the group column is named", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) d |> dplyr::group_by(y) |> dplyr::distinct(x, y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() on a grouped table with .keep_all preserves group_vars", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) d |> dplyr::group_by(y) |> dplyr::distinct(x, .keep_all = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() on a grouped table with no dots dedups per-group and keeps groups", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) d |> dplyr::group_by(y) |> dplyr::distinct()
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() with a computed key on a grouped table not matching the group name still prepends the group", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) d |> dplyr::group_by(x) |> dplyr::distinct(w = x + 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA / NaN: both count as a single distinct value (one row kept)
# =============================================================================

test_that("distinct() collapses duplicate NA key rows to one", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(NA_real_, NA_real_, 1, 2), y = c(1, 2, 3, 4))
  pipeline <- function(d) dplyr::distinct(d, x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() collapses duplicate computed NaN key rows to one", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(0, 0, 1, 2))
  pipeline <- function(d) dplyr::distinct(d, w = x / x - 1)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Edge cases: 0-row input, all-duplicate input, string keys
# =============================================================================

test_that("distinct() on a 0-row table returns 0 rows with the right schema", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()[0, ]
  pipeline <- function(d) dplyr::distinct(d, x, y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() collapses an all-duplicate table to a single row", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = rep(1, 5), y = rep("a", 5))
  pipeline <- function(d) dplyr::distinct(d)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct() on string keys matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    s = c("apple", "banana", "apple", "cherry", "banana"),
    v = 1:5
  )
  pipeline <- function(d) dplyr::distinct(d, s)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("distinct(x, x) duplicate dot names collapse to one key column", {
  skip_if_no_gpu()
  df <- oracle_distinct_df()
  pipeline <- function(d) dplyr::distinct(d, x, x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# distinct() does not trigger the CPU fallback notifier
# =============================================================================

test_that("distinct() never triggers cuplyr_fallback_notify()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(oracle_distinct_df())

  expect_no_error(gt |> dplyr::distinct())
  expect_no_error(gt |> dplyr::distinct(x, y))
  expect_no_error(gt |> dplyr::distinct(x, .keep_all = TRUE))
  expect_no_error(gt |> dplyr::distinct(w = x + 1))
  expect_no_error(gt |> dplyr::group_by(y) |> dplyr::distinct(x))
})

# =============================================================================
# Factor keys (Phase 11 L2): now testable -- see the file header comment
# =============================================================================

oracle_distinct_factor_df <- function() {
  tibble::tibble(
    g = factor(c("a", "b", "a", "c", "b", "a"), levels = c("a", "b", "c")),
    v = c(10, 20, 30, 40, 50, 60)
  )
}

test_that("distinct() on a factor key column matches dplyr and preserves levels", {
  skip_if_no_gpu()
  df <- oracle_distinct_factor_df()
  pipeline <- function(d) dplyr::distinct(d, g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  result <- dplyr::distinct(tbl_gpu(df), g) |> collect()
  expect_s3_class(result$g, "factor")
  expect_equal(levels(result$g), c("a", "b", "c"))
})

test_that("distinct(.keep_all = TRUE) keeps a non-key factor column with its levels", {
  skip_if_no_gpu()
  df <- oracle_distinct_factor_df()
  pipeline <- function(d) dplyr::distinct(d, v = v %% 2, .keep_all = TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  result <- dplyr::distinct(tbl_gpu(df), v = v %% 2, .keep_all = TRUE) |> collect()
  expect_s3_class(result$g, "factor")
  expect_equal(levels(result$g), c("a", "b", "c"))
})

test_that("distinct() on a grouped table with a factor group column preserves levels", {
  skip_if_no_gpu()
  df <- oracle_distinct_factor_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::distinct()
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
