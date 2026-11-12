# Mirrored dplyr-oracle tests for near() semantics (Phase 4, wave 2)
#
# Every test compares a `dplyr::near()`-using pipeline run against a plain
# tibble (the oracle) with the same pipeline run against a `tbl_gpu`, in both
# eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# `dplyr:::near`'s own source and R/ir.R's `"near"` registry entry):
#
#   1. `near(x, y, tol = .Machine$double.eps^0.5)` is EXACTLY `abs(x - y) <
#      tol` -- verified by inspecting `dplyr:::near`'s own function body,
#      nothing more elaborate. Fully desugared at parse time into existing
#      `abs()`/`-`/`<` IR nodes -- no new C++ handler.
#   2. NA propagates normally: `near(NA, 1)` is `NA` (matches ordinary
#      arithmetic/comparison NA propagation, not a special near()-specific
#      rule).
#   3. `tol` defaults to `.Machine$double.eps^0.5` when omitted, and can be
#      overridden positionally or by name.

# =============================================================================
# Shared fixture
# =============================================================================

oracle_near_df <- function() {
  data.frame(
    x = c(1, 2, NA, 4, sqrt(2)^2, 100, -5.5, 0, NA, 3.14159),
    y = c(1.0000000001, 2.5, 3, NA, 2, 100.0001, -5.5, 0.0000000001, 7, 3.14159000001)
  )
}

# =============================================================================
# Basic column/column and column/scalar forms
# =============================================================================

test_that("near() column/column (default tol) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::near(x, y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("near() column/scalar (default tol) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::near(x, 2))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("near() with an explicit custom tol matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::near(x, y, tol = 0.001))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("near() with a very tight tol (mostly FALSE) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::near(x, y, tol = 1e-12))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# sqrt(2)^2 == 2 floating-point trap (the motivating example from dplyr's own
# docs: `sqrt(2)^2 == 2` is FALSE, but `near(sqrt(2)^2, 2)` is TRUE)
# =============================================================================

test_that("near() correctly handles the sqrt(2)^2 == 2 floating point trap, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::mutate(d, exact = x == y, approx = dplyr::near(x, y))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false(result$gpu$exact[5])
  expect_true(result$gpu$approx[5])
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA propagation
# =============================================================================

test_that("near() propagates NA normally, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::near(x, y))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.na(result$gpu$z[3]))
  expect_true(is.na(result$gpu$z[4]))
  expect_true(is.na(result$gpu$z[9]))
})

# =============================================================================
# Usage inside filter()
# =============================================================================

test_that("near() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::near(x, y, tol = 0.001))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("near() negated inside filter() matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::filter(d, !dplyr::near(x, y, tol = 0.001))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Chaining with other expressions
# =============================================================================

test_that("near() combined with & inside a predicate matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_near_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::near(x, y, tol = 0.001) & x > 0)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
