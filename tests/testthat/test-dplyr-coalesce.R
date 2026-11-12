# Mirrored dplyr-oracle tests for coalesce() semantics (Phase 4, wave 1)
#
# Every test compares a `dplyr::coalesce()`-using pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a `tbl_gpu`,
# in both eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `type_coalesce()` and `src/expr_eval.hpp`'s `coalesce` handler):
#
#   1. First non-NA element-wise across args, left to right.
#   2. Type unification follows the same vctrs-style common-type rule as
#      if_else()/case_when() (int/double mix OK, STRING/numeric mix errors).
#   3. A scalar argument recycles/broadcasts against the column-length args.
#   4. All-NA input across every arg stays NA (typed).
#   5. coalesce() composes with if_else() (nested).

# =============================================================================
# Shared fixture
# =============================================================================

oracle_coalesce_df <- function() {
  data.frame(
    x = c(1, NA, 3, NA, 5, NA, 7, 8, NA, 10),
    y = c(NA, 2, NA, 4, NA, 6, 7, NA, 9, 10),
    z = c(NA, NA, NA, 4, 5, 6, NA, 8, NA, NA),
    s = c(NA, "b", NA, "d", NA, "f", "g", NA, "i", "j"),
    t = c("a", NA, "c", NA, "e", NA, NA, "h", NA, "j"),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Basic first-non-NA-wins semantics
# =============================================================================

test_that("coalesce() of two columns matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(x, y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("coalesce() of three columns matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(x, y, z))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("coalesce() on STRING columns matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(s, t))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Scalar recycling
# =============================================================================

test_that("coalesce() with a trailing scalar fallback matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(x, y, z, 0))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("coalesce() with a single scalar arg broadcasts NA-free, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(x, 99))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false(anyNA(result$gpu$w))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# All-NA-across-every-arg stays NA (typed)
# =============================================================================

test_that("coalesce() stays NA when every arg is NA for a row, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  # Row 1 (index 1): x=1 so not all-NA overall, but row 8 has x=8 (not NA
  # either) -- add a genuinely all-NA row via a dedicated small fixture so
  # the trap is unambiguous.
  df2 <- data.frame(a = c(1, NA, NA), b = c(NA, NA, NA), c = c(NA, NA, 3))
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(a, b, c))
  result <- expect_same_as_dplyr(df2, pipeline)
  expect_true(is.na(result$gpu$w[2]))
  expect_same_as_dplyr_lazy(df2, pipeline)
})

# =============================================================================
# Type unification: int/double mix OK; STRING vs numeric errors (both sides)
# =============================================================================

test_that("coalesce() promotes int/double mix to double, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::mutate(d, w = dplyr::coalesce(x, 5L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$w, "double")
})

test_that("coalesce() errors on STRING vs numeric arg mix, matching dplyr's own error", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, w = dplyr::coalesce(x, s)))
  expect_error(dplyr::mutate(gt, w = dplyr::coalesce(x, s)), "coalesce")
})

# =============================================================================
# Composition with if_else()
# =============================================================================

test_that("coalesce() composed with a nested if_else() matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) {
    dplyr::mutate(d, w = dplyr::coalesce(x, dplyr::if_else(is.na(y), 0, y)))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Usage inside filter()
# =============================================================================

test_that("coalesce() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_coalesce_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::coalesce(x, y, z, 0) > 5)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
