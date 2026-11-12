# Mirrored dplyr-oracle tests for na_if() semantics (Phase 4, wave 1)
#
# Every test compares a `dplyr::na_if()`-using pipeline run against a plain
# tibble (the oracle) with the same pipeline run against a `tbl_gpu`, in both
# eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `type_na_if()` and `src/expr_eval.hpp`'s `na_if` handler):
#
#   1. `na_if(x, y)`'s OUTPUT always keeps x's own type exactly -- e.g.
#      `na_if(1:3, 2)` (integer x vs a double literal y) stays integer; y is
#      only ever compared against, never promoted into the result. This is
#      unlike if_else()/case_when()/coalesce(), which unify to a common
#      type across their value-position args.
#   2. y can be a column (same length as x, compared element-wise -- the
#      GPU-column analogue of dplyr's "y recycled to x's length" vector
#      rule) or a scalar literal (broadcast against every row).
#   3. A pre-existing NA in x stays NA (a comparison against NA is never
#      TRUE, so those rows are never "double-NA'd" incorrectly).
#   4. A bare, untyped `NA` for y is a well-defined no-op-ish comparison
#      (matches x's type, `x == NA` is NA everywhere, so nothing converts).
#   5. A TYPED NA for y that mismatches x's type (e.g. `NA_character_`
#      against a numeric x) still errors, exactly like a non-NA STRING/
#      numeric mismatch -- only a bare untyped NA adopts the sibling's type.
#   6. STRING vs numeric type mismatch errors on both sides (message text
#      differs, but both error).

# =============================================================================
# Shared fixture
# =============================================================================

oracle_na_if_df <- function() {
  data.frame(
    x = c(1, 2, NA, 4, 5, 2, 7, 8, 9, 2),
    y = c(1, 99, 3, 4, 99, 2, 99, 8, 99, 99),
    s = c("a", "b", NA, "d", "b", "f", "g", "b", "i", "j"),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Basic column/scalar and column/column forms
# =============================================================================

test_that("na_if() against a scalar matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(x, 2))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("na_if() against another column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(x, y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("na_if() on a STRING column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(s, "b"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Pre-existing NA in x stays NA
# =============================================================================

test_that("na_if() leaves a pre-existing NA in x untouched, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(x, 999))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.na(result$gpu$z[3]))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Output type: ALWAYS x's own type, never promoted by y
# =============================================================================

test_that("na_if() keeps x's integer type even when y is a double literal, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, 2L, 3L, 4L, 5L))
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(i, 2))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "integer")
})

# =============================================================================
# Bare NA for y: well-defined no-op (matches x's type, converts nothing)
# =============================================================================

test_that("na_if() with a bare NA for y is a no-op matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(x, NA))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("na_if() with a bare NA for y on a STRING column is a no-op matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::na_if(s, NA))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Type strictness: STRING vs numeric errors on both sides (message text
# differs, but both error) -- including via a TYPED (non-bare) NA
# =============================================================================

test_that("na_if() errors on STRING y against a numeric x, matching dplyr's own error", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::na_if(x, s)))
  expect_error(dplyr::mutate(gt, z = dplyr::na_if(x, s)), "na_if")
})

test_that("na_if() errors on a typed NA_character_ y against a numeric x, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::na_if(x, NA_character_)))
  expect_error(dplyr::mutate(gt, z = dplyr::na_if(x, NA_character_)), "na_if")
})

# =============================================================================
# Chaining / usage inside filter()
# =============================================================================

test_that("na_if() chained with a later mutate dot referencing its output matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) {
    dplyr::mutate(d, a = dplyr::na_if(x, 2), b = dplyr::coalesce(a, 0))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("na_if() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_na_if_df()
  pipeline <- function(d) dplyr::filter(d, is.na(dplyr::na_if(x, 2)))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
