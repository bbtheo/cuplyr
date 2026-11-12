# Mirrored dplyr-oracle tests for if_else() semantics (Phase 4, wave 1)
#
# Every test compares a `dplyr::if_else()`-using pipeline run against a plain
# tibble (the oracle) with the same pipeline run against a `tbl_gpu`, in both
# eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `type_if_else()`/`ir_unify_types()` and
# `src/expr_eval.hpp`'s `if_else` handler for where each lands):
#
#   1. Type strictness is NOT "yes/no must be identical types": dplyr's
#      vctrs-based if_else() freely promotes across integer/double/logical
#      (`if_else(cond, 1L, 2.5)` succeeds as a double), and only errors on a
#      genuinely incompatible mix (character vs numeric).
#   2. A NA condition produces NA in the output (or the `missing=` value if
#      supplied) -- this is NOT the same as treating NA like FALSE (which
#      would silently select the `no` branch instead). Verified against
#      `cudf::copy_if_else()`'s own documented null-mask rule (which DOES
#      pick the "false" side for a null mask) -- if_else()'s handler runs a
#      second copy_if_else() pass keyed on is.na(cond) specifically to
#      correct for this divergence.
#   3. NA in the yes/no branch itself propagates normally.
#   4. Recycling: scalar yes/no broadcast against a vector cond; mismatched
#      vector lengths are a dplyr error (not exercised here: cuplyr's mutate
#      columns are always table-length, so this dplyr-specific vctrs
#      recycling error doesn't have a GPU-column analogue).

# =============================================================================
# Shared fixture
# =============================================================================

oracle_if_else_df <- function() {
  data.frame(
    x = c(5, NA, 3, 8, -2, 1, 9, 4, 6, 2),
    y = c(2, 4, NA, 1, 6, 3, 20, 4, NA, 5),
    i = c(1L, 2L, 3L, NA, 5L, 6L, 7L, NA, 9L, 10L),
    s = c("banana", "apple", NA, "cherry", "date", NA, "fig", "grape", "honeydew", "kiwi"),
    flag = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE, NA, FALSE, TRUE, FALSE),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Basic column/column and column/scalar forms
# =============================================================================

test_that("if_else() column/column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, x, y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() column/scalar matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, 1, 0))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() scalar/scalar (both literal) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, 100L, -100L))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() with a nested arithmetic condition matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else((x - y) > 0, x, y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() on a STRING column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(is.na(s), "MISSING", s))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() with logical yes/no produces a logical column, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, TRUE, FALSE))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA-condition trap: NA cond -> NA output (never silently the `no` branch)
# =============================================================================

test_that("if_else() with a NA condition produces NA (not the `no` branch), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  # `x > 2` is NA wherever x is NA -- exercises the trap directly.
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, 1L, 2L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(anyNA(result$gpu$z))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() `missing=` substitutes for a NA condition, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::if_else(x > 2, 1L, 2L, missing = 99L))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false(anyNA(result$gpu$z))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() `missing=` with a column value matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::if_else(x > 2, x, y, missing = 0))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() NA in the yes/no branch itself propagates, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, y, x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() with a bare NA branch adopts the sibling's type, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, x, NA))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Type unification: int/double mix succeeds (NOT strict-identical), matching
# dplyr's vctrs-based common-type promotion
# =============================================================================

test_that("if_else() promotes int/double mix to double, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, 1L, 2.5))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "double")
})

test_that("if_else() preserves integer type when yes/no are both integer, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::if_else(x > 2, 1L, 2L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "integer")
})

# =============================================================================
# Type strictness: STRING vs non-STRING errors on both sides (message text
# differs, but both error)
# =============================================================================

test_that("if_else() errors on STRING vs numeric branches, matching dplyr's own error", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::if_else(x > 2, s, x)))
  expect_error(dplyr::mutate(gt, z = dplyr::if_else(x > 2, s, x)), "if_else")
})

test_that("if_else() errors when `missing=` has an incompatible type, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::if_else(x > 2, 1, 2, missing = "x")))
  expect_error(dplyr::mutate(gt, z = dplyr::if_else(x > 2, 1, 2, missing = "x")), "if_else")
})

# =============================================================================
# Usage inside filter()
# =============================================================================

test_that("if_else() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::if_else(x > 2, TRUE, FALSE))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("if_else() with is.na() inside filter() matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::if_else(is.na(x), FALSE, x > 2))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Chaining / multiple dots
# =============================================================================

test_that("if_else() used across multiple mutate dots, later referencing earlier, matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_if_else_df()
  pipeline <- function(d) {
    dplyr::mutate(
      d,
      a = dplyr::if_else(x > 2, x, y),
      b = dplyr::if_else(a > 5, a, 0)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
