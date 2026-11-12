# Mirrored dplyr-oracle tests for case_when() semantics (Phase 4, wave 1)
#
# Every test compares a `dplyr::case_when()`-using pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a `tbl_gpu`,
# in both eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `ir_parse_case_when()`/`type_case_when()` and
# `src/expr_eval.hpp`'s `case_when` handler for where each lands):
#
#   1. First-match-wins: earlier `~` clauses take priority over later ones
#      for the same row.
#   2. Unmatched rows (no clause TRUE, no `.default=`) become a TYPED NA --
#      not a bare logical NA -- matching the common type of every RHS value.
#   3. `.default=` supplies the fallback value for unmatched rows instead of
#      NA.
#   4. A NA condition is treated as "no match" (falls through to the next
#      clause, or the default) -- NOT the same trap as if_else()'s NA
#      condition (which becomes a genuine NA output). Verified empirically
#      against `cudf::copy_if_else()`'s documented null-mask rule, which
#      picks the "else" side for a null mask -- exactly this fall-through
#      behavior, with NO adjustment needed (unlike if_else()'s handler,
#      which needs a corrective second pass).
#   5. RHS type unification follows the same vctrs-style common-type rule as
#      if_else() (int/double mix OK, STRING/numeric mix errors).
#   6. if_else() nested inside a case_when() RHS value works (both go
#      through the same IR/apply_handler machinery).

# =============================================================================
# Shared fixture
# =============================================================================

oracle_case_when_df <- function() {
  data.frame(
    x = c(5, NA, 3, 8, -2, 1, 9, 4, 6, 2),
    y = c(2, 4, NA, 1, 6, 3, 20, 4, NA, 5),
    s = c("banana", "apple", NA, "cherry", "date", NA, "fig", "grape", "honeydew", "kiwi"),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# First-match-wins, .default=, unmatched -> typed NA
# =============================================================================

test_that("case_when() first-match-wins matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::case_when(x < 0 ~ "neg", x < 5 ~ "small", .default = "big"))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_when() with .default= matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x < 2 ~ "a", x < 4 ~ "b", .default = "z"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_when() unmatched rows become a typed NA, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x < 2 ~ 1L, x < 4 ~ 2L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "integer")
  expect_true(anyNA(result$gpu$z))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_when() with TRUE ~ ... as a catch-all matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x < 2 ~ "a", TRUE ~ "b"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA-condition trap: NA treated as "no match", falls through (NOT the same
# behavior as if_else()'s NA-condition-becomes-NA trap)
# =============================================================================

test_that("case_when() treats a NA condition as no match (falls through), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  # `x < 2` is NA wherever x is NA -- the catch-all should still fire for
  # those rows (not propagate NA the way if_else()'s NA-cond trap does).
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x < 2 ~ "a", TRUE ~ "b"))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false(anyNA(result$gpu$z))
})

test_that("case_when() with a NA condition and no catch-all still falls through to NA-unmatched, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x < 2 ~ "a"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Type unification: int/double mix OK; STRING vs numeric errors (both sides)
# =============================================================================

test_that("case_when() promotes int/double RHS mix to double, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x < 2 ~ 1L, x < 4 ~ 2.5, .default = 3L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "double")
})

test_that("case_when() errors on STRING vs numeric RHS mix, matching dplyr's own error", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::case_when(x < 2 ~ "a", x < 4 ~ 1)))
  expect_error(dplyr::mutate(gt, z = dplyr::case_when(x < 2 ~ "a", x < 4 ~ 1)), "case_when")
})

test_that("case_when() errors when .default has an incompatible type, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::case_when(x < 2 ~ "a", .default = 5)))
  expect_error(dplyr::mutate(gt, z = dplyr::case_when(x < 2 ~ "a", .default = 5)), "case_when")
})

# =============================================================================
# Nested if_else() inside a case_when() RHS value
# =============================================================================

test_that("if_else() nested inside a case_when() RHS matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) {
    dplyr::mutate(
      d,
      z = dplyr::case_when(
        x < 3 ~ dplyr::if_else(x == 1, "one", "not-one"),
        TRUE ~ "big"
      )
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Long chains
# =============================================================================

test_that("case_when() with a long chain of clauses matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) {
    dplyr::mutate(
      d,
      z = dplyr::case_when(
        x < -1 ~ "very negative",
        x < 0 ~ "negative",
        x == 0 ~ "zero",
        x < 3 ~ "small",
        x < 6 ~ "medium",
        x < 9 ~ "large",
        .default = "huge"
      )
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_when() with column-vs-column conditions matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::case_when(x > y ~ "x_bigger", x < y ~ "y_bigger", .default = "tie"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Usage inside filter()
# =============================================================================

test_that("case_when() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_when_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::case_when(x < 3 ~ TRUE, .default = FALSE))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
