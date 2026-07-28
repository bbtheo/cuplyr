# Mirrored dplyr-oracle tests for the dplyr 1.2 "recode/replace" family --
# recode_values(), replace_values(), replace_when() (Phase 4, wave 2).
#
# Every test compares a dplyr-using pipeline run against a plain tibble (the
# oracle) with the same pipeline run against a `tbl_gpu`, in both eager and
# lazy exec modes, via `expect_same_as_dplyr()` / `expect_same_as_dplyr_lazy()`
# (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `ir_parse_recode_values()`/`ir_parse_replace_values()`/
# `ir_parse_replace_when()` and `src/expr_eval.hpp`'s `replace_when` handler):
#
#   1. `recode_values(x, ..., default=)` is the non-deprecated "create an
#      entirely new vector" equivalent of `case_match()` -- same value-match
#      (`%in%`-style) semantics, same common-type unification across every
#      clause's value plus `default`, unmatched rows become `default` (or a
#      typed NA).
#   2. `replace_values(x, ...)` is the TYPE STABLE "partially update x"
#      counterpart -- verified empirically: `typeof(replace_values(1:3, 2 ~
#      NA))` stays "integer" even where `typeof(case_match(1:3, 2 ~ NA,
#      .default = 1:3))` would too (case_match is also type-stable via
#      .default = x, but recode_values-with-a-double RHS is NOT, e.g.
#      `recode_values(1:3, 2 ~ 99.5)` promotes to double -- replace_values
#      always casts the replacement DOWN/up to x's own type instead). There
#      is no `default=` argument at all: unmatched rows keep their original
#      `x` value.
#   3. `replace_when(x, cond1 ~ val1, ...)` is replace_values()'s
#      boolean-condition analogue (case_when()'s clause shape, but type
#      stable to `x` with `x` itself as the default) -- a `NA` condition is
#      treated like FALSE (falls through, same as case_when()), and
#      unmatched rows keep their original `x` value.
#   4. All three reject a replacement/matched value whose type is
#      incompatible with `x` (STRING vs numeric), matching dplyr's own
#      type-safety, with a diagnosable error rather than a raw GPU cast
#      failure.

# =============================================================================
# Shared fixture
# =============================================================================

oracle_recode_df <- function() {
  data.frame(
    x = c(1L, 2L, 3L, NA, 5L, 2L, 1L, 4L, NA, 3L),
    s = c("NC", "NYC", "CA", NA, "NYC", "Unknown", "CA", "NC", "NYC", NA),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# recode_values(): common-type unify, unmatched -> default/typed NA
# =============================================================================

test_that("recode_values() with vector LHS clauses and default matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::recode_values(
      x, c(1, 3) ~ "odd", c(2, 4) ~ "even", default = "other"
    ))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("recode_values() with no default: unmatched rows become typed NA, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::recode_values(x, c(1, 3) ~ "odd", c(2, 4) ~ "even"))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.na(result$gpu$z[5]))  # x = 5, unmatched
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("recode_values() on a STRING column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::recode_values(
      s, "NC" ~ "North Carolina", "NYC" ~ "New York", "CA" ~ "California", default = "Other"
    ))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("recode_values() with NA ~ value replaces x's own NA elements, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::recode_values(s, NA ~ "unknown", default = s))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$z[4], "unknown")
  expect_equal(result$gpu$z[9], "NYC")
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# replace_values(): TYPE STABLE, no default=, unmatched keep original x
# =============================================================================

test_that("replace_values() is type stable (integer stays integer) even with a double RHS, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_values(x, c(1, 3) ~ 99))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "integer")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("replace_values() leaves unmatched rows as the original x value, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_values(x, c(1, 3) ~ 0L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$z[2], 2L)  # x = 2, unmatched: kept as-is
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("replace_values() with NA ~ value replaces x's own NA elements, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_values(x, NA ~ -1L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$z[4], -1L)
  expect_equal(result$gpu$z[9], -1L)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("replace_values() on a STRING column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_values(s, "NYC" ~ "New York"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# replace_when(): boolean conditions, type stable, NA cond treated as FALSE
# =============================================================================

test_that("replace_when() is type stable (integer stays integer) even with a double RHS, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_when(x, x > 3 ~ 99))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_type(result$gpu$z, "integer")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("replace_when() with a NA condition falls through (kept as original x), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_when(x, x > 3 ~ 0L))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.na(result$gpu$z[4]))  # x is NA -> condition NA -> kept as NA
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("replace_when() with is.na(x) explicitly catching NA matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::replace_when(x, is.na(x) ~ -1L, x > 3 ~ 99L))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("replace_when() with multiple clauses (first match wins) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) {
    dplyr::mutate(d, z = dplyr::replace_when(x, x >= 1 ~ 100L, x >= 3 ~ 200L))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Type strictness: incompatible replacement/matched value errors
# =============================================================================

test_that("replace_values() errors on a STRING LHS against a numeric x, matching dplyr's own type-safety", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::replace_values(x, "a" ~ 1L)))
  expect_error(dplyr::mutate(gt, z = dplyr::replace_values(x, "a" ~ 1L)), "replace_values")
})

test_that("replace_when() errors on a STRING replacement value against a numeric x, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(df, z = dplyr::replace_when(x, x > 1 ~ "a")))
  expect_error(dplyr::mutate(gt, z = dplyr::replace_when(x, x > 1 ~ "a")), "replace_when")
})

# =============================================================================
# Chaining / usage inside filter()
# =============================================================================

test_that("replace_when() chained with a later mutate dot referencing its output matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) {
    dplyr::mutate(d, a = dplyr::replace_when(x, x > 3 ~ 0L), b = a == 0L)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("recode_values() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_recode_df()
  pipeline <- function(d) {
    dplyr::filter(d, dplyr::recode_values(x, c(1, 3) ~ TRUE, default = FALSE))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
