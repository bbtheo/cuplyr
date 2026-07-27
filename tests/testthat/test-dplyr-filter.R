# Mirrored dplyr-oracle tests for filter() semantics (Phase 1, task T4)
#
# Every test in this file compares a `dplyr::filter()` pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a `tbl_gpu`,
# in both eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R). The data below is
# deliberately NA-heavy (numeric, string, integer, logical columns each
# carry NAs) to stress the three-valued-logic / NA-propagation rules
# documented in scratchpad/phase1_expression_engine.md section 2.3.
#
# This file does NOT duplicate what test-filter.R already covers (basic
# comparisons, chained filters, data residency, the cuplyr.fallback
# notifier, the IR-cutover smoke tests) -- see that file's own oracle-parity
# section for the handful of overlapping shapes it already pins.

# =============================================================================
# Shared fixture
# =============================================================================

# One fixture reused by most tests below: numeric columns with NAs, a string
# column with NAs, an integer column with NAs, and a logical column with NAs.
oracle_filter_df <- function() {
  data.frame(
    x = c(5, NA, 3, 8, NA, 1, 9, 4, 6, 2),
    y = c(2, 4, NA, 1, 6, 3, 20, 4, NA, 5),
    z = c(1, 10, 2, 3, 20, 5, NA, 8, 4, 7),
    s = c("banana", "apple", NA, "cherry", "date", NA, "fig", "grape", "honeydew", "kiwi"),
    i = c(1L, 2L, 3L, NA, 5L, 6L, 7L, NA, 9L, 10L),
    flag = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE, NA, FALSE, TRUE, FALSE),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Compound predicates: &, |, !, nested, xor()
# =============================================================================

test_that("filter() compound & matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 2 & y < 5)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() compound | matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 6 | y > 6)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() ! (negation) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, !(x > 4))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() ! combined with is.na matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, !is.na(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() nested x > 2 & (y < 5 | z == 1) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 2 & (y < 5 | z == 1))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() deeply nested compound predicate matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) {
    dplyr::filter(d, (x > 1 & x < 9) | (is.na(x) & (y > 2 | z == 1)))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() xor() matches dplyr (logical columns with NA)", {
  skip_if_no_gpu()
  # T4 divergence found + fixed: BITWISE_XOR's AST-derived output type for
  # two BOOL8 operands was INT32 (C++ integer-promotes bool before `^`),
  # which made gpu_filter_expr()'s "mask must be logical" check reject every
  # xor() predicate. Fixed in src/expr_eval.hpp::materialize() by casting the
  # BITWISE_XOR result back to BOOL8 (see that file's inline comment).
  df <- data.frame(
    a = c(TRUE, FALSE, NA, TRUE, FALSE, NA, TRUE),
    b = c(TRUE, TRUE, TRUE, FALSE, FALSE, NA, NA)
  )
  pipeline <- function(d) dplyr::filter(d, xor(a, b))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA semantics
# =============================================================================

test_that("filter() x > 5 drops NA rows like dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 5)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() is.na(x) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, is.na(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() !is.na(x) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, !is.na(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() x > 2 | is.na(x) matches dplyr (NA rescued by OR)", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 2 | is.na(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() is.na() on a string column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, is.na(s))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() is.na() on a logical column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, is.na(flag))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() on a logical column directly (flag) matches dplyr", {
  skip_if_no_gpu()
  # A bare logical-column predicate: NA rows must drop, same as any other
  # NA-valued mask (apply_boolean_mask() keeps non-null-and-true only).
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, flag)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# %in%
# =============================================================================

test_that("filter() %in% numeric set matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x %in% c(3, 4, 9))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% set containing NA matches dplyr (NA row is kept)", {
  skip_if_no_gpu()
  # R's %in% never returns NA: `NA %in% c(3, NA)` is TRUE (the NA row is
  # rescued into the result), `NA %in% c(3, 4)` is FALSE. Verified against
  # the dplyr oracle, not asserted by construction.
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x %in% c(3, NA))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% set WITHOUT NA drops the column's NA rows (via dplyr oracle)", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x %in% c(3, 4, 9, 6))
  result <- expect_same_as_dplyr(df, pipeline)
  # Sanity: the NA rows of x are genuinely excluded here (not incidentally
  # matching some other row), i.e. this test exercises the FALSE branch of
  # the never-NA rule, complementing the TRUE-branch test above.
  expect_false(anyNA(result$gpu$x))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% string set matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s %in% c("apple", "fig", "kiwi"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% string set containing NA matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s %in% c("apple", NA))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% empty set (typed) matches dplyr (T4 divergence, fixed)", {
  skip_if_no_gpu()
  # T4 finding: ir_lit_from_r() rejected ANY length-0 vector, even in the
  # %in% RHS (allow_vector = TRUE) position, so `x %in% numeric(0)` errored
  # in cuplyr at parse time while dplyr correctly returns an empty result
  # (numeric(0) has no candidates, so nothing matches -- not an error).
  # Fixed in R/ir.R::ir_lit_from_r() + src/expr_eval.hpp's %in% handler
  # (empty/NULL RHS short-circuits to an all-FALSE mask).
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x %in% numeric(0))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% empty set (c(), a NULL literal) matches dplyr (T4 divergence, fixed)", {
  skip_if_no_gpu()
  # A second, distinct code path to the same "empty set" outcome: `c()`
  # parses as a NULL-valued literal (not a zero-length *typed* vector), and
  # previously crashed the %in% handler trying to coerce a NULL SEXP into an
  # Rcpp::NumericVector. Same fix (short-circuit before touching the type).
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x %in% c())
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() x %in% single_value matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x %in% 4)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# between()
# =============================================================================

test_that("filter() between() matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::between(x, 2, 6))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() between() with NA in x propagates NA (row dropped) like dplyr", {
  skip_if_no_gpu()
  # between(x, l, r) desugars to (x >= l) & (x <= r); with x == NA this is
  # NA & NA == NA, so the row is dropped by apply_boolean_mask() -- same as
  # dplyr's between(), which is built on the same >=/<=/& semantics.
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, dplyr::between(x, 1, 9))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false(anyNA(result$gpu$x))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# String filters
# =============================================================================

test_that("filter() string == literal matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s == "fig")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() string != literal matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s != "fig")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() string < literal matches dplyr (T4 divergence, fixed)", {
  skip_if_no_gpu()
  # T4 finding: check_filter_comparison_types()'s STRING-vs-STRING guard
  # compared a *named* single-element schema-type subset against a bare
  # "STRING" with identical() -- which considers the "names" attribute, so
  # `identical(c(s = "STRING"), "STRING")` was FALSE even when the column
  # genuinely was STRING, and EVERY string-vs-string-literal comparison
  # (a fully supported shape) incorrectly errored with "Cannot compare
  # column 's' (STRING) with a character literal." Fixed with unname() in
  # R/filter.R::check_filter_comparison_types().
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s < "fig")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() string > literal matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s > "fig")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() string <= / >= literal matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, s <= "fig" | s >= "kiwi")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() string column-vs-column == matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    a = c("x", "y", "z", NA, "w"),
    b = c("y", "y", "a", "q", NA),
    stringsAsFactors = FALSE
  )
  pipeline <- function(d) dplyr::filter(d, a == b)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() string column-vs-column < matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    a = c("x", "y", "z", NA, "w"),
    b = c("y", "y", "a", "q", NA),
    stringsAsFactors = FALSE
  )
  pipeline <- function(d) dplyr::filter(d, a < b)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Column-vs-column mixed with scalars
# =============================================================================

test_that("filter() mixes column-vs-column and column-vs-scalar predicates", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > y & z < 10)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() (x - y) > 0 style arithmetic-in-predicate matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, (x - y) > 0)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Env-variable usage
# =============================================================================

test_that("filter() with an env-variable threshold matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  threshold <- 5
  pipeline <- function(d) dplyr::filter(d, x > threshold)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() %in% with an env-variable vector matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  keep_set <- c(3, 6, 9)
  pipeline <- function(d) dplyr::filter(d, x %in% keep_set)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .data / .env pronouns
# =============================================================================

test_that("filter() .data pronoun matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, .data$x > 2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() .env pronoun with a name colliding with a column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  x <- 3  # deliberately collides with the column name "x"
  pipeline <- function(d) dplyr::filter(d, x > .env$x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Multiple dots (implicit AND) mixed with compound dots
# =============================================================================

test_that("filter() multiple dots (implicit AND) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 1, y < 10)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() multiple dots mixed with a compound expression dot matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, x > 1, y < 10 | is.na(y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Constant edge cases
# =============================================================================

test_that("filter(TRUE) matches dplyr (no-op)", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, TRUE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(FALSE) matches dplyr (drops everything)", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, FALSE)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(NA) matches dplyr (NA drops all rows)", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) dplyr::filter(d, NA)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Grouped table: filter preserves groups metadata
# =============================================================================

test_that("filter() on a grouped table preserves group_vars() like dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(flag) |> dplyr::filter(x > 2)
  }
  # expect_same_as_dplyr()/_lazy() already assert group_vars() equality
  # between the GPU and oracle results before collecting (helper-oracle.R);
  # this test exists specifically to exercise that assertion for filter().
  expect_same_as_dplyr(df, pipeline, arrange_by = c("x", "y", "z", "i"))
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("x", "y", "z", "i"))
})

test_that("filter() on a table grouped by multiple columns preserves group_vars()", {
  skip_if_no_gpu()
  df <- oracle_filter_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(flag, i) |> dplyr::filter(x > 1 | is.na(x))
  }
  expect_same_as_dplyr(df, pipeline, arrange_by = c("x", "y", "z"))
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("x", "y", "z"))
})

# =============================================================================
# 0-row input / all-NA column
# =============================================================================

test_that("filter() on a 0-row input matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_df()[0, ]
  pipeline <- function(d) dplyr::filter(d, x > 2 | is.na(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() with an all-NA column: is.na() matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4), all_na = rep(NA_real_, 4))
  pipeline <- function(d) dplyr::filter(d, is.na(all_na))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() with an all-NA column: a comparison drops every row, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4), all_na = rep(NA_real_, 4))
  pipeline <- function(d) dplyr::filter(d, all_na > 0)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() with an all-NA column: %in% never returns NA (always FALSE here)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4), all_na = rep(NA_real_, 4))
  pipeline <- function(d) dplyr::filter(d, all_na %in% c(1, 2))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
