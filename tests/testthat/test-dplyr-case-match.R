# Mirrored dplyr-oracle tests for case_match() semantics (Phase 4, wave 2)
#
# Every test compares a `dplyr::case_match()`-using pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a `tbl_gpu`, in
# both eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `ir_parse_value_match_clauses()`/`ir_parse_case_match()` and
# `src/expr_eval.hpp`'s `%in%` handler, which needed a real pre-existing bug
# fix for this feature -- see the dedicated regression test below):
#
#   1. `case_match()` is DEPRECATED in dplyr 1.2.1 (superseded by
#      `recode_values()`/`replace_values()`, see test-dplyr-recode-values.R)
#      -- calling it emits a lifecycle warning, suppressed below with
#      `suppressWarnings()` since it's still exported and functional, and is
#      explicitly in this wave's scope.
#   2. Matching is by VALUE (a `%in%`-style membership test), not `==`: a
#      clause's left-hand side can be a vector (`c(1, 3) ~ "odd"`), matching
#      ANY element of `.x` equal to ANY value in the vector.
#   3. NA matches NA: `case_match(x, NA ~ "unknown", .default = x)` replaces
#      `x`'s own NA elements specifically (base R's `NA %in% NA` is `TRUE`) --
#      this is NOT the same as an empty/no-op match.
#   4. Unmatched rows become `.default`, or a typed NA if `.default` isn't
#      supplied.
#   5. The common-type unification across every clause's value plus
#      `.default` follows the same vctrs-style promotion rule as
#      `case_when()` (int/double/logical mix freely; STRING only unifies
#      with STRING).
#   6. A left-hand-side value of an incompatible type (e.g. a STRING literal
#      against a numeric `.x`) errors, matching dplyr's own type-safety.

# =============================================================================
# Shared fixture
# =============================================================================

oracle_case_match_df <- function() {
  data.frame(
    x = c(1L, 2L, NA, 3L, 4L, 1L, 5L, 2L, NA, 3L),
    s = c("a", "b", NA, "c", "a", "d", "b", NA, "c", "a"),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Basic multi-clause matching (numeric)
# =============================================================================

test_that("case_match() with vector LHS clauses and .default matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(
      d, z = dplyr::case_match(x, c(1, 3) ~ "odd", c(2, 4) ~ "even", .default = "other")
    ))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_match() with no .default: unmatched rows become typed NA, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(d, z = dplyr::case_match(x, c(1, 3) ~ "odd", c(2, 4) ~ "even")))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.na(result$gpu$z[7]))  # x = 5, unmatched
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_match() on a STRING column matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(
      d, z = dplyr::case_match(s, c("a", "d") ~ "group1", "b" ~ "group2", .default = "other")
    ))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA matching: NA on the LHS matches x's own NA elements (NOT an empty match)
# =============================================================================

test_that("case_match() with NA ~ value replaces x's own NA elements, matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(d, z = dplyr::case_match(x, NA ~ 99L, .default = x)))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$z[3], 99L)
  expect_equal(result$gpu$z[9], 99L)
  expect_equal(result$gpu$z[1], 1L)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_match() with a mixed NA+value LHS vector matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(d, z = dplyr::case_match(x, c(1, NA) ~ 0L, .default = x)))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("regression: bare scalar NA as the WHOLE %in% RHS matches x's own NA rows (not always FALSE)", {
  skip_if_no_gpu()
  # Direct regression test for the %in% handler fix this feature depended on
  # (src/expr_eval.hpp): `x %in% NA` used to be wrongly routed into the
  # "empty RHS set" always-FALSE shortcut instead of matching x's NA rows.
  df <- oracle_case_match_df()
  pipeline <- function(d) dplyr::mutate(d, z = x %in% NA)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(result$gpu$z[3])
  expect_true(result$gpu$z[9])
  expect_false(result$gpu$z[1])
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .default = x itself (the "replace some values" idiom shown in dplyr's docs)
# =============================================================================

test_that("case_match() with .default = x (partial recode idiom) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(d, z = dplyr::case_match(s, "b" ~ "B", .default = s)))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Type strictness: LHS type must be compatible with x
# =============================================================================

test_that("case_match() errors on a STRING LHS against a numeric x, matching dplyr's own type-safety", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  gt <- tbl_gpu(df)

  expect_error(suppressWarnings(dplyr::mutate(df, z = dplyr::case_match(x, "a" ~ 1L))))
  expect_error(dplyr::mutate(gt, z = dplyr::case_match(x, "a" ~ 1L)), "case_match")
})

# =============================================================================
# Not-yet-supported named arguments fall back to a hard error (mutate() has
# no CPU fallback), matching the same scope-narrowing precedent as
# case_when()'s .ptype=/.size=/.unmatched=
# =============================================================================

test_that("case_match()'s .ptype= falls back (rather than erroring) and works end-to-end", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # Phase 11, task L1 (universal expression-level CPU fallback): `.ptype=`
  # is still not natively parsed (ir_parse_case_match() returns NULL for
  # it), but that no longer means "mutate() has no CPU fallback" -- it now
  # transparently falls back to real dplyr::mutate() and succeeds.
  df <- oracle_case_match_df()
  gt <- tbl_gpu(df)

  # case_match() itself is deprecated upstream (in favor of
  # recode_values()) and warns about that on every call, on TOP OF this
  # task's own "fell back to CPU evaluation" notification -- muffle just
  # the deprecation warning (matching this file's own suppressWarnings()
  # convention elsewhere) so only the fallback notification reaches
  # expect_warning() below.
  withCallingHandlers(
    expect_warning(
      result <- dplyr::mutate(gt, z = dplyr::case_match(x, 1 ~ 1L, .default = 0L, .ptype = integer())) |>
        collect(),
      "fell back to CPU evaluation"
    ),
    warning = function(w) {
      if (grepl("deprecated", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
  expect_equal(
    result$z,
    suppressWarnings(dplyr::case_match(df$x, 1 ~ 1L, .default = 0L, .ptype = integer()))
  )
})

# =============================================================================
# Chaining / usage inside filter()
# =============================================================================

test_that("case_match() chained with a later mutate dot referencing its output matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::mutate(
      d,
      a = dplyr::case_match(x, c(1, 3) ~ "small", .default = "big"),
      b = a == "small"
    ))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("case_match() works inside filter(), matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_case_match_df()
  pipeline <- function(d) {
    suppressWarnings(dplyr::filter(d, dplyr::case_match(x, c(1, 3) ~ TRUE, .default = FALSE)))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
