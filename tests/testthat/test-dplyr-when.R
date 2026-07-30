# Mirrored dplyr-oracle tests for when_all()/when_any() semantics
# (Phase 4, wave 2).
#
# Every test compares a dplyr-using pipeline run against a plain tibble (the
# oracle) with the same pipeline run against a `tbl_gpu`, in both eager and
# lazy exec modes, via `expect_same_as_dplyr()` / `expect_same_as_dplyr_lazy()`
# (see helper-oracle.R).
#
# Empirical traps verified against dplyr 1.2.1 BEFORE implementation (see
# R/ir.R's `ir_parse_when_reduce()`):
#
#   1. `when_any(x, y, z)` is `x | y | z`; `when_all(x, y, z)` is `x & y & z`
#      -- generalized elementwise reduction to N logical inputs, fully
#      desugared at parse time into existing `|`/`&` IR nodes (no new C++
#      handler).
#   2. `na_rm = FALSE` (the default) propagates NA exactly like plain `|`/`&`
#      three-valued logic (SQL-style: `TRUE | NA` is `TRUE`, `FALSE | NA` is
#      `NA`, `FALSE & NA` is `FALSE`, `TRUE & NA` is `NA`).
#   3. `na_rm = TRUE` does NOT just drop NAs from the *result* -- verified
#      empirically it treats NA as each operator's identity element instead:
#      `when_any(NA, NA, na_rm = TRUE)` is `FALSE` (not `NA`), `when_all(NA,
#      NA, na_rm = TRUE)` is `TRUE` (not `NA`). Implemented via
#      `coalesce()`-ing each input against that identity value before
#      folding.
#   4. `when_any()` is particularly useful inside `filter()`/`filter_out()`
#      to combine multiple comma-separated conditions with `|` instead of
#      the default `&`.

# =============================================================================
# Shared fixture
# =============================================================================

oracle_when_df <- function() {
  data.frame(
    a = c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, NA, NA, NA),
    b = c(TRUE, FALSE, NA, TRUE, FALSE, NA, TRUE, FALSE, NA),
    x = c(1L, 5L, 10L, 15L, 20L, 3L, 8L, 12L, 25L)
  )
}

# =============================================================================
# Two-input reduction: equivalent to plain | and &
# =============================================================================

test_that("when_any(a, b) matches a | b (and dplyr) exactly, including NA propagation", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::when_any(a, b))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$z, df$a | df$b)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("when_all(a, b) matches a & b (and dplyr) exactly, including NA propagation", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::when_all(a, b))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$z, df$a & df$b)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# N-ary reduction (3+ inputs), including nested expressions as inputs
# =============================================================================

test_that("when_any() with 3 expression inputs matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::when_any(x > 20, x < 5, a & b))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("when_all() with 3 expression inputs matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::when_all(x > 0, x < 30, a))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# na_rm = TRUE: NA treated as the operator's identity element, not dropped
# from the result
# =============================================================================

test_that("when_any(na_rm = TRUE) matches dplyr, including the all-NA row (FALSE, not NA)", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::when_any(a, b, na_rm = TRUE))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false(result$gpu$z[9])  # a=NA, b=NA -> FALSE under na_rm=TRUE
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("when_all(na_rm = TRUE) matches dplyr, including the all-NA row (TRUE, not NA)", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) dplyr::mutate(d, z = dplyr::when_all(a, b, na_rm = TRUE))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(result$gpu$z[9])  # a=NA, b=NA -> TRUE under na_rm=TRUE
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Usage inside filter() -- when_any()'s documented primary use case
# =============================================================================

test_that("when_any() inside filter() matches dplyr's equivalent OR-of-ANDs pipeline", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) {
    dplyr::filter(d, dplyr::when_any(x < 5, x > 20))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("when_all() inside filter() matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_when_df()
  pipeline <- function(d) {
    dplyr::filter(d, dplyr::when_all(x > 0, x < 20))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Not-yet-supported size= falls back to a hard mutate() error
# =============================================================================

test_that("when_any()'s size= falls back and still errors (real dplyr's own error, not cuplyr's)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # size= has no native parse support (ir_parse_when_reduce() returns NULL
  # for it) -- Phase 11, task L1 (universal expression-level CPU fallback)
  # means this is no longer mutate()'s own "only supports" hard error: it
  # now falls back to real dplyr::mutate(), which ALSO errors here (size=3
  # doesn't match this fixture's 9 rows) -- just with dplyr's own message,
  # not cuplyr's.
  df <- oracle_when_df()
  gt <- tbl_gpu(df)

  expect_warning(
    expect_error(
      dplyr::mutate(gt, z = dplyr::when_any(x > 1, size = 3)),
      "must have size 3"
    ),
    "fell back to CPU evaluation"
  )
})
