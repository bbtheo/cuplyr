# Mirrored dplyr-oracle tests for mutate() semantics (Phase 1, task T6)
#
# Every test in this file compares a `dplyr::mutate()` pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a `tbl_gpu`, in
# both eager and lazy exec modes, via `expect_same_as_dplyr()` /
# `expect_same_as_dplyr_lazy()` (see helper-oracle.R). The data below carries
# NAs across numeric/integer/logical/character columns to stress the same
# NA-propagation rules documented in scratchpad/phase1_expression_engine.md
# section 2.3 that test-dplyr-filter.R already exercises for filter().
#
# This file does NOT duplicate test-mutate.R's existing coverage (basic
# +-*/^, column-to-column ops, replacement/order, data residency, the T5
# cutover smoke tests for nesting/%%/sqrt/log/abs/scalar-on-left) -- see that
# file's "Expression-engine smoke tests" section for the handful of
# overlapping shapes it already pins.
#
# T6 divergences found + fixed (both in R/ir.R's ir_call_registry, no C++
# changes needed -- gpu_mutate_expr() already casts the computed column to
# the R-declared output_type, so correcting the *type* function alone fixes
# the observed value):
#
#   1. round(): registered as `type_arg1` (preserve the argument's type), on
#      the theory (stated in phase1_expression_engine.md section 1.3) that
#      "round(1L) is integer" in R. Empirically FALSE: `typeof(round(1L))`,
#      `typeof(round(TRUE))`, and `typeof(round(1L, 2))` are all "double" --
#      base R's round() always returns a double, regardless of input type.
#      cuplyr's round(int_col) was therefore staying INT32 while dplyr's
#      round(int_col) is FLOAT64. Fixed: "round"'s type function is now
#      `type_float64` unconditionally.
#   2. abs(): registered as `type_arg1`, correct for abs(-1L) (integer) and
#      abs(-1.5) (double), but WRONG for a logical column: `typeof(abs(TRUE))`
#      is "integer" in R (abs() promotes logical to integer first), while
#      cuplyr's abs(flag) was staying BOOL8. Fixed: a dedicated `type_abs()`
#      that maps BOOL8 -> INT32 and otherwise preserves the argument's type.
#
# Both were caught by the "type parity" tests below (BEFORE the fix, they
# failed with a column-type mismatch under ignore_col_types = FALSE); see
# git history for the failing-test-first commit sequence.

# =============================================================================
# Shared fixture
# =============================================================================

# Numeric/integer/logical/character columns, each carrying NAs.
oracle_mutate_df <- function() {
  data.frame(
    x = c(5, NA, 3, 8, -2, 1, 9, 4, 6, 2),
    y = c(2, 4, NA, 1, 6, 3, 20, 4, NA, 5),
    z = c(1, 10, 2, 3, 20, 5, NA, 8, 4, 7),
    s = c("banana", "apple", NA, "cherry", "date", NA, "fig", "grape", "honeydew", "kiwi"),
    i = c(1L, 2L, 3L, NA, 5L, 6L, 7L, NA, 9L, 10L),
    flag = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE, NA, FALSE, TRUE, FALSE),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Deep nesting / parenthesized groupings
# =============================================================================

test_that("mutate() deeply nested arithmetic matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = ((x + y) * z - x / 2) ^ 2)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() deeply nested arithmetic matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = ((x + y) * z - x / 2) ^ 2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() parenthesized groupings match dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = (x + (y - (z * 2))) / (x + 1))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() parenthesized groupings match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = (x + (y - (z * 2))) / (x + 1))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Unary math functions
# =============================================================================

test_that("mutate() sqrt/exp/floor/ceiling match dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) {
    dplyr::mutate(d, sq = sqrt(z), e = exp(z / 10), fl = floor(x), ce = ceiling(x))
  }
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() sqrt/exp/floor/ceiling match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) {
    dplyr::mutate(d, sq = sqrt(z), e = exp(z / 10), fl = floor(x), ce = ceiling(x))
  }
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() sin/cos/tan match dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, s = sin(x), c = cos(x), t = tan(x))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() sin/cos/tan match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, s = sin(x), c = cos(x), t = tan(x))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() log()/log2()/log10()/log(x,base) match dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 4, 8, 16, 100, NA))
  pipeline <- function(d) {
    dplyr::mutate(d, l = log(x), l2 = log2(x), l10 = log10(x), lb = log(x, 3))
  }
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() log()/log2()/log10()/log(x,base) match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 4, 8, 16, 100, NA))
  pipeline <- function(d) {
    dplyr::mutate(d, l = log(x), l2 = log2(x), l10 = log10(x), lb = log(x, 3))
  }
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() round(x) at .5 boundaries matches dplyr's banker's rounding (eager)", {
  skip_if_no_gpu()
  # Exact-in-binary values so both R and cudf's HALF_EVEN agree without
  # floating-point representation noise (scratchpad/phase1_expression_engine.md
  # section 0: round.hpp / cudf::round_decimal / cudf::round use HALF_EVEN,
  # matching R's banker's rounding).
  df <- data.frame(x = c(0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, NA))
  pipeline <- function(d) dplyr::mutate(d, r = round(x))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() round(x) at .5 boundaries matches dplyr's banker's rounding (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, NA))
  pipeline <- function(d) dplyr::mutate(d, r = round(x))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() round(x, 2) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.005, 2.675, 3.14159, -1.125, NA))
  pipeline <- function(d) dplyr::mutate(d, r = round(x, 2))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() round(x, 2) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.005, 2.675, 3.14159, -1.125, NA))
  pipeline <- function(d) dplyr::mutate(d, r = round(x, 2))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() round(int_col) promotes to double like dplyr (T6 divergence, fixed)", {
  skip_if_no_gpu()
  # T6 finding: ir_call_registry's "round" entry used type_arg1 (preserve
  # the argument's type), on the theory that round(1L) is integer in R --
  # empirically false (typeof(round(1L)) is "double"; base R's round()
  # always returns a double). Fixed in R/ir.R: "round" now always types as
  # FLOAT64, matching dplyr exactly (no ignore_col_types escape hatch).
  df <- data.frame(x = 1:10)
  pipeline <- function(d) dplyr::mutate(d, r = round(x))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() round(int_col) promotes to double like dplyr (T6 divergence, fixed, lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = 1:10)
  pipeline <- function(d) dplyr::mutate(d, r = round(x))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# %% and %/% (incl. negative operands, integer/double mixes)
# =============================================================================

test_that("mutate() %% with negative operands matches dplyr's sign semantics (eager)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(-7, 7, -7, 7, -10, 10), y = c(3, 3, -3, -3, 4, -4))
  pipeline <- function(d) dplyr::mutate(d, m = x %% y)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() %% with negative operands matches dplyr's sign semantics (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(-7, 7, -7, 7, -10, 10), y = c(3, 3, -3, -3, 4, -4))
  pipeline <- function(d) dplyr::mutate(d, m = x %% y)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() %/% with negative operands matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(-7, 7, -7, 7, -10, 10), y = c(3, 3, -3, -3, 4, -4))
  pipeline <- function(d) dplyr::mutate(d, fd = x %/% y)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() %/% with negative operands matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(-7, 7, -7, 7, -10, 10), y = c(3, 3, -3, -3, 4, -4))
  pipeline <- function(d) dplyr::mutate(d, fd = x %/% y)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() %% / %/% on integer/double mixes match dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(i = c(-7L, 7L, 10L, -10L))
  pipeline <- function(d) dplyr::mutate(d, m = i %% 3, fd = i %/% 3, m2 = i %% 3L, fd2 = i %/% 3L)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() %% / %/% on integer/double mixes match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(i = c(-7L, 7L, 10L, -10L))
  pipeline <- function(d) dplyr::mutate(d, m = i %% 3, fd = i %/% 3, m2 = i %% 3L, fd2 = i %/% 3L)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Unary minus
# =============================================================================

test_that("mutate() unary minus (-x) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = -x)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() unary minus (-x) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = -x)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() unary minus on a sub-expression (-(x+y)) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = -(x + y))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() unary minus on a sub-expression (-(x+y)) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = -(x + y))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() x * -2 matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = x * -2)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() x * -2 matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = x * -2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() unary minus preserves INT32 like dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10)
  pipeline <- function(d) dplyr::mutate(d, w = -i)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() unary minus preserves INT32 like dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10)
  pipeline <- function(d) dplyr::mutate(d, w = -i)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Scalar-on-left
# =============================================================================

test_that("mutate() scalar-on-left 10 - x matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = 10 - x)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() scalar-on-left 10 - x matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = 10 - x)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() scalar-on-left 2 / x matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = 2 / x)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() scalar-on-left 2 / x matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = 2 / x)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() scalar-on-left 3 ^ x matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = 3 ^ x)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() scalar-on-left 3 ^ x matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = 3 ^ x)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# NA literals
# =============================================================================

test_that("mutate(y = NA) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, y2 = NA)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate(y = NA) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, y2 = NA)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(y = NA_real_) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, y2 = NA_real_)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate(y = NA_real_) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, y2 = NA_real_)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(y = x + NA) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, y2 = x + NA)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate(y = x + NA) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, y2 = x + NA)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Type parity
# =============================================================================

test_that("mutate() integer preservation through + - * matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10, j = 10:1)
  pipeline <- function(d) dplyr::mutate(d, a = i + j, b = i - j, c = i * j)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() integer preservation through + - * matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10, j = 10:1)
  pipeline <- function(d) dplyr::mutate(d, a = i + j, b = i - j, c = i * j)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() promotion through / and ^ matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10, j = 10:1)
  pipeline <- function(d) dplyr::mutate(d, a = i / j, b = i ^ 2L)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() promotion through / and ^ matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10, j = 10:1)
  pipeline <- function(d) dplyr::mutate(d, a = i / j, b = i ^ 2L)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() logical arithmetic (flag + 1L) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(flag = c(TRUE, FALSE, TRUE, NA))
  pipeline <- function(d) dplyr::mutate(d, s = flag + 1L)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() logical arithmetic (flag + 1L) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(flag = c(TRUE, FALSE, TRUE, NA))
  pipeline <- function(d) dplyr::mutate(d, s = flag + 1L)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() abs(-int) stays integer like dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(i = c(-3L, 5L, -7L, 0L, NA))
  pipeline <- function(d) dplyr::mutate(d, a = abs(i))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() abs(-int) stays integer like dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(i = c(-3L, 5L, -7L, 0L, NA))
  pipeline <- function(d) dplyr::mutate(d, a = abs(i))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() abs(logical) promotes to integer like dplyr (T6 divergence, fixed, eager)", {
  skip_if_no_gpu()
  # T6 finding: ir_call_registry's "abs" entry used type_arg1 (preserve the
  # argument's type verbatim), correct for abs(-1L)/abs(-1.5) but wrong for a
  # logical column: typeof(abs(TRUE)) is "integer" in R, not "logical".
  # Fixed in R/ir.R with a dedicated type_abs() that special-cases BOOL8 ->
  # INT32 and otherwise preserves the argument's type.
  df <- data.frame(flag = c(TRUE, FALSE, NA, TRUE))
  pipeline <- function(d) dplyr::mutate(d, a = abs(flag))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() abs(logical) promotes to integer like dplyr (T6 divergence, fixed, lazy)", {
  skip_if_no_gpu()
  df <- data.frame(flag = c(TRUE, FALSE, NA, TRUE))
  pipeline <- function(d) dplyr::mutate(d, a = abs(flag))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() floor(int_col) promotes to double like dplyr (eager)", {
  skip_if_no_gpu()
  # Pinning the "floor/ceil output type" check the task calls out explicitly:
  # ir_call_registry already declares floor/ceil as FLOAT64 (matching R's
  # typeof(floor(1L)) == "double"), so this is a parity confirmation, not a
  # fix -- included so a future regression here is caught by the oracle.
  df <- data.frame(i = 1:10)
  pipeline <- function(d) dplyr::mutate(d, f = floor(i), c = ceiling(i))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() floor(int_col) promotes to double like dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(i = 1:10)
  pipeline <- function(d) dplyr::mutate(d, f = floor(i), c = ceiling(i))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() INT32 arithmetic overflow wraps, unlike dplyr's NA (documented divergence, not fixed)", {
  skip_if_no_gpu()
  # Not an oracle-comparison test: R promotes an overflowing integer
  # addition to NA with a warning ("NAs produced by integer overflow"),
  # while cuplyr's INT32 arithmetic is plain fixed-width C++ int32_t
  # addition, which silently wraps around -- the same divergence flagged in
  # R/mutate.R's roxygen ("this means INT32 arithmetic can silently wrap
  # around on overflow ... rather than promoting to a wider type"). Per the
  # T6 task brief this is a documented, not fixed, divergence -- pin the
  # actual GPU behavior directly instead of comparing to the (deliberately
  # different) dplyr oracle.
  #
  # Two sub-cases, because R's own integer type reserves the INT32_MIN bit
  # pattern (-2147483648) as its NA_INTEGER sentinel -- it isn't a
  # representable integer *value* in R at all (`-2147483647L - 1L` itself
  # overflows and warns). So:
  #  - .Machine$integer.max + 1L wraps to INT32_MIN at the bit level, which
  #    R's collect() then reads back as NA (a coincidental, silent collision
  #    with R's own sentinel, NOT cuplyr detecting the overflow).
  #  - .Machine$integer.max + 2L wraps to INT32_MIN + 1, which IS a normal,
  #    representable, non-NA R integer (-2147483647L) -- this is the case
  #    that unambiguously demonstrates real wraparound rather than an NA
  #    coincidence.
  df <- data.frame(i = c(.Machine$integer.max, .Machine$integer.max, 1L))
  gpu_df <- tbl_gpu(df)

  result1 <- dplyr::mutate(gpu_df, w = i + 1L) |> collect()
  expect_equal(result1$w[3], 2L)
  expect_true(is.na(result1$w[1]))  # INT32_MIN bit pattern == R's NA_INTEGER

  result2 <- dplyr::mutate(gpu_df, w = i + 2L) |> collect()
  expect_equal(result2$w[3], 3L)
  expect_equal(result2$w[1], -2147483647L)  # genuine wraparound, not NA

  # dplyr/R's own behavior for comparison: NA with a warning, no wrap at all.
  r_result <- suppressWarnings(dplyr::mutate(df, w = i + 1L))
  expect_true(is.na(r_result$w[1]))
  expect_warning(dplyr::mutate(df, w = i + 1L), "overflow")
})

# =============================================================================
# Multiple dots with cross-references
# =============================================================================

test_that("mutate() multiple dots with cross-references (a, b, c) match dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, a = x + 1, b = a * a, c = b - x)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() multiple dots with cross-references (a, b, c) match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, a = x + 1, b = a * a, c = b - x)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Replace-in-place chains (later dot sees the NEW value, not the original)
# =============================================================================

test_that("mutate() replace-in-place chain (x = x+1, y = x*2) matches dplyr (eager)", {
  skip_if_no_gpu()
  # dplyr semantics: `y = x * 2` sees the just-updated `x` (x+1), not the
  # original column -- this is the sequential-evaluation contract
  # gpu_mutate_expr()'s append-and-remap bookkeeping is built around
  # (scratchpad/phase1_expression_engine.md section 2.2).
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, x = x + 1, y = x * 2)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() replace-in-place chain (x = x+1, y = x*2) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, x = x + 1, y = x * 2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Env variables and .data pronoun
# =============================================================================

test_that("mutate() with an env-variable operand matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  bump <- 100
  pipeline <- function(d) dplyr::mutate(d, w = x + bump)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() with an env-variable operand matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  bump <- 100
  pipeline <- function(d) dplyr::mutate(d, w = x + bump)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .data pronoun matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = .data$x + .data$y)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() .data pronoun matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, w = .data$x + .data$y)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .env pronoun with a name colliding with a column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  x <- 1000  # deliberately collides with the column name "x"
  pipeline <- function(d) dplyr::mutate(d, w = x + .env$x)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() .env pronoun with a name colliding with a column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  x <- 1000
  pipeline <- function(d) dplyr::mutate(d, w = x + .env$x)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Comparisons / logicals as mutate outputs
# =============================================================================

test_that("mutate() comparison/logical outputs (big, both) match dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, big = x > 5, both = big & (y < 2))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() comparison/logical outputs (big, both) match dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, big = x > 5, both = big & (y < 2))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() is.na(x) as an output column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, x_missing = is.na(x))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() is.na(x) as an output column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, x_missing = is.na(x))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# %in% and between() as mutate outputs
# =============================================================================

test_that("mutate() %in% as an output column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, in_set = x %in% c(3, 4, 9))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() %in% as an output column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, in_set = x %in% c(3, 4, 9))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() %in% with NA in the set as an output column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, in_set = x %in% c(3, NA))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() %in% with NA in the set as an output column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, in_set = x %in% c(3, NA))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() between() as an output column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, mid = dplyr::between(x, 2, 6))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() between() as an output column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, mid = dplyr::between(x, 2, 6))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# String columns
# =============================================================================

test_that("mutate() string column copy matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, s2 = s)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() string column copy matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, s2 = s)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() string comparison output column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, eq = s == "fig")
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() string comparison output column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) dplyr::mutate(d, eq = s == "fig")
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Grouped table: mutate preserves groups metadata
# =============================================================================

test_that("mutate() on a grouped table preserves group_vars() like dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(flag) |> dplyr::mutate(w = x + y)
  }
  # expect_same_as_dplyr()/_lazy() already assert group_vars() equality
  # between the GPU and oracle results before collecting (helper-oracle.R).
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() on a grouped table preserves group_vars() like dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(flag) |> dplyr::mutate(w = x + y)
  }
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() on a table grouped by multiple columns preserves group_vars()", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(flag, i) |> dplyr::mutate(w = x * 2)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# 0-row table / all-NA column arithmetic
# =============================================================================

test_that("mutate() on a 0-row input matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()[0, ]
  pipeline <- function(d) dplyr::mutate(d, w = x + y, big = x > 5)
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() on a 0-row input matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- oracle_mutate_df()[0, ]
  pipeline <- function(d) dplyr::mutate(d, w = x + y, big = x > 5)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() all-NA column arithmetic matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4), all_na = rep(NA_real_, 4))
  pipeline <- function(d) dplyr::mutate(d, w = all_na + x, is_missing = is.na(all_na))
  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() all-NA column arithmetic matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4), all_na = rep(NA_real_, 4))
  pipeline <- function(d) dplyr::mutate(d, w = all_na + x, is_missing = is.na(all_na))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .keep / .before / .after control args (Phase 3, task 4)
#
# Semantics verified empirically against dplyr 1.2.1 before implementation
# (see R/mutate.R's roxygen "`.keep` semantics" section for the full write
# up): columns created or modified by `...` and every group column are
# ALWAYS kept, regardless of `.keep`; `.keep` only controls which *other*
# columns of `.data` additionally survive ("all" = all of them, "used" =
# referenced as an input to any expression, "unused" = the complement,
# "none" = none). Column order: original columns keep their original
# relative position (a modified column does NOT move); genuinely new
# columns are appended at the end, or repositioned via `.before`/`.after`
# (only the new ones move -- a modified EXISTING column stays put even
# when `.before`/`.after` is given).
# =============================================================================

keep_position_df <- function() {
  tibble::tibble(a = 1:4, b = 5:8, c = 9:12, d = 13:16)
}

test_that("mutate() .keep='all' (default) keeps every column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b", "c", "d", "e"))
})

test_that("mutate() .keep='used' keeps created/modified + input columns matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, .keep = "used")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b", "e"))
})

test_that("mutate() .keep='used' keeps created/modified + input columns matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, .keep = "used")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .keep='used' with a column that is BOTH used and modified matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  # b is used as an input AND replaced in place -- must appear exactly
  # once, not duplicated, and no new column is created.
  pipeline <- function(d) dplyr::mutate(d, b = a + b, .keep = "used")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b"))
})

test_that("mutate() .keep='used' with a column that is BOTH used and modified matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, b = a + b, .keep = "used")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .keep='unused' keeps created/modified + unreferenced columns matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, .keep = "unused")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("c", "d", "e"))
})

test_that("mutate() .keep='unused' keeps created/modified + unreferenced columns matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, .keep = "unused")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .keep='unused' with a column that is BOTH used and modified matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, b = a + b, .keep = "unused")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("b", "c", "d"))
})

test_that("mutate() .keep='none' keeps only created/modified columns matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, .keep = "none")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), "e")
})

test_that("mutate() .keep='none' keeps only created/modified columns matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, .keep = "none")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .keep='none' with a bare column dot keeps that column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, d, .keep = "none")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("d", "e"))
})

test_that("mutate() .keep='none' with a bare column dot keeps that column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + b, d, .keep = "none")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .keep='none' on a grouped table always retains group columns matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) d |> dplyr::group_by(a) |> dplyr::mutate(e = b + c, .keep = "none")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "e"))
})

test_that("mutate() .keep='none' on a grouped table always retains group columns matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) d |> dplyr::group_by(a) |> dplyr::mutate(e = b + c, .keep = "none")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .keep='unused' on a grouped table always retains group columns matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) d |> dplyr::group_by(a) |> dplyr::mutate(e = b + c, .keep = "unused")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "d", "e"))
})

test_that("mutate() .after positions a new column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, .after = b)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b", "e", "c", "d"))
})

test_that("mutate() .after positions a new column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, .after = b)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .before positions a new column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, .before = b)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "e", "b", "c", "d"))
})

test_that("mutate() .before positions a new column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, .before = b)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .after does NOT move a modified existing column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  # a is modified in place, not new -- .after must not relocate it.
  pipeline <- function(d) dplyr::mutate(d, a = a + 1, .after = c)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b", "c", "d"))
})

test_that("mutate() .after moves only NEW columns when mixed with a modification matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, b = b * 2, .after = c)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b", "c", "e", "d"))
})

test_that("mutate() .after with multiple new columns moves them together in dot order matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, f = b + 1, .after = a)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "e", "f", "b", "c", "d"))
})

test_that("mutate() .after with multiple new columns moves them together in dot order matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = a + 1, f = b + 1, .after = a)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() .after combined with .keep='used' filters the repositioned list matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  # e uses c; .after=a positions e right after a in the "keep=all" order
  # (a, e, b, c, d); .keep="used" then filters that list down to {c, e},
  # preserving their RELATIVE order from the positioned list -- i.e. "e"
  # before "c", not the reverse.
  pipeline <- function(d) dplyr::mutate(d, e = c + 1, .after = a, .keep = "used")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("e", "c"))
})

test_that("mutate() .after combined with .keep='used' filters the repositioned list matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) dplyr::mutate(d, e = c + 1, .after = a, .keep = "used")
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() on a grouped table with .after keeps the group column in place matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- keep_position_df()
  pipeline <- function(d) d |> dplyr::group_by(b) |> dplyr::mutate(e = a + 1, .after = a)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "e", "b", "c", "d"))
})

test_that("mutate() errors when both .before and .after are supplied", {
  skip_if_no_gpu()
  gt <- tbl_gpu(keep_position_df())
  expect_error(
    dplyr::mutate(gt, e = a + 1, .before = b, .after = c),
    "Can't supply both"
  )
})

test_that("mutate() .by= is GPU-native (Phase 5, task W6) -- no CPU fallback", {
  skip_if_no_gpu()
  df <- keep_position_df()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(df)

  result <- expect_no_error(dplyr::mutate(gt, e = a + b, .by = a))
  oracle <- dplyr::mutate(df, e = a + b, .by = a)
  expect_equal(tibble::as_tibble(collect(result)), tibble::as_tibble(oracle))
})
