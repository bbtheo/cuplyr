# Tests for the C++ expression evaluator (src/expr_eval.hpp, src/ops_expr.cpp)
#
# Task T2 of scratchpad/phase1_expression_engine.md: the evaluator is fully
# dormant here -- nothing in R/filter.R or R/mutate.R calls gpu_compute_column
# ()/gpu_filter_expr()/gpu_mutate_expr() yet (that's T3/T5). These tests call
# the three Rcpp exports directly on hand-built, already-bound IR (plain R
# lists constructed with ir_col()/ir_lit()/ir_call() + ir_bind(), all internal
# functions from R/ir.R, directly callable here since testthat runs inside the
# package namespace -- see test-ir.R for the same pattern).
#
# All require a GPU (skip_if_no_gpu()), unlike test-ir.R's pure-R IR tests.

# -----------------------------------------------------------------------------
# Local helpers
# -----------------------------------------------------------------------------

# Evaluate a bound (or bindable) IR node via gpu_compute_column() and pull the
# single resulting column back to R.
compute_ir <- function(tbl, ir) {
  bound <- ir_bind(ir, tbl$schema)
  ptr <- gpu_compute_column(tbl$ptr, bound)
  gpu_collect(ptr, "v")$v
}

# Same, but also returns the GPU column type (gpu_col_types()'s short code,
# e.g. "dbl"/"int"/"lgl") for tests that need to pin the *actual* output type.
compute_ir_typed <- function(tbl, ir) {
  bound <- ir_bind(ir, tbl$schema)
  ptr <- gpu_compute_column(tbl$ptr, bound)
  list(value = gpu_collect(ptr, "v")$v, type = gpu_col_types(ptr))
}

# Evaluate a bound (or bindable) IR node as a filter mask via
# gpu_filter_expr() and collect the filtered table back to R.
filter_ir <- function(tbl, ir) {
  bound <- ir_bind(ir, tbl$schema)
  ptr <- gpu_filter_expr(tbl$ptr, bound)
  gpu_collect(ptr, tbl$schema$names)
}

# gpu_mutate_expr()'s final-assembly column order (input names, with any
# replaced-in-place kept at their original position, then newly-appended
# names in expression order) -- mirrors the C++ final-assembly loop in
# ops_expr.cpp exactly, so tests can gpu_collect() with the right names.
final_mutate_names <- function(input_names, out_names) {
  c(input_names, out_names[!out_names %in% input_names])
}

# -----------------------------------------------------------------------------
# The six section 0 empirical checks
# -----------------------------------------------------------------------------
# All six ops here have no column reference (both operands are IR literals),
# which still routes through gpu_compute_column(): the expression is built
# and computed against `tbl`'s row count purely for broadcasting (matches
# dplyr's own "constant expression still yields one value per row").

test_that("[section 0, check 1] %% lowers to PYMOD (R's sign rule, not C's MOD)", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))

  ir <- ir_call("%%", list(ir_lit(-7, "FLOAT64"), ir_lit(3, "FLOAT64")))
  result <- compute_ir(tbl, ir)

  expect_equal(unique(result), -7 %% 3)  # R: 2 (a naive C `%` would give -1)
  expect_equal(unique(result), 2)
})

test_that("[section 0, check 2] %/% lowers to FLOOR_DIV (matches R's %/%)", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))

  ir <- ir_call("%/%", list(ir_lit(-7, "FLOAT64"), ir_lit(3, "FLOAT64")))
  result <- compute_ir(tbl, ir)

  expect_equal(unique(result), -7 %/% 3)  # R: -3
  expect_equal(unique(result), -3)
})

test_that("[section 0, check 3] POW on two INT32 operands promotes to FLOAT64, like R's 2L^2L", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))

  ir <- ir_call("^", list(ir_lit(2L, "INT32"), ir_lit(2L, "INT32")))
  out <- compute_ir_typed(tbl, ir)

  # R: 2L^2L is a *double* (4), never integer -- cudf's POW matches: verified
  # empirically that the AST op promotes INT32^INT32 to FLOAT64 ("dbl").
  expect_equal(out$type, "dbl")
  expect_equal(unique(out$value), 4)
  expect_identical(2L^2L, 4)  # pin R's own behavior alongside it
})

test_that("[section 0, check 4] bool + bool yields INT32, like R's TRUE+TRUE == 2L", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))

  ir <- ir_call("+", list(ir_lit(TRUE, "BOOL8"), ir_lit(TRUE, "BOOL8")))
  out <- compute_ir_typed(tbl, ir)

  expect_equal(out$type, "int")
  expect_equal(unique(out$value), 2)
  expect_identical(TRUE + TRUE, 2L)  # pin R's own behavior alongside it
})

test_that("[section 0, check 5] compute_column on a 0-row table returns a 0-row column", {
  skip_if_no_gpu()
  tbl0 <- tbl_gpu(data.frame(x = numeric(0)))

  ir <- ir_call("+", list(ir_col("x"), ir_lit(1, "FLOAT64")))
  result <- compute_ir(tbl0, ir)

  expect_length(result, 0)
})

test_that("[section 0, check 6] KNOWN DIVERGENCE: cuDF NaN is not null; R's is.na(NaN) is TRUE", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))

  # NaN as a *computed value* (0/0 via TRUE_DIV), not as an uploaded R NaN --
  # numeric_to_gpu() (transfer_io.cpp) already nulls out both NA and NaN on
  # ingestion (Rcpp::NumericVector::is_na() is true for NaN too, matching
  # R's own is.na() semantics), so a GPU-resident, *non-null* NaN can only be
  # produced by an on-GPU computation, never by uploading an R vector
  # containing NaN. Confirmed empirically: gpu_collect() on such an upload
  # already shows NA at that position (verified during T2 development).
  division_ir <- ir_call("/", list(ir_lit(0, "FLOAT64"), ir_lit(0, "FLOAT64")))
  division_result <- compute_ir(tbl, division_ir)
  expect_true(all(is.nan(division_result)))

  isna_ir <- ir_call("is.na", list(division_ir))
  isna_result <- compute_ir(tbl, isna_ir)

  # Pinned, per section 0: this is a KNOWN, DOCUMENTED, NOT-FIXED-IN-PHASE-1
  # divergence from R. is.na(NaN) is TRUE in R; IS_NULL(NaN) is FALSE in
  # cuDF, because NaN is a valid (non-null) floating-point value there.
  expect_true(is.na(NaN))              # R's actual behavior
  expect_false(unique(isna_result))    # cuDF's actual behavior (divergent)
})

# -----------------------------------------------------------------------------
# Null propagation (default nullable semantics, section 0's ast_operator
# ground truth): NA propagates through every binary op exactly like R.
# -----------------------------------------------------------------------------

test_that("null propagation: NA + 1 is NA, NA > 5 is NA (matches R)", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, NA, 3)))

  add_result <- compute_ir(tbl, ir_call("+", list(ir_col("x"), ir_lit(1, "FLOAT64"))))
  expect_equal(add_result, c(1, NA, 3) + 1)
  expect_true(is.na(add_result[2]))

  gt_result <- compute_ir(tbl, ir_call(">", list(ir_col("x"), ir_lit(0, "FLOAT64"))))
  expect_equal(gt_result, c(1, NA, 3) > 0)
  expect_true(is.na(gt_result[2]))
})

# -----------------------------------------------------------------------------
# Three-valued & and | (NULL_LOGICAL_AND / NULL_LOGICAL_OR): all 9 combos of
# {TRUE, FALSE, NA} x {TRUE, FALSE, NA}, checked against R's own operators.
# -----------------------------------------------------------------------------

test_that("& matches R's three-valued logic for all 9 TRUE/FALSE/NA combos", {
  skip_if_no_gpu()
  vals <- c(TRUE, FALSE, NA)
  a <- rep(vals, each = 3)
  b <- rep(vals, times = 3)
  tbl <- tbl_gpu(data.frame(a = a, b = b))

  result <- compute_ir(tbl, ir_call("&", list(ir_col("a"), ir_col("b"))))

  expect_equal(result, a & b)
})

test_that("| matches R's three-valued logic for all 9 TRUE/FALSE/NA combos", {
  skip_if_no_gpu()
  vals <- c(TRUE, FALSE, NA)
  a <- rep(vals, each = 3)
  b <- rep(vals, times = 3)
  tbl <- tbl_gpu(data.frame(a = a, b = b))

  result <- compute_ir(tbl, ir_call("|", list(ir_col("a"), ir_col("b"))))

  expect_equal(result, a | b)
})

test_that("!NA is NA (NOT's default null propagation matches R)", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(a = c(TRUE, FALSE, NA)))

  result <- compute_ir(tbl, ir_call("!", list(ir_col("a"))))

  expect_equal(result, !c(TRUE, FALSE, NA))
  expect_true(is.na(result[3]))
})

test_that("is.na() lowers to IS_NULL and matches R's is.na() for real NA (not NaN)", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, NA, 3)))

  result <- compute_ir(tbl, ir_call("is.na", list(ir_col("x"))))

  expect_equal(result, is.na(c(1, NA, 3)))
})

# -----------------------------------------------------------------------------
# Unary minus: SUB(typed-zero, x) -- must preserve INT32, not promote to
# FLOAT64 the way a naive floating-point zero would.
# -----------------------------------------------------------------------------

test_that("unary minus preserves INT32", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(i = c(1L, -2L, 3L)))

  out <- compute_ir_typed(tbl, ir_call("-", list(ir_col("i"))))

  expect_equal(out$type, "int")
  expect_equal(out$value, c(-1L, 2L, -3L))
})

# -----------------------------------------------------------------------------
# %in%'s never-NA rule (section 2.3's handler table): R's %in% never returns
# NA, unlike a naive contains()+null-propagation would.
# -----------------------------------------------------------------------------

test_that("%in% never returns NA: NA %in% c(1,2) is FALSE, NA %in% c(NA,1) is TRUE", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))
  na_lit <- ir_lit(NULL, NULL, na = TRUE)

  no_na_in_set <- compute_ir(tbl, ir_call("%in%", list(na_lit, ir_lit(c(1, 2), "FLOAT64"))))
  expect_true(all(!no_na_in_set))
  expect_identical(unique(no_na_in_set), FALSE)
  expect_identical(NA %in% c(1, 2), FALSE)  # pin R's own behavior alongside it

  na_in_set <- compute_ir(tbl, ir_call("%in%", list(na_lit, ir_lit(c(NA, 1), "FLOAT64"))))
  expect_true(all(na_in_set))
  expect_identical(unique(na_in_set), TRUE)
  expect_identical(NA %in% c(NA, 1), TRUE)  # pin R's own behavior alongside it
})

test_that("%in% works for a real column against a numeric set", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, 2, 3, 4)))

  result <- compute_ir(tbl, ir_call("%in%", list(ir_col("x"), ir_lit(c(2, 4), "FLOAT64"))))

  expect_equal(result, c(1, 2, 3, 4) %in% c(2, 4))
})

# -----------------------------------------------------------------------------
# String comparisons: non-AST handler (binary_operation + get_compare_op()),
# exercised end-to-end through gpu_filter_expr().
# -----------------------------------------------------------------------------

test_that("string == lowers to a filter mask via the non-AST handler", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(s = c("foo", "bar", "foo", "baz")))

  ir <- ir_call("==", list(ir_col("s"), ir_lit("foo", "STRING")))
  result <- filter_ir(tbl, ir)

  expect_equal(result$s, c("foo", "foo"))
})

# -----------------------------------------------------------------------------
# round(): banker's rounding via round_decimal()/round() (HALF_EVEN).
#
# HEADER-SIGNATURE SURPRISE (not anticipated by section 0): cudf::round_decimal
# ONLY accepts integral or fixed-point columns -- it throws "Only integral or
# fixed point currently supported by round_decimal" for FLOAT32/FLOAT64,
# despite its docstring examples showing floating-point input and despite
# section 0 treating it as a straightforward, non-deprecated replacement for
# the deprecated cudf::round(). Verified empirically during T2 development.
# apply_handler()'s "round" case therefore dispatches to the *deprecated*
# cudf::round() for floating-point columns (which does support HALF_EVEN on
# floats) and to round_decimal() for integral/fixed-point columns (where it's
# both correct and the intended, non-deprecated call). See src/expr_eval.hpp.
# -----------------------------------------------------------------------------

test_that("round() uses HALF_EVEN (banker's rounding): round(2.5)==2, round(3.5)==4", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(dummy = 1:3))

  r25 <- compute_ir(tbl, ir_call("round", list(ir_lit(2.5, "FLOAT64"))))
  r35 <- compute_ir(tbl, ir_call("round", list(ir_lit(3.5, "FLOAT64"))))

  expect_equal(unique(r25), 2)
  expect_equal(unique(r35), 4)
  expect_identical(round(2.5), 2)  # pin R's own (also HALF_EVEN) behavior
  expect_identical(round(3.5), 4)
})

test_that("round() works on a real FLOAT64 column with mixed .5 values", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(0.5, 1.5, 2.5, 3.5, -0.5)))

  result <- compute_ir(tbl, ir_call("round", list(ir_col("x"))))

  expect_equal(result, c(0, 2, 2, 4, 0))
})

# -----------------------------------------------------------------------------
# Nested arithmetic: fully AST-native, collapses to one compute_column() call
# ("one-kernel path").
# -----------------------------------------------------------------------------

test_that("nested arithmetic (x+y)*z computes correctly via the fused AST path", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, 2, 3), y = c(4, 5, 6), z = c(2, 2, 2)))

  ir <- ir_call("*", list(ir_call("+", list(ir_col("x"), ir_col("y"))), ir_col("z")))
  result <- compute_ir(tbl, ir)

  expect_equal(result, (c(1, 2, 3) + c(4, 5, 6)) * c(2, 2, 2))
})

# -----------------------------------------------------------------------------
# A non-AST node nested inside an AST tree: exercises the materialize-then-
# reference path (build_ast() hits a non-expressible %in% child, materializes
# it via apply_handler(), and references the result by column_reference from
# the enclosing `&` operation() -- both `&` and `>` still lower to ONE
# compute_column() call, with the %in% subtree as its one materialized input).
# -----------------------------------------------------------------------------

test_that("a handler op nested inside an AST tree materializes and is referenced correctly", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, 2, 3), y = c(4, -5, 6)))

  ir <- ir_call("&", list(
    ir_call("%in%", list(ir_col("x"), ir_lit(c(1, 2), "FLOAT64"))),
    ir_call(">", list(ir_col("y"), ir_lit(0, "FLOAT64")))
  ))
  result <- compute_ir(tbl, ir)

  expect_equal(result, (c(1, 2, 3) %in% c(1, 2)) & (c(4, -5, 6) > 0))
})

# -----------------------------------------------------------------------------
# gpu_filter_expr() end-to-end: NA rows are dropped (apply_boolean_mask()
# keeps a row iff the mask is non-null AND true).
# -----------------------------------------------------------------------------

test_that("gpu_filter_expr() drops NA mask rows, matching dplyr's filter() contract", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, NA, -1, 5)))

  ir <- ir_call(">", list(ir_col("x"), ir_lit(0, "FLOAT64")))
  result <- filter_ir(tbl, ir)

  # NA > 0 is NA (dropped), -1 > 0 is FALSE (dropped): only 1 and 5 survive.
  expect_equal(result$x, c(1, 5))
})

test_that("gpu_filter_expr() errors if the expression is not logical", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, 2, 3)))

  ir <- ir_call("+", list(ir_col("x"), ir_lit(1, "FLOAT64")))
  bound <- ir_bind(ir, tbl$schema)

  expect_error(gpu_filter_expr(tbl$ptr, bound), "filter\\(\\) expressions must be logical")
})

# -----------------------------------------------------------------------------
# gpu_mutate_expr(): sequential evaluation with append-and-remap, so a later
# expression sees an earlier expression's output (even when that output
# replaces an existing column, i.e. runs through the "replace-in-place, then
# append" final assembly lifted from ops_mutate_batch.cpp:141-202).
# -----------------------------------------------------------------------------

test_that("gpu_mutate_expr() lets a later expression refer to an earlier (replacing) output", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(1, 2, 3), y = c(10, 20, 30)))
  n_input <- length(tbl$schema$names)

  # expr 1: x <- x + 1 (replaces "x"; ir_bind() resolves "x" against the
  # ORIGINAL schema, since this is the first expression in the chain)
  expr1 <- ir_bind(ir_call("+", list(ir_col("x"), ir_lit(1, "FLOAT64"))), tbl$schema)

  # expr 2: y_new <- x * 2, referring to the NEW x (expr 1's output). Per
  # section 2.2's contract, gpu_mutate_expr() appends expr 1's result at ctx
  # index n_input (regardless of "x" being a replacement, not a new name) --
  # so a col node manually pointed at that index is exactly what a real
  # running-schema binder (T5) would have produced for "x" at this point in
  # the chain. Hand-built here since T5 (the R-side binder) doesn't exist yet.
  new_x_ref <- ir_col("x")
  new_x_ref$index <- n_input
  expr2 <- ir_call("*", list(new_x_ref, ir_lit(2, "FLOAT64")))

  out_names <- c("x", "y_new")
  out_types <- c("FLOAT64", "FLOAT64")
  ptr <- gpu_mutate_expr(tbl$ptr, list(expr1, expr2), out_names, out_types, tbl$schema$names)

  result <- gpu_collect(ptr, final_mutate_names(tbl$schema$names, out_names))

  expect_equal(result$x, c(2, 3, 4))       # x + 1
  expect_equal(result$y, c(10, 20, 30))    # untouched
  expect_equal(result$y_new, c(4, 6, 8))   # (x + 1) * 2, i.e. refers to the NEW x
  # Replaced columns keep their original position; new columns append at the
  # end -- matches dplyr's own mutate() column-order contract.
  expect_equal(names(result), c("x", "y", "y_new"))
})

test_that("gpu_mutate_expr() casts to the R-declared output type when it differs", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(x = c(2L, 4L, 6L)))

  # x / 2 is TRUE_DIV -> FLOAT64 on the GPU, but declare the output as INT32
  # (as e.g. integer-preserving mutate paths might for a divisible column)
  # to exercise the "R schema is authoritative, cast if it differs" step.
  expr <- ir_bind(ir_call("/", list(ir_col("x"), ir_lit(2, "FLOAT64"))), tbl$schema)

  ptr <- gpu_mutate_expr(tbl$ptr, list(expr), "halved", "INT32", tbl$schema$names)
  result <- gpu_collect(ptr, final_mutate_names(tbl$schema$names, "halved"))

  expect_equal(result$halved, c(1L, 2L, 3L))
  expect_type(result$halved, "integer")
})
