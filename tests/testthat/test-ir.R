# Tests for the expression IR (R/ir.R)
#
# Pure R: no skip_if_no_gpu(), no GPU calls anywhere in this file. The IR is
# fully dormant in Phase 1 task T1 -- nothing else in the package calls it
# yet -- so these tests exercise ir.R's functions directly.

quo_in <- function(expr, env = rlang::current_env()) {
  rlang::new_quosure(expr, env)
}

schema_xyz <- list(names = c("x", "y", "z"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))

# -----------------------------------------------------------------------------
# Node constructor shapes
# -----------------------------------------------------------------------------

test_that("ir_col() produces the documented shape", {
  node <- ir_col("mpg")

  expect_s3_class(node, "cuplyr_ir_col")
  expect_s3_class(node, "cuplyr_ir")
  expect_equal(node$kind, "col")
  expect_equal(node$name, "mpg")
})

test_that("ir_lit() produces the documented shape", {
  node <- ir_lit(5, "FLOAT64")

  expect_s3_class(node, "cuplyr_ir_lit")
  expect_s3_class(node, "cuplyr_ir")
  expect_equal(node$kind, "lit")
  expect_equal(node$value, 5)
  expect_equal(node$type, "FLOAT64")
  expect_false(node$na)
})

test_that("ir_call() produces the documented shape", {
  args <- list(ir_col("x"), ir_lit(1, "FLOAT64"))
  node <- ir_call("+", args)

  expect_s3_class(node, "cuplyr_ir_call")
  expect_s3_class(node, "cuplyr_ir")
  expect_equal(node$kind, "call")
  expect_equal(node$op, "+")
  expect_identical(node$args, args)
  expect_equal(node$meta, list())
})

# -----------------------------------------------------------------------------
# ir_lit_from_r()
# -----------------------------------------------------------------------------

test_that("ir_lit_from_r() types a scalar double as FLOAT64", {
  node <- ir_lit_from_r(3.5)
  expect_equal(node$type, "FLOAT64")
  expect_equal(node$value, 3.5)
  expect_false(node$na)
})

test_that("ir_lit_from_r() types a scalar integer as INT32", {
  node <- ir_lit_from_r(3L)
  expect_equal(node$type, "INT32")
  expect_equal(node$value, 3L)
})

test_that("ir_lit_from_r() types a scalar logical as BOOL8", {
  node <- ir_lit_from_r(TRUE)
  expect_equal(node$type, "BOOL8")
})

test_that("ir_lit_from_r() types a scalar character as STRING", {
  node <- ir_lit_from_r("a")
  expect_equal(node$type, "STRING")
})

test_that("ir_lit_from_r() marks a bare NA with na = TRUE and type = NULL", {
  node <- ir_lit_from_r(NA)
  expect_true(node$na)
  expect_null(node$type)
  expect_null(node$value)
})

test_that("ir_lit_from_r() rejects length > 1 unless allow_vector", {
  expect_error(ir_lit_from_r(c(1, 2, 3)), "length")
})

test_that("ir_lit_from_r() allows length > 1 when allow_vector = TRUE", {
  node <- ir_lit_from_r(c(4, 6), allow_vector = TRUE)
  expect_equal(node$value, c(4, 6))
  expect_equal(node$type, "FLOAT64")
})

test_that("ir_lit_from_r() rejects length 0 without allow_vector", {
  expect_error(ir_lit_from_r(numeric(0)), "length")
})

test_that("ir_lit_from_r() allows length 0 when allow_vector = TRUE (T4: %in% empty set)", {
  # `x %in% numeric(0)` is well-defined, ordinary R (always FALSE) -- not a
  # user error -- so the %in% RHS position must accept a length-0 vector
  # just like any other length != 1 vector. See src/expr_eval.hpp's %in%
  # handler for the "always FALSE" lowering this feeds.
  node <- ir_lit_from_r(numeric(0), allow_vector = TRUE)
  expect_equal(node$value, numeric(0))
  expect_equal(node$type, "FLOAT64")
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): symbols, columns-shadow-environment
# -----------------------------------------------------------------------------

test_that("bare column reference parses to ir_col", {
  schema <- list(names = "mpg", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(mpg)), schema)

  expect_equal(ir$kind, "col")
  expect_equal(ir$name, "mpg")
})

test_that("a symbol matching a column name is a column ref even if also bound in env", {
  env <- rlang::new_environment(list(x = 999))
  schema <- list(names = "x", types = "FLOAT64")

  ir <- ir_parse_quo(quo_in(quote(x), env), schema)

  expect_equal(ir$kind, "col")
  expect_equal(ir$name, "x")
})

test_that("a symbol not matching any column resolves from the environment", {
  env <- rlang::new_environment(list(threshold = 42))
  schema <- list(names = "mpg", types = "FLOAT64")

  ir <- ir_parse_quo(quo_in(quote(threshold), env), schema)

  expect_equal(ir$kind, "lit")
  expect_equal(ir$value, 42)
})

test_that("an unresolvable symbol re-raises dplyr's 'object not found' shape", {
  env <- rlang::new_environment()
  schema <- list(names = "mpg", types = "FLOAT64")

  expect_error(
    ir_parse_quo(quo_in(quote(totally_undefined_symbol), env), schema),
    "object 'totally_undefined_symbol' not found"
  )
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): .data / .env pronouns
# -----------------------------------------------------------------------------

test_that(".data[[\"x\"]] forces a column reference", {
  schema <- list(names = c("mpg", "cyl"), types = c("FLOAT64", "INT32"))
  ir <- ir_parse_quo(quo_in(quote(.data[["mpg"]])), schema)

  expect_equal(ir$kind, "col")
  expect_equal(ir$name, "mpg")
})

test_that(".data$x forces a column reference", {
  schema <- list(names = c("mpg", "cyl"), types = c("FLOAT64", "INT32"))
  ir <- ir_parse_quo(quo_in(quote(.data$mpg)), schema)

  expect_equal(ir$kind, "col")
  expect_equal(ir$name, "mpg")
})

test_that(".data[[\"x\"]] on a missing column errors with the pinned message", {
  schema <- list(names = c("mpg", "cyl"), types = c("FLOAT64", "INT32"))

  expect_error(
    ir_parse_quo(quo_in(quote(.data[["nonexistent"]])), schema),
    "Column 'nonexistent' not found\\.\nAvailable columns: mpg, cyl"
  )
})

test_that(".env$x forces an environment lookup even when a same-named column exists", {
  env <- rlang::new_environment(list(x = 123))
  schema <- list(names = "x", types = "FLOAT64")

  ir <- ir_parse_quo(quo_in(quote(.env$x), env), schema)

  expect_equal(ir$kind, "lit")
  expect_equal(ir$value, 123)
})

test_that(".env[[\"x\"]] forces an environment lookup", {
  env <- rlang::new_environment(list(x = 7))
  schema <- list(names = "x", types = "FLOAT64")

  ir <- ir_parse_quo(quo_in(quote(.env[["x"]]), env), schema)

  expect_equal(ir$kind, "lit")
  expect_equal(ir$value, 7)
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): literals
# -----------------------------------------------------------------------------

test_that("a bare NA literal parses with na = TRUE", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(NA)), schema)

  expect_equal(ir$kind, "lit")
  expect_true(ir$na)
  expect_null(ir$type)
})

test_that("a negative number literal parses as unary minus of the positive literal", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(-3)), schema)

  expect_equal(ir$kind, "call")
  expect_equal(ir$op, "-")
  expect_length(ir$args, 1)
  expect_equal(ir$args[[1]]$kind, "lit")
  expect_equal(ir$args[[1]]$value, 3)
})

test_that("a plain numeric/character/logical literal round-trips", {
  schema <- list(names = "x", types = "FLOAT64")

  expect_equal(ir_parse_quo(quo_in(quote(3.5)), schema)$value, 3.5)
  expect_equal(ir_parse_quo(quo_in(quote("a")), schema)$value, "a")
  expect_equal(ir_parse_quo(quo_in(quote(TRUE)), schema)$value, TRUE)
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): nested arithmetic
# -----------------------------------------------------------------------------

test_that("nested arithmetic ((x+y)*z - 3) parses into a matching IR tree", {
  ir <- ir_parse_quo(quo_in(quote((x + y) * z - 3)), schema_xyz)

  expect_equal(ir$op, "-")
  expect_equal(ir$args[[2]]$value, 3)

  mul_node <- ir$args[[1]]
  expect_equal(mul_node$op, "*")
  expect_equal(mul_node$args[[2]]$name, "z")

  add_node <- mul_node$args[[1]]
  expect_equal(add_node$op, "+")
  expect_equal(add_node$args[[1]]$name, "x")
  expect_equal(add_node$args[[2]]$name, "y")
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): comparisons and logic ops
# -----------------------------------------------------------------------------

test_that("all six comparison operators parse to matching ir_call nodes", {
  ops <- c("==", "!=", "<", "<=", ">", ">=")
  schema <- list(names = c("x", "y"), types = c("FLOAT64", "FLOAT64"))

  for (op in ops) {
    expr <- rlang::call2(op, quote(x), quote(y))
    ir <- ir_parse_quo(quo_in(expr), schema)

    expect_equal(ir$kind, "call", info = op)
    expect_equal(ir$op, op, info = op)
    expect_equal(ir$args[[1]]$name, "x", info = op)
    expect_equal(ir$args[[2]]$name, "y", info = op)
  }
})

test_that("& and | parse to ir_call nodes", {
  schema <- list(names = c("x", "y"), types = c("BOOL8", "BOOL8"))

  and_ir <- ir_parse_quo(quo_in(quote(x & y)), schema)
  expect_equal(and_ir$op, "&")

  or_ir <- ir_parse_quo(quo_in(quote(x | y)), schema)
  expect_equal(or_ir$op, "|")
})

test_that("! parses to a unary ir_call node", {
  schema <- list(names = "x", types = "BOOL8")
  ir <- ir_parse_quo(quo_in(quote(!x)), schema)

  expect_equal(ir$op, "!")
  expect_length(ir$args, 1)
  expect_equal(ir$args[[1]]$name, "x")
})

test_that("xor() parses to an ir_call node", {
  schema <- list(names = c("x", "y"), types = c("BOOL8", "BOOL8"))
  ir <- ir_parse_quo(quo_in(quote(xor(x, y))), schema)

  expect_equal(ir$op, "xor")
  expect_equal(ir$args[[1]]$name, "x")
  expect_equal(ir$args[[2]]$name, "y")
})

test_that("is.na(x) parses to a unary ir_call node", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(is.na(x))), schema)

  expect_equal(ir$op, "is.na")
  expect_length(ir$args, 1)
  expect_equal(ir$args[[1]]$name, "x")
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): between() desugar
# -----------------------------------------------------------------------------

test_that("between(x, l, r) desugars to (x >= l) & (x <= r)", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(between(x, 1, 5))), schema)

  expect_equal(ir$op, "&")

  ge_node <- ir$args[[1]]
  expect_equal(ge_node$op, ">=")
  expect_equal(ge_node$args[[1]]$name, "x")
  expect_equal(ge_node$args[[2]]$value, 1)

  le_node <- ir$args[[2]]
  expect_equal(le_node$op, "<=")
  expect_equal(le_node$args[[1]]$name, "x")
  expect_equal(le_node$args[[2]]$value, 5)
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): %in%
# -----------------------------------------------------------------------------

test_that("x %in% c(4, 6) constant-folds the RHS into a set literal", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(x %in% c(4, 6))), schema)

  expect_equal(ir$op, "%in%")
  expect_equal(ir$args[[1]]$name, "x")
  expect_equal(ir$args[[2]]$kind, "lit")
  expect_equal(ir$args[[2]]$value, c(4, 6))
})

test_that("a length > 1 literal outside %in%'s RHS errors mentioning length", {
  schema <- list(names = "x", types = "FLOAT64")

  expect_error(
    ir_parse_quo(quo_in(quote(x + c(1, 2, 3))), schema),
    "length"
  )
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): namespace stripping
# -----------------------------------------------------------------------------

test_that("dplyr::between() strips the namespace and desugars identically", {
  schema <- list(names = "x", types = "FLOAT64")

  plain <- ir_parse_quo(quo_in(quote(between(x, 1, 5))), schema)
  namespaced <- ir_parse_quo(quo_in(quote(dplyr::between(x, 1, 5))), schema)

  expect_equal(namespaced, plain)
})

test_that("base::sqrt() strips the namespace", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(base::sqrt(x))), schema)

  expect_equal(ir$op, "sqrt")
  expect_equal(ir$args[[1]]$name, "x")
})

# -----------------------------------------------------------------------------
# ir_parse_quo(): unsupported calls, constant folding, && error
# -----------------------------------------------------------------------------

test_that("an unregistered call referencing a column returns NULL (fallback)", {
  schema <- list(names = "x", types = "STRING")
  ir <- ir_parse_quo(quo_in(quote(paste0(x, "y"))), schema)

  expect_null(ir)
})

test_that("a column-free call constant-folds via eval_tidy", {
  # prod() (unlike max()/min()/sum()/mean(), Phase 5 window-class registry
  # entries as of this task) is still genuinely unregistered, so this
  # exercises the "call not in ir_call_registry at all" constant-fold
  # branch, not a registered function's own arg-parsing path.
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(prod(c(1, 2)))), schema)

  expect_equal(ir$kind, "lit")
  expect_equal(ir$value, 2)
})

test_that("&& errors with the design's pinned message", {
  schema <- list(names = c("x", "y"), types = c("BOOL8", "BOOL8"))

  expect_error(
    ir_parse_quo(quo_in(quote(x && y)), schema),
    "'&&' is not supported in filter\\(\\); use '&'"
  )
})

test_that("|| errors analogously to &&", {
  schema <- list(names = c("x", "y"), types = c("BOOL8", "BOOL8"))

  expect_error(
    ir_parse_quo(quo_in(quote(x || y)), schema),
    "'\\|\\|' is not supported in filter\\(\\); use '\\|'"
  )
})

# -----------------------------------------------------------------------------
# ir_cols()
# -----------------------------------------------------------------------------

test_that("ir_cols() collects unique columns from a deep tree", {
  ir <- ir_parse_quo(quo_in(quote((x + y) * z - x)), schema_xyz)

  expect_equal(sort(ir_cols(ir)), c("x", "y", "z"))
})

test_that("ir_cols() is empty for a literal-only subtree", {
  ir <- ir_call("+", list(ir_lit_from_r(1), ir_lit_from_r(2)))
  expect_equal(ir_cols(ir), character())
})

# -----------------------------------------------------------------------------
# ir_infer_type()
# -----------------------------------------------------------------------------

test_that("ir_infer_type() delegates + on two INT32 columns to infer_mutate_output_type", {
  schema <- list(names = c("a", "b"), types = c("INT32", "INT32"))
  ir <- ir_call("+", list(ir_col("a"), ir_col("b")))

  expect_equal(ir_infer_type(ir, schema), "INT32")
})

test_that("ir_infer_type() delegates ^ to FLOAT64 promotion (D5 rule 4)", {
  schema <- list(names = c("a", "b"), types = c("INT32", "INT32"))
  ir <- ir_call("^", list(ir_col("a"), ir_col("b")))

  expect_equal(ir_infer_type(ir, schema), "FLOAT64")
})

test_that("ir_infer_type() re-raises infer_mutate_output_type's STRING error verbatim", {
  schema <- list(names = "s", types = "STRING")
  ir <- ir_call("+", list(ir_col("s"), ir_lit_from_r(1)))

  expect_error(
    ir_infer_type(ir, schema),
    "mutate\\(\\) does not support arithmetic on STRING columns"
  )
})

test_that("ir_infer_type() is BOOL8 for comparisons", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_call(">", list(ir_col("x"), ir_lit_from_r(5)))

  expect_equal(ir_infer_type(ir, schema), "BOOL8")
})

test_that("ir_infer_type() is FLOAT64 for sqrt() even on an INT32 column", {
  schema <- list(names = "n", types = "INT32")
  ir <- ir_call("sqrt", list(ir_col("n")))

  expect_equal(ir_infer_type(ir, schema), "FLOAT64")
})

test_that("ir_infer_type() preserves INT32 for abs()", {
  schema <- list(names = "n", types = "INT32")
  ir <- ir_call("abs", list(ir_col("n")))

  expect_equal(ir_infer_type(ir, schema), "INT32")
})

test_that("ir_infer_type() preserves argument type for unary minus", {
  schema <- list(names = "n", types = "INT32")
  ir <- ir_call("-", list(ir_col("n")))

  expect_equal(ir_infer_type(ir, schema), "INT32")
})

test_that("ir_infer_type() defaults a bare NA literal to BOOL8 in isolation", {
  # T6 finding (phase1_expression_engine.md task T6, dplyr-oracle mutate
  # coverage): this was pinned at FLOAT64 during T1, on the theory that an
  # isolated bare NA is always asked as part of a call's arg_types where
  # FLOAT64 is a promotion-union no-op. Empirically wrong for the OTHER
  # place this default is reached: `mutate(y = NA)`, a bare NA as an entire
  # top-level expression with no enclosing call at all -- `typeof(NA)` is
  # "logical" in R, and dplyr's `mutate(y = NA)` produces a logical column.
  # Corrected to BOOL8; the promotion-union case is now handled separately
  # by ir_infer_type()'s "call" branch, which substitutes a sibling arg's
  # type for an isolated NA before it ever reaches a `type` function.
  ir <- ir_lit_from_r(NA)
  expect_equal(ir_infer_type(ir, list(names = character(), types = character())), "BOOL8")
})

# -----------------------------------------------------------------------------
# ir_cost()
# -----------------------------------------------------------------------------

test_that("ir_cost() orders scalar compare < col-col compare < nested", {
  scalar_compare <- ir_call(">", list(ir_col("x"), ir_lit_from_r(5)))
  colcol_compare <- ir_call(">", list(ir_col("x"), ir_col("y")))
  nested <- ir_parse_quo(quo_in(quote((x + y) * z - 3)), schema_xyz)

  expect_lt(ir_cost(scalar_compare), ir_cost(colcol_compare))
  expect_lt(ir_cost(colcol_compare), ir_cost(nested))
})

test_that("ir_cost() base cases: literal = 1, column = 2", {
  expect_equal(ir_cost(ir_lit_from_r(1)), 1L)
  expect_equal(ir_cost(ir_col("x")), 2L)
})

# -----------------------------------------------------------------------------
# ir_is_const()
# -----------------------------------------------------------------------------

test_that("ir_is_const() is TRUE for literal-only trees, FALSE otherwise", {
  const_ir <- ir_call("+", list(ir_lit_from_r(1), ir_lit_from_r(2)))
  col_ir <- ir_call("+", list(ir_col("x"), ir_lit_from_r(2)))

  expect_true(ir_is_const(const_ir))
  expect_false(ir_is_const(col_ir))
  expect_false(ir_is_const(ir_col("x")))
  expect_true(ir_is_const(ir_lit_from_r(1)))
})

# -----------------------------------------------------------------------------
# ir_bind()
# -----------------------------------------------------------------------------

test_that("ir_bind() fills 0-based indices on every col node", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  ir <- ir_call("+", list(ir_col("b"), ir_col("c")))

  bound <- ir_bind(ir, schema)

  expect_equal(bound$args[[1]]$index, 1L)
  expect_equal(bound$args[[2]]$index, 2L)
})

test_that("ir_bind() errors on a column missing from the binding schema", {
  schema <- list(names = "a", types = "FLOAT64")
  ir <- ir_col("nonexistent")

  expect_error(ir_bind(ir, schema), "Column 'nonexistent' not found")
})

test_that("ir_bind() leaves literal nodes untouched", {
  schema <- list(names = "a", types = "FLOAT64")
  ir <- ir_lit_from_r(5)

  bound <- ir_bind(ir, schema)
  expect_equal(bound, ir)
})

# -----------------------------------------------------------------------------
# ir_deparse()
# -----------------------------------------------------------------------------

test_that("ir_deparse() renders a readable expression", {
  ir <- ir_call("+", list(ir_col("x"), ir_lit_from_r(5)))
  text <- ir_deparse(ir)

  expect_true(is.character(text))
  expect_match(text, "x")
  expect_match(text, "5")
  expect_match(text, "\\+")
})

test_that("ir_deparse() renders NA literals as 'NA'", {
  expect_equal(ir_deparse(ir_lit_from_r(NA)), "NA")
})

# -----------------------------------------------------------------------------
# Phase 4: if_else()/case_when()/coalesce()/na_if() -- parse shapes and type
# inference, pure R (no GPU). See test-dplyr-if-else.R/test-dplyr-case-when.R/
# test-dplyr-coalesce.R/test-dplyr-na-if.R for the GPU oracle-parity coverage
# of these same ops' runtime values.
# -----------------------------------------------------------------------------

test_that("if_else() parses to an ir_call with cond/yes/no", {
  ir <- ir_parse_quo(quo_in(quote(if_else(x > 1, x, y))), schema_xyz)

  expect_equal(ir$kind, "call")
  expect_equal(ir$op, "if_else")
  expect_length(ir$args, 3)
  expect_equal(ir$args[[1]]$op, ">")
  expect_equal(ir$args[[2]]$name, "x")
  expect_equal(ir$args[[3]]$name, "y")
})

test_that("if_else() with missing= parses to a 4-arg ir_call", {
  ir <- ir_parse_quo(quo_in(quote(if_else(x > 1, x, y, missing = 0))), schema_xyz)
  expect_length(ir$args, 4)
})

test_that("if_else() rejects the wrong arg count (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(if_else(x > 1, x))), schema_xyz)
  expect_null(ir)
})

test_that("if_else() unifies int/double value types to FLOAT64, matching dplyr", {
  schema <- list(names = "x", types = "BOOL8")
  ir <- ir_call("if_else", list(ir_col("x"), ir_lit_from_r(1L), ir_lit_from_r(2.5)))
  expect_equal(ir_infer_type(ir, schema), "FLOAT64")
})

test_that("if_else() preserves INT32 when both value args are integer", {
  schema <- list(names = "x", types = "BOOL8")
  ir <- ir_call("if_else", list(ir_col("x"), ir_lit_from_r(1L), ir_lit_from_r(2L)))
  expect_equal(ir_infer_type(ir, schema), "INT32")
})

test_that("if_else() errors combining STRING and FLOAT64 value args", {
  schema <- list(names = "x", types = "BOOL8")
  ir <- ir_call("if_else", list(ir_col("x"), ir_lit_from_r("a"), ir_lit_from_r(1)))
  expect_error(ir_infer_type(ir, schema), "if_else")
})

test_that("if_else() adopts a bare NA value arg's type from its sibling at parse time", {
  ir <- ir_parse_quo(quo_in(quote(if_else(x > 1, x, NA))), schema_xyz)
  # ir_resolve_value_nas() should have already typed the NA literal (the
  # 3rd arg, `no`) as FLOAT64 (x's type in schema_xyz) -- so a later
  # ir_infer_type() walk sees a normal typed literal, not an isolated NA.
  expect_equal(ir$args[[3]]$type, "FLOAT64")
  expect_true(ir$args[[3]]$na)
})

test_that("case_when() parses formula dots into an interleaved cond/val ir_call", {
  ir <- ir_parse_quo(quo_in(quote(case_when(x < 2 ~ "a", x < 4 ~ "b"))), schema_xyz)

  expect_equal(ir$kind, "call")
  expect_equal(ir$op, "case_when")
  expect_equal(ir$meta$n_when, 2)
  # args: cond1, val1, cond2, val2, default (synthesized NA)
  expect_length(ir$args, 5)
  expect_equal(ir$args[[1]]$op, "<")
  expect_equal(ir$args[[2]]$value, "a")
  expect_equal(ir$args[[3]]$op, "<")
  expect_equal(ir$args[[4]]$value, "b")
  expect_true(ir$args[[5]]$na)  # no .default= supplied: synthesized NA
})

test_that("case_when() with .default= uses it as the trailing default arg", {
  ir <- ir_parse_quo(quo_in(quote(case_when(x < 2 ~ "a", .default = "z"))), schema_xyz)
  expect_length(ir$args, 3)
  expect_equal(ir$args[[3]]$value, "z")
})

test_that("case_when() falls back (NULL) for an unsupported named arg like .ptype=", {
  ir <- ir_parse_quo(quo_in(quote(case_when(x < 2 ~ 1, .ptype = double()))), schema_xyz)
  expect_null(ir)
})

test_that("case_when() falls back (NULL) for a one-sided formula", {
  ir <- ir_parse_quo(quo_in(quote(case_when(~"a"))), schema_xyz)
  expect_null(ir)
})

test_that("case_when() type unification promotes int/double RHS mix to FLOAT64", {
  schema <- list(names = "x", types = "BOOL8")
  ir <- ir_call(
    "case_when",
    list(ir_col("x"), ir_lit_from_r(1L), ir_lit_from_r(2.5)),
    meta = list(n_when = 1)
  )
  expect_equal(ir_infer_type(ir, schema), "FLOAT64")
})

test_that("case_when() errors combining STRING and numeric RHS values", {
  schema <- list(names = "x", types = "BOOL8")
  ir <- ir_call(
    "case_when",
    list(ir_col("x"), ir_lit_from_r("a"), ir_lit_from_r(1)),
    meta = list(n_when = 1)
  )
  expect_error(ir_infer_type(ir, schema), "case_when")
})

test_that("coalesce() parses a variadic arg list", {
  ir <- ir_parse_quo(quo_in(quote(coalesce(x, y, 0))), schema_xyz)
  expect_equal(ir$op, "coalesce")
  expect_length(ir$args, 3)
})

test_that("coalesce() requires at least one argument", {
  ir <- ir_parse_quo(quo_in(quote(coalesce())), schema_xyz)
  expect_null(ir)
})

test_that("coalesce() type unification promotes int/double mix to FLOAT64", {
  schema <- list(names = character(), types = character())
  ir <- ir_call("coalesce", list(ir_lit_from_r(1L), ir_lit_from_r(2.5)))
  expect_equal(ir_infer_type(ir, schema), "FLOAT64")
})

test_that("coalesce() errors combining STRING and numeric args", {
  schema <- list(names = character(), types = character())
  ir <- ir_call("coalesce", list(ir_lit_from_r("a"), ir_lit_from_r(1)))
  expect_error(ir_infer_type(ir, schema), "coalesce")
})

test_that("na_if() parses a 2-arg column/scalar call", {
  ir <- ir_parse_quo(quo_in(quote(na_if(x, 2))), schema_xyz)
  expect_equal(ir$op, "na_if")
  expect_length(ir$args, 2)
})

test_that("na_if() rejects the wrong arg count (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(na_if(x))), schema_xyz)
  expect_null(ir)
})

test_that("na_if() output type is always x's own type, never promoted by y", {
  schema <- list(names = "i", types = "INT32")
  ir <- ir_call("na_if", list(ir_col("i"), ir_lit_from_r(2.5)))
  expect_equal(ir_infer_type(ir, schema), "INT32")
})

test_that("na_if() errors at parse time on a STRING y against a numeric x", {
  schema <- list(names = c("x", "s"), types = c("FLOAT64", "STRING"))
  expect_error(
    ir_parse_quo(quo_in(quote(na_if(x, s))), schema),
    "na_if"
  )
})

test_that("na_if() errors at parse time on a typed NA_character_ y against a numeric x", {
  schema <- list(names = "x", types = "FLOAT64")
  expect_error(
    ir_parse_quo(quo_in(quote(na_if(x, NA_character_))), schema),
    "na_if"
  )
})

test_that("na_if() accepts a bare untyped NA for y without error", {
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(na_if(x, NA))), schema)
  expect_equal(ir$op, "na_if")
  expect_equal(ir$args[[2]]$type, "FLOAT64")
})

# -----------------------------------------------------------------------------
# Phase 4 wave 2: near()/case_match()/recode_values()/replace_values()/
# replace_when()/when_all()/when_any() -- parse shapes and type inference,
# pure R (no GPU). See test-dplyr-near.R/test-dplyr-case-match.R/
# test-dplyr-recode-values.R/test-dplyr-when.R for the GPU oracle-parity
# coverage of these same ops' runtime values.
# -----------------------------------------------------------------------------

test_that("near() desugars to abs(x - y) < tol at parse time", {
  ir <- ir_parse_quo(quo_in(quote(near(x, y))), schema_xyz)
  expect_equal(ir$op, "<")
  expect_equal(ir$args[[1]]$op, "abs")
  expect_equal(ir$args[[1]]$args[[1]]$op, "-")
  expect_true(ir$args[[2]]$kind == "lit")
  expect_equal(ir$args[[2]]$value, sqrt(.Machine$double.eps))
})

test_that("near() with an explicit tol uses it instead of the default", {
  ir <- ir_parse_quo(quo_in(quote(near(x, y, tol = 0.5))), schema_xyz)
  expect_equal(ir$args[[2]]$value, 0.5)
})

test_that("near() rejects the wrong arg count (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(near(x))), schema_xyz)
  expect_null(ir)
})

test_that("case_match() desugars to a case_when node with %in% conditions", {
  ir <- ir_parse_quo(quo_in(quote(case_match(x, c(1, 3) ~ "a", .default = "z"))), schema_xyz)
  expect_equal(ir$op, "case_when")
  expect_equal(ir$meta$n_when, 1)
  expect_equal(ir$args[[1]]$op, "%in%")
  expect_equal(ir$args[[1]]$args[[1]]$name, "x")
  expect_equal(ir$args[[1]]$args[[2]]$value, c(1, 3))
  expect_equal(ir$args[[2]]$value, "a")
  expect_equal(ir$args[[3]]$value, "z")
})

test_that("case_match() with a column-referencing LHS is unsupported (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(case_match(x, y ~ "a"))), schema_xyz)
  expect_null(ir)
})

test_that("case_match() with an unsupported named arg (.ptype=) falls back", {
  ir <- ir_parse_quo(quo_in(quote(case_match(x, 1 ~ "a", .ptype = character()))), schema_xyz)
  expect_null(ir)
})

test_that("case_match() errors at parse time on a STRING LHS against a numeric x", {
  expect_error(
    ir_parse_quo(quo_in(quote(case_match(x, "a" ~ 1))), schema_xyz),
    "case_match"
  )
})

test_that("recode_values() desugars identically to case_match(), with default= instead of .default=", {
  ir <- ir_parse_quo(quo_in(quote(recode_values(x, c(1, 3) ~ "a", default = "z"))), schema_xyz)
  expect_equal(ir$op, "case_when")
  expect_equal(ir$args[[2]]$value, "a")
  expect_equal(ir$args[[3]]$value, "z")
})

test_that("recode_values() with an unsupported named arg (unmatched=) falls back", {
  ir <- ir_parse_quo(quo_in(quote(recode_values(x, 1 ~ "a", unmatched = "error"))), schema_xyz)
  expect_null(ir)
})

test_that("replace_values() desugars to a replace_when node with %in% conditions and x as arg[1]", {
  ir <- ir_parse_quo(quo_in(quote(replace_values(x, c(1, 3) ~ 0))), schema_xyz)
  expect_equal(ir$op, "replace_when")
  expect_equal(ir$args[[1]]$name, "x")
  expect_equal(ir$args[[2]]$op, "%in%")
  expect_equal(ir$args[[3]]$value, 0)
})

test_that("replace_values() output type is always x's own type (type_replace_when)", {
  schema <- list(names = "i", types = "INT32")
  ir <- ir_call("replace_when", list(ir_col("i"), ir_call("%in%", list(ir_col("i"), ir_lit_from_r(2))), ir_lit_from_r(99.5)))
  expect_equal(ir_infer_type(ir, schema), "INT32")
})

test_that("replace_values() with from=/to= is unsupported this wave (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(replace_values(x, from = 1, to = 2))), schema_xyz)
  expect_null(ir)
})

test_that("replace_when() parses boolean-condition clauses with x as arg[1]", {
  ir <- ir_parse_quo(quo_in(quote(replace_when(x, x > 1 ~ 99))), schema_xyz)
  expect_equal(ir$op, "replace_when")
  expect_equal(ir$args[[1]]$name, "x")
  expect_equal(ir$args[[2]]$op, ">")
  expect_equal(ir$args[[3]]$value, 99)
})

test_that("replace_when() errors at parse time on a STRING replacement value against a numeric x", {
  expect_error(
    ir_parse_quo(quo_in(quote(replace_when(x, x > 1 ~ "a"))), schema_xyz),
    "replace_when"
  )
})

test_that("when_any()/when_all() desugar to a folded chain of |/&", {
  ir_any <- ir_parse_quo(quo_in(quote(when_any(x > 1, y > 1, z > 1))), schema_xyz)
  expect_equal(ir_any$op, "|")

  ir_all <- ir_parse_quo(quo_in(quote(when_all(x > 1, y > 1, z > 1))), schema_xyz)
  expect_equal(ir_all$op, "&")
})

test_that("when_any(na_rm = TRUE) wraps each input in coalesce(., FALSE)", {
  ir <- ir_parse_quo(quo_in(quote(when_any(x > 1, y > 1, na_rm = TRUE))), schema_xyz)
  expect_equal(ir$op, "|")
  expect_equal(ir$args[[1]]$op, "coalesce")
  expect_false(ir$args[[1]]$args[[2]]$value)
})

test_that("when_all(na_rm = TRUE) wraps each input in coalesce(., TRUE)", {
  ir <- ir_parse_quo(quo_in(quote(when_all(x > 1, y > 1, na_rm = TRUE))), schema_xyz)
  expect_equal(ir$op, "&")
  expect_equal(ir$args[[1]]$op, "coalesce")
  expect_true(ir$args[[1]]$args[[2]]$value)
})

test_that("when_any() with zero dots is unsupported this wave (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(when_any())), schema_xyz)
  expect_null(ir)
})

test_that("when_any()'s size= is unsupported this wave (falls back)", {
  ir <- ir_parse_quo(quo_in(quote(when_any(x > 1, size = 3))), schema_xyz)
  expect_null(ir)
})

# -----------------------------------------------------------------------------
# Phase 5 (window functions, scratchpad/phase5_window_design.md sections 0-1):
# registry window= entries, ir_has_window(), ir_bind()'s window guard, and
# the parse-hook desugars that PRODUCE window IR (cummean/cumall/cumany/
# consecutive_id/ntile). Pure R, no GPU: this file's own module comment
# ("dormant in Phase 1 task T1... nothing else in the package calls
# ir_parse_quo()") still applies -- window functions are additionally
# dormant on top of that, since no verb is wired to plan_window_stages()
# yet (see test-window-plan.R for that machinery's own tests).
# -----------------------------------------------------------------------------

test_that("window-class registry entries exist with window non-NULL and lower NULL", {
  window_ops <- c(
    "lag", "lead", "row_number", "min_rank", "dense_rank", "percent_rank",
    "cume_dist", "cumsum", "cummax", "cummin", "cumprod",
    "mean", "sum", "min", "max", "n", "sd", "var", "first", "last", "nth"
  )
  for (op in window_ops) {
    entry <- ir_call_registry[[op]]
    expect_false(is.null(entry), info = op)
    expect_false(is.null(entry$window), info = op)
    expect_null(entry$lower, info = op)
  }
})

test_that("window-class registry entries' `kind` matches the section 4 lowering family", {
  expect_equal(ir_call_registry[["lag"]]$window$kind, "shift")
  expect_equal(ir_call_registry[["lead"]]$window$kind, "shift")
  expect_equal(ir_call_registry[["row_number"]]$window$kind, "rank")
  expect_equal(ir_call_registry[["min_rank"]]$window$kind, "rank")
  expect_equal(ir_call_registry[["dense_rank"]]$window$kind, "rank")
  expect_equal(ir_call_registry[["percent_rank"]]$window$kind, "rank")
  expect_equal(ir_call_registry[["cume_dist"]]$window$kind, "rank")
  expect_equal(ir_call_registry[["cumsum"]]$window$kind, "scan")
  expect_equal(ir_call_registry[["cummax"]]$window$kind, "scan")
  expect_equal(ir_call_registry[["cummin"]]$window$kind, "scan")
  expect_equal(ir_call_registry[["cumprod"]]$window$kind, "scan")
  expect_equal(ir_call_registry[["mean"]]$window$kind, "agg")
  expect_equal(ir_call_registry[["sum"]]$window$kind, "agg")
  expect_equal(ir_call_registry[["n"]]$window$kind, "agg")
  expect_equal(ir_call_registry[["first"]]$window$kind, "agg")
  expect_equal(ir_call_registry[["last"]]$window$kind, "agg")
  expect_equal(ir_call_registry[["nth"]]$window$kind, "agg")
})

test_that("desugar-only entries (cummean/cumall/cumany/consecutive_id) have window=NULL", {
  for (op in c("cummean", "cumall", "cumany", "consecutive_id")) {
    entry <- ir_call_registry[[op]]
    expect_false(is.null(entry), info = op)
    expect_null(entry$window, info = op)
    expect_null(entry$lower, info = op)
  }
  # ntile() never appears in the registry at all -- it's dispatched
  # directly from ir_parse_expr(), see ir_parse_ntile()'s own docs.
  expect_null(ir_call_registry[["ntile"]])
})

test_that("ir_parse_quo() parses a bare window call (mean(x)) into a window ir_call", {
  ir <- ir_parse_quo(quo_in(quote(mean(x))), schema_xyz)
  expect_equal(ir$kind, "call")
  expect_equal(ir$op, "mean")
  expect_equal(ir$args[[1]]$name, "x")
})

test_that("ir_parse_quo() parses row_number() with 0 args", {
  ir <- ir_parse_quo(quo_in(quote(row_number())), schema_xyz)
  expect_equal(ir$op, "row_number")
  expect_equal(length(ir$args), 0)
})

test_that("ir_parse_quo() parses row_number(x) with 1 arg", {
  ir <- ir_parse_quo(quo_in(quote(row_number(x))), schema_xyz)
  expect_equal(ir$op, "row_number")
  expect_equal(ir$args[[1]]$name, "x")
})

test_that("ir_parse_quo() rejects row_number() with 2 args", {
  ir <- ir_parse_quo(quo_in(quote(row_number(x, y))), schema_xyz)
  expect_null(ir)
})

test_that("n() parses with 0 args", {
  ir <- ir_parse_quo(quo_in(quote(n())), schema_xyz)
  expect_equal(ir$op, "n")
  expect_equal(length(ir$args), 0)
})

test_that("lag(x) defaults n=1L, default=NULL", {
  ir <- ir_parse_quo(quo_in(quote(lag(x))), schema_xyz)
  expect_equal(ir$op, "lag")
  expect_equal(ir$args[[1]]$name, "x")
  expect_equal(ir$meta$n, 1L)
  expect_null(ir$meta$default)
})

test_that("lag(x, 2) and lag(x, n = 2) both set meta$n = 2L", {
  ir1 <- ir_parse_quo(quo_in(quote(lag(x, 2))), schema_xyz)
  ir2 <- ir_parse_quo(quo_in(quote(lag(x, n = 2))), schema_xyz)
  expect_equal(ir1$meta$n, 2L)
  expect_equal(ir2$meta$n, 2L)
})

test_that("lag(x, default = 0) works even though n= is skipped (named-arg matching)", {
  ir <- ir_parse_quo(quo_in(quote(lag(x, default = 0))), schema_xyz)
  expect_equal(ir$op, "lag")
  expect_equal(ir$meta$n, 1L)
  expect_equal(ir$meta$default$value, 0)
})

test_that("lead(x, n = 3, default = -1) sets both meta fields", {
  ir <- ir_parse_quo(quo_in(quote(lead(x, n = 3, default = -1))), schema_xyz)
  expect_equal(ir$op, "lead")
  expect_equal(ir$meta$n, 3L)
  expect_equal(ir$meta$default$value, -1)
})

test_that("lag(x, order_by = y) parses to a window call with an order_override (Phase 5, W4)", {
  ir <- ir_parse_quo(quo_in(quote(lag(x, order_by = y))), schema_xyz)
  expect_false(is.null(ir))
  expect_identical(ir$kind, "call")
  expect_identical(ir$op, "lag")
  expect_identical(ir$meta$order_override, list(col = "y", desc = FALSE))
})

test_that("lag(x, order_by = desc(y)) parses with a descending order_override", {
  ir <- ir_parse_quo(quo_in(quote(lag(x, order_by = desc(y)))), schema_xyz)
  expect_false(is.null(ir))
  expect_identical(ir$meta$order_override, list(col = "y", desc = TRUE))
})

test_that("lag(x, order_by = x + y) (non-column order_by) falls back", {
  ir <- ir_parse_quo(quo_in(quote(lag(x, order_by = x + y))), schema_xyz)
  expect_null(ir)
})

test_that("lag(x, n = 2.5) (non-whole n) falls back", {
  ir <- ir_parse_quo(quo_in(quote(lag(x, n = 2.5))), schema_xyz)
  expect_null(ir)
})

test_that("lag(x, default = y) (non-constant default) falls back", {
  ir <- ir_parse_quo(quo_in(quote(lag(x, default = y))), schema_xyz)
  expect_null(ir)
})

test_that("first(x)/last(x) parse to bare window calls", {
  ir_first <- ir_parse_quo(quo_in(quote(first(x))), schema_xyz)
  expect_equal(ir_first$op, "first")
  expect_equal(ir_first$args[[1]]$name, "x")

  ir_last <- ir_parse_quo(quo_in(quote(last(x))), schema_xyz)
  expect_equal(ir_last$op, "last")
})

test_that("nth(x, 2) sets meta$n = 2L", {
  ir <- ir_parse_quo(quo_in(quote(nth(x, 2))), schema_xyz)
  expect_equal(ir$op, "nth")
  expect_equal(ir$args[[1]]$name, "x")
  expect_equal(ir$meta$n, 2L)
})

test_that("nth(x, -1) (negative element position) is accepted", {
  ir <- ir_parse_quo(quo_in(quote(nth(x, -1))), schema_xyz)
  expect_equal(ir$meta$n, -1L)
})

test_that("first(x, order_by = y) parses to a window call with an order_override (Phase 5, W4)", {
  ir <- ir_parse_quo(quo_in(quote(first(x, order_by = y))), schema_xyz)
  expect_false(is.null(ir))
  expect_identical(ir$op, "first")
  expect_identical(ir$meta$order_override, list(col = "y", desc = FALSE))
})

test_that("first(x, order_by = x + y) (non-column order_by) falls back", {
  ir <- ir_parse_quo(quo_in(quote(first(x, order_by = x + y))), schema_xyz)
  expect_null(ir)
})

test_that("nth(x) missing n falls back", {
  ir <- ir_parse_quo(quo_in(quote(nth(x))), schema_xyz)
  expect_null(ir)
})

test_that("ir_has_window() is TRUE for a bare window call", {
  expect_true(ir_has_window(ir_call("mean", list(ir_col("x")))))
})

test_that("ir_has_window() is FALSE for ordinary arithmetic", {
  expect_false(ir_has_window(ir_call("+", list(ir_col("x"), ir_col("y")))))
})

test_that("ir_has_window() is TRUE when a window call is nested deep inside ordinary arithmetic", {
  ir <- ir_parse_quo(quo_in(quote(x - mean(x) * 2)), schema_xyz)
  expect_true(ir_has_window(ir))
})

test_that("ir_has_window() is TRUE for doubly-nested window calls (lag(cumsum(x)))", {
  ir <- ir_parse_quo(quo_in(quote(lag(cumsum(x)))), schema_xyz)
  expect_true(ir_has_window(ir))
})

test_that("ir_has_window() works on a plain node list (a call's $args)", {
  args <- list(ir_col("x"), ir_call("mean", list(ir_col("y"))))
  expect_true(ir_has_window(args))
  expect_false(ir_has_window(list(ir_col("x"), ir_lit_from_r(1))))
})

test_that("ir_has_window(NULL) / ir_has_window(list()) are FALSE", {
  expect_false(ir_has_window(NULL))
  expect_false(ir_has_window(list()))
})

test_that("ir_bind() stops with an internal error on a bare window call", {
  ir <- ir_call("mean", list(ir_col("x")))
  expect_error(ir_bind(ir, schema_xyz), "internal: window op 'mean' reached lowering")
})

test_that("ir_bind() stops on a window call nested inside ordinary arithmetic", {
  ir <- ir_parse_quo(quo_in(quote(x - cumsum(x))), schema_xyz)
  expect_error(ir_bind(ir, schema_xyz), "internal: window op 'cumsum' reached lowering")
})

test_that("ir_bind() is unaffected for ordinary (non-window) expressions", {
  ir <- ir_parse_quo(quo_in(quote(x + y * 2)), schema_xyz)
  bound <- ir_bind(ir, schema_xyz)
  expect_equal(bound$args[[1]]$index, 0L)
})

test_that("cummean(x) desugars to cumsum(x) / row_number()", {
  ir <- ir_parse_quo(quo_in(quote(cummean(x))), schema_xyz)
  expect_equal(ir$op, "/")
  expect_equal(ir$args[[1]]$op, "cumsum")
  expect_equal(ir$args[[1]]$args[[1]]$name, "x")
  expect_equal(ir$args[[2]]$op, "row_number")
  expect_equal(length(ir$args[[2]]$args), 0)
})

test_that("cumall(x) desugars to the tri-state cummin encoding", {
  ir <- ir_parse_quo(quo_in(quote(cumall(x))), schema_xyz)
  expect_equal(ir$op, "if_else")
  # cond: s == 1L, where s = cummin(encoded)
  cond <- ir$args[[1]]
  expect_equal(cond$op, "==")
  s <- cond$args[[1]]
  expect_equal(s$op, "cummin")
  encoded <- s$args[[1]]
  expect_equal(encoded$op, "if_else")
  expect_equal(encoded$args[[1]]$op, "is.na")
  # yes-branch is the bare (adopted-type) NA literal
  expect_true(ir$args[[2]]$na)
  expect_equal(ir$args[[2]]$type, "BOOL8")
  # no-branch: s == 2L
  expect_equal(ir$args[[3]]$op, "==")
})

test_that("cumany(x) desugars to the tri-state cummax encoding", {
  ir <- ir_parse_quo(quo_in(quote(cumany(x))), schema_xyz)
  s <- ir$args[[1]]$args[[1]]
  expect_equal(s$op, "cummax")
})

test_that("consecutive_id(x) desugars to cumsum() with lag()/row_number() nested inside", {
  ir <- ir_parse_quo(quo_in(quote(consecutive_id(x))), schema_xyz)
  expect_equal(ir$op, "cumsum")
  expect_true(ir_has_window(ir))

  # The nested lag(x) and row_number() are buried inside the increment
  # expression -- confirm both are reachable via window_arg_cols()-style
  # inspection (ir_has_window on the increment subtree).
  increment <- ir$args[[1]]
  expect_true(ir_has_window(increment))
})

test_that("consecutive_id(a, b) (multi-column) falls back (NULL)", {
  ir <- ir_parse_quo(quo_in(quote(consecutive_id(x, y))), schema_xyz)
  expect_null(ir)
})

test_that("ntile(x, 3) desugars with row_number(x)/sum(...) nested window calls", {
  ir <- ir_parse_quo(quo_in(quote(ntile(x, 3))), schema_xyz)
  expect_true(ir_has_window(ir))
  # Outermost shape: if_else(len == 0, NA, bins)
  expect_equal(ir$op, "if_else")
  expect_equal(ir$args[[1]]$op, "==")
  len_node <- ir$args[[1]]$args[[1]]
  expect_equal(len_node$op, "sum")
})

test_that("ntile(n = 3) (x omitted) uses bare row_number()/n() instead of ranking", {
  ir <- ir_parse_quo(quo_in(quote(ntile(n = 3))), schema_xyz)
  expect_true(ir_has_window(ir))
  len_node <- ir$args[[1]]$args[[1]]
  expect_equal(len_node$op, "n")
  expect_equal(length(len_node$args), 0)
})

test_that("ntile() with missing n falls back", {
  ir <- ir_parse_quo(quo_in(quote(ntile(x))), schema_xyz)
  expect_null(ir)
})

test_that("ntile(x, 0) (non-positive n) falls back", {
  ir <- ir_parse_quo(quo_in(quote(ntile(x, 0))), schema_xyz)
  expect_null(ir)
})

# =============================================================================
# sum(x, na.rm = ) -- Phase 5, task W9 (ir_parse_sum_call())
# =============================================================================

test_that("sum(x) (1-arg) parses to a bare window call, unchanged", {
  ir <- ir_parse_quo(quo_in(quote(sum(x))), schema_xyz)
  expect_equal(ir$op, "sum")
  expect_equal(ir$args[[1]]$kind, "col")
  expect_equal(ir$args[[1]]$name, "x")
})

test_that("sum(x, na.rm = TRUE) desugars x to coalesce(x, 0)", {
  ir <- ir_parse_quo(quo_in(quote(sum(x, na.rm = TRUE))), schema_xyz)
  expect_equal(ir$op, "sum")
  expect_equal(ir$args[[1]]$op, "coalesce")
  expect_equal(ir$args[[1]]$args[[1]]$name, "x")
  expect_equal(ir$args[[1]]$args[[2]]$value, 0L)
})

test_that("sum(x, na.rm = FALSE) is a pure no-op (same as bare sum(x))", {
  ir <- ir_parse_quo(quo_in(quote(sum(x, na.rm = FALSE))), schema_xyz)
  expect_equal(ir$op, "sum")
  expect_equal(ir$args[[1]]$kind, "col")
  expect_equal(ir$args[[1]]$name, "x")
})

test_that("sum(x, foo = TRUE) (wrong second-arg name) falls back", {
  ir <- ir_parse_quo(quo_in(quote(sum(x, foo = TRUE))), schema_xyz)
  expect_null(ir)
})

test_that("sum(x, na.rm = y > 0) (non-literal-logical na.rm) falls back", {
  ir <- ir_parse_quo(quo_in(quote(sum(x, na.rm = y > 0))), schema_xyz)
  expect_null(ir)
})

test_that("sum(x, TRUE) (unnamed second arg) falls back", {
  ir <- ir_parse_quo(quo_in(quote(sum(x, TRUE))), schema_xyz)
  expect_null(ir)
})

test_that("sum() (0 args) falls back", {
  ir <- ir_parse_quo(quo_in(quote(sum())), schema_xyz)
  expect_null(ir)
})

test_that("sum(x, na.rm = TRUE, extra = 1) (3 args) falls back", {
  ir <- ir_parse_quo(quo_in(quote(sum(x, na.rm = TRUE, extra = 1))), schema_xyz)
  expect_null(ir)
})

# =============================================================================
# Quosure embedded inside an expression tree (Phase 5, task W9): arises from
# `rlang::expr(fn(!!some_quosure))`, exactly what real dplyr:::tally_n()
# does internally for add_tally(wt=)/add_count(wt=) -- see
# ir_parse_expr()'s "Case 0" docs.
# =============================================================================

test_that("a quosure embedded as a call argument is transparently unwrapped", {
  wt <- rlang::quo(x)
  embedded_call <- rlang::expr(base::sum(!!wt, na.rm = TRUE))
  ir <- ir_parse_quo(rlang::new_quosure(embedded_call, rlang::current_env()), schema_xyz)
  expect_equal(ir$op, "sum")
  expect_equal(ir$args[[1]]$op, "coalesce")
  expect_equal(ir$args[[1]]$args[[1]]$name, "x")
})

test_that("an embedded quosure resolves against its OWN environment, not the outer call's", {
  local_env <- new.env()
  local_env$outer_var <- 42
  wt <- rlang::new_quosure(quote(outer_var), local_env)
  embedded_call <- rlang::expr(base::sum(!!wt))
  ir <- ir_parse_quo(rlang::new_quosure(embedded_call, rlang::empty_env()), schema_xyz)
  expect_equal(ir$op, "sum")
  expect_equal(ir$args[[1]]$kind, "lit")
  expect_equal(ir$args[[1]]$value, 42)
})
