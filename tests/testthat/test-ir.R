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
  schema <- list(names = "x", types = "FLOAT64")
  ir <- ir_parse_quo(quo_in(quote(max(c(1, 2)))), schema)

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
