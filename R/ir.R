# Expression IR - rlang quosure -> IR -> (eventually) cudf::compute_column
#
# See scratchpad/phase1_expression_engine.md, section 1, for the full design
# authority this file implements verbatim (node shapes 1.1, parse rule 1.2,
# type inference 1.3) and section 6 for the ir_call_registry extension
# contract that T2/T3 will consume via each entry's `lower` field.
#
# This file is dormant in Phase 1 task T1: nothing else in the package calls
# ir_parse_quo()/ir_infer_type()/etc. yet. filter.R/mutate.R keep using their
# existing parsers until T3/T5 cut them over.

# -----------------------------------------------------------------------------
# Node Constructors
# -----------------------------------------------------------------------------

#' Create an IR column-reference node
#'
#' @param name Character scalar, the column name
#' @return A `cuplyr_ir_col` node: `list(kind = "col", name = <chr1>)`. Gains
#'   an `index` field (0-based) only after [ir_bind()] runs.
#' @keywords internal
ir_col <- function(name) {
  structure(
    list(kind = "col", name = name),
    class = c("cuplyr_ir_col", "cuplyr_ir")
  )
}

#' Create an IR literal node
#'
#' @param value Atomic value (length 1, or length > 1 only for a
#'   set-argument literal such as `%in%`'s RHS), or `NULL`
#' @param type GPU type string, or `NULL` for a bare `NA` (type is adopted
#'   from the sibling operand at build/lowering time; defaults to `FLOAT64`
#'   when inferred in isolation, see [ir_infer_type()])
#' @param na `TRUE` iff this literal is the bare, untyped `NA` constant
#' @return A `cuplyr_ir_lit` node
#' @keywords internal
ir_lit <- function(value, type, na = FALSE) {
  structure(
    list(kind = "lit", value = value, type = type, na = na),
    class = c("cuplyr_ir_lit", "cuplyr_ir")
  )
}

#' Create an IR function-call node
#'
#' @param op Canonical cuplyr op name (not necessarily the R function name
#'   that produced it -- see [ir_call_registry] and `ir_op_alias()`, e.g.
#'   `ceiling` normalizes to `ceil`)
#' @param args List of IR nodes (the call's arguments, already parsed)
#' @param meta Forward-compatibility slot for future lowering metadata
#'   (e.g. window-function args); unused in Phase 1
#' @return A `cuplyr_ir_call` node
#' @keywords internal
ir_call <- function(op, args, meta = list()) {
  structure(
    list(kind = "call", op = op, args = args, meta = meta),
    class = c("cuplyr_ir_call", "cuplyr_ir")
  )
}

#' Build an IR literal node from an already-evaluated R value
#'
#' Implements the length/NA rules from section 1.2: a bare, untyped `NA`
#' (the logical `NA` constant) becomes `na = TRUE` with `type = NULL`
#' (adopts the sibling operand's type later; [ir_infer_type()] defaults it
#' to `FLOAT64` when asked in isolation). Any other length-1 atomic value is
#' typed via [gpu_type_from_r()]. Length != 1 is only legal when
#' `allow_vector = TRUE` (the `%in%` RHS set-argument position); otherwise
#' it's a diagnosable user error.
#'
#' @param value An R value (atomic length 1, an atomic vector, or `NULL`)
#' @param allow_vector `TRUE` iff a length != 1 vector is acceptable here
#'   (used for `%in%`'s RHS)
#' @return A `cuplyr_ir_lit` node
#' @keywords internal
ir_lit_from_r <- function(value, allow_vector = FALSE) {
  if (is.null(value)) {
    return(ir_lit(value = NULL, type = NULL, na = FALSE))
  }

  n <- length(value)

  if (n != 1) {
    if (!allow_vector) {
      stop("Expected a numeric scalar, but got a value of length ", n,
           ". (Vector literals are only supported on the right-hand side ",
           "of %in%.)", call. = FALSE)
    }
    if (n == 0) {
      stop("Expected a literal of length >= 1, but got length 0.",
           call. = FALSE)
    }
    return(ir_lit(value = value, type = gpu_type_from_r(value), na = FALSE))
  }

  if (is.logical(value) && is.na(value)) {
    return(ir_lit(value = NULL, type = NULL, na = TRUE))
  }

  ir_lit(value = value, type = gpu_type_from_r(value), na = FALSE)
}

# -----------------------------------------------------------------------------
# Parsing: rlang quosure -> IR (or NULL for "unsupported, caller should
# fall back")
# -----------------------------------------------------------------------------

# Canonical op-name aliasing, applied to the R function name before registry
# lookup (section 1.1: "op is a canonical cuplyr op name, not an R function
# name: the parser normalizes (ceiling -> ceil, ...)").
ir_op_alias <- function(fn_name) {
  switch(fn_name, ceiling = "ceil", fn_name)
}

# TRUE iff `expr` is `.data$name` or `.data[["name"]]`.
is_data_pronoun_call <- function(expr) {
  is.call(expr) &&
    length(expr) == 3 &&
    (identical(expr[[1]], as.name("$")) || identical(expr[[1]], as.name("[["))) &&
    identical(expr[[2]], as.name(".data"))
}

# TRUE iff `expr` is `.env$name` or `.env[["name"]]`.
is_env_pronoun_call <- function(expr) {
  is.call(expr) &&
    length(expr) == 3 &&
    (identical(expr[[1]], as.name("$")) || identical(expr[[1]], as.name("[["))) &&
    identical(expr[[2]], as.name(".env"))
}

# Extract the bare name out of a `.data$name`/`.data[["name"]]` (or `.env`
# equivalent) call's third element, which is a symbol for `$` and a string
# for `[[`.
pronoun_call_name <- function(expr) {
  as.character(expr[[3]])
}

# TRUE iff `expr` contains a symbol matching a schema column name anywhere
# in its subtree (used to decide whether an unrecognized call's subtree can
# be constant-folded per section 1.2's governing rule). A forced `.data`
# reference anywhere in the subtree is conservatively always treated as a
# column reference, even if the referenced column is missing (an
# unsupported call is not the place to raise the "not found" diagnostic;
# the caller falls back instead).
expr_has_column_ref <- function(expr, schema) {
  if (is.symbol(expr)) {
    return(as.character(expr) %in% schema$names)
  }
  if (is_data_pronoun_call(expr)) {
    return(TRUE)
  }
  if (is.call(expr)) {
    return(any(vapply(as.list(expr), expr_has_column_ref, logical(1), schema = schema)))
  }
  FALSE
}

# The recursive walker behind ir_parse_quo(). Carries the quosure's
# environment explicitly (rather than re-deriving a quosure at every
# recursive step) and an `allow_vector` flag threaded down only to the
# `%in%` RHS argument position.
ir_parse_expr <- function(expr, env, schema, allow_vector = FALSE) {
  # Case 1: "(" is transparent.
  if (rlang::is_call(expr, "(") && length(expr) == 2) {
    return(ir_parse_expr(expr[[2]], env, schema, allow_vector))
  }

  # Case 2: bare symbol. Columns shadow the environment.
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    if (nm %in% schema$names) {
      return(ir_col(nm))
    }
    val <- tryCatch(
      rlang::eval_tidy(rlang::new_quosure(expr, env)),
      error = function(e) stop(conditionMessage(e), call. = FALSE)
    )
    return(ir_lit_from_r(val, allow_vector = allow_vector))
  }

  # Case 3: .data[["x"]] / .data$x - forced column reference.
  if (is_data_pronoun_call(expr)) {
    nm <- pronoun_call_name(expr)
    if (!nm %in% schema$names) {
      stop("Column '", nm, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    return(ir_col(nm))
  }

  # Case 4: .env$x / .env[["x"]] - forced environment lookup.
  if (is_env_pronoun_call(expr)) {
    nm <- pronoun_call_name(expr)
    val <- tryCatch(
      get(nm, envir = env, inherits = TRUE),
      error = function(e) stop("object '", nm, "' not found", call. = FALSE)
    )
    return(ir_lit_from_r(val, allow_vector = allow_vector))
  }

  # Case 5: literal constant (numeric/character/logical/NA/NULL).
  if (is.null(expr)) {
    return(ir_lit_from_r(NULL))
  }
  if (is.atomic(expr) && length(expr) == 1) {
    return(ir_lit_from_r(expr, allow_vector = allow_vector))
  }

  # Case 6: call f(...).
  if (rlang::is_call(expr)) {
    head <- expr[[1]]

    if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
      fn_name <- as.character(head[[3]])  # strip namespace: pkg::fn -> fn
    } else if (is.symbol(head)) {
      fn_name <- as.character(head)
    } else {
      return(NULL)  # exotic call head (e.g. a function literal): unsupported
    }

    if (fn_name %in% c("&&", "||")) {
      replacement <- if (fn_name == "&&") "&" else "|"
      stop("'", fn_name, "' is not supported in filter(); use '", replacement, "'",
           call. = FALSE)
    }

    canonical <- ir_op_alias(fn_name)
    args_raw <- as.list(expr)[-1]

    if (canonical %in% names(ir_call_registry)) {
      entry <- ir_call_registry[[canonical]]

      if (!is.na(entry$arity) && length(args_raw) != entry$arity) {
        return(NULL)  # wrong arg count for this op: not this shape, fall back
      }

      parsed_args <- vector("list", length(args_raw))
      for (i in seq_along(args_raw)) {
        # %in%'s RHS (2nd argument) is the one set-argument position where
        # a vector literal is legal (section 1.2/1.1).
        arg_allow_vector <- identical(canonical, "%in%") && i == 2L
        parsed <- ir_parse_expr(args_raw[[i]], env, schema, allow_vector = arg_allow_vector)
        if (is.null(parsed)) {
          return(NULL)
        }
        parsed_args[[i]] <- parsed
      }

      if (!is.null(entry$parse)) {
        return(entry$parse(parsed_args, schema))
      }
      return(ir_call(canonical, parsed_args))
    }

    # f not in the registry: constant-fold if the subtree has no column
    # reference (how c(4, 6), nrow(df), Sys.Date() work), else unsupported.
    if (!expr_has_column_ref(expr, schema)) {
      val <- tryCatch(
        rlang::eval_tidy(rlang::new_quosure(expr, env)),
        error = function(e) stop(conditionMessage(e), call. = FALSE)
      )
      return(ir_lit_from_r(val, allow_vector = allow_vector))
    }

    return(NULL)
  }

  NULL
}

#' Parse an rlang quosure into cuplyr IR
#'
#' Implements the parse rule of section 1.2: columns shadow the
#' environment; any subtree with no column reference is constant-folded by
#' evaluating it in the quosure's environment. Returns `NULL` for anything
#' the IR doesn't (yet) understand -- callers (filter/mutate parsers, from
#' T3/T5 onward) fall back to the pre-existing CPU-eval or error path.
#'
#' @param quo An `rlang` quosure
#' @param schema List with `names`/`types` (as from [current_schema()])
#' @return An IR node, or `NULL` if unsupported
#' @keywords internal
ir_parse_quo <- function(quo, schema) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)
  ir_parse_expr(expr, env, schema, allow_vector = FALSE)
}

# -----------------------------------------------------------------------------
# Tree Helpers
# -----------------------------------------------------------------------------

#' Collect the unique columns referenced by an IR subtree
#'
#' @param ir An IR node (or `NULL`)
#' @return Unique character vector of column names, in first-encountered
#'   order
#' @keywords internal
ir_cols <- function(ir) {
  if (is.null(ir)) {
    return(character())
  }
  switch(ir$kind,
    col = ir$name,
    lit = character(),
    call = unique(unlist(lapply(ir$args, ir_cols), use.names = FALSE))
  )
}

#' Estimate the relative evaluation cost of an IR subtree
#'
#' Used by `reorder_filters()` to run cheaper predicates first. Literal = 1,
#' column reference = 2, and every call node adds 1 on top of the summed
#' cost of its arguments -- so a scalar comparison (`x > 5`, cost 4) is
#' cheaper than a column-column comparison (`x > y`, cost 5), which in turn
#' is cheaper than any deeper nesting.
#'
#' @param ir An IR node (or `NULL`)
#' @return Integer cost
#' @keywords internal
ir_cost <- function(ir) {
  if (is.null(ir)) {
    return(0L)
  }
  switch(ir$kind,
    lit = 1L,
    col = 2L,
    call = sum(vapply(ir$args, ir_cost, integer(1))) + 1L
  )
}

#' Is an IR subtree constant (no column references)?
#'
#' @param ir An IR node
#' @return `TRUE` iff [ir_cols()] is empty
#' @keywords internal
ir_is_const <- function(ir) {
  length(ir_cols(ir)) == 0
}

#' Bind 0-based column indices onto every `col` node in an IR tree
#'
#' Per section 2.1's decision, indices are resolved against a schema only
#' at lowering time (never at parse time), since projection pushdown can
#' shift column positions in between. Returns a new tree (does not mutate
#' in place, matching R's copy-on-modify semantics) with an `index` field
#' (0-based, for C++) added to every `col` node.
#'
#' @param ir An IR node (or `NULL`)
#' @param schema List with `names`/`types` to resolve column positions
#'   against
#' @return The IR tree with `index` filled in on every `col` node
#' @keywords internal
ir_bind <- function(ir, schema) {
  if (is.null(ir)) {
    return(ir)
  }

  if (ir$kind == "col") {
    idx <- match(ir$name, schema$names)
    if (is.na(idx)) {
      stop("Column '", ir$name, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    ir$index <- idx - 1L
    return(ir)
  }

  if (ir$kind == "lit") {
    return(ir)
  }

  ir$args <- lapply(ir$args, ir_bind, schema = schema)
  ir
}

#' Deparse an IR node back to readable text
#'
#' For diagnostics (`show_query()`, error messages) -- not intended to be a
#' faithful/round-trippable R deparser, just a readable rendering.
#'
#' @param ir An IR node (or `NULL`)
#' @return Character scalar
#' @keywords internal
ir_deparse <- function(ir) {
  if (is.null(ir)) {
    return("NULL")
  }

  switch(ir$kind,
    col = ir$name,
    lit = {
      if (isTRUE(ir$na)) {
        "NA"
      } else if (is.null(ir$value)) {
        "NULL"
      } else if (length(ir$value) > 1) {
        paste0("c(", paste(format(ir$value), collapse = ", "), ")")
      } else {
        format(ir$value)
      }
    },
    call = {
      arg_strs <- vapply(ir$args, ir_deparse, character(1))
      paste0(ir$op, "(", paste(arg_strs, collapse = ", "), ")")
    }
  )
}

#' Infer the GPU output type of an IR node
#'
#' Recurses; for arithmetic call nodes this delegates to the existing
#' [infer_mutate_output_type()] unchanged, so D5's rules (and its pinned
#' STRING/DICTIONARY32/TIMESTAMP_* error messages) survive verbatim. Every
#' other op class is resolved via its [ir_call_registry] entry's `type`
#' function (section 1.3).
#'
#' @param ir An IR node
#' @param schema List with `names`/`types` to resolve column types against
#' @return GPU type string
#' @keywords internal
ir_infer_type <- function(ir, schema) {
  if (ir$kind == "col") {
    idx <- match(ir$name, schema$names)
    if (is.na(idx)) {
      stop("Column '", ir$name, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    return(schema$types[idx])
  }

  if (ir$kind == "lit") {
    # A bare NA (type = NULL) defaults to FLOAT64 when asked in isolation;
    # in practice it's asked as part of a call's arg_types, where its
    # contribution is a no-op for infer_mutate_output_type()'s promotion
    # union (which is exactly why FLOAT64 -- the union identity value for
    # the promotion rules in section 1.3 -- is a safe default here too).
    if (is.null(ir$type)) {
      return("FLOAT64")
    }
    return(ir$type)
  }

  entry <- ir_call_registry[[ir$op]]
  if (is.null(entry)) {
    stop("Unknown IR op: '", ir$op, "'", call. = FALSE)
  }

  arg_types <- vapply(ir$args, ir_infer_type, character(1), schema = schema)
  entry$type(arg_types)
}

# -----------------------------------------------------------------------------
# ir_call_registry - the extension contract (section 6)
# -----------------------------------------------------------------------------
#
# Adding a function is two edits, never a parser change:
#   R side: one entry here -- arity (int or NA for variable), an optional
#     `parse` desugaring hook (NULL if the op needs none), a `type`
#     function computing the GPU output type from argument types, and a
#     `lower` spec: list(ast_op = "<ENUM>") for an AST-native op (T2 looks
#     the enum name up directly), or list(handler = "<name>") for a
#     non-AST cuDF call (T2 adds one case in apply_handler()). `lower` is
#     NULL for ops that are always fully desugared at parse time (between,
#     log2, log10) and therefore never survive to reach lowering under
#     their own op name.
#   C++ side: nothing for ast_op entries; one apply_handler() case for
#     handler entries.

# Small type-fn builders shared by several registry entries below.
type_arith_op <- function(op) {
  force(op)
  function(arg_types) infer_mutate_output_type(op, arg_types, NULL)
}
type_bool8 <- function(arg_types) "BOOL8"
type_float64 <- function(arg_types) "FLOAT64"
type_arg1 <- function(arg_types) arg_types[1]

# Unary `-x` preserves the argument's type (section 1.3); binary `x - y`
# defers to infer_mutate_output_type() like the other D5-governed ops
# (+, *, %%, %/%). Both share the single registry key "-" (arity = NA)
# since R's parser also uses the same call head for both arities.
type_minus <- function(arg_types) {
  if (length(arg_types) == 1) {
    return(arg_types[1])
  }
  infer_mutate_output_type("-", arg_types, NULL)
}

#' Registry of supported expression-IR operations
#'
#' See the module-level comment above and section 6 of
#' `scratchpad/phase1_expression_engine.md` for the extension contract this
#' implements. Each entry: `list(arity, parse, type, lower)`.
#'
#' @keywords internal
ir_call_registry <- list(
  "+" = list(arity = 2L, parse = NULL, type = type_arith_op("+"), lower = list(ast_op = "ADD")),
  "-" = list(arity = NA_integer_, parse = NULL, type = type_minus, lower = list(ast_op = "SUB")),
  "*" = list(arity = 2L, parse = NULL, type = type_arith_op("*"), lower = list(ast_op = "MUL")),
  "/" = list(arity = 2L, parse = NULL, type = type_float64, lower = list(ast_op = "TRUE_DIV")),
  "^" = list(arity = 2L, parse = NULL, type = type_float64, lower = list(ast_op = "POW")),
  "%%" = list(arity = 2L, parse = NULL, type = type_arith_op("%%"), lower = list(ast_op = "PYMOD")),
  "%/%" = list(arity = 2L, parse = NULL, type = type_arith_op("%/%"), lower = list(ast_op = "FLOOR_DIV")),

  "==" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "EQUAL")),
  "!=" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "NOT_EQUAL")),
  "<"  = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "LESS")),
  "<=" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "LESS_EQUAL")),
  ">"  = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "GREATER")),
  ">=" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "GREATER_EQUAL")),

  "&" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "NULL_LOGICAL_AND")),
  "|" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "NULL_LOGICAL_OR")),
  "!" = list(arity = 1L, parse = NULL, type = type_bool8, lower = list(ast_op = "NOT")),
  "xor" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "BITWISE_XOR")),

  "is.na" = list(arity = 1L, parse = NULL, type = type_bool8, lower = list(ast_op = "IS_NULL")),

  "between" = list(
    arity = 3L,
    parse = function(args, schema) {
      x <- args[[1]]
      lo <- args[[2]]
      hi <- args[[3]]
      ir_call("&", list(ir_call(">=", list(x, lo)), ir_call("<=", list(x, hi))))
    },
    type = type_bool8,
    lower = NULL  # always desugared at parse time; never lowered as "between"
  ),

  "%in%" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(handler = "in")),

  "sqrt" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "SQRT")),

  "log" = list(
    arity = NA_integer_,  # log(x) or log(x, base)
    parse = function(args, schema) {
      if (length(args) == 2) {
        base_node <- args[[2]]
        if (!identical(base_node$kind, "lit") || isTRUE(base_node$na) ||
            is.null(base_node$value) || length(base_node$value) != 1) {
          return(NULL)  # non-constant base: unsupported, fall back
        }
        return(ir_call("/", list(
          ir_call("log", list(args[[1]])),
          ir_lit_from_r(log(base_node$value))
        )))
      }
      ir_call("log", args)
    },
    type = type_float64,
    lower = list(ast_op = "LOG")
  ),
  "log2" = list(
    arity = 1L,
    parse = function(args, schema) {
      ir_call("/", list(ir_call("log", args), ir_lit_from_r(log(2))))
    },
    type = type_float64,
    lower = NULL  # always desugared at parse time
  ),
  "log10" = list(
    arity = 1L,
    parse = function(args, schema) {
      ir_call("/", list(ir_call("log", args), ir_lit_from_r(log(10))))
    },
    type = type_float64,
    lower = NULL  # always desugared at parse time
  ),

  "exp" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "EXP")),
  "abs" = list(arity = 1L, parse = NULL, type = type_arg1, lower = list(ast_op = "ABS")),
  "floor" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "FLOOR")),
  "ceil" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "CEIL")),
  "sin" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "SIN")),
  "cos" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "COS")),
  "tan" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "TAN")),

  "round" = list(arity = NA_integer_, parse = NULL, type = type_arg1, lower = list(handler = "round"))
)
