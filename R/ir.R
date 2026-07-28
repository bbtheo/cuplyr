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
#' A length-0 vector is legal in the `allow_vector = TRUE` position too:
#' `x %in% numeric(0)` (or `character(0)`, etc.) is well-defined, ordinary R
#' -- always `FALSE`, for every element including `NA` -- not a user error,
#' so it must not be rejected here (T4 correctness finding: this used to
#' `stop()` for *any* length-0 vector, `allow_vector` or not, which broke
#' `filter(x %in% <empty vector>)` parity with dplyr; the `%in%` handler in
#' `src/expr_eval.hpp` special-cases an empty/`NULL` RHS set to "always
#' FALSE" to match).
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

    # case_when()'s dots are `~` formulas, not ordinary expressions the
    # generic per-arg parse loop below understands (a formula's LHS/RHS need
    # pulling apart before recursing, and `.default=` is a named value
    # argument, not a formula at all) -- so it's special-cased here, before
    # canonical/registry dispatch, the same way `&&`/`||` are. See
    # ir_parse_case_when()'s own docs for exactly what it accepts.
    if (identical(fn_name, "case_when")) {
      return(ir_parse_case_when(expr, env, schema))
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

#' Parse a `case_when()` call into IR (Phase 4)
#'
#' `case_when(cond1 ~ val1, cond2 ~ val2, ..., .default = val)` dots are
#' two-sided formulas (`~` calls), not ordinary expressions -- so unlike
#' every other `ir_call_registry` entry, this is invoked directly from
#' `ir_parse_expr()`'s call-handling branch (see the `identical(fn_name,
#' "case_when")` check there) rather than going through the registry's
#' generic per-arg parse loop. Recognizes exactly: any number of two-sided
#' `cond ~ value` clauses, plus an optional `.default = value` named
#' argument. Returns `NULL` (unsupported, caller falls back per the normal
#' contract) for anything else this Phase-4 wave doesn't implement yet:
#' `.ptype=`, `.size=`, `.unmatched=`, one-sided formulas, non-formula dots,
#' or zero clauses.
#'
#' Normalizes into a single flat `ir_call("case_when", args)` where `args`
#' is `list(cond1, val1, cond2, val2, ..., valN, default)` -- always
#' terminated by a default (a bare untyped-NA literal is synthesized when
#' the user didn't supply `.default=`, matching dplyr's "unmatched rows are
#' a typed NA" rule). `meta$n_when` records the clause count for
#' diagnostics/`show_query()`; the C++ side derives the same count from
#' `length(args)` being odd (`2 * n_when + 1`), so `meta` isn't load-bearing
#' there.
#'
#' Bare untyped `NA` literals among the value positions (any `val_i` or the
#' `.default`/synthesized default) are resolved against a sibling value's
#' type via [ir_resolve_value_nas()] -- the same value-position type
#' adoption if_else()/coalesce() use, generalized to `case_when`'s N+1
#' value slots.
#'
#' @param expr The raw `case_when(...)` call
#' @param env The quosure's environment (conditions/values are parsed
#'   against this, exactly like any other IR subtree)
#' @param schema List with `names`/`types`
#' @return An `ir_call("case_when", ...)` node, or `NULL` if unsupported
#' @keywords internal
ir_parse_case_when <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }

  # .ptype=/.size=/.unmatched=/etc.: not implemented this wave -- fall back.
  unsupported_named <- setdiff(arg_names[nzchar(arg_names)], ".default")
  if (length(unsupported_named) > 0) {
    return(NULL)
  }

  default_pos <- which(arg_names == ".default")
  default_raw <- if (length(default_pos) == 1) args_raw[[default_pos]] else NULL
  clause_raw <- if (length(default_pos) == 1) args_raw[-default_pos] else args_raw

  if (length(clause_raw) == 0) {
    return(NULL)  # nothing to match against
  }

  conds <- vector("list", length(clause_raw))
  vals <- vector("list", length(clause_raw))

  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)  # not a two-sided formula: unsupported shape
    }
    cond_ir <- ir_parse_expr(f[[2]], env, schema)
    if (is.null(cond_ir)) {
      return(NULL)
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    conds[[i]] <- cond_ir
    vals[[i]] <- val_ir
  }

  default_ir <- if (!is.null(default_raw)) {
    parsed_default <- ir_parse_expr(default_raw, env, schema)
    if (is.null(parsed_default)) {
      return(NULL)
    }
    parsed_default
  } else {
    ir_lit(value = NULL, type = NULL, na = TRUE)  # synthesized untyped NA
  }

  values <- ir_resolve_value_nas(c(vals, list(default_ir)), schema)
  n_when <- length(conds)

  interleaved <- vector("list", n_when * 2L)
  interleaved[seq(1L, n_when * 2L, by = 2L)] <- conds
  interleaved[seq(2L, n_when * 2L, by = 2L)] <- values[seq_len(n_when)]

  ir_call("case_when", c(interleaved, values[n_when + 1L]), meta = list(n_when = n_when))
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
    # A bare NA (type = NULL) defaults to BOOL8 when asked in isolation --
    # matching `typeof(NA)` ("logical") in R, so `mutate(y = NA)` types `y`
    # as a logical column exactly like dplyr (T6 finding: this used to
    # default to FLOAT64 on the theory that it's always asked as part of a
    # call's arg_types where its contribution is a promotion-union no-op;
    # empirically wrong for a bare NA as an entire top-level mutate
    # expression, which never reaches a call node at all). See the "call"
    # branch below for the sibling-adoption logic that keeps this default
    # from over-eagerly promoting a *nested* NA in e.g. `x + NA`.
    if (is.null(ir$type)) {
      return("BOOL8")
    }
    return(ir$type)
  }

  entry <- ir_call_registry[[ir$op]]
  if (is.null(entry)) {
    stop("Unknown IR op: '", ir$op, "'", call. = FALSE)
  }

  arg_types <- vapply(ir$args, ir_infer_type, character(1), schema = schema)

  # A bare, untyped NA literal argument (kind="lit", na=TRUE, type=NULL) is
  # a type no-op in R's arithmetic: `1L + NA` is integer, not double (NA is
  # coerced to the OTHER operand's type, it never forces a promotion). Since
  # each arg's type is inferred independently above, an isolated NA arg
  # would otherwise surface as the "in isolation" BOOL8 default from the
  # branch above and incorrectly feed into infer_mutate_output_type()'s
  # promotion union. Substitute it with a sibling arg's type first (mirrors
  # src/expr_eval.hpp's resolve_static_type(), which does the same
  # adoption when building the actual AST) -- T6 finding: without this,
  # `mutate(y = int_col + NA)` came back FLOAT64 instead of matching
  # dplyr's INT32.
  is_isolated_na <- vapply(ir$args, function(a) {
    identical(a$kind, "lit") && isTRUE(a$na) && is.null(a$type)
  }, logical(1))
  if (any(is_isolated_na) && !all(is_isolated_na)) {
    arg_types[is_isolated_na] <- arg_types[!is_isolated_na][1]
  }

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

# abs() preserves its argument's type -- EXCEPT a logical (BOOL8) argument,
# which R promotes to integer (`typeof(abs(TRUE))` is "integer", not
# "logical"; base R's abs() coerces via as.integer()/as.double() before
# taking the absolute value, it never returns a logical). Verified against
# the dplyr oracle (T6 finding; phase1_expression_engine.md section 1.3's
# "abs, round | argument type" table entry was correct for abs() only on
# the non-logical case).
type_abs <- function(arg_types) {
  if (identical(arg_types[1], "BOOL8")) "INT32" else arg_types[1]
}

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

# -----------------------------------------------------------------------------
# Phase 4: value-position type unification (if_else/case_when/coalesce) and
# the shared bare-NA-adoption helper they (and na_if) all need.
#
# Verified empirically against dplyr 1.2.1 (see the Phase 4 task's trap
# list): `if_else()`/`case_when()`/`coalesce()` unify their "value position"
# arguments (if_else's yes/no/missing; case_when's val_i's + default;
# coalesce's whole arg list) via a vctrs-style common type, NOT the strict
# "must be identical type" rule -- e.g. `if_else(cond, 1L, 2.5)` succeeds as
# a double (dplyr's vctrs::vec_ptype2() freely promotes across
# integer/double/logical), it only errors on a genuinely incompatible mix
# like character vs double. This mirrors -- but is not literally the same
# function as -- infer_mutate_output_type()'s D5 arithmetic promotion
# (BOOL8 < INT32 < INT64 < FLOAT64); STRING here can only unify with STRING
# (no numeric<->STRING promotion path exists either way).
# -----------------------------------------------------------------------------

ir_value_type_rank <- c(BOOL8 = 1L, INT32 = 2L, INT64 = 3L, FLOAT64 = 4L)

# @param types Character vector of GPU types for every "value position" arg
#   of an if_else()/case_when()/coalesce() call (already resolved -- no
#   remaining bare/isolated NA placeholders, see ir_resolve_value_nas())
# @param op_name Verb name, for the error message
# @return The unified GPU type
# @keywords internal
ir_unify_types <- function(types, op_name) {
  uniq <- unique(types)
  if (length(uniq) == 0) {
    return("BOOL8")  # typeof(NA) in R, when every value position was an isolated NA
  }
  if (length(uniq) == 1) {
    return(uniq)
  }

  if ("STRING" %in% uniq) {
    stop(op_name, "(): can't combine ",
         paste(paste0("<", uniq, ">"), collapse = " and "),
         " -- STRING values can't be combined with a non-STRING value.",
         call. = FALSE)
  }

  ranks <- ir_value_type_rank[uniq]
  if (anyNA(ranks)) {
    stop(op_name, "(): can't unify types ", paste(uniq, collapse = ", "), call. = FALSE)
  }

  names(ranks)[which.max(ranks)]
}

type_if_else <- function(arg_types) {
  ir_unify_types(arg_types[-1], "if_else")  # drop cond's own (BOOL8) type
}

type_case_when <- function(arg_types) {
  n <- length(arg_types)
  value_idx <- c(seq(2L, n - 1L, by = 2L), n)  # val_1, val_2, ..., default
  ir_unify_types(arg_types[value_idx], "case_when")
}

type_coalesce <- function(arg_types) {
  ir_unify_types(arg_types, "coalesce")
}

# na_if()'s output keeps `x`'s own type exactly (verified empirically: e.g.
# `na_if(1:3, 2)` -- an integer `x` against a double literal `y` -- stays
# integer; `y` is just compared against, never promoted into the result).
type_na_if <- function(arg_types) {
  arg_types[1]
}

#' Resolve a bare untyped `NA` literal's type from a sibling value-position arg
#'
#' `if_else()`/`case_when()`/`coalesce()`/`na_if()` each have several "value
#' position" argument slots that must share a common output type (if_else's
#' yes/no/missing; case_when's val_i's + default; coalesce's whole arg list;
#' na_if's x/y). A bare, untyped `NA` in one of those slots (e.g. `if_else(
#' cond, x, NA)`) must adopt a *sibling* value-position arg's type -- exactly
#' like arithmetic's `1L + NA` being integer, generalized from 2 args to N
#' (section 1.2/1.3).
#'
#' This is done here, at PARSE time (mutating each isolated node's own
#' `type` field directly, in place), rather than relying on
#' [ir_infer_type()]'s own generic isolated-NA substitution (which already
#' exists for arithmetic's 2-arg case): that generic mechanism picks the
#' FIRST non-isolated arg among a call node's ENTIRE `args` list, which for
#' `if_else` would incorrectly be `cond` (always the first arg, always
#' non-isolated, and NOT a value-position arg) rather than a true sibling
#' value. Calling this here on ONLY the value-position args (never `cond`)
#' avoids that pitfall entirely, and once it runs, the isolated node's
#' `type` field is no longer `NULL` -- so `ir_infer_type()`'s later generic
#' walk of the finished node is a no-op for these args, as intended.
#'
#' A typed `NA` (e.g. `NA_character_`, `NA_integer_`) is untouched: it
#' already carries its own `type` and is not "isolated" by this function's
#' definition (matching dplyr: `na_if(numeric_x, NA_character_)` still
#' errors, only a bare untyped `NA` adopts a sibling's type).
#'
#' @param nodes A list of IR nodes -- the "value position" args only (e.g.
#'   if_else()'s yes/no/missing, never its cond)
#' @param schema List with `names`/`types`
#' @return `nodes`, with every isolated bare-NA node's `type` field filled
#'   in from the first non-isolated sibling's inferred type (returned
#'   unchanged if none, or all, of `nodes` are isolated)
#' @keywords internal
ir_resolve_value_nas <- function(nodes, schema) {
  is_isolated_na <- vapply(nodes, function(n) {
    identical(n$kind, "lit") && isTRUE(n$na) && is.null(n$type)
  }, logical(1))

  if (!any(is_isolated_na) || all(is_isolated_na)) {
    return(nodes)
  }

  fallback_type <- ir_infer_type(nodes[[which(!is_isolated_na)[1]]], schema)

  for (i in which(is_isolated_na)) {
    nodes[[i]]$type <- fallback_type
  }
  nodes
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
  "abs" = list(arity = 1L, parse = NULL, type = type_abs, lower = list(ast_op = "ABS")),
  "floor" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "FLOOR")),
  "ceil" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "CEIL")),
  "sin" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "SIN")),
  "cos" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "COS")),
  "tan" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "TAN")),

  # round() ALWAYS returns a double in R, regardless of the input's type
  # (`typeof(round(1L))`, `typeof(round(TRUE))`, `typeof(round(1L, 2))` are
  # all "double") -- unlike abs(), which does preserve non-BOOL8 input
  # types. Verified against the dplyr oracle (T6 finding; corrects
  # phase1_expression_engine.md section 1.3's table, which grouped round()
  # with abs() as "argument type" -- empirically wrong for round()).
  "round" = list(arity = NA_integer_, parse = NULL, type = type_float64, lower = list(handler = "round")),

  # --- Phase 4: conditional & vector functions ---
  # `if_else(cond, yes, no, missing = NULL)`: 3 or 4 positional args (the
  # `arity = NA` + validating `parse` hook is the same pattern `log(x[,
  # base])` already uses above). `ir_resolve_value_nas()` fixes up a bare
  # untyped NA among yes/no/missing before the call node is built (never
  # touching `cond`, see that function's own docs for why that matters).
  "if_else" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) < 3 || length(args) > 4) {
        return(NULL)  # wrong arg count: not this shape, fall back
      }
      cond <- args[[1]]
      values <- ir_resolve_value_nas(args[-1], schema)
      ir_call("if_else", c(list(cond), values))
    },
    type = type_if_else,
    lower = list(handler = "if_else")
  ),

  # `case_when(...)` itself never reaches this entry's `parse`/generic
  # per-arg loop -- see `ir_parse_case_when()`, invoked directly from
  # `ir_parse_expr()`'s call-handling branch because its dots are `~`
  # formulas. This entry exists only so `ir_infer_type()` can look up
  # `type`/`lower` for the `ir_call("case_when", ...)` node
  # `ir_parse_case_when()` builds.
  "case_when" = list(
    arity = NA_integer_,
    parse = NULL,
    type = type_case_when,
    lower = list(handler = "case_when")
  ),

  # `coalesce(...)`: variadic, at least one argument.
  "coalesce" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) < 1) {
        return(NULL)
      }
      values <- ir_resolve_value_nas(args, schema)
      ir_call("coalesce", values)
    },
    type = type_coalesce,
    lower = list(handler = "coalesce")
  ),

  # `na_if(x, y)`: output keeps x's own type (type_na_if); a STRING/
  # non-STRING mismatch between x and y is diagnosed here, at parse time,
  # with a message naming both sides and their types (mirroring dplyr's own
  # "Can't convert `y` <type> to match type of `x` <type>." wording) rather
  # than surfacing as a raw cudf type error later.
  "na_if" = list(
    arity = 2L,
    parse = function(args, schema) {
      resolved <- ir_resolve_value_nas(args, schema)
      x <- resolved[[1]]
      y <- resolved[[2]]
      # unname(): a `col` node's type comes back from ir_infer_type() as
      # schema$types[idx] -- a NAMED single-element subset when
      # current_schema()'s own $types vector is named (as it is for any
      # real tbl_gpu), and identical() treats a "names" attribute as
      # significant (identical(c(s = "STRING"), "STRING") is FALSE even
      # though the values are equal). Same landmine, same fix, as
      # check_filter_comparison_types()'s existing unname() call
      # (R/filter.R) -- without it, every na_if() STRING/non-STRING
      # mismatch silently skipped this check and only surfaced as a raw
      # cudf cast error at GPU-execution time instead.
      x_type <- unname(ir_infer_type(x, schema))
      y_type <- unname(ir_infer_type(y, schema))
      if (identical(x_type, "STRING") != identical(y_type, "STRING")) {
        stop("na_if(): can't convert `y` (", y_type, ") to match the type of ",
             "`x` (", x_type, ").", call. = FALSE)
      }
      ir_call("na_if", list(x, y))
    },
    type = type_na_if,
    lower = list(handler = "na_if")
  )
)
