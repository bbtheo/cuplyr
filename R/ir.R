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
  # An empty/missing argument slot (e.g. the elided row-index in `x[, 1]`)
  # is a special R sentinel: forcing it (is.symbol()/is.call(), even just
  # reading it as a plain value) always raises "argument ... is missing,
  # with no default", regardless of context -- rlang::is_missing() is the
  # one check that inspects it safely. Found via Phase 11 task L1 test
  # coverage (mutate(y = scale(x)[, 1]), which reaches this walker once
  # `[` isn't found in ir_call_registry) -- a pre-existing landmine, not
  # something L1 itself introduces (see the identical guard in
  # R/across.R's raw-expression walkers).
  if (rlang::is_missing(expr)) {
    return(FALSE)
  }
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
  # Case 0: a quosure embedded INSIDE the expression tree (as opposed to
  # the top-level quosure ir_parse_quo() itself unwraps) -- unwrap to its
  # own expr/env and recurse. This arises whenever a language tree is built
  # by unquoting an already-captured quosure into another call, e.g.
  # `rlang::expr(base::sum(!!wt, na.rm = TRUE))` where `wt <-
  # rlang::enquo(wt_arg)` (exactly what real, unmodified
  # `dplyr:::tally_n()` does for `add_tally(wt = )`/`add_count(wt = )`'s
  # internal `mutate()` call, Phase 5 task W9) -- `!!wt` splices the
  # quosure OBJECT itself into that argument position (a one-sided
  # formula, `is.call()`-true but with head `` ` ~` ``, not the wrapped
  # function's own name), not just its bare expression. Ordinary tidy-eval
  # (`rlang::eval_tidy()`) resolves nested quosures like this
  # transparently; this walker doesn't use tidy-eval for structural
  # dispatch (it inspects the raw language tree directly), so it needs this
  # explicit unwrap instead.
  if (rlang::is_quosure(expr)) {
    return(ir_parse_expr(rlang::quo_get_expr(expr), rlang::quo_get_env(expr), schema, allow_vector))
  }

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

    # Phase 4 wave 2: case_match()/recode_values()/replace_values()/
    # replace_when()'s dots are ALSO `~` formulas (case_match()/
    # recode_values() additionally take a leading `.x`/`x` vector argument
    # before the formulas), and when_all()/when_any()'s dots need their
    # `na_rm=`/`size=` named controls pulled out before the remaining dots
    # are parsed as ordinary logical expressions -- none of that fits the
    # generic per-arg parse loop below, so all are special-cased here
    # exactly like case_when() above. See each parse function's own docs.
    if (identical(fn_name, "case_match")) {
      return(ir_parse_case_match(expr, env, schema))
    }
    if (identical(fn_name, "recode_values")) {
      return(ir_parse_recode_values(expr, env, schema))
    }
    if (identical(fn_name, "replace_values")) {
      return(ir_parse_replace_values(expr, env, schema))
    }
    if (identical(fn_name, "replace_when")) {
      return(ir_parse_replace_when(expr, env, schema))
    }
    if (fn_name %in% c("when_all", "when_any")) {
      return(ir_parse_when_reduce(expr, env, schema, fn_name))
    }

    # Phase 5 (window functions): lag()/lead() (n=/default=/order_by= named
    # controls), first()/last()/nth() (order_by=/default=/na_rm=), and
    # ntile() (x= is optional with a non-trivial default, and its
    # "supplied or not" distinction changes the desugar -- see
    # ir_parse_ntile()'s own docs) all need real argument-name-aware
    # matching the generic per-arg registry loop below can't do (it only
    # ever sees positional args, see ir_parse_shift_call()'s own docs) --
    # so, like case_when()/case_match()/etc. above, these are special-cased
    # here, before registry dispatch.
    if (fn_name %in% c("lag", "lead")) {
      return(ir_parse_shift_call(expr, env, schema, fn_name))
    }
    if (fn_name %in% c("first", "last", "nth")) {
      return(ir_parse_first_last_nth(expr, env, schema, fn_name))
    }
    if (identical(fn_name, "ntile")) {
      return(ir_parse_ntile(expr, env, schema))
    }
    if (identical(fn_name, "order_by")) {
      return(ir_parse_order_by_call(expr, env, schema))
    }
    if (identical(fn_name, "with_order")) {
      return(ir_parse_with_order_call(expr, env, schema))
    }

    # Phase 5, task W9: `sum()`'s ordinary registry entry only accepts the
    # bare 1-arg form (see ir_parse_sum_call()'s own docs for why the 2-arg
    # `sum(x, na.rm = TRUE)` shape needs real argument-NAME inspection the
    # generic per-arg registry loop below can't do) -- special-cased here
    # alongside lag()/lead()/first()/last()/nth()/ntile() above, for the
    # same reason.
    if (identical(fn_name, "sum")) {
      return(ir_parse_sum_call(expr, env, schema))
    }

    # Rolling windows (scratchpad/todo.md "Rolling-window benchmark
    # target"): cuplyr's own roll_*() names, and an alias for
    # data.table::froll*() (verified against benchmark_rolling.R's actual
    # cuplyr probe call shape) -- both need real argument-name matching
    # (na.rm=/fill=/align=/... can be omitted or out of position), so both
    # get dedicated dispatch like lag()/lead()/ntile() above. See the
    # file-level comment above ir_parse_roll_call()'s definition.
    if (fn_name %in% c("roll_mean", "roll_sum", "roll_min", "roll_max", "roll_sd", "roll_median")) {
      return(ir_parse_roll_call(expr, env, schema, fn_name))
    }
    if (fn_name %in% c("frollmean", "frollsum", "frollmin", "frollmax", "frollsd", "frollmedian")) {
      return(ir_parse_froll_call(expr, env, schema, fn_name))
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
# Phase 4 wave 2: case_match()/recode_values()/replace_values()/
# replace_when()/when_all()/when_any() -- verified empirically against
# dplyr 1.2.1 first (see the Phase 4 wave 2 task notes/scratchpad/todo.md).
#
# case_match()/recode_values(): "match .x/x by VALUE against a set of old
# values" desugars EXACTLY into `%in%` conditions feeding a `case_when()`
# chain -- `case_match(x, c(1,3) ~ "a", .default = "z")` is, at the IR
# level, indistinguishable from `case_when(x %in% c(1,3) ~ "a", .default =
# "z")`. This reuses `%in%`'s own NA-matching semantics verbatim (`NA %in%
# NA` is TRUE in base R, so `case_match(x, NA ~ "unknown", ...)` correctly
# replaces x's own NA elements) -- which required a real pre-existing `%in%`
# handler bug fix (a bare scalar `NA` literal RHS, e.g. `x %in% NA`, was
# wrongly routed into the "empty RHS set" always-FALSE shortcut instead of
# matching x's NA rows; see `src/expr_eval.hpp`'s `%in%` handler). Because
# both fully reduce to an existing "case_when" node, neither needs its own
# registry entry, `type` function, or C++ handler.
#
# replace_values()/replace_when(): same idea (value-match / boolean-cond
# clauses respectively) but TYPE STABLE to `x` (dplyr's own docs: "type
# stable, unlike case_match()") and with `x` itself as the implicit,
# non-overridable "default" for unmatched rows -- case_when()'s own type
# rule (vctrs common-type unify across every value + default) is WRONG for
# this, so both lower through a dedicated "replace_when" IR op instead (see
# `type_replace_when()` below and the `replace_when` handler in
# `src/expr_eval.hpp`).
#
# when_all()/when_any(): `when_any(x, y, z)` is `x | y | z` generalized to N
# inputs (`when_all()` the `&` analogue) -- fully desugars into a folded
# chain of the existing `|`/`&` ops, needing no IR op of its own at all.
# `na_rm = TRUE` does NOT simply drop NAs from the *result*; verified
# empirically it treats NA as each operator's identity element instead
# (`when_any(NA, NA, na_rm = TRUE)` is `FALSE`, not `NA`; `when_all(NA, NA,
# na_rm = TRUE)` is `TRUE`) -- implemented by `coalesce()`-ing every input
# against that identity value before folding, reusing coalesce()'s existing
# handler too.
#
# Every one of these follows wave 1's own scope-narrowing precedent: any
# named argument this wave doesn't implement (recode_values()'s
# `from=`/`to=`/`unmatched=`/`ptype=`, case_match()'s `.ptype=`,
# replace_values()'s `from=`/`to=`, when_all()/when_any()'s `size=`) being
# present at all makes the parse function return `NULL` (caller falls
# back/errors), exactly like case_when()'s own `.ptype=`/`.size=`/
# `.unmatched=` handling above.
# -----------------------------------------------------------------------------

# na_if()'s output keeps `x`'s own type; replace_when()/replace_values() are
# the same idea generalized to N clauses -- the result is always `x`'s own
# type (arg 1), never a vctrs-style unify across the replacement values.
type_replace_when <- function(arg_types) {
  arg_types[1]
}

#' Shared clause-parsing helper for case_match()/recode_values() (Phase 4 wave 2)
#'
#' Both functions share an identical clause shape once "match by value" is
#' expressed as `%in%`: `f(x, old_values_1 ~ new_value_1, ..., <default_key>
#' = value)`. Builds `x %in% old_values_i` as each clause's condition
#' (reusing the existing `%in%` op verbatim, including its NA-matching
#' semantics) and hands the interleaved cond/value list to `ir_call(
#' "case_when", ...)` directly -- case_match()/recode_values() are, at the
#' IR level, indistinguishable from case_when() once desugared this way.
#'
#' Only the two-sided-formula clause interface is implemented this wave:
#' recode_values()'s `from=`/`to=`/`unmatched=`/`ptype=` vector-lookup-table
#' interface, and case_match()'s `.ptype=`, are NOT implemented -- ANY named
#' argument other than `default_key` present in the call makes this return
#' `NULL` (caller falls back).
#'
#' A clause's LHS ("old value(s)") must resolve to a literal (a constant, a
#' `c(...)` vector, or an environment lookup) -- an LHS referencing a column
#' is unsupported this wave (mirrors `%in%`'s own RHS restriction, since
#' that's exactly the op this desugars into). A STRING-vs-non-STRING type
#' mismatch between `x` and a clause's LHS is diagnosed here (mirroring
#' `na_if()`'s own STRING compatibility check) rather than surfacing as a
#' raw cudf cast error at lowering time.
#'
#' @param expr The raw `case_match(...)`/`recode_values(...)` call
#' @param env The quosure's environment
#' @param schema List with `names`/`types`
#' @param default_key `".default"` (case_match) or `"default"`
#'   (recode_values) -- the one named control argument this wave supports
#' @return An `ir_call("case_when", ...)` node, or `NULL` if unsupported
#' @keywords internal
ir_parse_value_match_clauses <- function(expr, env, schema, default_key) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) < 1) {
    return(NULL)  # need at least `.x`/`x`
  }
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }

  # First positional (unnamed) argument is .x/x -- case_match()/
  # recode_values() both always take it first and unnamed in every
  # realistic call site (the same simplifying assumption if_else()/
  # case_when() already make for their own positional args).
  if (nzchar(arg_names[1])) {
    return(NULL)
  }
  x_raw <- args_raw[[1]]
  rest_raw <- args_raw[-1]
  rest_names <- arg_names[-1]

  unsupported_named <- setdiff(rest_names[nzchar(rest_names)], default_key)
  if (length(unsupported_named) > 0) {
    return(NULL)
  }

  default_pos <- which(rest_names == default_key)
  default_raw <- if (length(default_pos) == 1) rest_raw[[default_pos]] else NULL
  clause_raw <- if (length(default_pos) == 1) rest_raw[-default_pos] else rest_raw

  if (length(clause_raw) == 0) {
    return(NULL)  # nothing to match against
  }

  x_ir <- ir_parse_expr(x_raw, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))

  conds <- vector("list", length(clause_raw))
  vals <- vector("list", length(clause_raw))

  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)  # not a two-sided formula: unsupported shape
    }
    lhs_ir <- ir_parse_expr(f[[2]], env, schema, allow_vector = TRUE)
    if (is.null(lhs_ir) || !identical(lhs_ir$kind, "lit")) {
      return(NULL)  # non-literal (e.g. column-referencing) LHS: unsupported
    }
    if (!isTRUE(lhs_ir$na)) {
      lhs_type <- unname(ir_infer_type(lhs_ir, schema))
      if (identical(x_type, "STRING") != identical(lhs_type, "STRING")) {
        stop("case_match()/recode_values(): can't convert the matched value (",
             lhs_type, ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    conds[[i]] <- ir_call("%in%", list(x_ir, lhs_ir))
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

#' Parse `case_match(.x, old_values ~ new_value, ..., .default=, .ptype=)`
#'
#' See [ir_parse_value_match_clauses()] -- `case_match()` is deprecated in
#' dplyr 1.2.1 in favor of `recode_values()`/`replace_values()` (verified:
#' calling it emits a lifecycle warning), but is still exported and
#' functional, and is explicitly in this wave's scope, so it's implemented
#' identically to `recode_values()`'s formula interface, just with `.default`
#' as the named key instead of `default`.
#'
#' @keywords internal
ir_parse_case_match <- function(expr, env, schema) {
  ir_parse_value_match_clauses(expr, env, schema, default_key = ".default")
}

#' Parse `recode_values(x, old_values ~ new_value, ..., default=, unmatched=, ptype=)`
#'
#' See [ir_parse_value_match_clauses()]. `from=`/`to=`/`unmatched=`/`ptype=`
#' are not implemented this wave.
#'
#' @keywords internal
ir_parse_recode_values <- function(expr, env, schema) {
  ir_parse_value_match_clauses(expr, env, schema, default_key = "default")
}

#' Parse `replace_values(x, old_values ~ new_value, ..., from=, to=)`
#'
#' Value-match clauses like [ir_parse_value_match_clauses()] (`x %in%
#' old_values_i`), but TYPE STABLE: the result always keeps `x`'s own type,
#' and unmatched rows keep `x` itself (there is no `.default=`/`default=`
#' argument at all) -- expressed as a dedicated `ir_call("replace_when",
#' ...)` node (see `type_replace_when()` and this file's module comment
#' above) rather than reusing `case_when`'s node, since case_when's own
#' common-type-unify rule is wrong here.
#'
#' Only the two-sided-formula clause interface is implemented this wave:
#' `from=`/`to=` (the vector-lookup-table interface) being present at all
#' makes this return `NULL` (fall back).
#'
#' @keywords internal
ir_parse_replace_values <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) < 2) {
    return(NULL)  # need x + at least one clause
  }
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }
  if (nzchar(arg_names[1])) {
    return(NULL)
  }
  if (any(nzchar(arg_names[-1]))) {
    return(NULL)  # from=/to=: not implemented this wave
  }

  x_raw <- args_raw[[1]]
  clause_raw <- args_raw[-1]

  x_ir <- ir_parse_expr(x_raw, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))

  interleaved <- vector("list", length(clause_raw) * 2L)
  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)
    }
    lhs_ir <- ir_parse_expr(f[[2]], env, schema, allow_vector = TRUE)
    if (is.null(lhs_ir) || !identical(lhs_ir$kind, "lit")) {
      return(NULL)
    }
    if (!isTRUE(lhs_ir$na)) {
      lhs_type <- unname(ir_infer_type(lhs_ir, schema))
      if (identical(x_type, "STRING") != identical(lhs_type, "STRING")) {
        stop("replace_values(): can't convert the matched value (", lhs_type,
             ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    if (!(identical(val_ir$kind, "lit") && isTRUE(val_ir$na) && is.null(val_ir$type))) {
      val_type <- unname(ir_infer_type(val_ir, schema))
      if (identical(x_type, "STRING") != identical(val_type, "STRING")) {
        stop("replace_values(): can't convert a replacement value (", val_type,
             ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    interleaved[[2 * i - 1]] <- ir_call("%in%", list(x_ir, lhs_ir))
    interleaved[[2 * i]] <- val_ir
  }

  ir_call("replace_when", c(list(x_ir), interleaved), meta = list(n_when = length(clause_raw)))
}

#' Parse `replace_when(x, cond1 ~ val1, ..., condN ~ valN)`
#'
#' Boolean-condition clauses -- exactly `case_when()`'s own clause shape,
#' see [ir_parse_case_when()] -- but TYPE STABLE to `x`, with `x` itself as
#' the implicit, non-overridable final "default" (unmatched rows keep their
#' original `x` value; there is no `.default=`/`default=` argument at all).
#' Lowered via the same dedicated "replace_when" op as
#' [ir_parse_replace_values()].
#'
#' No named arguments are supported this wave (the real `replace_when()`
#' signature is just `replace_when(x, ...)` anyway) -- any named dot present
#' makes this return `NULL` (fall back).
#'
#' @keywords internal
ir_parse_replace_when <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) < 2) {
    return(NULL)  # need x + at least one clause
  }
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }
  if (nzchar(arg_names[1])) {
    return(NULL)
  }
  if (any(nzchar(arg_names[-1]))) {
    return(NULL)  # no named controls supported this wave
  }

  x_raw <- args_raw[[1]]
  clause_raw <- args_raw[-1]

  x_ir <- ir_parse_expr(x_raw, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))

  interleaved <- vector("list", length(clause_raw) * 2L)
  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)
    }
    cond_ir <- ir_parse_expr(f[[2]], env, schema)
    if (is.null(cond_ir)) {
      return(NULL)
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    if (!(identical(val_ir$kind, "lit") && isTRUE(val_ir$na) && is.null(val_ir$type))) {
      val_type <- unname(ir_infer_type(val_ir, schema))
      if (identical(x_type, "STRING") != identical(val_type, "STRING")) {
        stop("replace_when(): can't convert a replacement value (", val_type,
             ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    interleaved[[2 * i - 1]] <- cond_ir
    interleaved[[2 * i]] <- val_ir
  }

  ir_call("replace_when", c(list(x_ir), interleaved), meta = list(n_when = length(clause_raw)))
}

#' Parse `when_all(..., na_rm=FALSE, size=NULL)` / `when_any(...)`
#'
#' See this file's module comment above for the full empirical rationale.
#' Fully desugars at parse time into a right-folded chain of the existing
#' `|`/`&` ops -- no new IR op, registry entry, or C++ handler needed.
#'
#' `size=` (only meaningful for the zero-input edge case) is NOT implemented
#' this wave -- its presence, or zero dots, makes this return `NULL`.
#'
#' @param fn_name `"when_all"` or `"when_any"`
#' @keywords internal
ir_parse_when_reduce <- function(expr, env, schema, fn_name) {
  args_raw <- as.list(expr)[-1]
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }

  unsupported_named <- setdiff(arg_names[nzchar(arg_names)], "na_rm")
  if (length(unsupported_named) > 0) {
    return(NULL)  # size=: not implemented this wave
  }

  na_rm_pos <- which(arg_names == "na_rm")
  na_rm <- FALSE
  if (length(na_rm_pos) == 1) {
    na_rm_raw <- args_raw[[na_rm_pos]]
    na_rm_val <- tryCatch(
      rlang::eval_tidy(rlang::new_quosure(na_rm_raw, env)),
      error = function(e) NULL
    )
    if (!is.logical(na_rm_val) || length(na_rm_val) != 1 || is.na(na_rm_val)) {
      return(NULL)
    }
    na_rm <- na_rm_val
  }

  dot_raw <- if (length(na_rm_pos) == 1) args_raw[-na_rm_pos] else args_raw
  dot_names <- if (length(na_rm_pos) == 1) arg_names[-na_rm_pos] else arg_names
  if (any(nzchar(dot_names))) {
    return(NULL)  # dots should be positional
  }
  if (length(dot_raw) == 0) {
    return(NULL)  # zero-input edge case: not implemented this wave
  }

  fold_op <- if (identical(fn_name, "when_all")) "&" else "|"
  identity_value <- identical(fn_name, "when_all")  # TRUE for &, FALSE for |

  parsed <- vector("list", length(dot_raw))
  for (i in seq_along(dot_raw)) {
    node <- ir_parse_expr(dot_raw[[i]], env, schema)
    if (is.null(node)) {
      return(NULL)
    }
    if (na_rm) {
      node <- ir_call("coalesce", list(node, ir_lit_from_r(identity_value)))
    }
    parsed[[i]] <- node
  }

  Reduce(function(a, b) ir_call(fold_op, list(a, b)), parsed)
}

# -----------------------------------------------------------------------------
# Phase 5 (window functions, scratchpad/phase5_window_design.md section 1.1):
# lag()/lead(), first()/last()/nth(), ntile(), order_by()/with_order() --
# parsed via dedicated top-level dispatch (see the `fn_name %in% c(...)`
# checks in ir_parse_expr()'s call-handling branch above), not the
# registry's generic per-arg loop, because they have named controls that
# can be supplied out of position or omitted entirely (`lag(x, default =
# 0)` skipping `n=`, `nth(x, k, order_by = y)`, ...) -- something the
# generic loop can't express, since it only ever sees `as.list(expr)[-1]`
# positionally (see ir_parse_case_when()'s sibling functions above for the
# file's established "named args need dedicated dispatch" precedent).
# Argument matching itself reuses rlang::call_match() (rlang >= 1.1.0,
# already a hard dependency) against a small prototype function whose
# formals mirror the real dplyr signature -- this gets R's own
# argument-matching semantics (named, positional, or mixed, in any order)
# for free, rather than hand-rolling it.
#
# Phase 5 W4: `order_by=` (on `lag()`/`lead()`/`first()`/`last()`/`nth()`)
# and the standalone `order_by()`/`with_order()` wrappers are now
# implemented, but ONLY when the order argument resolves to a bare column
# reference (or `desc(column)`/`-column`) via `ir_parse_order_ref()` --
# a general expression (e.g. `order_by(a + b, ...)`) is still unsupported
# and falls back, same as before.
# -----------------------------------------------------------------------------

#' Evaluate a raw (unparsed) expression as a constant in `env`
#'
#' No data mask is supplied, so a genuine column reference (or anything
#' else that isn't resolvable from `env` alone) raises an ordinary R
#' error, caught here and reported as "not a constant" (`ok = FALSE`)
#' rather than propagating -- the same "no data mask" trick
#' `resolve_slice_dots()` (`R/slice.R`) already relies on to distinguish
#' plain R values from column references.
#'
#' @param raw_expr A language object (not a quosure)
#' @param env The environment to evaluate against
#' @return `list(ok = <logical>, value = <the evaluated R value, or NULL>)`
#' @keywords internal
ir_eval_constant <- function(raw_expr, env) {
  tryCatch(
    list(ok = TRUE, value = rlang::eval_tidy(rlang::new_quosure(raw_expr, env))),
    error = function(e) list(ok = FALSE, value = NULL)
  )
}

#' Parse `lag(x, n = 1L, default = NULL, order_by = NULL)` / `lead(...)`
#'
#' Builds a window-class `ir_call(fn_name, list(x_ir), meta = list(n = ,
#' default = , order_override = ))` node (never lowered directly -- see the
#' [ir_call_registry] entry / [ir_bind()]'s guard). `n` must be a
#' non-negative whole-number constant (dplyr itself requires this);
#' `default`, if supplied, must be a CONSTANT expression -- evaluable via
#' [ir_eval_constant()] with no data mask, so `default = -1`, `default = 2L
#' + 3L`, or any other column-free expression all work (not just a bare
#' literal token), but `default = first(x)` (a real, column-referencing
#' dplyr default) is not implemented this wave.
#'
#' `order_by=`, when supplied, must resolve via [ir_parse_order_ref()] (a
#' bare column, `desc(column)`, or `-column`); anything else (a general
#' expression) is not implemented this wave and returns `NULL` (Phase 5
#' W4: `order_by=` re-frames this spec's own frame exactly like
#' `order_by()`/`with_order()` would if wrapped around the same call --
#' see [apply_order_override()]'s docs -- so `lag(x, order_by = y)` and
#' `order_by(y, lag(x))` produce the same `meta$order_override`).
#'
#' @keywords internal
ir_parse_shift_call <- function(expr, env, schema, fn_name) {
  proto <- function(x, n = 1L, default = NULL, order_by = NULL) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  order_override <- NULL
  if (!is.null(present_args$order_by)) {
    order_override <- ir_parse_order_ref(present_args$order_by, schema)
    if (is.null(order_override)) {
      return(NULL)  # unsupported order_by= shape: not implemented this wave
    }
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  n_value <- 1L
  if (!is.null(present_args$n)) {
    n_eval <- ir_eval_constant(present_args$n, env)
    n_val <- n_eval$value
    if (!n_eval$ok || !is.numeric(n_val) || length(n_val) != 1 || is.na(n_val) ||
        n_val != round(n_val) || n_val < 0) {
      return(NULL)
    }
    n_value <- as.integer(n_val)
  }

  default_ir <- NULL
  if (!is.null(present_args$default)) {
    default_eval <- ir_eval_constant(present_args$default, env)
    if (!default_eval$ok) {
      return(NULL)  # non-constant default: not implemented this wave
    }
    default_val <- default_eval$value
    if (!is.null(default_val)) {
      if (!is.atomic(default_val) || length(default_val) != 1) {
        return(NULL)  # non-scalar default: not implemented this wave
      }
      default_ir <- ir_lit_from_r(default_val)
    }
  }

  ir_call(fn_name, list(x_ir), meta = list(n = n_value, default = default_ir,
                                            order_override = order_override))
}

#' Parse `first(x, ...)` / `last(x, ...)` / `nth(x, n, ...)`
#'
#' The bare `x` (and, for `nth()`, a constant-integer `n`) shape, plus
#' `order_by=` (Phase 5 W4: see [ir_parse_order_ref()]/
#' [apply_order_override()] -- `first(x, order_by = y)` re-frames this
#' spec's frame to be ordered by `y`, matching dplyr's own
#' "first/last/nth row of the group IN `order_by`'s ORDER, not the
#' group's current order" semantics, verified empirically against dplyr
#' 1.2.1). `default=`/`na_rm=` being present at all -- even at their own
#' default value -- still makes this return `NULL` (not implemented this
#' wave). Builds a window-class `ir_call(fn_name, list(x_ir), meta =
#' list(order_override = ))` node (`nth()` additionally carries `n = <int>`,
#' the (1-based, possibly negative) element position).
#'
#' @keywords internal
ir_parse_first_last_nth <- function(expr, env, schema, fn_name) {
  proto <- if (identical(fn_name, "nth")) {
    function(x, n, order_by = NULL, default = NULL, na_rm = FALSE) NULL
  } else {
    function(x, order_by = NULL, default = NULL, na_rm = FALSE) NULL
  }

  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args$default) || !is.null(present_args$na_rm)) {
    return(NULL)  # not implemented this wave
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  order_override <- NULL
  if (!is.null(present_args$order_by)) {
    order_override <- ir_parse_order_ref(present_args$order_by, schema)
    if (is.null(order_override)) {
      return(NULL)  # unsupported order_by= shape: not implemented this wave
    }
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  if (identical(fn_name, "nth")) {
    if (is.null(present_args$n)) {
      return(NULL)
    }
    n_eval <- ir_eval_constant(present_args$n, env)
    n_val <- n_eval$value
    if (!n_eval$ok || !is.numeric(n_val) || length(n_val) != 1 || is.na(n_val) ||
        n_val != round(n_val) || n_val == 0) {
      return(NULL)
    }
    return(ir_call("nth", list(x_ir),
                    meta = list(n = as.integer(n_val), order_override = order_override)))
  }

  ir_call(fn_name, list(x_ir), meta = list(order_override = order_override))
}

#' Parse a window "order by" reference (`order_by=` argument, or
#' `order_by()`/`with_order()`'s own first argument) into a schema column
#' reference
#'
#' Recognizes a bare column name, `desc(column)`/`dplyr::desc(column)`, or
#' `-column` (dplyr's own `desc()` shorthand, also usable directly as an
#' `arrange()`/`order_by()` sort key) and resolves it against `schema`.
#' Anything else -- a general expression such as `order_by(a + b, ...)` --
#' is not implemented this wave (mirroring `R/window.R`'s rank-family
#' design note: only a bare, already-existing column doubles as an order
#' key so far, no `..winarg..`-style hoisting for order columns yet) and
#' returns `NULL`, same as any other unsupported IR shape (callers fall
#' back).
#'
#' @param expr A language object (not a quosure)
#' @param schema List with `names`/`types`
#' @return `list(col = <chr1>, desc = <lgl1>)`, or `NULL`
#' @keywords internal
ir_parse_order_ref <- function(expr, schema) {
  if (is.symbol(expr)) {
    col <- as.character(expr)
    if (!(col %in% schema$names)) {
      return(NULL)
    }
    return(list(col = col, desc = FALSE))
  }

  if (is.call(expr) && length(expr) == 2) {
    head <- expr[[1]]
    if (is.call(head) && length(head) == 3 && identical(head[[1]], as.symbol("::"))) {
      head <- head[[3]]  # strip namespace: dplyr::desc -> desc
    }
    if (is.symbol(head) && as.character(head) %in% c("desc", "-")) {
      inner <- expr[[2]]
      if (!is.symbol(inner)) {
        return(NULL)
      }
      col <- as.character(inner)
      if (!(col %in% schema$names)) {
        return(NULL)
      }
      return(list(col = col, desc = TRUE))
    }
  }

  NULL
}

#' Attach an `order_by()`/`with_order()`/`order_by=`-supplied ordering to a
#' directly-wrapped window call
#'
#' No-op (returns `ir_node` unchanged) unless `ir_node` is ITSELF a single
#' window-class `ir_call` node -- e.g. `order_by(y, x + 1)` (an
#' order-independent expression) or `order_by(y, cumsum(x) + lag(x))` (a
#' compound expression, not one bare window call) both pass `ir_node`
#' through untouched, matching dplyr's own behavior for the first case
#' (reordering has no observable effect on an order-independent
#' expression) and this wave's documented scope limit for the second (only
#' a SINGLE directly-wrapped window call gets the override; see
#' [ir_parse_order_by_call()]'s docs).
#'
#' Sets `ir_node$meta$order_override <- order_ref`, read by
#' `R/window.R`'s `extract_one_window_call()` in preference to a
#' rank-kind spec's usual "value column doubles as order column" default.
#'
#' Rejected outright (returns `NULL`, the file's usual "not implemented"
#' signal) when `ir_node` is an `"agg"`-kind call -- `mean()`/`sum()`/
#' `min()`/`max()`/`n()`/`sd()`/`var()`/`first()`/`last()`/`nth()`.
#' Verified empirically against dplyr 1.2.1: `order_by()`/`with_order()`
#' themselves ERROR for every one of these (not just the order-invariant
#' ones like `mean()`) -- `order_by(y, first(x))`,
#' `with_order(y, first, x)`, and even `order_by(y, n())` all raise
#' `vec_slice()`/argument-matching errors in real dplyr, because
#' `order_by()`/`with_order()` un-shuffle their wrapped call's result via
#' `vec_slice(result, order(order_by))`, which requires the result to have
#' ONE VALUE PER INPUT ROW -- true for the `"scan"`/`"shift"`/`"rank"`
#' kinds, never true for `"agg"` (a reduction). `first()`/`last()`/`nth()`'s
#' OWN `order_by=` ARGUMENT (handled directly by
#' `ir_parse_first_last_nth()`, never via this function) is a completely
#' different, genuinely-supported dplyr feature and is unaffected by this
#' rejection.
#'
#' A `"rank"`-kind call gets one of three treatments, also verified
#' empirically against dplyr 1.2.1:
#' \itemize{
#'   \item `row_number(x)` (a VALUE argument, ties = "sequential" broken by
#'     row position): the override genuinely changes the result (ties in
#'     `x` get broken by `order_ref`'s order instead of original row
#'     position -- confirmed empirically: `order_by(y, row_number(x))` on
#'     data with ties in `x` differs from bare `row_number(x)`). This
#'     would need a two-column rank frame (primary key `x`, tie-break key
#'     `order_ref`), not implemented this wave -- rejected (`NULL`) rather
#'     than silently producing dplyr's OWN tie-break order (original row
#'     position) when a different one was actually requested.
#'   \item `min_rank()`/`dense_rank()`/`percent_rank()`/`cume_dist()`
#'     (MIN/DENSE/MAX methods, none of which break ties by position at
#'     all): `order_by()`/`with_order()` has NO OBSERVABLE EFFECT on any of
#'     these (confirmed empirically: `order_by(y, min_rank(x))` is
#'     IDENTICAL to bare `min_rank(x)`, for any `y`) -- the override is
#'     silently dropped (`ir_node` returned unchanged, exactly like an
#'     order-independent plain expression).
#'   \item bare `row_number()` (0 args, purely position-based): real dplyr
#'     itself errors here (`with_order()`'s `x` argument has nothing to
#'     bind to, since `row_number()` takes none) -- this is NOT replicated;
#'     the override is silently dropped here too (same as the value-only
#'     rank functions above), a deliberate, harmless LOOSENING relative to
#'     dplyr (produces the same value bare `row_number()` would, rather
#'     than reproducing dplyr's own argument-binding quirk as an error).
#' }
#'
#' @param ir_node An IR node
#' @param order_ref A `list(col =, desc =)` as returned by
#'   [ir_parse_order_ref()]
#' @return `ir_node`, possibly with `meta$order_override` set, or `NULL`
#' @keywords internal
apply_order_override <- function(ir_node, order_ref) {
  if (!identical(ir_node$kind, "call")) {
    return(ir_node)
  }
  entry <- ir_call_registry[[ir_node$op]]
  if (is.null(entry) || is.null(entry$window)) {
    return(ir_node)
  }

  kind <- entry$window$kind
  if (identical(kind, "agg")) {
    return(NULL)
  }
  if (identical(kind, "rank")) {
    if (identical(ir_node$op, "row_number") && length(ir_node$args) > 0) {
      return(NULL)  # row_number(x): tie-break frame change not implemented this wave
    }
    return(ir_node)  # value-only rank fns (+ bare row_number()): no observable effect
  }

  ir_node$meta$order_override <- order_ref
  ir_node
}

#' Parse `order_by(order_by, call)`
#'
#' dplyr's `order_by(order_by, call)` re-evaluates `call` as if the data
#' were sorted by `order_by`, then un-sorts the result back to the
#' original row order (verified empirically against dplyr 1.2.1:
#' `order_by(x, cumsum(y))` on `y = c(3,1,4,1,5,9,2,6)` matches computing
#' `cumsum` after a STABLE ascending sort of `x`, then scattering back to
#' the original positions) -- exactly the "frame" concept
#' `R/window.R`'s `window_spec()` already carries via `order_cols`/
#' `order_desc`. Parses `order_by` into a column reference (see
#' [ir_parse_order_ref()]) and folds it into the parsed `call`'s own IR
#' node via [apply_order_override()].
#'
#' @keywords internal
ir_parse_order_by_call <- function(expr, env, schema) {
  proto <- function(order_by, call) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$order_by) || is.null(present_args$call)) {
    return(NULL)
  }

  order_ref <- ir_parse_order_ref(present_args$order_by, schema)
  if (is.null(order_ref)) {
    return(NULL)
  }

  call_ir <- ir_parse_expr(present_args$call, env, schema)
  if (is.null(call_ir)) {
    return(NULL)
  }

  apply_order_override(call_ir, order_ref)
}

#' Parse `with_order(order_by, fun, x)`
#'
#' dplyr 1.2.1's actual `with_order()` signature is `function(order_by,
#' fun, x, ...)` (verified empirically -- no `order_by_desc=` parameter in
#' this version; wrap `order_by` in `desc()` for descending order, exactly
#' like [order_by()]). `fun` is a BARE function reference (e.g. the
#' symbol `cumsum`, not a call): `with_order(order_by, fun, x)` means
#' `fun(x)`, evaluated as if the data were sorted by `order_by` (confirmed
#' empirically identical to `order_by(order_by, fun(x))`). Only the
#' 3-argument form (no extra `...` passed through to `fun`) is implemented
#' this wave. `fun` may be namespace-qualified (`dplyr::lag`) -- one leading
#' `pkg::` qualifier is stripped before the bare-symbol check, same
#' convention as `R/summarise.R`'s `strip_ns_prefix()`/`decompose_agg_call()`
#' -- since `dplyr` is not necessarily attached wherever `with_order()` is
#' called from, `fun = lag` would otherwise silently resolve to `stats::lag`
#' (a real footgun, caught empirically while writing this wave's oracle
#' tests). Reconstructs the equivalent call and delegates to the same
#' [apply_order_override()] [ir_parse_order_by_call()] uses.
#'
#' @keywords internal
ir_parse_with_order_call <- function(expr, env, schema) {
  proto <- function(order_by, fun, x) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$order_by) || is.null(present_args$fun) || is.null(present_args$x)) {
    return(NULL)
  }

  fun_expr <- present_args$fun
  if (is.call(fun_expr) && length(fun_expr) == 3 && identical(fun_expr[[1]], as.symbol("::"))) {
    fun_expr <- fun_expr[[3]]  # strip namespace: dplyr::lag -> lag
  }
  if (!is.symbol(fun_expr)) {
    return(NULL)  # a non-bare-symbol `fun` (e.g. an anonymous function): not implemented this wave
  }

  order_ref <- ir_parse_order_ref(present_args$order_by, schema)
  if (is.null(order_ref)) {
    return(NULL)
  }

  reconstructed <- as.call(list(fun_expr, present_args$x))
  call_ir <- ir_parse_expr(reconstructed, env, schema)
  if (is.null(call_ir)) {
    return(NULL)
  }

  apply_order_override(call_ir, order_ref)
}

# -----------------------------------------------------------------------------
# Rolling windows (Theo-requested benchmark task, scratchpad/todo.md
# "Rolling-window benchmark target"): a fifth window kind, `"rolling"`,
# alongside shift/rank/scan/agg. Surfaced under two call shapes, both
# needing real argument-NAME matching (same reason as lag()/lead()/ntile()
# above -- na.rm=/fill=/align= are named controls that can be omitted or
# supplied out of position), so both get dedicated top-level dispatch
# rather than the generic per-arg registry loop:
#   * cuplyr's own names -- `roll_mean(x, n, na.rm = FALSE)`,
#     `roll_sum()`/`roll_min()`/`roll_max()`/`roll_sd()`/`roll_median()` --
#     RcppRoll/slider-adjacent naming picked as the dplyr-ecosystem-idiomatic
#     surface (see ir_parse_roll_call()).
#   * an ALIAS for `data.table::frollmean(x, n, ...)` and its five
#     siblings (`frollsum`/`frollmin`/`frollmax`/`frollsd`/`frollmedian`) --
#     this is the exact call shape `benchmark/benchmark_rolling.R`'s cuplyr
#     probe constructs (`rlang::call2(paste0("froll", workload), sym("x"),
#     window, .ns = "data.table")`), verified by reading that script before
#     picking this alias surface. `pkg::fn(...)` is already stripped to a
#     bare `fn_name` by `ir_parse_expr()`'s call-head handling before any
#     dispatch runs (`data.table::frollmean` arrives here as plain
#     `"frollmean"`), so `ir_parse_froll_call()` only has to translate
#     data.table's richer argument set (`fill=`/`algo=`/`align=`/`hasNA=`/
#     `adaptive=`) down to the same `list(n=, na_rm=)` meta shape, rejecting
#     (NULL, i.e. "not implemented this wave, fall back") any non-default
#     `align=`/`fill=`/`adaptive=` -- this task is right-aligned-only, per
#     scratchpad/todo.md's explicit scope cut. `algo=`/`hasNA=` are accepted
#     but ignored: both are pure performance hints in data.table (exact vs.
#     approximate *computation strategy*, and a "does x contain NA at all"
#     hint respectively) that never change the correct *result*, and cuplyr
#     always computes the exact result regardless.
#
# Both shapes desugar to the SAME five (well, six, see roll_median below)
# canonical registry ops (`"roll_mean"`/`"roll_sum"`/`"roll_min"`/
# `"roll_max"`/`"roll_sd"`/`"roll_median"`), reusing `window_spec()`'s
# EXISTING `n`/`na_rm` fields verbatim (no new field needed: `n` already
# carries lag()/lead()'s shift amount and nth()'s element position, `na_rm`
# already threads through generically in R/window.R's
# extract_one_window_call() even though no pre-existing window op had set
# it to anything but FALSE before this task) -- see
# scratchpad/phase5_window_design.md section 2's `window_spec()` field
# docs, all reused unchanged.
#
# Empirically verified (a standalone probe against this environment's
# installed cudf 25.12, NOT re-derived from the header comments alone,
# since "min_periods" is ambiguous between "row count in window" and
# "valid (non-null) count in window" until tested):
#   * `cudf::rolling_window(x, preceding = w, following = 0, min_periods =
#     w, agg)` reproduces R's na.rm=FALSE poisoning FOR FREE, with no
#     separate is-null-indicator masking pass at all (unlike the W4
#     cum*/rank sticky-NA fix this file's window entries otherwise mirror):
#     the window is a FIXED w-row range, so "w valid values required"
#     among exactly w physical rows already means "zero nulls tolerated" --
#     confirmed on `x = c(1,2,NA,4,5)`, w=3: every output is NA the moment
#     the row-range includes the NA, matching data.table's own
#     `frollsum(x, 3, na.rm = FALSE)`.
#   * na.rm=TRUE is NOT simply "cudf's own default behavior" (cudf rolling
#     aggregations always skip nulls when computing, regardless of
#     min_periods): using `min_periods = 1` alone would incorrectly
#     compute a value for the leading (w-1) rows too, using FEWER than w
#     physical rows -- confirmed empirically that real
#     `data.table::frollmean(x, n, na.rm = TRUE)` still keeps exactly the
#     leading (n-1) positions NA (a ROW-COUNT requirement, independent of
#     na.rm), and only skips individual NA *values* once a window has
#     become physically complete. `src/ops_window.cpp`'s rolling branch
#     therefore computes the aggregate with `min_periods = 1` (skip nulls,
#     as low a valid-count bar as possible) and separately masks to NA any
#     row whose PHYSICAL row-count-in-window (a second rolling COUNT_ALL
#     pass, `null_policy::INCLUDE`, `min_periods = 1`, which counts rows
#     regardless of their own nullity) is `< w` -- verified bit-for-bit
#     against `data.table::frollmean(c(1,2,NA,4,5), 3, na.rm = TRUE)` ==
#     `c(NA, NA, 1.5, 3, 4.5)`.
#   * Grouped rolling comes "almost free": `cudf::grouped_rolling_window()`
#     clamps window boundaries at GROUP edges (not just table edges) when
#     given a `group_keys` table, and this file's existing frame
#     architecture already produces a stable-sorted-by-group-only `work`
#     view (no order_cols for a rolling frame, exactly like the scan
#     family) whose ROW ORDER is the group's original relative order --
#     exactly what `grouped_rolling_window()` requires ("presorted by the
#     group_key values"). `src/ops_window.cpp` therefore uses
#     `grouped_rolling_window()` UNIFORMLY for both the truly-grouped and
#     the ungrouped case (the ungrouped case's substituted constant
#     zero-key column is trivially "one sorted group"), the same "one code
#     path via a substituted key column" idiom this file already uses for
#     the scan/rank/agg families -- no separate `cudf::rolling_window()`
#     call needed at all.
#   * `cudf::make_median_aggregation<T>()` has NO `rolling_aggregation`
#     explicit template instantiation in this environment's compiled
#     `libcudf.so` (confirmed via `nm -DC libcudf.so | grep
#     make_median_aggregation`: only `aggregation`/`reduce_aggregation`/
#     `groupby_aggregation` instantiations exist) -- MEDIAN is genuinely
#     unsupported by `cudf::rolling_window`/`grouped_rolling_window` in
#     cudf 25.12, not merely undocumented. `roll_median()` is therefore
#     registered (so it parses, types, and participates in window
#     decomposition identically to its five siblings) but
#     `window_spec_lowerable()` (R/lower.R) explicitly excludes it, routing
#     any `roll_median()`/`data.table::frollmedian()` mutate() call through
#     the existing `mutate_window_fallback()` CPU path -- which needs
#     `roll_median()` (cuplyr's own name) to be a REAL, correct, exported R
#     function for that CPU re-run to work at all; see `R/roll.R`.
#   * MEAN/SUM/MIN/MAX/STD all have confirmed `rolling_aggregation`
#     instantiations (`nm -DC libcudf.so`), matching scratchpad/todo.md's
#     expectation exactly. `roll_sd()` uses `make_std_aggregation<
#     rolling_aggregation>(1)` (ddof = 1, sample standard deviation,
#     matching R's own `sd()` and this file's existing groupby `sd()`
#     entry's ddof choice).
# -----------------------------------------------------------------------------

#' Parse `roll_mean(x, n, na.rm = FALSE)` and its five siblings
#'
#' cuplyr's own rolling-window surface (RcppRoll/slider-adjacent naming --
#' see the file-level comment above this section for the full design
#' rationale). Right-aligned only: there is no `align=` parameter on this
#' surface at all (unlike the `frollmean()`-alias path below, which has to
#' reject a non-default `align=` explicitly since data.table's own
#' signature offers one).
#'
#' @param fn_name One of `"roll_mean"`/`"roll_sum"`/`"roll_min"`/
#'   `"roll_max"`/`"roll_sd"`/`"roll_median"`
#' @keywords internal
ir_parse_roll_call <- function(expr, env, schema, fn_name) {
  proto <- function(x, n, na.rm = FALSE) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$x) || is.null(present_args$n)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  n_value <- ir_parse_roll_window_width(present_args$n, env)
  if (is.null(n_value)) {
    return(NULL)
  }

  na_rm <- ir_parse_roll_na_rm(present_args$na.rm, env)
  if (is.null(na_rm)) {
    return(NULL)
  }

  ir_call(fn_name, list(x_ir), meta = list(n = n_value, na_rm = na_rm))
}

#' Parse `data.table::frollmean(x, n, fill=, algo=, align=, na.rm=, hasNA=,
#' adaptive=)` and its five siblings as an alias for `roll_mean()`/etc.
#'
#' See the file-level comment above this section: only the default
#' `align = "right"`, `fill = NA`, `adaptive = FALSE`, `partial = FALSE`,
#' `give.names = FALSE` are supported this wave (a non-default value
#' returns `NULL`, i.e. "not implemented, fall back" -- mutate()'s usual
#' contract for an unrecognized shape: `partial = TRUE` changes windowing
#' semantics entirely -- allows a genuinely incomplete window to produce a
#' value instead of `NA` -- and `give.names = TRUE` changes the RETURN
#' SHAPE to a named `data.table`/list instead of a plain vector, so both
#' must be rejected, not silently ignored); `algo=`/`has.nf=`/`hasNA=` are
#' accepted and ignored (pure performance hints, verified against
#' data.table's own docs to never change the *result*). Proto signature
#' verified against this environment's actual installed
#' `data.table::frollmean` (`args(data.table::frollmean)`), not assumed
#' from memory/docs alone -- note the real signature has `has.nf=` (NOT
#' `hasNA=`, which is a separate, no-default legacy/back-compat argument
#' that's ALSO present, verified empirically).
#'
#' @param dt_fn_name One of `"frollmean"`/`"frollsum"`/`"frollmin"`/
#'   `"frollmax"`/`"frollsd"`/`"frollmedian"`
#' @keywords internal
ir_parse_froll_call <- function(expr, env, schema, dt_fn_name) {
  canonical <- switch(dt_fn_name,
    frollmean = "roll_mean", frollsum = "roll_sum", frollmin = "roll_min",
    frollmax = "roll_max", frollsd = "roll_sd", frollmedian = "roll_median"
  )

  proto <- function(x, n, fill = NA, algo = c("fast", "exact"),
                     align = c("right", "left", "center"), na.rm = FALSE,
                     has.nf = NA, adaptive = FALSE, partial = FALSE,
                     give.names = FALSE, hasNA) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$x) || is.null(present_args$n)) {
    return(NULL)
  }

  if (!is.null(present_args$align)) {
    align_eval <- ir_eval_constant(present_args$align, env)
    if (!align_eval$ok || !identical(align_eval$value, "right")) {
      return(NULL)  # align="left"/"center": not implemented this wave
    }
  }
  if (!is.null(present_args$fill)) {
    fill_eval <- ir_eval_constant(present_args$fill, env)
    if (!fill_eval$ok || length(fill_eval$value) != 1 || !is.na(fill_eval$value)) {
      return(NULL)  # a non-NA fill: not implemented this wave
    }
  }
  if (!is.null(present_args$adaptive)) {
    adaptive_eval <- ir_eval_constant(present_args$adaptive, env)
    if (!adaptive_eval$ok || !isFALSE(adaptive_eval$value)) {
      return(NULL)  # adaptive=TRUE (per-row window vector): not implemented this wave
    }
  }
  if (!is.null(present_args$partial)) {
    partial_eval <- ir_eval_constant(present_args$partial, env)
    if (!partial_eval$ok || !isFALSE(partial_eval$value)) {
      return(NULL)  # partial=TRUE: different windowing semantics, not implemented this wave
    }
  }
  if (!is.null(present_args$give.names)) {
    give_names_eval <- ir_eval_constant(present_args$give.names, env)
    if (!give_names_eval$ok || !isFALSE(give_names_eval$value)) {
      return(NULL)  # give.names=TRUE: changes return shape entirely, not implemented this wave
    }
  }
  # algo=/has.nf=/hasNA=: accepted, deliberately unread (see docs above).

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  n_value <- ir_parse_roll_window_width(present_args$n, env)
  if (is.null(n_value)) {
    return(NULL)
  }

  na_rm <- ir_parse_roll_na_rm(present_args$na.rm, env)
  if (is.null(na_rm)) {
    return(NULL)
  }

  ir_call(canonical, list(x_ir), meta = list(n = n_value, na_rm = na_rm))
}

# A rolling window width must be a positive whole-number CONSTANT (never a
# column reference/vector of multiple sizes -- data.table's own "adaptive"
# multi-window-size shape, rejected above via `adaptive=`, is the only
# vector-`n` case that would otherwise reach here, and `length(...) != 1`
# rejects it defensively too). Returns an integer scalar, or `NULL` (not
# implemented this wave / not a constant).
ir_parse_roll_window_width <- function(n_expr, env) {
  n_eval <- ir_eval_constant(n_expr, env)
  n_val <- n_eval$value
  if (!n_eval$ok || !is.numeric(n_val) || length(n_val) != 1 || is.na(n_val) ||
      n_val != round(n_val) || n_val < 1) {
    return(NULL)
  }
  as.integer(n_val)
}

# `na.rm=` must be a constant TRUE/FALSE (absent -> FALSE, dplyr/data.table's
# shared default). Returns a logical scalar, or `NULL` (not a constant).
ir_parse_roll_na_rm <- function(na_rm_expr, env) {
  if (is.null(na_rm_expr)) {
    return(FALSE)
  }
  na_rm_eval <- ir_eval_constant(na_rm_expr, env)
  na_rm_val <- na_rm_eval$value
  if (!na_rm_eval$ok || !is.logical(na_rm_val) || length(na_rm_val) != 1 || is.na(na_rm_val)) {
    return(NULL)
  }
  na_rm_val
}

#' Parse `ntile(x = row_number(), n)`
#'
#' Fully desugars at parse time into dplyr's own `ntile()` algorithm
#' (verified against `dplyr:::ntile`'s actual body, not just its docs;
#' cross-checked numerically against a battery of cases including ties, a
#' non-multiple group size, all-`NA`, and a zero-row input) expressed in
#' plain IR plus two nested window calls -- `row_number(x)` (or bare
#' `row_number()` when `x` is omitted) and, only when `x` is supplied,
#' `sum(if_else(is.na(x), 0L, 1L))` (the non-missing count; when `x` is
#' omitted, dplyr's own `len` is just the group size, i.e. plain `n()`).
#' Both are ordinary window-class `ir_call` nodes at this point -- their
#' actual extraction into `..win*..` temp columns happens later, in
#' `R/window.R`'s `ir_extract_windows()`/`plan_window_stages()`, not here.
#'
#' Whether `x` was actually supplied changes the desugar (dplyr's own
#' source only re-ranks `x` -- `x <- row_number(x)` -- when it's NOT the
#' default), so this dispatches on `rlang::call_match(..., defaults =
#' FALSE)` (which omits an unsupplied formal entirely, unlike `defaults =
#' TRUE`) rather than merely on whether `x`'s resolved value looks like
#' `row_number()`.
#'
#' `n` must be a constant, positive, whole-number scalar (dplyr's own
#' `check_number_whole()` requirement); `n` missing entirely (dplyr's `n`
#' has no default) falls back, since there's nothing sensible to build.
#'
#' The division-then-floor steps in dplyr's own formula (`(x + (larger_size
#' - 1L)) / larger_size`, then a final `as.integer(floor(bins))`) are
#' expressed here via `%/%` (floor division) directly instead: every
#' numerator/denominator pair in both branches is non-negative by
#' construction (ranks start at 1, sizes are always >= 1), and floor
#' division equals true-division-then-floor for non-negative operands, so
#' this produces exactly the same result while keeping the whole expression
#' integer-typed throughout (matching ntile()'s own always-integer output)
#' instead of needing a separate cast step the IR has no primitive for.
#'
#' @keywords internal
ir_parse_ntile <- function(expr, env, schema) {
  proto <- function(x = row_number(), n) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (is.null(present_args$n)) {
    return(NULL)  # n is mandatory (no default); dplyr itself errors here
  }
  n_eval <- ir_eval_constant(present_args$n, env)
  n_val <- n_eval$value
  if (!n_eval$ok || !is.numeric(n_val) || length(n_val) != 1 || is.na(n_val) ||
      n_val != round(n_val) || n_val <= 0) {
    return(NULL)
  }
  n_lit <- ir_lit_from_r(as.integer(n_val))

  x_supplied <- !is.null(present_args$x)

  if (x_supplied) {
    x_ir <- ir_parse_expr(present_args$x, env, schema)
    if (is.null(x_ir)) {
      return(NULL)
    }
    rn <- ir_call("row_number", list(x_ir))
    indicator <- ir_call("if_else", list(
      ir_call("is.na", list(x_ir)), ir_lit_from_r(0L), ir_lit_from_r(1L)
    ))
    len <- ir_call("sum", list(indicator))
  } else {
    rn <- ir_call("row_number", list())
    len <- ir_call("n", list())
  }

  n_larger         <- ir_call("%%", list(len, n_lit))
  size             <- ir_call("/", list(len, n_lit))
  larger_size      <- ir_call("ceil", list(size))
  smaller_size     <- ir_call("floor", list(size))
  larger_threshold <- ir_call("*", list(larger_size, n_larger))

  branch_a <- ir_call("%/%", list(
    ir_call("+", list(rn, ir_call("-", list(larger_size, ir_lit_from_r(1L))))),
    larger_size
  ))
  branch_b <- ir_call("+", list(
    ir_call("%/%", list(
      ir_call("+", list(
        ir_call("-", list(rn, larger_threshold)),
        ir_call("-", list(smaller_size, ir_lit_from_r(1L)))
      )),
      smaller_size
    )),
    n_larger
  ))

  bins <- ir_call("if_else", list(
    ir_call("<=", list(rn, larger_threshold)),
    branch_a,
    branch_b
  ))

  ir_call("if_else", list(
    ir_call("==", list(len, ir_lit_from_r(0L))),
    ir_lit(value = NULL, type = "INT32", na = TRUE),
    bins
  ))
}

#' Parse `sum(x)` / `sum(x, na.rm = <literal>)` (Phase 5, task W9)
#'
#' `sum()`'s ordinary registry entry (`arity = 1L`, agg-family window kind)
#' only accepts the bare single-argument form -- the overwhelmingly common
#' shape, and the one this window-agg family's `na.rm = FALSE` default (see
#' `src/ops_window.cpp`'s "sticky-NA"-style whole-group-NA-propagation
#' rule, matching R's own `sum()` default) is built for. But real,
#' unmodified `dplyr::add_tally(wt = )` (a plain, non-generic dplyr
#' function we don't control -- `dplyr:::tally_n()`) UNCONDITIONALLY emits
#' `base::sum(<wt>, na.rm = TRUE)` as its internal `mutate()` dot,
#' regardless of whether the weight column actually has any NAs -- so this
#' 2-arg shape must parse too, or `add_tally(wt = )` hard-errors. (This is
#' the Phase 5 task W9 fix-forward found while deleting
#' `grouped_aggregate_mutate()`, which used to accept this exact shape via
#' a real NA-excluding `summarise()`-then-join-back call.)
#'
#' `na.rm = TRUE` desugars `x` to `coalesce(x, 0)` before the window `sum`
#' call: replacing every `NA` with the additive identity is exactly
#' equivalent to excluding it from the sum (matching what R's own
#' `sum(x, na.rm = TRUE)` computes), and since the desugared column has no
#' nulls left at all, the window kernel's whole-group-NA rule never fires
#' -- so no C++ change is needed. `na.rm = FALSE` (if named explicitly) is
#' a pure no-op, since propagating NA is already this family's own default.
#'
#' A second argument with any other name (or a non-literal-logical value)
#' doesn't match this shape -- returns `NULL`, same as any other
#' unsupported call shape (`mutate()` hard-errors on it; `filter()` falls
#' back to CPU evaluation).
#'
#' @keywords internal
ir_parse_sum_call <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) == 0 || length(args_raw) > 2) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(args_raw[[1]], env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  if (length(args_raw) == 1) {
    return(ir_call("sum", list(x_ir)))
  }

  arg_names <- names(args_raw)
  if (is.null(arg_names) || !identical(arg_names, c("", "na.rm"))) {
    return(NULL)  # only the exact `sum(x, na.rm = <literal>)` shape matches
  }

  na_rm_val <- tryCatch(
    rlang::eval_tidy(rlang::new_quosure(args_raw[[2]], env)),
    error = function(e) NULL
  )
  if (!is.logical(na_rm_val) || length(na_rm_val) != 1 || is.na(na_rm_val)) {
    return(NULL)
  }

  if (isTRUE(na_rm_val)) {
    x_ir <- ir_call("coalesce", list(x_ir, ir_lit_from_r(0L)))
  }

  ir_call("sum", list(x_ir))
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

#' Does an IR (sub)tree contain a window-function call anywhere?
#'
#' Phase 5 (window functions, `scratchpad/phase5_window_design.md` section
#' 1.1): `TRUE` iff any `call` node in the subtree has a non-`NULL` `window`
#' field in its [ir_call_registry] entry. Used by [ir_bind()]'s window
#' guard, by the decomposition machinery in `R/window.R`
#' (`ir_extract_windows()`/`plan_window_stages()`), and will be used by the
#' as-yet-unwired grouped `mutate()`/`filter()` (W3/W5) to detect when a
#' parsed expression needs the window decomposition path at all.
#'
#' Accepts either a single IR node (identified by the `cuplyr_ir` class
#' every [ir_col()]/[ir_lit()]/[ir_call()] node carries) or a plain
#' (possibly unnamed) list of IR nodes -- e.g. a call node's own `$args` --
#' so callers can check a call's argument list directly without unpacking
#' it first.
#'
#' @param ir An IR node, a list of IR nodes, or `NULL`
#' @return Logical scalar
#' @keywords internal
ir_has_window <- function(ir) {
  if (is.null(ir)) {
    return(FALSE)
  }

  if (inherits(ir, "cuplyr_ir")) {
    if (!identical(ir$kind, "call")) {
      return(FALSE)
    }
    entry <- ir_call_registry[[ir$op]]
    if (!is.null(entry) && !is.null(entry$window)) {
      return(TRUE)
    }
    return(any(vapply(ir$args, ir_has_window, logical(1))))
  }

  # A plain list of IR nodes (e.g. a call node's `$args`).
  any(vapply(ir, ir_has_window, logical(1)))
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

  # Phase 5 (window functions) invariant: a window-class op (registry entry
  # has a non-NULL `window` field, see ir_call_registry's window entries and
  # ir_has_window()) must NEVER reach lowering directly -- it can only ever
  # appear as an intermediate node that R/window.R's ir_extract_windows()/
  # plan_window_stages() rewrites away (into an ir_col() reference to a
  # materialized window-spec's output column) before a verb's own
  # ast_mutate()/ast_filter()/etc. node is built. Nothing in the package
  # wires window ops into a verb yet (W1 is dormant), so this guard should
  # be unreachable in practice; it exists so that a future verb (or a
  # currently-live mutate()/filter() call that now happens to *parse*
  # successfully because "mean"/"lag"/etc. are registry entries) fails
  # loudly here instead of silently mis-executing as an ordinary AST op.
  entry <- ir_call_registry[[ir$op]]
  if (!is.null(entry) && !is.null(entry$window)) {
    stop("internal: window op '", ir$op, "' reached lowering", call. = FALSE)
  }

  ir$args <- lapply(ir$args, ir_bind, schema = schema)
  ir
}

#' Create a side-tagged IR column reference for a non-equi `join_by()`
#' condition (Phase 7 J7)
#'
#' A plain [ir_col()] node is resolved against a *single* schema by name
#' ([ir_bind()]) -- fine for `filter()`/`mutate()`, which only ever see one
#' table. A join condition (`join_by(a >= b)`) references columns from TWO
#' tables at once, and -- crucially -- a self-join (`inner_join(x, x,
#' join_by(a >= b))`) has IDENTICAL left/right schemas, so a name alone
#' can't say which side `a`/`b` belong to. `join_by()`'s own pre-normalized
#' spec already tells us explicitly, though: `x`/`y` are parallel vectors
#' where `x[i]` is ALWAYS the left-table column and `y[i]` is ALWAYS the
#' right-table column for condition `i` (see `join_by_is_non_equi()`'s own
#' roxygen, R/join.R). `join_cond_ir_from_spec()` (R/join.R) uses that
#' knowledge directly when building each condition's IR, tagging every
#' `col` node with an explicit `side` (`0L` = left/x, `1L` = right/y) at
#' construction time -- deliberately NOT the design doc's original
#' "resolve left-first-then-right by lookup order" sketch, which silently
#' breaks on a self-join. [ir_bind_join()] then resolves `$index` against
#' the correct side's schema using this tag, never by cross-schema lookup.
#' @param name Character scalar, the column name
#' @param side `0L` (left/x) or `1L` (right/y)
#' @return A `cuplyr_ir_col` node with an extra `side` field
#' @keywords internal
ir_col_join <- function(name, side) {
  node <- ir_col(name)
  node$side <- as.integer(side)
  node
}

#' Bind 0-based, per-side column indices onto a non-equi join condition IR
#' tree (Phase 7 J7)
#'
#' The join-condition analogue of [ir_bind()]. Every `col` node in a
#' condition IR built by `join_cond_ir_from_spec()` (R/join.R) already
#' carries an explicit `side` tag (see [ir_col_join()]'s docs for why this
#' package deviates from the design doc's lookup-order sketch: a self-join
#' has identical left/right schemas, so name-based resolution against
#' `left_schema` first would silently mis-bind a right-only column, or
#' bind a same-named column to the wrong side entirely). This function
#' simply resolves each `col` node's `$index` against `left_schema` (side
#' `0L`) or `right_schema` (side `1L`) directly, using that tag -- never a
#' generic single-schema lookup.
#' @param ir An IR node (or `NULL`), every `col` descendant carrying a
#'   `side` field ([ir_col_join()])
#' @param left_schema,right_schema List with `names`/`types` for the join's
#'   left/right inputs
#' @return The IR tree with `index` filled in on every `col` node
#' @keywords internal
ir_bind_join <- function(ir, left_schema, right_schema) {
  if (is.null(ir)) {
    return(ir)
  }

  if (ir$kind == "col") {
    schema <- if (identical(ir$side, 1L)) right_schema else left_schema
    idx <- match(ir$name, schema$names)
    if (is.na(idx)) {
      side_name <- if (identical(ir$side, 1L)) "Right" else "Left"
      stop(side_name, " join condition column '", ir$name, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    ir$index <- idx - 1L
    return(ir)
  }

  # kind == "call": every non-equi join condition IR is built exclusively
  # from comparison ops folded with "&" (join_cond_ir_from_spec()) -- never
  # a literal (join_by() itself only ever accepts column-to-column
  # comparisons, e.g. `join_by(a >= b)`; `join_by(a >= 5)` is a dplyr parse
  # error before this package ever sees it) -- so no "lit" branch is needed
  # here, unlike ir_bind()'s generic single-schema version.
  ir$args <- lapply(ir$args, ir_bind_join, left_schema = left_schema, right_schema = right_schema)
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
type_int32 <- function(arg_types) "INT32"
type_arg1 <- function(arg_types) arg_types[1]

# Window-class `sum()`'s output type -- reuses make_aggregation()'s own
# switch verbatim (R/ast.R): INT32 promotes to INT64, everything else
# (including INT64 itself, matching that function's existing behavior)
# defaults to FLOAT64. Phase 5 (window functions) note: this intentionally
# mirrors that pre-existing table rather than "fixing" it, per
# scratchpad/phase5_window_design.md section 4's "output types reuse
# make_aggregation()'s table verbatim" instruction.
type_window_sum <- function(arg_types) {
  if (identical(arg_types[1], "INT32")) "INT64" else "FLOAT64"
}

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

# -----------------------------------------------------------------------------
# Phase 5 (window functions): cummean()/cumall()/cumany()/consecutive_id()
# registry `parse` hooks -- these are ordinary registry entries (single
# positional `x`, no ambiguous named controls), so unlike lag()/lead()/
# first()/last()/nth()/ntile() above they need no dedicated top-level
# dispatch: the registry's own generic per-arg loop in ir_parse_expr()
# already parses their one argument correctly before calling `entry$parse`.
#
# Each fully desugars into plain IR containing nested window-class calls
# (cumsum/cummin/cummax/row_number/sum/n) -- see R/window.R's
# ir_extract_windows()/plan_window_stages() for how those get decomposed
# into actual `..win*..` temp columns later; nothing here needs to know
# about that. Every formula below was verified empirically against dplyr
# 1.2.1 (not re-derived from the design doc's table alone) via a battery of
# hand-checked cases including NA-adjacency runs and ties; see the W1 task
# notes for the exact cases run.
# -----------------------------------------------------------------------------

#' Desugar `cummean(x)` into `cumsum(x) / row_number()`
#'
#' Verified against dplyr 1.2.1: `cummean(c(1,2,NA,4))` is `1, 1.5, NA, NA`
#' -- i.e. cummean inherits cumsum()'s own (sticky, once cudf's scan gets
#' its Phase-5 sticky-NA fix in W2 -- see phase5_window_design.md section
#' 4) NA propagation for free, simply by dividing by row_number() (which is
#' never NA), with no special-casing needed here.
#'
#' @keywords internal
ir_parse_cummean <- function(args, schema) {
  ir_call("/", list(ir_call("cumsum", list(args[[1]])), ir_call("row_number", list())))
}

#' Desugar `cumall(x)`/`cumany(x)` into a tri-state `cummin`/`cummax` encoding
#'
#' Verified against dplyr 1.2.1: `cumall(c(T,T,NA,T,F,T))` is `T,T,NA,NA,
#' F,F`; `cumany(c(F,NA,T,F))` is `F,NA,T,T` -- both are exactly `cummin`/
#' `cummax` (respectively) over the tri-state encoding FALSE=0, NA=1,
#' TRUE=2, then decoded back to BOOL8 (`scratchpad/phase5_window_design.md`
#' section 0's ground truth table, cross-checked directly here rather than
#' taken on faith).
#'
#' `if_else()`'s bare-NA value-position adoption (`ir_resolve_value_nas()`)
#' is applied explicitly to the decode step's `yes`/`no` pair here, exactly
#' as if_else()'s own registry `parse` hook would -- this desugar builds
#' `ir_call("if_else", ...)` nodes directly rather than routing back through
#' `ir_parse_expr()`, so that adoption has to be done by hand.
#'
#' @param scan_op `"cummin"` (cumall) or `"cummax"` (cumany)
#' @keywords internal
ir_parse_cumall_cumany <- function(args, schema, scan_op) {
  x <- args[[1]]
  encoded <- ir_call("if_else", list(
    ir_call("is.na", list(x)),
    ir_lit_from_r(1L),
    ir_call("if_else", list(x, ir_lit_from_r(2L), ir_lit_from_r(0L)))
  ))
  s <- ir_call(scan_op, list(encoded))
  decode_values <- ir_resolve_value_nas(
    list(ir_lit(value = NULL, type = NULL, na = TRUE), ir_call("==", list(s, ir_lit_from_r(2L)))),
    schema
  )
  ir_call("if_else", c(list(ir_call("==", list(s, ir_lit_from_r(1L)))), decode_values))
}

#' Desugar single-column `consecutive_id(x)` into a two-stage `lag`/
#' `row_number`/`cumsum` formula
#'
#' Multi-column `consecutive_id(a, b, ...)` is not implemented this wave
#' (returns `NULL`, i.e. falls back) -- this is enforced for free by the
#' registry entry's `arity = 1L` (a 2+-arg call fails the generic arity
#' check in `ir_parse_expr()` before `parse` is ever invoked), so this
#' function itself only ever sees exactly one argument.
#'
#' Verified against dplyr 1.2.1, INCLUDING NA-adjacency (the tricky part:
#' two consecutive `NA`s are treated as "the same value", i.e. NOT a new
#' group, which the `is.na(x) & is.na(lx)` disjunct captures):
#' `consecutive_id(c(1,NA,NA,2))` is `1,2,2,3`; `consecutive_id(c(NA,NA,1,
#' 1,NA))` is `1,1,2,2,3`; `consecutive_id(c(1,2,2,NA,NA,2))` is
#' `1,2,2,3,3,4`.
#'
#' Builds `cumsum(if_else(rn==1L, 1L, if_else(same_as_prev, 0L, 1L)))`
#' where `rn = row_number()` and `same_as_prev = coalesce(x == lag(x),
#' FALSE) | (is.na(x) & is.na(lag(x)))`. `lag(x)`/`row_number()` are
#' themselves window calls nested inside `cumsum()`'s argument -- exactly
#' the "nested window calls needing 2 decomposition passes" shape `R/
#' window.R`'s `ir_extract_windows()`/`plan_window_stages()` handles
#' generically (see test-window-plan.R), so this desugar doesn't need to
#' know anything about staging itself.
#'
#' @keywords internal
ir_parse_consecutive_id <- function(args, schema) {
  x <- args[[1]]
  lx <- ir_call("lag", list(x), meta = list(n = 1L, default = NULL))
  rn <- ir_call("row_number", list())

  same_as_prev <- ir_call("|", list(
    ir_call("coalesce", list(ir_call("==", list(x, lx)), ir_lit_from_r(FALSE))),
    ir_call("&", list(ir_call("is.na", list(x)), ir_call("is.na", list(lx))))
  ))

  increment <- ir_call("if_else", list(
    ir_call("==", list(rn, ir_lit_from_r(1L))),
    ir_lit_from_r(1L),
    ir_call("if_else", list(same_as_prev, ir_lit_from_r(0L), ir_lit_from_r(1L)))
  ))

  ir_call("cumsum", list(increment))
}

#' Registry of supported expression-IR operations
#'
#' See the module-level comment above and section 6 of
#' `scratchpad/phase1_expression_engine.md` for the extension contract this
#' implements. Each entry: `list(arity, parse, type, lower)`.
#'
#' Phase 5 (`scratchpad/phase5_window_design.md` section 1.1) adds a fifth
#' field, `window`: `NULL` for every pre-Phase-5 entry (unchanged), or
#' `list(kind = "shift"|"rank"|"scan"|"agg")` for a window-class op. `window`
#' non-`NULL` and `lower` non-`NULL` are mutually exclusive -- a window-class
#' entry always has `lower = NULL` (window ops are never `ir_call` handlers
#' in `src/expr_eval.hpp`, see this file's own module comment for that
#' boundary, and are structurally prevented from reaching lowering at all
#' by [ir_bind()]'s guard). `ir_infer_type()` needs no change: a window
#' entry's `type` function is looked up and called exactly like any other
#' entry's.
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
  ),

  # --- Phase 4 wave 2: near(), replace_when() (case_match()/recode_values()/
  # replace_values()/when_all()/when_any() all fully desugar at parse time
  # into "case_when"/"&"/"|" nodes -- see the module comment above
  # ir_parse_value_match_clauses() and never reach the registry under their
  # own op name at all) ---

  # `near(x, y, tol = .Machine$double.eps^0.5)`: verified against dplyr's
  # own source (`dplyr:::near`) -- `abs(x - y) < tol`, nothing more. Always
  # desugared at parse time (like `between`/`log2`/`log10` above), so "near"
  # never reaches lowering under its own op name.
  "near" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) < 2 || length(args) > 3) {
        return(NULL)  # wrong arg count: not this shape, fall back
      }
      x <- args[[1]]
      y <- args[[2]]
      tol <- if (length(args) == 3) args[[3]] else ir_lit_from_r(sqrt(.Machine$double.eps))
      ir_call("<", list(ir_call("abs", list(ir_call("-", list(x, y)))), tol))
    },
    type = type_bool8,
    lower = NULL  # always desugared at parse time; never lowered as "near"
  ),

  # `replace_when(x, ...)`/`replace_values(x, ...)`: never reach this
  # entry's `parse` -- both are built directly by their own dedicated parse
  # functions (`ir_parse_replace_when()`/`ir_parse_replace_values()`), since
  # `x` (arg 1) is a required leading positional argument the generic
  # per-arg registry loop has no special handling for. This entry exists
  # only so `ir_infer_type()`/lowering can look up `type`/`lower` for the
  # `ir_call("replace_when", ...)` node those parse functions build.
  "replace_when" = list(
    arity = NA_integer_,
    parse = NULL,
    type = type_replace_when,
    lower = list(handler = "replace_when")
  ),

  # --- Phase 5 (window functions): window-class entries ---
  # See scratchpad/phase5_window_design.md sections 1.1/4. Every entry here
  # has `lower = NULL` (window ops never reach lowering under their own op
  # name -- ir_bind()'s guard enforces this defensively) and a `window`
  # field carrying the broad C++ lowering family ("shift"/"rank"/"scan"/
  # "agg") that R/window.R's ir_extract_windows() reads (only for "rank",
  # to decide whether the value column doubles as the frame's order
  # column) -- the family itself has no other R-side effect this wave;
  # W2/W4 are where each family actually gets lowered.
  #
  # lag()/lead(): parsed by ir_parse_shift_call() (dedicated dispatch,
  # never via this entry's own `parse`/generic loop -- see that function's
  # docs for why named args force this). `type = type_arg1`: the shifted
  # value keeps x's own type (the fill default, when absent, is an invalid
  # scalar of that same type -- W2's C++ concern, not this entry's).
  "lag" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
               lower = NULL, window = list(kind = "shift")),
  "lead" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                lower = NULL, window = list(kind = "shift")),

  # row_number(): 0- or 1-arg (dplyr: bare `row_number()` inside a mutate/
  # filter is sequential position; `row_number(x)` ranks by x, ties broken
  # by original position -- rank_method::FIRST). Needs its own `parse` hook
  # (arity = NA) purely to re-validate the 0-or-1 arg count, since the
  # registry's generic arity check only fires for a single fixed arity;
  # the args themselves are ordinary positional parses, so no dedicated
  # top-level dispatch is needed (unlike lag()/lead()'s named controls).
  "row_number" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) > 1) {
        return(NULL)
      }
      ir_call("row_number", args)
    },
    type = type_int32,
    lower = NULL,
    window = list(kind = "rank")
  ),
  "min_rank" = list(arity = 1L, parse = NULL, type = type_int32,
                     lower = NULL, window = list(kind = "rank")),
  "dense_rank" = list(arity = 1L, parse = NULL, type = type_int32,
                       lower = NULL, window = list(kind = "rank")),
  "percent_rank" = list(arity = 1L, parse = NULL, type = type_float64,
                         lower = NULL, window = list(kind = "rank")),
  "cume_dist" = list(arity = 1L, parse = NULL, type = type_float64,
                      lower = NULL, window = list(kind = "rank")),

  # cumsum/cummax/cummin/cumprod: single-arg scan family. cumsum's type
  # reuses "+"'s own D5 promotion function verbatim (calling
  # infer_mutate_output_type("+", arg_types, NULL) with a single input type
  # gives exactly BOOL8/INT32 -> INT32, INT64 -> INT64, else FLOAT64 --
  # dplyr's own cumsum() promotion, confirmed by inspection of that
  # function's Rule 5 branch); cummax/cummin preserve the argument's type
  # (matching base R); cumprod always returns FLOAT64 (matching base R:
  # `typeof(cumprod(1:3))` is "double").
  "cumsum" = list(arity = 1L, parse = NULL, type = type_arith_op("+"),
                   lower = NULL, window = list(kind = "scan")),
  "cummax" = list(arity = 1L, parse = NULL, type = type_arg1,
                   lower = NULL, window = list(kind = "scan")),
  "cummin" = list(arity = 1L, parse = NULL, type = type_arg1,
                   lower = NULL, window = list(kind = "scan")),
  "cumprod" = list(arity = 1L, parse = NULL, type = type_float64,
                    lower = NULL, window = list(kind = "scan")),

  # mean/sum/min/max/n/sd/var: agg-family window entries (aggregate +
  # broadcast, see phase5_window_design.md section 4). These double as
  # summarise()'s own aggregation function names, but summarise()'s parser
  # (parse_aggregations(), R/summarise.R) never reaches ir_call_registry at
  # all -- it has its own independent dispatch table -- so registering
  # these here only affects expressions parsed via ir_parse_quo() (mutate()/
  # filter(), and summarise()'s temp-column sub-expression preprocessing,
  # which parses the *argument* of an aggregation call, never the
  # aggregation function name itself). Output types reuse
  # make_aggregation()'s own table verbatim (R/ast.R) except where that
  # table doesn't cover a case at all (sum(): type_window_sum reimplements
  # its exact switch branch, since arg_types here come from a window call's
  # own arg rather than an aggregation struct's `input_type` field).
  "mean" = list(arity = 1L, parse = NULL, type = type_float64,
                 lower = NULL, window = list(kind = "agg")),
  "sum" = list(arity = 1L, parse = NULL, type = type_window_sum,
                lower = NULL, window = list(kind = "agg")),
  "min" = list(arity = 1L, parse = NULL, type = type_arg1,
                lower = NULL, window = list(kind = "agg")),
  "max" = list(arity = 1L, parse = NULL, type = type_arg1,
                lower = NULL, window = list(kind = "agg")),
  "n" = list(arity = 0L, parse = NULL, type = type_int32,
              lower = NULL, window = list(kind = "agg")),
  "sd" = list(arity = 1L, parse = NULL, type = type_float64,
               lower = NULL, window = list(kind = "agg")),
  "var" = list(arity = 1L, parse = NULL, type = type_float64,
                lower = NULL, window = list(kind = "agg")),

  # median()/n_distinct() (Phase 6, task 6.2): join the agg family "for
  # free" once summarise() learned them, reusing get_groupby_agg()
  # (src/ops_groupby.cpp, genuinely shared source between the summarise()
  # and window-mutate() paths) -- verified empirically against dplyr 1.2.1
  # (test-dplyr-summarise.R's window-parity oracle tests). Like every
  # other "agg"-family window function, neither supports a na.rm=/na_rm=
  # ARGUMENT here (arity is fixed at 1L, same as mean()/sum()/...; a
  # `median(x, na.rm = TRUE)` mutate() call falls through to mutate()'s own
  # "unsupported named argument" error, exactly like `mean(x, na.rm =
  # TRUE)` already does) -- median() always gets na.rm=FALSE's whole-
  # group-null propagation (src/ops_window.cpp's na_propagates list),
  # n_distinct() always counts NA as its own distinct value
  # (null_policy::INCLUDE, get_groupby_agg()'s "n_distinct" branch).
  # quantile()/any()/all() are NOT added here -- see is_agg_fn()'s own
  # docs (src/ops_window.cpp) for why.
  "median" = list(arity = 1L, parse = NULL, type = type_float64,
                   lower = NULL, window = list(kind = "agg")),
  "n_distinct" = list(arity = 1L, parse = NULL, type = type_int32,
                       lower = NULL, window = list(kind = "agg")),

  # first()/last()/nth(): agg-family, but order-sensitive (dplyr's own
  # "first"/"last"/"nth row *of the group, in its current order*" -- W1
  # only supports the no-order_by/no-default/no-na_rm shape, hence the
  # dedicated dispatch in ir_parse_first_last_nth(); type = type_arg1
  # (preserves the argument's own type, per phase5_window_design.md
  # section 4's "-> arg type", NOT make_aggregation()'s generic table,
  # which doesn't cover these three functions at all).
  "first" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                  lower = NULL, window = list(kind = "agg")),
  "last" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                 lower = NULL, window = list(kind = "agg")),
  "nth" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                lower = NULL, window = list(kind = "agg")),

  # --- Phase 5: pure parse-time desugars that happen to produce window IR
  # (window = NULL / lower = NULL, exactly like "between"/"log2"/"near"
  # above -- these op names never survive to appear in a finished tree
  # under their own name, only the window-class calls their `parse` hooks
  # build do) ---
  "cummean" = list(arity = 1L, parse = ir_parse_cummean, type = type_float64,
                    lower = NULL, window = NULL),
  "cumall" = list(
    arity = 1L,
    parse = function(args, schema) ir_parse_cumall_cumany(args, schema, "cummin"),
    type = type_bool8,
    lower = NULL,
    window = NULL
  ),
  "cumany" = list(
    arity = 1L,
    parse = function(args, schema) ir_parse_cumall_cumany(args, schema, "cummax"),
    type = type_bool8,
    lower = NULL,
    window = NULL
  ),
  "consecutive_id" = list(arity = 1L, parse = ir_parse_consecutive_id, type = type_int32,
                           lower = NULL, window = NULL),

  # --- Rolling windows (Theo-requested benchmark task, a sixth window
  # kind alongside shift/rank/scan/agg): roll_mean()/roll_sum()/roll_min()/
  # roll_max()/roll_sd()/roll_median(), parsed via the dedicated
  # ir_parse_roll_call()/ir_parse_froll_call() dispatch above (never via
  # this entry's own `parse`/generic loop -- na.rm= is a named control the
  # generic per-arg loop can't express, same reason as lag()/lead()).
  # Output types: roll_mean/roll_sd/roll_median always FLOAT64 (matching
  # data.table::frollmean()/frollsd()/frollmedian(), always double);
  # roll_sum reuses type_window_sum verbatim (identical promotion rule to
  # window sum()/summarise's sum(): INT32 -> INT64, else FLOAT64);
  # roll_min/roll_max preserve the argument's own type (type_arg1, matching
  # cudf::rolling_window()'s own documented "MIN/MAX return a column of the
  # same type as the input" rule). `window_spec_lowerable()` (R/lower.R)
  # excludes "roll_median" specifically -- cudf's rolling_window()/
  # grouped_rolling_window() have no MEDIAN aggregation instantiation in
  # this environment (verified via `nm -DC libcudf.so`) -- routing it
  # through the existing CPU-fallback path instead (see R/roll.R).
  "roll_mean" = list(arity = 1L, parse = NULL, type = type_float64,
                      lower = NULL, window = list(kind = "rolling")),
  "roll_sum" = list(arity = 1L, parse = NULL, type = type_window_sum,
                     lower = NULL, window = list(kind = "rolling")),
  "roll_min" = list(arity = 1L, parse = NULL, type = type_arg1,
                     lower = NULL, window = list(kind = "rolling")),
  "roll_max" = list(arity = 1L, parse = NULL, type = type_arg1,
                     lower = NULL, window = list(kind = "rolling")),
  "roll_sd" = list(arity = 1L, parse = NULL, type = type_float64,
                    lower = NULL, window = list(kind = "rolling")),
  "roll_median" = list(arity = 1L, parse = NULL, type = type_float64,
                        lower = NULL, window = list(kind = "rolling"))

  # ntile() is dispatched directly from ir_parse_expr() (ir_parse_ntile()),
  # never through this registry at all (its "x supplied or not" distinction
  # needs raw-call inspection the generic per-arg loop can't provide) -- so,
  # unlike cummean()/cumall()/cumany()/consecutive_id() above, it has no
  # entry here; ir_infer_type()/ir_bind() never need to look up "ntile" as
  # an op name, since it never appears in a finished tree either.
)
