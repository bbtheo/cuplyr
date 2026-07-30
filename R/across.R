# across()/if_any()/if_all()/pick() and tidy-eval context functions
# (Phase 10 of the dplyr-parity roadmap, scratchpad/todo.md)
#
# Design (verified empirically against dplyr 1.2.1, see
# scratchpad/workflow_state.md's Phase 10 entry for the full empirical
# ledger): across()/if_any()/if_all() are expanded at the QUOSURE level,
# before any dot ever reaches ir_parse_quo()/parse_mutate_dots()/
# parse_aggregations() -- one across() dot becomes N ordinary named dots
# (one per matched column x function combination), and one if_any()/
# if_all() dot becomes a single dot whose expression is an OR/AND fold of
# the same per-column predicate. No new IR node type, no new C++: every
# expanded dot is just ordinary mutate/filter/summarise input from that
# point on.
#
# Scope cuts (deliberately deferred, matching this task's own brief):
#   - pick() used as a genuine data-frame value (`rowSums(pick(...))`,
#     `arrange(pick(...))`, ...) -- no GPU representation for "a handful of
#     columns as one value" exists in this package's IR, so ANY dot that
#     references pick()/cur_group()/cur_group_rows()/cur_data()/
#     cur_data_all()/c_across() anywhere in its expression tree, or nests
#     across()/if_any()/if_all() inside another call (rather than being the
#     WHOLE dot), or supplies `across(.unpack = TRUE)`, is routed through a
#     whole-CALL CPU fallback (dots_need_fallback()/verb_across_fallback())
#     that simply re-runs the real, original dplyr call on the CPU --
#     dplyr's own pick()/cur_group()/across(.unpack=) already do the right
#     thing there, so there is no reason to reimplement them.
#   - cur_group_id() is the one context function that gets a genuine
#     GPU-native lowering (substitute_cur_group_id()): dplyr's own
#     implementation is exactly "the dense (1-based) rank of each row's own
#     group-key tuple in group_data()'s row order", and group_data() is
#     already GPU-native (R/group-by.R's compute_group_data(), Phase 9) --
#     so this reuses that plus an ordinary left_join() to broadcast each
#     group's id back onto every one of its rows. Supported inside
#     mutate()/filter() (row-level verbs); inside summarise() it is routed
#     to the whole-call fallback instead (see substitute_cur_group_id()'s
#     own docs for why).
#   - cur_group()/cur_group_rows() always fall back (no clean GPU
#     lowering: the former returns a 1-row tibble, the latter a row-index
#     vector, per current group -- genuinely row/group-context objects,
#     not columns).
#   - c_across() is rowwise()'s own companion; rowwise() itself is already
#     a full CPU-fallback verb (R/fallback.R, Phase 2), so c_across() needs
#     no code here at all -- it only ever appears inside a rowwise-mutate
#     call, which never reaches this file.

# =============================================================================
# Small AST helpers
# =============================================================================

# The bare (namespace-stripped) name of a call's head, or NULL if `expr`
# isn't a call with a symbol (possibly `pkg::`-qualified) head -- e.g.
# `quote(dplyr::across(x, mean))` -> "across", `quote(x + 1)` -> "+",
# `quote(x)` -> NULL (not a call at all).
# @keywords internal
call_head_name <- function(expr) {
  if (!is.call(expr)) {
    return(NULL)
  }
  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]
  }
  if (!is.symbol(head)) {
    return(NULL)
  }
  as.character(head)
}

# TRUE iff `expr` is a one-sided formula (`~ rhs`), e.g. `~ .x + 1`.
# @keywords internal
is_formula_expr <- function(expr) {
  is.call(expr) && identical(expr[[1]], as.name("~")) && length(expr) == 2
}

# TRUE iff `expr` is an (unevaluated) function definition -- covers both
# `function(x) ...` and `\(x) ...` (the backslash form parses to the same
# `` `function` ``-headed call as of R 4.1).
# @keywords internal
is_lambda_expr <- function(expr) {
  is.call(expr) && identical(expr[[1]], as.name("function"))
}

# Recursively substitute every bare `.x`/`.` symbol in `expr` with
# `replacement` (a column symbol), and every `cur_column()` call (arity 0)
# with the literal string `col_name` -- the two textual substitutions
# across()'s per-column function-template instantiation needs (section
# "Implementation" of this task's own brief). Never descends into a nested
# across()/if_any()/if_all()/pick() (dots_need_fallback() already routed
# any dot containing one of those to the CPU fallback before this is ever
# called, so encountering one here would be a logic error, not a case to
# handle silently) -- but since that invariant is enforced by the caller,
# this function just recurses uniformly; there is nothing further to guard.
#
# @param expr A language object (already quote()/expr()'d, never a quosure)
# @param replacement A symbol (or literal) to substitute `.x`/`.` with
# @param col_name Character scalar, this column's name (for `cur_column()`)
# @return `expr` with every substitution applied
# @keywords internal
substitute_across_placeholders <- function(expr, replacement, col_name) {
  if (is.symbol(expr)) {
    if (identical(expr, as.symbol(".x")) || identical(expr, as.symbol("."))) {
      return(replacement)
    }
    return(expr)
  }
  if (is.call(expr)) {
    if (identical(call_head_name(expr), "cur_column") && length(expr) == 1) {
      return(col_name)
    }
    parts <- as.list(expr)
    parts <- lapply(parts, substitute_across_placeholders,
                     replacement = replacement, col_name = col_name)
    return(as.call(parts))
  }
  expr
}

# Recursively substitute every zero-arg call to `fn_name` (bare or
# `pkg::`-qualified, e.g. `cur_group_id()`) with `replacement` (a symbol).
# Used by substitute_cur_group_id() below.
# @keywords internal
substitute_zero_arg_call <- function(expr, fn_name, replacement) {
  if (is.call(expr)) {
    if (identical(call_head_name(expr), fn_name) && length(expr) == 1) {
      return(replacement)
    }
    parts <- as.list(expr)
    parts <- lapply(parts, substitute_zero_arg_call, fn_name = fn_name, replacement = replacement)
    return(as.call(parts))
  }
  expr
}

# TRUE iff `expr` contains a zero-arg call to `fn_name` (bare or
# `pkg::`-qualified) anywhere in its tree.
# @keywords internal
expr_contains_zero_arg_call <- function(expr, fn_name) {
  if (!is.call(expr)) {
    return(FALSE)
  }
  if (identical(call_head_name(expr), fn_name) && length(expr) == 1) {
    return(TRUE)
  }
  for (part in as.list(expr)) {
    if (expr_contains_zero_arg_call(part, fn_name)) {
      return(TRUE)
    }
  }
  FALSE
}

# =============================================================================
# Whole-call CPU-fallback detection (pick()/cur_group()/cur_group_rows()/
# cur_data()/cur_data_all()/c_across()/nested-across-if_any-if_all/.unpack=)
# =============================================================================

# Function names that ALWAYS require the whole-call CPU fallback, in every
# verb, wherever they appear in a dot's expression tree (never just at the
# top level) -- see this file's own module docs for why each one is
# deferred rather than natively lowered.
deferred_fallback_fns <- c("pick", "cur_group", "cur_group_rows",
                            "cur_data", "cur_data_all", "c_across")

# TRUE iff any dot in `dots` needs the whole-call CPU fallback.
#
# @param dots A list of quosures (as captured by `rlang::enquos(...)`,
#   BEFORE any Phase 10 expansion/substitution)
# @param treat_cur_group_id_as_deferred Logical. `FALSE` (mutate()/
#   filter()'s own call site) -- `cur_group_id()` gets its own GPU-native
#   substitution (substitute_cur_group_id()), so it must NOT trigger this
#   fallback. `TRUE` (summarise()'s call site) -- summarise()'s per-GROUP
#   (not per-row) shape means the row-level substitution trick doesn't
#   directly produce a valid `fn(col)` aggregation dot, so `cur_group_id()`
#   inside `summarise()` is deferred to the whole-call fallback instead.
# @return Logical scalar
# @keywords internal
dots_need_fallback <- function(dots, treat_cur_group_id_as_deferred = FALSE) {
  extra_names <- if (treat_cur_group_id_as_deferred) {
    c(deferred_fallback_fns, "cur_group_id")
  } else {
    deferred_fallback_fns
  }
  any(vapply(
    dots,
    function(q) expr_needs_fallback(rlang::quo_get_expr(q), extra_names, top_level = TRUE),
    logical(1)
  ))
}

# @keywords internal
expr_needs_fallback <- function(expr, extra_names, top_level = FALSE) {
  if (!is.call(expr)) {
    return(FALSE)
  }
  head_name <- call_head_name(expr)
  if (!is.null(head_name)) {
    if (head_name %in% extra_names) {
      return(TRUE)
    }
    if (!top_level && head_name %in% c("across", "if_any", "if_all")) {
      return(TRUE)
    }
    if (top_level && identical(head_name, "across") && across_call_has_unpack(expr)) {
      return(TRUE)
    }
  }
  for (part in as.list(expr)[-1]) {
    if (expr_needs_fallback(part, extra_names, top_level = FALSE)) {
      return(TRUE)
    }
  }
  FALSE
}

# TRUE iff a top-level `across(...)` call expression supplies a truthy
# `.unpack = ` argument (Phase 10 scope cut: `.unpack` -- unpacking a
# data-frame-valued `.fns` result into multiple columns -- is not
# implemented; deferred to the whole-call CPU fallback instead of a hard
# error, since dplyr itself supports it).
# @keywords internal
across_call_has_unpack <- function(expr) {
  proto <- function(.cols, .fns, ..., .names = NULL, .unpack = FALSE) NULL
  matched <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(matched) || is.null(matched$.unpack)) {
    return(FALSE)
  }
  val <- tryCatch(eval(matched$.unpack, baseenv()), error = function(e) NULL)
  isTRUE(val)
}

# Internal: run the real dplyr verb on the CPU, for a call whose dots
# contain a deferred across()-family shape (dots_need_fallback() == TRUE).
# A thin wrapper over gpu_fallback() (R/fallback.R) -- the exact same
# "materialize, collect, restore group_by(), re-run the real verb, re-
# upload" primitive every other CPU-fallback verb in this package already
# uses.
#
# @param verb_name Character, for the fallback notification
# @param .data A tbl_gpu
# @param call_fn A function of one argument (a tibble, already grouped if
#   `.data` was grouped) that calls the real dplyr verb with the ORIGINAL
#   (pre-expansion) dots and every other original argument
# @return A new tbl_gpu (see gpu_fallback())
# @keywords internal
verb_across_fallback <- function(verb_name, .data, call_fn) {
  gpu_fallback(verb_name, .data, call_fn)
}

# =============================================================================
# cur_group_id(): GPU-native substitution (mutate()/filter() only)
# =============================================================================

# Substitute every `cur_group_id()` call in `dots` with a reference to a
# real, materialized helper column giving each row's own 1-based dense
# group-key rank -- dplyr's own `cur_group_id()` contract, verified
# empirically to match `group_data()`'s row order exactly (ascending
# group-key order, matching `compute_group_data()`'s existing, already
# GPU-native implementation, R/group-by.R, Phase 9). Ungrouped `.data`
# (or `.by=` resolving to zero columns) makes every row's id `1`
# (verified empirically against dplyr 1.2.1), handled directly with no
# join at all.
#
# No-op (returns `.data`/`dots` unchanged) if no dot references
# `cur_group_id()` at all -- the common case, and the only one that avoids
# materializing `.data`.
#
# @param .data A tbl_gpu (may be lazy with pending ops; materialized here
#   only if actually needed)
# @param dots A list of quosures (mutate()/filter()'s own dots, BEFORE
#   across()/if_any()/if_all() expansion -- cur_group_id() substitution
#   runs first, so any across()-generated dot inheriting `cur_column()`-
#   substituted text never itself needs to worry about cur_group_id())
# @param group_cols Character vector, the effective group columns
#   (`verb_groups` -- `.by=` columns if supplied, `.data$groups`
#   otherwise)
# @return `list(data = <possibly-materialized-and-joined tbl_gpu>, dots =
#   <possibly-rewritten dots>)`
# @keywords internal
substitute_cur_group_id <- function(.data, dots, group_cols) {
  has_any <- any(vapply(
    dots, function(q) expr_contains_zero_arg_call(rlang::quo_get_expr(q), "cur_group_id"),
    logical(1)
  ))
  if (!has_any) {
    return(list(data = .data, dots = dots, used = FALSE))
  }

  if (has_pending_ops(.data)) {
    .data <- compute(.data)
  }

  helper_name <- unique_temp_name("..cur_group_id..", current_schema(.data)$names)

  if (length(group_cols) == 0) {
    # Ungrouped (or `.by=` resolving to no columns): every row is group 1.
    step <- make_mutate_expr(helper_name, ir_lit_from_r(1L), .data$schema)
    new_data <- push_op(.data, ast_mutate(input_node(.data), list(step)))
  } else {
    tmp_grouped <- new_tbl_gpu(
      ptr = .data$ptr, schema = .data$schema, groups = group_cols,
      exec_mode = .data$exec_mode
    )
    gd <- compute_group_data(tmp_grouped)
    keys <- gd[setdiff(names(gd), ".rows")]
    keys[[helper_name]] <- seq_len(nrow(keys))
    keys_gpu <- tbl_gpu(keys, lazy = FALSE)

    joined <- dplyr::left_join(.data, keys_gpu, by = group_cols)
    # left_join.tbl_gpu() preserves `.data`'s own groups already (Phase 7/8);
    # rebuild explicitly regardless, so this never depends on that being
    # true for every future join code path.
    new_data <- new_tbl_gpu(
      ptr = joined$ptr, schema = joined$schema, groups = .data$groups,
      exec_mode = joined$exec_mode
    )
  }

  new_dots <- lapply(dots, function(q) {
    rlang::new_quosure(
      substitute_zero_arg_call(rlang::quo_get_expr(q), "cur_group_id", as.name(helper_name)),
      rlang::quo_get_env(q)
    )
  })
  names(new_dots) <- names(dots)

  list(data = new_data, dots = new_dots, used = TRUE, helper_name = helper_name)
}

# =============================================================================
# tidyselect resolution for across()/if_any()/if_all()'s `.cols` (and, by
# construction, the SAME rule pick() itself uses: `everything()` and every
# other tidyselect helper only ever "sees" the CURRENT GROUP's non-group
# columns -- verified empirically: even an EXPLICIT bare reference to a
# group column inside `.cols` is a "column doesn't exist" error, not just
# `everything()` implicitly skipping it).
# =============================================================================

# A one-row proxy value for a GPU schema type, so `where(is.numeric)`-style
# predicates evaluate correctly against tidyselect's data mask without
# needing any real GPU data.
# @keywords internal
schema_type_proxy_value <- function(type) {
  switch(type,
    BOOL8 = TRUE,
    INT32 = 1L,
    INT64 = 1L,
    FLOAT64 = 1,
    FLOAT32 = 1,
    STRING = "a",
    TIMESTAMP_DAYS = as.Date("1970-01-01"),
    TIMESTAMP_MICROSECONDS = as.POSIXct("1970-01-01", tz = "UTC"),
    DICTIONARY32 = factor("a"),
    NA
  )
}

# Resolve a `.cols`/`pick()`-style tidyselect expression against the
# columns of `schema` EXCLUDING `group_cols` (see this section's own
# header comment). Returns the selected column names, in tidyselect's own
# resolved order (duplicates/renames are not meaningful for across()'s own
# `.cols`, so `allow_rename = FALSE`).
#
# @param cols_expr A language object (the `.cols` argument's raw
#   expression)
# @param env The environment to evaluate the tidyselect expression in
# @param schema list(names=, types=)
# @param group_cols Character vector, columns to exclude from the
#   selectable universe entirely
# @return Character vector of selected column names (possibly empty)
# @keywords internal
resolve_across_cols <- function(cols_expr, env, schema, group_cols) {
  eligible <- setdiff(schema$names, group_cols)
  if (length(eligible) == 0) {
    return(character())
  }
  eligible_types <- unname(schema$types[match(eligible, schema$names)])

  proxy <- stats::setNames(
    lapply(eligible_types, schema_type_proxy_value),
    eligible
  )
  proxy_df <- tibble::as_tibble(proxy)

  sel <- tidyselect::eval_select(
    expr = cols_expr, data = proxy_df, env = env, allow_rename = FALSE
  )
  eligible[sel]
}

# =============================================================================
# across()'s `.fns` -> a list of "function specs", each instantiable per
# selected column
# =============================================================================

# One `.fns` entry, resolved to something instantiable per column.
# @keywords internal
make_fn_spec <- function(label, build) {
  list(label = label, build = build)
}

# Parse across()'s `.fns` argument (a single function-like value, or a
# `list()` of them) into a list of `make_fn_spec()` entries. Returns `NULL`
# if `fns_expr` is a shape this file doesn't understand syntactically (a
# bound variable holding a function/list, or anything else that isn't
# literally a symbol/formula/lambda/`list(...)` call at the quosure level)
# -- callers treat `NULL` as a hard, diagnosable error (dots_need_fallback()
# already screens out every shape genuinely meant to defer to the CPU, so
# reaching here with an unparseable `.fns` is a real "cuplyr doesn't
# understand this yet" case, not a deferred one).
#
# `missing_ok = TRUE` (across()'s own contract, verified empirically:
# `across(cols)` with `.fns` entirely omitted is IDENTITY, not an error --
# distinct from an explicit `.fns = NULL`, which real dplyr itself
# rejects) lets the caller signal "no `.fns` argument was supplied at
# all"; this function then returns a single identity spec.
#
# @param fns_expr The `.fns` argument's raw expression, or `NULL` if
#   entirely omitted (see `missing_ok`)
# @param env Environment to resolve any leftover free variables in a
#   formula/lambda body against
# @param extra_args A (possibly empty) named `list()` of already-evaluated
#   extra arguments from across()'s own `...` (the deprecated-but-still-
#   supported "extra args passed to a bare function name" form, e.g.
#   `across(x, mean, na.rm = TRUE)`) -- appended only to bare-symbol
#   (plain function name) specs, matching dplyr's own historical behavior
# @return A list of `make_fn_spec()` entries, or `NULL`
# @keywords internal
parse_across_fns <- function(fns_expr, env, extra_args = list()) {
  if (is.null(fns_expr)) {
    return(list(make_fn_spec(NA_character_, function(col_sym, col_name) col_sym)))
  }

  if (identical(call_head_name(fns_expr), "list")) {
    elts <- as.list(fns_expr)[-1]
    elt_names <- names(elts)
    if (is.null(elt_names)) elt_names <- rep("", length(elts))

    specs <- vector("list", length(elts))
    for (i in seq_along(elts)) {
      label <- if (nzchar(elt_names[i])) elt_names[i] else as.character(i)
      one <- parse_single_fn(elts[[i]], env, extra_args)
      if (is.null(one)) return(NULL)
      specs[[i]] <- make_fn_spec(label, one)
    }
    return(specs)
  }

  one <- parse_single_fn(fns_expr, env, extra_args)
  if (is.null(one)) return(NULL)
  list(make_fn_spec(NA_character_, one))
}

# Parse a SINGLE function-like `.fns` value (bare symbol / formula /
# lambda) into a `build(col_sym, col_name)` closure producing the
# instantiated expression for one column. Returns `NULL` for anything else
# (a bound variable, a call to something other than `function`, ...).
# @keywords internal
parse_single_fn <- function(fn_expr, env, extra_args = list()) {
  if (is.symbol(fn_expr)) {
    fn_name <- as.character(fn_expr)
    return(function(col_sym, col_name) {
      as.call(c(list(as.name(fn_name), col_sym), extra_args))
    })
  }

  if (is_formula_expr(fn_expr)) {
    rhs <- fn_expr[[2]]
    return(function(col_sym, col_name) {
      substitute_across_placeholders(rhs, col_sym, col_name)
    })
  }

  if (is_lambda_expr(fn_expr)) {
    formals_pairlist <- fn_expr[[2]]
    formal_names <- names(formals_pairlist)
    formal_names <- formal_names[formal_names != "..."]
    if (length(formal_names) != 1) {
      return(NULL)
    }
    param <- as.name(formal_names[1])
    body_expr <- fn_expr[[3]]
    return(function(col_sym, col_name) {
      substituted <- substitute_across_placeholders(body_expr, col_sym, col_name)
      # The lambda's own formal (whatever it's actually called, e.g. `v` in
      # `\(v) v + 1`) also needs substituting -- substitute_across_placeholders()
      # only rewrites the canonical `.x`/`.` spelling, so do the formal's own
      # name as a second pass.
      substitute_symbol(substituted, formal_names[1], col_sym)
    })
  }

  NULL
}

# Recursively substitute every bare occurrence of symbol named `sym_name`
# with `replacement` -- used for a lambda's own (arbitrarily-named) formal
# parameter, distinct from substitute_across_placeholders()'s fixed `.x`/
# `.`/`cur_column()` handling.
# @keywords internal
substitute_symbol <- function(expr, sym_name, replacement) {
  if (is.symbol(expr)) {
    if (identical(as.character(expr), sym_name)) {
      return(replacement)
    }
    return(expr)
  }
  if (is.call(expr)) {
    parts <- as.list(expr)
    parts <- lapply(parts, substitute_symbol, sym_name = sym_name, replacement = replacement)
    return(as.call(parts))
  }
  expr
}

# =============================================================================
# .names glue substitution -- across() only ever exposes `{.col}`/`{.fn}`
# in this position (verified empirically; no arbitrary glue expressions),
# so plain fixed-string substitution is enough, with no `glue` dependency.
# =============================================================================

# @keywords internal
apply_names_template <- function(template, col, fn_label) {
  out <- gsub("{.col}", col, template, fixed = TRUE)
  gsub("{.fn}", fn_label, out, fixed = TRUE)
}

# =============================================================================
# across() / if_any() / if_all() top-level expansion
# =============================================================================

# Prototype for `rlang::call_match()`-based argument resolution, shared by
# across()/if_any()/if_all() (if_any()/if_all() have no `.unpack`).
# @keywords internal
across_proto <- function(.cols, .fns, ..., .names = NULL, .unpack = FALSE) NULL
if_any_all_proto <- function(.cols, .fns, ..., .names = NULL) NULL

# Expand one top-level `across(.cols, .fns, ..., .names=, .unpack=)` dot
# into a named list of ordinary quosures, one per (selected column x
# resolved `.fns` entry) combination, in column-major order (matching
# dplyr's own `across()` output column order: every function for the
# first column, then every function for the second, ...).
#
# @param expr The `across(...)` call expression (already confirmed to be
#   `across`-headed by the caller)
# @param env The dot's own quosure environment
# @param schema Current schema (`list(names=, types=)`)
# @param group_cols Character vector, columns excluded from `.cols`'s
#   selectable universe
# @return A named list of quosures (possibly length 0, if `.cols` matched
#   no columns -- verified empirically: this is a silent no-op, not an
#   error)
# @keywords internal
expand_one_across <- function(expr, env, schema, group_cols) {
  matched <- rlang::call_match(expr, across_proto, defaults = FALSE)
  present <- as.list(matched)[-1]

  if (is.null(present$.cols)) {
    stop("across() requires a `.cols` argument.", call. = FALSE)
  }
  cols <- resolve_across_cols(present$.cols, env, schema, group_cols)
  if (length(cols) == 0) {
    return(stats::setNames(list(), character()))
  }

  extra_args <- present[!names(present) %in% c(".cols", ".fns", ".names", ".unpack")]
  extra_args <- lapply(extra_args, eval, envir = env)

  fns_expr <- present$.fns
  specs <- parse_across_fns(fns_expr, env, extra_args)
  if (is.null(specs)) {
    stop(
      "across()'s `.fns` must be a function, a formula (`~ .x + 1`), or a ",
      "list of them: ", rlang::expr_deparse(expr), call. = FALSE
    )
  }

  names_tmpl <- if (!is.null(present$.names)) {
    val <- tryCatch(eval(present$.names, envir = env), error = function(e) NULL)
    if (!is.character(val) || length(val) != 1) {
      stop("across()'s `.names` must be a single glue string.", call. = FALSE)
    }
    val
  } else if (length(specs) == 1) {
    "{.col}"
  } else {
    "{.col}_{.fn}"
  }

  out_names <- character()
  out_quos <- list()
  for (col in cols) {
    col_sym <- as.name(col)
    for (spec in specs) {
      built_expr <- spec$build(col_sym, col)
      out_name <- apply_names_template(names_tmpl, col, spec$label %||% "")
      out_names <- c(out_names, out_name)
      out_quos <- c(out_quos, list(rlang::new_quosure(built_expr, env)))
    }
  }
  names(out_quos) <- out_names
  out_quos
}

# Expand a top-level `if_any(.cols, .fns, ..., .names=)` / `if_all(...)`
# dot into a SINGLE quosure: the OR-fold (`if_any`) or AND-fold (`if_all`)
# of each selected column's own instantiated predicate -- exactly dplyr's
# own semantics (verified empirically: `NA` propagates through `|`/`&`
# with R's three-valued logic, identical to `filter()`'s own `&`/`|`
# handling already documented in R/filter.R).
#
# @param expr The `if_any(...)`/`if_all(...)` call expression
# @param env The dot's own quosure environment
# @param schema,group_cols As `expand_one_across()`
# @param mode `"if_any"` or `"if_all"`
# @return A single quosure (the folded predicate), or `NULL` if `.cols`
#   matched no columns (there is no sensible empty-fold value to return
#   without knowing the row count up front, so this is treated as a
#   deferred shape by the caller rather than guessed at)
# @keywords internal
expand_if_any_all <- function(expr, env, schema, group_cols, mode) {
  matched <- rlang::call_match(expr, if_any_all_proto, defaults = FALSE)
  present <- as.list(matched)[-1]

  if (is.null(present$.cols)) {
    stop(mode, "() requires a `.cols` argument.", call. = FALSE)
  }
  cols <- resolve_across_cols(present$.cols, env, schema, group_cols)
  if (length(cols) == 0) {
    return(NULL)
  }

  extra_args <- present[!names(present) %in% c(".cols", ".fns", ".names")]
  extra_args <- lapply(extra_args, eval, envir = env)

  fns_expr <- present$.fns
  specs <- parse_across_fns(fns_expr, env, extra_args)
  if (is.null(specs) || length(specs) != 1) {
    stop(
      mode, "()'s `.fns` must be a single function or formula: ",
      rlang::expr_deparse(expr), call. = FALSE
    )
  }
  spec <- specs[[1]]

  fold_op <- if (identical(mode, "if_any")) as.name("|") else as.name("&")

  combined <- NULL
  for (col in cols) {
    piece <- spec$build(as.name(col), col)
    combined <- if (is.null(combined)) piece else as.call(list(fold_op, combined, piece))
  }

  rlang::new_quosure(combined, env)
}

# =============================================================================
# The main driver, called from mutate.tbl_gpu()/summarise.tbl_gpu()/
# filter.tbl_gpu()/transmute.tbl_gpu() -- ASSUMES the caller has already
# confirmed `!dots_need_fallback(dots, ...)` (this function raises real
# errors for genuinely malformed input past that screen, it does not
# itself decide to defer anything).
# =============================================================================

#' Expand `across()`/`if_any()`/`if_all()` dots at the quosure level
#'
#' Walks `dots` in order; a top-level, unnamed `across(...)` dot is
#' replaced by [expand_one_across()]'s (possibly multi-element, possibly
#' empty) result, a top-level `if_any(...)`/`if_all(...)` dot is replaced
#' by [expand_if_any_all()]'s single folded quosure (keeping the ORIGINAL
#' dot's own name, if any -- e.g. `mutate(flag = if_any(...))`), and every
#' other dot is passed through unchanged. `across()` is disallowed as a
#' top-level `filter()` dot (matching real dplyr 1.2.1 exactly, verified
#' empirically -- it raises at EVALUATION time there, not parse time, but
#' the message ("must be a logical vector... use if_any()/if_all()
#' instead") is the same either way, so raising it up front here is
#' equivalent and cheaper). A NAMED top-level `across()` dot (e.g.
#' `mutate(z = across(x, y))`) is likewise unsupported here (dplyr's own
#' behavior for this shape is a genuinely strange packed-column name, not
#' a shape worth reproducing) -- callers should have already routed it to
#' the whole-call fallback via [dots_need_fallback()] (this function
#' itself will error for it, not fallback, if it slips through).
#'
#' @param dots A list of quosures (already past cur_group_id substitution,
#'   if applicable)
#' @param schema Current schema
#' @param group_cols Character vector, the effective group columns to
#'   exclude from `.cols`'s selectable universe
#' @param verb Character, `"filter"`/`"mutate"`/`"summarise"`/`"transmute"`
#' @return A new (possibly longer, possibly shorter) named list of
#'   quosures
#' @keywords internal
expand_across_dots <- function(dots, schema, group_cols, verb) {
  out_quos <- list()
  out_names <- character()

  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    nm <- names(dots)[i]
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)
    head_name <- call_head_name(expr)

    is_named <- !is.null(nm) && !is.na(nm) && nzchar(nm)

    if (!is_named && identical(head_name, "across")) {
      if (identical(verb, "filter")) {
        stop(
          "`across()` can't be used in filter() -- use `if_any()`/`if_all()` ",
          "instead.\nExpression: ", rlang::expr_deparse(expr), call. = FALSE
        )
      }
      expanded <- expand_one_across(expr, env, schema, group_cols)
      out_quos <- c(out_quos, expanded)
      out_names <- c(out_names, names(expanded))
      next
    }

    if (is_named && identical(head_name, "across")) {
      stop("A named `across()` dot is not supported.", call. = FALSE)
    }

    if (identical(head_name, "if_any") || identical(head_name, "if_all")) {
      combined <- expand_if_any_all(expr, env, schema, group_cols, head_name)
      if (is.null(combined)) {
        stop(
          head_name, "() matched no columns -- can't fold an empty selection.",
          call. = FALSE
        )
      }
      out_quos <- c(out_quos, list(combined))
      out_names <- c(out_names, if (is_named) nm else "")
      next
    }

    out_quos <- c(out_quos, list(quo))
    out_names <- c(out_names, if (is_named) nm else "")
  }

  names(out_quos) <- out_names
  out_quos
}
