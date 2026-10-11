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
substitute_across_placeholders <- function(expr, replacement, col_name, subst_dot = TRUE) {
  # An empty/missing argument slot (e.g. the elided row-index in `x[, 1]`,
  # which as.list()/recursion below can surface as one of `expr`'s own
  # "parts") is a special R sentinel: FORCING it (is.symbol()/is.call()/
  # identical(), even just reading it as a plain value) always raises
  # "argument ... is missing, with no default", regardless of context.
  # rlang::is_missing() is the one check that inspects it safely. Found via
  # Phase 11 task L1 test coverage (mutate(y = scale(x)[, 1])) -- a
  # pre-existing landmine in every one of this file's raw-expression
  # walkers, not something L1 itself introduces.
  if (rlang::is_missing(expr)) {
    return(expr)
  }
  if (is.symbol(expr)) {
    if (subst_dot && (identical(expr, as.symbol(".x")) || identical(expr, as.symbol(".")))) {
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
                     replacement = replacement, col_name = col_name,
                     subst_dot = subst_dot)
    return(as.call(parts))
  }
  expr
}

# Recursively substitute every zero-arg call to `fn_name` (bare or
# `pkg::`-qualified, e.g. `cur_group_id()`) with `replacement` (a symbol).
# Used by substitute_cur_group_id() below.
# @keywords internal
substitute_zero_arg_call <- function(expr, fn_name, replacement) {
  # See substitute_across_placeholders()'s identical guard just above for
  # why this is needed (the empty-argument-slot/rlang::is_missing() landmine).
  if (rlang::is_missing(expr)) {
    return(expr)
  }
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
  # See substitute_across_placeholders()'s identical guard for why this is
  # needed (the empty-argument-slot/rlang::is_missing() landmine) --
  # without it, `substitute_cur_group_id()`'s own `has_any` gate check
  # (called on EVERY mutate()/filter() dot, whether or not it references
  # cur_group_id() at all) crashes for a dot like `scale(x)[, 1]`.
  if (rlang::is_missing(expr)) {
    return(FALSE)
  }
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
  dot_names <- names(dots) %||% rep("", length(dots))
  any(vapply(
    seq_along(dots),
    function(i) {
      expr <- rlang::quo_get_expr(dots[[i]])
      # A NAMED top-level across() packs a data-frame column, which a
      # tbl_gpu can't hold: dplyr's own result (or, for filter(), its own
      # named-input error) comes from the whole-call fallback.
      if (!is.na(dot_names[i]) && nzchar(dot_names[i]) &&
          identical(call_head_name(expr), "across")) {
        return(TRUE)
      }
      expr_needs_fallback(expr, extra_names, top_level = TRUE)
    },
    logical(1)
  ))
}

# @keywords internal
expr_needs_fallback <- function(expr, extra_names, top_level = FALSE) {
  # See substitute_across_placeholders()'s identical guard for why this is
  # needed (the empty-argument-slot/rlang::is_missing() landmine) --
  # without it, dots_need_fallback() (called on EVERY mutate()/filter()/
  # summarise() dot) crashes for a dot like `scale(x)[, 1]`.
  if (rlang::is_missing(expr)) {
    return(FALSE)
  }
  # A quosure spliced into the tree (`f({{ var }})`); `{{ }}` of a missing
  # argument is a quosure wrapping the empty argument.
  if (rlang::is_quosure(expr)) {
    return(expr_needs_fallback(rlang::quo_get_expr(expr), extra_names, top_level))
  }
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

# TRUE iff a top-level `across(...)` call expression supplies any `.unpack = `
# argument other than a literal FALSE (Phase 10 scope cut: `.unpack` -- unpacking a
# data-frame-valued `.fns` result into multiple columns -- is not
# implemented; deferred to the whole-call CPU fallback instead of a hard
# error, since dplyr itself supports it).
# @keywords internal
across_call_has_unpack <- function(expr) {
  proto <- function(.cols, .fns, ..., .names = NULL, .unpack = FALSE) NULL
  matched <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(matched) || !".unpack" %in% names(matched)) {
    return(FALSE)
  }
  # Anything but a literal FALSE (TRUE, a glue string, a variable, an invalid
  # value dplyr must reject) goes to dplyr.
  !identical(matched$.unpack, FALSE)
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
# @param dots A list of quosures. mutate()/transmute() pass their dots
#   AFTER across()/if_any()/if_all() expansion (so the helper column is never
#   selectable and their CPU fallbacks can re-run the unsubstituted dots);
#   filter() passes them BEFORE expansion (so the if_any()/if_all() piece
#   check can type cur_group_id() pieces) and excludes the helper from
#   `.cols` itself
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
    # slice(.preserve = TRUE) fixed the group keys: numbering must run over
    # them (empty groups included), as group_indices() does.
    fixed <- if (identical(group_cols, .data$groups)) fixed_keys_of(.data)
    tmp_grouped <- new_tbl_gpu(
      ptr = .data$ptr, schema = .data$schema, groups = group_cols,
      exec_mode = .data$exec_mode, fixed_keys = fixed
    )
    gd <- compute_group_data(tmp_grouped)
    keys <- gd[setdiff(names(gd), ".rows")]
    keys[[helper_name]] <- seq_len(nrow(keys))
    keys_gpu <- tbl_gpu(keys, lazy = FALSE)

    joined <- dplyr::left_join(.data, keys_gpu, by = group_cols)
    # A lazy join is only a pending op; the rebuild below reads `$ptr`
    if (has_pending_ops(joined)) {
      joined <- compute(joined)
    }
    # left_join.tbl_gpu() preserves `.data`'s own groups already (Phase 7/8);
    # rebuild explicitly regardless, so this never depends on that being
    # true for every future join code path.
    new_data <- new_tbl_gpu(
      ptr = joined$ptr, schema = joined$schema, groups = .data$groups,
      exec_mode = joined$exec_mode, fixed_keys = fixed
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
# resolved order. Renames (`c(new = old)`) are allowed: the result is named
# by the NEW names, its values are the OLD (source) column names.
#
# @param cols_expr A language object (the `.cols` argument's raw
#   expression)
# @param env The environment to evaluate the tidyselect expression in
# @param schema list(names=, types=)
# @param group_cols Character vector, columns to exclude from the
#   selectable universe entirely
# @return Named character vector: values are the selected source columns,
#   names the output names (possibly empty)
# @keywords internal
resolve_across_cols <- function(cols_expr, env, schema, group_cols) {
  eligible <- setdiff(schema$names, group_cols)
  if (length(eligible) == 0) {
    return(stats::setNames(character(), character()))
  }
  eligible_types <- unname(schema$types[match(eligible, schema$names)])

  proxy <- stats::setNames(
    lapply(eligible_types, schema_type_proxy_value),
    eligible
  )
  proxy_df <- tibble::as_tibble(proxy)

  sel <- tidyselect::eval_select(
    expr = cols_expr, data = proxy_df, env = env, allow_rename = TRUE
  )
  if (anyDuplicated(names(sel))) {
    rlang::abort("Names must be unique.")
  }
  stats::setNames(eligible[sel], names(sel))
}

# Resolve a TOP-LEVEL `pick(...)` / `dplyr::pick(...)` dot (group_by(),
# arrange()) to column names. dplyr expands pick() once, on the FULL current
# columns (no group exclusion: unlike mutate()/summarise(), these verbs see
# the ungrouped data), in the quosure's environment. Returns the selected
# column names (character(0) for `pick(NULL)`), or NULL when the dot is not a
# natively resolvable pick: not a pick call, no arguments, a rename, or a
# tidyselect error -- the caller then falls back to dplyr, which raises its
# own error (or succeeds) for those.
# @param quo A quosure (one dot)
# @param schema list(names=, types=)
# @keywords internal
resolve_pick_call <- function(quo, schema) {
  expr <- rlang::quo_get_expr(quo)
  if (!rlang::is_call(expr) || length(expr) < 2L) {
    return(NULL)
  }
  head <- expr[[1]]
  is_pick <- identical(head, quote(pick)) ||
    (rlang::is_call(head, "::") && identical(head[[2]], quote(dplyr)) &&
       identical(head[[3]], quote(pick)))
  if (!is_pick) {
    return(NULL)
  }
  args <- as.list(expr)[-1]
  if (!is.null(names(args)) && any(names(args) != "")) {
    return(NULL)
  }
  # The proxies carry only the column TYPE, so a selection that looks at
  # values (a lambda or user predicate in where()) cannot be resolved here
  if (!pick_selection_is_type_only(expr, rlang::quo_get_env(quo), schema$names)) {
    return(NULL)
  }
  sel <- tryCatch(
    resolve_across_cols(rlang::call2("c", !!!args), rlang::quo_get_env(quo), schema, character()),
    error = function(e) NULL
  )
  if (is.null(sel) || !identical(unname(names(sel)), unname(as.character(sel)))) {
    return(NULL)
  }
  unname(as.character(sel))
}

# Is every `where()` predicate in a pick() selection a known type-only
# predicate, and every other call a tidyselect helper or operator? `x` is a
# pick() call (or an argument of one); embedded `{{ }}` quosures are walked in
# their own environment. Anything else (a lambda, a user function, an unknown
# call) could depend on column values, which the type proxies don't have.
# A bare symbol must name a column: tidyselect would otherwise look it up in
# the environment, where it may be a predicate function (`pick(is_complete)`
# is still accepted, with a deprecation warning) or an external vector. Of the
# other non-calls only constants (names, positions, NULL) are safe; a function
# spliced in with `!!` would also be applied as a predicate.
# @param cols Column names of the schema the selection resolves against
# @keywords internal
pick_selection_is_type_only <- function(x, env, cols) {
  # `pick(a, )`: an empty argument slot can't be classified
  if (rlang::is_missing(x)) {
    return(FALSE)
  }
  if (rlang::is_quosure(x)) {
    return(pick_selection_is_type_only(rlang::quo_get_expr(x), rlang::quo_get_env(x), cols))
  }
  if (is.symbol(x)) {
    return(as.character(x) %in% cols)
  }
  if (!is.call(x)) {
    return(is.atomic(x) || is.null(x))
  }
  name <- pick_call_name(x[[1]])
  args <- as.list(x)[-1]
  walk_args <- function() {
    all(vapply(args, pick_selection_is_type_only, logical(1), env = env, cols = cols))
  }
  if (identical(name, "pick")) {
    return(walk_args())
  }
  if (identical(name, "where")) {
    return(length(args) == 1L && pick_predicate_is_type_only(args[[1]], env))
  }
  if (is.null(name)) {
    return(FALSE)
  }
  if (name %in% c("c", "-", "!", "&", "|", "(", ":")) {
    return(walk_args())
  }
  # other tidyselect helpers take names/patterns, never column values
  name %in% c(
    "everything", "last_col", "starts_with", "ends_with", "contains", "matches",
    "num_range", "all_of", "any_of", "group_cols"
  )
}

# `f` / `pkg::f` call head -> "f" (tidyselect/dplyr/base namespaces only)
# @keywords internal
pick_call_name <- function(head) {
  if (is.symbol(head)) {
    return(as.character(head))
  }
  if (rlang::is_call(head, "::", n = 2L) && is.symbol(head[[3]]) &&
        as.character(head[[2]]) %in% c("tidyselect", "dplyr", "base")) {
    return(as.character(head[[3]]))
  }
  NULL
}

# A `where()` predicate resolving (in `env`) to one of base R's type tests.
# @keywords internal
pick_predicate_is_type_only <- function(pred, env) {
  if (rlang::is_quosure(pred)) {
    return(pick_predicate_is_type_only(rlang::quo_get_expr(pred), rlang::quo_get_env(pred)))
  }
  if (!is.symbol(pred) && !rlang::is_call(pred, "::", n = 2L)) {
    return(FALSE)
  }
  fn <- tryCatch(eval(pred, env), error = function(e) NULL)
  is.function(fn) && any(vapply(
    list(is.numeric, is.double, is.integer, is.character, is.logical, is.factor),
    identical, logical(1), fn
  ))
}

# `.cols` of a missing-`.cols` across()/if_any()/if_all(): dplyr's deprecated
# everything() default, with its lifecycle warning.
# @keywords internal
default_across_cols <- function(fn) {
  lifecycle::deprecate_warn(
    "1.1.0", I(paste0("Using `", fn, "()` without supplying `.cols`")),
    details = "Please supply `.cols` instead.",
    env = asNamespace("dplyr"), user_env = rlang::global_env()
  )
  rlang::expr(tidyselect::everything())
}

# =============================================================================
# across()'s `.fns` -> a list of "function specs", each instantiable per
# selected column
# =============================================================================

# One `.fns` entry, resolved to something instantiable per column.
# `env`, when set, is the environment the instantiated expression must be
# evaluated in (a formula/closure VALUE keeps its own environment, a literal
# lambda uses the dot's).
# @keywords internal
make_fn_spec <- function(label, build, env = NULL) {
  list(label = label, build = build, env = env)
}

# Function names the IR / aggregation layer can lower natively. A `.fns`
# VALUE identical to one of these is emitted as a plain `name(col)` call.
# @keywords internal
across_known_fn_names <- function() {
  union(names(ir_call_registry), c("quantile", "any", "all", "count"))
}

# Call heads an inlined VALUE body may use: everything the IR lowers by name.
# A body with any other call (UseMethod, missing, sapply, a nested function,
# `{`, `pkg::fn`, ...) is not inlined, since dplyr calls the value in a real
# frame instead.
# @keywords internal
across_native_heads <- function() {
  c(across_known_fn_names(), "(", "ceiling", "paste0", "paste", "startsWith", "endsWith")
}

# TRUE iff every call in `expr` has a bare-symbol head the IR knows by name.
# @keywords internal
across_expr_native <- function(expr) {
  if (rlang::is_quosure(expr)) {
    return(across_expr_native(rlang::quo_get_expr(expr)))
  }
  if (rlang::is_missing(expr) || !is.call(expr)) {
    return(TRUE)
  }
  head <- expr[[1]]
  if (!is.symbol(head) || !as.character(head) %in% across_native_heads()) {
    return(FALSE)
  }
  all(vapply(as.list(expr)[-1], across_expr_native, logical(1)))
}

# The registry name `value` is, or NULL. The IR lowers a call by its bare
# name (it drops any `pkg::`), as the function dplyr itself sees under that
# name, so `value` must be that function: `stats::lag` is not the IR's `lag`,
# `dplyr::lag` is. A bare name must also resolve to `value` from `env`, so
# the emitted `name(col)` means the same thing if an expression fallback
# evaluates it.
# @keywords internal
across_known_fn_name <- function(value, env, fn_expr = NULL) {
  qualified <- is.call(fn_expr) && length(fn_expr) == 3 &&
    identical(fn_expr[[1]], as.name("::")) && is.symbol(fn_expr[[3]])
  candidates <- c(across_known_fn_names(), "identity")
  if (qualified) {
    candidates <- intersect(as.character(fn_expr[[3]]), candidates)
  }
  dplyr_ns <- asNamespace("dplyr")
  for (nm in candidates) {
    canonical <- get0(nm, envir = dplyr_ns, mode = "function")
    if (is.null(canonical) || !identical(canonical, value)) {
      next
    }
    if (qualified || identical(get0(nm, envir = env, mode = "function"), value)) {
      return(nm)
    }
  }
  NULL
}

# TRUE iff an inlined body would read a data column or the pronouns where
# dplyr's call of the VALUE would see its own environment instead.
# @param vars Free variables of the body (all.vars()), minus its argument
# @param scope Every column name in scope: the pre-call schema plus every
#   column an earlier dot of the same call creates
# @keywords internal
across_free_vars_hit_data <- function(vars, scope) {
  # An NA in `scope` means an earlier dot may have added columns we can't
  # name (an unnamed dot that returns a data frame): any free variable may
  # be one of them.
  if (anyNA(scope)) return(length(vars) > 0)
  any(vars %in% c(scope, ".data", ".env"))
}

# The arguments dplyr's lambda/formula inlining rules reject.
# @keywords internal
across_formula_inlinable <- function(fml_expr) {
  nms <- all.names(fml_expr)
  !(any(grepl("\\.\\.[0-9]|\\.y", nms)) || "return" %in% nms)
}

# Build spec for an inlined formula right-hand side.
# @keywords internal
across_formula_build <- function(rhs) {
  function(col_sym, col_name) substitute_across_placeholders(rhs, col_sym, col_name)
}

# Build spec for an inlined one-formal lambda body.
# @keywords internal
across_lambda_build <- function(formal, body_expr) {
  function(col_sym, col_name) {
    substituted <- substitute_across_placeholders(body_expr, col_sym, col_name, subst_dot = FALSE)
    substitute_symbol(substituted, formal, col_sym)
  }
}

# `spec` when its instantiated body is fully native, else NULL.
# @keywords internal
across_native_spec <- function(spec) {
  probe <- spec$build(as.name("..across_probe.."), "..across_probe..")
  if (across_expr_native(probe)) spec else NULL
}

# Dispatch on a `.fns` VALUE (evaluated in the dot's environment). dplyr
# never inlines a value: it calls it, with the value's own environment and
# a real call frame. So a value is turned into an expression only when that
# is unobservable: it is a function the IR knows by identity, or its body
# only uses calls the IR lowers by name and none of its free variables is a
# column in scope or a pronoun. Returns a make_fn_spec(), or NULL for "let
# dplyr handle it".
#
# @param value The evaluated function / formula
# @param fn_expr The expression it came from (NULL if there is none)
# @param scope Column names in scope (see across_free_vars_hit_data())
# @keywords internal
across_value_spec <- function(value, fn_expr, env, scope, extra_args) {
  has_extra <- length(extra_args) > 0
  if (inherits(value, "formula")) {
    if (has_extra || !rlang::is_formula(value, lhs = FALSE) || !across_formula_inlinable(value)) {
      return(NULL)
    }
    rhs <- value[[2]]
    if (across_free_vars_hit_data(setdiff(all.vars(rhs), c(".x", ".")), scope)) {
      return(NULL)
    }
    return(across_native_spec(make_fn_spec(NULL, across_formula_build(rhs), env = environment(value))))
  }
  if (!is.function(value)) {
    return(NULL)
  }

  known <- across_known_fn_name(value, env, fn_expr)
  if (!is.null(known)) {
    if (identical(known, "identity") && !has_extra) {
      return(make_fn_spec(NULL, function(col_sym, col_name) col_sym))
    }
    # Keep a `pkg::` qualification: the IR ignores it, but an expression
    # fallback that evaluates the call must find the same function.
    head <- if (is.call(fn_expr) && identical(fn_expr[[1]], as.name("::"))) fn_expr else as.name(known)
    return(make_fn_spec(NULL, function(col_sym, col_name) {
      as.call(c(list(head, col_sym), extra_args))
    }))
  }

  if (has_extra || is.primitive(value) || isNamespace(environment(value))) {
    return(NULL)
  }
  fmls <- formals(value)
  body_expr <- body(value)
  if (length(fmls) != 1 || identical(names(fmls), "...") || "return" %in% all.names(body_expr)) {
    return(NULL)
  }
  formal <- names(fmls)[1]
  if (across_free_vars_hit_data(setdiff(all.vars(body_expr), formal), scope)) {
    return(NULL)
  }
  across_native_spec(make_fn_spec(NULL, across_lambda_build(formal, body_expr), env = environment(value)))
}

# Resolve ONE `.fns` expression (a list element or the whole argument) to a
# make_fn_spec(), or NULL when it needs dplyr: a literal function/formula is
# inlined iff dplyr would inline it; anything else is evaluated and
# dispatched on its value. A quosure (from `{{ fn }}`) is looked through to
# its expression and environment, as if it had been written there.
# @keywords internal
parse_single_fn <- function(fn_expr, env, scope, extra_args = list()) {
  if (rlang::is_missing(fn_expr)) {
    return(NULL)
  }
  if (rlang::is_quosure(fn_expr)) {
    return(parse_single_fn(rlang::quo_get_expr(fn_expr), rlang::quo_get_env(fn_expr), scope, extra_args))
  }
  if (!inherits(fn_expr, "formula")) {
    if (is_formula_expr(fn_expr)) {
      if (length(extra_args) > 0 || !across_formula_inlinable(fn_expr)) {
        return(NULL)
      }
      return(make_fn_spec(NULL, across_formula_build(fn_expr[[2]]), env = env))
    }
    if (is_lambda_expr(fn_expr)) {
      formals_pairlist <- fn_expr[[2]]
      body_expr <- fn_expr[[3]]
      if (length(extra_args) > 0 || length(formals_pairlist) != 1 ||
          "return" %in% all.names(body_expr)) {
        return(NULL)
      }
      return(make_fn_spec(NULL, across_lambda_build(names(formals_pairlist)[1], body_expr), env = env))
    }
  }
  value <- tryCatch(
    if (is.language(fn_expr) && !inherits(fn_expr, "formula")) eval(fn_expr, env) else fn_expr,
    error = function(e) NULL
  )
  if (is.null(value)) {
    return(NULL)
  }
  across_value_spec(value, fn_expr, env, scope, extra_args)
}

# Parse across()'s `.fns` argument into `list(specs, is_list)`, where
# `specs` is a list of make_fn_spec() entries (labelled for `{.fn}`) and
# `is_list` says whether the argument was a list (which changes the default
# `.names`). Returns NULL when dplyr must run the call instead (an
# unsupported or invalid shape, an error while evaluating `.fns`, ...).
#
# `fns_expr = NULL` means `.fns` was omitted (across()'s identity).
#
# @param fns_expr The `.fns` argument's raw expression, or `NULL`
# @param env The dot's environment
# @param scope Column names in scope (pre-call schema plus names earlier
#   dots create)
# @param extra_args Already-checked extra arguments from across()'s `...`
# @keywords internal
parse_across_fns <- function(fns_expr, env, scope, extra_args = list()) {
  if (is.null(fns_expr)) {
    return(list(
      specs = list(make_fn_spec("1", function(col_sym, col_name) col_sym)),
      is_list = FALSE
    ))
  }

  # `{{ fn }}` in a wrapper: behave exactly as if the caller's expression
  # had been written here, in the caller's environment.
  if (rlang::is_quosure(fns_expr)) {
    return(parse_across_fns(rlang::quo_get_expr(fns_expr), rlang::quo_get_env(fns_expr), scope, extra_args))
  }

  if (identical(call_head_name(fns_expr), "list") && !inherits(fns_expr, "formula")) {
    elts <- as.list(fns_expr)[-1]
    if (length(elts) == 0) {
      return(NULL)
    }
    elt_names <- names(elts)
    if (is.null(elt_names)) elt_names <- rep("", length(elts))
    specs <- vector("list", length(elts))
    for (i in seq_along(elts)) {
      one <- parse_single_fn(elts[[i]], env, scope, extra_args)
      if (is.null(one)) return(NULL)
      one$label <- if (nzchar(elt_names[i])) elt_names[i] else as.character(i)
      specs[[i]] <- one
    }
    return(list(specs = specs, is_list = TRUE))
  }

  # A literal function/formula
  if (!inherits(fns_expr, "formula") && (is_formula_expr(fns_expr) || is_lambda_expr(fns_expr))) {
    one <- parse_single_fn(fns_expr, env, scope, extra_args)
    if (is.null(one)) return(NULL)
    one$label <- "1"
    return(list(specs = list(one), is_list = FALSE))
  }

  # Anything else is a value: evaluate once, dispatch on its type
  value <- tryCatch(
    if (is.language(fns_expr) && !inherits(fns_expr, "formula")) eval(fns_expr, env) else fns_expr,
    error = function(e) NULL
  )
  if (is.null(value)) {
    return(NULL)
  }
  if (is.list(value) && !inherits(value, "formula") && !is.data.frame(value)) {
    if (length(value) == 0) {
      return(NULL)
    }
    vnames <- names(value)
    if (is.null(vnames)) vnames <- rep("", length(value))
    specs <- vector("list", length(value))
    for (i in seq_along(value)) {
      one <- across_value_spec(value[[i]], NULL, env, scope, extra_args)
      if (is.null(one)) return(NULL)
      one$label <- if (nzchar(vnames[i])) vnames[i] else as.character(i)
      specs[[i]] <- one
    }
    return(list(specs = specs, is_list = TRUE))
  }
  one <- across_value_spec(value, fns_expr, env, scope, extra_args)
  if (is.null(one)) return(NULL)
  one$label <- "1"
  list(specs = list(one), is_list = FALSE)
}

# TRUE for a numeric constant expression such as `-1` or `1/3`.
# @keywords internal
across_const_expr <- function(a) {
  if (is.numeric(a) && length(a) == 1) {
    return(TRUE)
  }
  is.call(a) && is.symbol(a[[1]]) && as.character(a[[1]]) %in% c("-", "+", "*", "/", "^", "(") &&
    all(vapply(as.list(a)[-1], across_const_expr, logical(1)))
}

# across()'s extra `...` arguments, if they can be passed natively: every one
# must be a constant (including simple numeric arithmetic such as `-1` or
# `1/3`) or a symbol that is not a column in scope and evaluates cleanly in
# `env`. NULL means "let dplyr evaluate them" (they are data-masked there,
# once per group).
# @param scope Column names in scope (see across_free_vars_hit_data())
# @keywords internal
across_extra_args <- function(args, env, scope) {
  out <- list()
  for (i in seq_along(args)) {
    a <- args[[i]]
    val <- if (is.symbol(a)) {
      nm <- as.character(a)
      if (anyNA(scope) || nm %in% c(scope, ".data", ".env")) return(NULL)
      tryCatch(eval(a, env), error = function(e) return(NULL))
    } else if (is.atomic(a) && length(a) == 1) {
      a
    } else if (across_const_expr(a)) {
      eval(a, baseenv())
    } else {
      return(NULL)
    }
    if (is.null(val)) return(NULL)
    out[i] <- list(val)
  }
  names(out) <- names(args)
  out
}

# Recursively substitute every bare occurrence of symbol named `sym_name`
# with `replacement` -- used for a lambda's own (arbitrarily-named) formal
# parameter, distinct from substitute_across_placeholders()'s fixed `.x`/
# `.`/`cur_column()` handling.
# @keywords internal
substitute_symbol <- function(expr, sym_name, replacement) {
  if (rlang::is_missing(expr)) {
    return(expr)
  }
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
# @param scope Column names in scope for the free-variable guards: `schema`'s
#   plus every name an earlier dot of the same call creates
# @return A named list of quosures (possibly length 0, if `.cols` matched
#   no columns -- verified empirically: this is a silent no-op, not an
#   error). Attributes: `"needs_fallback"` is `TRUE` when dplyr must run the
#   whole call (an unsupported `.fns`/extra-argument shape, duplicate output
#   names); `"hazard"` is `TRUE` when an output name is read by a later
#   expansion (see the loop below).
# @keywords internal
expand_one_across <- function(expr, env, schema, group_cols, scope = schema$names) {
  matched <- rlang::call_match(expr, across_proto, defaults = FALSE)
  present <- as.list(matched)[-1]

  # An explicit `.cols = NULL` selects nothing; only an absent `.cols` takes
  # the deprecated everything() default.
  cols_expr <- if (".cols" %in% names(present)) present$.cols else default_across_cols("across")
  cols <- resolve_across_cols(cols_expr, env, schema, group_cols)
  if (length(cols) == 0) {
    return(stats::setNames(list(), character()))
  }

  # Anything this file can't express natively hands the whole call to dplyr
  fallback <- function() across_fallback_result()

  extra_args <- across_extra_args(
    present[!names(present) %in% c(".cols", ".fns", ".names", ".unpack")], env, scope
  )
  if (is.null(extra_args)) {
    return(fallback())
  }

  fns_given <- ".fns" %in% names(present)
  if (fns_given && is.null(present[[".fns"]])) {
    return(fallback())
  }
  parsed <- parse_across_fns(if (fns_given) present[[".fns"]] else NULL, env, scope, extra_args)
  if (is.null(parsed)) {
    return(fallback())
  }
  specs <- parsed$specs

  names_tmpl <- if (!is.null(present$.names)) {
    val <- tryCatch(eval(present$.names, envir = env), error = function(e) NULL)
    if (!is.character(val) || length(val) != 1 || is.na(val)) {
      return(fallback())
    }
    # Only {.col}/{.fn} are native; any other glue is dplyr's to evaluate
    if (grepl("[{}]", gsub("{.fn}", "", gsub("{.col}", "", val, fixed = TRUE), fixed = TRUE))) {
      return(fallback())
    }
    val
  } else if (parsed$is_list) {
    "{.col}_{.fn}"
  } else {
    "{.col}"
  }

  out_names <- character()
  out_quos <- list()
  # dplyr evaluates every function of one across() against the data as it was
  # before the across(); the expanded dots run in sequence, so an output name
  # that a later expansion reads would change that expansion's values. The
  # source column itself counts as read even when only a string names it
  # (`get(cur_column())` falls back above); `.data[["y"]]` reads `y`, and a
  # non-literal `.data[[nm]]` is treated as reading every column.
  hazard <- FALSE
  for (i in seq_along(cols)) {
    col <- cols[[i]]
    col_sym <- as.name(col)
    for (spec in specs) {
      built_expr <- spec$build(col_sym, col)
      # get()/get0()/mget() look a column up by string behind the IR's back
      # (it would fail to resolve the name): dplyr runs those.
      if (any(c("get", "get0", "mget") %in% all.names(built_expr))) {
        return(fallback())
      }
      out_name <- apply_names_template(names_tmpl, names(cols)[i], spec$label %||% "")
      reads <- across_string_reads(built_expr)
      if (col %in% out_names || any(out_names %in% c(all.vars(built_expr), reads$names)) ||
          (length(out_names) > 0 && reads$dynamic)) {
        hazard <- TRUE
      }
      out_names <- c(out_names, out_name)
      out_quos <- c(out_quos, list(rlang::new_quosure(built_expr, spec$env %||% env)))
    }
  }
  if (anyDuplicated(out_names)) {
    return(fallback())
  }
  names(out_quos) <- out_names
  attr(out_quos, "needs_fallback") <- FALSE
  attr(out_quos, "hazard") <- hazard
  out_quos
}

# An empty expansion flagged for the whole-call CPU fallback.
# @keywords internal
across_fallback_result <- function() {
  out <- stats::setNames(list(), character())
  attr(out, "needs_fallback") <- TRUE
  out
}

# The columns `expr` reads through `.data[[...]]`, which all.vars() can't see:
# `names` are the string-literal ones, `dynamic` says some index is not a
# literal (so any column may be read). Other `[[` uses (a list, a vector) are
# not column reads.
# @keywords internal
across_string_reads <- function(expr) {
  out <- list(names = character(), dynamic = FALSE)
  walk <- function(e) {
    if (rlang::is_missing(e) || !is.call(e)) {
      return(invisible())
    }
    if (identical(e[[1]], as.name("[[")) && length(e) >= 3 && identical(e[[2]], as.name(".data"))) {
      idx <- e[[3]]
      if (is.character(idx) && length(idx) == 1) {
        out$names <<- c(out$names, idx)
      } else {
        out$dynamic <<- TRUE
      }
    }
    for (part in as.list(e)) walk(part)
  }
  walk(expr)
  out
}

# Rewrite one across() dot's expansions so they all read the data as it was
# before the dot (mutate()/transmute()): every expansion is computed into a
# temp column first, then assigned to its real name. The caller drops the
# temps.
#
# @param expanded The expansion (named list of quosures)
# @param taken Names the temps must not collide with
# @return List of quosures (temps, then assignments) with a `"temps"`
#   attribute naming the temp columns
# @keywords internal
stage_across_expansion <- function(expanded, taken) {
  n <- length(expanded)
  temps <- character(n)
  for (k in seq_len(n)) {
    temps[k] <- unique_temp_name(paste0("..across_", k, ".."), c(taken, temps[seq_len(k - 1)]))
  }
  computed <- expanded
  attributes(computed) <- NULL
  names(computed) <- temps
  assigned <- lapply(seq_len(n), function(k) {
    rlang::new_quosure(as.name(temps[k]), rlang::quo_get_env(expanded[[k]]))
  })
  names(assigned) <- names(expanded)
  out <- c(computed, assigned)
  attr(out, "temps") <- temps
  out
}

# dplyr's wording for the class of a non-logical if_any()/if_all() piece.
# @keywords internal
across_type_label <- function(type) {
  switch(type,
    FLOAT64 = , FLOAT32 = "a double vector",
    INT32 = , INT64 = "an integer vector",
    STRING = "a character vector",
    DICTIONARY32 = "a <factor> object",
    TIMESTAMP_DAYS = "a <Date> object",
    TIMESTAMP_SECONDS = , TIMESTAMP_MILLISECONDS = , TIMESTAMP_MICROSECONDS = ,
    TIMESTAMP_NANOSECONDS = "a <POSIXct> object",
    paste0("a ", tolower(type), " vector")
  )
}

# if_any()/if_all() require every per-column predicate to be logical. dplyr
# checks the values; here the piece's inferred type stands in for them, so a
# piece the IR can't type is left to the normal parse path.
# @param piece The instantiated predicate for `col`
# @keywords internal
check_across_predicate_piece <- function(piece, col, env, schema) {
  # mutate() expands before substituting cur_group_id(), so a piece may still
  # contain the call; type it as the INT32 helper it will become.
  if (expr_contains_zero_arg_call(piece, "cur_group_id")) {
    helper <- unique_temp_name("..cur_group_id..", schema$names)
    piece <- substitute_zero_arg_call(piece, "cur_group_id", as.name(helper))
    schema$names <- c(schema$names, helper)
    schema$types <- c(schema$types, "INT32")
  }
  type <- tryCatch(suppressWarnings({
    ir <- ir_parse_quo(rlang::new_quosure(piece, env), schema)
    if (is.null(ir)) NULL else ir_infer_type(ir, schema)
  }), error = function(e) NULL)
  if (is.null(type) || identical(unname(type), "BOOL8")) {
    return(invisible())
  }
  rlang::abort(sprintf("`%s` must be a logical vector, not %s.", col,
                       across_type_label(unname(type))))
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
# @param verb The calling verb; only `"summarise"` changes the result
# @return A single quosure (the folded predicate; the literal FALSE/TRUE
#   if `.cols` matched no columns)
# @keywords internal
expand_if_any_all <- function(expr, env, schema, group_cols, mode, verb = "filter",
                              scope = schema$names) {
  matched <- rlang::call_match(expr, if_any_all_proto, defaults = FALSE)
  present <- as.list(matched)[-1]

  cols_expr <- if (".cols" %in% names(present)) present$.cols else default_across_cols(mode)
  cols <- unname(resolve_across_cols(cols_expr, env, schema, group_cols))
  if (length(cols) == 0) {
    # dplyr: if_any() over nothing is FALSE, if_all() is TRUE
    value <- identical(mode, "if_all")
    # summarise() only knows aggregations, so a constant there is an
    # aggregation over the first column's NA pattern; everywhere else it is
    # the literal.
    if (identical(verb, "summarise") && length(schema$names) > 0) {
      col <- as.name(c(setdiff(schema$names, group_cols), schema$names)[1])
      probe <- if (value) {
        rlang::expr(all(is.na(!!col) | !is.na(!!col), na.rm = TRUE))
      } else {
        rlang::expr(any(is.na(!!col) & !is.na(!!col), na.rm = TRUE))
      }
      return(rlang::new_quosure(probe, env))
    }
    return(rlang::new_quosure(value, env))
  }

  fallback <- function() {
    structure(rlang::new_quosure(FALSE, env), needs_fallback = TRUE)
  }

  extra_args <- across_extra_args(
    present[!names(present) %in% c(".cols", ".fns", ".names")], env, scope
  )
  if (is.null(extra_args)) {
    return(fallback())
  }
  fns_given <- ".fns" %in% names(present)
  if (fns_given && is.null(present[[".fns"]])) {
    return(fallback())
  }
  parsed <- parse_across_fns(if (fns_given) present[[".fns"]] else NULL, env, scope, extra_args)
  if (is.null(parsed)) {
    return(fallback())
  }

  fold_op <- if (identical(mode, "if_any")) as.name("|") else as.name("&")

  # One quosure carries one environment, so the functions must share it
  envs <- lapply(parsed$specs, function(sp) sp$env %||% env)
  if (!all(vapply(envs, identical, logical(1), envs[[1]]))) {
    return(fallback())
  }
  piece_env <- envs[[1]]

  # Every function of a list contributes one piece per column
  combined <- NULL
  for (col in cols) {
    for (spec in parsed$specs) {
      piece <- spec$build(as.name(col), col)
      check_across_predicate_piece(piece, col, piece_env, schema)
      combined <- if (is.null(combined)) piece else as.call(list(fold_op, combined, piece))
    }
  }

  rlang::new_quosure(combined, piece_env)
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
#' `mutate(z = across(x, y))`) packs a data-frame column and never gets
#' here: [dots_need_fallback()] sends the whole call to the CPU fallback.
#'
#' The result carries a `"needs_fallback"` attribute. It is `TRUE` when the
#' schema or the `.fns` values show that dplyr must run the call:
#' an unsupported `.fns`/extra-argument shape (a non-inlinable lambda, a
#' value whose free variables are columns, ...), duplicate output names, or,
#' in `summarise()`/`filter()`, an `across()` that renames onto a column that a
#' later expansion of the same `across()` reads (`across(c(y = x, z = y),
#' ...)`): dplyr evaluates every function against the pre-`across()` data,
#' while the expanded dots would run in sequence. That needs the schema, so
#' [dots_need_fallback()] can't see it; the verb calls then re-run the
#' ORIGINAL dots on the CPU. In `mutate()`/`transmute()` the same hazard is
#' staged instead: the expansions are computed into temp columns first and
#' assigned afterwards, and the `"staged_temps"` attribute names the temps the
#' caller must drop (and whose CPU fallbacks must use the ORIGINAL dots).
#'
#' @param dots A list of quosures (already past cur_group_id substitution,
#'   if applicable)
#' @param schema Current schema
#' @param group_cols Character vector, the effective group columns to
#'   exclude from `.cols`'s selectable universe
#' @param verb Character, `"filter"`/`"mutate"`/`"summarise"`/`"transmute"`
#' @return A new (possibly longer, possibly shorter) named list of
#'   quosures, with the `"needs_fallback"` attribute described above
#' @keywords internal
expand_across_dots <- function(dots, schema, group_cols, verb) {
  out_quos <- list()
  out_names <- character()
  needs_fallback <- FALSE
  staged_temps <- character()
  mutating <- verb %in% c("mutate", "transmute")
  # Names in scope for the guards of a later dot: the table's columns plus
  # everything an earlier dot of this call creates (dplyr evaluates a value's
  # free variables and the extra `...` args against that data).
  scope <- schema$names

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
      expanded <- expand_one_across(expr, env, schema, group_cols, scope)
      needs_fallback <- needs_fallback || isTRUE(attr(expanded, "needs_fallback"))
      if (isTRUE(attr(expanded, "hazard"))) {
        if (mutating) {
          # Stage: compute every expansion from the pre-dot data into temps
          expanded <- stage_across_expansion(
            expanded, c(schema$names, out_names, names(dots), staged_temps)
          )
          staged_temps <- c(staged_temps, attr(expanded, "temps"))
        } else {
          needs_fallback <- TRUE
        }
      }
      out_quos <- c(out_quos, expanded)
      out_names <- c(out_names, names(expanded))
      scope <- union(scope, names(expanded))
      next
    }

    if (identical(head_name, "if_any") || identical(head_name, "if_all")) {
      combined <- expand_if_any_all(expr, env, schema, group_cols, head_name, verb, scope)
      needs_fallback <- needs_fallback || isTRUE(attr(combined, "needs_fallback"))
      out_quos <- c(out_quos, list(combined))
      # dplyr names an unnamed if_any()/if_all() dot after the original call
      out_names <- c(out_names, if (is_named) nm else if (identical(verb, "filter")) "" else rlang::as_label(expr))
      if (!identical(verb, "filter")) scope <- union(scope, out_names[length(out_names)])
      next
    }

    out_quos <- c(out_quos, list(quo))
    out_names <- c(out_names, if (is_named) nm else "")
    if (!identical(verb, "filter")) {
      # An unnamed dot other than a column already in scope can return a data
      # frame, whose columns dplyr adds under their own names: from here on
      # the set of columns is unknown (NA, see across_free_vars_hit_data()).
      unknown <- !is_named && !(is.symbol(expr) && as.character(expr) %in% scope)
      scope <- union(scope, if (is_named) nm else if (unknown) NA_character_ else rlang::as_label(expr))
    }
  }

  names(out_quos) <- out_names
  attr(out_quos, "needs_fallback") <- needs_fallback
  attr(out_quos, "staged_temps") <- staged_temps
  out_quos
}
