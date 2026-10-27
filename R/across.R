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
#     cur_data_all()/c_across() (beyond the native rewrites below) anywhere in its expression tree, or nests
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
#     mutate()/filter() (row-level verbs), numbered over group_data() for
#     group_by() (`.drop = FALSE` empty groups included) and in first-appearance
#     order for `.by`. In summarise() only a dot that is exactly
#     `cur_group_id()` is native (the output row number, see
#     summarise_group_id_split()); any other use falls back.
#   - cur_group()/cur_group_rows() are genuinely group-context objects (a 1-row
#     tibble, a row-index vector), so they fall back except for two shapes that
#     are plain expressions (rewrite_context_dots()): `cur_group()$key` /
#     `cur_group()[["key"]]` for a group column, and `length(cur_group_rows())`
#     (= n()).
#   - c_across() is rowwise()'s own companion. It only ever appears inside a
#     rowwise call, which R/rowwise.R rewrites (sum/mean/min/max/prod of
#     c_across() become horizontal GPU ops, F49) before it reaches this file;
#     a c_across() that survives the rewrite sends the whole call to dplyr.

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
# `pkg::`-qualified) anywhere in its tree. `.scan`: first rule `fn_name` out
# with all.names() (C), which lists every symbol of every call the walk can
# reach, heads included, so a dot without the name costs no R-level walk.
# @keywords internal
expr_contains_zero_arg_call <- function(expr, fn_name, .scan = TRUE) {
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
  if (.scan && !fn_name %in% all.names(expr)) {
    return(FALSE)
  }
  if (identical(call_head_name(expr), fn_name) && length(expr) == 1) {
    return(TRUE)
  }
  for (part in as.list(expr)) {
    if (expr_contains_zero_arg_call(part, fn_name, .scan = FALSE)) {
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

# The names the dot screens look for (colwise_rewrite_dots() aside, which looks
# for function objects): the cur_group*()/pick()/c_across() family that
# rewrite_context_dots(), substitute_cur_group_id() and dots_need_fallback()
# handle, and the across()/if_any()/if_all() that expand_across_dots() expands. A
# dot list that mentions none of them (dots_are_plain()) passes every one of
# those screens unchanged, so the verbs skip them.
dots_screen_fns <- c(deferred_fallback_fns, "cur_group_id", "across", "if_any", "if_all")

# TRUE iff no dot mentions a name in dots_screen_fns.
# @keywords internal
dots_are_plain <- function(dots) {
  !dots_mention(dots, dots_screen_fns)
}

# Does any dot of `dots` (quosures) mention one of the names `fns` anywhere in its
# expression, as a symbol: a call head, an argument, inside an embedded quosure?
# all.names() is a C walk, so this is the cheap pre-check the dot walkers below
# (and expand_across_dots(), substitute_cur_group_id(), rewrite_context_dots(),
# summarise_has_cur_group_id()) use before their R-level walks: a dot that does
# not mention a name the walk looks for cannot match it. Conservative: a symbol
# that is not a call head (a column called `across`) also counts.
# @keywords internal
dots_mention <- function(dots, fns) {
  for (i in seq_along(dots)) {
    if (any(fns %in% all.names(rlang::quo_get_expr(dots[[i]])))) {
      return(TRUE)
    }
  }
  FALSE
}

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
  # Every shape below is a call to one of these names (a named top-level
  # across(), expr_needs_fallback()'s own all.names() gate): a call that
  # mentions none of them needs nothing from the R-level walk.
  if (!dots_mention(dots, c(extra_names, "across", "if_any", "if_all"))) {
    return(FALSE)
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
  # Every name the walk below can match is a call head, which all.names() (C)
  # lists: a dot with none of them needs no R-level walk.
  if (top_level && !any(c(extra_names, "across", "if_any", "if_all") %in% all.names(expr))) {
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
# cur_group()$key and length(cur_group_rows()): syntactic rewrites
# =============================================================================

# Rewrite the two shapes of cur_group()/cur_group_rows() that are plain column
# or count expressions, before the whole-call fallback screen:
#   - `cur_group()$key` / `cur_group()[["key"]]` -> the group column `key`
#     (`key` a column of `group_cols` that no earlier dot of the call writes:
#     cur_group() keeps the old key)
#   - `length(cur_group_rows())` -> `dplyr::n()`
# Bare or `pkg::`-qualified. Everything else is left alone and so still reaches
# the fallback. Function bodies and formulas are not entered (their names may
# be shadowed).
#
# @param dots A list of quosures
# @param group_cols Character vector, the effective group columns
# @param sequential `TRUE` for mutate()/transmute(), whose dots can write a
#   group column before a later dot reads cur_group()
# @param name_unnamed Name a rewritten unnamed dot after the original
#   expression (filter() dots must stay unnamed)
# @return The dots, rewritten (the originals stay with the caller for the CPU
#   fallbacks)
# @keywords internal
rewrite_context_dots <- function(dots, group_cols, sequential = FALSE, name_unnamed = FALSE) {
  has_ctx <- dots_mention(dots, c("cur_group", "cur_group_rows")) &&
    any(vapply(dots, function(q) {
      expr_contains_zero_arg_call(rlang::quo_get_expr(q), "cur_group") ||
        expr_contains_zero_arg_call(rlang::quo_get_expr(q), "cur_group_rows")
    }, logical(1)))
  if (!has_ctx) {
    return(dots)
  }
  nms <- names(dots) %||% rep("", length(dots))
  names(dots) <- nms
  for (i in seq_along(dots)) {
    keys <- group_cols
    if (sequential && i > 1L) {
      keys <- set_diff(keys, nms[seq_len(i - 1L)])
    }
    old <- rlang::quo_get_expr(dots[[i]])
    expr <- rewrite_context_expr(old, keys)
    # An unnamed dot is named after the expression the user wrote
    if (name_unnamed && !identical(expr, old) && !nzchar(nms[i])) {
      names(dots)[i] <- rlang::as_label(old)
    }
    dots[[i]] <- rlang::new_quosure(expr, rlang::quo_get_env(dots[[i]]))
  }
  dots
}

# @keywords internal
rewrite_context_expr <- function(expr, keys) {
  if (rlang::is_missing(expr) || !is.call(expr)) {
    return(expr)
  }
  head <- call_head_name(expr)
  if (identical(head, "length") && length(expr) == 2L) {
    arg <- expr[[2]]
    if (is.call(arg) && length(arg) == 1L && identical(call_head_name(arg), "cur_group_rows")) {
      return(quote(dplyr::n()))
    }
  }
  if (is.symbol(expr[[1]]) && as.character(expr[[1]]) %in% c("$", "[[") && length(expr) == 3L) {
    obj <- expr[[2]]
    if (is.call(obj) && length(obj) == 1L && identical(call_head_name(obj), "cur_group")) {
      key <- expr[[3]]
      key <- if (is.symbol(key) && identical(as.character(expr[[1]]), "$")) {
        as.character(key)
      } else if (is.character(key) && length(key) == 1L) {
        key
      } else {
        NULL
      }
      if (!is.null(key) && key %in% keys) {
        return(as.name(key))
      }
    }
  }
  if (identical(head, "function") || identical(head, "~")) {
    return(expr)
  }
  parts <- as.list(expr)
  for (j in seq_along(parts)[-1]) {
    if (!rlang::is_missing(parts[[j]])) {
      parts[j] <- list(rewrite_context_expr(parts[[j]], keys))
    }
  }
  as.call(parts)
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
# @param by Logical, `TRUE` when `group_cols` come from `.by=` (first-appearance
#   numbering, no empty groups) rather than group_by()
# @param rowwise `TRUE` for a rowwise `.data` (F49): the id is the row number
# @return `list(data = <possibly-materialized-and-joined tbl_gpu>, dots =
#   <possibly-rewritten dots>)`
# @keywords internal
substitute_cur_group_id <- function(.data, dots, group_cols, by = FALSE, rowwise = FALSE) {
  has_any <- dots_mention(dots, "cur_group_id") && any(vapply(
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

  if (rowwise) {
    # A rowwise table: every row is its own group, numbered in row order (F49).
    new_data <- rowwise_with_row_ids(.data, helper_name)
  } else if (length(group_cols) == 0) {
    # Ungrouped (or `.by=` resolving to no columns): every row is group 1.
    step <- make_mutate_expr(helper_name, ir_lit_from_r(1L), .subset2(.data, "schema"))
    new_data <- push_op(.data, ast_mutate(input_node(.data), list(step)))
  } else {
    # group_by() groups: numbering runs over group_data() as the table has it,
    # so `.drop = FALSE` empty groups count and slice(.preserve = TRUE) fixed
    # keys are kept. `.by` groups (vec_group_loc() in dplyr) have no empty
    # groups and are numbered in first-appearance order.
    fixed <- if (!by) fixed_keys_of(.data)
    tmp_grouped <- new_tbl_gpu(
      ptr = .subset2(.data, "ptr"), schema = .subset2(.data, "schema"), groups = group_cols,
      exec_mode = .subset2(.data, "exec_mode"), fixed_keys = fixed,
      group_drop = if (by) TRUE else .subset2(.data, "group_drop")
    )
    gd <- compute_group_data(tmp_grouped)
    if (by && nrow(gd) > 1L) {
      # Rows within a group are ascending, so the first row is each group's head
      rows <- unclass(gd$.rows)
      first_row <- unlist(rows, use.names = FALSE)[cumsum(c(1L, lengths(rows)[-length(rows)]))]
      gd <- gd[order(first_row), , drop = FALSE]
    }
    keys <- gd[set_diff(names(gd), ".rows")]
    keys[[helper_name]] <- seq_len(nrow(keys))
    keys_gpu <- tbl_gpu(keys, lazy = FALSE)

    joined <- dplyr::left_join(.data, keys_gpu, by = group_cols)
    # A lazy join is only a pending op; the rebuild below reads `$ptr`
    if (has_pending_ops(joined)) {
      joined <- compute(joined)
    }
    # Keep every field of `.data` (group_drop, fixed_keys, frame meta, ...);
    # only the data and its schema change.
    new_data <- .data
    new_data <- set_tbl_gpu_field(new_data, "ptr", joined$ptr)
    new_data <- set_tbl_gpu_field(new_data, "schema", joined$schema)
    new_data <- set_tbl_gpu_field(new_data, "lazy_ops", NULL)
    new_data <- set_tbl_gpu_field(new_data, "exec_mode", joined$exec_mode)
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
#   names the output names (possibly empty); or `NULL` when the selection
#   is not provably type-only (selection_is_type_only()): a predicate that may
#   read values needs the real data, which the caller hands to dplyr (a
#   whole-call fallback)
# @keywords internal
resolve_across_cols <- function(cols_expr, env, schema, group_cols) {
  # A running schema that lost track of which columns exist (see
  # across_run_step()): dplyr resolves `.cols` on the real data.
  if (isTRUE(schema$unknown_set)) {
    return(NULL)
  }
  eligible <- set_diff(schema$names, group_cols)
  if (length(eligible) == 0) {
    return(stats::setNames(character(), character()))
  }
  if (!selection_is_type_only(cols_expr, env, eligible)) {
    return(NULL)
  }
  keep <- match(eligible, schema$names)
  types <- schema$types[keep]
  # A column of unknown type (NA, created by an earlier dot) can only be told
  # apart by name: any other selection waits for the real data.
  if (anyNA(types)) {
    if (!pick_selection_is_type_only(cols_expr, env, eligible) ||
          "where" %in% all.names(cols_expr)) {
      return(NULL)
    }
    types[is.na(types)] <- "FLOAT64"
  }
  select_on <- function(keep_rows) {
    proxy <- tidyselect_proxy(list(
      names = eligible[keep_rows], types = types[keep_rows],
      factor_levels = schema$factor_levels
    ))
    tidyselect::eval_select(expr = cols_expr, data = proxy, env = env, allow_rename = TRUE)
  }
  all_rows <- rep(TRUE, length(eligible))

  sel <- tryCatch(
    select_on(all_rows),
    error = function(e) {
      # The selection may only be broken by a column an earlier dot removed:
      # dplyr raises its own error for that, on the real data.
      if (!is.null(schema$base_names) && !all(schema$base_names %in% schema$names)) {
        return(NULL)
      }
      stop(e)
    }
  )
  if (is.null(sel)) {
    return(NULL)
  }
  # A column an earlier dot may not have created at all (see
  # across_run_set()): the selection must come out the same without it, and
  # must not pick it, or dplyr decides on the real data.
  absent <- eligible %in% schema$maybe
  if (any(absent)) {
    if (any(absent[sel])) {
      return(NULL)
    }
    without <- tryCatch(select_on(!absent), error = function(e) NULL)
    if (is.null(without) ||
          !identical(names(without), names(sel)) ||
          !identical(eligible[!absent][without], eligible[sel])) {
      return(NULL)
    }
  }
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
  across_registry_names()$known
}

# Call heads an inlined VALUE body may use: everything the IR lowers by name.
# A body with any other call (UseMethod, missing, sapply, a nested function,
# `{`, `pkg::fn`, ...) is not inlined, since dplyr calls the value in a real
# frame instead.
# @keywords internal
across_native_heads <- function() {
  across_registry_names()$heads
}

# Both name sets above derive only from the IR registry, which is built when
# the namespace loads, so they're computed once per session. The cache is
# keyed on the registry's names and ir_internal_ops (identical() is a pointer
# check while they're unchanged), so a test that mocks the registry still
# sees its own entries.
across_registry_cache <- new.env(parent = emptyenv())

# @keywords internal
across_registry_names <- function() {
  cache <- across_registry_cache
  reg_names <- names(ir_call_registry)
  if (!identical(cache$reg_names, reg_names) || !identical(cache$internal, ir_internal_ops)) {
    # F50: the factor conversions are parsed by ir_parse_factor_conversion()
    # (no registry entry), so `across(where(is.factor), as.character)` can name them.
    known <- set_union(set_diff(reg_names, ir_internal_ops), c("quantile", "any", "all", "count",
                                     "as.character", "as.integer", "as.numeric", "as.double"))
    cache$value <- list(
      known = known,
      candidates = c(known, "identity"),
      heads = c(known, "(", "ceiling", "paste0", "paste", "startsWith", "endsWith")
    )
    cache$reg_names <- reg_names
    cache$internal <- ir_internal_ops
  }
  cache$value
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
# name (it drops any `pkg::`), so `value` must be one of that name's native
# canonical functions (F77's table, ir_native_canonical_fns()): `stats::lag`
# is not the IR's `lag`, `dplyr::lag` is. Never an inheriting lookup from a
# namespace: its chain reaches the global env, where a user's own `median`
# would pass for stats'. A bare name must also resolve to `value` from `env`,
# so the emitted `name(col)` means the same thing if an expression fallback
# evaluates it. The name the value was written as (`mean`) is tried first. An
# alias or a value with no name (a list held in a variable, a scoped verb's
# function) is matched against every registry name, first by the binding of
# that name where the value was defined (its own namespace, base for a
# primitive; a non-inheriting lookup), which rejects nearly all of them in C.
# A function exported under a name its own namespace does not bind is missed
# there: it is then not known, which costs speed only.
# @keywords internal
across_known_fn_name <- function(value, env, fn_expr = NULL) {
  # a canonical function is base's or a namespace's: a closure of the user's
  # (its top env is the global env) is none
  home <- baseenv()
  if (!is.primitive(value)) {
    home <- topenv(environment(value))
    if (!identical(home, baseenv()) && !isNamespace(home)) {
      return(NULL)
    }
  }
  qualified <- is.call(fn_expr) && length(fn_expr) == 3 &&
    identical(fn_expr[[1]], as.name("::")) && is.symbol(fn_expr[[3]])
  candidates <- across_registry_names()$candidates
  if (qualified) {
    nm <- as.character(fn_expr[[3]])
    return(if (nm %in% candidates && across_is_native_fn(nm, value)) nm)
  }
  if (is.symbol(fn_expr)) {
    nm <- as.character(fn_expr)
    if (nm %in% candidates && across_is_native_fn(nm, value) &&
        identical(get0(nm, envir = env, mode = "function"), value)) {
      return(nm)
    }
  }
  for (nm in candidates) {
    if (identical(get0(nm, envir = home, mode = "function", inherits = FALSE), value) &&
        across_is_native_fn(nm, value) &&
        identical(get0(nm, envir = env, mode = "function"), value)) {
      return(nm)
    }
  }
  NULL
}

# Is `value` one of the native canonical functions of `nm`?
# @keywords internal
across_is_native_fn <- function(nm, value) {
  for (canon in ir_native_canonical_fns(nm)) {
    if (identical(canon, value)) {
      return(TRUE)
    }
  }
  FALSE
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
    if (across_free_vars_hit_data(set_diff(all.vars(rhs), c(".x", ".")), scope)) {
      return(NULL)
    }
    # (a head in the value's own environment can be a function of the user's
    # that shadows a native op: dplyr calls the value, so that one wins)
    if (ir_shadow_hit(rhs, environment(value))) {
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
  if (across_free_vars_hit_data(set_diff(all.vars(body_expr), formal), scope)) {
    return(NULL)
  }
  if (ir_shadow_hit(body_expr, environment(value))) {
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
  if (is.null(cols)) {
    # value-dependent selection: dplyr resolves it on the real data
    return(across_fallback_result())
  }
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

# if_any()/if_all() require every per-column predicate to be logical. dplyr
# checks the values; here the piece's inferred type stands in for them. A piece
# the IR can't parse is left to the normal parse path; one whose type is
# unknown (it reads a column the running schema couldn't type) is dplyr's to
# check on the real values.
# @param piece The instantiated predicate for `col`
# @return `TRUE` when dplyr must run the call (a piece of known non-logical
#   type is dplyr's error to raise), otherwise invisibly `FALSE`
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
    return(invisible(FALSE))
  }
  # Known non-logical (or an NA type): dplyr raises the error itself, with
  # its own class, call and group context, from the CPU fallback.
  TRUE
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
  cols <- resolve_across_cols(cols_expr, env, schema, group_cols)
  if (is.null(cols)) {
    return(structure(rlang::new_quosure(FALSE, env), needs_fallback = TRUE))
  }
  cols <- unname(cols)
  if (length(cols) == 0) {
    # dplyr: if_any() over nothing is FALSE, if_all() is TRUE
    value <- identical(mode, "if_all")
    # summarise() only knows aggregations, so a constant there is an
    # aggregation over the first column's NA pattern; everywhere else it is
    # the literal.
    if (identical(verb, "summarise") && length(schema$names) > 0) {
      col <- as.name(c(set_diff(schema$names, group_cols), schema$names)[1])
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
      if (check_across_predicate_piece(piece, col, piece_env, schema)) {
        return(fallback())
      }
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
#' Expansion stops at the first dot that needs dplyr (the result is then partial
#' and only its `"needs_fallback"` attribute is meaningful).
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
#'   quosures, with the `"needs_fallback"` attribute described above, and a
#'   `"provenance"` attribute: `list(origin, kind)`, one entry per returned
#'   quosure. `origin` is the index of the input dot it came from; `kind` is
#'   `"dot"` (the input dot, passed through unchanged), `"across"` (an
#'   expansion of an unnamed `across()` dot, staged or not) or `"combined"`
#'   (an `if_any()`/`if_all()` dot). mutate()/transmute() use it to tell
#'   which original dot a value computed from the expanded dots belongs to.
#' @keywords internal
expand_across_dots <- function(dots, schema, group_cols, verb) {
  # No dot mentions across()/if_any()/if_all(): every dot passes through as it
  # is, and nothing below it (the running schema, the staging) is ever used.
  if (length(dots) > 0L && !dots_mention(dots, c("across", "if_any", "if_all"))) {
    return(expand_plain_dots(dots))
  }
  out_quos <- list()
  out_names <- character()
  prov_origin <- integer()
  prov_kind <- character()
  needs_fallback <- FALSE
  staged_temps <- character()
  mutating <- verb %in% c("mutate", "transmute")
  # Names in scope for the guards of a later dot: the table's columns plus
  # everything an earlier dot of this call creates (dplyr evaluates a value's
  # free variables and the extra `...` args against that data).
  scope <- schema$names

  # dplyr resolves an across()/if_any()/if_all() against the columns as they
  # stand at its dot. `run` is that schema, tracked only up to the last such
  # dot, so a call without one pays nothing.
  selecting <- vapply(seq_along(dots), function(i) {
    nm <- names(dots)[i]
    head <- call_head_name(rlang::quo_get_expr(dots[[i]]))
    is_named <- !is.null(nm) && !is.na(nm) && nzchar(nm)
    (!is_named && identical(head, "across")) || isTRUE(head %in% c("if_any", "if_all"))
  }, logical(1))
  last_selecting <- if (identical(verb, "filter") || !any(selecting)) 0L else max(which(selecting))
  run <- schema
  run$base_names <- schema$names

  for (i in seq_along(dots)) {
    # dplyr reruns the original dots, so what the later ones would expand to
    # is moot: against the schema as it was before this dot they could raise
    # an error dplyr doesn't.
    if (needs_fallback) {
      break
    }
    quo <- dots[[i]]
    nm <- names(dots)[i]
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)
    head_name <- call_head_name(expr)

    is_named <- !is.null(nm) && !is.na(nm) && nzchar(nm)
    track <- i < last_selecting

    if (!is_named && identical(head_name, "across")) {
      if (identical(verb, "filter")) {
        stop(
          "`across()` can't be used in filter() -- use `if_any()`/`if_all()` ",
          "instead.\nExpression: ", rlang::expr_deparse(expr), call. = FALSE
        )
      }
      expanded <- expand_one_across(expr, env, run, group_cols, scope)
      needs_fallback <- needs_fallback || isTRUE(attr(expanded, "needs_fallback"))
      # The outputs' types come from the expansion as written, against the
      # columns before this dot
      if (track && !needs_fallback) {
        after <- run
        for (k in seq_along(expanded)) {
          after <- across_run_set_inferred(after, names(expanded)[k], expanded[[k]], run, verb)
        }
        run_next <- after
      }
      if (isTRUE(attr(expanded, "hazard"))) {
        if (mutating) {
          # Stage: compute every expansion from the pre-dot data into temps
          expanded <- stage_across_expansion(
            expanded, c(schema$names, run$names, out_names, names(dots), staged_temps)
          )
          staged_temps <- c(staged_temps, attr(expanded, "temps"))
        } else {
          needs_fallback <- TRUE
        }
      }
      out_quos <- c(out_quos, expanded)
      out_names <- c(out_names, names(expanded))
      prov_origin <- c(prov_origin, rep(i, length(expanded)))
      prov_kind <- c(prov_kind, rep("across", length(expanded)))
      scope <- set_union(scope, names(expanded))
      if (track && !needs_fallback) run <- run_next
      next
    }

    if (identical(head_name, "if_any") || identical(head_name, "if_all")) {
      combined <- expand_if_any_all(expr, env, run, group_cols, head_name, verb, scope)
      needs_fallback <- needs_fallback || isTRUE(attr(combined, "needs_fallback"))
      out_quos <- c(out_quos, list(combined))
      prov_origin <- c(prov_origin, i)
      prov_kind <- c(prov_kind, "combined")
      # dplyr names an unnamed if_any()/if_all() dot after the original call
      out_names <- c(out_names, if (is_named) nm else if (identical(verb, "filter")) "" else rlang::as_label(expr))
      if (!identical(verb, "filter")) scope <- set_union(scope, out_names[length(out_names)])
      if (track && !needs_fallback) {
        run <- across_run_set(run, out_names[length(out_names)],
                              if (mutating) "BOOL8" else NA_character_)
      }
      next
    }

    out_quos <- c(out_quos, list(quo))
    out_names <- c(out_names, if (is_named) nm else "")
    prov_origin <- c(prov_origin, i)
    prov_kind <- c(prov_kind, "dot")
    if (!identical(verb, "filter")) {
      # An unnamed dot other than a column already in scope can return a data
      # frame, whose columns dplyr adds under their own names: from here on
      # the set of columns is unknown (NA, see across_free_vars_hit_data()).
      unknown <- !is_named && !(is.symbol(expr) && as.character(expr) %in% scope)
      # c(), not set_union(): `scope` is only matched against (%in%, anyNA()),
      # and a union per dot is quadratic in a wide call's dot count
      scope <- c(scope, if (is_named) nm else if (unknown) NA_character_ else rlang::as_label(expr))
      if (track) run <- across_run_step(run, nm, quo, verb)
    }
  }

  names(out_quos) <- out_names
  attr(out_quos, "needs_fallback") <- needs_fallback
  attr(out_quos, "staged_temps") <- staged_temps
  attr(out_quos, "provenance") <- list(origin = prov_origin, kind = prov_kind)
  out_quos
}

# expand_across_dots() of dots none of which mentions across()/if_any()/if_all():
# the loop's "dot" branch for each of them, without its bookkeeping of the
# running schema and the names in scope (only a later across() would read them).
# @keywords internal
expand_plain_dots <- function(dots) {
  n <- length(dots)
  in_names <- names(dots)
  out_quos <- vector("list", n)
  out_names <- character(n)
  for (i in seq_len(n)) {
    out_quos[[i]] <- dots[[i]]
    nm <- in_names[i]
    out_names[[i]] <- if (!is.null(nm) && !is.na(nm) && nzchar(nm)) nm else ""
  }
  names(out_quos) <- out_names
  attr(out_quos, "needs_fallback") <- FALSE
  attr(out_quos, "staged_temps") <- character()
  attr(out_quos, "provenance") <- list(origin = seq_len(n), kind = rep("dot", n))
  out_quos
}

# =============================================================================
# The running schema of expand_across_dots()
#
# `run` is a schema (names, types, factor_levels, tz) plus `base_names` (the
# columns before the call) and `unknown_set` (set once an earlier dot may have
# added columns we can't name, or put one where we can't say). A type of NA
# means "exists, type not known": a created column whose type the native path
# can't be asked about without running something (a column-free call), whose
# factor levels/zone aren't known exactly, or whose R type may not be the type
# the native path gives it (`median(<int>)`, lubridate's `year()`: see
# across_r_typed()). `maybe` lists the columns that may not exist at all: a
# dot whose value can't be known without running it may evaluate to NULL,
# which creates nothing (`z = v` with `v <- NULL`). UNKNOWN is always safe:
# resolve_across_cols() sends a selection that could depend on it to dplyr.
# =============================================================================

# Add or overwrite one column of a running schema (an overwrite keeps its
# position). `maybe` marks a column that may not exist (see above): it also
# stands for an overwritten column that the dot may have removed.
# @keywords internal
across_run_set <- function(run, name, type, levels = NULL, zone = "UTC", maybe = FALSE) {
  idx <- match(name, run$names)
  # Setting a column that may not exist: if it doesn't (an earlier dot was
  # NULL), dplyr's mask appends it at the end; if it does, it stays put. Only
  # the last column is in the same place either way.
  if (!is.na(idx) && name %in% run$maybe && idx < length(run$names)) {
    run$unknown_set <- TRUE
  }
  run$maybe <- if (maybe) set_union(run$maybe, name) else set_diff(run$maybe, name)
  run <- set_schema_factor_levels(run, name, levels)
  if (is.na(idx)) {
    run$names <- c(run$names, name)
    run$types <- c(unname(run$types), type)
  } else {
    run$types <- unname(run$types)
    run$types[idx] <- type
  }
  set_schema_tz(run, name, zone)
}

# Remove one column from a running schema (`name = NULL`).
# @keywords internal
across_run_drop <- function(run, name) {
  run$maybe <- set_diff(run$maybe, name)
  keep <- run$names != name
  run$names <- run$names[keep]
  run$types <- unname(run$types)[keep]
  run <- set_schema_factor_levels(run, name, NULL)
  set_schema_tz(run, name, "UTC")
}

# Calls that are safe to hand to ir_parse_quo() twice: it evaluates a
# column-free call it doesn't recognize, which must not happen just to learn a
# type.
# @keywords internal
across_pure_heads <- function() {
  c(set_diff(names(ir_call_registry), ir_internal_ops), ir_foldable_window_fns, "c", "~", "(", ":",
    "case_when", "case_match", "recode", "recode_values", "replace_values", "replace_when",
    "when_all", "when_any", "order_by", "with_order")
}

# The type, factor levels and zone the native path gives the column that `quo`
# computes on `run`, or `list(type = NA, maybe = TRUE)` when that can't be said
# without running anything (the column may not even exist, see above), or
# `list(type = NA, maybe = FALSE)` when R's type for it may be another one.
# @keywords internal
across_infer_output <- function(quo, run) {
  unknown <- list(type = NA_character_, maybe = TRUE)
  expr <- rlang::quo_get_expr(quo)
  if (isTRUE(run$unknown_set) || anyDuplicated(run$names) > 0L ||
        expr_has_mask_only_call(expr) || expr_reaches_mask(expr)) {
    return(unknown)
  }
  literal <- is.atomic(expr) && length(expr) == 1L
  if (!literal && !expr_has_column_ref(expr, run)) {
    return(unknown)
  }
  # an operand of unknown type makes the result's type unknown
  no_type <- run$names[is.na(run$types)]
  if (length(no_type) > 0L && (any(all.vars(expr) %in% no_type) || ".data" %in% all.vars(expr))) {
    return(unknown)
  }
  pure_heads <- across_pure_heads()
  pure <- function(e) {
    if (rlang::is_missing(e) || !is.call(e)) {
      return(TRUE)
    }
    head <- call_head_name(e)
    if (is.null(head) || (!head %in% pure_heads && !expr_has_column_ref(e, run))) {
      return(FALSE)
    }
    # the head of `pkg::fn` is the call `::`(pkg, fn), not an argument
    all(vapply(as.list(e)[-1], pure, logical(1)))
  }
  if (!pure(expr)) {
    return(unknown)
  }
  tryCatch(suppressWarnings({
    ir <- ir_parse_quo(quo, run)
    if (is.null(ir) || identical(ir_factor_rule(ir, run), "cpu")) {
      return(unknown)
    }
    type <- unname(make_mutate_expr(".", ir, run)$output_type)
    if (length(type) != 1L || is.na(type)) {
      return(unknown)
    }
    # The column exists (the IR computes it), but a type predicate must see
    # the type R gives it, which may not be this one.
    if (!across_r_typed(expr, rlang::quo_get_env(quo), run) ||
          !type %in% c("BOOL8", "INT32", "FLOAT64", "STRING", "DICTIONARY32",
                       "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")) {
      return(list(type = NA_character_, maybe = FALSE))
    }
    levels <- NULL
    if (identical(type, "DICTIONARY32")) {
      # only a bare copy of a factor with known levels stays a native factor
      levels <- if (identical(ir$kind, "col")) run$factor_levels[[ir$name]] else NULL
      if (is.null(levels)) {
        return(unknown)
      }
    }
    zone <- if (identical(type, "TIMESTAMP_MICROSECONDS")) ir_infer_tz(ir, run) else "UTC"
    list(type = type, levels = levels, zone = zone)
  }), error = function(e) unknown)
}

# Functions whose native type is the type R gives their result, for every
# input the native path takes, with the package a `pkg::fn` call may name.
# Each was checked against dplyr 1.2.1 and lubridate on plain tibbles, for
# logical, integer, double, character, Date and POSIXct inputs, n odd and
# even. Left out because R's type differs: ntile() (integer in R), year(),
# month(), second(), yday() and wday() (doubles in lubridate), and cuplyr's
# roll_*()/data.table's froll*(). across_r_typed() adds the cases that
# depend on the input type.
# @keywords internal
across_r_typed_fns <- c(
  "(" = "base", "~" = "base", "c" = "base", ":" = "base",
  "+" = "base", "-" = "base", "*" = "base", "/" = "base", "^" = "base",
  "%%" = "base", "%/%" = "base",
  "==" = "base", "!=" = "base", "<" = "base", "<=" = "base", ">" = "base",
  ">=" = "base", "&" = "base", "|" = "base", "!" = "base", xor = "base",
  is.na = "base", "%in%" = "base",
  abs = "base", sqrt = "base", exp = "base", log = "base", log2 = "base",
  log10 = "base", floor = "base", ceiling = "base", sin = "base", cos = "base",
  tan = "base", round = "base",
  cumsum = "base", cumprod = "base", cummax = "base", cummin = "base",
  mean = "base", sum = "base", max = "base", min = "base",
  median = "stats", sd = "stats", var = "stats",
  toupper = "base", tolower = "base", nchar = "base", substr = "base",
  trimws = "base", paste = "base", paste0 = "base", grepl = "base",
  sub = "base", gsub = "base", startsWith = "base", endsWith = "base",
  as.Date = "base", as.POSIXct = "base",
  between = "dplyr", near = "dplyr", cumall = "dplyr", cumany = "dplyr",
  cummean = "dplyr", when_any = "dplyr", when_all = "dplyr",
  n_distinct = "dplyr", row_number = "dplyr", min_rank = "dplyr",
  dense_rank = "dplyr", percent_rank = "dplyr", cume_dist = "dplyr",
  consecutive_id = "dplyr", lag = "dplyr", lead = "dplyr", first = "dplyr",
  last = "dplyr", nth = "dplyr", if_else = "dplyr", case_when = "dplyr",
  coalesce = "dplyr", na_if = "dplyr", case_match = "dplyr",
  recode = "dplyr", recode_values = "dplyr", replace_values = "dplyr", replace_when = "dplyr",
  as_date = "lubridate", day = "lubridate", hour = "lubridate",
  minute = "lubridate", quarter = "lubridate"
)

# Of those, the ones whose result type in R doesn't depend on their
# arguments' (logical, character, nchar()'s integer): what's inside them
# can't change the type.
across_r_fixed_type_fns <- c(
  "==", "!=", "<", "<=", ">", ">=", "&", "|", "!", "xor", "is.na", "%in%",
  "between", "near", "grepl", "startsWith", "endsWith", "cumall", "cumany",
  "when_any", "when_all", "toupper", "tolower", "substr", "trimws", "paste",
  "paste0", "sub", "gsub", "nchar"
)

# TRUE when R's type for `expr` (one that ir_parse_quo() parses on `run`) is
# surely the type the native path gives it: every call is in
# across_r_typed_fns (a `pkg::` prefix must name the function's package),
# outside these cases where R's type depends on the input:
# - median() of anything but a double or a timestamp: an integer or a
#   logical gives an integer/logical for an odd count and a double for an
#   even one, per group;
# - max()/min()/cummax()/cummin() of a logical and unary `-` of a logical
#   are integers in R;
# - a value from the environment of a class other than Date/POSIXct.
# @keywords internal
across_r_typed <- function(expr, env, run) {
  arg_type <- function(a) {
    tryCatch(
      ir_infer_type(ir_parse_expr(a, env, run, allow_vector = FALSE), run),
      error = function(e) NA_character_
    )
  }
  plain <- function(v) {
    is.atomic(v) && !is.factor(v) && (is.null(oldClass(v)) || inherits(v, c("Date", "POSIXct")))
  }
  if (rlang::is_quosure(expr)) {
    return(across_r_typed(rlang::quo_get_expr(expr), rlang::quo_get_env(expr), run))
  }
  if (rlang::is_missing(expr) || is.null(expr) || is_data_pronoun_call(expr)) {
    return(TRUE)
  }
  # a constant, possibly a classed one injected with `!!`
  if (is.atomic(expr)) {
    return(plain(expr))
  }
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    return(nm %in% run$names || plain(get0(nm, envir = env, inherits = TRUE)))
  }
  if (is_env_pronoun_call(expr)) {
    return(plain(get0(pronoun_call_name(expr), envir = env, inherits = TRUE)))
  }
  if (!is.call(expr)) {
    return(FALSE)
  }
  head <- expr[[1]]
  fn <- call_head_name(expr)
  if (is.null(fn) || !fn %in% names(across_r_typed_fns)) {
    return(FALSE)
  }
  if (is.call(head) && !identical(as.character(head[[2]]), across_r_typed_fns[[fn]])) {
    return(FALSE)
  }
  if (fn %in% across_r_fixed_type_fns) {
    return(TRUE)
  }
  args <- as.list(expr)[-1]
  if (fn %in% c("median", "max", "min", "cummax", "cummin") ||
        (identical(fn, "-") && length(args) == 1L)) {
    arg_names <- names(args) %||% rep("", length(args))
    x <- if ("x" %in% arg_names) args["x"] else utils::head(args[!nzchar(arg_names)], 1L)
    if (length(x) == 0L) {
      return(FALSE)
    }
    type <- arg_type(x[[1]])
    ok <- if (identical(fn, "median")) {
      isTRUE(type %in% c("FLOAT64", "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS"))
    } else {
      !is.na(type) && type != "BOOL8"
    }
    if (!ok) {
      return(FALSE)
    }
  }
  all(vapply(args, across_r_typed, logical(1), env = env, run = run))
}

# Record the column `quo` computes under `name`; `on` is the schema it is
# computed against (an across() dot's expansions all see the pre-dot one).
# @keywords internal
across_run_set_inferred <- function(run, name, quo, on, verb) {
  info <- if (verb %in% c("mutate", "transmute")) {
    across_infer_output(quo, on)
  } else {
    list(type = NA_character_, maybe = TRUE)
  }
  across_run_set(run, name, info$type, info$levels, info$zone %||% "UTC", isTRUE(info$maybe))
}

# Advance `run` over one ordinary (non-across) dot. `nm` is the dot's name
# (empty when unnamed).
# @keywords internal
across_run_step <- function(run, nm, quo, verb) {
  if (isTRUE(run$unknown_set)) {
    return(run)
  }
  expr <- rlang::quo_get_expr(quo)
  is_named <- !is.null(nm) && !is.na(nm) && nzchar(nm)
  if (!is_named && is.symbol(expr) && as.character(expr) %in% run$names) {
    # A column an earlier dot may have removed: if it did, the symbol is
    # whatever the environment holds, which dplyr appends at the end (or
    # unpacks, for a data frame, or skips, for NULL).
    if (as.character(expr) %in% run$maybe) {
      run$unknown_set <- TRUE
    }
    return(run)
  }
  mutating <- verb %in% c("mutate", "transmute")
  if (is.null(expr)) {
    if (!is_named || !mutating) {
      run$unknown_set <- TRUE
      return(run)
    }
    return(across_run_drop(run, nm))
  }
  if (is_named) {
    return(across_run_set_inferred(run, nm, quo, run, verb))
  }
  # Unnamed: a column named after the expression, unless it returns a data
  # frame. Only an expression the IR parses is surely a column.
  info <- if (mutating) across_infer_output(quo, run) else list(type = NA_character_)
  if (is.na(info$type)) {
    run$unknown_set <- TRUE
    return(run)
  }
  across_run_set(run, rlang::as_label(expr), info$type, info$levels, info$zone %||% "UTC")
}
