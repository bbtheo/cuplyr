# Execution Helpers - Unified eager/lazy op dispatch
#
# This module implements the "one node, two schedules" execution model
# (see scratchpad/unification_design.md, decisions D1/D3/D4). Every verb
# builds a single AST node over `input_node(.data)` and hands it to
# `push_op()`, which is the *only* place that branches on `exec_mode`:
# lazy mode stores the node in `$lazy_ops` with no GPU work; eager mode
# lowers and executes the node immediately via `lower_and_execute()`.
#
# `R/lower.R` (lowering) and `R/ast.R` (schema inference) are shared
# verbatim by both schedules.

#' Get the schema a new op should be built against
#'
#' Pending-lazy-AST-aware: if `.data` is lazy and has pending operations,
#' infers the schema from the AST (`$lazy_ops`); otherwise returns the
#' table's base `$schema`. This is the canonical replacement for the
#' `current_schema`/`current schema` blocks duplicated inline across verbs.
#'
#' @param .data A `tbl_gpu` object
#' @return List with `names` and `types` (and possibly `factor_levels`)
#' @keywords internal
current_schema <- function(.data) {
  if (identical(.data$exec_mode, "lazy") && has_pending_ops(.data)) {
    infer_schema(.data$lazy_ops)
  } else {
    .data$schema
  }
}

#' Build the AST node that a new op should be layered on top of
#'
#' - lazy + pending ops: returns `.data$lazy_ops` as-is.
#' - lazy + no pending ops: returns a fresh `ast_source(.data$schema)`,
#'   matching the historical lazy-verb convention (e.g. the now-deleted
#'   `filter_lazy()`, `select_lazy()`) of leaving the source pointer unset; `compute()`/
#'   `collect()` supply `.data$ptr` as the lowering fallback `source_ptr`.
#' - eager: returns a fresh `ast_source(.data$schema, source_ptr = .data$ptr)`
#'   so `push_op()` can call `lower_and_execute()` immediately without a
#'   separate source table.
#'
#' @param .data A `tbl_gpu` object
#' @return An `ast_node` to use as the `input` of a new node
#' @keywords internal
input_node <- function(.data) {
  if (identical(.data$exec_mode, "lazy")) {
    if (has_pending_ops(.data)) {
      return(.data$lazy_ops)
    }
    return(ast_source(.data$schema))
  }

  ast_source(.data$schema, source_ptr = .data$ptr)
}

#' Restrict factor level metadata to columns that survive an op
#'
#' Implements the factor_levels half of D3: `schema$factor_levels` is
#' always recomputed from the *input* schema's factor_levels, filtered to
#' the names present in the node's output schema. This fixes the bug where
#' eager `select()` of a factor column dropped `factor_levels`, causing
#' `collect()` to return integer codes instead of a factor.
#'
#' @param node The AST node being applied (used to translate factor_levels
#'   keys through a `rename` node's name map, see `ast_rename()`,
#'   `R/ast.R` -- without this, a renamed factor column's levels would be
#'   silently dropped here since its key in `input_schema$factor_levels`
#'   is the OLD name, which no longer appears in `surviving_names`)
#' @param input_schema The schema of the op's input (pre-op)
#' @param surviving_names Character vector of column names in the new schema
#' @return A named list of factor levels, or `NULL` if none survive
#' @keywords internal
propagate_factor_levels <- function(node, input_schema, surviving_names) {
  factor_levels <- input_schema$factor_levels
  if (is.null(factor_levels) || length(factor_levels) == 0) {
    return(NULL)
  }

  if (identical(node$type, "rename")) {
    nm <- names(factor_levels)
    idx <- match(nm, node$old_names)
    renamed <- !is.na(idx)
    nm[renamed] <- node$new_names[idx[renamed]]
    names(factor_levels) <- nm
  }

  factor_levels <- factor_levels[names(factor_levels) %in% surviving_names]

  if (length(factor_levels) == 0) {
    return(NULL)
  }

  factor_levels
}

#' Propagate group columns across an AST node
#'
#' Per D3: `summarise` and `join` always yield no groups (`character()`);
#' every other node type keeps whichever of the existing `groups` still
#' appear in the node's (post-op) schema. This single rule reproduces
#' today's per-verb behavior (select drops de-selected group columns;
#' filter/mutate/arrange preserve all of them).
#'
#' `rename` needs one extra step before that final `intersect()`: a
#' group column that got renamed (e.g. `rename(gdf, grp = g)` where `g` is
#' grouped) must be translated to its NEW name first, or the plain
#' `intersect()` would miss it entirely (`schema$names` has the new name,
#' `groups` still has the old one) and silently drop the grouping.
#'
#' @param node The AST node being applied
#' @param groups Character vector of the input's group columns
#' @param schema The node's output schema (post-op), as from `infer_schema()`
#' @return Character vector of group columns to carry forward
#' @keywords internal
propagate_groups <- function(node, groups, schema) {
  if (node$type %in% c("summarise", "join")) {
    return(character())
  }

  if (identical(node$type, "rename")) {
    idx <- match(groups, node$old_names)
    renamed <- !is.na(idx)
    groups[renamed] <- node$new_names[idx[renamed]]
  }

  intersect(groups, schema$names)
}

#' Apply an AST node to a tbl_gpu, dispatching on exec_mode
#'
#' The only exec_mode branch point (D1). Lazy mode stores `node` as the new
#' `$lazy_ops` and does no GPU work; eager mode lowers and executes `node`
#' immediately. Both modes uniformly recompute `schema` (via `infer_schema()`,
#' with factor_levels reattached per D3) and `groups` (via
#' `propagate_groups()`).
#'
#' @param .data A `tbl_gpu` object
#' @param node The new AST node (already wrapping `input_node(.data)`)
#' @return A new `tbl_gpu` object
#' @keywords internal
push_op <- function(.data, node) {
  input_schema <- current_schema(.data)

  new_schema <- infer_schema(node)
  new_schema$factor_levels <- propagate_factor_levels(node, input_schema, new_schema$names)

  new_groups <- propagate_groups(node, .data$groups, new_schema)

  if (identical(.data$exec_mode, "lazy")) {
    return(new_tbl_gpu(
      ptr = .data$ptr,
      schema = new_schema,
      lazy_ops = node,
      groups = new_groups,
      exec_mode = .data$exec_mode
    ))
  }

  new_ptr <- wrap_gpu_call(node$type, lower_and_execute(node, .data$ptr))

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = new_groups,
    exec_mode = .data$exec_mode
  )
}

#' Build one side of a join's AST, always anchoring the source pointer
#'
#' Like `input_node()`, but *unconditionally* attaches `tbl$ptr` to the leaf
#' `ast_source` node, even when `tbl` is lazy with no pending ops (the case
#' where `input_node()` deliberately leaves the pointer unset and relies on
#' `lower_and_execute()`'s single `source_ptr` argument as a fallback, filled
#' in later by `compute()`/`collect()`). A join has *two* independent input
#' trees, so that single-fallback mechanism can't serve both leaves --
#' instead, each side's own concrete GPU pointer must be anchored explicitly
#' at construction time. This is safe because `tbl$ptr` is always a valid
#' pointer to that side's own (possibly stale, pre-pending-ops) base table:
#' `push_op()` carries it forward unchanged across lazy operations, only
#' `compute()`/`collect()` ever replace it.
#'
#' @param tbl A `tbl_gpu` object (one side of a join)
#' @return An `ast_node` with its leaf `source_ptr` set to `tbl$ptr`
#' @keywords internal
join_input_node <- function(tbl) {
  node <- if (has_pending_ops(tbl)) tbl$lazy_ops else ast_source(tbl$schema)
  set_ast_source_ptr(node, tbl$ptr)
}

#' Apply a two-input join AST node, dispatching on exec_mode
#'
#' The join analogue of `push_op()` (D1). Joins can't reuse `push_op()`
#' as-is because it is built around a single `.data` input: it reads exactly
#' one `exec_mode`/`ptr`/`schema` to decide the schedule. A join has two
#' independent inputs (`x`/`y`) whose *either* side being lazy forces the
#' whole join lazy (mirroring the pre-unification behavior), and whose eager
#' execution needs both sides' pointers anchored -- see `join_input_node()`.
#' Per D3, joins always drop groups (`character()`) and never carry
#' `factor_levels` (this matches `build_join_schema()`, which never
#' propagated them).
#'
#' @param join_type One of `"left"`, `"inner"`, `"full"`, `"right"`
#' @param x,y The two `tbl_gpu` join inputs
#' @param join_spec List with `left`/`right` key column names, as returned by
#'   `parse_join_by()`
#' @param suffix,keep,na_matches As documented on the join verbs
#' @return A new `tbl_gpu`
#' @keywords internal
push_join <- function(join_type, x, y, join_spec, suffix, keep, na_matches) {
  join_ast <- ast_join(
    join_type,
    join_input_node(x),
    join_input_node(y),
    join_spec,
    keep = keep,
    suffix = suffix,
    na_matches = na_matches
  )

  new_schema <- infer_schema(join_ast)

  if (identical(x$exec_mode, "lazy") || identical(y$exec_mode, "lazy")) {
    return(new_tbl_gpu(
      ptr = NULL,
      schema = new_schema,
      lazy_ops = join_ast,
      groups = character(),
      exec_mode = "lazy"
    ))
  }

  warn_if_join_too_large(join_type, x, y, join_spec, suffix, keep)

  new_ptr <- wrap_gpu_call(
    paste0(join_type, "_join"),
    lower_and_execute(join_ast, x$ptr)
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = character(),
    exec_mode = "eager"
  )
}

#' Auto-name unnamed dots, warning as dplyr's verbs do
#'
#' `mutate()` and `summarise()` both accept unnamed expressions (e.g.
#' `mutate(mpg * 2)`) and, matching dplyr, fall back to the expression's
#' deparsed text as the output column name -- while warning that an explicit
#' name would be clearer. Historically this logic was duplicated verbatim
#' inside both `mutate.tbl_gpu()` (twice -- once for its since-deleted eager
#' path, once for its since-deleted lazy path) and `summarise_lazy()` (now
#' also deleted); this is the single shared implementation, parameterized by
#' `verb` so each caller's warning names itself correctly.
#'
#' @param dots A list of quosures, as returned by `rlang::enquos(...)`
#' @param verb Character, the verb name to use in the warning text (e.g.
#'   `"mutate"`, `"summarise"`)
#' @return `dots`, with `names(dots)` filled in for any unnamed entries
#' @keywords internal
auto_name_dots <- function(dots, verb) {
  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }

  for (i in seq_along(dots)) {
    new_name <- dot_names[i]

    if (is.na(new_name) || new_name == "") {
      new_name <- rlang::quo_text(dots[[i]])
      warning("Unnamed ", verb, " expression '", new_name, "' will use expression as column name.\n",
              "Consider using explicit names: ", verb, "(name = ", new_name, ")",
              call. = FALSE)
      dot_names[i] <- new_name
    }
  }

  names(dots) <- dot_names
  dots
}

#' Resolve a verb's `.by=`/`by=` quosure into group column names
#'
#' Implements `scratchpad/phase5_window_design.md` section 6: the shared
#' resolution logic for on-the-fly grouping across `mutate()`/`filter()`
#' (and, from task W7 onward, the `slice()` family). `.by=`/`by=` is
#' mutually exclusive with an already-grouped `.data` -- verified
#' empirically against dplyr 1.2.1, both `mutate()` and `filter()` raise
#' the identical message below verbatim (including the backticks) when
#' `.data` is grouped and `.by=` is also supplied.
#'
#' @param .data A `tbl_gpu` object
#' @param by_quo A quosure for the verb's `.by=`/`by=` argument, as
#'   returned by `rlang::enquo(.by)` -- NOT yet evaluated
#' @param arg_name Character scalar, the argument's name as written in the
#'   verb's own signature (`".by"` for `mutate()`/`filter()`/`slice()`,
#'   `"by"` for the `slice_*()` family) -- interpolated into the
#'   grouped-`.data` error message so it names the argument the caller
#'   actually used
#' @return A character vector of column names: `character()` when `by_quo`
#'   is NULL (absent or explicit `.by = NULL`, matching dplyr: both behave
#'   identically, see the roxygen on the calling verb), otherwise the
#'   tidyselect-resolved column names in selection order (via
#'   `tidyselect::eval_select()`'s own `names()`, so `c(g, h)`,
#'   `starts_with("g")`, etc. all work exactly as they do for real
#'   dplyr's own `.by=`)
#' @keywords internal
resolve_by <- function(.data, by_quo, arg_name = ".by") {
  if (rlang::quo_is_null(by_quo)) {
    return(character())
  }

  if (length(.data$groups) > 0) {
    stop("Can't supply `", arg_name, "` when `.data` is a grouped data frame.",
         call. = FALSE)
  }

  nm <- current_schema(.data)$names
  loc <- tidyselect::eval_select(by_quo, stats::setNames(nm, nm))
  names(loc)
}

#' Notify about a CPU-eval fallback, gated by `options(cuplyr.fallback = )`
#'
#' Per D4: default is silent (keeps existing eager behavior unchanged,
#' e.g. `filter(rep(TRUE, n))` is legitimate dplyr, not an error). Set
#' `options(cuplyr.fallback = "warn")` or `"error"` to surface fallbacks.
#'
#' @param verb Character, the verb name (e.g. `"filter"`, `"summarise"`)
#' @param expr_text Character, the offending expression as text
#' @return `invisible(NULL)`
#' @keywords internal
cuplyr_fallback_notify <- function(verb, expr_text) {
  mode <- getOption("cuplyr.fallback", "silent")
  msg <- sprintf(
    "cuplyr: %s() fell back to CPU evaluation for: %s",
    verb, expr_text
  )

  switch(mode,
    "warn" = warning(msg, call. = FALSE),
    "error" = stop(msg, call. = FALSE),
    invisible(NULL)
  )

  invisible(NULL)
}
