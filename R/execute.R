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
#' @param input_schema The schema of the op's input (pre-op)
#' @param surviving_names Character vector of column names in the new schema
#' @return A named list of factor levels, or `NULL` if none survive
#' @keywords internal
propagate_factor_levels <- function(input_schema, surviving_names) {
  factor_levels <- input_schema$factor_levels
  if (is.null(factor_levels) || length(factor_levels) == 0) {
    return(NULL)
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
#' @param node The AST node being applied
#' @param groups Character vector of the input's group columns
#' @param schema The node's output schema (post-op), as from `infer_schema()`
#' @return Character vector of group columns to carry forward
#' @keywords internal
propagate_groups <- function(node, groups, schema) {
  if (node$type %in% c("summarise", "join")) {
    return(character())
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
  new_schema$factor_levels <- propagate_factor_levels(input_schema, new_schema$names)

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
