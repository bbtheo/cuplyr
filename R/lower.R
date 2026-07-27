# AST Lowering - Convert AST to GPU Operations
#
# This module translates the optimized AST into actual GPU operations
# by calling the C++ functions in the cuplyr package.

#' Lower and execute an AST against a GPU table
#'
#' @param ast Optimized AST root node
#' @param source_ptr External pointer to source GPU table
#' @return External pointer to result GPU table
#' @keywords internal
lower_and_execute <- function(ast, source_ptr) {
  if (is.null(ast)) {
    return(source_ptr)
  }

  switch(ast$type,
    "source" = if (!is.null(ast$source_ptr)) ast$source_ptr else source_ptr,
    "select" = lower_select(ast, source_ptr),
    "filter" = lower_filter(ast, source_ptr),
    "mutate" = lower_mutate(ast, source_ptr),
    "arrange" = lower_arrange(ast, source_ptr),
    "summarise" = lower_summarise(ast, source_ptr),
    "join" = lower_join(ast, source_ptr),
    "barrier" = lower_and_execute(ast$input, source_ptr),
    stop("Unknown AST node type: ", ast$type, call. = FALSE)
  )
}

#' Lower select node
#' @keywords internal
lower_select <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)
  indices <- match(ast$columns, input_schema$names) - 1L
  gpu_select(input_ptr, indices)
}

#' Lower filter node
#'
#' Task T3 (Phase 1 expression-engine cutover, `scratchpad/phase1_expression_engine.md`
#' section 5): every predicate in the node is a `cuplyr_ir` node (see
#' `R/ir.R`, `R/ast.R::make_predicate()`). All of them are folded together
#' with `&` into a single expression -- one filter node is always exactly
#' one fused GPU kernel now, regardless of how many predicates it holds, so
#' there is no separate "fused vs sequential" branch (and no `ast$fused`
#' flag) to check.
#' @keywords internal
lower_filter <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  if (length(ast$predicates) == 0) {
    return(input_ptr)
  }

  combined_ir <- ast$predicates[[1]]$ir
  for (pred in ast$predicates[-1]) {
    combined_ir <- ir_call("&", list(combined_ir, pred$ir))
  }

  bound_ir <- ir_bind(combined_ir, input_schema)
  gpu_filter_expr(input_ptr, bound_ir)
}

#' Lower mutate node
#' @keywords internal
lower_mutate <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  gpu_mutate_batch(input_ptr, ast$expressions, input_schema)
}

#' Lower arrange node
#' @keywords internal
lower_arrange <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  col_indices <- integer(length(ast$sort_specs))
  descending <- logical(length(ast$sort_specs))

  for (i in seq_along(ast$sort_specs)) {
    spec <- ast$sort_specs[[i]]
    col_indices[i] <- match(spec$col_name, input_schema$names) - 1L
    descending[i] <- isTRUE(spec$descending)
  }

  # Handle grouped arrange (.by_group = TRUE): dplyr's arrange.grouped_df
  # always prepends the group columns ASCENDING ahead of the user's own sort
  # keys (D8), even if the user's dots already sort by one of those columns
  # (possibly with desc()) -- that user key simply becomes a later, no-op
  # tiebreaker since the ascending group prepend already fully orders it.
  if (length(ast$groups) > 0) {
    group_indices <- match(ast$groups, input_schema$names) - 1L
    col_indices <- c(group_indices, col_indices)
    descending <- c(rep(FALSE, length(group_indices)), descending)
  }

  gpu_arrange(input_ptr, col_indices, descending)
}

#' Lower summarise node
#' @keywords internal
lower_summarise <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  # Build aggregation specs
  group_indices <- if (length(ast$groups) > 0) {
    match(ast$groups, input_schema$names) - 1L
  } else {
    integer(0)
  }

  agg_col_indices <- integer(length(ast$aggregations))
  agg_fns <- character(length(ast$aggregations))

  for (i in seq_along(ast$aggregations)) {
    agg <- ast$aggregations[[i]]
    agg_fns[i] <- agg$fn
    if (agg$fn == "n") {
      agg_col_indices[i] <- 0L  # n() doesn't need a column
    } else {
      agg_col_indices[i] <- match(agg$input_col, input_schema$names) - 1L
    }
  }

  gpu_summarise(input_ptr, group_indices, agg_col_indices, agg_fns)
}

#' Lower join node
#' @keywords internal
lower_join <- function(ast, source_ptr) {
  left_ptr <- lower_and_execute(ast$left, source_ptr)
  right_ptr <- lower_and_execute(ast$right, source_ptr)

  left_schema <- infer_schema(ast$left)
  right_schema <- infer_schema(ast$right)

  left_key_idx <- match(ast$by$left, left_schema$names) - 1L
  right_key_idx <- match(ast$by$right, right_schema$names) - 1L

  right_drop <- if (!isTRUE(ast$keep)) ast$by$right else character(0)
  right_drop_idx <- if (length(right_drop) > 0) {
    match(right_drop, right_schema$names) - 1L
  } else {
    integer(0)
  }

  switch(ast$join_type,
    "left" = gpu_left_join(left_ptr, right_ptr, left_key_idx, right_key_idx, right_drop_idx),
    "inner" = gpu_inner_join(left_ptr, right_ptr, left_key_idx, right_key_idx, right_drop_idx),
    "full" = gpu_full_join(left_ptr, right_ptr, left_key_idx, right_key_idx, right_drop_idx),
    "right" = {
      # Implement right join via swapped left join, then reorder columns.
      # Shared with the eager path in right_join.tbl_gpu() (R/join.R) via
      # build_right_join_via_left() / resolve_right_join_select_idx() --
      # right_join.tbl_gpu() builds this exact ast_join("right", ...) node
      # whenever either input is lazy, so this branch is the live lazy path
      # and must stay in lockstep with the eager one (including the
      # keep = FALSE join-key coalescing fix in build_right_join_via_left()).
      plan <- build_right_join_via_left(left_schema, right_schema, ast$by,
                                        suffix = ast$suffix, keep = ast$keep)
      out <- gpu_left_join(right_ptr, left_ptr, right_key_idx, left_key_idx,
                           integer(0))
      idx <- resolve_right_join_select_idx(plan, left_schema, right_schema, ast$suffix)
      gpu_select(out, idx - 1L)
    },
    stop("Unknown join type: ", ast$join_type, call. = FALSE)
  )
}
