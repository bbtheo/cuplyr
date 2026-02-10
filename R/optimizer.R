# AST Optimizer for GPU Operations
#
# This module implements optimization passes for the lazy evaluation AST:
# 1. Projection pruning (push down selects)
# 2. Mutate fusion
# 3. Dead column pruning (drop unused mutate outputs)
# 4. Filter pushdown (across mutate when safe)
# 5. Filter reordering
# 6. Filter fusion
#
# Pass order matters! Run in sequence:
# projection -> mutate fusion -> dead column pruning -> filter pushdown
# -> filter reorder -> filter fusion

#' Optimize an AST for GPU execution
#'
#' Applies all optimization passes in the correct order.
#'
#' @param ast Root AST node
#' @return Optimized AST
#' @keywords internal
optimize_ast <- function(ast) {
  if (is.null(ast)) return(NULL)
  if (!inherits(ast, "ast_node")) {
    stop("Invalid lazy_ops: expected an AST node or NULL.", call. = FALSE)
  }

  # Handle barriers: optimize segments separately
  ast <- optimize_with_barriers(ast)

  ast
}

#' Optimize AST respecting barrier nodes
#'
#' @param ast Root AST node
#' @return Optimized AST
#' @keywords internal
optimize_with_barriers <- function(ast) {
  if (is.null(ast)) return(NULL)

  if (is_barrier(ast)) {
    # Optimize below the barrier, not across
    if (!is.null(ast$input)) {
      ast$input <- optimize_with_barriers(ast$input)
    }
    return(ast)
  }

  # Find the next barrier down the tree
  barrier_node <- find_next_barrier(ast)

  if (is.null(barrier_node)) {
    # No barrier: optimize entire subtree
    optimize_segment(ast)
  } else {
    # Optimize from root to barrier, then recurse below barrier
    segment_root <- extract_segment_above_barrier(ast, barrier_node)
    optimized_segment <- optimize_segment(segment_root)

    # Reconnect: find the bottom of optimized segment and attach barrier
    barrier_node$input <- optimize_with_barriers(barrier_node$input)

    attach_ast_bottom(optimized_segment, barrier_node)
  }
}

#' Optimize a single segment (no barriers)
#'
#' @param ast Root of segment
#' @return Optimized segment
#' @keywords internal
optimize_segment <- function(ast) {
  if (is.null(ast)) return(NULL)

  ast |>
    push_down_projections() |>
    fuse_mutates() |>
    prune_dead_columns() |>
    push_down_filters() |>
    reorder_filters() |>
    fuse_filters()
}

#' Find the next barrier node walking down from root
#'
#' @param ast AST node
#' @return Barrier node or NULL
#' @keywords internal
find_next_barrier <- function(ast) {
  if (is.null(ast)) return(NULL)
  if (is_barrier(ast)) return(ast)
  find_next_barrier(ast$input)
}

#' Extract segment above a barrier (returns copy)
#'
#' @param ast Root node
#' @param barrier Target barrier node
#' @return Copy of segment with NULL input at barrier point
#' @keywords internal
extract_segment_above_barrier <- function(ast, barrier) {
  if (is.null(ast) || identical(ast, barrier)) {
    return(NULL)
  }

  # Shallow copy the node
  new_node <- ast
  new_node$input <- extract_segment_above_barrier(ast$input, barrier)
  new_node
}

#' Find the bottom node of an AST (deepest non-NULL input)
#'
#' @param ast AST node
#' @return Bottom node
#' @keywords internal
find_ast_bottom <- function(ast) {
  if (is.null(ast$input)) return(ast)
  find_ast_bottom(ast$input)
}

#' Attach a node at the bottom of an AST
#'
#' Returns a new AST with the bottom input replaced by `new_input`.
#'
#' @param ast AST node
#' @param new_input Node to attach at bottom
#' @return AST with bottom input replaced
#' @keywords internal
attach_ast_bottom <- function(ast, new_input) {
  if (is.null(ast)) return(NULL)
  if (is.null(ast$input)) {
    ast$input <- new_input
    return(ast)
  }
  ast$input <- attach_ast_bottom(ast$input, new_input)
  ast
}

# -----------------------------------------------------------------------------
# Pass 1: Projection Pruning
# -----------------------------------------------------------------------------

#' Push down column projections to reduce data width early
#'
#' @param ast Root AST node
#' @param required_cols Columns required by parent (NULL = all output cols)
#' @param group_cols Group columns that must be preserved
#' @return Optimized AST with select nodes inserted
#' @keywords internal
push_down_projections <- function(ast, required_cols = NULL, group_cols = character()) {
  if (is.null(ast)) return(NULL)

  if (is.null(required_cols)) {
    required_cols <- infer_schema(ast)$names
  }

  # Always preserve group columns
  required_cols <- union(required_cols, group_cols)

  switch(ast$type,
    "source" = {
      keep <- intersect(ast$schema$names, required_cols)
      if (length(keep) < length(ast$schema$names)) {
        ast_select(ast, keep)
      } else {
        ast
      }
    },
    "filter" = {
      pred_cols <- unique(unlist(lapply(ast$predicates, `[[`, "col_name")))
      rhs_cols <- unlist(lapply(ast$predicates, function(p) {
        if (isTRUE(p$is_col_compare)) p$value else NULL
      }))
      needed <- union(required_cols, union(pred_cols, rhs_cols))
      ast$input <- push_down_projections(ast$input, needed, group_cols)

      extra_cols <- setdiff(union(pred_cols, rhs_cols), required_cols)
      if (length(extra_cols) > 0) {
        ast_select(ast, required_cols)
      } else {
        ast
      }
    },
    "mutate" = {
      expr_inputs <- unique(unlist(lapply(ast$expressions, `[[`, "input_cols")))
      outputs <- vapply(ast$expressions, `[[`, character(1), "output_col")
      needed <- union(setdiff(required_cols, outputs), expr_inputs)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "summarise" = {
      agg_inputs <- unique(unlist(lapply(ast$aggregations, `[[`, "input_col")))
      agg_inputs <- agg_inputs[!is.na(agg_inputs)]  # n() has no input
      needed <- union(ast$groups, agg_inputs)
      ast$input <- push_down_projections(ast$input, needed, ast$groups)
      ast
    },
    "group_by" = {
      ast$input <- push_down_projections(ast$input, required_cols, ast$groups)
      ast
    },
    "select" = {
      ast$input <- push_down_projections(ast$input, ast$columns, group_cols)
      ast
    },
    "arrange" = {
      sort_cols <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
      needed <- union(required_cols, sort_cols)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "join" = {
      left_schema <- infer_schema(ast$left)
      right_schema <- infer_schema(ast$right)
      info <- build_join_output_info(left_schema, right_schema, ast$by,
                                     suffix = ast$suffix, keep = ast$keep)
      req <- intersect(required_cols, info$names)
      idx <- match(req, info$names)
      idx <- idx[!is.na(idx)]
      left_needed <- unique(c(ast$by$left, info$source_names[idx][info$origin[idx] == "left"]))
      right_needed <- unique(c(ast$by$right, info$source_names[idx][info$origin[idx] == "right"]))

      ast$left <- push_down_projections(ast$left, left_needed, group_cols)
      ast$right <- push_down_projections(ast$right, right_needed, group_cols)
      ast
    },
    {
      if (!is.null(ast$input)) {
        ast$input <- push_down_projections(ast$input, required_cols, group_cols)
      }
      ast
    }
  )
}

# -----------------------------------------------------------------------------
# Pass 2: Mutate Fusion
# -----------------------------------------------------------------------------

#' Fuse consecutive mutate nodes when safe
#'
#' @param ast Root AST node
#' @return AST with fused mutates
#' @keywords internal
fuse_mutates <- function(ast) {
  if (is.null(ast)) return(NULL)

  if (ast$type != "mutate") {
    if (!is.null(ast$input)) {
      ast$input <- fuse_mutates(ast$input)
    }
    return(ast)
  }

  # Recursively process input first
  ast$input <- fuse_mutates(ast$input)

  # Check if input is also a mutate
  if (!is.null(ast$input) && ast$input$type == "mutate") {
    combined <- try_fuse_mutate_pair(ast$input, ast)
    if (!is.null(combined)) {
      return(combined)
    }
  }

  ast
}

#' Try to fuse two consecutive mutate nodes
#'
#' @param lower Lower (earlier) mutate node
#' @param upper Upper (later) mutate node
#' @return Fused mutate node or NULL if fusion not possible
#' @keywords internal
try_fuse_mutate_pair <- function(lower, upper) {
  combined_exprs <- c(lower$expressions, upper$expressions)

  # Guard 1: Expression count limit
  if (length(combined_exprs) > 8) {
    return(NULL)
  }

  # Guard 2: Check for dependencies
  lower_outputs <- vapply(lower$expressions, `[[`, character(1), "output_col")
  upper_inputs <- unique(unlist(lapply(upper$expressions, `[[`, "input_cols")))

  has_dependency <- any(upper_inputs %in% lower_outputs)

  if (has_dependency) {
    combined_exprs <- toposort_expressions(combined_exprs)
    if (is.null(combined_exprs)) {
      return(NULL)  # Cycle detected or too complex
    }
  }

  # Guard 3: Count intermediate columns
  intermediate_uses <- table(upper_inputs[upper_inputs %in% lower_outputs])
  n_intermediates <- length(intermediate_uses)
  max_reuse <- if (length(intermediate_uses) > 0) max(intermediate_uses) else 0

  if (n_intermediates > 4 || max_reuse > 3) {
    return(NULL)
  }

  ast_mutate(lower$input, combined_exprs)
}

#' Topologically sort expressions to respect dependencies
#'
#' @param exprs List of expression structures
#' @return Sorted list or NULL if cycle detected
#' @keywords internal
toposort_expressions <- function(exprs) {
  if (length(exprs) == 0) return(list())

  outputs <- vapply(exprs, `[[`, character(1), "output_col")
  names(exprs) <- outputs

  # Kahn's algorithm
  in_degree <- integer(length(exprs))
  names(in_degree) <- outputs

  for (i in seq_along(exprs)) {
    deps <- intersect(exprs[[i]]$input_cols, outputs)
    in_degree[i] <- length(deps)
  }

  result <- list()
  queue <- names(in_degree)[in_degree == 0]

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]
    result <- c(result, list(exprs[[current]]))

    for (i in seq_along(exprs)) {
      if (current %in% exprs[[i]]$input_cols) {
        in_degree[i] <- in_degree[i] - 1
        if (in_degree[i] == 0) {
          queue <- c(queue, outputs[i])
        }
      }
    }
  }

  if (length(result) != length(exprs)) {
    return(NULL)  # Cycle detected
  }

  result
}

# -----------------------------------------------------------------------------
# Pass 3: Dead Column Pruning
# -----------------------------------------------------------------------------

#' Prune unused mutate outputs to reduce intermediate width
#'
#' Walks the AST from root to leaves, tracking columns required downstream.
#' Any mutate expression whose output is not required is dropped; empty mutate
#' nodes are removed entirely.
#'
#' @param ast Root AST node
#' @param required_cols Columns required by parent (NULL = all output cols)
#' @param group_cols Group columns that must be preserved
#' @return AST with unused mutate outputs pruned
#' @keywords internal
prune_dead_columns <- function(ast, required_cols = NULL, group_cols = character()) {
  if (is.null(ast)) return(NULL)

  if (is.null(required_cols)) {
    required_cols <- infer_schema(ast)$names
  }

  required_cols <- union(required_cols, group_cols)

  switch(ast$type,
    "source" = {
      ast
    },
    "filter" = {
      pred_cols <- unique(unlist(lapply(ast$predicates, `[[`, "col_name")))
      rhs_cols <- unlist(lapply(ast$predicates, function(p) {
        if (isTRUE(p$is_col_compare)) p$value else NULL
      }))
      needed <- union(required_cols, union(pred_cols, rhs_cols))
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "mutate" = {
      exprs <- ast$expressions
      if (length(exprs) == 0) {
        return(prune_dead_columns(ast$input, required_cols, group_cols))
      }

      required <- required_cols
      keep <- logical(length(exprs))

      for (i in rev(seq_along(exprs))) {
        expr <- exprs[[i]]
        if (expr$output_col %in% required) {
          keep[i] <- TRUE
          required <- union(setdiff(required, expr$output_col), expr$input_cols)
        }
      }

      if (!any(keep)) {
        return(prune_dead_columns(ast$input, required, group_cols))
      }

      ast$expressions <- exprs[keep]
      ast$input <- prune_dead_columns(ast$input, required, group_cols)
      ast
    },
    "summarise" = {
      agg_inputs <- unique(unlist(lapply(ast$aggregations, `[[`, "input_col")))
      agg_inputs <- agg_inputs[!is.na(agg_inputs)]
      needed <- union(ast$groups, agg_inputs)
      ast$input <- prune_dead_columns(ast$input, needed, ast$groups)
      ast
    },
    "group_by" = {
      ast$input <- prune_dead_columns(ast$input, required_cols, ast$groups)
      ast
    },
    "select" = {
      keep <- intersect(ast$columns, required_cols)
      if (length(keep) == 0) {
        keep <- ast$columns
      }
      ast$columns <- keep
      ast$input <- prune_dead_columns(ast$input, keep, group_cols)
      ast
    },
    "arrange" = {
      sort_cols <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
      needed <- union(required_cols, sort_cols)
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "join" = {
      left_schema <- infer_schema(ast$left)
      right_schema <- infer_schema(ast$right)
      info <- build_join_output_info(left_schema, right_schema, ast$by,
                                     suffix = ast$suffix, keep = ast$keep)
      req <- intersect(required_cols, info$names)
      idx <- match(req, info$names)
      idx <- idx[!is.na(idx)]
      left_needed <- unique(c(ast$by$left, info$source_names[idx][info$origin[idx] == "left"]))
      right_needed <- unique(c(ast$by$right, info$source_names[idx][info$origin[idx] == "right"]))

      ast$left <- prune_dead_columns(ast$left, left_needed, group_cols)
      ast$right <- prune_dead_columns(ast$right, right_needed, group_cols)
      ast
    },
    {
      if (!is.null(ast$input)) {
        ast$input <- prune_dead_columns(ast$input, required_cols, group_cols)
      }
      ast
    }
  )
}

# -----------------------------------------------------------------------------
# Pass 4: Filter Pushdown
# -----------------------------------------------------------------------------

#' Push filters below mutates when predicates do not depend on mutate outputs
#'
#' @param ast Root AST node
#' @return AST with filters pushed down across mutates where safe
#' @keywords internal
push_down_filters <- function(ast) {
  if (is.null(ast)) return(NULL)

  # Recurse first
  if (!is.null(ast$input)) {
    ast$input <- push_down_filters(ast$input)
  }

  if (ast$type != "filter") {
    return(ast)
  }

  input <- ast$input
  if (is.null(input)) {
    return(ast)
  }

  pred_cols <- unique(unlist(lapply(ast$predicates, `[[`, "col_name")))
  rhs_cols <- unlist(lapply(ast$predicates, function(p) {
    if (isTRUE(p$is_col_compare)) p$value else NULL
  }))
  filter_cols <- unique(c(pred_cols, rhs_cols))

  if (input$type == "mutate") {
    mutate_outputs <- vapply(input$expressions, `[[`, character(1), "output_col")

    # Do not push if filter references any mutate outputs (including replacements)
    if (length(intersect(filter_cols, mutate_outputs)) > 0) {
      return(ast)
    }

    # Swap: filter below mutate
    new_filter <- ast_filter(input$input, ast$predicates)
    input$input <- new_filter
    return(input)
  }

  if (input$type == "select") {
    # Only push if select keeps all predicate columns
    if (!all(filter_cols %in% input$columns)) {
      return(ast)
    }

    new_filter <- ast_filter(input$input, ast$predicates)
    input$input <- new_filter
    return(input)
  }

  if (input$type == "join") {
    left_schema <- infer_schema(input$left)
    right_schema <- infer_schema(input$right)
    info <- build_join_output_info(left_schema, right_schema, input$by,
                                   suffix = input$suffix, keep = input$keep)

    classify_predicate <- function(pred) {
      cols <- pred$col_name
      if (isTRUE(pred$is_col_compare)) {
        cols <- c(cols, pred$value)
      }
      sides <- unique(info$origin[match(cols, info$names)])
      sides <- sides[!is.na(sides)]
      if (length(sides) != 1) return("both")
      sides
    }

    left_preds <- list()
    right_preds <- list()
    stay_preds <- list()

    for (pred in ast$predicates) {
      side <- classify_predicate(pred)
      if (side == "left" && input$join_type %in% c("inner", "left")) {
        left_preds <- c(left_preds, list(pred))
      } else if (side == "right" && input$join_type %in% c("inner", "right")) {
        right_preds <- c(right_preds, list(pred))
      } else {
        stay_preds <- c(stay_preds, list(pred))
      }
    }

    if (length(left_preds) > 0) {
      input$left <- ast_filter(input$left, left_preds)
    }
    if (length(right_preds) > 0) {
      input$right <- ast_filter(input$right, right_preds)
    }

    if (length(stay_preds) == 0) {
      return(input)
    }

    ast$predicates <- stay_preds
    ast$input <- input
    return(ast)
  }

  ast
}

# -----------------------------------------------------------------------------
# Pass 5: Filter Reordering
# -----------------------------------------------------------------------------

#' Reorder filters by estimated cost (cheapest first)
#'
#' @param ast Root AST node
#' @return AST with reordered filters
#' @keywords internal
reorder_filters <- function(ast) {
  if (is.null(ast)) return(NULL)

  if (ast$type != "filter") {
    if (!is.null(ast$input)) {
      ast$input <- reorder_filters(ast$input)
    }
    return(ast)
  }

  # Collect consecutive filter chain
  filters <- list()
  current <- ast

  while (!is.null(current) && current$type == "filter") {
    filters <- c(filters, list(current$predicates))
    current <- current$input
  }

  if (length(filters) <= 1) {
    ast$input <- reorder_filters(ast$input)
    return(ast)
  }

  # Flatten predicates
  all_preds <- unlist(filters, recursive = FALSE)

  # Safety check: don't reorder non-deterministic predicates
  safe_to_reorder <- all(vapply(all_preds, function(p) {
    isTRUE(p$is_deterministic)
  }, logical(1)))

  if (safe_to_reorder) {
    costs <- vapply(all_preds, `[[`, integer(1), "estimated_cost")
    all_preds <- all_preds[order(costs)]
  }

  # Rebuild single filter node
  result <- ast_filter(current, all_preds)
  result$input <- reorder_filters(result$input)
  result
}

# -----------------------------------------------------------------------------
# Pass 6: Filter Fusion
# -----------------------------------------------------------------------------

#' Mark filter nodes for fused AND-mask lowering
#'
#' @param ast Root AST node
#' @return AST with fused flags set
#' @keywords internal
fuse_filters <- function(ast) {
  if (is.null(ast)) return(NULL)

  if (ast$type != "filter" || length(ast$predicates) <= 1) {
    if (!is.null(ast$input)) {
      ast$input <- fuse_filters(ast$input)
    }
    return(ast)
  }

  all_simple <- all(vapply(ast$predicates, function(p) {
    !isTRUE(p$is_col_compare) && p$op %in% c("==", "!=", ">", ">=", "<", "<=")
  }, logical(1)))

  if (all_simple && length(ast$predicates) <= 4) {
    ast$fused <- TRUE
  } else {
    ast$fused <- FALSE
  }

  ast$input <- fuse_filters(ast$input)
  ast
}
