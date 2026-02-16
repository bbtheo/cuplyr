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
    "head" = lower_head(ast, source_ptr),
    "summarise" = lower_summarise(ast, source_ptr),
    "join" = lower_join(ast, source_ptr),
    "group_by" = lower_and_execute(ast$input, source_ptr),
    "ungroup" = lower_and_execute(ast$input, source_ptr),
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
#' @keywords internal
lower_filter <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)

  # Cache schema once
  input_schema <- infer_schema(ast$input)

  if (isTRUE(ast$fused) && length(ast$predicates) > 1) {
    # Fused filter: single mask operation
    lower_filter_fused(input_ptr, ast$predicates, input_schema)
  } else {
    # Sequential filters
    result <- input_ptr
    for (pred in ast$predicates) {
      col_idx <- match(pred$col_name, input_schema$names) - 1L
      if (isTRUE(pred$is_col_compare)) {
        rhs_idx <- match(pred$value, input_schema$names) - 1L
        result <- gpu_filter_col(result, col_idx, pred$op, rhs_idx)
      } else {
        result <- gpu_filter_scalar(result, col_idx, pred$op, pred$value)
      }
    }
    result
  }
}

#' Lower fused filter (AND mask)
#' @keywords internal
lower_filter_fused <- function(ptr, predicates, schema) {
  # Build predicate specs for C++
  col_indices <- integer(length(predicates))
  ops <- character(length(predicates))
  values <- numeric(length(predicates))

  for (i in seq_along(predicates)) {
    pred <- predicates[[i]]
    col_indices[i] <- match(pred$col_name, schema$names) - 1L
    ops[i] <- pred$op
    values[i] <- as.numeric(pred$value)
  }

  # Call fused filter (falls back to sequential if not available)
  if (exists("gpu_filter_fused", mode = "function")) {
    gpu_filter_fused(ptr, col_indices, ops, values)
  } else {
    # Fallback to sequential
    result <- ptr
    for (i in seq_along(predicates)) {
      result <- gpu_filter_scalar(result, col_indices[i], ops[i], values[i])
    }
    result
  }
}

#' Lower mutate node
#' @keywords internal
lower_mutate <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  if (length(ast$expressions) == 1) {
    lower_single_mutate(input_ptr, ast$expressions[[1]], input_schema)
  } else if (length(ast$expressions) > 1 && exists("gpu_mutate_batch", mode = "function")) {
    # Use batched mutate if available
    gpu_mutate_batch(input_ptr, ast$expressions, input_schema)
  } else {
    # Fallback: apply expressions sequentially
    result <- input_ptr
    current_schema <- input_schema

    for (expr in ast$expressions) {
      result <- lower_single_mutate(result, expr, current_schema)
      # Update schema for next expression
      current_schema <- update_schema_after_mutate(current_schema, expr)
    }
    result
  }
}

#' Lower a single mutate expression
#' @keywords internal
lower_single_mutate <- function(ptr, expr, schema) {
  output_col <- expr$output_col
  input_cols <- expr$input_cols
  op <- expr$op

  existing_idx <- match(output_col, schema$names)
  is_replace <- !is.na(existing_idx)
  replace_idx <- if (is_replace) existing_idx - 1L else -1L

  if (op == "copy") {
    source_idx <- match(input_cols[1], schema$names) - 1L
    if (is_replace) {
      gpu_copy_column_replace(ptr, source_idx, replace_idx)
    } else {
      gpu_copy_column(ptr, source_idx)
    }
  } else if (!is.null(expr$scalar)) {
    # Column op scalar
    col_idx <- match(input_cols[1], schema$names) - 1L
    if (is_replace) {
      gpu_mutate_binary_scalar_replace(ptr, col_idx, op, expr$scalar, replace_idx)
    } else {
      gpu_mutate_binary_scalar(ptr, col_idx, op, expr$scalar)
    }
  } else {
    # Column op column
    col_idx1 <- match(input_cols[1], schema$names) - 1L
    col_idx2 <- match(input_cols[2], schema$names) - 1L
    if (is_replace) {
      gpu_mutate_binary_cols_replace(ptr, col_idx1, op, col_idx2, replace_idx)
    } else {
      gpu_mutate_binary_cols(ptr, col_idx1, op, col_idx2)
    }
  }
}

#' Update schema after a mutate expression
#' @keywords internal
update_schema_after_mutate <- function(schema, expr) {
  existing_idx <- match(expr$output_col, schema$names)

  if (!is.na(existing_idx)) {
    schema$types[existing_idx] <- expr$output_type
  } else {
    schema$names <- c(schema$names, expr$output_col)
    schema$types <- c(schema$types, expr$output_type)
  }

  schema
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

  # Handle grouped arrange
  if (length(ast$groups) > 0) {
    group_indices <- match(ast$groups, input_schema$names) - 1L
    # Prepend group columns (not already in sort)
    sort_col_names <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
    new_groups <- setdiff(ast$groups, sort_col_names)
    if (length(new_groups) > 0) {
      new_indices <- match(new_groups, input_schema$names) - 1L
      col_indices <- c(new_indices, col_indices)
      descending <- c(rep(FALSE, length(new_indices)), descending)
    }
  }

  gpu_arrange(input_ptr, col_indices, descending)
}

#' Lower head node
#' @keywords internal
lower_head <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)
  head_df <- gpu_head(input_ptr, ast$n, input_schema$names)
  df_to_gpu(head_df)
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
      # Implement right join via swapped left join, then reorder columns
      out <- gpu_left_join(right_ptr, left_ptr, right_key_idx, left_key_idx,
                           integer(0))
      out_schema <- build_join_schema(right_schema, left_schema,
                                      list(left = ast$by$right, right = ast$by$left),
                                      suffix = rev(ast$suffix), keep = TRUE)
      desired <- build_join_schema(left_schema, right_schema, ast$by,
                                   suffix = ast$suffix, keep = ast$keep)$names
      idx <- match(desired, out_schema$names) - 1L
      if (any(is.na(idx))) {
        stop("Right join column reordering failed. Missing columns: ",
             paste(desired[is.na(idx)], collapse = ", "),
             call. = FALSE)
      }
      gpu_select(out, idx)
    },
    stop("Unknown join type: ", ast$join_type, call. = FALSE)
  )
}
