# AST Node Infrastructure for Lazy Evaluation
#
# This module defines the Abstract Syntax Tree (AST) node types used for
# lazy evaluation of GPU operations. Each dplyr verb creates an AST node
# that is only executed when collect() or compute() is called.

# -----------------------------------------------------------------------------
# Base Node Constructor
# -----------------------------------------------------------------------------

#' Create an AST node
#'
#' @param type Character string identifying the node type
#' @param ... Additional named arguments stored on the node
#' @param input The input AST node (NULL for source nodes)
#' @return An ast_node object
#' @keywords internal
ast_node <- function(type, ..., input = NULL) {
  structure(
    list(
      type = type,
      input = input,
      ...
    ),
    class = c(paste0("ast_", type), "ast_node")
  )
}

# -----------------------------------------------------------------------------
# Node Type Constructors
# -----------------------------------------------------------------------------

#' Create a source AST node (leaf node)
#'
#' @param schema List with names and types vectors
#' @return An ast_source node
#' @keywords internal
ast_source <- function(schema, source_ptr = NULL) {
  ast_node("source", schema = schema, source_ptr = source_ptr, input = NULL)
}

#' Create a filter AST node
#'
#' @param input Input AST node
#' @param predicates List of predicate structures
#' @return An ast_filter node
#' @keywords internal
ast_filter <- function(input, predicates) {
  ast_node("filter", input = input, predicates = predicates)
}

#' Create a mutate AST node
#'
#' @param input Input AST node
#' @param expressions List of expression structures
#' @return An ast_mutate node
#' @keywords internal
ast_mutate <- function(input, expressions) {
  ast_node("mutate", input = input, expressions = expressions)
}

#' Create a select AST node
#'
#' @param input Input AST node
#' @param columns Character vector of column names to keep
#' @return An ast_select node
#' @keywords internal
ast_select <- function(input, columns) {

  ast_node("select", input = input, columns = columns)
}

#' Create an arrange AST node
#'
#' @param input Input AST node
#' @param sort_specs List of sort specifications (col_name, descending)
#' @param groups Character vector of group columns (for .by_group)
#' @return An ast_arrange node
#' @keywords internal
ast_arrange <- function(input, sort_specs, groups = character()) {
  ast_node("arrange", input = input, sort_specs = sort_specs, groups = groups)
}

#' Create a group_by AST node (metadata only)
#'
#' @param input Input AST node
#' @param groups Character vector of grouping columns
#' @return An ast_group_by node
#' @keywords internal
ast_group_by <- function(input, groups) {
  ast_node("group_by", input = input, groups = groups)
}

#' Create an ungroup AST node (metadata only)
#'
#' @param input Input AST node
#' @return An ast_ungroup node
#' @keywords internal
ast_ungroup <- function(input) {
 ast_node("ungroup", input = input)
}

#' Create a summarise AST node
#'
#' @param input Input AST node
#' @param aggregations List of aggregation structures
#' @param groups Character vector of group columns (stored explicitly)
#' @return An ast_summarise node
#' @keywords internal
ast_summarise <- function(input, aggregations, groups) {
  ast_node("summarise", input = input, aggregations = aggregations, groups = groups)
}

#' Create a head/limit AST node
#'
#' @param input Input AST node
#' @param n Number of rows to keep
#' @return An ast_head node
#' @keywords internal
ast_head <- function(input, n) {
  ast_node("head", input = input, n = n)
}

#' Create a barrier AST node (optimization fence)
#'
#' @param input Input AST node
#' @return An ast_barrier node
#' @keywords internal
ast_barrier <- function(input) {
  ast_node("barrier", input = input)
}

#' Create a join AST node
#'
#' @param type Join type: "inner", "left", "right", "full"
#' @param left Left input AST node
#' @param right Right input AST node
#' @param by Join specification list(left = <chr>, right = <chr>)
#' @param keep Logical, keep both key columns when names match
#' @param suffix Character vector of length 2
#' @param na_matches Character, "na" or "never"
#' @return An ast_join node
#' @keywords internal
ast_join <- function(type, left, right, by, keep = FALSE,
                     suffix = c(".x", ".y"), na_matches = "na") {
  ast_node("join", left = left, right = right, join_type = type, by = by,
           keep = keep, suffix = suffix, na_matches = na_matches)
}

# -----------------------------------------------------------------------------
# Predicate and Expression Structures
# -----------------------------------------------------------------------------

#' Create a filter predicate structure
#'
#' @param col_name Column name for LHS
#' @param op Comparison operator
#' @param value Scalar value or column name for RHS
#' @param is_col_compare TRUE if RHS is a column name
#' @return A predicate list structure
#' @keywords internal
make_predicate <- function(col_name, op, value, is_col_compare = FALSE) {
  list(
    col_name = col_name,
    op = op,
    value = value,
    is_col_compare = is_col_compare,
    estimated_cost = if (is_col_compare) 2L else 1L,
    is_deterministic = TRUE,
    na_sensitive = op %in% c("==", "!=")
  )
}

#' Create a mutate expression structure
#'
#' @param output_col Output column name
#' @param input_cols Character vector of input column names
#' @param op Operation: "+", "-", "*", "/", "^", "copy", or function name
#' @param scalar Numeric scalar or NULL
#' @param input_types Character vector of input column types
#' @return An expression list structure
#' @keywords internal
make_mutate_expr <- function(output_col, input_cols, op, scalar = NULL,
                              input_types = NULL) {
  list(
    output_col = output_col,
    input_cols = input_cols,
    op = op,
    scalar = scalar,
    input_types = input_types,
    output_type = infer_mutate_output_type(op, input_types, scalar)
  )
}

#' Infer output type for a mutate expression
#'
#' @param op The operation
#' @param input_types Types of input columns
#' @param scalar Scalar value if any
#' @return GPU type string
#' @keywords internal
infer_mutate_output_type <- function(op, input_types, scalar) {
  if (is.null(input_types)) {
    return("FLOAT64")
  }

  if (op == "copy") {
    return(input_types[1])
  }

  # Arithmetic with any FLOAT64 -> FLOAT64
  if (any(input_types == "FLOAT64") || is.double(scalar)) {
    return("FLOAT64")
  }

  # INT32 op INT32 -> INT32 (except division)
  if (all(input_types == "INT32") && op != "/") {
    return("INT32")
  }

  # Division always produces FLOAT64
  if (op == "/") {
    return("FLOAT64")
  }

  # Default to FLOAT64 for safety
  "FLOAT64"
}

#' Create an aggregation structure for summarise
#'
#' @param output_col Output column name
#' @param input_col Input column name
#' @param fn Aggregation function name (sum, mean, min, max, n)
#' @param input_type Input column type
#' @return An aggregation list structure
#' @keywords internal
make_aggregation <- function(output_col, input_col, fn, input_type = NULL) {
  output_type <- switch(fn,
    "n" = "INT32",
    "sum" = if (!is.null(input_type) && input_type == "INT32") "INT64" else "FLOAT64",
    "mean" = "FLOAT64",
    "min" = input_type %||% "FLOAT64",
    "max" = input_type %||% "FLOAT64",
    "FLOAT64"
  )

  list(
    output_col = output_col,
    input_col = input_col,
    fn = fn,
    input_type = input_type,
    output_type = output_type
  )
}

# -----------------------------------------------------------------------------
# Schema Inference
# -----------------------------------------------------------------------------

#' Infer output schema from an AST node
#'
#' @param node An AST node
#' @return List with names and types vectors
#' @export
infer_schema <- function(node) {
  UseMethod("infer_schema")
}

#' @export
infer_schema.ast_source <- function(node) {
  node$schema
}

#' @export
infer_schema.ast_filter <- function(node) {
  infer_schema(node$input)
}

#' @export
infer_schema.ast_select <- function(node) {
  input_schema <- infer_schema(node$input)
  idx <- match(node$columns, input_schema$names)
  list(
    names = input_schema$names[idx],
    types = input_schema$types[idx]
  )
}

#' @export
infer_schema.ast_mutate <- function(node) {
  input_schema <- infer_schema(node$input)
  result_names <- input_schema$names
  result_types <- input_schema$types

  for (expr in node$expressions) {
    existing_idx <- match(expr$output_col, result_names)
    if (!is.na(existing_idx)) {
      result_types[existing_idx] <- expr$output_type
    } else {
      result_names <- c(result_names, expr$output_col)
      result_types <- c(result_types, expr$output_type)
    }
  }

  list(names = result_names, types = result_types)
}

#' @export
infer_schema.ast_arrange <- function(node) {
  infer_schema(node$input)
}

#' @export
infer_schema.ast_group_by <- function(node) {
  infer_schema(node$input)
}

#' @export
infer_schema.ast_ungroup <- function(node) {
  infer_schema(node$input)
}

#' @export
infer_schema.ast_head <- function(node) {
  infer_schema(node$input)
}

#' @export
infer_schema.ast_barrier <- function(node) {
  infer_schema(node$input)
}

#' @export
infer_schema.ast_summarise <- function(node) {
  input_schema <- infer_schema(node$input)

  # Group columns come first
  if (length(node$groups) > 0) {
    group_idx <- match(node$groups, input_schema$names)
    group_names <- input_schema$names[group_idx]
    group_types <- input_schema$types[group_idx]
  } else {
    group_names <- character(0)
    group_types <- character(0)
  }

  # Then aggregation columns
  agg_names <- vapply(node$aggregations, `[[`, character(1), "output_col")
  agg_types <- vapply(node$aggregations, `[[`, character(1), "output_type")

  list(
    names = c(group_names, agg_names),
    types = c(group_types, agg_types)
  )
}

#' @export
infer_schema.ast_join <- function(node) {
  left_schema <- infer_schema(node$left)
  right_schema <- infer_schema(node$right)
  build_join_schema(left_schema, right_schema, node$by,
                    suffix = node$suffix, keep = node$keep)
}

#' @export
infer_schema.NULL <- function(node) {
  list(names = character(0), types = character(0))
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Recursively find all function calls in an R expression
#'
#' @param expr An R expression (from parse())
#' @return Character vector of function names
#' @keywords internal
find_calls <- function(expr) {
  if (is.call(expr)) {
    fn_name <- as.character(expr[[1]])
    # Recurse into arguments
    arg_calls <- unlist(lapply(as.list(expr)[-1], find_calls))
    unique(c(fn_name, arg_calls))
  } else if (is.recursive(expr)) {
    unique(unlist(lapply(expr, find_calls)))
  } else {
    character(0)
  }
}

#' Check if an expression is opaque (contains unknown functions)
#'
#' @param expr_text Expression as text string
#' @return TRUE if expression contains unknown/unsafe functions
#' @keywords internal
is_opaque_expression <- function(expr_text) {
  safe_ops <- c("+", "-", "*", "/", "^", "(", "c")
  safe_fns <- c("desc", "n", "sum", "mean", "min", "max", "sqrt", "abs", "log",
                "log10", "log2", "exp", "ceiling", "floor", "round", "trunc",
                "is.na", "!is.na", "!", "%%", "%/%")

  parsed <- tryCatch(parse(text = expr_text)[[1]], error = function(e) NULL)
  if (is.null(parsed)) return(TRUE)

  calls <- find_calls(parsed)
  unknown <- setdiff(calls, c(safe_ops, safe_fns))

  length(unknown) > 0
}

#' Check if an AST node is an optimization barrier
#'
#' @param node An AST node
#' @return TRUE if node is a barrier
#' @keywords internal
is_barrier <- function(node) {
  if (is.null(node)) return(FALSE)
  node$type %in% c("arrange", "head", "barrier", "summarise")
}

#' Get the depth of an AST tree
#'
#' @param node Root AST node
#' @return Integer depth
#' @keywords internal
ast_depth <- function(node) {
  if (is.null(node)) {
    return(0L)
  }
  if (node$type == "join") {
    return(1L + max(ast_depth(node$left), ast_depth(node$right)))
  }
  if (is.null(node$input)) {
    return(1L)
  }
  1L + ast_depth(node$input)
}

#' Count nodes in an AST tree
#'
#' @param node Root AST node
#' @return Integer count
#' @keywords internal
ast_count <- function(node) {
  if (is.null(node)) {
    return(0L)
  }
  if (node$type == "join") {
    return(1L + ast_count(node$left) + ast_count(node$right))
  }
  1L + ast_count(node$input)
}

#' Attach a source pointer to the source node in an AST
#'
#' @param node Root AST node
#' @param ptr External pointer to GPU table
#' @return AST node with source pointer attached
#' @keywords internal
set_ast_source_ptr <- function(node, ptr) {
  if (is.null(node)) return(NULL)
  if (node$type == "source") {
    node$source_ptr <- ptr
    return(node)
  }
  if (!is.null(node$input)) {
    node$input <- set_ast_source_ptr(node$input, ptr)
  }
  if (!is.null(node$left)) {
    node$left <- set_ast_source_ptr(node$left, ptr)
  }
  if (!is.null(node$right)) {
    node$right <- set_ast_source_ptr(node$right, ptr)
  }
  node
}

# -----------------------------------------------------------------------------
# Print Methods
# -----------------------------------------------------------------------------

#' @export
print.ast_node <- function(x, ..., indent = 0) {
  prefix <- paste0(rep("| ", indent), collapse = "")
  cat(prefix, "ast_", x$type, sep = "")

  # Print type-specific info
  switch(x$type,
    "source" = {
      cat(" [", length(x$schema$names), " cols]", sep = "")
    },
    "filter" = {
      cat(" [", length(x$predicates), " predicates]", sep = "")
    },
    "mutate" = {
      cat(" [", length(x$expressions), " expressions]", sep = "")
    },
    "select" = {
      cat(" [", length(x$columns), " cols]", sep = "")
    },
    "arrange" = {
      cat(" [", length(x$sort_specs), " keys]", sep = "")
    },
    "summarise" = {
      cat(" [", length(x$groups), " groups, ",
          length(x$aggregations), " aggs]", sep = "")
    },
    "head" = {
      cat(" [n=", x$n, "]", sep = "")
    },
    "join" = {
      cat(" [", x$join_type, " join]", sep = "")
    }
  )

  cat("\n")

  if (x$type == "join") {
    print(x$left, indent = indent + 1)
    print(x$right, indent = indent + 1)
  } else if (!is.null(x$input)) {
    print(x$input, indent = indent + 1)
  }

  invisible(x)
}

#' Format AST as a string for debugging
#'
#' @param node Root AST node
#' @return Character string representation
#' @keywords internal
ast_to_string <- function(node) {
  if (is.null(node)) return("NULL")

  info <- switch(node$type,
    "source" = paste0("source[", paste(node$schema$names, collapse = ","), "]"),
    "filter" = paste0("filter[", length(node$predicates), "]"),
    "mutate" = paste0("mutate[", paste(vapply(node$expressions, `[[`, character(1), "output_col"), collapse = ","), "]"),
    "select" = paste0("select[", paste(node$columns, collapse = ","), "]"),
    "arrange" = paste0("arrange[", length(node$sort_specs), "]"),
    "summarise" = paste0("summarise[", length(node$aggregations), "]"),
    "head" = paste0("head[", node$n, "]"),
    "group_by" = paste0("group_by[", paste(node$groups, collapse = ","), "]"),
    "ungroup" = "ungroup",
    "barrier" = "barrier",
    "join" = paste0("join[", node$join_type, "]"),
    node$type
  )

  if (node$type == "join") {
    paste0(info, " -> left(", ast_to_string(node$left),
           "), right(", ast_to_string(node$right), ")")
  } else if (!is.null(node$input)) {
    paste0(info, " -> ", ast_to_string(node$input))
  } else {
    info
  }
}
