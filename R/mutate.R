#' Create or modify columns in a GPU table
#'
#' Adds new columns or modifies existing columns in a GPU table using
#' arithmetic expressions, similar to `dplyr::mutate()`. All computations
#' are performed on the GPU for maximum performance.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Name-value pairs of expressions. The name gives the column name
#'   (new or existing), and the value is an arithmetic expression involving
#'   existing columns and/or scalar values.
#'
#' @return A `tbl_gpu` object with the new or modified columns. If a column
#'   name already exists, it is replaced. New columns are appended.
#'
#' @details
#' ## Supported arithmetic operators
#' \itemize{
#'   \item `+` - addition
#'   \item `-` - subtraction
#'   \item `*` - multiplication
#'   \item `/` - division
#'   \item `^` - exponentiation (power)
#' }
#'
#' ## Column replacement behavior
#' When the output column name matches an existing column, the existing
#' column is replaced in-place (preserving column order). For example,
#' `mutate(x = x + 1)` will modify `x` rather than creating a duplicate.
#'
#' ## Current limitations
#' \itemize{
#'   \item Only binary operations are supported (col op value or col op col)
#'   \item Complex expressions like `(x + y) * z` are not yet supported
#'   \item Functions like `sqrt()`, `log()`, `abs()` are not yet implemented
#'   \item Result type is always FLOAT64 (double precision)
#' }
#'
#' ## Performance
#' GPU arithmetic operations are highly vectorized and can process
#' billions of elements per second. Memory bandwidth is typically
#' the limiting factor, not compute.
#'
#' @seealso
#' \code{\link{filter.tbl_gpu}} for filtering rows,
#' \code{\link{select.tbl_gpu}} for selecting columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr mutate
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Add a new column
#'   result <- gpu_mtcars |>
#'     mutate(kpl = mpg * 0.425) |>
#'     collect()
#'
#'   # Modify an existing column
#'   adjusted <- gpu_mtcars |>
#'     mutate(mpg = mpg + 5) |>
#'     collect()
#'
#'   # Combine two columns
#'   gpu_cars <- tbl_gpu(cars)
#'   result <- gpu_cars |>
#'     mutate(ratio = dist / speed) |>
#'     collect()
#'
#'   # Chain multiple mutations
#'   result <- gpu_mtcars |>
#'     mutate(power_weight = hp / wt) |>
#'     mutate(efficiency = mpg * power_weight) |>
#'     collect()
#' }
mutate.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)

  if (length(dots) == 0) return(.data)

  # Lazy path: build AST instead of executing
  if (.data$exec_mode == "lazy") {
    return(mutate_lazy(.data, dots))
  }

  # Eager path: execute immediately
  result <- .data

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]

    # Handle unnamed expressions: use expression text as column name (dplyr behavior)
    if (is.null(new_name) || new_name == "") {
      new_name <- rlang::quo_text(expr)
      # Warn user about auto-generated name
      warning("Unnamed mutate expression '", new_name, "' will use expression as column name.\n",
              "Consider using explicit names: mutate(name = ", new_name, ")",
              call. = FALSE)
    }

    result <- mutate_one(result, new_name, expr)
  }

  result
}

# Lazy mutate: build AST node
mutate_lazy <- function(.data, dots) {
  expressions <- list()

  # Get current schema (from AST or base)
  current_schema <- if (!is.null(.data$lazy_ops)) {
    infer_schema(.data$lazy_ops)
  } else {
    .data$schema
  }

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]

    if (is.null(new_name) || new_name == "") {
      new_name <- rlang::quo_text(expr)
      warning("Unnamed mutate expression '", new_name, "' will use expression as column name.\n",
              "Consider using explicit names: mutate(name = ", new_name, ")",
              call. = FALSE)
    }

    parsed <- parse_mutate_exprs(new_name, expr, current_schema)

    if (is.null(parsed)) {
      # Opaque expression - fall back to eager
      warning("Opaque mutate expression '", new_name, "', falling back to eager execution",
              call. = FALSE)
      .data <- as_eager(.data)
      for (j in seq_along(dots)) {
        nm <- names(dots)[j]
        if (is.null(nm) || nm == "") nm <- rlang::quo_text(dots[[j]])
        .data <- mutate_one(.data, nm, dots[[j]])
      }
      return(as_lazy(.data))
    }

    expressions <- c(expressions, parsed)

    # Update schema for subsequent expressions
    for (expr_step in parsed) {
      current_schema <- update_schema_for_expr(current_schema, expr_step)
    }
  }

  # Initialize AST if needed
  if (is.null(.data$lazy_ops)) {
    .data$lazy_ops <- ast_source(.data$schema)
  }

  # Add mutate node
  .data$lazy_ops <- ast_mutate(.data$lazy_ops, expressions)

  # Update schema
  .data$schema <- current_schema

  .data
}

# Parse a mutate expression into expression structure
parse_mutate_expr <- function(new_name, expr, schema) {
  expr_chr <- rlang::quo_text(expr)

  # Check if expression is just a column name (simple copy)
  if (expr_chr %in% schema$names) {
    col_type <- schema$types[match(expr_chr, schema$names)]
    return(make_mutate_expr(new_name, expr_chr, "copy",
                            input_types = col_type))
  }

  # Parse arithmetic: col op value or col op col
  ops <- c("+", "-", "*", "/", "^")
  op_found <- NULL
  op_pos <- NULL

  for (op in ops) {
    pos <- regexpr(op, expr_chr, fixed = TRUE)
    if (pos > 0) {
      if (is.null(op_pos) || pos < op_pos) {
        op_found <- op
        op_pos <- pos
      }
    }
  }

  if (is.null(op_found)) {
    return(NULL)  # Opaque expression
  }

  lhs <- trimws(substr(expr_chr, 1, op_pos - 1))
  rhs <- trimws(substr(expr_chr, op_pos + 1, nchar(expr_chr)))

  # Validate LHS is a column
  if (!lhs %in% schema$names) {
    return(NULL)
  }

  lhs_type <- schema$types[match(lhs, schema$names)]

  # Check if RHS is a column
  if (rhs %in% schema$names) {
    rhs_type <- schema$types[match(rhs, schema$names)]
    return(make_mutate_expr(new_name, c(lhs, rhs), op_found,
                            input_types = c(lhs_type, rhs_type)))
  }

  # Try to parse RHS as scalar
  value <- tryCatch(eval(parse(text = rhs)), error = function(e) NULL)
  if (is.null(value) || !is.numeric(value) || length(value) != 1) {
    return(NULL)
  }

  make_mutate_expr(new_name, lhs, op_found, scalar = value,
                   input_types = lhs_type)
}

# Parse a mutate expression into one or more expression structures
parse_mutate_exprs <- function(new_name, expr, schema) {
  tree <- parse_mutate_exprs_tree(new_name, expr, schema)
  if (!is.null(tree)) {
    return(tree)
  }

  linear <- parse_linear_mutate_exprs(new_name, expr, schema)
  if (!is.null(linear)) {
    return(linear)
  }

  parsed <- parse_mutate_expr(new_name, expr, schema)
  if (is.null(parsed)) {
    return(NULL)
  }
  list(parsed)
}

# Parse nested arithmetic expression trees into sequential steps
parse_mutate_exprs_tree <- function(new_name, expr, schema) {
  expr_obj <- rlang::quo_get_expr(expr)
  node <- parse_mutate_node(expr_obj)
  if (is.null(node)) return(NULL)

  steps <- list()
  current_schema <- schema

  emit_col_scalar <- function(lhs_col, op, scalar) {
    lhs_idx <- match(lhs_col, current_schema$names)
    if (is.na(lhs_idx)) return(FALSE)
    lhs_type <- current_schema$types[lhs_idx]
    step <- make_mutate_expr(new_name, lhs_col, op, scalar = scalar,
                             input_types = lhs_type)
    steps <<- c(steps, list(step))
    current_schema <<- update_schema_for_expr(current_schema, step)
    TRUE
  }

  emit_col_col <- function(lhs_col, op, rhs_col) {
    lhs_idx <- match(lhs_col, current_schema$names)
    rhs_idx <- match(rhs_col, current_schema$names)
    if (is.na(lhs_idx) || is.na(rhs_idx)) return(FALSE)
    lhs_type <- current_schema$types[lhs_idx]
    rhs_type <- current_schema$types[rhs_idx]
    step <- make_mutate_expr(new_name, c(lhs_col, rhs_col), op,
                             input_types = c(lhs_type, rhs_type))
    steps <<- c(steps, list(step))
    current_schema <<- update_schema_for_expr(current_schema, step)
    TRUE
  }

  build <- function(n) {
    if (n$type == "col" || n$type == "scalar") {
      return(FALSE)
    }

    if (n$type != "call") return(FALSE)

    lhs_node <- parse_mutate_node(n$lhs)
    rhs_node <- parse_mutate_node(n$rhs)
    if (is.null(lhs_node) || is.null(rhs_node)) return(FALSE)

    op <- n$op

    if (lhs_node$type == "call" && rhs_node$type == "call") {
      return(FALSE)
    }

    if (lhs_node$type == "call") {
      if (!build(lhs_node)) return(FALSE)
      if (rhs_node$type == "col") {
        return(emit_col_col(new_name, op, rhs_node$name))
      }
      if (rhs_node$type == "scalar") {
        return(emit_col_scalar(new_name, op, rhs_node$value))
      }
      return(FALSE)
    }

    if (rhs_node$type == "call") {
      if (!build(rhs_node)) return(FALSE)
      if (lhs_node$type == "col") {
        return(emit_col_col(lhs_node$name, op, new_name))
      }
      if (lhs_node$type == "scalar") {
        if (op %in% c("+", "*")) {
          return(emit_col_scalar(new_name, op, lhs_node$value))
        }
      }
      return(FALSE)
    }

    if (lhs_node$type == "col" && rhs_node$type == "col") {
      return(emit_col_col(lhs_node$name, op, rhs_node$name))
    }

    if (lhs_node$type == "col" && rhs_node$type == "scalar") {
      return(emit_col_scalar(lhs_node$name, op, rhs_node$value))
    }

    if (lhs_node$type == "scalar" && rhs_node$type == "col") {
      if (op %in% c("+", "*")) {
        return(emit_col_scalar(rhs_node$name, op, lhs_node$value))
      }
    }

    FALSE
  }

  if (!build(node)) return(NULL)

  steps
}

# Parse left-associative chains of + or - into multiple expressions
parse_linear_mutate_exprs <- function(new_name, expr, schema) {
  expr_obj <- rlang::quo_get_expr(expr)
  flat <- flatten_left_assoc_ops(expr_obj)
  if (is.null(flat)) {
    return(NULL)
  }

  terms <- flat$terms
  ops <- flat$ops

  if (length(terms) < 2) {
    return(NULL)
  }

  # First term must be a column
  if (!rlang::is_symbol(terms[[1]])) {
    return(NULL)
  }
  lhs_name <- as.character(terms[[1]])
  if (!lhs_name %in% schema$names) {
    return(NULL)
  }

  expressions <- list()
  current_schema <- schema

  for (i in 2:length(terms)) {
    op <- ops[i - 1]
    term <- terms[[i]]

    lhs_type <- current_schema$types[match(lhs_name, current_schema$names)]

    if (rlang::is_symbol(term)) {
      rhs_name <- as.character(term)
      if (!rhs_name %in% current_schema$names) {
        return(NULL)
      }
      rhs_type <- current_schema$types[match(rhs_name, current_schema$names)]
      expr_step <- make_mutate_expr(new_name, c(lhs_name, rhs_name), op,
                                    input_types = c(lhs_type, rhs_type))
    } else if (is.numeric(term) && length(term) == 1) {
      expr_step <- make_mutate_expr(new_name, lhs_name, op, scalar = term,
                                    input_types = lhs_type)
    } else {
      return(NULL)
    }

    expressions <- c(expressions, list(expr_step))
    current_schema <- update_schema_for_expr(current_schema, expr_step)
    lhs_name <- new_name
  }

  expressions
}

# Flatten left-associative + / - chains, return terms and ops or NULL
flatten_left_assoc_ops <- function(expr) {
  if (!rlang::is_call(expr)) {
    return(NULL)
  }

  op <- rlang::call_name(expr)
  if (is.null(op) || !op %in% c("+", "-")) {
    return(NULL)
  }

  terms <- list()
  ops <- character()
  current <- expr

  while (rlang::is_call(current) && rlang::call_name(current) %in% c("+", "-")) {
    current_op <- rlang::call_name(current)
    rhs <- current[[3]]

    # Only handle left-associative chains (reject nested right ops)
    if (rlang::is_call(rhs) && rlang::call_name(rhs) %in% c("+", "-")) {
      return(NULL)
    }

    terms <- c(list(rhs), terms)
    ops <- c(current_op, ops)
    current <- current[[2]]
  }

  terms <- c(list(current), terms)
  list(terms = terms, ops = ops)
}

# Update schema after adding an expression
update_schema_for_expr <- function(schema, expr) {
  existing_idx <- match(expr$output_col, schema$names)

  if (!is.na(existing_idx)) {
    schema$types[existing_idx] <- expr$output_type
  } else {
    schema$names <- c(schema$names, expr$output_col)
    schema$types <- c(schema$types, expr$output_type)
  }

  schema
}

# -----------------------------------------------------------------------------
# Structured mutate evaluation (eager path)
# -----------------------------------------------------------------------------

parse_mutate_node <- function(expr) {
  if (rlang::is_call(expr) && identical(rlang::call_name(expr), "(") && length(expr) == 2) {
    return(parse_mutate_node(expr[[2]]))
  }
  if (rlang::is_symbol(expr)) {
    return(list(type = "col", name = as.character(expr)))
  }
  if (is.numeric(expr) && length(expr) == 1) {
    return(list(type = "scalar", value = expr))
  }
  if (rlang::is_call(expr)) {
    op <- rlang::call_name(expr)
    if (op %in% c("+", "-") && length(expr) == 2) {
      arg_node <- parse_mutate_node(expr[[2]])
      if (!is.null(arg_node) && arg_node$type == "scalar") {
        value <- if (op == "-") -arg_node$value else arg_node$value
        return(list(type = "scalar", value = value))
      }
    }
    if (!is.null(op) && op %in% c("+", "-", "*", "/", "^") && length(expr) >= 3) {
      return(list(type = "call", op = op, lhs = expr[[2]], rhs = expr[[3]]))
    }
  }
  NULL
}

mutate_apply_col_scalar <- function(.data, output_col, lhs_col, op, scalar) {
  existing_idx <- match(output_col, .data$schema$names)
  is_replace <- !is.na(existing_idx)
  replace_idx <- if (is_replace) existing_idx - 1L else -1L
  col_idx <- match(lhs_col, .data$schema$names)
  if (is.na(col_idx)) {
    stop("Column '", lhs_col, "' not found.\n",
         "Available columns: ", paste(.data$schema$names, collapse = ", "),
         call. = FALSE)
  }
  col_idx <- col_idx - 1L

  if (is_replace) {
    new_ptr <- gpu_mutate_binary_scalar_replace(.data$ptr, col_idx, op, scalar, replace_idx)
  } else {
    new_ptr <- gpu_mutate_binary_scalar(.data$ptr, col_idx, op, scalar)
  }

  lhs_type <- .data$schema$types[match(lhs_col, .data$schema$names)]
  expr_step <- make_mutate_expr(output_col, lhs_col, op, scalar = scalar,
                                input_types = lhs_type)
  new_schema <- update_schema_for_expr(.data$schema, expr_step)
  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}

mutate_apply_col_col <- function(.data, output_col, lhs_col, op, rhs_col) {
  existing_idx <- match(output_col, .data$schema$names)
  is_replace <- !is.na(existing_idx)
  replace_idx <- if (is_replace) existing_idx - 1L else -1L
  col_idx1 <- match(lhs_col, .data$schema$names)
  col_idx2 <- match(rhs_col, .data$schema$names)
  if (is.na(col_idx1)) {
    stop("Column '", lhs_col, "' not found.\n",
         "Available columns: ", paste(.data$schema$names, collapse = ", "),
         call. = FALSE)
  }
  if (is.na(col_idx2)) {
    stop("Column '", rhs_col, "' not found.\n",
         "Available columns: ", paste(.data$schema$names, collapse = ", "),
         call. = FALSE)
  }
  col_idx1 <- col_idx1 - 1L
  col_idx2 <- col_idx2 - 1L

  if (is_replace) {
    new_ptr <- gpu_mutate_binary_cols_replace(.data$ptr, col_idx1, op, col_idx2, replace_idx)
  } else {
    new_ptr <- gpu_mutate_binary_cols(.data$ptr, col_idx1, op, col_idx2)
  }

  lhs_type <- .data$schema$types[match(lhs_col, .data$schema$names)]
  rhs_type <- .data$schema$types[match(rhs_col, .data$schema$names)]
  expr_step <- make_mutate_expr(output_col, c(lhs_col, rhs_col), op,
                                input_types = c(lhs_type, rhs_type))
  new_schema <- update_schema_for_expr(.data$schema, expr_step)
  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}

mutate_eval_expr <- function(.data, output_col, expr) {
  node <- parse_mutate_node(expr)
  if (is.null(node)) return(NULL)

  if (node$type == "col") {
    return(mutate_copy_column(.data, output_col, node$name))
  }
  if (node$type == "scalar") {
    return(NULL)
  }

  lhs_node <- parse_mutate_node(node$lhs)
  rhs_node <- parse_mutate_node(node$rhs)
  if (is.null(lhs_node) || is.null(rhs_node)) return(NULL)

  op <- node$op

  if (lhs_node$type == "call" && rhs_node$type == "call") {
    return(NULL)
  }

  if (lhs_node$type == "call") {
    data_step <- mutate_eval_expr(.data, output_col, node$lhs)
    if (is.null(data_step)) return(NULL)

    if (rhs_node$type == "col") {
      return(mutate_apply_col_col(data_step, output_col, output_col, op, rhs_node$name))
    }
    if (rhs_node$type == "scalar") {
      return(mutate_apply_col_scalar(data_step, output_col, output_col, op, rhs_node$value))
    }
  }

  if (rhs_node$type == "call") {
    data_step <- mutate_eval_expr(.data, output_col, node$rhs)
    if (is.null(data_step)) return(NULL)

    if (lhs_node$type == "col") {
      return(mutate_apply_col_col(data_step, output_col, lhs_node$name, op, output_col))
    }
    if (lhs_node$type == "scalar") {
      if (op %in% c("+", "*")) {
        return(mutate_apply_col_scalar(data_step, output_col, output_col, op, lhs_node$value))
      }
      return(NULL)
    }
  }

  if (lhs_node$type == "col" && rhs_node$type == "col") {
    return(mutate_apply_col_col(.data, output_col, lhs_node$name, op, rhs_node$name))
  }

  if (lhs_node$type == "col" && rhs_node$type == "scalar") {
    return(mutate_apply_col_scalar(.data, output_col, lhs_node$name, op, rhs_node$value))
  }

  if (lhs_node$type == "scalar" && rhs_node$type == "col") {
    if (op %in% c("+", "*")) {
      return(mutate_apply_col_scalar(.data, output_col, rhs_node$name, op, lhs_node$value))
    }
  }

  NULL
}

# Internal: Parse and execute a single mutate expression
#
# Parses a quosure containing an arithmetic expression and calls the
# appropriate GPU binary operation function.
#
# @param .data A tbl_gpu object
# @param new_name Name for the output column
# @param expr A quosure with an arithmetic expression
# @return A tbl_gpu with the new/modified column
# @keywords internal
mutate_one <- function(.data, new_name, expr) {
  expr_chr <- rlang::quo_text(expr)
  expr_obj <- rlang::quo_get_expr(expr)

  linear <- parse_linear_mutate_exprs(new_name, expr, .data$schema)
  if (!is.null(linear)) {
    apply_expr <- function(.data_step, expr_step) {
      output_col <- expr_step$output_col
      input_cols <- expr_step$input_cols
      op <- expr_step$op
      existing_idx <- match(output_col, .data_step$schema$names)
      is_replace <- !is.na(existing_idx)
      replace_idx <- if (is_replace) existing_idx - 1L else -1L

      if (op == "copy") {
        source_idx <- match(input_cols[1], .data_step$schema$names) - 1L
        if (is_replace) {
          new_ptr <- gpu_copy_column_replace(.data_step$ptr, source_idx, replace_idx)
        } else {
          new_ptr <- gpu_copy_column(.data_step$ptr, source_idx)
        }
      } else if (!is.null(expr_step$scalar)) {
        col_idx <- match(input_cols[1], .data_step$schema$names) - 1L
        if (is_replace) {
          new_ptr <- gpu_mutate_binary_scalar_replace(.data_step$ptr, col_idx, op,
                                                      expr_step$scalar, replace_idx)
        } else {
          new_ptr <- gpu_mutate_binary_scalar(.data_step$ptr, col_idx, op, expr_step$scalar)
        }
      } else {
        col_idx1 <- match(input_cols[1], .data_step$schema$names) - 1L
        col_idx2 <- match(input_cols[2], .data_step$schema$names) - 1L
        if (is_replace) {
          new_ptr <- gpu_mutate_binary_cols_replace(.data_step$ptr, col_idx1, op,
                                                    col_idx2, replace_idx)
        } else {
          new_ptr <- gpu_mutate_binary_cols(.data_step$ptr, col_idx1, op, col_idx2)
        }
      }

      new_schema <- update_schema_for_expr(.data_step$schema, expr_step)
      new_tbl_gpu(
        ptr = new_ptr,
        schema = new_schema,
        groups = .data_step$groups,
        exec_mode = .data_step$exec_mode
      )
    }

    result <- .data
    for (expr_step in linear) {
      result <- apply_expr(result, expr_step)
    }
    return(result)
  }

  # Try structured evaluation for nested arithmetic expressions
  structured <- mutate_eval_expr(.data, new_name, expr_obj)
  if (!is.null(structured)) {
    return(structured)
  }

  # Check if expression is just a column name (simple copy)
  if (expr_chr %in% .data$schema$names) {
    return(mutate_copy_column(.data, new_name, expr_chr))
  }

  # Parse simple arithmetic: col op value or col op col
  # Find the first operator
  ops <- c("+", "-", "*", "/", "^")
  op_found <- NULL
  op_pos <- NULL

  for (op in ops) {
    pos <- regexpr(op, expr_chr, fixed = TRUE)
    if (pos > 0) {
      if (is.null(op_pos) || pos < op_pos) {
        op_found <- op
        op_pos <- pos
      }
    }
  }

  if (is.null(op_found)) {
    stop("mutate() only supports column copies or arithmetic operations: +, -, *, /, ^\n",
         "Expression: ", expr_chr, call. = FALSE)
  }

  lhs <- trimws(substr(expr_chr, 1, op_pos - 1))
  rhs <- trimws(substr(expr_chr, op_pos + 1, nchar(expr_chr)))

  lhs_idx <- tryCatch(col_index(.data, lhs), error = function(e) NULL)
  rhs_idx <- tryCatch(col_index(.data, rhs), error = function(e) NULL)

  if (is.null(lhs_idx)) {
    stop("Column '", lhs, "' not found.\n",
         "Available columns: ", paste(.data$schema$names, collapse = ", "),
         call. = FALSE)
  }

  existing_idx <- match(new_name, .data$schema$names)

  if (!is.null(rhs_idx)) {
    # Column to column operation
    if (!is.na(existing_idx)) {
      new_ptr <- gpu_mutate_binary_cols_replace(.data$ptr, lhs_idx, op_found, rhs_idx, existing_idx - 1L)
    } else {
      new_ptr <- gpu_mutate_binary_cols(.data$ptr, lhs_idx, op_found, rhs_idx)
    }
  } else {
    # Column to scalar operation
    value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
      stop("Cannot parse value: ", rhs, call. = FALSE)
    })
    if (!is.numeric(value) || length(value) != 1) {
      stop("mutate() currently only supports numeric scalar operations.\n",
           "Got: ", class(value)[1], " of length ", length(value), call. = FALSE)
    }
    if (!is.na(existing_idx)) {
      new_ptr <- gpu_mutate_binary_scalar_replace(.data$ptr, lhs_idx, op_found, as.double(value), existing_idx - 1L)
    } else {
      new_ptr <- gpu_mutate_binary_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
    }
  }

  if (!is.na(existing_idx)) {
    new_schema <- .data$schema
    new_schema$types[existing_idx] <- "FLOAT64"
  } else {
    # Adding new column
    new_schema <- .data$schema
    new_schema$names <- c(new_schema$names, new_name)
    new_schema$types <- c(new_schema$types, "FLOAT64")
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}

# Internal: Copy a column with a new name
#
# @param .data A tbl_gpu object
# @param new_name Name for the new column
# @param source_col Name of the column to copy
# @return A tbl_gpu with the copied column
# @keywords internal
mutate_copy_column <- function(.data, new_name, source_col) {
  source_idx <- col_index(.data, source_col)
  source_type <- .data$schema$types[source_idx + 1L]  # col_index returns 0-based

  # Check if we're replacing an existing column
  existing_idx <- match(new_name, .data$schema$names)

  if (!is.na(existing_idx)) {
    # Replacing existing column with a copy
    new_ptr <- gpu_copy_column_replace(.data$ptr, source_idx, existing_idx - 1L)
    new_schema <- .data$schema
    new_schema$types[existing_idx] <- source_type
  } else {
    # Adding new column as copy
    new_ptr <- gpu_copy_column(.data$ptr, source_idx)

    new_schema <- .data$schema
    new_schema$names <- c(new_schema$names, new_name)
    new_schema$types <- c(new_schema$types, source_type)
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}
