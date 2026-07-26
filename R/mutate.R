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

  # Schema accumulates across dots within this call so a later expression can
  # reference an earlier one's output (e.g. `mutate(a = x + y, b = a * 2)`).
  # This mirrors what infer_schema.ast_mutate() will independently recompute
  # from the finished `expressions` list once push_op() calls infer_schema();
  # keeping a running local copy here is only needed to validate/parse later
  # dots against the right column set, not to set `.data$schema` directly
  # (push_op() owns that per D1/D3).
  schema <- current_schema(.data)
  expressions <- list()

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

    parsed <- parse_mutate_exprs(new_name, expr, schema)

    if (is.null(parsed)) {
      stop("mutate() only supports column copies or arithmetic operations: +, -, *, /, ^\n",
           "Expression: ", rlang::quo_text(expr), call. = FALSE)
    }

    expressions <- c(expressions, parsed)

    # Update the running schema so subsequent dots see this dot's output(s).
    for (expr_step in parsed) {
      schema <- update_schema_for_expr(schema, expr_step)
    }
  }

  # One ast_mutate node carries every expression from every dot (including
  # the synthetic intermediate steps a single dot may expand into via the
  # tree/linear-chain parsers below) -- the same shape mutate_lazy() used to
  # build, which fuse_mutates()/lower_mutate()'s gpu_mutate_batch() call
  # consume as a single fused execution.
  push_op(.data, ast_mutate(input_node(.data), expressions))
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

  # Validate LHS is a column. This is the last-resort parser in the
  # tree -> linear -> here fallback chain (parse_mutate_exprs()), so an
  # unknown LHS at this point is a diagnosable user error, not a genuinely
  # opaque expression -- stop() with the exact message the now-deleted
  # mutate_one() used to raise here.
  if (!lhs %in% schema$names) {
    stop("Column '", lhs, "' not found.\n",
         "Available columns: ", paste(schema$names, collapse = ", "),
         call. = FALSE)
  }

  lhs_type <- schema$types[match(lhs, schema$names)]

  # Check if RHS is a column
  if (rhs %in% schema$names) {
    rhs_type <- schema$types[match(rhs, schema$names)]
    return(make_mutate_expr(new_name, c(lhs, rhs), op_found,
                            input_types = c(lhs_type, rhs_type)))
  }

  # Try to parse RHS as scalar. Same rationale as above: once we know LHS is
  # a real column and an operator was found, a bad RHS is diagnosable, so we
  # stop() with mutate_one()'s exact former messages rather than returning
  # NULL (which would otherwise be misreported as "only supports column
  # copies or arithmetic operations").
  value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
    stop("Cannot parse value: ", rhs, call. = FALSE)
  })
  if (!is.numeric(value) || length(value) != 1) {
    stop("mutate() currently only supports numeric scalar operations.\n",
         "Got: ", class(value)[1], " of length ", length(value), call. = FALSE)
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
# Nested-expression node parsing (shared by parse_mutate_exprs_tree())
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
