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
      new_ptr <- wrap_gpu_call(
        "mutate_binary_cols_replace",
        gpu_mutate_binary_cols_replace(.data$ptr, lhs_idx, op_found, rhs_idx, existing_idx - 1L)
      )
    } else {
      new_ptr <- wrap_gpu_call(
        "mutate_binary_cols",
        gpu_mutate_binary_cols(.data$ptr, lhs_idx, op_found, rhs_idx)
      )
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
      new_ptr <- wrap_gpu_call(
        "mutate_binary_scalar_replace",
        gpu_mutate_binary_scalar_replace(.data$ptr, lhs_idx, op_found, as.double(value), existing_idx - 1L)
      )
    } else {
      new_ptr <- wrap_gpu_call(
        "mutate_binary_scalar",
        gpu_mutate_binary_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
      )
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
    groups = .data$groups
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
    new_ptr <- wrap_gpu_call(
      "copy_column_replace",
      gpu_copy_column_replace(.data$ptr, source_idx, existing_idx - 1L)
    )
    new_schema <- .data$schema
    new_schema$types[existing_idx] <- source_type
  } else {
    # Adding new column as copy
    new_ptr <- wrap_gpu_call(
      "copy_column",
      gpu_copy_column(.data$ptr, source_idx)
    )

    new_schema <- .data$schema
    new_schema$names <- c(new_schema$names, new_name)
    new_schema$types <- c(new_schema$types, source_type)
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = .data$groups
  )
}
