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
    stop("mutate() only supports arithmetic operations: +, -, *, /, ^\n",
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

  if (!is.null(rhs_idx)) {
    # Column to column operation
    new_ptr <- gpu_mutate_binary_cols(.data$ptr, lhs_idx, op_found, rhs_idx)
  } else {
    # Column to scalar operation
    value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
      stop("Cannot parse value: ", rhs, call. = FALSE)
    })
    if (!is.numeric(value) || length(value) != 1) {
      stop("mutate() currently only supports numeric scalar operations.\n",
           "Got: ", class(value)[1], " of length ", length(value), call. = FALSE)
    }
    new_ptr <- gpu_mutate_binary_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
  }

  # Check if we're replacing an existing column
  existing_idx <- match(new_name, .data$schema$names)

  if (!is.na(existing_idx)) {
    # Replace existing column: reorder to put new column in original position
    n_orig <- length(.data$schema$names)
    new_col_idx <- n_orig  # 0-based index of newly appended column

    # Build new column order: replace old column position with new column
    indices <- seq_len(n_orig) - 1L  # 0-based
    indices[existing_idx] <- new_col_idx
    indices <- indices[indices != new_col_idx | seq_along(indices) == existing_idx]

    new_ptr <- gpu_select(new_ptr, as.integer(indices))

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
