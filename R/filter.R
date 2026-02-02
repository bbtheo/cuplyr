#' Filter rows of a GPU table
#'
#' Selects rows from a GPU table where conditions are TRUE, similar to
#' `dplyr::filter()`. Filtering is performed entirely on the GPU for
#' maximum performance on large datasets.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Logical expressions to filter by. Each expression should be
#'   a comparison of the form `column <op> value` or `column <op> column`.
#'   Multiple conditions are combined with AND (all must be TRUE).
#' @param .preserve Ignored. Included for compatibility with dplyr generic.
#'
#' @return A `tbl_gpu` object containing only rows where all conditions are TRUE.
#'   The GPU memory for the filtered result is newly allocated.
#'
#' @details
#' ## Supported comparison operators
#' \itemize{
#'   \item `==` - equal to
#'   \item `!=` - not equal to
#'   \item `>` - greater than
#'   \item `>=` - greater than or equal to
#'   \item `<` - less than
#'   \item `<=` - less than or equal to
#' }
#'
#' ## Current limitations
#' \itemize{
#'   \item Only simple comparisons are supported (column op value/column)
#'   \item Compound expressions with `&` or `|` are not yet supported
#'   \item String comparisons are not yet implemented
#'   \item Only numeric scalar values on the right-hand side
#' }
#'
#' ## Performance
#' Filtering on GPU is highly parallel and can process billions of rows
#' per second. For best performance, chain multiple filter conditions
#' rather than using compound expressions.
#'
#' @seealso
#' \code{\link{mutate.tbl_gpu}} for creating new columns,
#' \code{\link{select.tbl_gpu}} for selecting columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr filter
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Filter with single condition
#'   efficient_cars <- gpu_mtcars |>
#'     filter(mpg > 25)
#'
#'   # Multiple conditions (combined with AND)
#'   result <- gpu_mtcars |>
#'     filter(mpg > 20) |>
#'     filter(cyl == 4) |>
#'     collect()
#'
#'   # Compare two columns
#'   gpu_cars <- tbl_gpu(cars)
#'   fast_stops <- gpu_cars |>
#'     filter(dist < speed) |>
#'     collect()
#' }
filter.tbl_gpu <- function(.data, ..., .preserve = FALSE) {
  dots <- rlang::enquos(...)

  if (length(dots) == 0) return(.data)

  result <- .data
  for (expr in dots) {
    result <- filter_one(result, expr)
  }

  result
}

# Internal: Parse and execute a single filter expression
#
# Parses a quosure containing a comparison expression and calls the
# appropriate GPU filter function.
#
# @param .data A tbl_gpu object
# @param expr A quosure with a comparison expression
# @return A filtered tbl_gpu object
# @keywords internal
filter_one <- function(.data, expr) {
  expr_chr <- rlang::quo_text(expr)

  # Try to evaluate the expression first to check for boolean literal/vector
  eval_result <- tryCatch({
    rlang::eval_tidy(expr)
  }, error = function(e) NULL)

  # Check if it's a logical value (TRUE/FALSE or logical vector)
  if (!is.null(eval_result) && is.logical(eval_result)) {
    return(filter_logical(.data, eval_result))
  }

  # Parse simple comparison: col op value or col op col
  # Order matters: check two-char operators before single-char
  ops <- c("==", "!=", ">=", "<=", ">", "<")
  op_found <- NULL
  for (op in ops) {
    if (grepl(op, expr_chr, fixed = TRUE)) {
      op_found <- op
      break
    }
  }

  if (is.null(op_found)) {
    stop("filter() only supports comparisons: ==, !=, >, >=, <, <=\n",
         "Or logical values: TRUE, FALSE, logical vectors\n",
         "Expression: ", expr_chr, call. = FALSE)
  }

  parts <- strsplit(expr_chr, op_found, fixed = TRUE)[[1]]
  if (length(parts) != 2) {
    stop("Invalid filter expression: ", expr_chr,
         "\nExpected format: column ", op_found, " value", call. = FALSE)
  }

  lhs <- trimws(parts[1])
  rhs <- trimws(parts[2])

  lhs_idx <- tryCatch(col_index(.data, lhs), error = function(e) NULL)
  rhs_idx <- tryCatch(col_index(.data, rhs), error = function(e) NULL)

  if (is.null(lhs_idx)) {
    stop("Column '", lhs, "' not found.\n",
         "Available columns: ", paste(.data$schema$names, collapse = ", "),
         call. = FALSE)
  }

  if (!is.null(rhs_idx)) {
    # Column to column comparison
    new_ptr <- wrap_gpu_call(
      "filter_col",
      gpu_filter_col(.data$ptr, lhs_idx, op_found, rhs_idx)
    )
  } else {
    # Column to scalar comparison
    value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
      stop("Cannot parse value: ", rhs, call. = FALSE)
    })
    if (!is.numeric(value) || length(value) != 1) {
      stop("filter() currently only supports numeric scalar comparisons.\n",
           "Got: ", class(value)[1], " of length ", length(value), call. = FALSE)
    }
    new_ptr <- wrap_gpu_call(
      "filter_scalar",
      gpu_filter_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
    )
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups
  )
}

# Internal: Filter by logical value or vector
#
# @param .data A tbl_gpu object
# @param logical_val A logical scalar or vector
# @return A filtered tbl_gpu object
# @keywords internal
filter_logical <- function(.data, logical_val) {
  n_rows <- dim(.data)[1]

  if (length(logical_val) == 1) {
    # Single boolean: TRUE keeps all rows, FALSE keeps none
    if (isTRUE(logical_val)) {
      new_ptr <- wrap_gpu_call("filter_bool_true", gpu_filter_bool(.data$ptr, TRUE))
    } else {
      new_ptr <- wrap_gpu_call("filter_bool_false", gpu_filter_bool(.data$ptr, FALSE))
    }
  } else {
    # Logical vector: use as mask
    if (length(logical_val) != n_rows) {
      stop("Logical vector length (", length(logical_val),
           ") must match number of rows (", n_rows, ")", call. = FALSE)
    }

    # Check for all TRUE or all FALSE (optimize common cases)
    if (all(logical_val, na.rm = TRUE) && !any(is.na(logical_val))) {
      new_ptr <- wrap_gpu_call("filter_bool_all_true", gpu_filter_bool(.data$ptr, TRUE))
    } else if (!any(logical_val, na.rm = TRUE)) {
      new_ptr <- wrap_gpu_call("filter_bool_all_false", gpu_filter_bool(.data$ptr, FALSE))
    } else {
      # Mixed: apply mask
      new_ptr <- wrap_gpu_call("filter_mask", gpu_filter_mask(.data$ptr, logical_val))
    }
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups
  )
}
