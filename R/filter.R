# Filter verb for tbl_gpu

#' Filter rows of a GPU table
#'
#' @param .data A tbl_gpu object
#' @param ... Filter expressions (e.g., x > 0, y == 5)
#' @param .preserve Ignored (for compatibility)
#' @return A filtered tbl_gpu
#' @export
#' @importFrom dplyr filter
#' @examples
#' if (interactive()) {
#'   gpu_df <- tbl_gpu(mtcars)
#'   gpu_df |> filter(mpg > 20)
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

# Parse and execute a single filter expression
filter_one <- function(.data, expr) {
  expr_chr <- rlang::quo_text(expr)

  # Parse simple comparison: col op value or col op col
  # Supported: ==, !=, >, >=, <, <=
  ops <- c("==", "!=", ">=", "<=", ">", "<")
  op_found <- NULL
  for (op in ops) {
    if (grepl(op, expr_chr, fixed = TRUE)) {
      op_found <- op
      break
    }
  }

  if (is.null(op_found)) {
    stop("filter() only supports comparisons: ==, !=, >, >=, <, <=")
  }

  parts <- strsplit(expr_chr, op_found, fixed = TRUE)[[1]]
  if (length(parts) != 2) {
    stop("Invalid filter expression: ", expr_chr)
  }

  lhs <- trimws(parts[1])
  rhs <- trimws(parts[2])

  lhs_idx <- tryCatch(col_index(.data, lhs), error = function(e) NULL)
  rhs_idx <- tryCatch(col_index(.data, rhs), error = function(e) NULL)

  if (is.null(lhs_idx)) {
    stop("Column '", lhs, "' not found in filter expression")
  }

  if (!is.null(rhs_idx)) {
    # Column to column comparison
    new_ptr <- gpu_filter_col(.data$ptr, lhs_idx, op_found, rhs_idx)
  } else {
    # Column to scalar comparison
    value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
      stop("Cannot parse value: ", rhs)
    })
    if (!is.numeric(value) || length(value) != 1) {
      stop("filter() currently only supports numeric scalar comparisons")
    }
    new_ptr <- gpu_filter_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups
  )
}
