# Mutate verb for tbl_gpu

#' Create or modify columns in a GPU table
#'
#' @param .data A tbl_gpu object
#' @param ... Name-value pairs of expressions (e.g., z = x + y)
#' @return A tbl_gpu with new/modified columns
#' @export
#' @importFrom dplyr mutate
#' @examples
#' if (interactive()) {
#'   gpu_df <- tbl_gpu(mtcars)
#'   gpu_df |> mutate(kpl = mpg * 0.425)
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

# Parse and execute a single mutate expression
mutate_one <- function(.data, new_name, expr) {
  expr_chr <- rlang::quo_text(expr)

  # Parse simple arithmetic: col op value or col op col
  # Supported: +, -, *, /, ^
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
    stop("mutate() only supports arithmetic: +, -, *, /, ^")
  }

  lhs <- trimws(substr(expr_chr, 1, op_pos - 1))
  rhs <- trimws(substr(expr_chr, op_pos + 1, nchar(expr_chr)))

  lhs_idx <- tryCatch(col_index(.data, lhs), error = function(e) NULL)
  rhs_idx <- tryCatch(col_index(.data, rhs), error = function(e) NULL)

  if (is.null(lhs_idx)) {
    stop("Column '", lhs, "' not found in mutate expression")
  }

  if (!is.null(rhs_idx)) {
    # Column to column operation
    new_ptr <- gpu_mutate_binary_cols(.data$ptr, lhs_idx, op_found, rhs_idx)
  } else {
    # Column to scalar operation
    value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
      stop("Cannot parse value: ", rhs)
    })
    if (!is.numeric(value) || length(value) != 1) {
      stop("mutate() currently only supports numeric scalar operations")
    }
    new_ptr <- gpu_mutate_binary_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
  }

  # Check if we're replacing an existing column
  existing_idx <- match(new_name, .data$schema$names)

  if (!is.na(existing_idx)) {
    # Replace existing column: select columns with new one in place of old
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
