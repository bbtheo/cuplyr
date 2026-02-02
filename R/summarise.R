#' Summarise groups in a GPU table
#'
#' Computes aggregations on groups defined by [group_by()]. Operations are
#' performed entirely on the GPU for maximum performance.
#'
#' @param .data A grouped `tbl_gpu` object created by [group_by()].
#' @param ... Name-value pairs of summary functions. The name will be the
#'   name of the variable in the result. The value must be a single aggregation
#'   expression in the form `fun(column)`.
#' @param .groups Controls grouping structure of the result. Currently only
#'   "drop" is supported (default).
#'
#' @return A `tbl_gpu` object with one row per group containing the grouping
#'   columns and computed aggregations.
#'
#' @details
#' ## Supported aggregation functions
#' \itemize{
#'   \item `sum(x)` - Sum of values
#'   \item `mean(x)` - Arithmetic mean
#'   \item `min(x)` - Minimum value
#'   \item `max(x)` - Maximum value
#'   \item `n()` - Count of rows in each group
#'   \item `sd(x)` - Standard deviation
#'   \item `var(x)` - Variance
#' }
#'
#' ## NA handling
#' By default, NA values are excluded from aggregations. This matches
#' the default behavior of R's base aggregation functions.
#'
#' ## Ungrouped summarise
#' If `.data` is not grouped, summarise will compute aggregations over all
#' rows, returning a single-row table.
#'
#' @seealso
#' \code{\link{group_by.tbl_gpu}} for grouping data,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr summarise summarize
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Simple aggregation over all rows
#'   total <- gpu_mtcars |>
#'     summarise(avg_mpg = mean(mpg)) |>
#'     collect()
#'
#'   # Grouped aggregation
#'   by_cyl <- gpu_mtcars |>
#'     group_by(cyl) |>
#'     summarise(
#'       avg_mpg = mean(mpg),
#'       max_hp = max(hp),
#'       count = n()
#'     ) |>
#'     collect()
#'
#'   # Multiple grouping columns
#'   by_cyl_gear <- gpu_mtcars |>
#'     group_by(cyl, gear) |>
#'     summarise(
#'       mean_mpg = mean(mpg),
#'       min_wt = min(wt)
#'     ) |>
#'     collect()
#' }
summarise.tbl_gpu <- function(.data, ..., .groups = "drop") {

  dots <- rlang::enquos(...)

  if (length(dots) == 0) {
    stop("summarise() requires at least one aggregation expression.",
         call. = FALSE)
  }

  # Pre-process: create temporary columns for expressions inside agg functions
  preprocess_result <- preprocess_agg_expressions(.data, dots)
  working_data <- preprocess_result$data
  processed_dots <- preprocess_result$dots

  # Parse the aggregation expressions (now with simple column refs)
  agg_info <- parse_agg_expressions(working_data, processed_dots)

  # Get group column indices (0-based for C++)
  group_indices <- if (length(.data$groups) > 0) {
    match(.data$groups, working_data$schema$names) - 1L
  } else {
    integer(0)
  }

  # Call C++ function to perform the grouped aggregation
  result_ptr <- wrap_gpu_call(
    "summarise",
    gpu_summarise(
      working_data$ptr,
      group_indices,
      agg_info$col_indices,
      agg_info$agg_types
    )
  )

  # Build the result schema
  # First the group columns, then the aggregation results
  result_names <- c(.data$groups, names(dots))
  result_types <- c(
    .data$schema$types[match(.data$groups, .data$schema$names)],
    agg_info$result_types
  )

  new_tbl_gpu(
    ptr = result_ptr,
    schema = list(names = result_names, types = result_types),
    lazy_ops = list(),
    groups = character()  # Result is ungrouped
  )
}

#' @rdname summarise.tbl_gpu
#' @export
summarize.tbl_gpu <- summarise.tbl_gpu

# Internal: Pre-process aggregation expressions
#
# Detects expressions inside aggregation functions (e.g., sum(carb == 4))
# and creates temporary columns for them using mutate operations.
#
# @param .data A tbl_gpu object
# @param dots Quosures from summarise()
# @return List with modified data and simplified dots
# @keywords internal
preprocess_agg_expressions <- function(.data, dots) {
  working_data <- .data
  new_dots <- vector("list", length(dots))
  names(new_dots) <- names(dots)

  # Supported aggregation functions
  agg_functions <- c("sum", "mean", "min", "max", "n", "sd", "var", "count")

  # Counter for temporary column names
  temp_col_counter <- 0

  for (i in seq_along(dots)) {
    expr <- dots[[i]]
    expr_text <- rlang::quo_text(expr)

    # Check for n() which has no column argument
    if (grepl("^n\\(\\)$", trimws(expr_text))) {
      new_dots[[i]] <- expr
      next
    }

    # Parse function(argument) pattern
    match_result <- regmatches(
      expr_text,
      regexec("^([a-zA-Z_][a-zA-Z0-9_]*)\\((.+)\\)$", expr_text)
    )[[1]]

    if (length(match_result) != 3) {
      new_dots[[i]] <- expr
      next
    }

    func_name <- match_result[2]
    arg_text <- trimws(match_result[3])

    # Check if argument is a simple column name
    if (arg_text %in% working_data$schema$names) {
      # Simple column reference, keep as-is
      new_dots[[i]] <- expr
      next
    }

    # Argument is an expression - need to create a temporary column
    # Check if it contains comparison or arithmetic operators
    has_operator <- grepl("==|!=|>=|<=|>|<|\\+|-|\\*|/|\\^", arg_text)

    if (!has_operator) {
      # Not an expression we can handle, keep as-is (will error later if invalid)
      new_dots[[i]] <- expr
      next
    }

    # Create a temporary column name
    temp_col_counter <- temp_col_counter + 1
    temp_col_name <- paste0(".temp_agg_", temp_col_counter)

    # Parse the expression and create the temporary column via mutate
    # We need to handle comparison operators (==, !=, >, <, >=, <=)
    # and arithmetic operators (+, -, *, /, ^)
    working_data <- create_temp_column(working_data, temp_col_name, arg_text)

    # Create a new quosure with the temp column name
    new_expr_text <- paste0(func_name, "(", temp_col_name, ")")
    new_dots[[i]] <- rlang::parse_quo(new_expr_text, env = rlang::base_env())
  }

  list(data = working_data, dots = new_dots)
}

# Internal: Create a temporary column from an expression
#
# @param .data A tbl_gpu object
# @param col_name Name for the new column
# @param expr_text Expression text (e.g., "carb == 4")
# @return Modified tbl_gpu with new column
# @keywords internal
create_temp_column <- function(.data, col_name, expr_text) {
  # Detect comparison operators (order matters - check two-char first)
  compare_ops <- c("==", "!=", ">=", "<=", ">", "<")
  arith_ops <- c("+", "-", "*", "/", "^")

  op_found <- NULL
  op_type <- NULL

  # Check comparison operators first
  for (op in compare_ops) {
    if (grepl(op, expr_text, fixed = TRUE)) {
      op_found <- op
      op_type <- "compare"
      break
    }
  }

  # If no comparison op, check arithmetic
  if (is.null(op_found)) {
    for (op in arith_ops) {
      # For arithmetic, need to be careful with negative numbers
      # Use regex to find operator not at start
      pattern <- paste0("(?<!^)", gsub("([+*^])", "\\\\\\1", op))
      if (grepl(pattern, expr_text, perl = TRUE)) {
        op_found <- op
        op_type <- "arith"
        break
      }
    }
  }

  if (is.null(op_found)) {
    stop("Cannot parse expression: ", expr_text,
         "\nExpected a comparison or arithmetic expression.",
         call. = FALSE)
  }

  # Split on operator
  parts <- strsplit(expr_text, op_found, fixed = TRUE)[[1]]
  if (length(parts) != 2) {
    stop("Invalid expression: ", expr_text, call. = FALSE)
  }

  lhs <- trimws(parts[1])
  rhs <- trimws(parts[2])

  # Determine if lhs/rhs are columns or values
  lhs_is_col <- lhs %in% .data$schema$names
  rhs_is_col <- rhs %in% .data$schema$names

  if (!lhs_is_col && !rhs_is_col) {
    stop("Expression must reference at least one column: ", expr_text, call. = FALSE)
  }

  # Get column index (0-based)
  lhs_idx <- if (lhs_is_col) match(lhs, .data$schema$names) - 1L else NULL

  if (op_type == "compare") {
    # For comparison, result is boolean (0/1 for summing)
    if (rhs_is_col) {
      # Column to column comparison
      rhs_idx <- match(rhs, .data$schema$names) - 1L
      new_ptr <- wrap_gpu_call(
        "summarise_compare_cols",
        gpu_compare_cols(.data$ptr, lhs_idx, op_found, rhs_idx)
      )
    } else {
      # Column to scalar comparison
      value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
        stop("Cannot parse value: ", rhs, call. = FALSE)
      })
      new_ptr <- wrap_gpu_call(
        "summarise_compare_scalar",
        gpu_compare_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
      )
    }
    new_type <- "INT32"  # Boolean stored as int for summing
  } else {
    # Arithmetic operation
    if (rhs_is_col) {
      rhs_idx <- match(rhs, .data$schema$names) - 1L
      new_ptr <- wrap_gpu_call(
        "summarise_mutate_binary_cols",
        gpu_mutate_binary_cols(.data$ptr, lhs_idx, op_found, rhs_idx)
      )
    } else {
      value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
        stop("Cannot parse value: ", rhs, call. = FALSE)
      })
      new_ptr <- wrap_gpu_call(
        "summarise_mutate_binary_scalar",
        gpu_mutate_binary_scalar(.data$ptr, lhs_idx, op_found, as.double(value))
      )
    }
    new_type <- "FLOAT64"
  }

  # Create new tbl_gpu with added column
  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(
      names = c(.data$schema$names, col_name),
      types = c(.data$schema$types, new_type)
    ),
    lazy_ops = .data$lazy_ops,
    groups = .data$groups
  )
}

# Internal: Parse aggregation expressions
#
# Extracts column indices, aggregation types, and result types from
# summarise expressions.
#
# @param .data A tbl_gpu object
# @param dots Quosures from summarise()
# @return List with col_indices, agg_types, and result_types
# @keywords internal
parse_agg_expressions <- function(.data, dots) {
  col_indices <- integer(length(dots))
  agg_types <- character(length(dots))
  result_types <- character(length(dots))

  # Supported aggregation functions and their types
  agg_functions <- c("sum", "mean", "min", "max", "n", "sd", "var", "count")

  for (i in seq_along(dots)) {
    expr <- dots[[i]]
    expr_text <- rlang::quo_text(expr)

    # Check for n() which has no column argument
    if (grepl("^n\\(\\)$", trimws(expr_text))) {
      col_indices[i] <- 0L  # Use first column (ignored for count)
      agg_types[i] <- "n"
      result_types[i] <- "INT64"
      next
    }

    # Parse function(column) pattern
    match_result <- regmatches(
      expr_text,
      regexec("^([a-zA-Z_][a-zA-Z0-9_]*)\\(([^)]+)\\)$", expr_text)
    )[[1]]

    if (length(match_result) != 3) {
      stop("Invalid aggregation expression: ", expr_text,
           "\nExpected format: function(column), e.g., mean(mpg)",
           call. = FALSE)
    }

    func_name <- match_result[2]
    col_name <- trimws(match_result[3])

    # Validate function
    if (!func_name %in% agg_functions) {
      stop("Unsupported aggregation function: ", func_name,
           "\nSupported functions: ", paste(agg_functions, collapse = ", "),
           call. = FALSE)
    }

    # Validate column
    col_idx <- match(col_name, .data$schema$names)
    if (is.na(col_idx)) {
      stop("Column '", col_name, "' not found.",
           "\nAvailable columns: ", paste(.data$schema$names, collapse = ", "),
           call. = FALSE)
    }

    col_indices[i] <- col_idx - 1L  # Convert to 0-based

    # Map function to aggregation type
    agg_types[i] <- switch(func_name,
      "sum" = "sum",
      "mean" = "mean",
      "min" = "min",
      "max" = "max",
      "sd" = "std",
      "var" = "variance",
      "count" = "n",
      func_name
    )

    # Determine result type
    input_type <- .data$schema$types[col_idx]
    result_types[i] <- get_agg_result_type(agg_types[i], input_type)
  }

  list(
    col_indices = col_indices,
    agg_types = agg_types,
    result_types = result_types
  )
}

# Internal: Determine result type for an aggregation
#
# @param agg_type The aggregation type (sum, mean, etc.)
# @param input_type The input column type
# @return The expected result type
# @keywords internal
get_agg_result_type <- function(agg_type, input_type) {
  switch(agg_type,
    "sum" = {
      # Sum promotes integers to int64, keeps float64
      if (input_type %in% c("INT32", "INT64")) "INT64" else "FLOAT64"
    },
    "mean" = "FLOAT64",  # Mean is always float
    "min" = input_type,  # Min/max preserve type
    "max" = input_type,
    "n" = "INT64",       # Count is always int64
    "std" = "FLOAT64",   # Std dev is always float
    "variance" = "FLOAT64",  # Variance is always float
    "FLOAT64"  # Default to float64
  )
}
