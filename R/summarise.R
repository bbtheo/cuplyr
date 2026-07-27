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
  # (e.g. sum(carb == 4)). Per D4, if a temp column is needed and .data is
  # lazy with pending ops, this materializes (compute()) before creating the
  # temp column and notifies via cuplyr_fallback_notify().
  preprocess_result <- preprocess_agg_expressions(.data, dots)
  working_data <- preprocess_result$data
  processed_dots <- preprocess_result$dots

  # Parse the aggregation expressions (now with simple column refs), emitting
  # aggregation structs that carry the cudf-accepted function name (fixes
  # lazy sd()/var(), which used to reach C++ as "sd"/"var" and error out).
  aggregations <- parse_aggregations(current_schema(working_data), processed_dots)

  push_op(working_data, ast_summarise(input_node(working_data), aggregations,
                                      working_data$groups))
}

#' @rdname summarise.tbl_gpu
#' @export
summarize.tbl_gpu <- summarise.tbl_gpu

# Internal: Strip a leading `pkg::` qualifier from an aggregation call's
# text (e.g. `dplyr::n()` -> `n()`, `stats::sd(mpg)` -> `sd(mpg)`), so the
# shape-matching regexes below (and in parse_aggregations()) see the bare
# call regardless of how the user namespaced it.
#
# @param expr_text Character, expression text from rlang::quo_text()
# @return Character, with any leading `identifier::` removed
# @keywords internal
strip_ns_prefix <- function(expr_text) {
  sub("^[a-zA-Z_][a-zA-Z0-9_.]*::", "", trimws(expr_text))
}

# Internal: Pre-process aggregation expressions
#
# Detects expressions inside aggregation functions (e.g., sum(carb == 4))
# and creates temporary columns for them using mutate operations.
#
# Per D4: if a dot needs a temp column and `.data` is lazy with pending
# operations, this materializes (`compute()`) once, up front, before
# building any temp column, and notifies via `cuplyr_fallback_notify()` --
# `create_temp_column()` always operates on a materialized GPU pointer
# rather than reaching into an in-progress AST.
#
# @param .data A tbl_gpu object
# @param dots Quosures from summarise()
# @return List with modified data and simplified dots
# @keywords internal
preprocess_agg_expressions <- function(.data, dots) {
  working_data <- .data
  new_dots <- vector("list", length(dots))
  names(new_dots) <- names(dots)

  for (i in seq_along(dots)) {
    expr <- dots[[i]]
    expr_text <- rlang::quo_text(expr)
    match_text <- strip_ns_prefix(expr_text)

    # Check for n() which has no column argument
    if (grepl("^n\\(\\)$", match_text)) {
      new_dots[[i]] <- expr
      next
    }

    # Parse function(argument) pattern
    match_result <- regmatches(
      match_text,
      regexec("^([a-zA-Z_][a-zA-Z0-9_]*)\\((.+)\\)$", match_text)
    )[[1]]

    if (length(match_result) != 3) {
      new_dots[[i]] <- expr
      next
    }

    func_name <- match_result[2]
    arg_text <- trimws(match_result[3])

    # Check if argument is a simple column name (a pure schema lookup: safe
    # to call even while lazy ops are still pending, no GPU work involved).
    if (arg_text %in% current_schema(working_data)$names) {
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

    # Materialize before building the temp column if lazy ops are pending.
    if (has_pending_ops(working_data)) {
      working_data <- compute(working_data)
      cuplyr_fallback_notify("summarise", expr_text)
    }

    # Create a temporary column name (index-based, so it stays unique even
    # though not every dot ends up needing one).
    temp_col_name <- paste0(".temp_agg_", i)

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
    # Arithmetic operation. Routed through the same gpu_mutate_batch() path
    # as mutate() (D6): build a single make_mutate_expr() step and lower it
    # via one-expression batch call rather than the deleted
    # gpu_mutate_binary_cols()/gpu_mutate_binary_scalar() kernels.
    if (rhs_is_col) {
      rhs_idx <- match(rhs, .data$schema$names) - 1L
      expr_step <- make_mutate_expr(
        col_name, c(lhs, rhs), op_found,
        input_types = c(.data$schema$types[lhs_idx + 1L],
                         .data$schema$types[rhs_idx + 1L])
      )
    } else {
      value <- tryCatch(eval(parse(text = rhs)), error = function(e) {
        stop("Cannot parse value: ", rhs, call. = FALSE)
      })
      expr_step <- make_mutate_expr(
        col_name, lhs, op_found, scalar = as.double(value),
        input_types = .data$schema$types[lhs_idx + 1L]
      )
    }
    new_ptr <- wrap_gpu_call(
      "summarise_mutate_batch",
      gpu_mutate_batch(.data$ptr, list(expr_step), .data$schema)
    )
    new_type <- expr_step$output_type
  }

  # Create new tbl_gpu with added column
  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(
      names = c(.data$schema$names, col_name),
      types = c(.data$schema$types, new_type)
    ),
    lazy_ops = .data$lazy_ops,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}

# Internal: Parse aggregation expressions into make_aggregation() structs
#
# The one parser used by both eager and lazy scheduling (D1): given a
# schema and quosures already simplified by preprocess_agg_expressions()
# (so every dot's argument is either a bare column name or the no-arg
# n()), validates function/column names -- raising the same three
# diagnosable errors the pre-unification eager parser did, verbatim -- and
# emits aggregation structs (via make_aggregation(), which also folds in
# former get_agg_result_type()'s result-type inference) carrying the
# *cudf*-accepted function name. This is the fix for lazy sd()/var(),
# which used to reach C++ as "sd"/"var" (gpu_summarise()/get_groupby_agg()
# in src/ops_groupby.cpp only accept "std"/"variance") and error out with
# "Unknown aggregation type"; the eager path already translated these
# names and continues to do so here.
#
# A leading `pkg::` qualifier (`dplyr::n()`, `stats::sd(mpg)`, ...) is
# stripped before shape-matching so namespaced aggregation calls resolve
# the same way as their bare equivalents.
#
# @param schema Current schema (list(names=, types=))
# @param dots Quosures from summarise(), already temp-column-preprocessed
# @return List of aggregation structs (see make_aggregation())
# @keywords internal
parse_aggregations <- function(schema, dots) {
  # dplyr-facing aggregation function names (order matches the historical
  # error message from parse_agg_expressions()) and their cudf-accepted
  # equivalents (src/ops_groupby.cpp::get_groupby_agg()).
  agg_functions <- c("sum", "mean", "min", "max", "n", "sd", "var", "count")
  agg_fn_map <- c(
    sum = "sum", mean = "mean", min = "min", max = "max",
    n = "n", sd = "std", var = "variance", count = "n"
  )

  aggregations <- vector("list", length(dots))

  for (i in seq_along(dots)) {
    expr <- dots[[i]]
    expr_text <- rlang::quo_text(expr)
    match_text <- strip_ns_prefix(expr_text)

    # Check for n() which has no column argument
    if (grepl("^n\\(\\)$", match_text)) {
      aggregations[[i]] <- make_aggregation(names(dots)[i], NA_character_, "n")
      next
    }

    # Parse function(column) pattern
    match_result <- regmatches(
      match_text,
      regexec("^([a-zA-Z_][a-zA-Z0-9_]*)\\(([^)]+)\\)$", match_text)
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
    col_idx <- match(col_name, schema$names)
    if (is.na(col_idx)) {
      stop("Column '", col_name, "' not found.",
           "\nAvailable columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }

    input_type <- schema$types[col_idx]
    cudf_fn <- unname(agg_fn_map[func_name])

    aggregations[[i]] <- make_aggregation(names(dots)[i], col_name, cudf_fn, input_type)
  }

  aggregations
}
