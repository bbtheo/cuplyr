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
#'   \item Only simple comparisons are parsed directly into a GPU predicate
#'     (column op value/column)
#'   \item String comparisons are not parsed directly into a GPU predicate
#'   \item Only numeric scalar values on the right-hand side are parsed
#'     directly into a GPU predicate
#' }
#'
#' ## `TRUE`/`FALSE` literals
#' `filter(TRUE)` is a no-op: it returns `.data` unchanged (no GPU work is
#' performed). `filter(FALSE)` returns an empty (zero-row) result.
#'
#' ## Expressions that aren't a simple comparison (CPU fallback)
#' Compound expressions with `&`/`|`, calls like `between()` or `%in%`, and
#' other shapes that don't match `column op value`/`column op column`
#' still are not parsed into a GPU predicate at the parser level. Such an
#' expression is handed to a CPU-side fallback: it's evaluated with
#' `rlang::eval_tidy()` and, if the result is a logical scalar or vector,
#' applied back to the GPU table as a boolean mask (materializing first,
#' via `compute()`, if `.data` is lazy with pending operations). Because
#' this evaluation happens with *no data mask*, it can only succeed for
#' expressions that evaluate to a logical value on their own without
#' referencing table columns -- e.g. a pre-computed logical vector or
#' `filter(rep(TRUE, n))`. An expression that references a table column
#' (e.g. `filter(mpg > 20 & cyl == 4)` or `filter(cyl %in% c(4, 6))`) will
#' still error, now with the "only supports comparisons" message, since the
#' column symbol can't resolve outside the table. This fallback path is
#' legitimate, supported behavior for the cases it does cover, not an
#' error condition -- see `getOption("cuplyr.fallback")` below to be
#' notified when it's used.
#'
#' ## `options(cuplyr.fallback = ...)`
#' Controls whether the CPU fallback described above (used by `filter()`
#' and by `summarise()`'s temp-column preprocessing) notifies the caller.
#' One of:
#' \itemize{
#'   \item `"silent"` (default) - no notification
#'   \item `"warn"` - emits a `warning()` naming the verb and expression
#'   \item `"error"` - `stop()`s instead of falling back
#' }
#'
#' ## Performance
#' Filtering on GPU is highly parallel and can process billions of rows
#' per second. For best performance, prefer simple comparisons (parsed
#' directly into a GPU predicate) over expressions that require the CPU
#' fallback.
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

  # Schema is stable across this whole call: filter() never adds/removes
  # columns, so a single current_schema() snapshot is valid for every dot,
  # even across an intervening eval-mask fallback materialization.
  schema <- current_schema(.data)
  predicates <- list()

  for (quo in dots) {
    parsed <- parse_filter_expr(quo, schema)

    if (is.null(parsed)) {
      # Opaque expression (not a comparison shape): flush any predicates
      # accumulated so far into a real ast_filter node, then fall back to
      # CPU-side evaluation of this expression (D4).
      if (length(predicates) > 0) {
        .data <- push_op(.data, ast_filter(input_node(.data), predicates))
        predicates <- list()
      }
      .data <- filter_eval_mask(.data, quo)
    } else {
      # Comparison predicate(s) (including the no-op TRUE case, which
      # contributes an empty list, and the impossible FALSE predicate):
      # accumulate into a single ast_filter node's predicate list, the
      # same structure filter_lazy() used to build.
      predicates <- c(predicates, parsed)
    }
  }

  if (length(predicates) > 0) {
    .data <- push_op(.data, ast_filter(input_node(.data), predicates))
  }

  .data
}

# Parse a filter expression into predicate structure(s)
#
# Shared by eager and lazy modes (D1/S4). Returns:
#   - list() for a literal TRUE (no filter needed)
#   - list(<predicate>) for a literal FALSE (impossible predicate) or a
#     recognized `column <op> value`/`column <op> column` comparison
#   - NULL if the expression is not a comparison shape (caller falls back
#     to filter_eval_mask())
# stop()s for the two diagnosable cases that used to live in filter_one():
# an unknown LHS column, and a non-numeric/non-scalar RHS value -- both
# with their exact original message text.
#
# @param quo A quosure containing a filter expression
# @param schema The current schema (names/types) to validate against
# @return A list of predicate structures, or NULL
# @keywords internal
parse_filter_expr <- function(quo, schema) {
  expr_chr <- rlang::quo_text(quo)
  expr_obj <- rlang::quo_get_expr(quo)

  # Handle boolean literals without evaluating in a data mask
  if (identical(expr_obj, TRUE)) {
    return(list())  # TRUE = no filter needed
  }
  if (identical(expr_obj, FALSE)) {
    # FALSE = filter everything - create impossible predicate
    if (length(schema$names) > 0) {
      return(list(make_predicate(schema$names[1], "!=", schema$names[1],
                                 is_col_compare = TRUE)))
    }
    return(list())
  }

  # Parse comparison expression
  ops <- c("==", "!=", ">=", "<=", ">", "<")
  op_found <- NULL

  for (op in ops) {
    if (grepl(op, expr_chr, fixed = TRUE)) {
      op_found <- op
      break
    }
  }

  if (is.null(op_found)) {
    return(NULL)  # Opaque expression
  }

  parts <- strsplit(expr_chr, op_found, fixed = TRUE)[[1]]
  if (length(parts) != 2) {
    return(NULL)
  }

  lhs <- trimws(parts[1])
  rhs <- trimws(parts[2])

  # Validate LHS is a column (diagnosable case moved in from filter_one())
  if (!lhs %in% schema$names) {
    stop("Column '", lhs, "' not found.\n",
         "Available columns: ", paste(schema$names, collapse = ", "),
         call. = FALSE)
  }

  # Check if RHS is a column
  if (rhs %in% schema$names) {
    return(list(make_predicate(lhs, op_found, rhs, is_col_compare = TRUE)))
  }

  # Try to parse RHS as value
  value <- tryCatch(eval(parse(text = rhs)), error = function(e) NULL)
  if (is.null(value)) {
    return(NULL)  # Can't parse RHS at all - opaque, fall back
  }
  if (!is.numeric(value) || length(value) != 1) {
    # Diagnosable case moved in from filter_one()
    stop("filter() currently only supports numeric scalar comparisons.\n",
         "Got: ", class(value)[1], " of length ", length(value), call. = FALSE)
  }

  list(make_predicate(lhs, op_found, value, is_col_compare = FALSE))
}

# Internal: CPU-eval fallback for filter expressions that aren't a
# recognized comparison shape.
#
# Materializes any pending lazy operations first (so the fallback always
# operates on a real GPU table), eval_tidy()s the quosure (no explicit data
# mask - matches the historical behavior of only supporting expressions
# that evaluate directly, e.g. `rep(TRUE, n)`, not column references),
# validates the result is logical, and applies it via gpu_filter_bool()
# (scalar/uniform case) or gpu_filter_mask() (mixed mask), same as the
# renamed-from filter_logical() did. Notifies via cuplyr_fallback_notify()
# (D4: silent by default).
#
# @param .data A tbl_gpu object
# @param quo A quosure whose evaluated result must be logical
# @return A filtered tbl_gpu object
# @keywords internal
filter_eval_mask <- function(.data, quo) {
  if (identical(.data$exec_mode, "lazy") && has_pending_ops(.data)) {
    .data <- compute(.data)
  }

  expr_chr <- rlang::quo_text(quo)
  eval_result <- tryCatch(rlang::eval_tidy(quo), error = function(e) NULL)

  if (is.null(eval_result) || !is.logical(eval_result)) {
    stop("filter() only supports comparisons: ==, !=, >, >=, <, <=\n",
         "Or logical values: TRUE, FALSE, logical vectors\n",
         "Expression: ", expr_chr, call. = FALSE)
  }

  cuplyr_fallback_notify("filter", expr_chr)

  n_rows <- dim(.data)[1]

  if (length(eval_result) == 1) {
    # Single boolean: TRUE keeps all rows, FALSE keeps none
    if (isTRUE(eval_result)) {
      new_ptr <- wrap_gpu_call("filter_bool_true", gpu_filter_bool(.data$ptr, TRUE))
    } else {
      new_ptr <- wrap_gpu_call("filter_bool_false", gpu_filter_bool(.data$ptr, FALSE))
    }
  } else {
    # Logical vector: use as mask
    if (length(eval_result) != n_rows) {
      stop("Logical vector length (", length(eval_result),
           ") must match number of rows (", n_rows, ")", call. = FALSE)
    }

    # Check for all TRUE or all FALSE (optimize common cases)
    if (all(eval_result, na.rm = TRUE) && !any(is.na(eval_result))) {
      new_ptr <- wrap_gpu_call("filter_bool_all_true", gpu_filter_bool(.data$ptr, TRUE))
    } else if (!any(eval_result, na.rm = TRUE)) {
      new_ptr <- wrap_gpu_call("filter_bool_all_false", gpu_filter_bool(.data$ptr, FALSE))
    } else {
      # Mixed: apply mask
      new_ptr <- wrap_gpu_call("filter_mask", gpu_filter_mask(.data$ptr, eval_result))
    }
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}
