#' Arrange rows of a GPU table by column values
#'
#' Orders the rows of a GPU table by the values of specified columns,
#' similar to `dplyr::arrange()`. Sorting is performed entirely on the GPU
#' using a memory-efficient two-phase algorithm.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names or expressions to sort by. Use `desc(column)` or
#'   `-column` for descending order. Multiple columns are sorted in order
#'   of precedence (first column is primary sort key).
#' @param .by_group If `TRUE` and `.data` is grouped, sort within groups by
#'   prepending group columns to the sort specification. Default is `FALSE`.
#' @param .locale The locale to sort character vectors in. `NULL` (the default)
#'   and `"C"` sort by byte order on the GPU; any other locale is sorted by
#'   dplyr on the CPU (via stringi) through the usual fallback.
#'
#' @return A `tbl_gpu` object with rows reordered. The GPU memory for the
#'   sorted result is newly allocated (approximately 2x table size peak memory).
#'
#' @details
#' ## Sort order
#' - Default is ascending order

#' - Use `desc(column)` or `-column` for descending order
#' - Multiple columns: first column is primary key, second is tiebreaker, etc.
#' - Sorting is stable: ties preserve their original relative order
#'
#' ## NA handling
#' - `NA` values are always placed last, for both ascending order and
#'   `desc()`/descending order, matching dplyr exactly (verified empirically
#'   against dplyr 1.2.1; fixed in Phase 6 -- older versions of this package
#'   placed `NA` first for descending sorts, a divergence from dplyr)
#'
#' ## Memory usage
#' The arrange operation requires approximately 2x the table size in GPU memory:
#' - Original table
#' - Sort indices (4 bytes per row)
#' - New sorted table
#'
#' For very large tables, consider filtering to reduce size before sorting.
#'
#' ## Supported column types
#' All column types supported by `tbl_gpu` can be sorted:
#' numeric, integer, character, logical, Date, POSIXct.
#'
#' Note: Character sorting on the GPU uses binary/UTF-8 ordering (dplyr's
#' `"C"` locale). A non-`"C"` `.locale` falls back to dplyr.
#'
#' @seealso
#' \code{\link{filter.tbl_gpu}} for filtering rows,
#' \code{\link{select.tbl_gpu}} for selecting columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr arrange desc
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Sort by single column (ascending)
#'   sorted <- gpu_mtcars |>
#'     arrange(mpg) |>
#'     collect()
#'
#'   # Sort descending
#'   sorted_desc <- gpu_mtcars |>
#'     arrange(desc(mpg)) |>
#'     collect()
#'
#'   # Multiple columns: primary and secondary sort keys
#'   sorted_multi <- gpu_mtcars |>
#'     arrange(cyl, desc(mpg)) |>
#'     collect()
#'
#'   # With grouped data
#'   grouped_sort <- gpu_mtcars |>
#'     group_by(cyl) |>
#'     arrange(mpg, .by_group = TRUE) |>
#'     collect()
#' }
arrange.tbl_gpu <- function(.data, ..., .by_group = FALSE, .locale = NULL) {
  dots <- rlang::enquos(...)
  if (length(dots) > 0L) check_unique_input_names(.data)

  # The GPU sort is byte order, i.e. dplyr's "C" locale. Any other `.locale`
  # (or an invalid one, which dplyr must reject, even with no dots) reruns the
  # whole call in dplyr, and so does the deprecated system-locale option, which
  # only applies while `.locale` is NULL
  legacy_locale <- is.null(.locale) && isTRUE(getOption("dplyr.legacy_locale"))
  if (legacy_locale || !(is.null(.locale) || identical(.locale, "C"))) {
    return(gpu_fallback("arrange", .data, function(tbl) {
      dplyr::arrange(tbl, !!!dots, .by_group = .by_group, .locale = .locale)
    }))
  }

  if (length(dots) == 0) {
    # `arrange(group_by(df, g), .by_group = TRUE)` still sorts by the groups
    if (isTRUE(.by_group) && length(.data$groups) > 0) {
      return(push_op(.data, ast_arrange(input_node(.data), list(), .data$groups)))
    }
    return(.data)
  }

  # Get current schema
  schema <- current_schema(.data)

  # Classify every dot first: `x` / `desc(x)` / `-x` and a resolvable
  # top-level pick() are native (pick() expands to one ascending spec per
  # column, selected on the full data); anything else is evaluated by dplyr.
  # Decided before any op is pushed, so the fallback reruns the whole call.
  sort_specs <- list()
  for (quo in dots) {
    picked <- resolve_pick_call(quo, schema)
    if (!is.null(picked)) {
      sort_specs <- c(sort_specs, lapply(picked, function(col) {
        list(col_name = col, descending = FALSE)
      }))
    } else if (arrange_dot_is_native(quo)) {
      sort_specs <- c(sort_specs, list(parse_arrange_expr(quo, .data = .data, schema = schema)))
    } else {
      return(gpu_fallback("arrange", .data, function(tbl) {
        dplyr::arrange(tbl, !!!dots, .by_group = .by_group, .locale = .locale)
      }))
    }
  }

  # `arrange(pick(NULL))` and friends select no column: nothing to sort by,
  # unless `.by_group = TRUE` (then the same node as the no-dots case)
  if (length(sort_specs) == 0 && !(isTRUE(.by_group) && length(.data$groups) > 0)) {
    return(.data)
  }

  # Validate columns exist
  for (spec in sort_specs) {
    if (!spec$col_name %in% schema$names) {
      stop("Column '", spec$col_name, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
  }

  # Handle .by_group: group columns are prepended ascending (D8, matches
  # dplyr's arrange.grouped_df); the user's own desc() on a group column
  # survives as a later no-op tiebreaker via lower_arrange()'s dedup logic.
  groups_for_arrange <- character()
  if (isTRUE(.by_group) && length(.data$groups) > 0) {
    groups_for_arrange <- .data$groups
  }

  push_op(.data, ast_arrange(input_node(.data), sort_specs, groups_for_arrange))
}

# Internal: is this dot one of the shapes parse_arrange_expr() handles
# natively -- `x`, `desc(x)` / `dplyr::desc(x)`, `-x` with a bare symbol?
# @keywords internal
arrange_dot_is_native <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  if (is.symbol(expr)) {
    return(TRUE)
  }
  if (!rlang::is_call(expr) || length(expr) != 2L || !is.symbol(expr[[2]])) {
    return(FALSE)
  }
  head <- expr[[1]]
  if (rlang::is_call(head, "::") && identical(head[[2]], quote(dplyr))) {
    head <- head[[3]]
  }
  is.symbol(head) && as.character(head) %in% c("desc", "-")
}

# Internal: Parse a single arrange expression
#
# Extracts column name and sort direction from a quosure.
# Supports: bare column names, desc(column), and -column.
#
# @param quo A quosure containing an arrange expression
# @param .data The tbl_gpu object (for context)
# @param schema The current schema (for lazy mode)
# @return A list with `col_name` (character) and `descending` (logical)
# @keywords internal
parse_arrange_expr <- function(quo, .data, schema = NULL) {
  if (is.null(schema)) {
    schema <- .data$schema
  }
  expr <- rlang::quo_get_expr(quo)

  if (is.symbol(expr)) {
    # Simple column: arrange(x)
    return(list(col_name = as.character(expr), descending = FALSE))
  }

  if (is.call(expr)) {
    fn_expr <- expr[[1]]
    # Handle both desc() and dplyr::desc() - namespaced calls return length > 1
    if (is.call(fn_expr) && identical(fn_expr[[1]], as.symbol("::"))) {
      fn <- as.character(fn_expr[[3]])
    } else {
      fn <- as.character(fn_expr)
    }

    if (fn == "desc") {
      # desc(x)
      if (length(expr) != 2) {
        stop("desc() requires exactly one argument", call. = FALSE)
      }
      inner <- expr[[2]]
      if (!is.symbol(inner)) {
        stop("desc() argument must be a column name, got: ",
             deparse(inner), call. = FALSE)
      }
      return(list(col_name = as.character(inner), descending = TRUE))
    }

    if (fn == "-") {
      # -x shorthand for desc(x) (dplyr compatibility)
      if (length(expr) != 2) {
        stop("Unary minus requires exactly one argument", call. = FALSE)
      }
      inner <- expr[[2]]
      if (!is.symbol(inner)) {
        stop("Unary minus must be applied to a column name, got: ",
             deparse(inner), call. = FALSE)
      }
      return(list(col_name = as.character(inner), descending = TRUE))
    }
  }

  stop("arrange() expressions must be column names or desc(column).\n",
       "Got: ", deparse(expr), call. = FALSE)
}
