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
#' - `NA` values are placed last for ascending order
#' - `NA` values are placed first for descending order
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
#' Note: Character sorting uses binary/UTF-8 ordering, not locale-aware collation.
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
arrange.tbl_gpu <- function(.data, ..., .by_group = FALSE) {
  dots <- rlang::enquos(...)


  if (length(dots) == 0) {
    return(.data)
  }

  # Parse sort specifications
  sort_specs <- lapply(dots, parse_arrange_expr, .data = .data)

  col_indices <- integer(length(sort_specs))
  descending <- logical(length(sort_specs))

  for (i in seq_along(sort_specs)) {
    spec <- sort_specs[[i]]
    col_idx <- match(spec$col_name, .data$schema$names)
    if (is.na(col_idx)) {
      stop("Column '", spec$col_name, "' not found.\n",
           "Available columns: ", paste(.data$schema$names, collapse = ", "),
           call. = FALSE)
    }
    col_indices[i] <- col_idx - 1L  # 0-indexed for C++
    descending[i] <- spec$descending
  }

  # Handle grouped arrange (.by_group = TRUE prepends group columns)
  if (isTRUE(.by_group) && length(.data$groups) > 0) {
    group_indices <- match(.data$groups, .data$schema$names) - 1L

    # Check if user explicitly specified ordering for any group columns
    user_col_names <- vapply(sort_specs, `[[`, character(1), "col_name")

    # Build descending vector for group columns, respecting user's explicit ordering
    group_descending <- logical(length(.data$groups))
    for (i in seq_along(.data$groups)) {
      grp <- .data$groups[i]
      user_idx <- match(grp, user_col_names)
      if (!is.na(user_idx)) {
        # User explicitly specified this group column - use their ordering
        group_descending[i] <- sort_specs[[user_idx]]$descending
      }
      # Otherwise default FALSE (ascending) is already set
    }

    # Remove user columns that are group columns (they're handled above)
    keep <- !user_col_names %in% .data$groups

    col_indices <- c(group_indices, col_indices[keep])
    descending <- c(group_descending, descending[keep])
  }

  new_ptr <- wrap_gpu_call(
    "arrange",
    gpu_arrange(.data$ptr, col_indices, descending)
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups
  )
}

# Internal: Parse a single arrange expression
#
# Extracts column name and sort direction from a quosure.
# Supports: bare column names, desc(column), and -column.
#
# @param quo A quosure containing an arrange expression
# @param .data The tbl_gpu object (for context, currently unused)
# @return A list with `col_name` (character) and `descending` (logical)
# @keywords internal
parse_arrange_expr <- function(quo, .data) {
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
