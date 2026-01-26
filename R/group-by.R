#' Group a GPU table by one or more columns
#'
#' Marks columns to group by for subsequent operations like [summarise()].
#' The grouping is stored as metadata and does not perform any computation
#' until an aggregation is requested.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names to group by. Can be unquoted column names or
#'   tidyselect expressions.
#' @param .add If `FALSE` (default), will override existing groups. If `TRUE`,
#'   will add to existing groups.
#' @param .drop Ignored. Included for compatibility with dplyr generic.
#'
#' @return A grouped `tbl_gpu` object. The object has the same data but with
#'   grouping columns recorded for use by [summarise()].
#'
#' @details
#' Unlike operations like [filter()] or [mutate()], `group_by()` does not
#' perform any GPU computation. It simply records which columns should be
#' used for grouping in subsequent aggregation operations.
#'
#' The actual groupby computation happens when you call [summarise()] on the
#' grouped table. This lazy approach allows you to chain multiple operations
#' before executing the expensive groupby operation.
#'
#' @seealso
#' \code{\link{summarise.tbl_gpu}} for aggregating grouped data,
#' \code{\link{ungroup.tbl_gpu}} for removing grouping
#'
#' @export
#' @importFrom dplyr group_by
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Group by a single column
#'   by_cyl <- gpu_mtcars |>
#'     group_by(cyl)
#'
#'   # Group by multiple columns
#'   by_cyl_gear <- gpu_mtcars |>
#'     group_by(cyl, gear)
#'
#'   # Use with summarise for aggregation
#'   result <- gpu_mtcars |>
#'     group_by(cyl) |>
#'     summarise(mean_mpg = mean(mpg)) |>
#'     collect()
#' }
group_by.tbl_gpu <- function(.data, ..., .add = FALSE, .drop = TRUE) {
  dots <- rlang::enquos(...)

  if (length(dots) == 0) {
    return(.data)
  }

  # Parse column names from the expressions
  new_groups <- parse_group_cols(.data, dots)

  # Validate all columns exist
  missing <- setdiff(new_groups, .data$schema$names)
  if (length(missing) > 0) {
    stop("Column(s) not found: ", paste(missing, collapse = ", "),
         "\nAvailable columns: ", paste(.data$schema$names, collapse = ", "),
         call. = FALSE)
  }

  # Combine with existing groups if .add = TRUE
  if (.add && length(.data$groups) > 0) {
    new_groups <- unique(c(.data$groups, new_groups))
  }

  # Create new tbl_gpu with updated groups
  new_tbl_gpu(
    ptr = .data$ptr,
    schema = .data$schema,
    lazy_ops = .data$lazy_ops,
    groups = new_groups
  )
}

#' Remove grouping from a GPU table
#'
#' Removes all grouping information from a grouped `tbl_gpu` object.
#'
#' @param x A `tbl_gpu` object.
#' @param ... Ignored. Included for compatibility with dplyr generic.
#'
#' @return An ungrouped `tbl_gpu` object.
#'
#' @export
#' @importFrom dplyr ungroup
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   grouped <- gpu_mtcars |>
#'     group_by(cyl)
#'
#'   # Remove grouping
#'   ungrouped <- grouped |>
#'     ungroup()
#'
#'   # Verify groups are removed
#'   length(group_vars(ungrouped))  # 0
#' }
ungroup.tbl_gpu <- function(x, ...) {
  new_tbl_gpu(
    ptr = x$ptr,
    schema = x$schema,
    lazy_ops = x$lazy_ops,
    groups = character()
  )
}

#' Get grouping variables from a GPU table
#'
#' Returns the names of columns used for grouping.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A character vector of grouping column names.
#'
#' @export
#' @importFrom dplyr group_vars
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   grouped <- gpu_mtcars |>
#'     group_by(cyl, gear)
#'
#'   group_vars(grouped)  # c("cyl", "gear")
#' }
group_vars.tbl_gpu <- function(x) {
  x$groups
}

#' Get grouping information from a GPU table
#'
#' Returns a list of symbols representing the grouping columns.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A list of symbols for the grouping columns.
#'
#' @export
#' @importFrom dplyr groups
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   grouped <- gpu_mtcars |>
#'     group_by(cyl)
#'
#'   groups(grouped)  # list(as.symbol("cyl"))
#' }
groups.tbl_gpu <- function(x) {
  lapply(x$groups, as.symbol)
}

# Internal: Parse column names from group_by expressions
#
# Handles both bare column names and tidyselect expressions
#
# @param .data A tbl_gpu object
# @param dots Quosures from group_by()
# @return Character vector of column names
# @keywords internal
parse_group_cols <- function(.data, dots) {
  col_names <- character()

  for (expr in dots) {
    expr_text <- rlang::quo_text(expr)

    # Check if it's a simple column name
    if (expr_text %in% .data$schema$names) {
      col_names <- c(col_names, expr_text)
    } else {
      # Try tidyselect evaluation
      tryCatch({
        # Create a named vector for tidyselect
        col_positions <- stats::setNames(
          seq_along(.data$schema$names),
          .data$schema$names
        )
        selected <- tidyselect::eval_select(expr, data = col_positions)
        col_names <- c(col_names, names(selected))
      }, error = function(e) {
        # If tidyselect fails, assume it's a column name
        col_names <<- c(col_names, expr_text)
      })
    }
  }

  unique(col_names)
}
