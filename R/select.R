#' Select columns from a GPU table
#'
#' Keeps only the specified columns from a GPU table, similar to
#' `dplyr::select()`. Supports tidyselect syntax for flexible column selection.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names or tidyselect expressions specifying which columns
#'   to keep. Supports:
#'   \itemize{
#'     \item Column names: `select(x, y, z)`
#'     \item Negative selection: `select(-x)` (not yet supported)
#'     \item Range: `select(x:z)` (not yet supported)
#'     \item Helpers: `starts_with()`, `ends_with()`, `contains()`, etc.
#'     \item Renaming while selecting: `select(new_name = old_name)`
#'   }
#'
#' @return A `tbl_gpu` object containing only the selected columns.
#'   Column order matches the order specified in the selection; any
#'   `new = old` selections are renamed in the result.
#'
#' @details
#' Column selection creates a new GPU table with only the selected columns.
#' The original data remains in GPU memory until garbage collected.
#'
#' ## Renaming
#' `select(new_name = old_name)` both selects and renames `old_name`. This
#' is schema-only (see [rename.tbl_gpu()]): the underlying selection still
#' runs as one `gpu_select()` gather, with the rename applied on top of it
#' as a separate, schema-only [ast_rename()] step whenever the tidyselect
#' resolution renamed a column.
#'
#' ## Performance
#' Select operations involve copying column data to a new table structure.
#' For very wide tables, selecting fewer columns can significantly reduce
#' memory usage and improve performance of subsequent operations.
#'
#' @seealso
#' \code{\link{filter.tbl_gpu}} for filtering rows,
#' \code{\link{mutate.tbl_gpu}} for creating columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr select
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Select specific columns
#'   result <- gpu_mtcars |>
#'     select(mpg, cyl, hp) |>
#'     collect()
#'
#'   # Select with tidyselect helpers
#'   result <- gpu_mtcars |>
#'     select(starts_with("d")) |>
#'     collect()
#'
#'   # Reorder columns
#'   result <- gpu_mtcars |>
#'     select(hp, mpg, wt) |>
#'     collect()
#' }
select.tbl_gpu <- function(.data, ...) {
  # Get current schema (from AST or base)
  schema <- current_schema(.data)

  # tidyselect needs a named vector
  name_vec <- stats::setNames(schema$names, schema$names)
  vars <- tidyselect::eval_select(rlang::expr(c(...)), name_vec)

  if (length(vars) == 0) {
    stop("select() resulted in no columns.", call. = FALSE)
  }

  # `names(vars)` are the OUTPUT names tidyselect resolved -- for a plain
  # selection these are identical to the selected columns' own names, but
  # for `select(new = old)` they carry the rename. Uniqueness of the
  # renamed set is already validated by eval_select() itself (unlike
  # rename()'s partial-touch case, select()'s output IS the full result,
  # so there's no separate untouched-column collision to check -- see
  # R/rename.R's check_rename_unique() doc comment for the contrast).
  selected_cols <- schema$names[vars]
  output_names <- names(vars)

  select_node <- ast_select(input_node(.data), selected_cols)

  if (identical(selected_cols, output_names)) {
    return(push_op(.data, select_node))
  }

  changed <- selected_cols != output_names
  node <- ast_rename(select_node, selected_cols[changed], output_names[changed])
  push_op(.data, node)
}
