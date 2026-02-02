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
#'   }
#'
#' @return A `tbl_gpu` object containing only the selected columns.
#'   Column order matches the order specified in the selection.
#'
#' @details
#' Column selection creates a new GPU table with only the selected columns.
#' The original data remains in GPU memory until garbage collected.
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
  # tidyselect needs a named vector
  name_vec <- setNames(.data$schema$names, .data$schema$names)
  vars <- tidyselect::eval_select(rlang::expr(c(...)), name_vec)

  if (length(vars) == 0) {
    stop("select() resulted in no columns.", call. = FALSE)
  }

  indices <- as.integer(vars) - 1L
  new_ptr <- wrap_gpu_call("select", gpu_select(.data$ptr, indices))

  new_schema <- list(
    names = .data$schema$names[vars],
    types = .data$schema$types[vars]
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = intersect(.data$groups, new_schema$names)
  )
}
