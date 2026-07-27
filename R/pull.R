# pull() (Phase 3, task 3)
#
# Terminal verb: no AST node, since there is nothing further to build a
# pipeline on top of. Pending lazy ops are materialized via compute()
# first, then the whole table is collected (reusing collect.tbl_gpu()'s
# existing factor/INT64 handling rather than duplicating it for a single
# column) and the requested column is picked out in R.
#
# Semantics verified empirically against dplyr 1.2.1 before implementation
# (dplyr::pull.data.frame() is, verbatim: `tidyselect::vars_pull(names(.data),
# !!enquo(var))`, then the same for `name` when supplied -- reused here
# unchanged):
#   - var: by name (bare symbol/string), by position (including negative,
#     counting from the end), default `-1` (last column). Note `var`'s
#     data mask maps each name to its POSITION, not its value, so e.g.
#     `pull(df, x + 1)` means "one past x's position", not "x's values
#     plus 1".
#   - name=: another column (by the same var-resolution rules) whose
#     values become the result's names.
#   - Grouping is ignored entirely (`pull(group_by(df, g), x)` returns the
#     same plain vector as the ungrouped table, in original row order --
#     confirmed empirically, matches dplyr).

#' Pull a single column from a GPU table
#'
#' Extracts one column as a plain R vector, similar to `dplyr::pull()`.
#' Materializes any pending lazy operations first via [compute()], then
#' [collect()]s the table and picks out the requested column.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param var A variable, as a name, position (positive or negative,
#'   counting from the end), or expression evaluated against column
#'   *positions* (see `dplyr::pull()`'s own contract). Defaults to the
#'   last column.
#' @param name An optional column (by the same rules as `var`) whose
#'   values become the names of the returned vector.
#' @param ... Not used; for extensibility, must be empty.
#'
#' @return A plain R vector (not a `tbl_gpu`), named if `name` was
#'   supplied. Grouping (if any) is ignored, matching `dplyr::pull()`.
#'
#' @export
#' @importFrom dplyr pull
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   gpu_mtcars |> pull(mpg)
#' }
pull.tbl_gpu <- function(.data, var = -1, name = NULL, ...) {
  var_quo <- rlang::enquo(var)
  name_quo <- rlang::enquo(name)

  materialized <- if (has_pending_ops(.data)) compute(.data) else .data
  schema <- materialized$schema

  var_name <- rlang::inject(tidyselect::vars_pull(schema$names, !!var_quo))

  result_tbl <- collect(materialized)

  if (rlang::quo_is_null(name_quo)) {
    return(result_tbl[[var_name]])
  }

  name_col <- rlang::inject(tidyselect::vars_pull(schema$names, !!name_quo))
  stats::setNames(result_tbl[[var_name]], as.character(result_tbl[[name_col]]))
}
