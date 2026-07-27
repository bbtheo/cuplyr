# GPU-native relocate() (Phase 3, task 3)
#
# Column reorder, so it reuses the existing ast_select node (no new node
# type needed for that part); relocate() can ALSO rename while relocating
# (e.g. relocate(df, a = x) -- verified empirically against dplyr 1.2.1,
# and mirrors dplyr's own relocate.data.frame(), which is built on the same
# tidyselect::eval_relocate(..., allow_rename = TRUE) call used here), so
# when the resolved positions carry a rename, an ast_rename node (R/ast.R,
# R/rename.R) is stacked on top -- same pattern select.tbl_gpu() uses for
# its own `new = old` rename fix (R/select.R).
#
# Semantics verified empirically before implementation:
#   - Default (neither .before nor .after given): moves selected columns to
#     the front.
#   - .before=/.after=: moves selected columns to just before/after the
#     target; both given at once errors ("Can't supply both `.before` and
#     `.after`.").
#   - tidyselect helpers (incl. everything()) work in all three slots
#     (the `...` selection, `.before`, and `.after`).
#   - Relocating a group column preserves it in $groups at its new
#     position (position doesn't affect group membership); renaming one
#     while relocating updates group_vars() (via the same ast_rename
#     translation used by rename()).

#' Reorder columns of a GPU table
#'
#' Moves columns of a GPU table to a new position, similar to
#' `dplyr::relocate()`. The reorder itself runs via [ast_select()] (schema
#' + a `gpu_select()` gather, no new C++ binding needed); a rename
#' (`relocate(df, a = x)`) is schema-only, like [rename.tbl_gpu()].
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Columns to move (tidyselect syntax; may include `new = old`
#'   renames).
#' @param .before,.after Destination, as a tidyselect specification of an
#'   existing column; the selected columns are placed immediately before/
#'   after it. At most one of `.before`/`.after` may be supplied; if
#'   neither is given, selected columns move to the front.
#'
#' @return A `tbl_gpu` object with columns reordered (and any requested
#'   columns renamed).
#'
#' @export
#' @importFrom dplyr relocate
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   result <- gpu_mtcars |>
#'     relocate(hp, .after = mpg) |>
#'     collect()
#' }
relocate.tbl_gpu <- function(.data, ..., .before = NULL, .after = NULL) {
  before_quo <- rlang::enquo(.before)
  after_quo <- rlang::enquo(.after)

  schema <- current_schema(.data)
  name_vec <- stats::setNames(schema$names, schema$names)

  loc <- tidyselect::eval_relocate(
    expr = rlang::expr(c(...)), data = name_vec,
    before = before_quo, after = after_quo,
    before_arg = ".before", after_arg = ".after"
  )

  reordered_cols <- schema$names[loc]
  output_names <- names(loc)

  select_node <- ast_select(input_node(.data), reordered_cols)

  if (identical(reordered_cols, output_names)) {
    return(push_op(.data, select_node))
  }

  changed <- reordered_cols != output_names
  node <- ast_rename(select_node, reordered_cols[changed], output_names[changed])
  push_op(.data, node)
}
