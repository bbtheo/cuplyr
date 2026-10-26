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

  quos <- rlang::enquos(...)
  call <- rlang::current_env()
  sel <- resolve_selection(
    .data, schema, c(as.list(quos), list(before_quo, after_quo)), "relocate",
    function(data) {
      tidyselect::eval_relocate(
        expr = rlang::expr(c(!!!quos)), data = data,
        before = before_quo, after = after_quo,
        before_arg = ".before", after_arg = ".after", error_call = call
      )
    }
  )
  loc <- sel$loc
  .data <- sel$data

  if (length(schema$names) == 0L) {
    # nothing to move (a zero-column table keeps its row count as is); the
    # selection above was still validated
    return(.data)
  }

  out <- push_select_rename(.data, schema$names, loc)

  if (length(.subset2(.data, "groups")) > 0L) {
    out <- set_groups_by_position(out, .data, loc, schema$names)
  }
  if (is_rowwise_gpu(.data) && !is_rowwise_gpu(out)) {
    out <- rowwise_mark(out, character())  # (a rowwise table with duplicated names, no vars)
  }
  rowwise_column_order(out)
}
