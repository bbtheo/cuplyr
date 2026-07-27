# GPU-native rename()/rename_with() (Phase 3, task 3)
#
# Both verbs are pure schema operations: renaming never touches GPU column
# data (column names live entirely in the R-side schema, see ast_rename()'s
# doc comment in R/ast.R), so both build a single ast_rename node and hand
# it to push_op() -- no C++ binding needed at all, matching the "schema-only
# batch" scope for this task (see scratchpad/todo.md Phase 3).
#
# Semantics verified empirically against dplyr 1.2.1 before implementation:
#   - rename(df, new = old): only names change, positions/values untouched;
#     renaming a grouped column updates group_vars(); tidyselect-style
#     rename(df, all_of(c(new = "old"))) works (same tidyselect::eval_rename()
#     dplyr's own rename.data.frame() uses).
#   - Renaming to a name that collides with an EXISTING, un-renamed column
#     errors ("Names must be unique.") -- but a same-call swap
#     (rename(df, x = y, y = x)) is valid, since both original names are
#     replaced simultaneously. tidyselect::eval_rename() itself only
#     catches collisions among the renamed dots (e.g. two dots targeting the
#     same new name); it does NOT check the renamed names against
#     surviving un-renamed columns (verified: eval_rename(c(y = x), ...)
#     alone raises nothing even when the input has an untouched `y`), so
#     check_rename_unique() below does that final check explicitly, exactly
#     the way dplyr::rename_with.data.frame() does with
#     vctrs::vec_as_names(repair = "check_unique").
#   - rename_with(df, .fn, .cols=, ...): .fn is applied ONCE to the vector
#     of selected (old) names (not per-name), and must return a same-length
#     character vector; duplicate results in the final (renamed + untouched)
#     name set error the same way. `.cols` selection does not itself permit
#     renaming (`allow_rename = FALSE`, matching dplyr's own contract).

# Check that a full (renamed + untouched) name vector has no duplicates,
# throwing the same vctrs "Names must be unique." error dplyr's own
# rename()/rename_with() raise (they both ultimately go through
# vctrs::vec_as_names(repair = "check_unique")).
#
# @param names Character vector of the full post-rename column name set
# @return `invisible(NULL)`, or throws
# @keywords internal
check_rename_unique <- function(names) {
  vctrs::vec_as_names(names, repair = "check_unique")
  invisible(NULL)
}

#' Rename columns of a GPU table
#'
#' Renames columns of a GPU table, similar to `dplyr::rename()`. Runs
#' entirely as a schema-only operation -- no GPU work at all, since column
#' names are R-side metadata, never part of the underlying GPU table data.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Use `new_name = old_name` to rename columns. Supports
#'   tidyselect renaming helpers, e.g. `rename(df, all_of(c(a = "x")))`.
#'
#' @return A `tbl_gpu` object with the requested columns renamed. Column
#'   order, data, and types are unchanged. If `.data` is grouped and a
#'   group column is renamed, `group_vars()` reflects the new name.
#'
#' @details
#' Renaming a column to a name already used by a different, un-renamed
#' column errors, matching `dplyr::rename()`. A same-call swap (e.g.
#' `rename(df, x = y, y = x)`) is valid, since both original names are
#' replaced at once.
#'
#' @export
#' @importFrom dplyr rename
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   result <- gpu_mtcars |>
#'     rename(miles_per_gallon = mpg) |>
#'     collect()
#' }
rename.tbl_gpu <- function(.data, ...) {
  schema <- current_schema(.data)
  name_vec <- stats::setNames(schema$names, schema$names)

  loc <- tidyselect::eval_rename(rlang::expr(c(...)), name_vec)

  if (length(loc) == 0) {
    # No dots: dplyr's own rename() is a no-op in this case -- return .data
    # unchanged rather than pushing a needless (barrier) rename node that
    # would only block lazy optimization for nothing.
    return(.data)
  }

  old_names <- schema$names[loc]
  new_names <- names(loc)

  full_names <- schema$names
  full_names[loc] <- new_names
  check_rename_unique(full_names)

  node <- ast_rename(input_node(.data), old_names, new_names)
  push_op(.data, node)
}

#' Rename columns of a GPU table with a function
#'
#' Renames columns of a GPU table by applying a function to their names,
#' similar to `dplyr::rename_with()`. Runs entirely as a schema-only
#' operation, like [rename.tbl_gpu()].
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param .fn A function (or one-sided formula, e.g. `~ paste0(.x, "_2")`)
#'   applied once to the character vector of selected (old) names; must
#'   return a character vector of the same length.
#' @param .cols Tidyselect specification of which columns to rename
#'   (default: all columns). Does not itself support renaming.
#' @param ... Additional arguments passed to `.fn`.
#'
#' @return A `tbl_gpu` object with the selected columns renamed via `.fn`.
#'   Column order, data, and types are unchanged.
#'
#' @export
#' @importFrom dplyr rename_with
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   result <- gpu_mtcars |>
#'     rename_with(toupper) |>
#'     collect()
#' }
rename_with.tbl_gpu <- function(.data, .fn, .cols = dplyr::everything(), ...) {
  cols_quo <- rlang::enquo(.cols)
  fn <- rlang::as_function(.fn)

  schema <- current_schema(.data)
  name_vec <- stats::setNames(schema$names, schema$names)

  sel <- tidyselect::eval_select(cols_quo, name_vec, allow_rename = FALSE)
  old_names <- schema$names[sel]

  new_names <- fn(old_names, ...)
  if (!is.character(new_names)) {
    stop("`.fn` must return a character vector, not ", class(new_names)[1],
         ".", call. = FALSE)
  }
  if (length(new_names) != length(old_names)) {
    stop("`.fn` must return a vector of length ", length(old_names),
         ", not ", length(new_names), ".", call. = FALSE)
  }

  full_names <- schema$names
  full_names[sel] <- new_names
  check_rename_unique(full_names)

  if (length(sel) == 0 || identical(old_names, new_names)) {
    return(.data)
  }

  node <- ast_rename(input_node(.data), old_names, new_names)
  push_op(.data, node)
}
