# dplyr's backend "extension contract" (Phase 9, task 1)
#
# dplyr documents three low-level generics (`dplyr_row_slice()`,
# `dplyr_col_modify()`, `dplyr_reconstruct()`) that a backend can implement
# so that dplyr's own internals -- and third-party packages built on top of
# dplyr's data-frame abstraction -- can manipulate a custom table class
# generically, without knowing about any of its specific verb methods.
# `tbl_gpu` implements every verb it supports directly (filter.tbl_gpu(),
# mutate.tbl_gpu(), etc.), so these three generics are never on the path
# for cuplyr's OWN verb dispatch -- they exist purely so code that calls
# them directly (or indirectly, via some other package's dplyr-backend
# support) doesn't hit "no applicable method" on a tbl_gpu.
#
# Also here: `[.tbl_gpu` (the 1-d, column-selecting form dplyr's own
# `tbl_nongroup_vars()`-adjacent internals sometimes rely on) -- before this
# method existed, `tbl_gpu` (a plain classed `list`) fell through to base
# R's *list* `[` method, silently returning `list(ptr = <pointer>)` for
# `gt[1]` instead of a 1-column tbl_gpu. `names<-.tbl_gpu` already existed
# (R/tbl-gpu.R).

#' Modify rows of a GPU table by position (dplyr backend generic)
#'
#' Implements dplyr's `dplyr_row_slice()` extension-contract generic for
#' `tbl_gpu` -- a thin wrapper over the same `gpu_slice_indices()` primitive
#' [slice()][slice.tbl_gpu] itself uses for arbitrary positive-index
#' gathers (duplicates/reordering allowed). Per dplyr's own generic
#' (`dplyr_row_slice()`, not this method), `i` is always numeric (1-based
#' positions) or logical (a row mask); this method materializes any
#' pending lazy operations first, since there is no lazy AST node for an
#' arbitrary literal index gather.
#'
#' Matches dplyr's own `.drop = TRUE` (default) contract exactly, verified
#' empirically: slicing down to a subset of a `.drop = TRUE` grouped
#' table's rows recomputes `group_data()` from the surviving rows alone
#' (now-empty groups vanish). Real dplyr's `.drop = FALSE` case is
#' NOT matched here, though: dplyr's `grouped_df` carries its group-keys
#' table as a fixed attribute that a row-slice narrows via set intersection
#' (keeping now-EMPTY groups that existed before the slice, for ANY column
#' type, not just factors); `group_data.tbl_gpu()` (R/group-by.R) instead
#' always recomputes fresh from the actual GPU data on every call, so a
#' `.drop = FALSE` group whose only rows get sliced away here will vanish
#' from `group_data()` too, rather than surviving as an empty group -- see
#' `group_by.tbl_gpu()`'s own "`.drop` semantics" docs for why this is a
#' deliberate, documented deferral (a fixed, carried group-keys attribute
#' is a bigger structural change than this task's on-demand metadata query).
#'
#' @param data A `tbl_gpu` object.
#' @param i A numeric (1-based positions) or logical (row mask) vector.
#' @param ... Ignored. Included for compatibility with the dplyr generic.
#' @return A new `tbl_gpu`.
#' @export
#' @importFrom dplyr dplyr_row_slice
#' @keywords internal
dplyr_row_slice.tbl_gpu <- function(data, i, ...) {
  if (is.logical(i)) {
    i <- which(i)
  }

  materialized <- if (has_pending_ops(data)) compute(data) else data
  new_ptr <- wrap_gpu_call("dplyr_row_slice", gpu_slice_indices(materialized$ptr, as.double(i)))

  # NOT routed through `dplyr::dplyr_reconstruct()` -- see
  # `dplyr_reconstruct.tbl_gpu()`'s own docs below for why the top-level
  # generic is unsafe to call with a tbl_gpu `data` argument; this result
  # is already a fully-formed tbl_gpu, so there is nothing to reconstruct.
  new_tbl_gpu(
    ptr = new_ptr,
    schema = materialized$schema,
    groups = materialized$groups,
    group_drop = materialized$group_drop,
    exec_mode = materialized$exec_mode
  )
}

#' Add, remove, or replace columns of a GPU table with literal values (dplyr
#' backend generic)
#'
#' Implements dplyr's `dplyr_col_modify()` extension-contract generic for
#' `tbl_gpu`. Unlike [mutate()][mutate.tbl_gpu], `cols` here is a named list
#' of already-computed R VALUES (not expressions) -- a `NULL` entry removes
#' that column, a non-`NULL` entry adds or replaces it (recycled to
#' `nrow(data)` via [vctrs::vec_recycle_common()], matching
#' `dplyr:::dplyr_col_modify.data.frame()`'s own recycling contract,
#' including its error class on a size mismatch). Existing columns being
#' replaced stay at their original position; genuinely new columns are
#' appended at the end -- matching dplyr's own contract exactly.
#'
#' Implemented as: drop any column being removed/replaced from `data`,
#' upload the new/replacement columns as a small `tbl_gpu` (reusing the
#' ordinary `tbl_gpu()` upload path -- no expression IR needed, these are
#' already concrete values), and `bind_cols()` the two back together in the
#' right column order.
#'
#' @param data A `tbl_gpu` object.
#' @param cols A named list: `NULL` entries remove that column, other
#'   entries are recycled to `nrow(data)` and added/replace that column.
#' @return A new `tbl_gpu`.
#' @export
#' @importFrom dplyr dplyr_col_modify
#' @keywords internal
dplyr_col_modify.tbl_gpu <- function(data, cols) {
  if (length(cols) == 0) {
    return(data)
  }

  materialized <- if (has_pending_ops(data)) compute(data) else data
  nr <- dim(materialized)[1]

  cols <- do.call(vctrs::vec_recycle_common, c(cols, list(.size = nr)))

  is_removed <- vapply(cols, is.null, logical(1))
  removed_names <- names(cols)[is_removed]
  cols <- cols[!is_removed]

  schema_names <- materialized$schema$names
  existing_replaced <- intersect(names(cols), schema_names)
  drop_names <- union(removed_names, existing_replaced)

  base <- materialized
  if (length(drop_names) > 0) {
    keep <- setdiff(schema_names, drop_names)
    base <- dplyr::select(materialized, dplyr::all_of(keep))
  }

  if (length(cols) == 0) {
    return(base)
  }

  upload <- tbl_gpu(tibble::as_tibble(cols), lazy = FALSE)
  # `bind_cols()` unqualified (NOT `dplyr::bind_cols()`): dplyr's own
  # top-level `bind_cols()` is a plain function that calls
  # `vctrs::vec_cbind()` directly (never `UseMethod()`) -- this package's
  # OWN `bind_cols()` (R/bind.R) is a real S3 generic that dispatches to
  # `bind_cols.tbl_gpu()`; calling the `dplyr::` one here would try (and
  # fail) to treat a `tbl_gpu` as a plain vctrs-compatible value.
  combined <- bind_cols(base, upload)

  final_order <- c(
    intersect(schema_names, names(combined)),
    setdiff(names(combined), schema_names)
  )
  # NOT routed through `dplyr::dplyr_reconstruct()` -- see
  # `dplyr_reconstruct.tbl_gpu()`'s own docs below for why the top-level
  # generic is unsafe to call with a tbl_gpu `data` argument; `select()`
  # already returns a fully-formed tbl_gpu.
  dplyr::select(combined, dplyr::all_of(final_order))
}

#' Restore `tbl_gpu`-specific attributes after a generic transformation
#' (dplyr backend generic)
#'
#' Implements dplyr's `dplyr_reconstruct()` extension-contract generic for
#' `tbl_gpu`.
#'
#' ## An important structural gotcha (verified empirically)
#' dplyr's own top-level `dplyr_reconstruct()` function is NOT a plain
#' generic: its body is `data <- dplyr_new_data_frame(data);
#' dplyr_reconstruct_dispatch(data, template)` -- it unconditionally coerces
#' `data` via `vctrs::new_data_frame()` (dispatching on `template`'s class,
#' not `data`'s own) *before* any S3 method ever runs. For a genuinely
#' row-oriented backend (a real data frame, or dbplyr's lazy-query object,
#' whose top-level list elements really are one implicit "row" of
#' metadata), that's harmless. For `tbl_gpu` -- a FIXED six-field metadata
#' handle (`ptr`/`schema`/`lazy_ops`/`groups`/`exec_mode`/`group_drop`), not
#' a row-oriented list -- `new_data_frame()` instead produces a degenerate
#' `data.frame` whose "columns" are literally those six fields (verified
#' directly: `dplyr:::dplyr_new_data_frame(a_tbl_gpu)` returns a 0-row
#' `data.frame(ptr=, schema=, lazy_ops=, groups=, exec_mode=,
#' group_drop=)`). This method recovers the original `tbl_gpu` from that
#' exact shape (`new_data_frame()` doesn't deep-copy anything, so the GPU
#' pointer/schema survive intact) rather than trying to re-upload it as if
#' it were real tabular data, which would error (or silently corrupt) the
#' moment `df_to_gpu()` saw an external pointer as a "column".
#'
#' Because of this, [dplyr_row_slice.tbl_gpu()] and
#' [dplyr_col_modify.tbl_gpu()] (above) deliberately do NOT call
#' `dplyr::dplyr_reconstruct()` themselves -- their own results are already
#' fully-formed `tbl_gpu` objects, so routing them through the top-level
#' function would trigger this exact mangling for no benefit. This method
#' exists for the (rarer) case where something else calls
#' `dplyr::dplyr_reconstruct()` directly on a tbl_gpu.
#'
#' @param data The dplyr-generic's (possibly mangled, per above) `data`
#'   argument.
#' @param template A `tbl_gpu` object to restore grouping/exec_mode from.
#' @return A `tbl_gpu` object.
#' @export
#' @importFrom dplyr dplyr_reconstruct
#' @keywords internal
dplyr_reconstruct.tbl_gpu <- function(data, template) {
  if (is_tbl_gpu(data)) {
    return(data)
  }

  tbl_gpu_fields <- c("ptr", "schema", "lazy_ops", "groups", "exec_mode", "group_drop")
  if (is.data.frame(data) && identical(names(data), tbl_gpu_fields)) {
    return(new_tbl_gpu(
      ptr = data$ptr, schema = data$schema, lazy_ops = data$lazy_ops,
      groups = data$groups, exec_mode = data$exec_mode, group_drop = data$group_drop
    ))
  }

  # `data` is a genuine, different data frame (e.g. from generic code that
  # produced a real transformed dataset upstream, not a mangled tbl_gpu) --
  # re-upload it and restore template's grouping/exec_mode.
  out <- tbl_gpu(data, lazy = identical(template$exec_mode, "lazy"))

  template_groups <- template$groups
  if (length(template_groups) > 0 && all(template_groups %in% names(out))) {
    out <- dplyr::group_by(
      out, !!!rlang::syms(template_groups),
      .drop = isTRUE(template$group_drop %||% TRUE)
    )
  }

  out
}

#' Select columns of a GPU table by position, name, or logical mask
#'
#' `tbl_gpu` is a plain classed `list`, not a `data.frame` subclass, so
#' without this method `gt[1]` fell through to base R's *list* `[` method
#' (returning `list(ptr = <pointer to the GPU table>)`, not a 1-column
#' `tbl_gpu`) -- silently wrong and dangerous, since the leaked `ptr` looks
#' like it could be used directly. Only the 1-d form (`x[j]`, selecting
#' columns) is supported -- matching data.frame/tibble's own single-index
#' `[` convention -- via [dplyr::select()], not a 2-d `x[i, j]` row+column
#' form (use [slice()][slice.tbl_gpu]/[filter()][filter.tbl_gpu] for rows,
#' [select()][select.tbl_gpu] for columns instead).
#'
#' @param x A `tbl_gpu` object.
#' @param i A numeric (position, positive to keep or negative to drop),
#'   character (name), or logical (mask) vector selecting columns.
#' @param ... Ignored (present only so a 2-d call, e.g. `x[i, j]`, can be
#'   detected via [nargs()] and rejected with a clear error instead of
#'   silently misinterpreting `j`).
#' @return A new `tbl_gpu` with only the selected columns.
#' @export
`[.tbl_gpu` <- function(x, i, ...) {
  if (nargs() > 2) {
    stop(
      "2-d `[` indexing (`x[i, j]`) is not supported for tbl_gpu; ",
      "use slice()/filter() for rows and select() for columns instead.",
      call. = FALSE
    )
  }

  if (missing(i)) {
    return(x)
  }

  nm <- current_schema(x)$names
  col_positions <- stats::setNames(seq_along(nm), nm)
  selected <- tryCatch(
    col_positions[i],
    error = function(e) {
      stop("Invalid column index for `[.tbl_gpu`: ", conditionMessage(e), call. = FALSE)
    }
  )

  if (anyNA(selected)) {
    stop("Undefined columns selected.", call. = FALSE)
  }

  dplyr::select(x, dplyr::all_of(names(selected)))
}
