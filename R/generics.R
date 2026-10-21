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
# Also here: `[.tbl_gpu`, tibble's `[` for both the 1-d (column) and the 2-d
# (`x[i, j]`, `drop =`) forms -- before this method existed, `tbl_gpu` (a
# plain classed `list`) fell through to base R's *list* `[` method, silently
# returning `list(ptr = <pointer>)` for `gt[1]` instead of a 1-column
# tbl_gpu. `names<-.tbl_gpu` already existed (R/tbl-gpu.R).

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

  if (length(materialized$schema$names) == 0L) {
    # Zero-column table: nothing for cudf to gather, only the R-side count.
    n <- materialized[["nrow"]]
    if (is.null(n) || any(i < 1L | i > n)) {
      stop("Can't slice a tbl_gpu with no columns: unknown row count or ",
           "out-of-range positions.", call. = FALSE)
    }
    return(new_tbl_gpu(
      ptr = materialized$ptr,
      schema = materialized$schema,
      groups = materialized$groups,
      group_drop = materialized$group_drop,
      exec_mode = materialized$exec_mode,
      nrow = length(i),
      rowwise = is_rowwise_gpu(materialized)
    ) |> carry_frame_meta(materialized))
  }

  new_ptr <- wrap_gpu_call("dplyr_row_slice", gpu_slice_indices(materialized$ptr, as.double(i)))

  # NOT routed through `dplyr::dplyr_reconstruct()` -- see
  # `dplyr_reconstruct.tbl_gpu()`'s own docs below for why the top-level
  # generic is unsafe to call with a tbl_gpu `data` argument; this result
  # is already a fully-formed tbl_gpu, so there is nothing to reconstruct.
  carry_frame_meta(new_tbl_gpu(
    ptr = new_ptr,
    schema = materialized$schema,
    groups = materialized$groups,
    group_drop = materialized$group_drop,
    exec_mode = materialized$exec_mode,
    rowwise = is_rowwise_gpu(materialized)
  ), materialized)
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

  # Existing columns being replaced stay in place, new ones are appended.
  final_order <- c(
    setdiff(schema_names, removed_names),
    setdiff(names(cols), schema_names)
  )
  apply_col_changes(data, materialized, final_order, cols)
}

# Build the table with columns `final_names`, where the columns named in
# `new_cols` (a named list of already-recycled R vectors) are uploaded and
# every other one is carried over from `m`, the materialized (no pending ops)
# table of `x`. Shared by `dplyr_col_modify()` and `$<-`/`[[<-`.
#
# Uploads the new columns as a small `tbl_gpu`, drops the replaced columns from
# `m`, and `bind_cols()`s the two back together. `x`'s groups (those that
# survive) and `.drop`, and its exec_mode, are kept: a lazy table stays lazy.
apply_col_changes <- function(x, m, final_names, new_cols) {
  if (length(final_names) == 0L) {
    stop("A tbl_gpu cannot have zero columns.", call. = FALSE)
  }
  base_names <- setdiff(final_names, names(new_cols))
  work <- set_tbl_gpu_field(ungrouped(m), "group_drop", TRUE)
  work <- set_tbl_gpu_field(work, "exec_mode", "eager")

  combined <- if (length(base_names) == 0L) {
    NULL
  } else if (identical(base_names, .subset2(work, "schema")$names)) {
    work
  } else {
    push_op(work, ast_select(input_node(work), base_names))
  }

  if (length(new_cols) > 0L) {
    # `bind_cols()` unqualified (NOT `dplyr::bind_cols()`): this package's own
    # (R/bind.R) binds a tbl_gpu on the GPU; dplyr's would try (and fail) to
    # treat a `tbl_gpu` as a plain vctrs-compatible value.
    upload <- tbl_gpu(tibble::as_tibble(new_cols, .name_repair = "minimal"), lazy = FALSE)
    # "minimal": the names were validated by tibble (or by the caller) already,
    # and "unique" would rewrite a legal `...N` name.
    combined <- if (is.null(combined)) upload else bind_cols(combined, upload, .name_repair = "minimal")
  }

  # An index bug here would surface from the GPU layer as an "out of memory"
  # hint, so say what happened.
  if (!all(final_names %in% .subset2(combined, "schema")$names)) {
    stop("Internal error: columns ", paste0("`", setdiff(final_names, .subset2(combined, "schema")$names), "`", collapse = ", "),
         " were lost while modifying a tbl_gpu.", call. = FALSE)
  }
  if (!identical(.subset2(combined, "schema")$names, final_names)) {
    combined <- push_op(combined, ast_select(input_node(combined), final_names))
  }

  combined <- with_surviving_groups(x, combined)
  set_tbl_gpu_field(combined, "exec_mode", .subset2(x, "exec_mode"))
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

  # (`nrow` is only present on a zero-column table, `fixed_keys` on a table
  # whose group keys were fixed, `bare_df` on a table from a plain data.frame)
  optional <- c("nrow", "fixed_keys", "unspecified", "bare_df", "rowwise")
  if (is.data.frame(data) &&
      identical(setdiff(names(data), optional), setdiff(tbl_gpu_fields, optional))) {
    # dplyr_reconstruct.grouped_df()/.data.frame(): the template's groups
    # (those that survive), `.drop`, attributes and class win over `data`'s.
    groups <- intersect(template$groups, data$schema$names)
    out <- new_tbl_gpu(
      ptr = data$ptr, schema = data$schema, lazy_ops = data$lazy_ops,
      groups = groups, exec_mode = data$exec_mode,
      group_drop = if (length(groups) > 0L) template$group_drop %||% TRUE else TRUE,
      nrow = data[["nrow"]], fixed_keys = data[["fixed_keys"]],
      unspecified = data[["unspecified"]],
      rowwise = is_rowwise_gpu(template)
    )
    return(carry_frame_meta(out, template))
  }

  # `data` is a genuine, different data frame (e.g. from generic code that
  # produced a real transformed dataset upstream, not a mangled tbl_gpu) --
  # re-upload it and restore template's grouping/exec_mode.
  out <- tbl_gpu(data, lazy = identical(template$exec_mode, "lazy"))

  template_groups <- template$groups
  if (is_rowwise_gpu(template)) {
    return(rowwise_mark(out, intersect(template_groups, names(out))))
  }
  if (length(template_groups) > 0 && all(template_groups %in% names(out))) {
    out <- dplyr::group_by(
      out, !!!rlang::syms(template_groups),
      .drop = isTRUE(template$group_drop %||% TRUE)
    )
  }

  out
}

#' Subset a GPU table with tibble's `[` semantics
#'
#' `tbl_gpu` is a plain classed `list`, not a `data.frame` subclass, so a
#' `[` method is needed (base R's list `[` would leak the raw `ptr`).
#' Follows the `[` method of tibble (`tibble:::[.tbl_df`), and for a grouped
#' table dplyr's `[.grouped_df`:
#' - `x[j]` selects columns; `x[i, j]`, `x[i, ]`, `x[, j]`, `x[]` are the
#'   2-d forms. `j` is resolved first (names, positions, negative, logical)
#'   with tibble's `vctrs_error_subscript_*` errors, messages and calls.
#' - Rows (`i`): positive positions gather in the given order (duplicates
#'   kept; out-of-range or `NA` gives an all-`NA` row), `0` is dropped,
#'   negative positions drop rows (out-of-range ones are ignored), a logical
#'   recycles from length 1 or must have length `nrow` (`NA` gives an `NA`
#'   row), a character is read as a row position (no row names).
#' - `drop = TRUE` with exactly one result column returns that column as a
#'   vector (the only shape that executes a lazy table); with several
#'   columns it is ignored, except that a grouped table comes back
#'   ungrouped. In the 1-d form `drop=` warns and is ignored (a grouped table
#'   is still ungrouped, as in dplyr).
#'
#' Row selection is one GPU gather (an `ast_slice(mode = "bracket")` node,
#' lazy or eager); column selection is the ordinary select node. Groups are
#' the original group variables that survive, `.drop` is kept.
#'
#' Row subscripts are validated against the real row count at call time
#' whenever that is known without executing anything (an eager table, or a
#' lazy one with no pending operations). On a lazy table with pending
#' operations the row subscript is resolved when the plan runs, so its errors
#' (same class, message and `[` call) surface at `collect()`/`compute()`.
#'
#' A zero-column result (`x[, 0]`, `x[character()]`, `x[, FALSE]`) is an
#' ungrouped zero-column table that keeps the row count (a lazy table with
#' pending operations is computed first).
#'
#' Divergences from tibble, all loud errors: duplicated columns
#' (`x[, c("a", "a")]`) because a `tbl_gpu` cannot represent them, and matrix
#' subscripts (`x[m]`, and the soft-deprecated one-column matrix row subscript
#' `x[m, ]`, which tibble still accepts with a deprecation notice).
#'
#' @param x A `tbl_gpu` object.
#' @param i Row subscript (or, in the 1-d form, the column subscript).
#' @param j Column subscript.
#' @param drop If `TRUE` and the result has one column, return the column.
#' @return A `tbl_gpu`, or a vector when `drop = TRUE` selects one column.
#' @export
`[.tbl_gpu` <- function(x, i, j, drop = FALSE) {
  # Subscript labels and the error call are the user's, as in tibble.
  i_arg <- substitute(i)
  j_arg <- substitute(j)
  # sys.call() of an internally dispatched method names the method, not `[`.
  this_call <- sys.call()
  this_call[[1L]] <- quote(`[`)
  this_env <- environment()

  i_missing <- missing(i)
  j_missing <- missing(j)
  if (i_missing) i_arg <- NULL
  if (j_missing) j_arg <- NULL
  if (!i_missing && is.null(i)) i <- integer()
  if (!j_missing && is.null(j)) j <- integer()

  ungroup_after <- FALSE
  n_real_args <- nargs() - !missing(drop)
  if (n_real_args <= 2L) {
    # dplyr's `[.grouped_df` sees the user's `drop` and ungroups on TRUE even
    # though tibble ignores it in the 1-d form.
    ungroup_after <- isTRUE(drop)
    if (!missing(drop)) {
      rlang::warn(
        "`drop` argument ignored for subsetting a tibble with `x[j]`, it has an effect only for `x[i, j]`.",
        use_cli_format = FALSE
      )
      drop <- FALSE
    }
    j <- if (i_missing) NULL else i
    j_arg <- i_arg
    i_arg <- NULL
    j_missing <- i_missing
    i_missing <- TRUE
    if (!j_missing && is.matrix(j)) {
      stop("Matrix subscripts `x[m]` are not supported for tbl_gpu.", call. = FALSE)
    }
  }

  # Row count, when known without executing anything (a zero-column table's
  # lives on the R side, base_dim() reads it).
  n_known <- if (has_pending_ops(x)) NA_integer_ else base_dim(x)[1]

  # A zero-column input keeps zero columns, whatever `j` is.
  zero_cols <- length(current_schema(x)$names) == 0L
  if (!j_missing) {
    schema <- current_schema(x)
    loc <- bracket_col_location(j, schema$names, j_arg, this_env)
    zero_cols <- zero_cols || length(loc) == 0L
    if (!zero_cols && anyDuplicated(loc)) {
      stop("`[` with duplicated columns is not supported for tbl_gpu (column names must be unique).",
           call. = FALSE)
    }
    if (!zero_cols && !identical(loc, seq_along(schema$names))) {
      keep_groups <- x$groups
      drop_flag <- x$group_drop %||% TRUE
      x <- push_op(x, ast_select_at(input_node(x), schema$names, loc))
      x$groups <- intersect(keep_groups, schema$names[loc])
      # dplyr: once no group column is left the table is ungrouped, and a later
      # group_by() starts from the default `.drop` again.
      x$group_drop <- if (length(x$groups) > 0L) drop_flag else TRUE
    }
  }

  if (!i_missing && !isTRUE(i) && zero_cols && !is.na(n_known)) {
    # only the number of selected rows is kept: no need to gather any column
    n_rows <- length(bracket_row_positions(i, n_known, i_arg, this_env))
    return(zero_column_result(x, "[", n = n_rows))
  }

  if (!i_missing && !isTRUE(i)) {
    drop_flag <- x$group_drop %||% TRUE
    node <- if (is.na(n_known)) {
      # nrow unknown until the plan runs: resolve (and validate) at lowering,
      # reporting against the user's subscript and call.
      ast_slice(input_node(x), mode = "bracket", raw_indices = i,
                bracket_arg = i_arg, bracket_call = this_call)
    } else {
      ast_slice(input_node(x), mode = "bracket",
                raw_indices = bracket_row_positions(i, n_known, i_arg, this_env),
                rows_resolved = TRUE)
    }
    x <- push_op(x, node)
    x$group_drop <- drop_flag
  }

  # Zero columns: the rows above were selected on the full table, the result
  # keeps only their count.
  if (zero_cols) {
    return(zero_column_result(x, "["))
  }

  if (isTRUE(drop) && length(x$schema$names) == 1L) {
    return(dplyr::pull(dplyr::ungroup(x), 1L))
  }
  if (isTRUE(drop) || ungroup_after) {
    x <- dplyr::ungroup(x)
  }
  rowwise_column_order(x)
}

# Column subscript -> integer positions, with tibble's error classes, labels
# and call (`vectbl_as_col_location()` + `subclass_col_index_errors()`).
bracket_col_location <- function(j, names, j_arg, call) {
  withCallingHandlers(
    vctrs::vec_as_location(j, length(names), names, missing = "error", call = call),
    vctrs_error_subscript = function(cnd) {
      cnd$subscript_arg <- j_arg
      cnd$subscript_elt <- "column"
      rlang::cnd_signal(cnd)
    }
  )
}

# Row subscript -> 1-based positions against `n` rows (NA = all-NA row).
# Mirrors `vectbl_as_row_index()`: out-of-range positives become NA,
# out-of-range negatives are ignored, everything else goes to vctrs, with
# the errors relabelled as `subclass_row_index_errors()` does.
bracket_row_positions <- function(i, n, i_arg, call) {
  if (is.character(i)) {
    # Tibble coerces character subscripts by its own rules, so ask it on a
    # one-column stub (seq_len() is compact, cost is the result size).
    return(tibble::new_tibble(list(.r = seq_len(n)), nrow = n)[i, ]$.r)
  }
  if (is.numeric(i)) {
    if (any(i > 0, na.rm = TRUE)) {
      i[which(i > n)] <- NA
    } else if (any(i < 0, na.rm = TRUE)) {
      i <- i[!(i < -n)]
      if (length(i) == 0L) i <- seq_len(n)
    }
  }
  withCallingHandlers(
    vctrs::vec_as_location(i, n, missing = "propagate", call = call),
    vctrs_error_subscript = function(cnd) {
      cnd$subscript_arg <- i_arg
      cnd$subscript_elt <- "row"
      rlang::cnd_signal(cnd)
    }
  )
}

# =============================================================================
# `$`, `[[`, `$<-`, `[[<-`, `[<-`: the base-R object contract (F04)
#
# A tbl_gpu is a classed list whose six elements are internal fields, but to
# users it is a table, so `gt$x` / `gt[["x"]]` return COLUMN x (tibble's
# semantics, warnings and errors), and assignment is a column modification on
# the GPU.
#
# The package itself reads and writes the fields with `x$ptr`, `x$groups <- `,
# ... hundreds of times, through these same methods. The rule that keeps those
# working: a field name (`tbl_gpu_fields`) accessed from code whose top-level
# environment is the cuplyr namespace is the raw list element; everything else
# (user code, other packages, even for a column that happens to be named
# `groups`) is a column. testthat evaluates test files in a child of the
# global environment, not of the namespace, so the package's own tests keep
# internal access through a marker binding (`.cuplyr_internal_access`, defined
# in the test helpers). The check is `switch()` on the name first, so the
# common internal access costs one extra `topenv()`.
#
# Subscript validation, recycling and error messages are tibble's own: each
# method runs the same operation on a "stub" tibble (the right names and
# row count, placeholder columns) and acts on the outcome. The user's
# subscripts reach tibble as the ORIGINAL promises, forwarded through `...`
# (`[[` takes `(x, ...)`, `[[<-` `(x, ..., value)`): each is evaluated once, in
# its own environment, and tibble's `substitute()` labels (`Can't extract
# column with `nm``) show the user's expressions. Rebuilding a call from
# `substitute(i)` and evaluating it somewhere else would evaluate the
# expression twice and in the wrong frame (`lapply(tbls, `[[`, nm)`). The
# assigned value is passed on as `value` (replacement methods must end in a
# `value` argument), so tibble's messages about it name `value`.
# =============================================================================

# `nrow` is the row count of a zero-column table (new_tbl_gpu()); `fixed_keys`
# the group keys `slice(.preserve = TRUE)` fixed, `unspecified` the names of
# vctrs::unspecified() columns, `bare_df` that the table came from a plain
# data.frame (F31), `rowwise` that it is a rowwise table (F49); all four only
# present when set.
tbl_gpu_fields <- c("ptr", "schema", "lazy_ops", "groups", "exec_mode", "group_drop", "nrow",
                    "fixed_keys", "unspecified", "bare_df", "rowwise")

# The cuplyr namespace (this file is evaluated in it).
cuplyr_ns <- environment()

# Is `env` (a caller's environment) package-internal code? True inside the
# namespace, and in the package's own tests: testthat evaluates them in
# environments below the global environment, where the test helpers
# (tests/testthat/helper-gpu.R) define the marker `.cuplyr_internal_access`.
# The search stops at the global environment, so user code, other packages and
# functions re-homed to `globalenv()` never count (the marker can also be
# visible from there, since pkgload loads the helpers into the namespace).
internal_caller <- function(env) {
  if (identical(parent.env(env), cuplyr_ns) || identical(topenv(env), cuplyr_ns)) {
    return(TRUE)
  }
  global <- globalenv()
  while (!identical(env, global) && !identical(env, emptyenv())) {
    if (exists(".cuplyr_internal_access", envir = env, inherits = FALSE)) {
      return(TRUE)
    }
    env <- parent.env(env)
  }
  FALSE
}

is_tbl_gpu_field <- function(name) {
  is.character(name) && length(name) == 1L && !is.na(name) && name %in% tbl_gpu_fields
}

#' @export
`$.tbl_gpu` <- function(x, name) {
  if (switch(name, ptr = , schema = , lazy_ops = , groups = , exec_mode = ,
             group_drop = , nrow = , fixed_keys = , unspecified = , bare_df = , rowwise = TRUE, FALSE) &&
      internal_caller(parent.frame())) {
    return(.subset2(x, name))
  }
  col <- match(name, current_schema(x)$names)
  if (is.na(col)) {
    rlang::warn(paste0("Unknown or uninitialised column: `", name, "`."))
    return(NULL)
  }
  fetch_column(x, col)
}

#' @export
`[[.tbl_gpu` <- function(x, ...) {
  nd <- ...length()
  if (nd == 1L && !missing(..1) && is_tbl_gpu_field(..1) && internal_caller(parent.frame())) {
    return(.subset2(x, ..1))
  }

  nms <- current_schema(x)$names
  # `exact=` is tibble's to warn about; every other argument is a subscript.
  n_sub <- nd - sum(...names() == "exact")
  # sys.call() of an internally dispatched method names the method, not `[[`.
  this_call <- sys.call()
  this_call[[1L]] <- quote(`[[`)

  if (n_sub <= 1L) {
    # tibble resolves the column subscript against columns whose value is
    # their own position, so its errors, warnings and messages are reproduced
    # as is.
    pos_stub <- tibble::new_tibble(stats::setNames(as.list(seq_along(nms)), nms), nrow = 1L)
    pos <- with_user_call(`[[`(pos_stub, ...), this_call)
    return(if (is.null(pos)) NULL else fetch_column(x, pos))
  }

  # x[[i, j]]: tibble validates the row and column on a stub whose every column
  # is the row numbers (a compact sequence, shared by all columns), then one
  # row of the column is gathered on the GPU.
  m <- if (has_pending_ops(x)) compute(x) else x
  n <- gpu_dim(.subset2(m, "ptr"))[[1L]]
  rows <- seq_len(n)
  row_stub <- tibble::new_tibble(stats::setNames(rep(list(rows), length(nms)), nms), nrow = n)
  row <- with_user_call(`[[`(row_stub, ...), this_call)
  if (is.null(row)) {
    return(NULL)
  }
  # The column position, from the column subscript alone: `col_of()` has
  # tibble's formals, so `j` matches by name or position like it does there.
  # Its promise was forced by the call above, so it is not evaluated again,
  # and the warnings that call raised are muffled.
  pos_stub <- tibble::new_tibble(stats::setNames(as.list(seq_along(nms)), nms), nrow = 1L)
  col_of <- function(i, j, ..., exact = TRUE) `[[`(pos_stub, j)
  pos <- suppressWarnings(col_of(...))
  fetch_cell(m, pos, row)
}

# Errors tibble raises against a subscript stub name the stub's call
# (`pos_stub[[...]]`); they are re-signalled with the user's call, class and
# message unchanged. Errors from evaluating the user's own subscript name some
# other call and pass through as they are.
with_user_call <- function(expr, call) {
  tryCatch(expr, error = function(e) {
    stub_call <- conditionCall(e)
    if (is.call(stub_call) && any(c("pos_stub", "row_stub") %in% all.names(stub_call))) {
      e$call <- call
    }
    stop(e)
  })
}

# One column of `x` as an R vector: ungroup, select it, collect that
# one-column table. Lazy tables only append a node to the plan (so the
# optimizer prunes everything else), eager ones gather one column on the GPU.
fetch_column <- function(x, col) {
  .subset2(collect_frame(select_one_column(x, col)), 1L)
}

# One cell: the same, with a one-row gather before the transfer.
fetch_cell <- function(x, col, row) {
  one <- select_one_column(x, col)
  one <- push_op(one, ast_slice(input_node(one), mode = "bracket",
                                raw_indices = row, rows_resolved = TRUE))
  .subset2(collect_frame(one), 1L)
}

select_one_column <- function(x, col) {
  names <- current_schema(x)$names
  x <- set_tbl_gpu_field(x, "groups", character())
  push_op(x, ast_select_at(input_node(x), names, col))
}

#' @export
`$<-.tbl_gpu` <- function(x, name, value) {
  if (switch(name, ptr = , schema = , lazy_ops = , groups = , exec_mode = ,
             group_drop = , nrow = , fixed_keys = , unspecified = , bare_df = , rowwise = TRUE, FALSE) &&
      internal_caller(parent.frame())) {
    return(set_tbl_gpu_field(x, name, value))
  }
  gpu_assign_column(x, value, function(stub) {
    eval(call("$<-", stub, as.name(name), quote(value)))
  })
}

#' @export
`[[<-.tbl_gpu` <- function(x, ..., value) {
  nd <- ...length()
  if (nd == 1L && !missing(..1) && is_tbl_gpu_field(..1) && internal_caller(parent.frame())) {
    return(set_tbl_gpu_field(x, ..1, value))
  }
  if (nd > 1L) {
    stop(
      "`[[<-` with a row and a column (`x[[i, j]] <- value`) is not supported for tbl_gpu.\n",
      "Use `mutate()` to change values.",
      call. = FALSE
    )
  }
  # The subscript stays an unevaluated promise that tibble labels its
  # messages with.
  gpu_assign_column(x, value, function(stub) `[[<-`(stub, ..., value = value))
}

#' @export
`[<-.tbl_gpu` <- function(x, i, j, value) {
  # Whole columns only: `x[j] <- v` and `x[, j] <- v`, through the same
  # validated stub assignment as `$<-` / `[[<-` (so recycling, new columns,
  # `NULL` drops and a grouped table's vars behave like the tibble). Every cell
  # form errors: a row subscript, an empty `x[] <-` / `x[, ] <-`, and a matrix
  # subscript (tibble's `x[is.na(x)] <- 0`).
  nsub <- nargs() - 2L  # the subscripts, empty ones included
  if (nsub == 2L && !missing(i)) {
    stop_cell_assign("with a row subscript (`x[i, j] <- value`)")
  }
  if ((nsub == 1L && missing(i)) || (nsub == 2L && missing(j))) {
    stop_cell_assign("with an empty subscript (`x[] <- value`, `x[, ] <- value`)")
  }
  if (nsub == 1L) {
    if (is.matrix(i) || is.array(i)) {
      stop_cell_assign("with a matrix subscript (`x[m] <- value`)")
    }
    out <- gpu_assign_column(x, value, function(stub) {
      stub[i] <- value
      stub
    })
  } else {
    out <- gpu_assign_column(x, value, function(stub) {
      stub[, j] <- value
      stub
    })
  }
  # `[<-.rowwise_df` lists the surviving rowwise vars in column order.
  rowwise_column_order(out)
}

stop_cell_assign <- function(what) {
  stop(
    "`[<-` is not supported for tbl_gpu ", what, ": it cannot modify cells in place.\n",
    "Use `mutate()` to change values, and `x[j] <- value` / `x[, j] <- value` to set whole columns.",
    call. = FALSE
  )
}

set_tbl_gpu_field <- function(x, name, value) {
  cl <- oldClass(x)
  x <- unclass(x)
  x[[name]] <- value
  class(x) <- cl
  x
}

# Stub tibble for validating an assignment: the table's column names, `n`
# rows, and a placeholder column that no assigned value can be identical to.
assign_marker <- structure(0L, class = "cuplyr_assign_marker")

assign_stub <- function(nms, n) {
  tibble::new_tibble(stats::setNames(rep(list(assign_marker), length(nms)), nms), nrow = n)
}

# A plain, non-missing, finite length-one logical/integer/double/character: a
# value that mutate() takes as a literal and types like tibble would (no
# attributes: not a factor, Date, ...; no NA/NaN/Inf, whose literals are not
# typed or kept the same way).
is_plain_scalar <- function(v) {
  length(v) == 1L && is.atomic(v) && is.null(attributes(v)) &&
    (is.logical(v) || is.numeric(v) || is.character(v)) &&
    !is.na(v) && !(is.double(v) && is.infinite(v))
}

# `x$col <- value` / `x[[j]] <- value` for tbl_gpu. `assign_fn(stub)` performs
# the assignment on a stub tibble, so tibble validates the subscript, recycles
# `value` to the row count and says what columns result.
gpu_assign_column <- function(x, value, assign_fn) {
  nms <- current_schema(x)$names

  if (is.null(value)) {
    # Dropping a column (or nothing): no row count needed, so no execution.
    keep <- names(assign_fn(assign_stub(nms, 0L)))
    if (identical(keep, nms)) {
      return(x)
    }
    if (length(keep) == 0L) {
      stop("A tbl_gpu cannot have zero columns.", call. = FALSE)
    }
    return(with_surviving_groups(x, push_op(ungrouped(x), ast_select(input_node(x), keep))))
  }

  if (is_plain_scalar(value) && length(nms) > 0L) {
    # A length-one value recycles to any row count, so tibble's check does not
    # need the real one, and the column is a literal in a mutate(): a lazy plan
    # is extended, not run, and nothing is built on the host. (Not on a
    # zero-column table: mutate() can't know its row count there.)
    res <- assign_fn(assign_stub(nms, 1L))
    changed <- changed_columns(res)
    # (Not over a factor column: mutate() would keep the old column's levels.)
    if (length(changed) == 1L && is_plain_scalar(.subset2(res, changed)) &&
        !changed %in% names(current_schema(x)$factor_levels)) {
      u <- ungrouped(x)
      out <- rlang::inject(dplyr::mutate(u, !!changed := !!.subset2(res, changed)))
      if (identical(names(res), current_schema(out)$names)) {
        return(with_surviving_groups(x, out))
      }
    }
  }

  # The row count is the one AFTER pending operations.
  m <- if (has_pending_ops(x)) compute(x) else x
  # dim(), not gpu_dim(): cudf reports 0 rows for a zero-column table.
  n <- dim(m)[[1L]]
  if (is.na(n)) {
    check_zero_column_input(m, current_schema(m), "$<-")
  }
  res <- assign_fn(assign_stub(nms, n))

  final_names <- names(res)
  changed <- changed_columns(res)
  new_cols <- lapply(stats::setNames(changed, changed), function(nm) {
    v <- .subset2(res, nm)
    if (is.data.frame(v) || is.matrix(v) || is.array(v) || (is.list(v) && !inherits(v, "POSIXlt"))) {
      stop(
        "Can't assign a data frame, matrix or list as a tbl_gpu column (`", nm, "`).\n",
        "A GPU column must be an atomic vector.",
        call. = FALSE
      )
    }
    v
  })
  apply_col_changes(x, m, final_names, new_cols)
}

# Names of the columns of a stub-assignment result that were assigned (they no
# longer hold the placeholder).
changed_columns <- function(res) {
  nms <- names(res)
  nms[!vapply(nms, function(nm) identical(.subset2(res, nm), assign_marker), logical(1))]
}

# `x` without its groups (what the one-column helpers and the assignment work on).
# A rowwise table loses its rowwise-ness too: with_surviving_groups() and
# set_groups_by_position() put it back from the table they are given.
ungrouped <- function(x) {
  set_tbl_gpu_field(set_tbl_gpu_field(x, "groups", character()), "rowwise", NULL)
}

# `out`, a table derived from the ungrouped `x`, regrouped by the group
# columns of `x` that it still has, keeping `x`'s `.drop` while any survive.
# Like dplyr_col_modify() + dplyr_reconstruct(out, x), `x` also decides the
# attributes and class (F31).
with_surviving_groups <- function(x, out) {
  keep <- intersect(.subset2(x, "groups"), current_schema(out)$names)
  out <- reset_frame_meta(out, x)
  out <- set_tbl_gpu_field(out, "groups", keep)
  out <- set_tbl_gpu_field(out, "rowwise", if (is_rowwise_gpu(x)) TRUE)
  set_tbl_gpu_field(out, "group_drop",
                    if (length(keep) > 0L) .subset2(x, "group_drop") %||% TRUE else TRUE)
}
