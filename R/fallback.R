# Transparent CPU fallback layer (Phase 2)
#
# Goal: every dplyr generic that doesn't yet have a GPU-native tbl_gpu method
# should still *work* on a tbl_gpu, by collecting to a tibble, running the
# real dplyr verb on the CPU, and re-uploading the result. This makes the
# entire dplyr API surface usable today; later phases replace individual
# fallbacks with GPU-native implementations one verb at a time (see
# scratchpad/todo.md Phase 3+).
#
# `gpu_fallback()` below is the single shared implementation every method in
# this file is built on -- mirroring the "one shared helper, many thin verb
# wrappers" shape used by `push_op()` (R/execute.R) for GPU-native verbs.

#' Run a dplyr verb on the CPU and re-upload the result
#'
#' The core Phase-2 fallback primitive. Materializes any pending lazy
#' operations, collects `.data` to a tibble (restoring `group_by()` structure
#' from `.data$groups` so grouped semantics apply on the CPU exactly as they
#' would on a real `grouped_df`), applies `expr_fn` (which should call the
#' real dplyr verb), and re-uploads the result via [tbl_gpu()].
#'
#' @param verb_name Character, the verb name to use in the fallback
#'   notification (e.g. `"transmute"`, `"slice_min"`), gated by
#'   `options(cuplyr.fallback = )` (see [cuplyr_fallback_notify()]).
#' @param .data A `tbl_gpu` object.
#' @param expr_fn A function of one argument (a tibble, already grouped if
#'   `.data` was grouped) that calls the corresponding real dplyr verb and
#'   returns its result.
#' @param as_is Logical. If `TRUE`, `expr_fn`'s return value is always
#'   returned unchanged, never re-uploaded -- for a verb that returns
#'   metadata *about* the grouping/data rather than a transformed dataset,
#'   even when that metadata happens to be a small data frame (dplyr's own
#'   grouping-metadata queries, e.g. `group_data()`/`group_keys()`, used to
#'   be implemented this way -- Phase 9 task 1 made them GPU-native instead,
#'   see R/group-by.R -- but the parameter stays for any future fallback
#'   verb with the same "always a plain R object" shape).
#' @return Either a new `tbl_gpu` (grouping, execution mode, and factor
#'   levels restored/round-tripped), or `expr_fn`'s raw return value
#'   unchanged when: `as_is = TRUE`; the result isn't a data frame at all
#'   (e.g. `pull()`'s vector, `group_split()`'s list); or the result is a
#'   data frame that contains list-columns (e.g. `nest_by()`/`group_nest()`/
#'   `nest_join()`'s nested-tibble columns), which have no GPU representation.
#'
#'   When the result is a grouped data frame whose `group_data()` keeps groups
#'   with no row that a plain regroup of its data would not have (dplyr's
#'   `dplyr_col_modify.grouped_df()` does this for a verb that leaves the group
#'   columns alone, after `filter()`/`slice()` with `.preserve = TRUE`), the
#'   returned `tbl_gpu` carries those keys as `$fixed_keys` (see
#'   `fixed_keys_of()`); verbs that regroup get none.
#' @keywords internal
gpu_fallback <- function(verb_name, .data, expr_fn, ..., as_is = FALSE) {
  if (!is_tbl_gpu(.data)) {
    stop("gpu_fallback() requires a tbl_gpu object.", call. = FALSE)
  }

  was_lazy <- identical(.data$exec_mode, "lazy")

  # Materialize any pending lazy ops first, per Phase 2 design: the fallback
  # always operates on a real, fully-computed GPU table.
  materialized <- if (has_pending_ops(.data)) compute(.data) else .data

  cuplyr_fallback_notify(verb_name, sprintf("%s()", verb_name))

  # dplyr sees the frame it would have: the input's attributes and class
  # (a bare data.frame stays one), so its own result carries them (F31).
  tbl <- restore_frame_meta(collect_frame(materialized), materialized)

  # Restore dplyr grouping metadata on the CPU side so grouped semantics
  # (per-group aggregation, grouped slice_min/slice_max, etc.) work exactly
  # as they would on a real grouped_df.
  if (length(materialized$groups) > 0) {
    drop <- isTRUE(materialized$group_drop %||% TRUE)
    keys <- fixed_keys_of(materialized)
    tbl <- if (is.null(keys)) {
      dplyr::group_by(tbl, !!!rlang::syms(materialized$groups), .drop = drop)
    } else {
      with_fixed_groups(tbl, keys, drop)
    }
  }

  result <- expr_fn(tbl)

  # A `rowwise_df` result has no `tbl_gpu` representation (re-uploading would
  # drop its per-row grouping). It is returned as-is, so the next verb
  # dispatches to dplyr's own method; cuplyr never claims a `rowwise()`
  # result stays GPU-backed.
  if (as_is || !is.data.frame(result) || has_list_column(result) ||
      inherits(result, "rowwise_df")) {
    return(result)
  }

  # The verb may change grouping (e.g. summarise()-like verbs drop it,
  # reframe()/rowwise() may set it) -- read it back from the actual result
  # rather than assuming it's preserved.
  new_groups <- dplyr::group_vars(result)

  new_tbl <- tbl_gpu(dplyr::ungroup(result), lazy = FALSE)
  # dplyr's own result decides the attributes and class; ungroup() of a
  # grouped one would have dropped them (F31). The class rides on the upload.
  if (length(new_groups) > 0) {
    new_tbl <- with_frame_attrs(strip_frame_attrs(new_tbl), frame_attrs(result))
  }

  # `.drop` is read back the same way: a verb that keeps x's grouping (joins,
  # filter, ...) keeps its `.drop = FALSE` too.
  if (length(new_groups) > 0) {
    new_tbl <- dplyr::group_by(new_tbl, !!!rlang::syms(new_groups),
                               .drop = dplyr::group_by_drop_default(result))
    new_tbl$fixed_keys <- result_fixed_keys(result, new_groups)
  }

  if (was_lazy) {
    new_tbl <- as_lazy(new_tbl)
  }

  new_tbl
}

# The group keys of the grouped_df `result` when dplyr kept groups a plain
# regroup of its data would not have (the empty groups of `filter()`/`slice()`
# `.preserve = TRUE`, which `dplyr_col_modify.grouped_df()` leaves in
# `group_data()` for a verb that doesn't write a group column), else NULL.
# dplyr's own result is the oracle, so a verb that regroups (writing a group
# column, rows_*, and on a `.drop = TRUE` table arrange() or a join) yields
# NULL and `.drop = FALSE` factor expansion is not mistaken for fixed keys.
# Skipped when no group is empty.
# @keywords internal
result_fixed_keys <- function(result, new_groups) {
  gd <- dplyr::group_data(result)
  if (!any(lengths(gd$.rows) == 0L)) {
    return(NULL)
  }
  keys <- tibble::as_tibble(gd[new_groups])
  plain <- dplyr::grouped_df(dplyr::ungroup(result), new_groups,
                             drop = dplyr::group_by_drop_default(result))
  plain_keys <- tibble::as_tibble(dplyr::group_data(plain)[new_groups])
  if (nrow(keys) == nrow(plain_keys) && all(vctrs::vec_equal(keys, plain_keys, na_equal = TRUE))) {
    return(NULL)
  }
  keys
}

# TRUE if any column of `df` is a list-column (e.g. nest_by()/group_nest()'s
# nested-tibble column, nest_join()'s nested-match column). List columns have
# no cuDF representation, so gpu_fallback() returns these results as a plain
# data frame instead of attempting (and failing) to re-upload.
# @keywords internal
has_list_column <- function(df) {
  any(vapply(df, is.list, logical(1)))
}

# Collect the "other side" of a two-table verb (join, set operation, rows_*)
# to a plain data frame if it's a tbl_gpu (materializing pending lazy ops via
# collect()'s own compute() call), or pass a plain data.frame through as-is.
# @keywords internal
collect_other_side <- function(y) {
  if (is_tbl_gpu(y)) {
    return(restore_frame_meta(collect_frame(y), y))
  }
  y
}

# =============================================================================
# rowwise()
#
# reframe() moved to R/reframe.R (native for the quantile()/range()/literal
# shapes, F22; the rest is the whole-call fallback there).
#
# distinct() moved to R/distinct.R (GPU-native, Phase 3) -- see ast_distinct
# (R/ast.R), lower_distinct() (R/lower.R), gpu_distinct() (src/ops_distinct.cpp).
# transmute() moved to R/mutate.R, and glimpse() to R/glimpse.R (both
# GPU-native, Phase 3 task 4) -- see transmute.tbl_gpu()/parse_mutate_dots()
# (R/mutate.R) and glimpse.tbl_gpu() (R/glimpse.R).
# =============================================================================

#' CPU fallback methods for dplyr verbs without a GPU-native implementation
#'
#' `cuplyr` implements every dplyr generic on `tbl_gpu`, but not every one of
#' them runs natively on the GPU yet (see `scratchpad/todo.md` for the
#' phase-by-phase rollout). The methods documented here provide a
#' *transparent CPU fallback*: they materialize any pending lazy operations,
#' [collect()] `.data` to a tibble (restoring `group_by()` structure from
#' `.data`'s grouping so grouped semantics apply exactly as on a real
#' `grouped_df`), run the real dplyr verb on the CPU, and re-upload the
#' result via [tbl_gpu()] -- restoring grouping (read back from the verb's
#' actual result, since some verbs change it), execution mode (lazy stays
#' lazy), and factor levels.
#'
#' Verbs whose result isn't a data frame at all (e.g. [dplyr::pull()]'s
#' vector, [dplyr::group_split()]'s list), or whose result is a data frame
#' containing list-columns with no GPU representation (e.g.
#' [dplyr::nest_by()], [dplyr::group_nest()], [dplyr::nest_join()]'s nested
#' match column), are returned as plain R objects rather than re-uploaded.
#'
#' A GPU key match for `nest_join()`/`group_nest()`/`nest_by()` was measured and
#' does not pay (the time is building the nested tibbles on the host), so they
#' stay CPU fallbacks.
#'
#' Every fallback calls [cuplyr_fallback_notify()], so
#' `options(cuplyr.fallback = "warn")` (or `"error"`) can be used to surface
#' (or forbid) accidental fallbacks in code that's expected to stay fully
#' GPU-native.
#'
#' @param .data,x,y,tbl,.tbl A `tbl_gpu` object (or, for `y`/two-table verbs,
#'   optionally a plain `data.frame`).
#' @param ... Passed through to the corresponding dplyr verb.
#' @name fallback-verbs
#' @keywords internal
NULL

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rowwise
rowwise.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  gpu_fallback("rowwise", .data, function(tbl) {
    rlang::inject(dplyr::rowwise(tbl, !!!dots))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr do
#
# Phase 11 L7: do() IS an ordinary S3 generic (dplyr exports do.data.frame/
# do.grouped_df/do.rowwise_df/do.NULL, but no do.default) -- confirmed
# empirically (test-first, per the repo's STRICT mandate) that calling
# dplyr::do() on a tbl_gpu before this method existed hard-errored with "no
# applicable method for 'do' applied to an object of class \"c('tbl_gpu',
# 'tbl')\"", never reaching any fallback at all. gpu_fallback() already
# restores group_by() structure on the collected tibble when `.data` was
# grouped, so dispatching through it here is enough to route to real
# dplyr's do.grouped_df for a grouped tbl_gpu, or do.data.frame for an
# ungrouped one -- no bespoke per-group logic needed.
#
# `do()`'s named-dots form (e.g. `do(models = lm(y ~ x, data = .))`)
# produces a data frame with LIST columns (one model object per group) --
# `gpu_fallback()`'s existing `has_list_column()` check already returns
# that as a plain tibble rather than attempting (and failing) to re-upload
# it, exactly like `nest_by()`/`group_nest()`.
do.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  gpu_fallback("do", .data, function(tbl) {
    rlang::inject(dplyr::do(tbl, !!!dots))
  })
}

# =============================================================================
# slice() family
#
# slice()/slice_head()/slice_tail()/slice_min()/slice_max()/slice_sample()
# moved to R/slice.R (GPU-native for ungrouped input, Phase 3 task 2;
# grouped/`by=` Phase 5 task W7; slice_sample() Phase 5 task W8) -- see
# ast_slice (R/ast.R), lower_slice() (R/lower.R), src/ops_slice.cpp, and
# (for slice_sample()) slice_sample_native() in R/slice.R, which has no
# AST node at all. Each of those verbs still calls through to
# gpu_fallback() itself for the cases that remain out of scope: grouped
# input, on-the-fly `by=`/`.by=` grouping, and (for slice()) index
# expressions the GPU-native path can't evaluate without a data mask;
# slice_sample() falls back only for `weight_by=` (needs arbitrary R-side
# probability weighting `sample.int()` doesn't need to reproduce here,
# since gpu_fallback() re-runs real dplyr, which already gets it right).
# =============================================================================

# =============================================================================
# rename() / rename_with() / relocate() / pull()
#
# Moved to R/rename.R, R/relocate.R, R/pull.R (GPU-native/schema-only,
# Phase 3 task 3) -- see ast_rename (R/ast.R), lower_rename() (R/lower.R).
# rename()/rename_with() build an ast_rename node directly; relocate()
# reuses ast_select (a plain column reorder) and stacks an ast_rename node
# on top only when the relocate also renamed a column; pull() is a
# terminal verb with no AST node (materializes + collects, then picks out
# the requested column in R).
# =============================================================================

# =============================================================================
# count() / tally() / add_count() / add_tally()
#
# Moved to R/count.R (GPU-native, Phase 3 task 5) -- see count.tbl_gpu(),
# tally.tbl_gpu(), add_count.tbl_gpu() (all sugar over group_by()/
# summarise()/mutate()/arrange(), no CPU round-trip). add_count()'s
# per-row broadcast is mutate()'s general window/aggregate machinery
# (Phase 5; the narrow join-back-based grouped_aggregate_mutate() this used
# to route through was deleted in Phase 5 task W9). add_tally() is a plain
# (non-generic) dplyr function and needs no tbl_gpu method at all:
# R/group-by.R's tbl_vars.tbl_gpu() plus mutate()'s window/aggregate
# support are enough for the real dplyr::add_tally() to work transparently.
# =============================================================================
# Two-table verbs: joins without a GPU-native path, set operations, rows_*
#
# semi_join()/anti_join() moved to R/join.R (GPU-native via
# cudf::filtered_join, Phase 7 J2) -- see semi_join.tbl_gpu()/
# anti_join.tbl_gpu(), ast_join()'s "semi"/"anti" join_type (R/ast.R),
# lower_join() (R/lower.R), gpu_semi_anti_join() (src/ops_join.cpp).
# cross_join() moved to R/join.R (GPU-native via cudf::cross_join,
# Phase 7 J3) -- see cross_join.tbl_gpu(), ast_join()'s "cross" join_type
# (R/ast.R), lower_join() (R/lower.R), gpu_cross_join() (src/ops_join.cpp).
# union()/union_all()/intersect()/setdiff()/setequal()/symdiff() moved to
# R/sets.R (GPU-native, composed from bind_rows()/distinct()/semi_join()/
# anti_join(), Phase 8) -- see union.tbl_gpu() et al., and
# check_set_op_compatible() for the dplyr-mirrored column-compatibility
# checks. Each still falls back to CPU evaluation (via gpu_fallback(),
# same shape as every verb below) for the narrow factor-column edge cases
# R/sets.R's own docs describe.
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr nest_join
nest_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, keep = NULL,
                               name = NULL, ...,
                               na_matches = c("na", "never"), unmatched = "drop") {
  rlang::check_dots_empty0(...)
  # dplyr names the list column after the `y` expression (`as_label(enexpr(y))`);
  # taken before `y` is collected so it isn't the collected copy's name.
  name <- name %||% rlang::as_label(rlang::enexpr(y))
  user_env <- rlang::caller_env()
  y_tbl <- collect_other_side(y)
  # `by = character()`: dplyr's own deprecation warning would blame the
  # calling package (cuplyr), so warn here with the user's frame and mute
  # the copy dplyr raises inside the fallback -- exactly one warning.
  cross_by <- is_cross_by(by)
  if (cross_by) {
    warn_join_cross_by(user_env)
  }
  gpu_fallback("nest_join", x, function(tbl) {
    mute_cross_by_warning(
      dplyr::nest_join(tbl, y_tbl, by = by, copy = copy, keep = keep, name = name,
                       na_matches = na_matches, unmatched = unmatched),
      cross_by
    )
  })
}

# rows_insert()/rows_append()/rows_update()/rows_patch()/rows_upsert()/
# rows_delete() moved to R/rows.R (GPU-native, F26); their data-dependent errors
# and shapes outside the native allowlist still run through gpu_fallback().

# =============================================================================
# Grouping introspection & iteration
#
# group_data()/group_keys()/group_rows()/group_indices()/group_size()/
# n_groups() moved to R/group-by.R (GPU-native, Phase 9 task 1) -- see
# compute_group_data() (the shared `gpu_group_info()`-backed query) and
# each verb's own method there. The iteration family below
# (group_map()/group_modify()/group_nest()/nest_by())
# inherently produces R-side objects (a user-function's arbitrary return
# value, or a list-column data frame) and stays on the CPU fallback --
# group_map()'s .f needs real data frames, group_nest()'s list-column is
# caught by has_list_column(). group_split() is GPU-native (below).
# group_trim() is GPU-native (F27, R/group-by.R).
# =============================================================================

#' Split a GPU table into one GPU table per group (dplyr backend method)
#'
#' Native: [dplyr::group_split()] on a `tbl_gpu` groups by `...` (or uses the
#' existing groups), computes the group row indices on the GPU
#' (`compute_group_data()`: sorted keys, NA last, `.drop = FALSE` and
#' preserved empty groups), gathers the rows into group order and cuts the
#' table at the group sizes (`gpu_split_rows()`). The chunks are `tbl_gpu`
#' tables, as dplyr's chunks have the class of the ungrouped input, and the
#' `list_of` ptype is a 0-row `tbl_gpu`. A table left with no columns (every
#' column dropped by `.keep = FALSE`) is the one case that still goes through
#' the CPU fallback.
#'
#' @inheritParams dplyr::group_split
#' @param keep Defunct, use `.keep`.
#' @return A [vctrs::list_of()] of `tbl_gpu` tables.
#' @export
#' @importFrom dplyr group_split
group_split.tbl_gpu <- function(.tbl, ..., .keep = TRUE, keep = lifecycle::deprecated()) {
  if (lifecycle::is_present(keep)) {
    lifecycle::deprecate_stop("1.0.0", "group_split(keep = )", "group_split(.keep = )",
                              env = asNamespace("dplyr"))
  }

  if (length(.tbl$groups) > 0L) {
    if (...length() > 0L) {
      # same call dplyr's warn_ignores_dots() makes, so the text matches
      cli::cli_warn(paste(
        "Calling {.fn group_split} on a {.cls grouped_df} ignores `...`.",
        "Please use {.code group_by(..., .add = TRUE) |> group_split()}."
      ))
    }
    data <- .tbl
  } else {
    data <- group_by(.tbl, ...)
  }
  if (has_pending_ops(data)) {
    data <- compute(data)
  }

  out <- ungroup(data)
  if (!isTRUE(.keep) && length(data$groups) > 0L) {
    out <- dplyr::select(out, -dplyr::all_of(data$groups))
    if (has_pending_ops(out)) {
      out <- compute(out)
    }
  }
  if (length(out$schema$names) == 0L) {
    return(gpu_fallback("group_split", .tbl, function(tbl) {
      dplyr::group_split(tbl, ..., .keep = .keep)
    }))
  }

  rows <- compute_group_data(data)$.rows
  sizes <- lengths(rows)
  # no reordering needed when everything is one group (ungrouped input)
  idx0 <- if (length(data$groups) == 0L) integer() else as.integer(unlist(rows, use.names = FALSE)) - 1L
  ptrs <- wrap_gpu_call("group_split", gpu_split_rows(out$ptr, idx0, as.integer(sizes)))

  # vctrs::unspecified() columns: the select() above and group_by()'s
  # computed-key mutate both drop the marker, so take it from before them.
  # A computed key that overwrites an unspecified column is no longer one.
  unspecified <- union(
    data[["unspecified"]],
    setdiff(.tbl[["unspecified"]], data$groups)
  )
  wrap <- function(ptr) {
    carry_frame_meta(
      new_tbl_gpu(ptr = ptr, schema = out$schema, exec_mode = .tbl$exec_mode,
                  unspecified = unspecified),
      out
    )
  }
  ptype <- wrap(wrap_gpu_call("group_split", gpu_slice_head(out$ptr, 0, FALSE)))
  vctrs::new_list_of(lapply(ptrs, wrap), ptype = ptype)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_map
group_map.tbl_gpu <- function(.data, .f, ..., .keep = FALSE) {
  gpu_fallback("group_map", .data, function(tbl) {
    dplyr::group_map(tbl, .f, ..., .keep = .keep)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_modify
group_modify.tbl_gpu <- function(.data, .f, ..., .keep = FALSE) {
  gpu_fallback("group_modify", .data, function(tbl) {
    dplyr::group_modify(tbl, .f, ..., .keep = .keep)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_nest
group_nest.tbl_gpu <- function(.tbl, ..., .key = "data", keep = FALSE) {
  # `...` is forwarded as is (not enquos()+inject()) so `.drop = TRUE` reaches
  # group_by() as a value; dplyr itself raises the grouped-input dots warning.
  gpu_fallback("group_nest", .tbl, function(tbl) {
    dplyr::group_nest(tbl, ..., .key = .key, keep = keep)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr nest_by
nest_by.tbl_gpu <- function(.data, ..., .key = "data", .keep = FALSE) {
  # `...` forwarded unchanged, see group_nest.tbl_gpu().
  gpu_fallback("nest_by", .data, function(tbl) {
    dplyr::nest_by(tbl, ..., .key = .key, .keep = .keep)
  })
}

# =============================================================================
# Superseded sampling/top-n verbs
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr sample_n
sample_n.tbl_gpu <- function(tbl, size, replace = FALSE, weight = NULL,
                              .env = NULL, ...) {
  weight_quo <- rlang::enquo(weight)
  gpu_fallback("sample_n", tbl, function(t) {
    rlang::inject(dplyr::sample_n(t, size, replace = replace, weight = !!weight_quo, ...))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr sample_frac
sample_frac.tbl_gpu <- function(tbl, size = 1, replace = FALSE, weight = NULL,
                                 .env = NULL, ...) {
  weight_quo <- rlang::enquo(weight)
  gpu_fallback("sample_frac", tbl, function(t) {
    rlang::inject(dplyr::sample_frac(t, size, replace = replace, weight = !!weight_quo, ...))
  })
}
