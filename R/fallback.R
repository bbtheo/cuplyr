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
#'   returned unchanged, never re-uploaded -- used for verbs that return
#'   metadata *about* the grouping/data rather than a transformed dataset
#'   (e.g. `group_data()`, `group_keys()`), even though that metadata happens
#'   to be a small data frame.
#' @return Either a new `tbl_gpu` (grouping, execution mode, and factor
#'   levels restored/round-tripped), or `expr_fn`'s raw return value
#'   unchanged when: `as_is = TRUE`; the result isn't a data frame at all
#'   (e.g. `pull()`'s vector, `group_split()`'s list); or the result is a
#'   data frame that contains list-columns (e.g. `nest_by()`/`group_nest()`/
#'   `nest_join()`'s nested-tibble columns), which have no GPU representation.
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

  tbl <- collect(materialized)

  # Restore dplyr grouping metadata on the CPU side so grouped semantics
  # (per-group aggregation, grouped slice_min/slice_max, etc.) work exactly
  # as they would on a real grouped_df.
  if (length(materialized$groups) > 0) {
    tbl <- dplyr::group_by(tbl, !!!rlang::syms(materialized$groups))
  }

  result <- expr_fn(tbl)

  if (as_is || !is.data.frame(result) || has_list_column(result)) {
    return(result)
  }

  # The verb may change grouping (e.g. summarise()-like verbs drop it,
  # reframe()/rowwise() may set it) -- read it back from the actual result
  # rather than assuming it's preserved.
  new_groups <- dplyr::group_vars(result)

  new_tbl <- tbl_gpu(dplyr::ungroup(result), lazy = FALSE)

  if (length(new_groups) > 0) {
    new_tbl <- dplyr::group_by(new_tbl, !!!rlang::syms(new_groups))
  }

  if (was_lazy) {
    new_tbl <- as_lazy(new_tbl)
  }

  new_tbl
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
    return(collect(y))
  }
  y
}

# =============================================================================
# transmute() / reframe() / rowwise()
#
# distinct() moved to R/distinct.R (GPU-native, Phase 3) -- see ast_distinct
# (R/ast.R), lower_distinct() (R/lower.R), gpu_distinct() (src/ops_distinct.cpp).
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
#' Verbs that only report metadata about the grouping/data (e.g.
#' [dplyr::group_data()], [dplyr::group_keys()]) are likewise always
#' returned as plain R objects.
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
#' @importFrom dplyr transmute
transmute.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  gpu_fallback("transmute", .data, function(tbl) {
    rlang::inject(dplyr::transmute(tbl, !!!dots))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr reframe
reframe.tbl_gpu <- function(.data, ..., .by = NULL) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(.by)
  gpu_fallback("reframe", .data, function(tbl) {
    rlang::inject(dplyr::reframe(tbl, !!!dots, .by = !!by_quo))
  })
}

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
#' @importFrom dplyr glimpse
glimpse.tbl_gpu <- function(x, width = NULL, ...) {
  materialized <- if (has_pending_ops(x)) compute(x) else x

  cuplyr_fallback_notify("glimpse", "glimpse()")

  tbl <- collect(materialized)
  if (length(materialized$groups) > 0) {
    tbl <- dplyr::group_by(tbl, !!!rlang::syms(materialized$groups))
  }

  dplyr::glimpse(tbl, width = width, ...)

  # glimpse() is a side-effecting print method: like dplyr's own
  # glimpse.data.frame(), it returns its input unchanged (invisibly) rather
  # than a transformed result, so there is nothing to re-upload -- return the
  # original tbl_gpu so pipe chains continue to operate on GPU.
  invisible(x)
}

# =============================================================================
# slice() family
#
# slice()/slice_head()/slice_tail()/slice_min()/slice_max() moved to
# R/slice.R (GPU-native for ungrouped input, Phase 3 task 2) -- see
# ast_slice (R/ast.R), lower_slice() (R/lower.R), src/ops_slice.cpp. Each of
# those verbs still calls through to gpu_fallback() itself for the cases
# that remain out of scope: grouped input, on-the-fly `by=`/`.by=`
# grouping, and (for slice()) index expressions the GPU-native path can't
# evaluate without a data mask. slice_sample() is untouched here: it always
# uses the CPU fallback regardless of grouping (needs dplyr's own RNG
# stream reproduced bit-for-bit -- a separate task).
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr slice_sample
slice_sample.tbl_gpu <- function(.data, ..., n, prop, by = NULL,
                                  weight_by = NULL, replace = FALSE) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  weight_quo <- rlang::enquo(weight_by)
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop
  gpu_fallback("slice_sample", .data, function(tbl) {
    rlang::inject(dplyr::slice_sample(tbl, !!!dots, !!!np, by = !!by_quo,
                                       weight_by = !!weight_quo, replace = replace))
  })
}

# =============================================================================
# rename() / rename_with() / relocate() / pull()
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rename
rename.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  gpu_fallback("rename", .data, function(tbl) {
    rlang::inject(dplyr::rename(tbl, !!!dots))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rename_with
rename_with.tbl_gpu <- function(.data, .fn, .cols = dplyr::everything(), ...) {
  cols_quo <- rlang::enquo(.cols)
  gpu_fallback("rename_with", .data, function(tbl) {
    rlang::inject(dplyr::rename_with(tbl, .fn, .cols = !!cols_quo, ...))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr relocate
relocate.tbl_gpu <- function(.data, ..., .before = NULL, .after = NULL) {
  dots <- rlang::enquos(...)
  before_quo <- rlang::enquo(.before)
  after_quo <- rlang::enquo(.after)
  gpu_fallback("relocate", .data, function(tbl) {
    rlang::inject(dplyr::relocate(tbl, !!!dots, .before = !!before_quo, .after = !!after_quo))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr pull
pull.tbl_gpu <- function(.data, var = -1, name = NULL, ...) {
  var_quo <- rlang::enquo(var)
  name_quo <- rlang::enquo(name)
  gpu_fallback("pull", .data, function(tbl) {
    rlang::inject(dplyr::pull(tbl, var = !!var_quo, name = !!name_quo, ...))
  }, as_is = TRUE)
}

# =============================================================================
# count() / tally() / add_count()
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr count
count.tbl_gpu <- function(x, ..., wt = NULL, sort = FALSE, name = NULL) {
  dots <- rlang::enquos(...)
  wt_quo <- rlang::enquo(wt)
  gpu_fallback("count", x, function(tbl) {
    rlang::inject(dplyr::count(tbl, !!!dots, wt = !!wt_quo, sort = sort, name = name))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr tally
tally.tbl_gpu <- function(x, wt = NULL, sort = FALSE, name = NULL) {
  wt_quo <- rlang::enquo(wt)
  gpu_fallback("tally", x, function(tbl) {
    rlang::inject(dplyr::tally(tbl, wt = !!wt_quo, sort = sort, name = name))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr add_count
add_count.tbl_gpu <- function(x, ..., wt = NULL, sort = FALSE, name = NULL) {
  dots <- rlang::enquos(...)
  wt_quo <- rlang::enquo(wt)
  gpu_fallback("add_count", x, function(tbl) {
    rlang::inject(dplyr::add_count(tbl, !!!dots, wt = !!wt_quo, sort = sort, name = name))
  })
}

# =============================================================================
# Two-table verbs: joins without a GPU-native path, set operations, rows_*
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr semi_join
semi_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("semi_join", x, function(tbl) {
    dplyr::semi_join(tbl, y_tbl, by = by, copy = copy, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr anti_join
anti_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("anti_join", x, function(tbl) {
    dplyr::anti_join(tbl, y_tbl, by = by, copy = copy, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr nest_join
nest_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, keep = NULL,
                               name = NULL, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("nest_join", x, function(tbl) {
    dplyr::nest_join(tbl, y_tbl, by = by, copy = copy, keep = keep, name = name, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr cross_join
cross_join.tbl_gpu <- function(x, y, ..., copy = FALSE, suffix = c(".x", ".y")) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("cross_join", x, function(tbl) {
    dplyr::cross_join(tbl, y_tbl, ..., copy = copy, suffix = suffix)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr union
union.tbl_gpu <- function(x, y, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("union", x, function(tbl) {
    dplyr::union(tbl, y_tbl, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr union_all
union_all.tbl_gpu <- function(x, y, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("union_all", x, function(tbl) {
    dplyr::union_all(tbl, y_tbl, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr intersect
intersect.tbl_gpu <- function(x, y, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("intersect", x, function(tbl) {
    dplyr::intersect(tbl, y_tbl, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr setdiff
setdiff.tbl_gpu <- function(x, y, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("setdiff", x, function(tbl) {
    dplyr::setdiff(tbl, y_tbl, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr setequal
setequal.tbl_gpu <- function(x, y, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("setequal", x, function(tbl) {
    dplyr::setequal(tbl, y_tbl, ...)
  }, as_is = TRUE)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr symdiff
symdiff.tbl_gpu <- function(x, y, ...) {
  y_tbl <- collect_other_side(y)
  gpu_fallback("symdiff", x, function(tbl) {
    dplyr::symdiff(tbl, y_tbl, ...)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rows_insert
rows_insert.tbl_gpu <- function(x, y, by = NULL, ..., conflict = c("error", "ignore"),
                                 copy = FALSE, in_place = FALSE) {
  if (isTRUE(in_place)) {
    stop("rows_insert(in_place = TRUE) is not supported for tbl_gpu; ",
         "assign the result instead.", call. = FALSE)
  }
  y_tbl <- collect_other_side(y)
  gpu_fallback("rows_insert", x, function(tbl) {
    dplyr::rows_insert(tbl, y_tbl, by = by, ..., conflict = conflict, copy = copy)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rows_append
rows_append.tbl_gpu <- function(x, y, ..., copy = FALSE, in_place = FALSE) {
  if (isTRUE(in_place)) {
    stop("rows_append(in_place = TRUE) is not supported for tbl_gpu; ",
         "assign the result instead.", call. = FALSE)
  }
  y_tbl <- collect_other_side(y)
  gpu_fallback("rows_append", x, function(tbl) {
    dplyr::rows_append(tbl, y_tbl, ..., copy = copy)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rows_update
rows_update.tbl_gpu <- function(x, y, by = NULL, ..., unmatched = c("error", "ignore"),
                                 copy = FALSE, in_place = FALSE) {
  if (isTRUE(in_place)) {
    stop("rows_update(in_place = TRUE) is not supported for tbl_gpu; ",
         "assign the result instead.", call. = FALSE)
  }
  y_tbl <- collect_other_side(y)
  gpu_fallback("rows_update", x, function(tbl) {
    dplyr::rows_update(tbl, y_tbl, by = by, ..., unmatched = unmatched, copy = copy)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rows_patch
rows_patch.tbl_gpu <- function(x, y, by = NULL, ..., unmatched = c("error", "ignore"),
                                copy = FALSE, in_place = FALSE) {
  if (isTRUE(in_place)) {
    stop("rows_patch(in_place = TRUE) is not supported for tbl_gpu; ",
         "assign the result instead.", call. = FALSE)
  }
  y_tbl <- collect_other_side(y)
  gpu_fallback("rows_patch", x, function(tbl) {
    dplyr::rows_patch(tbl, y_tbl, by = by, ..., unmatched = unmatched, copy = copy)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rows_upsert
rows_upsert.tbl_gpu <- function(x, y, by = NULL, ..., copy = FALSE, in_place = FALSE) {
  if (isTRUE(in_place)) {
    stop("rows_upsert(in_place = TRUE) is not supported for tbl_gpu; ",
         "assign the result instead.", call. = FALSE)
  }
  y_tbl <- collect_other_side(y)
  gpu_fallback("rows_upsert", x, function(tbl) {
    dplyr::rows_upsert(tbl, y_tbl, by = by, ..., copy = copy)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr rows_delete
rows_delete.tbl_gpu <- function(x, y, by = NULL, ..., unmatched = c("error", "ignore"),
                                 copy = FALSE, in_place = FALSE) {
  if (isTRUE(in_place)) {
    stop("rows_delete(in_place = TRUE) is not supported for tbl_gpu; ",
         "assign the result instead.", call. = FALSE)
  }
  y_tbl <- collect_other_side(y)
  gpu_fallback("rows_delete", x, function(tbl) {
    dplyr::rows_delete(tbl, y_tbl, by = by, ..., unmatched = unmatched, copy = copy)
  })
}

# =============================================================================
# Grouping introspection: group_data/group_keys/group_indices/group_size/
# n_groups (always returned as-is: metadata about the grouping, not a
# transformed dataset) and group_split/group_map/group_modify/group_nest/
# group_trim (transforming verbs; group_split/group_map naturally return a
# plain list, group_nest's list-column is caught by has_list_column())
# =============================================================================

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_data
group_data.tbl_gpu <- function(.data) {
  gpu_fallback("group_data", .data, function(tbl) {
    dplyr::group_data(tbl)
  }, as_is = TRUE)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_keys
group_keys.tbl_gpu <- function(.tbl, ...) {
  dots <- rlang::enquos(...)
  gpu_fallback("group_keys", .tbl, function(tbl) {
    rlang::inject(dplyr::group_keys(tbl, !!!dots))
  }, as_is = TRUE)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_indices
group_indices.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  gpu_fallback("group_indices", .data, function(tbl) {
    rlang::inject(dplyr::group_indices(tbl, !!!dots))
  }, as_is = TRUE)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_size
group_size.tbl_gpu <- function(x) {
  gpu_fallback("group_size", x, function(tbl) {
    dplyr::group_size(tbl)
  }, as_is = TRUE)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr n_groups
n_groups.tbl_gpu <- function(x) {
  gpu_fallback("n_groups", x, function(tbl) {
    dplyr::n_groups(tbl)
  }, as_is = TRUE)
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_split
group_split.tbl_gpu <- function(.tbl, ..., .keep = TRUE) {
  dots <- rlang::enquos(...)
  gpu_fallback("group_split", .tbl, function(tbl) {
    rlang::inject(dplyr::group_split(tbl, !!!dots, .keep = .keep))
  })
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
  dots <- rlang::enquos(...)
  gpu_fallback("group_nest", .tbl, function(tbl) {
    rlang::inject(dplyr::group_nest(tbl, !!!dots, .key = .key, keep = keep))
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr group_trim
group_trim.tbl_gpu <- function(.tbl, .drop = dplyr::group_by_drop_default(.tbl)) {
  gpu_fallback("group_trim", .tbl, function(tbl) {
    dplyr::group_trim(tbl, .drop = .drop)
  })
}

#' @rdname fallback-verbs
#' @export
#' @importFrom dplyr nest_by
nest_by.tbl_gpu <- function(.data, ..., .key = "data", .keep = FALSE) {
  dots <- rlang::enquos(...)
  gpu_fallback("nest_by", .data, function(tbl) {
    rlang::inject(dplyr::nest_by(tbl, !!!dots, .key = .key, .keep = .keep))
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
