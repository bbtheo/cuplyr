# Join operations for tbl_gpu

#' Parse a join specification into a 4-vector spec
#'
#' Returns `list(left, right, op, filter)`: `left`/`right`/`op` are parallel
#' character vectors (one entry per key/condition pair), `filter` is a single
#' scalar describing the whole join's filter semantics (`"none"` for a plain
#' equi/inequality join; `"max"`/`"min"` for `closest()` joins -- Phase 7 J7).
#' Every form this function currently understands (`NULL`, an unnamed
#' character vector, or a named character vector) produces an all-equi spec:
#' `op` is `"=="` for every key and `filter` is `"none"`. `join_by()` objects
#' (class `dplyr_join_by`) are rejected for now -- non-equi condition support
#' lands in Phase 7 J7, which lifts this rejection.
#' @keywords internal
parse_join_by <- function(by, x, y) {
  if (is.null(by)) {
    common <- intersect(x$schema$names, y$schema$names)
    if (length(common) == 0) {
      stop("No common columns for natural join. Specify `by` argument.",
           call. = FALSE)
    }
    return(list(left = common, right = common,
                op = rep("==", length(common)), filter = "none"))
  }

  if (inherits(by, "dplyr_join_by")) {
    stop("`join_by()` is not supported yet for tbl_gpu joins.", call. = FALSE)
  }

  if (is.character(by) && is.null(names(by))) {
    return(list(left = by, right = by,
                op = rep("==", length(by)), filter = "none"))
  }

  if (is.character(by) && !is.null(names(by))) {
    left_cols <- names(by)
    right_cols <- unname(by)
    empty_names <- left_cols == ""
    left_cols[empty_names] <- right_cols[empty_names]
    return(list(left = left_cols, right = right_cols,
                op = rep("==", length(left_cols)), filter = "none"))
  }

  stop("Invalid `by` specification. Use NULL, character vector, or named vector.",
       call. = FALSE)
}

#' Is a join spec entirely equi (`==`)?
#'
#' A spec with no condition pairs at all (`length(spec$op) == 0`) counts as
#' equi -- vacuously true, and matches `join_is_equi()`'s only current caller
#' ([resolve_join_keep()]) needing a sensible answer for the natural-join
#' (`by = NULL`, zero common columns already rejected earlier) edge case.
#' @keywords internal
join_is_equi <- function(spec) {
  length(spec$op) == 0 || all(spec$op == "==")
}

#' The equi (`==`) subset of a join spec
#'
#' Every current spec is 100% equi (Phase 7 J1); this returns `spec`
#' unchanged in that case. Kept as a named subsetting helper (rather than
#' inlining `spec$op == "=="` at call sites) so the future non-equi callers
#' (Phase 7 J7's hash-join key selection) have one place to read the equi
#' subset from.
#' @keywords internal
join_equi_spec <- function(spec) {
  is_eq <- spec$op == "=="
  list(left = spec$left[is_eq], right = spec$right[is_eq],
       op = spec$op[is_eq], filter = spec$filter)
}

#' The non-equi subset of a join spec
#'
#' Dormant until Phase 7 J7 (`build_join_ast()`/`gpu_cond_join()`), which
#' will lower this subset via `mixed_join`/`conditional_join`. Always empty
#' for now, since no code path currently produces `op` values other than
#' `"=="`.
#' @keywords internal
join_cond_spec <- function(spec) {
  is_eq <- spec$op == "=="
  list(left = spec$left[!is_eq], right = spec$right[!is_eq],
       op = spec$op[!is_eq], filter = spec$filter)
}

#' Resolve `keep`'s `NULL` default against a join spec
#'
#' Mirrors dplyr's own `keep = NULL` default resolution: `NULL` means `FALSE`
#' for an equi join and `TRUE` for a non-equi one (inequality/rolling/overlap
#' joins can't drop either side's key column, since there's no single shared
#' key value to keep). Explicit `keep = FALSE` on a non-equi join is a hard
#' error with dplyr's own text -- dormant in practice until Phase 7 J7 (no
#' spec is non-equi yet), but exercised here directly against a hand-built
#' spec in tests.
#' @param keep `NULL`, `TRUE`, or `FALSE`, as passed by the user
#' @param spec A join spec as returned by [parse_join_by()]
#' @return `TRUE`/`FALSE`
#' @keywords internal
resolve_join_keep <- function(keep, spec) {
  if (is.null(keep)) {
    return(!join_is_equi(spec))
  }

  if (isFALSE(keep) && !join_is_equi(spec)) {
    stop("Can't set `keep = FALSE` when using an inequality, rolling, or overlap join.",
         call. = FALSE)
  }

  keep
}

validate_join_cols <- function(cols, tbl, side) {
  missing <- setdiff(cols, tbl$schema$names)
  if (length(missing) > 0) {
    stop(side, " join columns not found: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  if (length(unique(cols)) != length(cols)) {
    stop(side, " join columns must be unique.", call. = FALSE)
  }
}

validate_key_types <- function(x, y, join_spec) {
  left_types <- x$schema$types[match(join_spec$left, x$schema$names)]
  right_types <- y$schema$types[match(join_spec$right, y$schema$names)]

  numeric_types <- c("INT8", "INT16", "INT32", "INT64", "FLOAT32", "FLOAT64", "BOOL8")

  for (i in seq_along(left_types)) {
    lt <- left_types[i]
    rt <- right_types[i]

    if (identical(lt, rt)) {
      next
    }

    if (lt %in% numeric_types && rt %in% numeric_types) {
      next
    }

    stop(sprintf(
      "Key column type mismatch: %s (%s) vs %s (%s). Cannot join incompatible types.",
      join_spec$left[i], lt, join_spec$right[i], rt
    ), call. = FALSE)
  }
}

build_join_output_info <- function(left_schema, right_schema, join_spec,
                                   suffix = c(".x", ".y"), keep = FALSE) {
  left_names <- left_schema$names
  right_names <- right_schema$names

  drop_right <- if (!isTRUE(keep)) join_spec$right else character(0)

  right_keep <- setdiff(right_names, drop_right)
  conflicts <- intersect(left_names, right_keep)

  left_out <- ifelse(left_names %in% conflicts,
                     paste0(left_names, suffix[1]),
                     left_names)
  right_out <- ifelse(right_keep %in% conflicts,
                      paste0(right_keep, suffix[2]),
                      right_keep)

  list(
    names = c(left_out, right_out),
    types = c(left_schema$types, right_schema$types[match(right_keep, right_names)]),
    origin = c(rep("left", length(left_out)), rep("right", length(right_out))),
    source_names = c(left_names, right_keep)
  )
}

build_join_schema <- function(left_schema, right_schema, join_spec,
                              suffix = c(".x", ".y"), keep = FALSE) {
  info <- build_join_output_info(left_schema, right_schema, join_spec,
                                 suffix = suffix, keep = keep)
  list(names = info$names, types = info$types)
}

estimate_gpu_bytes <- function(nrow, types) {
  bytes_per_type <- vapply(types, function(type) {
    switch(type,
      "FLOAT64" = 8,
      "FLOAT32" = 4,
      "INT64" = 8,
      "INT32" = 4,
      "INT16" = 2,
      "INT8" = 1,
      "BOOL8" = 1,
      "STRING" = 32,
      "TIMESTAMP_DAYS" = 4,
      "TIMESTAMP_MICROSECONDS" = 8,
      "TIMESTAMP_NANOSECONDS" = 8,
      "DICTIONARY32" = 4,
      8
    )
  }, numeric(1))

  data_bytes <- sum(nrow * bytes_per_type)
  mask_bytes <- ceiling(nrow / 8) * length(types)

  data_bytes + mask_bytes
}

warn_if_join_too_large <- function(join_type, x, y, join_spec, suffix, keep) {
  dims_x <- tryCatch(dim(x), error = function(e) NULL)
  dims_y <- tryCatch(dim(y), error = function(e) NULL)
  if (is.null(dims_x) || is.null(dims_y)) {
    return(invisible(NULL))
  }

  n_left <- dims_x[1]
  n_right <- dims_y[1]

  est_rows <- switch(join_type,
    "left" = n_left,
    "right" = n_right,
    "inner" = min(n_left, n_right),
    "full" = n_left + n_right,
    "semi" = n_left,
    "anti" = n_left,
    "cross" = n_left * n_right,
    n_left
  )

  # semi/anti never merge columns (output is x's own schema verbatim, see
  # infer_schema.ast_join's semi/anti branch) -- build_join_output_info()'s
  # merged-schema estimate would be wrong here (and keep/suffix are unused
  # for these join types), so estimate straight from x's own schema.
  schema <- if (join_type %in% c("semi", "anti")) {
    x$schema
  } else {
    build_join_schema(x$schema, y$schema, join_spec,
                      suffix = suffix, keep = keep)
  }
  est_bytes <- estimate_gpu_bytes(est_rows, schema$types)

  mem <- gpu_memory_state()
  if (!isTRUE(mem$available) || is.na(mem$free_bytes)) {
    return(invisible(NULL))
  }

  if (est_bytes > mem$free_bytes * 0.8) {
    warning(
      sprintf(
        "Join output is estimated at ~%.2f GB with only %.2f GB free on GPU. ",
        est_bytes / 1e9,
        mem$free_bytes / 1e9
      ),
      "Many-to-many joins can exceed this estimate; consider filtering or ",
      "calling gpu_gc() before joining.",
      call. = FALSE
    )
  }
}

#' Format a scalar value for a "not one of ..." style error message
#'
#' Strings render quoted (`"bogus"`); anything else falls back to `deparse()`
#' so the message stays readable without crashing on non-character input.
#' @keywords internal
format_join_arg_value <- function(value) {
  if (is.character(value) && length(value) == 1) {
    return(paste0('"', value, '"'))
  }
  paste(deparse(value), collapse = " ")
}

#' Stop for a join argument value that is legal but not yet implemented
#'
#' `multiple=`/`unmatched=`/`relationship=` all gained real signatures in
#' Phase 7 J1 but their actual cardinality-changing behavior only lands in
#' J5. A value that fails dplyr's own validation gets dplyr's own error
#' text (see the `validate_join_*()` functions below); a value that PASSES
#' validation but isn't the default is a legal dplyr call this package can't
#' honor correctly yet, so it must hard-stop rather than silently ignore the
#' argument and return possibly-wrong rows.
#' @keywords internal
join_arg_not_supported_yet <- function(arg_name, value) {
  stop(sprintf(
    "`%s = %s` is not supported yet for tbl_gpu joins (Phase 7 J5); only the default is currently honored.",
    arg_name, format_join_arg_value(value)
  ), call. = FALSE)
}

#' Validate `multiple=`, replicating dplyr's exact bad-value text
#'
#' Note dplyr's own message (from `vctrs::vec_locate_matches()`) omits the
#' offending value for this argument specifically (unlike `unmatched=`/
#' `relationship=`/`na_matches=`, which all name it) -- this is dplyr's
#' actual behavior, verified empirically against dplyr 1.2.1, not an
#' oversight here.
#' @keywords internal
validate_join_multiple <- function(multiple) {
  if (!is.character(multiple) || length(multiple) != 1) {
    stop("`multiple` must be a string.", call. = FALSE)
  }
  if (!multiple %in% c("all", "any", "first", "last")) {
    stop('`multiple` must be one of "all", "any", "first", or "last".',
         call. = FALSE)
  }
  multiple
}

#' Validate `unmatched=`, replicating dplyr's exact bad-value text
#'
#' `max_length` is 2 only for `inner_join()` (independent x-side/y-side
#' checks); every other join type allows length 1 only. `full_join()` has no
#' `unmatched=` parameter at all in dplyr (it never drops rows), so it never
#' calls this.
#' @keywords internal
validate_join_unmatched <- function(unmatched, max_length = 1L) {
  if (!is.character(unmatched)) {
    stop(sprintf("`unmatched` must be a character vector, not %s.",
                 format_join_arg_value(unmatched)), call. = FALSE)
  }
  if (length(unmatched) < 1 || length(unmatched) > max_length) {
    expected <- if (max_length == 1L) "length 1" else paste0("length 1 or ", max_length)
    stop(sprintf("`unmatched` must be %s, not %d.", expected, length(unmatched)),
         call. = FALSE)
  }
  bad <- setdiff(unmatched, c("drop", "error"))
  if (length(bad) > 0) {
    stop(sprintf('`unmatched` must be one of "drop" or "error", not "%s".', bad[1]),
         call. = FALSE)
  }
  unmatched
}

#' Validate `relationship=`, replicating dplyr's exact bad-value text
#' @keywords internal
validate_join_relationship <- function(relationship) {
  if (is.null(relationship)) {
    return(relationship)
  }
  if (!is.character(relationship) || length(relationship) != 1) {
    stop("`relationship` must be a string or character vector.", call. = FALSE)
  }
  valid <- c("one-to-one", "one-to-many", "many-to-one", "many-to-many")
  if (!relationship %in% valid) {
    stop(sprintf(
      '`relationship` must be one of "one-to-one", "one-to-many", "many-to-one", or "many-to-many", not "%s".',
      relationship
    ), call. = FALSE)
  }
  relationship
}

#' Validate `na_matches=`, replicating dplyr's exact bad-value text
#' @keywords internal
validate_join_na_matches <- function(na_matches) {
  if (!is.character(na_matches) || length(na_matches) != 1) {
    stop("`na_matches` must be a string or character vector.", call. = FALSE)
  }
  if (!na_matches %in% c("na", "never")) {
    stop(sprintf('`na_matches` must be one of "na" or "never", not "%s".', na_matches),
         call. = FALSE)
  }
  na_matches
}

#' @export
#' @importFrom dplyr left_join
left_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), ..., keep = NULL,
                              na_matches = "na", multiple = "all",
                              unmatched = "drop", relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  unmatched <- validate_join_unmatched(unmatched, max_length = 1L)
  relationship <- validate_join_relationship(relationship)

  if (!identical(multiple, "all")) join_arg_not_supported_yet("multiple", multiple)
  if (!identical(unmatched, "drop")) join_arg_not_supported_yet("unmatched", unmatched)
  if (!is.null(relationship)) join_arg_not_supported_yet("relationship", relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  keep <- resolve_join_keep(keep, join_spec)

  push_join("left", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = unmatched,
           relationship = relationship)
}

#' @export
#' @importFrom dplyr inner_join
inner_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), ..., keep = NULL,
                               na_matches = "na", multiple = "all",
                               unmatched = "drop", relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  unmatched <- validate_join_unmatched(unmatched, max_length = 2L)
  relationship <- validate_join_relationship(relationship)

  if (!identical(multiple, "all")) join_arg_not_supported_yet("multiple", multiple)
  if (!(length(unmatched) == 1 && identical(unmatched, "drop"))) {
    join_arg_not_supported_yet("unmatched", unmatched)
  }
  if (!is.null(relationship)) join_arg_not_supported_yet("relationship", relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  keep <- resolve_join_keep(keep, join_spec)

  push_join("inner", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = unmatched,
           relationship = relationship)
}

#' @export
#' @importFrom dplyr full_join
full_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), ..., keep = NULL,
                              na_matches = "na", multiple = "all",
                              relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  # full_join() has no `unmatched=` in dplyr (every row, matched or not, is
  # always kept), so it isn't accepted here either -- passing it hits `...`
  # and is silently ignored, same as dplyr's own `...`-must-be-empty dots
  # (dplyr hard-errors on that; this is a narrower gap, not a behavior this
  # task needs to close).
  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  relationship <- validate_join_relationship(relationship)

  if (!identical(multiple, "all")) join_arg_not_supported_yet("multiple", multiple)
  if (!is.null(relationship)) join_arg_not_supported_yet("relationship", relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  keep <- resolve_join_keep(keep, join_spec)

  push_join("full", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = "drop",
           relationship = relationship)
}

#' @export
#' @importFrom dplyr right_join
right_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), ..., keep = NULL,
                               na_matches = "na", multiple = "all",
                               unmatched = "drop", relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  unmatched <- validate_join_unmatched(unmatched, max_length = 1L)
  relationship <- validate_join_relationship(relationship)

  if (!identical(multiple, "all")) join_arg_not_supported_yet("multiple", multiple)
  if (!identical(unmatched, "drop")) join_arg_not_supported_yet("unmatched", unmatched)
  if (!is.null(relationship)) join_arg_not_supported_yet("relationship", relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  keep <- resolve_join_keep(keep, join_spec)

  # right_join(x, y) is native (Phase 7 J4): gpu_right_join()
  # (src/ops_join.cpp) computes matched (x, y) pairs via cudf::inner_join()
  # plus unmatched-y rows via a filtered_join anti-join, assembled so the
  # shared stable-sort/gather/coalesce pipeline (build_join_result(), used
  # by every mutating join type) reproduces dplyr's row-order contract:
  # x-matched rows in x's order, then unmatched-y rows appended last. This
  # is just another "right"-typed ast_join node -- push_join()
  # (R/execute.R) dispatches it through the same unified eager/lazy path as
  # the other join types, and lower_join()'s "right" case (R/lower.R) calls
  # gpu_right_join() with exactly the same argument shape as
  # gpu_left_join()/gpu_inner_join()/gpu_full_join(), so both schedules
  # agree on the resulting schema.
  push_join("right", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = unmatched,
           relationship = relationship)
}

#' `semi_join()`/`anti_join()`: native via `cudf::filtered_join`
#'
#' Native GPU implementation via `cudf::filtered_join` (Phase 7 J2). Unlike
#' the mutating joins, dplyr's own `semi_join()`/`anti_join()` signature has
#' no `suffix`/`keep`/`multiple`/`unmatched`/`relationship` arguments at all
#' (verified via `args(dplyr:::semi_join.data.frame)`: `x, y, by, copy, ...,
#' na_matches`) -- there's nothing to merge, so no column-collision handling
#' is needed. The result is x's own schema verbatim (a row subset of x),
#' and -- verified empirically against dplyr 1.2.1 -- x's `group_by()`
#' grouping is PRESERVED (these are filters, not the column-merging
#' mutating joins that `push_join()`/`propagate_groups()`'s D3 rule always
#' clears groups for); `push_join()` special-cases `join_type %in%
#' c("semi", "anti")` to carry `x$groups` forward instead of dropping them.
#' @export
#' @importFrom dplyr semi_join
semi_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, ...,
                              na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  na_matches <- validate_join_na_matches(na_matches)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  push_join("semi", x, y, join_spec, suffix = c(".x", ".y"), keep = FALSE,
           na_matches = na_matches)
}

#' See `semi_join.tbl_gpu()`'s roxygen just above -- same native
#' `filtered_join`-based implementation and grouping-preservation rationale,
#' `is_anti = TRUE` in the shared `gpu_semi_anti_join()` C++ entry point.
#' @export
#' @importFrom dplyr anti_join
anti_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, ...,
                              na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  na_matches <- validate_join_na_matches(na_matches)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  push_join("anti", x, y, join_spec, suffix = c(".x", ".y"), keep = FALSE,
           na_matches = na_matches)
}

#' `cross_join()`: native via `cudf::cross_join`
#'
#' Native GPU implementation (Phase 7 J3). Unlike every other join verb,
#' dplyr's own `cross_join()` signature has no `by`/`keep`/`na_matches`/
#' `multiple`/`unmatched`/`relationship` at all (verified via
#' `args(dplyr::cross_join)`: `x, y, ..., copy, suffix`) -- there are no keys
#' to match, every row of `x` is paired with every row of `y`. Verified
#' empirically against dplyr 1.2.1:
#' - Output column order is left-major: all of `x`'s columns (in their
#'   original order), then all of `y`'s (in their original order).
#' - Every COMMON column name (not just would-be join keys, since there are
#'   none) gets BOTH suffixes -- e.g. two tables both having `id`/`val`
#'   produces `id.x`/`val.x`/`id.y`/`val.y`, never a bare shared name. This
#'   is exactly `build_join_output_info()`'s own behavior for an all-common
#'   spec with `keep = TRUE` (its `drop_right` is always `character(0)` when
#'   `keep = TRUE`, regardless of what the spec's `left`/`right` contain), so
#'   an empty join spec (no keys at all) plus `keep = TRUE` reproduces it
#'   exactly -- see `ast_join()`'s `by`/`keep` fields below.
#' - Grouping is NOT unconditionally cleared, unlike `left_join()`/
#'   `inner_join()`/`full_join()`/`right_join()` (D3): dplyr's own
#'   `cross_join.data.frame()` ends with `dplyr_reconstruct(x_out, x)`,
#'   which keeps `x`'s grouping over exactly the group-var NAMES that
#'   survive, unchanged, in the output. A group column whose name is unique
#'   to `x` stays grouped; one whose name collides with `y`'s (and gets
#'   suffixed away, e.g. `"g"` -> `"g.x"`) is silently dropped from the
#'   grouping. `y`'s own grouping is always irrelevant. See `push_join()`'s
#'   (`R/execute.R`) `preserves_x_groups` handling, which reuses the exact
#'   same `intersect(x$groups, new_schema$names)` formula already used for
#'   `semi_join()`/`anti_join()` -- a colliding group column is renamed away
#'   from the new schema's names by `build_join_output_info()`'s own
#'   suffixing rule, so the intersection naturally excludes it with no
#'   cross-specific logic needed.
#' - 0-row `x` or `y` produces a 0-row result with the correct (merged)
#'   schema -- cudf's `cross_join()` handles this natively, no R-side
#'   special-casing needed.
#' - `copy = TRUE` with a plain `data.frame` `y` works like every other join.
#'
#' `gpu_cross_join()` (`src/ops_join.cpp`) throws if either side has 0
#' *columns* (verified in the `cudf/join/join.hpp` doc comment: "throw
#' cudf::logic_error if the number of columns in either left or right table
#' is 0") -- guarded here with a clearer, dplyr-flavored message before ever
#' reaching the GPU call, for both the eager and lazy schedules (the 0-column
#' case is knowable from the schemas alone, no execution needed).
#' @export
#' @importFrom dplyr cross_join
cross_join.tbl_gpu <- function(x, y, ..., copy = FALSE, suffix = c(".x", ".y")) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (length(x$schema$names) == 0 || length(y$schema$names) == 0) {
    stop("cross_join() requires both tables to have at least one column.",
         call. = FALSE)
  }

  # No keys at all -- an empty spec, `keep = TRUE` so build_join_output_info()
  # suffixes every common name (see roxygen above).
  join_spec <- list(left = character(0), right = character(0),
                    op = character(0), filter = "none")

  push_join("cross", x, y, join_spec, suffix = suffix, keep = TRUE,
           na_matches = "na")
}
