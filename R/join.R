# Join operations for tbl_gpu

# Parse join specification
parse_join_by <- function(by, x, y) {
  if (is.null(by)) {
    common <- intersect(x$schema$names, y$schema$names)
    if (length(common) == 0) {
      stop("No common columns for natural join. Specify `by` argument.",
           call. = FALSE)
    }
    return(list(left = common, right = common))
  }

  if (inherits(by, "dplyr_join_by")) {
    stop("`join_by()` is not supported yet for tbl_gpu joins.", call. = FALSE)
  }

  # Already-parsed spec (e.g. `swapped_by` built by
  # build_right_join_via_left() for the right_join()-via-left_join() plan):
  # a plain list with character `left`/`right` elements. Accept and return
  # as-is so callers can pass a pre-built spec straight through without
  # re-encoding it as a named character vector.
  if (is.list(by) && !is.data.frame(by) &&
      is.character(by$left) && is.character(by$right)) {
    return(list(left = by$left, right = by$right))
  }

  if (is.character(by) && is.null(names(by))) {
    return(list(left = by, right = by))
  }

  if (is.character(by) && !is.null(names(by))) {
    left_cols <- names(by)
    right_cols <- unname(by)
    empty_names <- left_cols == ""
    left_cols[empty_names] <- right_cols[empty_names]
    return(list(left = left_cols, right = right_cols))
  }

  stop("Invalid `by` specification. Use NULL, character vector, or named vector.",
       call. = FALSE)
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

#' Build the join spec and desired output schema for right_join(),
#' implemented via a swapped left_join()
#'
#' `right_join(x, y)` is implemented as `left_join(y, x)` with the key sides
#' swapped and `keep = TRUE` forced (so both original key columns survive the
#' join uncombined), followed by reordering/selecting down to the column set
#' and names a native right join would produce. This helper computes the two
#' pieces of that plan that are pure functions of the input schemas (i.e. that
#' don't require the swapped join to have actually run yet).
#'
#' `right_join.tbl_gpu()` (this file) always builds a `"right"`-typed
#' `ast_join` node and hands it to `push_join()` (`R/execute.R`), which
#' lowers it -- in both the eager and lazy schedules -- via `lower_join()`'s
#' `"right"` case (`R/lower.R`). That is the single caller of this helper: it
#' calls `gpu_left_join()` on the swapped sides, then this helper's plan
#' (via `resolve_right_join_select_idx()`) to reorder/rename the raw output.
#'
#' Unlike the swapped join itself, the reorder/select plan resolved by
#' [resolve_right_join_select_idx()] is a pure function of the schemas too --
#' it matches desired output columns to the swapped join's (formulaic, always
#' `keep = TRUE`) raw output columns by origin table + raw source name, not
#' by the swapped join's actual runtime column names -- so callers may
#' compute it before or after actually running the swapped join.
#'
#' @param left_schema Schema of the (original) left table.
#' @param right_schema Schema of the (original) right table.
#' @param join_spec List with `left`/`right` character vectors of join key
#'   column names, as returned by `parse_join_by()`.
#' @param suffix Suffix pair for non-key name collisions, as in `right_join()`.
#' @param keep Whether to keep both key columns, as in `right_join()`.
#' @return A list with:
#'   - `swapped_by`: the join spec to pass to `left_join(y, x, by = ...)`
#'   - `desired_names`: the output column names a native right join would have
#'   - `desired_types`: the output column GPU types, parallel to `desired_names`
#'   - `desired_origin`: for each `desired_names` entry, whether it is
#'     sourced from the original `"left"` (`x`) or `"right"` (`y`) table
#'   - `desired_source`: for each `desired_names` entry, the raw column name
#'     in its origin table's schema (before any suffixing)
#' @keywords internal
build_right_join_via_left <- function(left_schema, right_schema, join_spec,
                                      suffix = c(".x", ".y"), keep = FALSE) {
  swapped_by <- list(left = join_spec$right, right = join_spec$left)

  desired_info <- build_join_output_info(left_schema, right_schema, join_spec,
                                         suffix = suffix, keep = keep)

  desired_origin <- desired_info$origin
  desired_source <- desired_info$source_names

  if (!isTRUE(keep)) {
    # `keep = FALSE`'s single output key column carries x's (the original
    # left table's) display name and position, per dplyr's `by = c(x = y)`
    # naming convention -- that part of desired_info is correct as-is.
    # But because a right_join is driven by y (every row of y survives) and
    # the swapped left_join(y, x) NA-fills x's columns for right-only rows,
    # sourcing that key column's *values* from x (as build_join_output_info
    # naturally does, since it's the left-origin copy) reproduces the
    # coalescing gap documented in scratchpad/unification_design.md Part C:
    # right-only rows would get NA instead of the key value. y's raw key
    # column is always present (right_join keeps every y row), so re-point
    # just the key columns' value source to the right (y) side; the display
    # name/type contract (desired_names/desired_types) is untouched.
    source_in_left <- ifelse(desired_info$origin == "left",
                             desired_info$source_names, NA_character_)
    key_pos <- match(join_spec$left, source_in_left)
    desired_origin[key_pos] <- "right"
    desired_source[key_pos] <- join_spec$right
  }

  list(
    swapped_by = swapped_by,
    desired_names = desired_info$names,
    desired_types = desired_info$types,
    desired_origin = desired_origin,
    desired_source = desired_source
  )
}

#' Resolve the column reorder/select for a right-join-via-swapped-left-join
#'
#' The swapped `left_join(y, x, ...)` used to implement `right_join(x, y)` is
#' always executed with `keep = TRUE` internally (so both original join-key
#' columns survive, uncombined), regardless of what the caller's `keep`
#' actually requested. That means its raw output column names never contain
#' a bare, unsuffixed key name when the two tables' keys share a name -- only
#' the suffixed variants (e.g. `"id.x"`/`"id.y"`) do. A plain name-based match
#' of [build_right_join_via_left()]'s `desired_names` (which, for
#' `keep = FALSE`, wants a single unsuffixed key column) against those raw
#' names would therefore always fail to find it.
#'
#' Instead, this matches columns by *origin* (which original table, `x` or
#' `y`, they are sourced from) and *raw source name* -- a pair that uniquely
#' identifies a physical column regardless of what suffix it ends up
#' displayed with. The desired schema's origins are relative to the
#' original, unswapped `x`/`y`; the swapped join's own output-info origins
#' are relative to *its* call (`"left"` = original `y`, `"right"` = original
#' `x`), so they're translated back before comparing.
#'
#' Key-coalescing note: because the swapped join is a left join keeping every
#' row of the original right table (`y`), the copy of the key column sourced
#' from the original *left* table (`x`) is `NULL` for right-only rows (`x`
#' has no matching row at all). [build_right_join_via_left()] handles this by
#' re-pointing the `keep = FALSE` key column's *source* (not its display
#' name, which still follows dplyr's `by = c(x_col = y_col)` convention) to
#' the always-present `y` copy for such rows, so `desired_origin`/
#' `desired_source` here already reflect the coalesced source -- this
#' function just needs to find the matching raw column, whichever side it
#' now points at. This mirrors the same-shaped `full_join()` join-key
#' coalescing fix in `src/ops_join.cpp` (see `test-join.R`).
#'
#' @param plan The list returned by [build_right_join_via_left()].
#' @param left_schema,right_schema Schemas of the original (unswapped) left
#'   and right tables.
#' @param suffix Suffix pair as passed to `right_join()`.
#' @return Integer vector, 1-based indices into the swapped join's raw output
#'   columns that select/reorder/rename it into `plan$desired_names`.
#' @keywords internal
resolve_right_join_select_idx <- function(plan, left_schema, right_schema, suffix) {
  actual_info <- build_join_output_info(right_schema, left_schema, plan$swapped_by,
                                        suffix = rev(suffix), keep = TRUE)

  # actual_info$origin is relative to the swapped call: "left" = the original
  # right table (y), "right" = the original left table (x). Flip it back to
  # original left/right terms so it's comparable to plan$desired_origin.
  actual_origin <- ifelse(actual_info$origin == "left", "right", "left")
  actual_key <- paste(actual_origin, actual_info$source_names, sep = "\r")
  desired_key <- paste(plan$desired_origin, plan$desired_source, sep = "\r")

  idx <- match(desired_key, actual_key)
  if (any(is.na(idx))) {
    stop("Right join column reordering failed. Missing columns: ",
         paste(plan$desired_names[is.na(idx)], collapse = ", "),
         call. = FALSE)
  }
  idx
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
    n_left
  )

  schema <- build_join_schema(x$schema, y$schema, join_spec,
                              suffix = suffix, keep = keep)
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

#' @export
#' @importFrom dplyr left_join
left_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), ..., keep = FALSE,
                              na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!identical(na_matches, "na")) {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  push_join("left", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches)
}

#' @export
#' @importFrom dplyr inner_join
inner_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), ..., keep = FALSE,
                               na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!identical(na_matches, "na")) {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  push_join("inner", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches)
}

#' @export
#' @importFrom dplyr full_join
full_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), ..., keep = FALSE,
                              na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!identical(na_matches, "na")) {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  push_join("full", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches)
}

#' @export
#' @importFrom dplyr right_join
right_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), ..., keep = FALSE,
                               na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!identical(na_matches, "na")) {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  # right_join(x, y) is implemented as a swapped left_join(y, x) (keeping
  # every row of y, NA-filling unmatched x columns), reordered/renamed down
  # to the column set and names a native right join would produce. This is
  # just another "right"-typed ast_join node -- push_join() (R/execute.R)
  # dispatches it through the same unified eager/lazy path as the other join
  # types, and lower_join()'s "right" case (R/lower.R) performs the actual
  # swap + positional gpu_select() using the shared plan helpers
  # (build_right_join_via_left() / resolve_right_join_select_idx()), so both
  # schedules agree on the resulting schema.
  push_join("right", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches)
}
