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
#' don't require the swapped join to have actually run yet), shared by both
#' callers:
#'   - `right_join.tbl_gpu()` (eager path, this file), which executes the
#'     swapped join via the `left_join()` S3 generic on live `tbl_gpu` objects
#'   - `lower_join()` (`R/lower.R`), which lowers a lazy `ast_join` node by
#'     calling `gpu_left_join()` directly on raw GPU table pointers
#'
#' Callers are responsible for running the swapped join and then resolving
#' the reorder/select against its *actual* resulting column names via
#' [resolve_right_join_select_idx()] -- in that order, matching the original,
#' non-shared implementations this replaces (the swapped join must run first;
#' only then can columns be selected out of its result).
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
#' @keywords internal
build_right_join_via_left <- function(left_schema, right_schema, join_spec,
                                      suffix = c(".x", ".y"), keep = FALSE) {
  swapped_by <- list(left = join_spec$right, right = join_spec$left)

  desired_names <- build_join_schema(left_schema, right_schema, join_spec,
                                     suffix = suffix, keep = keep)$names

  list(swapped_by = swapped_by, desired_names = desired_names)
}

#' Resolve the column reorder/select for a right-join-via-swapped-left-join
#'
#' Given the desired final output column names (from
#' [build_right_join_via_left()]) and the actual column names produced by the
#' already-executed swapped `left_join()`, returns the 1-based indices that
#' select/reorder the latter into the former, erroring if any desired column
#' is missing (which should not happen for valid inputs -- this is a
#' consistency check).
#'
#' @param desired_names Character vector, target output column names.
#' @param current_names Character vector, actual column names of the
#'   executed swapped join's result.
#' @return Integer vector, 1-based indices into `current_names`.
#' @keywords internal
resolve_right_join_select_idx <- function(desired_names, current_names) {
  idx <- match(desired_names, current_names)
  if (any(is.na(idx))) {
    stop("Right join column reordering failed. Missing columns: ",
         paste(desired_names[is.na(idx)], collapse = ", "),
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

  if (identical(x$exec_mode, "lazy") || identical(y$exec_mode, "lazy")) {
    left_ast <- if (!is.null(x$lazy_ops)) x$lazy_ops else ast_source(x$schema)
    right_ast <- if (!is.null(y$lazy_ops)) y$lazy_ops else ast_source(y$schema)

    left_ast <- set_ast_source_ptr(left_ast, x$ptr)
    right_ast <- set_ast_source_ptr(right_ast, y$ptr)

    join_ast <- ast_join("left", left_ast, right_ast, join_spec,
                         keep = keep, suffix = suffix, na_matches = na_matches)

    left_schema <- infer_schema(left_ast)
    right_schema <- infer_schema(right_ast)
    new_schema <- build_join_schema(left_schema, right_schema, join_spec,
                                    suffix = suffix, keep = keep)

    return(new_tbl_gpu(
      ptr = NULL,
      schema = new_schema,
      lazy_ops = join_ast,
      groups = character(),
      exec_mode = "lazy"
    ))
  }

  if (!is.null(x$lazy_ops)) x <- compute(x)
  if (!is.null(y$lazy_ops)) y <- compute(y)

  left_key_idx <- match(join_spec$left, x$schema$names) - 1L
  right_key_idx <- match(join_spec$right, y$schema$names) - 1L
  right_drop <- if (!isTRUE(keep)) join_spec$right else character(0)
  right_drop_idx <- if (length(right_drop) > 0) {
    match(right_drop, y$schema$names) - 1L
  } else {
    integer(0)
  }

  warn_if_join_too_large("left", x, y, join_spec, suffix, keep)

  new_ptr <- wrap_gpu_call(
    "left_join",
    gpu_left_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  )
  new_schema <- build_join_schema(x$schema, y$schema, join_spec,
                                  suffix = suffix, keep = keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = character(),
    exec_mode = "eager"
  )
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

  if (identical(x$exec_mode, "lazy") || identical(y$exec_mode, "lazy")) {
    left_ast <- if (!is.null(x$lazy_ops)) x$lazy_ops else ast_source(x$schema)
    right_ast <- if (!is.null(y$lazy_ops)) y$lazy_ops else ast_source(y$schema)

    left_ast <- set_ast_source_ptr(left_ast, x$ptr)
    right_ast <- set_ast_source_ptr(right_ast, y$ptr)

    join_ast <- ast_join("inner", left_ast, right_ast, join_spec,
                         keep = keep, suffix = suffix, na_matches = na_matches)

    left_schema <- infer_schema(left_ast)
    right_schema <- infer_schema(right_ast)
    new_schema <- build_join_schema(left_schema, right_schema, join_spec,
                                    suffix = suffix, keep = keep)

    return(new_tbl_gpu(
      ptr = NULL,
      schema = new_schema,
      lazy_ops = join_ast,
      groups = character(),
      exec_mode = "lazy"
    ))
  }

  if (!is.null(x$lazy_ops)) x <- compute(x)
  if (!is.null(y$lazy_ops)) y <- compute(y)

  left_key_idx <- match(join_spec$left, x$schema$names) - 1L
  right_key_idx <- match(join_spec$right, y$schema$names) - 1L
  right_drop <- if (!isTRUE(keep)) join_spec$right else character(0)
  right_drop_idx <- if (length(right_drop) > 0) {
    match(right_drop, y$schema$names) - 1L
  } else {
    integer(0)
  }

  warn_if_join_too_large("inner", x, y, join_spec, suffix, keep)

  new_ptr <- wrap_gpu_call(
    "inner_join",
    gpu_inner_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  )
  new_schema <- build_join_schema(x$schema, y$schema, join_spec,
                                  suffix = suffix, keep = keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = character(),
    exec_mode = "eager"
  )
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

  if (identical(x$exec_mode, "lazy") || identical(y$exec_mode, "lazy")) {
    left_ast <- if (!is.null(x$lazy_ops)) x$lazy_ops else ast_source(x$schema)
    right_ast <- if (!is.null(y$lazy_ops)) y$lazy_ops else ast_source(y$schema)

    left_ast <- set_ast_source_ptr(left_ast, x$ptr)
    right_ast <- set_ast_source_ptr(right_ast, y$ptr)

    join_ast <- ast_join("full", left_ast, right_ast, join_spec,
                         keep = keep, suffix = suffix, na_matches = na_matches)

    left_schema <- infer_schema(left_ast)
    right_schema <- infer_schema(right_ast)
    new_schema <- build_join_schema(left_schema, right_schema, join_spec,
                                    suffix = suffix, keep = keep)

    return(new_tbl_gpu(
      ptr = NULL,
      schema = new_schema,
      lazy_ops = join_ast,
      groups = character(),
      exec_mode = "lazy"
    ))
  }

  if (!is.null(x$lazy_ops)) x <- compute(x)
  if (!is.null(y$lazy_ops)) y <- compute(y)

  left_key_idx <- match(join_spec$left, x$schema$names) - 1L
  right_key_idx <- match(join_spec$right, y$schema$names) - 1L
  right_drop <- if (!isTRUE(keep)) join_spec$right else character(0)
  right_drop_idx <- if (length(right_drop) > 0) {
    match(right_drop, y$schema$names) - 1L
  } else {
    integer(0)
  }

  warn_if_join_too_large("full", x, y, join_spec, suffix, keep)

  new_ptr <- wrap_gpu_call(
    "full_join",
    gpu_full_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  )
  new_schema <- build_join_schema(x$schema, y$schema, join_spec,
                                  suffix = suffix, keep = keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = character(),
    exec_mode = "eager"
  )
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

  plan <- build_right_join_via_left(x$schema, y$schema, join_spec,
                                    suffix = suffix, keep = keep)

  out <- left_join(y, x, by = plan$swapped_by, copy = copy,
                   suffix = rev(suffix), keep = TRUE, na_matches = na_matches)

  resolve_right_join_select_idx(plan$desired_names, out$schema$names)

  out <- dplyr::select(out, dplyr::all_of(plan$desired_names))
  out$schema$names <- plan$desired_names
  out
}
