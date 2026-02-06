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
    return(list(left = names(by), right = unname(by)))
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

  new_ptr <- gpu_left_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  new_schema <- build_join_schema(x$schema, y$schema, join_spec,
                                  suffix = suffix, keep = keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = list(),
    groups = character(),
    exec_mode = "eager"
  )
}

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

  new_ptr <- gpu_inner_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  new_schema <- build_join_schema(x$schema, y$schema, join_spec,
                                  suffix = suffix, keep = keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = list(),
    groups = character(),
    exec_mode = "eager"
  )
}

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

  new_ptr <- gpu_full_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  new_schema <- build_join_schema(x$schema, y$schema, join_spec,
                                  suffix = suffix, keep = keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = list(),
    groups = character(),
    exec_mode = "eager"
  )
}

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

  # Swap for left_join implementation
  by_swapped <- list(left = join_spec$right, right = join_spec$left)
  out <- left_join(y, x, by = by_swapped, copy = copy,
                   suffix = rev(suffix), keep = TRUE, na_matches = na_matches)

  desired <- build_join_schema(x$schema, y$schema, join_spec,
                               suffix = suffix, keep = keep)$names
  current <- out$schema$names
  idx <- match(desired, current)
  if (any(is.na(idx))) {
    stop("Right join column reordering failed. Missing columns: ",
         paste(desired[is.na(idx)], collapse = ", "), call. = FALSE)
  }

  out <- dplyr::select(out, dplyr::all_of(desired))
  out$schema$names <- desired
  out
}
