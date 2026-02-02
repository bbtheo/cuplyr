#' Join two GPU tables
#'
#' These functions provide GPU-accelerated implementations of dplyr's
#' mutating joins. They combine columns from two GPU tables based on
#' matching keys.
#'
#' @param x A `tbl_gpu` object (left table).
#' @param y A `tbl_gpu` object (right table), or a data frame if `copy = TRUE`.
#' @param by A join specification created with [dplyr::join_by()], or a character
#'   vector of variables to join by. If `NULL`, the default, performs a natural
#'   join using all variables with common names.
#' @param copy If `y` is not a `tbl_gpu` and `copy = TRUE`, it will be converted
#'   to a GPU table. If `FALSE` (the default), an error is thrown.
#' @param suffix Character vector of length 2 specifying suffixes to add to
#'   disambiguate column names from `x` and `y`. Default is `c(".x", ".y")`.
#' @param keep Should the join keys from both tables be preserved in the output?
#'   Default is `FALSE`, which drops the join key columns from `y` when they
#'   have the same name as the corresponding column in `x`.
#' @param na_matches Should NA values match other NA values? Default is `"na"`
#'   which treats NA as a value. `"never"` is not yet supported.
#' @param ... Additional arguments (currently unused).
#'
#' @return A `tbl_gpu` object containing the joined data.
#'
#' @details
#' ## Join Types
#' - `left_join()`: Returns all rows from `x`, and all columns from `x` and `y`.
#'   Rows in `x` with no match in `y` will have NA values in the new columns.
#' - `right_join()`: Returns all rows from `y`, and all columns from `x` and `y`.
#'   Rows in `y` with no match in `x` will have NA values in the new columns.
#' - `inner_join()`: Returns all rows from `x` where there are matching values
#'   in `y`, and all columns from `x` and `y`.
#' - `full_join()`: Returns all rows and columns from both `x` and `y`. Rows
#'   without matches are filled with NA values.
#'
#' ## Performance
#' GPU joins can be significantly faster than CPU joins for large datasets.
#' The join algorithm uses hash-based matching on the GPU.
#'
#' ## Grouping
#' Joins drop any existing grouping. Use `group_by()` after joining to
#' re-establish groups.
#'
#' @seealso
#' [dplyr::left_join()], [dplyr::right_join()], [dplyr::inner_join()],
#' [dplyr::full_join()] for the dplyr equivalents.
#'
#' @examples
#' if (has_gpu()) {
#'   left_df <- data.frame(key = 1:3, x = c("a", "b", "c"))
#'   right_df <- data.frame(key = 2:4, y = c("x", "y", "z"))
#'
#'   gpu_left <- tbl_gpu(left_df)
#'   gpu_right <- tbl_gpu(right_df)
#'
#'   # Left join - keeps all rows from left table
#'   result <- left_join(gpu_left, gpu_right, by = "key") |> collect()
#'
#'   # Inner join - keeps only matching rows
#'   result <- inner_join(gpu_left, gpu_right, by = "key") |> collect()
#'
#'   # Join with different key names
#'   df1 <- data.frame(id = 1:3, val = letters[1:3])
#'   df2 <- data.frame(key = 2:4, other = LETTERS[1:3])
#'   result <- tbl_gpu(df1) |>
#'     left_join(tbl_gpu(df2), by = c("id" = "key")) |>
#'     collect()
#' }
#'
#' @name join
NULL

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Parse the `by` argument into left/right column lists
#' @noRd
parse_join_by <- function(by, x, y) {
  # Case 1: by = NULL -> natural join (common columns)
  if (is.null(by)) {
    common <- intersect(x$schema$names, y$schema$names)
    if (length(common) == 0) {
      stop("No common columns for natural join. Specify `by` argument.",
           call. = FALSE)
    }
    return(list(left = common, right = common))
  }

  # Case 2: by = c("a", "b") -> same column names in both tables
  # Case 3: by = c("a" = "b") -> left$a joins to right$b
  if (is.character(by)) {
    if (is.null(names(by))) {
      return(list(left = by, right = by))
    } else {
      left_cols <- names(by)
      right_cols <- unname(by)
      empty_names <- left_cols == ""
      left_cols[empty_names] <- right_cols[empty_names]
      if (length(left_cols) != length(right_cols)) {
        stop("`by` must map left and right columns 1:1.", call. = FALSE)
      }
      return(list(left = left_cols, right = right_cols))
    }
  }

  if (inherits(by, "dplyr_join_by")) {
    stop("`join_by()` is not supported yet for tbl_gpu joins.", call. = FALSE)
  }

  stop("Unsupported `by` specification. Use a character vector or NULL.",
       call. = FALSE)
}

#' Validate columns exist in table
#' @noRd
validate_join_cols <- function(cols, tbl, side) {
  missing <- setdiff(cols, tbl$schema$names)
  if (length(missing) > 0) {
    stop(sprintf("%s table missing join columns: %s\nAvailable: %s",
                 side, paste(missing, collapse = ", "),
                 paste(tbl$schema$names, collapse = ", ")),
         call. = FALSE)
  }
  if (anyDuplicated(cols)) {
    stop(sprintf("%s join columns must be unique.", side), call. = FALSE)
  }
}

#' Validate key column types are compatible
#' @noRd
validate_key_types <- function(x, y, join_spec) {
  left_types <- x$schema$types[match(join_spec$left, x$schema$names)]
  right_types <- y$schema$types[match(join_spec$right, y$schema$names)]

  # Define compatible type groups
  numeric_types <- c("INT32", "INT64", "FLOAT32", "FLOAT64")

  for (i in seq_along(left_types)) {
    lt <- left_types[i]
    rt <- right_types[i]

    # Same type is always ok
    if (lt == rt) next

    # Numeric types are compatible with each other
    if (lt %in% numeric_types && rt %in% numeric_types) next

    # Otherwise incompatible
    stop(sprintf(
      "Key column type mismatch: %s (%s) vs %s (%s). Cannot join incompatible types.",
      join_spec$left[i], lt, join_spec$right[i], rt
    ), call. = FALSE)
  }
}

#' Build result schema after join
#' @noRd
build_join_schema <- function(x, y, join_spec, suffix = c(".x", ".y"), keep = FALSE) {
  left_names <- x$schema$names
  left_types <- x$schema$types

  # Determine which right columns to drop
 # Only drop right key columns when: keep = FALSE AND names match
  drop_right <- join_spec$right[!keep & join_spec$left == join_spec$right]
  right_drop_idx <- match(drop_right, y$schema$names)
  right_keep_idx <- setdiff(seq_along(y$schema$names), right_drop_idx)
  right_names <- y$schema$names[right_keep_idx]
  right_types <- y$schema$types[right_keep_idx]

  # Handle name conflicts with suffix
  conflicts <- intersect(left_names, right_names)
  if (length(conflicts) > 0) {
    left_names[left_names %in% conflicts] <- paste0(
      left_names[left_names %in% conflicts], suffix[1]
    )
    right_names[right_names %in% conflicts] <- paste0(
      right_names[right_names %in% conflicts], suffix[2]
    )
  }

  list(
    names = c(left_names, right_names),
    types = c(left_types, right_types)
  )
}

#' Compute which right columns to drop (0-based indices for C++)
#' @noRd
get_right_drop_indices <- function(y, join_spec, keep) {
  if (keep) {
    return(integer(0))
  }
  # Only drop when left and right key names match
  drop_cols <- join_spec$right[join_spec$left == join_spec$right]
  if (length(drop_cols) == 0) {
    return(integer(0))
  }
  match(drop_cols, y$schema$names) - 1L
}

#' Estimate GPU bytes for a result based on row count + column types
#' @noRd
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

#' Warn if estimated join output may exceed available GPU memory
#' @noRd
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

  schema <- build_join_schema(x, y, join_spec, suffix, keep)
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

# -----------------------------------------------------------------------------
# S3 Methods
# -----------------------------------------------------------------------------

#' @rdname join
#' @export
#' @importFrom dplyr left_join
left_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), keep = FALSE,
                              na_matches = c("na", "never"), ...) {
  na_matches <- match.arg(na_matches)
  if (na_matches != "na") {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  if (!is_tbl_gpu(y)) {
    if (copy) {
      y <- tbl_gpu(y)
    } else {
      stop("y must be a tbl_gpu. Use `copy = TRUE` to convert automatically.",
           call. = FALSE)
    }
  }

  join_spec <- parse_join_by(by, x, y)

  # Informative message for natural join (matches dplyr behavior)
  if (is.null(by)) {
    message("Joining with `by = join_by(",
            paste(join_spec$left, collapse = ", "), ")`")
  }

  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  warn_if_join_too_large("left", x, y, join_spec, suffix, keep)

  left_key_idx <- match(join_spec$left, x$schema$names) - 1L
  right_key_idx <- match(join_spec$right, y$schema$names) - 1L
  right_drop_idx <- get_right_drop_indices(y, join_spec, keep)

  new_ptr <- wrap_gpu_call(
    "left_join",
    gpu_left_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  )
  new_schema <- build_join_schema(x, y, join_spec, suffix, keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = character()
  )
}

#' @rdname join
#' @export
#' @importFrom dplyr right_join
right_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), keep = FALSE,
                               na_matches = c("na", "never"), ...) {
  na_matches <- match.arg(na_matches)
  if (na_matches != "na") {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  if (!is_tbl_gpu(y)) {
    if (copy) {
      y <- tbl_gpu(y)
    } else {
      stop("y must be a tbl_gpu. Use `copy = TRUE` to convert automatically.",
           call. = FALSE)
    }
  }

  # Implement via swapped left_join
  # Reverse the by specification
  by_reversed <- by
  if (is.character(by) && !is.null(names(by)) && any(names(by) != "")) {
    by_reversed <- setNames(names(by), by)
    empty <- names(by_reversed) == ""
    by_reversed[empty] <- by[empty]
  }

  # Note: suffix is reversed because we're swapping x and y
  out <- left_join(y, x, by = by_reversed, copy = FALSE,
                   suffix = rev(suffix), keep = keep,
                   na_matches = na_matches, ...)

  # Reorder columns to match dplyr right_join output: x columns first, then y columns
  # Must account for potential suffix modifications to column names
  out_names <- out$schema$names
  x_orig_names <- x$schema$names

  # Find the actual output names for x columns (may have suffix applied)
  x_names_in_output <- character(length(x_orig_names))
  for (i in seq_along(x_orig_names)) {
    nm <- x_orig_names[i]
    if (nm %in% out_names) {
      x_names_in_output[i] <- nm
    } else if (paste0(nm, suffix[1]) %in% out_names) {
      x_names_in_output[i] <- paste0(nm, suffix[1])
    } else {
      # Column was a join key that got dropped
      x_names_in_output[i] <- NA_character_
    }
  }

  x_cols_present <- x_names_in_output[!is.na(x_names_in_output)]
  y_cols_in_output <- setdiff(out_names, x_cols_present)

  desired_order <- c(x_cols_present, y_cols_in_output)
  select(out, dplyr::all_of(desired_order))
}

#' @rdname join
#' @export
#' @importFrom dplyr inner_join
inner_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), keep = FALSE,
                               na_matches = c("na", "never"), ...) {
  na_matches <- match.arg(na_matches)
  if (na_matches != "na") {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  if (!is_tbl_gpu(y)) {
    if (copy) {
      y <- tbl_gpu(y)
    } else {
      stop("y must be a tbl_gpu. Use `copy = TRUE` to convert automatically.",
           call. = FALSE)
    }
  }

  join_spec <- parse_join_by(by, x, y)

  if (is.null(by)) {
    message("Joining with `by = join_by(",
            paste(join_spec$left, collapse = ", "), ")`")
  }

  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  warn_if_join_too_large("inner", x, y, join_spec, suffix, keep)

  left_key_idx <- match(join_spec$left, x$schema$names) - 1L
  right_key_idx <- match(join_spec$right, y$schema$names) - 1L
  right_drop_idx <- get_right_drop_indices(y, join_spec, keep)

  new_ptr <- wrap_gpu_call(
    "inner_join",
    gpu_inner_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  )
  new_schema <- build_join_schema(x, y, join_spec, suffix, keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = character()
  )
}

#' @rdname join
#' @export
#' @importFrom dplyr full_join
full_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), keep = FALSE,
                              na_matches = c("na", "never"), ...) {
  na_matches <- match.arg(na_matches)
  if (na_matches != "na") {
    stop("`na_matches = \"never\"` is not supported yet for tbl_gpu joins.",
         call. = FALSE)
  }

  if (!is_tbl_gpu(y)) {
    if (copy) {
      y <- tbl_gpu(y)
    } else {
      stop("y must be a tbl_gpu. Use `copy = TRUE` to convert automatically.",
           call. = FALSE)
    }
  }

  join_spec <- parse_join_by(by, x, y)

  if (is.null(by)) {
    message("Joining with `by = join_by(",
            paste(join_spec$left, collapse = ", "), ")`")
  }

  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)
  warn_if_join_too_large("full", x, y, join_spec, suffix, keep)

  left_key_idx <- match(join_spec$left, x$schema$names) - 1L
  right_key_idx <- match(join_spec$right, y$schema$names) - 1L
  right_drop_idx <- get_right_drop_indices(y, join_spec, keep)

  new_ptr <- wrap_gpu_call(
    "full_join",
    gpu_full_join(x$ptr, y$ptr, left_key_idx, right_key_idx, right_drop_idx)
  )
  new_schema <- build_join_schema(x, y, join_spec, suffix, keep)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = character()
  )
}
