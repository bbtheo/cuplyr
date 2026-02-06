# Bind operations for tbl_gpu
# bind_rows: vertical concatenation (stacking rows)
# bind_cols: horizontal concatenation (adding columns)

# =============================================================================
# bind_cols - S3 generic that dispatches based on first argument
# =============================================================================

#' Bind multiple data frames/tables by column
#'
#' Combines objects horizontally by adding columns. For tbl_gpu objects,
#' operations are performed on the GPU. For other objects, delegates to dplyr.
#'
#' @param ... Objects to bind (tbl_gpu, data.frame, or a list of these)
#' @param .name_repair How to handle duplicate column names
#'
#' @return Combined data frame or tbl_gpu
#' @export
bind_cols <- function(..., .name_repair = c("unique", "universal",
                                             "check_unique", "minimal")) {
  UseMethod("bind_cols")
}

#' @export
bind_cols.tbl_gpu <- function(..., .name_repair = c("unique", "universal",
                                                     "check_unique", "minimal")) {
  dots <- list(...)
  .name_repair <- match.arg(.name_repair)
  bind_cols_gpu(dots, .name_repair = .name_repair)
}


#' Bind multiple data frames/tables by row
#'
#' Combines objects vertically by stacking rows. For tbl_gpu objects,
#' operations are performed on the GPU. For other objects, delegates to dplyr.
#'
#' @param ... Objects to bind (tbl_gpu, data.frame, or a list of these)
#' @param .id Optional column name to identify source tables
#'
#' @return Combined data frame or tbl_gpu
#' @export
bind_rows <- function(..., .id = NULL) {
  UseMethod("bind_rows")
}

#' @export
bind_rows.tbl_gpu <- function(..., .id = NULL) {
  dots <- list(...)
  bind_rows_gpu(dots, .id = .id)
}



# =============================================================================
# GPU implementations
# =============================================================================

#' GPU bind_cols implementation
#' @param dots List of tables to bind
#' @param .name_repair Name repair method
#' @return A tbl_gpu
#' @keywords internal
bind_cols_gpu <- function(dots, .name_repair = "unique") {
  # Flatten if passed a list (but not a tbl_gpu or data.frame)
  if (length(dots) == 1 && is.list(dots[[1]]) &&
      !is_tbl_gpu(dots[[1]]) && !inherits(dots[[1]], "data.frame")) {
    dots <- dots[[1]]
  }

  # Filter NULL entries
  dots <- dots[!vapply(dots, is.null, logical(1))]

  if (length(dots) == 0) {
    stop("No tables provided to bind_cols", call. = FALSE)
  }

  # Convert any data.frames to tbl_gpu
  dots <- lapply(dots, function(x) {
    if (is.data.frame(x) && !is_tbl_gpu(x)) {
      tbl_gpu(x)
    } else if (!is_tbl_gpu(x)) {
      stop("All inputs must be tbl_gpu or data.frame", call. = FALSE)
    } else {
      x
    }
  })

  if (length(dots) == 1) {
    return(dots[[1]])
  }

  # Materialize any lazy tables
  dots <- lapply(dots, function(x) {
    if (!is.null(x$lazy_ops) && length(x$lazy_ops) > 0 &&
        identical(x$exec_mode, "lazy")) {
      compute(x)
    } else {
      x
    }
  })

  # Collect column names and handle duplicates
  all_names <- unlist(lapply(dots, function(x) x$schema$names))
  repaired_names <- repair_names(all_names, .name_repair)

  # Collect types (unname to avoid issues)
  all_types <- unname(unlist(lapply(dots, function(x) x$schema$types)))

  # Collect pointers
  ptrs <- lapply(dots, function(x) x$ptr)

  # Call C++ function
  new_ptr <- gpu_bind_cols_impl(ptrs)

  # Preserve groups from first table
  groups <- dots[[1]]$groups

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = repaired_names, types = all_types),
    groups = groups,
    exec_mode = "eager"
  )
}

#' GPU bind_rows implementation
#' @param dots List of tables to bind
#' @param .id Optional column name to identify source
#' @return A tbl_gpu
#' @keywords internal
bind_rows_gpu <- function(dots, .id = NULL) {
  # Get source names BEFORE flattening (to preserve names)
  source_names <- names(dots)

  # Flatten if passed a list (but not a tbl_gpu or data.frame)
  if (length(dots) == 1 && is.list(dots[[1]]) &&
      !is_tbl_gpu(dots[[1]]) && !inherits(dots[[1]], "data.frame")) {
    # Get names from the inner list if the outer one has none
    if (is.null(source_names) || all(source_names == "")) {
      source_names <- names(dots[[1]])
    }
    dots <- dots[[1]]
  }

  # Filter NULL entries
  dots <- dots[!vapply(dots, is.null, logical(1))]

  if (length(dots) == 0) {
    stop("No tables provided to bind_rows", call. = FALSE)
  }

  # Set up source names for .id column
  if (is.null(source_names) || length(source_names) != length(dots)) {
    source_names <- as.character(seq_along(dots))
  } else {
    source_names[source_names == ""] <- as.character(which(source_names == ""))
  }

  # Convert any data.frames to tbl_gpu
  dots <- lapply(dots, function(x) {
    if (is.data.frame(x) && !is_tbl_gpu(x)) {
      tbl_gpu(x)
    } else if (!is_tbl_gpu(x)) {
      stop("All inputs must be tbl_gpu or data.frame", call. = FALSE)
    } else {
      x
    }
  })

  # Materialize any lazy tables
  dots <- lapply(dots, function(x) {
    if (!is.null(x$lazy_ops) && length(x$lazy_ops) > 0 &&
        identical(x$exec_mode, "lazy")) {
      compute(x)
    } else {
      x
    }
  })

  if (length(dots) == 1 && is.null(.id)) {
    # Single table, no .id - just return with groups cleared
    result <- dots[[1]]
    result$groups <- character()
    return(result)
  }

  # Compute unified schema
  unified <- compute_unified_schema(dots)

  # Align each table to unified schema
  aligned <- lapply(dots, function(tbl) {
    align_to_schema(tbl, unified)
  })

  # Collect pointers
  ptrs <- lapply(aligned, function(x) x$ptr)

  # Concatenate
  new_ptr <- gpu_bind_rows_aligned(ptrs)

  # Build result
  result <- new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = unified$names, types = unified$types),
    groups = character(),
    exec_mode = "eager"
  )

  # Add .id column if requested
  if (!is.null(.id)) {
    result <- add_id_column(result, .id, dots, source_names)
  }

  result
}

# =============================================================================
# Helper functions
# =============================================================================

#' Repair duplicate column names
#' @param names Character vector of column names
#' @param method Repair method
#' @return Character vector with repaired names
#' @keywords internal
repair_names <- function(names, method = "unique") {
  if (method == "minimal") {
    return(names)
  }

  if (method == "check_unique") {
    if (anyDuplicated(names)) {
      dups <- names[duplicated(names)]
      stop("Column names must be unique. Duplicates: ",
           paste(unique(dups), collapse = ", "), call. = FALSE)
    }
    return(names)
  }

  if (method == "unique" || method == "universal") {
    # Use vctrs if available for better name repair
    if (requireNamespace("vctrs", quietly = TRUE)) {
      return(vctrs::vec_as_names(names, repair = method))
    }
    # Fallback: add numeric suffixes to duplicates
    make.unique(names, sep = "_")
  } else {
    names
  }
}

#' Compute unified schema from multiple tables
#' @param tables List of tbl_gpu objects
#' @return List with names and types for unified schema
#' @keywords internal
compute_unified_schema <- function(tables) {
  # Union of all column names (preserving order from first occurrence)
  all_names <- character()
  name_types <- list()

  for (tbl in tables) {
    for (i in seq_along(tbl$schema$names)) {
      nm <- tbl$schema$names[i]
      ty <- tbl$schema$types[i]

      if (!(nm %in% all_names)) {
        all_names <- c(all_names, nm)
        name_types[[nm]] <- ty
      } else {
        # Column exists - check type compatibility and promote if needed
        existing_type <- name_types[[nm]]
        promoted <- promote_types(existing_type, ty)
        name_types[[nm]] <- promoted
      }
    }
  }

  unified_types <- vapply(all_names, function(nm) name_types[[nm]], character(1))

  list(names = all_names, types = unname(unified_types))
}

#' Promote types for bind_rows compatibility
#' @param type1 First type string
#' @param type2 Second type string
#' @return Promoted type string
#' @keywords internal
promote_types <- function(type1, type2) {
  if (identical(type1, type2)) return(type1)

  # Define type hierarchy for numeric types
  numeric_order <- c("BOOL8", "INT32", "INT64", "FLOAT64")

  if (type1 %in% numeric_order && type2 %in% numeric_order) {
    idx1 <- match(type1, numeric_order)
    idx2 <- match(type2, numeric_order)
    return(numeric_order[max(idx1, idx2)])
  }

  # STRING can coerce from any type (widest)
  if (type1 == "STRING" || type2 == "STRING") {
    return("STRING")
  }

  # Timestamp types - use more precise
  timestamp_types <- c("TIMESTAMP_DAYS", "TIMESTAMP_SECONDS",
                       "TIMESTAMP_MILLISECONDS", "TIMESTAMP_MICROSECONDS",
                       "TIMESTAMP_NANOSECONDS")
  if (type1 %in% timestamp_types && type2 %in% timestamp_types) {
    idx1 <- match(type1, timestamp_types)
    idx2 <- match(type2, timestamp_types)
    return(timestamp_types[max(idx1, idx2)])
  }

  stop(sprintf("Cannot promote incompatible types: %s and %s", type1, type2),
       call. = FALSE)
}

#' Align a table to a target schema
#' @param tbl A tbl_gpu object
#' @param target_schema List with names and types
#' @return A tbl_gpu aligned to the target schema
#' @keywords internal
align_to_schema <- function(tbl, target_schema) {
  current_names <- tbl$schema$names
  current_types <- unname(tbl$schema$types)  # Remove names for comparison
  target_names <- target_schema$names
  target_types <- unname(target_schema$types)

  # Check if already aligned (common case for same-schema tables)
  if (identical(current_names, target_names) &&
      identical(current_types, target_types)) {
    return(tbl)
  }

  nrows <- nrow(tbl)

  # Identify missing columns
  missing_cols <- setdiff(target_names, current_names)

  # If there are missing columns, add them
  if (length(missing_cols) > 0) {
    missing_types <- target_types[match(missing_cols, target_names)]
    tbl <- add_null_columns(tbl, missing_cols, missing_types)
    current_names <- tbl$schema$names
    current_types <- unname(tbl$schema$types)
  }

  # Now handle type coercion for each column (skip STRING - can't cast to STRING)
  for (i in seq_along(target_names)) {
    target_name <- target_names[i]
    target_type <- target_types[i]

    # Skip STRING columns - cudf::cast doesn't support string output
    if (target_type == "STRING") {
      next
    }

    current_idx <- match(target_name, current_names)
    current_type <- current_types[current_idx]

    if (!identical(current_type, target_type)) {
      # Need to cast this column
      tbl <- cast_column(tbl, target_name, target_type)
      current_types <- unname(tbl$schema$types)
    }
  }

  # Reorder columns to match target
  col_indices <- match(target_names, tbl$schema$names) - 1L
  new_ptr <- gpu_select(tbl$ptr, col_indices)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = target_names, types = target_types),
    groups = character(),
    exec_mode = "eager"
  )
}

#' Add null columns to a table
#' @param tbl A tbl_gpu object
#' @param col_names Names of columns to add
#' @param col_types Types of columns to add
#' @return A tbl_gpu with additional null columns
#' @keywords internal
add_null_columns <- function(tbl, col_names, col_types) {
  nrows <- nrow(tbl)

  # Create null columns and bind them
  null_tbls <- lapply(seq_along(col_names), function(i) {
    null_ptr <- gpu_make_null_column(nrows, col_types[i])
    new_tbl_gpu(
      ptr = null_ptr,
      schema = list(names = col_names[i], types = col_types[i]),
      groups = character(),
      exec_mode = "eager"
    )
  })

  # Bind null columns to original table
  all_tbls <- c(list(tbl), null_tbls)
  ptrs <- lapply(all_tbls, function(x) x$ptr)

  new_ptr <- gpu_bind_cols_impl(ptrs)

  new_names <- c(tbl$schema$names, col_names)
  new_types <- c(tbl$schema$types, col_types)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = new_names, types = new_types),
    groups = character(),
    exec_mode = "eager"
  )
}

#' Cast a column to a different type
#' @param tbl A tbl_gpu object
#' @param col_name Name of column to cast
#' @param target_type Target type string
#' @return A tbl_gpu with the column cast to the target type
#' @keywords internal
cast_column <- function(tbl, col_name, target_type) {
  col_idx <- match(col_name, tbl$schema$names) - 1L

  if (is.na(col_idx) || col_idx < 0) {
    stop("Column not found: ", col_name, call. = FALSE)
  }

  new_ptr <- gpu_cast_column(tbl$ptr, col_idx, target_type)

  new_types <- tbl$schema$types
  new_types[col_idx + 1L] <- target_type

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = tbl$schema$names, types = new_types),
    groups = tbl$groups,
    exec_mode = "eager"
  )
}

#' Add .id column to identify source tables
#' @param result The combined tbl_gpu
#' @param id_col_name Name for the .id column
#' @param original_tables List of original tables (for row counts)
#' @param source_names Names/identifiers for each source
#' @return A tbl_gpu with .id column prepended
#' @keywords internal
add_id_column <- function(result, id_col_name, original_tables, source_names) {
  # Create character vector with source identifiers
  row_counts <- vapply(original_tables, nrow, integer(1))
  id_values <- rep(source_names, times = row_counts)

  # Create data frame with .id column and transfer to GPU
  id_df <- data.frame(x = id_values, stringsAsFactors = FALSE)
  names(id_df) <- id_col_name
  id_tbl <- tbl_gpu(id_df)

  # Bind .id column at front
  bind_cols(id_tbl, result, .name_repair = "minimal")
}
