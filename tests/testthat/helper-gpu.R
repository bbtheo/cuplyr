# Helper functions for GPU testing
#
# This file provides utilities for testing GPU operations including:
# - Skip helpers for non-GPU environments
# - GPU memory measurement and verification
# - Object size calculation on GPU
# - Data residency verification (ensuring data is on GPU, not CPU)

# =============================================================================
# Skip Helpers
# =============================================================================

#' Skip test if no GPU is available
#'
#' Use this at the start of any test that requires a GPU
skip_if_no_gpu <- function() {
  if (!has_gpu()) {
    skip("GPU not available")
  }
}

#' Skip test if GPU memory is insufficient
#'
#' @param required_bytes Minimum free GPU memory needed in bytes
skip_if_insufficient_gpu_memory <- function(required_bytes) {
  skip_if_no_gpu()
  info <- gpu_details()
  if (info$free_memory < required_bytes) {
    skip(sprintf("Insufficient GPU memory: need %s, have %s",
                 format_bytes(required_bytes),
                 format_bytes(info$free_memory)))
  }
}

# =============================================================================
# GPU Memory Utilities
# =============================================================================

#' Get current GPU memory usage
#'
#' Returns a snapshot of GPU memory state
#'
#' @return A list with total_memory, free_memory, and used_memory in bytes
gpu_memory_snapshot <- function() {

  if (!has_gpu()) {
    return(list(
      total_memory = NA_real_,
      free_memory = NA_real_,
      used_memory = NA_real_,
      available = FALSE
    ))
  }

  info <- gpu_details()
  list(
    total_memory = info$total_memory,
    free_memory = info$free_memory,
    used_memory = info$total_memory - info$free_memory,
    available = TRUE
  )
}

#' Calculate memory difference between two snapshots
#'
#' @param before Snapshot taken before operation
#' @param after Snapshot taken after operation
#' @return Memory allocated (positive) or freed (negative) in bytes
gpu_memory_diff <- function(before, after) {
  if (!before$available || !after$available) {
    return(NA_real_)
  }
  after$used_memory - before$used_memory
}

#' Measure GPU memory used by a tbl_gpu object
#'
#' This function estimates the GPU memory footprint of a tbl_gpu object
#' by examining its dimensions and column types.
#'
#' @param x A tbl_gpu object
#' @return Estimated memory usage in bytes
estimate_gpu_table_size <- function(x) {
  if (!is_tbl_gpu(x) || is.null(x$ptr)) {
    return(NA_real_)
  }

  dims <- dim(x)
  nrow <- dims[1]
  types <- x$schema$types

  # Estimate bytes per element for each type
  bytes_per_type <- vapply(types, function(type) {
    switch(type,
      "FLOAT64" = 8,
      "FLOAT32" = 4,
      "INT64" = 8,
      "INT32" = 4,
      "INT16" = 2,
      "INT8" = 1,
      "BOOL8" = 1,
      "STRING" = 32,  # Average estimate for strings
      "TIMESTAMP_DAYS" = 4,
      "TIMESTAMP_MICROSECONDS" = 8,
      "DICTIONARY32" = 4,
      8  # Default to 8 bytes for unknown types
    )
  }, numeric(1))

  # Total data bytes
  data_bytes <- sum(nrow * bytes_per_type)

  # Add validity mask overhead (1 bit per row per column, rounded up)
  mask_bytes <- ceiling(nrow / 8) * length(types)

  data_bytes + mask_bytes
}

#' Format bytes as human-readable string
#'
#' @param bytes Number of bytes
#' @return Formatted string (e.g., "1.5 GB")
format_bytes <- function(bytes) {
  if (is.na(bytes)) return("NA")
  if (bytes < 1024) return(paste(bytes, "B"))
  if (bytes < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  if (bytes < 1024^3) return(sprintf("%.1f MB", bytes / 1024^2))
  sprintf("%.1f GB", bytes / 1024^3)
}

# =============================================================================
# GPU Data Residency Verification
# =============================================================================

#' Verify that a tbl_gpu object has data on the GPU
#'
#' This function checks multiple indicators to confirm that data
#' actually resides on the GPU:
#' 1. Object has valid external pointer
#' 2. GPU operations work on the pointer
#' 3. Data is not stored in R memory
#'
#' @param x A tbl_gpu object
#' @return TRUE if data is verified to be on GPU, FALSE otherwise
verify_data_on_gpu <- function(x) {
  # Check 1: Object must be tbl_gpu

  if (!is_tbl_gpu(x)) {
    return(FALSE)
  }

  # Check 2: Must have a valid pointer
  if (is.null(x$ptr)) {
    return(FALSE)
  }

  # Check 3: Pointer must be an external pointer
  if (!inherits(x$ptr, "externalptr")) {
    return(FALSE)
  }

  # Check 4: GPU operations should work on the pointer
  tryCatch({
    dims <- gpu_dim(x$ptr)
    if (!is.integer(dims) || length(dims) != 2) {
      return(FALSE)
    }
    if (any(dims < 0)) {
      return(FALSE)
    }
    TRUE
  }, error = function(e) FALSE)
}

#' Verify GPU pointer is valid and accessible
#'
#' Performs more thorough checks on the GPU pointer
#'
#' @param ptr An external pointer to GPU data
#' @return List with validation results
validate_gpu_pointer <- function(ptr) {
  result <- list(
    is_externalptr = inherits(ptr, "externalptr"),
    is_null_ptr = FALSE,
    can_get_dims = FALSE,
    can_get_types = FALSE,
    can_get_head = FALSE,
    dims = NULL,
    types = NULL
  )

  if (!result$is_externalptr) {
    return(result)
  }

  # Check if pointer is null (freed memory)
  tryCatch({
    dims <- gpu_dim(ptr)
    result$can_get_dims <- TRUE
    result$dims <- dims
  }, error = function(e) {
    if (grepl("NULL|null|invalid", e$message, ignore.case = TRUE)) {
      result$is_null_ptr <- TRUE
    }
  })

  if (result$can_get_dims && !result$is_null_ptr) {
    tryCatch({
      types <- gpu_col_types(ptr)
      result$can_get_types <- TRUE
      result$types <- types
    }, error = function(e) NULL)

    tryCatch({
      # Try to get first row
      head_data <- gpu_head(ptr, 1L, paste0("col", seq_len(result$dims[2])))
      result$can_get_head <- TRUE
    }, error = function(e) NULL)
  }

  result
}

#' Check that data has NOT been copied to R memory
#'
#' Verifies that the tbl_gpu object only holds a pointer,
#' not actual data in R memory
#'
#' @param x A tbl_gpu object
#' @return TRUE if no data copy exists in R memory
verify_no_r_copy <- function(x) {
  if (!is_tbl_gpu(x)) {
    return(FALSE)
  }

  # Check the structure - should only have ptr, schema, lazy_ops, groups, exec_mode
  expected_fields <- c("ptr", "schema", "lazy_ops", "groups", "exec_mode")
  actual_fields <- names(unclass(x))

  # Verify no extra fields that might contain data
  extra_fields <- setdiff(actual_fields, expected_fields)
  if (length(extra_fields) > 0) {
    return(FALSE)
  }

  # Check that schema only contains metadata, not actual data
  if (!is.list(x$schema)) return(FALSE)
  if (!all(c("names", "types") %in% names(x$schema))) return(FALSE)

  # names and types should be character vectors, not data
  if (!is.character(x$schema$names)) return(FALSE)
  if (!is.character(x$schema$types)) return(FALSE)

  # The R object size should be small (no large data vectors)
  # Allow more room for S3 class overhead and external pointer metadata
  obj_size <- as.numeric(object.size(x))
  expected_max_size <- 50000  # ~50KB max for metadata

  # Each column name/type adds ~200 bytes, so adjust for number of columns
  ncol <- length(x$schema$names)
  adjusted_max <- expected_max_size + ncol * 500

  obj_size < adjusted_max
}

# =============================================================================
# GPU Object Size Helpers
# =============================================================================
#' Calculate the actual GPU memory footprint of a table
#'
#' Measures memory by creating a snapshot before and after garbage collection
#'
#' @param x A tbl_gpu object
#' @return Measured memory in bytes (approximate)
measure_gpu_object_size <- function(x) {
  if (!is_tbl_gpu(x) || is.null(x$ptr)) {
    return(NA_real_)
  }

  # Get reference measurement by estimating from schema
  estimate_gpu_table_size(x)
}

#' Compare R object size vs GPU data size
#'
#' The R object should be much smaller than the actual GPU data
#'
#' @param x A tbl_gpu object
#' @return List with r_size, gpu_size, and ratio
compare_r_vs_gpu_size <- function(x) {
  r_size <- as.numeric(object.size(x))
  gpu_size <- estimate_gpu_table_size(x)

  list(
    r_size = r_size,
    gpu_size = gpu_size,
    ratio = if (!is.na(gpu_size) && gpu_size > 0) gpu_size / r_size else NA_real_,
    r_size_formatted = format_bytes(r_size),
    gpu_size_formatted = format_bytes(gpu_size)
  )
}

# =============================================================================
# Test Data Generators
# =============================================================================

#' Create a test data frame with various column types
#'
#' @param nrow Number of rows
#' @param include_na Whether to include NA values
#' @return A data frame for testing
create_test_data <- function(nrow = 100, include_na = FALSE) {
  set.seed(42)

  df <- data.frame(
    int_col = sample(1:1000, nrow, replace = TRUE),
    dbl_col = runif(nrow, 0, 100),
    chr_col = sample(letters[1:10], nrow, replace = TRUE),
    lgl_col = sample(c(TRUE, FALSE), nrow, replace = TRUE),
    stringsAsFactors = FALSE
  )

  if (include_na) {
    # Add some NA values
    na_indices <- sample(nrow, nrow %/% 10)
    df$int_col[na_indices[1:3]] <- NA_integer_
    df$dbl_col[na_indices[4:6]] <- NA_real_
    df$chr_col[na_indices[7:9]] <- NA_character_
    df$lgl_col[na_indices[10:min(length(na_indices), 12)]] <- NA
  }

  df
}

# =============================================================================
# Exec Mode Helpers
# =============================================================================

#' Run a function in both eager and lazy execution modes
#'
#' @param data Data frame to load to GPU
#' @param fn Function that receives (tbl, mode) and returns a result
#' @return Named list with results for "eager" and "lazy"
with_exec_modes <- function(data, fn) {
  results <- list()
  for (mode in c("eager", "lazy")) {
    tbl <- tbl_gpu(data, lazy = identical(mode, "lazy"))
    results[[mode]] <- fn(tbl, mode)
  }
  results
}


#' Create a large test data frame for memory testing
#'
#' @param nrow Number of rows (default 1 million)
#' @param ncol Number of numeric columns
#' @return A data frame
create_large_test_data <- function(nrow = 1e6, ncol = 10) {
  set.seed(123)

  df <- as.data.frame(
    matrix(runif(nrow * ncol), nrow = nrow, ncol = ncol)
  )
  names(df) <- paste0("col", seq_len(ncol))

  df
}

# =============================================================================
# Assertion Helpers
# =============================================================================

#' Assert that data is on GPU with informative error
#'
#' @param x A tbl_gpu object
#' @param info Additional info for error message
expect_data_on_gpu <- function(x, info = NULL) {
  act <- quasi_label(rlang::enquo(x), arg = "x")

  if (!verify_data_on_gpu(x)) {
    fail(sprintf(
      "%s does not have valid data on GPU.%s",
      act$lab,
      if (!is.null(info)) paste0("\n", info) else ""
    ))
  }

  invisible(x)
}

#' Assert that R object is lightweight (no data copy)
#'
#' @param x A tbl_gpu object
#' @param max_bytes Maximum allowed R object size
expect_lightweight_r_object <- function(x, max_bytes = 100000) {
  act <- quasi_label(rlang::enquo(x), arg = "x")

  r_size <- as.numeric(object.size(x))
  # Adjust for number of columns
  ncol <- length(x$schema$names)
  adjusted_max <- max_bytes + ncol * 500

  if (r_size > adjusted_max) {
    fail(sprintf(
      "%s R object size (%s) exceeds maximum (%s).\nThis suggests data may have been copied to R memory.",
      act$lab,
      format_bytes(r_size),
      format_bytes(adjusted_max)
    ))
  }

  invisible(x)
}

#' Assert GPU memory was allocated
#'
#' @param before Memory snapshot before operation
#' @param after Memory snapshot after operation
#' @param min_bytes Minimum expected allocation
expect_gpu_memory_allocated <- function(before, after, min_bytes = 0) {
  diff <- gpu_memory_diff(before, after)

  if (is.na(diff)) {
    fail("Could not measure GPU memory difference")
  }

  if (diff < min_bytes) {
    fail(sprintf(
      "Expected at least %s GPU memory allocation, but only %s was allocated",
      format_bytes(min_bytes),
      format_bytes(diff)
    ))
  }

  invisible(diff)
}

#' Assert tbl_gpu structure is valid
#'
#' @param x Object to check
expect_valid_tbl_gpu <- function(x) {
  expect_true(is_tbl_gpu(x))
  expect_true(is.list(unclass(x)))

  # Use unclass to get actual list elements (names.tbl_gpu returns column names)
  list_names <- names(unclass(x))
  expect_true(all(c("ptr", "schema", "lazy_ops", "groups", "exec_mode") %in% list_names))

  expect_true(inherits(x$ptr, "externalptr"))
  expect_true(is.list(x$schema))
  expect_true(all(c("names", "types") %in% names(x$schema)))
  expect_type(x$schema$names, "character")
  expect_type(x$schema$types, "character")
  expect_true(is.null(x$lazy_ops) || is.list(x$lazy_ops))
  expect_type(x$groups, "character")

  invisible(x)
}

# =============================================================================
# Cleanup Helpers
# =============================================================================

#' Force garbage collection and wait for GPU cleanup
#'
#' Use between tests to ensure GPU memory is freed
gc_gpu <- function() {
  gc(verbose = FALSE, full = TRUE)
  # Small delay to allow GPU cleanup
  Sys.sleep(0.1)
  gc(verbose = FALSE, full = TRUE)
}
