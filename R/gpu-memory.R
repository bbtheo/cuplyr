#' Estimate GPU memory usage of a tbl_gpu object
#'
#' Calculates the estimated GPU memory footprint of a GPU table based on
#' its dimensions and column types. This is useful for understanding memory
#' requirements before working with large datasets.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return The estimated memory usage in bytes (numeric), or `NA` if the
#'   object is not a valid `tbl_gpu` or has no data on GPU.
#'
#' @details
#' The estimate includes:
#' \itemize{
#'   \item Column data (varies by type: 8 bytes for FLOAT64, 4 bytes for INT32, etc.)
#'   \item Validity bitmasks for NA handling (1 bit per row per column)
#' }
#'
#' String columns use an average estimate of 32 bytes per element, which may
#' vary significantly based on actual string lengths.
#'
#' This is an estimate and actual GPU memory usage may be higher due to:
#' \itemize{
#'   \item Memory alignment requirements
#'   \item RMM memory pool overhead
#'   \item Temporary allocations
#' }
#'
#' @seealso
#' \code{\link{gpu_details}} for overall GPU memory info,
#' \code{\link{gpu_object_info}} for detailed object information
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   size <- gpu_memory_usage(gpu_mtcars)
#'   cat("Estimated GPU memory:", round(size / 1024, 1), "KB\n")
#' }
gpu_memory_usage <- function(x) {
  if (!is_tbl_gpu(x) || is.null(x$ptr)) {
    return(NA_real_)
  }

  dims <- tryCatch(dim(x), error = function(e) NULL)
  if (is.null(dims)) {
    return(NA_real_)
  }

  nrow <- dims[1]
  types <- x$schema$types

  # Bytes per element for each GPU type
  bytes_per_type <- vapply(types, function(type) {
    switch(type,
      "FLOAT64" = 8,
      "FLOAT32" = 4,
      "INT64" = 8,
      "INT32" = 4,
      "INT16" = 2,
      "INT8" = 1,
      "BOOL8" = 1,
      "STRING" = 32,  # Average estimate
      "TIMESTAMP_DAYS" = 4,
      "TIMESTAMP_MICROSECONDS" = 8,
      "TIMESTAMP_NANOSECONDS" = 8,
      "DICTIONARY32" = 4,
      8  # Default
    )
  }, numeric(1))

  # Data bytes
  data_bytes <- sum(nrow * bytes_per_type)

  # Validity mask overhead (1 bit per row per column, rounded up to bytes)
  mask_bytes <- ceiling(nrow / 8) * length(types)

  data_bytes + mask_bytes
}

#' Get detailed information about a GPU table object
#'
#' Returns comprehensive information about a `tbl_gpu` object including
#' its dimensions, column types, estimated memory usage, and verification
#' that data resides on the GPU.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{valid}{Logical: TRUE if the object is a valid tbl_gpu with GPU data}
#'   \item{nrow}{Number of rows}
#'   \item{ncol}{Number of columns}
#'   \item{column_names}{Character vector of column names}
#'   \item{column_types}{Character vector of GPU column types}
#'   \item{estimated_gpu_bytes}{Estimated GPU memory usage in bytes}
#'   \item{estimated_gpu_mb}{Estimated GPU memory usage in megabytes}
#'   \item{r_object_bytes}{Size of the R object (should be small)}
#'   \item{data_on_gpu}{Logical: TRUE if data is verified to be on GPU}
#'   \item{pointer_valid}{Logical: TRUE if the external pointer is valid}
#' }
#'
#' @seealso
#' \code{\link{gpu_memory_usage}} for just the memory estimate,
#' \code{\link{verify_gpu_data}} to check if data is on GPU
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   info <- gpu_object_info(gpu_mtcars)
#'   cat("Rows:", info$nrow, "\n")
#'   cat("GPU memory:", round(info$estimated_gpu_mb, 2), "MB\n")
#'   cat("Data on GPU:", info$data_on_gpu, "\n")
#' }
gpu_object_info <- function(x) {
  result <- list(
    valid = FALSE,
    nrow = NA_integer_,
    ncol = NA_integer_,
    column_names = character(0),
    column_types = character(0),
    estimated_gpu_bytes = NA_real_,
    estimated_gpu_mb = NA_real_,
    r_object_bytes = as.numeric(utils::object.size(x)),
    data_on_gpu = FALSE,
    pointer_valid = FALSE
  )

  if (!is_tbl_gpu(x)) {
    return(result)
  }

  result$valid <- TRUE

  if (!is.null(x$ptr) && inherits(x$ptr, "externalptr")) {
    result$pointer_valid <- TRUE

    dims <- tryCatch(gpu_dim(x$ptr), error = function(e) NULL)
    if (!is.null(dims)) {
      result$nrow <- dims[1]
      result$ncol <- dims[2]
      result$data_on_gpu <- TRUE
    }
  }

  result$column_names <- x$schema$names
  result$column_types <- x$schema$types

  gpu_bytes <- gpu_memory_usage(x)
  result$estimated_gpu_bytes <- gpu_bytes
  result$estimated_gpu_mb <- gpu_bytes / (1024 * 1024)

  result
}

#' Verify that data resides on GPU
#'
#' Performs checks to confirm that a `tbl_gpu` object has its data
#' stored on the GPU, not in R memory. This is useful for debugging
#' and ensuring GPU operations are working correctly.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return `TRUE` if all checks pass and data is verified to be on GPU,
#'   `FALSE` otherwise.
#'
#' @details
#' This function performs multiple verification steps:
#' \enumerate{
#'   \item Object has the `tbl_gpu` class
#'   \item Object has a valid external pointer
#'   \item GPU operations (dim, types) work on the pointer
#'   \item R object is small (no data copy in R memory)
#' }
#'
#' @seealso
#' \code{\link{gpu_object_info}} for detailed object information,
#' \code{\link{tbl_gpu}} for creating GPU tables
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Should return TRUE
#'   verify_gpu_data(gpu_mtcars)
#'
#'   # Regular data frames return FALSE
#'   verify_gpu_data(mtcars)
#' }
verify_gpu_data <- function(x) {
  # Check 1: Must be tbl_gpu
  if (!is_tbl_gpu(x)) {
    return(FALSE)
  }

  # Check 2: Must have a valid pointer
  if (is.null(x$ptr)) {
    return(FALSE)
  }

  # Check 3: Pointer must be external pointer
  if (!inherits(x$ptr, "externalptr")) {
    return(FALSE)
  }

  # Check 4: GPU operations should work
  dims <- tryCatch(gpu_dim(x$ptr), error = function(e) NULL)
  if (is.null(dims) || !is.integer(dims) || length(dims) != 2) {
    return(FALSE)
  }

  # Check 5: R object should be lightweight (no data copy)
  r_size <- as.numeric(utils::object.size(x))
  ncol <- length(x$schema$names)
  max_expected <- 10000 + ncol * 200  # Base + per-column metadata

  if (r_size > max_expected) {
    return(FALSE)
  }

  TRUE
}

#' Compare R object size vs GPU data size
#'
#' Computes the ratio of GPU memory usage to R object size for a `tbl_gpu`
#' object. A high ratio confirms that data is stored on GPU, not in R.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A list with:
#' \describe{
#'   \item{r_bytes}{Size of the R object in bytes}
#'   \item{gpu_bytes}{Estimated GPU memory in bytes}
#'   \item{ratio}{GPU size divided by R size (should be > 1 if data is on GPU)}
#' }
#'
#' @seealso
#' \code{\link{verify_gpu_data}} for boolean verification,
#' \code{\link{gpu_object_info}} for detailed information
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Create a larger dataset
#'   df <- data.frame(matrix(runif(10000), ncol = 10))
#'   gpu_df <- tbl_gpu(df)
#'
#'   comparison <- gpu_size_comparison(gpu_df)
#'   cat("R object:", round(comparison$r_bytes / 1024, 1), "KB\n")
#'   cat("GPU data:", round(comparison$gpu_bytes / 1024, 1), "KB\n")
#'   cat("Ratio:", round(comparison$ratio, 1), "x\n")
#' }
gpu_size_comparison <- function(x) {
  r_bytes <- as.numeric(utils::object.size(x))
  gpu_bytes <- gpu_memory_usage(x)

  list(
    r_bytes = r_bytes,
    gpu_bytes = gpu_bytes,
    ratio = if (!is.na(gpu_bytes) && gpu_bytes > 0) gpu_bytes / r_bytes else NA_real_
  )
}

#' Get GPU memory snapshot
#'
#' Returns the current GPU memory state including total, free, and used memory.
#' Useful for monitoring GPU memory usage during operations.
#'
#' @return A list with:
#' \describe{
#'   \item{available}{Logical: TRUE if GPU is available}
#'   \item{total_bytes}{Total GPU memory in bytes}
#'   \item{free_bytes}{Free GPU memory in bytes}
#'   \item{used_bytes}{Used GPU memory in bytes}
#'   \item{total_gb}{Total GPU memory in gigabytes}
#'   \item{free_gb}{Free GPU memory in gigabytes}
#'   \item{used_gb}{Used GPU memory in gigabytes}
#' }
#'
#' @seealso
#' \code{\link{gpu_details}} for device information,
#' \code{\link{gpu_memory_usage}} for per-object memory estimates
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Check memory before allocation
#'   before <- gpu_memory_state()
#'
#'   # Allocate some GPU data
#'   gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))
#'
#'   # Check memory after allocation
#'   after <- gpu_memory_state()
#'
#'   cat("Memory used:", after$used_gb - before$used_gb, "GB\n")
#' }
gpu_memory_state <- function() {
  info <- gpu_details()

  if (!isTRUE(info$available)) {
    return(list(
      available = FALSE,
      total_bytes = NA_real_,
      free_bytes = NA_real_,
      used_bytes = NA_real_,
      total_gb = NA_real_,
      free_gb = NA_real_,
      used_gb = NA_real_
    ))
  }

  list(
    available = TRUE,
    total_bytes = info$total_memory,
    free_bytes = info$free_memory,
    used_bytes = info$total_memory - info$free_memory,
    total_gb = info$total_memory / 1e9,
    free_gb = info$free_memory / 1e9,
    used_gb = (info$total_memory - info$free_memory) / 1e9
  )
}

#' Force GPU memory cleanup
#'
#' Triggers R garbage collection to free GPU memory held by unreferenced
#' `tbl_gpu` objects. Use this between operations when GPU memory is limited
#' or before large allocations.
#'
#' @param verbose Logical. If TRUE, prints memory freed. Default FALSE.
#' @param aggressive Logical. If TRUE (default), runs multiple GC passes with
#'   short delays to more aggressively trigger finalizers. If FALSE, runs a
#'   lighter cleanup pass.
#'
#' @return Invisibly returns a list with memory state before and after cleanup,
#'   and the amount freed in bytes and gigabytes.
#'
#' @details
#' GPU memory is automatically freed when `tbl_gpu` objects are garbage
#' collected by R. However, R's garbage collector doesn't know about GPU
#' memory pressure and may not run immediately. This function forces
#' garbage collection and allows time for GPU cleanup.
#'
#' Call this function:
#' \itemize{
#'   \item Between benchmark iterations
#'   \item After removing large GPU objects with `rm()`
#'   \item When you see out-of-memory errors
#'   \item Before allocating large new GPU tables
#' }
#'
#' @seealso
#' \code{\link{gpu_memory_state}} for checking current memory usage
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Create and discard a GPU table
#'   gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))
#'   rm(gpu_df)
#'
#'   # Force cleanup
#'   gpu_gc(verbose = TRUE)
#' }
gpu_gc <- function(verbose = FALSE, aggressive = TRUE) {
  before <- gpu_memory_state()

  if (aggressive) {
    # Multiple GC passes with delays to ensure finalizers run
    for (i in 1:5) {
      gc(verbose = FALSE, full = TRUE)
      Sys.sleep(0.2)
    }
  } else {
    gc(verbose = FALSE, full = TRUE)
    Sys.sleep(0.1)
    gc(verbose = FALSE, full = TRUE)
  }

  after <- gpu_memory_state()

  freed_bytes <- after$free_bytes - before$free_bytes
  freed_gb <- freed_bytes / 1e9

  if (verbose && !is.na(freed_bytes)) {
    if (freed_gb >= 0.01) {
      message(sprintf("GPU memory freed: %.2f GB", freed_gb))
    } else if (freed_bytes > 0) {
      message(sprintf("GPU memory freed: %.0f bytes", freed_bytes))
    } else {
      message("No GPU memory freed")
    }
  }

  invisible(list(
    before = before,
    after = after,
    freed_bytes = freed_bytes,
    freed_gb = freed_gb
  ))
}
