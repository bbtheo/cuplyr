# Tests for GPU memory management and data residency verification
#
# These tests specifically verify:
# - Data actually resides on GPU (not in R memory)
# - GPU memory allocation and deallocation
# - Memory footprint estimation
# - No accidental data copies to R memory

# =============================================================================
# Data Residency Verification Tests
# =============================================================================

test_that("tbl_gpu data is on GPU, not in R memory", {
  skip_if_no_gpu()

  # Create a reasonably sized dataset
  df <- data.frame(
    a = runif(10000),
    b = runif(10000),
    c = runif(10000)
  )

  gpu_df <- tbl_gpu(df)

  # Verify using our helper functions
  expect_true(verify_data_on_gpu(gpu_df))
  expect_true(verify_no_r_copy(gpu_df))
})

test_that("GPU pointer is valid and accessible", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  validation <- validate_gpu_pointer(gpu_df$ptr)

  expect_true(validation$is_externalptr)
  expect_false(validation$is_null_ptr)
  expect_true(validation$can_get_dims)
  expect_true(validation$can_get_types)
  expect_true(validation$can_get_head)

  # Dimensions should match
  expect_equal(validation$dims[1], nrow(mtcars))
  expect_equal(validation$dims[2], ncol(mtcars))
})

test_that("R object size is much smaller than GPU data", {
  skip_if_no_gpu()

  # Create large data
  df <- data.frame(matrix(runif(100000), ncol = 10))
  gpu_df <- tbl_gpu(df)

  sizes <- compare_r_vs_gpu_size(gpu_df)

  # GPU data should be much larger than R object
  expect_true(sizes$ratio > 10,
              info = sprintf("Expected GPU size (%s) to be much larger than R size (%s)",
                             sizes$gpu_size_formatted, sizes$r_size_formatted))

  # R object should be small (just metadata)
  expect_true(sizes$r_size < 50000,
              info = sprintf("R object size (%s) unexpectedly large",
                             sizes$r_size_formatted))
})

test_that("tbl_gpu stores only external pointer, not data", {
  skip_if_no_gpu()

  df <- create_large_test_data(nrow = 50000, ncol = 5)
  gpu_df <- tbl_gpu(df)

  # Check structure (use unclass to get list elements, not column names)
  expect_true("ptr" %in% names(unclass(gpu_df)))
  expect_true(inherits(gpu_df$ptr, "externalptr"))

  # Schema should only contain metadata
  expect_true(is.character(gpu_df$schema$names))
  expect_true(is.character(gpu_df$schema$types))
  expect_equal(length(gpu_df$schema$names), 5)
  expect_equal(length(gpu_df$schema$types), 5)

  # No other large data structures
  expect_true(is.null(gpu_df$lazy_ops) || is.list(gpu_df$lazy_ops))
  if (is.list(gpu_df$lazy_ops)) {
    expect_length(gpu_df$lazy_ops, 0)
  }
  expect_type(gpu_df$groups, "character")
})

# =============================================================================
# GPU Memory Allocation Tests
# =============================================================================

test_that("tbl_gpu() allocates GPU memory", {
  skip_if_no_gpu()

  gc_gpu()

  # Measured at the RMM allocator, not with device-level free memory
  # (cudaMemGetInfo). Each column here is 781 KiB, and the CUDA driver serves
  # allocations of up to 1 MiB from 2 MiB pages the process already holds, so
  # when earlier tests leave partly used pages behind, device-level usage does
  # not move at all (the 0 B failures in F56). RMM sees every allocation.
  df <- create_large_test_data(nrow = 100000, ncol = 5)
  tracked <- gpu_alloc_tracking(tbl_gpu(df))
  gpu_df <- tracked$value
  expect_s3_class(gpu_df, "tbl_gpu")
  expect_equal(dim(gpu_df), c(100000L, 5L))

  # 100000 rows * 5 cols * 8 bytes = 4 MB of column data, still held by the
  # table when tbl_gpu() returns (no NAs, so no null masks).
  expected_min <- 100000 * 5 * 8
  expect_true(tracked$held_bytes >= expected_min,
              info = sprintf("Upload holds %s on the GPU, expected at least %s",
                             format_bytes(tracked$held_bytes),
                             format_bytes(expected_min)))
  expect_true(tracked$held_bytes < 2 * expected_min,
              info = sprintf("Upload holds %s on the GPU, expected about %s",
                             format_bytes(tracked$held_bytes),
                             format_bytes(expected_min)))
  expect_true(tracked$allocated_bytes >= tracked$held_bytes)
})

test_that("tbl_gpu() raises device-level GPU memory use", {
  skip_if_no_gpu()

  # Device-level check that the data really occupies device memory. Columns
  # are 7.6 MiB each: the CUDA driver maps fresh pages for every allocation
  # above 1 MiB, so memory the process already holds cannot absorb them
  # (unlike the sub-1 MiB columns of the test above; see F56).
  #
  # A memory pool keeps its pages, so this claim about the driver is checked in
  # a process with cuplyr's pool switched off; the allocator-level equivalent
  # under the pool is the next test.
  out <- run_with_pool_off(function() {
    df <- as.data.frame(matrix(runif(1e6 * 5), ncol = 5))
    gc(full = TRUE)
    info0 <- gpu_details()
    gpu_df <- tbl_gpu(df)
    info1 <- gpu_details()
    list(diff = (info1$total_memory - info1$free_memory) -
           (info0$total_memory - info0$free_memory),
         dim = dim(gpu_df))
  })

  expected_min <- 1000000 * 5 * 8
  expect_true(out$diff >= expected_min * 0.5,
              info = sprintf("Memory increase (%s) less than expected (%s)",
                             format_bytes(out$diff), format_bytes(expected_min)))
  expect_equal(out$dim, c(1000000L, 5L))
})

test_that("tbl_gpu() raises allocator-level GPU memory use under the pool", {
  skip_if_no_gpu()

  gc_gpu()

  # What the table holds, measured as bytes allocated from cuplyr's resource
  # (pool_used_bytes; device-level `used_bytes` when there is none).
  df <- create_large_test_data(nrow = 1000000, ncol = 5)
  before <- gpu_alloc_snapshot()
  gpu_df <- tbl_gpu(df)
  after <- gpu_alloc_snapshot()

  diff <- gpu_memory_diff(before, after)
  expected_min <- 1000000 * 5 * 8
  expect_true(diff >= expected_min * 0.95,
              info = sprintf("Memory increase (%s) less than expected (%s)",
                             format_bytes(diff), format_bytes(expected_min)))
  expect_true(diff < 2 * expected_min,
              info = sprintf("Memory increase (%s) more than expected (%s)",
                             format_bytes(diff), format_bytes(expected_min)))
  expect_equal(dim(gpu_df), c(1000000L, 5L))
})

test_that("GPU memory is freed on garbage collection", {
  skip_if_no_gpu()

  gc_gpu()
  before <- gpu_alloc_snapshot()

  # Create and discard in a local scope
  local({
    df <- create_large_test_data(nrow = 100000, ncol = 10)
    gpu_df <- tbl_gpu(df)
    expect_data_on_gpu(gpu_df)
  })

  gc_gpu()
  after <- gpu_alloc_snapshot()

  # Allocator level (bytes allocated from cuplyr's resource): the 8 MB table is
  # gone, so usage is back to baseline well inside the tolerance.
  expect_lt(after$used_memory - before$used_memory, 100000 * 10 * 8 * 0.5)

  # Memory should be approximately back to baseline
  diff <- abs(after$used_memory - before$used_memory)
  tolerance <- 100 * 1024 * 1024  # 100 MB tolerance

  expect_true(diff < tolerance,
              info = sprintf(
                "Memory not freed: before=%s, after=%s, diff=%s",
                format_bytes(before$used_memory),
                format_bytes(after$used_memory),
                format_bytes(diff)
              ))
})

test_that("Multiple GPU tables can coexist with separate memory", {
  skip_if_no_gpu()

  gc_gpu()
  baseline <- gpu_alloc_snapshot()

  # Create first table
  gpu_df1 <- tbl_gpu(create_large_test_data(nrow = 10000, ncol = 5))
  after_first <- gpu_alloc_snapshot()

  # Create second table
  gpu_df2 <- tbl_gpu(create_large_test_data(nrow = 10000, ncol = 5))
  after_second <- gpu_alloc_snapshot()

  # Allocator-level usage (bytes allocated from cuplyr's resource) rises with
  # each table by its data size: 10000 rows * 5 columns * 8 bytes each.
  one_table <- 10000 * 5 * 8
  first_diff <- gpu_memory_diff(baseline, after_first)
  total_diff <- gpu_memory_diff(baseline, after_second)
  expect_true(first_diff >= one_table * 0.95,
              info = sprintf("First table diff: %s", format_bytes(first_diff)))
  expect_true(total_diff >= 2 * one_table * 0.95,
              info = sprintf("Total memory diff: %s", format_bytes(total_diff)))

  # Both tables should work independently
  expect_data_on_gpu(gpu_df1)
  expect_data_on_gpu(gpu_df2)

  # Dimensions should be correct
  expect_equal(dim(gpu_df1), c(10000L, 5L))
  expect_equal(dim(gpu_df2), c(10000L, 5L))
})

# =============================================================================
# Memory After Operations Tests
# =============================================================================

test_that("filter() creates new GPU allocation", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path

  gpu_df <- tbl_gpu(create_large_test_data(nrow = 10000, ncol = 5))
  before <- gpu_memory_snapshot()

  filtered <- dplyr::filter(gpu_df, col1 > 0.5)

  after <- gpu_memory_snapshot()

  # New memory should be allocated for filtered result
  expect_false(identical(gpu_df$ptr, filtered$ptr))

  # Both should have data on GPU
  expect_data_on_gpu(gpu_df)
  expect_data_on_gpu(filtered)
})

test_that("mutate() creates new GPU allocation", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path

  gpu_df <- tbl_gpu(mtcars)

  mutated <- dplyr::mutate(gpu_df, new_col = mpg + 10)

  # Different pointers
  expect_false(identical(gpu_df$ptr, mutated$ptr))

  # Both valid
  expect_data_on_gpu(gpu_df)
  expect_data_on_gpu(mutated)

  # Different dimensions
  expect_equal(ncol(gpu_df), 11)
  expect_equal(ncol(mutated), 12)
})

test_that("select() creates new GPU allocation", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path

  gpu_df <- tbl_gpu(mtcars)

  selected <- dplyr::select(gpu_df, mpg, cyl)

  # Different pointers
  expect_false(identical(gpu_df$ptr, selected$ptr))

  # Both valid
  expect_data_on_gpu(gpu_df)
  expect_data_on_gpu(selected)
})

test_that("chained operations maintain GPU residency", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = FALSE)

  # Chain multiple operations
  result <- gpu_df |>
    dplyr::filter(mpg > 15) |>
    dplyr::mutate(kpl = mpg * 0.425) |>
    dplyr::select(mpg, kpl, cyl)

  # Final result should be on GPU
  expect_data_on_gpu(result)
  expect_true(verify_no_r_copy(result))
  expect_lightweight_r_object(result)
})

# =============================================================================
# Memory Size Estimation Tests
# =============================================================================

test_that("estimate_gpu_table_size() returns reasonable estimates", {
  skip_if_no_gpu()

  # Known size: 100 rows, 3 double columns = 100 * 3 * 8 = 2400 bytes + overhead
  df <- data.frame(a = runif(100), b = runif(100), c = runif(100))
  gpu_df <- tbl_gpu(df)

  size <- estimate_gpu_table_size(gpu_df)

  expect_true(!is.na(size))
  expect_true(size > 2400)  # At least the raw data
  expect_true(size < 10000)  # But not unreasonably large
})

test_that("estimate_gpu_table_size() scales with data", {
  skip_if_no_gpu()

  df1 <- data.frame(a = runif(1000))
  df2 <- data.frame(a = runif(10000))

  gpu_df1 <- tbl_gpu(df1)
  gpu_df2 <- tbl_gpu(df2)

  size1 <- estimate_gpu_table_size(gpu_df1)
  size2 <- estimate_gpu_table_size(gpu_df2)

  # Size should scale roughly linearly
  expect_true(size2 > size1 * 5)
  expect_true(size2 < size1 * 15)
})

test_that("estimate_gpu_table_size() accounts for column types", {
  skip_if_no_gpu()

  # Same rows, different types
  df_double <- data.frame(a = runif(1000))  # 8 bytes each
  df_int <- data.frame(a = 1:1000)  # 4 bytes each

  gpu_double <- tbl_gpu(df_double)
  gpu_int <- tbl_gpu(df_int)

  size_double <- estimate_gpu_table_size(gpu_double)
  size_int <- estimate_gpu_table_size(gpu_int)

  # Double should be roughly twice the size
  ratio <- size_double / size_int
  expect_true(ratio > 1.5 && ratio < 2.5,
              info = sprintf("Expected ratio ~2, got %.2f", ratio))
})

# =============================================================================
# GPU Object Size Helper Tests
# =============================================================================

test_that("measure_gpu_object_size() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  size <- measure_gpu_object_size(gpu_df)

  expect_true(!is.na(size))
  expect_true(size > 0)
})

test_that("compare_r_vs_gpu_size() provides useful comparison", {
  skip_if_no_gpu()

  df <- create_large_test_data(nrow = 10000, ncol = 10)
  gpu_df <- tbl_gpu(df)

  comparison <- compare_r_vs_gpu_size(gpu_df)

  expect_true(comparison$r_size > 0)
  expect_true(comparison$gpu_size > 0)
  expect_true(comparison$ratio > 1)

  # Formatted strings should be present
  expect_type(comparison$r_size_formatted, "character")
  expect_type(comparison$gpu_size_formatted, "character")
})

# =============================================================================
# External Pointer Validity Tests
# =============================================================================

test_that("GPU operations fail gracefully on invalid pointer", {
  skip_if_no_gpu()

  # Create a fake invalid external pointer
  fake_ptr <- new("externalptr")

  expect_error(gpu_dim(fake_ptr))
})

test_that("GPU pointer remains valid through operations", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = FALSE)

  # Perform many operations with simple filter expressions
  thresholds <- c(11, 12, 13, 14, 15)
  for (thresh in thresholds) {
    current <- gpu_df
    current <- dplyr::filter(current, mpg > 10)
    current <- dplyr::filter(current, mpg > 11)
    current <- dplyr::filter(current, mpg > 12)
    # Each result should be valid
    expect_data_on_gpu(current)
  }

  # Original should still be valid
  expect_data_on_gpu(gpu_df)
})

test_that("GPU pointer is unique per allocation", {
  skip_if_no_gpu()

  gpu_df1 <- tbl_gpu(mtcars)
  gpu_df2 <- tbl_gpu(mtcars)
  gpu_df3 <- tbl_gpu(iris[, 1:4])

  # All should have different pointers
  expect_false(identical(gpu_df1$ptr, gpu_df2$ptr))
  expect_false(identical(gpu_df1$ptr, gpu_df3$ptr))
  expect_false(identical(gpu_df2$ptr, gpu_df3$ptr))

  # But all should be valid
  expect_data_on_gpu(gpu_df1)
  expect_data_on_gpu(gpu_df2)
  expect_data_on_gpu(gpu_df3)
})

# =============================================================================
# No Accidental CPU Copy Tests
# =============================================================================

test_that("tbl_gpu creation does not keep data in R", {
  skip_if_no_gpu()

  # Large data to make copies obvious
  df <- create_large_test_data(nrow = 50000, ncol = 10)
  original_size <- as.numeric(object.size(df))

  gpu_df <- tbl_gpu(df)
  gpu_size <- as.numeric(object.size(gpu_df))

  # GPU object should be much smaller (no data copy)
  expect_true(gpu_size < original_size / 10,
              info = sprintf("GPU object (%s) should be much smaller than data (%s)",
                             format_bytes(gpu_size), format_bytes(original_size)))
})

test_that("filter() does not copy data to R", {
  skip_if_no_gpu()

  df <- create_large_test_data(nrow = 50000, ncol = 10)
  gpu_df <- tbl_gpu(df, lazy = FALSE)

  filtered <- dplyr::filter(gpu_df, col1 > 0.5)
  filtered_size <- as.numeric(object.size(filtered))

  # Filtered result should also be lightweight
  expect_true(filtered_size < 50000)
  expect_true(verify_no_r_copy(filtered))
})

test_that("mutate() does not copy data to R", {
  skip_if_no_gpu()

  df <- create_large_test_data(nrow = 50000, ncol = 10)
  gpu_df <- tbl_gpu(df, lazy = FALSE)

  mutated <- dplyr::mutate(gpu_df, new_col = col1 + col2)
  mutated_size <- as.numeric(object.size(mutated))

  expect_true(mutated_size < 50000)
  expect_true(verify_no_r_copy(mutated))
})

test_that("select() does not copy data to R", {
  skip_if_no_gpu()

  df <- create_large_test_data(nrow = 50000, ncol = 10)
  gpu_df <- tbl_gpu(df, lazy = FALSE)

  selected <- dplyr::select(gpu_df, col1, col2)
  selected_size <- as.numeric(object.size(selected))

  expect_true(selected_size < 50000)
  expect_true(verify_no_r_copy(selected))
})

# =============================================================================
# Memory Stress Tests
# =============================================================================

test_that("Many small allocations don't leak memory", {
  skip_if_no_gpu()

  gc_gpu()
  before <- gpu_alloc_snapshot()

  # Create many small tables
  for (i in 1:50) {
    local({
      df <- data.frame(x = runif(1000))
      gpu_df <- tbl_gpu(df)
      expect_data_on_gpu(gpu_df)
    })

    # Periodic GC
    if (i %% 10 == 0) gc_gpu()
  }

  gc_gpu()
  after <- gpu_alloc_snapshot()

  # Memory should be approximately back to baseline
  diff <- after$used_memory - before$used_memory
  tolerance <- 50 * 1024 * 1024  # 50 MB tolerance

  expect_true(abs(diff) < tolerance,
              info = sprintf("Memory leak detected: %s", format_bytes(diff)))

  # Allocator level (bytes allocated from cuplyr's resource) sees every byte: the
  # 50 tables hold 400 KB in all, and less than half of that may remain.
  expect_lt(diff, 50 * 1000 * 8 * 0.5)
})

test_that("Large allocation followed by GC frees memory", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(2 * 1024 * 1024 * 1024)  # Need 2GB free

  gc_gpu()
  before <- gpu_alloc_snapshot()

  # Create large table
  local({
    df <- create_large_test_data(nrow = 1000000, ncol = 10)  # ~80 MB
    gpu_df <- tbl_gpu(df)
    expect_data_on_gpu(gpu_df)

    during <- gpu_alloc_snapshot()
    allocation <- gpu_memory_diff(before, during)
    expect_true(allocation > 50 * 1024 * 1024)  # At least 50 MB allocated
  })

  gc_gpu()
  after <- gpu_alloc_snapshot()

  # Memory should be freed
  diff <- after$used_memory - before$used_memory
  expect_true(diff < 50 * 1024 * 1024,
              info = sprintf("Large allocation not freed: %s remaining",
                             format_bytes(diff)))
  # Allocator level: the 80 MB are gone, not just under the 50 MB tolerance.
  expect_lt(diff, 0.1 * 1000000 * 10 * 8)
})
