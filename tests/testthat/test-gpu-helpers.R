# Tests for GPU memory helper functions
#
# These tests verify the exported helper functions:
# - gpu_memory_usage()
# - gpu_object_info()
# - verify_gpu_data()
# - gpu_size_comparison()
# - gpu_memory_state()

# =============================================================================
# gpu_memory_usage() Tests
# =============================================================================

test_that("gpu_memory_usage() returns numeric for valid tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  size <- gpu_memory_usage(gpu_df)

  expect_type(size, "double")
  expect_true(!is.na(size))
  expect_true(size > 0)
})

test_that("gpu_memory_usage() returns NA for non-tbl_gpu", {
  size <- gpu_memory_usage(mtcars)
  expect_true(is.na(size))

  size <- gpu_memory_usage(NULL)
  expect_true(is.na(size))

  size <- gpu_memory_usage(list(a = 1))
  expect_true(is.na(size))
})

test_that("gpu_memory_usage() scales with data size", {
  skip_if_no_gpu()

  small <- tbl_gpu(data.frame(x = runif(100)))
  large <- tbl_gpu(data.frame(x = runif(10000)))

  small_size <- gpu_memory_usage(small)
  large_size <- gpu_memory_usage(large)

  # Large should be ~100x bigger
  ratio <- large_size / small_size
  expect_true(ratio > 50 && ratio < 150)
})

test_that("gpu_memory_usage() accounts for column count", {
  skip_if_no_gpu()

  one_col <- tbl_gpu(data.frame(a = runif(1000)))
  five_col <- tbl_gpu(data.frame(
    a = runif(1000), b = runif(1000), c = runif(1000),
    d = runif(1000), e = runif(1000)
  ))

  one_size <- gpu_memory_usage(one_col)
  five_size <- gpu_memory_usage(five_col)

  # Five columns should be ~5x bigger
  ratio <- five_size / one_size
  expect_true(ratio > 3 && ratio < 7)
})

test_that("gpu_memory_usage() accounts for column types", {
  skip_if_no_gpu()

  int_col <- tbl_gpu(data.frame(x = 1:1000))  # INT32 = 4 bytes
  dbl_col <- tbl_gpu(data.frame(x = as.double(1:1000)))  # FLOAT64 = 8 bytes

  int_size <- gpu_memory_usage(int_col)
  dbl_size <- gpu_memory_usage(dbl_col)

  # Double should be ~2x bigger
  ratio <- dbl_size / int_size
  expect_true(ratio > 1.5 && ratio < 2.5)
})

# =============================================================================
# gpu_object_info() Tests
# =============================================================================

test_that("gpu_object_info() returns complete info for valid tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  info <- gpu_object_info(gpu_df)

  expect_type(info, "list")
  expect_true(info$valid)
  expect_equal(info$nrow, 32L)
  expect_equal(info$ncol, 11L)
  expect_equal(info$column_names, names(mtcars))
  expect_equal(length(info$column_types), 11)
  expect_true(!is.na(info$estimated_gpu_bytes))
  expect_true(!is.na(info$estimated_gpu_mb))
  expect_true(info$r_object_bytes > 0)
  expect_true(info$data_on_gpu)
  expect_true(info$pointer_valid)
})

test_that("gpu_object_info() handles non-tbl_gpu gracefully", {
  info <- gpu_object_info(mtcars)

  expect_false(info$valid)
  expect_true(is.na(info$nrow))
  expect_false(info$data_on_gpu)
})

test_that("gpu_object_info() shows small R object size", {
  skip_if_no_gpu()

  # Create large GPU table
  df <- data.frame(matrix(runif(50000), ncol = 10))
  gpu_df <- tbl_gpu(df)

  info <- gpu_object_info(gpu_df)

  # R object should be small (metadata only)
  expect_true(info$r_object_bytes < 50000)

  # GPU data should be larger
  expect_true(info$estimated_gpu_bytes > info$r_object_bytes)
})

# =============================================================================
# verify_gpu_data() Tests
# =============================================================================

test_that("verify_gpu_data() returns TRUE for valid tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  expect_true(verify_gpu_data(gpu_df))
})

test_that("verify_gpu_data() returns FALSE for non-tbl_gpu", {
  expect_false(verify_gpu_data(mtcars))
  expect_false(verify_gpu_data(iris))
  expect_false(verify_gpu_data(NULL))
  expect_false(verify_gpu_data(list(a = 1)))
})

test_that("verify_gpu_data() returns FALSE for tbl_gpu with NULL pointer", {
  # Create fake tbl_gpu with NULL pointer
  fake <- structure(
    list(
      ptr = NULL,
      schema = list(names = "x", types = "FLOAT64"),
      lazy_ops = list(),
      groups = character()
    ),
    class = c("tbl_gpu", "tbl_lazy", "tbl")
  )

  expect_false(verify_gpu_data(fake))
})

test_that("verify_gpu_data() works after operations", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path

  gpu_df <- tbl_gpu(mtcars)

  filtered <- dplyr::filter(gpu_df, mpg > 20)
  expect_true(verify_gpu_data(filtered))

  mutated <- dplyr::mutate(gpu_df, new = mpg + 1)
  expect_true(verify_gpu_data(mutated))

  selected <- dplyr::select(gpu_df, mpg, cyl)
  expect_true(verify_gpu_data(selected))
})

# =============================================================================
# gpu_size_comparison() Tests
# =============================================================================

test_that("gpu_size_comparison() returns expected structure", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  comp <- gpu_size_comparison(gpu_df)

  expect_type(comp, "list")
  expect_true("r_bytes" %in% names(comp))
  expect_true("gpu_bytes" %in% names(comp))
  expect_true("ratio" %in% names(comp))
})

test_that("gpu_size_comparison() shows GPU is larger than R object", {
  skip_if_no_gpu()

  df <- data.frame(matrix(runif(10000), ncol = 10))
  gpu_df <- tbl_gpu(df)

  comp <- gpu_size_comparison(gpu_df)

  expect_true(comp$gpu_bytes > comp$r_bytes)
  expect_true(comp$ratio > 1)
})

test_that("gpu_size_comparison() ratio increases with data size", {
  skip_if_no_gpu()

  small <- tbl_gpu(data.frame(x = runif(100)))
  large <- tbl_gpu(data.frame(x = runif(100000)))

  small_comp <- gpu_size_comparison(small)
  large_comp <- gpu_size_comparison(large)

  # Ratio should be higher for larger data
  # (R object grows slower than GPU data)
  expect_true(large_comp$ratio > small_comp$ratio)
})

test_that("gpu_size_comparison() handles non-tbl_gpu", {
  comp <- gpu_size_comparison(mtcars)

  expect_true(comp$r_bytes > 0)  # R size still computed
  expect_true(is.na(comp$gpu_bytes))
  expect_true(is.na(comp$ratio))
})

# =============================================================================
# gpu_memory_state() Tests
# =============================================================================

test_that("gpu_memory_state() returns expected structure with GPU", {
  skip_if_no_gpu()

  state <- gpu_memory_state()

  expect_type(state, "list")
  expect_true(state$available)
  expect_true(!is.na(state$total_bytes))
  expect_true(!is.na(state$free_bytes))
  expect_true(!is.na(state$used_bytes))
  expect_true(!is.na(state$total_gb))
  expect_true(!is.na(state$free_gb))
  expect_true(!is.na(state$used_gb))
})

test_that("gpu_memory_state() values are consistent", {
  skip_if_no_gpu()

  state <- gpu_memory_state()

  # used = total - free
  expect_equal(state$used_bytes, state$total_bytes - state$free_bytes)

  # GB conversions are correct
  expect_equal(state$total_gb, state$total_bytes / 1e9)
  expect_equal(state$free_gb, state$free_bytes / 1e9)
})

test_that("gpu_memory_state() reflects allocations", {
  skip_if_no_gpu()

  gc_gpu()  # From helper
  before <- gpu_memory_state()

  # Allocate GPU data
  gpu_df <- tbl_gpu(data.frame(matrix(runif(100000), ncol = 10)))

  after <- gpu_memory_state()

  # Used memory should increase
  expect_true(after$used_bytes >= before$used_bytes)
})

test_that("gpu_memory_state() returns NA values without GPU", {
  skip_if(has_gpu(), "GPU is available, skipping no-GPU test")

  state <- gpu_memory_state()

  expect_false(state$available)
  expect_true(is.na(state$total_bytes))
  expect_true(is.na(state$free_bytes))
})

# =============================================================================
# Combined Helper Function Tests
# =============================================================================

test_that("helper functions work together for complete verification", {
  skip_if_no_gpu()

  df <- data.frame(matrix(runif(10000), ncol = 10))
  gpu_df <- tbl_gpu(df)

  # All verification methods should agree
  expect_true(verify_gpu_data(gpu_df))

  info <- gpu_object_info(gpu_df)
  expect_true(info$valid)
  expect_true(info$data_on_gpu)

  comp <- gpu_size_comparison(gpu_df)
  expect_true(comp$ratio > 1)

  size <- gpu_memory_usage(gpu_df)
  expect_true(size > 0)
})

test_that("helper functions work on operation results", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path

  result <- mtcars |>
    tbl_gpu() |>
    dplyr::filter(mpg > 20) |>
    dplyr::mutate(kpl = mpg * 0.425) |>
    dplyr::select(mpg, kpl)

  # All should verify correctly
  expect_true(verify_gpu_data(result))

  info <- gpu_object_info(result)
  expect_equal(info$ncol, 2L)
  expect_true(info$nrow < 32)  # Filtered

  size <- gpu_memory_usage(result)
  expect_true(size > 0)
})
