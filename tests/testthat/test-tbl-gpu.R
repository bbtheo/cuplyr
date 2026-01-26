# Tests for tbl_gpu creation and S3 methods
#
# These tests verify:
# - GPU table creation from data frames
# - Data type conversions
# - S3 methods (dim, names, print, etc.)
# - Data is actually on GPU (not in R memory)

# =============================================================================
# tbl_gpu() Constructor Tests
# =============================================================================

test_that("tbl_gpu() creates valid tbl_gpu object from data frame", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(length(gpu_df$schema$names), ncol(mtcars))
  expect_equal(gpu_df$schema$names, names(mtcars))
})

test_that("tbl_gpu() data actually resides on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Verify data is on GPU
expect_data_on_gpu(gpu_df)

  # Verify pointer is valid
  validation <- validate_gpu_pointer(gpu_df$ptr)
  expect_true(validation$is_externalptr)
  expect_false(validation$is_null_ptr)
  expect_true(validation$can_get_dims)
  expect_true(validation$can_get_types)
})

test_that("tbl_gpu() R object is lightweight (no data copy)", {
  skip_if_no_gpu()

  # Create a reasonably sized data frame
  df <- data.frame(
    a = runif(10000),
    b = runif(10000),
    c = runif(10000)
  )

  gpu_df <- tbl_gpu(df)

  # Verify R object doesn't contain data
  expect_true(verify_no_r_copy(gpu_df))

  # R object should be much smaller than original data
  sizes <- compare_r_vs_gpu_size(gpu_df)
  expect_true(sizes$ratio > 10,
              info = sprintf("GPU data (%s) should be much larger than R object (%s)",
                             sizes$gpu_size_formatted, sizes$r_size_formatted))
})

test_that("tbl_gpu() handles numeric columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1.5, 2.5, 3.5))
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "FLOAT64")
  expect_equal(dim(gpu_df)[1], 3)

  # Verify round-trip
  result <- collect(gpu_df)
  expect_equal(result$x, df$x)
})

test_that("tbl_gpu() handles integer columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1L, 2L, 3L))
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "INT32")

  result <- collect(gpu_df)
  expect_equal(result$x, df$x)
})

test_that("tbl_gpu() handles character columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c("a", "b", "c"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "STRING")

  result <- collect(gpu_df)
  expect_equal(result$x, df$x)
})

test_that("tbl_gpu() handles logical columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(TRUE, FALSE, TRUE))
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "BOOL8")

  result <- collect(gpu_df)
  expect_equal(result$x, df$x)
})

test_that("tbl_gpu() handles Date columns", {
  skip_if_no_gpu()

  df <- data.frame(x = as.Date(c("2024-01-01", "2024-01-02", NA)))
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "TIMESTAMP_DAYS")

  result <- collect(gpu_df)
  expect_true(inherits(result$x, "Date"))
  expect_equal(result$x, df$x)
})

test_that("tbl_gpu() handles POSIXct columns", {
  skip_if_no_gpu()

  df <- data.frame(x = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-02 12:34:56"), tz = "UTC"))
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "TIMESTAMP_MICROSECONDS")

  result <- collect(gpu_df)
  expect_true(inherits(result$x, "POSIXct"))
  expect_equal(result$x, df$x)
})

test_that("tbl_gpu() maps factor columns to DICTIONARY32", {
  skip_if_no_gpu()

  df <- data.frame(x = factor(c("a", "b", "a")))
  gpu_df <- tbl_gpu(df)

  expect_equal(unname(gpu_df$schema$types), "DICTIONARY32")
})

test_that("tbl_gpu() handles NA values in numeric columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1.0, NA, 3.0))
  gpu_df <- tbl_gpu(df)

  result <- collect(gpu_df)
  expect_true(is.na(result$x[2]))
  expect_equal(result$x[1], 1.0)
  expect_equal(result$x[3], 3.0)
})

test_that("tbl_gpu() handles NA values in integer columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1L, NA_integer_, 3L))
  gpu_df <- tbl_gpu(df)

  result <- collect(gpu_df)
  expect_true(is.na(result$x[2]))
})

test_that("tbl_gpu() handles NA values in character columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c("a", NA_character_, "c"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)

  result <- collect(gpu_df)
  expect_true(is.na(result$x[2]))
  expect_equal(result$x[1], "a")
  expect_equal(result$x[3], "c")
})

test_that("tbl_gpu() handles mixed column types", {
  skip_if_no_gpu()

  df <- data.frame(
    int_col = 1:3,
    dbl_col = c(1.1, 2.2, 3.3),
    chr_col = c("a", "b", "c"),
    stringsAsFactors = FALSE
  )

  gpu_df <- tbl_gpu(df)

  expect_equal(length(gpu_df$schema$types), 3)
  expect_equal(unname(gpu_df$schema$types[1]), "INT32")
  expect_equal(unname(gpu_df$schema$types[2]), "FLOAT64")
  expect_equal(unname(gpu_df$schema$types[3]), "STRING")
})

test_that("tbl_gpu() handles empty data frame", {
  skip_if_no_gpu()

  df <- data.frame(x = numeric(0), y = character(0))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df)[1], 0)
  expect_equal(dim(gpu_df)[2], 2)
})

test_that("tbl_gpu() handles single row data frame", {
  skip_if_no_gpu()

  df <- data.frame(x = 1, y = "a", stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)

  expect_equal(dim(gpu_df), c(1L, 2L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 1)
})

test_that("tbl_gpu() handles single column data frame", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10)
  gpu_df <- tbl_gpu(df)

  expect_equal(dim(gpu_df)[2], 1)
})

test_that("tbl_gpu() returns input if already tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  gpu_df2 <- tbl_gpu(gpu_df)

  # Should be the same object
  expect_identical(gpu_df$ptr, gpu_df2$ptr)
})

test_that("tbl_gpu() preserves column names", {
  skip_if_no_gpu()

  df <- data.frame(
    col_with_underscore = 1:3,
    CamelCase = 4:6,
    `spaced name` = 7:9,
    check.names = FALSE
  )

  gpu_df <- tbl_gpu(df)

  expect_equal(names(gpu_df), names(df))
})

# =============================================================================
# is_tbl_gpu() Tests
# =============================================================================

test_that("is_tbl_gpu() returns TRUE for tbl_gpu objects", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  expect_true(is_tbl_gpu(gpu_df))
})

test_that("is_tbl_gpu() returns FALSE for regular data frames", {
  expect_false(is_tbl_gpu(mtcars))
  expect_false(is_tbl_gpu(iris))
})

test_that("is_tbl_gpu() returns FALSE for tibbles", {
  expect_false(is_tbl_gpu(tibble::as_tibble(mtcars)))
})

test_that("is_tbl_gpu() returns FALSE for non-data objects", {
  expect_false(is_tbl_gpu(1:10))
  expect_false(is_tbl_gpu("string"))
  expect_false(is_tbl_gpu(list(a = 1)))
  expect_false(is_tbl_gpu(NULL))
})

# =============================================================================
# as_tbl_gpu() Tests
# =============================================================================

test_that("as_tbl_gpu() converts data frames to tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- as_tbl_gpu(mtcars)
  expect_true(is_tbl_gpu(gpu_df))
  expect_data_on_gpu(gpu_df)
})

test_that("as_tbl_gpu() works with tibbles", {
  skip_if_no_gpu()

  tbl <- tibble::tibble(x = 1:10, y = letters[1:10])
  gpu_df <- as_tbl_gpu(tbl)

  expect_true(is_tbl_gpu(gpu_df))
  expect_equal(dim(gpu_df), c(10L, 2L))
})

# =============================================================================
# dim() Method Tests
# =============================================================================

test_that("dim.tbl_gpu() returns correct dimensions", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  dims <- dim(gpu_df)

  expect_equal(dims, c(32L, 11L))
  expect_type(dims, "integer")
})

test_that("dim.tbl_gpu() works after operations", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # After filter
  filtered <- dplyr::filter(gpu_df, mpg > 20)
  dims <- dim(filtered)
  expect_equal(dims[2], 11L)
  expect_true(dims[1] < 32L)  # Should have fewer rows
})

test_that("nrow() and ncol() work with tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_equal(nrow(gpu_df), 32L)
  expect_equal(ncol(gpu_df), 11L)
})

# =============================================================================
# names() Method Tests
# =============================================================================

test_that("names.tbl_gpu() returns column names", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_equal(names(gpu_df), names(mtcars))
})

test_that("names<-.tbl_gpu() can set column names", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars[, 1:3])
  names(gpu_df) <- c("new1", "new2", "new3")

  expect_equal(names(gpu_df), c("new1", "new2", "new3"))
})

# =============================================================================
# print() Method Tests
# =============================================================================

test_that("print.tbl_gpu() produces output", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  output <- capture.output(print(gpu_df))

  # Should produce some output
  expect_true(length(output) > 0)

  # Should show dimensions
  output_text <- paste(output, collapse = "\n")
  expect_match(output_text, "Rows:|32")
  expect_match(output_text, "Columns:|11")
})

test_that("print.tbl_gpu() shows column types", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  output <- capture.output(print(gpu_df))
  output_text <- paste(output, collapse = "\n")

  # Should show type indicators
  expect_match(output_text, "<dbl>|<int>|<chr>")
})

test_that("print.tbl_gpu() shows column values preview", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1.5, 2.5, 3.5))
  gpu_df <- tbl_gpu(df)

  output <- capture.output(print(gpu_df))
  output_text <- paste(output, collapse = "\n")

  # Should show some actual values
  expect_match(output_text, "1\\.5|2\\.5|3\\.5")
})

test_that("print.tbl_gpu() handles large tables gracefully", {
  skip_if_no_gpu()

  df <- data.frame(matrix(1:1000, ncol = 10))
  gpu_df <- tbl_gpu(df)

  # Should not error or hang
  output <- capture.output(print(gpu_df))
  expect_true(length(output) > 0)
})

# =============================================================================
# GPU Memory Allocation Tests
# =============================================================================

test_that("tbl_gpu() allocates GPU memory", {
  skip_if_no_gpu()

  gc_gpu()

  before <- gpu_memory_snapshot()
  gpu_df <- tbl_gpu(mtcars)
  after <- gpu_memory_snapshot()

  # Verify memory was allocated
  expect_true(after$used_memory >= before$used_memory)
})

test_that("tbl_gpu() GPU memory is freed on garbage collection", {
  skip_if_no_gpu()

  # Force clean state
  gc_gpu()
  before <- gpu_memory_snapshot()

  # Create and immediately discard GPU table
  local({
    gpu_df <- tbl_gpu(create_large_test_data(nrow = 100000, ncol = 5))
    expect_data_on_gpu(gpu_df)
  })

  # Force garbage collection
  gc_gpu()
  after <- gpu_memory_snapshot()

  # Memory should be approximately back to before
  # (allowing some tolerance for other processes)
  diff <- after$used_memory - before$used_memory
  tolerance <- 50 * 1024 * 1024  # 50 MB tolerance

  expect_true(
    diff < tolerance,
    info = sprintf("Memory not freed after GC: before=%s, after=%s, diff=%s",
                   format_bytes(before$used_memory),
                   format_bytes(after$used_memory),
                   format_bytes(diff))
  )
})

test_that("Multiple tbl_gpu objects can coexist", {
  skip_if_no_gpu()

  gpu_df1 <- tbl_gpu(mtcars)
  gpu_df2 <- tbl_gpu(iris[, 1:4])
  gpu_df3 <- tbl_gpu(cars)

  # All should have data on GPU
  expect_data_on_gpu(gpu_df1)
  expect_data_on_gpu(gpu_df2)
  expect_data_on_gpu(gpu_df3)

  # All should have different pointers
  expect_false(identical(gpu_df1$ptr, gpu_df2$ptr))
  expect_false(identical(gpu_df2$ptr, gpu_df3$ptr))
})
