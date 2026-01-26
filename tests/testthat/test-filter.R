# Tests for filter.tbl_gpu()
#
# These tests verify:
# - Filter operations with all comparison operators
# - Column-to-scalar comparisons
# - Column-to-column comparisons
# - Chained filter operations
# - Data remains on GPU after filtering

# =============================================================================
# Basic Filter Operations
# =============================================================================

test_that("filter() with > operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 20)

  expect_data_on_gpu(filtered)

  result <- collect(filtered)
  expect_true(all(result$mpg > 20))
  expect_true(nrow(result) > 0)
  expect_true(nrow(result) < 32)
})

test_that("filter() with >= operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg >= 21)

  result <- collect(filtered)
  expect_true(all(result$mpg >= 21))
})

test_that("filter() with < operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg < 15)

  result <- collect(filtered)
  expect_true(all(result$mpg < 15))
})

test_that("filter() with <= operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg <= 15)

  result <- collect(filtered)
  expect_true(all(result$mpg <= 15))
})

test_that("filter() with == operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, cyl == 4)

  result <- collect(filtered)
  expect_true(all(result$cyl == 4))
})

test_that("filter() with != operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, cyl != 4)

  result <- collect(filtered)
  expect_true(all(result$cyl != 4))
})

# =============================================================================
# Column-to-Column Comparisons
# =============================================================================

test_that("filter() with column-to-column comparison works", {
  skip_if_no_gpu()

  gpu_cars <- tbl_gpu(cars)
  filtered <- dplyr::filter(gpu_cars, dist < speed)

  expect_data_on_gpu(filtered)

  result <- collect(filtered)
  expect_true(all(result$dist < result$speed))
})

test_that("filter() column-to-column with different operators", {
  skip_if_no_gpu()

  df <- data.frame(a = c(1, 5, 3, 7), b = c(2, 3, 3, 4))
  gpu_df <- tbl_gpu(df)

  # >
  result1 <- collect(dplyr::filter(gpu_df, a > b))
  expect_true(all(result1$a > result1$b))

  # >=
  result2 <- collect(dplyr::filter(gpu_df, a >= b))
  expect_true(all(result2$a >= result2$b))

  # ==
  result3 <- collect(dplyr::filter(gpu_df, a == b))
  expect_true(all(result3$a == result3$b))
})

# =============================================================================
# Chained Filters
# =============================================================================

test_that("chained filter() operations work", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- gpu_df |>
    dplyr::filter(mpg > 15) |>
    dplyr::filter(cyl == 4)

  expect_data_on_gpu(filtered)

  result <- collect(filtered)
  expect_true(all(result$mpg > 15))
  expect_true(all(result$cyl == 4))
})

test_that("multiple chained filters reduce row count progressively", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  step1 <- dplyr::filter(gpu_df, mpg > 15)
  step2 <- dplyr::filter(step1, cyl == 4)
  step3 <- dplyr::filter(step2, hp < 100)

  n1 <- dim(step1)[1]
  n2 <- dim(step2)[1]
  n3 <- dim(step3)[1]

  expect_true(n1 >= n2)
  expect_true(n2 >= n3)
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("filter() with no matching rows returns empty table", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 1000)

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 0)
  expect_equal(dim(filtered)[2], 11)

  result <- collect(filtered)
  expect_equal(nrow(result), 0)
})

test_that("filter() with all matching rows returns all rows", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 0)

  expect_equal(dim(filtered)[1], 32)

  result <- collect(filtered)
  expect_equal(nrow(result), 32)
})

test_that("filter() with scalar TRUE returns all rows", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, TRUE)

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 32)

  result <- collect(filtered)
  expect_equal(nrow(result), 32)
  expect_equal(result, mtcars, ignore_attr = TRUE)
})

test_that("filter() with scalar FALSE returns no rows", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, FALSE)

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 0)
  expect_equal(dim(filtered)[2], 11)

  result <- collect(filtered)
  expect_equal(nrow(result), 0)
  expect_equal(names(result), names(mtcars))
})

test_that("filter() with vector of TRUE returns all rows", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, rep(TRUE, nrow(mtcars)))

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 32)

  result <- collect(filtered)
  expect_equal(nrow(result), 32)
})

test_that("filter() with vector of FALSE returns no rows", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, rep(FALSE, nrow(mtcars)))

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 0)

  result <- collect(filtered)
  expect_equal(nrow(result), 0)
})

test_that("filter() with no conditions returns unchanged table", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df)

  expect_equal(dim(filtered), dim(gpu_df))
})

test_that("filter() preserves all columns", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 20)

  expect_equal(names(filtered), names(mtcars))
  expect_equal(dim(filtered)[2], ncol(mtcars))

  result <- collect(filtered)
  expect_equal(names(result), names(mtcars))
})

test_that("filter() with decimal values works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 21.4)

  result <- collect(filtered)
  expect_true(all(result$mpg > 21.4))
})

test_that("filter() with negative values works", {
  skip_if_no_gpu()

  df <- data.frame(x = c(-5, -2, 0, 2, 5))
  gpu_df <- tbl_gpu(df)

  filtered <- dplyr::filter(gpu_df, x > -3)
  result <- collect(filtered)
  expect_true(all(result$x > -3))
})

test_that("filter() with zero comparison works", {
  skip_if_no_gpu()

  df <- data.frame(x = c(-5, -2, 0, 2, 5))
  gpu_df <- tbl_gpu(df)

  filtered <- dplyr::filter(gpu_df, x >= 0)
  result <- collect(filtered)
  expect_true(all(result$x >= 0))
  expect_equal(nrow(result), 3)
})

# =============================================================================
# Data Residency Tests
# =============================================================================

test_that("filter() result stays on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 20)

  # Verify data is on GPU, not in R memory
  expect_data_on_gpu(filtered)
  expect_true(verify_no_r_copy(filtered))
  expect_lightweight_r_object(filtered)
})

test_that("filter() creates new GPU allocation (immutable)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 20)

  # Should have different pointers (original unchanged)
  expect_false(identical(gpu_df$ptr, filtered$ptr))

  # Original should still work
  expect_equal(dim(gpu_df)[1], 32)
  expect_data_on_gpu(gpu_df)
})

test_that("chained filters all stay on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  step1 <- dplyr::filter(gpu_df, mpg > 15)
  step2 <- dplyr::filter(step1, cyl == 4)
  step3 <- dplyr::filter(step2, hp < 100)

  expect_data_on_gpu(step1)
  expect_data_on_gpu(step2)
  expect_data_on_gpu(step3)
})

# =============================================================================
# Error Handling
# =============================================================================

test_that("filter() errors on non-existent column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::filter(gpu_df, nonexistent > 5),
    "not found"
  )
})

test_that("filter() errors on non-numeric comparison value", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Character value should error
  expect_error(
    dplyr::filter(gpu_df, mpg > "twenty"),
    "numeric scalar"
  )
})

test_that("filter() errors on unsupported operator", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # %in% is not supported
  expect_error(
    dplyr::filter(gpu_df, cyl %in% c(4, 6)),
    "only supports"
  )
})

# =============================================================================
# Integer Column Tests
# =============================================================================

test_that("filter() works with integer columns", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10)
  gpu_df <- tbl_gpu(df)

  filtered <- dplyr::filter(gpu_df, x > 5)
  result <- collect(filtered)

  expect_equal(result$x, 6:10)
})

test_that("filter() with integer comparison value works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, cyl == 4L)  # Integer literal

  result <- collect(filtered)
  expect_true(all(result$cyl == 4))
})

# =============================================================================
# Large Data Tests
# =============================================================================

test_that("filter() works with large datasets", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(500 * 1024 * 1024)  # Need 500MB

  df <- create_large_test_data(nrow = 100000, ncol = 10)
  gpu_df <- tbl_gpu(df)

  # Filter to approximately half the rows
  filtered <- dplyr::filter(gpu_df, col1 > 0.5)

  expect_data_on_gpu(filtered)

  # Should have roughly half the rows (with some variance)
  dims <- dim(filtered)
  expect_true(dims[1] > 40000)
  expect_true(dims[1] < 60000)

  result <- collect(filtered)
  expect_true(all(result$col1 > 0.5))
})
