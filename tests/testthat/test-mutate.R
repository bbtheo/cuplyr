# Tests for mutate.tbl_gpu()
#
# These tests verify:
# - Creating new columns with arithmetic operations
# - Modifying existing columns
# - Column-to-scalar operations
# - Column-to-column operations
# - Data remains on GPU after mutation

# =============================================================================
# Basic Arithmetic Operations
# =============================================================================

test_that("mutate() with + operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg + 10)

  expect_data_on_gpu(mutated)

  result <- collect(mutated)
  expect_true("new_col" %in% names(result))
  expect_equal(result$new_col, mtcars$mpg + 10)
})

test_that("mutate() with - operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg - 5)

  result <- collect(mutated)
  expect_equal(result$new_col, mtcars$mpg - 5)
})

test_that("mutate() with * operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, kpl = mpg * 0.425)

  result <- collect(mutated)
  expect_equal(result$kpl, mtcars$mpg * 0.425, tolerance = 1e-10)
})

test_that("mutate() with / operator works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg / 2)

  result <- collect(mutated)
  expect_equal(result$new_col, mtcars$mpg / 2, tolerance = 1e-10)
})

test_that("mutate() with ^ operator (power) works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg ^ 2)

  result <- collect(mutated)
  expect_equal(result$new_col, mtcars$mpg ^ 2, tolerance = 1e-10)
})

# =============================================================================
# Column-to-Column Operations
# =============================================================================

test_that("mutate() with column-to-column addition works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, sum_col = mpg + cyl)

  result <- collect(mutated)
  expect_equal(result$sum_col, mtcars$mpg + mtcars$cyl)
})

test_that("mutate() with column-to-column division works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, power_weight = hp / wt)

  result <- collect(mutated)
  expect_equal(result$power_weight, mtcars$hp / mtcars$wt, tolerance = 1e-10)
})

test_that("mutate() with column-to-column multiplication works", {
  skip_if_no_gpu()

  gpu_cars <- tbl_gpu(cars)
  mutated <- dplyr::mutate(gpu_cars, product = speed * dist)

  result <- collect(mutated)
  expect_equal(result$product, cars$speed * cars$dist)
})

test_that("mutate() with column-to-column subtraction works", {
  skip_if_no_gpu()

  df <- data.frame(a = c(10, 20, 30), b = c(1, 2, 3))
  gpu_df <- tbl_gpu(df)
  mutated <- dplyr::mutate(gpu_df, diff = a - b)

  result <- collect(mutated)
  expect_equal(result$diff, c(9, 18, 27))
})

# =============================================================================
# Modifying Existing Columns
# =============================================================================

test_that("mutate() can replace an existing column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, mpg = mpg + 5)

  expect_data_on_gpu(mutated)

  result <- collect(mutated)

  # Column should still be named 'mpg'
  expect_true("mpg" %in% names(result))

  # Value should be modified
  expect_equal(result$mpg, mtcars$mpg + 5)

  # Column order should be preserved
  expect_equal(names(result), names(mtcars))
})

test_that("mutate() column replacement preserves column order", {
  skip_if_no_gpu()

  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  gpu_df <- tbl_gpu(df)

  # Replace middle column
  mutated <- dplyr::mutate(gpu_df, b = b * 2)

  result <- collect(mutated)
  expect_equal(names(result), c("a", "b", "c"))
  expect_equal(result$b, c(8, 10, 12))
})

# =============================================================================
# Chained Mutations
# =============================================================================

test_that("chained mutate() operations work", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- gpu_df |>
    dplyr::mutate(power_weight = hp / wt) |>
    dplyr::mutate(efficiency = mpg * power_weight)

  expect_data_on_gpu(mutated)

  result <- collect(mutated)

  # Both new columns should exist
  expect_true("power_weight" %in% names(result))
  expect_true("efficiency" %in% names(result))

  # Values should be correct
  expected_pw <- mtcars$hp / mtcars$wt
  expected_eff <- mtcars$mpg * expected_pw

  expect_equal(result$power_weight, expected_pw, tolerance = 1e-10)
  expect_equal(result$efficiency, expected_eff, tolerance = 1e-10)
})

test_that("mutate() can use previously created column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # First mutation creates new column
  step1 <- dplyr::mutate(gpu_df, kpl = mpg * 0.425)

  # Second mutation uses the new column
  step2 <- dplyr::mutate(step1, kpl_rounded = kpl * 10)

  result <- collect(step2)
  expect_true("kpl" %in% names(result))
  expect_true("kpl_rounded" %in% names(result))
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("mutate() with no expressions returns unchanged table", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df)

  expect_equal(dim(mutated), dim(gpu_df))
  expect_equal(names(mutated), names(gpu_df))
})

test_that("mutate() with zero scalar works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, zeros = mpg * 0)

  result <- collect(mutated)
  expect_true(all(result$zeros == 0))
})

test_that("mutate() with one scalar works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, same = mpg * 1)

  result <- collect(mutated)
  expect_equal(result$same, mtcars$mpg)
})

test_that("mutate() with negative scalar works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, negated = mpg * -1)

  result <- collect(mutated)
  expect_equal(result$negated, -mtcars$mpg)
})

test_that("mutate() with decimal scalar works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, fractional = mpg * 0.123)

  result <- collect(mutated)
  expect_equal(result$fractional, mtcars$mpg * 0.123, tolerance = 1e-10)
})

test_that("mutate() preserves row count", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg + 10)

  expect_equal(dim(mutated)[1], nrow(mtcars))
})

test_that("mutate() adds column at end for new columns", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg + 10)

  result <- collect(mutated)
  expect_equal(tail(names(result), 1), "new_col")
})

# =============================================================================
# Data Residency Tests
# =============================================================================

test_that("mutate() result stays on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg + 10)

  expect_data_on_gpu(mutated)
  expect_true(verify_no_r_copy(mutated))
  expect_lightweight_r_object(mutated)
})

test_that("mutate() creates new GPU allocation (immutable)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg + 10)

  # Should have different pointers
  expect_false(identical(gpu_df$ptr, mutated$ptr))

  # Original should be unchanged
  expect_equal(ncol(gpu_df), 11)
  expect_data_on_gpu(gpu_df)
})

test_that("chained mutations all stay on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  step1 <- dplyr::mutate(gpu_df, col1 = mpg + 10)
  step2 <- dplyr::mutate(step1, col2 = cyl * 2)
  step3 <- dplyr::mutate(step2, col3 = hp / wt)

  expect_data_on_gpu(step1)
  expect_data_on_gpu(step2)
  expect_data_on_gpu(step3)
})

# =============================================================================
# Error Handling
# =============================================================================

test_that("mutate() errors on non-existent column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::mutate(gpu_df, new_col = nonexistent + 5),
    "not found"
  )
})

test_that("mutate() errors on non-numeric scalar", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::mutate(gpu_df, new_col = mpg + "ten"),
    "numeric scalar"
  )
})

test_that("mutate() errors on vector scalar", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::mutate(gpu_df, new_col = mpg + c(1, 2, 3)),
    "numeric scalar|length"
  )
})

test_that("mutate() errors on unsupported operation", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # %% is not supported
  expect_error(
    dplyr::mutate(gpu_df, new_col = mpg %% 5),
    "only supports"
  )
})

# =============================================================================
# Result Type Tests
# =============================================================================

test_that("mutate() result type is FLOAT64", {
  skip_if_no_gpu()

  df <- data.frame(int_col = 1:10)
  gpu_df <- tbl_gpu(df)

  mutated <- dplyr::mutate(gpu_df, new_col = int_col + 1)

  # Result type should be FLOAT64
  expect_equal(unname(mutated$schema$types[2]), "FLOAT64")
})

test_that("mutate() handles division by zero", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 0))
  gpu_df <- tbl_gpu(df)

  # Division by column containing zero
  mutated <- dplyr::mutate(gpu_df, y = x / x)

  result <- collect(mutated)
  # 0/0 should be NaN
  expect_true(is.nan(result$y[3]) || is.na(result$y[3]))
})

# =============================================================================
# Large Data Tests
# =============================================================================

test_that("mutate() works with large datasets", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(500 * 1024 * 1024)

  df <- create_large_test_data(nrow = 100000, ncol = 10)
  gpu_df <- tbl_gpu(df)

  mutated <- dplyr::mutate(gpu_df, new_col = col1 + col2)

  expect_data_on_gpu(mutated)
  expect_equal(dim(mutated)[1], 100000)
  expect_equal(dim(mutated)[2], 11)  # Original + 1 new

  result <- collect(mutated)
  expect_equal(result$new_col, df$col1 + df$col2, tolerance = 1e-10)
})

# =============================================================================
# Multiple Column Operations
# =============================================================================

test_that("mutate() with multiple expressions in single call works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df,
                           col1 = mpg + 10,
                           col2 = cyl * 2)

  result <- collect(mutated)

  expect_true("col1" %in% names(result))
  expect_true("col2" %in% names(result))
  expect_equal(result$col1, mtcars$mpg + 10)
  expect_equal(result$col2, mtcars$cyl * 2)
})

# =============================================================================
# Column Copy Tests
# =============================================================================

test_that("mutate() can copy a column with a new name", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, am_2 = am)

  expect_data_on_gpu(mutated)

  result <- collect(mutated)

  # New column should exist

  expect_true("am_2" %in% names(result))

  # Values should be identical to original column
  expect_equal(result$am_2, mtcars$am)

  # Original column should still exist
  expect_true("am" %in% names(result))
  expect_equal(result$am, mtcars$am)

  # Should have one more column than original
  expect_equal(ncol(result), ncol(mtcars) + 1)
})

test_that("mutate() can copy multiple columns with new names", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df,
                           mpg_copy = mpg,
                           cyl_copy = cyl)

  result <- collect(mutated)

  expect_equal(result$mpg_copy, mtcars$mpg)
  expect_equal(result$cyl_copy, mtcars$cyl)
})
