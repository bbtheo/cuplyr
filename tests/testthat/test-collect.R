# Tests for collect.tbl_gpu()
#
# These tests verify:
# - Data transfer from GPU to R
# - Type conversions
# - NA handling
# - Return type is tibble
# - Data integrity after round-trip

# =============================================================================
# Basic collect() Tests
# =============================================================================

test_that("collect() returns a tibble", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- collect(gpu_df)

  expect_s3_class(result, "tbl_df")
  expect_s3_class(result, "tbl")
  expect_s3_class(result, "data.frame")
})

test_that("collect() returns correct dimensions", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- collect(gpu_df)

  expect_equal(nrow(result), nrow(mtcars))
  expect_equal(ncol(result), ncol(mtcars))
})

test_that("collect() returns correct column names", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- collect(gpu_df)

  expect_equal(names(result), names(mtcars))
})

test_that("collect() returns correct data values", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- collect(gpu_df)

  for (col in names(mtcars)) {
    expect_equal(result[[col]], mtcars[[col]],
                 info = paste("Column", col, "values differ"))
  }
})

test_that("collect() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- df |>
    dplyr::filter(mpg > 20) |>
    dplyr::mutate(kpl = mpg * 0.425) |>
    dplyr::select(mpg, kpl, hp)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::filter(mpg > 20) |>
      dplyr::mutate(kpl = mpg * 0.425) |>
      dplyr::select(mpg, kpl, hp) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})

# =============================================================================
# Type Conversion Tests
# =============================================================================

test_that("collect() converts numeric columns correctly", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1.5, 2.5, 3.5))
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_type(result$x, "double")
  expect_equal(result$x, df$x)
})

test_that("collect() converts integer columns correctly", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1L, 2L, 3L))
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_type(result$x, "integer")
  expect_equal(result$x, df$x)
})

test_that("collect() converts character columns correctly", {
  skip_if_no_gpu()

  df <- data.frame(x = c("apple", "banana", "cherry"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_type(result$x, "character")
  expect_equal(result$x, df$x)
})

test_that("collect() handles empty strings", {
  skip_if_no_gpu()

  df <- data.frame(x = c("a", "", "c"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(result$x, df$x)
  expect_equal(result$x[2], "")
})

test_that("collect() handles long strings", {
  skip_if_no_gpu()

  long_string <- paste(rep("abcdefghij", 100), collapse = "")
  df <- data.frame(x = c(long_string, "short"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(result$x[1], long_string)
})

# =============================================================================
# NA Handling Tests
# =============================================================================

test_that("collect() preserves NA in numeric columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1.0, NA, 3.0, NA, 5.0))
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_true(is.na(result$x[2]))
  expect_true(is.na(result$x[4]))
  expect_equal(result$x[1], 1.0)
  expect_equal(result$x[3], 3.0)
  expect_equal(result$x[5], 5.0)
})

test_that("collect() preserves NA in integer columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1L, NA_integer_, 3L))
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_true(is.na(result$x[2]))
  expect_equal(result$x[1], 1L)
  expect_equal(result$x[3], 3L)
})

test_that("collect() preserves NA in character columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c("a", NA_character_, "c"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_true(is.na(result$x[2]))
  expect_equal(result$x[1], "a")
  expect_equal(result$x[3], "c")
})

test_that("collect() handles all-NA columns", {
  skip_if_no_gpu()

  df <- data.frame(x = rep(NA_real_, 5))
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_true(all(is.na(result$x)))
})

test_that("collect() preserves NA positions correctly", {
  skip_if_no_gpu()

  df <- data.frame(
    a = c(1, NA, 3, NA, 5),
    b = c(NA, 2, NA, 4, NA)
  )
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(which(is.na(result$a)), c(2, 4))
  expect_equal(which(is.na(result$b)), c(1, 3, 5))
})

# =============================================================================
# Mixed Column Type Tests
# =============================================================================

test_that("collect() handles mixed column types", {
  skip_if_no_gpu()

  df <- data.frame(
    int_col = 1:5,
    dbl_col = c(1.1, 2.2, 3.3, 4.4, 5.5),
    chr_col = letters[1:5],
    stringsAsFactors = FALSE
  )

  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_type(result$int_col, "integer")
  expect_type(result$dbl_col, "double")
  expect_type(result$chr_col, "character")

  expect_equal(result$int_col, df$int_col)
  expect_equal(result$dbl_col, df$dbl_col)
  expect_equal(result$chr_col, df$chr_col)
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("collect() handles empty table", {
  skip_if_no_gpu()

  df <- data.frame(x = numeric(0), y = character(0))
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(nrow(result), 0)
  expect_equal(ncol(result), 2)
  expect_equal(names(result), c("x", "y"))
})

test_that("collect() handles single row", {
  skip_if_no_gpu()

  df <- data.frame(x = 42, y = "test", stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(nrow(result), 1)
  expect_equal(result$x, 42)
  expect_equal(result$y, "test")
})

test_that("collect() handles single column", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:100)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(ncol(result), 1)
  expect_equal(result$x, 1:100)
})

test_that("collect() errors on NULL pointer", {
  skip_if_no_gpu()

  # Create a fake tbl_gpu with NULL pointer
  fake_gpu <- structure(
    list(
      ptr = NULL,
      schema = list(names = "x", types = "FLOAT64"),
      lazy_ops = list(),
      groups = character()
    ),
    class = c("tbl_gpu", "tbl_lazy", "tbl")
  )

  expect_error(collect(fake_gpu), "NULL|pointer")
})

# =============================================================================
# Round-trip Integrity Tests
# =============================================================================

test_that("collect() maintains data integrity on round-trip", {
  skip_if_no_gpu()

  # Original data
  original <- mtcars

  # Round-trip: R -> GPU -> R
  gpu_df <- tbl_gpu(original)
  result <- collect(gpu_df)

  # Compare each column
  for (col in names(original)) {
    expect_equal(result[[col]], original[[col]],
                 tolerance = 1e-10,
                 info = paste("Round-trip integrity failed for column:", col))
  }
})

test_that("collect() maintains data integrity after operations", {
  skip_if_no_gpu()

  original <- mtcars
  gpu_df <- tbl_gpu(original)

  # Apply operations
  result <- gpu_df |>
    dplyr::filter(mpg > 15) |>
    dplyr::select(mpg, cyl, hp) |>
    dplyr::mutate(hp_per_cyl = hp / cyl) |>
    collect()

  # Verify the operations were correct
  expected <- original[original$mpg > 15, c("mpg", "cyl", "hp")]
  expected$hp_per_cyl <- expected$hp / expected$cyl

  expect_equal(result$mpg, expected$mpg)
  expect_equal(result$cyl, expected$cyl)
  expect_equal(result$hp, expected$hp)
  expect_equal(result$hp_per_cyl, expected$hp_per_cyl, tolerance = 1e-10)
})

test_that("collect() can be called multiple times", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result1 <- collect(gpu_df)
  result2 <- collect(gpu_df)
  result3 <- collect(gpu_df)

  expect_equal(result1, result2)
  expect_equal(result2, result3)
})

test_that("collect() does not modify GPU data", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Collect multiple times
  collect(gpu_df)
  collect(gpu_df)

  # GPU data should still be valid
  expect_data_on_gpu(gpu_df)

  result <- collect(gpu_df)
  expect_equal(nrow(result), nrow(mtcars))
})

# =============================================================================
# Precision Tests
# =============================================================================

test_that("collect() maintains numeric precision", {
  skip_if_no_gpu()

  # Test with various magnitudes
  df <- data.frame(
    small = c(1e-10, 2e-10, 3e-10),
    medium = c(1.0, 2.0, 3.0),
    large = c(1e10, 2e10, 3e10)
  )

  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(result$small, df$small, tolerance = 1e-15)
  expect_equal(result$medium, df$medium, tolerance = 1e-15)
  expect_equal(result$large, df$large, tolerance = 1e-5)  # Relative tolerance
})

test_that("collect() handles special numeric values", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(0, -0, Inf, -Inf)
  )

  # Note: NaN handling may vary
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(result$x[1], 0)
  expect_true(is.infinite(result$x[3]) && result$x[3] > 0)
  expect_true(is.infinite(result$x[4]) && result$x[4] < 0)
})

# =============================================================================
# Large Data Tests
# =============================================================================

test_that("collect() works with large datasets", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(500 * 1024 * 1024)

  df <- create_large_test_data(nrow = 100000, ncol = 10)
  gpu_df <- tbl_gpu(df)

  result <- collect(gpu_df)

  expect_equal(nrow(result), 100000)
  expect_equal(ncol(result), 10)

  # Spot check some values
  expect_equal(result$col1, df$col1, tolerance = 1e-10)
  expect_equal(result$col5, df$col5, tolerance = 1e-10)
})

test_that("collect() after operations on large data", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(500 * 1024 * 1024)

  df <- create_large_test_data(nrow = 100000, ncol = 5)
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::filter(col1 > 0.5) |>
    dplyr::mutate(sum = col1 + col2) |>
    dplyr::select(col1, sum) |>
    collect()

  # Verify filter worked
  expect_true(all(result$col1 > 0.5))

  # Verify mutate worked
  expected_sum <- df$col1[df$col1 > 0.5] + df$col2[df$col1 > 0.5]
  expect_equal(result$sum, expected_sum, tolerance = 1e-10)
})

# =============================================================================
# Memory Behavior Tests
# =============================================================================

test_that("collect() creates new R memory (not view)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- collect(gpu_df)

  # Modifying result should not affect GPU data
  result$mpg[1] <- -999

  # Collect again - should have original value
  result2 <- collect(gpu_df)
  expect_equal(result2$mpg[1], mtcars$mpg[1])
})

test_that("collect() result can be modified independently", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result1 <- collect(gpu_df)
  result2 <- collect(gpu_df)

  # Modify result1
  result1$new_col <- 1:nrow(result1)

  # result2 should be unaffected
  expect_false("new_col" %in% names(result2))
})
