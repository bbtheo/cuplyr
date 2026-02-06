# Tests for select.tbl_gpu()
#
# These tests verify:
# - Column selection by name
# - Column reordering
# - tidyselect helper functions
# - Data remains on GPU after selection

# =============================================================================
# Basic Column Selection
# =============================================================================

test_that("select() with single column works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg)

  expect_data_on_gpu(selected)
  expect_equal(names(selected), "mpg")
  expect_equal(dim(selected)[2], 1)
})

test_that("select() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- dplyr::select(df, mpg, cyl, hp)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::select(mpg, cyl, hp) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})

test_that("select() with multiple columns works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl, hp)

  expect_data_on_gpu(selected)
  expect_equal(names(selected), c("mpg", "cyl", "hp"))
  expect_equal(dim(selected)[2], 3)
})

test_that("select() preserves row count", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl)

  expect_equal(dim(selected)[1], nrow(mtcars))
})

test_that("select() reorders columns", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, hp, mpg, wt)

  expect_equal(names(selected), c("hp", "mpg", "wt"))

  result <- collect(selected)
  expect_equal(result$hp, mtcars$hp)
  expect_equal(result$mpg, mtcars$mpg)
  expect_equal(result$wt, mtcars$wt)
})

test_that("select() preserves data values", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl)

  result <- collect(selected)
  expect_equal(result$mpg, mtcars$mpg)
  expect_equal(result$cyl, mtcars$cyl)
})

# =============================================================================
# tidyselect Helper Functions
# =============================================================================

test_that("select() with starts_with() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, starts_with("d"))

  expect_data_on_gpu(selected)

  # Should select 'disp' and 'drat'
  expected_cols <- names(mtcars)[startsWith(names(mtcars), "d")]
  expect_equal(names(selected), expected_cols)
})

test_that("select() with ends_with() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, ends_with("t"))

  # Should select columns ending with 't'
  expected_cols <- names(mtcars)[endsWith(names(mtcars), "t")]
  expect_equal(names(selected), expected_cols)
})

test_that("select() with contains() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, contains("ar"))

  # Should select columns containing 'ar' (gear, carb)
  expected_cols <- names(mtcars)[grepl("ar", names(mtcars))]
  expect_equal(names(selected), expected_cols)
})

test_that("select() with everything() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, everything())

  expect_equal(names(selected), names(mtcars))
  expect_equal(dim(selected)[2], ncol(mtcars))
})

test_that("select() with mixed selection and helpers works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, starts_with("c"))

  # Should include mpg and columns starting with 'c' (cyl, carb)
  expect_true("mpg" %in% names(selected))
  expect_true("cyl" %in% names(selected))
  expect_true("carb" %in% names(selected))
})

# =============================================================================
# Select with Custom Column Names
# =============================================================================

test_that("select() works with column names containing dots", {
  skip_if_no_gpu()

  df <- data.frame(col.one = 1:3, col.two = 4:6, check.names = FALSE)
  gpu_df <- tbl_gpu(df)

  selected <- dplyr::select(gpu_df, col.one)

  expect_equal(names(selected), "col.one")
})

test_that("select() works with column names containing underscores", {
  skip_if_no_gpu()

  df <- data.frame(col_one = 1:3, col_two = 4:6)
  gpu_df <- tbl_gpu(df)

  selected <- dplyr::select(gpu_df, col_one)

  expect_equal(names(selected), "col_one")
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("select() all columns returns same structure", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb)

  expect_equal(names(selected), names(mtcars))
})

test_that("select() single column from large table works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg)

  expect_equal(dim(selected)[2], 1)
  expect_equal(dim(selected)[1], 32)
})

test_that("select() errors when no columns match", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::select(gpu_df, starts_with("xyz")),
    "no columns|resulted in no"
  )
})

# =============================================================================
# Data Residency Tests
# =============================================================================

test_that("select() result stays on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl, hp)

  expect_data_on_gpu(selected)
  expect_true(verify_no_r_copy(selected))
  expect_lightweight_r_object(selected)
})

test_that("select() creates new GPU allocation", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl)

  # Should have different pointers
  expect_false(identical(gpu_df$ptr, selected$ptr))

  # Original should be unchanged
  expect_equal(ncol(gpu_df), 11)
  expect_data_on_gpu(gpu_df)
})

test_that("select() reduces schema appropriately", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, mpg, cyl)

  expect_equal(length(selected$schema$names), 2)
  expect_equal(length(selected$schema$types), 2)
  expect_equal(selected$schema$names, c("mpg", "cyl"))
})

# =============================================================================
# Chained Operations
# =============================================================================

test_that("select() followed by filter() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- gpu_df |>
    dplyr::select(mpg, cyl, hp) |>
    dplyr::filter(mpg > 20)

  expect_data_on_gpu(result)
  expect_equal(names(result), c("mpg", "cyl", "hp"))

  collected <- collect(result)
  expect_true(all(collected$mpg > 20))
})

test_that("filter() followed by select() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- gpu_df |>
    dplyr::filter(mpg > 20) |>
    dplyr::select(mpg, cyl)

  expect_data_on_gpu(result)
  expect_equal(names(result), c("mpg", "cyl"))

  collected <- collect(result)
  expect_true(all(collected$mpg > 20))
})

test_that("select() followed by mutate() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- gpu_df |>
    dplyr::select(mpg, hp, wt) |>
    dplyr::mutate(power_weight = hp / wt)

  expect_data_on_gpu(result)
  expect_equal(names(result), c("mpg", "hp", "wt", "power_weight"))
})

# =============================================================================
# Multiple select() Calls
# =============================================================================

test_that("chained select() calls work", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- gpu_df |>
    dplyr::select(mpg, cyl, hp, wt) |>
    dplyr::select(mpg, hp)

  expect_equal(names(result), c("mpg", "hp"))
  expect_equal(dim(result)[2], 2)
})

test_that("select() can further narrow selection", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  step1 <- dplyr::select(gpu_df, mpg, cyl, hp, wt, gear)
  expect_equal(ncol(step1), 5)

  step2 <- dplyr::select(step1, mpg, hp)
  expect_equal(ncol(step2), 2)

  step3 <- dplyr::select(step2, mpg)
  expect_equal(ncol(step3), 1)
})

# =============================================================================
# Type Preservation Tests
# =============================================================================

test_that("select() preserves column types", {
  skip_if_no_gpu()

  df <- data.frame(
    int_col = 1:3,
    dbl_col = c(1.1, 2.2, 3.3),
    chr_col = c("a", "b", "c"),
    stringsAsFactors = FALSE
  )

  gpu_df <- tbl_gpu(df)
  selected <- dplyr::select(gpu_df, dbl_col, chr_col)

  expect_equal(unname(selected$schema$types), c("FLOAT64", "STRING"))

  result <- collect(selected)
  expect_type(result$dbl_col, "double")
  expect_type(result$chr_col, "character")
})

# =============================================================================
# Large Data Tests
# =============================================================================

test_that("select() works with large datasets", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(500 * 1024 * 1024)

  df <- create_large_test_data(nrow = 100000, ncol = 10)
  gpu_df <- tbl_gpu(df)

  # Select only 2 columns
  selected <- dplyr::select(gpu_df, col1, col5)

  expect_data_on_gpu(selected)
  expect_equal(dim(selected)[1], 100000)
  expect_equal(dim(selected)[2], 2)

  result <- collect(selected)
  expect_equal(result$col1, df$col1)
  expect_equal(result$col5, df$col5)
})

test_that("select() reduces memory footprint", {
  skip_if_no_gpu()

  df <- create_large_test_data(nrow = 10000, ncol = 10)
  gpu_df <- tbl_gpu(df)

  selected <- dplyr::select(gpu_df, col1, col2)

  # Estimated size should be smaller
  full_size <- estimate_gpu_table_size(gpu_df)
  selected_size <- estimate_gpu_table_size(selected)

  expect_true(selected_size < full_size)
  # Should be approximately 2/10 of original
  expect_true(selected_size < full_size * 0.4)
})
