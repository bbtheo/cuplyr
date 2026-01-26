# Tests for group_by.tbl_gpu()
#
# These tests verify:
# - Group assignment and retrieval
# - Multiple grouping columns
# - Adding groups vs replacing groups
# - Group preservation through operations
# - Ungrouping

# =============================================================================
# Basic Group Operations
# =============================================================================

test_that("group_by() assigns single column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)

  expect_equal(dplyr::group_vars(grouped), "cyl")
  expect_data_on_gpu(grouped)
})

test_that("group_by() assigns multiple columns", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl, gear)

  expect_equal(dplyr::group_vars(grouped), c("cyl", "gear"))
  expect_data_on_gpu(grouped)
})

test_that("group_by() with no arguments returns unchanged table", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- dplyr::group_by(gpu_df)

  expect_equal(dplyr::group_vars(result), character(0))
  expect_equal(dim(result), dim(gpu_df))
})

test_that("group_by() replaces existing groups by default", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped1 <- dplyr::group_by(gpu_df, cyl)
  grouped2 <- dplyr::group_by(grouped1, gear)

  expect_equal(dplyr::group_vars(grouped2), "gear")
})

test_that("group_by() with .add = TRUE adds to existing groups", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped1 <- dplyr::group_by(gpu_df, cyl)
  grouped2 <- dplyr::group_by(grouped1, gear, .add = TRUE)

  expect_equal(dplyr::group_vars(grouped2), c("cyl", "gear"))
})

# =============================================================================
# Ungroup Operations
# =============================================================================

test_that("ungroup() removes all groups", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl, gear)
  ungrouped <- dplyr::ungroup(grouped)

  expect_equal(dplyr::group_vars(ungrouped), character(0))
  expect_data_on_gpu(ungrouped)
})

test_that("ungroup() on ungrouped table returns same table", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  result <- dplyr::ungroup(gpu_df)

  expect_equal(dplyr::group_vars(result), character(0))
  expect_equal(dim(result), dim(gpu_df))
})

# =============================================================================
# groups() and group_vars() Helpers
# =============================================================================

test_that("groups() returns list of symbols", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl, gear)

  grps <- dplyr::groups(grouped)
  expect_type(grps, "list")
  expect_length(grps, 2)
  expect_true(is.symbol(grps[[1]]))
  expect_true(is.symbol(grps[[2]]))
  expect_equal(as.character(grps[[1]]), "cyl")
  expect_equal(as.character(grps[[2]]), "gear")
})

test_that("group_vars() returns character vector", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)

  vars <- dplyr::group_vars(grouped)
  expect_type(vars, "character")
  expect_equal(vars, "cyl")
})

# =============================================================================
# Group Preservation Through Operations
# =============================================================================

test_that("filter() preserves groups", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)
  filtered <- dplyr::filter(grouped, mpg > 20)

  expect_equal(dplyr::group_vars(filtered), "cyl")
})

test_that("select() preserves groups when group columns selected", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)
  selected <- dplyr::select(grouped, cyl, mpg)

  # Check groups are preserved
  expect_equal(dplyr::group_vars(selected), "cyl")
})

# =============================================================================
# Error Handling
# =============================================================================

test_that("group_by() errors on non-existent column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::group_by(gpu_df, nonexistent),
    "not found"
  )
})

test_that("group_by() errors with helpful message listing available columns", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::group_by(gpu_df, bad_column),
    "Available columns"
  )
})

# =============================================================================
# Data Residency Tests
# =============================================================================

test_that("group_by() keeps data on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)

  expect_data_on_gpu(grouped)
  expect_true(verify_no_r_copy(grouped))
})

test_that("group_by() shares pointer with original (no copy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)

  # Should have same pointer - grouping is metadata only
  expect_identical(gpu_df$ptr, grouped$ptr)
})

test_that("ungroup() shares pointer with original (no copy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl)
  ungrouped <- dplyr::ungroup(grouped)

  expect_identical(grouped$ptr, ungrouped$ptr)
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("group_by() with duplicate column names deduplicates", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl, cyl)

  expect_equal(dplyr::group_vars(grouped), "cyl")
})

test_that("group_by() works with single-row table", {
  skip_if_no_gpu()

  df <- data.frame(x = 1, y = 2)
  gpu_df <- tbl_gpu(df)
  grouped <- dplyr::group_by(gpu_df, x)

  expect_equal(dplyr::group_vars(grouped), "x")
})

test_that("group_by() works with many columns", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  grouped <- dplyr::group_by(gpu_df, cyl, vs, am, gear, carb)

  expect_equal(dplyr::group_vars(grouped), c("cyl", "vs", "am", "gear", "carb"))
})
