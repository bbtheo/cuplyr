# Tests for arrange.tbl_gpu
#
# These tests verify:
# - Basic ascending and descending sorts
# - Multi-column sorting
# - NA handling
# - Stability (ties preserve original order)
# - Grouped arrangement with .by_group
# - Edge cases (empty tables, single row)

# =============================================================================
# Basic Sorting Tests
# =============================================================================

test_that("arrange() sorts ascending by default", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, 2))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, c(1, 2, 3))
})

test_that("arrange() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, 2), y = c("c", "a", "b"))
  expected <- df |>
    dplyr::arrange(dplyr::desc(x), y)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::arrange(dplyr::desc(x), y) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})

test_that("arrange() sorts descending with desc()", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 3, 2))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(dplyr::desc(x)) |>
    collect()

  expect_equal(result$x, c(3, 2, 1))
})

test_that("arrange() sorts descending with unary minus", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 3, 2))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(-x) |>
    collect()

  expect_equal(result$x, c(3, 2, 1))
})

test_that("arrange() with no columns returns data unchanged", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, 2), y = c("c", "a", "b"))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange() |>
    collect()

  expect_equal(result$x, df$x)
  expect_equal(result$y, df$y)
})

# =============================================================================
# Multi-Column Sorting Tests
# =============================================================================

test_that("arrange() sorts by multiple columns", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, 1, 2, 2),
    y = c(2, 1, 2, 1)
  )
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x, y) |>
    collect()

  expect_equal(result$x, c(1, 1, 2, 2))
  expect_equal(result$y, c(1, 2, 1, 2))
})

test_that("arrange() handles mixed asc/desc in multi-column sort", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, 1, 2, 2),
    y = c(1, 2, 1, 2)
  )
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x, dplyr::desc(y)) |>
    collect()

  expect_equal(result$x, c(1, 1, 2, 2))
  expect_equal(result$y, c(2, 1, 2, 1))
})

# =============================================================================
# NA Handling Tests
# =============================================================================

test_that("arrange() places NA last for ascending sort", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, NA, 1, 3))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x[1:3], c(1, 2, 3))
  expect_true(is.na(result$x[4]))
})

test_that("arrange() places NA first for descending sort", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, NA, 1, 3))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(dplyr::desc(x)) |>
    collect()

  expect_true(is.na(result$x[1]))
  expect_equal(result$x[2:4], c(3, 2, 1))
})

test_that("arrange() handles all-NA column", {
  skip_if_no_gpu()

  df <- data.frame(x = c(NA_real_, NA_real_, NA_real_))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_true(all(is.na(result$x)))
  expect_equal(length(result$x), 3)
})

# =============================================================================
# Stability Tests
# =============================================================================

test_that("arrange() is stable (ties preserve original order)", {
  skip_if_no_gpu()

  # Create data where x has ties, and we track original order via y
  df <- data.frame(
    x = c(1, 1, 1, 2, 2),
    order_marker = c(1, 2, 3, 4, 5)
  )
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  # Ties within x=1 should preserve original order (1, 2, 3)
  expect_equal(result$order_marker[1:3], c(1, 2, 3))
  # Ties within x=2 should preserve original order (4, 5)
  expect_equal(result$order_marker[4:5], c(4, 5))
})

# =============================================================================
# Column Type Tests
# =============================================================================

test_that("arrange() works with integer columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3L, 1L, 2L))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, c(1L, 2L, 3L))
})

test_that("arrange() works with character columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c("banana", "apple", "cherry"), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, c("apple", "banana", "cherry"))
})

test_that("arrange() works with logical columns", {
  skip_if_no_gpu()

  df <- data.frame(x = c(TRUE, FALSE, TRUE, FALSE))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  # FALSE (0) should come before TRUE (1)
  expect_equal(result$x, c(FALSE, FALSE, TRUE, TRUE))
})

test_that("arrange() works with Date columns", {
  skip_if_no_gpu()

  df <- data.frame(x = as.Date(c("2024-03-01", "2024-01-01", "2024-02-01")))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, as.Date(c("2024-01-01", "2024-02-01", "2024-03-01")))
})

# =============================================================================
# Grouped Arrangement Tests
# =============================================================================

test_that("arrange() with .by_group = FALSE ignores groups", {
  skip_if_no_gpu()

  df <- data.frame(
    g = c("b", "a", "b", "a"),
    x = c(2, 1, 1, 2)
  )
  gpu_df <- tbl_gpu(df) |>
    dplyr::group_by(g)

  result <- gpu_df |>
    dplyr::arrange(x, .by_group = FALSE) |>
    collect()

  # Should sort by x globally, ignoring groups
  expect_equal(result$x, c(1, 1, 2, 2))
})

test_that("arrange() with .by_group = TRUE sorts within groups", {
  skip_if_no_gpu()

  df <- data.frame(
    g = c("b", "a", "b", "a"),
    x = c(2, 1, 1, 2)
  )
  gpu_df <- tbl_gpu(df) |>
    dplyr::group_by(g)

  result <- gpu_df |>
    dplyr::arrange(x, .by_group = TRUE) |>
    collect()

  # Should be sorted by group first (a, a, b, b), then by x within groups
  expect_equal(result$g, c("a", "a", "b", "b"))
  # Within group "a": 1, 2; within group "b": 1, 2
  expect_equal(result$x, c(1, 2, 1, 2))
})

test_that("arrange() with .by_group deduplicates overlapping columns", {
  skip_if_no_gpu()

  df <- data.frame(
    g = c("b", "a", "b", "a"),
    x = c(2, 1, 1, 2)
  )
  gpu_df <- tbl_gpu(df) |>
    dplyr::group_by(g)

  # User specifies g again - should not sort by g twice
  result <- gpu_df |>
    dplyr::arrange(g, x, .by_group = TRUE) |>
    collect()

  expect_equal(result$g, c("a", "a", "b", "b"))
  expect_equal(result$x, c(1, 2, 1, 2))
})

test_that("arrange() preserves grouping metadata", {
  skip_if_no_gpu()

  df <- data.frame(g = c("b", "a"), x = c(2, 1))
  gpu_df <- tbl_gpu(df) |>
    dplyr::group_by(g)

  result <- dplyr::arrange(gpu_df, x)

  expect_equal(result$groups, "g")
})

# =============================================================================
# Edge Case Tests
# =============================================================================

test_that("arrange() handles empty table", {
  skip_if_no_gpu()

  df <- data.frame(x = numeric(0), y = character(0))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(nrow(result), 0)
  expect_equal(names(result), c("x", "y"))
})

test_that("arrange() handles single row table", {
  skip_if_no_gpu()

  df <- data.frame(x = 42, y = "only")
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, 42)
  expect_equal(result$y, "only")
})

test_that("arrange() handles already sorted data", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3, 4, 5))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, c(1, 2, 3, 4, 5))
})

test_that("arrange() handles reverse sorted data", {
  skip_if_no_gpu()

  df <- data.frame(x = c(5, 4, 3, 2, 1))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x) |>
    collect()

  expect_equal(result$x, c(1, 2, 3, 4, 5))
})

# =============================================================================
# Error Handling Tests
# =============================================================================

test_that("arrange() errors on non-existent column", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3))
  gpu_df <- tbl_gpu(df)

  expect_error(
    dplyr::arrange(gpu_df, nonexistent),
    "not found"
  )
})

test_that("arrange() errors on invalid desc() argument", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3))
  gpu_df <- tbl_gpu(df)

  expect_error(
    dplyr::arrange(gpu_df, dplyr::desc(x + 1)),
    "column name"
  )
})

test_that("arrange() errors on complex expressions", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3))
  gpu_df <- tbl_gpu(df)

  expect_error(
    dplyr::arrange(gpu_df, x + 1),
    "column names"
  )
})

# =============================================================================
# Schema Preservation Tests
# =============================================================================

test_that("arrange() preserves schema", {
  skip_if_no_gpu()

  df <- data.frame(
    int_col = c(3L, 1L, 2L),
    dbl_col = c(3.0, 1.0, 2.0),
    chr_col = c("c", "a", "b"),
    stringsAsFactors = FALSE
  )
  gpu_df <- tbl_gpu(df)

  result <- dplyr::arrange(gpu_df, int_col)

  expect_equal(result$schema$names, gpu_df$schema$names)
  expect_equal(result$schema$types, gpu_df$schema$types)
})

# =============================================================================
# Large Data Tests
# =============================================================================

test_that("arrange() handles larger datasets", {
  skip_if_no_gpu()

  n <- 10000
  df <- data.frame(
    x = sample(1:100, n, replace = TRUE),
    y = runif(n)
  )
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::arrange(x, y) |>
    collect()

  # Verify sorted
  expect_true(all(diff(result$x) >= 0))

  # Verify within-group sorting for y
  for (val in unique(result$x)) {
    subset_y <- result$y[result$x == val]
    expect_true(all(diff(subset_y) >= 0))
  }
})
