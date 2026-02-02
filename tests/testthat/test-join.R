# Tests for join operations on tbl_gpu
#
# These tests verify:
# - left_join(), right_join(), inner_join(), full_join()
# - Single and multiple key joins
# - Different key name joins
# - Natural joins (by = NULL)
# - Suffix handling for column name conflicts
# - Edge cases: empty tables, type mismatches, many-to-many

# =============================================================================
# Basic Join Operations
# =============================================================================

test_that("left_join() with single key works", {
  skip_if_no_gpu()

  left_df <- data.frame(
    key = c(1, 2, 3, 4, 5),
    val_left = c("a", "b", "c", "d", "e"),
    stringsAsFactors = FALSE
  )
  right_df <- data.frame(
    key = c(2, 3, 4, 6, 7),
    val_right = c("x", "y", "z", "w", "v"),
    stringsAsFactors = FALSE
  )

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = "key") |>
    collect()

  expected <- dplyr::left_join(left_df, right_df, by = "key")

  expect_equal(nrow(result), nrow(expected))
  expect_equal(sort(names(result)), sort(names(expected)))
  # Check matching rows
  expect_equal(sum(!is.na(result$val_right)), 3)  # keys 2, 3, 4 match
})

test_that("inner_join() with single key works", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:5, x = letters[1:5], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 3:7, y = LETTERS[1:5], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::inner_join(gpu_right, by = "key") |>
    collect()

  expected <- dplyr::inner_join(left_df, right_df, by = "key")

  expect_equal(nrow(result), nrow(expected))  # Should be 3 rows (keys 3, 4, 5)
  expect_equal(nrow(result), 3)
})

test_that("right_join() works", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::right_join(gpu_right, by = "key") |>
    collect()

  expected <- dplyr::right_join(left_df, right_df, by = "key")

  expect_equal(nrow(result), nrow(expected))  # Should be 3 rows (keys 2, 3, 4)
  expect_true("x" %in% names(result))
  expect_true("y" %in% names(result))
})

test_that("full_join() works", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::full_join(gpu_right, by = "key") |>
    collect()

  expected <- dplyr::full_join(left_df, right_df, by = "key")

  expect_equal(nrow(result), nrow(expected))  # Should be 4 rows (keys 1, 2, 3, 4)
  expect_equal(nrow(result), 4)
})

# =============================================================================
# Multiple Key Joins
# =============================================================================

test_that("left_join() with multiple keys works", {
  skip_if_no_gpu()

  left_df <- data.frame(
    k1 = c(1, 1, 2, 2),
    k2 = c("a", "b", "a", "b"),
    val = 1:4,
    stringsAsFactors = FALSE
  )
  right_df <- data.frame(
    k1 = c(1, 2),
    k2 = c("a", "b"),
    other = c(10, 20),
    stringsAsFactors = FALSE
  )

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = c("k1", "k2")) |>
    collect()

  expected <- dplyr::left_join(left_df, right_df, by = c("k1", "k2"))

  expect_equal(nrow(result), nrow(expected))
  # Two rows should match (1,a) and (2,b)
  expect_equal(sum(!is.na(result$other)), 2)
})

# =============================================================================
# Different Key Names
# =============================================================================

test_that("join with different key names works", {
  skip_if_no_gpu()

  left_df <- data.frame(id = 1:3, x = c("a", "b", "c"), stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = c("x", "y", "z"), stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = c("id" = "key")) |>
    collect()

  # When key names differ, both columns should be kept
  expect_true("id" %in% names(result))
  expect_true("key" %in% names(result))
  expect_equal(nrow(result), 3)
})

# =============================================================================
# Natural Join
# =============================================================================

test_that("natural join (by = NULL) works", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  expect_message(
    result <- dplyr::left_join(gpu_left, gpu_right),
    "Joining with"
  )

  result <- collect(result)
  expect_equal(nrow(result), 3)
  expect_true("key" %in% names(result))
})

# =============================================================================
# Suffix Handling
# =============================================================================

test_that("join with suffix handles column name conflicts", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, val = c(1, 2, 3))
  right_df <- data.frame(key = 2:4, val = c(10, 20, 30))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = "key", suffix = c("_L", "_R")) |>
    collect()

  expect_true("val_L" %in% names(result))
  expect_true("val_R" %in% names(result))
})

test_that("join with default suffix works", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, val = c(1, 2, 3))
  right_df <- data.frame(key = 2:4, val = c(10, 20, 30))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = "key") |>
    collect()

  expect_true("val.x" %in% names(result))
  expect_true("val.y" %in% names(result))
})

# =============================================================================
# keep Parameter
# =============================================================================

test_that("join with keep = TRUE preserves both key columns", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = "key", keep = TRUE) |>
    collect()

  # Both key columns should be present (with suffixes since names match)
  expect_true("key.x" %in% names(result) || "key" %in% names(result))
  expect_true("key.y" %in% names(result))
})

test_that("join with different key names keeps both columns by default", {
  skip_if_no_gpu()

  left_df <- data.frame(id = 1:3, x = c("a", "b", "c"), stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = c("x", "y", "z"), stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = c("id" = "key")) |>
    collect()

  # When key names differ, right key column should be kept
  expect_true("id" %in% names(result))
  expect_true("key" %in% names(result))
})

# =============================================================================
# copy Parameter
# =============================================================================

test_that("join with copy = TRUE converts data.frame to tbl_gpu", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)

  result <- gpu_left |>
    dplyr::left_join(right_df, by = "key", copy = TRUE) |>
    collect()

  expect_equal(nrow(result), 3)
  expect_true("y" %in% names(result))
})

test_that("join without copy = TRUE errors for data.frame", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)

  expect_error(
    dplyr::left_join(gpu_left, right_df, by = "key"),
    "tbl_gpu"
  )
})

# =============================================================================
# Empty Tables
# =============================================================================

test_that("join with empty left table returns empty result", {
  skip_if_no_gpu()

  left_df <- data.frame(key = integer(0), x = character(0), stringsAsFactors = FALSE)
  right_df <- data.frame(key = 1:3, y = letters[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = "key") |>
    collect()

  expect_equal(nrow(result), 0)
  expect_true("key" %in% names(result))
  expect_true("y" %in% names(result))
})

test_that("join with empty right table returns NAs for right columns", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = integer(0), y = character(0), stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::left_join(gpu_right, by = "key") |>
    collect()

  expect_equal(nrow(result), 3)
  expect_true(all(is.na(result$y)))
})

test_that("inner_join with empty table returns empty result", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = integer(0), y = character(0), stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::inner_join(gpu_right, by = "key") |>
    collect()

  expect_equal(nrow(result), 0)
})

# =============================================================================
# Error Handling
# =============================================================================

test_that("join errors with non-existent column", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(key = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  expect_error(
    dplyr::left_join(gpu_left, gpu_right, by = "nonexistent"),
    "missing"
  )
})

test_that("join errors on incompatible key types", {
  skip_if_no_gpu()

  left_df <- data.frame(key = c("a", "b", "c"), x = 1:3, stringsAsFactors = FALSE)
  right_df <- data.frame(key = 1:3, y = letters[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  expect_error(
    dplyr::left_join(gpu_left, gpu_right, by = "key"),
    "type mismatch"
  )
})

test_that("natural join errors when no common columns", {
  skip_if_no_gpu()

  left_df <- data.frame(a = 1:3, x = letters[1:3], stringsAsFactors = FALSE)
  right_df <- data.frame(b = 2:4, y = LETTERS[1:3], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  expect_error(
    dplyr::left_join(gpu_left, gpu_right),
    "common"
  )
})

# =============================================================================
# Many-to-Many Joins
# =============================================================================

test_that("many-to-many join produces correct row count", {
  skip_if_no_gpu()

  # Each key appears twice in both tables -> 4 matches per key
  left_df <- data.frame(key = c(1, 1, 2, 2), x = letters[1:4], stringsAsFactors = FALSE)
  right_df <- data.frame(key = c(1, 1, 2, 2), y = LETTERS[1:4], stringsAsFactors = FALSE)

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::inner_join(gpu_right, by = "key") |>
    collect()

  expected <- dplyr::inner_join(left_df, right_df, by = "key")
  expect_equal(nrow(result), nrow(expected))  # Should be 8 rows (2*2 + 2*2)
})

# =============================================================================
# Numeric Precision
# =============================================================================

test_that("join preserves numeric precision", {
  skip_if_no_gpu()

  set.seed(42)
  left_df <- data.frame(key = 1:100, val = runif(100))
  right_df <- data.frame(key = 50:150, other = runif(101))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- gpu_left |>
    dplyr::inner_join(gpu_right, by = "key") |>
    collect()

  expected <- dplyr::inner_join(left_df, right_df, by = "key")

  # Sort both by key to compare values (join order is not guaranteed)
  result <- result[order(result$key), ]
  expected <- expected[order(expected$key), ]

  expect_equal(result$val, expected$val, tolerance = 1e-10)
})

# =============================================================================
# GPU Data Residency
# =============================================================================

test_that("join result stays on GPU", {
  skip_if_no_gpu()

  left_df <- data.frame(key = 1:100, x = runif(100))
  right_df <- data.frame(key = 50:150, y = runif(101))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- dplyr::left_join(gpu_left, gpu_right, by = "key")

  expect_data_on_gpu(result)
  expect_valid_tbl_gpu(result)
})

# =============================================================================
# Grouping Behavior
# =============================================================================

test_that("joins drop groups", {
  skip_if_no_gpu()

  left_df <- data.frame(
    grp = c("a", "a", "b", "b"),
    key = 1:4,
    x = runif(4)
  )
  right_df <- data.frame(key = 2:5, y = runif(4))

  gpu_left <- tbl_gpu(left_df) |>
    dplyr::group_by(grp)

  gpu_right <- tbl_gpu(right_df)

  result <- dplyr::left_join(gpu_left, gpu_right, by = "key")

  # Join should drop groups
  expect_equal(length(result$groups), 0)
})
