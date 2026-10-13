# Integration tests for cuplyr
#
# These tests verify:
# - Complete pipelines work correctly
# - Chained operations maintain data integrity
# - Real-world usage patterns
# - Correct behavior compared to dplyr on data frames

# =============================================================================
# Complete Pipeline Tests
# =============================================================================

test_that("complete filter-mutate-select-collect pipeline works", {
  skip_if_no_gpu()

  result <- mtcars |>
    tbl_gpu() |>
    dplyr::filter(mpg > 20) |>
    dplyr::mutate(kpl = mpg * 0.425) |>
    dplyr::select(mpg, kpl, cyl, hp) |>
    collect()

  # Verify filter
  expect_true(all(result$mpg > 20))

  # Verify mutate
  expect_equal(result$kpl, result$mpg * 0.425, tolerance = 1e-10)

  # Verify select
  expect_equal(names(result), c("mpg", "kpl", "cyl", "hp"))

  # Verify result is a data.frame (F31: the input's class, as dplyr keeps it)
  expect_identical(class(result), "data.frame")
})

test_that("pipeline results match dplyr on data.frame", {
  skip_if_no_gpu()

  # GPU result
  gpu_result <- mtcars |>
    tbl_gpu() |>
    dplyr::filter(mpg > 20) |>
    dplyr::mutate(power_weight = hp / wt) |>
    dplyr::select(mpg, hp, wt, power_weight) |>
    collect()

  # dplyr result
  dplyr_result <- mtcars |>
    dplyr::filter(mpg > 20) |>
    dplyr::mutate(power_weight = hp / wt) |>
    dplyr::select(mpg, hp, wt, power_weight)

  # Compare results
  expect_equal(nrow(gpu_result), nrow(dplyr_result))
  expect_equal(gpu_result$mpg, dplyr_result$mpg)
  expect_equal(gpu_result$hp, dplyr_result$hp)
  expect_equal(gpu_result$wt, dplyr_result$wt)
  expect_equal(gpu_result$power_weight, dplyr_result$power_weight, tolerance = 1e-10)
})

test_that("multiple filter operations produce correct results", {
  skip_if_no_gpu()

  # GPU result
  gpu_result <- mtcars |>
    tbl_gpu() |>
    dplyr::filter(mpg > 15) |>
    dplyr::filter(cyl == 4) |>
    dplyr::filter(hp < 100) |>
    collect()

  # dplyr result
  dplyr_result <- mtcars |>
    dplyr::filter(mpg > 15) |>
    dplyr::filter(cyl == 4) |>
    dplyr::filter(hp < 100)

  expect_equal(nrow(gpu_result), nrow(dplyr_result))
  expect_equal(gpu_result$mpg, dplyr_result$mpg)
  expect_equal(gpu_result$cyl, dplyr_result$cyl)
  expect_equal(gpu_result$hp, dplyr_result$hp)
})

test_that("multiple mutate operations produce correct results", {
  skip_if_no_gpu()

  # GPU result
  gpu_result <- mtcars |>
    tbl_gpu() |>
    dplyr::mutate(col1 = mpg + 10) |>
    dplyr::mutate(col2 = hp / wt) |>
    dplyr::mutate(col3 = col1 * 2) |>
    collect()

  # Verify calculations
  expect_equal(gpu_result$col1, mtcars$mpg + 10)
  expect_equal(gpu_result$col2, mtcars$hp / mtcars$wt, tolerance = 1e-10)
  expect_equal(gpu_result$col3, (mtcars$mpg + 10) * 2, tolerance = 1e-10)
})

# =============================================================================
# Real-World Usage Pattern Tests
# =============================================================================

test_that("data analysis workflow pattern works", {
  skip_if_no_gpu()

  # Simulating a data analysis workflow with fixed seed for reproducibility
  set.seed(42)
  df <- data.frame(
    customer_id = 1:100,
    revenue = runif(100, 100, 1000),
    cost = runif(100, 50, 500),
    region_code = sample(1:5, 100, replace = TRUE)
  )

  result <- df |>
    tbl_gpu() |>
    dplyr::filter(revenue > 500) |>
    dplyr::mutate(profit = revenue - cost) |>
    dplyr::mutate(margin = profit / revenue) |>
    dplyr::filter(margin > 0.3) |>
    dplyr::select(customer_id, revenue, profit, margin) |>
    collect()

  # Verify the workflow produced valid results
  expect_true(all(result$revenue > 500))
  expect_true(all(result$margin > 0.3))

  # Verify profit = revenue - cost (using collected values)
  # Since we don't have cost in the result, verify margin formula instead
  expect_equal(result$margin, result$profit / result$revenue, tolerance = 1e-10)

  # Verify result has expected columns
  expect_equal(names(result), c("customer_id", "revenue", "profit", "margin"))
})
test_that("cars dataset analysis works", {
  skip_if_no_gpu()

  result <- cars |>
    tbl_gpu() |>
    dplyr::filter(speed > 10) |>
    dplyr::mutate(stopping_ratio = dist / speed) |>
    dplyr::filter(stopping_ratio < 5) |>
    collect()

  # Verify
  expect_true(all(result$speed > 10))
  expect_true(all(result$stopping_ratio < 5))
  expect_equal(result$stopping_ratio, result$dist / result$speed, tolerance = 1e-10)
})

test_that("iris dataset analysis works", {
  skip_if_no_gpu()

  # Use only numeric columns
  iris_numeric <- iris[, 1:4]

  result <- iris_numeric |>
    tbl_gpu() |>
    dplyr::filter(Sepal.Length > 5) |>
    dplyr::mutate(Sepal.Area = Sepal.Length * Sepal.Width) |>
    dplyr::select(Sepal.Length, Sepal.Width, Sepal.Area) |>
    collect()

  expect_true(all(result$Sepal.Length > 5))
  expect_equal(result$Sepal.Area,
               result$Sepal.Length * result$Sepal.Width,
               tolerance = 1e-10)
})

# =============================================================================
# Intermediate Result Tests
# =============================================================================

test_that("intermediate results can be inspected", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Save intermediate results
  step1 <- dplyr::filter(gpu_df, mpg > 20)
  step2 <- dplyr::mutate(step1, kpl = mpg * 0.425)
  step3 <- dplyr::select(step2, mpg, kpl)

  # Verify each step is valid GPU data
  expect_data_on_gpu(step1)
  expect_data_on_gpu(step2)
  expect_data_on_gpu(step3)

  # Verify dimensions change appropriately
  expect_equal(dim(step1)[2], 11)
  expect_equal(dim(step2)[2], 12)  # +1 for kpl
  expect_equal(dim(step3)[2], 2)

  # Final collection should work
  result <- collect(step3)
  expect_identical(class(result), "data.frame")
})

test_that("branching pipelines work correctly", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Create two branches from the same source
  branch1 <- gpu_df |>
    dplyr::filter(cyl == 4) |>
    collect()

  branch2 <- gpu_df |>
    dplyr::filter(cyl == 8) |>
    collect()

  # Both should be correct
  expect_true(all(branch1$cyl == 4))
  expect_true(all(branch2$cyl == 8))

  # Should be mutually exclusive (no overlap in cyl values)
  expect_equal(length(intersect(unique(branch1$cyl), unique(branch2$cyl))), 0)

  # Combined row count should equal original cyl==4 + cyl==8 count
  expect_equal(
    nrow(branch1) + nrow(branch2),
    sum(mtcars$cyl == 4) + sum(mtcars$cyl == 8)
  )
})

# =============================================================================
# Column Order Tests
# =============================================================================

test_that("column order is preserved through operations", {
  skip_if_no_gpu()

  df <- data.frame(z = 1:5, y = 6:10, x = 11:15)
  gpu_df <- tbl_gpu(df)

  filtered <- dplyr::filter(gpu_df, z > 2)
  result <- collect(filtered)

  expect_equal(names(result), c("z", "y", "x"))
})

test_that("select reorders columns correctly", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Reorder to different order
  result <- gpu_df |>
    dplyr::select(hp, mpg, wt, cyl) |>
    collect()

  expect_equal(names(result), c("hp", "mpg", "wt", "cyl"))

  # Values should still be correct
  expect_equal(result$hp, mtcars$hp)
  expect_equal(result$mpg, mtcars$mpg)
})

test_that("mutate adds columns at correct position", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars[, 1:3])  # mpg, cyl, disp

  result <- gpu_df |>
    dplyr::mutate(new_col = mpg + 1) |>
    collect()

  # New column should be at the end
  expect_equal(names(result), c("mpg", "cyl", "disp", "new_col"))
})

test_that("mutate replaces column in correct position", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars[, 1:3])  # mpg, cyl, disp

  result <- gpu_df |>
    dplyr::mutate(cyl = cyl * 2) |>
    collect()

  # cyl should stay in position 2
  expect_equal(names(result), c("mpg", "cyl", "disp"))
  expect_equal(ncol(result), 3)
  expect_equal(result$cyl, mtcars$cyl[1:32] * 2)
})

# =============================================================================
# Large Data Integration Tests
# =============================================================================

test_that("complete pipeline works with large data", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(500 * 1024 * 1024)

  df <- create_large_test_data(nrow = 100000, ncol = 10)

  result <- df |>
    tbl_gpu() |>
    dplyr::filter(col1 > 0.5) |>
    dplyr::filter(col2 < 0.8) |>
    dplyr::mutate(sum = col1 + col2) |>
    dplyr::mutate(diff = col1 - col2) |>
    dplyr::select(col1, col2, sum, diff) |>
    collect()

  # Verify filter conditions
  expect_true(all(result$col1 > 0.5))
  expect_true(all(result$col2 < 0.8))

  # Verify calculations
  expect_equal(result$sum, result$col1 + result$col2, tolerance = 1e-10)
  expect_equal(result$diff, result$col1 - result$col2, tolerance = 1e-10)
})

# =============================================================================
# Edge Case Integration Tests
# =============================================================================

test_that("pipeline with filtering to zero rows works", {
  skip_if_no_gpu()

  result <- mtcars |>
    tbl_gpu() |>
    dplyr::filter(mpg > 1000) |>
    dplyr::mutate(new_col = mpg + 1) |>
    dplyr::select(mpg, new_col) |>
    collect()

  expect_equal(nrow(result), 0)
  expect_equal(names(result), c("mpg", "new_col"))
})

test_that("pipeline preserves NA values correctly", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, NA, 3, NA, 5),
    y = c(10, 20, NA, 40, 50)
  )

  # Filter doesn't include NAs in comparison
  result <- df |>
    tbl_gpu() |>
    dplyr::filter(x > 0) |>
    dplyr::mutate(sum = x + y) |>
    collect()

  # NAs in x column should be filtered out
  expect_false(any(is.na(result$x)))

  # sum should have NA where y is NA (using is.na() not == NA)
  na_positions <- which(is.na(result$y))
  if (length(na_positions) > 0) {
    expect_true(all(is.na(result$sum[na_positions])))
  }
})

test_that("single-row pipeline works", {
  skip_if_no_gpu()

  df <- data.frame(x = 42, y = 100)

  result <- df |>
    tbl_gpu() |>
    dplyr::mutate(z = x + y) |>
    dplyr::select(x, z) |>
    collect()

  expect_equal(nrow(result), 1)
  expect_equal(result$x, 42)
  expect_equal(result$z, 142)
})

# =============================================================================
# Comparison Operators Integration
# =============================================================================

test_that("all comparison operators work in pipeline", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10)

  # Test each operator
  gt <- df |> tbl_gpu() |> dplyr::filter(x > 5) |> collect()
  gte <- df |> tbl_gpu() |> dplyr::filter(x >= 5) |> collect()
  lt <- df |> tbl_gpu() |> dplyr::filter(x < 5) |> collect()
  lte <- df |> tbl_gpu() |> dplyr::filter(x <= 5) |> collect()
  eq <- df |> tbl_gpu() |> dplyr::filter(x == 5) |> collect()
  neq <- df |> tbl_gpu() |> dplyr::filter(x != 5) |> collect()

  expect_equal(gt$x, 6:10)
  expect_equal(gte$x, 5:10)
  expect_equal(lt$x, 1:4)
  expect_equal(lte$x, 1:5)
  expect_equal(eq$x, 5)
  expect_equal(neq$x, c(1:4, 6:10))
})

test_that("all arithmetic operators work in pipeline", {
  skip_if_no_gpu()

  df <- data.frame(x = c(10, 20, 30), y = c(2, 4, 5))

  result <- df |>
    tbl_gpu() |>
    dplyr::mutate(add = x + 5) |>
    dplyr::mutate(sub = x - 3) |>
    dplyr::mutate(mul = x * 2) |>
    dplyr::mutate(div = x / 2) |>
    dplyr::mutate(pow = y ^ 2) |>
    collect()

  expect_equal(result$add, c(15, 25, 35))
  expect_equal(result$sub, c(7, 17, 27))
  expect_equal(result$mul, c(20, 40, 60))
  expect_equal(result$div, c(5, 10, 15))
  expect_equal(result$pow, c(4, 16, 25))
})

# =============================================================================
# Mixed Type Pipeline Tests
# =============================================================================

test_that("pipeline works with mixed column types", {
  skip_if_no_gpu()

  df <- data.frame(
    int_col = 1:10,
    dbl_col = seq(0.1, 1.0, by = 0.1),
    chr_col = letters[1:10],
    stringsAsFactors = FALSE
  )

  result <- df |>
    tbl_gpu() |>
    dplyr::filter(int_col > 5) |>
    dplyr::mutate(product = int_col * dbl_col) |>
    dplyr::select(int_col, dbl_col, product) |>
    collect()

  expect_true(all(result$int_col > 5))
  expect_equal(result$product, result$int_col * result$dbl_col, tolerance = 1e-10)
})

# =============================================================================
# Reproducibility Tests
# =============================================================================

test_that("same pipeline produces same results", {
  skip_if_no_gpu()

  run_pipeline <- function() {
    mtcars |>
      tbl_gpu() |>
      dplyr::filter(mpg > 20) |>
      dplyr::mutate(kpl = mpg * 0.425) |>
      dplyr::select(mpg, kpl, cyl) |>
      collect()
  }

  result1 <- run_pipeline()
  result2 <- run_pipeline()
  result3 <- run_pipeline()

  expect_equal(result1, result2)
  expect_equal(result2, result3)
})

# =============================================================================
# GPU Residency Through Pipeline
# =============================================================================

test_that("data stays on GPU throughout entire pipeline", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # Build pipeline step by step
  step1 <- dplyr::filter(gpu_df, mpg > 15)
  expect_data_on_gpu(step1)

  step2 <- dplyr::mutate(step1, kpl = mpg * 0.425)
  expect_data_on_gpu(step2)

  step3 <- dplyr::select(step2, mpg, kpl, cyl)
  expect_data_on_gpu(step3)

  # Only collect at the end
  result <- collect(step3)
  expect_identical(class(result), "data.frame")

  # Verify intermediate objects are lightweight
  expect_lightweight_r_object(step1)
  expect_lightweight_r_object(step2)
  expect_lightweight_r_object(step3)
})
