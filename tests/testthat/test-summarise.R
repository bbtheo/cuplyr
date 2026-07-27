# Tests for summarise.tbl_gpu()
#
# These tests verify:
# - Basic aggregation functions (sum, mean, min, max, n, sd, var)
# - Grouped summarise
# - Ungrouped summarise (over all rows)
# - Multiple aggregations
# - Correct result values

# =============================================================================
# Basic Ungrouped Aggregations
# =============================================================================

test_that("summarise() with sum() works ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3, 4, 5))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::summarise(total = sum(x)) |>
    collect()

  expect_equal(nrow(result), 1)
  expect_equal(result$total, 15)
})

test_that("eager summarise result has no pending lazy ops", {
  skip_if_no_gpu()

  out <- tbl_gpu(data.frame(x = 1:5)) |>
    dplyr::summarise(total = sum(x))

  expect_null(out$lazy_ops)
  expect_false(has_pending_ops(out))
})

test_that("summarise() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(total_hp = sum(hp), avg_mpg = mean(mpg), .groups = "drop") |>
    dplyr::arrange(cyl)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(total_hp = sum(hp), avg_mpg = mean(mpg)) |>
      dplyr::arrange(cyl) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})

test_that("summarise() with mean() works ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, 4, 6, 8, 10))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::summarise(avg = mean(x)) |>
    collect()

  expect_equal(nrow(result), 1)
  expect_equal(result$avg, 6, tolerance = 1e-10)
})

test_that("summarise() with min() works ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(5, 2, 8, 1, 9))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::summarise(minimum = min(x)) |>
    collect()

  expect_equal(result$minimum, 1)
})

test_that("summarise() with max() works ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(5, 2, 8, 1, 9))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::summarise(maximum = max(x)) |>
    collect()

  expect_equal(result$maximum, 9)
})

test_that("summarise() with n() works ungrouped", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::summarise(count = n()) |>
    collect()

  expect_equal(result$count, 32)
})

test_that("summarise(n = n()) declares a schema type matching the actual GPU column type (S7)", {
  skip_if_no_gpu()

  # Ground truth (see R/summarise.R::make_aggregation() / src/ops_groupby.cpp
  # gpu_summarise()): both the grouped (cudf groupby count aggregation, which
  # yields cudf::size_type = INT32) and ungrouped (explicit scalar column,
  # built as INT32 in the C++) paths produce an INT32 "n" column, in both
  # eager and lazy exec modes. The schema declaration in make_aggregation()
  # ("n" -> "INT32") must match this for both.
  df <- data.frame(g = rep(1:2, 5), x = 1:10)

  for (mode in c("eager", "lazy")) {
    tbl <- tbl_gpu(df, lazy = identical(mode, "lazy"))

    # --- ungrouped ---
    res_ungrouped <- tbl |> dplyr::summarise(count = n())
    if (identical(mode, "lazy")) res_ungrouped <- compute(res_ungrouped)
    count_idx <- match("count", res_ungrouped$schema$names)
    expect_equal(unname(res_ungrouped$schema$types[count_idx]), "INT32",
                 info = paste(mode, "ungrouped: schema$types"))
    expect_equal(unname(gpu_col_types(res_ungrouped$ptr)[count_idx]), "int",
                 info = paste(mode, "ungrouped: actual GPU column type"))
    expect_type(collect(res_ungrouped)$count, "integer")

    # --- grouped ---
    res_grouped <- tbl |> dplyr::group_by(g) |> dplyr::summarise(count = n())
    if (identical(mode, "lazy")) res_grouped <- compute(res_grouped)
    count_idx_g <- match("count", res_grouped$schema$names)
    expect_equal(unname(res_grouped$schema$types[count_idx_g]), "INT32",
                 info = paste(mode, "grouped: schema$types"))
    expect_equal(unname(gpu_col_types(res_grouped$ptr)[count_idx_g]), "int",
                 info = paste(mode, "grouped: actual GPU column type"))
    expect_type(collect(res_grouped)$count, "integer")
  }
})

test_that("summarise(n = n()) matches dplyr's integer type exactly, ungrouped and grouped (S7)", {
  skip_if_no_gpu()

  df <- data.frame(g = rep(1:2, 5), x = 1:10)

  pipeline_ungrouped <- function(d) dplyr::summarise(d, count = dplyr::n())
  pipeline_grouped <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(count = dplyr::n(), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline_ungrouped, ignore_col_types = FALSE)
  expect_same_as_dplyr(df, pipeline_grouped, arrange_by = "g", ignore_col_types = FALSE)
  expect_same_as_dplyr_lazy(df, pipeline_ungrouped, ignore_col_types = FALSE)
  expect_same_as_dplyr_lazy(df, pipeline_grouped, arrange_by = "g", ignore_col_types = FALSE)
})

test_that("summarise() with multiple aggregations works ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3, 4, 5))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::summarise(
      total = sum(x),
      avg = mean(x),
      min_val = min(x),
      max_val = max(x)
    ) |>
    collect()

  expect_equal(nrow(result), 1)
  expect_equal(result$total, 15)
  expect_equal(result$avg, 3, tolerance = 1e-10)
  expect_equal(result$min_val, 1)
  expect_equal(result$max_val, 5)
})

# =============================================================================
# Grouped Aggregations
# =============================================================================

test_that("summarise() with single group works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_mpg = mean(mpg)) |>
    collect()

  # Should have one row per unique cyl value
  expect_equal(nrow(result), 3)
  expect_true("cyl" %in% names(result))
  expect_true("avg_mpg" %in% names(result))

  # Check values against R computation
  expected <- aggregate(mpg ~ cyl, data = mtcars, mean)
  for (cyl_val in c(4, 6, 8)) {
    gpu_val <- result$avg_mpg[result$cyl == cyl_val]
    r_val <- expected$mpg[expected$cyl == cyl_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() with multiple groups works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl, gear) |>
    dplyr::summarise(avg_mpg = mean(mpg)) |>
    collect()

  # Should have one row per unique cyl-gear combination
  expected_rows <- nrow(unique(mtcars[, c("cyl", "gear")]))
  expect_equal(nrow(result), expected_rows)
  expect_true(all(c("cyl", "gear", "avg_mpg") %in% names(result)))
})

test_that("summarise() with n() grouped works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(count = n()) |>
    collect()

  # Check counts match R
  r_counts <- table(mtcars$cyl)
  for (cyl_val in c(4, 6, 8)) {
    gpu_count <- result$count[result$cyl == cyl_val]
    r_count <- as.numeric(r_counts[as.character(cyl_val)])
    expect_equal(gpu_count, r_count)
  }
})

test_that("summarise() with sum() grouped works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(total_hp = sum(hp)) |>
    collect()

  # Check sums match R
  expected <- aggregate(hp ~ cyl, data = mtcars, sum)
  for (cyl_val in c(4, 6, 8)) {
    gpu_val <- result$total_hp[result$cyl == cyl_val]
    r_val <- expected$hp[expected$cyl == cyl_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() with min() and max() grouped works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(
      min_mpg = min(mpg),
      max_mpg = max(mpg)
    ) |>
    collect()

  # Check values match R
  for (cyl_val in c(4, 6, 8)) {
    subset_data <- mtcars[mtcars$cyl == cyl_val, ]
    gpu_min <- result$min_mpg[result$cyl == cyl_val]
    gpu_max <- result$max_mpg[result$cyl == cyl_val]
    expect_equal(gpu_min, min(subset_data$mpg), tolerance = 1e-10)
    expect_equal(gpu_max, max(subset_data$mpg), tolerance = 1e-10)
  }
})

test_that("summarise() with multiple aggregations grouped works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(
      avg_mpg = mean(mpg),
      total_hp = sum(hp),
      count = n()
    ) |>
    collect()

  expect_equal(nrow(result), 3)
  expect_true(all(c("cyl", "avg_mpg", "total_hp", "count") %in% names(result)))
})

# =============================================================================
# Variance and Standard Deviation
# =============================================================================

test_that("summarise() with sd() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::summarise(sd_mpg = sd(mpg)) |>
    collect()

  # Compare with R (note: R uses n-1 denominator)
  expected_sd <- sd(mtcars$mpg)
  expect_equal(result$sd_mpg, expected_sd, tolerance = 0.1)
})

test_that("summarise() with var() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::summarise(var_mpg = var(mpg)) |>
    collect()

  expected_var <- var(mtcars$mpg)
  expect_equal(result$var_mpg, expected_var, tolerance = 0.1)
})

# =============================================================================
# S6: lazy sd()/var() and namespaced aggregation calls (eager/lazy unification)
# =============================================================================

test_that("summarise() with sd() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  pipeline <- function(d) d |> dplyr::summarise(sd_mpg = sd(mpg))

  expect_same_as_dplyr(mtcars, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, ignore_col_types = TRUE)
})

test_that("summarise() with var() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  pipeline <- function(d) d |> dplyr::summarise(var_mpg = var(mpg))

  expect_same_as_dplyr(mtcars, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, ignore_col_types = TRUE)
})

test_that("summarise() with grouped sd()/var() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(sd_mpg = sd(mpg), var_hp = var(hp), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("lazy summarise() with sum() of a comparison expression matches dplyr (temp-column path)", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    d |>
      dplyr::filter(mpg > 0) |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(high_carb = sum(carb > 3), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("lazy summarise() temp-column path materializes pending ops and notifies fallback", {
  skip_if_no_gpu()

  withr::local_options(cuplyr.fallback = "warn")

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE) |>
    dplyr::filter(mpg > 0)

  expect_warning(
    result <- gpu_df |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(high_carb = sum(carb > 3)) |>
      collect(),
    "fell back to CPU evaluation"
  )

  expect_equal(nrow(result), 3)
})

test_that("summarise() accepts dplyr::n() (namespaced call) in eager and lazy modes", {
  skip_if_no_gpu()

  pipeline_ungrouped <- function(d) d |> dplyr::summarise(count = dplyr::n())
  pipeline_grouped <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(count = dplyr::n(), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline_ungrouped, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline_ungrouped, ignore_col_types = TRUE)
  expect_same_as_dplyr(mtcars, pipeline_grouped, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline_grouped, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("summarise() accepts stats::sd() (namespaced call) in eager and lazy modes", {
  skip_if_no_gpu()

  pipeline_ungrouped <- function(d) d |> dplyr::summarise(sd_mpg = stats::sd(mpg))
  pipeline_grouped <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(sd_mpg = stats::sd(mpg), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline_ungrouped, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline_ungrouped, ignore_col_types = TRUE)
  expect_same_as_dplyr(mtcars, pipeline_grouped, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline_grouped, arrange_by = "cyl", ignore_col_types = TRUE)
})

# =============================================================================
# Widened aggregation sub-expressions (Phase 1 T7 -- create_temp_column() is
# now a single IR-based implementation, so any expression shape ir_parse_quo()
# understands is a valid aggregation argument, not just one bare operator)
# =============================================================================

test_that("summarise() accepts a compound logical sub-expression (sum(x > 3 & y < 2)) ungrouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    dplyr::summarise(d, high_carb_light = sum(carb > 3 & wt < 4))
  }

  expect_same_as_dplyr(mtcars, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, ignore_col_types = TRUE)
})

test_that("summarise() accepts a compound logical sub-expression (sum(x > 3 & y < 2)) grouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(high_carb_light = sum(carb > 3 & wt < 4), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("summarise() accepts a math-function sub-expression (mean(sqrt(x))) ungrouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    dplyr::summarise(d, mean_sqrt_hp = mean(sqrt(hp)))
  }

  expect_same_as_dplyr(mtcars, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, ignore_col_types = TRUE)
})

test_that("summarise() accepts a math-function sub-expression (mean(sqrt(x))) grouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(mean_sqrt_hp = mean(sqrt(hp)), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("summarise() accepts a modulo comparison sub-expression (sum(x %% 2 == 0)) ungrouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    dplyr::summarise(d, even_carb = sum(carb %% 2 == 0))
  }

  expect_same_as_dplyr(mtcars, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, ignore_col_types = TRUE)
})

test_that("summarise() accepts a modulo comparison sub-expression (sum(x %% 2 == 0)) grouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(even_carb = sum(carb %% 2 == 0), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("summarise() accepts a column-vs-column arithmetic sub-expression (mean(x / y)) ungrouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    dplyr::summarise(d, power_weight = mean(hp / wt))
  }

  expect_same_as_dplyr(mtcars, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, ignore_col_types = TRUE)
})

test_that("summarise() accepts a column-vs-column arithmetic sub-expression (mean(x / y)) grouped", {
  skip_if_no_gpu()

  pipeline <- function(d) {
    d |>
      dplyr::group_by(cyl) |>
      dplyr::summarise(power_weight = mean(hp / wt), .groups = "drop")
  }

  expect_same_as_dplyr(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(mtcars, pipeline, arrange_by = "cyl", ignore_col_types = TRUE)
})

test_that("summarise() raises a diagnosable error for a genuinely unparseable aggregation sub-expression", {
  skip_if_no_gpu()

  expect_error(
    tbl_gpu(mtcars) |> dplyr::summarise(x = sum(paste0(carb))),
    "Invalid aggregation expression"
  )
})

# =============================================================================
# Unnamed expression auto-naming (S10)
# =============================================================================

test_that("summarise() warns and auto-names an unnamed aggregation (eager)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = FALSE)

  expect_warning(
    result <- dplyr::summarise(gpu_df, mean(mpg)),
    "Unnamed summarise expression 'mean\\(mpg\\)' will use expression as column name"
  )

  collected <- collect(result)
  expect_true("mean(mpg)" %in% names(collected))
  expect_equal(collected[["mean(mpg)"]], mean(mtcars$mpg))
})

test_that("summarise() warns and auto-names an unnamed aggregation (lazy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE)

  expect_warning(
    result <- dplyr::summarise(gpu_df, mean(mpg)),
    "Unnamed summarise expression 'mean\\(mpg\\)' will use expression as column name"
  )

  collected <- collect(result)
  expect_true("mean(mpg)" %in% names(collected))
  expect_equal(collected[["mean(mpg)"]], mean(mtcars$mpg))
})

# =============================================================================
# Result Structure
# =============================================================================

test_that("summarise() returns ungrouped result", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg = mean(mpg))

  expect_equal(dplyr::group_vars(result), character(0))
})

test_that("summarise() result is on GPU", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg = mean(mpg))

  expect_data_on_gpu(result)
})

test_that("summarise() can be chained with collect()", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg = mean(mpg)) |>
    collect()

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3)
})

# =============================================================================
# Error Handling
# =============================================================================

test_that("summarise() errors with no expressions", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::summarise(gpu_df),
    "requires at least one"
  )
})

test_that("summarise() errors on non-existent column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::summarise(gpu_df, avg = mean(nonexistent)),
    "not found"
  )
})

test_that("summarise() errors on unsupported function", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::summarise(gpu_df, result = median(mpg)),
    "Unsupported"
  )
})

test_that("summarise() errors with invalid expression format", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::summarise(gpu_df, result = mpg + 1),
    "Invalid aggregation"
  )
})

# =============================================================================
# Integration Tests
# =============================================================================

test_that("filter() then summarise() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::filter(mpg > 20) |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_hp = mean(hp)) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::filter(mpg > 20) |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_hp = mean(hp))

  expect_equal(nrow(result), nrow(r_result))
})

test_that("mutate() then summarise() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::mutate(hp_per_cyl = hp / cyl) |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_hpc = mean(hp_per_cyl)) |>
    collect()

  expect_true("avg_hpc" %in% names(result))
  expect_equal(nrow(result), 3)
})

test_that("select() then summarise() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::select(cyl, mpg, hp) |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_mpg = mean(mpg)) |>
    collect()

  expect_equal(names(result), c("cyl", "avg_mpg"))
})

# =============================================================================
# Large Data Tests
# =============================================================================

test_that("summarise() works with large datasets", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(200 * 1024 * 1024)  # Need 200MB

  # Create large data with known groups
  n <- 100000
  df <- data.frame(
    group = rep(1:100, each = n / 100),
    value = rnorm(n)
  )
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::group_by(group) |>
    dplyr::summarise(avg = mean(value)) |>
    collect()

  expect_equal(nrow(result), 100)
})

# =============================================================================
# Expressions Inside Aggregations
# =============================================================================

test_that("summarise() with comparison expression sum(col == value) works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(vs) |>
    dplyr::summarise(n_carb_4 = sum(carb == 4)) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::group_by(vs) |>
    dplyr::summarise(n_carb_4 = sum(carb == 4))

  expect_equal(nrow(result), 2)
  for (vs_val in c(0, 1)) {
    gpu_val <- result$n_carb_4[result$vs == vs_val]
    r_val <- r_result$n_carb_4[r_result$vs == vs_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() with comparison expression sum(col != value) works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(not_auto = sum(am != 0)) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(not_auto = sum(am != 0))

  expect_equal(nrow(result), 3)
  for (cyl_val in c(4, 6, 8)) {
    gpu_val <- result$not_auto[result$cyl == cyl_val]
    r_val <- r_result$not_auto[r_result$cyl == cyl_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() with comparison expression sum(col > value) works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(high_mpg = sum(mpg > 20)) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(high_mpg = sum(mpg > 20))

  for (cyl_val in c(4, 6, 8)) {
    gpu_val <- result$high_mpg[result$cyl == cyl_val]
    r_val <- r_result$high_mpg[r_result$cyl == cyl_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() with mean of comparison expression works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(pct_auto = mean(am == 1)) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(pct_auto = mean(am == 1))

  for (cyl_val in c(4, 6, 8)) {
    gpu_val <- result$pct_auto[result$cyl == cyl_val]
    r_val <- r_result$pct_auto[r_result$cyl == cyl_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() with multiple expression aggregations works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(vs) |>
    dplyr::summarise(
      count = n(),
      min_am = min(am),
      n_carb_4 = sum(carb == 4)
    ) |>
    collect()

  # Compare with R - use aggregate for simpler test
  r_counts <- as.data.frame(table(mtcars$vs))
  r_carb4 <- aggregate(carb ~ vs, data = mtcars, FUN = function(x) sum(x == 4))

  expect_equal(nrow(result), 2)
  expect_true(all(c("vs", "count", "min_am", "n_carb_4") %in% names(result)))

  for (vs_val in c(0, 1)) {
    gpu_carb4 <- result$n_carb_4[result$vs == vs_val]
    r_carb4_val <- r_carb4$carb[r_carb4$vs == vs_val]
    expect_equal(gpu_carb4, r_carb4_val, tolerance = 1e-10)
  }
})

test_that("summarise() with arithmetic expression works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_hp_per_cyl = mean(hp / cyl)) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg_hp_per_cyl = mean(hp / cyl))

  for (cyl_val in c(4, 6, 8)) {
    gpu_val <- result$avg_hp_per_cyl[result$cyl == cyl_val]
    r_val <- r_result$avg_hp_per_cyl[r_result$cyl == cyl_val]
    expect_equal(gpu_val, r_val, tolerance = 1e-10)
  }
})

test_that("summarise() ungrouped with expression works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::summarise(n_6cyl = sum(cyl == 6)) |>
    collect()

  expect_equal(nrow(result), 1)
  expect_equal(result$n_6cyl, sum(mtcars$cyl == 6), tolerance = 1e-10)
})

test_that("summarise() with column-to-column comparison works", {
  skip_if_no_gpu()

  # Create test data where we can compare columns
  df <- data.frame(
    group = rep(c("A", "B"), each = 5),
    x = c(1, 2, 3, 4, 5, 5, 4, 3, 2, 1),
    y = c(2, 2, 3, 3, 4, 4, 4, 4, 4, 4)
  )
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::group_by(group) |>
    dplyr::summarise(x_lt_y = sum(x < y)) |>
    collect()

  # Compare with R
  r_result <- df |>
    dplyr::group_by(group) |>
    dplyr::summarise(x_lt_y = sum(x < y))

  expect_equal(nrow(result), 2)
  # Note: group order may differ between GPU and R results
  expect_equal(sort(result$x_lt_y), sort(r_result$x_lt_y))
})

# =============================================================================
# summarize() alias
# =============================================================================

test_that("summarize() is alias for summarise()", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result1 <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(avg = mean(mpg)) |>
    collect()

  result2 <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarize(avg = mean(mpg)) |>
    collect()

  expect_equal(result1, result2)
})

# =============================================================================
# Complex Aggregation Expressions
# =============================================================================

test_that("summarise() with n(), max of expression, and sum of comparison works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(vs) |>
    dplyr::summarise(
      n = n(),
      max_am_100 = max(am * 100),
      n_carb_4 = sum(carb == 4)
    ) |>
    collect()

  # Compare with R dplyr
  r_result <- mtcars |>
    dplyr::group_by(vs) |>
    dplyr::summarise(
      n = dplyr::n(),
      max_am_100 = max(am * 100),
      n_carb_4 = sum(carb == 4)
    )

  expect_equal(nrow(result), 2)
  expect_true(all(c("vs", "n", "max_am_100", "n_carb_4") %in% names(result)))

  # Check values for each group

  for (vs_val in c(0, 1)) {
    gpu_row <- result[result$vs == vs_val, ]
    r_row <- r_result[r_result$vs == vs_val, ]

    expect_equal(gpu_row$n, r_row$n, tolerance = 1e-10)
    expect_equal(gpu_row$max_am_100, r_row$max_am_100, tolerance = 1e-10)
    expect_equal(gpu_row$n_carb_4, r_row$n_carb_4, tolerance = 1e-10)
  }
})

test_that("summarise() with min/max of arithmetic expressions works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  result <- gpu_df |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(
      min_hp_wt = min(hp / wt),
      max_mpg_10 = max(mpg * 10)
    ) |>
    collect()

  # Compare with R
  r_result <- mtcars |>
    dplyr::group_by(cyl) |>
    dplyr::summarise(
      min_hp_wt = min(hp / wt),
      max_mpg_10 = max(mpg * 10)
    )

  for (cyl_val in c(4, 6, 8)) {
    gpu_row <- result[result$cyl == cyl_val, ]
    r_row <- r_result[r_result$cyl == cyl_val, ]

    expect_equal(gpu_row$min_hp_wt, r_row$min_hp_wt, tolerance = 1e-10)
    expect_equal(gpu_row$max_mpg_10, r_row$max_mpg_10, tolerance = 1e-10)
  }
})
