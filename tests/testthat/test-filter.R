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

test_that("filter() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- dplyr::filter(df, mpg > 20, cyl == 4)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::filter(mpg > 20) |>
      dplyr::filter(cyl == 4) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
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

  # S4 note: filter(TRUE) is now a genuine no-op (parse_filter_expr()
  # returns an empty predicate list, so no ast_filter node is pushed and
  # `.data` is returned unchanged -- same GPU pointer as the input). None
  # of the assertions below check pointer/object freshness, so no
  # relaxation was needed here; see test-filter.R's "creates new GPU
  # allocation (immutable)" test below for the case that does.
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

  # Comparing a numeric column against a character literal is a clear type
  # mismatch, caught before reaching the GPU (S5, Phase 1 expression-engine
  # cutover: check_filter_comparison_types() in R/filter.R). The pre-IR
  # parser's "numeric scalar" message is gone -- this is a genuine error
  # contract change (scratchpad/phase1_expression_engine.md section 4).
  expect_error(
    dplyr::filter(gpu_df, mpg > "twenty"),
    "Cannot compare column 'mpg'"
  )
})

test_that("filter() with %in% works (IR cutover, task T3)", {
  skip_if_no_gpu()

  # %in% used to fall back to the "only supports comparisons" error; the
  # Phase 1 expression-engine cutover parses it directly into a GPU
  # predicate (ir_call_registry's "%in%" entry, already implemented in T2),
  # so this is now rewritten as a correctness test rather than an error
  # pin (scratchpad/phase1_expression_engine.md section 4 / task T3 item 7).
  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, cyl %in% c(4, 6))

  result <- collect(filtered)
  expect_true(all(result$cyl %in% c(4, 6)))
  expect_equal(nrow(result), sum(mtcars$cyl %in% c(4, 6)))
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

# =============================================================================
# Lazy variants of TRUE / FALSE / vector-mask cases (S4)
# =============================================================================

test_that("filter() with scalar TRUE returns all rows (lazy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE)
  filtered <- dplyr::filter(gpu_df, TRUE)

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 32)

  result <- collect(filtered)
  expect_equal(nrow(result), 32)
  expect_equal(result, mtcars, ignore_attr = TRUE)
})

test_that("filter() with scalar FALSE returns no rows (lazy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE)
  filtered <- dplyr::filter(gpu_df, FALSE)

  # FALSE builds a real (impossible) predicate, so this stays a pending AST
  # node rather than executing immediately. dim() reads gpu_dim() off the
  # *unmaterialized* source pointer for lazy tables with pending ops (a
  # pre-existing, orthogonal limitation, not introduced by S4) so only the
  # column count is meaningful before collect()/compute(); row count is
  # checked after materializing.
  expect_data_on_gpu(filtered)
  expect_true(has_pending_ops(filtered))
  expect_equal(dim(filtered)[2], 11)

  result <- collect(filtered)
  expect_equal(nrow(result), 0)
  expect_equal(names(result), names(mtcars))
})

test_that("filter() with vector of TRUE returns all rows (lazy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE)
  filtered <- dplyr::filter(gpu_df, rep(TRUE, nrow(mtcars)))

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 32)

  result <- collect(filtered)
  expect_equal(nrow(result), 32)
})

test_that("filter() with vector of FALSE returns no rows (lazy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE)
  filtered <- dplyr::filter(gpu_df, rep(FALSE, nrow(mtcars)))

  expect_data_on_gpu(filtered)
  expect_equal(dim(filtered)[1], 0)

  result <- collect(filtered)
  expect_equal(nrow(result), 0)
})

# =============================================================================
# Oracle parity (S4)
# =============================================================================

test_that("filter() column-vs-column predicate chain matches dplyr (eager and lazy)", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(5, 3, 8, 1, 9, 4),
    y = c(2, 4, 1, 6, 3, 4),
    z = c(1, 10, 2, 3, 20, 5)
  )

  pipeline <- function(d) {
    d |> dplyr::filter(x > y, z <= 5)
  }

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# IR cutover smoke tests (task T3): a handful of compound-predicate shapes
# that were previously opaque (CPU-fallback or error) and now parse
# directly into one fused GPU predicate. Comprehensive coverage (string
# comparisons, is.na()/between() oracle parity, arbitrary nesting) is
# task T4's job (test-dplyr-filter.R); these just prove the cutover is
# live in the same commit that made it live.
# =============================================================================

test_that("filter() with a single & expression in one dot works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, mpg > 20 & cyl == 4)

  result <- collect(filtered)
  expect_true(all(result$mpg > 20 & result$cyl == 4))
  expect_equal(nrow(result), sum(mtcars$mpg > 20 & mtcars$cyl == 4))
})

test_that("filter() with a single | expression in one dot works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, cyl == 4 | cyl == 8)

  result <- collect(filtered)
  expect_true(all(result$cyl == 4 | result$cyl == 8))
  expect_equal(nrow(result), sum(mtcars$cyl == 4 | mtcars$cyl == 8))
})

test_that("filter() with is.na() works", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, NA, 3, NA, 5))
  gpu_df <- tbl_gpu(df)
  filtered <- dplyr::filter(gpu_df, is.na(x))

  result <- collect(filtered)
  expect_equal(nrow(result), 2)
})

test_that("filter() with between() works", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  filtered <- dplyr::filter(gpu_df, dplyr::between(mpg, 15, 20))

  result <- collect(filtered)
  expect_true(all(result$mpg >= 15 & result$mpg <= 20))
  expect_equal(nrow(result), sum(mtcars$mpg >= 15 & mtcars$mpg <= 20))
})

test_that("filter() combines multiple dots and a compound expression into one fused predicate", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(5, 3, 8, 1, 9, 4),
    y = c(2, 4, 1, 6, 3, 4),
    z = c(1, 10, 2, 3, 20, 5)
  )

  pipeline <- function(d) {
    d |> dplyr::filter(x > y, z <= 5 | z > 15)
  }

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Fallback notifier (D4)
# =============================================================================

test_that("filter() vector-mask fallback warns under cuplyr.fallback = 'warn'", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gpu_df <- tbl_gpu(mtcars)

  expect_warning(
    dplyr::filter(gpu_df, rep(TRUE, nrow(mtcars))),
    "fell back to CPU evaluation"
  )
})

test_that("filter() vector-mask fallback errors under cuplyr.fallback = 'error'", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::filter(gpu_df, rep(TRUE, nrow(mtcars))),
    "fell back to CPU evaluation"
  )
})

test_that("filter() vector-mask fallback is silent by default", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL)

  gpu_df <- tbl_gpu(mtcars)

  expect_no_warning(dplyr::filter(gpu_df, rep(TRUE, nrow(mtcars))))
})
