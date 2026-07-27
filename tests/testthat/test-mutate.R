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

test_that("mutate() handles nested arithmetic with parentheses", {
  skip_if_no_gpu()

  df <- data.frame(
    fare_amount = c(10, 20, 40),
    tip_amount = c(1, 2, 8),
    tolls_amount = c(0, 1, 2),
    VendorID = c(1, 2, 1)
  )
  gpu_df <- tbl_gpu(df)

  mutated <- dplyr::mutate(gpu_df, tip_pct = (tip_amount / fare_amount) * 100)
  result <- collect(mutated)

  expect_equal(result$tip_pct, (df$tip_amount / df$fare_amount) * 100,
               tolerance = 1e-10)
})

test_that("mutate() handles nested arithmetic with parentheses in lazy mode", {
  skip_if_no_gpu()

  df <- data.frame(
    fare_amount = c(10, 20, 40),
    tip_amount = c(1, 2, 8),
    tolls_amount = c(0, 1, 2),
    VendorID = c(1, 2, 1)
  )
  gpu_df <- tbl_gpu(df, lazy = TRUE)

  mutated <- dplyr::mutate(gpu_df, tip_pct = (tip_amount / fare_amount) * 100)
  result <- collect(mutated)

  expect_equal(result$tip_pct, (df$tip_amount / df$fare_amount) * 100,
               tolerance = 1e-10)
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

test_that("lazy single-expression replace-in-place mutate matches dplyr (S5a)", {
  skip_if_no_gpu()

  # lower_mutate() always routes through gpu_mutate_expr(), even for a
  # single expression that replaces an existing column in place -- this
  # exercises that path against the dplyr oracle.
  df <- data.frame(x = c(1, 2, 3, 4, 5))

  pipeline <- function(d) {
    d |> dplyr::mutate(x = x * 2)
  }

  expect_same_as_dplyr_lazy(df, pipeline)
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

test_that("mutate() supports chained addition of multiple columns", {
  skip_if_no_gpu()

  df <- data.frame(a = c(1, 2, 3), b = c(10, 20, 30), c = c(100, 200, 300))
  expected <- df |>
    dplyr::mutate(total = a + b + c)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::mutate(total = a + b + c) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})

test_that("mutate() matches dplyr in eager and lazy modes", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- df |>
    dplyr::mutate(kpl = mpg * 0.425) |>
    dplyr::mutate(ratio = hp / wt)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::mutate(kpl = mpg * 0.425) |>
      dplyr::mutate(ratio = hp / wt) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
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

  # Phase 1 expression-engine cutover (T5): `mpg + "ten"` now parses fine as
  # IR (a `+` call over a column and a STRING literal) -- the error moves
  # from parse time to type-inference time, via infer_mutate_output_type()'s
  # existing STRING rejection (D5 rule 2, pinned verbatim at test-ast.R:205-
  # 209), reached through ir_infer_type(). See
  # scratchpad/phase1_expression_engine.md section 4's error-contract
  # ledger: the message changes, but the underlying "no arithmetic on
  # non-numeric input" rejection survives.
  gpu_df <- tbl_gpu(mtcars)

  expect_error(
    dplyr::mutate(gpu_df, new_col = mpg + "ten"),
    "does not support arithmetic on STRING"
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

test_that("mutate() no longer errors on %% (now supported by the expression engine)", {
  skip_if_no_gpu()

  # Phase 1 expression-engine cutover (T5): mutate() and filter() share the
  # same ir_parse_quo() parser, and %% has been in ir_call_registry (PYMOD)
  # since T1/T2 -- so this now WORKS instead of erroring. See
  # scratchpad/phase1_expression_engine.md section 4's error-contract
  # ledger (test-mutate.R:440 "CHANGES").
  gpu_df <- tbl_gpu(mtcars)
  mutated <- dplyr::mutate(gpu_df, new_col = mpg %% 5)

  result <- collect(mutated)
  expect_equal(result$new_col, mtcars$mpg %% 5, tolerance = 1e-10)
})

test_that("mutate() errors on unsupported operation", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # strsplit() has no ir_call_registry entry and its argument references a
  # real column, so ir_parse_quo() can't constant-fold it either -- a
  # genuinely opaque expression, replacing the old %% example (which the
  # IR now understands, see the test above).
  expect_error(
    dplyr::mutate(gpu_df, new_col = strsplit(mpg, "")),
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

  # Result type should be FLOAT64 because the literal `1` is a double in R
  # (dplyr agrees: `1:10 + 1` is also double). See D5 rule 5 /
  # infer_mutate_output_type().
  expect_equal(unname(mutated$schema$types[2]), "FLOAT64")
})

test_that("mutate() preserves INT32 for int_col + 1L (integer literal scalar)", {
  skip_if_no_gpu()

  df <- data.frame(int_col = 1:10)
  gpu_df <- tbl_gpu(df)

  mutated <- dplyr::mutate(gpu_df, new_col = int_col + 1L)

  # Unlike `int_col + 1` above, an integer literal scalar keeps
  # INT32 + INT32 -> INT32 (D5 rule 5): this is the S5b milestone -- mutate
  # type inference now flows uniformly through make_mutate_expr() /
  # infer_mutate_output_type() in both exec modes, so eager no longer
  # hard-codes FLOAT64.
  expect_equal(unname(mutated$schema$types[2]), "INT32")

  result <- collect(mutated)
  expect_type(result$new_col, "integer")
  expect_equal(result$new_col, df$int_col + 1L)
})

test_that("mutate() with int_col / 2L is FLOAT64 (division always promotes)", {
  skip_if_no_gpu()

  df <- data.frame(int_col = 1:10)
  gpu_df <- tbl_gpu(df)

  mutated <- dplyr::mutate(gpu_df, new_col = int_col / 2L)

  expect_equal(unname(mutated$schema$types[2]), "FLOAT64")

  result <- collect(mutated)
  expect_equal(result$new_col, df$int_col / 2L)
})

test_that("mutate() with int_col ^ 2L is FLOAT64 (power always promotes)", {
  skip_if_no_gpu()

  # S5c: `^` now follows D5 rule 4 and always promotes to FLOAT64, matching
  # R's own `2L^2L` -> double. This flips the S5b-pinned INT32 behavior.
  df <- data.frame(int_col = 1:10)
  gpu_df <- tbl_gpu(df)

  mutated <- dplyr::mutate(gpu_df, new_col = int_col ^ 2L)

  expect_equal(unname(mutated$schema$types[2]), "FLOAT64")

  result <- collect(mutated)
  expect_equal(result$new_col, df$int_col ^ 2L)
})

test_that("mutate() logical arithmetic type parity with dplyr (oracle, eager)", {
  skip_if_no_gpu()

  # BOOL8 + BOOL8 -> INT32 (D5 rule 5), matching R's `TRUE + TRUE == 2L`.
  # Verified empirically that gpu_mutate_expr()'s compute_column() path
  # accepts BOOL8 operands with an INT32 output type.
  df <- data.frame(flag = c(TRUE, FALSE, TRUE, TRUE))

  pipeline <- function(d) {
    d |> dplyr::mutate(s = flag + flag)
  }

  expect_same_as_dplyr(df, pipeline, ignore_col_types = FALSE)
})

test_that("mutate() logical arithmetic type parity with dplyr (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(flag = c(TRUE, FALSE, TRUE, TRUE))

  pipeline <- function(d) {
    d |> dplyr::mutate(s = flag + flag)
  }

  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = FALSE)
})

test_that("mutate() type parity with dplyr for integer-preserving arithmetic (oracle, eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10, y = 10:1)

  pipeline <- function(d) {
    d |> dplyr::mutate(z = x + 1L)
  }

  # The milestone this step unlocks: mutate() type parity WITHOUT the
  # ignore_col_types = TRUE escape hatch, because both x and the output are
  # now INT32/integer end to end.
  expect_same_as_dplyr(df, pipeline, ignore_col_types = FALSE)
})

test_that("mutate() type parity with dplyr for integer-preserving arithmetic (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10, y = 10:1)

  pipeline <- function(d) {
    d |> dplyr::mutate(z = x + 1L)
  }

  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = FALSE)
})

test_that("mutate() chained expression in a single call matches dplyr (oracle, eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3), y = c(10, 20, 30))

  pipeline <- function(d) {
    # `b` references `a`, created earlier in the SAME mutate() call -- this
    # exercises the current_schema()-threading in mutate.tbl_gpu() that lets
    # later dots see earlier dots' output columns/types.
    d |> dplyr::mutate(a = x + y, b = a * 2)
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() chained expression in a single call matches dplyr (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3), y = c(10, 20, 30))

  pipeline <- function(d) {
    d |> dplyr::mutate(a = x + y, b = a * 2)
  }

  expect_same_as_dplyr_lazy(df, pipeline)
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

# =============================================================================
# Unnamed expression auto-naming (S10)
# =============================================================================

test_that("mutate() warns and auto-names an unnamed expression (eager)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = FALSE)

  expect_warning(
    mutated <- dplyr::mutate(gpu_df, mpg * 2),
    "Unnamed mutate expression 'mpg \\* 2' will use expression as column name"
  )

  result <- collect(mutated)
  expect_true("mpg * 2" %in% names(result))
  expect_equal(result[["mpg * 2"]], mtcars$mpg * 2)
})

test_that("mutate() warns and auto-names an unnamed expression (lazy)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars, lazy = TRUE)

  expect_warning(
    mutated <- dplyr::mutate(gpu_df, mpg * 2),
    "Unnamed mutate expression 'mpg \\* 2' will use expression as column name"
  )

  result <- collect(mutated)
  expect_true("mpg * 2" %in% names(result))
  expect_equal(result[["mpg * 2"]], mtcars$mpg * 2)
})

# =============================================================================
# Expression-engine smoke tests (Phase 1 T5) -- the full capability wave
# (arbitrary nesting, round(), log(x, base), multi-dot chains, ...) is
# T6's job; these just spot-check that the ir_parse_quo() cutover carries
# mutate() the same expressive surface filter() already has.
# =============================================================================

test_that("mutate() nested arithmetic ((x+y)*z-1) matches dplyr (oracle, eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3), y = c(10, 20, 30), z = c(2, 3, 4))

  pipeline <- function(d) {
    d |> dplyr::mutate(w = (x + y) * z - 1)
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() nested arithmetic ((x+y)*z-1) matches dplyr (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3), y = c(10, 20, 30), z = c(2, 3, 4))

  pipeline <- function(d) {
    d |> dplyr::mutate(w = (x + y) * z - 1)
  }

  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() %% and %/% match dplyr (oracle, eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(-7, 7, 10, -10), y = c(3, 3, 4, 4))

  pipeline <- function(d) {
    d |> dplyr::mutate(m = x %% y, fd = x %/% y)
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() %% and %/% match dplyr (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(-7, 7, 10, -10), y = c(3, 3, 4, 4))

  pipeline <- function(d) {
    d |> dplyr::mutate(m = x %% y, fd = x %/% y)
  }

  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() sqrt/log/abs match dplyr (oracle, eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 4, 9, 16), y = c(-2, -1, 1, 2))

  pipeline <- function(d) {
    d |> dplyr::mutate(s = sqrt(x), l = log(x), a = abs(y))
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() sqrt/log/abs match dplyr (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 4, 9, 16), y = c(-2, -1, 1, 2))

  pipeline <- function(d) {
    d |> dplyr::mutate(s = sqrt(x), l = log(x), a = abs(y))
  }

  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() scalar-on-left (10 - x) matches dplyr (oracle, eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3, 4))

  pipeline <- function(d) {
    d |> dplyr::mutate(y = 10 - x)
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() scalar-on-left (10 - x) matches dplyr (oracle, lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3, 4))

  pipeline <- function(d) {
    d |> dplyr::mutate(y = 10 - x)
  }

  expect_same_as_dplyr_lazy(df, pipeline)
})
