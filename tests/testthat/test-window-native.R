# Tests for the C++ window kernel + ast_window node (Phase 5, task W2)
#
# scratchpad/phase5_window_design.md section 2.2: the six-step stable-sort /
# gather / groupby-scan-or-shift / scatter-back algorithm, implemented in
# src/ops_window.cpp. No verb is wired to window functions yet (that's
# W3/W5/W6/W7), so every test here hand-builds an `ast_window` node (via
# `window_spec()`, R/window.R) and drives it through `push_op()`/
# `lower_and_execute()` directly -- the same pattern test-execute.R uses for
# hand-built `ast_select` nodes.
#
# This file also implements the four W2 empirical checks from the design
# doc (E1/E2/E5/E6 -- E3/E4 are W4's, the rank/agg families):
#   E1: grouped cumsum on a shuffled 100k-row table == dplyr (validates the
#       entire perm/gather/scan/scatter round trip).
#   E2: grouped row_number() with duplicate + NA group keys.
#   E5: cudf's own group-scan NA is non-sticky (verified against cudf
#       source, scratchpad/phase5_window_design.md section 0); this file
#       tests that our sticky-NA fix reproduces R's sticky
#       cumsum(c(1,2,NA,4)) == 1,3,NA,NA.
#   E6: lag()/lead() fill-value type + NA default.

# -----------------------------------------------------------------------------
# Local helpers
# -----------------------------------------------------------------------------

# Build a tbl_gpu, wrap `specs` in one ast_window node over it, push it
# through push_op() (eager: lowers+executes immediately; lazy: stores the
# AST and lowers on collect()), and collect the result back to R.
run_window <- function(df, specs, group_cols = character(), lazy = FALSE) {
  tbl <- tbl_gpu(df, lazy = lazy)
  node <- ast_window(input_node(tbl), specs, group_cols)
  result <- push_op(tbl, node)
  collect(result)
}

# -----------------------------------------------------------------------------
# E1: grouped cumsum on a shuffled 100k-row table == dplyr
# -----------------------------------------------------------------------------

test_that("[E1] grouped cumsum() on a shuffled 100k-row table matches dplyr", {
  skip_if_no_gpu()

  set.seed(101)
  n <- 100000L
  df <- data.frame(
    grp = sample(1:200, n, replace = TRUE),
    x = rnorm(n)
  )
  # Shuffle rows so the input is NOT already grouped/sorted -- this is what
  # actually exercises the gather/scatter round trip (an already-sorted
  # input could accidentally "work" even with a broken permutation).
  df <- df[sample(n), ]
  rownames(df) <- NULL

  expected <- dplyr::mutate(dplyr::group_by(df, grp), cs = cumsum(x))$cs

  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  result <- run_window(df, list(spec), group_cols = "grp")

  expect_equal(result$cs, expected, tolerance = 1e-9)
  expect_equal(result$grp, df$grp)
  expect_equal(result$x, df$x)
})

# -----------------------------------------------------------------------------
# E2: grouped row_number() with duplicate + NA group keys
# -----------------------------------------------------------------------------

test_that("[E2] grouped row_number() handles duplicate + NA group keys", {
  skip_if_no_gpu()

  df <- data.frame(
    grp = c(1, 1, NA, 2, 2, NA, 1, 2, NA),
    x = 1:9
  )

  expected <- dplyr::mutate(dplyr::group_by(df, grp), rn = dplyr::row_number())$rn

  spec <- window_spec("rn", "row_number", output_type = "INT32")
  result <- run_window(df, list(spec), group_cols = "grp")

  expect_equal(result$rn, as.integer(expected))
})

test_that("[E2b] grouped row_number() with duplicate keys, no NAs", {
  skip_if_no_gpu()

  df <- data.frame(grp = c(2, 1, 2, 1, 1, 2), x = 1:6)
  expected <- dplyr::mutate(dplyr::group_by(df, grp), rn = dplyr::row_number())$rn

  spec <- window_spec("rn", "row_number", output_type = "INT32")
  result <- run_window(df, list(spec), group_cols = "grp")

  expect_equal(result$rn, as.integer(expected))
})

# -----------------------------------------------------------------------------
# E5: sticky-NA fix reproduces R's cumsum(c(1,2,NA,4)) == 1,3,NA,NA
# -----------------------------------------------------------------------------

test_that("[E5] cumsum() sticky-NA fix matches R's sticky semantics, ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, NA, 4))
  expected <- cumsum(df$x)
  expect_equal(expected, c(1, 3, NA, NA))  # pin R's own sticky behavior

  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  result <- run_window(df, list(spec))

  expect_equal(result$cs, expected)
})

test_that("[E5b] cumsum() sticky-NA fix matches R's sticky semantics, grouped", {
  skip_if_no_gpu()

  df <- data.frame(grp = c(1, 1, 1, 1, 2, 2, 2), x = c(1, 2, NA, 4, 5, NA, 1))
  expected <- dplyr::mutate(dplyr::group_by(df, grp), cs = cumsum(x))$cs

  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  result <- run_window(df, list(spec), group_cols = "grp")

  expect_equal(result$cs, expected)
})

test_that("[E5c] cummax()/cummin()/cumprod() sticky-NA fix, ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, NA, 5, 2))

  spec_max <- window_spec("mx", "cummax", value_col = "x", output_type = "FLOAT64")
  spec_min <- window_spec("mn", "cummin", value_col = "x", output_type = "FLOAT64")
  spec_prod <- window_spec("pr", "cumprod", value_col = "x", output_type = "FLOAT64")

  result <- run_window(df, list(spec_max, spec_min, spec_prod))

  expect_equal(result$mx, cummax(df$x))
  expect_equal(result$mn, cummin(df$x))
  expect_equal(result$pr, cumprod(df$x))
})

# -----------------------------------------------------------------------------
# E6: lag()/lead() fill-value type + NA default
# -----------------------------------------------------------------------------

test_that("[E6] lag() with no default fills NA of the value column's type", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5)
  spec <- window_spec("lg", "lag", value_col = "x", output_type = "INT32", n = 1L)
  result <- run_window(df, list(spec))

  expect_equal(result$lg, c(NA_integer_, 1:4))
  expect_type(result$lg, "integer")
})

test_that("[E6b] lag() with an explicit non-NA default uses it, correct type", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5)
  spec <- window_spec("lg", "lag", value_col = "x", output_type = "INT32", n = 1L,
                       default = ir_lit_from_r(0L))
  result <- run_window(df, list(spec))

  expect_equal(result$lg, c(0L, 1:4))
  expect_type(result$lg, "integer")
})

test_that("[E6c] lead() with an explicit default, default = NA behaves like no default", {
  skip_if_no_gpu()

  df <- data.frame(x = c(10.5, 20.5, 30.5, 40.5))
  spec_na <- window_spec("ld_na", "lead", value_col = "x", output_type = "FLOAT64", n = 1L,
                          default = ir_lit(value = NULL, type = NULL, na = TRUE))
  spec_nodefault <- window_spec("ld", "lead", value_col = "x", output_type = "FLOAT64", n = 1L)

  result <- run_window(df, list(spec_na, spec_nodefault))

  expect_equal(result$ld_na, c(20.5, 30.5, 40.5, NA_real_))
  expect_equal(result$ld, c(20.5, 30.5, 40.5, NA_real_))
})

# -----------------------------------------------------------------------------
# Additional coverage (task list, section 5)
# -----------------------------------------------------------------------------

test_that("ungrouped cumsum() matches base R", {
  skip_if_no_gpu()

  df <- data.frame(x = c(5, 3, 1, 4, 2))
  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  result <- run_window(df, list(spec))

  expect_equal(result$cs, cumsum(df$x))
})

test_that("lag() with n = 2 matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10)
  expected <- dplyr::lag(df$x, n = 2)

  spec <- window_spec("lg", "lag", value_col = "x", output_type = "INT32", n = 2L)
  result <- run_window(df, list(spec))

  expect_equal(result$lg, expected)
})

test_that("lead() with a default matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:6)
  expected <- dplyr::lead(df$x, n = 1, default = -1L)

  spec <- window_spec("ld", "lead", value_col = "x", output_type = "INT32", n = 1L,
                       default = ir_lit_from_r(-1L))
  result <- run_window(df, list(spec))

  expect_equal(result$ld, expected)
})

test_that("mixed specs (cumsum + lag) in one node share a single (no-order) frame", {
  skip_if_no_gpu()

  df <- data.frame(x = c(10, 20, 30, 40, 50))
  spec_cs <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  spec_lag <- window_spec("lg", "lag", value_col = "x", output_type = "FLOAT64", n = 1L)

  result <- run_window(df, list(spec_cs, spec_lag))

  expect_equal(result$cs, cumsum(df$x))
  expect_equal(result$lg, dplyr::lag(df$x, n = 1))
})

test_that("grouped lag() preserves original row order on shuffled input", {
  skip_if_no_gpu()

  set.seed(202)
  df <- data.frame(grp = sample(1:5, 500, replace = TRUE), x = rnorm(500))
  df <- df[sample(nrow(df)), ]
  rownames(df) <- NULL

  expected <- dplyr::mutate(dplyr::group_by(df, grp), lg = dplyr::lag(x))$lg

  spec <- window_spec("lg", "lag", value_col = "x", output_type = "FLOAT64", n = 1L)
  result <- run_window(df, list(spec), group_cols = "grp")

  expect_equal(result$lg, expected)
  expect_equal(result$grp, df$grp)
  expect_equal(result$x, df$x)
})

test_that("0-row table: window node returns 0 rows with the appended columns", {
  skip_if_no_gpu()

  df <- data.frame(grp = integer(0), x = numeric(0))
  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  result <- run_window(df, list(spec), group_cols = "grp")

  expect_equal(nrow(result), 0)
  expect_true("cs" %in% names(result))
})

test_that("single-row groups: row_number()/cumsum()/lag() all resolve trivially", {
  skip_if_no_gpu()

  df <- data.frame(grp = c(1, 2, 3), x = c(10, 20, 30))
  spec_rn <- window_spec("rn", "row_number", output_type = "INT32")
  spec_cs <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  spec_lg <- window_spec("lg", "lag", value_col = "x", output_type = "FLOAT64", n = 1L)

  result <- run_window(df, list(spec_rn, spec_cs, spec_lg), group_cols = "grp")

  expect_equal(result$rn, c(1L, 1L, 1L))
  expect_equal(result$cs, df$x)
  expect_equal(result$lg, c(NA_real_, NA_real_, NA_real_))
})

test_that("all-one-group: grouped cumsum()/row_number() equal the ungrouped case", {
  skip_if_no_gpu()

  df <- data.frame(grp = rep(1, 6), x = c(5, 3, 8, 1, 9, 2))
  spec_rn <- window_spec("rn", "row_number", output_type = "INT32")
  spec_cs <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")

  result <- run_window(df, list(spec_rn, spec_cs), group_cols = "grp")

  expect_equal(result$rn, 1:6)
  expect_equal(result$cs, cumsum(df$x))
})

test_that("lazy exec mode produces the same result as eager", {
  skip_if_no_gpu()

  df <- data.frame(grp = c(1, 1, 2, 2, 1), x = c(1, 2, 3, 4, 5))
  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")

  eager <- run_window(df, list(spec), group_cols = "grp", lazy = FALSE)
  lazy <- run_window(df, list(spec), group_cols = "grp", lazy = TRUE)

  expect_equal(tibble::as_tibble(eager), tibble::as_tibble(lazy))
})

# -----------------------------------------------------------------------------
# ast_window()/infer_schema()/is_barrier() plumbing (pure R, no GPU)
# -----------------------------------------------------------------------------

test_that("ast_window() appends one column per spec, never replaces", {
  source <- ast_source(list(names = c("x", "grp"), types = c("FLOAT64", "INT32")))
  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  node <- ast_window(source, list(spec), group_cols = "grp")

  expect_s3_class(node, "ast_window")
  expect_equal(node$type, "window")

  schema <- infer_schema(node)
  expect_equal(schema$names, c("x", "grp", "cs"))
  expect_equal(unname(schema$types), c("FLOAT64", "INT32", "FLOAT64"))
})

test_that("is_barrier() treats window as a barrier", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  spec <- window_spec("cs", "cumsum", value_col = "x", output_type = "FLOAT64")
  node <- ast_window(source, list(spec))

  expect_true(is_barrier(node))
})

# -----------------------------------------------------------------------------
# Defensive rejections (W2 scope: shift + scan families only)
# -----------------------------------------------------------------------------

test_that("lower_window() rejects a ranked row_number(x) spec (W4 scope)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, 2))
  spec <- window_spec("rn", "row_number", value_col = "x", order_cols = "x",
                       order_desc = FALSE, output_type = "INT32")

  expect_error(run_window(df, list(spec)), "W4")
})

test_that("lower_window() rejects an agg-family spec (mean(), W4 scope)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, 2))
  spec <- window_spec("m", "mean", value_col = "x", output_type = "FLOAT64")

  expect_error(run_window(df, list(spec)), "W4")
})
