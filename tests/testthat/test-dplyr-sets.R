# Oracle parity tests for Phase 8: set operations (union(), union_all(),
# intersect(), setdiff(), symdiff(), setequal()).
#
# Mirrors dplyr's own test-sets.R (installed package source, see
# scratchpad/workflow_state.md Phase 8 kickoff note) section by section,
# adapted to this package's oracle-test conventions, plus a matrix of
# cuplyr-specific empirically-verified semantics (type promotion strictness,
# NA-row matching, column-order tolerance, factor handling, error text) the
# task called out as needing verification against dplyr 1.2.1 directly.

# =============================================================================
# Shared helper: run `op_fn` eagerly AND lazily on tbl_gpu, compare both
# against the dplyr-on-data.frame oracle. Mirrors test-dplyr-join-by.R's own
# `compare_join_by()` (self-sufficient per this project's testthat
# conventions, rather than reaching into another test file).
# =============================================================================

compare_set_op <- function(x_df, y_df, op_fn, ..., arrange_by = NULL) {
  expected <- op_fn(x_df, y_df, ...)

  eager <- op_fn(tbl_gpu(x_df), tbl_gpu(y_df), ...) |> collect()
  lazy <- op_fn(tbl_gpu(x_df, lazy = TRUE), tbl_gpu(y_df, lazy = TRUE), ...) |> collect()

  expected_df <- as.data.frame(expected)
  eager_df <- as.data.frame(eager)
  lazy_df <- as.data.frame(lazy)

  if (!is.null(arrange_by)) {
    ord <- function(df) do.call(order, c(as.list(df[arrange_by]), list(na.last = TRUE)))
    expected_df <- expected_df[ord(expected_df), , drop = FALSE]
    eager_df <- eager_df[ord(eager_df), , drop = FALSE]
    lazy_df <- lazy_df[ord(lazy_df), , drop = FALSE]
    rownames(expected_df) <- NULL
    rownames(eager_df) <- NULL
    rownames(lazy_df) <- NULL
  }

  expect_equal(eager_df, expected_df)
  expect_equal(lazy_df, expected_df)
}

set_op_fns <- list(
  union = dplyr::union, union_all = dplyr::union_all,
  intersect = dplyr::intersect, setdiff = dplyr::setdiff,
  symdiff = dplyr::symdiff
)

# =============================================================================
# 1. Mirrors dplyr's own test-sets.R directly (section names kept close to
#    the original so a diff against dplyr's file stays legible).
# =============================================================================

test_that("x used as basis of output (dplyr #3839)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = 1:4, y = 1)
  df2 <- tibble::tibble(y = 1, x = c(4, 2))

  compare_set_op(df1, df2, dplyr::intersect)
  compare_set_op(df1, df2, dplyr::union)
  compare_set_op(df1, df2, dplyr::union_all)
  compare_set_op(df1, df2, dplyr::setdiff)
  compare_set_op(df1, df2, dplyr::symdiff)
})

test_that("set operations (apart from union_all) remove duplicates", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = c(1, 1, 2))
  df2 <- tibble::tibble(x = 2)

  compare_set_op(df1, df2, dplyr::intersect)
  compare_set_op(df1, df2, dplyr::union)
  compare_set_op(df1, df2, dplyr::union_all)
  compare_set_op(df1, df2, dplyr::setdiff)
  compare_set_op(df1, df2, dplyr::symdiff)
})

test_that("standard coercion rules are used (dplyr #799)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = 1:2, y = c(1, 1))
  df2 <- tibble::tibble(x = 1:2, y = 1:2)

  gx <- tbl_gpu(df1)
  gy <- tbl_gpu(df2)

  expect_equal(nrow(collect(dplyr::intersect(gx, gy))), 1)
  expect_equal(nrow(collect(dplyr::union(gx, gy))), 3)
  expect_equal(nrow(collect(dplyr::union_all(gx, gy))), 4)
  expect_equal(nrow(collect(dplyr::setdiff(gx, gy))), 1)
  expect_equal(nrow(collect(dplyr::symdiff(gx, gy))), 2)
})

test_that("grouping metadata is reconstructed (dplyr #3587)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = 1:4, g = rep(1:2, each = 2))
  df2 <- tibble::tibble(x = 3:6, g = rep(2:3, each = 2))

  gx <- tbl_gpu(df1) |> dplyr::group_by(g)
  gy <- tbl_gpu(df2)

  expect_equal(dplyr::group_vars(dplyr::intersect(gx, gy)), "g")
  expect_equal(dplyr::group_vars(dplyr::union(gx, gy)), "g")
  expect_equal(dplyr::group_vars(dplyr::union_all(gx, gy)), "g")
  expect_equal(dplyr::group_vars(dplyr::setdiff(gx, gy)), "g")
  expect_equal(dplyr::group_vars(dplyr::symdiff(gx, gy)), "g")
})

test_that("extra arguments in ... error", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(var = 1:3))
  gy <- tbl_gpu(tibble::tibble(var = 2:4))

  for (nm in names(set_op_fns)) {
    fn <- set_op_fns[[nm]]
    expect_error(fn(gx, gy, z = 3), "must be empty", info = nm)
  }
})

test_that("incompatible data frames error (dplyr #903): different column count", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(x = 1))
  gy <- tbl_gpu(tibble::tibble(x = 1, y = 1))

  for (nm in names(set_op_fns)) {
    fn <- set_op_fns[[nm]]
    expect_error(fn(gx, gy), "not compatible", info = nm)
    expect_error(fn(gx, gy), "Different number of columns: 1 vs 2", info = nm)
  }
})

# --- setequal --------------------------------------------------------------

test_that("setequal ignores column and row order", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = 1:2, y = 3:4)
  df2 <- df1[2:1, 2:1]

  expect_true(dplyr::setequal(tbl_gpu(df1), tbl_gpu(df2)))
  expect_true(dplyr::setequal(tbl_gpu(df2), tbl_gpu(df1)))
})

test_that("setequal ignores duplicated rows (dplyr #6057)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = 1)
  df2 <- df1[c(1, 1, 1), ]

  expect_true(dplyr::setequal(tbl_gpu(df1), tbl_gpu(df2)))
  expect_true(dplyr::setequal(tbl_gpu(df2), tbl_gpu(df1)))
})

test_that("setequal uses coercion rules (dplyr #6114)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x = 1)
  df2 <- tibble::tibble(x = 1L)

  expect_true(dplyr::setequal(tbl_gpu(df1), tbl_gpu(df2)))
  expect_true(dplyr::setequal(tbl_gpu(df2), tbl_gpu(df1)))
})

test_that("setequal tibbles must have same rows and columns", {
  skip_if_no_gpu()
  # Different rows are the definition of not equal -- not an error.
  expect_false(dplyr::setequal(tbl_gpu(tibble::tibble(x = 1:2)), tbl_gpu(tibble::tibble(x = 2:3))))

  # Different or incompatible columns are an error, like the other set ops.
  expect_error(
    dplyr::setequal(tbl_gpu(tibble::tibble(x = 1:2)), tbl_gpu(tibble::tibble(y = 1:2))),
    "not compatible"
  )
  expect_error(
    dplyr::setequal(tbl_gpu(tibble::tibble(x = 1:2)), tbl_gpu(tibble::tibble(x = c("a", "b")))),
    "not compatible"
  )
})

test_that("setequal checks y is a data frame", {
  skip_if_no_gpu()
  expect_error(
    dplyr::setequal(tbl_gpu(mtcars), 1),
    "must be a data frame"
  )
})

test_that("setequal checks for extra arguments", {
  skip_if_no_gpu()
  gt <- tbl_gpu(mtcars)
  expect_error(dplyr::setequal(gt, gt, z = 2), "must be empty")
})

# =============================================================================
# 2. cuplyr-specific semantics matrix (empirically verified against dplyr
#    1.2.1 directly, see R/sets.R's own header comment / roxygen for the
#    full write-up).
# =============================================================================

test_that("y may be a plain data.frame (auto-uploaded, no copy= needed)", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(x = 1:3))
  y_df <- tibble::tibble(x = c(2, 4))

  result <- dplyr::union(gx, y_df) |> collect()
  expect_equal(sort(result$x), c(1, 2, 3, 4))
})

test_that("column order may differ between x and y (matched by name)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(a = 1:3, b = c("x", "y", "z"))
  df2 <- tibble::tibble(b = c("y", "w"), a = c(2L, 4L))

  compare_set_op(df1, df2, dplyr::union)
  compare_set_op(df1, df2, dplyr::union_all)
  compare_set_op(df1, df2, dplyr::intersect)
  compare_set_op(df1, df2, dplyr::setdiff)
  compare_set_op(df1, df2, dplyr::symdiff)
})

test_that("missing/extra column names error, naming the offending columns", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(a = 1, b = 2))
  gy <- tbl_gpu(tibble::tibble(a = 1, c = 2))

  expect_error(
    dplyr::union(gx, gy),
    "Cols in `y` but not `x`: `c`", fixed = TRUE
  )
  expect_error(
    dplyr::union(gx, gy),
    "Cols in `x` but not `y`: `b`", fixed = TRUE
  )
})

test_that("incompatible column types error, naming the column and both types", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(a = 1L))
  gy <- tbl_gpu(tibble::tibble(a = "1"))

  expect_error(
    dplyr::union(gx, gy),
    "Incompatible types for column `a`: integer vs character",
    fixed = TRUE
  )
})

test_that("numeric type promotion: logical < integer < double", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(a = TRUE, b = 1L))
  gy <- tbl_gpu(tibble::tibble(a = 1L, b = 1.5))

  result <- dplyr::union(gx, gy) |> collect()
  expect_type(result$a, "integer")
  expect_type(result$b, "double")
})

test_that("Date vs. POSIXct promotes to POSIXct natively (Phase 11 L3, no fallback)", {
  skip_if_no_gpu()
  # Was a CPU fallback before Phase 11 L3 (no native TIMESTAMP_DAYS ->
  # TIMESTAMP_MICROSECONDS cast existed yet); gpu_cast_column() gained the
  # 5 timestamp targets, so check_set_op_compatible()'s `$needs_fallback`
  # never fires for this combination any more -- asserting `"error"` below
  # would itself fail loudly if a fallback were ever (re-)triggered.
  df1 <- tibble::tibble(a = as.Date("2020-01-01"))
  df2 <- tibble::tibble(a = as.POSIXct("2020-01-02", tz = "UTC"))
  gx <- tbl_gpu(df1)
  gy <- tbl_gpu(df2)

  oracle <- dplyr::union(df1, df2)

  withr::local_options(cuplyr.fallback = "error")
  result <- dplyr::union(gx, gy)
  result_df <- collect(result)
  expect_s3_class(result_df$a, "POSIXct")
  expect_equal(sort(as.numeric(result_df$a)), sort(as.numeric(oracle$a)))
})

test_that("NA rows count as matching for intersect()/setdiff()", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(a = c(1, NA, 3))
  df2 <- tibble::tibble(a = c(NA, 3))

  compare_set_op(df1, df2, dplyr::intersect, arrange_by = "a")
  compare_set_op(df1, df2, dplyr::setdiff)
})

test_that("0-row inputs work for every set op", {
  skip_if_no_gpu()
  df0 <- tibble::tibble(a = integer(), b = character())
  df1 <- tibble::tibble(a = 1L, b = "x")

  compare_set_op(df0, df1, dplyr::union)
  compare_set_op(df1, df0, dplyr::union)
  compare_set_op(df0, df0, dplyr::union)
  compare_set_op(df0, df1, dplyr::union_all)
  compare_set_op(df0, df1, dplyr::intersect)
  compare_set_op(df0, df1, dplyr::setdiff)
  compare_set_op(df1, df0, dplyr::setdiff)
  compare_set_op(df0, df1, dplyr::symdiff)

  expect_true(dplyr::setequal(tbl_gpu(df0), tbl_gpu(df0)))
  expect_false(dplyr::setequal(tbl_gpu(df0), tbl_gpu(df1)))
})

test_that("factor columns with identical levels are GPU-native (no fallback)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  df1 <- tibble::tibble(a = factor(c("a", "b"), levels = c("a", "b", "c")))
  df2 <- tibble::tibble(a = factor(c("b", "c"), levels = c("a", "b", "c")))
  gx <- tbl_gpu(df1)
  gy <- tbl_gpu(df2)

  result <- expect_no_error(dplyr::union(gx, gy) |> collect())
  expect_s3_class(result$a, "factor")
  expect_equal(levels(result$a), c("a", "b", "c"))
  expect_equal(sort(as.character(result$a)), c("a", "b", "c"))

  expect_no_error(dplyr::union_all(gx, gy))
  expect_no_error(dplyr::intersect(gx, gy))
  expect_no_error(dplyr::setdiff(gx, gy))
  expect_no_error(dplyr::symdiff(gx, gy))
  expect_no_error(dplyr::setequal(gx, gy))
})

test_that("factor columns with different levels are GPU-native as of Phase 11 L2 (level union)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(a = factor(c("a", "b")))
  df2 <- tibble::tibble(a = factor(c("b", "c")))
  gx <- tbl_gpu(df1)
  gy <- tbl_gpu(df2)

  oracle <- dplyr::union(df1, df2)

  # gpu_remap_codes() (src/ops_bind.cpp) makes this native now -- no fallback.
  withr::local_options(cuplyr.fallback = "error")
  result <- dplyr::union(gx, gy)
  result_df <- collect(result)

  expect_s3_class(result_df$a, "factor")
  expect_equal(levels(result_df$a), levels(oracle$a))
  expect_equal(sort(as.character(result_df$a)), sort(as.character(oracle$a)))
})

test_that("factor vs. character columns are GPU-native as of Phase 11 L2 (decode to character)", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(a = factor(c("a", "b")))
  df2 <- tibble::tibble(a = c("b", "c"))
  gx <- tbl_gpu(df1)
  gy <- tbl_gpu(df2)

  oracle <- dplyr::union(df1, df2)

  # gpu_decode_factor() (src/ops_bind.cpp) makes this native now -- no fallback.
  withr::local_options(cuplyr.fallback = "error")
  result <- dplyr::union(gx, gy)
  result_df <- collect(result)

  expect_type(result_df$a, "character")
  expect_equal(sort(result_df$a), sort(oracle$a))
})

test_that("factor vs. numeric columns is a genuine incompatible-type error", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(a = factor(c("a", "b"))))
  gy <- tbl_gpu(tibble::tibble(a = 1:2))

  expect_error(
    dplyr::union(gx, gy),
    "Incompatible types for column `a`: factor vs integer",
    fixed = TRUE
  )
})

test_that("set ops are GPU-native and never trigger a fallback notification (happy path)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gx <- tbl_gpu(tibble::tibble(a = 1:3, b = c("x", "y", "z")))
  gy <- tbl_gpu(tibble::tibble(a = c(2L, 4L), b = c("y", "w")))
  y_df <- tibble::tibble(a = c(2L, 4L), b = c("y", "w"))

  expect_no_error(dplyr::union(gx, gy))
  expect_no_error(dplyr::union_all(gx, gy))
  expect_no_error(dplyr::intersect(gx, gy))
  expect_no_error(dplyr::setdiff(gx, gy))
  expect_no_error(dplyr::symdiff(gx, gy))
  expect_no_error(dplyr::setequal(gx, gy))
  # A plain data.frame `y` (auto-uploaded) is native too.
  expect_no_error(dplyr::union(gx, y_df))
  # Lazy inputs stay native.
  expect_no_error(dplyr::union(as_lazy(gx), gy) |> collect())
  expect_no_error(dplyr::intersect(as_lazy(gx), as_lazy(gy)) |> collect())
})
