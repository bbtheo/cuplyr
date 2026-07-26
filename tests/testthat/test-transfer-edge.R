# Tests for transfer-layer hygiene on zero-length data (P3, unification_design.md Part C)
#
# These exercise src/transfer_io.cpp host<->device transfer functions with
# n == 0 for every supported column type: numeric_to_gpu, integer_to_gpu,
# character_to_gpu, logical_to_gpu, date_to_gpu, posixct_to_gpu, and the
# factor path (stored as INT32 codes). Each type is round-tripped via
# tbl_gpu() -> collect(), checked with dim(), and exercised with a simple
# op (filter) to make sure the zero-row path doesn't crash or corrupt data
# through gpu_head()/gpu_collect() (including the FLOAT32/INT64/
# TIMESTAMP_DAYS/TIMESTAMP_MICROSECONDS branches which historically lacked
# an nrow > 0 guard before their cudaMemcpy calls).

# =============================================================================
# Per-type zero-row round trips
# =============================================================================

test_that("zero-row numeric column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = numeric(0))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_type(result$x, "double")

  filtered <- dplyr::filter(gpu_df, x > 0)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row integer column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = integer(0))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_type(result$x, "integer")

  filtered <- dplyr::filter(gpu_df, x > 0L)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row character column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = character(0), stringsAsFactors = FALSE)
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_type(result$x, "character")

  filtered <- dplyr::filter(gpu_df, TRUE)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row logical column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = logical(0))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_type(result$x, "logical")

  filtered <- dplyr::filter(gpu_df, TRUE)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row Date column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = as.Date(character(0)))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_s3_class(result$x, "Date")

  filtered <- dplyr::filter(gpu_df, TRUE)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row POSIXct column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = as.POSIXct(character(0), tz = "UTC"))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_s3_class(result$x, "POSIXct")

  filtered <- dplyr::filter(gpu_df, TRUE)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row factor column round-trips", {
  skip_if_no_gpu()

  df <- data.frame(x = factor(character(0), levels = c("a", "b", "c")))
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 1L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_s3_class(result$x, "factor")
  expect_equal(levels(result$x), c("a", "b", "c"))

  filtered <- dplyr::filter(gpu_df, TRUE)
  expect_equal(nrow(collect(filtered)), 0)
})

test_that("zero-row table with all supported types round-trips together", {
  skip_if_no_gpu()

  df <- data.frame(
    n = numeric(0),
    i = integer(0),
    c = character(0),
    l = logical(0),
    d = as.Date(character(0)),
    p = as.POSIXct(character(0), tz = "UTC"),
    stringsAsFactors = FALSE
  )
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 6L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_equal(ncol(result), 6)
  expect_equal(names(result), c("n", "i", "c", "l", "d", "p"))

  # print() exercises gpu_head() on the zero-row table (used by print.tbl_gpu)
  expect_no_error(print(gpu_df))

  filtered <- dplyr::filter(gpu_df, n > 0)
  expect_equal(nrow(collect(filtered)), 0)
})

# =============================================================================
# Zero-column data frame
# =============================================================================

test_that("tbl_gpu() handles a zero-column data frame", {
  skip_if_no_gpu()

  df <- data.frame()
  gpu_df <- tbl_gpu(df)

  expect_valid_tbl_gpu(gpu_df)
  expect_equal(dim(gpu_df), c(0L, 0L))

  result <- collect(gpu_df)
  expect_equal(nrow(result), 0)
  expect_equal(ncol(result), 0)
})
