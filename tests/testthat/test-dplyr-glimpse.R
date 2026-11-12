# Tests for glimpse() (Phase 3, task 4) -- GPU-native, R/glimpse.R.
#
# Unlike most of this file's siblings, this is NOT an oracle-comparison
# suite (expect_same_as_dplyr()): dplyr's own glimpse() formats through
# pillar, and cuplyr's bounded-preview implementation deliberately doesn't
# try to reproduce that formatting byte-for-byte (see glimpse.tbl_gpu()'s
# roxygen "Details" in R/glimpse.R). Instead these tests pin the structural
# facts via `expect_output()` regexes (row/column counts, grouping, column
# names/types) plus a direct proof that no full gpu_collect() ever runs.

glimpse_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2),
    x = c(10L, 20L, 30L),
    y = c(1.5, 2.5, 3.5),
    s = c("a", "b", "c")
  )
}

test_that("glimpse() prints Rows/Columns and returns its input invisibly", {
  skip_if_no_gpu()
  gt <- tbl_gpu(glimpse_df())

  out <- capture.output(result <- dplyr::glimpse(gt))

  expect_true(any(grepl("^Rows: 3$", out)))
  expect_true(any(grepl("^Columns: 4$", out)))
  expect_identical(result, gt)
})

test_that("glimpse() prints every column name and its type", {
  skip_if_no_gpu()
  gt <- tbl_gpu(glimpse_df())

  out <- capture.output(dplyr::glimpse(gt))

  # gpu_col_types() (also used by print.tbl_gpu()) reports R-style
  # abbreviations (<dbl>/<int>/<chr>), matching dplyr's own glimpse() type
  # labels -- not the GPU-side type strings (FLOAT64/INT32/STRING).
  expect_true(any(grepl("^\\$ g\\s+<dbl>", out)))
  expect_true(any(grepl("^\\$ x\\s+<int>", out)))
  expect_true(any(grepl("^\\$ y\\s+<dbl>", out)))
  expect_true(any(grepl("^\\$ s\\s+<chr>", out)))
})

test_that("glimpse() prints a Groups line for a grouped table", {
  skip_if_no_gpu()
  gt <- tbl_gpu(glimpse_df()) |> dplyr::group_by(g)

  out <- capture.output(dplyr::glimpse(gt))

  expect_true(any(grepl("^Groups: g \\[2\\]$", out)))
})

test_that("glimpse() prints no Groups line for an ungrouped table", {
  skip_if_no_gpu()
  gt <- tbl_gpu(glimpse_df())

  out <- capture.output(dplyr::glimpse(gt))

  expect_false(any(grepl("^Groups:", out)))
})

test_that("glimpse() materializes pending lazy ops before reporting counts", {
  skip_if_no_gpu()
  lazy_gt <- tbl_gpu(glimpse_df(), lazy = TRUE) |> dplyr::filter(g == 1)

  out <- capture.output(dplyr::glimpse(lazy_gt))

  expect_true(any(grepl("^Rows: 2$", out)))
  expect_true(any(grepl("^Columns: 4$", out)))
})

test_that("glimpse() shows values from the actual data, not placeholders", {
  skip_if_no_gpu()
  gt <- tbl_gpu(glimpse_df())

  out <- capture.output(dplyr::glimpse(gt))
  s_line <- out[grepl("\\$ s\\b", out)]

  expect_true(length(s_line) == 1)
  expect_true(grepl('"a"', s_line))
  expect_true(grepl('"b"', s_line))
  expect_true(grepl('"c"', s_line))
})

test_that("glimpse() never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(glimpse_df())

  expect_no_error(capture.output(dplyr::glimpse(gt)))
})

test_that("glimpse() never calls gpu_collect() (no full-table collect)", {
  skip_if_no_gpu()
  gt <- tbl_gpu(glimpse_df())

  testthat::local_mocked_bindings(
    gpu_collect = function(...) stop("gpu_collect() should never be called by glimpse()"),
    .package = "cuplyr"
  )

  expect_no_error(capture.output(dplyr::glimpse(gt)))

  # Sanity check the mock is actually wired up: collect() SHOULD hit it.
  expect_error(collect(gt), "gpu_collect\\(\\) should never be called")
})

test_that("glimpse() stays fast on a large lazy table (no full materialize of preview data)", {
  skip_if_no_gpu()
  big <- tbl_gpu(data.frame(x = seq_len(2e6)))
  lazy_big <- as_lazy(big) |> dplyr::filter(x > 10)

  elapsed <- system.time(capture.output(dplyr::glimpse(lazy_big)))["elapsed"]

  expect_lt(unname(elapsed), 5)
})
