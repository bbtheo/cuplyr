# F85 round 3: the review's non-blocking notes that change behaviour.

test_that("a read that fails with a pushed filter names the filter", {
  skip_if_no_gpu()
  path <- pq_write(tibble::tibble(i = 1:3, s = c("a", "b", "c")))
  q <- filter(read_parquet_gpu(path, lazy = TRUE), i > 1L, s == "b")
  local_mocked_bindings(gpu_parquet_read = function(...) stop("reader says no"))
  err <- tryCatch(collect(q), error = function(e) e)
  expect_s3_class(err, "cuplyr_error_parquet_invalid")
  msg <- conditionMessage(err)
  expect_match(msg, basename(path), fixed = TRUE)
  expect_match(msg, "reader says no", fixed = TRUE)
  expect_match(msg, "i > 1", fixed = TRUE)
  expect_match(msg, "cuplyr.parquet_pushdown", fixed = TRUE)
})

test_that("a read that fails without a pushed filter doesn't mention one", {
  skip_if_no_gpu()
  path <- pq_write(tibble::tibble(i = 1:3))
  q <- read_parquet_gpu(path, lazy = TRUE)
  local_mocked_bindings(gpu_parquet_read = function(...) stop("reader says no"))
  err <- tryCatch(collect(q), error = function(e) e)
  expect_s3_class(err, "cuplyr_error_parquet_invalid")
  expect_false(grepl("cuplyr.parquet_pushdown", conditionMessage(err), fixed = TRUE))
})

test_that("a first row group with too much column data is not read for a preview", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = 1:1000, v = seq_len(1000L) * 0.5)
  path <- pq_write(df)
  x <- read_parquet_gpu(path, lazy = TRUE)

  # small row group: previewed
  parquet_last_read(reset = TRUE)
  capture.output(print(x))
  expect_false(is.null(parquet_last_read()))

  # the same file, with the byte limit below the size of its row group
  local_mocked_bindings(scan_preview_max_bytes = 16)
  parquet_last_read(reset = TRUE)
  out <- capture.output(print(x))
  expect_null(parquet_last_read())
  expect_true(any(grepl("id", out)))
})

test_that("files libcudf won't read together raise the mismatch error, naming them", {
  skip_if_no_gpu()
  # same column names and types, but libcudf compares the footers' schemas
  # element by element, and these two writers' differ
  duck <- pq_fixture("duckdb_zero_bloom.parquet")
  ours <- pq_write(collect(read_parquet_gpu(duck, lazy = FALSE)))
  for (lazy in c(TRUE, FALSE)) {
    err <- tryCatch(collect(read_parquet_gpu(c(duck, ours), lazy = lazy)), error = function(e) e)
    expect_s3_class(err, "cuplyr_error_parquet_schema_mismatch")
    expect_s3_class(err, "cuplyr_error_parquet")
    expect_match(conditionMessage(err), basename(duck), fixed = TRUE)
    expect_match(conditionMessage(err), basename(ours), fixed = TRUE)
    expect_identical(basename(err$path), basename(c(duck, ours)))
  }
})
