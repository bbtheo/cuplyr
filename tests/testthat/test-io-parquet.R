# F85 (cuplyr's own checks, beside the PM's acceptance files): footer encodings,
# odd labels and zones, the plan display, dim().

test_that("netstrings round-trip any label and reject anything malformed", {
  x <- c("a", "", "b:c", "3:ab", "line\nbreak", "é中")
  expect_identical(parquet_decode_netstrings(parquet_encode_netstrings(x)), x)
  expect_identical(parquet_decode_netstrings(""), character())
  for (bad in c("3:ab", "99:ab", "x:ab", ":ab", "2ab", "1:a2", "-1:a", "1234567890:a")) {
    expect_null(parquet_decode_netstrings(bad), info = bad)
  }
})

test_that("an ARROW:schema value that is not valid gives no information, never an error", {
  for (bad in c("", "AAAA", "////", "not base64 !!", strrep("A", 64), "/////wAAAAA=")) {
    res <- parquet_arrow_tz(bad)
    expect_identical(res$names, character(), info = bad)
  }
})

test_that("labels with any character, ordered factors and local-time zones survive a file", {
  skip_if_no_gpu()
  lv <- c("b:c", "", "3:ab", "line\nbreak", "é中", "z")
  df <- tibble::tibble(
    f = factor(c(lv[1:5], NA, "z"), levels = lv, ordered = TRUE),
    t_local = as.POSIXct(c("2021-01-01 10:00:00", NA, "2021-06-01 00:00:00", rep(NA, 4)), tz = ""),
    t_none = structure(c(0, 1, NA, 3, 4, 5, 6), class = c("POSIXct", "POSIXt")),
    t_plus = as.POSIXct(rep("2021-01-01 10:00:00", 7), tz = "Etc/GMT+5")
  )
  path <- pq_write(df)
  got <- collect(read_parquet_gpu(path, lazy = FALSE))
  expect_identical(tibble::as_tibble(got), df)
})

test_that("a factor with an NA level is written as strings, never mislabelled", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = addNA(factor(c("a", NA, "b"))))
  expect_warning(path <- pq_write(df), class = "cuplyr_warning_parquet_factor_na_level")
  got <- collect(read_parquet_gpu(path, lazy = FALSE))
  expect_identical(as.character(got$f), c("a", NA, "b"))
})

test_that("a grouped row_number() filter reads like dplyr", {
  skip_if_no_gpu()
  expect_parquet_same_as_dplyr(
    pq_sem_df(), function(x) filter(group_by(x, g), dplyr::row_number() == 1L))
})

test_that("explain() names the file and the columns of a scan", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  out <- capture.output(explain(select(read_parquet_gpu(path), id, s)))
  expect_true(any(grepl(paste0("parquet[", basename(path), "; cols="), out, fixed = TRUE)))
})

test_that("print() shows the row count of the file, not of the preview", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = seq_len(3000L))
  path <- pq_write(df, row_group_size = 500L)
  out <- capture.output(print(read_parquet_gpu(path)))
  expect_true(any(grepl("3,?000 . 1", out)))
  expect_length(grep("^\\$ id", out), 1L)
})

test_that("dim() of a bare scan follows the footer when only the data changed", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  write_parquet_gpu(tbl_gpu(data.frame(a = 1:3), lazy = FALSE), path)
  q <- read_parquet_gpu(path)
  expect_identical(dim(q), c(3L, 1L))
  write_parquet_gpu(tbl_gpu(data.frame(a = 1:5), lazy = FALSE), path)
  expect_identical(dim(q), c(5L, 1L))
  expect_identical(nrow(read_parquet_gpu(c(path, path))), 10L)
})

test_that("a scan under a join prints its schema without reading", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  j <- left_join(read_parquet_gpu(path), tbl_gpu(data.frame(id = 1:2, w = c(1, 2))), by = "id")
  parquet_last_read(reset = TRUE)
  out <- capture.output(print(j))
  expect_null(parquet_last_read())
  expect_true(any(grepl("not materialized", out)))
})

test_that("write_parquet_gpu() refuses what it cannot name", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  dup <- tbl_gpu(data.frame(a = 1:2, b = 3:4), lazy = FALSE)
  names(dup) <- c("a", "a")
  expect_error(write_parquet_gpu(dup, path), class = "cuplyr_error_parquet_write")
  expect_false(file.exists(path))
  expect_error(write_parquet_gpu(data.frame(a = 1), path), class = "cuplyr_error_parquet")
})

test_that("columns = character() is refused at the call, naming the file", {
  skip_if_no_gpu()
  path <- pq_write(data.frame(a = 1:3))
  expect_error(read_parquet_gpu(path, columns = character()),
               class = "cuplyr_error_parquet_columns", regexp = basename(path), fixed = TRUE)
})

test_that("an error about several files says which ones it could be", {
  skip_if_no_gpu()
  a <- pq_write(data.frame(x = 1:2))
  b <- pq_write(data.frame(x = 3:4))
  expect_match(parquet_files_label(c(a, b)), "one of the parquet files", fixed = TRUE)
  expect_match(parquet_files_label(a), basename(a), fixed = TRUE)
  expect_identical(collect(read_parquet_gpu(c(a, b), lazy = FALSE))$x, 1:4)
})

test_that("a file that is not writable is refused and left alone", {
  skip_if_no_gpu()
  skip_if(identical(Sys.info()[["effective_user"]], "root"), "root ignores file modes")
  path <- tempfile(fileext = ".parquet")
  withr::defer({Sys.chmod(path, "644"); unlink(path)})
  write_parquet_gpu(tbl_gpu(data.frame(x = 1:3), lazy = FALSE), path)
  Sys.chmod(path, "444")
  expect_error(write_parquet_gpu(tbl_gpu(data.frame(x = 9L), lazy = FALSE), path),
               class = "cuplyr_error_parquet_write", regexp = basename(path), fixed = TRUE)
  expect_identical(collect(read_parquet_gpu(path, lazy = FALSE))$x, 1:3)
})

test_that("a preview of a file with columns R cannot hold shows a placeholder, not a failed plan", {
  skip_if_no_gpu()
  x <- read_parquet_gpu(pq_fixture("arrow_nested.parquet"), lazy = TRUE)
  out <- capture.output(print(x))
  expect_false(any(grepl("Plan failed", out)))
  expect_true(any(grepl("cannot be shown in R", out)))
  out <- capture.output(glimpse(x))
  expect_false(any(grepl("Plan failed", out)))
  # the data itself is still refused
  expect_error(collect(x), class = "cuplyr_error_parquet_unsupported_type")
})
