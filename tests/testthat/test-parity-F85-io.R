# F85: read_parquet_gpu() / write_parquet_gpu(): round trips, the type mapping
# on files arrow-R wrote (tests/testthat/fixtures/parquet), plan state, errors
# and previews.

schema_of <- function(x) {
  s <- current_schema(x)
  stats::setNames(s$types, s$names)
}

# --- round trips -------------------------------------------------------------

test_that("every type round-trips through write and read, with NA", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  path <- pq_write(df)
  for (lazy in c(TRUE, FALSE)) {
    got <- tibble::as_tibble(collect(read_parquet_gpu(path, lazy = lazy)))
    expect_identical(got, tibble::as_tibble(df))
  }
  expect_identical(
    unname(schema_of(read_parquet_gpu(path, lazy = TRUE))),
    c("INT32", "STRING", "STRING", "INT32", "FLOAT64", "BOOL8",
      "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS", "DICTIONARY32")
  )
})

test_that("factor levels, unused levels, ordered and POSIXct zones are kept", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    f = factor(c("b", NA, "a"), levels = c("z", "b", "a")),
    o = factor(c("lo", "hi", NA), levels = c("lo", "hi"), ordered = TRUE),
    t_utc = as.POSIXct(c("2021-01-01 10:00:00", NA, "2021-06-01 00:00:00"), tz = "UTC"),
    t_ny = as.POSIXct(c("2021-01-01 10:00:00", NA, "2021-06-01 00:00:00"), tz = "America/New_York")
  )
  path <- pq_write(df)
  got <- collect(read_parquet_gpu(path, lazy = FALSE))
  expect_identical(levels(got$f), c("z", "b", "a"))
  expect_false(is.ordered(got$f))
  expect_true(is.ordered(got$o))
  expect_identical(attr(got$t_ny, "tzone"), "America/New_York")
  expect_identical(attr(got$t_utc, "tzone"), "UTC")
  expect_identical(tibble::as_tibble(got), df)
})

test_that("a factor column is physically INT32 codes on the GPU, never a real dictionary", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  x <- read_parquet_gpu(path, lazy = FALSE)
  idx <- match("f", current_schema(x)$names) - 1L
  expect_identical(gpu_column_type(x$ptr, idx), "INT32")
})

test_that("integer64 round-trips exactly", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  v <- bit64::as.integer64(c("9007199254740993", NA, "-9007199254740993"))
  df <- tibble::tibble(b = v)
  path <- pq_write(df)
  x <- read_parquet_gpu(path, lazy = FALSE)
  expect_identical(unname(schema_of(x)), "INT64")
  expect_identical(as.character(collect(x)$b), as.character(v))
})

test_that("empty and one-row files keep the schema", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  for (d in list(df[0L, ], df[1L, ])) {
    path <- pq_write(d)
    for (lazy in c(TRUE, FALSE)) {
      got <- tibble::as_tibble(collect(read_parquet_gpu(path, lazy = lazy)))
      expect_identical(got, tibble::as_tibble(d))
    }
  }
})

test_that("write_parquet_gpu() computes a lazy plan, drops groups, returns its input", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  q <- tbl_gpu(df, lazy = TRUE) |> filter(i > 1L) |> select(id, i)
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  ret <- withVisible(write_parquet_gpu(q, path))
  expect_false(ret$visible)
  expect_true(identical(ret$value, q) || is_lazy(ret$value))
  expect_true(has_pending_ops(q))
  expect_identical(tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE))),
                   tibble::as_tibble(filter(df, i > 1L) |> select(id, i)))

  g <- group_by(tbl_gpu(df, lazy = FALSE), g)
  path2 <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path2))
  write_parquet_gpu(g, path2)
  back <- read_parquet_gpu(path2, lazy = FALSE)
  expect_identical(dplyr::group_vars(back), character())
  expect_identical(nrow(collect(back)), nrow(df))
})

test_that("compression options round-trip and a bad one is a classed error", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  for (cmp in c("snappy", "zstd", "none")) {
    path <- pq_write(df, compression = cmp)
    expect_identical(tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE))),
                     tibble::as_tibble(df), info = cmp)
  }
  path <- tempfile(fileext = ".parquet")
  expect_error(write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path, compression = "bogus"),
               class = "cuplyr_error_parquet")
  expect_false(file.exists(path))
})

test_that("writing to a directory that does not exist is a classed error naming the file", {
  skip_if_no_gpu()
  path <- file.path(tempfile("no-such-dir"), "x.parquet")
  expect_error(write_parquet_gpu(tbl_gpu(data.frame(a = 1L), lazy = FALSE), path),
               class = "cuplyr_error_parquet_write", regexp = "x.parquet", fixed = TRUE)
})

test_that("columns = reads only those columns, in the order of the file or as asked", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  path <- pq_write(df)
  for (lazy in c(TRUE, FALSE)) {
    got <- collect(read_parquet_gpu(path, columns = c("s", "id"), lazy = lazy))
    expect_named(got, c("s", "id"))
    expect_identical(got$id, df$id)
  }
  expect_error(read_parquet_gpu(path, columns = "nope"), class = "cuplyr_error_parquet")
})

test_that("a vector of paths reads the files in order; a different schema is a classed error", {
  skip_if_no_gpu()
  a <- pq_write(data.frame(x = 1:2, y = c("p", "q")))
  b <- pq_write(data.frame(x = 3:5, y = c("r", NA, "t")))
  c <- pq_write(data.frame(x = c(1.5, 2.5), y = c("u", "v")))
  for (lazy in c(TRUE, FALSE)) {
    got <- collect(read_parquet_gpu(c(a, b), lazy = lazy))
    expect_identical(got$x, 1:5)
    expect_identical(got$y, c("p", "q", "r", NA, "t"))
  }
  expect_error(read_parquet_gpu(c(a, c)), class = "cuplyr_error_parquet_schema_mismatch",
               regexp = basename(c), fixed = TRUE)
})

test_that("a vector of files with factor metadata keeps or refuses the levels, never mixes them", {
  skip_if_no_gpu()
  a <- pq_write(data.frame(f = factor(c("x", "y"))))
  b <- pq_write(data.frame(f = factor(c("y", "z"))))
  res <- tryCatch(collect(read_parquet_gpu(c(a, b), lazy = FALSE)), error = function(e) e)
  if (inherits(res, "error")) {
    expect_s3_class(res, "cuplyr_error_parquet")
  } else {
    expect_identical(as.character(res$f), c("x", "y", "y", "z"))
  }
})

# --- independent files: written by arrow-R, committed ------------------------

test_that("a file arrow-R wrote reads with the documented types", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_types.parquet")
  x <- read_parquet_gpu(path, lazy = FALSE)
  got <- schema_of(x)
  expect_identical(got[c("b", "i32", "i8", "i16", "u8", "u16", "u32", "u64", "i64", "f32", "f64",
                         "s", "d", "ts_s", "ts_ms", "ts_us", "ts_ns", "ts_ny", "ts_naive", "dec")],
                   c(b = "BOOL8", i32 = "INT32", i8 = "INT32", i16 = "INT32", u8 = "INT32", u16 = "INT32",
                     u32 = "FLOAT64", u64 = "INT64", i64 = "INT64", f32 = "FLOAT64", f64 = "FLOAT64",
                     s = "STRING", d = "TIMESTAMP_DAYS", ts_s = "TIMESTAMP_MICROSECONDS",
                     ts_ms = "TIMESTAMP_MICROSECONDS", ts_us = "TIMESTAMP_MICROSECONDS",
                     ts_ns = "TIMESTAMP_MICROSECONDS", ts_ny = "TIMESTAMP_MICROSECONDS",
                     ts_naive = "TIMESTAMP_MICROSECONDS", dec = "FLOAT64"))
})

test_that("values from an arrow-R file are exact, NA included", {
  skip_if_no_gpu()
  res <- tibble::as_tibble(collect(read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = FALSE)))
  ts0 <- as.POSIXct("2021-03-04 05:06:07", tz = "UTC")

  expect_identical(res$b, c(TRUE, NA, FALSE, TRUE))
  expect_identical(res$i32, c(1L, NA, -3L, 2147483647L))
  expect_identical(res$i8, c(1L, NA, -128L, 127L))
  expect_identical(res$i16, c(1L, NA, -32768L, 32767L))
  expect_identical(res$u8, c(0L, NA, 200L, 255L))
  expect_identical(res$u16, c(0L, NA, 40000L, 65535L))
  expect_identical(res$u32, c(0, NA, 3e9, 4294967295))
  expect_identical(res$f32, c(0.5, NA, 0.10000000149011612, -2.25))
  expect_identical(res$f64, c(1.5, NA, -0.25, 1e300))
  expect_identical(res$s, c("a", NA, "", "é"))
  expect_identical(res$d, as.Date("2020-01-01") + c(0, NA, 366, -1))
  expect_equal(as.numeric(res$ts_s), as.numeric(ts0) + c(0, NA, 1, 2))
  expect_equal(as.numeric(res$ts_ms), as.numeric(ts0) + c(0, NA, 0.25, 2))
  expect_equal(as.numeric(res$ts_us), as.numeric(ts0) + c(0, NA, 0.5, 2))
  expect_equal(as.numeric(res$ts_ns), as.numeric(ts0) + c(0, NA, 0.25, 2))
  expect_s3_class(res$ts_ns, "POSIXct")
  expect_identical(res$dec, c(1.23, NA, -0.3, 12345678.9))

  if (requireNamespace("bit64", quietly = TRUE)) {
    expect_s3_class(res$i64, "integer64")
    expect_s3_class(res$u64, "integer64")
    expect_identical(as.character(res$i64), c("1", NA, "-3", "9007199254740992"))
    expect_identical(as.character(res$u64), c("0", NA, "3000000000", "9007199254740992"))
  }
})

test_that("a time zone stored by arrow-R comes back as tzone", {
  skip_if_no_gpu()
  res <- collect(read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = FALSE))
  expect_identical(attr(res$ts_ny, "tzone"), "America/New_York")
  expect_identical(attr(res$ts_us, "tzone"), "UTC")
  # a timestamp with no zone (isAdjustedToUTC = false): no tzone, as in arrow-R
  expect_null(attr(res$ts_naive, "tzone"))
})

test_that("a factor written by arrow-R reads as a string column (its levels are not in the metadata)", {
  skip_if_no_gpu()
  x <- read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = FALSE)
  expect_identical(schema_of(x)[["fct"]], "STRING")
  expect_identical(schema_of(x)[["ord"]], "STRING")
  res <- collect(x)
  expect_identical(res$fct, c("b", NA, "a", "c"))
  expect_identical(res$ord, c("lo", "hi", NA, "lo"))
})

test_that("verbs work on every mapped type from an arrow-R file", {
  skip_if_no_gpu()
  df <- tibble::as_tibble(collect(read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = FALSE)))
  x <- read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = TRUE)
  got <- x |> select(i8, u16, f32, ts_ms, dec, fct) |>
    filter(i8 > 0L, f32 > 0) |>
    mutate(z = i8 + 1L) |>
    arrange(desc(z)) |>
    collect()
  want <- df |> select(i8, u16, f32, ts_ms, dec, fct) |>
    filter(i8 > 0L, f32 > 0) |>
    mutate(z = i8 + 1L) |>
    arrange(desc(z))
  expect_equal(tibble::as_tibble(got), want)
})

test_that("where() predicates see the collected class for every column of a scan", {
  skip_if_no_gpu()
  x <- read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = TRUE)
  df <- tibble::as_tibble(collect(read_parquet_gpu(pq_fixture("arrow_types.parquet"), lazy = FALSE)))
  expect_identical(names(select(x, where(is.numeric))), names(select(df, where(is.numeric))))
  expect_identical(names(select(x, where(is.character))), names(select(df, where(is.character))))
  expect_identical(names(select(x, where(is.factor))), names(select(df, where(is.factor))))
  expect_identical(names(select(x, where(~ inherits(.x, "POSIXct")))),
                   names(select(df, where(~ inherits(.x, "POSIXct")))))
  # an unsupported column is not numeric, and selecting around it reads fine
  n <- read_parquet_gpu(pq_fixture("arrow_nested.parquet"), lazy = TRUE)
  expect_identical(names(select(n, where(is.numeric))), "x")
})

test_that("a list or struct column is a classed error naming it, unless not read", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_nested.parquet")
  expect_error(read_parquet_gpu(path, lazy = FALSE),
               class = "cuplyr_error_parquet_unsupported_type", regexp = "\\bl\\b")
  expect_error(collect(read_parquet_gpu(path, lazy = TRUE)),
               class = "cuplyr_error_parquet_unsupported_type", regexp = basename(path), fixed = TRUE)
  expect_identical(collect(read_parquet_gpu(path, columns = "x", lazy = FALSE))$x, 1:3)
  expect_identical(collect(read_parquet_gpu(path, columns = "x", lazy = TRUE))$x, 1:3)
})

test_that("nanoseconds finer than a microsecond warn with a classed warning", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_ts_subus.parquet")
  expect_warning(res <- collect(read_parquet_gpu(path, lazy = FALSE)),
                 class = "cuplyr_warning_collect_precision")
  expect_s3_class(res$t, "POSIXct")
})

test_that("a UINT64 past INT64_MAX is a classed error naming the column", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_u64_big.parquet")
  expect_error(collect(read_parquet_gpu(path, lazy = FALSE)),
               class = "cuplyr_error_unsupported_gpu_type", regexp = "\\bu\\b")
  expect_identical(collect(read_parquet_gpu(path, columns = "id", lazy = FALSE))$id, 1:2)
})

# --- plan state ---------------------------------------------------------------

test_that("the schema is known from the footer without reading data", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  parquet_last_read(reset = TRUE)
  x <- read_parquet_gpu(path, lazy = TRUE)
  expect_null(parquet_last_read())
  expect_identical(current_schema(x)$names, names(pq_sem_df()))
  expect_identical(ncol(x), ncol(pq_sem_df()))
  expect_identical(nrow(x), 12L)
  expect_null(parquet_last_read())
})

test_that("a schema that changed after read_parquet_gpu() is a classed error at read", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  write_parquet_gpu(tbl_gpu(data.frame(a = 1:3, b = 4:6), lazy = FALSE), path)
  q <- read_parquet_gpu(path, lazy = TRUE)
  write_parquet_gpu(tbl_gpu(data.frame(a = c(1.5, 2.5)), lazy = FALSE), path)
  expect_error(collect(q), class = "cuplyr_error_parquet_schema_changed",
               regexp = basename(path), fixed = TRUE)
  unlink(path)
  expect_error(collect(q), class = "cuplyr_error_parquet_not_found",
               regexp = basename(path), fixed = TRUE)
})

test_that("new data with the same schema is read, as the docs say", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  write_parquet_gpu(tbl_gpu(data.frame(a = 1:3), lazy = FALSE), path)
  q <- read_parquet_gpu(path, lazy = TRUE)
  write_parquet_gpu(tbl_gpu(data.frame(a = 7:8), lazy = FALSE), path)
  expect_identical(collect(q)$a, 7:8)
})

test_that("lazy and eager reads agree, and tbl_gpu(df) is unchanged", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  path <- pq_write(df)
  expect_identical(tibble::as_tibble(collect(read_parquet_gpu(path, lazy = TRUE))),
                   tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE))))
  expect_identical(tibble::as_tibble(collect(tbl_gpu(df, lazy = FALSE))), tibble::as_tibble(df))
})

# --- errors -------------------------------------------------------------------

test_that("a missing file is a classed error naming it", {
  skip_if_no_gpu()
  path <- file.path(tempdir(), "definitely-missing.parquet")
  expect_error(read_parquet_gpu(path), class = "cuplyr_error_parquet_not_found",
               regexp = "definitely-missing.parquet", fixed = TRUE)
  expect_error(read_parquet_gpu(path), class = "cuplyr_error_parquet")
})

test_that("a file that isn't parquet is a classed error naming it", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  writeLines("this is not parquet", path)
  expect_error(read_parquet_gpu(path), class = "cuplyr_error_parquet_invalid",
               regexp = basename(path), fixed = TRUE)
})

test_that("a truncated file (corrupt footer) is a classed error naming it", {
  skip_if_no_gpu()
  src <- pq_write(pq_sem_df())
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  bytes <- readBin(src, "raw", n = file.size(src))
  writeBin(bytes[seq_len(length(bytes) %/% 2L)], path)
  expect_error(read_parquet_gpu(path), class = "cuplyr_error_parquet_invalid",
               regexp = basename(path), fixed = TRUE)
})

test_that("out of GPU memory during a read is the classed OOM error", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  withr::defer(gpu_oom_inject(0L))
  gpu_oom_inject(1000L)
  expect_error(read_parquet_gpu(path, lazy = FALSE), class = "cuplyr_error_gpu_oom")
  gpu_oom_inject(0L)
  expect_identical(nrow(collect(read_parquet_gpu(path, lazy = FALSE))), 12L)
})

# --- previews -----------------------------------------------------------------

test_that("print() and glimpse() of a bare scan read at most a preview", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = seq_len(5000L), v = seq_len(5000L) * 0.5)
  path <- pq_write(df, row_group_size = 100L)
  x <- read_parquet_gpu(path, lazy = TRUE)
  parquet_last_read(reset = TRUE)
  out <- capture.output(print(x))
  expect_lte(parquet_last_read()$rows_read, 100)
  expect_true(any(grepl("5,?000", out)))

  parquet_last_read(reset = TRUE)
  out <- capture.output(glimpse(x))
  expect_lte(parquet_last_read()$rows_read, 100)
  expect_true(any(grepl("5,?000", out)))

  parquet_last_read(reset = TRUE)
  invisible(capture.output(print(select(x, id))))
  expect_lte(parquet_last_read()$rows_read, 100)
})

test_that("print() and glimpse() of any other plan over a file don't read it", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = seq_len(5000L), v = seq_len(5000L) * 0.5)
  path <- pq_write(df, row_group_size = 100L)
  q <- filter(read_parquet_gpu(path, lazy = TRUE), id > 2L)
  parquet_last_read(reset = TRUE)
  out <- capture.output(print(q))
  expect_null(parquet_last_read())
  expect_true(any(grepl("id", out)))
  out <- capture.output(glimpse(q))
  expect_null(parquet_last_read())
  expect_true(any(grepl("id", out)))
})

# --- added after round 1's review (B1-B6, B8, B9, N3) --------------------------

test_that("a pandas index column does not reorder or refuse columns =", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_pandas_index.parquet")
  for (lazy in c(TRUE, FALSE)) {
    got <- collect(read_parquet_gpu(path, columns = c("b", "id"), lazy = lazy))
    expect_named(got, c("b", "id"))
    expect_identical(got$b, 11:13)
    expect_identical(got$id, 1:3)
    expect_identical(collect(read_parquet_gpu(path, columns = "b", lazy = lazy))$b, 11:13)
    all <- collect(read_parquet_gpu(path, lazy = lazy))
    expect_named(all, c("id", "b"))
    expect_identical(all$id, 1:3)
  }
})

test_that("a stored value equal to R's NA reads as missing on the GPU too", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_r_na_sentinels.parquet")
  for (lazy in c(TRUE, FALSE)) {
    x <- read_parquet_gpu(path, lazy = lazy)
    expect_identical(collect(x)$i, c(NA, 1L, NA, -1L))
    expect_identical(collect(filter(x, !is.na(i)))$id, c(2L, 4L))
    expect_identical(collect(filter(x, i < 0L))$id, 4L)
    s <- collect(summarise(x, s = sum(i, na.rm = TRUE), nas = sum(is.na(i))))
    expect_equal(s$s, 0)
    expect_equal(s$nas, 2)
  }
  skip_if_not_installed("bit64")
  for (lazy in c(TRUE, FALSE)) {
    x <- read_parquet_gpu(path, lazy = lazy)
    expect_identical(as.character(collect(x)$b), c(NA, "1", NA, "-1"))
    expect_identical(collect(filter(x, !is.na(b)))$id, c(2L, 4L))
    expect_identical(collect(filter(x, b < 0L))$id, 4L)
  }
})

test_that("a failed write leaves the file it was overwriting", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  withr::defer({Sys.chmod(path, "644"); unlink(path)})
  write_parquet_gpu(tbl_gpu(data.frame(x = 1:3), lazy = FALSE), path)
  before <- readBin(path, "raw", n = file.size(path))

  expect_error(write_parquet_gpu(tbl_gpu(data.frame(x = 9L), lazy = FALSE), path, compression = "bogus"),
               class = "cuplyr_error_parquet")
  expect_identical(readBin(path, "raw", n = file.size(path)), before)

  skip_if(identical(Sys.info()[["effective_user"]], "root"), "root ignores file modes")
  Sys.chmod(dirname(path), "755")
  Sys.chmod(path, "444")
  res <- tryCatch(write_parquet_gpu(tbl_gpu(data.frame(x = 9L), lazy = FALSE), path), error = function(e) e)
  expect_true(file.exists(path))
  if (inherits(res, "error")) {
    expect_s3_class(res, "cuplyr_error_parquet_write")
    expect_match(conditionMessage(res), basename(path), fixed = TRUE)
    expect_identical(readBin(path, "raw", n = file.size(path)), before)
  } else {
    # a write through a temporary and a rename may legitimately succeed here
    expect_identical(collect(read_parquet_gpu(path, lazy = FALSE))$x, 9L)
  }
})

test_that("a successful overwrite replaces the file and leaves no temporary behind", {
  skip_if_no_gpu()
  d <- withr::local_tempdir()
  path <- file.path(d, "a.parquet")
  write_parquet_gpu(tbl_gpu(data.frame(x = 1:3), lazy = FALSE), path)
  write_parquet_gpu(tbl_gpu(data.frame(x = 7:8), lazy = FALSE), path)
  expect_identical(collect(read_parquet_gpu(path, lazy = FALSE))$x, 7:8)
  expect_identical(list.files(d, all.files = TRUE, no.. = TRUE), "a.parquet")
})

test_that("nrow() of a scan whose file is gone is the classed error", {
  skip_if_no_gpu()
  path <- tempfile(fileext = ".parquet")
  write_parquet_gpu(tbl_gpu(data.frame(x = 1:3), lazy = FALSE), path)
  g <- read_parquet_gpu(path, lazy = TRUE)
  unlink(path)
  expect_error(nrow(g), class = "cuplyr_error_parquet_not_found",
               regexp = basename(path), fixed = TRUE)
  expect_identical(ncol(g), 1L)
})

test_that("out of GPU memory during a read names the file, eager and lazy", {
  skip_if_no_gpu()
  path <- pq_write(data.frame(x = 1:1000))
  withr::defer(gpu_oom_inject(0L))
  for (lazy in c(FALSE, TRUE)) {
    gpu_oom_inject(1000L)
    err <- tryCatch(collect(read_parquet_gpu(path, lazy = lazy)), error = function(e) e)
    gpu_oom_inject(0L)
    expect_s3_class(err, "cuplyr_error_gpu_oom")
    expect_match(conditionMessage(err), basename(path), fixed = TRUE)
  }
})

test_that("print() and glimpse() do not read a file whose first row group is large", {
  skip_if_no_gpu()
  n <- 200000L
  df <- data.frame(id = seq_len(n), v = seq_len(n) * 0.5)
  path <- pq_write(df, row_group_size = n)
  x <- read_parquet_gpu(path, lazy = TRUE)
  parquet_last_read(reset = TRUE)
  out <- capture.output(print(x))
  expect_null(parquet_last_read())
  expect_true(any(grepl("id", out)))
  out <- capture.output(glimpse(x))
  expect_null(parquet_last_read())
  expect_true(any(grepl("id", out)))
})

test_that("a NUL byte in the footer's arrow schema means no information, not an error", {
  skip_if_no_gpu()
  skip_if_not_installed("jsonlite")
  f <- pq_fixture("arrow_ts_subus.parquet")
  b64 <- gpu_parquet_probe(f)[[1L]]$kv[["ARROW:schema"]]
  r <- jsonlite::base64_dec(b64)
  for (at in seq(40L, length(r))) {
    bad <- r
    bad[at] <- as.raw(0L)
    expect_no_error(parquet_arrow_tz(jsonlite::base64_enc(bad)))
  }
})

test_that("a factor with an NA level warns at write, naming the column", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = addNA(factor(c("a", NA, "b"))), n = 1:3)
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path))
  expect_warning(write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path),
                 class = "cuplyr_warning_parquet_factor_na_level", regexp = "\\bf\\b")
  res <- collect(read_parquet_gpu(path, lazy = FALSE))
  expect_identical(as.character(res$f), c("a", NA, "b"))
})

test_that("nanoseconds round to the nearest microsecond and a wide decimal is the nearest double", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_precision.parquet")
  w <- NULL
  res <- withCallingHandlers(
    collect(read_parquet_gpu(path, lazy = FALSE)),
    warning = function(x) {
      w[[length(w) + 1L]] <<- x
      invokeRestart("muffleWarning")
    }
  )
  expect_true(any(vapply(w, inherits, NA, "cuplyr_warning_collect_precision")))
  base <- as.numeric(as.POSIXct("2021-01-01", tz = "UTC"))
  expect_equal(as.numeric(res$t)[[1L]], base + 1e-6, tolerance = 1e-12)
  expect_equal(as.numeric(res$t)[[2L]], -1e-6, tolerance = 1e-12)
  expect_lte(abs(as.numeric(res$t)[[3L]] - base), 1e-6 + 1e-9)
  expect_identical(res$dec[[1L]], 123456789012345678.91)
  expect_identical(res$dec[[2L]], 1.1)
  expect_true(is.na(res$dec[[3L]]))
})
