# F85 round 3: an `==` pushed into the reader is decided by the Bloom filter when
# the file has one, and a Bloom filter holds a value's bits. The fixture was
# written by DuckDB (which writes Bloom filters; libcudf's writer doesn't by
# default): each of its 8 row groups holds a random subset of each domain, so a
# literal is often within a row group's min/max but absent from it, and only the
# Bloom filter can drop that group.

bloom_fixture <- function() test_path("fixtures", "parquet-bloom", "duckdb_bloom_types.parquet")

test_that("every `==` literal keeps every row the Bloom filter is asked about", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  path <- bloom_fixture()
  oracle <- tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE)))
  scan <- read_parquet_gpu(path, lazy = TRUE)

  # as an R value of the literal's type, or the expression to compare with
  cases <- list(
    list("i", -3L), list("i", 0L), list("i", 1L), list("i", 2L), list("i", 7L), list("i", 1000L),
    list("i64", bit64::as.integer64("2147483648")), list("i64", bit64::as.integer64("1099511627776")),
    list("i64", bit64::as.integer64("9007199254740991")), list("i64", bit64::as.integer64("1")),
    list("i64", bit64::as.integer64("-5")),
    list("d", 0.5), list("d", 1), list("d", 2), list("d", 1e300), list("d", 0.1), list("d", -3),
    list("s", ""), list("s", "a"), list("s", "A"), list("s", " a"), list("s", "a "),
    list("s", "é"), list("s", "é"), list("s", "日本語"),
    list("dt", as.Date("2020-01-01")), list("dt", as.Date("2020-02-01")), list("dt", as.Date("2021-02-04")),
    list("ts", as.POSIXct("2021-03-04 05:06:07", tz = "UTC")), list("ts", as.POSIXct("2021-03-04 06:06:07", tz = "UTC")),
    list("ts", as.POSIXct("2021-03-05 05:06:07", tz = "UTC"))
  )

  dropped <- 0L
  for (case in cases) {
    col <- case[[1L]]
    lit <- case[[2L]]
    want <- oracle$id[which(oracle[[col]] == lit)]
    pred <- rlang::expr(!!rlang::sym(col) == !!lit)
    r <- pq_run(filter(scan, !!pred))
    # (a comparison on a double or string column stays above the scan)
    expect_length(r$read$predicates, if (col %in% c("d", "s")) 0L else 1L)
    expect_identical(sort(r$result$id), sort(want), info = deparse(pred))
    bloom <- r$read$row_groups_after_bloom
    stats <- r$read$row_groups_after_stats
    if (!is.na(bloom) && !is.na(stats) && bloom < stats) dropped <- dropped + 1L
  }
  # the Bloom stage really runs on this file, so this test can catch a wrong probe
  expect_gt(dropped, 0L)
})

test_that("an integer-valued literal finds a double column's values", {
  skip_if_no_gpu()
  path <- bloom_fixture()
  oracle <- tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE)))
  scan <- read_parquet_gpu(path, lazy = TRUE)
  for (pipe in list(function(x) filter(x, d == 2L), function(x) filter(x, 2L == d))) {
    r <- pq_run(pipe(scan))
    expect_length(r$read$predicates, 0L)
    expect_identical(sort(r$result$id), oracle$id[which(oracle$d == 2)])
  }
})

test_that("zero stays above the scan for `==` and for ordering", {
  skip_if_no_gpu()
  path <- pq_fixture("duckdb_zero_bloom.parquet")
  scan <- read_parquet_gpu(path, lazy = TRUE)
  zero <- 0
  neg_zero <- -0
  for (pipe in list(function(x) filter(x, d == 0), function(x) filter(x, 0 == d),
                    function(x) filter(x, d == 0L), function(x) filter(x, d == -0),
                    function(x) filter(x, d == zero), function(x) filter(x, d == neg_zero))) {
    r <- pq_run(pipe(scan))
    expect_length(r$read$predicates, 0L)
    expect_identical(nrow(r$result), 1536L)
  }
  # ordering is numeric, so both zeros count
  r <- pq_run(filter(scan, d >= 0))
  expect_length(r$read$predicates, 0L)
  expect_identical(nrow(r$result), nrow(collect(filter(read_parquet_gpu(path, lazy = FALSE), d >= 0))))
})

test_that("a column read through a cast is not given to the reader", {
  skip_if_no_gpu()
  # same instants, millisecond units: the scan's file type is not the schema's
  path <- bloom_fixture()
  scan <- read_parquet_gpu(path, lazy = TRUE)
  t0 <- as.POSIXct("2021-03-04 05:06:07", tz = "UTC")
  r <- pq_run(filter(scan, ts_ms == t0))
  expect_length(r$read$predicates, 0L)
  oracle <- tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE)))
  expect_identical(sort(r$result$id), oracle$id[which(as.numeric(oracle$ts_ms) == as.numeric(t0))])
})

test_that("is.na() stays above the scan, and !is.na() of an integer column is pushed", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = 1:4, i = c(1L, NA, 3L, NA), s = c("a", NA, "b", NA))
  path <- pq_write(df)
  scan <- read_parquet_gpu(path, lazy = TRUE)
  for (pipe in list(function(x) filter(x, is.na(s)), function(x) filter(x, !is.na(s)),
                    function(x) filter(x, is.na(i)))) {
    r <- pq_run(pipe(scan))
    expect_length(r$read$predicates, 0L)
    expect_equal(tibble::as_tibble(r$result), pipe(df))
  }
  r <- pq_run(filter(scan, !is.na(i)))
  expect_length(r$read$predicates, 1L)
  expect_equal(r$result$id, c(1L, 3L))
})
