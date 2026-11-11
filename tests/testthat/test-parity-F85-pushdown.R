# F85: pushdown into the scan is visible where it must happen, absent where it
# must not. Results are covered by test-parity-F85-semantics.R.

test_that("select() and what the plan uses become the scan's columns", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  scan <- read_parquet_gpu(path, lazy = TRUE)

  r <- pq_run(select(scan, id, i))
  expect_setequal(r$read$columns, c("id", "i"))
  expect_named(r$result, c("id", "i"))

  r <- pq_run(summarise(group_by(scan, g), m = sum(i, na.rm = TRUE)))
  expect_setequal(r$read$columns, c("g", "i"))

  r <- pq_run(select(mutate(scan, z = id * 2L), z))
  expect_setequal(r$read$columns, "id")

  # a plan that reads no column still reads one, for its row count
  r <- pq_run(summarise(scan, n = n()))
  expect_length(r$read$columns, 1L)
  expect_equal(r$result$n, 12L)

  r <- pq_run(scan)
  expect_setequal(r$read$columns, names(pq_sem_df()))
})

test_that("column order of the result never follows the scan's order", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  scan <- read_parquet_gpu(path, lazy = TRUE)
  r <- pq_run(select(scan, ts, id, s))
  expect_named(r$result, c("ts", "id", "s"))
  expect_equal(r$result$id, 1:12)
})

test_that("a filter directly on the scan is pushed: every convertible type", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  scan <- read_parquet_gpu(path, lazy = TRUE)
  cases <- list(
    int = list(filter(scan, i > 1L), "\\bi\\b"),
    int_dbl_lit = list(filter(scan, i > 1), "\\bi\\b"),
    date = list(filter(scan, dt >= as.Date("2020-01-05")), "\\bdt\\b"),
    ts = list(filter(scan, ts >= as.POSIXct("2021-03-04 05:06:10", tz = "UTC")), "\\bts\\b")
  )
  for (nm in names(cases)) {
    r <- pq_run(cases[[nm]][[1L]])
    expect_length(r$read$predicates, 1L)
    expect_match(r$read$predicates, cases[[nm]][[2L]], info = nm)
  }
})

test_that("the AND is split: convertible parts pushed, the rest applied after", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  scan <- read_parquet_gpu(path, lazy = TRUE)

  r <- pq_run(filter(scan, i > 1L, id %% 2L == 0L))
  expect_length(r$read$predicates, 1L)
  expect_match(r$read$predicates, "\\bi\\b")
  expect_equal(r$result$id, filter(pq_sem_df(), i > 1L, id %% 2L == 0L)$id)

  # (a double comparison stays above the scan: libcudf prunes on a NaN statistic)
  r <- pq_run(filter(scan, i > 1L & id %% 2L == 0L & d > 0))
  expect_length(r$read$predicates, 1L)
  expect_match(r$read$predicates, "\\bi\\b")
  expect_equal(r$result$id, filter(pq_sem_df(), i > 1L & id %% 2L == 0L & d > 0)$id)

  # a filter column the output does not keep is still applied and not returned
  r <- pq_run(select(filter(scan, i > 1L), id))
  expect_length(r$read$predicates, 1L)
  expect_false("d" %in% r$read$columns)
  expect_named(r$result, "id")
})

test_that("a filter moved below a select or a mutate of other columns is pushed", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  scan <- read_parquet_gpu(path, lazy = TRUE)
  r <- pq_run(filter(select(scan, id, i), i > 1L))
  expect_length(r$read$predicates, 1L)
  # double arithmetic: it cannot overflow, so the filter may go below it
  r <- pq_run(filter(mutate(scan, z = id + 1), i > 1L))
  expect_length(r$read$predicates, 1L)
})

test_that("a filter does not move below checked integer arithmetic and hide its warning", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = c(1L, 2147483647L, 3L), i = c(0L, 0L, 5L))
  path <- pq_write(df)
  pipe <- function(x) filter(mutate(x, z = id + 1L), i > 1L)
  expect_warning(oracle <- pipe(df), "integer overflow")
  parquet_last_read(reset = TRUE)
  expect_warning(got <- collect(pipe(read_parquet_gpu(path, lazy = TRUE))), "integer overflow")
  expect_length(parquet_last_read()$predicates, 0L)
  expect_equal(tibble::as_tibble(got), oracle)
})

test_that("nothing is pushed where it would change the answer", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  scan <- read_parquet_gpu(path, lazy = TRUE)
  cases <- list(
    grouped = filter(group_by(scan, g), i > 1L),
    by = filter(scan, i > 1L, .by = g),
    reducer = filter(scan, i > mean(i, na.rm = TRUE)),
    window = filter(scan, i == max(i, na.rm = TRUE), .by = g),
    mutated = filter(mutate(scan, i = i + 1L), i > 1L),
    new_col = filter(mutate(scan, z = i + 1L), z > 1L),
    slice_ = filter(slice_head(scan, n = 4L), i > 1L),
    distinct_ = filter(distinct(scan, g, .keep_all = TRUE), i > 1L),
    arrange_slice = filter(slice_head(arrange(scan, id), n = 4L), i > 1L),
    summarise_ = filter(summarise(scan, m = max(i, na.rm = TRUE), .by = g), m > 1L),
    factor_ = filter(scan, f == "a"),
    float_gt = filter(scan, d > 0),
    float_eq = filter(scan, d == 1.5),
    float_int_lit = filter(scan, d > 1L),
    float_ne = filter(scan, d != 1.5),
    float_not = filter(scan, !(d > 0)),
    float_is_na = filter(scan, is.na(d)),
    string_eq = filter(scan, s == "x"),
    string_ne = filter(scan, s != "x"),
    string_flip = filter(scan, "x" == s),
    string_order = filter(scan, s < "y")
  )
  for (nm in names(cases)) {
    r <- pq_run(cases[[nm]])
    expect_length(r$read$predicates, 0L)
  }
})

test_that("pushdown off and eager reads push nothing", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  withr::with_options(list(cuplyr.parquet_pushdown = FALSE), {
    r <- pq_run(select(filter(read_parquet_gpu(path, lazy = TRUE), i > 1L), id))
    expect_length(r$read$predicates, 0L)
    expect_setequal(r$read$columns, names(pq_sem_df()))
  })
  parquet_last_read(reset = TRUE)
  res <- read_parquet_gpu(path, lazy = FALSE) |> filter(i > 1L) |> collect()
  expect_length(parquet_last_read()$predicates, 0L)
  expect_equal(res$id, filter(pq_sem_df(), i > 1L)$id)
})

test_that("a pushed predicate skips row groups from statistics", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = 1:1000, v = rep(c(1.5, 2.5), 500L))
  path <- pq_write(df, row_group_size = 100L)
  scan <- read_parquet_gpu(path, lazy = TRUE)
  r <- pq_run(filter(scan, id > 950L))
  expect_equal(r$read$row_groups_total, 10L)
  expect_lte(r$read$row_groups_after_stats, 2L)
  expect_equal(r$result$id, 951:1000)
  # predicates that match everywhere prune nothing and still return the rows
  r <- pq_run(filter(scan, id > 0L))
  expect_equal(r$read$row_groups_after_stats, 10L)
  expect_equal(nrow(r$result), 1000L)
})

test_that("explain() shows the columns and the predicate of the scan", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  q <- read_parquet_gpu(path, lazy = TRUE) |> filter(i > 1L) |> select(id)
  out <- paste(capture.output(explain(q)), collapse = "\n")
  expect_match(out, "parquet")
  expect_match(out, "i\\s*>\\s*1")
  expect_match(out, "\\bid\\b")
})

test_that("a scan is read at compute(), not at read_parquet_gpu() or a verb", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  parquet_last_read(reset = TRUE)
  q <- read_parquet_gpu(path, lazy = TRUE)
  expect_null(parquet_last_read())
  q2 <- filter(select(q, id, i), i > 1L)
  expect_null(parquet_last_read())
  res <- compute(q2)
  expect_false(is.null(parquet_last_read()))
  expect_equal(collect(res)$id, filter(pq_sem_df(), i > 1L)$id)
})

test_that("== 0 on a double keeps -0 rows when the file has a Bloom filter", {
  skip_if_no_gpu()
  path <- pq_fixture("duckdb_zero_bloom.parquet")
  # row group 1 holds 1024 rows of -0 and no +0; row group 2 holds 512 rows of 0
  preds <- list(
    function(x) filter(x, d == 0),
    function(x) filter(x, 0 == d),
    function(x) filter(x, d == 0L),
    function(x) filter(x, d == -0)
  )
  for (pipe in preds) {
    expect_identical(nrow(collect(pipe(read_parquet_gpu(path, lazy = FALSE)))), 1536L)
    for (push in c(TRUE, FALSE)) {
      withr::with_options(list(cuplyr.parquet_pushdown = push), {
        expect_identical(nrow(collect(pipe(read_parquet_gpu(path, lazy = TRUE)))), 1536L)
      })
    }
  }
  # other doubles are still found
  expect_identical(nrow(collect(filter(read_parquet_gpu(path, lazy = TRUE), d == 1.5))), 1536L)
})

test_that("a double filter keeps every row when a footer min or max is NaN", {
  skip_if_no_gpu()
  # row group 0 has NaN min and max in its footer statistics; libcudf 25.12
  # prunes on them. The data pages hold -987654.25, 5, 1234567.75.
  path <- pq_fixture("nan_footer_stats.parquet")
  cases <- list(
    list(function(x) filter(x, d > 1), 2:6),
    list(function(x) filter(x, d < 0), 1L),
    list(function(x) filter(x, d == 5), 2L),
    list(function(x) filter(x, d >= 1234567), 3L),
    list(function(x) filter(x, id > 0L, d > 1), 2:6)
  )
  for (cs in cases) {
    expect_identical(collect(cs[[1L]](read_parquet_gpu(path, lazy = FALSE)))$id, cs[[2L]])
    for (push in c(TRUE, FALSE)) {
      withr::with_options(list(cuplyr.parquet_pushdown = push), {
        expect_identical(collect(cs[[1L]](read_parquet_gpu(path, lazy = TRUE)))$id, cs[[2L]])
      })
    }
  }
  r <- pq_run(filter(read_parquet_gpu(path, lazy = TRUE), d > 1))
  expect_length(r$read$predicates, 0L)
})
