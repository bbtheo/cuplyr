# F73: polars and DuckDB in the rolling-window benchmark (bench/rolling-lib.R).
#
# Pins the helpers that decide whether the polars and DuckDB numbers are fair
# and correct: data loaded once into each engine, the timed call leaves its
# result inside the engine, the untimed reset between calls, the thread count
# actually in effect, and results equal to data.table's (NA padding included).
# CPU only: no GPU is touched. bench/ is .Rbuildignore'd, so these tests skip
# outside a source checkout. On a machine with the benchmark library, a
# missing polars or duckdb there is a failure, not a skip.

f73_lib_file <- function() {
  root <- testthat::test_path("..", "..")
  if (!file.exists(file.path(root, "DESCRIPTION"))) {
    testthat::skip("not in a source checkout; bench/ is not shipped")
  }
  path <- file.path(root, "bench", "rolling-lib.R")
  testthat::expect_true(file.exists(path), label = "bench/rolling-lib.R exists")
  if (!file.exists(path)) testthat::skip("bench/rolling-lib.R missing (failure recorded)")
  normalizePath(path)
}

f73_bench_lib <- function() {
  Sys.getenv("CUPLYR_BENCH_LIB", "/home/theo/cuplr/scratchpad/benchmarks/rlib")
}

# Source rolling-lib.R with the benchmark library first on the path (for the
# caller's frame), and require the packages the test needs.
f73_load <- function(pkgs = character(), env = parent.frame()) {
  path <- f73_lib_file()
  lib <- f73_bench_lib()
  if (dir.exists(lib)) {
    withr::local_libpaths(lib, action = "prefix", .local_envir = env)
    for (p in pkgs) {
      ok <- requireNamespace(p, quietly = TRUE)
      testthat::expect_true(ok, label = paste(p, "is installed in the benchmark library"))
      if (!ok) testthat::skip(paste(p, "missing (failure recorded)"))
    }
  } else {
    for (p in pkgs) testthat::skip_if_not_installed(p)
  }
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

f73_stats <- c("mean", "median", "min", "max", "sum", "sd")

# data.table's result per group, in the original row order
f73_grouped_ref <- function(rw, stat, x, g, n) {
  out <- numeric(length(x))
  for (k in split(seq_along(x), g)) out[k] <- rw$rw_reference(stat, x[k], n)
  out
}

test_that("rw_time() calls reset before every call, outside the timed region", {
  rw <- f73_load("microbenchmark")
  log <- character()
  fn <- function() { log <<- c(log, "f"); 1 }
  reset <- function() { log <<- c(log, "r"); Sys.sleep(0.05) }
  ms <- rw$rw_time(fn, times = 3, warmup = 2, reset = reset)
  expect_length(ms, 3)
  expect_identical(log, rep(c("r", "f"), 5))
  expect_true(all(ms < 40), label = "the 50 ms reset is outside every timed sample")
  # the default keeps the F54 behaviour
  log <- character()
  expect_length(rw$rw_time(fn, times = 2, warmup = 1), 2)
  expect_identical(log, rep("f", 3))
})

test_that("run_cell() calls reset before every call of the timed closure, the probe included", {
  rw <- f73_load(c("microbenchmark", "data.table"))
  log <- character()
  x <- rw$rw_data(50)$x
  fn <- function() { log <<- c(log, "f"); data.table::frollmean(x, 3) }
  reset <- function() log <<- c(log, "r")
  res <- rw$run_cell(fn, identity, times = 3, ref = rw$rw_reference("mean", x, 3),
                     cell_budget = 20, probe_limit = 60, reset = reset)
  n_f <- sum(log == "f")
  expect_gte(n_f, 1 + 3)
  expect_identical(log, rep(c("r", "f"), n_f))
  expect_true(res$cmp$ok)
  expect_length(res$ms, 3)
})

test_that("DuckDB: the data is a DuckDB table loaded once, and the thread count read back is the one in effect", {
  rw <- f73_load(c("duckdb", "DBI"))
  df <- rw$rw_data(50)
  e1 <- rw$rw_duckdb_engine(df, threads = 1)
  withr::defer(e1$close())
  expect_identical(as.integer(e1$threads()), 1L)
  tabs <- DBI::dbGetQuery(e1$con, "SELECT table_name FROM duckdb_tables()")$table_name
  expect_true("rw_x" %in% tabs, label = "rw_x is a DuckDB table (not a registered R data frame)")
  views <- DBI::dbGetQuery(e1$con, "SELECT view_name FROM duckdb_views() WHERE NOT internal")$view_name
  expect_false("rw_x" %in% views)
  expect_identical(nrow(DBI::dbGetQuery(e1$con, "SELECT * FROM rw_x")), nrow(df))

  e0 <- rw$rw_duckdb_engine(df)
  withr::defer(e0$close())
  if (parallel::detectCores() > 1) expect_gt(e0$threads(), 1)
})

test_that("DuckDB: the timed SQL is a plain CREATE TEMP TABLE over a guarded row frame", {
  rw <- f73_load(c("duckdb", "DBI"))
  q <- rw$rw_duckdb_sql("mean", 101)
  expect_match(q, "CREATE TEMP TABLE", fixed = TRUE)
  expect_no_match(q, "OR REPLACE", fixed = TRUE)
  expect_match(q, "ROWS BETWEEN 100 PRECEDING AND CURRENT ROW", fixed = TRUE)
  expect_match(q, "ORDER BY i", fixed = TRUE)
  expect_match(q, "AVG(x)", fixed = TRUE)
  expect_match(q, "CASE WHEN", fixed = TRUE)
  expect_no_match(q, "PARTITION BY", fixed = TRUE)
  expect_match(rw$rw_duckdb_sql("mean", 11, grouped = TRUE), "PARTITION BY g", fixed = TRUE)
  aggs <- c(sum = "SUM(x)", min = "MIN(x)", max = "MAX(x)", sd = "STDDEV_SAMP(x)", median = "MEDIAN(x)")
  for (s in names(aggs)) expect_match(rw$rw_duckdb_sql(s, 11), aggs[[s]], fixed = TRUE)
  expect_error(rw$rw_duckdb_sql("custom", 11))
})

test_that("DuckDB: every statistic equals data.table, NA padding included", {
  rw <- f73_load(c("duckdb", "DBI", "data.table"))
  df <- rw$rw_data(60)
  e <- rw$rw_duckdb_engine(df)
  withr::defer(e$close())
  for (stat in f73_stats) {
    f <- e$fun(stat)
    for (n in c(3, 4, 11, 61)) {
      e$reset()
      res <- f(n)
      out <- e$extract(res)
      cmp <- rw$rw_compare(out, rw$rw_reference(stat, df$x, n))
      expect_true(cmp$ok, label = sprintf("DuckDB %s at w=%s equals data.table (NA mismatches %s, max diff %s)",
                                          stat, n, cmp$n_na_mismatch, cmp$max_abs_diff))
    }
  }
  e$reset()
})

test_that("DuckDB grouped: PARTITION BY g equals data.table by = g, in the original row order", {
  rw <- f73_load(c("duckdb", "DBI", "data.table"))
  dfg <- rw$rw_grouped_data(400, 7)
  e <- rw$rw_duckdb_engine(dfg)
  withr::defer(e$close())
  for (stat in f73_stats) {
    f <- e$fun(stat, grouped = TRUE)
    for (n in c(3, 4, 11)) {
      e$reset()
      res <- f(n)
      out <- e$extract(res)
      cmp <- rw$rw_compare(out, f73_grouped_ref(rw, stat, dfg$x, dfg$g, n))
      expect_true(cmp$ok, label = sprintf("DuckDB grouped %s at w=%s", stat, n))
    }
  }
  e$reset()
})

test_that("DuckDB: the timed call leaves its result inside DuckDB; reset drops it, untimed", {
  rw <- f73_load(c("duckdb", "DBI"))
  df <- rw$rw_data(100)
  e <- rw$rw_duckdb_engine(df)
  withr::defer(e$close())
  f <- e$fun("mean")
  e$reset()
  res <- f(11)
  expect_false(is.numeric(res) || is.data.frame(res),
               label = "the timed closure returns a token, not data fetched into R")
  expect_length(e$extract(res), nrow(df))
  # a plain CREATE: a second call without the reset fails, so the drop of the
  # previous result can never be inside a timed call
  expect_error(f(11))
  e$reset()
  expect_no_error(f(11))
  e$reset()
})

test_that("polars: every statistic equals data.table, and the result stays a polars DataFrame", {
  rw <- f73_load(c("polars", "data.table"))
  df <- rw$rw_data(60)
  p <- rw$rw_polars_engine(df)
  withr::defer(p$close())
  expect_identical(as.integer(p$threads()), as.integer(polars::polars_info()$thread_pool_size))
  for (stat in f73_stats) {
    f <- p$fun(stat)
    for (n in c(3, 4, 11, 61)) {
      p$reset()
      res <- f(n)
      expect_s3_class(res, "polars_data_frame")
      cmp <- rw$rw_compare(p$extract(res), rw$rw_reference(stat, df$x, n))
      expect_true(cmp$ok, label = sprintf("polars %s at w=%s equals data.table (NA mismatches %s, max diff %s)",
                                          stat, n, cmp$n_na_mismatch, cmp$max_abs_diff))
    }
  }
  expect_error(p$fun("custom"))
})

test_that("polars grouped: over(g) equals data.table by = g, in the original row order", {
  rw <- f73_load(c("polars", "data.table"))
  dfg <- rw$rw_grouped_data(400, 7)
  p <- rw$rw_polars_engine(dfg)
  withr::defer(p$close())
  for (stat in f73_stats) {
    f <- p$fun(stat, grouped = TRUE)
    for (n in c(3, 4, 11)) {
      res <- f(n)
      expect_s3_class(res, "polars_data_frame")
      cmp <- rw$rw_compare(p$extract(res), f73_grouped_ref(rw, stat, dfg$x, dfg$g, n))
      expect_true(cmp$ok, label = sprintf("polars grouped %s at w=%s", stat, n))
    }
  }
})

test_that("rw_polars_child() fixes the polars thread pool before polars loads, without cuplyr", {
  rw <- f73_load(c("polars", "callr"))
  libs <- .libPaths()
  pool <- function() polars::polars_info()$thread_pool_size
  expect_identical(as.integer(rw$rw_polars_child(1, pool, libs = libs)), 1L)
  # the default child ignores a POLARS_MAX_THREADS inherited from the parent
  withr::local_envvar(POLARS_MAX_THREADS = "3")
  n_default <- as.integer(rw$rw_polars_child(NULL, pool, libs = libs))
  if (parallel::detectCores() > 3) expect_gt(n_default, 3L)
  expect_identical(rw$rw_polars_child(NULL, function() Sys.getenv("POLARS_MAX_THREADS"), libs = libs), "")
  # arguments, library path, and no cuplyr (no CUDA context) in the child
  expect_identical(rw$rw_polars_child(1, function(a, b) a + b, args = list(a = 2, b = 3), libs = libs), 5)
  child_lib <- rw$rw_polars_child(1, function() .libPaths()[1], libs = libs)
  expect_identical(normalizePath(child_lib), normalizePath(libs[1]))
  expect_false(rw$rw_polars_child(1, function() "cuplyr" %in% loadedNamespaces(), libs = libs))
})
