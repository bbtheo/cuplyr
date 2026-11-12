# F121: the parquet-to-summary pipeline benchmark (bench/parquet-lib.R).
#
# Pins what makes the numbers fair and correct: the seeded data and its file
# layout, each engine's pipelines (only the needed columns read, results equal
# to an independent base-R reference), cuplyr running natively with its
# pushdown visible, page-cache eviction, the load measurement, and the timing
# loop (warm-up rep, untimed hooks, busy reps re-run, out-of-memory recorded).
# Helpers in helper-parity-F121.R.

test_that("sourcing parquet-lib.R has no side effects and defines the API", {
  path <- f121_file("bench/parquet-lib.R")
  before_search <- search()
  before_opts <- options()
  before_libs <- .libPaths()
  before_files <- list.files(tempdir(), all.files = TRUE)
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  expect_identical(search(), before_search)
  expect_identical(options(), before_opts)
  expect_identical(.libPaths(), before_libs)
  expect_identical(list.files(tempdir(), all.files = TRUE), before_files)
  api <- c("pp_engines", "pp_variants", "pp_constants", "pp_columns", "pp_gen_fact",
           "pp_gen_dim", "pp_context", "pp_close", "pp_threads", "pp_read_arrow", "pp_sql",
           "pp_run", "pp_normalize", "pp_compare", "pp_evict", "pp_warm", "pp_resident",
           "pp_cpu", "pp_busy", "pp_wait_quiet", "pp_time_cell", "pp_first_call")
  for (f in api) expect_true(is.function(e[[f]]), label = paste(f, "is a function"))
})

test_that("engines, variants, constants and needed columns", {
  pp <- f121_load()
  expect_identical(pp$pp_engines(), c("cuplyr", "duckdb", "dplyr", "data.table"))
  v <- pp$pp_variants()
  expect_s3_class(v, "data.frame")
  expect_identical(v$variant, f121_variants)
  expect_identical(v$pipeline, rep(c("p1", "p2", "p3"), each = 2L))
  expect_identical(v$filter, rep(c("push", "nopush"), 3L))
  expect_identical(v$pushed, rep(c(TRUE, FALSE), 3L))

  k <- pp$pp_constants()
  expect_identical(k$day_from, as.Date("2024-01-01"))
  expect_identical(k$qty_min, 2L)
  expect_identical(k$price_min, 150)
  expect_identical(k$channel_drop, "phone")
  expect_identical(k$top_n, 10L)

  expect_setequal(pp$pp_columns("p1_push"), c("day", "store", "qty", "price", "discount", "channel"))
  expect_setequal(pp$pp_columns("p1_nopush"), c("store", "qty", "price", "discount", "channel"))
  expect_setequal(pp$pp_columns("p2_push"), c("day", "store", "qty", "price", "discount", "channel"))
  expect_setequal(pp$pp_columns("p2_nopush"), c("store", "qty", "price", "discount", "channel"))
  expect_setequal(pp$pp_columns("p3_push"), c("day", "product", "qty", "price", "discount"))
  expect_setequal(pp$pp_columns("p3_nopush"), c("product", "qty", "price", "discount", "channel"))
  expect_error(pp$pp_columns("p9_push"))
})

test_that("pp_gen_fact() writes the seeded narrow and wide tables with the stated layout", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  nar <- as.data.frame(arrow::read_parquet(fx$narrow))
  wid <- as.data.frame(arrow::read_parquet(fx$wide))
  expect_identical(names(nar), c("day", "store", "product", "qty", "price", "discount", "channel", "category"))
  expect_identical(nrow(nar), fx$n)
  expect_s3_class(nar$day, "Date")
  for (col in c("store", "product", "qty")) expect_type(nar[[col]], "integer")
  for (col in c("price", "discount")) expect_type(nar[[col]], "double")
  for (col in c("channel", "category")) expect_type(nar[[col]], "character")
  expect_false(is.unsorted(nar$day))
  expect_identical(range(nar$day), as.Date(c("2020-01-01", "2024-12-31")))
  expect_true(all(nar$store %in% 1:1000) && all(nar$product %in% 1:50000) && all(nar$qty %in% 1:20))
  expect_true(min(nar$price) >= 1 && max(nar$price) <= 200)
  expect_setequal(unique(nar$channel), c("app", "phone", "store", "web"))
  expect_length(unique(nar$category), 20L)
  na_frac <- mean(is.na(nar$discount))
  expect_true(na_frac > 0.005 && na_frac < 0.05, label = sprintf("discount NA fraction %.4f", na_frac))
  expect_true(all(nar$discount >= 0 & nar$discount <= 0.5, na.rm = TRUE))
  expect_false(anyNA(nar[setdiff(names(nar), "discount")]))
  # both filters keep about the same share of rows (about 19%)
  k <- pp$pp_constants()
  keep_push <- mean(nar$day >= k$day_from & nar$qty >= k$qty_min)
  keep_nopush <- mean(nar$price >= k$price_min & nar$channel != k$channel_drop)
  expect_true(abs(keep_push - 0.19) < 0.03 && abs(keep_nopush - 0.19) < 0.03,
              label = sprintf("selectivity push %.3f nopush %.3f", keep_push, keep_nopush))

  # wide: the same 8 columns with identical values, then x01..x32
  expect_identical(ncol(wid), 40L)
  expect_identical(names(wid)[1:8], names(nar))
  expect_identical(names(wid)[9:40], sprintf("x%02d", 1:32))
  expect_identical(wid[1:8], nar)
  types <- vapply(wid[9:40], typeof, "")
  expect_true(sum(types == "integer") >= 8L && sum(types == "double") >= 8L)

  # file layout: row groups of row_group_rows rows (the last may be short), snappy
  con <- DBI::dbConnect(duckdb::duckdb(shared_home = FALSE))
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))
  md <- DBI::dbGetQuery(con, sprintf(
    "SELECT row_group_id, max(row_group_num_rows) AS rows, string_agg(DISTINCT compression, ',') AS codec FROM parquet_metadata('%s') GROUP BY row_group_id ORDER BY row_group_id",
    fx$narrow))
  expect_identical(nrow(md), as.integer(ceiling(fx$n / fx$rg)))
  expect_true(all(md$rows[-nrow(md)] == fx$rg))
  expect_identical(unique(md$codec), "SNAPPY")
})

test_that("pp_gen_fact() is deterministic and independent of the write batching", {
  pp <- f121_load()
  d <- f121_tmpdir()
  a <- file.path(d, "a.parquet"); b <- file.path(d, "b.parquet"); s <- file.path(d, "s.parquet")
  pp$pp_gen_fact(a, n = 23000L, wide = TRUE, seed = 7L, row_group_rows = 4000L, groups_per_write = 1L)
  pp$pp_gen_fact(b, n = 23000L, wide = TRUE, seed = 7L, row_group_rows = 4000L, groups_per_write = 4L)
  pp$pp_gen_fact(s, n = 23000L, wide = TRUE, seed = 8L, row_group_rows = 4000L, groups_per_write = 4L)
  ra <- as.data.frame(arrow::read_parquet(a))
  expect_identical(ra, as.data.frame(arrow::read_parquet(b)))
  expect_false(identical(ra$price, as.data.frame(arrow::read_parquet(s))$price))
  expect_identical(arrow::ParquetFileReader$create(b)$num_row_groups, 6L)
})

test_that("pp_gen_dim() writes the store dimension", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  dm <- as.data.frame(arrow::read_parquet(fx$dim))
  expect_identical(names(dm), c("store", "region"))
  expect_identical(dm$store, 1:1000)
  expect_type(dm$region, "character")
  expect_length(unique(dm$region), 10L)
})

test_that("pp_normalize() gives the canonical form and pp_compare() catches differences", {
  pp <- f121_load()
  raw <- data.frame(channel = c("web", "app", "app"), store = c(2L, 2L, 1L),
                    n = c(3L, 1L, 2L), revenue = c(1, 2, 3), avg_price = c(4, 5, 6),
                    max_qty = c(7L, 8L, 9L), stringsAsFactors = FALSE)
  nz <- pp$pp_normalize(raw, "p1_push")
  expect_identical(class(nz), "data.frame")
  expect_identical(names(nz), f121_cols$p1)
  expect_identical(nz$store, c(1L, 2L, 2L))
  expect_identical(nz$channel, c("app", "app", "web"))
  expect_type(nz$n, "double")
  expect_type(nz$max_qty, "double")
  expect_identical(rownames(nz), as.character(1:3))
  # a data.table and a tibble normalize the same way
  expect_identical(pp$pp_normalize(data.table::as.data.table(raw), "p1_push"), nz)
  expect_identical(pp$pp_normalize(tibble::as_tibble(raw), "p1_push"), nz)

  top <- data.frame(product = c(5L, 9L, 7L), n = c(1, 2, 3), revenue = c(10, 30, 20))
  tz <- pp$pp_normalize(top, "p3_nopush")
  expect_identical(names(tz), f121_cols$p3)
  expect_identical(tz$product, c(9L, 7L, 5L))

  expect_true(pp$pp_compare(nz, nz)$ok)
  near <- nz; near$revenue <- near$revenue * (1 + 1e-12)
  expect_true(pp$pp_compare(nz, near, tol = 1e-9)$ok)
  far <- nz; far$revenue[2] <- far$revenue[2] * (1 + 1e-6)
  res <- pp$pp_compare(nz, far, tol = 1e-9)
  expect_false(res$ok)
  expect_type(res$detail, "character")
  expect_true(nzchar(res$detail))
  expect_false(pp$pp_compare(nz, nz[-1, ])$ok)
  key <- nz; key$store[1] <- 99L
  expect_false(pp$pp_compare(nz, key)$ok)
  nas <- nz; nas$avg_price[3] <- NA
  expect_false(pp$pp_compare(nz, nas)$ok)
  expect_false(pp$pp_compare(nz, nz[, -1])$ok)
})

test_that("every CPU engine matches an independent base-R reference on every variant", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  k <- pp$pp_constants()
  f <- as.data.frame(arrow::read_parquet(fx$narrow))
  d <- as.data.frame(arrow::read_parquet(fx$dim))
  for (engine in c("duckdb", "dplyr", "data.table")) {
    ctx <- pp$pp_context(engine)
    for (tab in c("narrow", "wide")) {
      for (v in f121_variants) {
        ref <- f121_base_ref(v, f, d, k)
        res <- pp$pp_run(engine, v, fx[[tab]], fx$dim, ctx)
        expect_true(is.data.frame(res), label = paste(engine, tab, v, "returns a data frame"))
        nz <- pp$pp_normalize(res, v)
        expect_equal(nz, ref, tolerance = 1e-9, label = paste(engine, tab, v))
        expect_true(pp$pp_compare(nz, ref)$ok, label = paste(engine, tab, v, "pp_compare"))
      }
    }
    pp$pp_close(ctx)
  }
})

test_that("each engine returns its own kind of R data frame", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  ctx <- pp$pp_context("duckdb")
  withr::defer(pp$pp_close(ctx))
  expect_identical(class(pp$pp_run("duckdb", "p1_push", fx$narrow, fx$dim, ctx)), "data.frame")
  expect_s3_class(pp$pp_run("dplyr", "p1_push", fx$narrow, fx$dim, pp$pp_context("dplyr")), "tbl_df")
  expect_s3_class(pp$pp_run("data.table", "p1_push", fx$narrow, fx$dim, pp$pp_context("data.table")), "data.table")
})

test_that("cuplyr matches the base-R reference on every variant, runs natively and comes back as a tibble", {
  skip_if_no_gpu()
  pp <- f121_load()
  fx <- f121_fixture(pp)
  k <- pp$pp_constants()
  f <- as.data.frame(arrow::read_parquet(fx$narrow))
  d <- as.data.frame(arrow::read_parquet(fx$dim))
  ctx <- pp$pp_context("cuplyr")
  withr::local_options(cuplyr.fallback = "error")
  for (tab in c("narrow", "wide")) {
    for (v in f121_variants) {
      res <- pp$pp_run("cuplyr", v, fx[[tab]], fx$dim, ctx)
      expect_s3_class(res, "tbl_df")
      expect_false(inherits(res, "tbl_gpu"))
      ref <- f121_base_ref(v, f, d, k)
      expect_equal(pp$pp_normalize(res, v), ref, tolerance = 1e-9, label = paste("cuplyr", tab, v))
    }
  }
  pp$pp_close(ctx)
})

test_that("cuplyr's scan reads only the needed columns and gets the pushed filter", {
  skip_if_no_gpu()
  pp <- f121_load()
  fx <- f121_fixture(pp)
  ctx <- pp$pp_context("cuplyr")
  for (tab in c("narrow", "wide")) {
    for (v in c("p1_push", "p1_nopush", "p3_push", "p3_nopush")) {
      pp$pp_run("cuplyr", v, fx[[tab]], fx$dim, ctx)
      lr <- parquet_last_read()
      expect_identical(normalizePath(lr$paths), normalizePath(fx[[tab]]))
      expect_setequal(lr$columns, pp$pp_columns(v))
      if (grepl("_push$", v)) {
        expect_true(all(c("day >= 2024-01-01", "qty >= 2") %in% lr$predicates),
                    label = paste(tab, v, "pushes the date and integer atoms"))
        expect_lt(lr$row_groups_after_stats, lr$row_groups_total)
      } else {
        expect_length(lr$predicates, 0L)
        expect_identical(lr$rows_read, as.numeric(fx$n))
      }
    }
  }
})

test_that("dplyr and data.table read only the needed columns, through arrow", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  seen <- list()
  orig <- pp$pp_read_arrow
  pp$pp_read_arrow <- function(path, cols) {
    seen[[length(seen) + 1L]] <<- list(path = path, cols = cols)
    orig(path, cols)
  }
  for (engine in c("dplyr", "data.table")) {
    for (v in f121_variants) {
      seen <- list()
      pp$pp_run(engine, v, fx$wide, fx$dim, pp$pp_context(engine))
      paths <- vapply(seen, function(s) normalizePath(s$path), "")
      expect_true(normalizePath(fx$wide) %in% paths, label = paste(engine, v, "reads the fact file through pp_read_arrow"))
      fact_cols <- seen[[match(normalizePath(fx$wide), paths)]]$cols
      expect_setequal(fact_cols, pp$pp_columns(v))
      if (startsWith(v, "p2")) {
        expect_true(normalizePath(fx$dim) %in% paths, label = paste(engine, v, "reads the dimension file"))
      }
    }
  }
  got <- orig(fx$wide, c("store", "price"))
  expect_identical(names(got), c("store", "price"))
  expect_identical(nrow(got), fx$n)
})

test_that("DuckDB runs its own parquet reader through SQL with explicit columns", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  for (v in f121_variants) {
    sql <- pp$pp_sql(v, fx$wide, fx$dim)
    expect_type(sql, "character")
    expect_length(sql, 1L)
    expect_match(sql, "read_parquet(", fixed = TRUE)
    expect_match(sql, fx$wide, fixed = TRUE)
    expect_false(grepl("SELECT\\s+\\*", sql, ignore.case = TRUE))
    if (startsWith(v, "p2")) expect_match(sql, fx$dim, fixed = TRUE)
  }
  calls <- 0L
  orig <- pp$pp_sql
  pp$pp_sql <- function(...) { calls <<- calls + 1L; orig(...) }
  ctx <- pp$pp_context("duckdb")
  withr::defer(pp$pp_close(ctx))
  pp$pp_run("duckdb", "p2_nopush", fx$narrow, fx$dim, ctx)
  expect_identical(calls, 1L)
})

test_that("pp_threads() reads back each engine's thread count in effect", {
  pp <- f121_load()
  ctx <- pp$pp_context("duckdb")
  withr::defer(pp$pp_close(ctx))
  th <- pp$pp_threads("duckdb", ctx)
  expect_equal(unname(th[["threads"]]), as.numeric(DBI::dbGetQuery(ctx$con, "SELECT current_setting('threads') AS t")$t))
  expect_equal(unname(th[["threads"]]), parallel::detectCores())
  expect_equal(unname(pp$pp_threads("data.table", NULL)[["threads"]]), data.table::getDTthreads())
  expect_equal(unname(pp$pp_threads("dplyr", NULL)[["arrow_cpu"]]), arrow::cpu_count())
})

test_that("pp_evict() drops a file from the page cache and pp_warm() brings it back", {
  pp <- f121_load(character())
  d <- f121_tmpdir()
  if (f121_is_tmpfs(d)) skip("test directory is on tmpfs; eviction cannot be observed")
  p <- file.path(d, "blob.bin")
  writeBin(as.raw(sample.int(255L, 8e6, TRUE)), p)
  pp$pp_warm(p)
  expect_gte(pp$pp_resident(p), 0.99)
  pp$pp_evict(p)
  expect_lte(pp$pp_resident(p), 0.01)
  pp$pp_warm(p)
  expect_gte(pp$pp_resident(p), 0.99)
})

test_that("pp_busy() counts other processes' user-space load, not this process or kernel work", {
  # Round 1 review (B1): on this machine a cold read makes kernel threads
  # (dm-crypt, page cache) busy on the engine's behalf, so "everything but this
  # process" is not other people's load. foreign = other processes' user time
  # (all user + nice time minus this process's user time); kernel = system, irq
  # and softirq time minus this process's system time. Shares of all CPUs.
  pp <- f121_load(character())
  ncpu <- parallel::detectCores()
  c0 <- pp$pp_cpu()
  t0 <- proc.time()[["elapsed"]]
  x <- 0
  while (proc.time()[["elapsed"]] - t0 < 0.5) x <- x + sum(sqrt(seq_len(1e4)))
  c1 <- pp$pp_cpu()
  b <- pp$pp_busy(c0, c1)
  expect_true(all(c("total", "own", "kernel", "foreign") %in% names(b)))
  expect_true(all(b >= 0 & b <= 1))
  # one busy R thread is about 1/ncpu of the machine, and it is not foreign
  expect_gt(b[["own"]], 0.5 / ncpu)
  # the parts add up to the total (clock-tick rounding aside)
  expect_lt(abs(b[["own"]] + b[["kernel"]] + b[["foreign"]] - b[["total"]]), 0.02)

  # a busy loop in another user process is foreign
  system2("timeout", c("1.5", "sh", "-c", shQuote("while :; do :; done")), wait = FALSE)
  Sys.sleep(0.2)
  c0 <- pp$pp_cpu()
  Sys.sleep(1)
  c1 <- pp$pp_cpu()
  b2 <- pp$pp_busy(c0, c1)
  expect_gt(b2[["foreign"]], 0.6 / ncpu)
  expect_lt(b2[["own"]], 0.3 / ncpu)
})

test_that("pp_busy() splits a known /proc/stat delta into own, kernel and foreign", {
  # Round 2 review N16: pins that system/irq/softirq time outside this process
  # is kernel, not foreign. Jiffies at 100 per second; own times in seconds.
  pp <- f121_load(character())
  c0 <- c(total = 0, idle = 0, user = 0, sys = 0, own_user = 0, own_sys = 0)
  c1 <- c(total = 100, idle = 50, user = 20, sys = 30, own_user = 0.10, own_sys = 0.05)
  b <- pp$pp_busy(c0, c1)
  expect_equal(b[["total"]], 0.5)
  expect_equal(b[["own"]], 0.15)
  expect_equal(b[["kernel"]], 0.25)
  expect_equal(b[["foreign"]], 0.10)
})

test_that("pp_time_cell() times only the call, rotates engines, and keeps a warm-up rep", {
  pp <- f121_load(character())
  log <- character()
  fns <- list(
    a = function() { log <<- c(log, "a"); Sys.sleep(0.03); "A" },
    b = function() { log <<- c(log, "b"); Sys.sleep(0.03); "B" }
  )
  before <- function(engine) { log <<- c(log, paste0("before-", engine)); Sys.sleep(0.15); 0.5 }
  checks <- 0L
  check <- function(engine, res) { checks <<- checks + 1L; identical(res, toupper(engine)) }
  waits <- 0L
  out <- pp$pp_time_cell(fns, reps = 3L, before = before, check = check, busy_max = 0.15,
                         busy_fn = function(c0, c1) c(total = 0, own = 0, foreign = 0),
                         wait = function(...) { waits <<- waits + 1L; 0 })
  expect_s3_class(out, "data.frame")
  expect_true(all(c("engine", "rep", "secs", "busy", "resident", "quiet", "ok", "status") %in% names(out)))
  for (e in names(fns)) {
    r <- out[out$engine == e, ]
    expect_identical(sort(r$rep), 0:3, label = paste(e, "has rep 0 (warm-up) and reps 1-3"))
  }
  # the hook's 0.15 s sleep is outside the timed region
  expect_true(all(out$secs >= 0.025 & out$secs < 0.14), label = paste(round(out$secs, 3), collapse = " "))
  expect_true(all(out$resident == 0.5))
  expect_true(all(out$ok) && all(out$quiet) && all(out$status == "ok"))
  expect_identical(checks, nrow(out))
  expect_gte(waits, nrow(out))
  # every call is preceded by its own hook
  calls <- which(log %in% c("a", "b"))
  expect_true(all(log[calls - 1L] == paste0("before-", log[calls])))
  # the order rotates: each engine goes first in some rep
  firsts <- log[calls][seq(1L, length(calls), by = 2L)]
  expect_setequal(unique(firsts), c("a", "b"))
})

test_that("pp_time_cell() re-runs busy reps and caps the attempts", {
  pp <- f121_load(character())
  fns <- list(a = function() 1, b = function() 2)
  n <- 0L
  # the third measured call is busy
  busy_fn <- function(c0, c1) { n <<- n + 1L; f <- if (n == 3L) 0.5 else 0; c(total = f, own = 0, foreign = f) }
  out <- pp$pp_time_cell(fns, reps = 2L, busy_max = 0.15, busy_fn = busy_fn, wait = function(...) 0)
  timed <- out[out$rep >= 1L, ]
  for (e in names(fns)) expect_identical(sum(timed$engine == e & timed$quiet), 2L)
  expect_identical(sum(!out$quiet), 1L)
  expect_equal(out$busy[!out$quiet], 0.5)
  expect_identical(nrow(timed), 5L)

  always <- function(c0, c1) c(total = 0.9, own = 0, foreign = 0.9)
  out2 <- pp$pp_time_cell(fns, reps = 2L, busy_max = 0.15, max_attempts = 4L, busy_fn = always, wait = function(...) 0)
  for (e in names(fns)) {
    r <- out2[out2$engine == e & out2$rep >= 1L, ]
    expect_identical(nrow(r), 4L)
    expect_false(any(r$quiet))
  }
})

test_that("pp_time_cell() records out-of-memory, errors and wrong results without stopping the cell", {
  pp <- f121_load(character())
  oom <- function() stop(structure(class = c("cuplyr_error_gpu_oom", "error", "condition"),
                                   list(message = "GPU out of memory", call = NULL)))
  fns <- list(gpu = oom, bad = function() stop("boom"), wrong = function() 0, fine = function() 1)
  check <- function(engine, res) engine != "wrong"
  out <- pp$pp_time_cell(fns, reps = 2L, check = check, busy_fn = function(c0, c1) c(total = 0, own = 0, foreign = 0),
                         wait = function(...) 0)
  g <- out[out$engine == "gpu", ]
  expect_identical(nrow(g), 1L)
  expect_identical(g$status, "oom")
  expect_false(g$ok)
  expect_true(is.na(g$secs))
  b <- out[out$engine == "bad", ]
  expect_identical(nrow(b), 1L)
  expect_identical(b$status, "error")
  w <- out[out$engine == "wrong", ]
  expect_true(all(w$status == "mismatch") && !any(w$ok))
  f <- out[out$engine == "fine", ]
  expect_identical(sort(f$rep), 0:2)
  expect_true(all(f$ok))
})

test_that("pp_first_call() times the first calls in a fresh process", {
  pp <- f121_load(c("arrow", "duckdb", "DBI", "data.table", "callr"))
  fx <- f121_fixture(pp)
  root <- normalizePath(testthat::test_path("..", ".."))
  out <- pp$pp_first_call("duckdb", "p1_push", fx$narrow, fx$dim, calls = 3L, root = root, lib = f121_bench_lib())
  expect_s3_class(out, "data.frame")
  expect_true(all(c("engine", "variant", "call", "secs", "load_secs", "pid") %in% names(out)))
  expect_identical(out$call, 1:3)
  expect_true(all(out$engine == "duckdb" & out$variant == "p1_push"))
  expect_true(all(out$secs > 0) && all(out$load_secs >= 0))
  expect_false(any(out$pid == Sys.getpid()))
  expect_length(unique(out$pid), 1L)
})

test_that("pp_first_call() loads this checkout's cuplyr in the child", {
  skip_if_no_gpu()
  pp <- f121_load(c("arrow", "duckdb", "DBI", "data.table", "callr"))
  fx <- f121_fixture(pp)
  root <- normalizePath(testthat::test_path("..", ".."))
  out <- pp$pp_first_call("cuplyr", "p3_push", fx$narrow, fx$dim, calls = 3L, root = root, lib = f121_bench_lib())
  expect_identical(out$call, 1:3)
  expect_true(all(out$secs > 0))
  expect_false(any(out$pid == Sys.getpid()))
})
