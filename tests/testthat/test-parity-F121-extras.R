# F121 rounds 2 to 4: tests of what the coder added to bench/ (the load columns of
# pp_time_cell(), the read-back settings, data.table's two idioms on the join
# pipeline, the cuplyr probe helpers, the rowwise chunk's cap, fallback watch and
# cells, the idle probe, and the report's names and rendered method text). The PM's acceptance tests are in
# test-parity-F121-lib.R and test-parity-F121-rowwise.R.

test_that("pp_time_cell() records kernel load, the wait sample and the wait length per call", {
  pp <- f121_load(character())
  fns <- list(a = function() Sys.sleep(0.01), b = function() Sys.sleep(0.01))
  busy_fn <- function(c0, c1) c(total = 0.3, own = 0.1, kernel = 0.07, foreign = 0.05)
  wait <- function(limit) { Sys.sleep(0.05); 0.02 }
  out <- pp$pp_time_cell(fns, reps = 2L, busy_max = 0.15, busy_fn = busy_fn, wait = wait)
  expect_true(all(c("busy", "own", "kernel", "total", "wait_load", "wait_secs") %in% names(out)))
  expect_true(all(out$kernel == 0.07) && all(out$busy == 0.05) && all(out$own == 0.1) && all(out$total == 0.3))
  expect_true(all(out$wait_load == 0.02))
  expect_true(all(out$wait_secs >= 0.04 & out$wait_secs < 0.5))
  expect_true(all(out$quiet))
  # one limit for the wait and for the verdict: the wait is handed busy_max
  seen <- numeric()
  pp$pp_time_cell(fns, reps = 1L, busy_max = 0.2, busy_fn = busy_fn, wait = function(limit) { seen <<- c(seen, limit); 0 })
  expect_true(all(seen == 0.2))
  # a load function without a kernel part (as in the older tests) leaves it missing
  out2 <- pp$pp_time_cell(fns, reps = 1L, busy_fn = function(c0, c1) c(total = 0, own = 0, foreign = 0), wait = NULL)
  expect_true(all(is.na(out2$kernel)) && all(is.na(out2$wait_load)))
})

test_that("pp_engine_settings() reads DuckDB's file-cache settings and the debug stream pool back", {
  pp <- f121_load(c("DBI", "duckdb"))
  ctx <- pp$pp_context("duckdb")
  withr::defer(pp$pp_close(ctx))
  s <- pp$pp_engine_settings("duckdb", ctx)
  expect_setequal(names(s), c("cache_local_files", "enable_external_file_cache"))
  expect_identical(s[["cache_local_files"]],
                   as.character(DBI::dbGetQuery(ctx$con, "SELECT current_setting('cache_local_files') AS v")$v))
  withr::local_envvar(LIBCUDF_USE_DEBUG_STREAM_POOL = "1")
  expect_match(pp$pp_engine_settings("cuplyr")[["debug_stream_pool"]], "^set")
  withr::local_envvar(LIBCUDF_USE_DEBUG_STREAM_POOL = NA)
  expect_identical(pp$pp_engine_settings("cuplyr")[["debug_stream_pool"]], "unset")
  expect_length(pp$pp_engine_settings("dplyr"), 0L)
})

test_that("data.table's single and stepwise idioms give the reference result on every variant", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  k <- pp$pp_constants()
  f <- as.data.frame(arrow::read_parquet(fx$narrow))
  d <- as.data.frame(arrow::read_parquet(fx$dim))
  for (form in c("single", "stepwise")) for (v in f121_variants) {
    ref <- f121_base_ref(v, f, d, k)
    nz <- pp$pp_normalize(pp$pp_datatable(v, fx$narrow, fx$dim, form = form), v)
    expect_true(pp$pp_compare(nz, ref)$ok, label = paste(form, v))
  }
})

test_that("the cuplyr probe helpers build the scan and the fact side the report describes", {
  skip_if_no_gpu()
  pp <- f121_load()
  fx <- f121_fixture(pp)
  k <- pp$pp_constants()
  f <- as.data.frame(arrow::read_parquet(fx$narrow))
  d <- as.data.frame(arrow::read_parquet(fx$dim))
  withr::local_options(cuplyr.fallback = "error")
  # the lazy plan collects to the reference result
  for (v in c("p1_push", "p2_nopush", "p3_push")) {
    nz <- pp$pp_normalize(dplyr::collect(pp$pp_cuplyr_plan(v, fx$narrow, fx$dim)), v)
    expect_true(pp$pp_compare(nz, f121_base_ref(v, f, d, k))$ok, label = paste("plan", v))
  }
  # the scan: the variant's columns, and for a push variant the pushed filter
  sc <- dplyr::collect(pp$pp_cuplyr_scan("p1_push", fx$narrow))
  expect_setequal(names(sc), pp$pp_columns("p1_push"))
  expect_true(all(sc$day >= k$day_from & sc$qty >= k$qty_min))
  sn <- dplyr::collect(pp$pp_cuplyr_scan("p1_nopush", fx$narrow))
  expect_setequal(names(sn), pp$pp_columns("p1_nopush"))
  expect_identical(nrow(sn), nrow(f))
  # the fact side: filtered rows with the revenue, and the last read is the fact file
  fs <- dplyr::collect(pp$pp_cuplyr_fact_side("p2_push", fx$narrow))
  expect_named(fs, c("store", "channel", "price", "qty", "revenue"))
  expect_identical(nrow(fs), sum(f$day >= k$day_from & f$qty >= k$qty_min))
  lr <- utils::getFromNamespace("parquet_last_read", "cuplyr")()
  expect_identical(basename(lr$paths[[1L]]), basename(fx$narrow))
  expect_lt(lr$row_groups_after_stats, lr$row_groups_total)
})

# ---- round 2b: the rowwise chunk (Theo's addition) and review N12 ----------------------------

test_that("pp_rowwise_engine() names the engine each form needs", {
  pp <- f121_load(character())
  eng <- vapply(pp$pp_rowwise_forms(), pp$pp_rowwise_engine, "")
  expect_identical(unname(eng), c("cuplyr", "dplyr", "dplyr", "data.table", "duckdb"))
  expect_true(all(eng %in% pp$pp_engines()))
  expect_error(pp$pp_rowwise_engine("nope"))
})

test_that("the base-R rowwise reference equals the vectorised dplyr form, and the forms' code names the columns", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  ref <- pp$pp_rowwise_ref(fx$wide)
  expect_identical(names(ref), c("category", "n", "m_mean", "s_sum"))
  expect_identical(sum(ref$n), as.double(fx$n))
  expect_true(pp$pp_compare(pp$pp_normalize(pp$pp_run_rowwise("dplyr_vectorised", fx$wide), "rowwise"), ref)$ok)
  # a result with the columns in another order or an integer count normalizes to the same form
  shuffled <- data.frame(s_sum = ref$s_sum, category = ref$category, m_mean = ref$m_mean, n = as.integer(ref$n))[rev(seq_len(nrow(ref))), ]
  expect_identical(pp$pp_normalize(shuffled, "rowwise"), ref)
  expect_error(pp$pp_normalize(ref[c("category", "n")], "rowwise"), "lacks")
  # every form's code uses the columns of pp_constants()$rowwise_cols (written out, or through the constant)
  for (form in setdiff(pp$pp_rowwise_forms(), "duckdb")) {
    code <- paste(deparse(pp$pp_rowwise_fn(form)), collapse = "\n")
    expect_true(grepl("rowwise_cols", code, fixed = TRUE) || all(vapply(pp$pp_constants()$rowwise_cols, grepl, NA, x = code, fixed = TRUE)), label = form)
  }
  expect_true(all(vapply(pp$pp_constants()$rowwise_cols, grepl, NA, x = pp$pp_rowwise_sql("f.parquet"), fixed = TRUE)))
})

test_that("pp_time_cell() with cap = Inf sets no time limit and gives the same rows as without the argument", {
  pp <- f121_load(character())
  fns <- list(a = function() 1, b = function() 2)
  bf <- function(c0, c1) c(total = 0, own = 0, kernel = 0, foreign = 0)
  seen <- numeric()
  w <- function(...) 0
  a <- pp$pp_time_cell(fns, reps = 2L, busy_fn = bf, wait = w)
  b <- pp$pp_time_cell(fns, reps = 2L, busy_fn = bf, wait = w, cap = Inf)
  cols <- setdiff(names(a), c("secs", "started", "wait_secs"))
  expect_identical(a[cols], b[cols])
  # a finite cap that is not reached changes nothing either, and leaves no time limit behind:
  # the loop after the cell runs past the cap, which a limit left set would abort
  c3 <- pp$pp_time_cell(fns, reps = 2L, busy_fn = bf, wait = w, cap = 0.5)
  expect_identical(c3[cols], a[cols])
  t0 <- proc.time()[["elapsed"]]
  expect_no_error(while (proc.time()[["elapsed"]] - t0 < 1.2) NULL)
})

test_that("a capped call is recorded as capped, with its error kept apart from other errors", {
  pp <- f121_load(character())
  spin <- function() { t0 <- proc.time()[["elapsed"]]; while (proc.time()[["elapsed"]] - t0 < 4) NULL; 1 }
  out <- pp$pp_time_cell(list(spin = spin, boom = function() stop("boom"), fine = function() 1), reps = 1L, cap = 0.2,
                         busy_fn = function(c0, c1) c(total = 0, own = 0, kernel = 0, foreign = 0), wait = NULL)
  expect_identical(out$status[out$engine == "spin"], "capped")
  expect_identical(out$status[out$engine == "boom"], "error")
  expect_match(out$detail[out$engine == "spin"], "over 0.2 s", fixed = TRUE)
  # the capped row keeps the elapsed time at the abort (review N27): at least the cap, far
  # below the call's 4 s; it is not ok and not quiet, so no median can use it
  sp <- out[out$engine == "spin", ]
  expect_gte(sp$secs, 0.2)
  expect_lt(sp$secs, 2)
  expect_false(sp$ok)
  expect_false(sp$quiet)
  # an error that is not the cap keeps secs NA
  expect_true(is.na(out$secs[out$engine == "boom"]))
  # the limit is gone after the cell: a long call outside it is not aborted
  t0 <- proc.time()[["elapsed"]]
  expect_no_error(while (proc.time()[["elapsed"]] - t0 < 0.5) NULL)
})

# ---- round 3: the fallback watch, the idle probe, dplyr rowwise in its own cells (review B1, N26, N27) ----

f121_fake_fallback <- function(reason) {
  structure(class = c("cuplyr_cpu_fallback", "condition"),
            list(message = paste("fallback:", reason), call = NULL, reason = reason, verb = "mutate"))
}

test_that("pp_fallback_watch() records a fallback condition per call, in call order, without changing the result", {
  pp <- f121_load(character())
  i <- 0L
  fn <- function() {
    i <<- i + 1L
    if (i %in% c(2L, 4L)) signalCondition(f121_fake_fallback(paste("reason", i)))
    i
  }
  w <- pp$pp_fallback_watch(fn)
  expect_identical(w$calls(), character())
  expect_identical(vapply(1:4, function(j) w$fn(), 1L), 1:4)
  expect_identical(w$calls(), c("", "mutate(): reason 2", "", "mutate(): reason 4"))
  # the watched function may replace the one it wraps (as the driver does, in a list):
  # the wrapped function is taken when the watch is made, not at its first call
  fl <- list(a = function() 1L)
  wl <- pp$pp_fallback_watch(fl[["a"]])
  fl[["a"]] <- wl$fn
  expect_identical(fl$a(), 1L)
  expect_identical(wl$calls(), "")
  # a call that fails is recorded too
  w2 <- pp$pp_fallback_watch(function() { signalCondition(f121_fake_fallback("then failed")); stop("boom") })
  expect_error(w2$fn(), "boom")
  expect_identical(w2$calls(), "mutate(): then failed")
  # inside pp_time_cell() an engine's rows come in call order, so the records line up with them
  j <- 0L
  g <- function() { j <<- j + 1L; if (j == 3L) signalCondition(f121_fake_fallback("third")); j }
  w3 <- pp$pp_fallback_watch(g)
  out <- pp$pp_time_cell(list(other = function() 0, watched = w3$fn), reps = 3L,
                         busy_fn = function(c0, c1) c(total = 0, own = 0, kernel = 0, foreign = 0), wait = NULL)
  rows <- out[out$engine == "watched", ]
  expect_identical(length(w3$calls()), nrow(rows))
  expect_identical(nzchar(w3$calls()), c(FALSE, FALSE, TRUE, FALSE))
  expect_identical(rows$rep, 0:3)
})

test_that("pp_idle_probe() primes and then idles before every call, records the gap and the GPU state, and re-runs nothing", {
  pp <- f121_load(character())
  slept <- numeric()
  st <- 0L
  state <- function() { st <<- st + 1L; list(pstate = paste0("P", st), sm_mhz = 1, mem_mhz = 100 * st) }
  # every timed call follows an untimed call of the same function and then the pause, so
  # that the GPU has been idle for the pause whichever function ran before (review B1)
  seq_calls <- character()
  a <- function() { seq_calls <<- c(seq_calls, "a"); Sys.sleep(0.01) }
  out <- pp$pp_idle_probe(list(a = a, b = function() { seq_calls <<- c(seq_calls, "b"); stop("boom") }), idles = c(0, 0.2), reps = 2L,
                          wait = NULL, sleep = function(s) { slept <<- c(slept, s); Sys.sleep(s) }, state = state,
                          busy_fn = function(c0, c1) c(total = 0, own = 0, kernel = 0, foreign = 0.5), busy_max = 0.15)
  expect_identical(nrow(out), 2L * 2L * 2L)
  expect_identical(seq_calls, rep(c("a", "a", "b", "b"), 4L))
  expect_true(all(out$primed))
  expect_identical(slept, rep(rep(c(0, 0.2), each = 2L), 2L))
  expect_identical(out$name, rep(c("a", "b"), 4L))
  expect_identical(out$idle, rep(rep(c(0, 0.2), each = 2L), 2L))
  expect_identical(out$rep, rep(1:2, each = 4L))
  # the gap runs from the end of the untimed call to the start of the timed one, so it holds the
  # pause and little else
  expect_true(all(out$gap_s >= out$idle))
  expect_true(all(out$gap_s[out$name == "a"] - out$idle[out$name == "a"] < 0.15))
  a <- out[out$name == "a", ]
  expect_true(all(a$ok) && all(a$secs >= 0.01))
  # the load is recorded and judged, never re-run
  expect_true(all(a$busy == 0.5) && !any(a$quiet))
  b <- out[out$name == "b", ]
  expect_true(all(!b$ok) && all(b$status == "error") && all(is.na(b$secs)))
  # the state is read just before and just after every call
  expect_identical(out$pstate_before, paste0("P", seq(1L, 15L, by = 2L)))
  expect_identical(out$mem_mhz_after, 100 * seq(2L, 16L, by = 2L))
  # pp_gpu_state() always gives the three fields
  s <- pp$pp_gpu_state()
  expect_named(s, c("pstate", "sm_mhz", "mem_mhz"))
})

test_that("pp_fallback_probe() reports an error that is not a fallback, and keeps the error", {
  skip_if_no_gpu()
  pp <- f121_load(character())
  v <- pp$pp_fallback_probe(function() stop("boom"))
  expect_true(is.na(v$native))
  expect_match(v$reason, "boom")
  expect_s3_class(v$error, "error")
  ok <- pp$pp_fallback_probe(function() 1)
  expect_true(ok$native)
  expect_null(ok$error)
  # under cuplyr.fallback = "error" the fallback still reads as a fallback, and the option is left as it was
  withr::local_options(cuplyr.fallback = "error")
  g <- tbl_gpu(data.frame(a = c(1.234, 5.678)))
  fb <- pp$pp_fallback_probe(function() collect(mutate(g, b = signif(a, 2))))
  expect_false(fb$native)
  expect_identical(getOption("cuplyr.fallback"), "error")
})

test_that("the join with the dimension computed first reads the fact file, and gives the pipeline's result (review N12)", {
  skip_if_no_gpu()
  pp <- f121_load()
  fx <- f121_fixture(pp)
  k <- pp$pp_constants()
  f <- as.data.frame(arrow::read_parquet(fx$narrow))
  d <- as.data.frame(arrow::read_parquet(fx$dim))
  withr::local_options(cuplyr.fallback = "error")
  dm <- dplyr::compute(cuplyr::read_parquet_gpu(fx$dim))
  for (v in c("p2_push", "p2_nopush")) {
    nz <- pp$pp_normalize(pp$pp_cuplyr_under_join(v, fx$narrow, dm), v)
    expect_true(pp$pp_compare(nz, f121_base_ref(v, f, d, k))$ok, label = v)
    lr <- utils::getFromNamespace("parquet_last_read", "cuplyr")()
    expect_identical(basename(lr$paths[[1L]]), basename(fx$narrow))
  }
  lr <- utils::getFromNamespace("parquet_last_read", "cuplyr")()
  expect_setequal(lr$columns, c("price", "qty", "discount", "channel", "store"))
})

test_that("the driver's rowwise steps check, probe, time and cap at a small scale", {
  skip_if_no_gpu()
  if (!identical(Sys.getenv("CUPLYR_BENCH_RENDER"), "1")) skip("set CUPLYR_BENCH_RENDER=1 to run the driver")
  pp <- f121_load(c("arrow", "duckdb", "DBI", "data.table", "callr"))
  d <- f121_tmpdir()
  if (f121_is_tmpfs(d)) skip("scratch directory is on tmpfs; cold reps cannot be observed")
  root <- normalizePath(testthat::test_path("..", ".."))
  driver <- file.path(root, "bench", "parquet-pipeline.R")
  run <- function(pp_list) {
    assign("PP", pp_list, envir = globalenv())
    on.exit(rm("PP", envir = globalenv()), add = TRUE)
    withr::with_dir(root, source(driver, local = new.env(parent = globalenv())))
  }
  sizes <- c(10000, 20000)
  cfg <- list(steps = "gen", sizes = sizes, tables = "wide", row_group_rows = 5000L, data_dir = file.path(d, "data"),
              out_dir = file.path(d, "out"), root = root, lib = f121_bench_lib(), busy_max = 1)
  run(cfg)
  run(modifyList(cfg, list(steps = c("rowwise_check", "rowwise"), rowwise_sizes = sizes, rowwise_reps = 2L, cap = 1.5,
                           rowwise_warm_rows = sizes[[1L]], warmups = 1L)))
  out <- cfg$out_dir
  rd <- function(f) utils::read.csv(file.path(out, f), stringsAsFactors = FALSE)
  cr <- rd("rowwise-correctness.csv")
  expect_setequal(names(cr), c("form", "rows", "status", "detail", "result_rows"))
  expect_identical(nrow(cr), 2L * length(pp$pp_rowwise_forms()))
  # every vectorised form and cuplyr agree with the base-R reference; the per-row dplyr form is over the cap
  # at the first size and is not run at the second
  expect_true(all(cr$status[cr$form != "dplyr_rowwise"] == "ok"), label = paste(cr$detail, collapse = "; "))
  expect_identical(cr$status[cr$form == "dplyr_rowwise"], c("capped", "not_run"))
  expect_match(cr$detail[cr$form == "dplyr_rowwise" & cr$status == "not_run"], "capped at 10,000 rows", fixed = TRUE)
  nat <- rd("rowwise-native.csv")
  expect_identical(nat$rows, as.integer(sizes))
  expect_true(all(c("native", "reason") %in% names(nat)))
  tm <- rd("rowwise.csv")
  expect_true(all(c("form", "table", "rows", "cache", "rep", "secs", "busy", "resident", "quiet", "ok", "status", "cap") %in% names(tm)))
  expect_true(all(tm$table == "wide"))
  for (f in setdiff(pp$pp_rowwise_forms(), "dplyr_rowwise")) for (n in sizes) for (cache in c("warm", "cold")) {
    x <- tm[tm$form == f & tm$rows == n & tm$cache == cache, ]
    expect_true(0L %in% x$rep, label = paste(f, n, cache, "rep 0"))
    expect_gte(sum(x$rep >= 1L & x$quiet & x$ok), 2L, label = paste(f, n, cache))
    expect_true(all(if (cache == "cold") x$resident <= 0.01 else x$resident >= 0.99), label = paste(f, n, cache, "residency"))
  }
  dp <- tm[tm$form == "dplyr_rowwise", ]
  expect_true(all(dp$status[dp$rows == sizes[[1L]]] == "capped"))
  expect_true(all(dp$status[dp$rows == sizes[[2L]]] == "not_run"))
  # a capped call keeps its elapsed time at the abort (review N27); a form not run has none
  expect_true(all(dp$secs[dp$status == "capped"] >= 1.5))
  expect_true(all(is.na(dp$secs[dp$status == "not_run"])))
  # dplyr rowwise is timed in a cell of its own, the four other forms together (review B1):
  # no dplyr rowwise call starts while the shared cell runs
  fast <- setdiff(pp$pp_rowwise_forms(), "dplyr_rowwise")
  expect_true(all(dp$cell_forms[dp$status != "not_run"] == "dplyr_rowwise"))
  expect_true(all(tm$cell_forms[tm$form %in% fast] == paste(fast, collapse = "+")))
  ts <- function(x) as.numeric(as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%OS"))
  for (cache in c("warm", "cold")) {
    sh <- tm[tm$form %in% fast & tm$rows == sizes[[1L]] & tm$cache == cache, ]
    own <- dp[dp$rows == sizes[[1L]] & dp$cache == cache & dp$status != "not_run", ]
    expect_false(any(ts(own$started) > min(ts(sh$started)) & ts(own$started) < max(ts(sh$started))), label = paste(cache, "no dplyr rowwise call inside the shared cell"))
  }
  # every timed cuplyr call was watched for a fallback (review N26); the other forms were not
  expect_true(all(c("fallback", "fallback_reason") %in% names(tm)))
  cu <- tm[tm$form == "cuplyr_rowwise", ]
  expect_true(all(!is.na(cu$fallback)))
  expect_false(any(as.logical(cu$fallback)))
  expect_true(all(is.na(tm$fallback[tm$form != "cuplyr_rowwise"])))
  expect_true(file.exists(file.path(out, "rowwise-warmup.csv")))
  # the idle step: every function after every idle time, with the GPU state around each call
  run(modifyList(cfg, list(steps = "idle", idle_rows = sizes[[1L]], idle_secs = c(0, 0.5), idle_reps = 1L, warmups = 1L)))
  il <- rd("idle.csv")
  expect_identical(nrow(il), 3L * 2L)
  expect_setequal(il$name, c("cuplyr_rowwise", "cuplyr_scan", "duckdb"))
  expect_true(all(il$ok), label = paste(il$detail, collapse = "; "))
  expect_true(all(c("idle", "gap_s", "pstate_before", "mem_mhz_before", "pstate_after", "mem_mhz_after") %in% names(il)))
  expect_true(all(il$gap_s[il$idle == 0.5] >= 0.5))
  pv <- rd("provenance.csv")
  expect_true(all(c("rowwise_check", "rowwise", "idle") %in% pv$step))
})

# ---- round 4: the report's names, and what the method bullet renders to (review 3b B1, N41) ----

# The names the executed chunks assign in the report's own environment, where every inline `r`
# expression looks them up: assignments outside function bodies and local(), and for-loop variables.
f121_top_names <- function(lines) {
  p <- f63_parse_qmd(lines)
  found <- character()
  walk <- function(e) {
    if (!is.call(e)) return(invisible())
    f <- e[[1L]]
    if (identical(f, quote(`function`)) || identical(f, quote(local)) || identical(f, quote(`~`)) || identical(f, quote(quote))) return(invisible())
    if ((identical(f, quote(`<-`)) || identical(f, quote(`=`))) && is.symbol(e[[2L]])) found <<- c(found, as.character(e[[2L]]))
    if (identical(f, quote(`for`))) found <<- c(found, as.character(e[[2L]]))
    for (a in as.list(e)[-1L]) if (!missing(a)) walk(a)
  }
  for (ch in p$chunks) for (e in parse(text = ch$code, keep.source = FALSE)) walk(e)
  found
}

test_that("the report assigns the method bullet's reps once and keeps the replicate's names to itself", {
  lines <- readLines(f121_file("bench/parquet-pipeline.qmd"), warn = FALSE)
  top <- f121_top_names(lines)
  # the walker sees what it should: a plain assignment, one inside if/braces, a loop variable
  expect_true(all(c("rep_text", "rw_sup_text", "idle_tab", "nm") %in% top))
  # rep_text is the reps per size of the main method bullet, assigned once, and that bullet prints it
  expect_identical(sum(top == "rep_text"), 1L)
  expect_identical(sum(grepl("\\brep_text\\s*<-", lines, perl = TRUE)), 1L)
  bullet <- grep("the number of quiet reps its size needs", lines, fixed = TRUE, value = TRUE)
  expect_length(bullet, 1L)
  expect_match(bullet, "needs (`r rep_text`)", fixed = TRUE)
  # the replicate paragraph's working names stay inside its local()
  expect_false(any(c("rw_rep_text", "cur", "both", "side", "dk", "ex", "rp", "rpq", "lowest", "spread_text", "chance", "n_unmatched") %in% top))
  # names that held two different things at the top level (review 3b N41 and the round-4 sweep)
  for (nm in c("best_of", "g", "pt", "rsum", "nm")) expect_lte(sum(top == nm), 1L, label = nm)
})

test_that("the rendered report prints the reps per size in the method bullet and the replicate paragraph once", {
  if (!identical(Sys.getenv("CUPLYR_BENCH_RENDER"), "1")) skip("set CUPLYR_BENCH_RENDER=1 to render the report")
  quarto <- f63_quarto()
  if (!nzchar(quarto)) skip("quarto CLI not found")
  data <- f121_data_root()
  if (!all(file.exists(file.path(data, c("timings.csv", "rowwise.csv", "rowwise-superseded.csv"))))) skip("the measured data are not here")
  qmd <- f121_file("bench/parquet-pipeline.qmd")
  root <- normalizePath(testthat::test_path("..", ".."))
  # a staged copy, as the driver's render step does; the report only reads `data`
  stage <- withr::local_tempdir("f121-render-")
  file.copy(qmd, file.path(stage, "parquet-pipeline.qmd"))
  log <- suppressWarnings(system2(quarto, c("render", shQuote(file.path(stage, "parquet-pipeline.qmd")),
                                            "-P", shQuote(paste0("root:", root)), "-P", shQuote(paste0("lib:", f121_bench_lib())),
                                            "-P", shQuote(paste0("out_dir:", data))), stdout = TRUE, stderr = TRUE))
  html <- file.path(stage, "parquet-pipeline.html")
  expect_true(file.exists(html), label = paste(utils::tail(log, 20L), collapse = "\n"))
  if (!file.exists(html)) return(invisible())
  h <- paste(readLines(html, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  h <- gsub("(?s)<(script|style|pre)\\b.*?</\\1>", " ", h, perl = TRUE)
  txt <- gsub("\\s+", " ", gsub("<[^>]+>", " ", h))
  need <- regmatches(txt, regexpr("the number of quiet reps its size needs \\([^)]*\\)", txt))
  expect_length(need, 1L)
  expect_match(need, "needs \\(1e[0-9] rows: [0-9]+")
  expect_identical(sum(gregexpr("and so is a replicate of them", txt, fixed = TRUE)[[1L]] > 0L), 1L)
  expect_false(grepl("\\b(NA|NaN)\\b", txt), label = "an NA or NaN in the rendered text")
})
