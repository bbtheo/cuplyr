# F54 round 2: pins of the benchmark harness that the acceptance file leaves
# open (bench/rolling-lib.R): gc() before every call and outside the timed
# region, the cut logic, the runstats tag/normalize split, and the thread
# controls behind the single-thread entries.

rw_extra_lib_path <- function() testthat::test_path("..", "..", "bench", "rolling-lib.R")

load_extra_lib <- function(env = parent.frame()) {
  if (!file.exists(testthat::test_path("..", "..", "DESCRIPTION"))) skip("not in a source checkout; bench/ is not shipped")
  path <- rw_extra_lib_path()
  expect_true(file.exists(path))
  lib <- Sys.getenv("CUPLYR_BENCH_LIB", "/home/theo/cuplr/scratchpad/benchmarks/rlib")
  if (dir.exists(lib)) withr::local_libpaths(lib, action = "prefix", .local_envir = env)
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

test_that("rw_time() runs gc() before every call and keeps it out of the timed region", {
  rw <- load_extra_lib()
  events <- character()
  # rw_time() looks gc up in its enclosing environment: shadow it there.
  rw$gc <- function(...) {
    events <<- c(events, "gc")
    Sys.sleep(0.05)
    invisible(NULL)
  }
  f <- function() events <<- c(events, "fn")
  t <- rw$rw_time(f, times = 3, warmup = 2)
  expect_identical(events, rep(c("gc", "fn"), 5))
  # 50 ms per gc: none of it lands in a sample.
  expect_true(all(t < 40), info = paste(round(t, 1), collapse = ", "))
})

test_that("rw_plan_cell() records exactly what it says it times", {
  rw <- load_extra_lib()

  fast <- rw$rw_plan_cell(0.01, times = 20, cell_budget = 20, probe_limit = 60)
  expect_identical(fast$k, 20)
  expect_identical(fast$warmup, 2)
  expect_false(fast$probe_is_sample)
  expect_identical(fast$note, "")

  # A 6.2 s call: 3 samples fit in a 20 s budget, one warmup.
  cut <- rw$rw_plan_cell(6.2, times = 20, cell_budget = 20, probe_limit = 60)
  expect_identical(cut$k, 3)
  expect_identical(cut$warmup, 1)
  expect_false(cut$probe_is_sample)
  expect_match(cut$note, "times cut 20 -> 3")
  expect_match(cut$note, "cold probe call 6.2 s, not recorded")
  expect_match(cut$note, "1 untimed warmup")
  expect_match(cut$note, "3 timed")

  # A 32 s call (under the probe limit): one timed sample, no warmup.
  slow <- rw$rw_plan_cell(32, times = 20, cell_budget = 20, probe_limit = 60)
  expect_identical(slow$k, 1)
  expect_identical(slow$warmup, 0)
  expect_false(slow$probe_is_sample)

  # Over the limit: the cold call itself is the sample, and says so.
  cold <- rw$rw_plan_cell(90, times = 20, cell_budget = 20, probe_limit = 60)
  expect_identical(cold$k, 1)
  expect_true(cold$probe_is_sample)
  expect_match(cold$note, "cold first call")
})

test_that("runstats closures only tag; rw_normalize() shifts outside timing", {
  rw <- load_extra_lib()
  skip_if_not_installed("runstats")
  skip_if_not_installed("data.table")
  fns <- rw$rw_functions()
  set.seed(7)
  x <- rnorm(500)
  n <- 11
  for (stat in c("mean", "sd")) {
    raw <- fns[[stat]]$runstats(x, n)
    expect_identical(attr(raw, "rw_align"), "left")
    expect_identical(attr(raw, "rw_window"), n)
    # Left-aligned as the package returns it: the last n - 1 values are NA.
    expect_true(all(is.na(utils::tail(as.numeric(raw), n - 1))))
    expect_false(anyNA(as.numeric(raw)[seq_len(length(x) - n + 1)]))
    out <- rw$rw_normalize(raw, length(x))
    expect_null(attributes(out))
    expect_true(all(is.na(out[seq_len(n - 1)])))
    expect_true(rw$rw_compare(out, rw$rw_reference(stat, x, n))$ok, info = stat)
  }
  # Untagged output is not shifted.
  expect_identical(rw$rw_normalize(c(1, 2, 3, NA, NA), 5), c(1, 2, 3, NA, NA))
})

test_that("thread controls set and restore data.table and RcppRoll, and label twins", {
  rw <- load_extra_lib()
  skip_if_not_installed("data.table")
  before <- data.table::getDTthreads()
  restore <- rw$rw_set_threads("data.table", 1)
  expect_identical(data.table::getDTthreads(), 1L)
  restore()
  expect_identical(data.table::getDTthreads(), before)

  skip_if_not_installed("RcppRoll")
  before_opt <- getOption("RcppRoll.threads")
  restore <- rw$rw_set_threads("RcppRoll", 1)
  expect_identical(as.integer(RcppRoll::roll_threads()), 1L)
  restore()
  expect_identical(getOption("RcppRoll.threads"), before_opt)

  expect_error(rw$rw_set_threads("slider", 1), "no thread control")

  expect_identical(rw$rw_threaded_pkg("data.table"), "data.table")
  expect_identical(rw$rw_threaded_pkg("data.table (exact)"), "data.table")
  expect_identical(rw$rw_threaded_pkg("data.table::frollapply"), "data.table")
  expect_identical(rw$rw_threaded_pkg("RcppRoll"), "RcppRoll")
  expect_identical(rw$rw_threaded_pkg("roll"), "roll")
  expect_true(is.na(rw$rw_threaded_pkg("slider")))
  expect_true(is.na(rw$rw_threaded_pkg("RollingWindow")))
  for (nm in c("RcppRoll", "data.table (exact)", "data.table (by = g)")) {
    expect_identical(rw$rw_base_label(rw$rw_one_thread_label(nm)), nm)
  }
  expect_identical(rw$rw_one_thread_label("RcppRoll"), "RcppRoll (1 thread)")
  expect_identical(rw$rw_one_thread_label("data.table (exact)"), "data.table (exact, 1 thread)")
})
