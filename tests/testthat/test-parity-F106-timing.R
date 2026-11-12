# F106 acceptance tests: the shared timing helper behind the suite's
# sub-millisecond timing tests (helper-timing.R).
#
# Three timing tests failed the suite and a merge gate at random on unchanged
# code (scratchpad/parity/features/F106.md). Their causes differ:
#   - F68a/F68b timed with proc.time(), which is rounded down to whole
#     milliseconds (?proc.time), and sized each closure's batch from ONE call
#     timed with it: a sub-ms call read 0 or 1 ticks, so batches came out at
#     ~25 ms or ~250 ms instead of the documented 50 ms (F68b) and read in
#     steps of up to 4%, and one slow probe call (a GC, a descheduling) made
#     them shorter still (F68a's 2.22 ratio fits that);
#   - F68b's eager verbs sit 2-5% under their 0.5 ms bound, so a run that is
#     a few percent slow fails (handled by a bounded near-miss re-measure,
#     timing_retry(); a sustained slowdown still fails);
#   - F00's fixture was Sys.sleep(), whose overshoot is the kernel's; it is
#     now a busy-wait on the wall clock (timing_spin()). perf_time() itself
#     is unchanged.
# The helper times with a sub-microsecond clock, sizes each batch from a probe
# that itself lasts long enough to read, and re-sizes when the timed batches
# come in short. The bounds of the tests that use it are unchanged.

test_that("timing_spin() busy-waits for the time it is given", {
  for (sec in c(2e-4, 2e-3)) {
    a <- as.numeric(Sys.time())
    timing_spin(sec)
    b <- as.numeric(Sys.time())
    expect_gte(b - a, sec)
  }
  expect_invisible(timing_spin(0))
})

test_that("timing_now() resolves well below a millisecond", {
  expect_true(is.numeric(timing_now()) && length(timing_now()) == 1L)
  a <- timing_now()
  repeat {
    b <- timing_now()
    if (b > a) break
  }
  expect_lt(b - a, 1e-4)
  a <- timing_now()
  timing_spin(0.002)
  b <- timing_now()
  expect_gte(b - a, 0.002)
  expect_lt(b - a, 0.05)
})

test_that("timing_medians() returns one median per closure with its batch sizes and samples", {
  skip_if_no_gpu()
  t <- timing_medians(list(a = function() timing_spin(2e-4), b = function() timing_spin(4e-4)),
                      samples = 5L, target = 0.005)
  expect_named(t, c("a", "b"))
  expect_true(is.numeric(t))
  reps <- attr(t, "reps")
  expect_named(reps, c("a", "b"))
  expect_true(all(reps >= 1))
  s <- attr(t, "samples")
  expect_true(is.matrix(s))
  expect_equal(dim(s), c(5L, 2L))
  expect_equal(colnames(s), c("a", "b"))
  expect_equal(as.numeric(t), as.numeric(apply(s, 2, stats::median)))
})

test_that("timing_medians() resolves a 0.05 ms difference with 5 ms batches", {
  # With proc.time() a 5 ms batch of 0.2 ms calls reads in steps of 0.04 ms.
  skip_if_no_gpu()
  t <- timing_medians(list(a = function() timing_spin(2e-4), b = function() timing_spin(2.5e-4)),
                      samples = 9L, target = 0.005)
  expect_gt(t[["b"]] - t[["a"]], 4e-5)
  expect_lt(t[["b"]] - t[["a"]], 6e-5)
  expect_gte(t[["a"]], 2e-4)
  expect_lt(t[["a"]], 2.4e-4)
})

test_that("timing_medians() sizes batches to their target despite slow first calls", {
  # The first 6 calls take 20 ms, later ones 0.2 ms (a GC or a descheduling
  # during warm-up/probing). Sizing from those calls gives 1-call batches of
  # 0.2 ms, read by any clock with poor relative precision.
  skip_if_no_gpu()
  calls <- 0L
  slow_start <- function() {
    calls <<- calls + 1L
    timing_spin(if (calls <= 6L) 0.02 else 2e-4)
  }
  target <- 0.02
  t <- timing_medians(list(x = slow_start), samples = 5L, target = target)
  # the helper re-sizes batches that come in under half their target
  expect_gte(attr(t, "reps")[["x"]] * t[["x"]], 0.5 * target)
  expect_lt(attr(t, "reps")[["x"]] * t[["x"]], 3 * target)
  expect_lt(t[["x"]], 2.5e-4)
})

test_that("timing_retry() re-measures only near misses, and a bounded number of times", {
  seq_measure <- function(vals) {
    i <- 0L
    function() {
      i <<- i + 1L
      vals[[i]]
    }
  }
  ok <- function(v) v < 0.5
  near <- function(v) v < 0.55
  # passes first time: one measurement
  expect_equal(timing_retry(seq_measure(c(0.4, 0.1)), ok = ok, retry_if = near), list(0.4))
  # near miss, then a pass: stops at the pass
  expect_equal(timing_retry(seq_measure(c(0.52, 0.49, 0.1)), ok = ok, retry_if = near),
               list(0.52, 0.49))
  # far over the bound (a real regression): no re-measurement
  expect_equal(timing_retry(seq_measure(c(0.7, 0.1)), ok = ok, retry_if = near), list(0.7))
  # near misses every time: at most `attempts` measurements
  expect_equal(timing_retry(seq_measure(c(0.52, 0.53, 0.51, 0.1)), ok = ok, retry_if = near,
                            attempts = 3L),
               list(0.52, 0.53, 0.51))
  # a near miss followed by a far miss stops
  expect_equal(timing_retry(seq_measure(c(0.52, 0.9, 0.1)), ok = ok, retry_if = near),
               list(0.52, 0.9))
})

test_that("the sub-ms timing tests use the shared helper, not their own proc.time() timers", {
  for (f in c("test-parity-F68a-lazy-speed.R", "test-parity-F68b-floor.R")) {
    src <- readLines(test_path(f))
    code <- sub("#.*$", "", src)
    expect_false(any(grepl("proc.time", code, fixed = TRUE)), info = f)
    expect_true(any(grepl("timing_medians(", code, fixed = TRUE)), info = f)
  }
})
