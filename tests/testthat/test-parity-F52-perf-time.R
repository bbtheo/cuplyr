# F52 acceptance tests: perf_time()'s batching guarantee must not depend on
# the probe batch being representative.
#
# perf_time() sizes `reps` from a short probe run right after warmup (and after
# gc_gpu(), which sleeps 100 ms). If the probe measures the call as slower than
# the timed samples later see, `reps` comes out too small and every timed
# sample is shorter than `min_sample_s`. The merge gate hit this twice with
# Sys.sleep(0.0025): reps * median = 0.0440 < 0.045 (test-parity-F00-gate.R).
#
# These tests make the probe unrepresentative on purpose: the closure counts
# its calls and sleeps longer for the first few (warmup + probe), then settles
# to its steady per-call time. Whatever perf_time() does internally, the
# samples it reports must each be about `min_sample_s` long at the steady
# per-call time.

# A closure that sleeps `slow_s` for its first `n_slow` calls, then `fast_s`.
f52_slow_start <- function(n_slow, slow_s, fast_s) {
  calls <- 0L
  function() {
    calls <<- calls + 1L
    Sys.sleep(if (calls <= n_slow) slow_s else fast_s)
  }
}

test_that("perf_time() keeps samples >= ~min_sample_s when the probe overestimates the call", {
  # warmup (1 call) and the doubling probe (1 + 2 calls) all land in the slow
  # phase: the probe reads ~8 ms per call while steady calls take ~2.6 ms.
  fn <- f52_slow_start(n_slow = 4L, slow_s = 0.008, fast_s = 0.0025)
  t <- perf_time(fn, iterations = 3, warmup = 1)
  reps <- attr(t, "reps")
  expect_true(is.numeric(reps) && length(reps) == 1)
  # The reported per-call time is the steady one, not the probe's.
  expect_lt(as.numeric(t), 0.0035)
  expect_gt(as.numeric(t), 0.0024)
  # Same bound as the F00 gate test: one timed sample is ~50 ms.
  expect_gte(reps * as.numeric(t), 0.045)
})

test_that("perf_time() keeps samples >= ~min_sample_s for a longer slow start", {
  # Slow phase covers warmup, the whole probe and more: whatever the probe
  # reads, the reported samples must be sized for the steady per-call time.
  fn <- f52_slow_start(n_slow = 12L, slow_s = 0.006, fast_s = 0.0025)
  t <- perf_time(fn, iterations = 5, warmup = 2)
  reps <- attr(t, "reps")
  expect_lt(as.numeric(t), 0.0035)
  expect_gte(reps * as.numeric(t), 0.045)
})

test_that("perf_time() honours a non-default min_sample_s", {
  fn <- f52_slow_start(n_slow = 4L, slow_s = 0.008, fast_s = 0.0025)
  t <- perf_time(fn, iterations = 3, warmup = 1, min_sample_s = 0.03)
  expect_gte(attr(t, "reps") * as.numeric(t), 0.027)
})

test_that("perf_time() still times slow calls one per sample", {
  # The probe reads >= min_sample_s, so the call is not batched; it stays
  # unbatched even if later calls come in faster (slow benchmarks must time
  # exactly as before).
  fn <- f52_slow_start(n_slow = 1L, slow_s = 0.06, fast_s = 0.03)
  t <- perf_time(fn, iterations = 3, warmup = 0)
  expect_equal(attr(t, "reps"), 1)
  expect_gt(as.numeric(t), 0.029)
  expect_lt(as.numeric(t), 0.04)
})

test_that("perf_time(batch = FALSE) never batches", {
  fn <- f52_slow_start(n_slow = 1L, slow_s = 0.008, fast_s = 0.0025)
  t <- perf_time(fn, iterations = 3, warmup = 1, batch = FALSE)
  expect_equal(attr(t, "reps"), 1)
  expect_lt(as.numeric(t), 0.0045)
})
