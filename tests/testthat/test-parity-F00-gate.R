# F00 acceptance tests: merge-gate stability.
#
# These test the perf harness logic (helper-perf.R) and the build it runs
# against. They do not time real GPU pipelines; the real evidence for F00 is
# three consecutive green full gate runs.

# -----------------------------------------------------------------------------
# Timing resolution: perf_time() must resolve per-call times well below the
# 1 ms proc.time() tick, by timing samples that each run long enough.
# -----------------------------------------------------------------------------

test_that("perf_time() resolves sub-millisecond differences between calls", {
  # The fixture busy-waits on the wall clock, so each call lasts its time on any
  # core at any speed; only a descheduling across the deadline overshoots. It
  # replaces Sys.sleep(), whose overshoot is the kernel's (timer slack and
  # wake-up latency), differs between the two closures and between their two
  # sequential measurements: solo dev runs read a long - short difference of
  # 0.748 and 0.727 ms against the 0.75 ms cap (F106). The pair is measured 3
  # times, interleaved, and the median difference is asserted: a single-call
  # 1 ms timer reports 0.002 vs 0.003 here (difference 1 ms) or two equal
  # values (difference 0), and a constant per-call overhead in perf_time()
  # shows in `short`.
  short <- numeric(3)
  long <- numeric(3)
  raw <- perf_time(function() timing_spin(0.0022), iterations = 5, warmup = 1)
  expect_true(is.numeric(raw) && length(raw) == 1L)
  for (i in 1:3) {
    short[[i]] <- as.numeric(perf_time(function() timing_spin(0.0022), iterations = 5, warmup = 1))
    long[[i]] <- as.numeric(perf_time(function() timing_spin(0.0027), iterations = 5, warmup = 1))
  }
  diff_ms <- 1e3 * (long - short)
  d <- stats::median(long - short)
  s <- stats::median(short)
  lbl <- sprintf("long - short per pair: %s ms; short per pair: %s ms",
                 paste(sprintf("%.3f", diff_ms), collapse = " / "),
                 paste(sprintf("%.3f", 1e3 * short), collapse = " / "))
  expect_gt(d, 0.00025, label = lbl)
  expect_lt(d, 0.00075, label = lbl)
  expect_gt(s, 0.0021, label = lbl)
  # timing_spin(0.0022) is ~2.2 ms; 0.8 ms headroom for overshoot under load, still
  # tight enough to catch a constant per-call overhead in perf_time().
  expect_lt(s, 0.0030, label = lbl)
})

test_that("perf_time() batches short calls so each timed sample is >= ~50 ms", {
  t <- perf_time(function() Sys.sleep(0.0025), iterations = 3, warmup = 1)
  reps <- attr(t, "reps")
  expect_true(is.numeric(reps) && length(reps) == 1)
  expect_gte(reps, 2)
  # reps calls of ~2.6 ms each make up one timed sample.
  expect_gte(reps * as.numeric(t), 0.045)
})

test_that("perf_time() does not batch calls that are already long", {
  t <- perf_time(function() Sys.sleep(0.06), iterations = 3, warmup = 0)
  expect_equal(attr(t, "reps"), 1)
  expect_gt(as.numeric(t), 0.059)
  expect_lt(as.numeric(t), 0.07)
})

test_that("perf_time() still validates its arguments", {
  expect_error(perf_time("not a function"))
  expect_error(perf_time(function() NULL, iterations = 0))
  expect_error(perf_time(function() NULL, warmup = -1))
})

# -----------------------------------------------------------------------------
# Calibration scale bounds.
# -----------------------------------------------------------------------------

test_that("a faster calibration session never tightens the threshold", {
  # 2026-10-09: the kernel probe measured 0.046 s against a 0.065 s
  # baseline (raw scale 0.708, identical in three runs) while every
  # benchmark's raw ratio was 0.93-1.11. Scaling by 0.708 pushed
  # filter_chain_* over 1.5x on unchanged code.
  expect_identical(perf_effective_scale(0.708), 1)
  expect_identical(perf_effective_scale(0.1), 1)
  expect_identical(perf_effective_scale(1), 1)
})

test_that("a moderately slower calibration session widens the threshold", {
  expect_equal(perf_effective_scale(1.2), 1.2)
})

test_that("an implausibly slow calibration session fails loudly instead of widening or skipping", {
  # A probe at 3x (what the transfer probe showed on a -O0 build) must not
  # silently widen every threshold 3x, and must not skip the check either:
  # a skipped perf test passes the gate.
  expect_error(perf_effective_scale(2.5), class = "cuplyr_perf_calibration_error")
  expect_error(perf_effective_scale(10), class = "cuplyr_perf_calibration_error")
})

test_that("invalid calibration scales are errors", {
  expect_error(perf_effective_scale(NA_real_), class = "cuplyr_perf_calibration_error")
  expect_error(perf_effective_scale(0), class = "cuplyr_perf_calibration_error")
  expect_error(perf_effective_scale(-1), class = "cuplyr_perf_calibration_error")
  expect_error(perf_effective_scale(Inf), class = "cuplyr_perf_calibration_error")
})

test_that("perf_threshold() is baseline * effective scale * max_ratio", {
  # The 2026-10-09 failure: baseline 0.010 s, raw kernel scale 0.708.
  expect_equal(perf_threshold(0.010, 0.708), 0.015)
  expect_equal(perf_threshold(0.010, 1.2), 0.018)
  expect_equal(perf_threshold(0.010, 1, max_ratio = 2), 0.020)
  expect_error(perf_threshold(0.010, 3), class = "cuplyr_perf_calibration_error")
})

test_that("the gate's max ratio stays 1.5", {
  expect_identical(eval(formals(expect_no_perf_regression)$max_ratio), 1.5)
  expect_identical(eval(formals(perf_threshold)$max_ratio), 1.5)
})

# -----------------------------------------------------------------------------
# Build: perf numbers from an unoptimized (-O0) build are meaningless.
# -----------------------------------------------------------------------------

test_that("the loaded cuplyr build is optimized", {
  # tools/wt-new and tools/build used pkgbuild's debug default, which adds
  # -O0: every worktree and the integration worktree ran 2.5-5x slower
  # than the -O2 build the perf baseline was recorded with.
  opt <- perf_build_is_optimized()
  expect_true(is.logical(opt) && length(opt) == 1 && !is.na(opt))
  expect_true(opt)
})

test_that("perf_build_is_optimized() scans the producer flags of the file it is given", {
  # Round-2 review N1/N2: the scan itself must be tested, and a file with no
  # compiler producer strings at all (no -g, clang, compressed debug info)
  # must not pass vacuously.
  write_fake_so <- function(text) {
    path <- withr::local_tempfile(fileext = ".so", .local_envir = parent.frame())
    writeBin(c(as.raw(c(0x7f, 0x45, 0x4c, 0x46, 0x00)), charToRaw(text), as.raw(0x00)), path)
    path
  }
  opt <- write_fake_so("GNU C++20 14.3.0 -march=x86-64 -g -O2 -std=gnu++20 -fpic")
  mixed <- write_fake_so(paste0(
    "GNU C++20 14.3.0 -march=x86-64 -g -O2 -std=gnu++20 -fpic", "\x01",
    "GNU C++20 14.3.0 -march=x86-64 -g -O2 -O0 -std=gnu++20 -fpic"
  ))
  no_producer <- write_fake_so("just some bytes without compiler flags")
  expect_true(perf_build_is_optimized(opt))
  expect_false(perf_build_is_optimized(mixed))
  expect_false(perf_build_is_optimized(no_producer))
  expect_false(perf_build_is_optimized(file.path(tempdir(), "no-such-cuplyr.so")))
  expect_false(perf_build_is_optimized(NA_character_))
})

test_that("perf_loaded_dll_path() picks the most recently loaded cuplyr DLL", {
  # Round-2 review B3: after an in-process load_all() reload, pkgload keeps
  # the old DLL loaded and getLoadedDLLs()[["cuplyr"]] returns the stale one.
  fake_dll <- function(name, path) {
    structure(list(name = name, path = path, dynamicLookup = FALSE), class = "DLLInfo")
  }
  dlls <- structure(list(
    base = fake_dll("base", "/r/base.so"),
    cuplyr = fake_dll("cuplyr", "/tmp/old/cuplyr.so"),
    Rcpp = fake_dll("Rcpp", "/r/Rcpp.so"),
    cuplyr = fake_dll("cuplyr", "/tmp/new/cuplyr.so")
  ), class = "DLLInfoList")
  expect_identical(perf_loaded_dll_path(dlls), "/tmp/new/cuplyr.so")
  none <- structure(list(base = fake_dll("base", "/r/base.so")), class = "DLLInfoList")
  expect_identical(perf_loaded_dll_path(none), NA_character_)
})

test_that("perf_check_build() refuses an unoptimized build", {
  expect_error(perf_check_build(optimized = FALSE), class = "cuplyr_perf_unoptimized_build")
  expect_no_error(perf_check_build(optimized = TRUE))
})

# -----------------------------------------------------------------------------
# End to end through expect_no_perf_regression(), with a fake baseline file
# and mocked probes (no GPU timing): the bounds above must actually be what
# the gate applies, and bad states must error, never skip (round-1 review N4).
# -----------------------------------------------------------------------------

local_fake_perf <- function(calib_s, transfer_s = calib_s, optimized = TRUE,
                            frame = parent.frame()) {
  path <- withr::local_tempfile(fileext = ".json", .local_envir = frame)
  perf_write_baseline(path, list(
    bench_k = list(median_s = 0.010, recorded_at = "x", commit = "x", iterations = 5),
    bench_t = list(median_s = 0.010, recorded_at = "x", commit = "x", iterations = 5),
    `_calibration` = list(median_s = 0.010, recorded_at = "x", commit = "x", iterations = 5),
    `_calibration_transfer` = list(median_s = 0.010, recorded_at = "x", commit = "x", iterations = 5)
  ))
  record_var <- paste0("CUPLYR_PERF_", "RECORD")
  withr::local_envvar(
    c(CUPLYR_PERF = "1", CUPLYR_PERF_BASELINE = path),
    .local_envir = frame
  )
  withr::local_envvar(stats::setNames(NA, record_var), .local_envir = frame)
  rlang::local_bindings(
    perf_warm_gpu = function(...) invisible(FALSE),
    perf_current_calibration = function(...) calib_s,
    perf_current_calibration_transfer = function(...) transfer_s,
    perf_build_is_optimized = function(...) optimized,
    .env = environment(expect_no_perf_regression),
    .frame = frame
  )
}

test_that("gate: the 2026-10-09 case (raw 1.1x, kernel scale 0.72) passes", {
  local_fake_perf(calib_s = 0.0072)
  expect_success(suppressMessages(
    expect_no_perf_regression("bench_k", function() Sys.sleep(0.011), calibration = "kernel")
  ))
})

test_that("gate: a real 1.7x regression fails even when the probe reads fast", {
  local_fake_perf(calib_s = 0.0072)
  expect_failure(suppressMessages(
    expect_no_perf_regression("bench_k", function() Sys.sleep(0.017), calibration = "kernel")
  ))
})

test_that("gate: a moderately slow transfer probe widens the transfer threshold", {
  # scale 1.2 -> threshold 0.010 * 1.2 * 1.5 = 0.018 s
  local_fake_perf(calib_s = 0.010, transfer_s = 0.012)
  expect_success(suppressMessages(
    expect_no_perf_regression("bench_t", function() Sys.sleep(0.016), calibration = "transfer")
  ))
})

test_that("gate: the kernel probe never widens kernel thresholds", {
  # Round-2 review B2 option (b): the kernel probe is bimodal by R process
  # state (~0.71 in the gate, 1.185-1.26 standalone or on a second
  # measurement), so it must not widen anything. A 1.6x regression with
  # the probe reading 1.3 fails as a plain perf failure, and does not error.
  local_fake_perf(calib_s = 0.013)
  expect_failure(suppressMessages(
    expect_no_perf_regression("bench_k", function() Sys.sleep(0.016), calibration = "kernel")
  ))
})

test_that("gate: the kernel probe's slow mode is not an error", {
  # 1.3 is above the slow mode's observed maximum (1.26); it must neither
  # widen nor error, so unchanged code passes deterministically in either mode.
  local_fake_perf(calib_s = 0.013)
  expect_success(suppressMessages(
    expect_no_perf_regression("bench_k", function() Sys.sleep(0.011), calibration = "kernel")
  ))
})

test_that("gate: a probe far above the cap errors instead of widening or skipping", {
  local_fake_perf(calib_s = 0.030)
  expect_error(
    suppressMessages(
      expect_no_perf_regression("bench_k", function() Sys.sleep(0.011), calibration = "kernel")
    ),
    class = "cuplyr_perf_calibration_error"
  )
})

test_that("gate: an unoptimized build errors before anything is timed", {
  local_fake_perf(calib_s = 0.010, optimized = FALSE)
  ran <- FALSE
  expect_error(
    expect_no_perf_regression("bench_k", function() ran <<- TRUE, calibration = "kernel"),
    class = "cuplyr_perf_unoptimized_build"
  )
  expect_false(ran)
})
