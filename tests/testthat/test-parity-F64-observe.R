# F64 acceptance tests (PM-owned): the perf harness records the gate's
# observations for the merge-queue ratchet.
#
# With CUPLYR_PERF_OBSERVATIONS=<path>, expect_no_perf_regression() appends
# one TSV row per benchmark: id, calibration, current_s, raw_scale, scale,
# observed_s, iterations, reps. observed_s = current_s / scale is the value
# the 1.5x check compares against the baseline (the calibrated median), so it
# is what the ratchet may store. An id with no baseline entry is timed and
# recorded too (then still skipped), so new benchmarks can enter the baseline.
#
# Temp baselines only: CUPLYR_PERF_BASELINE points at a tempfile. The
# calibration entries are set from this process's own (cached) probe readings,
# so the kernel raw scale is exactly 1 and the transfer raw scale is exactly
# the factor chosen below.

f64o_transfer_factor <- 1.1

f64o_setup <- function(env = parent.frame(), observations = TRUE, calibration = TRUE) {
  skip_if_no_gpu()
  skip_if_not(perf_build_is_optimized(), "perf timing needs an optimized (-O2) build; load through tools/")
  base <- withr::local_tempfile(fileext = ".json", .local_envir = env)
  obs <- withr::local_tempfile(fileext = ".tsv", .local_envir = env)
  withr::local_envvar(
    CUPLYR_PERF = "1",
    CUPLYR_PERF_BASELINE = base,
    CUPLYR_PERF_OBSERVATIONS = if (observations) obs else NA,
    .local_envir = env
  )
  kernel_probe <- as.numeric(perf_current_calibration())
  transfer_probe <- as.numeric(perf_current_calibration_transfer())
  rec <- function(s) list(median_s = s, recorded_at = "2026-10-23T12:00:00+0000", commit = "abc1234", iterations = 5)
  entries <- list(
    f64_existing_kernel = rec(100),
    f64_existing_transfer = rec(100),
    f64_too_slow = rec(1e-9),
    `_calibration` = rec(kernel_probe),
    `_calibration_transfer` = rec(transfer_probe / f64o_transfer_factor)
  )
  if (!calibration) entries[c("_calibration", "_calibration_transfer")] <- NULL
  perf_write_baseline(base, entries)
  list(base = base, obs = obs)
}

f64o_fn <- function() {
  df <- tbl_gpu(data.frame(x = as.numeric(seq_len(1000))), lazy = FALSE)
  function() df |> dplyr::filter(x > 5) |> collect()
}

f64o_read <- function(path) {
  utils::read.delim(path, sep = "\t", stringsAsFactors = FALSE)
}

f64o_cols <- c("id", "calibration", "current_s", "raw_scale", "scale", "observed_s", "iterations", "reps")

test_that("F64 observe: a kernel benchmark records its median with scale 1", {
  ws <- f64o_setup()
  suppressMessages(expect_no_perf_regression("f64_existing_kernel", f64o_fn()))
  expect_true(file.exists(ws$obs))
  o <- f64o_read(ws$obs)
  expect_true(all(f64o_cols %in% names(o)))
  expect_equal(nrow(o), 1)
  expect_identical(o$id, "f64_existing_kernel")
  expect_identical(o$calibration, "kernel")
  expect_equal(o$scale, 1)
  expect_equal(o$raw_scale, 1)
  expect_true(o$current_s > 0 && o$current_s < 100)
  expect_equal(o$observed_s, o$current_s)
  expect_equal(o$iterations, 5)
  expect_true(o$reps >= 1)
})

test_that("F64 observe: a transfer benchmark divides its ambient scale out", {
  ws <- f64o_setup()
  suppressMessages(expect_no_perf_regression("f64_existing_transfer", f64o_fn(), calibration = "transfer"))
  o <- f64o_read(ws$obs)
  expect_equal(nrow(o), 1)
  expect_identical(o$calibration, "transfer")
  expect_equal(o$raw_scale, f64o_transfer_factor, tolerance = 1e-9)
  expect_equal(o$scale, f64o_transfer_factor, tolerance = 1e-9)
  expect_equal(o$observed_s, o$current_s / o$scale, tolerance = 1e-12)
  expect_true(o$observed_s < o$current_s)
})

test_that("F64 observe: a benchmark with no baseline entry is timed, recorded and still skipped", {
  ws <- f64o_setup()
  res <- tryCatch(
    suppressMessages(expect_no_perf_regression("f64_brand_new", f64o_fn())),
    skip = function(e) e
  )
  expect_s3_class(res, "skip")
  expect_match(conditionMessage(res), "f64_brand_new")
  o <- f64o_read(ws$obs)
  expect_equal(nrow(o), 1)
  expect_identical(o$id, "f64_brand_new")
  expect_true(is.finite(o$observed_s) && o$observed_s > 0)
  expect_equal(o$scale, 1)
})

test_that("F64 observe: rows append under a single header", {
  ws <- f64o_setup()
  suppressMessages(expect_no_perf_regression("f64_existing_kernel", f64o_fn()))
  suppressMessages(expect_no_perf_regression("f64_existing_transfer", f64o_fn(), calibration = "transfer"))
  lines <- readLines(ws$obs)
  expect_equal(sum(startsWith(lines, "id\t")), 1)
  o <- f64o_read(ws$obs)
  expect_identical(o$id, c("f64_existing_kernel", "f64_existing_transfer"))
})

test_that("F64 observe: nothing is recorded without CUPLYR_PERF_OBSERVATIONS", {
  ws <- f64o_setup(observations = FALSE)
  suppressMessages(expect_no_perf_regression("f64_existing_kernel", f64o_fn()))
  res <- tryCatch(
    suppressMessages(expect_no_perf_regression("f64_brand_new", f64o_fn())),
    skip = function(e) e
  )
  expect_s3_class(res, "skip")
  expect_false(file.exists(ws$obs))
})

test_that("F64 observe: a failing benchmark is still recorded", {
  ws <- f64o_setup()
  suppressMessages(expect_failure(expect_no_perf_regression("f64_too_slow", f64o_fn())))
  o <- f64o_read(ws$obs)
  expect_identical(o$id, "f64_too_slow")
  expect_true(o$observed_s > 1e-9)
})

test_that("F64 observe: a new id without a calibration entry is skipped and not recorded", {
  ws <- f64o_setup(calibration = FALSE)
  res <- tryCatch(
    suppressMessages(expect_no_perf_regression("f64_brand_new", f64o_fn())),
    skip = function(e) e
  )
  expect_s3_class(res, "skip")
  expect_false(file.exists(ws$obs))
})
