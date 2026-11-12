# F00 harness details not covered by the PM's acceptance file: the build-state
# scan in tools/ensure-optimized.R and the kernel/transfer bound arithmetic at
# whole-tick readings. No GPU needed.

local_fake_root <- function(text, frame = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = frame)
  dir.create(file.path(root, "src"))
  if (!is.null(text)) {
    writeBin(
      c(as.raw(c(0x7f, 0x45, 0x4c, 0x46, 0x00)), charToRaw(text), as.raw(0x00)),
      file.path(root, "src", "cuplyr.so")
    )
  }
  root
}

test_that("ensure-optimized classifies a .so by its producer strings", {
  # tools/ is in .Rbuildignore, so R CMD check runs this file without it.
  path <- testthat::test_path("..", "..", "tools", "ensure-optimized.R")
  skip_if_not(file.exists(path), "tools/ is not shipped in the package tarball")
  env <- new.env()
  sys.source(path, envir = env)
  state <- env$cuplyr_so_build_state

  expect_identical(state(local_fake_root(NULL)), "missing")
  expect_identical(state(local_fake_root("GNU C++20 14.3.0 -g -O2 -fpic")), "optimized")
  expect_identical(
    state(local_fake_root("GNU C++20 14.3.0 -g -O2 -fpic\x01GNU C++20 14.3.0 -g -O2 -O0 -fpic")),
    "unoptimized"
  )
  # No recorded flags: reported, not rebuilt (a rebuild would record none either).
  expect_identical(state(local_fake_root("clang version 19.1.0")), "unverifiable")
})

test_that("kernel probe: both observed modes give scale 1, the bound sits between ticks", {
  kernel <- function(x) perf_effective_scale(x / 0.065, list(id = "t", probe = "kernel", current_s = x, baseline_s = 0.065))
  # Readings built the way proc.time() differences are, so they carry its
  # floating-point noise (a 78 ms reading against a cap of exactly 1.2 x
  # 0.065 s landed on either side depending on that noise).
  tick <- function(ms, k0) 1e-3 * (k0 + ms) - 1e-3 * k0
  for (k0 in c(5000, 12345, 23456, 40000, 61234)) {
    for (ms in c(43, 46, 47, 77, 78, 79, 80, 82, 97)) expect_identical(kernel(tick(ms, k0)), 1)
    expect_error(kernel(tick(98, k0)), class = "cuplyr_perf_calibration_error")
  }
})

test_that("transfer probe: whole-tick readings widen up to 0.022 s and error from 0.023 s", {
  transfer <- function(x) perf_effective_scale(x / 0.019, list(id = "t", probe = "transfer", current_s = x, baseline_s = 0.019))
  expect_identical(transfer(0.017), 1)
  expect_equal(transfer(0.020), 0.020 / 0.019)
  expect_equal(transfer(0.022), 0.022 / 0.019)
  expect_error(transfer(0.023), class = "cuplyr_perf_calibration_error")
})

test_that("a reading on the cap up to floating-point noise is accepted", {
  expect_equal(perf_effective_scale(1.2 * (1 + 1e-12)), 1.2)
  expect_error(perf_effective_scale(1.2 * (1 + 1e-6)), class = "cuplyr_perf_calibration_error")
  kctx <- list(probe = "kernel")
  expect_identical(perf_effective_scale(1.5 * (1 + 1e-12), kctx), 1)
  expect_error(perf_effective_scale(1.5 * (1 + 1e-6), kctx), class = "cuplyr_perf_calibration_error")
})

test_that("the kernel probe error names the benchmark, probe, timings and bound", {
  err <- tryCatch(
    perf_effective_scale(3, list(id = "bench_k", probe = "kernel", current_s = 0.195, baseline_s = 0.065)),
    error = function(e) e
  )
  expect_s3_class(err, "cuplyr_perf_calibration_error")
  # rlang wraps long bullets; compare with whitespace collapsed.
  msg <- gsub("\\s+", " ", conditionMessage(err))
  expect_match(msg, "bench_k", fixed = TRUE)
  expect_match(msg, "kernel probe: current 0.195 s, baseline 0.065 s, raw scale 3", fixed = TRUE)
  expect_match(msg, "health-check bound 1.50", fixed = TRUE)
})

test_that("baselines are written at full precision", {
  # The fallback writer without jsonlite keeps 7 significant digits only.
  skip_if_not_installed("jsonlite")
  path <- withr::local_tempfile(fileext = ".json")
  perf_write_baseline(path, list(b = list(median_s = 0.00161234567, commit = "x")))
  expect_identical(perf_read_baseline(path)$b$median_s, 0.00161234567)
})
