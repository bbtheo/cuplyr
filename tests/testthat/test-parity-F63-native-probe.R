# F63 amendment (Theo, 2026-10-22): "exclude all cpu fallbacks from the
# timings". Every cuplyr timing cell first runs one untimed probe call with
# options(cuplyr.fallback = "error"); a cell whose call falls back is skipped
# with the recorded reason instead of being timed. bench/rolling-lib.R's
# rw_native_probe(call) is that probe.

f63_lib_env_gpu <- function() {
  path <- f63_require("bench/rolling-lib.R")
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

test_that("rw_native_probe() passes a native cuplyr call", {
  skip_if_no_gpu()
  rw <- f63_lib_env_gpu()
  gt <- tbl_gpu(data.frame(x = as.numeric(1:200)))
  for (stat in c("mean", "sum", "min", "max", "sd")) {
    cf <- rw$rw_cuplyr_fun(stat)
    res <- rw$rw_native_probe(function() cf(gt, 11))
    expect_true(isTRUE(res$native), label = paste("roll", stat, "is native"))
    expect_identical(res$reason, "")
  }
})

test_that("rw_native_probe() rejects roll_median's CPU fallback and records why", {
  skip_if_no_gpu()
  rw <- f63_lib_env_gpu()
  gt <- tbl_gpu(data.frame(x = as.numeric(1:200)))
  cf <- rw$rw_cuplyr_fun("median")
  withr::local_options(cuplyr.fallback = NULL)
  res <- rw$rw_native_probe(function() cf(gt, 11))
  expect_false(isTRUE(res$native))
  expect_type(res$reason, "character")
  expect_match(res$reason, "fell back", fixed = TRUE)
  # the option is restored afterwards
  expect_null(getOption("cuplyr.fallback"))
})

test_that("rw_native_probe() restores a pre-set cuplyr.fallback option", {
  skip_if_no_gpu()
  rw <- f63_lib_env_gpu()
  gt <- tbl_gpu(data.frame(x = as.numeric(1:200)))
  withr::local_options(cuplyr.fallback = "warn")
  cf <- rw$rw_cuplyr_fun("median")
  res <- rw$rw_native_probe(function() cf(gt, 11))
  expect_false(isTRUE(res$native))
  expect_identical(getOption("cuplyr.fallback"), "warn")
})
