# F73 round 2: behaviour of bench/rolling-lib.R's rw_polars_child() beyond the
# PM's acceptance tests (CPU only, no GPU). The polars child's progress lines
# must reach this process's stderr (the render log), and an error in the child
# must reach the caller.

f73x_lib_env <- function() {
  path <- f63_require("bench/rolling-lib.R")
  lib <- f63_bench_lib()
  if (dir.exists(lib)) withr::local_libpaths(lib, action = "prefix", .local_envir = parent.frame())
  skip_if_not_installed("callr")
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

test_that("rw_polars_child(show = TRUE) passes the child's stderr lines on to this process's stderr", {
  rw <- f73x_lib_env()
  say <- function() {
    cat("progress line one\n", file = stderr())
    cat("progress line two\n", file = stderr())
    cat("on stdout\n")
    42
  }
  msg <- utils::capture.output(res <- rw$rw_polars_child(1, say, libs = .libPaths(), show = TRUE), type = "message")
  expect_identical(res, 42)
  expect_true(all(c("progress line one", "progress line two") %in% msg))
  # not through stdout, where a knitr chunk would capture (and hide) it
  out <- utils::capture.output(invisible(rw$rw_polars_child(1, say, libs = .libPaths(), show = TRUE)), type = "output")
  expect_false(any(grepl("progress line", out)))
  # show = FALSE stays silent on both
  msg2 <- utils::capture.output(out2 <- utils::capture.output(invisible(rw$rw_polars_child(1, say, libs = .libPaths())), type = "output"), type = "message")
  expect_length(msg2, 0L)
  expect_length(out2, 0L)
})

test_that("rw_polars_child() raises an error that happens in the child, and does not hang on a lot of output", {
  rw <- f73x_lib_env()
  expect_error(rw$rw_polars_child(1, function() stop("boom in the child"), libs = .libPaths()), "boom in the child")
  chatty <- function() {
    for (i in 1:20000) cat("line", i, "padding padding padding padding\n", file = stderr())
    "done"
  }
  expect_identical(rw$rw_polars_child(NULL, chatty, libs = .libPaths()), "done")
})

test_that("rw_polars_child() does not busy-wait while the child has an unfinished output line", {
  rw <- f73x_lib_env()
  partial <- function() {
    cat("no newline yet", file = stderr())
    Sys.sleep(2)
    "partial-done"
  }
  for (show in c(TRUE, FALSE)) {
    t0 <- proc.time()
    utils::capture.output(res <- rw$rw_polars_child(1, partial, libs = .libPaths(), show = show), type = "message")
    d <- proc.time() - t0
    expect_identical(res, "partial-done")
    expect_gte(d[["elapsed"]], 2)
    # this process's own CPU time: a spinning drain loop costs about the child's
    # 2 s sleep, a waiting one a small fraction of it
    expect_lt(d[["user.self"]] + d[["sys.self"]], 0.5)
  }
})

test_that("rw_time_interleaved(prepare =) runs each closure's prepare before its every call, outside timing", {
  path <- f63_require("bench/rolling-lib.R")
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  log <- character()
  e$gc <- function(...) { log <<- c(log, "gc"); invisible(NULL) }
  fns <- list(a = function() { log <<- c(log, "a"); 1 }, b = function() { log <<- c(log, "b"); 2 })
  prep <- list(a = function() log <<- c(log, "pa"), b = function() { log <<- c(log, "pb"); Sys.sleep(0.05) })
  d <- e$rw_time_interleaved(fns, times = 2, warmup = 1, prepare = prep)
  # warmup a, warmup b, then round 1 a-b, round 2 b-a; gc, then prepare, then the call
  expect_equal(log, c("gc", "pa", "a", "gc", "pb", "b", "gc", "pa", "a", "gc", "pb", "b", "gc", "pb", "b", "gc", "pa", "a"))
  # b's prepare sleeps 50 ms; none of that is in b's timings
  expect_true(all(d$ms[d$label == "b"] < 40))
  # without prepare, nothing changes
  log <- character()
  e$rw_time_interleaved(fns["a"], times = 1, warmup = 1)
  expect_equal(log, c("gc", "a", "gc", "a"))
})
