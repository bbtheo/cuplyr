# F63 rounds 2-3: helpers added to bench/rolling-lib.R beyond the PM's
# acceptance tests (no GPU needed): rw_time_interleaved(), the skipped-cells
# table, and the probe outcome (CPU fallback vs any other error).

f63x_lib_env <- function() {
  path <- f63_require("bench/rolling-lib.R")
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

test_that("rw_time_interleaved() times every closure once per round and alternates the order", {
  rw <- f63x_lib_env()
  calls <- character()
  fns <- list(a = function() { calls <<- c(calls, "a"); 1 },
              b = function() { calls <<- c(calls, "b"); 2 })
  d <- rw$rw_time_interleaved(fns, times = 4, warmup = 1)
  expect_named(d, c("label", "sample", "ms"))
  expect_equal(nrow(d), 8L)
  expect_equal(as.integer(table(d$label)), c(4L, 4L))
  expect_equal(sort(unique(d$sample)), 1:4)
  expect_true(all(is.finite(d$ms)) && all(d$ms >= 0))
  # one warmup call per closure, then a-b, b-a, a-b, b-a
  expect_equal(calls, c("a", "b", "a", "b", "b", "a", "a", "b", "b", "a"))
  # the recorded order is the call order
  expect_equal(d$label, c("a", "b", "b", "a", "a", "b", "b", "a"))
})

test_that("rw_time_interleaved() runs gc() before every call, timed or not", {
  rw <- f63x_lib_env()
  log <- character()
  e <- rw
  # shadow gc() in the sourced environment to record the interleaving
  e$gc <- function(...) { log <<- c(log, "gc"); invisible(NULL) }
  fns <- list(a = function() { log <<- c(log, "a"); 1 })
  e$rw_time_interleaved(fns, times = 2, warmup = 1)
  expect_equal(log, c("gc", "a", "gc", "a", "gc", "a"))
})

test_that("rw_skip_table() builds the skipped-cells table, empty or not", {
  rw <- f63x_lib_env()
  empty <- rw$rw_skip_table(list())
  expect_equal(nrow(empty), 0L)
  expect_named(empty, c("chunk", "stat", "fn", "width", "reason"))
  rows <- list(rw$rw_skip_row("bench-main", "median", "cuplyr", 11, "fell back"),
               rw$rw_skip_row("bench-memory", "memory", "plain", NA, "fell back"))
  tab <- rw$rw_skip_table(rows)
  expect_equal(nrow(tab), 2L)
  expect_equal(tab$width, c(11, NA))
  expect_equal(rbind(empty, tab), tab, ignore_attr = TRUE)
})

# F63 round 3 (review N12): only a CPU fallback skips a cuplyr cell; any other
# probe error is a failed cell that counts against the correctness total.

test_that("rw_native_probe() tells a CPU fallback from any other error", {
  rw <- f63x_lib_env()
  ok <- rw$rw_native_probe(function() 1)
  expect_true(ok$native)
  expect_false(ok$fallback)
  # what cuplyr raises under options(cuplyr.fallback = "error")
  fb <- rw$rw_native_probe(function() stop("cuplyr: mutate() fell back to CPU evaluation for: roll_median(x, 11)", call. = FALSE))
  expect_false(fb$native)
  expect_true(fb$fallback)
  other <- rw$rw_native_probe(function() stop("std::bad_alloc: out_of_memory: CUDA error", call. = FALSE))
  expect_false(other$native)
  expect_false(other$fallback)
  expect_match(other$reason, "out_of_memory", fixed = TRUE)
})

test_that("rw_is_fallback() accepts a cuplyr_cpu_fallback condition", {
  rw <- f63x_lib_env()
  cnd <- structure(class = c("cuplyr_cpu_fallback", "error", "condition"),
                   list(message = "mutate() fell back", call = NULL))
  expect_true(rw$rw_is_fallback(cnd))
  expect_false(rw$rw_is_fallback(simpleError("something else")))
})

test_that("rw_probe_error_cell() is the error row safe_cell() records", {
  rw <- f63x_lib_env()
  res <- rw$rw_probe_error_cell(list(native = FALSE, reason = "boom", fallback = FALSE))
  expect_true(is.na(res$ms))
  expect_identical(res$times, 0)
  expect_match(res$note, "^error: probe call failed: boom")
  expect_false(res$cmp$ok)
})

test_that("rw_native_probe() still restores the option after a non-fallback error", {
  rw <- f63x_lib_env()
  withr::local_options(cuplyr.fallback = "warn")
  rw$rw_native_probe(function() stop("boom"))
  expect_identical(getOption("cuplyr.fallback"), "warn")
})
