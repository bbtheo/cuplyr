# F61 item 5: the perf gate covers integer arithmetic and rolling windows.
#
# Each benchmark is an expect_no_perf_regression() call in
# test-perf-regression.R. A new id has no baseline entry until the merge
# queue's ratchet seeds it, so this only checks that the ids are defined once
# and are kernel-calibrated. tools/perf shows that they run and report a time.

f61p_calls <- function() {
  path <- test_path("test-perf-regression.R")
  exprs <- parse(path, keep.source = FALSE)
  found <- list()
  walk <- function(e) {
    if (is.call(e)) {
      if (identical(e[[1]], as.name("expect_no_perf_regression"))) {
        found[[length(found) + 1L]] <<- e
      }
      for (a in as.list(e)[-1]) if (!missing(a)) walk(a)
    }
  }
  for (e in exprs) walk(e)
  found
}

test_that("the perf gate defines the F61 integer and rolling benchmarks", {
  calls <- f61p_calls()
  ids <- vapply(calls, function(cl) as.character(cl[[2]]), character(1))
  required <- c("int_arith_eager", "int_arith_lazy",
                "roll_mean_w11_eager", "roll_mean_w1001_eager",
                "roll_sd_w11_eager", "roll_sd_w1001_eager")
  expect_true(all(required %in% ids), info = paste(setdiff(required, ids), collapse = ", "))
  expect_false(anyDuplicated(ids) > 0L)
  for (cl in calls[ids %in% required]) {
    expect_identical(cl$calibration, "kernel", info = as.character(cl[[2]]))
  }
})
