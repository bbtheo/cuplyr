# Acceptance tests for #13: the out-of-memory error shows the GPU memory state
# and names what helps.
#
# Contract:
# - When a GPU step still runs out of device memory after cuplyr's own retry
#   (gc() + gpu_gc(), then once more), the error has class
#   `cuplyr_error_gpu_oom` (an rlang error), and:
#   * `$memory` is the `gpu_memory_state()` read after the retry failed, and
#     the message shows it: the device's total and free memory, and the bytes
#     this session holds in its memory resource and has in use, with the
#     resource's name. Sizes are formatted as the fallback notices format
#     them (`format(object_size, units = "auto", standard = "SI")`: "12 GB").
#     A state that couldn't be read says so; no "NA" in the message.
#   * The message names what helps: `gpu_gc()`, lazy mode
#     (`cuplyr.exec_mode`), and, under the "pool" resource only, the
#     `cuplyr.memory_resource` option.
#   * `$parent` is the original allocator error, so the message ends with it
#     ("Caused by error: ... out_of_memory ...").
#   * `$call` is the user's call (`filter()`, `collect()`, `tbl_gpu()`), as
#     for dplyr's own errors, not NULL.
# - A verb that wraps the errors of its expressions (mutate(), filter())
#   passes the OOM error through unwrapped.
# - The messages are covered by snapshot tests (run with NOT_CRAN set, so they
#   run under tools/test and the merge gate, see issue #86).

p13_snapshots_on <- function(envir = parent.frame()) {
  withr::local_envvar(NOT_CRAN = "true", .local_envir = envir)
}

p13_state <- function(resource = "async") {
  list(
    available = TRUE,
    total_bytes = 12e9,
    free_bytes = 1.5e9,
    used_bytes = 10.5e9,
    total_gb = 12,
    free_gb = 1.5,
    used_gb = 10.5,
    resource = resource,
    pool_bytes = 9.5e9,
    pool_used_bytes = 9e9
  )
}

p13_state_unavailable <- function() {
  list(
    available = FALSE,
    total_bytes = NA_real_, free_bytes = NA_real_, used_bytes = NA_real_,
    total_gb = NA_real_, free_gb = NA_real_, used_gb = NA_real_,
    resource = "cuda", pool_bytes = NA_real_, pool_used_bytes = NA_real_
  )
}

# Make every RMM allocation fail until the calling test ends, and report a
# fixed memory state.
p13_local_oom <- function(state = p13_state(), envir = parent.frame()) {
  testthat::local_mocked_bindings(gpu_memory_state = function() state, .env = envir)
  gpu_oom_inject(100000L)
  withr::defer(gpu_oom_inject(0L), envir = envir)
}

p13_eager <- function() tbl_gpu(data.frame(x = c(0.25, 0.75, 1.5)), lazy = FALSE)

test_that("an OOM after the retry is a classed error with the memory state, the advice and the original error", {
  skip_if_no_gpu()
  d <- p13_eager()
  err <- local({
    p13_local_oom()
    tryCatch(filter(d, x > 0.5), error = function(e) e)
  })
  expect_s3_class(err, "cuplyr_error_gpu_oom")
  expect_s3_class(err, "rlang_error")
  expect_identical(err$memory, p13_state())
  expect_identical(rlang::format_error_call(err$call), "`filter()`")
  expect_true(inherits(err$parent, "error"))
  expect_match(conditionMessage(err$parent), "out_of_memory", fixed = TRUE)

  msg <- conditionMessage(err)
  # the memory state, as the fallback notices format sizes
  expect_match(msg, "12 GB", fixed = TRUE)   # device total
  expect_match(msg, "1.5 GB", fixed = TRUE)  # device free
  expect_match(msg, "9.5 GB", fixed = TRUE)  # held by the session's resource
  expect_match(msg, "9 GB", fixed = TRUE)    # in use by the session
  expect_match(msg, "async", fixed = TRUE)
  # what helps
  expect_match(msg, "gpu_gc()", fixed = TRUE)
  expect_match(msg, "cuplyr.exec_mode", fixed = TRUE)
  expect_no_match(msg, "cuplyr.memory_resource", fixed = TRUE)
  # the retry and the original error are still said
  expect_match(msg, "retr", ignore.case = TRUE)
  expect_match(msg, "injected out_of_memory", fixed = TRUE)
  expect_no_match(msg, "\\bNA\\b")
})

test_that("under the pool resource the OOM error names cuplyr.memory_resource", {
  skip_if_no_gpu()
  d <- p13_eager()
  err <- local({
    p13_local_oom(p13_state("pool"))
    tryCatch(filter(d, x > 0.5), error = function(e) e)
  })
  expect_s3_class(err, "cuplyr_error_gpu_oom")
  msg <- conditionMessage(err)
  expect_match(msg, "pool", fixed = TRUE)
  expect_match(msg, "cuplyr.memory_resource", fixed = TRUE)
  expect_match(msg, "gpu_gc()", fixed = TRUE)
})

test_that("an OOM error with an unreadable memory state says so and shows no NA", {
  skip_if_no_gpu()
  d <- p13_eager()
  err <- local({
    p13_local_oom(p13_state_unavailable())
    tryCatch(filter(d, x > 0.5), error = function(e) e)
  })
  expect_s3_class(err, "cuplyr_error_gpu_oom")
  expect_identical(err$memory, p13_state_unavailable())
  msg <- conditionMessage(err)
  expect_no_match(msg, "\\bNA\\b")
  expect_match(msg, "gpu_gc()", fixed = TRUE)
})

test_that("the OOM error reports the user's call: collect(), compute(), tbl_gpu()", {
  skip_if_no_gpu()
  l <- filter(tbl_gpu(data.frame(x = c(0.25, 0.75, 1.5)), lazy = TRUE), x > 0.5)
  err <- local({
    p13_local_oom()
    tryCatch(collect(l), error = function(e) e)
  })
  expect_s3_class(err, "cuplyr_error_gpu_oom")
  expect_identical(rlang::format_error_call(err$call), "`collect()`")
  expect_identical(err$memory, p13_state())

  err <- local({
    p13_local_oom()
    tryCatch(compute(l), error = function(e) e)
  })
  expect_s3_class(err, "cuplyr_error_gpu_oom")
  expect_identical(rlang::format_error_call(err$call), "`compute()`")

  err <- local({
    p13_local_oom()
    tryCatch(tbl_gpu(data.frame(x = 1:3), lazy = FALSE), error = function(e) e)
  })
  expect_s3_class(err, "cuplyr_error_gpu_oom")
  expect_identical(rlang::format_error_call(err$call), "`tbl_gpu()`")
})

test_that("mutate() and filter() pass the OOM error through unwrapped", {
  skip_if_no_gpu()
  d <- p13_eager()
  for (pipeline in list(
    function(d) mutate(d, z = x + 1),
    function(d) mutate(group_by(d, x), z = x + 1),
    function(d) filter(d, x > 0.5),
    function(d) filter(d, x > mean(x))
  )) {
    err <- local({
      p13_local_oom()
      tryCatch(pipeline(d), error = function(e) e)
    })
    expect_s3_class(err, "cuplyr_error_gpu_oom")
    expect_false(inherits(err, "dplyr:::mutate_error"))
    expect_null(err$parent$parent)
  }
})

test_that("the session works again after an OOM error", {
  skip_if_no_gpu()
  d <- p13_eager()
  local({
    p13_local_oom()
    expect_error(filter(d, x > 0.5), class = "cuplyr_error_gpu_oom")
  })
  expect_identical(collect(filter(d, x > 0.5))$x, c(0.75, 1.5))
})

test_that("OOM error messages (snapshot)", {
  skip_if_no_gpu()
  p13_snapshots_on()
  d <- p13_eager()
  l <- filter(tbl_gpu(data.frame(x = c(0.25, 0.75, 1.5)), lazy = TRUE), x > 0.5)
  local({
    p13_local_oom()
    expect_snapshot(error = TRUE, filter(d, x > 0.5))
    expect_snapshot(error = TRUE, collect(l))
  })
  local({
    p13_local_oom(p13_state("pool"))
    expect_snapshot(error = TRUE, filter(d, x > 0.5))
  })
  local({
    p13_local_oom(p13_state_unavailable())
    expect_snapshot(error = TRUE, filter(d, x > 0.5))
  })
})
