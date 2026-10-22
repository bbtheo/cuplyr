# F58: OOM safety net.
#
# Contract (scratchpad/parity/features/F58.md, Design):
# - `wrap_gpu_call()` (every eager GPU entry point goes through it) retries an
#   out-of-memory failure exactly once, after one `gc()` + `gpu_gc()`. A second
#   OOM raises `cuplyr_error_gpu_oom`, whose message says a retry after garbage
#   collection was attempted and keeps the original error text.
# - It never retries anything else, never loops (a nested wrap that already
#   gave up passes its `cuplyr_error_gpu_oom` through untouched), and a warning
#   the failed attempt signalled is not signalled a second time by the retry.
# - Test hook: `gpu_oom_inject(n)` makes the next `n` RMM device allocations
#   throw `rmm::out_of_memory` (`n = 0` disarms); `gpu_oom_inject_remaining()`
#   is how many armed failures have not fired yet. One failed attempt consumes
#   one armed failure, so after `gpu_oom_inject(3)` a call that retries exactly
#   once and then gives up leaves 1.

f58_oom_condition <- function(msg = "std::bad_alloc: out_of_memory: RMM failure at: arena.hpp:42: Maximum pool size exceeded") {
  structure(
    class = c("rmm::out_of_memory", "C++Error", "error", "condition"),
    list(message = msg, call = NULL)
  )
}

# -----------------------------------------------------------------------------
# wrap_gpu_call(): retry policy (no GPU needed)
# -----------------------------------------------------------------------------

test_that("a non-OOM error is not retried and keeps its message", {
  k <- 0L
  err <- expect_error(
    wrap_gpu_call("op", {
      k <- k + 1L
      stop("boom: a cudf logic error")
    }),
    "boom: a cudf logic error"
  )
  expect_identical(k, 1L)
  expect_false(inherits(err, "cuplyr_error_gpu_oom"))
})

test_that("an OOM is retried once and the retry's value is returned", {
  k <- 0L
  out <- wrap_gpu_call("op", {
    k <- k + 1L
    if (k == 1L) stop(f58_oom_condition())
    "ok"
  })
  expect_identical(out, "ok")
  expect_identical(k, 2L)
})

test_that("an OOM recognised by its message only (plain error class) is retried", {
  k <- 0L
  out <- wrap_gpu_call("op", {
    k <- k + 1L
    if (k == 1L) stop("std::bad_alloc: cudaErrorMemoryAllocation: out of memory")
    1
  })
  expect_identical(out, 1)
  expect_identical(k, 2L)
})

test_that("a second OOM raises cuplyr_error_gpu_oom after exactly two attempts", {
  k <- 0L
  err <- expect_error(
    wrap_gpu_call("filter", {
      k <- k + 1L
      stop(f58_oom_condition())
    }),
    class = "cuplyr_error_gpu_oom"
  )
  expect_identical(k, 2L)
  msg <- conditionMessage(err)
  expect_match(msg, "retr", ignore.case = TRUE)
  expect_match(msg, "garbage collection|gc\\(\\)|gpu_gc\\(\\)", ignore.case = TRUE)
  expect_match(msg, "Maximum pool size exceeded", fixed = TRUE)
  expect_match(msg, "filter", fixed = TRUE)
})

test_that("garbage is collected before the retry", {
  finalized <- FALSE
  local({
    e <- new.env()
    reg.finalizer(e, function(e) finalized <<- TRUE)
    NULL
  })
  seen_at_retry <- NA
  k <- 0L
  wrap_gpu_call("op", {
    k <- k + 1L
    if (k == 1L) stop(f58_oom_condition())
    seen_at_retry <- finalized
    NULL
  })
  expect_true(seen_at_retry)
})

test_that("a nested wrap that already gave up is not retried again (no loop)", {
  k <- 0L
  expect_error(
    wrap_gpu_call("outer", {
      wrap_gpu_call("inner", {
        k <- k + 1L
        stop(f58_oom_condition())
      })
    }),
    class = "cuplyr_error_gpu_oom"
  )
  expect_identical(k, 2L)
})

test_that("a warning from the failed attempt is not signalled twice", {
  k <- 0L
  n_warn <- 0L
  out <- withCallingHandlers(
    wrap_gpu_call("op", {
      k <- k + 1L
      warning("NAs produced by integer overflow", call. = FALSE)
      if (k == 1L) stop(f58_oom_condition())
      "done"
    }),
    warning = function(w) {
      n_warn <<- n_warn + 1L
      invokeRestart("muffleWarning")
    }
  )
  expect_identical(out, "done")
  expect_identical(k, 2L)
  expect_identical(n_warn, 1L)
})

test_that("passthrough classes are still not reworded or retried", {
  k <- 0L
  expect_error(
    wrap_gpu_call("op", {
      k <- k + 1L
      rlang::abort("overflow", class = "cuplyr_error_sum_overflow")
    }),
    class = "cuplyr_error_sum_overflow"
  )
  expect_identical(k, 1L)
})

# -----------------------------------------------------------------------------
# End to end, with OOMs injected at the RMM allocator
# -----------------------------------------------------------------------------

f58_df <- function(n = 2000L) {
  set.seed(58)
  x <- runif(n)
  x[c(5, 50)] <- NA
  data.frame(
    g = sample.int(7L, n, replace = TRUE),
    k = sample.int(50L, n, replace = TRUE),
    x = x,
    s = sprintf("s%02d", sample.int(20L, n, replace = TRUE)),
    stringsAsFactors = FALSE
  )
}

f58_dfy <- function() {
  set.seed(59)
  data.frame(k = 1:40, v = runif(40))
}

f58_disarm <- function(env = parent.frame()) {
  withr::defer(gpu_oom_inject(0L), envir = env)
}

f58_same <- function(gpu, expected) {
  expect_equal(tibble::as_tibble(collect(gpu)), tibble::as_tibble(expected))
}

# Each case: the GPU call on prebuilt inputs, and the dplyr result to match.
f58_cases <- function() {
  df <- f58_df()
  dfy <- f58_dfy()
  list(
    filter = list(
      gpu = function(a, b) dplyr::filter(a, x > 0.5, g != 3L),
      cpu = function() dplyr::filter(df, x > 0.5, g != 3L)
    ),
    mutate = list(
      gpu = function(a, b) dplyr::mutate(a, y = x * 2 + k),
      cpu = function() dplyr::mutate(df, y = x * 2 + k)
    ),
    mutate_window = list(
      gpu = function(a, b) dplyr::mutate(dplyr::group_by(a, g), r = dplyr::row_number(), m = x - mean(x, na.rm = TRUE)),
      cpu = function() dplyr::mutate(dplyr::group_by(df, g), r = dplyr::row_number(), m = x - mean(x, na.rm = TRUE))
    ),
    summarise = list(
      gpu = function(a, b) dplyr::summarise(dplyr::group_by(a, g), m = mean(x, na.rm = TRUE), n = dplyr::n()),
      cpu = function() dplyr::summarise(dplyr::group_by(df, g), m = mean(x, na.rm = TRUE), n = dplyr::n())
    ),
    arrange = list(
      gpu = function(a, b) dplyr::arrange(a, g, dplyr::desc(x)),
      cpu = function() dplyr::arrange(df, g, dplyr::desc(x))
    ),
    left_join = list(
      gpu = function(a, b) dplyr::left_join(a, b, by = "k"),
      cpu = function() dplyr::left_join(df, dfy, by = "k")
    ),
    distinct = list(
      gpu = function(a, b) dplyr::distinct(a, g, k),
      cpu = function() dplyr::distinct(df, g, k)
    ),
    slice = list(
      gpu = function(a, b) dplyr::slice(a, c(10:20, 400L)),
      cpu = function() dplyr::slice(df, c(10:20, 400L))
    ),
    bind_rows = list(
      gpu = function(a, b) dplyr::bind_rows(a, a),
      cpu = function() dplyr::bind_rows(df, df)
    )
  )
}

test_that("one injected OOM is absorbed by the retry, for every eager verb", {
  skip_if_no_gpu()
  f58_disarm()
  df <- f58_df()
  cases <- f58_cases()
  a <- tbl_gpu(df, lazy = FALSE)
  b <- tbl_gpu(f58_dfy(), lazy = FALSE)
  for (nm in names(cases)) {
    gpu_oom_inject(1L)
    res <- cases[[nm]]$gpu(a, b)
    expect_identical(gpu_oom_inject_remaining(), 0L, label = paste(nm, "remaining"))
    gpu_oom_inject(0L)
    f58_same(res, cases[[nm]]$cpu())
  }
  # The inputs are untouched by the failed attempts.
  f58_same(a, df)
})

test_that("a persistent OOM raises cuplyr_error_gpu_oom after one retry, for every eager verb", {
  skip_if_no_gpu()
  f58_disarm()
  df <- f58_df()
  cases <- f58_cases()
  a <- tbl_gpu(df, lazy = FALSE)
  b <- tbl_gpu(f58_dfy(), lazy = FALSE)
  for (nm in names(cases)) {
    gpu_oom_inject(3L)
    expect_error(cases[[nm]]$gpu(a, b), class = "cuplyr_error_gpu_oom", label = nm)
    expect_identical(gpu_oom_inject_remaining(), 1L, label = paste(nm, "remaining"))
    gpu_oom_inject(0L)
  }
  # Nothing the user holds was freed or changed.
  f58_same(a, df)
  f58_same(b, f58_dfy())
  f58_same(dplyr::filter(a, x > 0.5), dplyr::filter(df, x > 0.5))
})

test_that("the upload in tbl_gpu() is covered", {
  skip_if_no_gpu()
  f58_disarm()
  df <- f58_df()
  gpu_oom_inject(1L)
  a <- tbl_gpu(df, lazy = FALSE)
  expect_identical(gpu_oom_inject_remaining(), 0L)
  gpu_oom_inject(0L)
  f58_same(a, df)

  gpu_oom_inject(3L)
  expect_error(tbl_gpu(df, lazy = FALSE), class = "cuplyr_error_gpu_oom")
  expect_identical(gpu_oom_inject_remaining(), 1L)
})

test_that("compute() of a lazy pipeline is covered, and the lazy input survives", {
  skip_if_no_gpu()
  f58_disarm()
  df <- f58_df()
  dfy <- f58_dfy()
  la <- tbl_gpu(df, lazy = TRUE)
  lb <- tbl_gpu(dfy, lazy = TRUE)
  pipe <- function(x, y) {
    x |>
      dplyr::filter(x > 0.2) |>
      dplyr::mutate(z = x + k) |>
      dplyr::left_join(y, by = "k") |>
      dplyr::arrange(z)
  }
  expected <- pipe(df, dfy)

  gpu_oom_inject(1L)
  res <- dplyr::compute(pipe(la, lb))
  expect_identical(gpu_oom_inject_remaining(), 0L)
  gpu_oom_inject(0L)
  f58_same(res, expected)

  gpu_oom_inject(3L)
  expect_error(dplyr::compute(pipe(la, lb)), class = "cuplyr_error_gpu_oom")
  expect_identical(gpu_oom_inject_remaining(), 1L)
  gpu_oom_inject(0L)

  # collect() of the lazy plan runs it too.
  gpu_oom_inject(1L)
  out <- dplyr::collect(pipe(la, lb))
  gpu_oom_inject(0L)
  expect_equal(tibble::as_tibble(out), tibble::as_tibble(expected))

  f58_same(la, df)
  f58_same(pipe(la, lb), expected)
})

test_that("no OOM: injection disarmed means nothing is consumed or retried", {
  skip_if_no_gpu()
  df <- f58_df()
  a <- tbl_gpu(df, lazy = FALSE)
  gpu_oom_inject(0L)
  expect_identical(gpu_oom_inject_remaining(), 0L)
  f58_same(dplyr::arrange(a, x), dplyr::arrange(df, x))
})

test_that("GPU work outside the node pipeline is covered too", {
  skip_if_no_gpu()
  f58_disarm()
  df <- f58_df()
  a <- tbl_gpu(df, lazy = FALSE)

  # slice_sample(): the rows are drawn once, so a retried call returns what the
  # seed gives without an OOM and leaves the RNG where it would have.
  set.seed(7)
  ref <- tibble::as_tibble(collect(dplyr::slice_sample(a, n = 5)))
  ref_next <- runif(1)
  set.seed(7)
  gpu_oom_inject(1L)
  res <- dplyr::slice_sample(a, n = 5)
  expect_identical(gpu_oom_inject_remaining(), 0L)
  gpu_oom_inject(0L)
  expect_identical(runif(1), ref_next)
  expect_equal(tibble::as_tibble(collect(res)), ref)

  gpu_oom_inject(3L)
  expect_error(dplyr::slice_sample(a, n = 5), class = "cuplyr_error_gpu_oom")
  expect_identical(gpu_oom_inject_remaining(), 1L)
  gpu_oom_inject(0L)

  # A set operation that aligns column types (INT32 with FLOAT64).
  di <- data.frame(x = c(1L, 2L, 700L))
  dd <- data.frame(x = c(2, 3.5))
  bi <- tbl_gpu(di, lazy = FALSE)
  bd <- tbl_gpu(dd, lazy = FALSE)
  gpu_oom_inject(1L)
  res <- dplyr::union(bi, bd)
  expect_identical(gpu_oom_inject_remaining(), 0L)
  gpu_oom_inject(0L)
  f58_same(res, dplyr::union(di, dd))

  gpu_oom_inject(3L)
  expect_error(dplyr::union(bi, bd), class = "cuplyr_error_gpu_oom")
  expect_identical(gpu_oom_inject_remaining(), 1L)
  gpu_oom_inject(0L)

  # summarise() of a `.drop = FALSE` grouping with an empty group.
  df2 <- data.frame(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), x = c(1, 2, 5))
  t2 <- tbl_gpu(df2, lazy = FALSE)
  gpu_oom_inject(1L)
  res <- dplyr::summarise(dplyr::group_by(t2, f, .drop = FALSE), n = dplyr::n(), s = sum(x))
  expect_identical(gpu_oom_inject_remaining(), 0L)
  gpu_oom_inject(0L)
  f58_same(res, dplyr::summarise(dplyr::group_by(df2, f, .drop = FALSE), n = dplyr::n(), s = sum(x)))

  gpu_oom_inject(3L)
  expect_error(dplyr::summarise(dplyr::group_by(t2, f, .drop = FALSE), n = dplyr::n(), s = sum(x)),
               class = "cuplyr_error_gpu_oom")
  expect_identical(gpu_oom_inject_remaining(), 1L)
  gpu_oom_inject(0L)

  f58_same(a, df)
})
