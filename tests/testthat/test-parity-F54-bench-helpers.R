# F54: helper functions of the rolling-window benchmark (bench/rolling-lib.R).
#
# bench/ is .Rbuildignore'd, so these tests skip when the file isn't there
# (an installed/built package). They pin the parts of the benchmark that
# decide whether its numbers are fair and its correctness table is right:
# the comparator, the padding normalizer, the timing harness, and the cuplyr
# timed closure (GPU-native, no collect, finished on return).

rw_lib_path <- function() {
  testthat::test_path("..", "..", "bench", "rolling-lib.R")
}

# Skip only outside a source checkout (bench/ is not shipped); in the source
# tree a missing bench/rolling-lib.R is a failure.
rw_require_lib <- function() {
  path <- rw_lib_path()
  in_source_tree <- file.exists(testthat::test_path("..", "..", "DESCRIPTION"))
  if (!in_source_tree) skip("not in a source checkout; bench/ is not shipped")
  expect_true(file.exists(path), label = "bench/rolling-lib.R exists")
  if (!file.exists(path)) skip("bench/rolling-lib.R missing (failure recorded)")
  path
}

rw_bench_lib <- function() {
  Sys.getenv("CUPLYR_BENCH_LIB", "/home/theo/cuplr/scratchpad/benchmarks/rlib")
}

load_rw_lib <- function(env = parent.frame()) {
  path <- rw_require_lib()
  lib <- rw_bench_lib()
  if (dir.exists(lib)) withr::local_libpaths(lib, action = "prefix", .local_envir = env)
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

test_that("sourcing rolling-lib.R has no side effects on .libPaths()", {
  path <- rw_require_lib()
  before <- .libPaths()
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  expect_identical(.libPaths(), before)
  expect_true(is.function(e$rw_compare))
})

test_that("rw_data() reproduces the post's data", {
  rw <- load_rw_lib()
  d <- rw$rw_data(1000)
  set.seed(12345)
  expected <- rnorm(1000)
  expect_s3_class(d, "data.frame")
  expect_identical(names(d), "x")
  expect_identical(d$x, expected)
})

test_that("rw_normalize() left-pads with NA and strips attributes", {
  rw <- load_rw_lib()
  expect_identical(rw$rw_normalize(c(1, 2, 3), 5), c(NA, NA, 1, 2, 3))
  expect_identical(rw$rw_normalize(c(NA, 1, 2), 3), c(NA, 1, 2))
  expect_identical(rw$rw_normalize(1:3, 3), c(1, 2, 3))
  m <- matrix(c(NA, 1, 2), ncol = 1)
  expect_identical(rw$rw_normalize(m, 3), c(NA, 1, 2))
  named <- c(a = 1, b = 2)
  expect_identical(rw$rw_normalize(named, 3), c(NA, 1, 2))
})

test_that("rw_compare() checks length, NA positions and tolerance", {
  rw <- load_rw_lib()
  ref <- c(NA, NA, 1, 2, 3)

  same <- rw$rw_compare(ref, ref)
  expect_true(same$ok)
  expect_identical(as.integer(same$n_na_mismatch), 0L)
  expect_equal(same$max_abs_diff, 0)

  tiny <- rw$rw_compare(ref + c(0, 0, 1e-12, 0, 0), ref)
  expect_true(tiny$ok)

  off <- rw$rw_compare(ref + c(0, 0, 1e-3, 0, 0), ref)
  expect_false(off$ok)
  expect_equal(off$max_abs_diff, 1e-3, tolerance = 1e-6)

  # A partial window computed where the reference pads with NA.
  partial <- rw$rw_compare(c(NA, 0.5, 1, 2, 3), ref)
  expect_false(partial$ok)
  expect_identical(as.integer(partial$n_na_mismatch), 1L)

  short <- rw$rw_compare(c(1, 2, 3), ref)
  expect_false(short$ok)

  # All-NA on both sides is equal, not an error.
  expect_true(rw$rw_compare(c(NA_real_, NA_real_), c(NA_real_, NA_real_))$ok)
})

test_that("rw_time() warms up, times every sample, returns ms", {
  rw <- load_rw_lib()
  calls <- 0L
  f <- function() {
    calls <<- calls + 1L
    Sys.sleep(0.01)
  }
  t <- rw$rw_time(f, times = 3, warmup = 2)
  expect_identical(calls, 5L)
  expect_type(t, "double")
  expect_length(t, 3)
  # 10 ms of sleep, reported in milliseconds (not seconds, not ns).
  expect_true(all(t >= 9 & t < 1000))
})

test_that("rw_reference() is data.table's right-aligned froll*", {
  rw <- load_rw_lib()
  skip_if_not_installed("data.table")
  x <- c(1, 4, 2, 8, 5, 7)
  expect_equal(rw$rw_reference("mean", x, 3), data.table::frollmean(x, 3))
  expect_equal(rw$rw_reference("sum", x, 3), c(NA, NA, 7, 14, 15, 20))
  expect_equal(rw$rw_reference("min", x, 3), c(NA, NA, 1, 2, 2, 5))
  expect_equal(rw$rw_reference("max", x, 3), c(NA, NA, 4, 8, 8, 8))
  expect_equal(rw$rw_reference("median", x, 3), c(NA, NA, 2, 4, 5, 7))
  expect_equal(rw$rw_reference("sd", x, 3)[3], sd(c(1, 4, 2)))
})

test_that("rw_functions() covers the post's statistics and returns full-length output", {
  rw <- load_rw_lib()
  skip_if_not_installed("data.table")
  fns <- rw$rw_functions()
  expect_true(all(c("mean", "median", "min", "max", "sum", "sd", "custom") %in% names(fns)))
  for (stat in c("mean", "median", "min", "max", "sum")) {
    expect_true("data.table" %in% names(fns[[stat]]), info = stat)
  }
  expect_true("data.table::frollapply" %in% names(fns$custom))

  set.seed(1)
  x <- rnorm(3000)
  for (n in c(11, 101)) {
    for (stat in setdiff(names(fns), "custom")) {
      ref <- rw$rw_reference(stat, x, n)
      for (fn_name in names(fns[[stat]])) {
        info <- paste(stat, fn_name, n)
        out <- rw$rw_normalize(fns[[stat]][[fn_name]](x, n), length(x))
        expect_length(out, length(x))
        expect_type(out, "double")
        # Every package must agree on complete windows (semantics of the
        # leading padding may differ and are reported, not tested).
        tail_idx <- n:length(x)
        if (stat != "sd") {
          expect_equal(out[tail_idx], ref[tail_idx], tolerance = 1e-8, info = info)
        }
        if (fn_name == "data.table") {
          expect_true(rw$rw_compare(out, ref)$ok, info = info)
        }
      }
    }
    ref_sum <- rw$rw_reference("sum", x, n)
    for (fn_name in names(fns$custom)) {
      out <- rw$rw_normalize(fns$custom[[fn_name]](x, n), length(x))
      expect_equal(out[n:length(x)], ref_sum[n:length(x)], tolerance = 1e-8,
                   info = paste("custom", fn_name, n))
    }
  }
})

test_that("rw_cuplyr_fun() runs natively on the GPU, never collects, matches data.table", {
  skip_if_no_gpu()
  rw <- load_rw_lib()
  skip_if_not_installed("data.table")
  df <- rw$rw_data(5000)
  gt <- tbl_gpu(df)

  for (alias in c(FALSE, TRUE)) {
    for (stat in c("mean", "sum", "min", "max", "sd")) {
      for (n in c(11, 101)) {
        info <- paste(stat, n, if (alias) "froll alias" else "roll_")
        f <- rw$rw_cuplyr_fun(stat, alias = alias)
        # The timed closure must not download anything: gpu_collect() is
        # what collect() and every CPU fallback route through.
        r <- with_mocked_bindings(
          {
            expect_no_condition(r <- f(gt, n), class = "cuplyr_cpu_fallback")
            r
          },
          gpu_collect = function(...) stop("gpu_collect() called inside the timed closure"),
          .package = "cuplyr"
        )
        expect_s3_class(r, "tbl_gpu")
        expect_false(is_lazy(r))
        expect_false(has_pending_ops(r))
        res <- dplyr::collect(r)
        expect_identical(names(res), c("x", "r"))
        cmp <- rw$rw_compare(res$r, rw$rw_reference(stat, df$x, n))
        expect_true(cmp$ok, info = info)
      }
    }
  }
})

test_that("rw_cuplyr_fun('median') is the labelled CPU fallback and is correct", {
  skip_if_no_gpu()
  rw <- load_rw_lib()
  skip_if_not_installed("data.table")
  df <- rw$rw_data(2000)
  gt <- tbl_gpu(df)
  f <- rw$rw_cuplyr_fun("median")
  expect_condition(r <- f(gt, 11), class = "cuplyr_cpu_fallback")
  expect_s3_class(r, "tbl_gpu")
  expect_true(rw$rw_compare(dplyr::collect(r)$r, rw$rw_reference("median", df$x, 11))$ok)
})

test_that("rw_gpu_finish() returns its input and works on any tbl_gpu", {
  skip_if_no_gpu()
  rw <- load_rw_lib()
  gt <- tbl_gpu(data.frame(x = c(1, 2, 3)))
  expect_identical(rw$rw_gpu_finish(gt), gt)
  empty <- tbl_gpu(data.frame(x = numeric(0)))
  expect_no_error(rw$rw_gpu_finish(empty))
})

test_that("rw_grouped_setup() entries agree with a per-group reference", {
  skip_if_no_gpu()
  rw <- load_rw_lib()
  skip_if_not_installed("data.table")
  dfg <- rw$rw_grouped_data(n_rows = 6000, n_groups = 20)
  expect_identical(names(dfg), c("g", "x"))
  expect_identical(nrow(dfg), 6000L)
  expect_lte(length(unique(dfg$g)), 20L)
  # Groups are interleaved, not pre-sorted.
  expect_true(is.unsorted(dfg$g))

  n <- 11
  ref <- stats::ave(dfg$x, dfg$g, FUN = function(v) data.table::frollmean(v, n))
  setup <- rw$rw_grouped_setup(dfg, n)
  expect_true(all(c("cuplyr", "data.table") %in% names(setup)))
  for (nm in names(setup)) {
    entry <- setup[[nm]]
    res <- entry$run()
    out <- entry$extract(res)
    expect_true(rw$rw_compare(out, ref)$ok, info = nm)
    # Running twice gives the same answer (data.table's := must not
    # accumulate state between timed samples).
    expect_true(rw$rw_compare(entry$extract(entry$run()), ref)$ok, info = nm)
  }
  if (requireNamespace("slider", quietly = TRUE)) {
    expect_true("slider" %in% names(setup))
  }
})

test_that("the cuplyr grouped run stays on the GPU", {
  skip_if_no_gpu()
  rw <- load_rw_lib()
  skip_if_not_installed("data.table")
  dfg <- rw$rw_grouped_data(n_rows = 3000, n_groups = 10)
  setup <- rw$rw_grouped_setup(dfg, 11)
  local_mocked_bindings(
    gpu_collect = function(...) stop("gpu_collect() called inside the timed closure"),
    .package = "cuplyr"
  )
  expect_no_condition(r <- setup$cuplyr$run(), class = "cuplyr_cpu_fallback")
  expect_s3_class(r, "tbl_gpu")
})
