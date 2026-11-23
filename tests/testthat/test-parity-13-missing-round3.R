# Extra cases for #13 (round 3): the summarise() check's verdicts and the dots it
# leaves alone, computed keys of group_by()/count()/distinct().

r3_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    }),
    error = function(e) e
  )
}

# The same class vector, parent class, call and full message as dplyr, eager and lazy.
r3_same_error <- function(df, pipeline, gpu_opts = list()) {
  cpu <- r3_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"))
  for (lazy in c(FALSE, TRUE)) {
    gpu <- withr::with_options(gpu_opts, r3_err(tbl_gpu(df, lazy = lazy), pipeline))
    expect_true(inherits(gpu, "error"))
    expect_identical(class(gpu), class(cpu))
    expect_identical(class(gpu$parent), class(cpu$parent))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu))
    expect_identical(rlang::format_error_call(gpu$call), rlang::format_error_call(cpu$call))
  }
}

r3_df <- function() {
  tibble::tibble(x = c(2L, 1L, 3L), y = c(1.5, NA, 3), g = c("a", "a", "b"))
}

test_that("group_by() names an unnamed computed dot that reads a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- r3_df()
  r3_same_error(df, function(d) group_by(d, x + yy))
  r3_same_error(df, function(d) group_by(d, x, .data$yy + 1))
  r3_same_error(df, function(d) group_by(d, a = x + 1, yy * 2))
})

test_that("summarise() keeps working on arguments R doesn't force and on data-frame dots", {
  skip_if_no_gpu()
  df <- r3_df()
  f <- function(d, ...) summarise(d, m = sum(x) + ..1)
  pipelines <- list(
    function(d) f(d, 5),
    function(d) summarise(d, m = mean(x, foo = zz)),
    function(d) summarise(d, dplyr::first(tibble::tibble(a = x)), b = mean(a)),
    function(d) summarise(d, (tibble::tibble(a = sum(x))), b = mean(a)),
    function(d) suppressWarnings(summarise(d, m = mean(g, na.rm = zz)))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("summarise() names a missing column read first by a reducer, also in strict mode", {
  skip_if_no_gpu()
  df <- r3_df()
  strict <- list(cuplyr.fallback = "error")
  k <- 2
  for (opts in list(list(), strict)) {
    r3_same_error(df, function(d) summarise(d, m = sd(xx)), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(d, m = var(xx, na.rm = TRUE)), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(d, m = mad(xx)), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(d, m = IQR(xx)), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(d, m = dplyr::first(xx)), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(d, m = n_distinct(x, xx)), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(d, m = mean(x) * k + xx), gpu_opts = opts)
    r3_same_error(df, function(d) summarise(group_by(d, g), m = mean(x) + xx), gpu_opts = opts)
  }
})

test_that("summarise() reports the first dot that fails, as dplyr does", {
  skip_if_no_gpu()
  df <- r3_df()
  # (dplyr reports `a`: it fails before `b` is evaluated)
  r3_same_error(df, function(d) summarise(d, a = log(g), b = mean(xx)))
  r3_same_error(df, function(d) summarise(d, a = sum(g), b = mean(xx)))
  # (`na.rm` is read before `x`)
  r3_same_error(df, function(d) summarise(d, m = var(x, na.rm = zz)))
  # (the promise fails first)
  f <- function(d, kk) summarise(d, m = kk + xx)
  r3_same_error(df, function(d) f(d, stop("boom")))
})

test_that("count(), add_count() and distinct() report a missing column in more computed keys as dplyr does", {
  skip_if_no_gpu()
  df <- r3_df()
  r3_same_error(df, function(d) count(d, a = x + 1, b = xx + 1))
  r3_same_error(df, function(d) count(d, a = x + .data$xx))
  r3_same_error(df, function(d) count(d, a = x + xx), gpu_opts = list(cuplyr.fallback = "error"))
  r3_same_error(df, function(d) distinct(d, x + xx))
  r3_same_error(df, function(d) distinct(d, a = mean(xx)))
  r3_same_error(df, function(d) add_count(d, x + xx))
  r3_same_error(df, function(d) add_count(group_by(d, g), a = x + xx))
})

test_that("count() and distinct() leave a `&&` key to dplyr", {
  skip_if_no_gpu()
  df <- r3_df()
  r3_same_error(df, function(d) count(d, a = x > 1 && y > 1))
  r3_same_error(df, function(d) distinct(d, a = x > 1 && y > 1))
  # (one row: `&&` of two length-1 columns is fine)
  p <- function(d) count(d, a = x > 1 && y > 1)
  expect_same_as_dplyr(df[1, ], p)
  p <- function(d) distinct(d, a = x > 1 && y > 1)
  expect_same_as_dplyr(df[1, ], p)
})
