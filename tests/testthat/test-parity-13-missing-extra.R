# Extra cases for #13 (round 2): missing-column checks that must not fire on
# names dplyr never reaches, and the arrange()/group_by() shapes they route.

x13_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    }),
    error = function(e) e
  )
}

# The same class vector, call, bullets and full message as dplyr, eager and lazy.
x13_same_error <- function(df, pipeline, gpu_opts = list()) {
  cpu <- x13_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"))
  for (lazy in c(FALSE, TRUE)) {
    gpu <- withr::with_options(gpu_opts, x13_err(tbl_gpu(df, lazy = lazy), pipeline))
    expect_true(inherits(gpu, "error"))
    expect_identical(class(gpu), class(cpu))
    expect_identical(class(gpu$parent), class(cpu$parent))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu))
    expect_identical(rlang::format_error_call(gpu$call), rlang::format_error_call(cpu$call))
  }
}

x13_df <- function() {
  tibble::tibble(x = c(2L, 1L, 3L), y = c(1.5, NA, 3), g = c("a", "a", "b"))
}

test_that("summarise() keeps working where the unknown name is behind a call that may not run it", {
  skip_if_no_gpu()
  df <- x13_df()
  v <- 3
  pipelines <- list(
    function(d) summarise(d, m = mean(x) + if (FALSE) zz else 0),
    function(d) summarise(d, m = TRUE || sum(zz)),
    function(d) summarise(d, m = mean(x), k = (function(z) 1)(zz)),
    function(d) summarise(d, m = mean(x) + .env$v),
    function(d) summarise(d, m = local({ w <- 2; mean(x) * w })),
    function(d) summarise(d, m = with(list(w = 2), mean(x) * w))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("summarise() names a missing column read through operators and reducers", {
  skip_if_no_gpu()
  df <- x13_df()
  x13_same_error(df, function(d) summarise(d, m = (mean(xx) + 1) * 2))
  x13_same_error(df, function(d) summarise(d, m = sum(x, na.rm = TRUE) > mean(xx)))
  x13_same_error(df, function(d) summarise(d, a = mean(x), b = -sum(.data$xx)))
  x13_same_error(df, function(d) summarise(d, a = mean(x), a = sum(x), b = mean(xx)))
})

test_that("arrange() raises dplyr's error for a missing column inside an expression or a pronoun", {
  skip_if_no_gpu()
  df <- x13_df()
  strict <- list(cuplyr.fallback = "error")
  x13_same_error(df, function(d) arrange(d, xx + 1), gpu_opts = strict)
  x13_same_error(df, function(d) arrange(d, x, desc(.data$xx)), gpu_opts = strict)
  x13_same_error(df, function(d) arrange(d, -.data$xx), gpu_opts = strict)
})

test_that("arrange() still sorts by an expression of columns on the CPU route", {
  skip_if_no_gpu()
  df <- x13_df()
  p <- function(d) arrange(d, x + y, desc(.data$x))
  expect_same_as_dplyr(df, p)
})

test_that("group_by() with a computed dot raises dplyr's error for a missing .data pronoun or bare name", {
  skip_if_no_gpu()
  df <- x13_df()
  x13_same_error(df, function(d) group_by(d, a = x + 1, .data$xx))
  x13_same_error(df, function(d) group_by(d, xx, yy, a = x + 1))
  k <- 1
  p <- function(d) group_by(d, a = x + 1, k)
  expect_same_as_dplyr(df, p, arrange_by = c("x"))
})
