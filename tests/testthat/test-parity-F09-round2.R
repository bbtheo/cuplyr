# F09 round 2: guards, namespaced values, call-frame functions, embraced .fns.

r2_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

r2_native <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  r2_both(df, pipeline)
}

test_that("guards see earlier dots: transmute, unnamed dots, and if_any()", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  df <- tibble::tibble(x = c(1, 2), y = c(3, 4))
  w <- 5
  f <- ~ .x + w
  r2_both(df, function(d) dplyr::transmute(d, w = y, dplyr::across(x, f)))
  r2_both(df, function(d) dplyr::mutate(d, w = 1, dplyr::across(c(x, y), list(a = f, b = ~ .x * 2))))
  k <- 3
  r2_both(df, function(d) dplyr::summarise(d, k = 1, dplyr::across(x, mean, trim = k)))
  # Without the clash the same call stays native
  r2_native(df, function(d) dplyr::mutate(d, v = 1, dplyr::across(x, f)))
})

test_that("extra arguments that are constant expressions stay native", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  df <- tibble::tibble(x = c(14.26, 25.5, 36.9))
  r2_native(df, function(d) dplyr::mutate(d, dplyr::across(x, round, -1)))
  r2_native(df, function(d) dplyr::mutate(d, dplyr::across(x, round, digits = 2 - 1)))
})

test_that("namespaced values: only the IR's own function is native", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 4, 9))
  r2_native(df, function(d) dplyr::mutate(d, dplyr::across(x, base::sqrt)))
  r2_native(df, function(d) dplyr::mutate(d, dplyr::across(x, dplyr::lag)))
  # dplyr's stats::lag() leaves a `tsp` attribute on the column, which a GPU
  # column can't carry, so compare the values
  expect_equal(collect(dplyr::mutate(tbl_gpu(df), dplyr::across(x, stats::lag)))$x, c(1, 4, 9))
  expect_equal(collect(dplyr::mutate(as_lazy(tbl_gpu(df)), dplyr::across(x, stats::lag)))$x, c(1, 4, 9))
  r2_both(df, function(d) dplyr::summarise(d, dplyr::across(x, stats::median)))
})

test_that("functions that need their own frame are called, not inlined", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 4, 9))
  dots <- function(...) ..1 * 2
  brace <- function(v) {
    v + 1
  }
  r2_both(df, function(d) dplyr::mutate(d, dplyr::across(x, dots)))
  r2_both(df, function(d) dplyr::mutate(d, dplyr::across(x, brace)))
  fml <- ~ sapply(.x, function(v) v * 2)
  r2_both(df, function(d) dplyr::mutate(d, dplyr::across(x, fml)))
  r2_both(df, function(d) dplyr::mutate(d, dplyr::across(x, list(r = rev))))
})

test_that("an embraced .fns works in summarise, if_any, a closure and a list", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 3), x = c(1, 4, 9, 2, 5))
  a <- 7
  wrap <- function(d, fn) dplyr::mutate(d, dplyr::across(x, {{ fn }}))
  wrap_g <- function(d, fn) dplyr::summarise(dplyr::group_by(d, g), dplyr::across(x, {{ fn }}))
  wrap_f <- function(d, fn) dplyr::filter(d, dplyr::if_any(x, {{ fn }}))
  fq <- ~ .x + a
  r2_both(df, function(d) wrap(d, function(v) v + a))
  r2_both(df, function(d) wrap(d, fq))
  r2_both(df, function(d) wrap(d, list(a = fq, b = mean)))
  r2_both(df, function(d) wrap_g(d, list(m = mean, s = ~ sum(.x))))
  r2_both(df, function(d) wrap_f(d, ~ .x > 2))
  r2_native(df, function(d) wrap(d, sqrt))
  r2_native(df, function(d) wrap_g(d, mean))
})

test_that("only .data[[...]] counts as string access to columns", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 4, 9), y = c(2, 3, 5))
  idx <- c(2, 3)
  r2_native(df, function(d) {
    dplyr::summarise(d, dplyr::across(c(x, y), list(a = ~ max(.x), b = ~ max(.x / idx[[1]]))))
  })
  r2_both(df, function(d) dplyr::mutate(d, dplyr::across(c(x, y), ~ .x + .data[["y"]])))
})
