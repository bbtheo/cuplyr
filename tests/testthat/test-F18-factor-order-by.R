# F18 rule: `-f` is arithmetic on a factor, also as a window's order_by.
# dplyr warns and ignores the order; `desc(f)` is a sort key and stays native.

test_that("a window's order_by = -f on a factor warns like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    f = factor(c("b", "a", "c", "a")),
    n = c(1L, 2L, 3L, 4L)
  )
  pipe <- function(d) dplyr::mutate(d, y = dplyr::lag(n, order_by = -f))
  expect_warning(expected <- pipe(df), "not meaningful for factors")
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(out <- dplyr::collect(pipe(tbl_gpu(df, lazy = lazy))), "not meaningful for factors")
    expect_equal(tibble::as_tibble(out), expected)
  }
})

test_that("a window's order_by = desc(f) on a factor stays on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    f = factor(c("b", "a", "c", "a")),
    n = c(1L, 2L, 3L, 4L)
  )
  pipe <- function(d) dplyr::mutate(d, y = dplyr::lag(n, order_by = dplyr::desc(f)))
  expect_equal(tibble::as_tibble(dplyr::collect(pipe(tbl_gpu(df)))), pipe(df))
})
