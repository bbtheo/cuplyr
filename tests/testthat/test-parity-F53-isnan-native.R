# F53 round 3 (coder): is.nan() shapes, and reducers over
# context calls keep their native summarise route (review-2 note N1).

test_that("is.nan() of a column-free argument folds and stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  th <- NaN
  pipe <- function(d) dplyr::mutate(d, a = is.nan(0/0), b = is.nan(-th))
  want <- pipe(df)
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(pipe(tbl_gpu(df, lazy = lazy)))
    expect_identical(got, want, info = paste("lazy =", lazy))
  }
})

# is.nan() of a column goes to the CPU route (F53-a2); parity only, no native pin.
test_that("is.nan() of a column expression matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  p <- function(d) dplyr::mutate(d, c = is.nan(x + 0/0), e = is.nan((x - x) / (x - x)))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("is.nan() over a NaN literal in a window default matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  p <- function(d) dplyr::mutate(d, c = is.nan(dplyr::lag(x, default = NaN)))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("reducers over row_number() stay native in summarise()", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  pipe <- function(d) dplyr::summarise(dplyr::group_by(d, g),
                                       m = max(dplyr::row_number()),
                                       s = sum(dplyr::row_number()))
  want <- pipe(df)
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(pipe(tbl_gpu(df, lazy = lazy)))
    expect_equal(tibble::as_tibble(got), tibble::as_tibble(want), info = paste("lazy =", lazy))
  }
})
