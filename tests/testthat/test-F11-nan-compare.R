test_that("NaN literals under comparison-like ops are NA, as in R", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    x = c(1, 2, NA), i = c(1L, 2L, NA),
    d = as.Date("2020-01-01") + c(0, 1, NA)
  )
  pipeline <- function(d) {
    dplyr::mutate(d,
      b = dplyr::between(x, 0, NaN),
      nr = dplyr::near(x, NaN),
      cw = dplyr::case_when(x > NaN ~ "a", .default = "z"),
      ie = dplyr::if_else(x == NaN, 1, 2),
      ii = i > NaN,
      dd = d > NaN,
      ar = (x + NaN) > 0,
      ar2 = abs(x * NaN - 1) <= 3,
      v = x + NaN
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, (x - NaN) < 5 | is.na(x)))
  expect_same_as_dplyr_lazy(df, function(d) dplyr::filter(d, dplyr::between(x, NaN, 5)))
})
