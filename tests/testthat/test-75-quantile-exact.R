# #75: a grouped quantile() is R's type 7 to the bit. The interpolation
# (1 - h) * x[lo] + h * x[hi] is written out from cudf's LOWER and HIGHER
# elements, each step its own operation, instead of cudf's LINEAR (a few ulps off).

test_that("quantile() over groups equals R's own, bit for bit", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(75)
  n <- 4000
  df <- data.frame(
    k = sample.int(30L, n, TRUE),
    x = c(round(rnorm(n - 2) * 10, 1), 1e300, -1e300),  # ties and far neighbours
    i = sample.int(1000L, n, TRUE)
  )
  probs <- unique(c(0, 1, 1 / 3, 2 / 3, 0.1, 0.7, 0.999, seq(0, 1, by = 0.05)))
  for (col in c("x", "i")) {
    got <- collect(reframe(group_by(tbl_gpu(df, lazy = FALSE), k), q = quantile(.data[[col]], probs)))
    want <- dplyr::reframe(dplyr::group_by(df, k), q = unname(quantile(.data[[col]], probs)))
    expect_identical(unname(got$q), want$q, info = col)
  }
})

test_that("a median and a quantile of one column share their request", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(7501)
  df <- data.frame(k = sample.int(5L, 200, TRUE), x = rnorm(200))
  got <- collect(summarise(group_by(tbl_gpu(df, lazy = FALSE), k),
                           m = median(x), a = quantile(x, 0.5), b = quantile(x, 0.25), c = quantile(x, 0.5)))
  want <- dplyr::summarise(dplyr::group_by(df, k), m = median(x), a = unname(quantile(x, 0.5)),
                           b = unname(quantile(x, 0.25)), c = unname(quantile(x, 0.5)))
  expect_equal(unname(as.list(got)), unname(as.list(want)))
  expect_identical(unname(got$a), unname(got$c))
})
