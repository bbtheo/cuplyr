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
  g <- group_by(tbl_gpu(df, lazy = FALSE), k)
  got_x <- collect(reframe(g, q = quantile(x, probs)))
  got_i <- collect(reframe(g, q = quantile(i, probs)))
  by_k <- dplyr::group_by(df, k)
  expect_identical(unname(got_x$q), dplyr::reframe(by_k, q = unname(quantile(x, probs)))$q)
  expect_identical(unname(got_i$q), dplyr::reframe(by_k, q = unname(quantile(i, probs)))$q)
})

test_that("a median and quantiles of one column share their request", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(7501)
  df <- data.frame(k = sample.int(5L, 200, TRUE), x = rnorm(200))
  got <- collect(summarise(group_by(tbl_gpu(df, lazy = FALSE), k),
                           m = median(x), a = quantile(x, 0.5), b = quantile(x, 0.25), c = quantile(x, 0.5)))
  want <- dplyr::summarise(dplyr::group_by(df, k), m = median(x), a = unname(quantile(x, 0.5)),
                           b = unname(quantile(x, 0.25)), c = unname(quantile(x, 0.5)))
  strip <- function(d) unname(lapply(as.list(d), unname))
  expect_equal(strip(got), strip(want))
  expect_identical(unname(got$a), unname(got$c))
})
