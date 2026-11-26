# Issue 166: arrange() reads an ascending double key without nulls as it is,
# with no NaN-as-null pass: libcudf sorts NaN after every number and ties NaN
# with NaN, so the stable sort already puts NaN rows last in their original
# order, as dplyr does. These pin that for the shapes where it could go wrong:
# NaN of either sign next to +-Inf, NaN ties broken by a later key, a key with
# no NaN after one with NaN, the single-key and multi-key sort paths, and a
# NaN computed on the GPU. `k` records the original row order.

n166a_df <- function() {
  data.frame(
    x = c(3, NaN, -Inf, 1, -NaN, Inf, NaN, 1, -2, NaN, 0, -0),
    y = c(2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 3, 1),
    k = 1:12
  )
}

test_that("an ascending NaN key without nulls sorts NaN last, in original order", {
  skip_if_no_gpu()
  df <- n166a_df()
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x))
  expect_same_as_dplyr_lazy(df, function(d) dplyr::arrange(d, x))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x, y))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x, dplyr::desc(y)))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, y, x))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, dplyr::desc(y), x))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, dplyr::desc(x), y))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(dplyr::group_by(d, y), x, .by_group = TRUE))
})

test_that("an ascending NaN key computed on the GPU sorts with the uploaded ones", {
  skip_if_no_gpu()
  df <- n166a_df()
  expect_same_as_dplyr(df, function(d) dplyr::arrange(dplyr::mutate(d, r = x / y - x / y), r, k))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(dplyr::mutate(d, r = (x - 1) / (y - 1)), r))
  expect_same_as_dplyr_lazy(df, function(d) dplyr::arrange(dplyr::mutate(d, r = (x - 1) / (y - 1)), r))
})

test_that("a longer ascending NaN key keeps every NaN row's original order", {
  skip_if_no_gpu()
  set.seed(1662)
  n <- 5000L
  x <- round(runif(n, -5, 5))
  x[sample.int(n, 400L)] <- NaN
  x[sample.int(n, 50L)] <- -NaN
  x[sample.int(n, 20L)] <- c(Inf, -Inf)
  df <- data.frame(x = x, y = sample.int(3L, n, TRUE), k = seq_len(n))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x, y))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, y, x))
  # with an NA too, the key is nulled: NaN and NA tie
  df$x[c(7L, 4000L)] <- NA
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x))
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x, y))
})
