# F34 round 2: nested-reducer routing and the unnamed-expression warning.

test_that("summarise(): unnamed splice that falls back does not warn", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3))
  expect_no_warning(
    out <- collect(summarise(tbl_gpu(df), tibble::tibble(y = sum(x) * 2, z = 3)))
  )
  expect_equal(out$y, 12)
  expect_equal(out$z, 3)
})

test_that("summarise(): unnamed expression on the GPU path still warns", {
  skip_if_no_gpu()
  expect_warning(
    summarise(tbl_gpu(data.frame(x = 1:3)), sum(x)),
    "will use expression as column name"
  )
})

test_that("nested reducer inside another window of the argument matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), n = c(1L, 2L, 3L, 4L, 5L))
  for (pipe in list(
    function(d) d |> summarise(a = sum(dplyr::consecutive_id(mean(n)))),
    function(d) d |> summarise(a = sum(cumsum(mean(n)))),
    function(d) d |> summarise(a = sum(dplyr::percent_rank(mean(n)))),
    function(d) d |> summarise(a = sum(n - sum(mean(n))))
  )) {
    expect_same_as_dplyr(df, pipe)
    expect_same_as_dplyr_lazy(df, pipe)
  }
})
