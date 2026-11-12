# across() output names that a later expansion of the same across() reads.
# dplyr evaluates every function against the pre-across() data, so these go
# to the CPU fallback instead of running as sequential dots.

test_that("an output name read by a later expansion matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30), z = c(2, 2, 2))
  pipelines <- list(
    function(d) dplyr::mutate(d, dplyr::across(c(y = x, z = y), ~ .x + y)),
    function(d) dplyr::mutate(d, dplyr::across(c(x, y), ~ .x + x)),
    function(d) dplyr::mutate(d, dplyr::across(c(a = x, b = y), ~ .x + z)),
    function(d) dplyr::transmute(d, dplyr::across(c(y = x, z = y), ~ .x + y)),
    function(d) dplyr::summarise(d, dplyr::across(c(y = x, z = y), ~ mean(.x) + sum(z))),
    function(d) {
      dplyr::mutate(dplyr::group_by(d, z), dplyr::across(c(y = x, x = y), identity),
                    n = dplyr::cur_group_id())
    }
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("if_any()/if_all() reject any non-logical piece", {
  skip_if_no_gpu()
  df <- data.frame(w = c(TRUE, FALSE, NA), y = 1:3, z = c(1.5, 2, 3), s = c("a", "b", "c"))
  for (d in list(tbl_gpu(df), tbl_gpu(df, lazy = TRUE))) {
    expect_error(dplyr::filter(d, dplyr::if_any(c(y, z), ~ .x + 1)), "`y` must be a logical vector, not a double vector")
    expect_error(dplyr::filter(d, dplyr::if_all(z, ~ .x * 2)), "`z` must be a logical vector, not a double vector")
    expect_error(dplyr::mutate(d, a = dplyr::if_any(s, identity)), "`s` must be a logical vector, not a character vector")
    expect_error(dplyr::mutate(d, a = dplyr::if_any(c(w, y), ~ 1)), "must be a logical vector")
  }
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, dplyr::if_any(c(w, y), ~ is.na(.x))))
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, dplyr::if_all(c(y, z), ~ .x > 1)))
})

test_that("if_any()/if_all() name a factor or Date column like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "b")), dt = as.Date("2020-01-01") + 0:1)
  d <- tbl_gpu(df)
  expect_error(dplyr::filter(d, dplyr::if_any(f, identity)), "not a <factor> object")
  expect_error(dplyr::filter(d, dplyr::if_any(dt, identity)), "not a <Date> object")
})

test_that("a cur_column() lookup of an overwritten source column matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 5:7)
  p <- function(d) dplyr::mutate(d, dplyr::across(c(y = x, z = y), ~ get(dplyr::cur_column()) * 10L))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})
