# F09 coder tests beyond the PM's acceptance file.

test_that("grouped cur_group_id() in a lazy mutate() runs (join result is materialized)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), z = c(2, 1, 2))
  p <- function(d) dplyr::mutate(dplyr::group_by(d, z), a = x, n = dplyr::cur_group_id())
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("if_any()/if_all() with a non-inlinable lambda falls back to dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 9), y = c(2, 6, 1), k = c(1L, 2L, 1L))
  for (pipe in list(
    function(d) dplyr::filter(d, dplyr::if_any(c(x, y), function(v) return(v > 5))),
    function(d) dplyr::filter(dplyr::group_by(d, k), dplyr::if_all(c(x, y), ~ ..1 > 0))
  )) {
    expect_same_as_dplyr(df, pipe)
    expect_same_as_dplyr_lazy(df, pipe)
  }
})

test_that("extra across() arguments: constants stay native, column references fall back", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  df <- tibble::tibble(x = c(1, NA, 3), y = c(4, 5, NA))
  expect_same_as_dplyr(df, function(d) dplyr::summarise(d, dplyr::across(c(x, y), mean, na.rm = TRUE)))
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_same_as_dplyr(df, function(d) dplyr::summarise(d, dplyr::across(c(x, y), mean, na.rm = TRUE)))
  })
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, dplyr::across(x, function(a, b) a + b, b = y)))
})

test_that("staged across() keeps .keep = used and group columns right", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(4, 5, 6))
  p <- function(d) dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, y), ~ .x / y), .keep = "used")
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})
