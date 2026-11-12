# F18 round 4 (R3-B1): a factor column spelled through `.data$`, `.data[[`
# or parentheses inside summarise() aggregations.

f18p_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")),
    o = factor(c("lo", "hi", "mid", NA, "hi", "lo"), levels = c("lo", "mid", "hi"), ordered = TRUE),
    g = c("a", "a", "b", "b", "b", "a")
  )
}

test_that("n_distinct() of a factor via pronoun/parentheses matches dplyr, eager and lazy", {
  skip_if_no_gpu()
  df <- f18p_df()
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(d) dplyr::summarise(d, k = dplyr::n_distinct(.data$f)),
    function(d) dplyr::summarise(d, k = dplyr::n_distinct((f))),
    function(d) dplyr::summarise(d, k = dplyr::n_distinct(.data$o, na.rm = TRUE))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
  grouped <- function(d) dplyr::summarise(dplyr::group_by(d, g), k = dplyr::n_distinct(.data[["f"]]))
  expect_same_as_dplyr(df, grouped, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, grouped, arrange_by = "g")
})

test_that("pronoun n_distinct() of a factor after a pending lazy op stays lazy-native", {
  skip_if_no_gpu()
  df <- f18p_df()
  withr::local_options(cuplyr.fallback = "error")
  p <- function(d) dplyr::summarise(dplyr::filter(d, g != "z"), k = dplyr::n_distinct(.data$f))
  expect_same_as_dplyr_lazy(df, p)
})

test_that("pronoun first()/max() of a factor still falls back with dplyr's result", {
  skip_if_no_gpu()
  df <- f18p_df()
  expect_same_as_dplyr(df, function(d) dplyr::summarise(d, v = dplyr::first(.data$f)))
  expect_same_as_dplyr(df, function(d) dplyr::summarise(d, v = max(.data$o, na.rm = TRUE)))
})
