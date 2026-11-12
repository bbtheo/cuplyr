test_that("arrange() validates .locale even with no dots", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(a = 1:3))
  expect_error(arrange(g, .locale = 1), "`.locale` must be a string or `NULL`")
  expect_error(arrange(g, .locale = c("en_US", "fr_BF")))
  expect_error(arrange(g, .locale = "x"), "stringi::stri_locale_list")
  expect_no_error(arrange(g, .locale = "C"))
})

test_that("arrange(.by_group = TRUE) with no dots sorts by the groups", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 1, 2, 1), x = 1:4)
  expect_same_as_dplyr(df, function(d) d |> group_by(g) |> arrange(.by_group = TRUE))
  expect_same_as_dplyr(df, function(d) d |> arrange(.by_group = TRUE))
  expect_same_as_dplyr_lazy(df, function(d) d |> group_by(g) |> arrange(.by_group = TRUE))
})

test_that("dplyr.legacy_locale sorts strings in the system locale", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("b", "B", "a", "A"), x = 1:4)
  withr::local_options(dplyr.legacy_locale = TRUE, lifecycle_verbosity = "quiet")
  expect_same_as_dplyr(df, function(d) d |> arrange(s))
  expect_same_as_dplyr(df, function(d) d |> arrange(s, .locale = "C"))
})

test_that("pick() with an empty argument slot falls back instead of crashing", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(2, 1, 3), b = 1:3)
  expect_same_as_dplyr(df, function(d) d |> arrange(pick(a, )))
  expect_same_as_dplyr(df, function(d) d |> arrange(pick(c(a, ), b)))
})
