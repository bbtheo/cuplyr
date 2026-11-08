# F79: a vector literal in mutate() is uploaded as a temp column; its name must
# not collide with an input column or with an output name.

test_that("a user column named like a vector temp does not replace the vector", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4, ..vec1.. = 11:14)
  v <- c(5L, 6L, 7L, 8L)
  for (pipeline in list(
    function(d) dplyr::mutate(d, z = v),
    function(d) dplyr::mutate(d, z = v, w = ..vec1.. + 1L),
    function(d) dplyr::mutate(d, z = c(TRUE, FALSE, TRUE, TRUE), z2 = c(1, 2, 3, 4))
  )) {
    expect_same_as_dplyr(df, pipeline)
    expect_same_as_dplyr_lazy(df, pipeline)
  }
})

test_that("an output named like a vector temp does not hide the vector", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4)
  pipeline <- function(d) dplyr::mutate(d, ..vec2.. = 1:4, ..vec1.. = c(9L, 8L, 7L, 6L), z = x + ..vec2..)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
  pipeline2 <- function(d) dplyr::mutate(d, a = c(1, 2, 3, 4), ..vec1.. = c(4, 3, 2, 1), b = a + ..vec1..)
  expect_same_as_dplyr(df, pipeline2)
})
