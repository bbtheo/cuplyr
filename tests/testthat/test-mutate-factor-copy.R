# A bare factor-column copy in mutate() carries the levels to the new name.

test_that("mutate(h = f) copies a factor column with its levels", {
  skip_if_no_gpu()
  fd <- tibble::tibble(f = factor(c("a", "b", "a"), levels = c("b", "a", "z")), x = 1:3)
  expect_same_as_dplyr(fd, function(d) dplyr::mutate(d, h = f))
  expect_same_as_dplyr_lazy(fd, function(d) dplyr::mutate(d, h = f))
  expect_same_as_dplyr(fd, function(d) dplyr::transmute(d, h = f, x))
  expect_same_as_dplyr(fd, function(d) dplyr::mutate(d, f = x))
  expect_same_as_dplyr_lazy(fd, function(d) dplyr::mutate(d, h = f, f = x + 1L))
})

# Only a bare copy keeps a factor on the GPU; any other factor-valued
# expression runs the whole call on the CPU instead of computing on codes.
test_that("factor-valued window and transmute expressions match dplyr", {
  skip_if_no_gpu()
  fd <- tibble::tibble(f = factor(c("a", NA, "b")), f2 = factor(c("x", "y", "z")),
                       g = c(1, 1, 2), x = 1:3)
  pipelines <- list(
    function(d) dplyr::mutate(d, h = dplyr::lag(f)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), h = dplyr::first(f)),
    function(d) dplyr::transmute(d, h = dplyr::coalesce(f, f2), x),
    function(d) dplyr::mutate(d, h = f, k = dplyr::if_else(x > 1L, h, f))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(fd, p)
    expect_same_as_dplyr_lazy(fd, p)
  }
})
