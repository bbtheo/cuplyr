# if_any()/if_all() pieces that use cur_group_id() (F08 round-3 review, N2)

test_that("if_any() piece using cur_group_id() is type-checked like dplyr", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(4, 5, 6))
  expect_error(
    dplyr::mutate(dplyr::group_by(d, g), a = dplyr::if_any(c(x, y), ~ .x + dplyr::cur_group_id())),
    "logical vector"
  )
  expect_error(
    dplyr::mutate(dplyr::group_by(tbl_gpu(d), g), a = dplyr::if_any(c(x, y), ~ .x + dplyr::cur_group_id())),
    "`x` must be a logical vector, not a double vector"
  )
})

test_that("if_any() piece comparing cur_group_id() stays valid", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(4, 5, 6))
  expect_same_as_dplyr(d, function(t) {
    dplyr::mutate(dplyr::group_by(t, g), a = dplyr::if_any(c(x, y), ~ .x > dplyr::cur_group_id() + 3))
  })
})

test_that("grouped distinct() with a renamed factor key keeps the factor", {
  skip_if_no_gpu()
  fd <- tibble::tibble(g = c(1, 1, 2), f = factor(c("b", "a", "b"), levels = c("b", "a", "z")))
  expect_same_as_dplyr(fd, function(t) dplyr::distinct(dplyr::group_by(t, g), h = f),
                       arrange_by = c("g", "h"))
})
