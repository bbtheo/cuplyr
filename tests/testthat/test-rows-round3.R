# F26 review round 3: pieces the acceptance tests do not pin.

test_that("rows_patch() gives NA where x and a GPU-computed y value are both missing", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  x <- tibble::tibble(id = 1:3, v = c(NA, 0, 5))
  y <- tibble::tibble(id = 1:2, v = c(0, 0))
  want <- dplyr::rows_patch(dplyr::mutate(x, v = v / v), dplyr::mutate(y, v = v / v), by = "id")
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::rows_patch(dplyr::mutate(tbl_gpu(x, lazy = lazy), v = v / v),
                             dplyr::mutate(tbl_gpu(y, lazy = lazy), v = v / v), by = "id")
    got <- tibble::as_tibble(collect(got))
    expect_identical(is.nan(got$v), is.nan(want$v))
    expect_identical(is.na(got$v), is.na(want$v))
    expect_equal(got, want)
  }
})

test_that("a .drop = FALSE rows_delete() that empties no group keeps summarise() native", {
  skip_if_no_gpu()
  x <- tibble::tibble(id = 1:4, g = c("a", "b", "a", "b"), v = c(1, 2, 3, 4))
  y <- tibble::tibble(id = c(1L, 4L))
  want <- dplyr::rows_delete(dplyr::group_by(x, g, .drop = FALSE), y, by = "id")
  for (lazy in c(FALSE, TRUE)) {
    gx <- dplyr::group_by(tbl_gpu(x, lazy = lazy), g, .drop = FALSE)
    out <- dplyr::rows_delete(gx, tbl_gpu(y), by = "id")
    # no group was lost, so no fixed keys: later verbs see a plain regroup
    expect_null(fixed_keys_of(as_eager(out)))
    expect_equal(dplyr::group_data(out), dplyr::group_data(want))
    withr::with_options(list(cuplyr.fallback = "error"), {
      s <- dplyr::summarise(out, n = dplyr::n(), v = sum(v))
      expect_equal(tibble::as_tibble(collect(s)),
                   dplyr::summarise(want, n = dplyr::n(), v = sum(v)))
    })
  }
  # emptying a group still fixes the keys
  out <- dplyr::rows_delete(dplyr::group_by(tbl_gpu(x), g, .drop = FALSE),
                            tbl_gpu(tibble::tibble(id = c(2L, 4L))), by = "id")
  want <- dplyr::rows_delete(dplyr::group_by(x, g, .drop = FALSE),
                             tibble::tibble(id = c(2L, 4L)), by = "id")
  expect_false(is.null(fixed_keys_of(out)))
  expect_equal(dplyr::group_data(out), dplyr::group_data(want))
})
