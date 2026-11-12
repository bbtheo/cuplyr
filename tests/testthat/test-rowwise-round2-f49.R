# F49 round-2 additions: small behaviours fixed with the review-1 findings.

test_that("group_trim() on a rowwise tbl_gpu returns it unchanged, natively", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  withr::local_options(cuplyr.fallback = "error")
  got <- dplyr::group_trim(dplyr::rowwise(tbl_gpu(df), x))
  expect_true(is_tbl_gpu(got))
  expect_identical(collect(got), dplyr::group_trim(dplyr::rowwise(df, x)))
})

test_that("the implicit-keep rowwise summarise() message lists the vars like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:2, y = 3:4, z = 5:6)
  for (vars in list("x", c("x", "y"), c("x", "y", "z"))) {
    want <- testthat::capture_messages(
      dplyr::summarise(dplyr::rowwise(df, dplyr::all_of(vars)), s = x + 1)
    )
    got <- testthat::capture_messages(
      dplyr::summarise(dplyr::rowwise(tbl_gpu(df), dplyr::all_of(vars)), s = x + 1)
    )
    expect_identical(got, want)
  }
})

test_that("whole-column [<- with several columns and NULL keeps a rowwise table rowwise", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6, z = 7:9)
  want <- dplyr::rowwise(df, x, y)
  want[c("y", "z")] <- list(0L, 1L)
  got <- dplyr::rowwise(tbl_gpu(df), x, y)
  got[c("y", "z")] <- list(0L, 1L)
  expect_identical(collect(got), want)

  want["y"] <- NULL
  got["y"] <- NULL
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(want))
  expect_identical(collect(got), want)

  # a row subscript still errors
  g <- dplyr::rowwise(tbl_gpu(df), x)
  expect_error(g[1, "y"] <- 5L, "mutate\\(\\)")
})
