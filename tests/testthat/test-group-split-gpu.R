# group_split() regressions: message parity with dplyr and the unspecified()
# marker on chunks.

test_that("group_split(keep =) is defunct with dplyr's message", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1)
  msg <- function(expr) conditionMessage(tryCatch(expr, error = identity))
  want <- msg(dplyr::group_split(df, keep = TRUE))
  expect_match(want, "dplyr 1.0.0")
  expect_identical(msg(dplyr::group_split(tbl_gpu(df), keep = TRUE)), want)
  expect_identical(
    msg(dplyr::group_split(dplyr::group_by(tbl_gpu(df), x), keep = TRUE)),
    msg(dplyr::group_split(dplyr::group_by(df, x), keep = TRUE))
  )
})

test_that("group_split() keeps vctrs::unspecified() columns on chunks", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), u = vctrs::unspecified(3))
  want <- dplyr::group_split(df, g)

  got <- dplyr::group_split(tbl_gpu(df), g)
  expect_length(got, length(want))
  for (i in seq_along(want)) {
    expect_identical(tibble::as_tibble(dplyr::collect(got[[i]])), want[[i]])
  }
  expect_identical(
    tibble::as_tibble(dplyr::collect(attr(got, "ptype"))),
    attr(want, "ptype")
  )

  got0 <- dplyr::group_split(tbl_gpu(df))
  expect_identical(tibble::as_tibble(dplyr::collect(got0[[1]])), dplyr::group_split(df)[[1]])
})

test_that("group_split() <grouped_df> ignoring `...` warns with dplyr's text", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), y = 1:3)
  msg <- function(expr) {
    conditionMessage(tryCatch(expr, warning = identity))
  }
  want <- msg(dplyr::group_split(dplyr::group_by(df, g), y))
  expect_match(want, "ignores `...`")
  expect_identical(msg(dplyr::group_split(dplyr::group_by(tbl_gpu(df), g), y)), want)
})

test_that("gpu_split_rows() rejects sizes beyond the table and bad indices", {
  skip_if_no_gpu()
  gt <- tbl_gpu(tibble::tibble(a = 1:3))
  expect_error(gpu_split_rows(gt$ptr, c(0L, 1L, 2L, 0L), c(2L, 2L)), "exceed")
  expect_error(gpu_split_rows(gt$ptr, c(0L, 1L, 5L), c(3L)), "out of range")
  expect_error(gpu_split_rows(gt$ptr, c(0L, 1L, -1L), c(3L)), "out of range")
  expect_error(gpu_split_rows(gt$ptr, integer(), c(2L, 2L)), "group_split")
})

test_that("group_split(.keep = FALSE) keeps vctrs::unspecified() columns on chunks", {
  skip_if_no_gpu()
  check <- function(got, want) {
    expect_length(got, length(want))
    for (i in seq_along(want)) {
      expect_identical(tibble::as_tibble(dplyr::collect(got[[i]])), want[[i]])
    }
    expect_identical(
      tibble::as_tibble(dplyr::collect(attr(got, "ptype"))),
      attr(want, "ptype")
    )
  }
  for (df in list(
    tibble::tibble(g = c(1, 1, 2), u = vctrs::unspecified(3)),
    tibble::tibble(g = c(1, 1, 2), x = 1:3, u = vctrs::unspecified(3))
  )) {
    want <- dplyr::group_split(df, g, .keep = FALSE)
    want_grp <- dplyr::group_split(dplyr::group_by(df, g), .keep = FALSE)
    for (mk in list(tbl_gpu, function(d) as_lazy(tbl_gpu(d)))) {
      check(dplyr::group_split(mk(df), g, .keep = FALSE), want)
      check(dplyr::group_split(dplyr::group_by(mk(df), g), .keep = FALSE), want_grp)
    }
  }
})

test_that("group_split() with computed keys keeps vctrs::unspecified() columns", {
  skip_if_no_gpu()
  check <- function(got, want) {
    expect_length(got, length(want))
    for (i in seq_along(want)) {
      expect_identical(tibble::as_tibble(dplyr::collect(got[[i]])), want[[i]])
    }
    expect_identical(
      tibble::as_tibble(dplyr::collect(attr(got, "ptype"))),
      attr(want, "ptype")
    )
  }
  df <- tibble::tibble(g = c(1, 1, 2), u = vctrs::unspecified(3))
  for (mk in list(tbl_gpu, function(d) as_lazy(tbl_gpu(d)))) {
    check(dplyr::group_split(mk(df), k = g * 2), dplyr::group_split(df, k = g * 2))
    check(
      dplyr::group_split(mk(df), k = g * 2, .keep = FALSE),
      dplyr::group_split(df, k = g * 2, .keep = FALSE)
    )
    # a computed key that overwrites the unspecified column is a plain double
    check(dplyr::group_split(mk(df), u = g * 2), dplyr::group_split(df, u = g * 2))
    check(
      dplyr::group_split(mk(df), u = g * 2, .keep = FALSE),
      dplyr::group_split(df, u = g * 2, .keep = FALSE)
    )
  }
})
