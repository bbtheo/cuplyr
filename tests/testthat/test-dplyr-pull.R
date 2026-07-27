# Tests for pull() (Phase 3, task 3)
#
# pull() is a terminal verb (returns a plain R vector, not a tbl_gpu) so it
# doesn't fit expect_same_as_dplyr()'s data-frame-in/data-frame-out oracle
# shape (see helper-oracle.R) -- each test instead compares the returned
# vector directly against dplyr::pull() on an equivalent tibble. See
# R/pull.R for the empirically-verified semantics (dplyr::pull.data.frame()
# is, verbatim, `tidyselect::vars_pull(names(.data), !!enquo(var))`, reused
# here unchanged).
#
# The dedicated "pull() returns a plain vector matching dplyr" case
# previously lived in test-fallback.R (from when pull() was CPU-fallback,
# Phase 2); it has moved here now that pull() has its own file/tests.

oracle_pull_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3, 4, 5),
    y = c(10, 20, 30, 40, 50),
    g = c(1, 1, 2, 2, 3),
    s = c("a", "b", "c", "d", "e")
  )
}

test_that("pull() by bare column name matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df)

  expect_equal(dplyr::pull(gt, x), dplyr::pull(df, x))
  expect_false(is_tbl_gpu(dplyr::pull(gt, x)))
})

test_that("pull() with no var defaults to the last column", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df)

  expect_equal(dplyr::pull(gt), dplyr::pull(df))
})

test_that("pull() by positive position matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df)

  expect_equal(dplyr::pull(gt, 1), dplyr::pull(df, 1))
  expect_equal(dplyr::pull(gt, 2), dplyr::pull(df, 2))
})

test_that("pull() by negative position matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df)

  expect_equal(dplyr::pull(gt, -1), dplyr::pull(df, -1))
  expect_equal(dplyr::pull(gt, -2), dplyr::pull(df, -2))
})

test_that("pull() with name= returns a named vector matching dplyr", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df)

  expect_equal(dplyr::pull(gt, y, name = s), dplyr::pull(df, y, name = s))
  expect_equal(dplyr::pull(gt, x, name = g), dplyr::pull(df, x, name = g))
})

test_that("pull() on a grouped table ignores grouping, matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  gdf <- dplyr::group_by(df, g)

  expect_equal(dplyr::pull(gt, x), dplyr::pull(gdf, x))
})

test_that("pull() materializes a lazy pipeline before pulling", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  lazy_gt <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 2) |> dplyr::mutate(z = x * 2)

  result <- dplyr::pull(lazy_gt, z)
  oracle <- dplyr::pull(dplyr::mutate(dplyr::filter(df, x > 2), z = x * 2), z)

  expect_equal(result, oracle)
})

test_that("pull() on an already-eager tbl_gpu works directly", {
  skip_if_no_gpu()
  df <- oracle_pull_df()
  gt <- tbl_gpu(df, lazy = FALSE)

  expect_equal(dplyr::pull(gt, y), dplyr::pull(df, y))
})
