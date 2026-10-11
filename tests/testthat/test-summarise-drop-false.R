# F33 round 2: zero-dots summarise on .drop = FALSE groups with pending ops, and
# no CPU-fallback notice for the GPU materialization.

f33r2_df <- data.frame(
  f = factor(c("a", "a", "c", "a"), levels = c("a", "b", "c")),
  g = factor(c("x", "y", "x", "x"), levels = c("x", "y", "z")),
  v = 1:4
)

test_that("zero-dots .drop = FALSE summarise after pending lazy ops matches dplyr", {
  skip_if_no_gpu()
  pipeline <- function(d) {
    d |> dplyr::filter(v > 1) |> dplyr::group_by(f, g, .drop = FALSE) |>
      dplyr::summarise(.groups = "drop_last")
  }
  expected <- pipeline(f33r2_df)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(pipeline(tbl_gpu(f33r2_df, lazy = lazy)))
    expect_equal(as.data.frame(got), as.data.frame(expected), ignore_attr = TRUE)
  }
})

test_that("a lazy .drop = FALSE summarise/count with pending ops gives no fallback notice", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  lz <- tbl_gpu(f33r2_df, lazy = TRUE) |> dplyr::filter(v > 1)
  expect_no_error(collect(dplyr::count(dplyr::group_by(lz, f, .drop = FALSE))))
  expect_no_error(collect(dplyr::summarise(dplyr::group_by(lz, f, .drop = FALSE))))
})

test_that("zero-dots .drop = FALSE result is ungrouped (drop = TRUE) with .groups = 'drop'", {
  skip_if_no_gpu()
  res <- tbl_gpu(f33r2_df) |> dplyr::group_by(f, .drop = FALSE) |> dplyr::summarise()
  expect_equal(nrow(collect(res)), 3L)
  expect_true(isTRUE(res$group_drop))
})
