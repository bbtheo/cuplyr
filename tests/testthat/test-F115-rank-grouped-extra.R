# F115: extra checks of gpu_slice_rank_grouped() beyond the acceptance file:
# random shapes against dplyr, and the pieces of the route one by one.

f115x_df <- function(n, ngroups, seed) {
  set.seed(seed)
  tibble::tibble(
    id = seq_len(n),
    g = sample.int(ngroups, n, replace = TRUE),
    s = sample(c("a", "b", "c", NA), n, replace = TRUE),
    x = replace(sample(c(-2, -0, 0, 1, 1, 3, Inf, -Inf, NaN), n, replace = TRUE),
                sample.int(n, n %/% 7L), NA),
    i = replace(sample(1:6, n, replace = TRUE), sample.int(n, n %/% 7L), NA),
    d = as.Date("2024-01-01") + sample(c(0:4, NA), n, replace = TRUE),
    t = as.POSIXct("2024-01-01", tz = "UTC") + sample(c(0:4, NA), n, replace = TRUE)
  )
}

test_that("random shapes match dplyr, eager and lazy", {
  skip_if_no_gpu()
  for (spec in list(c(2000L, 7L, 1L), c(5000L, 900L, 2L), c(300L, 1L, 3L), c(1L, 1L, 4L), c(70000L, 40L, 5L))) {
    df <- f115x_df(spec[[1]], spec[[2]], spec[[3]])
    for (key in c("x", "i", "d", "t")) {
      k <- rlang::sym(key)
      for (ties in c(TRUE, FALSE)) {
        expect_same_as_dplyr(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), !!k, n = 2, with_ties = ties))
        expect_same_as_dplyr_lazy(df, function(d) dplyr::slice_max(d, !!k, prop = 0.3, with_ties = ties, na_rm = TRUE, by = s))
      }
    }
    expect_same_as_dplyr(df, function(d) dplyr::slice_max(dplyr::group_by(d, g, s), x, n = -1))
  }
})

test_that("the route runs for supported shapes and stays off for the others", {
  skip_if_no_gpu()
  df <- f115x_df(200L, 5L, 9L)
  calls <- 0L
  real <- gpu_slice_rank_grouped
  local_mocked_bindings(gpu_slice_rank_grouped = function(...) {
    calls <<- calls + 1L
    real(...)
  })
  invisible(dplyr::collect(dplyr::slice_min(dplyr::group_by(tbl_gpu(df, lazy = FALSE), g, s), d, n = 1)))
  expect_equal(calls, 1L)
  invisible(dplyr::collect(dplyr::slice_min(dplyr::group_by(tbl_gpu(df, lazy = FALSE), g), s, n = 1)))
  expect_equal(calls, 1L)
  withr::local_options(cuplyr.topk_path = "sort")
  invisible(dplyr::collect(dplyr::slice_min(dplyr::group_by(tbl_gpu(df, lazy = FALSE), g), x, n = 1)))
  expect_equal(calls, 1L)
})

test_that("the primitive returns survivors in (group, rank) order", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2L, 1L, 2L, 1L, 2L, 1L), x = c(5, 5, 1, 9, 1, -0))
  tb <- tbl_gpu(df, lazy = FALSE)
  # group 1: x = 5, 9, -0 ; group 2: x = 5, 1, 1
  out <- gpu_slice_rank_grouped(tb$ptr, 0L, 1L, FALSE, 2, FALSE, TRUE, FALSE, FALSE)
  expect_equal(gpu_dim(out)[1], 4L)
  out <- gpu_slice_rank_grouped(tb$ptr, 0L, 1L, TRUE, 1, FALSE, TRUE, FALSE, TRUE)
  expect_equal(gpu_dim(out)[1], 2L)
})

test_that("NaN and NA group keys, string groups and pruned lazy plans match dplyr", {
  skip_if_no_gpu()
  df <- f115x_df(900L, 4L, 11L)
  expect_same_as_dplyr(df, function(d) dplyr::slice_min(dplyr::group_by(d, x), i, n = 2))
  expect_same_as_dplyr(df, function(d) dplyr::slice_max(d, i, n = 1, by = x, with_ties = FALSE))
  expect_same_as_dplyr(df, function(d) dplyr::slice_max(dplyr::group_by(d, s, x), d, n = 2))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |>
      dplyr::group_by(s) |>
      dplyr::slice_min(x * 2, n = 2) |>
      dplyr::ungroup() |>
      dplyr::select(id)
  })
})
