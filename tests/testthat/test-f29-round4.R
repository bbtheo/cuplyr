# F29 round 4: extra checks beside the protected acceptance tests.

# A NaN computed on the GPU (0 / 0), as opposed to an uploaded NaN (F55: both
# are real NaN now). Each group mixes a real NA with one or two NaNs, in both row orders.
f29r4_nan_dfs <- function() {
  list(
    nan_first = tibble::tibble(
      g = c("a", "a", "a", "a", "a", "b", "b", "b"),
      a = c(NA, 0, 2, 0, 1, 0, NA, 3),
      b = c(1, 0, 1, 0, 1, 0, 1, 1)
    ),
    na_first = tibble::tibble(
      g = c("a", "a", "a", "a", "b", "b", "b"),
      a = c(0, NA, 2, 1, NA, 0, 1),
      b = c(0, 1, 1, 1, 1, 0, 1)
    )
  )
}

test_that("grouped slice_min()/slice_max() tie a GPU NaN with NA, in row order, natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    min_n3 = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_min(r, n = 3),
    min_prop = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_min(r, prop = 1),
    min_by = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::slice_min(r, n = 3, by = g),
    max_n3 = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_max(r, n = 3),
    max_n5 = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_max(r, n = 5),
    max_n1 = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_max(r, n = 1),
    min_no_ties = function(d) {
      d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_min(r, n = 5, with_ties = FALSE)
    },
    min_no_ties3 = function(d) {
      d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_min(r, n = 3, with_ties = FALSE)
    },
    max_no_ties_by = function(d) {
      d |> dplyr::mutate(r = a / b) |> dplyr::slice_max(r, n = 4, with_ties = FALSE, by = g)
    },
    min_na_rm = function(d) {
      d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_min(r, n = 5, na_rm = TRUE)
    },
    max_na_rm = function(d) {
      d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |> dplyr::slice_max(r, prop = 1, na_rm = TRUE)
    },
    max_na_rm_no_ties = function(d) {
      d |> dplyr::mutate(r = a / b) |> dplyr::group_by(g) |>
        dplyr::slice_max(r, n = 5, na_rm = TRUE, with_ties = FALSE)
    },
    # A computed order_by expression, materialized under a temp name.
    min_expr = function(d) d |> dplyr::group_by(g) |> dplyr::slice_min(a / b, n = 5),
    max_expr_by = function(d) d |> dplyr::slice_max(a / b, n = 5, by = g),
    # The ungrouped path (gpu_slice_rank()).
    ungrouped_min = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::slice_min(r, n = 8),
    ungrouped_max = function(d) d |> dplyr::mutate(r = a / b) |> dplyr::slice_max(r, n = 8)
  )
  for (df in f29r4_nan_dfs()) {
    for (nm in names(pipes)) {
      expect_same_as_dplyr(df, pipes[[nm]])
      expect_same_as_dplyr_lazy(df, pipes[[nm]])
    }
  }
})

test_that("grouped slice_min()/slice_max() on a non-float order column are unchanged", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    g = c("a", "a", "a", "b", "b", "b"),
    i = c(2L, NA, 1L, NA, 3L, 3L),
    s = c("b", NA, "a", "c", NA, "c")
  )
  pipes <- list(
    int_min = function(d) d |> dplyr::group_by(g) |> dplyr::slice_min(i, n = 3),
    int_max_narm = function(d) d |> dplyr::group_by(g) |> dplyr::slice_max(i, n = 3, na_rm = TRUE),
    str_max = function(d) d |> dplyr::group_by(g) |> dplyr::slice_max(s, n = 2)
  )
  for (nm in names(pipes)) {
    expect_same_as_dplyr(df, pipes[[nm]])
    expect_same_as_dplyr_lazy(df, pipes[[nm]])
  }
})

test_that("ntile() leaves a GPU NaN out of the bucket count (R3 N3)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    g = c("a", "a", "a", "b", "b", "b", "b"),
    a = c(1, 2, 0, 0, 3, NA, 1),
    b = c(1, 1, 0, 0, 1, 1, 1)
  )
  ungrouped <- function(d) d |> dplyr::mutate(x = a / b, k = dplyr::ntile(x, 2))
  grouped <- function(d) d |> dplyr::mutate(x = a / b) |> dplyr::group_by(g) |> dplyr::mutate(k = dplyr::ntile(x, 2))
  for (p in list(ungrouped, grouped)) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})
