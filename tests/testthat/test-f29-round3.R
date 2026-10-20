# F29 round 3: extra checks beside the protected acceptance tests.

# A NaN computed on the GPU (0 / 0), unlike an uploaded NaN, which becomes null.
f29r3_nan_df <- function() {
  tibble::tibble(
    g = c("a", "a", "a", "a", "b", "b", "b"),
    a = c(3, 0, 1, NA, 2, 0, 3),
    b = c(1, 0, 1, 1, 1, 0, 1)
  )
}

test_that("slice_min()/slice_max() rank a GPU-computed NaN as NA, natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f29r3_nan_df()
  pipes <- list(
    max2 = function(d) d |> dplyr::mutate(x = a / b) |> dplyr::slice_max(x, n = 2),
    max5 = function(d) d |> dplyr::mutate(x = a / b) |> dplyr::slice_max(x, n = 5),
    min4 = function(d) d |> dplyr::mutate(x = a / b) |> dplyr::slice_min(x, n = 4),
    min_na_rm = function(d) d |> dplyr::mutate(x = a / b) |> dplyr::slice_min(x, n = 6, na_rm = TRUE),
    max_no_ties = function(d) {
      d |> dplyr::mutate(x = a / b) |> dplyr::slice_max(x, n = 6, with_ties = FALSE)
    },
    # The grouped path ranks through gpu_window()'s min_rank(); n values that
    # reach the missing rows are pinned in test-f29-round4.R.
    grouped_max = function(d) {
      d |> dplyr::mutate(x = a / b) |> dplyr::group_by(g) |> dplyr::slice_max(x, n = 1)
    }
  )
  for (nm in names(pipes)) {
    expect_same_as_dplyr(df, pipes[[nm]])
    expect_same_as_dplyr_lazy(df, pipes[[nm]])
  }
})

test_that("a NaN rank key leaves the other window functions' NaN values alone", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f29r3_nan_df()
  p <- function(d) {
    d |>
      dplyr::mutate(x = a / b) |>
      dplyr::group_by(g) |>
      dplyr::mutate(
        k = dplyr::min_rank(x), cs = cumsum(x), l = dplyr::lag(x),
        r = dplyr::row_number(dplyr::desc(x))
      )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("slice_sample(replace = TRUE) on an empty group raises dplyr's error", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")), x = 1:3)
  p <- function(d) d |> dplyr::group_by(f, .drop = FALSE) |> dplyr::slice_sample(n = 1, replace = TRUE)
  set.seed(3)
  want <- tryCatch(p(df), error = identity)
  want_next <- stats::runif(1)
  expect_s3_class(want, "rlang_error")
  for (lazy in c(FALSE, TRUE)) {
    set.seed(3)
    got <- suppressWarnings(suppressMessages(
      tryCatch(collect(p(tbl_gpu(df, lazy = lazy))), error = identity)
    ))
    expect_identical(stats::runif(1), want_next)
    expect_s3_class(got, "rlang_error")
    expect_match(conditionMessage(got), "Can't compute indices", fixed = TRUE)
    expect_match(conditionMessage(got), "In group 3", fixed = TRUE)
  }
  # a size of 0 draws nothing, so it stays native
  withr::local_options(cuplyr.fallback = "error")
  p0 <- function(d) d |> dplyr::group_by(f, .drop = FALSE) |> dplyr::slice_sample(n = 0, replace = TRUE)
  expect_equal(nrow(collect(p0(tbl_gpu(df)))), 0L)
})

test_that("top_n() with n bound to a variable named n runs natively", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 3, 8, 1, 9), g = c("a", "b", "a", "b", "a"))
  n <- 2
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_same_as_dplyr(df, function(d) dplyr::top_n(d, n, x))
    expect_same_as_dplyr_lazy(df, function(d) dplyr::group_by(d, g) |> dplyr::top_n(-n, x))
  })
  # a column named n is what dplyr's data mask finds, not the variable
  dn <- tibble::tibble(x = c(5, 3, 8), n = c(1, 2, 3))
  want <- tryCatch(dplyr::top_n(dn, n, x), error = identity)
  got <- suppressWarnings(suppressMessages(
    tryCatch(collect(dplyr::top_n(tbl_gpu(dn), n, x)), error = identity)
  ))
  expect_equal(inherits(got, "error"), inherits(want, "error"))
  if (!inherits(want, "error")) expect_equal(tibble::as_tibble(got), want)
})

test_that("top_n()'s rewrite uses dplyr's min_rank()/desc(), not the caller's", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 3, 8, 1, 9))
  # A wt with no column is folded to a constant in R, in wt's environment.
  min_rank <- function(x) 99L
  desc <- function(x) x
  k <- 5
  want_col <- dplyr::top_n(df, 2, x)
  want_const <- dplyr::top_n(df, 2, k)
  withr::local_options(cuplyr.fallback = "error")
  expect_equal(tibble::as_tibble(collect(dplyr::top_n(tbl_gpu(df), 2, x))), want_col)
  expect_equal(tibble::as_tibble(collect(dplyr::top_n(tbl_gpu(df), 2, k))), want_const)
  expect_equal(tibble::as_tibble(collect(dplyr::top_n(tbl_gpu(df), -2, k))), dplyr::top_n(df, -2, k))
})
