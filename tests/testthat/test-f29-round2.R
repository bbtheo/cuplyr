# F29 round 2: extra checks beside the protected acceptance tests.

test_that("string desc() ranks: native under a C collation, CPU (and right) otherwise", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("a", "B", "c", "b"), x = c(1, 2, 3, 4))
  p <- function(d) dplyr::mutate(d, r = dplyr::min_rank(dplyr::desc(s)))

  withr::with_collate("C", {
    withr::local_options(cuplyr.fallback = "error")
    expect_same_as_dplyr(df, p)
  })

  withr::local_collate("en_US.UTF-8")
  withr::local_options(cuplyr.fallback = "error")
  # a bare min_rank(s) is vec_rank(), C-ordered, so it stays native
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, r = dplyr::min_rank(s)))
  # a numeric desc() is unaffected by the collation
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, r = dplyr::min_rank(dplyr::desc(x))))
  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(expect_same_as_dplyr(df, p), "fell back")
})

test_that("slice_sample() keeps the empty groups of a .drop = FALSE table like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 2), x = c(1, 2, 3, 4, 5))
  p <- function(d) {
    d |>
      dplyr::group_by(g, .drop = FALSE) |>
      dplyr::filter(x > 2) |>
      dplyr::slice_sample(n = 1)
  }
  for (lazy in c(FALSE, TRUE)) {
    set.seed(2)
    want <- p(df)
    want_next <- stats::runif(1)
    set.seed(2)
    got <- collect(p(tbl_gpu(df, lazy = lazy)))
    expect_identical(stats::runif(1), want_next)
    expect_equal(tibble::as_tibble(dplyr::group_data(got)), tibble::as_tibble(dplyr::group_data(want)))
    expect_equal(dplyr::ungroup(tibble::as_tibble(got)), dplyr::ungroup(want))
  }
})

test_that("top_n() with a deterministic n is evaluated no more than once", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 3, 8, 1, 9), f = factor(c("a", "b", "c", "d", "e")))
  calls <- 0L
  n_fn <- function() {
    calls <<- calls + 1L
    2L
  }
  dplyr::top_n(df, n_fn(), f)
  want <- calls
  calls <- 0L
  suppressMessages(collect(dplyr::top_n(tbl_gpu(df), n_fn(), f)))
  expect_lte(calls, want)
})
