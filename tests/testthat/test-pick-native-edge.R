# Edge cases of native pick() resolution in group_by()/arrange() and of the
# `.by_group` arrange node under the lazy optimizer (F10, round 3).

pne_df <- function() {
  tibble::tibble(
    g = c(1, NA, 2, 2),
    a = c(3L, 1L, 2L, 4L),
    b = c("x", "y", NA, "z")
  )
}

pne_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

pne_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  pne_both(df, pipeline, ...)
}

test_that("pick() symbols that aren't columns are resolved by dplyr", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  is_complete <- function(x) !anyNA(x)
  cols <- c("b", "a")
  # predicate symbol nested in a helper call
  pne_both(pne_df(), function(d) dplyr::arrange(d, pick(c(is_complete, g))))
  pne_both(pne_df(), function(d) dplyr::group_by(d, pick(-is_complete)))
  # external character vector (deprecated tidyselect usage)
  pne_both(pne_df(), function(d) dplyr::arrange(d, pick(cols)))
  pne_both(pne_df(), function(d) dplyr::group_by(d, pick(cols)))
})

test_that("pick() through a {{ }} wrapper checks the embedded expression", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  is_complete <- function(x) !anyNA(x)
  sort_by <- function(data, sel) dplyr::arrange(data, pick({{ sel }}))
  group_on <- function(data, sel) dplyr::group_by(data, pick({{ sel }}))
  pne_both(pne_df(), function(d) sort_by(d, is_complete))
  pne_both(pne_df(), function(d) group_on(d, is_complete))
  pne_native(pne_df(), function(d) sort_by(d, c(b, a)))
  pne_native(pne_df(), function(d) group_on(d, NULL))
})

test_that("constant pick() selections stay native", {
  skip_if_no_gpu()
  cols <- c("b", "a")
  pne_native(pne_df(), function(d) dplyr::arrange(d, pick(!!cols)))
  pne_native(pne_df(), function(d) dplyr::group_by(d, pick("a", 1)))
  pne_native(pne_df(), function(d) dplyr::arrange(d, pick(g:a)))
})

test_that("arrange(pick(NULL), .by_group = TRUE) sorts by the groups natively", {
  skip_if_no_gpu()
  pne_native(pne_df(), function(d) {
    dplyr::arrange(dplyr::group_by(d, b), pick(NULL), .by_group = TRUE)
  })
})

test_that("lazy .by_group arrange keeps the group column under a join and select", {
  skip_if_no_gpu()
  x <- tibble::tibble(g = c(2, 1, 2, 1), k = 1:4, v = c(10, 20, 30, 40))
  y <- tibble::tibble(k = 1:4, w = c("a", "b", "c", "d"))
  pipes <- list(
    # right side of the join
    function(x, y) {
      y |> dplyr::left_join(
        x |> dplyr::group_by(g) |> dplyr::arrange(.by_group = TRUE) |> dplyr::ungroup(),
        by = "k"
      ) |> dplyr::select(k, w, v)
    },
    # pick(NULL) node, inner join
    function(x, y) {
      x |> dplyr::group_by(g) |> dplyr::arrange(pick(NULL), .by_group = TRUE) |>
        dplyr::ungroup() |> dplyr::inner_join(y, by = "k") |> dplyr::select(w)
    }
  )
  for (f in pipes) {
    expected <- f(x, y)
    lazy <- f(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE))
    expect_equal(tibble::as_tibble(dplyr::collect(lazy)), expected)
    expect_equal(tibble::as_tibble(dplyr::collect(dplyr::compute(lazy))), expected)
  }
})
