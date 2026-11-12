# F49 round 4: the optimizer must not change which rows a rowwise aggregate
# that raises flags sees (review B6, N15), and a lazy zero-row min()/max()
# raises its flag (review B7).

# Node types from the root down the `$input` chain.
f49r4_chain <- function(ast) {
  out <- character()
  while (!is.null(ast)) {
    out <- c(out, ast$type)
    ast <- ast$input
  }
  out
}

test_that("an optimized lazy plan keeps a filter above a mutate with an integer row aggregate", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, 7L, 3L), b = c(2L, 1L, 4L), d = c(1, 2, 3))
  rw <- dplyr::rowwise(tbl_gpu(df, lazy = TRUE))
  for (agg in c("sum(c(a, b))", "max(c(a, b))", "min(a, b)", "max(c(a, d), na.rm = TRUE)")) {
    q <- dplyr::filter(dplyr::mutate(rw, s = !!rlang::parse_expr(agg)), a < 5L)
    chain <- f49r4_chain(optimize_ast(q$lazy_ops))
    expect_identical(chain[[1L]], "filter", info = agg)
    expect_false("filter" %in% chain[-1L], info = agg)
  }
  # a non-flagging aggregate and a plain mutate still let the filter through
  # (`d + 1` is a double: an integer `a + b` is F42-checked and F70 keeps the
  # filter above it)
  for (e in c("mean(c(a, b))", "d + 1")) {
    q <- dplyr::filter(dplyr::mutate(rw, s = !!rlang::parse_expr(e)), a < 5L)
    chain <- f49r4_chain(optimize_ast(q$lazy_ops))
    expect_false(identical(chain[[1L]], "filter"), info = e)
  }
})

test_that("a lazy row-sum filter after another filter sees only the surviving rows", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, .Machine$integer.max, 3L), b = c(2L, 1L, 4L))
  p <- function(d) d |> dplyr::rowwise() |> dplyr::filter(a < 5L) |> dplyr::filter(sum(c(a, b)) > 0L)
  want <- p(df)
  expect_identical(collect(p(tbl_gpu(df, lazy = TRUE))), want)
  expect_identical(collect(p(tbl_gpu(df))), want)
})

test_that("a lazy double min(na.rm = TRUE) warns for an all-NA row a later filter removes", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, NA, 3), b = c(2, NA, 4))
  p <- function(d) d |> dplyr::rowwise() |> dplyr::mutate(m = min(c(a, b), na.rm = TRUE)) |>
    dplyr::filter(!is.na(a))
  want <- suppressWarnings(p(df))
  expect_warning(got <- collect(p(tbl_gpu(df, lazy = TRUE))), "no non-missing arguments to min")
  expect_identical(got, want)
})

test_that("a lazy rowwise min()/max() filter on a zero-row table warns like dplyr", {
  skip_if_no_gpu()
  df0 <- tibble::tibble(a = integer(), b = integer())
  p <- function(d) d |> dplyr::rowwise() |> dplyr::filter(max(c(a, b)) > 0L)
  want <- suppressWarnings(p(df0))
  withr::local_options(cuplyr.fallback = "error")
  expect_warning(got <- collect(p(tbl_gpu(df0, lazy = TRUE))), "no non-missing arguments to max")
  expect_identical(got, want)
})

test_that("a lazy zero-row integer min()/max() fails loudly, natively", {
  skip_if_no_gpu()
  df0 <- tibble::tibble(a = integer(), b = integer())
  withr::local_options(cuplyr.fallback = "error")
  q <- dplyr::mutate(dplyr::rowwise(tbl_gpu(df0, lazy = TRUE)), m = max(c(a, b)))
  expect_error(collect(q), class = "cuplyr_error_rowwise_retype")
  # an integer row aggregate whose later filter empties the table is fine
  df <- tibble::tibble(a = 1:3, b = 4:6)
  q <- dplyr::rowwise(tbl_gpu(df, lazy = TRUE)) |> dplyr::mutate(m = max(c(a, b))) |>
    dplyr::filter(b > 100L)
  expect_identical(collect(q), dplyr::rowwise(df) |> dplyr::mutate(m = max(c(a, b))) |>
                     dplyr::filter(b > 100L))
})
