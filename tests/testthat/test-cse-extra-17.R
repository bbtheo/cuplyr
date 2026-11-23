# #17: extra checks beside the acceptance file (qualified dplyr calls, windows
# that run on the GPU, a repeat that must not be shared across mutate() calls
# whose rows differ).

cse_n <- function(expr) {
  n0 <- gpu_cse_reuse_count()
  force(expr)
  gpu_cse_reuse_count() - n0
}

test_that("a repeated %in% under if_else() is computed once", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1.5, NA, 3, 4), y = c(1, 2, NA, 4), i = c(1L, 5L, NA, 9L)
  )
  st <- c(1L, 5L, 77L)
  p <- function(d) {
    dplyr::mutate(d,
      a = dplyr::if_else(i %in% st, x, y),
      b = dplyr::if_else(i %in% st, 1L, 0L)
    )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeat around a window aggregate that runs on the GPU is shared", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1.234, -2.5, 0.75, 0.31, 0.5, 0.06),
    y = c(3, 4, 5, 6, 7, 8),
    g = c("a", "b", "a", "b", "a", "b")
  )
  p <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(
        a = round(x - mean(y), 2) + 1,
        b = round(x - mean(y), 2) * 2
      )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_gte(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeat is only shared within one mutate() call's rows", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.123, 0.456, 0.789), k = c(1L, 2L, 3L))
  p <- function(d) {
    d |>
      dplyr::mutate(a = round(x, 2) + 1, b = round(x, 2) * 2) |>
      dplyr::filter(k > 1L) |>
      dplyr::mutate(c = round(x, 2) - 1, d = round(x, 2) / 2)
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  # each mutate() shares its own pair (the filter in between changes the rows,
  # so a memo that outlived the first call would hand back a 3-row column)
  expect_equal(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 2)
})

test_that("a shared checked chain's proven range does not hide a later overflow", {
  skip_if_no_gpu()
  # `q` is proven in range from the columns' ranges (its interval is filed under
  # its slot), but `r = q * 1000000L` leaves the int range for the larger rows:
  # the interval of `q`, not an assumption that it is small, must send `r` to
  # the exact path.
  df <- data.frame(
    a = c(1L, 2L, 3L, NA, 1000L, 7L),
    b = c(4L, NA, 6L, 7L, 900L, 1L),
    c = c(1L, 1L, NA, 2L, 5L, 3L)
  )
  p <- function(d) {
    dplyr::mutate(d, s = a + b, q = (a + b) * 2L - c, r = q * 1000000L)
  }
  expected <- suppressWarnings(p(df))
  expect_warning(p(df), "NAs produced by integer overflow")
  got <- NULL
  expect_warning(got <- collect(p(tbl_gpu(df, lazy = FALSE))), "NAs produced by integer overflow")
  expect_equal(tibble::as_tibble(got), tibble::as_tibble(expected))
  expect_equal(cse_n(suppressWarnings(collect(p(tbl_gpu(df, lazy = FALSE))))), 1)
})

test_that("gpu_cse_enable(FALSE) turns the memo off and gives the same answer", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.123, 0.456, NA, 0.789))
  p <- function(d) dplyr::mutate(d, a = round(x, 2) + 1, b = round(x, 2) * 2)
  on <- collect(p(tbl_gpu(df, lazy = FALSE)))
  prev <- gpu_cse_enable(FALSE)
  withr::defer(gpu_cse_enable(prev))
  n0 <- gpu_cse_reuse_count()
  off <- collect(p(tbl_gpu(df, lazy = FALSE)))
  expect_equal(gpu_cse_reuse_count() - n0, 0)
  expect_equal(on, off)
})
