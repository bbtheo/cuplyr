# F09 round 3: CPU fallbacks re-run the ORIGINAL dots; an unnamed dot that may
# return a data frame makes the columns in scope unknown; `.unpack` other than
# a literal FALSE goes to dplyr.

r3_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("an earlier unnamed data-frame dot: transmute, list of values, if_any()", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  df <- tibble::tibble(x = c(1, 2), y = c(3, 4))
  w <- 5
  f <- ~ .x + w
  suppressWarnings({
    r3_both(df, function(d) dplyr::transmute(d, tibble::tibble(w = x * 0 + 1), dplyr::across(x, f)))
    r3_both(df, function(d) {
      dplyr::mutate(d, tibble::tibble(w = y), dplyr::across(c(x, y), list(a = f, b = ~ .x * 2)))
    })
    p <- function(v) v > w
    r3_both(df, function(d) dplyr::mutate(d, tibble::tibble(w = x), keep = dplyr::if_any(c(x, y), p)))
  })
})

test_that("a bare in-scope column dot keeps the guards precise", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(x = c(1, 2), y = c(3, 4))
  w <- 5
  f <- ~ .x + w
  r3_both(df, function(d) dplyr::transmute(d, y, dplyr::across(x, f)))
})

test_that("expression fallback re-runs the original dots without leaking helpers", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(4, 5, 6))
  w <- 5
  f <- ~ .x + w
  r3_both(df, function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(x, f), id = dplyr::cur_group_id(),
                  s = identity(x))
  })
  # staged across() plus an unparseable later dot
  r3_both(df, function(d) {
    dplyr::mutate(d, dplyr::across(c(x = y, y = x), ~ .x * 2), s = identity(x))
  })
  r3_both(df, function(d) dplyr::transmute(d, dplyr::across(c(x, y), f), s = identity(x)))
})

test_that("`.unpack` other than a literal FALSE goes to dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2))
  expect_error(
    dplyr::collect(dplyr::mutate(tbl_gpu(df), dplyr::across(x, ~ .x + 1, .unpack = 1))),
    "`.unpack` must be"
  )
  r3_both(df, function(d) dplyr::mutate(d, dplyr::across(x, ~ .x + 1, .unpack = "{outer}_{inner}")))
  withr::local_options(cuplyr.fallback = "error")
  r3_both(df, function(d) dplyr::mutate(d, dplyr::across(x, ~ .x + 1, .unpack = FALSE)))
})
