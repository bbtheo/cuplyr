# F21 tier A: group_split() / group_map() / group_modify() / group_walk()
# contract against real dplyr. Chunks are compared after collect(), so these
# tests hold whether a chunk is a tbl_gpu or a tibble (that choice is pinned
# in test-parity-F21-native.R).

f21_df <- function() {
  tibble::tibble(
    x = 1:8,
    g = factor(c("b", "a", NA, "b", "a", "a", NA, "b"), levels = c("a", "b", "c")),
    s = c("B", "a", "a", NA, "B", "a", NA, "a"),
    h = c(2, 1, 2, 1, 2, 2, 1, 1),
    d = as.Date("2024-01-01") + 0:7
  )
}

# A chunk / ptype as dplyr would hand it out: collected if it is a tbl_gpu.
f21_frame <- function(x) {
  if (is_tbl_gpu(x)) dplyr::collect(x) else x
}

f21_expect_split <- function(res, oracle) {
  expect_s3_class(res, "vctrs_list_of")
  expect_identical(length(res), length(oracle))
  for (i in seq_along(oracle)) {
    expect_identical(f21_frame(res[[i]]), oracle[[i]], info = paste("chunk", i))
  }
  expect_identical(f21_frame(attr(res, "ptype")), attr(oracle, "ptype"))
}

f21_check <- function(df, fn) {
  oracle <- fn(tibble::as_tibble(df))
  f21_expect_split(fn(tbl_gpu(df)), oracle)
  f21_expect_split(fn(as_lazy(tbl_gpu(df))), oracle)
}

test_that("group_split() by a factor keeps the grouping column, sorted, NA last", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(d, g))
})

test_that("group_split() by a string key uses C-locale order with NA last", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(d, s))
})

test_that("group_split() by several keys and by a computed key", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(d, h, g))
  f21_check(f21_df(), function(d) dplyr::group_split(d, y = h * 2))
})

test_that("group_split(.keep = FALSE) drops the grouping columns from chunks and ptype", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(d, g, .keep = FALSE))
  f21_check(f21_df(), function(d) dplyr::group_split(d, h, s, .keep = FALSE))
})

test_that("group_split() on a grouped table splits on its groups", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(dplyr::group_by(d, g)))
  f21_check(f21_df(), function(d) dplyr::group_split(dplyr::group_by(d, s, h), .keep = FALSE))
})

test_that("group_split() respects .drop: empty factor levels give 0-row chunks only with .drop = FALSE", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(d, g, .drop = FALSE))
  f21_check(f21_df(), function(d) dplyr::group_split(d, g, .drop = TRUE))
  f21_check(f21_df(), function(d) dplyr::group_split(dplyr::group_by(d, g, .drop = FALSE)))
  f21_check(f21_df(), function(d) dplyr::group_split(dplyr::group_by(d, g, .drop = FALSE), .keep = FALSE))

  one <- tibble::tibble(f = factor("b", levels = c("a", "b", "c")))
  expect_length(dplyr::group_split(tbl_gpu(one), f, .drop = TRUE), 1L)
  expect_length(dplyr::group_split(tbl_gpu(one), f, .drop = FALSE), 3L)
})

test_that("group_split() keeps groups emptied by filter(.preserve = TRUE)", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) {
    dplyr::group_split(dplyr::filter(dplyr::group_by(d, h), x > 6, .preserve = TRUE))
  })
})

test_that("group_split() chunks line up with group_keys()", {
  skip_if_no_gpu()
  gt <- dplyr::group_by(tbl_gpu(f21_df()), g, .drop = FALSE)
  keys <- dplyr::group_keys(gt)
  res <- dplyr::group_split(gt)
  expect_identical(length(res), nrow(keys))
  for (i in seq_along(res)) {
    chunk <- f21_frame(res[[i]])
    if (nrow(chunk) > 0L) {
      expect_identical(unique(chunk$g), keys$g[i])
    }
  }
})

test_that("group_split() with no groups gives one chunk; 0-row inputs", {
  skip_if_no_gpu()
  f21_check(f21_df(), function(d) dplyr::group_split(d))
  f21_check(f21_df()[0, ], function(d) dplyr::group_split(d))
  f21_check(f21_df()[0, ], function(d) dplyr::group_split(d, g))
  f21_check(f21_df()[0, ], function(d) dplyr::group_split(dplyr::group_by(d, s)))
})

test_that("group_split() chunks follow dplyr's class and attribute rules", {
  skip_if_no_gpu()
  df <- as.data.frame(f21_df())
  attr(df, "foo") <- "bar"
  oracle_none <- dplyr::group_split(df)
  oracle_g <- dplyr::group_split(df, g)
  # a data.frame input gives tibble chunks; attributes survive only without vars
  f21_expect_split(dplyr::group_split(tbl_gpu(df)), oracle_none)
  f21_expect_split(dplyr::group_split(tbl_gpu(df), g), oracle_g)
})

test_that("group_split() on a grouped table with ... warns like dplyr and ignores them", {
  skip_if_no_gpu()
  gt <- dplyr::group_by(tbl_gpu(f21_df()), g)
  expect_warning(res <- dplyr::group_split(gt, h), class = "rlang_warning",
                 regexp = "ignores `...`")
  f21_expect_split(res, dplyr::group_split(dplyr::group_by(f21_df(), g)))
})

test_that("group_split(keep = ) is defunct like dplyr", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f21_df())
  expect_error(dplyr::group_split(gt, keep = TRUE), class = "lifecycle_error_deprecated")
  expect_error(dplyr::group_split(dplyr::group_by(gt, g), keep = TRUE),
               class = "lifecycle_error_deprecated")
})

test_that("testthat comparisons of tbl_gpu compare data, not GPU pointers", {
  skip_if_no_gpu()
  df <- f21_df()
  expect_equal(tbl_gpu(df), tbl_gpu(df))
  expect_identical(vctrs::vec_ptype(tbl_gpu(df)), vctrs::vec_ptype(tbl_gpu(df)))
  expect_failure(expect_equal(tbl_gpu(df), tbl_gpu(df[-1, ])))
  expect_failure(expect_equal(dplyr::group_by(tbl_gpu(df), g), tbl_gpu(df)))
  # the shape of the vendored conformance check: a list_of whose ptype is a tbl_gpu
  gt <- tbl_gpu(df[1:4, c("x", "h")])
  expect_identical(
    dplyr::group_split(gt, h),
    vctrs::list_of(dplyr::filter(gt, h == 1), dplyr::filter(gt, h == 2))
  )
})

test_that("group_map()/group_modify()/group_walk() match dplyr, empty groups included", {
  skip_if_no_gpu()
  df <- f21_df()
  for (drop in c(TRUE, FALSE)) {
    gt <- dplyr::group_by(tbl_gpu(df), g, .drop = drop)
    gd <- dplyr::group_by(df, g, .drop = drop)

    fm <- function(.x, .y) list(n = nrow(.x), sum = sum(.x$x), key = .y$g, tibble = inherits(.x, "tbl_df"))
    expect_identical(dplyr::group_map(gt, fm), dplyr::group_map(gd, fm))
    expect_identical(dplyr::group_map(gt, fm, .keep = TRUE), dplyr::group_map(gd, fm, .keep = TRUE))

    fmod <- function(.x, .y) tibble::tibble(n = nrow(.x), s = sum(.x$x))
    got <- dplyr::group_modify(gt, fmod)
    expect_identical(dplyr::group_vars(got), "g")
    expect_identical(tibble::as_tibble(dplyr::collect(got)),
                     tibble::as_tibble(dplyr::group_modify(gd, fmod)))

    seen <- integer()
    out <- withVisible(dplyr::group_walk(gt, function(.x, .y) seen <<- c(seen, nrow(.x))))
    expect_false(out$visible)
    expect_true(is_tbl_gpu(out$value))
    expect_identical(seen, vapply(dplyr::group_split(gd), nrow, integer(1)))
  }
})
