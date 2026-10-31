# F50 acceptance tests: factor conversions on the GPU.
#
# as.character(f) decodes the codes to their labels (a STRING column), and
# as.integer(f) / as.numeric(f) are the codes themselves, as in base R. The
# results are not factors, so everything downstream that works on strings or
# numbers works on them natively.
#
# The oracle's tolerance ignores integer vs double and is lenient on some
# classes, so the result types are checked explicitly.

f50d_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1", "10", "2"),
               levels = c("10", "1", "2", "unused")),
    o = factor(c("lo", "mid", "hi", "mid", NA, "lo", "hi", "mid"),
               levels = c("lo", "mid", "hi"), ordered = TRUE),
    s = c("1", "x", "10", "2", NA, "1", "2", "2"),
    g = c(1L, 1L, 2L, 2L, 1L, 2L, 1L, 2L),
    x = c(1.5, 2, 3, 4, 5, NA, 7, 8)
  )
}

f50d_native <- function(df, pipeline, ...) {
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_same_as_dplyr(df, pipeline, ...)
    expect_same_as_dplyr_lazy(df, pipeline, ...)
  })
}

f50d_collect_native <- function(df, pipeline, lazy = FALSE) {
  withr::with_options(list(cuplyr.fallback = "error"), {
    g <- tbl_gpu(df, lazy = lazy)
    dplyr::collect(pipeline(g))
  })
}

# ---------------------------------------------------------------------------
# as.character()
# ---------------------------------------------------------------------------

test_that("F50: mutate(as.character(f)) decodes labels on the GPU", {
  skip_if_no_gpu()
  df <- f50d_df()
  f50d_native(df, function(d) dplyr::mutate(d, s2 = as.character(f), s3 = as.character(o)))
  for (lazy in c(FALSE, TRUE)) {
    res <- f50d_collect_native(df, function(d) dplyr::mutate(d, s2 = as.character(f)), lazy = lazy)
    expect_type(res$s2, "character")
    expect_false(is.factor(res$s2))
    expect_identical(res$s2, c("1", "2", "10", NA, "2", "1", "10", "2"))
    expect_true(is.factor(res$f))
  }
})

test_that("F50: as.character(f) overwriting the factor column gives a character column", {
  skip_if_no_gpu()
  df <- f50d_df()
  f50d_native(df, function(d) dplyr::mutate(d, f = as.character(f)))
  res <- f50d_collect_native(df, function(d) dplyr::mutate(d, f = as.character(f)), lazy = TRUE)
  expect_type(res$f, "character")
  # and a later verb sees a plain string column
  f50d_native(df, function(d) dplyr::filter(dplyr::mutate(d, f = as.character(f)), f == "1" | nchar(f) > 1))
})

test_that("F50: string operations on as.character(f) run on the GPU", {
  skip_if_no_gpu()
  df <- f50d_df()
  f50d_native(df, function(d) dplyr::mutate(d,
    u = toupper(as.character(o)),
    n = nchar(as.character(f)),
    p = paste0(as.character(f), "_x")
  ))
  f50d_native(df, function(d) dplyr::filter(d, as.character(f) == s))
  f50d_native(df, function(d) dplyr::filter(d, nchar(as.character(f)) > 1))
  f50d_native(df, function(d) dplyr::mutate(d, y = dplyr::if_else(as.character(f) == "10", "ten", "not")))
  f50d_native(df, function(d) dplyr::mutate(d, y = dplyr::coalesce(as.character(f), "none")))
})

test_that("F50: as.character(f) inside summarise() aggregates on the GPU", {
  skip_if_no_gpu()
  df <- f50d_df()
  f50d_native(df, function(d) dplyr::summarise(d, k = dplyr::n_distinct(as.character(f))))
  f50d_native(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g), k = dplyr::n_distinct(as.character(o), na.rm = TRUE))
  }, arrange_by = "g")
})

test_that("F50: across(where(is.factor), as.character) runs on the GPU", {
  skip_if_no_gpu()
  df <- f50d_df()
  f50d_native(df, function(d) dplyr::mutate(d, dplyr::across(dplyr::where(is.factor), as.character)))
  f50d_native(df, function(d) dplyr::mutate(d, dplyr::across(c(f, o), ~ as.character(.x), .names = "{.col}_chr")))
  res <- f50d_collect_native(df, function(d) dplyr::mutate(d, dplyr::across(dplyr::where(is.factor), as.character)),
                             lazy = TRUE)
  expect_type(res$f, "character")
  expect_type(res$o, "character")
})

test_that("F50: as.character(f) handles empty-string, spaced and non-ASCII labels, NA codes and 0 rows", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    f = factor(c("", "a b", "été", NA, "a b"), levels = c("a b", "", "été", "zz")),
    x = 1:5
  )
  f50d_native(df, function(d) dplyr::mutate(d, s = as.character(f)))
  res <- f50d_collect_native(df, function(d) dplyr::mutate(d, s = as.character(f)))
  expect_identical(res$s, c("", "a b", "été", NA, "a b"))
  f50d_native(df[0, ], function(d) dplyr::mutate(d, s = as.character(f)))
  res0 <- f50d_collect_native(df[0, ], function(d) dplyr::mutate(d, s = as.character(f)))
  expect_type(res0$s, "character")
  dfna <- tibble::tibble(f = factor(c(NA, NA), levels = c("a", "b")))
  f50d_native(dfna, function(d) dplyr::mutate(d, s = as.character(f)))
})

# ---------------------------------------------------------------------------
# as.integer() / as.numeric(): the codes
# ---------------------------------------------------------------------------

test_that("F50: as.integer(f) and as.numeric(f) are the codes, natively, with base R's types", {
  skip_if_no_gpu()
  df <- f50d_df()
  f50d_native(df, function(d) dplyr::mutate(d, i = as.integer(f), n = as.numeric(f), d2 = as.double(o)))
  for (lazy in c(FALSE, TRUE)) {
    res <- f50d_collect_native(df, function(d) {
      dplyr::mutate(d, i = as.integer(f), n = as.numeric(f), d2 = as.double(o))
    }, lazy = lazy)
    expect_type(res$i, "integer")
    expect_type(res$n, "double")
    expect_type(res$d2, "double")
    expect_identical(res$i, c(2L, 3L, 1L, NA, 3L, 2L, 1L, 3L))
    expect_identical(res$d2, c(1, 2, 3, 2, NA, 1, 3, 2))
  }
  f50d_native(df, function(d) dplyr::filter(d, as.integer(f) >= 2L))
  f50d_native(df, function(d) dplyr::summarise(dplyr::group_by(d, g), m = max(as.integer(o), na.rm = TRUE)),
              arrange_by = "g")
})
