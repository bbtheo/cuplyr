# Acceptance tests for #70, item 1: an all-NA logical join key (what vctrs
# calls "unspecified") joins a character, Date, POSIXct or factor key, as in
# dplyr. The key takes the other side's type. A logical key with a non-NA value
# (or with no rows) still has no common type and errors like dplyr. dplyr also
# errors when such an x key is merged and the join returns no rows (it casts
# the empty logical slice), and that is reproduced too.

p70j_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

p70j_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

p70j_na <- tibble::tibble(k = NA, a = 1)
p70j_chr <- tibble::tibble(k = c("u", NA), b = 2:3)

# Runs `f(x, y)` on GPU tables in both modes and expects an error of `class`
# (eager raises at the verb or at collect, lazy at collect).
p70j_expect_error <- function(x, y, f, class, regexp = NULL) {
  for (lazy in c(FALSE, TRUE)) {
    run <- function() collect(f(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy)))
    if (is.null(regexp)) {
      expect_error(run(), class = class, info = paste("lazy =", lazy))
    } else {
      expect_error(run(), regexp = regexp, class = class, fixed = TRUE,
                   info = paste("lazy =", lazy))
    }
  }
}

test_that("the issue's example: an all-NA key left-joins a character key", {
  skip_if_no_gpu()
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, p70j_chr), by = "k"))
  out <- collect(left_join(tbl_gpu(p70j_na, lazy = FALSE), tbl_gpu(p70j_chr, lazy = FALSE), by = "k"))
  expect_identical(tibble::as_tibble(out), tibble::tibble(k = NA_character_, a = 1, b = 3L))
})

test_that("every mutating join type, with the all-NA key on either side", {
  skip_if_no_gpu()
  p70j_both(p70j_na, function(d) right_join(d, p70j_like(d, p70j_chr), by = "k"))
  p70j_both(p70j_na, function(d) full_join(d, p70j_like(d, p70j_chr), by = "k"))
  p70j_both(p70j_na, function(d) inner_join(d, p70j_like(d, p70j_chr), by = "k"))
  # the character key is x
  p70j_both(p70j_chr, function(d) left_join(d, p70j_like(d, p70j_na), by = "k"))
  p70j_both(p70j_chr, function(d) inner_join(d, p70j_like(d, p70j_na), by = "k"))
  p70j_both(p70j_chr, function(d) right_join(d, p70j_like(d, p70j_na), by = "k"))
  p70j_both(p70j_chr, function(d) full_join(d, p70j_like(d, p70j_na), by = "k"))
  # several all-NA rows
  p70j_both(tibble::tibble(k = c(NA, NA), a = 1:2),
            function(d) inner_join(d, p70j_like(d, p70j_chr), by = "k"))
})

test_that("filtering joins return x as it is", {
  skip_if_no_gpu()
  p70j_both(p70j_na, function(d) semi_join(d, p70j_like(d, p70j_chr), by = "k"))
  p70j_both(p70j_na, function(d) anti_join(d, p70j_like(d, p70j_chr), by = "k"))
  p70j_both(tibble::tibble(k = c(NA, NA), a = 1:2),
            function(d) anti_join(d, p70j_like(d, p70j_chr), by = "k"))
  p70j_both(p70j_chr, function(d) semi_join(d, p70j_like(d, p70j_na), by = "k"))
  p70j_both(p70j_chr, function(d) anti_join(d, p70j_like(d, p70j_na), by = "k"))
})

test_that("keep = TRUE keeps both keys' own types; na_matches = 'never' is honoured", {
  skip_if_no_gpu()
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, p70j_chr), by = "k", keep = TRUE))
  p70j_both(p70j_na, function(d) right_join(d, p70j_like(d, p70j_chr), by = "k", keep = TRUE))
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, p70j_chr), by = "k", na_matches = "never"))
  p70j_both(p70j_chr, function(d) left_join(d, p70j_like(d, p70j_na), by = "k", na_matches = "never"))
})

test_that("Date, POSIXct and factor keys", {
  skip_if_no_gpu()
  dates <- tibble::tibble(k = as.Date(c("2020-01-01", NA)), b = 1:2)
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, dates), by = "k"))
  p70j_both(dates, function(d) full_join(d, p70j_like(d, tibble::tibble(k = c(NA, NA), a = 1:2)), by = "k"))

  # the POSIXct side's zone is the key's zone
  times <- tibble::tibble(k = as.POSIXct(c("2020-01-01 10:00", NA), tz = "America/New_York"), b = 1:2)
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, times), by = "k"))
  p70j_both(times, function(d) inner_join(d, p70j_like(d, p70j_na), by = "k"))

  # the factor side's levels, in their own order, whichever side it is
  fct <- tibble::tibble(k = factor(c("u", NA), levels = c("v", "u")), b = 1:2)
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, fct), by = "k"))
  p70j_both(p70j_na, function(d) right_join(d, p70j_like(d, fct), by = "k"))
  p70j_both(fct, function(d) left_join(d, p70j_like(d, p70j_na), by = "k"))
  p70j_both(fct, function(d) full_join(d, p70j_like(d, p70j_na), by = "k"))
  out <- collect(right_join(tbl_gpu(p70j_na), tbl_gpu(fct), by = "k"))
  expect_identical(levels(out$k), c("v", "u"))
})

test_that("keys with different names, several keys, groups", {
  skip_if_no_gpu()
  p70j_both(tibble::tibble(j = NA, a = 1),
            function(d) left_join(d, p70j_like(d, p70j_chr), by = c(j = "k")))
  p70j_both(tibble::tibble(j = NA, a = 1),
            function(d) left_join(d, p70j_like(d, p70j_chr), by = dplyr::join_by(j == k), keep = TRUE))
  p70j_both(tibble::tibble(k = NA, j = 1L, a = 1),
            function(d) left_join(d, p70j_like(d, tibble::tibble(k = c("u", NA), j = 1L, b = 2:3)),
                                  by = c("k", "j")))
  p70j_both(p70j_na, function(d) left_join(group_by(d, a), p70j_like(d, p70j_chr), by = "k"))
  p70j_both(p70j_na, function(d) left_join(group_by(d, k), p70j_like(d, p70j_chr), by = "k"))
})

test_that("an all-NA key made by earlier verbs", {
  skip_if_no_gpu()
  p70j_both(tibble::tibble(a = 1:2),
            function(d) left_join(mutate(d, k = NA), p70j_like(d, p70j_chr), by = "k"))
  # the filter runs first, as written: the remaining key is all NA
  p70j_both(tibble::tibble(k = c(NA, TRUE), a = 1:2),
            function(d) left_join(filter(d, a == 1), p70j_like(d, p70j_chr), by = "k"))
})

test_that("vctrs::unspecified() keys join a character key", {
  skip_if_no_gpu()
  p70j_both(tibble::tibble(k = vctrs::unspecified(2), a = 1:2),
            function(d) left_join(d, p70j_like(d, p70j_chr), by = "k"))
  p70j_both(tibble::tibble(k = vctrs::unspecified()),
            function(d) left_join(d, p70j_like(d, p70j_chr), by = "k"))
  # no rows returned: a vctrs::unspecified() key still casts (dplyr has no error here)
  p70j_both(tibble::tibble(k = vctrs::unspecified(1), a = 1),
            function(d) inner_join(d, p70j_like(d, tibble::tibble(k = "u", b = 2L)), by = "k"))
})

test_that("inequality and rolling conditions keep both keys", {
  skip_if_no_gpu()
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, p70j_chr), by = dplyr::join_by(k >= k)))
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, p70j_chr), by = dplyr::join_by(closest(k >= k))))
  p70j_both(p70j_chr, function(d) left_join(d, p70j_like(d, p70j_na), by = dplyr::join_by(k <= k)))
  # no rows, no merged key: no error
  p70j_both(p70j_na, function(d) inner_join(d, p70j_like(d, tibble::tibble(k = "u", b = 2L)),
                                            by = dplyr::join_by(k >= k)))
})

test_that("nest_join() accepts the all-NA key", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    out <- nest_join(tbl_gpu(p70j_na, lazy = lazy), tbl_gpu(p70j_chr, lazy = lazy), by = "k",
                     name = "y")
    expected <- nest_join(p70j_na, p70j_chr, by = "k", name = "y")
    out <- tibble::as_tibble(collect(out))
    expect_identical(out$k, expected$k)
    expect_identical(out$a, expected$a)
    expect_equal(tibble::as_tibble(collect(out$y[[1]])), expected$y[[1]])
  }
})

test_that("a logical key with values, or with no rows, still has no common type", {
  skip_if_no_gpu()
  lj <- function(x, y) left_join(x, y, by = "k")
  p70j_expect_error(tibble::tibble(k = c(NA, TRUE)), p70j_chr, lj,
                    class = "dplyr_error_join_incompatible_type",
                    regexp = "Can't join `x$k` with `y$k` due to incompatible types")
  p70j_expect_error(p70j_chr, tibble::tibble(k = c(NA, FALSE)), lj,
                    class = "dplyr_error_join_incompatible_type")
  p70j_expect_error(tibble::tibble(k = logical()), p70j_chr, lj,
                    class = "dplyr_error_join_incompatible_type")
  p70j_expect_error(p70j_chr, tibble::tibble(k = logical(), z = integer()), lj,
                    class = "dplyr_error_join_incompatible_type")
  p70j_expect_error(tibble::tibble(k = c(NA, TRUE)), p70j_chr,
                    function(x, y) semi_join(x, y, by = "k"),
                    class = "dplyr_error_join_incompatible_type")
  # the message names the types, as dplyr's does
  expect_error(
    left_join(tbl_gpu(tibble::tibble(k = c(NA, TRUE)), lazy = FALSE),
              tbl_gpu(p70j_chr, lazy = FALSE), by = "k"),
    "`y$k` is a <character>", fixed = TRUE
  )
})

test_that("an eager join raises at the verb call", {
  skip_if_no_gpu()
  expect_error(
    left_join(tbl_gpu(tibble::tibble(k = c(NA, TRUE)), lazy = FALSE),
              tbl_gpu(p70j_chr, lazy = FALSE), by = "k"),
    class = "dplyr_error_join_incompatible_type"
  )
})

test_that("a filter after the join does not change whether the join works", {
  skip_if_no_gpu()
  # dplyr joins first: the TRUE key errors even though the filter drops its row
  bad <- tibble::tibble(k = c(NA, TRUE), a = 1:2)
  p70j_expect_error(bad, p70j_chr, function(x, y) filter(left_join(x, y, by = "k"), a == 1),
                    class = "dplyr_error_join_incompatible_type")
  # and an all-NA key joins even when the filter keeps nothing
  p70j_both(tibble::tibble(k = c(NA, NA), a = 1:2),
            function(d) filter(left_join(d, p70j_like(d, p70j_chr), by = "k"), a > 5))
  p70j_both(p70j_na,
            function(d) filter(inner_join(d, p70j_like(d, p70j_chr), by = "k"), b > 10))
  p70j_both(p70j_na,
            function(d) filter(right_join(d, p70j_like(d, p70j_chr), by = "k"), b > 10))
})

test_that("a merged all-NA x key errors like dplyr when the join returns no rows", {
  skip_if_no_gpu()
  no_match <- tibble::tibble(k = "u", b = 2L)
  msg <- "Can't convert `out[merge]$k` <logical> to match type of `k` <character>."
  p70j_expect_error(p70j_na, no_match, function(x, y) inner_join(x, y, by = "k"),
                    class = "vctrs_error_cast")
  expect_error(
    inner_join(tbl_gpu(p70j_na, lazy = FALSE), tbl_gpu(no_match, lazy = FALSE), by = "k") |> collect(),
    msg, fixed = TRUE
  )
  p70j_expect_error(p70j_na, p70j_chr,
                    function(x, y) inner_join(x, y, by = "k", na_matches = "never"),
                    class = "vctrs_error_cast")
  p70j_expect_error(p70j_na, tibble::tibble(k = character(), b = integer()),
                    function(x, y) right_join(x, y, by = "k"),
                    class = "vctrs_error_cast")
  p70j_expect_error(tibble::tibble(j = NA, a = 1), no_match,
                    function(x, y) inner_join(x, y, by = c(j = "k")),
                    class = "vctrs_error_cast", regexp = "`out[merge]$j`")
  p70j_expect_error(p70j_na, tibble::tibble(k = as.Date("2020-01-01"), b = 2L),
                    function(x, y) inner_join(x, y, by = "k"),
                    class = "vctrs_error_cast", regexp = "<date>")
  p70j_expect_error(tibble::tibble(k = NA, j = 1L, a = 1), tibble::tibble(k = "u", j = 1L, b = 2L),
                    function(x, y) inner_join(x, y, by = dplyr::join_by(k, j >= j)),
                    class = "vctrs_error_cast")

  # no error where dplyr has none: keep = TRUE, the all-NA key on the y side,
  # left/full joins (every x row stays)
  p70j_both(p70j_na, function(d) inner_join(d, p70j_like(d, no_match), by = "k", keep = TRUE))
  p70j_both(no_match, function(d) inner_join(d, p70j_like(d, p70j_na), by = "k"))
  p70j_both(p70j_na, function(d) left_join(d, p70j_like(d, tibble::tibble(k = character(), b = integer())), by = "k"))
  p70j_both(p70j_na, function(d) full_join(d, p70j_like(d, tibble::tibble(k = character(), b = integer())), by = "k"))
})
