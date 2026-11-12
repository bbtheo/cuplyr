# F53 item 2 acceptance tests: an explicit NA factor level (addNA()) survives
# upload, verbs and collect.
#
# dplyr: `addNA(factor(c("a", NA, "b")))` has levels a, b, NA and codes 1 3 2.
# The NA-level element is NOT missing (`is.na()` is FALSE), `f == NA_character_`
# is NA, `f %in% NA` matches it, `as.character()` gives NA_character_, and
# count()/group_by()/.drop = FALSE list <NA> as the third level.

f53_na_df <- function() {
  tibble::tibble(
    f = addNA(factor(c("a", NA, "b", NA, "a"))),
    x = c(1L, 2L, 3L, 4L, 5L)
  )
}

f53n_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("addNA() levels and codes round-trip exactly", {
  skip_if_no_gpu()
  d <- f53_na_df()
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(tbl_gpu(d, lazy = lazy))
    expect_identical(levels(got$f), levels(d$f))
    expect_identical(as.integer(got$f), as.integer(d$f))
    expect_identical(got$f, d$f)
  }
  # an NA level that is not last, an ordered factor, and a real NA next to it
  # codes: "b", the NA level, a real missing value
  f2 <- structure(c(1L, 2L, NA, 3L), levels = c("b", NA, "a"), class = "factor")
  o <- addNA(factor(c("lo", "hi", NA, "lo"), levels = c("lo", "hi"), ordered = TRUE))
  d2 <- tibble::tibble(f2 = f2, o = o)
  got2 <- collect(tbl_gpu(d2))
  expect_identical(got2$f2, d2$f2)
  expect_identical(got2$o, d2$o)
})

test_that("count(), group_by() and .drop = FALSE keep the NA level like dplyr", {
  skip_if_no_gpu()
  d <- f53_na_df()
  f53n_both(d, function(t) count(t, f))
  f53n_both(d, function(t) summarise(group_by(t, f), n = n(), s = sum(x)))
  f53n_both(d, function(t) count(filter(t, x > 1), f, .drop = FALSE))
  f53n_both(d, function(t) summarise(group_by(filter(t, x != 2L & x != 4L), f, .drop = FALSE),
                                     n = n()))
  f53n_both(d, function(t) distinct(t, f))
  f53n_both(d, function(t) arrange(t, f, x))
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(count(tbl_gpu(d, lazy = lazy), f))
    expect_identical(levels(got$f), c("a", "b", NA))
  }
})

test_that("NA-level comparisons and conversions match dplyr", {
  skip_if_no_gpu()
  d <- f53_na_df()
  f53n_both(d, function(t) filter(t, is.na(f)))
  f53n_both(d, function(t) filter(t, !is.na(f)))
  f53n_both(d, function(t) filter(t, f == NA_character_))
  f53n_both(d, function(t) filter(t, f %in% NA))
  f53n_both(d, function(t) filter(t, f %in% c("a", NA)))
  f53n_both(d, function(t) filter(t, f == "a"))
  f53n_both(d, function(t) mutate(t, s = as.character(f), m = is.na(f), i = f %in% NA))
})

test_that("verbs that keep the factor column keep its NA level", {
  skip_if_no_gpu()
  d <- f53_na_df()
  f53n_both(d, function(t) select(filter(t, x > 2L), f))
  f53n_both(d, function(t) mutate(t, y = x * 2L))
  f53n_both(d, function(t) slice(t, 2:3))
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(filter(tbl_gpu(d, lazy = lazy), x == 1L))
    expect_identical(levels(got$f), c("a", "b", NA))
  }
})
