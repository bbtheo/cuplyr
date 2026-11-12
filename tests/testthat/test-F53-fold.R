# F53 item 4 (coder tests): literal-only sub-expressions are folded in R.
# The acceptance file's `if_else(k == "", x, 0L)` is an error in dplyr itself
# (a size-1 condition against a size-n `true`), so the if_else shapes dplyr
# accepts are checked here, plus the rest of that test's pipelines.

f53f_df <- function() {
  tibble::tibble(g = c(1, 1, 2, 3, 3), x = c(1L, 2L, 3L, NA, 5L),
                 y = c(0.5, 1.5, -2, 4, NA), s = c("a", "", "b", "", "c"))
}

f53f_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("literal-only conditions inside if_else/case_when are folded", {
  skip_if_no_gpu()
  k <- ""
  d <- f53f_df()
  f53f_both(d, function(t) mutate(t, i = dplyr::if_else(x > 1L & k == "", x, 0L)))
  f53f_both(d, function(t) mutate(t, i = dplyr::if_else(x > 1L, x, if (k == "") 0L else 9L)))
  f53f_both(d, function(t) mutate(t, w = dplyr::case_when(k != "" ~ 1L, x > 2L ~ 2L, .default = x)))
  f53f_both(d, function(t) mutate(t, z = (k < "b") & (s == k)))
  f53f_both(d, function(t) summarise(t, n = sum(k == ""), m = sum(x[k == ""], na.rm = TRUE)))
  f53f_both(d, function(t) summarise(group_by(t, g), n = sum(x > 1L & k == "", na.rm = TRUE)))
})

test_that("folded literals keep R's types and typed NAs", {
  skip_if_no_gpu()
  k <- "abc"
  d <- f53f_df()
  f53f_both(d, function(t) mutate(t, a = 5L %/% 2L, b = 5L / 2L, c = 5L %% 2L,
                                  d = -3L, e = !TRUE, f = NA_character_ == k))
  f53f_both(d, function(t) mutate(t, a = x + (2L * 3L), b = y + (NA_real_ + 1)))
  f53f_both(d, function(t) filter(t, k == "abc", x > 1L))
  f53f_both(d, function(t) filter(t, k == "zzz" | x > 3L))
})

test_that("an integer overflow in a folded literal call is left to R's own warning", {
  skip_if_no_gpu()
  d <- f53f_df()
  big <- .Machine$integer.max
  expect_equal(
    suppressWarnings(collect(mutate(tbl_gpu(d), o = big + 1L)))$o,
    suppressWarnings(mutate(d, o = big + 1L)$o)
  )
})
