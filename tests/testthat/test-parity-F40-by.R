# F40 acceptance tests: `.by = where(...)` / `by = where(...)`.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles): `.by`
# is a tidyselect over the data, so where(fn) tests each column's values:
# - type tests (is.character, is.factor, ...) group by the matching columns,
#   resolved natively (no CPU fallback);
# - value predicates (anyNA, n_distinct) group by what the data says;
# - mixed selections (c(g, where(is.factor))) combine both.
# Every verb with `.by`/`by` takes it: mutate, summarise, filter, filter_out,
# reframe, slice and the slice_*() family.

f40b_df <- function() {
  data.frame(
    g = c(1L, 1L, 2L, 2L, 2L),
    x = c(1, NA, 3, 4, 5),
    s = c("a", "a", "b", "b", "a"),
    f = factor(c("u", "v", "u", "u", "v")),
    stringsAsFactors = FALSE
  )
}

f40b_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f40b_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f40b_both(df, pipeline, ...)
}

test_that(".by = where(<type test>) groups by the matching columns natively", {
  skip_if_no_gpu()
  df <- f40b_df()
  f40b_native(df, function(d) summarise(d, n = dplyr::n(), .by = where(is.character)),
              arrange_by = "s")
  f40b_native(df, function(d) mutate(d, n = dplyr::n(), .by = where(is.factor)))
  f40b_native(df, function(d) filter(d, x == max(x), .by = where(is.character)))
  f40b_native(df, function(d) filter_out(d, x > 2, .by = where(is.character)))
  f40b_native(df, function(d) slice_head(d, n = 1, by = c(g, where(is.factor))))
  f40b_native(df, function(d) slice(d, 1, .by = where(is.integer)))
  f40b_native(df, function(d) slice_max(d, x, n = 1, by = where(is.character)),
              arrange_by = c("s", "x"))
  f40b_native(df, function(d) mutate(d, m = sum(g),
                                     .by = c(where(is.character), where(is.factor))))
})

test_that("reframe(.by = where()) groups by the matching columns", {
  skip_if_no_gpu()
  df <- f40b_df()
  f40b_both(df, function(d) reframe(d, r = range(x, na.rm = TRUE), .by = where(is.integer)),
            arrange_by = c("g", "r"))
})

test_that(".by = where(<value predicate>) groups by what the data says", {
  skip_if_no_gpu()
  df <- f40b_df()
  f40b_both(df, function(d) mutate(d, n = dplyr::n(),
                                   .by = where(~ is.integer(.x) && dplyr::n_distinct(.x) == 2)))
  f40b_both(df, function(d) summarise(d, n = dplyr::n(),
                                      .by = where(~ !is.numeric(.x) && dplyr::n_distinct(.x) == 2)),
            arrange_by = c("s", "f"))
  f40b_both(df, function(d) filter(d, dplyr::n() > 2, .by = where(~ !anyNA(.x) && is.character(.x))))
  f40b_both(df, function(d) slice_tail(d, n = 1, by = where(~ dplyr::n_distinct(.x) == 2)))
  f40b_both(df, function(d) d |> filter(!is.na(x)) |>
              summarise(n = dplyr::n(), .by = where(~ !anyNA(.x) && !is.double(.x))),
            arrange_by = c("g", "s", "f"))
})

test_that(".by = where() on zero-row input", {
  skip_if_no_gpu()
  df <- f40b_df()[0, ]
  f40b_both(df, function(d) summarise(d, n = dplyr::n(), .by = where(is.character)))
  f40b_both(df, function(d) summarise(d, n = dplyr::n(), .by = where(~ !anyNA(.x))))
})

test_that(".by = where() keeps dplyr's errors", {
  skip_if_no_gpu()
  df <- f40b_df()
  g <- tbl_gpu(df)
  # grouped .data and .by together
  expect_error(mutate(group_by(g, g), n = dplyr::n(), .by = where(is.character)),
               "Can't supply `.by` when `.data` is a grouped data frame.", fixed = TRUE)
  # a predicate that doesn't return TRUE/FALSE
  expect_error(collect(mutate(g, n = dplyr::n(), .by = where(~ 1))), class = "rlang_error")
})
