# F40 acceptance tests: across()/if_any()/if_all() column selection.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - `.cols = where(fn)` applies fn to the column VALUES (excluding grouping
#   columns), so value predicates such as anyNA() or n_distinct() pick the
#   columns the data says, not what a type proxy says.
# - Pure type tests (is.numeric, is.ordered, ...) depend on the class only and
#   are resolved natively (no CPU fallback).
# - The across() selection sees columns an earlier dot of the same call
#   created (covered here only through value predicates).

f40a_df <- function() {
  data.frame(
    g = c(1L, 1L, 2L, 2L, 2L),
    x = c(1, NA, 3, 4, 4),
    y = c(1L, 2L, 3L, 3L, 3L),
    s = c("a", "b", "c", "d", NA),
    k = c(5, 5, 5, 5, 5),
    stringsAsFactors = FALSE
  )
}

f40a_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f40a_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f40a_both(df, pipeline, ...)
}

test_that("mutate(across(where(<value predicate>))) picks columns from the data", {
  skip_if_no_gpu()
  df <- f40a_df()
  f40a_both(df, function(d) mutate(d, across(where(~ anyNA(.x)), ~ is.na(.x))))
  f40a_both(df, function(d) mutate(d, across(where(~ !anyNA(.x)), ~ 0)))
  f40a_both(df, function(d) mutate(d, across(where(~ dplyr::n_distinct(.x) > 2), ~ 1L)))
  f40a_both(df, function(d) mutate(d, across(where(function(col) is.numeric(col) && !anyNA(col)),
                                              ~ .x * 2)))
  f40a_both(df, function(d) mutate(d, across(c(g, where(~ anyNA(.x))), ~ 0, .names = "{.col}_z")))
  # a column created earlier in the same call is visible to the predicate
  f40a_both(df, function(d) mutate(d, z = NA, across(where(~ anyNA(.x)), ~ 0)))
})

test_that("summarise()/transmute() across value predicates, ungrouped and grouped", {
  skip_if_no_gpu()
  df <- f40a_df()
  f40a_both(df, function(d) summarise(d, across(where(~ dplyr::n_distinct(.x) > 3), ~ 1L)))
  f40a_both(df, function(d) summarise(d, across(where(~ anyNA(.x)), ~ sum(is.na(.x)))))
  f40a_both(df, function(d) transmute(d, across(where(~ anyNA(.x)), ~ is.na(.x))))
  # grouped: group columns are excluded from the selection
  f40a_both(df, function(d) d |> group_by(g) |>
              summarise(across(where(~ dplyr::n_distinct(.x) > 1), ~ length(.x))),
            arrange_by = "g")
  f40a_both(df, function(d) d |> group_by(g) |>
              mutate(across(where(~ dplyr::n_distinct(.x) == 1), ~ -1)))
  f40a_both(df, function(d) d |> group_by(y) |>
              transmute(across(where(~ !anyNA(.x)), ~ .x)))
})

test_that("filter(if_any/if_all(where(<value predicate>))) picks columns from the data", {
  skip_if_no_gpu()
  df <- f40a_df()
  f40a_both(df, function(d) filter(d, if_any(where(~ anyNA(.x)) & where(is.numeric), ~ .x > 2)))
  f40a_both(df, function(d) filter(d, if_all(where(~ is.numeric(.x) && !anyNA(.x)), ~ .x > 1)))
  f40a_both(df, function(d) d |> group_by(g) |>
              filter(if_any(where(~ dplyr::n_distinct(.x) > 2), ~ .x > 3)))
  # mutate() and summarise() spellings of if_any()
  f40a_both(df, function(d) mutate(d, h = dplyr::if_any(where(~ anyNA(.x)) & where(is.numeric), ~ .x > 3)))
})

test_that("value predicates after earlier verbs see the transformed data (lazy chains)", {
  skip_if_no_gpu()
  df <- f40a_df()
  f40a_both(df, function(d) d |> filter(!is.na(x)) |>
              mutate(across(where(~ anyNA(.x)), ~ 0)))
  f40a_both(df, function(d) d |> mutate(x = dplyr::coalesce(x, 0)) |>
              summarise(across(where(~ anyNA(.x)), ~ dplyr::n())))
})

test_that("value predicates on zero-row input", {
  skip_if_no_gpu()
  df <- f40a_df()[0, ]
  f40a_both(df, function(d) mutate(d, across(where(~ !anyNA(.x)), ~ .x)))
  f40a_both(df, function(d) summarise(d, across(where(~ anyNA(.x)), ~ dplyr::n())))
})

test_that("pure type tests in across() stay native, including is.ordered", {
  skip_if_no_gpu()
  df <- f40a_df()
  df$o <- factor(c("lo", "hi", "lo", "hi", "hi"), levels = c("lo", "hi"), ordered = TRUE)
  df$f <- factor(c("u", "v", "u", "v", "u"))
  f40a_native(df, function(d) mutate(d, across(where(is.numeric), ~ .x + 1)))
  f40a_native(df, function(d) mutate(d, across(where(is.ordered), is.na)))
  f40a_native(df, function(d) mutate(d, across(where(~ is.factor(.x) && !is.ordered(.x)), ~ is.na(.x))))
  f40a_native(df, function(d) d |> group_by(g) |> summarise(across(where(is.double), max)),
              arrange_by = "g")
  f40a_native(df, function(d) filter(d, if_all(where(is.integer), ~ .x > 1)))
})

test_that("a predicate that returns a non-logical value raises dplyr's error", {
  skip_if_no_gpu()
  df <- f40a_df()
  g <- tbl_gpu(df)
  expect_error(mutate(df, across(where(~ 1), ~ 0)), class = "dplyr:::mutate_error")
  expect_error(collect(mutate(g, across(where(~ 1), ~ 0))), class = "dplyr:::mutate_error")
})
