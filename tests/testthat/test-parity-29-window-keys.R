# #29 acceptance tests: a window function in a computed key of distinct(),
# count() and add_count() (and in slice_min()/slice_max()'s `order_by`).
#
# Before #29 the key went straight to one ast_mutate node, and a window op
# (dense_rank(), row_number(), lag(), cumsum(), mean(), n(), ...) died at
# lowering with "internal: window op 'dense_rank' reached lowering".
#
# dplyr's contract (dplyr 1.2.1, R/group-by.R::add_computed_columns(), used by
# distinct_prepare() and group_by_prepare(), so by distinct(), count() and
# add_count()): every computed key is evaluated by mutate_cols() on the
# UNGROUPED data, so a window or aggregate sees the whole table even when the
# input is grouped. distinct() then keeps the first row of each key (group
# columns prepended), count() groups by the keys on top of the existing groups.
# slice_min()/slice_max() evaluate `order_by` in the (grouped) data mask.
#
# cuplyr contract: the result matches dplyr in eager and lazy mode, and the
# plain window keys below run on the GPU (no CPU fallback). Window functions
# are written dplyr::f(): testthat does not attach dplyr, and the oracle
# evaluates the same pipeline.

w29_df <- function() {
  data.frame(
    g = c("a", "b", "a", "b", "a", "b", "a", "b"),
    k = c(3L, 1L, 3L, NA, 2L, 1L, 3L, 2L),
    x = c(2.5, NA, 2.5, 1, -1, 0, 7, 1),
    stringsAsFactors = FALSE
  )
}

w29_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Same comparison, but any CPU fallback is an error.
w29_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# --- distinct() ---------------------------------------------------------------

test_that("distinct() with a window key matches dplyr (the fuzz case)", {
  skip_if_no_gpu()
  x <- data.frame(k = c(1L, 1L), a = c(3L, 7L))
  w29_both(x, function(t) {
    t |>
      mutate(m1 = factor("b"), .by = a) |>
      distinct(g2 = dplyr::dense_rank(k) > 1L)
  })
})

test_that("distinct() with ranking, offset and cumulative keys matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) distinct(t, r = dplyr::dense_rank(k)))
  w29_both(df, function(t) distinct(t, r = dplyr::min_rank(x)))
  w29_both(df, function(t) distinct(t, r = dplyr::row_number() %% 3L))
  w29_both(df, function(t) distinct(t, p = dplyr::lag(k)))
  w29_both(df, function(t) distinct(t, s = cumsum(x > 0)))
  # an unnamed window key is named by its label
  w29_both(df, function(t) distinct(t, dplyr::dense_rank(k)))
  # a bare column next to a window key, and a key built from an earlier one
  w29_both(df, function(t) distinct(t, g, r = dplyr::dense_rank(k)))
  w29_both(df, function(t) distinct(t, r = dplyr::dense_rank(k), r2 = r * 2L))
})

test_that("distinct() with an aggregate key matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) distinct(t, big = x > mean(x, na.rm = TRUE)))
  w29_both(df, function(t) distinct(t, g, m = dplyr::n()))
  w29_both(df, function(t) distinct(t, m = max(k, na.rm = TRUE) - k))
})

test_that("distinct(.keep_all = TRUE) with a window key matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) distinct(t, r = dplyr::dense_rank(k), .keep_all = TRUE))
  # a window key that replaces an existing column keeps its position
  w29_both(df, function(t) distinct(t, k = dplyr::dense_rank(k), .keep_all = TRUE))
})

test_that("grouped distinct() evaluates a window key over the whole table, like dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) t |> group_by(g) |> distinct(r = dplyr::dense_rank(k)))
  w29_both(df, function(t) t |> group_by(g) |> distinct(r = dplyr::row_number() > 4L))
  w29_both(df, function(t) t |> group_by(g) |> distinct(k, m = mean(x, na.rm = TRUE)))
  w29_both(df, function(t) t |> group_by(g) |> distinct(r = dplyr::dense_rank(k), .keep_all = TRUE))
})

test_that("distinct() with a window key on 0 rows matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()[0, ]
  w29_both(df, function(t) distinct(t, r = dplyr::dense_rank(k)))
  w29_both(df, function(t) t |> group_by(g) |> distinct(r = dplyr::dense_rank(k)))
})

test_that("distinct() with a plain window key runs on the GPU", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_native(df, function(t) distinct(t, r = dplyr::dense_rank(k)))
  w29_native(df, function(t) distinct(t, g, r = dplyr::row_number() > 4L))
  w29_native(df, function(t) t |> group_by(g) |> distinct(r = dplyr::dense_rank(k)))
})

# --- count() / add_count() ----------------------------------------------------

test_that("count() with a window key matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) count(t, r = dplyr::dense_rank(k)))
  w29_both(df, function(t) count(t, g, p = dplyr::lag(k)))
  w29_both(df, function(t) count(t, dplyr::dense_rank(k)))
  w29_both(df, function(t) count(t, big = x > mean(x, na.rm = TRUE)))
  w29_both(df, function(t) count(t, r = dplyr::dense_rank(k), sort = TRUE))
  w29_both(df, function(t) count(t, r = dplyr::dense_rank(k), wt = x))
  w29_both(df, function(t) count(t, r = dplyr::dense_rank(k), name = "rows"))
})

test_that("grouped count() evaluates a window key over the whole table, like dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) t |> group_by(g) |> count(r = dplyr::dense_rank(k)))
  w29_both(df, function(t) t |> group_by(g) |> count(r = dplyr::row_number() > 4L))
})

test_that("count() with a window key on 0 rows matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()[0, ]
  w29_both(df, function(t) count(t, r = dplyr::dense_rank(k)))
})

test_that("add_count() with a window key matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) add_count(t, r = dplyr::dense_rank(k)))
  w29_both(df, function(t) add_count(t, g, p = dplyr::lag(k)))
  w29_both(df, function(t) t |> group_by(g) |> add_count(r = dplyr::dense_rank(k)))
})

test_that("count() with a plain window key runs on the GPU", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_native(df, function(t) count(t, r = dplyr::dense_rank(k)))
  w29_native(df, function(t) t |> group_by(g) |> count(r = dplyr::dense_rank(k)))
})

# --- slice_min() / slice_max() order_by ---------------------------------------

test_that("slice_min()/slice_max() with a window order_by matches dplyr", {
  skip_if_no_gpu()
  df <- w29_df()
  w29_both(df, function(t) slice_min(t, dplyr::dense_rank(k), n = 2))
  w29_both(df, function(t) slice_max(t, dplyr::row_number(), n = 3))
  w29_both(df, function(t) t |> group_by(g) |> slice_max(dplyr::row_number(), n = 1))
  w29_both(df, function(t) t |> group_by(g) |> slice_min(dplyr::lag(x), n = 1))
})

# --- grouping metadata after a window key (review of #29) ---------------------
# The oracle compares group_vars() and the ungrouped data only, so these pipe
# the result into summarise(n()), which reads the group structure itself.
# dplyr's mutate branch (add_computed_columns() -> dplyr_col_modify()) regroups
# when a key dot names a group column, which drops the empty groups a
# `.drop = FALSE` filter() kept.

test_that("a window key that overwrites a group column regroups like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(10L, 10L, 20L, 30L), k = c(5L, 6L, 7L, 8L))
  kept <- function(t) t |> group_by(g, .drop = FALSE) |> filter(k < 7L)
  w29_both(df, function(t) kept(t) |> distinct(g = dplyr::dense_rank(k)) |>
             summarise(n = dplyr::n()))
  w29_both(df, function(t) kept(t) |> distinct(g = dplyr::dense_rank(k), .keep_all = TRUE) |>
             summarise(n = dplyr::n(), k = sum(k)))
})

test_that("a window key next to a bare group column regroups like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(10L, 10L, 20L, 30L), k = c(5L, 6L, 7L, 8L))
  kept <- function(t) t |> group_by(g, .drop = FALSE) |> filter(k < 7L)
  w29_both(df, function(t) kept(t) |> distinct(g, r = dplyr::dense_rank(k)) |>
             summarise(n = dplyr::n()))
})

test_that("a window key keeps a .drop = FALSE grouping that no key writes", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("a", "a", "b"), levels = c("a", "b", "z")),
    k = c(5L, 6L, 7L)
  )
  w29_both(df, function(t) t |> group_by(f, .drop = FALSE) |>
             distinct(r = dplyr::dense_rank(k)) |> summarise(n = dplyr::n()))
  w29_both(df, function(t) t |> group_by(f, .drop = FALSE) |>
             count(r = dplyr::dense_rank(k)))
})
