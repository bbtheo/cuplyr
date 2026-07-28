# Mirrored dplyr-oracle tests for order_by()/with_order() and lag()/lead()'s
# own `order_by=` argument, inside mutate() (Phase 5, task W4).
#
# Semantics verified empirically against dplyr 1.2.1 (see R/ir.R's
# apply_order_override() for the full narrative):
#   - order_by(order_by, call) re-evaluates `call` as if the rows were
#     sorted (ascending, or descending under desc()) by `order_by`, then
#     un-sorts the result back to the ORIGINAL row order.
#   - with_order(order_by, fun, x) == order_by(order_by, fun(x)) (dplyr
#     1.2.1's actual signature has no `order_by_desc=` parameter -- wrap
#     `order_by` in desc() for descending order instead, same as
#     order_by()).
#   - lag(x, order_by = y) / lead(x, order_by = y) is the SAME frame
#     rewrite, just spelled as the shift function's own named argument
#     instead of the separate wrapper.
#   - order_by()/with_order() genuinely ERROR in dplyr itself when wrapping
#     any "agg"-kind function (mean()/sum()/min()/max()/n()/sd()/var()/
#     first()/last()/nth()) -- the un-shuffle step requires the wrapped
#     call to return one value per input row, never true for a reduction.
#     cuplyr matches this by rejecting the same shape (a hard mutate()
#     error) instead of silently ignoring the requested ordering.
#   - order_by()/with_order() wrapping row_number(x) (ties = "sequential",
#     tie-broken by row position) genuinely changes the result (ties get
#     broken by the given order instead of original position) -- not
#     implemented this wave (would need a two-key rank frame); cuplyr
#     rejects this shape too, rather than silently reproducing dplyr's own
#     (different) tie-break order.
#   - order_by()/with_order() wrapping min_rank()/dense_rank()/
#     percent_rank()/cume_dist() (none of which break ties by position at
#     all) has NO OBSERVABLE EFFECT in dplyr -- cuplyr silently drops the
#     override for these (matches dplyr's actual output, not a rejection).

# =============================================================================
# order_by()
# =============================================================================

test_that("order_by(x, cumsum(x)) matches dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(x = c(3, 1, 4, 1, 5, 9, 2, 6))
  pipeline <- function(d) dplyr::mutate(d, cs = dplyr::order_by(x, cumsum(x)))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("order_by(desc(x), cumsum(x)) matches dplyr (descending order)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(3, 1, 4, 1, 5, 9, 2, 6))
  pipeline <- function(d) dplyr::mutate(d, cs = dplyr::order_by(dplyr::desc(x), cumsum(x)))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("order_by(y, cumsum(x)) matches dplyr, grouped, with a different order column", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b"),
    x = c(3, 1, 4, 1, 5, 9),
    y = c(6, 5, 4, 3, 2, 1)
  )
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(cs = dplyr::order_by(y, cumsum(x)))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("order_by(y, lag(cumsum(x))) resolves across decomposition stages", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5, 6), y = c(3, 1, 2, 6, 4, 5))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(z = dplyr::order_by(y, dplyr::lag(cumsum(x))))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("order_by(y, min_rank(x))/dense_rank()/percent_rank()/cume_dist() are no-ops, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a"), x = c(3, 1, 4), y = c(6, 5, 4))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      mr = dplyr::order_by(y, dplyr::min_rank(x)),
      dr = dplyr::order_by(y, dplyr::dense_rank(x)),
      pr = dplyr::order_by(y, dplyr::percent_rank(x)),
      cd = dplyr::order_by(y, dplyr::cume_dist(x))
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("order_by() wrapping an agg-kind function is a hard mutate() error, matching dplyr's own error", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), x = c(1, 2, 3), y = c(3, 1, 2))
  gt <- tbl_gpu(df)

  expect_error(dplyr::mutate(gt, z = dplyr::order_by(y, mean(x))))
  expect_error(dplyr::mutate(gt, z = dplyr::order_by(y, first(x))))
  expect_error(dplyr::mutate(gt, z = dplyr::order_by(y, n())))

  expect_error(df |> dplyr::group_by(g) |> dplyr::mutate(z = dplyr::order_by(y, mean(x))))
})

test_that("order_by() wrapping row_number(x) with ties in the ranked column is not yet supported", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 1, 1, 2, 2), y = c(3, 1, 2, 5, 4))
  expect_error(dplyr::mutate(tbl_gpu(df), z = dplyr::order_by(y, dplyr::row_number(x))))
})

# =============================================================================
# with_order()
# =============================================================================

test_that("with_order(x, cumsum, x) matches dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(x = c(3, 1, 4, 1, 5, 9, 2, 6))
  pipeline <- function(d) dplyr::mutate(d, cs = dplyr::with_order(x, cumsum, x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("with_order(desc(y), lag, x) matches dplyr, grouped", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b"),
    x = c(3, 1, 4, 1, 5, 9),
    y = c(6, 5, 4, 3, 2, 1)
  )
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(lg = dplyr::with_order(dplyr::desc(y), dplyr::lag, x))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# lag()/lead()'s own order_by= argument
# =============================================================================

test_that("lag(x, order_by = y) matches dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(x = c(3, 1, 4, 1, 5, 9, 2, 6))
  pipeline <- function(d) dplyr::mutate(d, l = dplyr::lag(x, order_by = x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("lag(x, n = 2, order_by = y, default = 0) matches dplyr, grouped", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b"),
    x = c(3, 1, 4, 1, 5, 9),
    y = c(6, 5, 4, 3, 2, 1)
  )
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(l = dplyr::lag(x, n = 2, order_by = y, default = 0))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("lead(x, order_by = y) matches dplyr, grouped", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b"),
    x = c(3, 1, 4, 1, 5, 9),
    y = c(6, 5, 4, 3, 2, 1)
  )
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(ld = dplyr::lead(x, order_by = y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("lag(x, order_by = desc(y)) (descending order_by) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(3, 1, 4, 1, 5, 9, 2, 6), y = c(8, 7, 6, 5, 4, 3, 2, 1))
  pipeline <- function(d) dplyr::mutate(d, l = dplyr::lag(x, order_by = dplyr::desc(y)))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# first()/last()/nth()'s own order_by= argument
# =============================================================================

test_that("first(x, order_by = y)/last()/nth() match dplyr, grouped", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c(1, 1, 1, 2, 2),
    x = c(30, 10, 20, 5, 1),
    y = c(3, 1, 2, 2, 1)
  )
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      f = dplyr::first(x, order_by = y),
      l = dplyr::last(x, order_by = y),
      n2 = dplyr::nth(x, 2, order_by = y)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
