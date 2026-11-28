# #74: row_number(x) and ntile(x, n) inside order_by()/with_order() run on the
# GPU. dplyr permutes x into the order of `order_by` (vec_order_radix():
# stable, NA and NaN tied last), ranks it with ties broken by that position,
# and permutes back: x's ties are broken by the order column, then by row.
# Helpers: helper-parity-74.R.

p74_ob_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L),
    x = c(2, 1, 2, NA, 2, 1, 2, 2, NaN),
    i = c(5L, 5L, 3L, 5L, NA, 1L, 1L, 1L, 1L),
    s = c("b", "a", "b", NA, "a", "a", "c", "a", "b"),
    k = c(3, 2, 1, 5, NaN, NA, 4, 4, 0),
    j = c(2L, 2L, 1L, NA, 3L, 1L, 3L, 2L, 3L)
  )
}

test_that("#74: order_by(o, row_number(x)), ungrouped, grouped and .by", {
  skip_if_no_gpu()
  df <- p74_ob_df()
  for (col in c("x", "i", "s")) {
    x <- as.symbol(col)
    p74_native(df, function(d) mutate(d, r = dplyr::order_by(k, row_number(!!x))))
    p74_native(df, function(d) mutate(d, r = dplyr::order_by(j, row_number(!!x))))
    p74_native(df, function(d) mutate(group_by(d, g), r = dplyr::order_by(k, row_number(!!x))))
    p74_native(df, function(d) mutate(d, r = dplyr::order_by(j, row_number(!!x)), .by = g))
  }
})

test_that("#74: order_by(desc(o), row_number(x)) and with_order()", {
  skip_if_no_gpu()
  df <- p74_ob_df()
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(desc(k), row_number(x))))
  p74_native(df, function(d) mutate(group_by(d, g), r = dplyr::order_by(desc(j), row_number(i))))
  p74_native(df, function(d) mutate(d, r = dplyr::with_order(k, row_number, x)))
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(j, dplyr::row_number(x)), .by = g))
  # row_number() of an expression, and arithmetic around the call
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(k, row_number(x * 2))))
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(k, row_number(x)) + 1L))
  # row_number(desc(x)) under an order
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(j, row_number(desc(x)))))
})

test_that("#74: order_by(o, ntile(x, n))", {
  skip_if_no_gpu()
  df <- p74_ob_df()
  for (n in c(1L, 2L, 3L, 20L)) {
    p74_native(df, function(d) mutate(d, r = dplyr::order_by(k, dplyr::ntile(x, !!n))))
    p74_native(df, function(d) mutate(group_by(d, g), r = dplyr::order_by(j, dplyr::ntile(i, !!n))))
  }
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(desc(k), dplyr::ntile(s, 2)), .by = g))
  p74_native(df, function(d) mutate(d, r = dplyr::with_order(j, dplyr::ntile, x, n = 2)))
})

test_that("#74: order_by() ranks next to other windows, in filter(), after a lazy filter", {
  skip_if_no_gpu()
  df <- p74_ob_df()
  p74_native(df, function(d) {
    mutate(d, r = dplyr::order_by(k, row_number(x)), c = dplyr::order_by(k, cumsum(j)), m = dplyr::min_rank(x), .by = g)
  })
  p74_native(df, function(d) filter(d, dplyr::order_by(k, row_number(x)) <= 2L, .by = g))
  p74_native(df, function(d) mutate(filter(d, j != 2L), r = dplyr::order_by(k, dplyr::ntile(x, 2))))
})

test_that("#74: order_by() ranks over an empty table, all-NA x, and all-tied x", {
  skip_if_no_gpu()
  df <- p74_ob_df()
  p74_native(df[0, ], function(d) mutate(d, r = dplyr::order_by(k, row_number(x))))
  p74_native(df[0, ], function(d) mutate(d, r = dplyr::order_by(k, dplyr::ntile(x, 2)), .by = g))
  na <- tibble::tibble(x = c(NA_real_, NA, NA), k = c(2, 1, 3))
  p74_native(na, function(d) mutate(d, r = dplyr::order_by(k, row_number(x)), t = dplyr::order_by(k, dplyr::ntile(x, 2))))
  tied <- tibble::tibble(x = c(1L, 1L, 1L, 1L), k = c(4, 2, NA, 1))
  p74_native(tied, function(d) mutate(d, r = dplyr::order_by(k, row_number(x)), t = dplyr::order_by(k, dplyr::ntile(x, 3))))
})

test_that("#74: order_by() forms dplyr rejects or the GPU does not take keep dplyr's result", {
  skip_if_no_gpu()
  df <- p74_ob_df()
  # a row-sized extra argument and a nested order_by() stay dplyr's
  p74_parity(df, function(d) mutate(d, r = dplyr::order_by(k, dplyr::ntile(x, 2) + j)))
  p74_parity(df, function(d) mutate(d, r = dplyr::order_by(k, dplyr::order_by(j, row_number(x)))))
  # ntile() with a bad n is dplyr's error
  for (mode in c(FALSE, TRUE)) {
    expect_error(collect(mutate(tbl_gpu(df, lazy = mode), r = dplyr::order_by(k, dplyr::ntile(x, 0)))))
  }
})
