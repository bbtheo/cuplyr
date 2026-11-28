# #74, the coder's own cases next to the acceptance tests (test-parity-74-*.R):
# the warning of min()/max(na.rm = TRUE) for several calls of one dot, the forms
# that stay on the CPU, and the rank under an order. Helpers: helper-parity-74.R.

test_that("#74: a min()/max(na.rm = TRUE) call written twice in a dot warns for each", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(3L, 3L, 1L, 1L, 2L, 2L),
    v = c(NA, NaN, 1, NA, NA, NA),
    w = c(1, 2, NA, NA, 3, 4)
  )
  # group order: sorted keys for group_by(), first appearance for `.by`
  p74_native_warn(df, function(d) mutate(d, r = max(v, na.rm = TRUE) - max(v, na.rm = TRUE), .by = g))
  p74_native_warn(df, function(d) mutate(group_by(d, g), r = max(v, na.rm = TRUE) - max(v, na.rm = TRUE)))
  # a hit in a later call names its own group; the first warning is the first group
  p74_native_warn(df, function(d) mutate(d, r = max(w, na.rm = TRUE) + min(v, na.rm = TRUE), .by = g))
  # an expression argument
  p74_native_warn(df, function(d) mutate(d, r = max(v * 2, na.rm = TRUE), .by = g))
  # two dots: the first dot's group is the first warning
  p74_native_warn(df, function(d) mutate(d, a = min(w, na.rm = TRUE), b = max(v, na.rm = TRUE), .by = g))
})

test_that("#74: min()/max(na.rm = TRUE) shapes the GPU leaves to dplyr keep its result", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), v = c(NA, 2, NA, NaN), s = c("a", NA, NA, "b"))
  # in filter(): the warning names the predicate
  p74_parity(df, function(d) filter(d, v < max(v, na.rm = TRUE), .by = g))
  # a call whose result feeds a later window stage
  p74_parity(df, function(d) mutate(d, m = max(v, na.rm = TRUE), l = dplyr::lag(m), .by = g))
  # a string, and a call over a window result
  p74_parity(df, function(d) mutate(d, m = max(s, na.rm = TRUE), .by = g))
  p74_parity(df, function(d) mutate(d, m = max(cumsum(v), na.rm = TRUE), .by = g))
})

test_that("#74: mean(na.rm = TRUE) next to a lazy filter and an order_by rank", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L), v = c(1, NA, 3, NaN, NA), k = c(2, 1, 3, 1, 2))
  p74_native(df, function(d) {
    mutate(filter(d, k < 3), m = mean(v, na.rm = TRUE), r = dplyr::order_by(k, row_number(v)), .by = g)
  })
  # ranks under desc(): ties of x broken by the order
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(desc(k), row_number(desc(v))), .by = g))
})
