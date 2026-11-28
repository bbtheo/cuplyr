# #74: window mean()/min()/max() with `na.rm = TRUE` in mutate() (grouped,
# `.by =` and ungrouped) run on the GPU. dplyr drops NA and NaN (is.na()) from
# each group: mean() of nothing left is NaN, max()/min() of nothing left is
# -Inf/Inf with one folded dplyr warning ("There were N warnings in
# `mutate()`", the first one's group named). Helpers: helper-parity-74.R.

p74_narm_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L),
    h = c("a", "a", "b", "b", "b", "a", "a", "b", "b"),
    v = c(1.5, NA, -2, NA, NA, NaN, 4, NaN, NA),
    i = c(3L, NA, 1L, NA, NA, 7L, NA, 2L, -5L),
    l = c(TRUE, NA, FALSE, NA, NA, TRUE, TRUE, NA, FALSE)
  )
}

test_that("#74: mean(na.rm = TRUE) in mutate(.by =), grouped and ungrouped", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  for (col in c("v", "i", "l")) {
    x <- as.symbol(col)
    p74_native(df, function(d) mutate(d, m = mean(!!x, na.rm = TRUE), .by = g))
    p74_native(df, function(d) mutate(group_by(d, g), m = mean(!!x, na.rm = TRUE)))
    p74_native(df, function(d) mutate(d, m = mean(!!x, na.rm = TRUE)))
    p74_native(df, function(d) mutate(d, m = mean(!!x, na.rm = TRUE), .by = c(g, h)))
  }
})

test_that("#74: mean(na.rm = TRUE): NaN and all-NA groups, arithmetic around it, later dots", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  # group 2 is all NA, group 4 only NaN/NA: NaN; group 3 drops its NaN: 4
  p74_native(df, function(d) mutate(d, m = mean(v, na.rm = TRUE), .by = g))
  p74_native(df, function(d) mutate(d, z = v - mean(v, na.rm = TRUE), .by = g))
  p74_native(df, function(d) mutate(group_by(d, g), m = mean(v, na.rm = TRUE), k = m * 2))
  p74_native(df, function(d) mutate(d, a = mean(v, na.rm = TRUE), b = mean(v), .by = g))
  # Inf and -Inf in one group: NaN, as in R
  inf <- tibble::tibble(g = c(1L, 1L, 1L, 2L), v = c(Inf, -Inf, NA, Inf))
  p74_native(inf, function(d) mutate(d, m = mean(v, na.rm = TRUE), .by = g))
  # na.rm given by a variable
  rm <- TRUE
  p74_native(df, function(d) mutate(d, m = mean(v, na.rm = rm), .by = g))
})

test_that("#74: mean(na.rm = TRUE) over an empty table and after a lazy filter", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  p74_native(df[0, ], function(d) mutate(d, m = mean(v, na.rm = TRUE), .by = g))
  p74_native(df[0, ], function(d) mutate(d, m = mean(i, na.rm = TRUE)))
  p74_native(df, function(d) mutate(filter(d, g != 1L), m = mean(v, na.rm = TRUE), .by = g))
  p74_native(df, function(d) mutate(filter(d, g > 10L), m = mean(v, na.rm = TRUE), .by = g))
})

test_that("#74: mean(na.rm = TRUE) in filter()", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  p74_native(df, function(d) filter(d, v > mean(v, na.rm = TRUE), .by = g))
  p74_native(df, function(d) filter(group_by(d, g), i >= mean(i, na.rm = TRUE)))
})

test_that("#74: max()/min(na.rm = TRUE) of a double, with dplyr's all-NA warning", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  # groups 2 and 4 have no non-missing value: -Inf/Inf and one folded warning
  p74_native_warn(df, function(d) mutate(d, m = max(v, na.rm = TRUE), .by = g))
  p74_native_warn(df, function(d) mutate(group_by(d, g), m = min(v, na.rm = TRUE)))
  p74_native_warn(df, function(d) mutate(d, m = max(v, na.rm = TRUE), .by = c(g, h)))
  p74_native_warn(df, function(d) mutate(d, a = max(v, na.rm = TRUE), b = min(v, na.rm = TRUE),
                                         .by = g))
  p74_native_warn(df, function(d) mutate(d, r = max(v, na.rm = TRUE) - min(v, na.rm = TRUE),
                                         .by = g))
  # every group has a value: no warning
  ok <- tibble::tibble(g = c(1L, 1L, 2L, 2L), v = c(NA, 2, NaN, -1))
  p74_native(ok, function(d) mutate(d, m = max(v, na.rm = TRUE), .by = g))
  p74_native(ok, function(d) mutate(group_by(d, g), m = min(v, na.rm = TRUE)))
  # ungrouped: one value left, or none (one warning, no group line)
  p74_native(ok, function(d) mutate(d, m = max(v, na.rm = TRUE)))
  none <- tibble::tibble(v = c(NA, NaN, NA))
  p74_native_warn(none, function(d) mutate(d, m = max(v, na.rm = TRUE)))
  p74_native_warn(none, function(d) mutate(d, m = min(v, na.rm = TRUE)))
})

test_that("#74: max()/min(na.rm = TRUE) of a double over a zero-row table", {
  skip_if_no_gpu()
  df <- p74_narm_df()[0, ]
  p74_native_warn(df, function(d) mutate(d, m = max(v, na.rm = TRUE)))
  p74_native_warn(df, function(d) mutate(d, m = min(v, na.rm = TRUE), .by = g))
})

test_that("#74: max()/min(na.rm = TRUE) of an integer or logical match dplyr", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  # an all-NA group makes dplyr's whole column a double (-Inf), so the type is
  # the data's to decide: a CPU fallback is allowed here
  p74_parity(df, function(d) mutate(d, m = max(i, na.rm = TRUE), .by = g))
  p74_parity(df, function(d) mutate(group_by(d, g), m = min(l, na.rm = TRUE)))
  full <- tibble::tibble(g = c(1L, 1L, 2L), i = c(NA, 4L, 2L))
  p74_parity(full, function(d) mutate(d, m = max(i, na.rm = TRUE), .by = g))
})

test_that("#74: other na.rm forms keep dplyr's result", {
  skip_if_no_gpu()
  df <- p74_narm_df()
  # na.rm = FALSE is the plain call
  p74_native(df, function(d) mutate(d, m = mean(v, na.rm = FALSE), .by = g))
  p74_native(df, function(d) mutate(d, m = max(v, na.rm = FALSE), .by = g))
  p74_parity(df, function(d) mutate(d, m = median(v, na.rm = TRUE), .by = g))
  p74_parity(df, function(d) mutate(d, m = sd(v, na.rm = TRUE), .by = g))
  # a na.rm that is not TRUE/FALSE: dplyr's result, wherever it runs
  p74_parity(df, function(d) mutate(d, m = mean(v, na.rm = "yes"), .by = g))
  p74_parity(df, function(d) mutate(d, m = mean(v, na.rm = NA), .by = g))
})
