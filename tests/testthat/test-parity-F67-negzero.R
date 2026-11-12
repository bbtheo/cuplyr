# F67: a negative zero keeps its sign on the GPU. R's `-0` is a call to unary
# minus, so the literal and `-x` on a zero column must both give IEEE -0, as
# R does. The sign of a zero is checked from 1 / v (identical(0, -0) is TRUE).

# Tests see the package namespace, which doesn't import these; bind dplyr's
# own functions here, as library(dplyr) would for a user.
for (fn in c("if_else", "case_when", "coalesce", "recode", "replace_values", "replace_when",
            "lag", "lead", "cummean")) {
  assign(fn, getExportedValue("dplyr", fn))
}

test_that("division by a -0 literal gives dplyr's infinities", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, -1, 0, NA, Inf, -2.5))
  f67_check(df, function(d) {
    mutate(d,
      a = x / -0, b = x / -0.0, c = x / -(0), e = 1 / (x * -0),
      f = x / (-1 * 0), g = x / (0 * -1), h = x / -(-0)
    )
  })
})

test_that("unary minus of a zero column gives the opposite signed zero", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0, -0, 1, -1, NA, Inf))
  f67_check(df, function(d) mutate(d, n = -x, nn = -(-x), r = 1 / -x, p = 1 / (-x + 0)))
})

test_that("a -0 constant column and -0 in value positions keep the sign", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, -1, 0, NA), y = c(NA, 2, NA, 3))
  f67_check(df, function(d) mutate(d, z = -0, w = x / z))
  f67_check(df, function(d) {
    mutate(d,
      a = if_else(x > 5, 1, -0), b = coalesce(y, -0),
      c = case_when(x > 0 ~ -0, .default = 1), e = (x - x) + -0
    )
  })
})

test_that("filter() and grouped mutate() see the -0 literal's sign", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, -1, 0, 2, -3), g = c("a", "a", "b", "b", "b"))
  f67_check(df, function(d) filter(d, x / -0 < 0))
  f67_check(df, function(d) filter(d, 1 / (-(x - x)) < 0))
  f67_check(df, function(d) d |> group_by(g) |> mutate(r = x / -0, s = 1 / (x * -0)))
})

test_that("integer -0L is plain 0", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, -1L, 0L, NA), x = c(1, -1, 0, NA))
  f67_check(df, function(d) {
    mutate(d, a = x / -0L, b = -i, c = i * -0L, e = i / -0L, f = i - -0L)
  })
})

test_that("sign(), atan2() and recode() with signed zeros match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, -1, 0, -0, NA))
  f67_check(df, function(d) mutate(d, s = sign(x * -0), t = sign(-x)), native = FALSE)
  f67_check(df, function(d) mutate(d, a = atan2(-0, x), b = atan2(x * -0, -1)), native = FALSE)
  f67_check(df, function(d) {
    mutate(d, r = recode(x, `0` = "zero", .default = "other"))
  }, native = FALSE)
})

test_that("a -0 that R evaluates (environment, !!, folded constant, lag default) keeps its sign", {
  skip_if_no_gpu()
  z <- -0
  k <- -1
  df <- data.frame(x = c(1, -1, 0, 2), y = c(NA, 2, NA, 3), g = c("a", "b", "a", "b"))
  f67_check(df, function(d) {
    mutate(d,
      a = x / z, b = x / (k * 0), m = x / !!(-0), c = if_else(x > 5, 1, z),
      e = case_when(x > 0 ~ z, .default = 1), w = coalesce(y, z)
    )
  })
  f67_check(df, function(d) {
    mutate(d, rv = replace_values(x, 0 ~ z), rw = replace_when(x, x == 0 ~ z))
  })
  f67_check(df, function(d) mutate(d, l = lag(x, default = z), l2 = lead(x, default = -0)))
  f67_check(df, function(d) d |> group_by(g) |> mutate(a = x / z, l = lag(x, default = z)))
  f67_check(df, function(d) filter(d, x / z < 0))
  f67_check(df, function(d) filter(d, x / (k * 0) > 0))
  f67_check(df[df$x != 0, ], function(d) summarise(d, n = sum(x / z > 0), p = sum(x / (k * 0) > 0)))
})

test_that("summarise() keeps aggregates that differ only in a zero's sign apart", {
  skip_if_no_gpu()
  z <- -0
  k <- -1
  df <- data.frame(x = c(1, 2, 3), g = c("a", "a", "b"))
  f67_check(df, function(d) {
    summarise(d, a = sum(x / z), b = sum(x / 0), n = sum(x / 0 < 0), m = sum(x / (k * 0) < 0))
  })
  f67_check(df, function(d) summarise(d, b = sum(x / 0), a = sum(x / z)))
  f67_check(df, function(d) {
    d |> group_by(g) |> summarise(b = sum(x / 0), a = sum(x / z), m = sum(x / (k * 0) < 0))
  })
})

test_that("sum(), mean() and cumsum() of negative zeros are +0, as in R", {
  skip_if_no_gpu()
  # R's sums start from +0, so they never return -0 (0 + -0 is +0).
  df <- data.frame(x = c(0, 0, 0, 0), y = c(-0, -0, -0, NA), g = c("a", "a", "b", "b"))
  f67_check(df, function(d) {
    summarise(d, s = sum(-x), m = mean(-x), sy = sum(y, na.rm = TRUE), my = mean(y, na.rm = TRUE))
  })
  f67_check(df, function(d) {
    d |> group_by(g) |> summarise(s = sum(-x), m = mean(-x), sy = sum(y, na.rm = TRUE))
  })
  f67_check(df, function(d) mutate(d, c = cumsum(-x), cy = cumsum(y), cm = cummean(-x)))
  f67_check(df, function(d) {
    d |> group_by(g) |> mutate(s = sum(-x), m = mean(-x), c = cumsum(-x))
  })
})

test_that("the colwise memo keeps functions that differ only in a zero's sign apart", {
  skip_if_no_gpu()
  z <- -0
  f_neg <- eval(bquote(function(v) v / .(z)))
  f_pos <- function(v) v / 0
  df <- data.frame(x = c(1, -1, 2))
  f67_check(df, function(d) mutate_at(d, "x", list(a = f_neg, b = f_pos)))
  f67_check(df, function(d) mutate_at(d, "x", list(b = f_pos, a = f_neg)))
})

test_that("long sum/mean/cumsum results with many negative zeros are +0", {
  skip_if_no_gpu()
  # Row-length results (a cumsum, a summarise over 1e5 groups) with -0 on
  # many rows, so a "find a -0 first" shortcut must still fix them.
  set.seed(6701)
  n <- 2e5
  g <- rep(seq_len(n / 2), each = 2)
  y <- ifelse(g %% 3 == 0, -0, runif(n))
  y[seq_len(1e5)] <- -0
  df <- data.frame(y = y, g = g, h = rep(1:10, each = n / 10))
  f67_check(df, function(d) d |> group_by(g) |> summarise(s = sum(y), m = mean(y)))
  f67_check(df, function(d) mutate(d, c = cumsum(y), cm = cummean(y)))
  f67_check(df, function(d) d |> group_by(h) |> mutate(c = cumsum(y), s = sum(y)))
})
