# F67 round 2: a -0 that reaches the GPU as a scalar (cudf's numeric_scalar
# stores +0 for any value == 0), and the types and edge rows of round().

for (fn in c("if_else", "case_when", "coalesce", "replace_values", "replace_when", "lag", "lead")) {
  assign(fn, getExportedValue("dplyr", fn))
}

test_that("an AST literal holds a real -0 on the device", {
  skip_if_no_gpu()
  z <- -0
  k <- -1
  df <- data.frame(x = c(1, -1, 0, 2, NA))
  # every slot that takes a scalar: arithmetic, comparison, the if_else scalar
  # overloads and the missing slot, fold_clauses values, shift defaults
  f67_check(df, function(d) mutate(d, a = x / z, b = 1 / (x * z), c = x / !!(-0), e = x / (k * 0)))
  f67_check(df, function(d) mutate(d, a = if_else(x > 1, z, x), b = if_else(x > 1, x, z),
                                   c = if_else(x > 1, 1, 2, missing = z)))
  f67_check(df, function(d) mutate(d, a = case_when(x > 0 ~ z, x < 0 ~ 5, .default = z),
                                   b = replace_values(x, 2 ~ z), c = replace_when(x, x > 1 ~ z)))
  f67_check(df, function(d) mutate(d, a = lag(x, default = z), b = lead(x, default = !!(-0)),
                                   c = lag(x, 2, default = k * 0)))
  f67_check(df, function(d) filter(d, x / z < 0))
  f67_check(df, function(d) filter(d, 1 / (x * !!(-0)) > 0))
  f67_check(df, function(d) mutate(d, a = z))
})

test_that("a -0 scalar still compares equal to 0", {
  skip_if_no_gpu()
  z <- -0
  df <- data.frame(x = c(1, -1, 0, -0, NA))
  f67_check(df, function(d) mutate(d, a = x == z, b = x > z, c = x >= !!(-0)))
  f67_check(df, function(d) filter(d, x == z))
})

test_that("round() of integer and logical columns is a double that keeps x", {
  skip_if_no_gpu()
  df <- data.frame(i = c(5L, -12L, 0L, NA, 1234L), l = c(TRUE, FALSE, NA, TRUE, FALSE))
  f67_check(df, function(d) mutate(d, a = round(i, 2), b = round(i, 0), c = round(i, -1),
                                   e = round(l, 1), f = round(i, 1.5), g = round(i, 400)))
  f67_check(df, function(d) mutate(d, a = round(i, -400), b = round(l, -2), c = round(i, -2.4)))
})

test_that("round() of a string is dplyr's error", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(s = c("a", "b")))
  expect_error(collect(mutate(g, r = round(s, 1))), "non-numeric")
  expect_error(collect(mutate(as_lazy(g), r = round(s))), "non-numeric")
})

test_that("round() keeps nulls, non-finite rows and an empty table", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.25, NA, -3.35, NaN, Inf, 0.5))
  f67_check(df, function(d) mutate(d, a = round(x, 1), b = round(x, -1), c = round(x, 320)))
  f67_check(df[0, , drop = FALSE], function(d) mutate(d, a = round(x, 1)))
})
