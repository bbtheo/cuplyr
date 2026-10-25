# F67 round 3: places that deduplicate by value must tell -0 from 0
# (identical() calls them equal), and R's sum(), mean() and cumsum() never
# return -0, through every verb that computes them on the GPU.

for (fn in c("cummean", "order_by", "last")) {
  assign(fn, getExportedValue("dplyr", fn))
}

test_that("a window argument in summarise() keeps 0 and -0 apart", {
  skip_if_no_gpu()
  z <- -0
  df <- data.frame(x = c(1, 2, 3), g = c("a", "a", "b"))
  f67_check(df, function(d) summarise(d, a = sum(cumsum(x / z)), b = sum(cumsum(x / 0))))
  f67_check(df, function(d) summarise(d, b = last(cumsum(x / 0)), a = last(cumsum(x / z))))
  f67_check(df, function(d) {
    d |> group_by(g) |> summarise(a = sum(cumsum(x / 0)), b = sum(cumsum(x / z)))
  })
})

test_that("reframe() keeps aggregates that differ only in a zero's sign apart", {
  skip_if_no_gpu()
  z <- -0
  df <- data.frame(x = c(1, 2, 3), g = c("a", "a", "b"))
  # (a column-free value from the environment sends reframe() to the CPU, so
  # the -0 is spliced in)
  f67_check(df, function(d) {
    d |> group_by(g) |> reframe(a = quantile(x / !!z, c(0.25, 0.75), names = FALSE), b = quantile(x / 0, c(0.25, 0.75), names = FALSE))
  })
  f67_check(df, function(d) {
    d |> group_by(g) |> reframe(b = quantile(x / 0, c(0.25, 0.75), names = FALSE), a = quantile(x / !!z, c(0.25, 0.75), names = FALSE))
  })
})

test_that("the colwise memo keeps extra arguments 0 and -0 apart", {
  # `<function />`(v, <extra>) is rewritten to the plain call `v / <extra>`
  schema <- list(names = "v", types = "FLOAT64")
  dot <- function(extra) rlang::new_quosure(as.call(list(`/`, as.name("v"), extra)), globalenv())
  for (dots in list(colwise_rewrite_dots(list(a = dot(0), b = dot(-0)), schema),
                    colwise_rewrite_dots(list(b = dot(-0), a = dot(0)), schema))) {
    expect_false(colwise_has_fn_head(rlang::quo_get_expr(dots$a)))
    expect_false(colwise_has_fn_head(rlang::quo_get_expr(dots$b)))
    expect_identical(rlang::eval_tidy(dots$a, list(v = 1)), Inf)
    expect_identical(rlang::eval_tidy(dots$b, list(v = 1)), -Inf)
  }
})

test_that("sum(), mean() and cumsum() of -0 are +0 in ungrouped mutate() and in filter()", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0, 0, 0, 0), y = c(-0, -0, -0, NA), g = c("a", "a", "b", "b"))
  f67_check(df, function(d) {
    mutate(d, s = sum(-x), m = mean(-x), sy = sum(y, na.rm = TRUE), c = cumsum(y), cm = cummean(y))
  })
  f67_check(df, function(d) mutate(d, c = order_by(g, cumsum(y))))
  f67_check(df, function(d) d |> group_by(g) |> filter(1 / sum(-x) > 0))
  f67_check(df, function(d) d |> group_by(g) |> mutate(c = cumsum(y), m = mean(y), s = sum(y)))
})

test_that("rowwise sum() and mean() of -0 operands are +0", {
  skip_if_no_gpu()
  df <- data.frame(a = c(-0, -0, 1), b = c(-0, NA, -1))
  f67_check(df, function(d) {
    d |> rowwise() |> mutate(s = sum(c(a, b)), m = mean(c(a, b)), sr = sum(c(a, b), na.rm = TRUE),
                             mr = mean(c(a, b), na.rm = TRUE))
  })
})

test_that("cumsum() of a -0 prefix is +0, and later values keep their signs", {
  skip_if_no_gpu()
  # A prefix sum can only be -0 while every value so far is -0; past that, R's
  # and the GPU's running sums are the same numbers.
  df <- data.frame(x = c(-0, -0, -2, 2, -0, NaN, 1), g = c(1L, 1L, 1L, 1L, 2L, 2L, 2L))
  f67_check(df, function(d) mutate(d, c = cumsum(x), m = cummean(x)))
  f67_check(df, function(d) d |> group_by(g) |> mutate(c = cumsum(x), s = sum(x), m = mean(x)))
  f67_check(df, function(d) d |> group_by(g) |> summarise(s = sum(x), m = mean(x)))
})
