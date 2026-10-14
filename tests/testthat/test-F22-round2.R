# F22 round 2: summarise()'s quantile() takes R's type-7 values at +-Inf, the
# quantiles of one column share a groupby request, aggregations of the same
# expression share a temp column, and the reframe() plan only rewrites the real
# stats::quantile / base::range over arguments that read a column.

f22r2_summ <- function(d, ...) {
  dplyr::summarise(d, ..., .groups = "drop")
}

test_that("summarise() quantile() gives R's value where the position is whole and a neighbour is infinite", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c(1, 1, 1, 2, 2, 3, 3, 3, 4),
    x = c(1, 2, Inf, -Inf, 3, Inf, Inf, 5, -Inf)
  )
  q <- function(d) {
    f22r2_summ(
      dplyr::group_by(d, g),
      q0 = stats::quantile(x, 0, names = FALSE),
      q25 = stats::quantile(x, 0.25, names = FALSE),
      q5 = stats::quantile(x, 0.5, names = FALSE),
      q1 = stats::quantile(x, 1, names = FALSE)
    )
  }
  expect_equal(
    tibble::as_tibble(collect(q(tbl_gpu(df)))),
    q(tibble::as_tibble(df))
  )
  # ungrouped, and the lazy plan
  ung <- function(d) {
    dplyr::summarise(d, a = stats::quantile(x, 0, names = FALSE),
                     b = stats::quantile(x, 0.5, names = FALSE),
                     c = stats::quantile(x, 1, names = FALSE))
  }
  expect_equal(tibble::as_tibble(collect(ung(tbl_gpu(df, lazy = TRUE)))),
               ung(tibble::as_tibble(df)))
})

test_that("several quantiles of one column, mixed with other aggregations and NAs", {
  skip_if_no_gpu()
  set.seed(5)
  df <- data.frame(g = sample(1:4, 60, TRUE), x = round(runif(60), 3), y = rnorm(60))
  df$x[c(3, 11, 40)] <- NA
  df$g[df$g == 4] <- 5
  df$x[df$g == 5] <- NA   # an all-NA group
  f <- function(d) {
    f22r2_summ(
      dplyr::group_by(d, g),
      a = stats::quantile(x, 0.9, na.rm = TRUE, names = FALSE),
      m = mean(y),
      b = stats::quantile(y, 0.1, names = FALSE),
      c = stats::quantile(x, 0.1, na.rm = TRUE, names = FALSE),
      q9 = stats::quantile(y, 0.9, names = FALSE),
      e = stats::median(x, na.rm = TRUE)
    )
  }
  expect_equal(tibble::as_tibble(collect(f(tbl_gpu(df)))), f(tibble::as_tibble(df)))
  expect_error(
    collect(dplyr::summarise(tbl_gpu(df), a = stats::quantile(x, 0.5), b = stats::quantile(x, 0.1))),
    "missing values"
  )
})

test_that("aggregations of the same expression share a temp column and keep their meaning", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), x = c(1, 2, 3, 5, 8), b = c(TRUE, FALSE, TRUE, TRUE, NA))
  f <- function(d) {
    f22r2_summ(
      dplyr::group_by(d, g),
      a = mean(x * 2), s = sd(x * 2), mx = max(x * 2), other = sum(x * 3),
      anyb = any(x > 2), sumb = sum(x > 2), allb = all(x > 2)
    )
  }
  expect_equal(tibble::as_tibble(collect(f(tbl_gpu(df)))), f(tibble::as_tibble(df)))
  expect_equal(tibble::as_tibble(collect(f(tbl_gpu(df, lazy = TRUE)))), f(tibble::as_tibble(df)))
})

test_that("a computed argument with many probabilities needs one temp column", {
  skip_if_no_gpu()
  set.seed(1)
  n <- 4e5
  df <- data.frame(g = sample(2000, n, TRUE), x = runif(n))
  probs <- seq(0, 1, 0.01)
  expected <- dplyr::reframe(dplyr::group_by(tibble::as_tibble(df), g),
                             q = unname(stats::quantile(x * 2, probs)))
  got <- withr::with_options(
    list(cuplyr.fallback = "error"),
    collect(reframe(group_by(tbl_gpu(df), g), q = quantile(x * 2, probs)))
  )
  expect_equal(tibble::as_tibble(got), expected)
})

test_that("the reframe plan only takes the real quantile()/range() over a column argument", {
  skip_if_no_gpu()
  schema <- list(names = c("g", "x"), types = c("FLOAT64", "FLOAT64"))
  plan <- function(...) cuplyr:::reframe_native_plan(rlang::enquos(...), schema)

  expect_false(is.null(plan(q = quantile(x, c(0.1, 0.9)))))
  expect_false(is.null(plan(r = base::range(x))))

  v <- c(1, 5, 9, 2)
  expect_null(plan(q = quantile(v, c(0.1, 0.9))))
  expect_null(plan(r = range(v)))
  expect_null(plan(q = quantile(x, c(0.1, 0.9)), m = mean(v)))

  local({
    range <- function(x, ...) max(x) - min(x)
    expect_null(plan(r = range(x)))
    expect_false(is.null(plan(r = base::range(x))))
  })
  local({
    quantile <- function(x, ...) 1
    expect_null(plan(q = quantile(x, c(0.1, 0.9))))
  })

  # a masked c() is not a constant the plan may evaluate once
  local({
    c <- function(...) 1:2
    expect_null(cuplyr:::reframe_constant_value(quote(c(0.1, 0.2)), environment(), "x"))
  })
  expect_equal(cuplyr:::reframe_constant_value(quote(c(0.1, 0.2)), environment(), "x"), c(0.1, 0.2))

  # an unclassed matrix/array is not a plain vector literal
  m <- matrix(1:2, 1)
  a <- array(1:2, 2)
  expect_null(cuplyr:::reframe_constant_value(quote(m), environment(), "x"))
  expect_null(cuplyr:::reframe_constant_value(quote(a), environment(), "x"))
  expect_equal(cuplyr:::reframe_constant_value(quote(1:2), environment(), "x"), 1:2)
})
