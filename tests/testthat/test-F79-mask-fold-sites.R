# F79: every place the parser evaluates a column-free piece of a dot with no
# data mask has to leave a call that reaches the mask by name to dplyr, or the
# value comes from the calling environment (or from nowhere) with no error.

f79f_same <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

f79f_df <- function() {
  tibble::tibble(
    x = 1:6,
    flag79 = rep(TRUE, 6L),
    s79 = rep("2024-01-03", 6L),
    w = as.Date("2024-01-01") + 0:5,
    t = c("ab", "cd", "ab", "cd", "ab", "cd")
  )
}

test_that("a window call that folds leaves a name-reaching argument to dplyr", {
  skip_if_no_gpu()
  df <- f79f_df()
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & dplyr::first(exists("flag79"))))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & dplyr::cumall(exists("flag79"))))
  f79f_same(df, function(d) dplyr::mutate(d, z = x > 1L & dplyr::last(exists("flag79"))))
})

test_that("a date conversion that folds leaves a name-reaching argument to dplyr", {
  skip_if_no_gpu()
  df <- f79f_df()
  s79 <- "2030-01-01"
  f79f_same(df, function(d) dplyr::filter(d, w > as.Date(get("s79"))))
  f79f_same(df, function(d) dplyr::mutate(d, z = w > as.Date(get("s79"))))
})

test_that("a parameter argument evaluated as a constant leaves a name-reaching call to dplyr", {
  skip_if_no_gpu()
  df <- f79f_df()
  f79f_same(df, function(d) dplyr::mutate(d, y = dplyr::lag(x, n = exists("flag79") + 1L)))
  f79f_same(df, function(d) dplyr::filter(d, grepl(if (exists("flag79")) "a" else "z", t)))
  f79f_same(df, function(d) dplyr::mutate(d, u = gsub("a", if (exists("flag79")) "Q" else "R", t, fixed = TRUE)))
  na_df <- tibble::tibble(x = c(1L, NA, 3L), flag79 = TRUE)
  f79f_same(na_df, function(d) dplyr::summarise(d, z = sum(x, na.rm = exists("flag79"))))
})

test_that("order_by() and with_order() leave a name-reaching argument to dplyr", {
  skip_if_no_gpu()
  df <- f79f_df()
  f79f_same(df, function(d) dplyr::mutate(d, y = dplyr::order_by(x, dplyr::lag(x, exists("flag79") + 1L))))
  f79f_same(df, function(d) dplyr::mutate(d, y = dplyr::with_order(x, dplyr::lag, x, exists("flag79") + 1L)))
})

test_that("the generic fold's other callers leave a name-reaching call to dplyr", {
  skip_if_no_gpu()
  df <- f79f_df()
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & identity(exists("flag79"))))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & base::exists("flag79")))
  expect_same_as_dplyr(df, function(d) dplyr::count(d, z = x > 1L & exists("flag79")), arrange_by = "z")
  f79f_same(df, function(d) dplyr::ungroup(dplyr::group_by(d, z = x > 1L & exists("flag79"))))
  f79f_same(df, function(d) dplyr::distinct(d, z = x > 1L & exists("flag79")))
  f79f_same(df, function(d) dplyr::slice_max(d, order_by = x * exists("flag79"), n = 1L))
  f79f_same(df, function(d) dplyr::summarise(d, z = sum(x * exists("flag79"))))
})

test_that("top_n() leaves an `n` that reaches the mask by name to dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3, 4), flag79 = TRUE)
  f79f_same(df, function(d) dplyr::top_n(d, if (exists("flag79")) 1L else 3L, x))
  f79f_same(df, function(d) dplyr::top_frac(d, if (exists("flag79")) 0.25 else 0.75, x))
})

test_that("with() reaches the mask only through plain symbols and stays native", {
  skip_if_no_gpu()
  df <- f79f_df()
  cfg <- list(thr = 3L)
  fallbacks <- function(run) {
    k <- 0L
    withCallingHandlers(collect(run()), cuplyr_cpu_fallback = function(cnd) k <<- k + 1L)
    k
  }
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_identical(fallbacks(function() dplyr::filter(gt, x > with(cfg, thr))), 0L)
    expect_identical(fallbacks(function() dplyr::filter(gt, x > 1L, with(list(flag = TRUE), flag))), 0L)
  }
  f79f_same(df, function(d) dplyr::filter(d, x > with(cfg, thr)))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L, with(list(flag = TRUE), flag)))
  # the body names a column: dplyr's
  f79f_same(df, function(d) dplyr::filter(d, x > with(cfg, x - 1L)))
  # the body reaches the mask by name: found by the walk
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & with(cfg, exists("flag79"))))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L, with(cfg, exists("flag79"))))
})

test_that("the mask walk counts calls, however the head is spelled", {
  expect_true(expr_reaches_mask(quote(exists("a"))))
  expect_true(expr_reaches_mask(quote(base::exists("a"))))
  expect_true(expr_reaches_mask(quote(base:::exists("a"))))
  expect_true(expr_reaches_mask(quote(1L + base:::get("a"))))
  expect_true(expr_reaches_mask(quote(rlang:::exec("exists", "a"))))
  expect_true(expr_reaches_mask(quote(identical(rlang::current_env(), 1L))))
  expect_true(expr_reaches_mask(quote(identical(rlang::caller_env(0), 1L))))
  expect_false(expr_reaches_mask(quote(base:::paste("a"))))
  # a plain symbol named like a listed function is no call of it
  expect_false(expr_reaches_mask(quote(exists)))
  expect_false(expr_reaches_mask(quote(exists > 1L & get < 2L)))
  expect_false(expr_reaches_mask(quote(base:::exists)))
  expect_false(expr_reaches_mask(quote(l$exists("a"))))
  # an embedded quosure is walked, at the top and inside a call
  q <- rlang::quo(exists("a"))
  expect_true(expr_reaches_mask(q))
  expect_true(expr_reaches_mask(rlang::expr(1L + !!q)))
  expect_false(expr_reaches_mask(rlang::expr(1L + !!rlang::quo(a))))
  # not a call
  expect_false(expr_reaches_mask(1L))
  expect_false(expr_reaches_mask(quote(expr = )))
})

test_that("a column named like a listed function is an ordinary column", {
  skip_if_no_gpu()
  df <- tibble::tibble(exists = 1:4, get = 4:1)
  for (lazy in c(FALSE, TRUE)) {
    k <- 0L
    withCallingHandlers(
      collect(dplyr::filter(tbl_gpu(df, lazy = lazy), exists > 1L & get > 1L)),
      cuplyr_cpu_fallback = function(cnd) k <<- k + 1L
    )
    expect_identical(k, 0L)
  }
  f79f_same(df, function(d) dplyr::filter(d, exists > 1L & get > 1L))
})

test_that("parse() and friends only build language and stay native", {
  skip_if_no_gpu()
  df <- f79f_df()
  fallbacks <- function(run) {
    k <- 0L
    withCallingHandlers(collect(run()), cuplyr_cpu_fallback = function(cnd) k <<- k + 1L)
    k
  }
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_identical(fallbacks(function() dplyr::filter(gt, x > 1L & length(parse(text = "flag79")) == 1L)), 0L)
    expect_identical(fallbacks(function() dplyr::filter(gt, x > 1L & is.symbol(str2lang("flag79")))), 0L)
    expect_identical(fallbacks(function() dplyr::filter(gt, x > 1L, length(str2expression("flag79")) == 1L)), 0L)
  }
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & length(parse(text = "flag79")) == 1L))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & is.symbol(str2lang("flag79"))))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L, length(str2expression("flag79")) == 1L))
  # evaluating what they build is still dplyr's
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & eval(parse(text = "flag79"))))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & eval(str2lang("flag79"))))
})

test_that("a function match.fun() returns can read the mask when called", {
  skip_if_no_gpu()
  df <- f79f_df()
  f79f_same(df, function(d) dplyr::filter(d, x > 1L & match.fun("exists")("flag79")))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L, match.fun("get")("flag79")))
  f79f_same(df, function(d) dplyr::filter(d, match.fun("exists")("flag79")))
  f79f_same(df, function(d) dplyr::filter(d, x > 1L, match.fun("exists")("flag79")))
})
