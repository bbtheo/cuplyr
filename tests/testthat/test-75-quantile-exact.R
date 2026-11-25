# #75: a grouped quantile() is R's type 7 to the bit. The interpolation
# (1 - h) * x[lo] + h * x[hi] is written out from cudf's LOWER and HIGHER
# elements, each step its own operation, instead of cudf's LINEAR (a few ulps off).

test_that("quantile() over groups equals R's own, bit for bit", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(75)
  n <- 4000
  df <- data.frame(
    k = sample.int(30L, n, TRUE),
    x = c(round(rnorm(n - 2) * 10, 1), 1e300, -1e300),  # ties and far neighbours
    i = sample.int(1000L, n, TRUE)
  )
  probs <- unique(c(0, 1, 1 / 3, 2 / 3, 0.1, 0.7, 0.999, seq(0, 1, by = 0.05)))
  g <- group_by(tbl_gpu(df, lazy = FALSE), k)
  got_x <- collect(reframe(g, q = quantile(x, probs)))
  got_i <- collect(reframe(g, q = quantile(i, probs)))
  by_k <- dplyr::group_by(df, k)
  expect_identical(unname(got_x$q), dplyr::reframe(by_k, q = unname(quantile(x, probs)))$q)
  expect_identical(unname(got_i$q), dplyr::reframe(by_k, q = unname(quantile(i, probs)))$q)
})

test_that("a median and quantiles of one column share their request", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(7501)
  df <- data.frame(k = sample.int(5L, 200, TRUE), x = rnorm(200))
  got <- collect(summarise(group_by(tbl_gpu(df, lazy = FALSE), k),
                           m = median(x), a = quantile(x, 0.5), b = quantile(x, 0.25), c = quantile(x, 0.5)))
  want <- dplyr::summarise(dplyr::group_by(df, k), m = median(x), a = unname(quantile(x, 0.5)),
                           b = unname(quantile(x, 0.25)), c = unname(quantile(x, 0.5)))
  strip <- function(d) unname(lapply(as.list(d), unname))
  expect_equal(strip(got), strip(want))
  expect_identical(unname(got$a), unname(got$c))
})

# the same, to the bit, in every shape a quantile() runs in (eager and lazy,
# grouped or not, na.rm, the integer and infinite extremes)
p75q_bits <- function(df, probs, col = "x", na_rm = FALSE, by = NULL, lazy = FALSE) {
  g <- tbl_gpu(df, lazy = lazy)
  if (!is.null(by)) g <- group_by(g, !!rlang::sym(by))
  got <- collect(rlang::inject(
    reframe(g, q = quantile(!!rlang::sym(col), !!probs, na.rm = !!na_rm))
  ))
  d <- if (is.null(by)) df else dplyr::group_by(df, !!rlang::sym(by))
  want <- rlang::inject(
    dplyr::reframe(d, q = unname(quantile(!!rlang::sym(col), !!probs, na.rm = !!na_rm)))
  )
  expect_identical(unname(got$q), want$q)
}

test_that("quantile() is R's type 7 to the bit, ungrouped and lazy", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(7511)
  df <- data.frame(k = rep(1:7, length.out = 300), x = round(rnorm(300), 2), i = sample.int(50L, 300, TRUE))
  probs <- c(0, 1 / 3, 0.5, 0.9, 0.999, 1)
  for (lazy in c(FALSE, TRUE)) {
    p75q_bits(df, probs, lazy = lazy)
    p75q_bits(df, probs, col = "i", lazy = lazy)
    p75q_bits(df, probs, by = "k", lazy = lazy)
    p75q_bits(df, probs, col = "i", by = "k", lazy = lazy)
  }
})

test_that("quantile() with na.rm and an all-NA group is R's, to the bit", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  df <- data.frame(k = rep(1:4, each = 5),
                   x = c(1, 2, NA, 4, 5,  NA, NA, NA, NA, NA,  3, 3, 3, NA, 9,  0.1, 0.2, 0.3, 0.4, NA))
  probs <- c(0, 0.25, 0.5, 0.8, 1)
  for (lazy in c(FALSE, TRUE)) {
    p75q_bits(df, probs, by = "k", na_rm = TRUE, lazy = lazy)
    p75q_bits(df, probs, na_rm = TRUE, lazy = lazy)
    got <- collect(summarise(group_by(tbl_gpu(df, lazy = lazy), k), q = quantile(x, 0.5, na.rm = TRUE)))
    want <- dplyr::summarise(dplyr::group_by(df, k), q = unname(quantile(x, 0.5, na.rm = TRUE)))
    expect_identical(unname(got$q), want$q)
  }
})

test_that("quantile() at the integer and infinite extremes is R's, to the bit", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  ints <- data.frame(k = rep(1:2, each = 4),
                     i = c(.Machine$integer.max, -.Machine$integer.max, 0L, 1L,
                           .Machine$integer.max, .Machine$integer.max, -1L, 7L))
  infs <- data.frame(k = rep(1:3, each = 4),
                     x = c(-Inf, 1, 2, Inf,  Inf, Inf, 1, 2,  -Inf, -Inf, 0, 5))
  probs <- c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
  for (lazy in c(FALSE, TRUE)) {
    p75q_bits(ints, probs, col = "i", by = "k", lazy = lazy)
    p75q_bits(ints, probs, col = "i", lazy = lazy)
    p75q_bits(infs, probs, by = "k", lazy = lazy)
  }
})

test_that("median() and quantile() calls of one column, separate or wide, are R's", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(7512)
  df <- data.frame(k = sample.int(6L, 120, TRUE), x = rnorm(120), y = rnorm(120))
  probs <- seq(0, 1, by = 0.025)
  wide <- stats::setNames(lapply(probs, function(p) rlang::expr(quantile(x, !!p))), paste0("q", seq_along(probs)))
  strip <- function(d) unname(lapply(as.list(d), unname))
  for (lazy in c(FALSE, TRUE)) {
    g <- group_by(tbl_gpu(df, lazy = lazy), k)
    got <- collect(summarise(g, m = median(x), !!!wide, my = median(y), qy = quantile(y, 0.3)))
    want <- dplyr::summarise(dplyr::group_by(df, k), m = median(x),
                             !!!lapply(wide, function(e) rlang::expr(unname(!!e))),
                             my = median(y), qy = unname(quantile(y, 0.3)))
    expect_identical(strip(got), strip(want))
    # ungrouped, a single row
    got <- collect(summarise(tbl_gpu(df, lazy = lazy), m = median(x), a = quantile(x, 0.1), b = quantile(x, 0.9)))
    want <- dplyr::summarise(df, m = median(x), a = unname(quantile(x, 0.1)), b = unname(quantile(x, 0.9)))
    expect_identical(strip(got), strip(want))
  }
})

test_that("a grouped quantile() over more (group, probability) pairs than rows is R's, to the bit", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  set.seed(7513)
  # the values are computed in chunks of groups: more pairs than one chunk holds
  n <- 20000
  df <- data.frame(k = sample.int(5000L, n, TRUE), x = rnorm(n))
  p <- seq(0, 1, by = 0.005)
  for (lazy in c(FALSE, TRUE)) p75q_bits(df, p, by = "k", lazy = lazy)
})

test_that("quantile() names follow options(OutDec) at every call", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE, cuplyr.fallback = "error")
  d <- data.frame(x = 1:4, y = c("a", "a", "b", "b"))
  labels <- function(p) {
    mine <- names(collect(summarise(group_by(tbl_gpu(d, lazy = FALSE), y), b = quantile(x, !!p)))$b)
    lazy <- names(collect(summarise(group_by(tbl_gpu(d, lazy = TRUE), y), b = quantile(x, !!p)))$b)
    theirs <- names(dplyr::summarise(dplyr::group_by(d, y), b = quantile(x, !!p))$b)
    expect_identical(mine, theirs)
    expect_identical(lazy, theirs)
    mine
  }
  expect_identical(labels(0.125), c("12.5%", "12.5%"))
  withr::with_options(list(OutDec = ","), {
    expect_identical(labels(0.125), c("12,5%", "12,5%"))
    expect_identical(labels(0.3333), c("33,33%", "33,33%"))
  })
  # back to "." the labels asked for under "," are not kept
  expect_identical(labels(0.125), c("12.5%", "12.5%"))
  expect_identical(labels(0.3333), c("33.33%", "33.33%"))
  # `digits` is part of the label too
  expect_identical(
    names(collect(summarise(group_by(tbl_gpu(d, lazy = FALSE), y), b = quantile(x, 1 / 3, digits = 3)))$b),
    names(dplyr::summarise(dplyr::group_by(d, y), b = quantile(x, 1 / 3, digits = 3))$b)
  )
})

test_that("the label worked out for a probability is quantile()'s own", {
  probs <- c(0, 1, 0.5, 0.125, 1 / 3, 2 / 3, 0.001, 0.999, 0.12345678, 1e-9, 0.1 + 0.2, 0.95, 0.975)
  for (dec in c(".", ",")) {
    withr::local_options(OutDec = dec)
    for (digits in c(1, 2, 3, 7, 10, 15, 17)) {
      for (p in probs) {
        expr <- rlang::expr(quantile(x, !!p, digits = !!digits))
        got <- extract_quantile_probs(expr, globalenv(), rlang::new_quosure(expr))$qnames
        expect_identical(got, names(stats::quantile(0, p, digits = digits)))
      }
    }
  }
})
