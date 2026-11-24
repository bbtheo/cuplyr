# Acceptance tests for #21, added after the first review: names of constants
# read by later dots, group keys, `y` evaluated once, attributes R checks
# against the length (`tsp`), reframe() and `.drop = FALSE` keys. dplyr 1.2.1
# on plain tibbles is the oracle.

p21r2_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("later dots see the names of a named constant", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3)
  p21r2_both(df, function(x) mutate(x, v = c(p = 1), w = v + 1, u = v))
  p21r2_both(df, function(x) mutate(x, s = stats::setNames(1:3, c("q", "r", "t")), s2 = s * 2L))
  p21r2_both(df, function(x) mutate(x, s = stats::setNames(1:3, c("q", "r", "t")), s2 = s, a2 = a + 1L))
  p21r2_both(df, function(x) transmute(x, v = c(p = 1), u = v))
})

test_that("group_keys() and group_data() keep the keys' attributes and names", {
  skip_if_no_gpu()
  lab <- tibble::tibble(
    s = structure(c("a", "b", "a"), label = "S"),
    b = stats::setNames(c(1L, 2L, 1L), c("x", "y", "z")),
    v = 1:3
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(lab, lazy = lazy)
    info <- paste("lazy =", lazy)
    expect_equal(group_keys(group_by(gt, s)), group_keys(group_by(lab, s)), info = info)
    expect_equal(group_keys(group_by(gt, b)), group_keys(group_by(lab, b)), info = info)
    expect_equal(group_keys(group_by(gt, s, b)), group_keys(group_by(lab, s, b)), info = info)
    expect_equal(group_data(group_by(gt, s))$s, group_data(group_by(lab, s))$s, info = info)
    expect_equal(group_data(group_by(gt, b))$b, group_data(group_by(lab, b))$b, info = info)
  }
})

test_that("set operations and rows_*() evaluate y once", {
  skip_if_no_gpu()
  x <- tibble::tibble(a = 1:3, b = stats::setNames(1:3, c("p", "q", "r")))
  n <- 0L
  yf <- function() {
    n <<- n + 1L
    x[2:3, ]
  }
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(x, lazy = lazy)
    n <- 0L
    out <- collect(union(gt, yf()))
    expect_identical(n, 1L)
    expect_equal(out, union(x, x[2:3, ]))
    n <- 0L
    out <- collect(rows_append(gt, yf()))
    expect_identical(n, 1L)
    expect_equal(out, rows_append(x, x[2:3, ]))
    n <- 0L
    out <- collect(intersect(gt, yf()))
    expect_identical(n, 1L)
    expect_equal(out, intersect(x, x[2:3, ]))
  }
})

test_that("an attribute R checks against the length never breaks collect()", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 4, 9), y = 1:3)
  p21r2_both(df, function(t) mutate(t, across(x, stats::lag)))
  p21r2_both(df, function(t) filter(mutate(t, across(x, stats::lag)), y > 1L))
  p21r2_both(df, function(t) arrange(mutate(t, across(x, stats::lag)), desc(y)))
  p21r2_both(df, function(t) filter(mutate(t, across(x, stats::lag)), y > 5L))
})

test_that("reframe() and .drop = FALSE keys keep their attributes", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = structure(c(1L, 1L, 2L), label = "G"),
    f = structure(factor(c("u", "u", "v"), levels = c("u", "v", "w")), label = "F"),
    x = c(1, 2, 3)
  )
  p21r2_both(df, function(t) reframe(t, m = range(x), .by = g))
  p21r2_both(df, function(t) reframe(group_by(t, g), m = mean(x)))
  p21r2_both(df, function(t) summarise(group_by(t, f, .drop = FALSE), n = n()))
  p21r2_both(df, function(t) count(t, f, .drop = FALSE))
})

test_that("set operations and rows_*() follow dplyr on constant attributes", {
  skip_if_no_gpu()
  x <- tibble::tibble(a = 1:3, s = structure(c("p", "q", "r"), label = "S"))
  y <- x[2:3, ]
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(x, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    info <- paste("lazy =", lazy)
    expect_equal(collect(arrange(union(gt, gy), a)), arrange(union(x, y), a), info = info)
    expect_equal(collect(arrange(union_all(gt, gy), a)), arrange(union_all(x, y), a), info = info)
    expect_equal(collect(intersect(gt, gy)), intersect(x, y), info = info)
    expect_equal(collect(setdiff(gt, gy)), setdiff(x, y), info = info)
    expect_equal(collect(rows_append(gt, gy)), rows_append(x, y), info = info)
    expect_equal(collect(rows_update(gt, gy, by = "a")), rows_update(x, y, by = "a"), info = info)
  }
})

test_that("a verb called through a wrapper keeps the names", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), b = stats::setNames(c(3L, 1L, 2L), c("x", "y", "z")))
  f <- function(d, ...) filter(d, ...)
  a <- function(d, ...) arrange(d, ...)
  p21r2_both(df, function(t) f(t, b > 1L))
  p21r2_both(df, function(t) a(t, b))
})
