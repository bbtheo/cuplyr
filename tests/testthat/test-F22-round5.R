# F22 round 5: R's type-7 quantile index for every column, not only those with
# Inf (INT32 columns, a far finite neighbour, median of a float), and the
# run-now notice of a lazy reframe() naming only the integer sum() outputs.

f22r5_check <- function(df, pipeline) {
  expected <- pipeline(tibble::as_tibble(df))
  expected[] <- lapply(expected, unname)
  for (lazy in c(FALSE, TRUE)) {
    got <- withr::with_options(
      list(cuplyr.fallback = "error"),
      collect(pipeline(tbl_gpu(df, lazy = lazy)))
    )
    got <- tibble::as_tibble(got)
    got[] <- lapply(got, unname)
    expect_identical(got, expected, info = if (lazy) "lazy" else "eager")
  }
}

test_that("a whole type-7 index takes the element next to an INT32 far neighbour", {
  skip_if_no_gpu()
  p <- seq(0, 1, 0.1)
  df <- data.frame(g = 1L, x = c(1:8, 1000000000L, 2000000000L, 2147483647L))
  f22r5_check(df, function(t) reframe(t, q = quantile(x, p), .by = g))
  f22r5_check(df, function(t) summarise(t, q = quantile(x, p[8])))
  df2 <- data.frame(g = c(1L, 1L, 2L), x = c(5L, 7L, 9L))
  f22r5_check(df2, function(t) reframe(t, q = quantile(x, c(0, 0.5, 1)), .by = g))
})

test_that("a whole type-7 index takes the element next to a far finite double", {
  skip_if_no_gpu()
  p <- seq(0, 1, 0.1)
  df <- data.frame(g = 1, x = c(0:7, 1e300, 1e300, 1e300))
  f22r5_check(df, function(t) reframe(t, q = quantile(x, p), .by = g))
  f22r5_check(df, function(t) summarise(t, q = quantile(x, p[8])))
  f22r5_check(df, function(t) summarise(t, m = median(x)))
})

test_that("the lazy integer sum() notice names only the sum outputs", {
  skip_if_no_gpu()
  lz <- tbl_gpu(data.frame(g = 1:2, i = c(2e9L, 2e9L), d = c(1, 2)), lazy = TRUE)
  lz <- mutate(lz, k = i - 1L)
  # F88: running the plan early on the GPU is a plan break, not a CPU fallback
  msg <- withr::with_options(list(cuplyr.fallback = "warn", cuplyr.plan_break = "warn"), {
    tryCatch(
      reframe(lz, s = sum(i), m = mean(d), q = quantile(d, c(.1, .9)), .by = g),
      warning = function(w) conditionMessage(w)
    )
  })
  expect_type(msg, "character")
  expect_match(msg, "integer sum()", fixed = TRUE)
  expect_match(msg, ": s", fixed = TRUE)
  expect_no_match(msg, "reframe_")
  expect_no_match(msg, "m,")
  expect_no_match(msg, "q")
})
