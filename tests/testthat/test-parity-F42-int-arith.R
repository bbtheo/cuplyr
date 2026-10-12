# F42 acceptance tests, part 1: integer arithmetic overflow and zero divisors.
#
# Base R contract (?Arithmetic, verified with real dplyr 1.2.1 on tibbles):
# - `+`, `-`, `*` of integer/logical operands give integer. A result outside
#   +-.Machine$integer.max (2^31 - 1) is NA with the warning "NAs produced by
#   integer overflow" (dplyr wraps it in "There was 1 warning in `mutate()`").
#   -2^31 is R's NA_integer_, so `-M - 1L` is an overflow too.
# - The NA is an ordinary NA afterwards: `(x + 1L) * 0L` is NA, `is.na(x + 1L)`
#   is TRUE, a `case_when()` condition on it is NA (falls to .default).
# - Unary minus and abs() can't overflow (no integer is -2^31).
# - `%/%` and `%%` by 0L give NA without a warning.
# - `/` and `^` give double (no overflow question).
# - cumsum() of integers is integer; at the first overflow it warns
#   "integer overflow in 'cumsum'; use 'cumsum(as.numeric(.))'" and every
#   later value (of that group) is NA, even if the running sum comes back in
#   range. dplyr::cummean() accumulates in double and never overflows.
#
# Non-overflowing integer arithmetic must stay on the GPU, and so must the
# overflow detection itself: those tests run with cuplyr.fallback = "error".
# expect_equal() does not tell integer from double, so column classes are
# compared explicitly.

M <- .Machine$integer.max

f42_parity <- function(df, pipeline, arrange_by = NULL, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline, arrange_by = arrange_by))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
}

f42_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 3),
    x = c(M, 1L, -M, NA, 7L),
    y = c(1L, 2L, -1L, 3L, 0L),
    l = c(TRUE, FALSE, TRUE, NA, TRUE)
  )
}

gpu_only <- function() withr::local_options(cuplyr.fallback = "error", .local_envir = parent.frame())

test_that("binary +, -, * overflow to NA (column and literal operands)", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df(), function(d) {
    mutate(d,
      a = x + y, s = x - 2L, m = x * 2L, mm = x * y,
      a1 = x + 1L, a2 = 1L + x, s2 = -2L - x, m2 = 3L * x
    )
  })
})

test_that("integer + logical is integer and overflows the same way", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df(), function(d) mutate(d, a = x + l, b = l * x, c = l + l, d2 = x + l + l, e = -x - l - l))
})

test_that("unary minus and abs() never overflow; values at the boundary", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df(), function(d) {
    mutate(d,
      n = -x, b = abs(x), z = x + 0L, o = x * 1L, r = (x - 1L) + 1L,
      lo = -M - 1L + 0L * y, hi = M + y - y
    )
  })
})

test_that("an overflowed value is a real NA in the rest of the expression", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df(), function(d) {
    mutate(d,
      z0 = (x + 1L) * 0L,
      nn = -(x - 2L),
      na = is.na(x + 1L),
      cw = dplyr::case_when(x + y > 0L ~ 1L, .default = 0L),
      ie = dplyr::if_else(x * 2L > 0L, 1L, 0L),
      co = dplyr::coalesce(x * 2L, 0L),
      dd = (x + 1L) / 2L,
      cmp = x * 2L == 2L
    )
  })
})

test_that("nested overflow: -(-M - 1L) and chained sums", {
  skip_if_no_gpu()
  gpu_only()
  df <- data.frame(a = c(-M, 5L, 1L), b = c(1L, M, 2L))
  f42_parity(df, function(d) {
    mutate(d, n = -(a - 1L), c = a + b + b, k = (b * 2L) %/% 2L)
  })
})

test_that("%/% and %% by zero give NA", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df(), function(d) {
    mutate(d, q = x %/% y, r = x %% y, q0 = 7L %/% (y - y), r0 = 7L %% (y - y),
           qn = -5L %/% 2L, rn = -5L %% 2L)
  })
})

test_that("grouped mutate and transmute use the same checked arithmetic", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df(), function(d) d |> group_by(g) |> mutate(z = x + 1L, w = x * y))
  f42_parity(f42_df(), function(d) transmute(d, z = x + y, w = x - 2L))
})

test_that("zero-row integer arithmetic keeps integer types", {
  skip_if_no_gpu()
  gpu_only()
  f42_parity(f42_df()[0, ], function(d) mutate(d, a = x + y, m = x * 2L, q = x %/% y))
})

test_that("overflow warns once per call with R's message; no warning otherwise (eager)", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f42_df())
  expect_warning(collect(mutate(gt, a = x + 1L)), "NAs produced by integer overflow")
  expect_warning(collect(mutate(gt, m = x * y * 2L)), "NAs produced by integer overflow")
  expect_warning(collect(transmute(gt, s = x - 2L)), "NAs produced by integer overflow")
  # NA inputs, zero divisors and in-range results are not overflows
  expect_no_warning(collect(mutate(gt, a = x + 0L, q = x %/% y, r = x %% y, n = -x)))
  expect_no_warning(collect(mutate(gt, a = y * 2L - 1L)))
})

test_that("overflow warns at collect() in a lazy plan", {
  skip_if_no_gpu()
  lz <- as_lazy(tbl_gpu(f42_df()))
  q <- mutate(lz, a = x + 1L)
  expect_warning(collect(q), "NAs produced by integer overflow")
  expect_no_warning(collect(mutate(lz, a = y + 1L, q = x %/% y)))
})

test_that("filter() and summarise() see the NA, not a wrapped value", {
  skip_if_no_gpu()
  f42_parity(f42_df(), function(d) filter(d, x * 2L > 0L))
  f42_parity(f42_df(), function(d) filter(d, x + y < 5L))
  f42_parity(f42_df(), function(d) summarise(d, s = sum(x * 2L), t = sum(x + 1L, na.rm = TRUE)))
  f42_parity(f42_df(), function(d) d |> group_by(g) |> summarise(s = sum(x - 2L)),
             arrange_by = "g")
})

test_that("cumsum() of integers is NA from the first overflow on (sticky, per group)", {
  skip_if_no_gpu()
  gpu_only()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2), x = c(M, 1L, -5L, M, -1L, 1L))
  f42_parity(df, function(d) mutate(d, c = cumsum(x)))
  f42_parity(df, function(d) d |> group_by(g) |> mutate(c = cumsum(x)))
  f42_parity(df, function(d) d |> group_by(g) |> mutate(c = cumsum(x) - x))
  # NA still propagates; in-range sums stay integer
  df2 <- data.frame(x = c(1L, NA, 3L), l = c(TRUE, TRUE, NA))
  f42_parity(df2, function(d) mutate(d, c = cumsum(x), cl = cumsum(l)))
})

test_that("cumsum() overflow warns; cummean() of large integers does not overflow", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), x = c(M, 1L, M))
  expect_warning(collect(mutate(tbl_gpu(df), c = cumsum(x))), "integer overflow")
  expect_warning(collect(mutate(as_lazy(tbl_gpu(df)), c = cumsum(x))), "integer overflow")
  f42_parity(df, function(d) mutate(d, cm = dplyr::cummean(x)))
  f42_parity(df, function(d) d |> group_by(g) |> mutate(cm = dplyr::cummean(x)))
  expect_no_warning(collect(mutate(tbl_gpu(df), cm = dplyr::cummean(x))))
})

# Added after round 1 (review B1): a typed integer NA operand gives NA rows
# but is not an overflow.
test_that("a typed NA_integer_ operand gives NA without an overflow warning", {
  skip_if_no_gpu()
  d <- data.frame(x = 1:3)
  f42_parity(d, function(t) mutate(t, z = x + NA_integer_, w = x * NA_integer_))
  expect_no_warning(collect(mutate(tbl_gpu(d), z = x + NA_integer_)))
  expect_no_warning(collect(mutate(as_lazy(tbl_gpu(d)), z = x * NA_integer_)))
  expect_no_warning(collect(filter(tbl_gpu(d), x - NA_integer_ > 0L)))
})
