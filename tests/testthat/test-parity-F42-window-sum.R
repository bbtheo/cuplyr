# F42 acceptance tests, part 2: sum() of integer/logical inside mutate()
# (the window/broadcast path), matching F34's summarise() rule.
#
# dplyr 1.2.1 contract (verified with real dplyr on tibbles):
# - mutate(s = sum(<int or lgl>)) is integer, ungrouped and grouped, with
#   na.rm, for a comparison (sum(y > 0L)), and for 0 rows. NA without na.rm
#   gives NA_integer_.
# - When a sum leaves +-.Machine$integer.max, base R's sum() returns a double
#   (no warning); grouped, one overflowing group makes the whole column double
#   (vctrs combine), and an expression on the sum follows (sum(x) + 1L is
#   double). Sums of exactly M or -M stay integer.
# - A lazy table runs a mutate() with an INT32 sum at the verb (F80), as an
#   eager one does, so the type comes from the data. Only a BOOL8 sum stays a
#   pending step with a static integer schema.
# - filter() on an overflowing sum compares the double value; there is no
#   schema to drift, so lazy filter() must keep matching dplyr.

M <- .Machine$integer.max

f42w_parity <- function(df, pipeline, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
}

f42w_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 3),
    y = c(1L, -1L, 2L, 3L, 10L),
    m = c(4L, NA, 2L, 3L, NA),
    l = c(TRUE, NA, FALSE, TRUE, TRUE)
  )
}

f42w_big <- function() {
  data.frame(g = c(1, 1, 2, 2), v = c(M, 5L, 1L, 2L))
}

test_that("ungrouped window sum() of int/logical is integer", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  f42w_parity(f42w_df(), function(d) {
    mutate(d, s = sum(y), sm = sum(m), smr = sum(m, na.rm = TRUE),
           sl = sum(l, na.rm = TRUE), sc = sum(y > 0L))
  })
  f42w_parity(f42w_df(), function(d) mutate(d, z = y - sum(y), w = sum(y) * 2L))
})

test_that("grouped window sum() of int/logical is integer", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  f42w_parity(f42w_df(), function(d) {
    d |> group_by(g) |>
      mutate(s = sum(y), sm = sum(m), smr = sum(m, na.rm = TRUE),
             sl = sum(l, na.rm = TRUE), z = y - sum(y))
  })
  f42w_parity(f42w_df(), function(d) mutate(d, s = sum(y), .by = g))
})

test_that("zero-row window sum() is integer", {
  skip_if_no_gpu()
  f42w_parity(f42w_df()[0, ], function(d) mutate(d, s = sum(y), sl = sum(l)))
  f42w_parity(f42w_df()[0, ], function(d) d |> group_by(g) |> mutate(s = sum(y)))
})

test_that("sums of exactly M and -M stay integer", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(g = c(1, 1, 2, 2), v = c(M - 5L, 5L, -M + 1L, -1L))
  f42w_parity(df, function(d) mutate(d, s = sum(v)))
  f42w_parity(df, function(d) d |> group_by(g) |> mutate(s = sum(v)))
})

test_that("eager: an overflowing window sum is double, like dplyr", {
  skip_if_no_gpu()
  big <- f42w_big()
  f42w_parity(big, function(d) mutate(d, s = sum(v)), lazy = FALSE)
  f42w_parity(big, function(d) d |> group_by(g) |> mutate(s = sum(v)), lazy = FALSE)
  f42w_parity(big, function(d) mutate(d, z = sum(v) + 1L), lazy = FALSE)
  f42w_parity(big, function(d) mutate(d, s = sum(v), .by = g), lazy = FALSE)
})

test_that("lazy: an overflowing window sum is double, like dplyr (the call runs at the verb)", {
  skip_if_no_gpu()
  big <- f42w_big()
  f42w_parity(big, function(d) mutate(d, s = sum(v)))
  f42w_parity(big, function(d) d |> group_by(g) |> mutate(s = sum(v)) |> mutate(t = s + 1L))
  f42w_parity(big, function(d) mutate(d, s = sum(v), .by = g))
  # the result is lazy with nothing pending, and a pending plan ran first
  lz <- as_lazy(tbl_gpu(big))
  out <- mutate(filter(lz, v > 0L), s = sum(v))
  expect_true(is_lazy(out))
  expect_false(has_pending_ops(out))
})

test_that("lazy: a BOOL8 window sum stays a pending step", {
  skip_if_no_gpu()
  lz <- as_lazy(tbl_gpu(data.frame(g = c(1, 1, 2), l = c(TRUE, FALSE, TRUE))))
  expect_true(has_pending_ops(mutate(lz, s = sum(l))))
  expect_true(has_pending_ops(mutate(lz, s = sum(l), .by = g)))
})

test_that("filter() on an overflowing integer sum keeps matching dplyr", {
  skip_if_no_gpu()
  big <- f42w_big()
  f42w_parity(big, function(d) filter(d, sum(v) > 0))
  f42w_parity(big, function(d) d |> group_by(g) |> filter(sum(v) > 10))
})

# Added after round 1 (review B2): integer arithmetic on an in-range window
# sum inside filter() is checked integer arithmetic, as in dplyr.
test_that("filter() on integer arithmetic over a window sum drops overflowed rows", {
  skip_if_no_gpu()
  d <- data.frame(g = c(1, 1, 2, 2), x = c(M - 5L, 5L, 1L, 2L))
  f42w_parity(d, function(t) filter(t, sum(x) + 1L > 0L, .by = g))
  f42w_parity(d, function(t) t |> group_by(g) |> filter(sum(x) + 1L > 0L))
  e <- data.frame(x = c(2L, 1L), y = c(1000000000L, 1000000000L))
  f42w_parity(e, function(t) filter(t, x * sum(y) > 0L))
  expect_warning(collect(filter(tbl_gpu(e), x * sum(y) > 0L)), "NAs produced by integer overflow")
})

# Added after round 2 (review B3, B4): an in-range sum nested inside another
# window call is still an integer, so arithmetic on it is checked integer
# arithmetic (filter and summarise).
test_that("filter() on arithmetic over a nested in-range window sum is integer arithmetic", {
  skip_if_no_gpu()
  e <- data.frame(x = c(2L, 1L), y = c(1000000000L, 1000000000L))
  f42w_parity(e, function(t) filter(t, max(x * sum(y)) > 0L))
  d <- data.frame(g = c(1, 1, 2, 2), x = c(2L, 1L, 1L, 1L),
                  y = c(1000000000L, 1000000000L, 1L, 1L))
  f42w_parity(d, function(t) filter(t, max(x * sum(y)) > 0L, .by = g))
  expect_warning(collect(filter(tbl_gpu(e), max(x * sum(y)) > 0L)), "NAs produced by integer overflow")
})

test_that("summarise() of a row-level window over arithmetic on a nested sum is integer", {
  skip_if_no_gpu()
  f42w_parity(data.frame(x = c(1L, 2L, 3L)), function(t) summarise(t, m = dplyr::last(cumsum(x * sum(x)))))
  f42w_parity(data.frame(x = c(50000L, 50000L)), function(t) summarise(t, m = dplyr::last(cumsum(x * sum(x)))))
})

# Added after round 3 (review B5, N18): when one integer window sum overflows,
# the sums that fit are still integers, so integer arithmetic on them still
# overflows to NA with a warning (dplyr evaluates per group; values verified
# with tools/r). Eager must match dplyr. Lazy summarise() may match dplyr or
# raise `cuplyr_error_sum_overflow` (F34's lazy rule), never a wrong value.
f42w_eager_or_lazy_error <- function(df, pipeline) {
  f42w_parity(df, pipeline, lazy = FALSE)
  expect_warning(pipeline(tbl_gpu(df)), "NAs produced by integer overflow")
  oracle <- suppressWarnings(as.data.frame(pipeline(tibble::as_tibble(df))))
  got <- tryCatch(
    suppressWarnings(as.data.frame(collect(pipeline(as_lazy(tbl_gpu(df)))))),
    cuplyr_error_sum_overflow = function(e) NULL
  )
  if (!is.null(got)) expect_equal(got, oracle, ignore_attr = TRUE)
}

test_that("summarise(): an overflowing sum doesn't turn in-range sums into doubles", {
  skip_if_no_gpu()
  e <- data.frame(x = c(1L, 1L), y = c(M, 5L), z = c(50000L, 50000L))
  f42w_eager_or_lazy_error(e, function(t) summarise(t, m = max(x * sum(y) + z * sum(z))))
  d <- data.frame(g = c(1, 1, 2, 2), x = c(M, 5L, 50000L, 50000L))
  f42w_eager_or_lazy_error(d, function(t) summarise(t, m = max(x * sum(x)), .by = g))
  f42w_eager_or_lazy_error(d, function(t) t |> group_by(g) |> summarise(m = max(x * sum(x))))
  f42w_eager_or_lazy_error(d, function(t) summarise(t, m = dplyr::last(cumsum(x * sum(x))), .by = g))
})

test_that("filter(): an overflowing group's sum doesn't hide another group's overflow", {
  skip_if_no_gpu()
  # F80: a lazy table runs an integer window sum at the verb, as eager does
  d <- data.frame(g = c(1, 1, 2, 2), x = c(M, 5L, 50000L, 50000L))
  f42w_parity(d, function(t) filter(t, x * sum(x) > 0L, .by = g))
  f42w_parity(d, function(t) t |> group_by(g) |> filter(x * sum(x) > 0L))
  expect_warning(filter(tbl_gpu(d, lazy = FALSE), x * sum(x) > 0L, .by = g), "NAs produced by integer overflow")
  expect_warning(filter(tbl_gpu(d, lazy = TRUE), x * sum(x) > 0L, .by = g), "NAs produced by integer overflow")
})
