# F42 extra coverage (the acceptance tests live in test-parity-F42-*.R):
# bare NA operands, cumsum() with an NA before the overflow, warnings from the
# other verbs that share the evaluator, and the eager window-sum fallback.

M <- .Machine$integer.max

f42x_same <- function(df, pipeline, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
  # The values are compared above with warnings muffled; the overflow warning
  # itself must match too (dplyr wraps it, so match on the text).
  overflow_warned <- function(expr) any(grepl("integer overflow", f42x_warnings(expr)))
  oracle <- overflow_warned(pipeline(tibble::as_tibble(df)))
  expect_identical(overflow_warned(collect(pipeline(tbl_gpu(df, lazy = FALSE)))), oracle,
                   info = "eager overflow warning differs from dplyr")
  if (lazy) {
    expect_identical(overflow_warned(collect(pipeline(as_lazy(tbl_gpu(df))))), oracle,
                     info = "lazy overflow warning differs from dplyr")
  }
}

# The messages of the warnings signalled while `expr` runs.
f42x_warnings <- function(expr) {
  msgs <- character()
  withCallingHandlers(expr, warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  msgs
}

test_that("bare NA operands keep integer arithmetic checked and integer", {
  skip_if_no_gpu()
  df <- data.frame(x = c(M, 1L, NA), y = c(1L, 2L, 3L))
  f42x_same(df, function(d) mutate(d, a = x + NA, b = NA + y, c = x * NA_integer_, q = y %/% NA))
})

test_that("integer %/% and %% by zero, logical operands", {
  skip_if_no_gpu()
  df <- data.frame(a = c(5L, -5L, 0L, NA), b = c(2L, 2L, 0L, 1L))
  f42x_same(df, function(d) mutate(d, q = a %/% b, r = a %% b, q2 = a %/% -b, r2 = a %% -b))
  dl <- data.frame(a = c(TRUE, FALSE, NA), b = c(TRUE, TRUE, FALSE))
  f42x_same(dl, function(d) mutate(d, q = a %/% b, r = a %% b, s = a - b, m = a * b))
})

test_that("cumsum() warns only for an overflow before any NA", {
  skip_if_no_gpu()
  expect_no_warning(collect(mutate(tbl_gpu(data.frame(x = c(1L, NA, M, M))), c = cumsum(x))))
  expect_warning(collect(mutate(tbl_gpu(data.frame(x = c(1L, M, NA))), c = cumsum(x))),
                 "integer overflow in 'cumsum'")
  f42x_same(data.frame(x = c(1L, NA, M, M)), function(d) mutate(d, c = cumsum(x)))
})

test_that("cummean() of logical and integer columns accumulates in double", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), x = c(M, M, 1L), l = c(TRUE, NA, TRUE))
  f42x_same(df, function(d) d |> group_by(g) |> mutate(a = dplyr::cummean(x), b = dplyr::cummean(l)))
})

test_that("filter() and summarise() temp columns warn on integer overflow", {
  skip_if_no_gpu()
  gt <- tbl_gpu(data.frame(x = c(M, 1L), y = c(1L, 2L)))
  expect_warning(collect(filter(gt, x * 2L > 0L)), "NAs produced by integer overflow")
  expect_warning(collect(summarise(gt, s = sum(x + y))), "NAs produced by integer overflow")
  expect_no_warning(collect(filter(gt, y * 2L > 0L)))
})

test_that("eager window sum() overflow reruns on the CPU for .keep, .before and .by", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  big <- data.frame(g = c(1, 1, 2, 2), v = c(M, 5L, 1L, 2L), w = 1:4)
  f42x_same(big, function(d) mutate(d, s = sum(v), .by = g, .keep = "used"), lazy = FALSE)
  f42x_same(big, function(d) mutate(d, s = sum(v), t = cumsum(w), .before = w), lazy = FALSE)
  # No overflow: stays native and integer.
  withr::local_options(cuplyr.fallback = "error")
  f42x_same(data.frame(g = c(1, 1, 2), v = c(1L, 2L, 3L)),
            function(d) mutate(d, s = sum(v), p = sum(v > 1L), q = v + sum(v)))
})

# --- round 2 -----------------------------------------------------------------

test_that("a fused chain of checked nodes warns once and nulls every overflowed row", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1L, M, -M, 70000L, NA), y = c(2L, 1L, -1L, 70000L, 3L),
                   z = c(5L, 5L, 5L, 5L, NA))
  f42x_same(df, function(d) mutate(d, a = (x + y) * 2L - x, b = x * y * z, c = -(x - y) + z))
  expect_identical(f42x_warnings(collect(mutate(tbl_gpu(df), a = (x + y) * 2L - x, b = x * y * z))),
                   "NAs produced by integer overflow")
  expect_identical(f42x_warnings(collect(mutate(as_lazy(tbl_gpu(df)), a = (x + y) * 2L - x))),
                   "NAs produced by integer overflow")
  # A chain longer than the fused-node cap still gives exact values.
  long <- data.frame(x = c(1L, 2L, 1000000000L))
  f42x_same(long, function(d) mutate(d, a = x + 1L + 1L + 1L + 1L + 1L + 1L + 1L + 1L + 1L + 1L + 1L))
})

test_that("an overflow next to an NA in another branch is still reported", {
  skip_if_no_gpu()
  df <- data.frame(x = c(70000L, NA, 3L), z = c(NA, 5L, 6L))
  f42x_same(df, function(d) mutate(d, w = (x * x) + z))
  expect_identical(f42x_warnings(collect(mutate(tbl_gpu(df), w = (x * x) + z))),
                   "NAs produced by integer overflow")
  # Missing values alone are not an overflow, with or without an NA literal.
  expect_length(f42x_warnings(collect(mutate(tbl_gpu(df), w = x + z, v = x * z, u = (x + 1L) - NA_integer_))), 0)
  expect_length(f42x_warnings(collect(mutate(as_lazy(tbl_gpu(df)), w = x - z))), 0)
  # R evaluates the other operand of an NA operation, so its overflow counts.
  expect_identical(f42x_warnings(collect(mutate(tbl_gpu(df), w = (x * x) + NA_integer_))),
                   "NAs produced by integer overflow")
})

test_that("typed NA operands give NA without a warning, in every verb and mode", {
  skip_if_no_gpu()
  d <- data.frame(x = 1:3, y = c(1L, NA, 3L))
  for (mk in list(identity, as_lazy)) {
    expect_length(f42x_warnings(collect(mutate(mk(tbl_gpu(d, lazy = FALSE)), a = x + NA_integer_, b = NA_integer_ * y))), 0)
    expect_length(f42x_warnings(collect(filter(mk(tbl_gpu(d, lazy = FALSE)), x - NA_integer_ > 0L))), 0)
    expect_length(f42x_warnings(collect(summarise(mk(tbl_gpu(d, lazy = FALSE)), s = sum(x + NA_integer_, na.rm = TRUE)))), 0)
  }
})

test_that("ancestors of a checked node still evaluate as one expression", {
  skip_if_no_gpu()
  df <- data.frame(x = c(M, 1L, 5L, NA), y = c(1L, 2L, 3L, 4L), z = c(5L, 0L, 9L, 9L))
  f42x_same(df, function(d) filter(d, x + y > 0L & z > 1L))
  f42x_same(df, function(d) filter(d, (x + y) * z > 10L | is.na(x)))
  f42x_same(df, function(d) mutate(d, a = (x + y) / 2, b = dplyr::if_else(x + y > 3L, z, 0L), c = abs(x - y) + 0.5))
})

test_that("a later dot reads an earlier checked dot (scratch columns stay aligned)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(M, 1L, 5L), y = c(1L, 2L, 3L), b = c(TRUE, FALSE, TRUE))
  f42x_same(df, function(d) mutate(d, a = x + y, c = a * 2L, e = (b + b) * y, f = e + a, g = c - f))
  f42x_same(df, function(d) transmute(d, a = (x + y) %/% 2L, c = a + 1L))
})

test_that("integer arithmetic on a window sum is checked in filter() and summarise()", {
  skip_if_no_gpu()
  d <- data.frame(g = c(1, 1, 2, 2), x = c(M - 5L, 5L, 1L, 2L))
  f42x_same(d, function(t) filter(t, sum(x) + 1L > 0L, .by = g))
  # The ungrouped sum overflows inside mean(): both modes rerun on the CPU (F80:
  # a lazy integer window sum runs at the verb).
  f42x_same(d, function(t) filter(t, x * 1000L > mean(sum(x))))
  f42x_same(d, function(t) filter(t, x * 1000L > mean(sum(x)), .by = g))
  # An overflowing sum is a double, and the arithmetic on it follows.
  big <- data.frame(g = c(1, 1), x = c(M, 5L))
  f42x_same(big, function(t) filter(t, sum(x) + 1L > 0L))
  f42x_same(d, function(t) summarise(t, m = max(x * sum(x))))
  f42x_same(d, function(t) summarise(t, m = max(x * sum(x)), .by = g))
  f42x_same(d, function(t) summarise(t, m = max(sum(x) + 1L), .by = g))
  expect_identical(f42x_warnings(collect(filter(tbl_gpu(d), sum(x) + 1L > 0L, .by = g))),
                   "NAs produced by integer overflow")
})

test_that("a zero-row table keeps integer arithmetic typed", {
  skip_if_no_gpu()
  f42x_same(data.frame(x = integer(), y = integer()), function(d) mutate(d, z = x + y, w = x * 2L - y))
})

test_that("the eager window-sum fallback does not repeat a warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  # x + 1L overflows on one row and the remaining sum overflows too: dplyr
  # warns once, and so must the rerun.
  d <- data.frame(x = c(M, M - 1L, M - 1L))
  expect_length(f42x_warnings(mutate(tbl_gpu(d), s = sum(x + 1L, na.rm = TRUE))), 1)
  expect_length(f42x_warnings(mutate(d, s = sum(x + 1L, na.rm = TRUE))), 1)
  f42x_same(d, function(t) mutate(t, s = sum(x + 1L, na.rm = TRUE)), lazy = FALSE)
})

# --- round 3 -----------------------------------------------------------------

test_that("a checked chain longer than the fused cap is exact on both sides of every cut", {
  skip_if_no_gpu()
  # Ten left-deep nodes, bottom up: n1 x * y, n2 - y, n3 + 1L, n4 - 1L, n5 + z,
  # n6 - z, n7 + 3L, n8 - 3L, n9 * w, n10 + v. The fused groups are cut in the
  # middle of the chain; each row overflows at a different node (n1, n5, n7,
  # n9, n10), row 8 overflows at n1 and comes back in range at n5 in INT64.
  df <- data.frame(
    x = c(50000L, 2L, M, 1000L, M - 10L, 3L, 3L, 50000L),
    y = c(50000L, 1L, 1L, 1000L, 1L, 4L, 4L, 50000L),
    z = c(0L, M, 0L, 0L, 0L, 5L, 5L, -M),
    w = c(1L, 1L, 1L, 10000L, 1L, 2L, 2L, 1L),
    v = c(0L, 0L, 0L, 0L, 100L, 1L, NA, 0L)
  )
  chain <- function(d) {
    mutate(d, a = (((((((((x * y) - y) + 1L) - 1L) + z) - z) + 3L) - 3L) * w) + v,
           b = x + (y + (z + (w + (v + (x + (y + 1L)))))))
  }
  f42x_same(df, chain)
  f42x_same(df, function(d) filter(d, (((((((((x * y) - y) + 1L) - 1L) + z) - z) + 3L) - 3L) * w) + v > 0L))
  expect_identical(f42x_warnings(collect(chain(tbl_gpu(df)))), "NAs produced by integer overflow")
  expect_identical(f42x_warnings(collect(chain(as_lazy(tbl_gpu(df))))), "NAs produced by integer overflow")
  ok <- df[6, ]
  expect_length(f42x_warnings(collect(chain(tbl_gpu(ok)))), 0)
})

test_that("filter() keeps a top-level sum's double next to a nested in-range sum", {
  skip_if_no_gpu()
  # Both sums sit in the same window stage; only the top-level one overflows.
  # Both modes rerun on the CPU (F80: a lazy integer window sum runs at the verb).
  e <- data.frame(x = c(1L, 2L), y = c(M, 5L), z = c(1L, 2L))
  f42x_same(e, function(t) filter(t, sum(y) > 0L, max(x * sum(z)) > 0L))
  # with nothing overflowing, both stay on the GPU
  ok <- data.frame(x = c(1L, 2L), y = c(7L, 5L), z = c(1L, 2L))
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(ok, function(t) filter(t, sum(y) > 0L, max(x * sum(z)) > 0L))
  expect_same_as_dplyr_lazy(ok, function(t) filter(t, sum(y) > 0L, max(x * sum(z)) > 0L))
})

test_that("filter() on an overflowing nested sum matches dplyr in both modes", {
  skip_if_no_gpu()
  e <- data.frame(x = c(1L, 2L), y = c(M, 5L))
  f42x_same(e, function(t) filter(t, max(x * sum(y)) > 0L))
  f42x_same(e, function(t) filter(t, max(x * sum(y)) > 0L, .by = x))
  # F80: the lazy table runs the sum at the verb, so collect() has nothing to raise
  expect_no_error(collect(filter(tbl_gpu(e, lazy = TRUE), max(x * sum(y)) > 0L)))
  # y + 1L overflows (one warning in dplyr) and the remaining sum overflows too:
  # the rerun must not repeat the attempt's warning.
  d <- data.frame(x = c(1L, 2L, 3L), y = c(M, M - 1L, M - 1L))
  expect_length(f42x_warnings(filter(d, max(x * sum(y + 1L, na.rm = TRUE)) > 0)), 1)
  expect_length(f42x_warnings(filter(tbl_gpu(d), max(x * sum(y + 1L, na.rm = TRUE)) > 0)), 1)
  f42x_same(d, function(t) filter(t, max(x * sum(y + 1L, na.rm = TRUE)) > 0), lazy = FALSE)
})

test_that("summarise() reruns an overflowing nested sum on the CPU, warning once", {
  skip_if_no_gpu()
  f42x_same(data.frame(x = c(M, 5L)), function(t) summarise(t, m = dplyr::last(cumsum(x * sum(x)))))
  d <- data.frame(x = c(M, M - 1L, M - 1L))
  f42x_same(d, function(t) summarise(t, m = max(x * sum(x + 1L, na.rm = TRUE))))
  expect_length(f42x_warnings(summarise(d, m = max(x * sum(x + 1L, na.rm = TRUE)))), 1)
  expect_length(f42x_warnings(summarise(tbl_gpu(d), m = max(x * sum(x + 1L, na.rm = TRUE)))), 1)
})

# --- round 4 -----------------------------------------------------------------

test_that("summarise() with a mixed-overflow window sum reruns on the CPU in both modes", {
  skip_if_no_gpu()
  # Group 1's sum overflows (a double), group 2's fits, and 50000L * 100000L
  # then overflows in integer arithmetic.
  d <- data.frame(g = c(1, 1, 2, 2), h = c(1, 1, 1, 1), x = c(M, 5L, 50000L, 50000L))
  f42x_same(d, function(t) summarise(t, m = max(x * sum(x)), .by = g))
  f42x_same(d, function(t) t |> group_by(h, g) |> summarise(m = max(x * sum(x)), .groups = "keep"))
  f42x_same(d, function(t) t |> group_by(h, g) |> summarise(n = n(), m = max(x * sum(x)), .groups = "drop_last"))
  lz <- as_lazy(tbl_gpu(d))
  expect_length(f42x_warnings(summarise(lz, m = max(x * sum(x)), .by = g)), 1)
  expect_identical(suppressWarnings(summarise(lz, m = max(x * sum(x)), .by = g))$exec_mode, "lazy")
})

test_that("a lazy mutate() with an overflowing sum runs at the verb, then summarise() matches dplyr", {
  skip_if_no_gpu()
  # The mutate()'s own overflow is a double now (F80: an integer sum runs at
  # the verb), so the summarise() after it sees that column and the nested
  # window sum of its own, and the pipeline is dplyr's.
  d <- data.frame(x = c(M, 5L))
  pipeline <- function(t) t |> mutate(s = sum(x)) |> summarise(m = max(x * sum(x)))
  oracle <- suppressWarnings(pipeline(tibble::as_tibble(d)))
  got <- suppressWarnings(collect(pipeline(as_lazy(tbl_gpu(d)))))
  expect_equal(tibble::as_tibble(got), oracle)
})

test_that("in-range window sums stay native in eager filter() and summarise()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  withr::local_options(cuplyr.fallback = "error")
  d <- data.frame(g = c(1, 1, 2, 2), x = c(M - 5L, 5L, 50000L, 50000L))
  f42x_same(d, function(t) filter(t, x * sum(x) > 0L, .by = g))
  # (A lazy summarise() materializes its window stages, with the notice.)
  f42x_same(d, function(t) summarise(t, m = max(x * sum(x)), .by = g), lazy = FALSE)
  expect_length(f42x_warnings(filter(tbl_gpu(d), x * sum(x) > 0L, .by = g)), 1)
})
