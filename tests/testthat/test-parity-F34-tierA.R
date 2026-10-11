# F34 acceptance tests, tier A: summarise() silent wrong results.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - A1: a factor grouping column stays a factor (levels kept) in the result,
#   whatever the dots do (temp columns, .by, .groups = "keep", key override).
# - A2: dplyr evaluates each dot per group. An aggregate inside an aggregate
#   reduces a length-1 value: sum(mean(n)) is the mean, sum(n()) the group
#   size, sd(mean(n)) is NA. When the argument is still length n
#   (sum(n - mean(n))), recycling makes it a per-row expression.
# - A3: sum() of an integer or logical column is integer (0L for 0 rows or
#   an all-NA group with na.rm, NA_integer_ for NA without na.rm). When a sum
#   leaves R's integer range (+-(2^31 - 1)), base R returns a double, and one
#   overflowing group makes the whole column double. cuplyr's lazy mode keeps
#   a static integer schema and raises `cuplyr_error_sum_overflow` instead
#   (design decision, see the F34 brief).
#
# Grouped summarise row order is cudf hash order in cuplyr (F33), hence
# arrange_by. expect_equal() does not tell integer from double, or a factor
# from its codes reliably, so column classes are compared explicitly.

f34_parity <- function(df, pipeline, arrange_by = NULL, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline, arrange_by = arrange_by))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
}

f34_df <- function() {
  data.frame(
    f = factor(c("b", "a", "b", "c", NA, "a"), levels = c("b", "a", "c", "z")),
    h = c("p", "p", "q", "q", "q", "p"),
    g = c(1, 1, 2, 2, 2, 3),
    n = c(1L, 2L, 3L, 4L, 5L, 6L),
    m = c(1L, 2L, NA, NA, NA, 6L),          # group g == 2 all NA
    x = c(1.5, 2, NA, 4, 5, 6),
    l = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE),
    stringsAsFactors = FALSE
  )
}

f34_big <- function() {
  data.frame(
    g = c(1, 1, 2, 2),
    v = c(.Machine$integer.max, 1L, 5L, 6L)
  )
}

# -----------------------------------------------------------------------------
# A1: factor group keys keep their levels
# -----------------------------------------------------------------------------

test_that("A1: factor key stays a factor when a dot builds a temp column", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(f) |> summarise(a = sum(n > 1), b = mean(x * 2, na.rm = TRUE))
  }, arrange_by = "f")
})

test_that("A1: factor key with a second key and .groups = 'keep'", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(f, h) |> summarise(a = sum(n > 1), .groups = "keep")
  }, arrange_by = c("f", "h"))
})

test_that("A1: factor .by key stays a factor with a temp column", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> summarise(a = mean(n * 2), .by = f)
  })
})

test_that("A1: factor key after pending lazy ops, with a temp column", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> filter(n > 1) |> group_by(f) |> summarise(a = sum(x > 1, na.rm = TRUE))
  }, arrange_by = "f")
})

test_that("A1: an overridden second key with a temp column leaves the factor key intact", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(f, h) |> summarise(h = sum(n > 1))
  }, arrange_by = "f")
})

test_that("A1: factor key with a window-in-summarise argument", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(f) |> summarise(s = sum(cumsum(x)))
  }, arrange_by = "f")
})

# -----------------------------------------------------------------------------
# A2: nested aggregates
# -----------------------------------------------------------------------------

test_that("A2: aggregate of an aggregate, ungrouped", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> summarise(s = sum(mean(n)), k = sum(n()), v = sd(mean(n)),
                   p = sum(mean(n) + 1))
  })
  # an inner na.rm (a separate call, so a fallback of the dot above can't hide it)
  f34_parity(f34_df(), function(d) d |> summarise(q = sum(mean(x, na.rm = TRUE))))
})

test_that("A2: aggregate of an aggregate, grouped", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(g) |>
      summarise(s = sum(mean(n)), k = sum(n()), v = var(mean(n)),
                p = sum(mean(n) + 1))
  }, arrange_by = "g")
  f34_parity(f34_df(), function(d) {
    d |> group_by(g) |> summarise(q = sum(max(x, na.rm = TRUE)))
  }, arrange_by = "g")
})

test_that("A2: aggregate of an aggregate with .by", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> summarise(s = sum(mean(n)), k = sum(n()), .by = g)
  })
})

test_that("A2: aggregate of an aggregate after pending lazy ops", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> filter(n > 1) |> group_by(g) |> summarise(s = sum(mean(n)), k = sum(n()))
  }, arrange_by = "g")
})

test_that("A2: length-n arguments with an inner aggregate still match dplyr, eager and lazy", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(g) |>
      summarise(s = sum(n - mean(n)), t = sum(n * mean(n)), c = sum(cumsum(n)))
  }, arrange_by = "g")
  f34_parity(f34_df(), function(d) {
    d |> summarise(s = sum(n - mean(n)), t = mean(x - min(x, na.rm = TRUE), na.rm = TRUE))
  })
})

test_that("A2: length-n window arguments stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f34_df()
  out <- tbl_gpu(df) |> group_by(g) |>
    summarise(s = sum(n - mean(n)), c = sum(cumsum(n))) |> collect()
  expect_equal(nrow(out), 3L)
})

# -----------------------------------------------------------------------------
# A3: sum() of integer / logical is integer
# -----------------------------------------------------------------------------

test_that("A3: sum() of integer and logical is integer, grouped", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(g) |>
      summarise(s = sum(n), s2 = sum(m, na.rm = TRUE), s3 = sum(m),
                sl = sum(l, na.rm = TRUE), sc = sum(n > 2))
  }, arrange_by = "g")
})

test_that("A3: sum() of integer and logical is integer, ungrouped and .by", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> summarise(s = sum(n), s3 = sum(m), sl = sum(l), sc = sum(x > 2, na.rm = TRUE))
  })
  f34_parity(f34_df(), function(d) {
    d |> summarise(s = sum(n), s2 = sum(m, na.rm = TRUE), .by = h)
  })
})

test_that("A3: sum() of integer over 0 rows is 0L", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> filter(n > 100) |> summarise(s = sum(n), sl = sum(l))
  })
})

test_that("A3: where(is.integer) after sum(<int>) selects the sum (F14)", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> summarise(s = sum(n), mx = mean(x)) |> select(where(is.integer))
  })
})

test_that("A3: integer arithmetic downstream of sum(<int>) stays integer", {
  skip_if_no_gpu()
  f34_parity(f34_df(), function(d) {
    d |> group_by(g) |> summarise(s = sum(n)) |> mutate(t = s + 1L)
  }, arrange_by = "g")
})

test_that("A3: sums at the int32 boundary stay integer", {
  skip_if_no_gpu()
  df <- data.frame(v = c(.Machine$integer.max - 1L, 1L))
  f34_parity(df, function(d) d |> summarise(s = sum(v)))
})

test_that("A3: an overflowing integer sum is double in eager mode, as in dplyr", {
  skip_if_no_gpu()
  f34_parity(f34_big(), function(d) d |> summarise(s = sum(v)), lazy = FALSE)
  f34_parity(f34_big(), function(d) d |> group_by(g) |> summarise(s = sum(v)),
             arrange_by = "g", lazy = FALSE)
  # -2^31 is NA_integer_ in R, so it is outside R's integer range
  neg <- data.frame(v = c(-.Machine$integer.max, -1L))
  f34_parity(neg, function(d) d |> summarise(s = sum(v)), lazy = FALSE)
})

test_that("A3: an overflowing integer sum in a lazy plan is a classed error", {
  skip_if_no_gpu()
  lz <- as_lazy(tbl_gpu(f34_big()))
  expect_error(collect(summarise(lz, s = sum(v))), class = "cuplyr_error_sum_overflow")
  expect_error(
    lz |> group_by(g) |> summarise(s = sum(v)) |> mutate(t = s + 1L) |> collect(),
    class = "cuplyr_error_sum_overflow"
  )
})

# Added after round 1 (review B1): a reducer whose length-1 result feeds
# another window (row-level or a non-idempotent reducer) inside an argument
# that is otherwise length n. dplyr runs that window over one value.
f34_nest_df <- function() {
  data.frame(g = c(1, 1, 2, 2, 2), n = c(1L, 2L, 3L, 4L, 5L))
}

test_that("A2: a reducer feeding another window inside the argument", {
  skip_if_no_gpu()
  df <- f34_nest_df()
  f34_parity(df, function(d) d |> summarise(a = sum(dplyr::cummean(mean(n)))))
  f34_parity(df, function(d) d |> summarise(b = sum(n - sum(mean(n)))))
  f34_parity(df, function(d) d |> summarise(e = sum(n - sd(mean(n)))))
  f34_parity(df, function(d) {
    d |> group_by(g) |> summarise(c = sum(dplyr::lag(mean(n)) + n, na.rm = TRUE))
  }, arrange_by = "g")
})

test_that("A2: common window arguments match dplyr and stay on the GPU", {
  skip_if_no_gpu()
  df <- f34_nest_df()
  pipe <- function(d) {
    d |> group_by(g) |>
      summarise(a = sum(dplyr::cummean(n)), b = sum(dplyr::ntile(n, 2) == 1), c = mean(n == max(n)))
  }
  f34_parity(df, pipe, arrange_by = "g")
  withr::local_options(cuplyr.fallback = "error")
  out <- collect(pipe(tbl_gpu(df)))
  expect_equal(nrow(out), 2L)
})
