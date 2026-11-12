# F67: round(x, digits) on the GPU equals R's round() exactly.
#
# R >= 4.0.0 (src/nmath/fround.c, R 4.6.1): for digits != 0 R computes the two
# candidates xd = floor(|x| * 10^d) / 10^d and xu = ceil(|x| * 10^d) / 10^d
# and keeps the one closer to |x| in double arithmetic; on a tie the one whose
# scaled value is even. Every input below comes from a column, so the GPU
# evaluates it (a column-free round() is folded in R).

# Tests see the package namespace, which doesn't import these; bind dplyr's
# own functions here, as library(dplyr) would for a user.
for (fn in c("if_else", "case_when", "coalesce", "recode")) {
  assign(fn, getExportedValue("dplyr", fn))
}

test_that("round(x, digits) matches R on the decimal ties from the brief", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.05, 7.55, 0.15, -1.05, 2.675, 0.125, -7.55, 0.05,
                         0.25, 0.35, 0.45, -0.15, 99.95, 1.005))
  f67_check(df, function(d) mutate(d, r1 = round(x, 1), r2 = round(x, 2)))
})

test_that("round(x, 1) over every 2-decimal value in [-99.99, 99.99]", {
  skip_if_no_gpu()
  v <- (0:9999) / 100
  df <- data.frame(x = c(v, -v))
  f67_check(df, function(d) mutate(d, r1 = round(x, 1)))
})

test_that("round() over 3- and 4-decimal grids at several magnitudes", {
  skip_if_no_gpu()
  v3 <- (0:19999) / 1000
  set.seed(67)
  k4 <- sample.int(99999, 5000)
  v4 <- c(k4 / 10000, k4 / 10000 + 1e3, k4 / 10000 + 1e6, k4 / 10000 + 1e9,
          k4 / 10000 + 12345)
  df <- data.frame(x = c(v3, -v3, v4, -v4))
  f67_check(df, function(d) {
    mutate(d, r1 = round(x, 1), r2 = round(x, 2), r3 = round(x, 3))
  })
})

test_that("round() of random doubles across magnitudes and digits", {
  skip_if_no_gpu()
  set.seed(6767)
  n <- 20000
  x <- runif(n, -1, 1) * 10^sample(-12:17, n, replace = TRUE)
  df <- data.frame(x = x)
  f67_check(df, function(d) {
    mutate(d,
      r0 = round(x), r1 = round(x, 1), r2 = round(x, 2), r4 = round(x, 4),
      r7 = round(x, 7), r10 = round(x, 10), r12 = round(x, 12),
      r15 = round(x, 15), r16 = round(x, 16), r20 = round(x, 20),
      m1 = round(x, -1), m3 = round(x, -3), m6 = round(x, -6)
    )
  })
})

test_that("round(x) with no digits keeps R's half-even rule, large values and signs", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.5, 1.5, 2.5, -0.5, -1.5, -0.4, 0.4, 2^50 + 0.25,
                         2^50 + 0.5, 2^52 + 1, -2^53, 1e300, 4503599627370495.5))
  f67_check(df, function(d) mutate(d, r = round(x), r0 = round(x, 0)))
})

test_that("round() with negative digits, on doubles and integers", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1250, 1350, -1250, 15, 25, -35, 149.99, 5e300, 1.5e300, 123456789.5),
    i = c(1250L, 1350L, -1250L, 15L, 25L, -35L, 150L, 2147483647L, -2147483647L, NA)
  )
  f67_check(df, function(d) {
    mutate(d,
      x1 = round(x, -1), x2 = round(x, -2), x300 = round(x, -300),
      x301 = round(x, -301), i1 = round(i, -1), i2 = round(i, -2), i0 = round(i)
    )
  })
})

test_that("round() with non-integer digits follows R (dig = floor(digits + 0.5))", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.123, 1.155, -2.5, 2^50 + 0.25, 2^50 + 0.5, 0.15))
  f67_check(df, function(d) {
    mutate(d,
      a = round(x, 1.5), b = round(x, 0.5), c = round(x, 1.4999),
      e = round(x, 0.4), f = round(x, -0.4), g = round(x, 2.5)
    )
  })
})

test_that("round() with very large and very negative digits", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.234, -1.234, 1.234567e-300, -9.87654321e-305, 0.1 + 0.2,
                         123456.123456789, 5, -5, 0, -0, 5e300, 1e-320, Inf, NA))
  f67_check(df, function(d) {
    mutate(d,
      d15 = round(x, 15), d16 = round(x, 16), d308 = round(x, 308),
      d310 = round(x, 310), d320 = round(x, 320), d323 = round(x, 323),
      d400 = round(x, 400), dinf = round(x, Inf),
      m308 = round(x, -308), m309 = round(x, -308.6), m400 = round(x, -400),
      minf = round(x, -Inf)
    )
  })
})

test_that("round() of tiny values with mid-range and two-step digits", {
  skip_if_no_gpu()
  set.seed(670)
  n <- 20000
  x <- runif(n, -1, 1) * 10^-sample(0:322, n, replace = TRUE)
  df <- data.frame(x = c(x, 5 * 10^-(1:300), 1.5 * 10^-(1:300)))
  f67_check(df, function(d) {
    mutate(d,
      d5 = round(x, 5), d50 = round(x, 50), d100 = round(x, 100), d200 = round(x, 200),
      d300 = round(x, 300), d305 = round(x, 305), d309 = round(x, 309),
      d315 = round(x, 315), d322 = round(x, 322)
    )
  })
})

test_that("round() keeps Inf, NA and GPU-computed NaN, and the sign of zero", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(Inf, -Inf, NA, 1.25, -0, 0, -0.04, -0.4, 0.04),
    y = c(Inf, -Inf, NA, 1.25, 2, 3, 4, 5, 6)
  )
  f67_check(df, function(d) {
    mutate(d,
      r1 = round(x, 1), r2 = round(x, 2), rm = round(x, -1), r0 = round(x),
      nan1 = round(y - y, 1), nan0 = round(y - y)
    )
  })
})

test_that("round() of integer and logical columns is a double", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, -15L, 25L, NA, 0L), b = c(TRUE, FALSE, NA, TRUE, FALSE))
  f67_check(df, function(d) {
    mutate(d, i1 = round(i, 1), i0 = round(i), im = round(i, -1), b1 = round(b, 1),
           b0 = round(b))
  })
})

test_that("round(x, digits) is native and exact in filter, grouped mutate and summarise", {
  skip_if_no_gpu()
  v <- (0:1999) / 100
  df <- data.frame(x = c(v, -v), g = rep(c("a", "b", "c", "d"), 1000))
  f67_check(df, function(d) filter(d, round(x, 1) == round(x, 2)))
  f67_check(df, function(d) filter(d, round(x, 1) * 10 == floor(x * 10)))
  f67_check(df, function(d) d |> group_by(g) |> mutate(r = round(x, 1), s = round(x * 3, 1)))
  # Exact reducers only: sum()/mean() of the rounded values would also compare
  # cudf's summation order with R's long-double accumulator.
  f67_check(df, function(d) {
    d |> group_by(g) |> summarise(
      n = n_distinct(round(x, 1)), m = max(round(x * 7, 2)), l = min(round(x / 3, 2))
    )
  })
  f67_check(df, function(d) summarise(d, n = n_distinct(round(x, 1)), m = min(round(x, -1))))
})

test_that("group_by() on a round() expression gives R's groups", {
  skip_if_no_gpu()
  # No value rounds to zero: a group holding both +0 and -0 keys takes the
  # sign of its first row in dplyr, while cudf's groupby picks any row (a
  # separate group-key bug, on dev before F67 too, reported by the F67 PM).
  v <- (6:2999) / 100
  df <- data.frame(x = c(v, -v))
  f67_check(df, function(d) d |> group_by(r = round(x, 1)) |> summarise(n = n()))
})

test_that("round() inside across() and other expressions", {
  skip_if_no_gpu()
  df <- data.frame(a = c(1.05, 7.55, 0.15), b = c(-1.05, 2.675, 0.125))
  f67_check(df, function(d) mutate(d, across(c(a, b), ~ round(.x, 1))))
  f67_check(df, function(d) mutate(d, s = round(a + b, 1), t = if_else(a > 1, round(a, 1), b)))
})

test_that("the digits argument: named, integer, environment values and folded constants", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.05, 7.55, 0.15, -1.05, 1250, 1350))
  k <- 1
  f67_check(df, function(d) {
    mutate(d,
      a = round(x, digits = 1), b = round(x, 1L), c = round(x, k), e = round(x, -k),
      f = round(x, k + 1), h = round(x, -(2))
    )
  })
  # Reordered named arguments take the CPU route by design (F19-a2); the
  # result must still be R's.
  f67_check(df, function(d) mutate(d, i = round(digits = 1, x = x)), native = FALSE)
})

test_that("round() with digits from a column or a missing digits matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.05, 7.55, 0.15, -1.05, 1250, NA), dd = c(1L, 1L, 1L, 1L, -2L, 1L))
  f67_check(df, function(d) mutate(d, r = round(x, dd)), native = FALSE)
  f67_check(df, function(d) mutate(d, r = round(x, NA)), native = FALSE)
  f67_check(df, function(d) mutate(d, r = round(x, NaN)), native = FALSE)
})

test_that("round(x, digits) on a zero-row table", {
  skip_if_no_gpu()
  df <- data.frame(x = double(), i = integer())
  f67_check(df, function(d) mutate(d, r = round(x, 1), s = round(i, -1)))
})

test_that("signif() matches R", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.05, 0.15, 7.55, -1.05, 123456, 1e-300, 0, -0, NA))
  f67_check(df, function(d) mutate(d, s1 = signif(x, 2), s2 = signif(x), s3 = signif(x, 1)),
            native = FALSE)
})

test_that("round(x, digits) holds a bounded amount of GPU memory", {
  skip_if_no_gpu()
  # Dev's cudf::round allocated only its output; R's rule needs a few
  # intermediates, but not one live column per step (a 1.2e8-row round(x, 2)
  # ran out of memory in round 1). Peak at most 4 FLOAT64 columns of x's size.
  n <- 1e7
  g <- tbl_gpu(data.frame(x = runif(n, -100, 100)))
  for (dg in c(2, -1, 310)) {
    tr <- gpu_alloc_tracking(mutate(g, r = round(x, !!dg)))
    expect_lte(tr$peak_bytes, 4 * 8 * n, label = sprintf("peak bytes of round(x, %s)", dg))
  }
})
