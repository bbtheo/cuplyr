# F123 extra: siblings of the acceptance cases that the protected file does not
# cover (summarise, transmute, .keep, filter constants, desugars, types).
# (dplyr is not attached under tools/test-file: its helpers are called as dplyr::fn())

f123x_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f123x_df <- function() {
  tibble::tibble(
    k = c(3L, 2L, 1L, 5L, 4L), g = c(1L, 1L, 2L, 2L, 2L), d = c(1.5, 2, NA, NA, NA),
    s = letters[1:5], l = c(TRUE, FALSE, NA, TRUE, TRUE)
  )
}

test_that("F123: a later dot reads the new column in %in% (seed 2406)", {
  skip_if_no_gpu()
  v1 <- 4L
  f123x_native(tibble::tibble(k = 1L, a = 1), function(t) mutate(t, k = k >= a, m1 = k %in% v1))
  f123x_native(f123x_df(), function(t) transmute(t, m = l %in% 4L, n = k %in% c(1.5, 2)))
})

test_that("F123: window calls over one value per group in transmute, summarise and .keep", {
  skip_if_no_gpu()
  df <- f123x_df()
  # (a window-bearing transmute() runs on the CPU, dplyr's own result)
  expect_same_as_dplyr(df, function(t) transmute(t, a = sum(dplyr::n()), b = dplyr::lag(dplyr::first(s))))
  # (summarise() sends a reducer of a one-value argument to dplyr)
  expect_same_as_dplyr(df, function(t) summarise(t, a = sum(dplyr::n()), b = var(dplyr::n()), c = dplyr::lag(sum(k))))
  expect_same_as_dplyr(df, function(t) summarise(t, a = sum(dplyr::n()), c = max(k), .by = g))
  f123x_native(df, function(t) mutate(t, a = dplyr::lag(max(k)), .keep = "used"))
})

test_that("F123: filter() with a constant predicate that still needs the group", {
  skip_if_no_gpu()
  df <- f123x_df()
  f123x_native(df, function(t) filter(t, dplyr::lag(max(k), default = 0) == 0))
  f123x_native(df, function(t) filter(t, is.na(dplyr::lag(dplyr::n()))))
  f123x_native(df, function(t) filter(t, is.na(var(dplyr::n()))))
})

test_that("F123: desugars over a one-value-per-group argument", {
  skip_if_no_gpu()
  df <- f123x_df()
  f123x_native(df, function(t) {
    mutate(t,
      x1 = dplyr::cumall(dplyr::n() > 1), x2 = dplyr::cumany(dplyr::n() > 1), x3 = dplyr::consecutive_id(dplyr::n()),
      x4 = dplyr::cummean(sum(k)), x5 = dplyr::cummean(dplyr::first(l)), x6 = dplyr::cumall(dplyr::first(l))
    )
  })
  f123x_native(df, function(t) mutate(group_by(t, g), x1 = dplyr::consecutive_id(dplyr::n()), x2 = cumsum(sum(k))))
  # not group-scalar: the desugars are untouched
  f123x_native(df, function(t) mutate(t, y1 = dplyr::cummean(k), y2 = dplyr::consecutive_id(k), y3 = dplyr::ntile(k, 2)))
})

test_that("F123: result types of the recycled window calls", {
  skip_if_no_gpu()
  df <- f123x_df()
  f123x_native(df, function(t) {
    mutate(t,
      a = dplyr::lag(dplyr::first(s)), b = dplyr::nth(dplyr::last(s), 2), c = dplyr::lag(dplyr::n(), 2, default = 2), d = sum(dplyr::first(l)),
      e = cumsum(dplyr::first(l)), f = sum(max(d), na.rm = TRUE), g2 = dplyr::lead(dplyr::first(l), 0)
    )
  })
})

test_that("F123: double %% and %/% inside larger expressions and with NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5.5, -5.5, NA, 0, 7), y = c(2, 2, 0, 0.1, NA), z = c(1L, 2L, 3L, NA, 0L))
  f123x_native(df, function(t) mutate(t, a = (x %% y) + 1, b = z %% x, c = z %/% y, d = x %/% 2L))
  f123x_native(df, function(t) filter(t, x %% 2 == 1.5))
})

test_that("F123: a type error under a recycled window call is dplyr's own error", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("a", "b", "c"))
  p <- function(t) summarise(t, a = sum(dplyr::lag(dplyr::first(s) * 2)))
  oracle <- tryCatch(p(df), error = identity)
  expect_s3_class(oracle, "error")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tryCatch(dplyr::collect(p(tbl_gpu(df, lazy = lazy))), error = identity)
    expect_s3_class(gpu, "error")
    expect_identical(class(gpu), class(oracle))
  }
})

test_that("F123: cur_group_id() beside a column of a similar name", {
  skip_if_no_gpu()
  df <- tibble::tibble(..cur_group_id.. = 1:3, ..cur_group_id..1 = c(2L, 2L, 5L), g = c(1L, 1L, 2L))
  f123x_native(df, function(t) {
    mutate(t, s = sum(..cur_group_id..), h = sum(dplyr::cur_group_id()), c = cumsum(..cur_group_id..1))
  })
  f123x_native(df, function(t) mutate(t, h = sum(dplyr::cur_group_id()), l = dplyr::lag(dplyr::cur_group_id(), 0), .by = g))
  f123x_native(df, function(t) filter(t, sum(dplyr::cur_group_id()) == 1, .by = g))
})

test_that("F123: an ordered window over a column stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = c(2L, 1L, 3L))
  f123x_native(df, function(t) mutate(t, a = dplyr::lag(k, order_by = k), b = dplyr::first(k, order_by = k)))
})

test_that("F123: double %% and %/% give R's zero (no -0) and R's NaN/NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(-3, -0.5, 0.5, 0, NA, 6, 1, NaN, Inf), y = c(3, -3, -3, -3, 0, Inf, 0.1, 2, 2))
  p <- function(t) {
    mutate(t, a = x %% y, b = x %/% y, c = x %% 3, d = x %/% -3,
           e = x %% -Inf, f = x %/% Inf, g = NA_real_ %% y, h = x %% NA_real_)
  }
  want <- p(df)
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
    for (nm in c("a", "b", "c", "d", "e", "f", "g", "h")) {
      expect_identical(is.na(got[[nm]]), is.na(want[[nm]]), info = nm)
      expect_identical(is.nan(got[[nm]]), is.nan(want[[nm]]), info = nm)
      # (1 / x tells -0 from +0)
      expect_identical(1 / got[[nm]], 1 / want[[nm]], info = nm)
    }
  }
})

test_that("F123: %in% of an integer or logical column against a wider set", {
  skip_if_no_gpu()
  df <- tibble::tibble(i = c(4L, 6L, NA, 7L, 0L), l = c(TRUE, FALSE, NA, TRUE, FALSE), d = c(4, 6.5, NA, 1, 0))
  f123x_native(df, function(t) mutate(t, a = i %in% c(4, 6), b = i %in% c(4.5, 6), c = i %in% c(NA, 6.5),
                                      e = i %in% c(3e9, NaN), f = i %in% 1.5))
  f123x_native(df, function(t) mutate(t, a = l %in% c(0, 1), b = l %in% 1L, c = l %in% c(NA, 2L),
                                      e = l %in% c(2.5, 4L), f = l %in% 0L))
  f123x_native(df, function(t) mutate(t, a = d %in% c(4L, 1L), b = d %in% TRUE, c = d %in% c(NA, 0L)))
})

# Same values as dplyr with NA and NaN told apart, and -0 told from +0, eager
# and lazy, on the GPU.
f123x_same_values <- function(df, p, cols) {
  withr::local_options(cuplyr.fallback = "error")
  # (base %% warns "probable complete loss of accuracy" for a huge x / y; cuplyr doesn't)
  want <- suppressWarnings(p(df))
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
    for (nm in cols) {
      info <- sprintf("%s, lazy = %s", nm, lazy)
      expect_identical(is.na(got[[nm]]), is.na(want[[nm]]), info = info)
      expect_identical(is.nan(got[[nm]]), is.nan(want[[nm]]), info = info)
      expect_equal(1 / got[[nm]], 1 / want[[nm]], info = info)
    }
  }
}

test_that("F123: double %% and %/% by a NaN literal give NaN, not NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, -2.5, NaN, Inf, 0, NA), i = c(1L, -2L, NA, 0L, 3L, 4L))
  f123x_same_values(df, function(t) {
    mutate(t, a = x %% NaN, b = x %/% NaN, c = i %% NaN, d = i %/% NaN, e = x %/% -NaN)
  }, c("a", "b", "c", "d", "e"))
})

test_that("F123: negative literal divisors and the finite-column gate of %/%", {
  skip_if_no_gpu()
  p <- function(t) {
    mutate(t, a = x %/% -2, b = x %% -2, c = x %/% -0.1, d = x %% -0.1, e = x %/% -1e300,
           f = x %/% -2L, g = x %/% -(3), h = x %% -0, k = x %/% -0, m = x %/% 5e307, n = x %% -Inf)
  }
  cols <- c("a", "b", "c", "d", "e", "f", "g", "h", "k", "m", "n")
  xs <- list(
    c(4, 9, NaN, Inf), c(NaN, -Inf, 4, 9), c(rep(1, 6), NaN, Inf), c(rep(-1, 40), NaN, -Inf, 3),
    c(-0, 0, -3, 3, NA), numeric(0), c(NA_real_, NA_real_)
  )
  for (x in xs) f123x_same_values(tibble::tibble(x = x), p, cols)
  f123x_same_values(tibble::tibble(x = c(-3L, 4L, NA, 0L)), p, cols)
  f123x_same_values(tibble::tibble(x = c(TRUE, FALSE, NA)), p, cols)
  # Huge dividends: `x %% -0.1` is left out where R warns "probable complete
  # loss of accuracy" (|x / y| > 1/eps, finite; its long-double value is the
  # documented limit). A finite column whose sum overflows takes the exact path.
  big <- setdiff(cols, "d")
  f123x_same_values(tibble::tibble(x = c(1e300, -1e300, 1.7e308, -7.5)), p, big)
  f123x_same_values(tibble::tibble(x = c(9007199254740993, -1e16, 2.5)), p, big)
  f123x_same_values(tibble::tibble(x = c(1.7e308, 1.7e308, Inf, 3)), p, big)
  # R's %% is NaN where x / y overflows (no warning there)
  f123x_same_values(tibble::tibble(x = c(1.7e308, -1.7e308, 3, NA), y = c(0.1, 0.5, 0.1, 0.1)),
                    function(t) mutate(t, a = x %% 0.1, b = x %% -0.5, c = x %% y, d = x %/% 0.1), c("a", "b", "c", "d"))
})

test_that("F123: %% keeps R's -0 where R returns x itself (|y| > 2^63)", {
  skip_if_no_gpu()
  # (R's branch for a huge divisor starts above 1 / LDBL_EPSILON: -0 %% 2^63 is
  # +0, -0 %% 2^64 is -0)
  df <- tibble::tibble(
    x = c(-0, 0, -0, 5, -5, -0, -0, -0),
    y = c(Inf, -Inf, 1e300, -1e300, 1e17, 2, 2^63, -2^64)
  )
  f123x_same_values(df, function(t) {
    mutate(t, a = x %% y, b = x %% Inf, c = x %% -Inf, d = x %% 1e300, e = x %% -1e17, f = x %% 2,
           g = x %/% y, h = x %% 2^63, k = x %% -2^64)
  }, c("a", "b", "c", "d", "e", "f", "g", "h", "k"))
})

test_that("F123: cur_group_id() windows through if_any()/if_all()/across() in every verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 2L), x = c(1, 2, 3, 9, 1), y = c(1, 1, 1, 9, 9))
  f123x_native(df, function(t) {
    mutate(group_by(t, g), a = dplyr::if_any(c(x, y), ~ .x > sum(dplyr::cur_group_id())),
           b = dplyr::if_all(c(x, y), ~ .x > dplyr::lag(dplyr::cur_group_id(), 0) + 1))
  })
  f123x_native(df, function(t) mutate(t, dplyr::across(c(x, y), ~ .x + cumsum(dplyr::cur_group_id())), .by = g))
  f123x_native(df, function(t) filter(group_by(t, g), dplyr::if_all(c(x, y), ~ .x >= sum(dplyr::cur_group_id()))))
  f123x_native(df, function(t) filter(t, dplyr::if_any(c(x, y), ~ .x > 3 * sum(dplyr::cur_group_id())), .by = g))
  f123x_native(df, function(t) {
    filter(group_by(t, g), x > 1, dplyr::if_any(c(x, y), ~ .x > dplyr::n() + sum(dplyr::cur_group_id())))
  })
  # (a window-bearing transmute() runs on the CPU; its result must still be dplyr's)
  expect_same_as_dplyr(df, function(t) transmute(group_by(t, g), a = sum(dplyr::cur_group_id())))
  expect_same_as_dplyr_lazy(df, function(t) transmute(group_by(t, g), a = sum(dplyr::cur_group_id())))
})

test_that("F123: the cur_group_id() helper is known to the parser only inside its verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  for (lazy in c(FALSE, TRUE)) {
    t <- tbl_gpu(df, lazy = lazy)
    r1 <- t |> group_by(g) |> mutate(a = sum(dplyr::cur_group_id()))
    expect_null(ir_group_scalar_state$cols)
    r2 <- t |> filter(dplyr::if_any(x, ~ .x > sum(dplyr::cur_group_id())), .by = g)
    expect_null(ir_group_scalar_state$cols)
    expect_equal(dplyr::collect(r1)$a, c(1L, 1L, 2L))
    expect_equal(nrow(dplyr::collect(r2)), 2L)
    # an error inside the verb still clears it
    expect_error(t |> group_by(g) |> mutate(a = sum(dplyr::cur_group_id()) + no_such_col))
    expect_null(ir_group_scalar_state$cols)
  }
  # a user column later named like a helper is a column again
  df2 <- tibble::tibble(..cur_group_id.. = c(1L, 2L, 3L))
  f123x_native(df2, function(t) mutate(t, s = sum(..cur_group_id..)))
})

test_that("F123: %% by a finite column divisor keeps R's -0 (|y| > 2^63, either sign)", {
  skip_if_no_gpu()
  # (a y column with no NaN/Inf takes the one-kernel form; -0 %% -2^64 is -0)
  df <- tibble::tibble(
    x = c(-0, -0, 0, 0, -0, -0, 5, -5, NA, 3, 7),
    y = c(-2^64, 2^64, -2^64, 2^64, -2, 2, -1e300, 1e300, 0, 0, NA)
  )
  f123x_same_values(df, function(t) mutate(t, a = x %% y, b = x %/% y), c("a", "b"))
  f123x_same_values(tibble::tibble(x = c(-0, 0, -3, 3, NA), y = c(3L, -3L, 3L, -3L, 0L)),
                    function(t) mutate(t, a = x %% y, b = x %/% y), c("a", "b"))
  f123x_same_values(tibble::tibble(x = c(-0, 0, -3, 3), y = c(TRUE, TRUE, FALSE, NA)),
                    function(t) mutate(t, a = x %% y, b = x %/% y), c("a", "b"))
  # all-null and empty divisors
  f123x_same_values(tibble::tibble(x = c(-0, 1), y = c(NA_real_, NA_real_)),
                    function(t) mutate(t, a = x %% y, b = x %/% y), c("a", "b"))
  f123x_same_values(tibble::tibble(x = numeric(0), y = numeric(0)),
                    function(t) mutate(t, a = x %% y, b = x %/% y), c("a", "b"))
})

# Values identical to base R's, -0 told from +0, eager and lazy, on the GPU.
f123x_identical <- function(df, p, cols) {
  withr::local_options(cuplyr.fallback = "error")
  want <- p(df)
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
    for (nm in cols) {
      info <- sprintf("%s, lazy = %s", nm, lazy)
      expect_identical(got[[nm]], want[[nm]], info = info)
      expect_identical(1 / got[[nm]], 1 / want[[nm]], info = info)
    }
  }
}

test_that("F123: an integer or logical zero under unary minus is +0 in every operand position", {
  skip_if_no_gpu()
  k <- 0L
  kk <- FALSE
  df <- tibble::tibble(
    x = c(5, -5, 0, -0, NA, 2.5), i = c(0L, 3L, -3L, NA, 0L, 7L),
    big = 2^64, y = c(2, -2, 0.5, 3, 1, -7)
  )
  f123x_identical(df, function(t) {
    mutate(t,
      a = x %/% -0L, b = x %/% -k, c = -0L %/% y, d = -k %% big, e = i %/% -(0L),
      f = -0L %/% -big, g = x %/% -kk, h = -(-0L) %/% y, m = x %% -0L, n = -0 %/% y,
      o = -0L %% -big, r = i %/% -0L
    )
  }, c("a", "b", "c", "d", "e", "f", "g", "h", "m", "n", "o", "r"))
})

test_that("F123: %/% is R's whole number for every divisor form and magnitude", {
  skip_if_no_gpu()
  xs <- c(
    2^52 + 1, 2^53 - 1, 2^53 + 2, -2^53 - 2, 2^62 + 2048, 2^63 - 1024, 2^63, 2^63 + 2048,
    2^64, 1e17 + 8, -1e17, 4503599627370497, 13510798882111490, 9007199254740994,
    3, -3, 0.3, 1, 1e-300, -1e-300, 5e-324, -5e-324, 0, -0, NA, NaN, Inf, -Inf, 1.7e308
  )
  ys <- c(
    3, -3, 7, 0.1, -0.1, 1.5, 0.75, 2, -0.5, 1e300, -1e300, 5e307, 1e-305, -1e-310,
    0, -0, Inf, -Inf, NaN
  )
  # (no NA divisor: NaN %/% NA is NaN in R and NA on the GPU, which keeps NA
  # as a null; that is the NA rule of every arithmetic op, not F123's)
  g <- expand.grid(x = xs, y = ys, KEEP.OUT.ATTRS = FALSE)
  f123x_identical(tibble::tibble(x = g$x, y = g$y), function(t) mutate(t, q = x %/% y), "q")
  for (yv in ys) {
    f123x_identical(tibble::tibble(x = xs), function(t) mutate(t, q = x %/% yv), "q")
  }
  # negative literals written as calls, and INT32 / BOOL8 operands
  f123x_identical(tibble::tibble(x = xs), function(t) {
    mutate(t, a = x %/% -3, b = x %/% -0.1, c = x %/% -1.5, d = x %/% -2, e = x %/% -1e300, f = x %/% -7L)
  }, c("a", "b", "c", "d", "e", "f"))
  ints <- c(7L, -7L, 0L, NA, 2147483647L, -2147483647L, 1L)
  # (a NaN dividend over an NA divisor is the NA rule above: left out)
  gi <- expand.grid(i = ints, y = ys[!is.nan(ys)], KEEP.OUT.ATTRS = FALSE)
  f123x_identical(tibble::tibble(i = gi$i, y = gi$y, b = gi$i > 0L),
                  function(t) mutate(t, q = i %/% y, r = y %/% i, s = y %/% b, u = b %/% y), c("q", "r", "s", "u"))
  # quotients at the floor's edges: just below and above whole numbers
  ed <- expand.grid(x = c(1, 0.3, 0.6, 0.9, 1.2, 2.4, 3.3, 9.9, -0.9, -2.4), y = c(0.1, 0.3, 0.03, -0.15),
                    KEEP.OUT.ATTRS = FALSE)
  f123x_identical(tibble::tibble(x = ed$x, y = ed$y), function(t) mutate(t, q = x %/% y), "q")
})

test_that("F123: %/% follows R's long-double rounding where it is not the exact floor", {
  skip_if_no_gpu()
  # Each x / y lies within R's 64-bit rounding of a whole number: R's
  # floor(q) * y is rounded to 64 bits, so its result is one off the floor of
  # the exact quotient (checked with exact rationals). Below |q| = 2^53, from
  # 2^53 to 2^63, and where floor(q) * y rounds to a power of two.
  x <- c(
    1971303.2002307896, -126036.1050824451, 9.294572723387779e+22, 10663425494.813238,
    2.5145262314521254e+20, 1.7582179205850653e+17,
    1.2924765881407295e+21, 0.12499999999999999, -1.7358161887353117e+17, -7.700887612024619e+18,
    0.5000000000000001, -804601649917243.4,
    -15.999999999999998, 0.25, 0.25, -0.015624999999999998
  )
  y <- c(
    2.5696212483511876e-09, 5.98591629815459e-09, 93137942.47312754, -0.018338485375945124,
    -13748588.44832895, -53.801588925321575,
    -74195.36136113135, -1.909133130972699e-18, 13.219157653103634, 28.77509825703885,
    6.991809335164288e-18, 0.04595606097211851,
    -1.5932744456844955e-15, 3.584968256350084e-18, 5.602913320092008e-20, 1.2330124986173446e-18
  )
  # a column divisor, the same rows among ordinary ones, and literal divisors
  f123x_identical(tibble::tibble(x = x, y = y), function(t) mutate(t, q = x %/% y), "q")
  f123x_identical(tibble::tibble(x = c(x, 7, -2.5, 1e6), y = c(y, 0.1, 3, 0.3)),
                  function(t) mutate(t, q = x %/% y), "q")
  for (i in c(4L, 6L, 9L, 12L)) {
    yv <- y[i]
    f123x_identical(tibble::tibble(x = c(x[i], 1, -3)), function(t) mutate(t, q = x %/% yv), "q")
  }
})
