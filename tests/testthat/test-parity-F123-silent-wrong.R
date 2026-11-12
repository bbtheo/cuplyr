# F123 acceptance: the silent-wrong results F118's campaign left undiagnosed.
#
# Case 1 (seeds 1252, 2046, 3121: row counts) is a fuzz-harness false
# positive, not a cuplyr bug: the generator draws the second table's join keys
# from dplyr's own intermediate result, so a key computed by a reduction
# (var/sd/sum) equals dplyr's double bit for bit and the GPU's only within the
# documented float tolerance; the exact-equality join then matches on one side
# only. The guards below pin the parts of that pipeline that are cuplyr's
# contract (exact keys join, `%in% NA_integer_` on a double).
#
# Case 2 (seed 2406): `%in%` cast the set to x's type on the GPU, so
# `TRUE %in% 4L` was TRUE and `1L %in% 1.5` was TRUE. Base R's match() compares
# in the common type of x and the set. The value-match verbs built on the same
# node (replace_values(), case_match(), recode_values()) cast the left-hand
# values to x's type with vctrs, which errors on a lossy cast.
#
# Case 3 (seed 829): a window function over an argument that has one value per
# group in dplyr (n(), cur_group_id(), an aggregate, or an expression of
# those) ran over n broadcast copies: `sum(n())` gave n^2, `lag(n())` gave n,
# `var(n())` gave 0. dplyr evaluates the call on the length-1 value and
# recycles it (R/mutate.R mutate_col(); R/rank.R; R/lead-lag.R).
#
# Case 4 (seed 847): double `%%` / `%/%` used cudf's PYMOD / FLOOR_DIV, which
# differ from R's myfmod() / myfloor() (src/main/arithmetic.c) for infinite
# divisors, NA by zero, and fractional divisors (`1 %/% 0.1` is 9 in R).

f123_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f123_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Same column values as dplyr with NA and NaN told apart (waldo's tolerance
# compare is not relied on for the missing-value mask), eager and lazy.
f123_same_mask <- function(df, pipeline, cols) {
  oracle <- pipeline(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tibble::as_tibble(dplyr::collect(pipeline(tbl_gpu(df, lazy = lazy))))
    for (col in cols) {
      o <- oracle[[col]]
      g <- gpu[[col]]
      info <- sprintf("column %s, lazy = %s", col, lazy)
      expect_identical(is.na(g), is.na(o), info = info)
      expect_identical(is.nan(g), is.nan(o), info = info)
      fin <- !is.na(o)
      expect_identical(g[fin] == Inf, o[fin] == Inf, info = info)
      expect_identical(g[fin] == -Inf, o[fin] == -Inf, info = info)
    }
  }
}

# ---------------------------------------------------------------------------
# Case 1: harness false positive (guards only)
# ---------------------------------------------------------------------------

test_that("F123 case 1: right_join on an exact aggregate key, then %in% NA_integer_", {
  skip_if_no_gpu()
  # var(c(1, 2, 3, 4, 5)) is exactly 2.5 in R and on the GPU.
  x <- tibble::tibble(b = c(1, 2, 3, 4, 5))
  y <- tibble::tibble(s1 = c(2.5, 2.5, 7))
  p <- function(t) {
    yy <- if (is_tbl_gpu(t)) tbl_gpu(y, lazy = identical(t$exec_mode, "lazy")) else y
    t |>
      summarise(s1 = var(b), s2 = dplyr::last(b), s3 = median(b)) |>
      right_join(yy, by = "s1") |>
      filter(.data[["s2"]] %in% NA_integer_)
  }
  f123_both(x, p)
  p2 <- function(t) {
    yy <- if (is_tbl_gpu(t)) tbl_gpu(y, lazy = identical(t$exec_mode, "lazy")) else y
    t |>
      summarise(s1 = var(b), s2 = dplyr::last(b)) |>
      right_join(yy, by = "s1") |>
      filter(s2 %in% c(0L, NA_integer_) | s1 > 3)
  }
  f123_both(x, p2, arrange_by = "s1")
})

test_that("F123 case 1: the seed-1252 aggregates agree with dplyr within tolerance", {
  skip_if_no_gpu()
  x <- tibble::tibble(b = c(-0.77, 2.68, -0.31, 1.28, 8.58), c = c(1e+06, 0, -1, 0.5, -1))
  # pmin() has no GPU implementation, so this mutate() runs on the CPU.
  f123_both(x, function(t) {
    t |>
      mutate(a = pmin(b * c, c, na.rm = TRUE)) |>
      summarise(s1 = var(.data[["a"]], na.rm = TRUE), s2 = dplyr::last(a), s3 = median(.data[["b"]], na.rm = TRUE))
  })
})

# ---------------------------------------------------------------------------
# Case 2: %in% compares in the common type
# ---------------------------------------------------------------------------

f123_in_df <- function() {
  tibble::tibble(
    k = c(1L, 1L, 2L, NA, 0L),
    a = c(1, 0.5, 1, 1, 3),
    l = c(TRUE, FALSE, NA, TRUE, FALSE),
    i = c(1L, 2L, NA, 4L, -3L),
    d = c(1, 2.5, NA, 4, -3),
    g = c(1L, 1L, 2L, 2L, 2L)
  )
}

test_that("F123 case 2: the seed-2406 shape (a later dot reads the new logical k)", {
  skip_if_no_gpu()
  v1 <- 4L
  f123_native(tibble::tibble(k = 1L, a = 1), function(t) mutate(t, k = k >= a, m1 = k %in% v1))
  f123_native(f123_in_df(), function(t) {
    mutate(t, k = dplyr::coalesce(k, 1L) >= a, m1 = k >= k & .data$k %in% .env$v1) |> ungroup()
  })
})

test_that("F123 case 2: logical x %in% an integer or double set", {
  skip_if_no_gpu()
  f123_native(f123_in_df(), function(t) {
    mutate(t,
      m1 = l %in% 4L, m2 = l %in% c(0L, 4L), m3 = l %in% 2.5, m4 = l %in% 1L,
      m5 = l %in% c(NA, 3L), m6 = l %in% c(0, 1), m7 = l %in% c(1.5, NA),
      m8 = l %in% NA_integer_, m9 = l %in% c(-1L, 2L)
    )
  })
})

test_that("F123 case 2: integer x %in% a double set with fractions", {
  skip_if_no_gpu()
  f123_native(f123_in_df(), function(t) {
    mutate(t,
      m1 = i %in% 1.5, m2 = i %in% c(2, 1.0000001), m3 = i %in% c(NA, 2.5),
      m4 = i %in% c(4, -3), m5 = i %in% 3e9, m6 = i %in% c(-2.5, 0.5),
      m7 = d %in% c(1L, 4L), m8 = i %in% TRUE, m9 = d %in% c(TRUE, NA)
    )
  })
})

test_that("F123 case 2: %in% in filter(), grouped mutate and conditionals", {
  skip_if_no_gpu()
  df <- f123_in_df()
  f123_native(df, function(t) filter(t, l %in% 2L))
  f123_native(df, function(t) filter(t, i %in% 1.5 | l %in% 4L))
  f123_native(df, function(t) filter(t, !(i %in% c(2.5, 4))))
  f123_native(df, function(t) group_by(t, g) |> mutate(m = l %in% 4L))
  f123_native(df, function(t) mutate(t, m = i %in% 1.5, .by = g))
  f123_native(df, function(t) {
    mutate(t, a1 = dplyr::if_else(l %in% 4L, 1, 2), a2 = dplyr::case_when(i %in% 2.5 ~ 1, .default = 2))
  })
  f123_native(df, function(t) summarise(t, n = sum(l %in% 4L), m = sum(i %in% c(1.5, 4))))
  f123_native(df[0, ], function(t) mutate(t, m = l %in% 4L, m2 = i %in% 1.5))
})

test_that("F123 case 2: value-match verbs raise dplyr's lossy-cast error", {
  skip_if_no_gpu()
  df <- f123_in_df()
  cases <- list(
    function(t) mutate(t, r = dplyr::replace_values(i, 1.5 ~ 9L)),
    function(t) mutate(t, r = dplyr::replace_values(l, 4L ~ NA)),
    function(t) mutate(t, r = suppressWarnings(dplyr::case_match(i, 1.5 ~ "a", .default = "b"))),
    function(t) mutate(t, r = dplyr::recode_values(l, 4L ~ "a", default = "b"))
  )
  for (p in cases) {
    oracle <- tryCatch(p(df), error = identity)
    expect_s3_class(oracle, "error")
    for (lazy in c(FALSE, TRUE)) {
      gpu <- tryCatch(dplyr::collect(p(tbl_gpu(df, lazy = lazy))), error = identity)
      expect_s3_class(gpu, "error")
      if (inherits(gpu, "error")) {
        expect_identical(class(gpu), class(oracle))
        expect_identical(conditionMessage(gpu$parent), conditionMessage(oracle$parent))
      }
    }
  }
})

test_that("F123 case 2: value-match verbs with lossless left-hand values", {
  skip_if_no_gpu()
  df <- f123_in_df()
  f123_native(df, function(t) mutate(t, r = dplyr::replace_values(i, 2 ~ 9L, 4 ~ 0L)))
  f123_native(df, function(t) mutate(t, r = dplyr::replace_values(l, 1L ~ FALSE)))
  f123_native(df, function(t) mutate(t, r = dplyr::recode_values(i, c(1, 4) ~ "a", default = "b")))
  f123_native(df, function(t) mutate(t, r = dplyr::recode_values(l, 0L ~ "no", 1L ~ "yes", default = "na")))
})

# ---------------------------------------------------------------------------
# Case 3: window functions over a one-value-per-group argument
# ---------------------------------------------------------------------------

f123_win_df <- function() {
  tibble::tibble(
    k = c(3L, 2L, 1L, 5L, 4L),
    g = c(1L, 1L, 2L, 2L, 2L),
    d = c(1.5, 2, NA, NA, NA),
    f = c(0.25, 0.5, 1, 2, 4)
  )
}

test_that("F123 case 3: the seed-829 shape (sum(n()) through count(wt =) and a join)", {
  skip_if_no_gpu()
  y <- tibble::tibble(n = numeric(0))
  p <- function(t) {
    yy <- if (is_tbl_gpu(t)) tbl_gpu(y, lazy = identical(t$exec_mode, "lazy")) else y
    t |> mutate(m2 = sum(dplyr::n())) |> count(wt = m2) |> left_join(yy, by = "n")
  }
  f123_native(tibble::tibble(k = 2:1), p)
  f123_native(tibble::tibble(k = 2:1), function(t) mutate(t, m2 = sum(dplyr::n())) |> count(wt = m2))
})

test_that("F123 case 3: aggregates of n() and cur_group_id() in mutate()", {
  skip_if_no_gpu()
  df <- f123_win_df()
  p <- function(t) {
    mutate(t,
      a1 = sum(dplyr::n()), a2 = sum(dplyr::cur_group_id()), a3 = sum(dplyr::n() * 2), a4 = sum(dplyr::n() > 1),
      a5 = var(dplyr::n()), a6 = sd(dplyr::n() * 1), a7 = mean(dplyr::n()), a8 = max(dplyr::n()), a9 = dplyr::n_distinct(dplyr::n()),
      a10 = sum(dplyr::n()) + k, a11 = sum(k) * sum(dplyr::n()), a12 = dplyr::first(dplyr::n())
    )
  }
  f123_native(df, p)
  f123_native(df, function(t) p(group_by(t, g)))
  f123_native(df, function(t) mutate(t, a1 = sum(dplyr::n()), a2 = var(dplyr::n()), a3 = sum(dplyr::n()) + k, .by = g))
  f123_native(df[0, ], p)
})

test_that("F123 case 3: scans, shifts and ranks of n() in mutate()", {
  skip_if_no_gpu()
  df <- f123_win_df()
  p <- function(t) {
    mutate(t,
      b1 = cumsum(dplyr::n()), b2 = dplyr::lag(dplyr::n()), b3 = dplyr::lead(dplyr::n()), b4 = dplyr::lag(dplyr::n(), default = 9L),
      b5 = dplyr::lag(dplyr::n(), 0), b6 = dplyr::row_number(dplyr::n()), b7 = cumprod(dplyr::n()), b8 = dplyr::cummean(dplyr::n()),
      b9 = cummax(dplyr::n()), b10 = dplyr::min_rank(dplyr::n()), b11 = dplyr::dense_rank(dplyr::n()), b12 = dplyr::cume_dist(dplyr::n()),
      b13 = dplyr::percent_rank(dplyr::n()), b14 = dplyr::nth(dplyr::n(), 2), b15 = dplyr::nth(dplyr::n(), 1), b16 = dplyr::nth(dplyr::n(), -1),
      b17 = cumsum(dplyr::n() > 1), b18 = dplyr::lead(dplyr::cur_group_id(), 2)
    )
  }
  f123_native(df, p)
  f123_native(df, function(t) p(group_by(t, g)))
  f123_native(df, function(t) mutate(t, b1 = cumsum(dplyr::n()), b2 = dplyr::lag(dplyr::n()), b3 = dplyr::row_number(dplyr::n()), .by = g))
})

test_that("F123 case 3: window functions of an aggregate of a column", {
  skip_if_no_gpu()
  df <- f123_win_df()
  p <- function(t) {
    mutate(t,
      c1 = sum(sum(k)), c2 = cumsum(mean(k)), c3 = dplyr::lag(max(k)), c4 = sum(sum(f)),
      c5 = dplyr::row_number(max(d)), c6 = dplyr::lag(max(d), default = 0), c7 = var(mean(f)),
      c8 = cumsum(sum(k) + dplyr::n()), c9 = dplyr::percent_rank(max(d))
    )
  }
  f123_native(df, p)
  f123_native(df, function(t) p(group_by(t, g)))
})

test_that("F123 case 3: filter() with a window over n()", {
  skip_if_no_gpu()
  df <- f123_win_df()
  f123_native(df, function(t) filter(t, sum(dplyr::n()) == 5))
  f123_native(df, function(t) filter(group_by(t, g), sum(dplyr::n()) == 3))
  f123_native(df, function(t) filter(t, is.na(dplyr::lag(dplyr::n()))))
  f123_native(df, function(t) filter(t, dplyr::row_number(dplyr::n()) == 1, .by = g))
})

test_that("F123 case 3: a later dot reads the recycled column, not a length-1 value", {
  skip_if_no_gpu()
  df <- f123_win_df()
  # In dplyr `a` is stored with n rows, so sum(a) is n * mean(k).
  f123_native(df, function(t) mutate(t, a = mean(k), b = sum(a), c = cumsum(a)))
  f123_native(df, function(t) mutate(group_by(t, g), a = dplyr::n(), b = sum(a), c = dplyr::lag(a)))
})

test_that("F123 case 3: ntile() and rolling windows over n()", {
  skip_if_no_gpu()
  df <- f123_win_df()
  f123_both(df, function(t) mutate(t, e1 = dplyr::ntile(dplyr::n(), 2)))
  f123_both(df, function(t) mutate(group_by(t, g), e1 = dplyr::ntile(dplyr::n(), 2)))
  f123_both(df, function(t) mutate(t, e2 = roll_sum(dplyr::n(), 2), e3 = roll_mean(dplyr::n() * 1, 1)))
})

test_that("F123 case 3: consecutive_id() of a one-value-per-group argument", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = c(3L, 2L, 1L), j = 1:3)
  f123_both(df, function(t) summarise(t, a = sum(dplyr::consecutive_id(max(k)))))
  f123_native(df, function(t) mutate(t, a = dplyr::consecutive_id(max(k)), .keep = "used"))
  f123_native(df, function(t) mutate(t, a = dplyr::consecutive_id(max(k)), .keep = "unused"))
  f123_native(df, function(t) filter(t, dplyr::consecutive_id(dplyr::n()) == 1L))
})

test_that("F123 case 3: a POSIXct one-value-per-group argument keeps its time zone", {
  skip_if_no_gpu()
  d0 <- as.POSIXct("2024-01-01", tz = "UTC")
  df <- tibble::tibble(t = as.POSIXct(c("2024-01-01 10:00:00", "2024-01-02 11:00:00"), tz = "America/New_York"))
  p <- function(t) {
    mutate(t, a = dplyr::lag(max(t)), b = dplyr::nth(dplyr::first(t), 2),
           c = dplyr::lead(dplyr::last(t), default = d0), e = dplyr::lag(max(t), 0))
  }
  f123_both(df, p)
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
    expect_identical(lapply(got, attr, "tzone"), lapply(p(df), attr, "tzone"))
  }
})

test_that("F123 case 3: a user column named like the cur_group_id() helper is a column", {
  skip_if_no_gpu()
  df <- tibble::tibble(..cur_group_id.. = 1:3, g = c(1L, 1L, 2L))
  f123_both(df, function(t) mutate(t, s = sum(..cur_group_id..), c = cumsum(..cur_group_id..)))
  f123_both(df, function(t) mutate(t, s = sum(..cur_group_id..), h = sum(dplyr::cur_group_id()), .by = g))
})

test_that("F123 case 3: order_by over a one-value-per-group argument errors like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("a", "b", "c"), k = c(2L, 1L, 3L))
  cases <- list(
    function(t) mutate(t, a = dplyr::lag(dplyr::n(), order_by = k)),
    function(t) mutate(t, a = dplyr::first(dplyr::n(), order_by = k)),
    function(t) mutate(t, a = dplyr::nth(max(k), 2, order_by = k)),
    function(t) mutate(t, a = dplyr::order_by(k, cumsum(dplyr::n())))
  )
  for (p in cases) {
    oracle <- tryCatch(p(df), error = identity)
    expect_s3_class(oracle, "error")
    for (lazy in c(FALSE, TRUE)) {
      gpu <- tryCatch(dplyr::collect(p(tbl_gpu(df, lazy = lazy))), error = identity)
      expect_s3_class(gpu, "error")
      if (inherits(gpu, "error")) {
        expect_identical(class(gpu), class(oracle))
        expect_identical(conditionMessage(gpu$parent), conditionMessage(oracle$parent))
      }
    }
  }
})

# ---------------------------------------------------------------------------
# Case 4: %% and %/% follow R for doubles
# ---------------------------------------------------------------------------

f123_mod_grid <- function(xs, ys) {
  g <- expand.grid(x = xs, y = ys, KEEP.OUT.ATTRS = FALSE)
  tibble::tibble(x = g$x, y = g$y)
}

test_that("F123 case 4: the seed-847 shape (integer %% -Inf)", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = -Inf, c = 6L)
  p <- function(t) mutate(t, d = c %% .data[["a"]])
  f123_native(df, p)
  f123_same_mask(df, p, "d")
})

test_that("F123 case 4: double %% and %/% by infinities, zeros and NA", {
  skip_if_no_gpu()
  df <- f123_mod_grid(
    c(6, -6, 0, 5.5, -5.5, Inf, -Inf, NA, 7e15, 3),
    c(Inf, -Inf, 0, 3, -3, 2.5, -2.5, NA, 1e17, -1e17, 1e300, 6)
  )
  p <- function(t) mutate(t, m = x %% y, q = x %/% y)
  f123_native(df, p)
  f123_same_mask(df, p, c("m", "q"))
  # A huge dividend only where R's result is exact (no "loss of accuracy").
  big <- f123_mod_grid(c(1e300, -1e300), c(Inf, -Inf, 0, NA, 1e300, -1e300))
  f123_native(big, p)
  f123_same_mask(big, p, c("m", "q"))
})

test_that("F123 case 4: mixed integer / double operands", {
  skip_if_no_gpu()
  df <- f123_mod_grid(
    c(7L, -7L, 0L, NA, 2147483647L, -2147483647L, 1L, 6L),
    c(Inf, -Inf, 0, NA, 2.5, -0.5, 3, -3)
  )
  p <- function(t) mutate(t, m = x %% y, q = x %/% y, m2 = y %% x, q2 = y %/% x)
  f123_native(df, p)
  f123_same_mask(df, p, c("m", "q", "m2", "q2"))
  df2 <- f123_mod_grid(c(6, -6, Inf, NA, 0, 2.5), c(0L, NA, 3L, -3L))
  f123_native(df2, p)
  f123_same_mask(df2, p, c("m", "q", "m2", "q2"))
})

test_that("F123 case 4: %/% by a fractional divisor is R's floor of the exact quotient", {
  skip_if_no_gpu()
  df <- f123_mod_grid(
    c(1, 0.3, 0.7, 2.6, -1, -0.3, 5, 123456.789, 0.001, -0.5, 10, 4.5, 1e6),
    c(0.1, -0.1, 0.2, 0.3, 0.001, 1.3, -1.3, 0.5, 123456.789, 3, 7)
  )
  p <- function(t) mutate(t, m = x %% y, q = x %/% y)
  f123_native(df, p)
  f123_same_mask(df, p, c("m", "q"))
})

test_that("F123 case 4: integer %% and %/% are unchanged", {
  skip_if_no_gpu()
  df <- f123_mod_grid(
    c(7L, -7L, 0L, NA, 2147483647L, -2147483647L, 1L, -1L),
    c(3L, -3L, 0L, NA, 1L, -1L, 2147483647L, 7L)
  )
  p <- function(t) mutate(t, m = x %% y, q = x %/% y)
  f123_native(df, p)
  f123_same_mask(df, p, c("m", "q"))
})

test_that("F123 case 4: %% and %/% in filter(), summarise(), grouped and literal divisors", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    x = c(1, 6, -6, 0.3, NA, 7, 2.6),
    y = c(0.1, Inf, -Inf, 3, 0, 0, 0.2),
    g = c(1L, 1L, 1L, 2L, 2L, 2L, 2L)
  )
  f123_native(df, function(t) filter(t, x %/% y == 9 | x %% y == Inf))
  f123_native(df, function(t) filter(t, is.nan(x %% y)))
  f123_native(df, function(t) summarise(t, s = sum(x %/% y, na.rm = TRUE), .by = g))
  f123_native(df, function(t) mutate(group_by(t, g), m = x %% Inf, q = x %/% -Inf, r = x %/% 0.1))
  f123_native(df, function(t) mutate(t, m = (x %% y) * 2 + 1, q = (x + 1) %/% (y - 1)))
  p <- function(t) mutate(t, m = x %% y, q = x %/% y, m2 = x %% -Inf, q2 = 1 %/% y)
  f123_same_mask(df, p, c("m", "q", "m2", "q2"))
})

test_that("F123 case 4: literal divisors over columns holding NaN and infinities", {
  skip_if_no_gpu()
  cols <- list(
    c(4, 9, NaN, Inf), c(NaN, Inf, 4, 9), c(1, Inf, -Inf, 5),
    c(rep(1, 6), NaN, Inf), c(rep(1, 30), NaN, -Inf), c(-Inf, NA, 2.5, NaN, -7)
  )
  p <- function(t) {
    mutate(t,
      q1 = x %/% 2, m1 = x %% 2, q2 = x %/% -2, m2 = x %% -2, q3 = x %/% 1e300,
      m3 = x %% 1e300, q4 = x %/% -1e300, q5 = x %/% 0.1, m5 = x %% 0.1,
      q6 = x %/% Inf, m6 = x %% -Inf, q7 = x %/% 0, m7 = x %% 0, q8 = x %/% 5e307
    )
  }
  outs <- c("q1", "m1", "q2", "m2", "q3", "m3", "q4", "q5", "m5", "q6", "m6", "q7", "m7", "q8")
  for (x in cols) {
    df <- tibble::tibble(x = x)
    f123_native(df, p)
    f123_same_mask(df, p, outs)
  }
})

test_that("F123 case 3: cur_group_id() windows inside filter()'s if_any()/if_all()", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3), y = c(1, 1, 1))
  f123_native(df, function(t) group_by(t, g) |> filter(if_any(c(x, y), ~ .x > sum(dplyr::cur_group_id()))))
  f123_native(df, function(t) filter(t, if_all(c(x, y), ~ .x + sum(dplyr::cur_group_id()) > 2), .by = g))
  f123_native(df, function(t) filter(t, if_any(c(x, y), ~ .x > sum(dplyr::cur_group_id())), .by = g))
  f123_native(df, function(t) group_by(t, g) |> mutate(a = dplyr::if_any(c(x, y), ~ .x > sum(dplyr::cur_group_id()))))
})

test_that("F123 case 4: an integer zero under unary minus is +0", {
  skip_if_no_gpu()
  k <- 0L
  df <- tibble::tibble(x = c(5, -5, 0, 2.5, NA), big = 2^64, i = c(0L, 3L, -3L, NA, 0L))
  p <- function(t) {
    mutate(t,
      a = x %/% -0L, b = x %/% -k, c = x %% -0L, g = -0L %% big, h = 1 / (-0L %% big),
      e = 1 / (i %/% -big), f = 1 / (-k %% big)
    )
  }
  f123_native(df, p)
  f123_same_mask(df, p, c("a", "b", "c", "g", "h", "e", "f"))
})

test_that("F123 case 4: %/% of large doubles is R's whole number", {
  skip_if_no_gpu()
  df <- f123_mod_grid(
    c(13510798882111490, 9007199254740994, 1e17, -13510798882111490, 4503599627370497,
      2^62 + 2048, -9007199254740994, 2^63 * 3),
    c(3, 1.5, -3, 7, 0.75, -1.5)
  )
  p <- function(t) mutate(t, q = x %/% y)
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))$q
    expect_identical(got, p(df)$q, info = sprintf("lazy = %s", lazy))
  }
  f123_same_mask(df, p, "q")
})
