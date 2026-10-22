# F55: one NaN rule on the GPU. NaN is a value (arithmetic, collect, is.nan(),
# group and join keys keep it); is.na(), the six comparisons, coalesce(),
# arrange() keys, cummax()/cummin() and na_matches = "never" join keys read it
# as missing, as dplyr/vctrs do.
#
# expect_equal() treats NaN and NA as equal, so f55_same() also compares
# is.nan() of every double column of the oracle and GPU results.

f55_df <- function() {
  # x / y is NaN in rows 1 and 5, NA in row 4.
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L),
    x = c(0, 1, 2, NA, 0, 3),
    y = c(0, 1, 1, 1, 0, 2)
  )
}

# w holds uploaded NaN and NA (never both in one g group, see sum()'s
# operand-order rule in the brief).
f55_up <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L),
    w = c(NaN, 1, NA, 2, 3, 4, NaN, NaN),
    i = c(1L, NA, 3L, 4L, 5L, 6L, 7L, 8L)
  )
}

# `native = TRUE` pins the eager run to the GPU (`cuplyr.fallback = "error"`).
# The lazy run isn't pinned: a lazy summarise() that builds a temp column
# computes the pending ops first and reports that through the same notice
# (notify_materialization(), R/summarise.R), with or without a NaN.
f55_same <- function(df, pipeline, native = TRUE, nan_skip = character(),
                     arrange_by = NULL) {
  for (lazy in c(FALSE, TRUE)) {
    run <- function() {
      if (lazy) {
        expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
      } else {
        expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
      }
    }
    res <- if (native && !lazy) {
      withr::with_options(list(cuplyr.fallback = "error"), run())
    } else {
      run()
    }
    for (nm in setdiff(names(res$oracle), nan_skip)) {
      if (is.double(res$oracle[[nm]])) {
        expect_identical(
          is.nan(res$gpu[[nm]]), is.nan(res$oracle[[nm]]),
          info = sprintf("is.nan(%s), %s", nm, if (lazy) "lazy" else "eager")
        )
      }
    }
  }
}

# -----------------------------------------------------------------------------
# B8 (F53 round 4): the silent-wrong cases, natively
# -----------------------------------------------------------------------------

test_that("B8: is.na(), a comparison and coalesce() of a GPU NaN in one mutate()", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) {
    dplyr::mutate(d,
      miss = is.na(x / y), nan = is.nan(x / y),
      big = x / y > 1, r0 = dplyr::coalesce(x / y, 0)
    )
  })
  # U2: the same without the is.nan() dot
  f55_same(df, function(d) {
    dplyr::mutate(d, miss = is.na(x / y), big = x / y > 1, r0 = dplyr::coalesce(x / y, 0))
  })
  f55_same(df, function(d) {
    dplyr::transmute(d, miss = is.na(x / y), big = x / y > 1, r0 = dplyr::coalesce(x / y, 0))
  })
  # grouped and .by
  f55_same(df, function(d) {
    dplyr::mutate(dplyr::group_by(d, g), miss = is.na(x / y), big = x / y > 1, r0 = dplyr::coalesce(x / y, 0))
  })
  f55_same(df, function(d) {
    dplyr::mutate(d, miss = is.na(x / y), nan = is.nan(x / y), .by = g)
  })
})

test_that("B8: sum(is.na(x / y)) and sum(is.nan(x / y)) count the NaN rows", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) {
    dplyr::summarise(d, n_na = sum(is.na(x / y)), n_nan = sum(is.nan(x / y)))
  })
  f55_same(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
      n_na = sum(is.na(x / y)), n_nan = sum(is.nan(x / y)),
      n_big = sum(x / y > 1, na.rm = TRUE), any_big = any(x / y > 1), all_pos = all(x / y >= 0)
    )
  })
})

test_that("B8: filter() drops the rows where a NaN comparison is NA", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) dplyr::filter(d, x / y > 1))
  f55_same(df, function(d) dplyr::filter(d, x / y != 1))
  f55_same(df, function(d) dplyr::filter(d, !(x / y > 1)))
  f55_same(df, function(d) dplyr::filter(d, xor(x / y > 0.5, TRUE)))
  f55_same(df, function(d) dplyr::filter(d, is.na(x / y)))
  f55_same(df, function(d) dplyr::filter(d, !is.na(x / y)))
  f55_same(df, function(d) dplyr::filter(d, !is.nan(x / y) | is.na(x / y)))
  f55_same(df, function(d) dplyr::filter(dplyr::group_by(d, g), !(x / y < 1)))
})

# -----------------------------------------------------------------------------
# The rule on a stored NaN column (computed by an earlier mutate())
# -----------------------------------------------------------------------------

test_that("all six comparisons on a stored NaN are NA, either side", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) {
    d <- dplyr::mutate(d, r = x / y)
    dplyr::mutate(d,
      gt = r > 1, ge = r >= 1, lt = r < 1, le = r <= 1, eq = r == 1, ne = r != 1,
      rev = 1 < r, self_eq = r == r, self_ne = r != r, vs_col = r != x, vs_int = r > g
    )
  })
  f55_same(df, function(d) dplyr::filter(dplyr::mutate(d, r = x / y), r != 1))
  f55_same(df, function(d) dplyr::filter(dplyr::mutate(d, r = x / y), !(r >= 1)))
  f55_same(df, function(d) dplyr::filter(dplyr::mutate(d, r = x / y), r == r))
})

test_that("if_else(), case_when(), between(), near() conditions see NaN as missing", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) {
    d <- dplyr::mutate(d, r = x / y)
    dplyr::mutate(d,
      a = dplyr::if_else(r > 0.5, "hi", "lo", missing = "m"),
      a2 = dplyr::if_else(r > 0.5, 1, 0),
      b = dplyr::case_when(r > 1 ~ "big", is.na(r) ~ "miss", .default = "small"),
      c = dplyr::between(r, 0.5, 1.5),
      e = dplyr::near(r, 1),
      f = dplyr::if_else(is.na(r), 0, r)
    )
  })
})

test_that("coalesce() treats NaN as missing in every argument", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) {
    d <- dplyr::mutate(d, r = x / y)
    dplyr::mutate(d,
      c1 = dplyr::coalesce(r, 0),
      c2 = dplyr::coalesce(r, x / y, -1),
      c3 = dplyr::coalesce(r, r),
      c4 = dplyr::coalesce(x, r),
      c5 = dplyr::coalesce(r, y)
    )
  })
})

test_that("across() and if_any() apply the rule per column", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) {
    d <- dplyr::mutate(d, r = x / y, q = y / x)
    dplyr::mutate(d, dplyr::across(c(r, q), list(na = is.na, nan = is.nan, big = ~ .x > 1)))
  })
  f55_same(df, function(d) {
    d <- dplyr::mutate(d, r = x / y, q = y / x)
    dplyr::summarise(d, dplyr::across(c(r, q), ~ sum(is.na(.x))))
  })
  f55_same(df, function(d) {
    d <- dplyr::mutate(d, r = x / y, q = y / x)
    dplyr::filter(d, dplyr::if_any(c(r, q), ~ .x > 1))
  })
})

test_that("count() and group_by() keyed by a NaN comparison keep the NA group", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) dplyr::count(d, k = x / y > 1))
  f55_same(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, k = x / y > 0.5), n = dplyr::n())
  })
})

# -----------------------------------------------------------------------------
# is.nan() is native
# -----------------------------------------------------------------------------

test_that("is.nan() runs on the GPU for every verb that takes an expression", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) dplyr::mutate(d, a = is.nan(x / y), b = is.nan(x), c = is.nan(g), e = is.nan(x > 1)))
  f55_same(df, function(d) dplyr::filter(d, is.nan(x / y)))
  f55_same(df, function(d) dplyr::summarise(d, n = sum(is.nan(x / y)), .by = g))
  f55_same(df, function(d) dplyr::count(d, nan = is.nan(x / y)))
  f55_same(df, function(d) dplyr::distinct(d, nan = is.nan(x / y)))
  f55_same(df, function(d) dplyr::count(d, g, wt = is.nan(x / y)))
  f55_same(df, function(d) {
    dplyr::mutate(d, r = x / y, z = dplyr::case_when(is.nan(r) ~ "nan", r > 1 ~ "big", .default = "other"))
  })
})

test_that("is.nan() is FALSE for NA and for non-double columns", {
  skip_if_no_gpu()
  df <- tibble::tibble(i = c(1L, NA), l = c(TRUE, NA), d = as.Date(c("2020-01-01", NA)), x = c(NA, 1))
  f55_same(df, function(d) {
    dplyr::mutate(d, ni = is.nan(i), nl = is.nan(l), nd = is.nan(d), nx = is.nan(x))
  })
})

# -----------------------------------------------------------------------------
# NaN literals and environment values reach the device as NaN
# -----------------------------------------------------------------------------

test_that("a NaN literal or env value is a NaN on the GPU, not NA", {
  skip_if_no_gpu()
  df <- f55_df()
  th <- NaN
  f55_same(df, function(d) dplyr::mutate(d, v = x * th, z = is.nan(v)))
  f55_same(df, function(d) dplyr::mutate(d, v = dplyr::if_else(x > 1, NaN, x), z = is.nan(v)))
  f55_same(df, function(d) dplyr::mutate(d, v = NaN, z = is.nan(v)))
  f55_same(df, function(d) dplyr::mutate(d, v = 0 / 0, z = is.nan(v)))
  f55_same(df, function(d) dplyr::transmute(d, v = x * th, z = is.nan(v)))
  f55_same(df, function(d) dplyr::mutate(dplyr::mutate(d, v = x * th), z = is.nan(v)))
  f55_same(df, function(d) dplyr::count(d, v = y * th, wt = is.nan(v)))
})

test_that("a comparison with a NaN literal is NA", {
  skip_if_no_gpu()
  df <- f55_df()
  th <- NaN
  f55_same(df, function(d) {
    dplyr::mutate(d, a = x > NaN, b = x != th, c = x == NaN, e = dplyr::coalesce(th, x), f = is.na(th + x))
  })
  f55_same(df, function(d) dplyr::filter(d, x != th))
  f55_same(df, function(d) dplyr::filter(d, !(x < th)))
})

# -----------------------------------------------------------------------------
# Upload and collect keep NaN distinct from NA
# -----------------------------------------------------------------------------

test_that("tbl_gpu() and collect() keep NaN and NA apart", {
  skip_if_no_gpu()
  df <- f55_up()
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(tbl_gpu(df, lazy = lazy))
    expect_identical(is.nan(got$w), is.nan(df$w))
    expect_identical(is.na(got$w), is.na(df$w))
  }
  f55_same(df, function(d) dplyr::mutate(d, na = is.na(w), nan = is.nan(w), w2 = w * 2, big = w > 1))
  f55_same(df, function(d) dplyr::filter(d, !is.na(w)))
  f55_same(df, function(d) dplyr::filter(d, w != 1))
  f55_same(df, function(d) dplyr::mutate(d, c = dplyr::coalesce(w, -1)))
})

test_that("a CPU-fallback result keeps a NaN dplyr computed", {
  skip_if_no_gpu()
  df <- f55_up()
  f55_same(df, function(d) dplyr::mutate(d, p = pmin(w, 2)), native = FALSE)
  f55_same(df, function(d) dplyr::mutate(d, m = mean(w, na.rm = TRUE), .by = g), native = FALSE)
})

# -----------------------------------------------------------------------------
# Aggregates over uploaded and computed NaN
# -----------------------------------------------------------------------------

test_that("sum/mean propagate NaN, na.rm drops it, n_distinct keeps it apart", {
  skip_if_no_gpu()
  df <- f55_up()
  f55_same(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
      s = sum(w), m = mean(w), s2 = sum(w, na.rm = TRUE), m2 = mean(w, na.rm = TRUE),
      nd = dplyr::n_distinct(w), n_na = sum(is.na(w)), n_nan = sum(is.nan(w))
    )
  })
  # min/max: na.rm = TRUE drops NaN; without it cuplyr gives NA where R gives
  # NaN (F22's accepted divergence), so is.nan() isn't compared for those.
  df2 <- dplyr::filter(df, g != 4L)
  f55_same(df2, function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
      mn = min(w, na.rm = TRUE), mx = max(w, na.rm = TRUE), mn0 = min(w), mx0 = max(w)
    )
  }, nan_skip = c("mn0", "mx0"))
  f55_same(df, function(d) dplyr::summarise(d, s = sum(w, na.rm = TRUE), m = mean(w), n = sum(is.nan(w))))
})

# -----------------------------------------------------------------------------
# Ordering and keys
# -----------------------------------------------------------------------------

test_that("arrange() ties NaN with NA, last, in both directions", {
  skip_if_no_gpu()
  df <- f55_df()
  f55_same(df, function(d) dplyr::arrange(dplyr::mutate(d, r = x / y), r))
  f55_same(df, function(d) dplyr::arrange(dplyr::mutate(d, r = x / y), dplyr::desc(r)))
  f55_same(df, function(d) dplyr::arrange(dplyr::mutate(d, r = x / y), dplyr::desc(r), g))
  f55_same(df, function(d) {
    dplyr::arrange(dplyr::group_by(dplyr::mutate(d, r = x / y), g), dplyr::desc(r), .by_group = TRUE)
  })
  up <- f55_up()
  f55_same(up, function(d) dplyr::arrange(d, w))
  f55_same(up, function(d) dplyr::arrange(d, dplyr::desc(w)))
})

test_that("count(), distinct() and group_by() keep NaN and NA as separate keys", {
  skip_if_no_gpu()
  df <- f55_up()
  f55_same(df, function(d) dplyr::count(d, w))
  f55_same(df, function(d) dplyr::distinct(d, w))
  f55_same(df, function(d) dplyr::summarise(dplyr::group_by(d, w), n = dplyr::n()))
})

test_that("joins: NaN matches NaN only; na_matches = 'never' matches neither", {
  skip_if_no_gpu()
  a <- tibble::tibble(k = c(NaN, NA, 1, NaN), a = 1:4)
  b <- tibble::tibble(k = c(NA, NaN, 1), b = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    ga <- tbl_gpu(a, lazy = lazy)
    gb <- tbl_gpu(b, lazy = lazy)
    chk <- function(got, want) {
      got <- tibble::as_tibble(collect(got))
      expect_equal(got, want)
      expect_identical(is.nan(got$k), is.nan(want$k))
    }
    chk(dplyr::left_join(ga, gb, by = "k"), dplyr::left_join(a, b, by = "k"))
    chk(dplyr::inner_join(ga, gb, by = "k"), dplyr::inner_join(a, b, by = "k"))
    chk(dplyr::inner_join(ga, gb, by = "k", na_matches = "never"),
        dplyr::inner_join(a, b, by = "k", na_matches = "never"))
    chk(dplyr::left_join(ga, gb, by = "k", na_matches = "never"),
        dplyr::left_join(a, b, by = "k", na_matches = "never"))
    chk(dplyr::semi_join(ga, gb, by = "k", na_matches = "never"),
        dplyr::semi_join(a, b, by = "k", na_matches = "never"))
    chk(dplyr::anti_join(ga, gb, by = "k", na_matches = "never"),
        dplyr::anti_join(a, b, by = "k", na_matches = "never"))
  }
})

test_that("cummax()/cummin() read NaN as missing", {
  skip_if_no_gpu()
  df <- f55_df()
  # cuplyr gives NA where R gives NaN (accepted NaN-as-NA value divergence),
  # so only values are compared for these columns.
  f55_same(df, function(d) {
    dplyr::mutate(dplyr::mutate(d, r = x / y), a = cummax(r), b = cummin(r))
  }, nan_skip = c("a", "b"))
  f55_same(f55_up(), function(d) dplyr::mutate(d, a = cummax(w), b = cummin(w)), nan_skip = c("a", "b"))
})

# -----------------------------------------------------------------------------
# Round-1 review findings (reviews/round-1.md B1-B6, N1)
# -----------------------------------------------------------------------------

f55_wagg <- function() tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L), w = c(1, NaN, 2, 3, 4))

test_that("B1: window min/max/median/sd/var over a NaN group are missing", {
  skip_if_no_gpu()
  df <- f55_wagg()
  # max()/min() without na.rm: cuplyr gives NA where R gives NaN (F22's
  # accepted divergence), so is.nan() isn't compared for those.
  f55_same(df, function(d) {
    dplyr::mutate(d, mx = max(w), mn = min(w), md = median(w), s = sd(w), v = var(w), .by = g)
  }, nan_skip = c("mx", "mn"))
  f55_same(df, function(d) {
    dplyr::mutate(dplyr::group_by(d, g), mx = max(w), md = median(w), s = sd(w))
  }, nan_skip = "mx")
  f55_same(df, function(d) dplyr::mutate(d, mx = max(w), mn = min(w), md = median(w)), nan_skip = c("mx", "mn"))
  f55_same(df, function(d) dplyr::filter(d, w == max(w), .by = g))
  f55_same(df, function(d) dplyr::filter(d, w > median(w), .by = g))
})

test_that("B2: summarise()/reframe() sd()/var() of a NaN group are NA", {
  skip_if_no_gpu()
  df <- f55_wagg()
  f55_same(df, function(d) dplyr::summarise(d, s = sd(w), v = var(w), .by = g))
  f55_same(df, function(d) dplyr::summarise(dplyr::group_by(d, g), s = sd(w), v = var(w)))
  f55_same(df, function(d) dplyr::reframe(d, s = sd(w), v = var(w), .by = g))
  f55_same(df, function(d) dplyr::summarise(d, s = sd(w), v = var(w)))
})

test_that("B3/B4/N1: na_if(), consecutive_id() and %in% treat NaN as equal to NaN, apart from NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(w = c(NaN, 1, NA, NaN, 2), y = c(NaN, 1, NaN, 0, NA), i = c(1L, NA, 3L, 4L, 5L))
  f55_same(df, function(d) dplyr::mutate(d, a = dplyr::na_if(w, NaN), b = dplyr::na_if(w, y)))
  f55_same(df, function(d) dplyr::mutate(d, a = dplyr::na_if(y / y, NaN), b = dplyr::na_if(w, 0 / 0)))
  f55_same(df, function(d) dplyr::mutate(d, id = dplyr::consecutive_id(w), id2 = dplyr::consecutive_id(y / y)))
  f55_same(df, function(d) {
    dplyr::mutate(d, i1 = w %in% c(NaN, 1), i2 = w %in% c(NA, 2), i3 = i %in% c(NaN, 1), i4 = (y / y) %in% NaN)
  })
})

test_that("B5: order_by= frames tie NaN with NA, last", {
  skip_if_no_gpu()
  df <- tibble::tibble(w = c(NA, NaN, 1), x = c(1, 2, 3))
  f55_same(df, function(d) {
    dplyr::mutate(d,
      l = dplyr::lag(x, order_by = w), ld = dplyr::lead(x, order_by = w),
      la = dplyr::last(x, order_by = w)
    )
  })
  df2 <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L, 2L), u = c(0, 1, 2, 0, 1, 0), v = c(0, 1, NA, 0, 1, 0), x = 1:6 * 1.5)
  f55_same(df2, function(d) {
    dplyr::mutate(d, r = u / v, l = dplyr::lag(x, order_by = r), f = dplyr::first(x, order_by = r), .by = g)
  })
})

test_that("B6: a mixed join_by() under na_matches = 'never' doesn't match NaN keys", {
  skip_if_no_gpu()
  a <- tibble::tibble(k = c(NaN, NA, 1), v = c(5, 5, 5))
  b <- tibble::tibble(k = c(NaN, NA, 1), u = c(1, 1, 1))
  for (lazy in c(FALSE, TRUE)) {
    ga <- tbl_gpu(a, lazy = lazy)
    gb <- tbl_gpu(b, lazy = lazy)
    chk <- function(got, want) {
      got <- tibble::as_tibble(collect(got))
      expect_equal(got, want)
      expect_identical(is.nan(got$k), is.nan(want$k))
    }
    jb <- dplyr::join_by(k == k, v >= u)
    chk(dplyr::inner_join(ga, gb, jb, na_matches = "never"), dplyr::inner_join(a, b, jb, na_matches = "never"))
    chk(dplyr::left_join(ga, gb, jb, na_matches = "never"), dplyr::left_join(a, b, jb, na_matches = "never"))
    chk(dplyr::inner_join(ga, gb, jb), dplyr::inner_join(a, b, jb))
    chk(dplyr::semi_join(ga, gb, jb, na_matches = "never"), dplyr::semi_join(a, b, jb, na_matches = "never"))
  }
})

# -----------------------------------------------------------------------------
# Round-2 review finding (reviews/round-2.md B7)
# -----------------------------------------------------------------------------

test_that("B7: roll_*() read NaN as missing, under both na.rm settings", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L),
    x = c(1, NaN, 3, 4, NaN, NaN, 6, NA)
  )
  f55_same(df, function(d) {
    dplyr::mutate(d,
      rmean = roll_mean(x, 2, na.rm = TRUE), rsum = roll_sum(x, 2, na.rm = TRUE),
      rmax = roll_max(x, 2), rmin_rm = roll_min(x, 2, na.rm = TRUE),
      rsd = roll_sd(x, 2), rmax_rm = roll_max(x, 2, na.rm = TRUE),
      rmean0 = roll_mean(x, 3), rsum0 = roll_sum(x, 2)
    )
  })
  f55_same(df, function(d) {
    dplyr::mutate(d, rmean = roll_mean(x, 2, na.rm = TRUE), rmin = roll_min(x, 2), .by = g)
  })
  f55_same(df, function(d) dplyr::filter(d, roll_sum(x, 2, na.rm = TRUE) > 2))
  f55_same(df, function(d) {
    dplyr::mutate(d, r = x / (x - x + 1), rr = roll_mean(r, 2, na.rm = TRUE))
  })
})
