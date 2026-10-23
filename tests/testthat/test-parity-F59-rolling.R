# F59 acceptance tests: rolling kernels whose cost doesn't depend on the
# window width (roll_sum/roll_mean/roll_sd, optionally roll_min/roll_max).
#
# Contract (scratchpad/parity/features/F59.md):
#   - semantics are R/roll.R's reference (base R over each right-aligned
#     window), frozen below as f59_ref() so a change to R/roll.R can't move
#     the goalposts: leading n - 1 rows of each group NA; na.rm = FALSE -> any
#     NA in the window gives NA; na.rm = TRUE -> NAs dropped, an all-NA window
#     is NA (sd needs 2 values); NaN counts as missing (F55); Inf follows base
#     R (sum/mean +-Inf or NaN, sd NaN);
#   - accuracy: all.equal(tolerance = 1e-8) against data.table's default
#     froll*() on the benchmark data, and against data.table's
#     algo = "exact" on adversarial data (data.table's fast frollsd itself
#     misses 1e-8 there, see the brief);
#   - width scaling is flat for mean/sum/sd.

f59_ref <- function(x, n, stat, na.rm = FALSE) {
  stat_fn <- switch(stat,
    mean = base::mean, sum = base::sum, min = base::min, max = base::max,
    sd = stats::sd, median = stats::median
  )
  n <- as.integer(n)
  nx <- length(x)
  out <- rep(NA_real_, nx)
  if (nx == 0 || n > nx) {
    return(out)
  }
  x <- as.double(x)
  for (i in seq.int(n, nx)) {
    w <- x[(i - n + 1L):i]
    if (na.rm) {
      w <- w[!is.na(w)]
      if (length(w) == 0) next
    } else if (anyNA(w)) {
      next
    }
    out[i] <- stat_fn(w)
  }
  out
}

# Per-group reference in original row order; NA is a group of its own.
f59_ref_grouped <- function(x, g, n, stat, na.rm = FALSE, engine = f59_ref) {
  out <- rep(NA_real_, length(x))
  for (idx in split(seq_along(x), factor(g, exclude = NULL))) {
    out[idx] <- engine(x[idx], n, stat, na.rm)
  }
  out
}

# NA/NaN/Inf positions must match exactly; finite values to `tol` with
# all.equal()'s mean relative difference (the benchmark's check).
f59_check <- function(got, ref, info, tol = 1e-8) {
  got <- as.double(got)
  expect_identical(length(got), length(ref), info = info)
  expect_identical(is.na(got), is.na(ref), info = paste(info, "(NA positions)"))
  expect_identical(is.nan(got), is.nan(ref), info = paste(info, "(NaN positions)"))
  inf <- !is.na(ref) & is.infinite(ref)
  expect_identical(got[inf], ref[inf], info = paste(info, "(Inf values)"))
  fin <- !is.na(ref) & is.finite(ref)
  if (any(fin)) {
    eq <- all.equal(got[fin], ref[fin], tolerance = tol)
    expect_true(isTRUE(eq), info = paste(info, ":", paste(eq, collapse = "; ")))
  }
}

f59_roll_call <- function(stat, n, na_rm, col = "x") {
  rlang::call2(paste0("roll_", stat), rlang::sym(col), n, na.rm = na_rm)
}

f59_gpu_roll <- function(df, stat, n, na_rm, lazy = FALSE, group = NULL) {
  gt <- tbl_gpu(df, lazy = lazy)
  if (!is.null(group)) gt <- dplyr::group_by(gt, !!!rlang::syms(group))
  gt |>
    dplyr::mutate(y = !!f59_roll_call(stat, n, na_rm)) |>
    collect() |>
    dplyr::pull(y)
}

f59_stats <- c("mean", "sum", "sd", "min", "max")

# The values exercising every special case at once: Inf of both signs, NA,
# NaN, runs of NA, repeated values, a zero.
f59_edge_x <- c(1, Inf, 3, -Inf, 5, NA, NaN, 8, Inf, Inf, -Inf, 2, 2, 2, 7,
                NA, NA, NA, 4, 1e6, -1e6, 0)

# =============================================================================
# Semantics against the frozen reference
# =============================================================================

test_that("F59: roll_*() match the R reference on edge values, ungrouped, eager and lazy", {
  skip_if_no_gpu()
  df <- data.frame(x = f59_edge_x)
  nx <- nrow(df)
  for (lazy in c(FALSE, TRUE)) {
    for (stat in f59_stats) {
      for (w in c(1L, 2L, 3L, 5L, nx - 1L, nx, nx + 1L)) {
        for (na_rm in c(FALSE, TRUE)) {
          info <- sprintf("roll_%s(x, %d, na.rm = %s), lazy = %s", stat, w, na_rm, lazy)
          f59_check(f59_gpu_roll(df, stat, w, na_rm, lazy = lazy),
                    f59_ref(df$x, w, stat, na_rm), info)
        }
      }
    }
  }
})

test_that("F59: roll_*() match the R reference with interleaved groups, NA keys, short and all-NA groups", {
  skip_if_no_gpu()
  set.seed(5901)
  n <- 400L
  # group 9 has one row, group 8 is all NA, NA is a group of its own,
  # group 7 is shorter than every width but 1/2.
  g <- sample(c(1:6, NA), n, replace = TRUE)
  x <- round(rnorm(n, 50, 10), 2)
  x[sample.int(n, 30)] <- NA
  x[sample.int(n, 10)] <- NaN
  x[sample.int(n, 4)] <- Inf
  x[sample.int(n, 4)] <- -Inf
  g <- c(g, 9L, 8L, 8L, 8L, 7L, 7L)
  x <- c(x, 3, NA, NaN, NA, 1, 2)
  # interleave the appended rows with the rest
  perm <- sample.int(length(x))
  df <- data.frame(g = g[perm], x = x[perm])
  for (lazy in c(FALSE, TRUE)) {
    for (stat in f59_stats) {
      for (w in c(1L, 2L, 3L, 7L, 25L)) {
        for (na_rm in c(FALSE, TRUE)) {
          info <- sprintf("grouped roll_%s(x, %d, na.rm = %s), lazy = %s", stat, w, na_rm, lazy)
          f59_check(f59_gpu_roll(df, stat, w, na_rm, lazy = lazy, group = "g"),
                    f59_ref_grouped(df$x, df$g, w, stat, na_rm), info)
        }
      }
    }
  }
})

test_that("F59: several roll specs in one mutate (shared column, mixed widths and na.rm, .by)", {
  skip_if_no_gpu()
  set.seed(5902)
  df <- data.frame(g = sample(1:4, 300, TRUE), x = rnorm(300), z = rnorm(300, 1e6))
  df$x[sample.int(300, 20)] <- NA
  pipe <- function(d) {
    dplyr::mutate(d,
      a = roll_mean(x, 3), b = roll_sd(x, 5, na.rm = TRUE), c = roll_sum(x, 3, na.rm = TRUE),
      d = roll_sd(z, 3), e = roll_mean(z, 40), f = roll_max(x, 4, na.rm = TRUE),
      h = roll_min(z, 6), .by = g
    )
  }
  expect_same_as_dplyr(df, pipe)
  expect_same_as_dplyr_lazy(df, pipe)
  pipe_ungrouped <- function(d) {
    dplyr::mutate(d, a = roll_sd(x, 4), b = roll_sum(z, 50), c = roll_mean(x, 2, na.rm = TRUE))
  }
  expect_same_as_dplyr(df, pipe_ungrouped)
  expect_same_as_dplyr_lazy(df, pipe_ungrouped)
})

test_that("F59: roll_*() inside filter() take the same path", {
  skip_if_no_gpu()
  set.seed(5903)
  df <- data.frame(g = sample(1:3, 200, TRUE), x = rnorm(200))
  df$x[c(5, 50, 51)] <- NA
  pipe <- function(d) dplyr::filter(d, roll_sd(x, 3) > 0.5, roll_sum(x, 4, na.rm = TRUE) < 2)
  expect_same_as_dplyr(df, pipe)
  expect_same_as_dplyr_lazy(df, pipe)
  pipe_g <- function(d) dplyr::filter(dplyr::group_by(d, g), roll_mean(x, 5, na.rm = TRUE) > 0)
  expect_same_as_dplyr(df, pipe_g)
  expect_same_as_dplyr_lazy(df, pipe_g)
})

test_that("F59: 0-row input and windows wider than the table/group", {
  skip_if_no_gpu()
  df0 <- data.frame(g = integer(0), x = numeric(0))
  for (stat in f59_stats) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f59_gpu_roll(df0, stat, 3L, FALSE, lazy = lazy)
      expect_identical(length(got), 0L, info = paste(stat, lazy))
      got_g <- f59_gpu_roll(df0, stat, 3L, TRUE, lazy = lazy, group = "g")
      expect_identical(length(got_g), 0L, info = paste(stat, lazy, "grouped"))
    }
  }
  df <- data.frame(g = c(1L, 1L, 2L, 2L, 2L), x = c(1, 2, 3, 4, 5))
  for (stat in f59_stats) {
    got <- f59_gpu_roll(df, stat, 4L, TRUE, group = "g")
    expect_true(all(is.na(got)), info = stat)
    expect_false(any(is.nan(got)), info = stat)
  }
})

test_that("F59: input/output types are unchanged (integer, logical, Date, POSIXct)", {
  skip_if_no_gpu()
  set.seed(5904)
  # Integer sums must be exact, including values near the INT32 limits.
  xi <- sample(c(-2147483647L, 2147483647L, -5L, 0L, 7L, NA), 20000, TRUE)
  df <- data.frame(x = xi)
  for (w in c(3L, 1000L)) {
    for (na_rm in c(FALSE, TRUE)) {
      got <- f59_gpu_roll(df, "sum", w, na_rm)
      expect_true(is.double(got))
      expect_identical(got, f59_ref(xi, w, "sum", na_rm), info = sprintf("int sum w=%d na.rm=%s", w, na_rm))
      f59_check(f59_gpu_roll(df, "mean", w, na_rm), f59_ref(xi, w, "mean", na_rm),
                sprintf("int mean w=%d na.rm=%s", w, na_rm))
    }
  }
  got_min <- f59_gpu_roll(data.frame(x = c(3L, 1L, NA, 4L, 2L)), "min", 2L, TRUE)
  expect_identical(got_min, c(NA, 1L, 1L, 4L, 2L))
  got_max <- f59_gpu_roll(data.frame(x = c(3L, 1L, NA, 4L, 2L)), "max", 2L, FALSE)
  expect_identical(got_max, c(NA, 3L, NA, NA, 4L))

  xl <- c(TRUE, FALSE, NA, TRUE, TRUE, FALSE, TRUE)
  dfl <- data.frame(x = xl)
  for (na_rm in c(FALSE, TRUE)) {
    expect_identical(f59_gpu_roll(dfl, "sum", 3L, na_rm), f59_ref(xl, 3L, "sum", na_rm))
    f59_check(f59_gpu_roll(dfl, "mean", 2L, na_rm), f59_ref(xl, 2L, "mean", na_rm), "lgl mean")
    f59_check(f59_gpu_roll(dfl, "sd", 3L, na_rm), f59_ref(xl, 3L, "sd", na_rm), "lgl sd")
  }

  d <- data.frame(
    d = as.Date("2020-01-01") + c(3, 1, NA, 5),
    t = as.POSIXct("2020-01-01", tz = "UTC") + c(3, 1, NA, 5)
  )
  got <- tbl_gpu(d) |>
    dplyr::mutate(a = roll_max(d, 2), b = roll_min(t, 2, na.rm = TRUE)) |>
    collect()
  expect_identical(got$a, as.Date(c(NA, "2020-01-04", NA, NA)))
  expect_equal(got$b, as.POSIXct("2020-01-01", tz = "UTC") + c(NA, 1, 1, 5))
})

# =============================================================================
# Accuracy: all.equal(tolerance = 1e-8)
# =============================================================================

test_that("F59: benchmark data matches data.table's default froll*() at w = 11/101/1001", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  set.seed(12345)
  x <- rnorm(1e6) # bench/rolling-lib.R rw_data(1e6)
  gt <- tbl_gpu(data.frame(x = x))
  for (w in c(11L, 101L, 1001L)) {
    got <- gt |>
      dplyr::mutate(m = roll_mean(x, w), s = roll_sum(x, w), d = roll_sd(x, w),
                    lo = roll_min(x, w), hi = roll_max(x, w)) |>
      collect()
    f59_check(got$m, data.table::frollmean(x, w), sprintf("bench mean w=%d", w))
    f59_check(got$s, data.table::frollsum(x, w), sprintf("bench sum w=%d", w))
    f59_check(got$d, data.table::frollsd(x, w), sprintf("bench sd w=%d", w))
    expect_identical(got$lo, data.table::frollmin(x, w), info = sprintf("bench min w=%d", w))
    expect_identical(got$hi, data.table::frollmax(x, w), info = sprintf("bench max w=%d", w))
  }
})

# data.table's algo = "exact" with cuplyr's NA rule: under na.rm = TRUE a
# window with no non-missing value (sd: fewer than 2) is NA, where data.table
# gives 0/NaN.
f59_dt_exact <- function(x, w, stat, na_rm) {
  fn <- switch(stat, mean = data.table::frollmean, sum = data.table::frollsum,
               sd = data.table::frollsd)
  out <- fn(x, w, algo = "exact", na.rm = na_rm)
  if (na_rm) {
    cnt <- data.table::frollsum(as.double(!is.na(x)), w)
    out[!is.na(cnt) & cnt < (if (stat == "sd") 2 else 1)] <- NA_real_
  }
  out
}

f59_adversarial <- function() {
  set.seed(5905)
  n <- 1e6
  offset_na <- 1e9 + rnorm(n)
  offset_na[sample.int(n, n * 0.02)] <- NA
  list(
    offset = 1e9 + rnorm(n),
    walk = cumsum(rnorm(n)) + 100,
    trend = seq_len(n) * 1e3 + rnorm(n),
    mixedmag = rnorm(n) * 10^sample(-3:6, n, TRUE),
    offset_na = offset_na
  )
}

test_that("F59: adversarial data (large offsets, drift, mixed magnitudes, NAs) matches data.table exact", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  data <- f59_adversarial()
  for (nm in names(data)) {
    x <- data[[nm]]
    gt <- tbl_gpu(data.frame(x = x))
    for (w in c(2L, 11L, 1001L)) {
      xs <- if (w > 100L) x[seq_len(5e5)] else x # keeps the O(n * w) exact reference affordable
      gts <- if (w > 100L) tbl_gpu(data.frame(x = xs)) else gt
      for (na_rm in if (nm == "offset_na") c(FALSE, TRUE) else FALSE) {
        got <- gts |>
          dplyr::mutate(m = roll_mean(x, w, na.rm = na_rm), s = roll_sum(x, w, na.rm = na_rm),
                        d = roll_sd(x, w, na.rm = na_rm)) |>
          collect()
        for (stat in c("mean", "sum", "sd")) {
          col <- c(mean = "m", sum = "s", sd = "d")[[stat]]
          f59_check(got[[col]], f59_dt_exact(xs, w, stat, na_rm),
                    sprintf("%s roll_%s w=%d na.rm=%s", nm, stat, w, na_rm))
        }
      }
    }
  }
})

test_that("F59: a long (1e7) random walk and 1e9 offset stay accurate at small widths", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  set.seed(5906)
  n <- 1e7
  for (nm in c("walk", "offset")) {
    x <- if (nm == "walk") cumsum(rnorm(n)) + 100 else 1e9 + rnorm(n)
    got <- tbl_gpu(data.frame(x = x)) |>
      dplyr::mutate(m2 = roll_mean(x, 2), d2 = roll_sd(x, 2), m = roll_mean(x, 11),
                    s = roll_sum(x, 11), d = roll_sd(x, 11)) |>
      collect()
    f59_check(got$m2, f59_dt_exact(x, 2L, "mean", FALSE), paste(nm, "1e7 mean w=2"))
    f59_check(got$d2, f59_dt_exact(x, 2L, "sd", FALSE), paste(nm, "1e7 sd w=2"))
    f59_check(got$m, f59_dt_exact(x, 11L, "mean", FALSE), paste(nm, "1e7 mean w=11"))
    f59_check(got$s, f59_dt_exact(x, 11L, "sum", FALSE), paste(nm, "1e7 sum w=11"))
    f59_check(got$d, f59_dt_exact(x, 11L, "sd", FALSE), paste(nm, "1e7 sd w=11"))
    rm(x, got)
    gc()
  }
})

test_that("F59: grouped adversarial data (groups at very different levels) matches data.table exact per group", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  set.seed(5907)
  n <- 1e6
  g <- sample.int(50, n, TRUE)
  level <- sample(c(1e9, -5e8, 1e3, 0, 7e5), 50, TRUE)
  x <- level[g] + rnorm(n) * sample(c(1, 1e-3, 1e3), 50, TRUE)[g]
  x[sample.int(n, 5000)] <- NA
  df <- data.frame(g = g, x = x)
  exact_engine <- function(xx, w, stat, na_rm) f59_dt_exact(xx, w, stat, na_rm)
  for (lazy in c(FALSE, TRUE)) {
    for (w in c(11L, 1001L)) {
      for (na_rm in c(FALSE, TRUE)) {
        got <- tbl_gpu(df, lazy = lazy) |>
          dplyr::group_by(g) |>
          dplyr::mutate(m = roll_mean(x, w, na.rm = na_rm), s = roll_sum(x, w, na.rm = na_rm),
                        d = roll_sd(x, w, na.rm = na_rm)) |>
          collect()
        for (stat in c("mean", "sum", "sd")) {
          col <- c(mean = "m", sum = "s", sd = "d")[[stat]]
          f59_check(got[[col]], f59_ref_grouped(x, g, w, stat, na_rm, engine = exact_engine),
                    sprintf("grouped roll_%s w=%d na.rm=%s lazy=%s", stat, w, na_rm, lazy))
        }
      }
    }
  }
})

test_that("F59: constant runs give sd 0 (never NaN or negative) and exact means", {
  skip_if_no_gpu()
  x <- c(rep(1e9 + 0.1, 3000), rep(0.3, 3000), rep(-7.77e5, 3000))
  gt <- tbl_gpu(data.frame(x = x))
  for (w in c(2L, 11L, 1001L)) {
    got <- gt |> dplyr::mutate(d = roll_sd(x, w), m = roll_mean(x, w)) |> collect()
    ref_d <- f59_ref(x, w, "sd")
    expect_false(any(is.nan(got$d)), info = paste("w =", w))
    expect_true(all(got$d >= 0, na.rm = TRUE), info = paste("w =", w))
    f59_check(got$d, ref_d, paste("constant sd w =", w))
    f59_check(got$m, f59_ref(x, w, "mean"), paste("constant mean w =", w))
  }
})

test_that("F59: a window part with no finite value (na.rm = TRUE) is exact (review round 1, B1)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.1, 0.2, 0.3, NA, 1e17, 7))
  df2 <- data.frame(x = c(1e200, NA, NA, 0.1, 0.2, 0.3))
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(df, lazy = lazy) |>
      dplyr::mutate(m = roll_mean(x, 3, na.rm = TRUE), s = roll_sum(x, 3, na.rm = TRUE)) |>
      collect()
    f59_check(got$m, f59_ref(df$x, 3, "mean", TRUE), paste("B1 mean lazy =", lazy), tol = 1e-14)
    f59_check(got$s, f59_ref(df$x, 3, "sum", TRUE), paste("B1 sum lazy =", lazy), tol = 1e-14)
    got2 <- tbl_gpu(df2, lazy = lazy) |>
      dplyr::mutate(d = roll_sd(x, 3, na.rm = TRUE), m = roll_mean(x, 3, na.rm = TRUE)) |>
      collect()
    f59_check(got2$d, f59_ref(df2$x, 3, "sd", TRUE), paste("B1 sd lazy =", lazy), tol = 1e-12)
    f59_check(got2$m, f59_ref(df2$x, 3, "mean", TRUE), paste("B1 mean2 lazy =", lazy), tol = 1e-14)
  }
})

# Per-element bound (review round 1, N2): all.equal()'s mean relative
# difference can hide a few bad rows in a long vector. Every row of a sum
# or mean must be within 1e-10 of the window's own magnitude (sum of |x|
# over the window's non-missing values), whatever ran before the window.
test_that("F59: every row of roll_sum/roll_mean is accurate relative to its own window (mixed magnitudes, NA)", {
  skip_if_no_gpu()
  set.seed(5912)
  n <- 20000L
  x <- ifelse(stats::runif(n) < 0.5, stats::rnorm(n), 1e9 + stats::rnorm(n))
  x[sample.int(n, n / 10)] <- NA
  g <- rep(1:4, each = n / 4)[sample.int(n)]
  df <- data.frame(g = g, x = x)
  for (grouped in c(FALSE, TRUE)) {
    for (w in c(3L, 11L, 300L, 2500L)) {
      for (na_rm in c(FALSE, TRUE)) {
        gt <- tbl_gpu(df)
        if (grouped) gt <- dplyr::group_by(gt, g)
        got <- gt |>
          dplyr::mutate(m = roll_mean(x, w, na.rm = na_rm), s = roll_sum(x, w, na.rm = na_rm)) |>
          collect()
        gkey <- if (grouped) g else rep(1L, n)
        ref_s <- f59_ref_grouped(x, gkey, w, "sum", na_rm)
        ref_m <- f59_ref_grouped(x, gkey, w, "mean", na_rm)
        scale_s <- f59_ref_grouped(abs(x), gkey, w, "sum", na_rm)
        scale_m <- f59_ref_grouped(abs(x), gkey, w, "mean", na_rm)
        info <- sprintf("grouped=%s w=%d na.rm=%s", grouped, w, na_rm)
        expect_identical(is.na(got$s), is.na(ref_s), info = info)
        expect_identical(is.na(got$m), is.na(ref_m), info = info)
        ok <- !is.na(ref_s)
        err_s <- abs(got$s[ok] - ref_s[ok]) / scale_s[ok]
        err_m <- abs(got$m[ok] - ref_m[ok]) / scale_m[ok]
        expect_lte(max(c(0, err_s)), 1e-10, label = paste("max row error, sum,", info))
        expect_lte(max(c(0, err_m)), 1e-10, label = paste("max row error, mean,", info))
      }
    }
  }
})

# The tests above use narrow windows, which may run on cudf's kernel. These
# use widths above every crossover the implementation may choose, so the
# width-independent path is pinned to the same reference (review round 2,
# N5), including several specs over one column in every order (B5).
f59_wide_x <- function(n, specials = TRUE, seed = 5913) {
  set.seed(seed)
  x <- round(stats::rnorm(n, 10, 3), 3)
  if (specials) {
    # specials only in a few short stretches, so most wide windows stay valid
    x[c(1500, 1503)] <- NA
    x[1510] <- NaN
    x[c(2400, 2401)] <- Inf
    x[2405] <- -Inf
    x[3300] <- Inf
    x[3301:3310] <- NA
  }
  x
}

test_that("F59: wide windows (above any crossover) match the R reference, with specials, grouped and ungrouped", {
  skip_if_no_gpu()
  n <- 4000L
  x <- f59_wide_x(n)
  g <- rep(c(1L, 2L, NA), length.out = n)  # three interleaved groups of ~1333 rows
  df <- data.frame(g = g, x = x)
  for (lazy in c(FALSE, TRUE)) {
    for (w in c(300L, 1001L)) {
      for (na_rm in c(FALSE, TRUE)) {
        for (grouped in c(FALSE, TRUE)) {
          gt <- tbl_gpu(df, lazy = lazy)
          if (grouped) gt <- dplyr::group_by(gt, g)
          got <- gt |>
            dplyr::mutate(
              s = roll_sum(x, w, na.rm = na_rm), m = roll_mean(x, w, na.rm = na_rm),
              d = roll_sd(x, w, na.rm = na_rm), lo = roll_min(x, w, na.rm = na_rm),
              hi = roll_max(x, w, na.rm = na_rm)
            ) |>
            collect()
          gkey <- if (grouped) g else rep(1L, n)
          for (stat in c("sum", "mean", "sd", "min", "max")) {
            col <- c(sum = "s", mean = "m", sd = "d", min = "lo", max = "hi")[[stat]]
            f59_check(got[[col]], f59_ref_grouped(x, gkey, w, stat, na_rm),
                      sprintf("wide roll_%s w=%d na.rm=%s grouped=%s lazy=%s", stat, w, na_rm, grouped, lazy))
          }
        }
      }
    }
  }
})

test_that("F59: several specs over one column at a wide width, in every order (review round 2, B5)", {
  skip_if_no_gpu()
  n <- 4000L  # f59_wide_x() places specials up to row 3310
  w <- 400L
  set.seed(5914)
  for (specials in c(FALSE, TRUE)) {
    x <- f59_wide_x(n, specials = specials)
    g <- sample(1:2, n, TRUE)
    df <- data.frame(g = g, x = x)
    exprs <- list(
      s = quote(roll_sum(x, 400L)), m = quote(roll_mean(x, 400L)),
      m2 = quote(roll_mean(x, 400L, na.rm = TRUE)), s2 = quote(roll_sum(x, 400L, na.rm = TRUE)),
      d = quote(roll_sd(x, 400L))
    )
    refs <- list(
      s = function(xx) f59_ref(xx, w, "sum"), m = function(xx) f59_ref(xx, w, "mean"),
      m2 = function(xx) f59_ref(xx, w, "mean", TRUE), s2 = function(xx) f59_ref(xx, w, "sum", TRUE),
      d = function(xx) f59_ref(xx, w, "sd")
    )
    orders <- list(c("s", "m"), c("m", "s"), c("s", "m", "m2"), c("m", "s", "m2", "s2"),
                   c("s2", "s", "m", "m2"), c("s", "d", "m"), c("d", "s", "s2", "m"))
    for (ord in orders) {
      for (grouped in c(FALSE, TRUE)) {
        for (lazy in c(FALSE, TRUE)) {
          gt <- tbl_gpu(df, lazy = lazy)
          if (grouped) gt <- dplyr::group_by(gt, g)
          got <- gt |> dplyr::mutate(!!!exprs[ord]) |> collect()
          for (nm in ord) {
            ref <- if (grouped) {
              out <- rep(NA_real_, n)
              for (idx in split(seq_len(n), g)) out[idx] <- refs[[nm]](x[idx])
              out
            } else {
              refs[[nm]](x)
            }
            f59_check(got[[nm]], ref, sprintf("order %s: %s specials=%s grouped=%s lazy=%s",
                                              paste(ord, collapse = ","), nm, specials, grouped, lazy))
          }
        }
      }
    }
  }
})

test_that("F59: roll_sd() overflow follows R (Inf, or a finite 1e150-scale value), never 0 (review round 2, B6)", {
  skip_if_no_gpu()
  x <- c(1, 2, 3, 1e200, 5, 6, 7, 8, 9, -1e200, 1, 2)
  for (lazy in c(FALSE, TRUE)) {
    for (w in c(2L, 3L)) {
      got <- tbl_gpu(data.frame(x = x), lazy = lazy) |> dplyr::mutate(s = roll_sd(x, w)) |> collect()
      f59_check(got$s, f59_ref(x, w, "sd"), sprintf("1e200 sd w=%d lazy=%s", w, lazy), tol = 1e-12)
    }
  }
  x2 <- rep(c(1e150, -1e150), 15000)
  got2 <- tbl_gpu(data.frame(x = x2)) |> dplyr::mutate(s = roll_sd(x, 30000)) |> collect()
  f59_check(got2$s, f59_ref(x2, 30000, "sd"), "x2 sd w=30000")
  x3 <- x2
  x3[2] <- NA
  got3 <- tbl_gpu(data.frame(x = x3)) |> dplyr::mutate(s = roll_sd(x, 30000, na.rm = TRUE)) |> collect()
  f59_check(got3$s, f59_ref(x3, 30000, "sd", TRUE), "x3 sd w=30000 na.rm")
})

test_that("F59: huge values next to NA/NaN/Inf/group boundaries never leak into a window (review round 3, B7)", {
  skip_if_no_gpu()
  cases <- list(
    list(x = c(2e154, 2e154, NA, 2e154, 2e154, 2e154), w = 2L, label = "x1 NA neighbour"),
    list(x = c(2e154, 2e154, Inf, 2e154), w = 2L, label = "x2 Inf neighbour"),
    list(x = 1e155 + c(1, 2, NaN, 4, 5, 6) * 1e150, w = 2L, label = "x3 NaN neighbour"),
    list(x = local({ v <- rep(2e154, 1000); v[601] <- NA; v }), w = 300L, label = "x5 wide"),
    list(x = c(1, 1 + 1e-7, -1.7e308, 5), w = 2L, label = "y huge neighbour")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- tbl_gpu(data.frame(x = cs$x), lazy = lazy) |>
        dplyr::mutate(s = roll_sd(x, cs$w), m = roll_mean(x, cs$w), t = roll_sum(x, cs$w)) |>
        collect()
      info <- sprintf("%s lazy=%s", cs$label, lazy)
      f59_check(got$s, f59_ref(cs$x, cs$w, "sd"), paste(info, "sd"), tol = 1e-10)
      f59_check(got$m, f59_ref(cs$x, cs$w, "mean"), paste(info, "mean"), tol = 1e-12)
      f59_check(got$t, f59_ref(cs$x, cs$w, "sum"), paste(info, "sum"), tol = 1e-12)
    }
  }
  dfg <- data.frame(g = c(1L, 1L, 2L, 2L), x = c(2e154, 2e154, NA, 2e154))
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(dfg, lazy = lazy) |> dplyr::group_by(g) |> dplyr::mutate(s = roll_sd(x, 2)) |> collect()
    f59_check(got$s, f59_ref_grouped(dfg$x, dfg$g, 2L, "sd"), paste("grouped boundary lazy =", lazy), tol = 1e-10)
  }
  z <- c(rep(1, 99), 1e308, -1e308, rep(1, 99))
  for (with_sd in c(FALSE, TRUE)) {
    gt <- tbl_gpu(data.frame(x = z))
    got <- if (with_sd) {
      gt |> dplyr::mutate(m = roll_mean(x, 100), s = roll_sd(x, 100)) |> collect()
    } else {
      gt |> dplyr::mutate(m = roll_mean(x, 100)) |> collect()
    }
    # Rows 101-199 hold both +1e308 and -1e308: R's own two-pass mean() is
    # off there (mean(z[3:102]) is 0.0199, the exact mean 0.98), as is any
    # other summation order, so only the NA/NaN pattern is compared on them.
    ref_z <- f59_ref(z, 100L, "mean")
    both <- 101:199
    expect_identical(is.na(got$m), is.na(ref_z), info = paste("z NA pattern, with sd =", with_sd))
    expect_false(any(is.nan(got$m)), info = paste("z no NaN, with sd =", with_sd))
    f59_check(got$m[-both], ref_z[-both], paste("z mean, with sd =", with_sd), tol = 1e-12)
    expect_equal(got$m[100], 1e306, tolerance = 1e-12, info = paste("z m[100], with sd =", with_sd))
  }
})

# =============================================================================
# Width scaling: cost independent of the window width
# =============================================================================

f59_time <- function(f, reps = 9L) {
  f()
  gpu_sync()
  t <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- Sys.time()
    f()
    gpu_sync()
    t[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  stats::median(t)
}

# Bounds are in units of a same-run cumsum(x) (`tc`), so they follow the
# machine. Calibrated on dev 5e30ef7 (scratchpad/parity/features/F59/runs/
# profile/calib-dev.txt), cudf's rolling kernel at w = 11 cost:
#   ungrouped 1e7: 1.02-1.21 tc (na.rm = TRUE 1.60-1.81 tc), sd 3.2 tc;
#   grouped 2e6:   1.10-1.30 tc (na.rm = TRUE 1.33-1.41 tc), sd 1.8 tc;
# and grew linearly with w (mean 64 tc at w = 4001, sd 765 tc).
# Contract (revised after round 1, see the brief):
#   - narrow windows are no slower than dev (w = 11, mean/sum/min/max):
#     <= 1.6 tc + 0.3 ms, or 2.5 tc + 0.3 ms under na.rm = TRUE;
#   - wide windows cost about the same as narrow ones: at every width up to
#     4001, mean/sum <= 3.5 tc + 0.5 ms and sd <= 5 tc + 0.5 ms (sd takes
#     the accurate path at every width).
# Revised after round 2: a 1%-NA ungrouped 1e7 cell under the same bounds,
# except sd with na.rm = TRUE (<= 6.5 tc + 0.5 ms; each window part then
# needs its own reference and counts). On dev (calib-na-dev.txt) cudf at
# w = 11 was 0.88-1.48 tc there; round 2's sd was 7.05 tc at every width.
test_that("F59: narrow windows are no slower than dev, and wide windows cost about the same (ungrouped 1e7, grouped 2e6)", {
  skip_if_no_gpu()
  set.seed(5908)
  cells <- list(
    list(label = "ungrouped 1e7", n = 1e7, groups = NULL),
    list(label = "grouped 2e6/100", n = 2e6, groups = 100L),
    list(label = "ungrouped 1e7, 1% NA", n = 1e7, groups = NULL, na = 0.01)
  )
  for (cell in cells) {
    df <- data.frame(g = sample.int(if (is.null(cell$groups)) 1L else cell$groups, cell$n, TRUE),
                     x = rnorm(cell$n))
    if (!is.null(cell$na)) df$x[sample.int(cell$n, cell$n * cell$na)] <- NA
    src <- tbl_gpu(df)
    if (!is.null(cell$groups)) src <- dplyr::group_by(src, g)
    tc <- f59_time(function() dplyr::compute(dplyr::mutate(src, r = cumsum(x))))
    for (stat in c("mean", "sum", "min", "max", "sd")) {
      for (na_rm in c(FALSE, TRUE)) {
        for (w in c(11L, 101L, 1001L, 4001L)) {
          call <- f59_roll_call(stat, w, na_rm)
          t <- f59_time(function() dplyr::compute(dplyr::mutate(src, r = !!call)))
          lbl <- sprintf("%s roll_%s(w=%d, na.rm=%s): %.2f ms = %.2f x cumsum (%.2f ms)",
                         cell$label, stat, w, na_rm, 1000 * t, t / tc, 1000 * tc)
          if (w == 11L && stat != "sd") {
            expect_lte(t, (if (na_rm) 2.5 else 1.6) * tc + 0.0003, label = lbl)
          }
          if (stat %in% c("mean", "sum")) expect_lte(t, 3.5 * tc + 0.0005, label = lbl)
          if (stat == "sd") {
            sd_cap <- if (!is.null(cell$na) && na_rm) 6.5 else 5
            expect_lte(t, sd_cap * tc + 0.0005, label = lbl)
          }
        }
      }
    }
    rm(src, df)
    gc()
  }
})
