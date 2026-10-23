# F59 coder-owned tests: reproductions of the round-1 review findings
# (B1 empty window parts, B4 data.table without frollmedian), plus branches
# the acceptance file does not reach (w > 4096, either side of the cudf/scan
# width threshold, windows around Inf).

ref_roll <- function(x, n, stat, na.rm = FALSE) {
  fn <- switch(stat, mean = base::mean, sum = base::sum, min = base::min,
               max = base::max, sd = stats::sd)
  out <- rep(NA_real_, length(x))
  if (n > length(x)) return(out)
  for (i in seq.int(n, length(x))) {
    w <- x[(i - n + 1L):i]
    if (na.rm) {
      w <- w[!is.na(w)]
      if (length(w) == 0) next
    } else if (anyNA(w)) next
    out[i] <- fn(w)
  }
  out
}

gpu_roll <- function(df, stat, n, na_rm, lazy = FALSE, group = NULL) {
  gt <- tbl_gpu(df, lazy = lazy)
  if (!is.null(group)) gt <- dplyr::group_by(gt, !!!rlang::syms(group))
  call <- rlang::call2(paste0("roll_", stat), quote(x), n, na.rm = na_rm)
  dplyr::pull(collect(dplyr::mutate(gt, y = !!call)), y)
}

# element by element: same NA pattern, same Inf values, finite values within
# `tol` of the reference, relative to `scale` (the window's own magnitude)
expect_rows_close <- function(got, ref, scale = abs(ref), tol, info) {
  expect_identical(is.na(got), is.na(ref), info = info)
  inf <- !is.na(ref) & is.infinite(ref)
  expect_identical(got[inf], ref[inf], info = info)
  fin <- !is.na(ref) & is.finite(ref)
  err <- abs(got[fin] - ref[fin]) / pmax(scale[fin], .Machine$double.xmin)
  expect_lte(max(c(0, err)), tol, label = info)
}

test_that("B1: a window part with no finite value (na.rm = TRUE) is exact, eager and lazy", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.1, 0.2, 0.3, NA, 1e17, 7))
  df2 <- data.frame(x = c(1e200, NA, NA, 0.1, 0.2, 0.3))
  for (lazy in c(FALSE, TRUE)) {
    for (stat in c("mean", "sum")) {
      expect_rows_close(gpu_roll(df, stat, 3, TRUE, lazy), ref_roll(df$x, 3, stat, TRUE),
                        tol = 1e-15, info = paste("df", stat, lazy))
      expect_rows_close(gpu_roll(df2, stat, 3, TRUE, lazy), ref_roll(df2$x, 3, stat, TRUE),
                        tol = 1e-15, info = paste("df2", stat, lazy))
    }
    sd2 <- gpu_roll(df2, "sd", 3, TRUE, lazy)
    expect_false(any(is.nan(sd2)), info = paste("df2 sd NaN", lazy))
    expect_rows_close(sd2, ref_roll(df2$x, 3, "sd", TRUE), tol = 1e-12,
                      info = paste("df2 sd", lazy))
  }
})

test_that("B1: mixed magnitudes with NA, na.rm = TRUE, every row within 1e-12 of its window", {
  skip_if_no_gpu()
  set.seed(5912)
  n <- 20000L
  x <- ifelse(stats::runif(n) < 0.5, stats::rnorm(n), 1e9 + stats::rnorm(n))
  x[sample.int(n, n / 10)] <- NA
  df <- data.frame(x = x)
  for (w in c(3L, 4L, 11L)) {
    for (stat in c("sum", "mean")) {
      expect_rows_close(gpu_roll(df, stat, w, TRUE), ref_roll(x, w, stat, TRUE),
                        scale = ref_roll(abs(x), w, stat, TRUE), tol = 1e-12,
                        info = paste(stat, w))
    }
  }
  expect_rows_close(gpu_roll(df, "sd", 3L, TRUE), ref_roll(x, 3L, "sd", TRUE),
                    tol = 1e-9, info = "sd 3")
})

test_that("windows wider than 4096 match the reference (NA-free and with NA)", {
  skip_if_no_gpu()
  set.seed(5913)
  n <- 12000L
  x <- stats::rnorm(n) + 100
  df <- data.frame(x = x)
  for (stat in c("mean", "sum", "sd", "min", "max")) {
    expect_rows_close(gpu_roll(df, stat, 5000L, FALSE), ref_roll(x, 5000L, stat), tol = 1e-12,
                      info = stat)
  }
  x[c(10, 6000, 6001)] <- NA
  df <- data.frame(x = x)
  for (stat in c("mean", "sum", "sd")) {
    expect_rows_close(gpu_roll(df, stat, 5000L, TRUE), ref_roll(x, 5000L, stat, TRUE),
                      tol = 1e-12, info = paste(stat, "na.rm"))
  }
})

test_that("widths either side of the cudf/scan threshold agree with the reference (NA, Inf, groups)", {
  skip_if_no_gpu()
  set.seed(5914)
  n <- 3000L
  g <- sample.int(3, n, TRUE)
  x <- stats::rnorm(n) * 10^sample(-2:3, n, TRUE)
  x[sample.int(n, 100)] <- NA
  x[sample.int(n, 3)] <- Inf
  df <- data.frame(g = g, x = x)
  for (w in c(31L, 64L, 65L, 100L, 128L, 129L, 257L, 600L)) {
    for (stat in c("mean", "sum", "min", "max")) {
      for (na_rm in c(FALSE, TRUE)) {
        ref <- rep(NA_real_, n)
        for (i in split(seq_len(n), g)) ref[i] <- ref_roll(x[i], w, stat, na_rm)
        got <- gpu_roll(df, stat, w, na_rm, group = "g")
        expect_rows_close(got, ref, tol = 1e-9,
                          info = sprintf("%s w=%d na.rm=%s", stat, w, na_rm))
      }
    }
  }
})

test_that("B4: roll_median() falls back to the sorted engine when data.table lacks frollmedian", {
  local_mocked_bindings(
    roll_has_frollmedian = function() FALSE,
    roll_median_dt = function(...) stop("data.table path used")
  )
  expect_identical(roll_median(c(1, 2, 3), 2), c(NA, 1.5, 2.5))
  x <- c(3, NA, 1, 4, 1, NaN, 5, 9, 2, 6)
  expect_identical(roll_median(x, 3, na.rm = TRUE), roll_median_sorted(x, 3, na.rm = TRUE))
  expect_identical(roll_median(x, 3), roll_median_sorted(x, 3))
})

test_that("B4: the helper checks the installed data.table version, not just its presence", {
  expect_type(roll_has_frollmedian(), "logical")
  skip_if_not_installed("data.table", "1.18.4")
  expect_true(roll_has_frollmedian())
})

# -----------------------------------------------------------------------------
# The scan path under every width. roll_native() sends mean/sum/min/max below a
# crossover width to cudf's kernel (as dev did), so the acceptance tests'
# narrow windows no longer exercise the scans for those four; forcing the path
# (the internal option cuplyr.roll_path = "scans") repeats their semantic
# cases on it.
# -----------------------------------------------------------------------------
expect_rows_match <- function(got, ref, info, tol = 1e-9) {
  got <- as.double(got)
  expect_identical(length(got), length(ref), info = info)
  expect_identical(is.na(got), is.na(ref), info = paste(info, "(NA)"))
  expect_identical(is.nan(got), is.nan(ref), info = paste(info, "(NaN)"))
  inf <- !is.na(ref) & is.infinite(ref)
  expect_identical(got[inf], ref[inf], info = paste(info, "(Inf)"))
  fin <- !is.na(ref) & is.finite(ref)
  if (any(fin)) {
    err <- abs(got[fin] - ref[fin]) / pmax(abs(ref[fin]), 1)
    expect_lte(max(err), tol, label = info)
  }
}

scan_stats <- c("mean", "sum", "sd", "min", "max")

test_that("forced scans: edge values (Inf of both signs, NA, NaN, runs of NA), eager and lazy", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
  x <- c(1, Inf, 3, -Inf, 5, NA, NaN, 8, Inf, Inf, -Inf, 2, 2, 2, 7,
         NA, NA, NA, 4, 1e6, -1e6, 0)
  df <- data.frame(x = x)
  nx <- length(x)
  for (lazy in c(FALSE, TRUE)) {
    for (stat in scan_stats) {
      for (w in c(1L, 2L, 3L, 5L, nx - 1L, nx, nx + 1L)) {
        for (na_rm in c(FALSE, TRUE)) {
          info <- sprintf("roll_%s(x, %d, na.rm = %s) lazy = %s", stat, w, na_rm, lazy)
          expect_rows_match(gpu_roll(df, stat, w, na_rm, lazy = lazy),
                            ref_roll(x, w, stat, na_rm), info)
        }
      }
    }
  }
})

test_that("forced scans: interleaved groups, NA keys, short and all-NA groups", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
  set.seed(5901)
  n <- 400L
  g <- sample(c(1:6, NA), n, replace = TRUE)
  x <- round(rnorm(n, 50, 10), 2)
  x[sample.int(n, 30)] <- NA
  x[sample.int(n, 10)] <- NaN
  x[sample.int(n, 4)] <- Inf
  x[sample.int(n, 4)] <- -Inf
  g <- c(g, 9L, 8L, 8L, 8L, 7L, 7L)
  x <- c(x, 3, NA, NaN, NA, 1, 2)
  perm <- sample.int(length(x))
  df <- data.frame(g = g[perm], x = x[perm])
  for (lazy in c(FALSE, TRUE)) {
    for (stat in scan_stats) {
      for (w in c(1L, 2L, 3L, 7L, 25L)) {
        for (na_rm in c(FALSE, TRUE)) {
          ref <- rep(NA_real_, nrow(df))
          for (i in split(seq_len(nrow(df)), factor(df$g, exclude = NULL))) {
            ref[i] <- ref_roll(df$x[i], w, stat, na_rm)
          }
          info <- sprintf("grouped roll_%s(x, %d, na.rm = %s) lazy = %s", stat, w, na_rm, lazy)
          expect_rows_match(gpu_roll(df, stat, w, na_rm, lazy = lazy, group = "g"), ref, info)
        }
      }
    }
  }
})

test_that("forced scans: several specs on one column, filter, 0 rows, integer and logical inputs", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
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
  pipe_f <- function(d) dplyr::filter(d, roll_sd(x, 3) > 0.5, roll_sum(x, 4, na.rm = TRUE) < 2)
  expect_same_as_dplyr(df, pipe_f)

  for (stat in scan_stats) {
    expect_identical(length(gpu_roll(data.frame(g = integer(0), x = numeric(0)), stat, 3L, FALSE)), 0L)
  }
  xi <- c(3L, 1L, NA, 4L, 1L, 5L, 9L, 2L, 6L)
  for (stat in c("mean", "sd")) {
    for (na_rm in c(FALSE, TRUE)) {
      expect_rows_match(gpu_roll(data.frame(x = xi), stat, 3L, na_rm), ref_roll(xi, 3L, stat, na_rm),
                        paste("int", stat, na_rm))
    }
  }
  xl <- c(TRUE, FALSE, NA, TRUE, TRUE, FALSE, TRUE)
  expect_rows_match(gpu_roll(data.frame(x = xl), "mean", 2L, TRUE), ref_roll(xl, 2L, "mean", TRUE), "lgl mean")
})

test_that("forced scans: a window that is exactly the table, and the last segment short", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
  set.seed(5915)
  for (n in c(7L, 8L, 9L, 23L, 24L, 25L)) {
    x <- stats::rnorm(n, 1e6)
    for (w in c(2L, 3L, 4L, 7L, 8L, n - 1L, n)) {
      if (w < 2L) next
      for (stat in c("mean", "sum", "sd")) {
        expect_rows_match(gpu_roll(data.frame(x = x), stat, w, FALSE), ref_roll(x, w, stat),
                          sprintf("n=%d w=%d %s", n, w, stat), tol = 1e-12)
      }
      g <- rep(1:3, length.out = n)
      ref <- rep(NA_real_, n)
      for (i in split(seq_len(n), g)) ref[i] <- ref_roll(x[i], min(w, 3L), "sd")
      expect_rows_match(gpu_roll(data.frame(g = g, x = x), "sd", min(w, 3L), FALSE, group = "g"), ref,
                        sprintf("grouped n=%d w=%d sd", n, w), tol = 1e-9)
    }
  }
})

test_that("B6: roll_sd() follows R when the variance overflows (Inf), every row, eager and lazy", {
  skip_if_no_gpu()
  # base R's var() accumulates in long double and returns Inf once the
  # variance passes the double range: sd(c(3, 1e200)) is Inf
  x <- c(1, 2, 3, 1e200, 5, 6, 7, 8, 9, -1e200, 1, 2)
  df <- data.frame(x = x)
  for (lazy in c(FALSE, TRUE)) {
    for (stat in c("mean", "sd", "sum")) {
      for (w in c(2L, 3L, 4L)) {
        expect_rows_match(gpu_roll(df, stat, w, FALSE, lazy = lazy), ref_roll(x, w, stat),
                          sprintf("1e200 %s w=%d lazy=%s", stat, w, lazy), tol = 1e-12)
      }
    }
    expect_rows_match(gpu_roll(df, "sd", 3L, TRUE, lazy = lazy), ref_roll(x, 3L, "sd", TRUE),
                      sprintf("1e200 sd na.rm lazy=%s", lazy), tol = 1e-12)
  }
})

test_that("B6: roll_sd() of +-1e150 over a wide window is finite (s1^2 must not overflow)", {
  skip_if_no_gpu()
  x2 <- rep(c(1e150, -1e150), 15000)
  ref2 <- ref_roll(x2, 30000L, "sd")
  expect_rows_close(gpu_roll(data.frame(x = x2), "sd", 30000L, FALSE), ref2, tol = 1e-12, info = "x2")
  x3 <- x2
  x3[2] <- NA
  ref3 <- ref_roll(x3, 30000L, "sd", TRUE)
  expect_rows_close(gpu_roll(data.frame(x = x3), "sd", 30000L, TRUE), ref3, tol = 1e-12, info = "x3")
})

test_that("B6: a large spread that overflows only in squares is finite; small windows nearby stay exact", {
  skip_if_no_gpu()
  set.seed(5919)
  # +-1e152 at w = 1000: the squared spread summed over a window overflows a
  # double, the variance (1e304) does not
  x <- c(stats::rnorm(1500), rep(c(1e152, -1e152), 1000), stats::rnorm(1500, 1e6))
  x[c(10, 2000, 4200)] <- NA
  g <- rep(1:2, length.out = length(x))
  df <- data.frame(g = g, x = x)
  for (na_rm in c(FALSE, TRUE)) {
    for (w in c(3L, 1000L)) {
      expect_rows_match(gpu_roll(df, "sd", w, na_rm), ref_roll(x, w, "sd", na_rm),
                        sprintf("spread sd w=%d na.rm=%s", w, na_rm), tol = 1e-10)
      ref <- rep(NA_real_, length(x))
      for (i in split(seq_along(x), g)) ref[i] <- ref_roll(x[i], w, "sd", na_rm)
      expect_rows_match(gpu_roll(df, "sd", w, na_rm, group = "g"), ref,
                        sprintf("grouped spread sd w=%d na.rm=%s", w, na_rm), tol = 1e-10)
    }
  }
})

test_that("B5: roll_sum() before roll_mean() at a scan width, every order, eager and lazy, grouped and ungrouped", {
  skip_if_no_gpu()
  set.seed(5918)
  df <- data.frame(g = sample(1:2, 1000, TRUE), x = stats::rnorm(1000))
  pipes <- list(
    function(d) dplyr::mutate(d, s = roll_sum(x, 200), m = roll_mean(x, 200)),
    function(d) dplyr::mutate(d, m = roll_mean(x, 200), s = roll_sum(x, 200),
                              m2 = roll_mean(x, 200, na.rm = TRUE)),
    function(d) dplyr::mutate(d, s = roll_sum(x, 200), s2 = roll_sum(x, 200), d = roll_sd(x, 200),
                              m = roll_mean(x, 200))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    gp <- function(d) p(dplyr::group_by(d, g))
    expect_same_as_dplyr(df, gp)
    expect_same_as_dplyr_lazy(df, gp)
  }
})

test_that("B5: the same orders on the forced scan path at a narrow width", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
  set.seed(5918)
  df <- data.frame(g = sample(1:2, 300, TRUE), x = stats::rnorm(300))
  df$x[c(5, 77)] <- NA
  pipes <- list(
    function(d) dplyr::mutate(d, s = roll_sum(x, 5), m = roll_mean(x, 5)),
    function(d) dplyr::mutate(d, m = roll_mean(x, 5), s = roll_sum(x, 5), m2 = roll_mean(x, 5, na.rm = TRUE)),
    function(d) dplyr::mutate(d, s2 = roll_sum(x, 5, na.rm = TRUE), s = roll_sum(x, 5), m = roll_mean(x, 5),
                              d = roll_sd(x, 5, na.rm = TRUE), d2 = roll_sd(x, 5))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    expect_same_as_dplyr(df, function(d) p(dplyr::group_by(d, g)))
  }
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, roll_sum(x, 5) > 0, roll_mean(x, 5) < 1))
})

test_that("forced scans: random group sizes (many shorter than a window), clean and with NA", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
  set.seed(5916)
  for (rep in 1:4) {
    sizes <- sample(c(1:6, 15, 40), 120, TRUE)
    g <- rep(seq_along(sizes), sizes)
    n <- length(g)
    x <- stats::rnorm(n) * 10^sample(0:3, n, TRUE) + sample(c(0, 1e6), n, TRUE)
    if (rep > 2) x[sample.int(n, n / 8)] <- NA
    perm <- sample.int(n)
    df <- data.frame(g = g[perm], x = x[perm])
    for (w in c(2L, 3L, 5L, 8L, 13L)) {
      for (stat in c("mean", "sum", "sd", "min", "max")) {
        for (na_rm in c(FALSE, TRUE)) {
          ref <- rep(NA_real_, n)
          for (i in split(seq_len(n), df$g)) ref[i] <- ref_roll(df$x[i], w, stat, na_rm)
          expect_rows_match(gpu_roll(df, stat, w, na_rm, group = "g"), ref,
                            sprintf("rep=%d %s w=%d na.rm=%s", rep, stat, w, na_rm), tol = 1e-8)
        }
      }
    }
  }
})

test_that("forced scans: mean, sum and sd of one column at one width share their scans correctly", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.roll_path = "scans")
  set.seed(5917)
  n <- 1500L
  x <- 1e9 + stats::rnorm(n)
  g <- sample.int(5, n, TRUE)
  for (with_na in c(FALSE, TRUE)) {
    xx <- x
    if (with_na) xx[sample.int(n, 120)] <- NA
    for (grouped in c(FALSE, TRUE)) {
      for (w in c(7L, 64L)) {
        gt <- tbl_gpu(data.frame(g = g, x = xx))
        if (grouped) gt <- dplyr::group_by(gt, g)
        got <- gt |>
          dplyr::mutate(m = roll_mean(x, w), s = roll_sum(x, w), d = roll_sd(x, w),
                        m2 = roll_mean(x, w, na.rm = TRUE), d2 = roll_sd(x, w, na.rm = TRUE)) |>
          collect()
        key <- if (grouped) g else rep(1L, n)
        ref_for <- function(stat, na_rm) {
          ref <- rep(NA_real_, n)
          for (i in split(seq_len(n), key)) ref[i] <- ref_roll(xx[i], w, stat, na_rm)
          ref
        }
        info <- sprintf("na=%s grouped=%s w=%d", with_na, grouped, w)
        expect_rows_match(got$m, ref_for("mean", FALSE), paste("mean", info), tol = 1e-12)
        expect_rows_match(got$s, ref_for("sum", FALSE), paste("sum", info), tol = 1e-12)
        expect_rows_match(got$d, ref_for("sd", FALSE), paste("sd", info), tol = 1e-10)
        expect_rows_match(got$m2, ref_for("mean", TRUE), paste("mean na.rm", info), tol = 1e-12)
        expect_rows_match(got$d2, ref_for("sd", TRUE), paste("sd na.rm", info), tol = 1e-10)
      }
    }
  }
})

test_that("the cuplyr.roll_path switch: option first, then the environment, unknown values error", {
  withr::local_options(cuplyr.roll_path = NULL)
  withr::local_envvar(CUPLYR_ROLL_PATH = NA)
  expect_identical(roll_path_option(), "auto")
  withr::local_envvar(CUPLYR_ROLL_PATH = "cudf")
  expect_identical(roll_path_option(), "cudf")
  withr::local_options(cuplyr.roll_path = "scans")
  expect_identical(roll_path_option(), "scans")
  withr::local_options(cuplyr.roll_path = "Scans")
  expect_error(roll_path_option(), "must be \"auto\", \"scans\" or \"cudf\"")
  withr::local_options(cuplyr.roll_path = NULL)
  withr::local_envvar(CUPLYR_ROLL_PATH = "gpu")
  expect_error(roll_path_option(), "not \"gpu\"")
  skip_if_no_gpu()
  expect_error(collect(dplyr::mutate(tbl_gpu(data.frame(x = 1:5 + 0.5)), y = roll_mean(x, 2))),
               "cuplyr.roll_path")
})

test_that("forcing either path gives the reference values", {
  skip_if_no_gpu()
  set.seed(5920)
  df <- data.frame(g = sample(1:3, 600, TRUE), x = stats::rnorm(600))
  df$x[c(3, 50, 51)] <- NA
  pipe <- function(d) {
    dplyr::mutate(dplyr::group_by(d, g), a = roll_mean(x, 30), b = roll_sum(x, 30, na.rm = TRUE),
                  c = roll_min(x, 30, na.rm = TRUE), d = roll_max(x, 30), e = roll_sd(x, 30))
  }
  for (path in c("auto", "scans", "cudf")) {
    withr::local_options(cuplyr.roll_path = path)
    expect_same_as_dplyr(df, pipe)
  }
})

test_that("several specs over one column at w = 400 with NA, NaN and Inf, in every order", {
  skip_if_no_gpu()
  # the protected order test's specials case, on a column long enough to hold
  # them (its f59_wide_x() writes rows up to 3310)
  set.seed(5913)
  n <- 4000L
  x <- round(stats::rnorm(n, 10, 3), 3)
  x[c(1500, 1503)] <- NA
  x[1510] <- NaN
  x[c(2400, 2401)] <- Inf
  x[2405] <- -Inf
  x[3300] <- Inf
  x[3301:3310] <- NA
  g <- sample(1:2, n, TRUE)
  exprs <- list(
    s = quote(roll_sum(x, 400L)), m = quote(roll_mean(x, 400L)),
    m2 = quote(roll_mean(x, 400L, na.rm = TRUE)), s2 = quote(roll_sum(x, 400L, na.rm = TRUE)),
    d = quote(roll_sd(x, 400L)), d2 = quote(roll_sd(x, 400L, na.rm = TRUE))
  )
  refs <- list(s = c("sum", FALSE), m = c("mean", FALSE), m2 = c("mean", TRUE),
               s2 = c("sum", TRUE), d = c("sd", FALSE), d2 = c("sd", TRUE))
  orders <- list(c("s", "m"), c("m", "s"), c("s", "m", "m2"), c("m", "s", "m2", "s2"),
                 c("s2", "s", "m", "m2"), c("s", "d", "m"), c("d", "s", "s2", "m"),
                 c("d2", "m2", "d", "s", "s2", "m"))
  for (ord in orders) {
    for (grouped in c(FALSE, TRUE)) {
      for (lazy in c(FALSE, TRUE)) {
        gt <- tbl_gpu(data.frame(g = g, x = x), lazy = lazy)
        if (grouped) gt <- dplyr::group_by(gt, g)
        got <- gt |> dplyr::mutate(!!!exprs[ord]) |> collect()
        key <- if (grouped) g else rep(1L, n)
        for (nm in ord) {
          ref <- rep(NA_real_, n)
          for (i in split(seq_len(n), key)) {
            ref[i] <- ref_roll(x[i], 400L, refs[[nm]][1], as.logical(refs[[nm]][2]))
          }
          expect_rows_match(got[[nm]], ref, sprintf("order %s: %s grouped=%s lazy=%s",
                                                    paste(ord, collapse = ","), nm, grouped, lazy))
        }
      }
    }
  }
})

# -----------------------------------------------------------------------------
# Review round 3, B7: a window must never read a value from outside itself and
# then discard it by multiplying it by 0 (0 * Inf is NaN). Every row is
# compared: the exact NA / NaN / Inf pattern, finite values within 1e-12 of
# the reference relative to the value itself (an expected 0 must be 0).
# -----------------------------------------------------------------------------
expect_rows_exact <- function(got, ref, info, tol = 1e-12) {
  got <- as.double(got)
  expect_identical(length(got), length(ref), info = info)
  expect_identical(is.na(got), is.na(ref), info = paste(info, "(NA)"))
  expect_identical(is.nan(got), is.nan(ref), info = paste(info, "(NaN)"))
  inf <- !is.na(ref) & is.infinite(ref)
  expect_identical(got[inf], ref[inf], info = paste(info, "(Inf)"))
  fin <- !is.na(ref) & is.finite(ref)
  if (any(fin)) {
    err <- abs(got[fin] - ref[fin]) / pmax(abs(ref[fin]), .Machine$double.xmin)
    expect_lte(max(err), tol, label = paste(info, "(max relative error)"))
  }
}

# Rows whose window holds a value above 1e307 in magnitude. There R's mean /
# sum (long double, and for mean a second correcting pass, both left to
# right) differ from any double summation in another order: mean(c(1e308,
# -1e308, 1)) is 0.5556 in R, 1/3 exactly, and the sum of 1.7e308 twice
# overflows a double (the pre-existing N4 of the round-3 review; cudf's
# kernel differs from R there too). sd has no such exemption.
huge_window <- function(x, w) {
  big <- !is.na(x) & abs(x) > 1e307
  vapply(seq_along(x), function(i) any(big[max(1L, i - w + 1L):i]), logical(1))
}
expect_sums_exact <- function(got, ref, x, w, info) {
  expect_identical(is.na(got) & !is.nan(got), is.na(ref) & !is.nan(ref), info = paste(info, "(NA)"))
  keep <- !huge_window(x, w)
  expect_rows_exact(got[keep], ref[keep], info)
}

b7_cases <- list(
  list(x = c(2e154, 2e154, NA, 2e154, 2e154, 2e154), w = 2L, label = "x1 NA neighbour"),
  list(x = c(2e154, 2e154, Inf, 2e154), w = 2L, label = "x2 Inf neighbour"),
  list(x = 1e155 + c(1, 2, NaN, 4, 5, 6) * 1e150, w = 2L, label = "x3 NaN neighbour"),
  list(x = local({ v <- rep(2e154, 1000); v[601] <- NA; v }), w = 300L, label = "x5 wide"),
  list(x = c(1, 1 + 1e-7, -1.7e308, 5), w = 2L, label = "y huge neighbour"),
  list(x = c(-1.7e308, NA, 1.7e308, 1.7e308, NA, 1, 1 + 1e-7, -1.7e308, NA, 3, 3), w = 2L,
       label = "opposite huge values around NA"),
  list(x = c(1e308, -1e308, 1, 2, -1e308, 1e308, 5, 5), w = 2L, label = "range above DBL_MAX, w = 2"),
  list(x = c(1e308, -1e308, 1, 2, -1e308, 1e308, 5, 5, 5), w = 3L, label = "range above DBL_MAX, w = 3")
)

test_that("B7: sd, mean and sum next to huge values, NA, NaN and Inf, every row, both paths, eager and lazy", {
  skip_if_no_gpu()
  for (path in c("auto", "scans")) {
    withr::local_options(cuplyr.roll_path = path)
    for (cs in b7_cases) {
      for (lazy in c(FALSE, TRUE)) {
        info <- sprintf("%s path=%s lazy=%s", cs$label, path, lazy)
        df <- data.frame(x = cs$x)
        # all at one width (mean / sum then share roll_sd's scans) ...
        got <- tbl_gpu(df, lazy = lazy) |>
          dplyr::mutate(s = roll_sd(x, cs$w), m = roll_mean(x, cs$w), t = roll_sum(x, cs$w),
                        s2 = roll_sd(x, cs$w, na.rm = TRUE), m2 = roll_mean(x, cs$w, na.rm = TRUE)) |>
          collect()
        expect_rows_exact(got$s, ref_roll(cs$x, cs$w, "sd"), paste(info, "sd"))
        expect_sums_exact(got$m, ref_roll(cs$x, cs$w, "mean"), cs$x, cs$w, paste(info, "mean"))
        expect_sums_exact(got$t, ref_roll(cs$x, cs$w, "sum"), cs$x, cs$w, paste(info, "sum"))
        expect_rows_exact(got$s2, ref_roll(cs$x, cs$w, "sd", TRUE), paste(info, "sd na.rm"))
        expect_sums_exact(got$m2, ref_roll(cs$x, cs$w, "mean", TRUE), cs$x, cs$w, paste(info, "mean na.rm"))
        # ... and each alone
        expect_rows_exact(gpu_roll(df, "sd", cs$w, FALSE, lazy = lazy), ref_roll(cs$x, cs$w, "sd"),
                          paste(info, "sd alone"))
        for (stat in c("mean", "sum")) {
          expect_sums_exact(gpu_roll(df, stat, cs$w, FALSE, lazy = lazy), ref_roll(cs$x, cs$w, stat),
                            cs$x, cs$w, paste(info, stat, "alone"))
        }
      }
    }
  }
})

test_that("B7: the next group's first row (NA, huge) never leaks into the last window of a group", {
  skip_if_no_gpu()
  dfs <- list(
    data.frame(g = c(1L, 1L, 2L, 2L), x = c(2e154, 2e154, NA, 2e154)),
    data.frame(g = c(1L, 1L, 2L, 2L, 2L), x = c(1, 1 + 1e-7, -1.7e308, 5, 5)),
    data.frame(g = rep(1:3, c(4L, 3L, 4L)),
               x = c(3, 3, 3 + 1e-9, 3, 1.7e308, 1.7e308, NA, -1.7e308, 7, 7, 7))
  )
  for (path in c("auto", "scans")) {
    withr::local_options(cuplyr.roll_path = path)
    for (k in seq_along(dfs)) {
      df <- dfs[[k]]
      for (lazy in c(FALSE, TRUE)) {
        got <- tbl_gpu(df, lazy = lazy) |> dplyr::group_by(g) |>
          dplyr::mutate(s = roll_sd(x, 2), m = roll_mean(x, 2), s2 = roll_sd(x, 2, na.rm = TRUE)) |>
          collect()
        for (nm in c("s", "m", "s2")) {
          stat <- if (nm == "m") "mean" else "sd"
          ref <- rep(NA_real_, nrow(df))
          for (i in split(seq_len(nrow(df)), df$g)) ref[i] <- ref_roll(df$x[i], 2L, stat, nm == "s2")
          info <- sprintf("df %d %s path=%s lazy=%s", k, nm, path, lazy)
          if (nm == "m") {
            # (windows never cross groups, so the ungrouped huge_window() is a superset)
            expect_sums_exact(got[[nm]], ref, df$x, 2L, info)
          } else {
            expect_rows_exact(got[[nm]], ref, info)
          }
        }
      }
    }
  }
})

test_that("B7: roll_mean() beside +-1e308 is the same with and without roll_sd() at its width", {
  skip_if_no_gpu()
  z <- c(rep(1, 99), 1e308, -1e308, rep(1, 99))
  ref_m <- ref_roll(z, 100L, "mean")
  ref_s <- ref_roll(z, 100L, "sd")
  # The windows holding both 1e308 and -1e308 cancel them, and whether the
  # ones beside them survive depends on the order of summation (R adds left
  # to right, the scans add each window part separately), so there the mean
  # is only within 1 of R's; every other row is exact.
  both <- seq_along(z) %in% 101:199
  for (with_sd in c(FALSE, TRUE)) {
    for (lazy in c(FALSE, TRUE)) {
      gt <- tbl_gpu(data.frame(x = z), lazy = lazy)
      got <- if (with_sd) {
        collect(dplyr::mutate(gt, m = roll_mean(x, 100), s = roll_sd(x, 100)))
      } else {
        collect(dplyr::mutate(gt, m = roll_mean(x, 100)))
      }
      info <- sprintf("z with_sd=%s lazy=%s", with_sd, lazy)
      expect_rows_exact(got$m[!both], ref_m[!both], info)
      expect_equal(got$m[100], 1e306, tolerance = 1e-15, info = info)
      expect_true(all(is.finite(got$m[both])), info = info)
      expect_lte(max(abs(got$m[both] - ref_m[both])), 1, label = info)
      if (with_sd) expect_rows_exact(got$s, ref_s, paste(info, "sd"))
    }
  }
})
