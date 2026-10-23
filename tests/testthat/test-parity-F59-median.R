# F59 acceptance tests: roll_median()'s CPU path is fast and still matches
# the old R reference (R/roll.R's roll_apply_generic(x, n, stats::median,
# na.rm), frozen below) exactly: identical(), including NA/NaN handling (F55:
# a NaN input is missing), integer and logical input, Inf, and overflow of
# the two middle values' midpoint.
#
# Two engines: data.table::frollmedian() when data.table is installed, and an
# internal data.table-free O(n log w) roll_median_sorted(x, n, na.rm).

f59m_ref <- function(x, n, na.rm = FALSE) {
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n != trunc(n) || n < 1) {
    stop("`n` must be a single positive whole number.", call. = FALSE)
  }
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
    out[i] <- stats::median(w)
  }
  out
}

# Random vectors over every special value, some integer, some logical.
f59m_cases <- function() {
  cases <- list()
  for (seed in 1:60) {
    set.seed(seed)
    len <- sample(c(0:5, 50, 400), 1)
    y <- sample(c(rnorm(5), NA, NaN, Inf, -Inf, 0, 2, 2), len, TRUE)
    if (seed %% 3 == 0) y <- suppressWarnings(as.integer(round(y * 3)))
    if (seed %% 7 == 0) y <- y > 0
    cases[[length(cases) + 1L]] <- y
  }
  cases[[length(cases) + 1L]] <- c(1e308, 1.5e308, 1.7e308, -1.7e308, -1.6e308, 1e308)
  cases[[length(cases) + 1L]] <- c(NA, NA, NA, 1, NaN, NaN, 2, 3)
  cases
}

f59m_widths <- c(1:6, 9L, 10L, 60L)

f59m_expect_engine <- function(engine, label) {
  for (y in f59m_cases()) {
    for (w in f59m_widths) {
      for (na_rm in c(FALSE, TRUE)) {
        expect_identical(
          engine(y, w, na_rm), f59m_ref(y, w, na_rm),
          info = sprintf("%s: w=%d na.rm=%s x=%s", label, w, na_rm,
                         paste(utils::head(y, 12), collapse = ","))
        )
      }
    }
  }
}

test_that("F59: roll_median() is identical to the old R reference (fuzz incl. NA/NaN/Inf/int/lgl)", {
  f59m_expect_engine(function(x, n, na.rm) roll_median(x, n, na.rm = na.rm), "roll_median")
})

test_that("F59: roll_median() midpoint of two huge values does not overflow", {
  x <- c(1e308, 1.5e308, 1.7e308, -1.7e308, -1.6e308)
  expect_identical(roll_median(x, 2), f59m_ref(x, 2))
  expect_identical(roll_median(x, 2)[2:3], c(1.25e308, 1.6e308))
  expect_identical(roll_median(x, 4, na.rm = TRUE), f59m_ref(x, 4, na.rm = TRUE))
})

test_that("F59: the data.table-free roll_median_sorted() is identical to the old R reference", {
  sorted <- getFromNamespace("roll_median_sorted", "cuplyr")
  f59m_expect_engine(function(x, n, na.rm) sorted(x, n, na.rm), "roll_median_sorted")
})

test_that("F59: both median engines validate n like before", {
  sorted <- getFromNamespace("roll_median_sorted", "cuplyr")
  msg <- "`n` must be a single positive whole number."
  for (bad in list(0, -1, 1.5, NA, c(2, 3), "3")) {
    expect_error(roll_median(1:5, bad), msg, fixed = TRUE)
    expect_error(sorted(1:5, bad, FALSE), msg, fixed = TRUE)
  }
  expect_identical(roll_median(numeric(0), 3), numeric(0))
  expect_identical(sorted(numeric(0), 3, FALSE), numeric(0))
  expect_identical(roll_median(1:3, 5), rep(NA_real_, 3))
  expect_identical(sorted(1:3, 5, TRUE), rep(NA_real_, 3))
})

test_that("F59: roll_median() at 1e6 rows takes well under a second; the sorted engine a few seconds", {
  set.seed(5910)
  x <- rnorm(1e6)
  x[sample.int(1e6, 1e4)] <- NA
  sorted <- getFromNamespace("roll_median_sorted", "cuplyr")
  r1 <- NULL
  if (requireNamespace("data.table", quietly = TRUE)) {
    t_dt <- system.time(r1 <- roll_median(x, 1001, na.rm = TRUE))[["elapsed"]]
    expect_lt(t_dt, 1)
  }
  t_sorted <- system.time(r2 <- sorted(x, 1001, TRUE))[["elapsed"]]
  expect_lt(t_sorted, 5)
  if (!is.null(r1)) expect_identical(r1, r2)
  # spot-check against the reference on a slice (the full reference takes ~30 s)
  idx <- 5000:6500
  expect_identical(r2[idx], f59m_ref(x[4000:6500], 1001, TRUE)[idx - 3999])
})

test_that("F59: roll_median() inside tbl_gpu pipelines (grouped, NA keys, eager and lazy) matches the reference", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  set.seed(5911)
  n <- 3000
  df <- data.frame(g = sample(c(1:5, NA), n, TRUE), x = round(rnorm(n), 3))
  df$x[sample.int(n, 100)] <- NA
  df$x[sample.int(n, 30)] <- NaN
  df$x[sample.int(n, 10)] <- Inf
  for (lazy in c(FALSE, TRUE)) {
    for (na_rm in c(FALSE, TRUE)) {
      got <- withr::with_options(list(cuplyr.fallback = "silent"), {
        tbl_gpu(df, lazy = lazy) |>
          dplyr::group_by(g) |>
          dplyr::mutate(m = roll_median(x, 7, na.rm = na_rm), m2 = data.table::frollmedian(x, 4, na.rm = na_rm)) |>
          collect()
      })
      ref <- ref2 <- rep(NA_real_, n)
      for (idx in split(seq_len(n), factor(df$g, exclude = NULL))) {
        ref[idx] <- f59m_ref(df$x[idx], 7, na_rm)
        ref2[idx] <- f59m_ref(df$x[idx], 4, na_rm)
      }
      expect_identical(got$m, ref, info = sprintf("lazy=%s na.rm=%s", lazy, na_rm))
      expect_identical(got$m2, ref2, info = sprintf("frollmedian lazy=%s na.rm=%s", lazy, na_rm))
    }
  }
})
