# F41 round 2 extras: what a later dot sees of a native timestamp mean/median,
# and transmute() with a window call.

# dplyr's answer for everything but `floored` (floored with the upload's
# warning), eager and lazy.
f41r2_check <- function(data, pipeline, floored = "m", tol = 0) {
  want <- tibble::as_tibble(dplyr::ungroup(pipeline(data)))
  rest <- setdiff(names(want), floored)
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(data, lazy = lazy)
    expect_warning(got <- collect(pipeline(g)), class = "cuplyr_warning_fractional_date")
    got <- tibble::as_tibble(dplyr::ungroup(got))
    expect_equal(names(got), names(want))
    for (nm in floored) {
      expect_equal(as.numeric(got[[nm]]), floor(as.numeric(want[[nm]])))
    }
    expect_equal(got[rest], want[rest], tolerance = tol)
  }
}

f41r2_dates <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    x = c(1, 2, 3, 4, 5),
    d = as.Date(c("2024-01-01", "2024-01-02", "2024-03-01", "2024-03-02", "2024-03-02"))
  )
}

test_that("a later dot never reads the floored column, whatever the spelling", {
  skip_if_no_gpu()
  df <- f41r2_dates()
  f41r2_check(df, function(t) mutate(t, m = mean(d), early = .data$m > d))
  f41r2_check(df, function(t) mutate(t, m = median(d), early = d < m, .by = g))
  f41r2_check(df, function(t) mutate(t, m = mean(d, na.rm = TRUE), early = d < m, on = d == m))
  f41r2_check(df, function(t) mutate(t, m = mean(d), early = dplyr::if_else(d < m, 1L, 0L)))
  # the reader comes after an unrelated dot, and feeds a third one
  f41r2_check(df, function(t) mutate(t, m = mean(d), y = x + 1, early = d < m, z = early & y > 2))
  # a column the mean overwrites
  want <- mutate(df, d = mean(d), early = d == as.Date(19759))
  expect_warning(got <- collect(mutate(tbl_gpu(df), d = mean(d), early = d == as.Date(19759))),
                 class = "cuplyr_warning_fractional_date")
  expect_equal(got$early, want$early)
})

test_that("a native timestamp mean/median stays native when nothing reads it", {
  skip_if_no_gpu()
  df <- f41r2_dates()
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    tb <- tbl_gpu(df, lazy = lazy)
    expect_warning(
      got <- collect(mutate(tb, m = mean(d), y = x + 1, m2 = m, .by = g)),
      class = "cuplyr_warning_fractional_date"
    )
    want <- mutate(df, m = mean(d), y = x + 1, m2 = m, .by = g)
    expect_equal(as.numeric(got$m), floor(as.numeric(want$m)))
    expect_equal(as.numeric(got$m2), floor(as.numeric(want$m2)))
    expect_equal(got$y, want$y)
  }
})

test_that("a later dot compares against the unrounded POSIXct mean", {
  skip_if_no_gpu()
  df <- tibble::tibble(t = as.POSIXct(c(0, 1e-6), origin = "1970-01-01", tz = "UTC"))
  want <- tibble::as_tibble(mutate(df, m = mean(t), early = t < m))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    got <- collect(mutate(g, m = mean(t), early = t < m))
    expect_identical(got$early, want$early)
    expect_lte(abs(as.numeric(got$m[1]) - as.numeric(want$m[1])), 1e-6)
  }
})

test_that("transmute() with a window call matches dplyr", {
  skip_if_no_gpu()
  df <- f41r2_dates()
  expect_same_as_dplyr(df, function(t) transmute(t, m = mean(x)))
  expect_same_as_dplyr(df, function(t) transmute(t, x, m = cumsum(x), e = x < m))
  expect_same_as_dplyr(df, function(t) t |> group_by(g) |> transmute(m = mean(x), y = x - m))
  expect_same_as_dplyr_lazy(df, function(t) transmute(t, m = mean(x)))
  # a mean that is a whole day
  whole <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-01-03")))
  expect_same_as_dplyr(whole, function(t) transmute(t, m = mean(d)))
  expect_same_as_dplyr(whole, function(t) transmute(t, d, m = mean(d), early = d < m))
})

test_that("transmute() reads dplyr's fractional Date mean in a later dot", {
  skip_if_no_gpu()
  df <- f41r2_dates()
  f41r2_check(df, function(t) transmute(t, d, m = mean(d), early = d < m))
  f41r2_check(df, function(t) t |> group_by(g) |> transmute(d, m = median(d), late = d >= m))
  f41r2_check(df, function(t) transmute(t, m = mean(d)), floored = "m")
})

test_that("a POSIXct mean rounds half away from zero, also past 2^52 microseconds", {
  skip_if_no_gpu()
  # half-microsecond means, either side of the epoch
  ties <- tibble::tibble(
    g = rep(1:4, each = 2),
    t = .POSIXct(c(0, 1e-6, -1e-6, 0, 2e-6, 3e-6, -3e-6, -2e-6), tz = "UTC")
  )
  got <- collect(summarise(tbl_gpu(ties), m = mean(t), .by = g))
  expect_equal(as.numeric(got$m), c(1e-6, -1e-6, 3e-6, -3e-6))
  # a one-row group: its mean is its value, odd microsecond counts included
  big <- tibble::tibble(
    k = 1:12,
    t = .POSIXct(7.2e9 + (0:11) * 1.000001e-6 + 1e-3, tz = "UTC")
  )
  gt <- tbl_gpu(big)
  stored <- collect(gt)$t
  got <- collect(summarise(gt, m = mean(t), .by = k))
  expect_identical(as.numeric(got$m), as.numeric(stored))
  got <- collect(mutate(gt, m = mean(t), .by = k))
  expect_identical(as.numeric(got$m), as.numeric(stored))
})
