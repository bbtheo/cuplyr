# F41 extras: timestamp mean()/median() details the acceptance file doesn't pin.

test_that("a fractional Date window mean names the user's output column", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-02", "2023-12-30", "1969-12-31"))
  )
  expect_warning(
    got <- collect(mutate(tbl_gpu(df), mm = mean(d), .by = g)),
    "`mm`",
    class = "cuplyr_warning_fractional_date"
  )
  want <- mutate(df, mm = mean(d), .by = g)
  expect_equal(as.numeric(got$mm), floor(as.numeric(want$mm)))
})

test_that("POSIXct mean/median round to microseconds, before and after 1970", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    t = as.POSIXct(c("1969-12-31 23:59:59.5", "1970-01-01 00:00:00.000003",
                     "2024-01-01 00:00:01", "2024-01-01 00:00:02", NA), tz = "UTC")
  )
  withr::local_options(cuplyr.fallback = "error")
  want <- summarise(df, m = mean(t, na.rm = TRUE), md = median(t, na.rm = TRUE), .by = g)
  got <- collect(summarise(tbl_gpu(df), m = mean(t, na.rm = TRUE), md = median(t, na.rm = TRUE), .by = g))
  # dplyr keeps sub-microsecond fractions of its double; the GPU rounds to the
  # microsecond like an upload does.
  expect_lte(max(abs(as.numeric(got$m) - as.numeric(want$m))), 1e-6)
  expect_lte(max(abs(as.numeric(got$md) - as.numeric(want$md))), 1e-6)
  expect_identical(got$g, want$g)
})

test_that("a timestamp mean/median inside a larger expression stays on the CPU and matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-02", "2023-12-30", "1969-12-31"))
  )
  expect_same_as_dplyr(df, function(t) filter(t, d >= mean(d)))
})

test_that("timestamp %in% against a non-literal or mixed set stays on the CPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(c("2024-01-01", NA)), x = c(19723, 1))
  expect_same_as_dplyr(df, function(t) mutate(t, y = d %in% x))
  expect_same_as_dplyr(df, function(t) mutate(t, y = x %in% as.Date("2024-01-01")))
})
