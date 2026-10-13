# F46 acceptance tests, part 2: cases that stay on (or may use) the CPU route
# must still give lubridate's exact results, eager and lazy.
#
# Nothing here is pinned native. Covers:
#   - floor/ceiling/round_date where the native path must NOT be used:
#     non-UTC zones (wall clock, DST days, a half-hour offset), multi-unit
#     strings, calendar units, week on a POSIXct, explicit week_start /
#     change_on_boundary, a non-default lubridate.week.start option, a Date
#     with a sub-day unit (lubridate returns a UTC POSIXct), a unit held in a
#     variable, a user's own floor_date(), and an invalid unit (lubridate's
#     error).
#   - a smoke test of other lubridate functions through the CPU fallback.
#
# 2024 US DST: 2024-03-10 07:00:00 UTC = 03:00 EDT (02:xx does not exist);
# 2024-11-03 05:30 UTC = 01:30 EDT and 06:30 UTC = 01:30 EST (repeated hour).

f46f_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f46f_instants <- function(tz) {
  x <- as.POSIXct(c("2024-03-10 04:59:59", "2024-03-10 06:59:59",
                    "2024-03-10 07:00:00", "2024-03-10 07:30:00",
                    "2024-11-03 04:00:00", "2024-11-03 05:30:00",
                    "2024-11-03 06:30:00", "2024-11-03 17:00:00",
                    "2024-11-04 03:00:00", "2024-01-31 12:00:00", NA),
                  tz = "UTC")
  attr(x, "tzone") <- tz
  x
}

f46f_df <- function(tz) {
  t <- f46f_instants(tz)
  data.frame(t = t, g = rep(1:2, length.out = length(t)), v = seq_along(t))
}

f46f_date_df <- function() {
  d <- c(as.Date("2024-01-01") + (-3:7), as.Date(c("1969-12-28", "2024-02-29")), NA)
  data.frame(dt = d, v = seq_along(d))
}

test_that("non-UTC zones give lubridate's wall-clock results (DST, half-hour offset)", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  for (tz in c("America/New_York", "Asia/Kolkata", "Etc/GMT+5")) {
    f46f_both(f46f_df(tz), function(d) {
      dplyr::mutate(d,
        f_m = lubridate::floor_date(t, "minute"),
        f_h = lubridate::floor_date(t, "hour"),
        c_h = lubridate::ceiling_date(t, "hour"),
        r_h = lubridate::round_date(t, "hour"),
        f_d = lubridate::floor_date(t, "day"),
        c_d = lubridate::ceiling_date(t, "day"),
        r_d = lubridate::round_date(t, "day")
      )
    })
  }
})

test_that("group_by(floor_date()) in a DST zone matches lubridate", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46f_both(f46f_df("America/New_York"), function(d) {
    d |>
      dplyr::group_by(day = lubridate::floor_date(t, "day")) |>
      dplyr::summarise(n = dplyr::n(), s = sum(v))
  })
})

test_that("units outside the native set match lubridate", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46f_both(f46f_df("UTC"), function(d) {
    dplyr::mutate(d,
      h2 = lubridate::floor_date(t, "2 hours"),
      m15 = lubridate::ceiling_date(t, "15 mins"),
      r6h = lubridate::round_date(t, "6 hours"),
      mo = lubridate::floor_date(t, "month"),
      mo_c = lubridate::ceiling_date(t, "month"),
      q = lubridate::floor_date(t, "quarter"),
      y = lubridate::round_date(t, "year"),
      wk = lubridate::floor_date(t, "week"),
      m_lower = lubridate::floor_date(t, "m")
    )
  })
  f46f_both(f46f_date_df(), function(d) {
    dplyr::mutate(d,
      mo = lubridate::floor_date(dt, "month"),
      bm = lubridate::floor_date(dt, "bimonth"),
      se = lubridate::floor_date(dt, "season"),
      d2 = lubridate::floor_date(dt, "2 days")
    )
  })
})

test_that("explicit week_start / change_on_boundary and the week-start option match lubridate", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46f_both(f46f_date_df(), function(d) {
    dplyr::mutate(d,
      w1 = lubridate::floor_date(dt, "week", week_start = 1),
      w7 = lubridate::ceiling_date(dt, "week", week_start = 7),
      cob_f = lubridate::ceiling_date(dt, "day", change_on_boundary = FALSE),
      cob_w = lubridate::ceiling_date(dt, "week", change_on_boundary = FALSE)
    )
  })
  f46f_both(f46f_df("UTC"), function(d) {
    dplyr::mutate(d,
      cob_t = lubridate::ceiling_date(t, "day", change_on_boundary = TRUE),
      cob_h = lubridate::ceiling_date(t, "hour", change_on_boundary = TRUE)
    )
  })
  withr::local_options(lubridate.week.start = 1)
  f46f_both(f46f_date_df(), function(d) {
    dplyr::mutate(d,
      f = lubridate::floor_date(dt, "week"),
      c = lubridate::ceiling_date(dt, "week"),
      r = lubridate::round_date(dt, "week")
    )
  })
})

test_that("a Date with a sub-day unit gives lubridate's UTC POSIXct", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46f_both(f46f_date_df(), function(d) {
    dplyr::mutate(d,
      f = lubridate::floor_date(dt, "hour"),
      c = lubridate::ceiling_date(dt, "minute"),
      r = lubridate::round_date(dt, "second")
    )
  })
})

test_that("a unit held in a variable matches lubridate", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  u <- "hour"
  f46f_both(f46f_df("UTC"), function(d) {
    dplyr::mutate(d, f = lubridate::floor_date(t, u))
  })
})

test_that("a user's own floor_date() is called, not lubridate's", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  floor_date <- function(x, unit) x + 1
  f46f_both(f46f_df("UTC"), function(d) {
    dplyr::mutate(d, f = floor_date(t, "hour"))
  })
})

test_that("an invalid unit raises lubridate's error", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- f46f_df("UTC")
  pipeline <- function(d) dplyr::mutate(d, f = lubridate::floor_date(t, "hr"))
  expect_error(pipeline(tibble::as_tibble(df)), "Invalid unit specification")
  expect_error(dplyr::collect(pipeline(tbl_gpu(df))), "Invalid unit specification")
  expect_error(dplyr::collect(pipeline(tbl_gpu(df, lazy = TRUE))), "Invalid unit specification")
})

# --- Fallback smoke test: other lubridate functions -------------------------

test_that("lubridate parsers and constructors work through the fallback", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(
    s = c("2024-03-10", "2024-11-03", "1969-12-31", NA),
    sh = c("2024-03-10 02:30:00", "2024-11-03 01:30:00", "1969-12-31 23:59:59", NA),
    y = c(2024L, 2024L, 1969L, NA), mo = c(3L, 11L, 12L, 1L),
    dd = c(10L, 3L, 31L, 1L), h = c(2L, 1L, 23L, 0L), mi = c(30L, 30L, 59L, 0L)
  )
  f46f_both(df, function(d) {
    dplyr::mutate(d,
      a = lubridate::ymd(s),
      b = lubridate::ymd_hms(sh),
      b_ny = lubridate::ymd_hms(sh, tz = "America/New_York"),
      c = lubridate::make_date(y, mo, dd),
      e = lubridate::make_datetime(y, mo, dd, h, mi),
      e_ny = lubridate::make_datetime(y, mo, dd, h, mi, tz = "America/New_York")
    )
  })
})

test_that("week(), period arithmetic and zone changes work through the fallback", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  `%m+%` <- lubridate::`%m+%`
  for (tz in c("UTC", "America/New_York")) {
    f46f_both(f46f_df(tz), function(d) {
      dplyr::mutate(d,
        wk = lubridate::week(t),
        m1 = t %m+% months(1),
        d1 = t + lubridate::days(1),
        h2 = t + lubridate::hours(2),
        w = lubridate::with_tz(t, "Asia/Tokyo"),
        f = lubridate::force_tz(t, "Europe/Paris"),
        mo = lubridate::floor_date(t, "month")
      )
    })
  }
  f46f_both(f46f_date_df(), function(d) {
    dplyr::mutate(d,
      wk = lubridate::week(dt),
      m1 = dt %m+% months(1),
      d1 = dt + lubridate::days(1)
    )
  })
})

# Added after round-1 review (B1): a classed shift operand inside a rounding
# call (Period, Duration, difftime) must give lubridate's result on whatever
# route it takes. A Period's numeric data part is 0, so treating it as a plain
# number silently drops the shift.
test_that("rounding a POSIXct shifted by a Period/Duration/difftime matches lubridate", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(
    t = as.POSIXct("2024-01-01 10:20:00", tz = "UTC") + c(0, 1800, 4000, -86400 * 20000, NA)
  )
  # One expression per mutate: a sibling that falls back would otherwise
  # send the whole mutate to the CPU and hide a wrong native answer.
  exprs <- rlang::exprs(
    a = lubridate::floor_date(t + lubridate::hours(1), "hour"),
    b = lubridate::round_date(t - lubridate::minutes(30), "hour"),
    c = lubridate::ceiling_date(t + lubridate::days(1), "day"),
    e = lubridate::floor_date(lubridate::hours(2) + t, "minute"),
    f = lubridate::floor_date(t + lubridate::dhours(1), "hour"),
    g = lubridate::round_date(t - as.difftime(90, units = "mins"), "hour")
  )
  for (nm in names(exprs)) {
    e <- exprs[[nm]]
    f46f_both(df, function(d) dplyr::mutate(d, !!nm := !!e))
  }
})

# Added after round-2 review (B3): a Date shifted by a Period/Duration, bare
# and inside each rounding function. Period/Duration literals are S4 objects
# with double storage; coercing one to a day count gives seconds. One
# expression per mutate so a falling-back sibling can't mask a wrong answer.
test_that("a Date shifted by a Period/Duration matches lubridate, bare and rounded", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- f46f_date_df()
  exprs <- rlang::exprs(
    b1 = dt + lubridate::days(1),
    b2 = dt + lubridate::ddays(1),
    b3 = dt - lubridate::weeks(1),
    b4 = lubridate::days(2) + dt,
    r1 = lubridate::floor_date(dt + lubridate::days(1), "week"),
    r2 = lubridate::ceiling_date(dt + lubridate::days(1), "day"),
    r3 = lubridate::round_date(dt - lubridate::weeks(1), "week"),
    r4 = lubridate::floor_date(dt + lubridate::ddays(1), "week"),
    r5 = lubridate::round_date(dt + lubridate::dweeks(1), "day")
  )
  for (nm in names(exprs)) {
    e <- exprs[[nm]]
    f46f_both(df, function(d) dplyr::mutate(d, !!nm := !!e))
  }
})
