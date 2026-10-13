# F46 acceptance tests: native lubridate floor_date()/ceiling_date()/
# round_date() for fixed units.
#
# lubridate 1.9.5 / timechange 0.4.0 behaviour (verified with tools/r):
#   - Units: "second" (also seconds/sec/secs/S/s), "minute" (minutes/min/mins/
#     M; lower-case "m" is MONTH), "hour" (hours/h/H), "day" (days/d/D),
#     "week" (weeks/w/W). Default unit: "seconds".
#   - round_date() ties go UP (timechange::time_round: `above - mid <=
#     mid - below`), also for instants before 1970. This is not
#     round-half-to-even.
#   - ceiling_date(): change_on_boundary defaults to TRUE for a Date (a value
#     already on the boundary moves up one unit) and FALSE for a POSIXct.
#   - "week" starts on getOption("lubridate.week.start", 7) = Sunday.
#   - The result keeps the input's class and tzone; a POSIXct without a tzone
#     attribute comes back with tzone "".
#   - A Date with a sub-day unit returns a UTC POSIXct (CPU here, see the
#     fallback file).
#
# "Native" is pinned with options(cuplyr.fallback = "error"): any CPU
# fallback inside these pipelines raises.

f46_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f46_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f46_both(df, pipeline, ...)
}

# UTC instants: boundaries, exact ties at every unit (several where
# round-half-to-even would differ from lubridate's ties-up), sub-second
# values, instants before 1970 and an NA.
f46_utc_df <- function(tz = "UTC") {
  base <- as.POSIXct("2024-01-01", tz = "UTC")
  epoch <- as.POSIXct("1970-01-01", tz = "UTC")
  t <- c(
    base + c(0, 30, 90, 1800, 3599, 5400, 43199, 43200, 86400,
             86400 + 43200, 2.5, 2.25, 1.75, 59.75),
    epoch + c(-1.5, -0.5, -1, -43200, -5400, -30),
    NA
  )
  attr(t, "tzone") <- tz
  data.frame(
    t = t,
    g = rep(c(1L, 2L, 3L), length.out = length(t)),
    v = seq_along(t)
  )
}

# Dates around Sunday boundaries (2023-12-31, 2024-01-07, 1969-12-28 and
# 1970-01-04 are Sundays), before 1970, and an NA.
f46_date_df <- function() {
  d <- c(as.Date("2024-01-01") + (-3:7),
         as.Date(c("1969-12-27", "1969-12-28", "1969-12-31", "1970-01-01",
                   "1970-01-04")),
         NA)
  data.frame(dt = d, g = rep(1:2, length.out = length(d)), v = seq_along(d))
}

test_that("floor/ceiling/round_date on a UTC POSIXct are native for second..day", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46_native(f46_utc_df(), function(d) {
    dplyr::mutate(d,
      f_s = lubridate::floor_date(t, "second"),
      c_s = lubridate::ceiling_date(t, "sec"),
      r_s = lubridate::round_date(t, "secs"),
      f_m = lubridate::floor_date(t, "minute"),
      c_m = lubridate::ceiling_date(t, "mins"),
      r_m = lubridate::round_date(t, "M"),
      f_h = lubridate::floor_date(t, "hour"),
      c_h = lubridate::ceiling_date(t, "h"),
      r_h = lubridate::round_date(t, "hours"),
      f_d = lubridate::floor_date(t, "day"),
      c_d = lubridate::ceiling_date(t, "days"),
      r_d = lubridate::round_date(t, "D")
    )
  })
})

test_that("default unit, named unit and other accepted spellings are native", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46_native(f46_utc_df(), function(d) {
    dplyr::mutate(d,
      f0 = lubridate::floor_date(t),
      c0 = lubridate::ceiling_date(t),
      r0 = lubridate::round_date(t),
      fu = lubridate::floor_date(t, unit = "S"),
      cu = lubridate::ceiling_date(x = t, unit = "minutes"),
      rH = lubridate::round_date(t, "H"),
      fd = lubridate::floor_date(t, "d"),
      fmin = lubridate::floor_date(t, "min"),
      cs = lubridate::ceiling_date(t, "s"),
      rsec = lubridate::round_date(t, "seconds")
    )
  })
})

test_that("round_date ties go up, before and after 1970 (not half-to-even)", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  epoch <- as.POSIXct("1970-01-01", tz = "UTC")
  df <- data.frame(t = epoch + c(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5,
                                 -90, -30, 30, 90, -5400, -1800, 1800, 5400,
                                 -129600, -43200, 43200, 129600))
  f46_native(df, function(d) {
    dplyr::mutate(d,
      r_s = lubridate::round_date(t, "second"),
      r_m = lubridate::round_date(t, "minute"),
      r_h = lubridate::round_date(t, "hour"),
      r_d = lubridate::round_date(t, "day")
    )
  })
})

test_that("UTC-equivalent zones are native and keep their tzone", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  for (tz in c("GMT", "Etc/UTC")) {
    f46_native(f46_utc_df(tz), function(d) {
      dplyr::mutate(d,
        f_h = lubridate::floor_date(t, "hour"),
        c_m = lubridate::ceiling_date(t, "minute"),
        r_d = lubridate::round_date(t, "day")
      )
    })
  }
})

test_that("a tz-less POSIXct in a UTC session is native and comes back with tzone \"\"", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  withr::local_envvar(TZ = "UTC")
  df <- f46_utc_df()
  attr(df$t, "tzone") <- NULL
  f46_native(df, function(d) {
    dplyr::mutate(d,
      f_h = lubridate::floor_date(t, "hour"),
      r_d = lubridate::round_date(t, "day")
    )
  })
})

test_that("floor/ceiling/round_date on a Date are native for day and week", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46_native(f46_date_df(), function(d) {
    dplyr::mutate(d,
      f_d = lubridate::floor_date(dt, "day"),
      c_d = lubridate::ceiling_date(dt, "day"),
      r_d = lubridate::round_date(dt, "days"),
      f_w = lubridate::floor_date(dt, "week"),
      c_w = lubridate::ceiling_date(dt, "weeks"),
      r_w = lubridate::round_date(dt, "week"),
      f_w2 = lubridate::floor_date(dt, "w"),
      c_w2 = lubridate::ceiling_date(dt, "W")
    )
  })
})

test_that("a bare floor_date() resolving to lubridate's own function is native", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  floor_date <- lubridate::floor_date
  ceiling_date <- lubridate::ceiling_date
  round_date <- lubridate::round_date
  f46_native(f46_utc_df(), function(d) {
    dplyr::mutate(d,
      f = floor_date(t, "hour"),
      c = ceiling_date(t, "hour"),
      r = round_date(t, "hour")
    )
  })
})

test_that("group_by(h = floor_date(t, 'hour')) |> summarise() is native end to end", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46_native(f46_utc_df(), function(d) {
    d |>
      dplyr::group_by(h = lubridate::floor_date(t, "hour")) |>
      dplyr::summarise(n = dplyr::n(), s = sum(v))
  })
  f46_native(f46_date_df(), function(d) {
    d |>
      dplyr::group_by(w = lubridate::floor_date(dt, "week")) |>
      dplyr::summarise(n = dplyr::n(), s = sum(v))
  })
})

test_that("filter predicates with floor/ceiling/round_date are native", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  cut <- as.POSIXct("2024-01-01 01:00:00", tz = "UTC")
  f46_native(f46_utc_df(), function(d) {
    dplyr::filter(d, lubridate::floor_date(t, "hour") == cut)
  })
  f46_native(f46_utc_df(), function(d) {
    dplyr::filter(d, lubridate::round_date(t, "day") > lubridate::floor_date(t, "day"))
  })
  f46_native(f46_date_df(), function(d) {
    dplyr::filter(d, lubridate::ceiling_date(dt, "week") > as.Date("2024-01-01"))
  })
})

test_that("summarise aggregates over floor/ceiling/round_date are native", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46_native(f46_utc_df(), function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(
        mx = max(lubridate::floor_date(t, "day"), na.rm = TRUE),
        mn = min(lubridate::round_date(t, "hour"), na.rm = TRUE),
        nd = dplyr::n_distinct(lubridate::ceiling_date(t, "minute"))
      )
  })
  f46_native(f46_utc_df("GMT"), function(d) {
    dplyr::summarise(d, mx = max(lubridate::floor_date(t, "hour"), na.rm = TRUE))
  })
})

test_that("grouped mutate, transmute and nested expressions are native", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  f46_native(f46_utc_df(), function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(h = lubridate::ceiling_date(t, "minute"))
  })
  f46_native(f46_date_df(), function(d) {
    dplyr::transmute(d, w = lubridate::floor_date(dt, "week"), v)
  })
  f46_native(f46_utc_df(), function(d) {
    dplyr::mutate(d,
      a = lubridate::floor_date(t + 3600, "hour"),
      b = lubridate::floor_date(t, "day") == lubridate::round_date(t, "day"),
      c = lubridate::ceiling_date(lubridate::floor_date(t, "minute"), "hour")
    )
  })
})

test_that("zero-row inputs are native and keep the column types", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- f46_utc_df()[0, ]
  df$dt <- as.Date(character(0))
  f46_native(df, function(d) {
    dplyr::mutate(d,
      f = lubridate::floor_date(t, "hour"),
      c = lubridate::ceiling_date(t, "day"),
      r = lubridate::round_date(t, "minute"),
      fw = lubridate::floor_date(dt, "week"),
      cd = lubridate::ceiling_date(dt, "day")
    )
  })
})

# Added after round-1 review (B2): a column-free call (no column reference)
# that is outside the native rounding scope must still constant-fold to a
# literal, as it did on dev, so the surrounding verb stays native.
test_that("column-free out-of-scope floor/ceiling/round_date calls fold to literals", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  ddf <- data.frame(dt = as.Date("2023-12-25") + 0:40, v = 1:41)
  f46_native(ddf, function(d) {
    dplyr::filter(d, dt >= lubridate::floor_date(as.Date("2024-01-10"), "month"))
  })
  f46_native(ddf, function(d) {
    dplyr::filter(d,
      dt < lubridate::ceiling_date(as.Date("2024-01-10"), "month"),
      dt >= lubridate::floor_date(as.Date("2024-01-03"), "week", week_start = 1)
    )
  })
  f46_native(f46_utc_df(), function(d) {
    dplyr::mutate(d,
      ny = lubridate::floor_date(
        as.POSIXct("2024-03-10 12:00:00", tz = "America/New_York"), "day"),
      q = lubridate::round_date(as.POSIXct("2024-05-17", tz = "UTC"), "quarter"),
      keep = t >= lubridate::floor_date(as.POSIXct("2024-01-01 05:00", tz = "UTC"), "2 hours")
    )
  })
})
