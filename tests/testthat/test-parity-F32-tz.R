# F32 acceptance tests: POSIXct time zone preservation.
#
# cuplyr keeps each POSIXct instant as UTC microseconds on the GPU; the
# column's `tzone` is R-side metadata that must survive upload, every verb
# and collect, and the datetime accessors must use it (wall clock in the
# column's zone, exact across DST switches).
#
# dplyr 1.2.1 / vctrs 0.7.3 / lubridate 1.9.5 behaviour (verified with
# tools/r, scratchpad/parity/features/F32/probe1.R, probe2.R):
#   - vec_ptype2(x, y) of two POSIXct keeps x's tzone, unless x is local
#     (""), then y's. Date + POSIXct -> POSIXct in the POSIXct's tzone.
#     bind_rows(), if_else(), coalesce(), case_when(), joins follow it.
#   - A Date cast to POSIXct(tz) is local midnight in tz (DST-dependent).
#   - t + 3600, lag(), min()/max()/first()/last() keep the tzone.
#   - as.Date(<POSIXct>) defaults to tz = "UTC" (R 4.6), so it gives the UTC
#     calendar day; lubridate::as_date() uses the column's tzone.
#   - A tz-less POSIXct (tzone "" or absent) is local time.
#
# The instants below sit on both sides of both 2024 US DST switches,
# including the repeated 01:30 hour of 3 November:
#   2024-03-10 06:59:59 UTC = 01:59:59 EST   2024-03-10 07:00:00 UTC = 03:00:00 EDT
#   2024-11-03 05:30:00 UTC = 01:30:00 EDT   2024-11-03 06:30:00 UTC = 01:30:00 EST
#   2024-11-04 03:00:00 UTC = 2024-11-03 22:00:00 EST

f32_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f32_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f32_instants <- function(tz) {
  x <- as.POSIXct(c("2024-03-10 06:59:59", "2024-03-10 07:00:00",
                    "2024-11-03 05:30:00", "2024-11-03 06:30:00",
                    "2024-11-04 03:00:00", NA), tz = "UTC")
  attr(x, "tzone") <- tz
  x
}

f32_df <- function(tz) {
  data.frame(t = f32_instants(tz), g = c(1, 1, 2, 2, 2, 2), v = 1:6)
}

# UTC, two fixed offsets (one with a half-hour offset), a DST zone.
f32_zones <- c("UTC", "Etc/GMT+5", "Asia/Kolkata", "America/New_York")

f32_accessors <- function(d) {
  dplyr::mutate(d,
    h = lubridate::hour(t), mi = lubridate::minute(t), s = lubridate::second(t),
    dd = lubridate::day(t), mo = lubridate::month(t), y = lubridate::year(t),
    wd = lubridate::wday(t), yd = lubridate::yday(t), q = lubridate::quarter(t)
  )
}

# -----------------------------------------------------------------------------
# 1. Upload / collect round trip
# -----------------------------------------------------------------------------

test_that("collect() keeps each POSIXct column's tzone", {
  skip_if_no_gpu()
  for (tz in c(f32_zones, "")) {
    f32_both(f32_df(tz), function(d) d)
  }
  # two columns in different zones in one table
  df <- data.frame(a = f32_instants("America/New_York"), b = f32_instants("Asia/Kolkata"))
  f32_both(df, function(d) d)
  out <- collect(tbl_gpu(df))
  expect_identical(attr(out$a, "tzone"), "America/New_York")
  expect_identical(attr(out$b, "tzone"), "Asia/Kolkata")
})

test_that("a POSIXct without a tzone attribute round-trips without one", {
  skip_if_no_gpu()
  df <- data.frame(g = 1:3)
  df$t <- .POSIXct(c(0, 3600, NA))
  expect_null(attr(df$t, "tzone"))
  out <- collect(tbl_gpu(df))
  expect_null(attr(out$t, "tzone"))
  expect_equal(unclass(out$t), unclass(df$t))
  out_lazy <- collect(dplyr::select(tbl_gpu(df, lazy = TRUE), t))
  expect_null(attr(out_lazy$t, "tzone"))
})

test_that("zero-row and all-NA POSIXct columns keep their tzone", {
  skip_if_no_gpu()
  df0 <- data.frame(t = f32_instants("America/New_York")[0], v = integer())
  f32_both(df0, function(d) d)
  f32_both(df0, function(d) dplyr::filter(d, v > 0L))
  dfna <- data.frame(t = f32_instants("Asia/Kolkata")[c(6, 6)], v = 1:2)
  f32_both(dfna, function(d) d)
})

test_that("pull(), $, head() and print() see the column's zone", {
  skip_if_no_gpu()
  df <- f32_df("America/New_York")
  gt <- tbl_gpu(df)
  expect_identical(dplyr::pull(gt, t), df$t)
  expect_identical(gt$t, df$t)
  expect_identical(attr(collect(head(gt, 2))$t, "tzone"), "America/New_York")
  printed <- paste(utils::capture.output(print(gt)), collapse = "\n")
  expect_match(printed, "2024-03-10 01:59:59", fixed = TRUE)
  expect_no_match(printed, "2024-03-10 06:59:59", fixed = TRUE)
})

# -----------------------------------------------------------------------------
# 2. Verbs carry the tzone
# -----------------------------------------------------------------------------

test_that("row and column verbs keep the tzone", {
  skip_if_no_gpu()
  for (tz in c("America/New_York", "Etc/GMT+5", "")) {
    df <- f32_df(tz)
    f32_both(df, function(d) dplyr::filter(d, v > 2L))
    f32_both(df, function(d) dplyr::arrange(d, dplyr::desc(t)))
    f32_both(df, function(d) dplyr::select(d, when = t, v))
    f32_both(df, function(d) dplyr::rename(d, when = t))
    f32_both(df, function(d) dplyr::slice(d, 2:4))
    f32_both(df, function(d) dplyr::distinct(d, t))
    f32_both(df, function(d) dplyr::relocate(d, v))
  }
})

test_that("mutate() copies and arithmetic keep the tzone", {
  skip_if_no_gpu()
  for (tz in c("America/New_York", "Asia/Kolkata", "")) {
    df <- f32_df(tz)
    f32_both(df, function(d) dplyr::mutate(d, t2 = t, t3 = t + 3600, t4 = t - 60))
    f32_both(df, function(d) dplyr::transmute(d, later = t + v))
    f32_both(df, function(d) dplyr::mutate(d, prev = dplyr::lag(t), nxt = dplyr::lead(t)))
    f32_both(df, function(d) {
      d |> dplyr::group_by(g) |> dplyr::mutate(prev = dplyr::lag(t))
    })
  }
})

test_that("summarise() min/max/first/last keep the tzone", {
  skip_if_no_gpu()
  for (tz in c("America/New_York", "Etc/GMT+5", "")) {
    df <- f32_df(tz)
    f32_both(df, function(d) {
      dplyr::summarise(d, lo = min(t, na.rm = TRUE), hi = max(t, na.rm = TRUE),
                       f = dplyr::first(t), l = dplyr::last(t))
    })
    f32_both(df, function(d) {
      d |> dplyr::group_by(g) |>
        dplyr::summarise(lo = min(t, na.rm = TRUE), hi = max(t, na.rm = TRUE))
    })
    # grouping by the POSIXct column keeps its tzone on the key
    f32_both(df, function(d) d |> dplyr::group_by(t) |> dplyr::summarise(n = dplyr::n()))
    f32_both(df, function(d) dplyr::count(d, t))
  }
})

test_that("a POSIXct literal compares as an instant", {
  skip_if_no_gpu()
  df <- f32_df("America/New_York")
  lit <- as.POSIXct("2024-03-10 03:00:00", tz = "America/New_York")
  f32_both(df, function(d) dplyr::filter(d, t >= lit))
  f32_both(df, function(d) dplyr::filter(d, t < as.POSIXct("2024-11-03 01:30:00", tz = "UTC")))
})

# -----------------------------------------------------------------------------
# 3. Datetime accessors use the column's zone
# -----------------------------------------------------------------------------

test_that("lubridate accessors give wall-clock values in the column's zone", {
  skip_if_no_gpu()
  for (tz in f32_zones) {
    f32_both(f32_df(tz), f32_accessors, ignore_col_types = TRUE)
  }
})

test_that("accessors on a tz-less column use the session's local zone", {
  skip_if_no_gpu()
  withr::local_timezone("America/Chicago")
  f32_both(f32_df(""), f32_accessors, ignore_col_types = TRUE)
  withr::local_timezone("UTC")
  f32_both(f32_df(""), f32_accessors, ignore_col_types = TRUE)
})

test_that("accessors respect the zone in grouped mutate, filter and summarise", {
  skip_if_no_gpu()
  df <- f32_df("America/New_York")
  f32_both(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(h = lubridate::hour(t))
  }, ignore_col_types = TRUE)
  f32_both(df, function(d) dplyr::filter(d, lubridate::hour(t) == 1L))
  f32_both(df, function(d) dplyr::filter(d, lubridate::day(t) == 3L))
  f32_both(df, function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::summarise(hmax = max(lubridate::hour(t), na.rm = TRUE))
  }, ignore_col_types = TRUE)
  # accessor on a derived POSIXct keeps using the zone
  f32_both(df, function(d) dplyr::mutate(d, h = lubridate::hour(t + 3600)),
           ignore_col_types = TRUE)
})

test_that("as_date() uses the column's zone, as.Date() uses UTC", {
  skip_if_no_gpu()
  for (tz in c("America/New_York", "Asia/Kolkata")) {
    df <- f32_df(tz)
    f32_both(df, function(d) dplyr::mutate(d, D = lubridate::as_date(t)))
    f32_both(df, function(d) dplyr::mutate(d, D = as.Date(t)))
  }
})

# -----------------------------------------------------------------------------
# 4. Combining zones follows vctrs
# -----------------------------------------------------------------------------

test_that("if_else(), coalesce() and case_when() take vctrs' common tzone", {
  skip_if_no_gpu()
  df <- f32_df("America/New_York")
  df$u <- f32_instants("UTC")[c(2, 1, 6, 4, 3, 5)]
  df$l <- f32_instants("")[c(3, 4, 5, 6, 1, 2)]
  f32_both(df, function(d) dplyr::mutate(d, z = dplyr::if_else(v > 3L, t, u)))
  f32_both(df, function(d) dplyr::mutate(d, z = dplyr::if_else(v > 3L, u, t)))
  f32_both(df, function(d) dplyr::mutate(d, z = dplyr::if_else(v > 3L, l, t)))
  f32_both(df, function(d) dplyr::mutate(d, z = dplyr::coalesce(t, u)))
  f32_both(df, function(d) dplyr::mutate(d, z = dplyr::coalesce(u, t)))
  f32_both(df, function(d) {
    dplyr::mutate(d, z = dplyr::case_when(v > 4L ~ u, .default = t))
  })
})

test_that("bind_rows() and union() take vctrs' common tzone", {
  skip_if_no_gpu()
  ny <- data.frame(t = f32_instants("America/New_York"), v = 1:6)
  utc <- data.frame(t = f32_instants("UTC")[1:3], v = 7:9)
  loc <- data.frame(t = f32_instants("")[4:5], v = 10:11)
  f32_both(ny, function(d) cuplyr::bind_rows(d, f32_like(d, utc)))
  f32_both(utc, function(d) cuplyr::bind_rows(d, f32_like(d, ny)))
  f32_both(loc, function(d) cuplyr::bind_rows(d, f32_like(d, ny)))
  f32_both(ny, function(d) dplyr::union(d, f32_like(d, utc)), arrange_by = c("t", "v"))
})

test_that("bind_rows() of a Date and a zoned POSIXct casts to local midnight", {
  skip_if_no_gpu()
  dates <- data.frame(t = as.Date(c("2024-03-10", "2024-11-03", "2024-06-01")), v = 1:3)
  ny <- data.frame(t = f32_instants("America/New_York")[1:2], v = 4:5)
  f32_both(dates, function(d) cuplyr::bind_rows(d, f32_like(d, ny)))
  f32_both(ny, function(d) cuplyr::bind_rows(d, f32_like(d, dates)))
})

# -----------------------------------------------------------------------------
# 5. Joins
# -----------------------------------------------------------------------------

# NY midnight of 2024-03-10 (EST) is 05:00 UTC, of 2024-11-03 and 2024-06-01
# (EDT) 04:00 UTC. The 00:00 UTC rows would match if the Date were cast to
# UTC midnight instead.
f32_x_date <- data.frame(k = as.Date(c("2024-03-10", "2024-11-03", "2024-06-01", NA)), a = 1:4)
f32_y_ny <- local({
  k <- as.POSIXct(c("2024-03-10 05:00:00", "2024-11-03 04:00:00", "2024-06-01 04:00:00",
                    "2024-03-10 00:00:00", "2024-06-01 00:00:00"), tz = "UTC")
  attr(k, "tzone") <- "America/New_York"
  data.frame(k = k, b = 1:5)
})

test_that("Date x zoned POSIXct keys cast the Date to midnight in that zone", {
  skip_if_no_gpu()
  for (join in list(dplyr::left_join, dplyr::inner_join, dplyr::full_join)) {
    f32_both(f32_x_date, function(d) join(d, f32_like(d, f32_y_ny), by = "k"),
             arrange_by = c("k", "a", "b"))
    f32_both(f32_y_ny, function(d) join(d, f32_like(d, f32_x_date), by = "k"),
             arrange_by = c("k", "a", "b"))
  }
  f32_both(f32_x_date, function(d) dplyr::semi_join(d, f32_like(d, f32_y_ny), by = "k"))
  f32_both(f32_x_date, function(d) {
    dplyr::left_join(d, f32_like(d, f32_y_ny), by = "k", keep = TRUE)
  }, arrange_by = c("a", "b"))
})

test_that("Date x tz-less POSIXct keys use local midnight", {
  skip_if_no_gpu()
  withr::local_timezone("America/Chicago")
  y <- data.frame(k = as.POSIXct(c("2024-03-10 00:00:00", "2024-03-10 06:00:00")), b = 1:2)
  f32_both(f32_x_date, function(d) dplyr::left_join(d, f32_like(d, y), by = "k"),
           arrange_by = c("a", "b"))
})

test_that("POSIXct keys and columns keep their zones through joins", {
  skip_if_no_gpu()
  x <- data.frame(t = f32_instants("America/New_York")[1:5], a = 1:5)
  y_utc <- data.frame(t = f32_instants("UTC")[c(1, 3, 5)], b = 1:3,
                      seen = f32_instants("Asia/Kolkata")[1:3])
  for (join in list(dplyr::left_join, dplyr::inner_join, dplyr::right_join, dplyr::full_join)) {
    # merged key takes x's zone; y's non-key POSIXct keeps its own
    f32_both(x, function(d) join(d, f32_like(d, y_utc), by = "t"), arrange_by = c("a", "b"))
    f32_both(y_utc, function(d) join(d, f32_like(d, x), by = "t"), arrange_by = c("a", "b"))
  }
  # keep = TRUE: each side's key keeps its own zone
  f32_both(x, function(d) {
    dplyr::left_join(d, f32_like(d, y_utc), by = "t", keep = TRUE)
  }, arrange_by = c("a", "b"))
})

test_that("lazy pipelines keep zones through the optimizer", {
  skip_if_no_gpu()
  df <- f32_df("America/New_York")
  y <- data.frame(v = c(2L, 4L, 6L), seen = f32_instants("Asia/Kolkata")[1:3])
  f32_both(df, function(d) {
    d |>
      dplyr::mutate(t2 = t + 60, h = lubridate::hour(t)) |>
      dplyr::filter(v > 1L) |>
      dplyr::left_join(f32_like(d, y), by = "v") |>
      dplyr::select(t2, h, seen, v)
  }, ignore_col_types = TRUE)
  out <- tbl_gpu(df, lazy = TRUE) |> dplyr::mutate(t2 = t + 1) |> compute() |> collect()
  expect_identical(attr(out$t2, "tzone"), "America/New_York")
})
