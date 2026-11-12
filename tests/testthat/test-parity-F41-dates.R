# F41 acceptance tests: Date/POSIXct expression gaps.
#
# Base R semantics (dplyr adds nothing; verified with tools/r, see
# scratchpad/parity/features/F41/probe2.R):
#   - `%in%` (match() via mtfrm.Date/mtfrm.POSIXct) compares the raw numbers:
#     POSIXct vs POSIXct compares instants whatever the tzone, Date vs Date
#     compares days, NA matches NA when the set holds NA. Mixed Date/POSIXct,
#     Date vs number/string compare raw values (CPU route is fine there).
#   - as.Date(<Date>) is the identity, as.Date(<POSIXct>) the UTC day,
#     as.POSIXct(<Date>) UTC midnight.
#   - mean()/median() of a POSIXct give a POSIXct with the input's tzone; of
#     a Date, a Date (possibly fractional: cuplyr floors it with
#     `cuplyr_warning_fractional_date`, Theo's F37 decision).
#
# cuplyr contract (F41 design): same-type timestamp `%in%`/replace_values(),
# as.Date()/as.POSIXct() inside comparisons, and mean()/median() of Date/POSIXct
# in summarise() and window mutate() run natively (no CPU fallback).

f41_df <- function() {
  w <- c("2024-01-01 00:00:00", "2024-01-01 12:00:01", NA,
         "2024-03-10 03:30:00", "2024-12-31 23:59:59", "1969-12-31 23:00:00")
  tibble::tibble(
    g  = c(1L, 1L, 2L, 2L, 3L, 3L),
    d  = as.Date(c("2024-01-01", "2024-01-03", NA, "2024-03-10", "2024-12-31", "1969-12-30")),
    t  = as.POSIXct(w, tz = "UTC"),
    tn = as.POSIXct(w, tz = "America/New_York"),
    x  = c(1, 2, 3, 4, 5, 6)
  )
}

f41_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

# Same comparison, but any CPU fallback is an error.
f41_native <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

f41_dset <- as.Date(c("2024-01-03", "2024-12-31", "2030-01-01"))
f41_sset <- as.POSIXct(c("2024-01-01 12:00:01", "2024-03-10 03:30:00"), tz = "UTC")
# 2024-01-01 00:00 in New York is 05:00 UTC: matches tn[1] (same instant), not t[1].
f41_sset_shift <- as.POSIXct("2024-01-01 05:00:00", tz = "UTC")
f41_nyset <- as.POSIXct(c("2024-01-01 00:00:00", "2024-12-31 23:59:59"), tz = "America/New_York")

# --- item 1: %in% / replace_values() against timestamp sets --------------------

test_that("%in% against a Date set is native and matches dplyr", {
  skip_if_no_gpu()
  df <- f41_df()
  dset <- f41_dset
  f41_native(df, function(t) filter(t, d %in% dset))
  f41_native(df, function(t) filter(t, !(d %in% dset)))
  f41_native(df, function(t) mutate(t, y = d %in% dset))
  f41_native(df, function(t) mutate(t, y = d %in% c(dset, NA)))
  f41_native(df, function(t) filter(t, d %in% as.Date(c("2024-03-10", NA))))
  f41_native(df, function(t) filter(t, d %in% as.Date("1969-12-30")))
  f41_native(df, function(t) t |> group_by(g) |> filter(d %in% dset))
  f41_native(df, function(t) mutate(t, y = (d + 1L) %in% dset))
})

test_that("%in% against a POSIXct set is native and compares instants", {
  skip_if_no_gpu()
  df <- f41_df()
  sset <- f41_sset
  shift <- f41_sset_shift
  nyset <- f41_nyset
  f41_native(df, function(t) filter(t, t %in% sset))
  f41_native(df, function(t) filter(t, tn %in% sset))
  f41_native(df, function(t) filter(t, tn %in% shift))
  f41_native(df, function(t) filter(t, t %in% shift))
  f41_native(df, function(t) filter(t, t %in% nyset))
  f41_native(df, function(t) filter(t, tn %in% nyset))
  f41_native(df, function(t) mutate(t, y = t %in% c(sset, NA), z = tn %in% nyset))
  f41_native(df, function(t) t |> group_by(g) |> mutate(y = tn %in% sset))
})

test_that("timestamp %in% on 0-row data and with an empty set matches dplyr", {
  skip_if_no_gpu()
  df <- f41_df()
  dset <- f41_dset
  sset <- f41_sset
  f41_both(df[0, ], function(t) filter(t, d %in% dset))
  f41_both(df[0, ], function(t) mutate(t, y = tn %in% sset))
  f41_both(df, function(t) mutate(t, y = d %in% as.Date(character())))
})

test_that("mixed-type and fractional %in% sets still match dplyr", {
  skip_if_no_gpu()
  df <- f41_df()
  dset <- f41_dset
  sset <- f41_sset
  frac <- structure(c(19723.5, 19725), class = "Date")
  f41_both(df, function(t) filter(t, d %in% sset))
  f41_both(df, function(t) filter(t, t %in% dset))
  f41_both(df, function(t) filter(t, tn %in% dset))
  f41_both(df, function(t) mutate(t, y = d %in% frac))
  f41_both(df, function(t) filter(t, d %in% c(19723, 19792)))
})

test_that("replace_values() over timestamps is native and keeps x's type and zone", {
  skip_if_no_gpu()
  df <- f41_df()
  f41_native(df, function(t) {
    mutate(t, y = dplyr::replace_values(d, as.Date("2024-01-01") ~ as.Date("2030-06-01")))
  })
  f41_native(df, function(t) {
    mutate(t, y = dplyr::replace_values(d, as.Date(c("2024-01-03", "2024-12-31")) ~ as.Date("2030-06-01")))
  })
  f41_native(df, function(t) {
    mutate(t, y = dplyr::replace_values(tn, as.POSIXct("2024-01-01 05:00:00", tz = "UTC") ~
                                   as.POSIXct("2030-01-01", tz = "UTC")))
  })
  f41_native(df, function(t) {
    mutate(t, y = dplyr::replace_values(t, as.POSIXct(c("2024-01-01 12:00:01", "2024-03-10 03:30:00"), tz = "UTC") ~
                                   as.POSIXct("2030-01-01 08:00:00", tz = "America/New_York")))
  })
})

# --- item 2: as.Date()/as.POSIXct() inside predicates ----------------------------

test_that("as.Date() of a Date or POSIXct inside a comparison is native", {
  skip_if_no_gpu()
  df <- f41_df()
  f41_native(df, function(t) filter(t, as.Date(d) > as.Date("2024-01-02")))
  f41_native(df, function(t) filter(t, as.Date(d) >= d))
  f41_native(df, function(t) mutate(t, y = as.Date(d) > as.Date("2024-01-02")))
  f41_native(df, function(t) filter(t, as.Date(t) == d))
  f41_native(df, function(t) filter(t, as.Date(t) == as.Date("2024-01-01")))
  f41_native(df, function(t) filter(t, as.Date(tn) == as.Date("2024-01-01")))
  f41_native(df, function(t) filter(t, as.Date(tn) < as.Date("1970-01-01")))
  f41_native(df, function(t) mutate(t, y = as.Date(tn) == d, z = as.Date(t) != as.Date(d)))
  f41_native(df, function(t) t |> group_by(g) |> filter(as.Date(d) > as.Date("2024-01-02")))
})

test_that("as.POSIXct() of a Date inside a comparison is native", {
  skip_if_no_gpu()
  df <- f41_df()
  f41_native(df, function(t) filter(t, as.POSIXct(d) > as.POSIXct("2024-01-02", tz = "UTC")))
  f41_native(df, function(t) filter(t, as.POSIXct(d) <= t))
  f41_native(df, function(t) mutate(t, y = as.POSIXct(d) == t))
  f41_both(df, function(t) filter(t, as.POSIXct(d) <= tn))
})

test_that("date class remains on filter (dplyr #273)", {
  skip_if_no_gpu()
  x1 <- data.frame(
    date = seq.Date(as.Date("2013-01-01"), by = "1 days", length.out = 2),
    var = c(5, 8)
  )
  x2 <- x1
  x2$date <- x2$date + 1
  f41_native(x1, function(t) filter(t, as.Date(date) > as.Date("2013-01-01")))
  f41_native(x2, function(t) filter(t, as.Date(date) > as.Date("2013-01-01")))
  res <- collect(filter(tbl_gpu(x2), as.Date(date) > as.Date("2013-01-01")))
  expect_equal(class(res$date), "Date")
})

# --- item 3: mean()/median() of Date/POSIXct --------------------------------------

test_that("mean()/median() of a POSIXct in summarise() is native and keeps the zone", {
  skip_if_no_gpu()
  df <- f41_df()
  f41_native(df, function(t) summarise(t, m = mean(t, na.rm = TRUE), md = median(tn, na.rm = TRUE)))
  f41_native(df, function(t) summarise(t, m = mean(tn), md = median(t)))
  f41_native(df, function(t) {
    t |> group_by(g) |> summarise(m = mean(tn), md = median(t), mr = mean(t, na.rm = TRUE))
  })
  f41_native(df, function(t) summarise(t, m = mean(tn, na.rm = TRUE), md = median(tn, na.rm = TRUE), .by = g))
  f41_native(df, function(t) {
    t |> group_by(g) |> summarise(m = mean(tn, na.rm = TRUE)) |>
      filter(m > as.POSIXct("2024-02-01", tz = "UTC"))
  })
})

test_that("mean()/median() of a tz-less POSIXct keeps an empty tzone", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 2L),
    t = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-06-01 10:00:00"))
  )
  f41_native(df, function(t) t |> group_by(g) |> summarise(m = mean(t), md = median(t)))
  f41_native(df, function(t) t |> group_by(g) |> mutate(m = mean(t)))
})

test_that("mean()/median() of a Date in summarise() is native", {
  skip_if_no_gpu()
  df <- f41_df()
  expect_no_warning(f41_native(df, function(t) {
    t |> group_by(g) |> summarise(m = mean(d), md = median(d), mr = mean(d, na.rm = TRUE))
  }))
  expect_no_warning(f41_native(df, function(t) summarise(t, md = median(d, na.rm = TRUE))))
  expect_no_warning(f41_native(df, function(t) summarise(t, m = median(d, na.rm = TRUE), .by = g)))
  # The mean is a Date in the schema too: a whole-day shift of it stays native.
  f41_native(df, function(t) {
    t |> group_by(g) |> summarise(m = mean(d, na.rm = TRUE)) |> mutate(y = m + 1L)
  })
})

test_that("a fractional Date mean/median is floored natively with the warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f41_df()
  # mean of the 5 non-NA days is 15865.2; dplyr keeps the fraction.
  want <- summarise(df, m = mean(d, na.rm = TRUE))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_warning(
      got <- collect(summarise(g, m = mean(d, na.rm = TRUE))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_s3_class(got$m, "Date")
    expect_equal(as.numeric(got$m), floor(as.numeric(want$m)))
  }

  df2 <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-02", "2023-12-30", "1969-12-31", "1969-12-31"))
  )
  want2 <- df2 |> group_by(g) |> summarise(md = median(d), m = mean(d))
  expect_warning(
    got2 <- collect(df2 |> tbl_gpu() |> group_by(g) |> summarise(md = median(d), m = mean(d))),
    class = "cuplyr_warning_fractional_date"
  )
  expect_equal(got2$g, want2$g)
  expect_equal(as.numeric(got2$md), floor(as.numeric(want2$md)))
  expect_equal(as.numeric(got2$m), floor(as.numeric(want2$m)))

  want3 <- df2 |> group_by(g) |> mutate(m = mean(d))
  expect_warning(
    got3 <- collect(df2 |> tbl_gpu() |> group_by(g) |> mutate(m = mean(d))),
    class = "cuplyr_warning_fractional_date"
  )
  expect_equal(as.numeric(got3$m), floor(as.numeric(want3$m)))
})

test_that("window mean()/median() of Date/POSIXct in mutate() is native", {
  skip_if_no_gpu()
  df <- f41_df()
  f41_native(df, function(t) t |> group_by(g) |> mutate(m = mean(tn), md = median(t)))
  f41_native(df, function(t) mutate(t, m = median(tn, na.rm = TRUE), .by = g))
  f41_native(df, function(t) mutate(t, m = mean(t, na.rm = TRUE), md = median(tn)))
  expect_no_warning(f41_native(df, function(t) t |> group_by(g) |> mutate(m = mean(d), md = median(d, na.rm = TRUE))))
  expect_no_warning(f41_native(df, function(t) mutate(t, md = median(d, na.rm = TRUE))))
})

test_that("mean()/median() of timestamps on empty input match dplyr", {
  skip_if_no_gpu()
  df <- f41_df()[0, ]
  f41_both(df, function(t) t |> group_by(g) |> summarise(m = mean(tn), md = median(d)))
  f41_both(df, function(t) t |> group_by(g) |> mutate(m = mean(tn), md = median(d)))

  # Ungrouped: one row. dplyr's mean is NaN with the class (NA in a GPU
  # timestamp column), its median NA.
  want <- summarise(df, m = mean(tn), md = median(t), dm = mean(d), dmd = median(d))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    got <- collect(summarise(g, m = mean(tn), md = median(t), dm = mean(d), dmd = median(d)))
    expect_equal(nrow(got), 1L)
    for (col in names(want)) {
      expect_equal(class(got[[col]]), class(want[[col]]), info = col)
      expect_equal(attr(got[[col]], "tzone"), attr(want[[col]], "tzone"), info = col)
      expect_true(is.na(got[[col]]), info = col)
    }
  }
})

test_that("timestamp aggregates inside larger expressions still match dplyr", {
  skip_if_no_gpu()
  df <- f41_df()
  f41_both(df, function(t) filter(t, t >= mean(t, na.rm = TRUE)))
  f41_both(df, function(t) t |> group_by(g) |> filter(d <= median(d, na.rm = TRUE)))
  f41_both(df, function(t) summarise(t, n = sum(d %in% f41_dset), m = max(tn, na.rm = TRUE)))
})

# Round-1 review B1 (a regression F41 introduced): a later dot must compare
# against dplyr's real (fractional) mean, not the floored stored column. Only
# `m` itself is floored (with the warning).
test_that("a later dot compares against dplyr's unfloored Date mean/median", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-01-02")))
  df2 <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-02", "2024-03-01", "2024-03-02", "2024-03-02"))
  )
  check <- function(data, pipeline) {
    want <- tibble::as_tibble(dplyr::ungroup(pipeline(data)))
    rest <- setdiff(names(want), "m")
    for (lazy in c(FALSE, TRUE)) {
      g <- tbl_gpu(data, lazy = lazy)
      expect_warning(got <- collect(pipeline(g)), class = "cuplyr_warning_fractional_date")
      got <- tibble::as_tibble(dplyr::ungroup(got))
      expect_equal(names(got), names(want))
      expect_equal(as.numeric(got$m), floor(as.numeric(want$m)))
      expect_equal(got[rest], want[rest])
    }
  }
  check(df, function(t) mutate(t, m = mean(d), early = d < m, on_mean = d == m))
  check(df2, function(t) mutate(t, m = mean(d), early = d < m, .by = g))
  check(df2, function(t) t |> group_by(g) |> mutate(m = median(d), late = d >= m))
  check(df, function(t) transmute(t, d, m = mean(d), early = d < m))
})
