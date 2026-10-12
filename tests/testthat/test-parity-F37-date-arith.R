# F37 acceptance tests: Date arithmetic in mutate()/filter()/summarise().
#
# Base R semantics (dplyr adds nothing; verified with tools/r, see
# scratchpad/parity/features/F37/probe1.R, probe2.R):
#   - Date +/- number and number + Date give a Date (double storage, a
#     fractional number is kept: as.Date("2024-01-01") + 0.5 is 19723.5).
#   - number - Date, Date + Date, Date * number, unary -Date are errors.
#   - Date - Date (and POSIXct - POSIXct) give a difftime.
#
# cuplyr contract (F37 design):
#   - Date +/- a whole number of days (an integer column or a whole numeric
#     literal) runs natively on the GPU, with no CPU fallback.
#   - Other Date shifts go through the CPU route. A TIMESTAMP_DAYS column
#     can't hold fractional days, so a fractional Date is floored at upload
#     (the calendar day R prints) with a `cuplyr_warning_fractional_date`.
#   - A difftime result can't be uploaded (F18): Date - Date gives dplyr's
#     answer when the difftime doesn't survive into the result, and
#     otherwise F18's `cuplyr_error_unsupported_class` error naming the
#     difftime and suggesting as.numeric().

f37_df <- function() {
  tibble::tibble(
    g  = c(1L, 1L, 2L, 2L, 2L),
    d  = as.Date(c("2024-01-01", NA, "2024-02-29", "2023-12-31", "1969-12-30")),
    d2 = as.Date(c("2024-01-10", "2024-01-05", NA, "2024-01-01", "1970-01-02")),
    n  = c(1L, 2L, NA, -3L, 40L),
    x  = c(1, 2, 3, NA, -2)
  )
}

f37_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

# Same comparison, but any CPU fallback is an error.
f37_native <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

f37_pct <- function() {
  tibble::tibble(
    t1 = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-02 12:00:00", NA), tz = "UTC"),
    t2 = as.POSIXct(c("2024-01-01 01:00:00", "2024-01-01 00:00:00", "2024-01-01 00:00:00"), tz = "UTC")
  )
}

# --- native whole-day shifts --------------------------------------------------

test_that("Date +/- whole-number literal is native and matches dplyr", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_native(df, function(t) mutate(t, y = d + 1L))
  f37_native(df, function(t) mutate(t, y = d + 7))
  f37_native(df, function(t) mutate(t, y = d - 30))
  f37_native(df, function(t) mutate(t, y = d - 1L))
  f37_native(df, function(t) mutate(t, y = 3L + d))
  f37_native(df, function(t) mutate(t, y = 365 + d))
  f37_native(df, function(t) transmute(t, y = d + 100))
})

test_that("Date +/- integer column is native, NAs propagate", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_native(df, function(t) mutate(t, y = d + n))
  f37_native(df, function(t) mutate(t, y = n + d))
  f37_native(df, function(t) mutate(t, y = d - n))
  f37_native(df, function(t) mutate(t, y = d2 - n + 2L))
})

test_that("native Date shifts work grouped, chained lazily and on 0 rows", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_native(df, function(t) t |> group_by(g) |> mutate(y = d + 1L))
  f37_native(df, function(t) {
    t |>
      mutate(y = d + 1L) |>
      filter(y > as.Date("2024-01-01")) |>
      mutate(z = y - 1, w = z == d)
  })
  f37_native(df[0, ], function(t) mutate(t, y = d + 1L, z = d - n))
})

test_that("comparisons against shifted Dates are native in filter() and mutate()", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_native(df, function(t) filter(t, d > d2 - 7))
  f37_native(df, function(t) filter(t, d + 1L <= d2))
  f37_native(df, function(t) filter(t, d + n >= as.Date("2024-01-01")))
  f37_native(df, function(t) mutate(t, y = d > d2 + 7))
})

# --- other shapes through the CPU route (regressions included) ---------------

test_that("Date comparisons, casts and aggregates of shifted Dates match dplyr", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_both(df, function(t) filter(t, d == as.Date("2024-01-01")))
  f37_both(df, function(t) mutate(t, y = as.integer(d), z = as.numeric(d)))
  f37_both(df, function(t) mutate(t, y = as.integer(d + 1L)))
  f37_both(df, function(t) {
    t |> group_by(g) |> summarise(y = max(d + 1L, na.rm = TRUE), z = min(d - 1, na.rm = TRUE))
  })
})

# Round-1 review B1/B2: before F37 these ran on the CPU (the shift sent the
# whole dot there); making the shift native must not expose GPU paths that
# can't take a Date (mean/median/sd aggregates, Date vs number/string).
test_that("mean/median/sd of a shifted Date match dplyr", {
  skip_if_no_gpu()
  # Days chosen so every mean/median below is a whole day: a fractional
  # Date result is floored at upload (tested separately), which would
  # differ from dplyr's fractional value.
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L),
    d = as.Date("2024-01-01") + c(0, 2, NA, 9, 11, 13),
    d2 = as.Date("2024-01-01") + c(1, NA, 3, 10, 20, 30),
    n = c(1L, 3L, 5L, 2L, NA, 4L)
  )
  f37_both(df, function(t) summarise(t, m = mean(d + 1L, na.rm = TRUE)))
  f37_both(df, function(t) {
    t |> group_by(g) |> summarise(m = median(d + n, na.rm = TRUE), s = sd(d + 1L, na.rm = TRUE))
  })
  f37_both(df, function(t) t |> group_by(g) |> mutate(m = mean(d + 1L, na.rm = TRUE)))
  f37_both(df, function(t) filter(t, d + 1L > mean(d2 - 1L, na.rm = TRUE)))
})

test_that("a shifted Date compared with a number or a date string matches dplyr", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_both(df, function(t) filter(t, d + 1L >= 19724.5))
  f37_both(df, function(t) mutate(t, y = d + 1L >= 19724.5))
  f37_both(df, function(t) filter(t, 19724 < d + 1L))
  f37_both(df, function(t) filter(t, d + 1L == "2024-01-02"))
  f37_both(df, function(t) filter(t, d - 7 >= "2023-12-26"))
})

test_that("Date + double column with whole values matches dplyr without a warning", {
  skip_if_no_gpu()
  df <- f37_df()
  expect_no_warning(f37_both(df, function(t) mutate(t, y = d + x)))
  expect_no_warning(f37_both(df, function(t) mutate(t, y = x + d, z = d - x)))
})

test_that("Date + NA gives an NA Date like dplyr", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_both(df, function(t) mutate(t, y = d + NA))
})

# --- fractional days -----------------------------------------------------------

test_that("fractional Date results are floored with a classed warning", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    d = as.Date(c("2024-01-01", "1969-12-31", NA, "2024-03-01")),
    xf = c(0.5, 0.5, 1.5, 2)
  )
  floor_date <- function(v) as.Date(floor(unclass(v)))

  for (lazy in c(FALSE, TRUE)) {
    expected <- dplyr::mutate(df, y = d + xf)
    expected$y <- floor_date(expected$y)
    expect_warning(
      res <- tbl_gpu(df, lazy = lazy) |> mutate(y = d + xf) |> collect(),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
    expect_equal(format(res$y), format(dplyr::mutate(df, y = d + xf)$y))

    expected <- dplyr::mutate(df, y = d + 0.5)
    expected$y <- floor_date(expected$y)
    expect_warning(
      res <- tbl_gpu(df, lazy = lazy) |> mutate(y = d + 0.5) |> collect(),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("uploading a fractional Date floors it with a classed warning", {
  skip_if_no_gpu()
  frac <- tibble::tibble(d = as.Date(c(19723.5, -0.5, NA, -1.25)))
  expect_warning(g <- tbl_gpu(frac), class = "cuplyr_warning_fractional_date")
  expect_equal(collect(g)$d, as.Date(c(19723, -1, NA, -2)))

  whole <- tibble::tibble(d = as.Date(c(19723, -1, NA)))
  expect_no_warning(g2 <- tbl_gpu(whole))
  expect_equal(collect(g2)$d, whole$d)
})

# --- Date - Date (difftime) ----------------------------------------------------

f37_expect_difftime_error <- function(expr) {
  err <- expect_error(expr, class = "cuplyr_error_unsupported_class")
  msg <- rlang::cnd_message(err)
  expect_match(msg, "difftime", fixed = TRUE)
  expect_match(msg, "as.numeric", fixed = TRUE)
}

test_that("a difftime result column gives F18's classed error with an as.numeric() hint", {
  skip_if_no_gpu()
  df <- f37_df()
  for (lazy in c(FALSE, TRUE)) {
    gdf <- function() tbl_gpu(df, lazy = lazy)
    f37_expect_difftime_error(gdf() |> mutate(y = d2 - d) |> collect())
    f37_expect_difftime_error(gdf() |> group_by(g) |> mutate(y = d2 - d) |> collect())
    f37_expect_difftime_error(gdf() |> transmute(y = d2 - d) |> collect())
    f37_expect_difftime_error(gdf() |> mutate(y = as.Date("2024-03-01") - d) |> collect())
    f37_expect_difftime_error(
      gdf() |> summarise(m = mean(d2 - d, na.rm = TRUE)) |> collect()
    )
    f37_expect_difftime_error(
      tbl_gpu(f37_pct(), lazy = lazy) |> mutate(y = t2 - t1) |> collect()
    )
  }
})

test_that("Date - Date matches dplyr when no difftime reaches the result", {
  skip_if_no_gpu()
  df <- f37_df()
  f37_both(df, function(t) mutate(t, y = as.numeric(d2 - d)))
  f37_both(df, function(t) mutate(t, y = d2 - d, y = as.numeric(y)))
  f37_both(df, function(t) mutate(t, big = d2 - d > 3))
  f37_both(df, function(t) filter(t, d2 - d > 3))
  f37_both(df, function(t) group_by(t, g) |> mutate(y = as.numeric(d2 - d)))
  f37_both(f37_pct(), function(t) mutate(t, y = as.numeric(t2 - t1, units = "mins")))
})

# --- invalid Date arithmetic errors like base R --------------------------------

test_that("invalid Date arithmetic errors like dplyr", {
  skip_if_no_gpu()
  df <- f37_df()
  expect_error(dplyr::mutate(df, y = -d))
  for (lazy in c(FALSE, TRUE)) {
    gdf <- function() tbl_gpu(df, lazy = lazy)
    expect_error(gdf() |> mutate(y = -d) |> collect())
    expect_error(gdf() |> mutate(y = n - d) |> collect())
    expect_error(gdf() |> mutate(y = d * 2) |> collect())
    expect_error(gdf() |> mutate(y = d + d2) |> collect())
  }
})

# --- consumers of a shifted Date that round 2 missed (review round 2, B4/B5) ---

test_that("%in% on a shifted Date matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-03", "2024-02-01", "2024-02-05", NA)),
    n = c(1L, 1L, 2L, 2L, 2L)
  )
  hol <- as.Date(c("2024-01-02", "2024-02-06"))
  f37_both(df, function(t) filter(t, (d + 1L) %in% hol))
  f37_both(df, function(t) mutate(t, y = (d - 1L) %in% hol))
  f37_both(df, function(t) mutate(t, z = d + 1L, w = z %in% hol))
  f37_both(df, function(t) group_by(t, g) |> filter((d + n) %in% hol))
})

test_that("a Date shifted inside an expression is still compared like dplyr", {
  skip_if_no_gpu()
  # No Date column in the schema: the Date comes from a literal one level down.
  dn <- tibble::tibble(n = c(1L, 40L, 70L, NA))
  f37_both(dn, function(t) filter(t, as.Date("2024-01-01") + n >= 19763.5))
  f37_both(dn, function(t) mutate(t, y = as.Date("2024-01-01") + n >= 19763.5))
  f37_both(dn, function(t) filter(t, as.Date("2024-01-01") + n > "2024-02-01"))

  # between() expands to two comparisons; they must get the same rules.
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-03", "2024-02-01", "2024-02-05", NA)),
    n = c(1L, 1L, 2L, 2L, 2L)
  )
  f37_both(df, function(t) {
    filter(t, dplyr::between(d + 1L, as.Date("2024-01-02"), as.Date("2024-02-02")))
  })
  expect_error(
    dplyr::filter(df, dplyr::between(d + 1L, 19724.5, 19730)),
    class = "vctrs_error_incompatible_type"
  )
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      tbl_gpu(df, lazy = lazy) |> filter(dplyr::between(d + 1L, 19724.5, 19730)) |> collect(),
      class = "vctrs_error_incompatible_type"
    )
    expect_error(
      tbl_gpu(df, lazy = lazy) |> mutate(y = dplyr::between(d + 1L, 19724, 19730)) |> collect(),
      class = "vctrs_error_incompatible_type"
    )
  }
})

# --- fractional Date values and unary minus (review round 3, B6/B7) -----------

test_that("a shifted Date compared with a fractional Date value matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    d = as.Date(c("2024-01-01", "2024-01-02", "2024-01-03", NA)),
    n = c(1L, 2L, 3L, NA)
  )
  # 19724.5: prints as 2024-01-02, but compares as a half day.
  mid <- seq(as.Date("2024-01-01"), as.Date("2024-01-04"), length.out = 3)[2]
  f37_both(df, function(t) filter(t, d + 1L >= mid))
  f37_both(df, function(t) mutate(t, y = d + 1L == mid, z = dplyr::na_if(d + 1L, mid)))
  f37_both(df, function(t) filter(t, as.Date("2024-01-01") + n > mid))
})

test_that("unary minus on a Date built from a literal errors like dplyr", {
  skip_if_no_gpu()
  dn <- tibble::tibble(n = c(1L, 40L, 70L))
  expect_error(
    dplyr::mutate(dn, y = -(as.Date("2024-01-01") + n)),
    "unary - is not defined"
  )
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      tbl_gpu(dn, lazy = lazy) |> mutate(y = -(as.Date("2024-01-01") + n)) |> collect(),
      "unary - is not defined"
    )
  }
})
