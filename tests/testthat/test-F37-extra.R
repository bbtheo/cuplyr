# F37 round 2: extra coverage next to the acceptance file
# (test-parity-F37-date-arith.R) for the review's B1-B3 shapes. The data are
# built so every mean/median is a whole number of days: a GPU Date holds whole
# days, so a fractional average is floored at upload (cuplyr_warning_fractional_date).

f37x_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-03", "2024-02-01", "2024-02-05")),
    n = c(1L, 1L, 2L, 2L)
  )
}

f37x_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("mean/median/sd/var of a Date go to the CPU, bare or shifted", {
  skip_if_no_gpu()
  df <- f37x_df()
  f37x_both(df, function(t) summarise(t, m = mean(d)))
  f37x_both(df, function(t) summarise(t, m = mean(d + 1L), s = sd(d + 1L), v = var(d - n)))
  f37x_both(df, function(t) t |> group_by(g) |> summarise(m = median(d + n)))
  f37x_both(df, function(t) t |> group_by(g) |> mutate(m = mean(d + 1L), s = sd(d)))
  f37x_both(df, function(t) filter(t, d + 1L > mean(d)))
  f37x_both(df, function(t) summarise(t, lo = min(d + 1L), hi = max(d - n)))
})

test_that("mean of a POSIXct goes to the CPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    t = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-01 02:00:00"), tz = "UTC")
  )
  f37x_both(df, function(t) summarise(t, m = mean(t)))
})

test_that("a shifted Date against a number or string, every comparison operator", {
  skip_if_no_gpu()
  df <- f37x_df()
  for (op in c("==", "!=", "<", "<=", ">", ">=")) {
    f37x_both(df, function(t) rlang::inject(filter(t, !!rlang::call2(op, quote(d + n), 19725))))
    f37x_both(df, function(t) rlang::inject(filter(t, !!rlang::call2(op, 19725L, quote(d - 1L)))))
    f37x_both(df, function(t) rlang::inject(mutate(t, y = !!rlang::call2(op, quote(d + 1L), "2024-01-04"))))
  }
  # a shifted Date against a Date stays native
  withr::local_options(cuplyr.fallback = "error")
  f37x_both(df, function(t) filter(t, d + 1L > as.Date("2024-01-03")))
})

test_that("whole Dates upload without a warning; fractional columns warn once, by name", {
  skip_if_no_gpu()
  whole <- tibble::tibble(a = as.Date("2024-01-01") + 0:2, b = as.Date(c(NA, 5, -3)))
  expect_no_warning(g <- tbl_gpu(whole))
  expect_equal(tibble::as_tibble(collect(g)), whole)

  frac <- tibble::tibble(
    a = as.Date(c(1.5, 2)), b = as.Date(c(3, 4)), c = as.Date(c(NA, -0.25)), x = c(1.5, 2)
  )
  warns <- list()
  g <- withCallingHandlers(
    tbl_gpu(frac),
    warning = function(w) {
      warns[[length(warns) + 1L]] <<- w
      invokeRestart("muffleWarning")
    }
  )
  expect_length(warns, 1L)
  expect_s3_class(warns[[1L]], "cuplyr_warning_fractional_date")
  expect_match(conditionMessage(warns[[1L]]), "`a`, `c`", fixed = TRUE)
  expect_false(grepl("`b`", conditionMessage(warns[[1L]]), fixed = TRUE))
  res <- collect(g)
  expect_equal(res$a, as.Date(c(1, 2)))
  expect_equal(res$c, as.Date(c(NA, -1)))
  expect_equal(res$x, frac$x)
})

test_that("assigning a fractional Date column with $<- floors it with the warning", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(k = 1:2))
  expect_warning(g$d <- as.Date(c(10.5, 11)), class = "cuplyr_warning_fractional_date")
  expect_equal(collect(g)$d, as.Date(c(10, 11)))
})

# --- round 3: the timestamp rule of the type inference (ir_check_ts_operands()) ---

f37x_df3 <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-03", "2024-02-01", "2024-02-05", NA)),
    d2 = as.Date(c("2024-01-05", "2024-01-03", "2024-03-01", "2024-01-05", "2024-01-09")),
    n = c(1L, 1L, 2L, 2L, 2L),
    b = c(TRUE, FALSE, TRUE, NA, FALSE)
  )
}

# dplyr errors with `class`; so does cuplyr, eager and lazy.
f37x_same_error <- function(df, pipeline, class) {
  expect_error(pipeline(df), class = class)
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(pipeline(tbl_gpu(df, lazy = lazy))), class = class)
  }
}

test_that("a Date column, bare or shifted, against a number or string matches dplyr", {
  skip_if_no_gpu()
  df <- f37x_df3()
  f37x_both(df, function(t) filter(t, d < 5))
  f37x_both(df, function(t) filter(t, d == "2024-01-03"))
  f37x_both(df, function(t) mutate(t, y = d >= 19724.5, z = d2 < 19727L))
  # the shifted Date reaches the comparison as a plain column of the same call
  f37x_both(df, function(t) mutate(t, y = d + 1L, z = y >= 19724.5))
  f37x_both(df, function(t) mutate(t, y = d + 1L, z = y > "2024-01-03"))
})

test_that("%in% with a Date on either side matches dplyr, bare or shifted", {
  skip_if_no_gpu()
  df <- f37x_df3()
  hol <- as.Date(c("2024-01-02", "2024-02-06", "2024-01-03"))
  f37x_both(df, function(t) filter(t, d %in% hol))
  f37x_both(df, function(t) mutate(t, y = d %in% "2024-01-03", z = n %in% hol))
  f37x_both(df, function(t) mutate(t, y = dplyr::case_match(d + 1L, hol ~ 1L, .default = 0L)))
})

test_that("between() of a shifted Date matches dplyr (dplyr::between)", {
  skip_if_no_gpu()
  df <- f37x_df3()
  f37x_both(df, function(t) {
    filter(t, dplyr::between(d + 1L, as.Date("2024-01-02"), as.Date("2024-02-02")))
  })
  f37x_same_error(df, function(t) filter(t, dplyr::between(d + 1L, 19724.5, 19730)),
                  "vctrs_error_incompatible_type")
  f37x_same_error(df, function(t) mutate(t, y = dplyr::between(d, 19724, 19730)),
                  "vctrs_error_incompatible_type")
  f37x_same_error(df, function(t) mutate(t, y = d + 1L, z = dplyr::between(y, 19724, 19730)),
                  "vctrs_error_incompatible_type")
})

test_that("a Date mixed with a number in if_else()/coalesce()/na_if() errors like dplyr", {
  skip_if_no_gpu()
  df <- f37x_df3()
  f37x_same_error(df, function(t) mutate(t, y = dplyr::if_else(b, d + 1L, 1)),
                  "vctrs_error_incompatible_type")
  f37x_same_error(df, function(t) mutate(t, y = dplyr::coalesce(d, 0)),
                  "vctrs_error_incompatible_type")
  f37x_same_error(df, function(t) mutate(t, y = dplyr::na_if(d + 1L, 19725)),
                  "vctrs_error_incompatible_type")
})

test_that("ops R doesn't define (or defines differently) on a Date match dplyr", {
  skip_if_no_gpu()
  df <- f37x_df3()
  for (pipeline in list(
    function(t) mutate(t, y = cummax(d + 1L)),
    function(t) mutate(t, y = cummin(d)),
    function(t) mutate(t, y = abs(d + 1L)),
    function(t) mutate(t, y = (d + 1L) / 2),
    function(t) mutate(t, y = b & d),
    function(t) mutate(t, y = sum(d + 1L)),
    function(t) summarise(t, y = sum(d + 1L)),
    function(t) summarise(t, y = quantile(d, 0.5))
  )) {
    expect_error(pipeline(df))
    for (lazy in c(FALSE, TRUE)) {
      expect_error(collect(pipeline(tbl_gpu(df, lazy = lazy))), class = "rlang_error")
    }
  }
  f37x_both(df, function(t) mutate(t, y = round(d + 1L), z = dplyr::cumall(d)))
})

test_that("ops that take a Date correctly stay native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f37x_df3()
  f37x_both(df, function(t) {
    mutate(t,
      a = d + 1L == d2, c = dplyr::if_else(b, d + 1L, d), e = dplyr::coalesce(d + 1L, d2),
      f = dplyr::lag(d + 1L), h = dplyr::min_rank(d + n), i = dplyr::n_distinct(d + 1L),
      j = is.na(d - 1L), k = d > NaN, l = d == NA_real_
    )
  })
  f37x_both(df, function(t) filter(t, dplyr::between(d + 1L, d2 - 3L, d2)))
  f37x_both(df, function(t) group_by(t, g) |> summarise(lo = min(d + n), k = dplyr::n_distinct(d - 1L)))
})

# --- round 4: fractional Date literals, unary signs, nested timestamps ---------
# A Date literal is an int32 count of days on the GPU, but an R Date can hold
# a fraction (seq(..., length.out = k) breakpoints, midpoints, mean()). Any
# fractional Date literal sends its expression to the CPU (ir_check_ts_lit()).

f37x_df4 <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L),
    d = as.Date(c("2024-01-01", "2024-01-02", "2024-01-03", NA)),
    d2 = as.Date(c("2024-01-02", NA, "2024-01-01", "2024-01-05")),
    n = c(1L, 2L, 3L, NA),
    b = c(TRUE, FALSE, NA, TRUE)
  )
}

# 19724.5: prints as 2024-01-02, compares as a half day
f37x_mid <- function() seq(as.Date("2024-01-01"), as.Date("2024-01-04"), length.out = 3)[2]

test_that("a Date column, bare or shifted, compared with a fractional Date matches dplyr", {
  skip_if_no_gpu()
  df <- f37x_df4()
  mid <- f37x_mid()
  neg <- as.Date(-0.5)
  hi <- as.Date("2024-01-03")
  for (op in c("==", "!=", "<", "<=", ">", ">=")) {
    f37x_both(df, function(t) rlang::inject(filter(t, !!rlang::call2(op, quote(d), mid))))
    f37x_both(df, function(t) rlang::inject(filter(t, !!rlang::call2(op, mid, quote(d + 1L)))))
    f37x_both(df, function(t) rlang::inject(mutate(t, y = !!rlang::call2(op, quote(d - n), mid))))
  }
  f37x_both(df, function(t) filter(t, d > neg))
  f37x_both(df, function(t) filter(t, d >= mid, n > 1L))
  f37x_both(df, function(t) filter(t, d >= mid | n > 2L))
  f37x_both(df, function(t) filter(t, d >= .env$mid))
  f37x_both(df, function(t) filter(t, mid + n > d))
  f37x_both(df, function(t) filter(t, dplyr::between(d, mid, hi)))
  f37x_both(df, function(t) filter(t, dplyr::between(d + 1L, as.Date("2024-01-01"), mid)))
  f37x_both(df, function(t) filter(t, d %in% c(mid, hi)))
  f37x_both(df, function(t) mutate(t, dplyr::across(c(d, d2), ~ .x > mid)))
  f37x_both(df, function(t) summarise(t, k = sum(d > mid, na.rm = TRUE)))
  f37x_both(df, function(t) group_by(t, g) |> summarise(k = sum(d + 1L > mid, na.rm = TRUE)))
  f37x_both(df, function(t) mutate(t, z = dplyr::na_if(d, mid)))
  dn <- tibble::tibble(n = c(1L, 40L, 70L, NA))
  f37x_both(dn, function(t) mutate(t, y = as.Date("2024-01-01") + n > as.Date(19763.5)))
})

test_that("a fractional Date value in a result is floored with the warning, eager and lazy", {
  skip_if_no_gpu()
  df <- f37x_df4()
  mid <- f37x_mid()
  neg <- as.Date(-0.5)
  floor_dates <- function(x) {
    x[] <- lapply(x, function(v) if (inherits(v, "Date")) as.Date(floor(unclass(v))) else v)
    x
  }
  for (pipeline in list(
    function(t) mutate(t, z = dplyr::if_else(b, d, mid)),
    function(t) mutate(t, z = dplyr::if_else(b, mid, d + 1L, missing = neg)),
    function(t) mutate(t, z = dplyr::case_when(b ~ mid, .default = d)),
    function(t) mutate(t, z = dplyr::coalesce(d + 1L, mid)),
    function(t) mutate(t, z = dplyr::replace_when(d, b ~ mid)),
    function(t) mutate(t, z = dplyr::lag(d, default = mid)),
    function(t) group_by(t, g) |> mutate(z = dplyr::lead(d + 1L, default = mid)),
    function(t) mutate(t, z = mid + n, w = n + mid, v = mid - n),
    function(t) mutate(t, y = neg),
    function(t) mutate(t, y = mid, z = y > d),
    function(t) transmute(t, y = c(mid, neg, mid, neg)),
    function(t) group_by(t, g) |> summarise(y = mid, m = max(d, na.rm = TRUE))
  )) {
    expected <- floor_dates(pipeline(df))
    for (lazy in c(FALSE, TRUE)) {
      expect_warning(
        got <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(tibble::as_tibble(got), tibble::as_tibble(expected), ignore_attr = TRUE)
    }
  }
})

test_that("whole-day Date literals stay native in every slot", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f37x_df4()
  lo <- as.Date("2024-01-01")
  f37x_both(df, function(t) filter(t, d >= as.Date("2024-01-02")))
  f37x_both(df, function(t) filter(t, d < Sys.Date() - 30))
  f37x_both(df, function(t) filter(t, d + 1L >= lo, dplyr::between(d, lo, lo + 2)))
  f37x_both(df, function(t) {
    mutate(t,
      a = dplyr::if_else(b, d, lo), c = dplyr::coalesce(d, as.Date(NA)),
      e = dplyr::lag(d, default = lo), f = lo + n, h = dplyr::na_if(d, lo), k = lo
    )
  })
  f37x_both(df, function(t) filter(t, d == as.Date(NA)))
})

# Base R: unary minus on a Date errors; unary plus returns the Date (`+.Date`).
test_that("unary minus on a Date errors like dplyr and unary plus matches it, bare, shifted or nested", {
  skip_if_no_gpu()
  df <- f37x_df4()
  dn <- tibble::tibble(n = c(1L, 40L, 70L))
  minus <- list(
    list(df, function(t) mutate(t, y = -d)),
    list(df, function(t) mutate(t, y = -(d + 1L))),
    list(df, function(t) mutate(t, y = -(d2 - n))),
    list(dn, function(t) mutate(t, y = -(-(as.Date("2024-01-01") + n)))),
    list(dn, function(t) mutate(t, y = +(-(as.Date("2024-01-01") + n)))),
    list(dn, function(t) filter(t, -(as.Date("2024-01-01") + n) > as.Date("1900-01-01")))
  )
  for (cs in minus) {
    expect_error(cs[[2]](cs[[1]]), "unary - is not defined")
    for (lazy in c(FALSE, TRUE)) {
      expect_error(collect(cs[[2]](tbl_gpu(cs[[1]], lazy = lazy))), "unary - is not defined")
    }
  }
  f37x_both(df, function(t) mutate(t, y = +d, z = +(d - n)))
  f37x_both(dn, function(t) mutate(t, y = +(as.Date("2024-01-01") + n)))
  f37x_both(dn, function(t) filter(t, +(as.Date("2024-01-01") + n) > as.Date("2024-02-01")))
})

test_that("a Date nested below + or - (no Date column) matches dplyr", {
  skip_if_no_gpu()
  dn <- tibble::tibble(n = c(1L, 40L, 70L, NA))
  f37x_both(dn, function(t) mutate(t, y = (as.Date("2024-01-01") + n) + NA))
  f37x_both(dn, function(t) mutate(t, y = NA + (as.Date("2024-01-01") + n)))
  f37x_both(dn, function(t) mutate(t, y = (as.Date("2024-01-01") + n) - NA))
  f37x_both(dn, function(t) mutate(t, y = (as.Date("2024-01-01") + n) + 1L, z = y + n))
  f37x_both(dn, function(t) {
    mutate(t, y = as.numeric((as.Date("2024-01-01") + n) - (as.Date("2024-01-01") + 1L)))
  })
  f37x_both(dn, function(t) filter(t, (as.Date("2024-01-01") + n) > 19800))
  expect_error(mutate(dn, y = 1L - (as.Date("2024-01-01") + n)))
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(mutate(tbl_gpu(dn, lazy = lazy), y = 1L - (as.Date("2024-01-01") + n))))
  }
})
