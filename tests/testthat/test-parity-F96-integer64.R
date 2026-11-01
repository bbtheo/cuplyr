# F96: bit64::integer64 is uploaded as an exact cudf INT64 column and collects
# back to integer64. Verbs on it either match dplyr-on-integer64 exactly or
# run dplyr on the CPU; nothing may come back silently wrong.
#
# testthat's expect_equal() compares integer64 vectors loosely (it accepts
# 2^53 + 1 vs 2^53, and an integer64 vs a double with the same value), so every
# comparison here also goes through f96_canon(): classes must be identical and
# integer64 columns are compared as exact decimal strings.

f96_canon <- function(x) {
  x <- tibble::as_tibble(dplyr::ungroup(x))
  list(
    names = names(x),
    classes = unname(vapply(x, function(col) paste(class(col), collapse = "/"), character(1))),
    values = unname(lapply(x, function(col) {
      if (inherits(col, "integer64")) as.character(col) else col
    }))
  )
}

# dplyr oracle, eager AND lazy, plus the exact integer64 check.
f96_oracle <- function(df, pipeline, arrange_by = NULL) {
  eager <- expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  expect_identical(f96_canon(eager$gpu), f96_canon(eager$oracle), label = "eager (exact)")
  lazy <- expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
  expect_identical(f96_canon(lazy$gpu), f96_canon(lazy$oracle), label = "lazy (exact)")
  invisible(eager)
}

# Second table for joins/binds: a tbl_gpu in the same mode when `t` is one.
f96_like <- function(t, y) {
  if (is_tbl_gpu(t)) tbl_gpu(y, lazy = identical(unclass(t)$exec_mode, "lazy")) else y
}

f96_i64 <- function(x) bit64::as.integer64(x)

f96_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 2L, 3L),
    x = f96_i64(c("0", "-1", "1", NA, "9007199254740993", "9007199254740992",
                  "-9007199254740993", "9223372036854775807", "-9223372036854775807")),
    y = f96_i64(c("0", "1", "1", "5", "9007199254740993", "9007199254740993",
                  NA, "9223372036854775807", "-1")),
    i = c(0L, -2L, 1L, 4L, NA, 7L, 3L, 2L, 1L),
    v = c(0.5, 1, 2, 3, 4, 5, 6, 7, NA)
  )
}

f96_schema_type <- function(t, col) {
  s <- unclass(t)$schema
  unname(s$types[match(col, s$names)])
}

# ---------------------------------------------------------------------------
# Upload / collect
# ---------------------------------------------------------------------------

test_that("the F91 repro: integer64 values survive tbl_gpu() |> collect()", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(x = f96_i64(c(1, NA, 3)))
  for (lazy in c(FALSE, TRUE)) {
    out <- tbl_gpu(df, lazy = lazy) |> collect()
    expect_s3_class(out$x, "integer64")
    expect_identical(as.character(out$x), c("1", NA, "3"))
    expect_identical(out, df)
  }
})

test_that("integer64 round-trips exactly as INT64, eager and lazy, with no warning", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  for (lazy in c(FALSE, TRUE)) {
    expect_no_warning(g <- tbl_gpu(df, lazy = lazy))
    expect_identical(f96_schema_type(g, "x"), "INT64")
    expect_identical(f96_schema_type(g, "y"), "INT64")
    expect_no_warning(out <- collect(g))
    expect_identical(out, df)
    expect_identical(as.character(out$x), as.character(df$x))
    # the int64 extremes (INT64_MIN is NA in bit64) and 2^53 + 1
    expect_identical(as.character(out$x[8:9]),
                     c("9223372036854775807", "-9223372036854775807"))
    expect_identical(as.character(out$x[5]), "9007199254740993")
    expect_true(is.na(out$x[4]))
  }
})

test_that("zero-row and all-NA integer64 columns round-trip", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  for (df in list(
    tibble::tibble(x = bit64::integer64(0)),
    tibble::tibble(x = f96_i64(c(NA, NA)), i = 1:2)
  )) {
    for (lazy in c(FALSE, TRUE)) {
      expect_no_warning(out <- collect(tbl_gpu(df, lazy = lazy)))
      expect_identical(out, df)
    }
  }
})

test_that("pull(), `$` and as.data.frame() give integer64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_identical(pull(g, x), df$x)
    expect_identical(g$x, df$x)
    expect_identical(as.data.frame(g)$y, df$y)
  }
})

test_that("print() and glimpse() show integer64 like a tibble does", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  g <- tbl_gpu(df, lazy = FALSE)
  expect_no_warning(out <- capture.output(print(g)))
  expect_true(any(grepl("<int64>", out, fixed = TRUE)))
  gpu_lines <- capture.output(glimpse(g))
  cpu_lines <- capture.output(glimpse(df))
  expect_identical(grep("^\\$ x ", gpu_lines, value = TRUE),
                   grep("^\\$ x ", cpu_lines, value = TRUE))
  # a lazy table with pending ops previews the same column
  gl <- tbl_gpu(df, lazy = TRUE) |> filter(g > 0L)
  expect_identical(grep("^\\$ x ", capture.output(glimpse(gl)), value = TRUE),
                   grep("^\\$ x ", cpu_lines, value = TRUE))
})

test_that("without bit64 an INT64 column collects to double, warning only past 2^53 and naming the column", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    small = f96_i64(c(1, NA, 3)),
    big = f96_i64(c("9007199254740993", "1", NA))
  )
  g <- tbl_gpu(df, lazy = FALSE)
  local_mocked_bindings(bit64_available = function() FALSE)
  expect_no_warning(out_small <- collect(select(g, small)))
  expect_identical(out_small$small, c(1, NA, 3))
  w <- expect_warning(out <- collect(g), class = "cuplyr_warning_int64_precision")
  expect_match(conditionMessage(w), "big", fixed = TRUE)
  expect_no_match(conditionMessage(w), "small", fixed = TRUE)
  expect_type(out$big, "double")
  expect_identical(out$big[2:3], c(1, NA))
  expect_equal(out$big[1], 9007199254740992)
})

# ---------------------------------------------------------------------------
# Row verbs: filter / arrange / distinct / group_by
# ---------------------------------------------------------------------------

test_that("filter() comparisons on integer64 match dplyr exactly", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  f96_oracle(df, function(t) filter(t, x > 0L))
  f96_oracle(df, function(t) filter(t, x <= -1))
  # a double literal next to 2^53 + 1: bit64 compares in integer64, so this is
  # FALSE for 9007199254740993 (a FLOAT64 comparison would say TRUE)
  f96_oracle(df, function(t) filter(t, x == 9007199254740992))
  f96_oracle(df, function(t) filter(t, x != 9007199254740992))
  # bit64 truncates a fractional double operand: 2 >= 2.5 is TRUE there
  f96_oracle(tibble::tibble(x = f96_i64(c(1, 2, 3, NA))), function(t) filter(t, x >= 2.5))
  f96_oracle(df, function(t) filter(t, x > i))
  f96_oracle(df, function(t) filter(t, x == y))
  f96_oracle(df, function(t) filter(t, x < y | is.na(y)))
  f96_oracle(df, function(t) filter(t, x >= v))
  f96_oracle(df, function(t) filter(t, is.na(x)))
  f96_oracle(df, function(t) filter(t, !is.na(x), g == 3L))
  f96_oracle(df, function(t) filter(group_by(t, g), x == max(x, na.rm = TRUE)))
})

test_that("integer64 values from the environment are exact in filter()/mutate()", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  k <- f96_i64("9007199254740993")
  f96_oracle(df, function(t) filter(t, x == k))
  f96_oracle(df, function(t) filter(t, x == !!k))
  f96_oracle(df, function(t) filter(t, i < bit64::as.integer64(2)))
  f96_oracle(df, function(t) filter(t, x %in% f96_i64(c("9007199254740993", "-1", NA))))
  f96_oracle(df, function(t) mutate(t, z = bit64::as.integer64(7)))
  f96_oracle(df, function(t) mutate(t, z = k))
  f96_oracle(df, function(t) mutate(t, z = i + k))
})

test_that("arrange() orders integer64 exactly (neighbours of 2^53 included)", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  f96_oracle(df, function(t) arrange(t, x))
  f96_oracle(df, function(t) arrange(t, desc(x)))
  f96_oracle(df, function(t) arrange(t, y, desc(x)))
  f96_oracle(df, function(t) arrange(group_by(t, g), x, .by_group = TRUE))
})

test_that("distinct(), count() and group_by() keep 2^53 and 2^53 + 1 apart", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  f96_oracle(df, function(t) distinct(t, y))
  f96_oracle(df, function(t) distinct(t, x, y))
  f96_oracle(df, function(t) count(t, y))
  f96_oracle(df, function(t) t |> group_by(y) |> summarise(n = n(), s = sum(i, na.rm = TRUE)))
  f96_oracle(df, function(t) t |> group_by(y, g) |> summarise(n = n(), .groups = "drop"))
  f96_oracle(df, function(t) t |> group_by(y) |> mutate(r = dplyr::row_number()))
  f96_oracle(df, function(t) select(t, where(is.numeric)))
})

# ---------------------------------------------------------------------------
# summarise / mutate on integer64 values
# ---------------------------------------------------------------------------

test_that("summarise() over integer64 matches dplyr-on-bit64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  # group 2 holds 1 and the int64 max: sum(na.rm = TRUE) overflows to NA (with
  # bit64's warning), group 3 does not
  suppressWarnings(f96_oracle(df, function(t) t |> group_by(g) |> summarise(
    n = n(), s = sum(x), s2 = sum(x, na.rm = TRUE),
    mn = min(x), mx = max(x), nd = n_distinct(x), nd2 = n_distinct(x, na.rm = TRUE)
  )))
  f96_oracle(df, function(t) t |> group_by(g) |> summarise(m = mean(y, na.rm = TRUE), f = dplyr::first(y)))
  f96_oracle(df, function(t) summarise(t, nd = n_distinct(y), n = n()))
  f96_oracle(df, function(t) t |> filter(g == 1L) |> summarise(s = sum(x), mx = max(y, na.rm = TRUE)))
  f96_oracle(df, function(t) t |> filter(g > 5L) |> summarise(s = sum(x), n = n()))
})

test_that("mutate() arithmetic and conversions on integer64 match dplyr-on-bit64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  suppressWarnings({
    # x + 1L overflows at the int64 max: NA with bit64's warning
    f96_oracle(df, function(t) mutate(t, a = x + 1L, b = x - y, c = x * 2L, d = -x))
    f96_oracle(df, function(t) mutate(t, a = x + 1.5, b = y / 2L, c = y %/% 2L, d = y %% 3L))
  })
  f96_oracle(df, function(t) mutate(t, z = x))
  f96_oracle(df, function(t) mutate(t, a = x > y, b = is.na(x), c = x == i))
  # as.numeric() of the int64 extremes: bit64 warns "integer precision lost"
  suppressWarnings(f96_oracle(df, function(t) mutate(t, a = as.numeric(y), b = as.character(x))))
  suppressWarnings(f96_oracle(df, function(t) mutate(t, a = as.integer(y))))
  f96_oracle(df, function(t) mutate(t, z = dplyr::if_else(x > 0L, x, f96_i64(0))))
  f96_oracle(df, function(t) mutate(t, l = dplyr::lag(x), r = dplyr::min_rank(x), rn = dplyr::row_number(y)))
  suppressWarnings(f96_oracle(df, function(t) t |> group_by(g) |> mutate(s = sum(y), r = dplyr::min_rank(x))))
})

# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------

test_that("joins on an integer64 key match exactly", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  keys <- tibble::tibble(
    x = f96_i64(c("9007199254740992", "9007199254740993", NA, "-1", "42")),
    w = 1:5
  )
  f96_oracle(df, function(t) inner_join(t, f96_like(t, keys), by = "x"))
  f96_oracle(df, function(t) left_join(t, f96_like(t, keys), by = "x"))
  f96_oracle(df, function(t) inner_join(t, f96_like(t, keys), by = "x", na_matches = "never"))
  f96_oracle(df, function(t) semi_join(t, f96_like(t, keys), by = "x"))
  f96_oracle(df, function(t) anti_join(t, f96_like(t, keys), by = "x"))
  f96_oracle(keys, function(t) right_join(t, f96_like(t, select(df, x, g)), by = "x"))
  f96_oracle(df, function(t) inner_join(t, f96_like(t, rename(keys, y = x)), by = "y"))
})

test_that("integer64 joined with integer gives an integer64 key, as in dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  ki <- tibble::tibble(x = c(-1L, 1L, NA, 99L), w = 1:4)
  f96_oracle(df, function(t) left_join(t, f96_like(t, ki), by = "x"))
  f96_oracle(ki, function(t) inner_join(t, f96_like(t, select(df, x, g)), by = "x"))
})

test_that("integer64 joined with double is dplyr's incompatible-type error", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- f96_df()
  kd <- tibble::tibble(x = c(-1, 1.5, NA), w = 1:3)
  expect_error(left_join(df, kd, by = "x"), class = "dplyr_error_join_incompatible_type")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(left_join(tbl_gpu(df, lazy = lazy), tbl_gpu(kd, lazy = lazy), by = "x")),
      class = "dplyr_error_join_incompatible_type"
    )
  }
})

# ---------------------------------------------------------------------------
# bind_rows / set operations (vctrs' common type)
# ---------------------------------------------------------------------------

test_that("bind_rows() gives vctrs' common type for integer64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  a <- tibble::tibble(k = f96_i64(c("9007199254740993", NA)), n = 1:2)
  b_i64 <- tibble::tibble(k = f96_i64(c("-9223372036854775807", "5")))
  b_int <- tibble::tibble(k = c(7L, NA))
  b_lgl <- tibble::tibble(k = c(NA, NA))
  for (lazy in c(FALSE, TRUE)) {
    for (pair in list(list(a, b_i64), list(a, b_int), list(b_int, a), list(a, b_lgl))) {
      want <- dplyr::bind_rows(pair[[1]], pair[[2]])
      got <- collect(bind_rows(tbl_gpu(pair[[1]], lazy = lazy), tbl_gpu(pair[[2]], lazy = lazy)))
      expect_identical(f96_canon(got), f96_canon(want))
    }
  }
})

test_that("bind_rows() of integer64 and double is vctrs' incompatible-type error", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  a <- tibble::tibble(k = f96_i64(c(1, NA)))
  b <- tibble::tibble(k = c(2.5, 3))
  expect_error(dplyr::bind_rows(a, b), class = "vctrs_error_incompatible_type")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(bind_rows(tbl_gpu(a, lazy = lazy), tbl_gpu(b, lazy = lazy))),
      class = "vctrs_error_incompatible_type"
    )
  }
})

test_that("set operations on integer64 match dplyr exactly", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  a <- tibble::tibble(k = f96_i64(c("9007199254740993", "9007199254740992", NA, "1")))
  b <- tibble::tibble(k = f96_i64(c("9007199254740992", NA, "2")))
  bi <- tibble::tibble(k = c(1L, 3L))
  f96_oracle(a, function(t) dplyr::union(t, f96_like(t, b)))
  f96_oracle(a, function(t) dplyr::union_all(t, f96_like(t, b)))
  f96_oracle(a, function(t) dplyr::intersect(t, f96_like(t, b)))
  f96_oracle(a, function(t) dplyr::setdiff(t, f96_like(t, b)))
  f96_oracle(a, function(t) dplyr::union(t, f96_like(t, bi)))
})

# ---------------------------------------------------------------------------
# Results that are INT64 inside libcudf keep dplyr's R types
# ---------------------------------------------------------------------------

test_that("counts, integer sums and roll_sum() keep the types dplyr gives", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, NA),
    i = c(1L, 2L, 3L, NA, 5L),
    b = c(TRUE, FALSE, NA, TRUE, TRUE),
    v = c(1.5, 2, 3, NA, 5)
  )
  f96_oracle(df, function(t) count(t, g))
  f96_oracle(df, function(t) tally(t))
  f96_oracle(df, function(t) add_count(t, g))
  f96_oracle(df, function(t) t |> group_by(g) |> summarise(
    n = n(), s = sum(i, na.rm = TRUE), sb = sum(b, na.rm = TRUE), nd = n_distinct(i)
  ))
  f96_oracle(df, function(t) t |> group_by(g) |> mutate(n = n(), s = sum(i), cs = cumsum(b)))
  f96_oracle(df, function(t) mutate(t, r = roll_sum(i, 2), rb = roll_sum(b, 2)))
  f96_oracle(df, function(t) t |> group_by(g) |> slice_min(v, n = 1, with_ties = TRUE))
  f96_oracle(df, function(t) t |> group_by(g) |> slice_max(i, n = 1))
  tz <- tibble::tibble(t = as.POSIXct(c("2020-01-01 10:00:00", NA), tz = "UTC"), i = 1:2)
  f96_oracle(tz, function(t) mutate(t, t2 = t + 1, t3 = t - 3600))
})

# ---------------------------------------------------------------------------
# Round-1 review findings (B1-B3): added by the PM after round 1
# ---------------------------------------------------------------------------

test_that("negative and near-max integer64 literals compare exactly (their bits are NaN doubles)", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(x = f96_i64(c(-5, -1, 0, 3, NA)), i = c(-5L, -1L, 0L, 3L, NA))
  k <- f96_i64(-1)
  m <- f96_i64("9223372036854775807")
  f96_oracle(df, function(t) filter(t, x > k))
  f96_oracle(df, function(t) filter(t, i > k))
  f96_oracle(df, function(t) filter(t, x < m))
  f96_oracle(df, function(t) filter(t, x == !!f96_i64(-5)))
  f96_oracle(df, function(t) mutate(t, b = x == k, c = i <= k, d = x != m))
  f96_oracle(df, function(t) summarise(t, n = sum(x > k, na.rm = TRUE)))
})

test_that("an integer64 lag()/lead() default on an integer or double column is cast like vctrs does", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(i = 1:3, v = c(1.5, 2.5, 3.5))
  d5 <- f96_i64(5)
  f96_oracle(df, function(t) mutate(t, li = dplyr::lag(i, default = d5), lv = dplyr::lead(v, default = d5)))
  f96_oracle(df, function(t) mutate(t, li = dplyr::lag(i, default = bit64::NA_integer64_)))
})

test_that("tidyselect predicates see an integer64 column as integer64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(x = f96_i64(c(1, 2)), v = c(1.5, 2.5), i = 1:2)
  f96_oracle(df, function(t) select(t, where(bit64::is.integer64)))
  f96_oracle(df, function(t) select(t, where(~ inherits(.x, "integer64"))))
  f96_oracle(df, function(t) mutate(t, across(where(bit64::is.integer64), as.character)))
  f96_oracle(df, function(t) relocate(t, where(bit64::is.integer64), .after = i))
})

# ---------------------------------------------------------------------------
# Round-2 review findings (B4-B5): added by the PM after round 2
# ---------------------------------------------------------------------------

# dplyr's desc() is -xtfrm(x), and bit64 has no xtfrm() method, so desc() of an
# integer64 ranks the raw bits there (NA becomes 0, a negative value is NaN).
# That is dplyr's answer and the oracle here.
test_that("desc() of an integer64 in rank functions, order_by and top_n() matches dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L), i = 1:6,
    x = f96_i64(c("5", NA, "3", "-1", "9007199254740993", "2"))
  )
  f96_oracle(df, function(t) mutate(t, r = dplyr::min_rank(dplyr::desc(x)),
                                    rn = dplyr::row_number(dplyr::desc(x)),
                                    dr = dplyr::dense_rank(dplyr::desc(x))))
  f96_oracle(df, function(t) t |> group_by(g) |> mutate(r = dplyr::min_rank(dplyr::desc(x))))
  f96_oracle(df, function(t) mutate(t, l = dplyr::lag(i, order_by = dplyr::desc(x))))
  f96_oracle(df, function(t) mutate(t, s = dplyr::order_by(dplyr::desc(x), cumsum(i))))
  f96_oracle(df, function(t) dplyr::top_n(t, 5, x))
  f96_oracle(df, function(t) filter(t, dplyr::min_rank(dplyr::desc(x)) <= 2L))
})

test_that("a Date/POSIXct compared with NA_integer64_ gives dplyr's result, not an internal error", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    d = as.Date("2020-01-01") + 0:4,
    p = as.POSIXct("2020-01-01", tz = "UTC") + c(0, 60, NA, 120, 180)
  )
  f96_oracle(df, function(t) filter(t, d > bit64::NA_integer64_))
  f96_oracle(df, function(t) mutate(t, z = p == bit64::NA_integer64_))
})
