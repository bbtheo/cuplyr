# F96 extras: the integer64 cases of the acceptance file that call dplyr
# functions by bare name (row_number(), first(), if_else(), min_rank()), written
# with dplyr:: so they run when dplyr is not attached, plus the INT64 gate and
# the schema-type rules. testthat's expect_equal() compares integer64 loosely,
# so every comparison also goes through x_canon() (exact strings).

x_canon <- function(x) {
  x <- tibble::as_tibble(dplyr::ungroup(x))
  list(
    names = names(x),
    classes = unname(vapply(x, function(col) paste(class(col), collapse = "/"), character(1))),
    values = unname(lapply(x, function(col) {
      if (inherits(col, "integer64")) as.character(col) else col
    }))
  )
}

x_oracle <- function(df, pipeline, arrange_by = NULL) {
  eager <- expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  expect_identical(x_canon(eager$gpu), x_canon(eager$oracle), label = "eager (exact)")
  lazy <- expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
  expect_identical(x_canon(lazy$gpu), x_canon(lazy$oracle), label = "lazy (exact)")
  invisible(eager)
}

x_i64 <- function(x) bit64::as.integer64(x)

x_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 2L, 3L),
    x = x_i64(c("0", "-1", "1", NA, "9007199254740993", "9007199254740992",
                "-9007199254740993", "9223372036854775807", "-9223372036854775807")),
    y = x_i64(c("0", "1", "1", "5", "9007199254740993", "9007199254740993",
                NA, "9223372036854775807", "-1")),
    i = c(0L, -2L, 1L, 4L, NA, 7L, 3L, 2L, 1L)
  )
}

# The pipeline runs without a CPU fallback, eager and lazy (a fallback is an
# error under this option; by default it is a silent condition), and matches
# dplyr exactly.
x_native <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  x_oracle(df, pipeline)
}

test_that("rank functions, lag/lead and window aggregates of integer64 match dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- x_df()
  x_oracle(df, function(t) dplyr::mutate(
    dplyr::group_by(t, y), r = dplyr::row_number()))
  x_oracle(df, function(t) dplyr::mutate(
    t, l = dplyr::lag(x), ld = dplyr::lead(y), r = dplyr::min_rank(x),
    d = dplyr::dense_rank(y), rn = dplyr::row_number(y)))
  suppressWarnings(x_oracle(df, function(t) dplyr::mutate(
    dplyr::group_by(t, g), s = sum(y), r = dplyr::min_rank(x))))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::if_else(x > 0L, x, x_i64(0))))
  suppressWarnings(x_oracle(df, function(t) dplyr::summarise(
    dplyr::group_by(t, g), m = mean(y, na.rm = TRUE), f = dplyr::first(y), l = dplyr::last(x))))
})

test_that("lag() with a default and first()/last() of integer64 take dplyr's route", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- x_df()
  x_oracle(df, function(t) dplyr::mutate(t, l = dplyr::lag(x, default = x_i64(7))))
  x_oracle(df, function(t) dplyr::mutate(t, l = dplyr::lead(x, 2L)))
})

test_that("the native integer64 operations raise no CPU fallback notice", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- x_df()
  k <- x_i64("9007199254740993")
  x_native(df, function(t) dplyr::filter(t, x > 0L))
  x_native(df, function(t) dplyr::filter(t, x == 9007199254740992))
  x_native(df, function(t) dplyr::filter(t, x == k, !is.na(y)))
  x_native(df, function(t) dplyr::arrange(t, dplyr::desc(x)))
  x_native(df, function(t) dplyr::mutate(t, r = dplyr::min_rank(x), l = dplyr::lag(y)))
  x_native(df, function(t) dplyr::summarise(dplyr::group_by(t, y), n = dplyr::n()))
})

test_that("a whole double literal is compared as integer64, a fractional one as bit64 does", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- x_df()
  x_oracle(df, function(t) dplyr::filter(t, x < 9007199254740993))
  x_oracle(df, function(t) dplyr::filter(t, 9007199254740992 <= x))
  x_oracle(df, function(t) dplyr::filter(t, x > -1.5))
  x_oracle(df, function(t) dplyr::filter(t, x != NA_real_))
  x_oracle(df, function(t) dplyr::filter(t, x != NA_integer_))
  suppressWarnings(x_oracle(df, function(t) dplyr::filter(t, x >= 2^63)))
})

test_that("an integer64 NA literal and a length-1 integer64 vector are exact", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- x_df()
  x_oracle(df, function(t) dplyr::mutate(t, z = bit64::NA_integer64_))
  x_oracle(df, function(t) dplyr::filter(t, x == bit64::NA_integer64_))
  x_oracle(df, function(t) dplyr::mutate(t, z = x_i64("9223372036854775807")))
})

test_that("integer64 keys joined with integer keys, and non-equi joins", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- x_df()
  ki <- tibble::tibble(x = c(0L, 1L, NA), w = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(dplyr::inner_join(tbl_gpu(df, lazy = lazy), tbl_gpu(ki, lazy = lazy), by = "x"))
    want <- dplyr::inner_join(df, ki, by = "x")
    expect_identical(x_canon(got), x_canon(want))
    got <- collect(dplyr::inner_join(tbl_gpu(df, lazy = lazy), tbl_gpu(ki, lazy = lazy),
                                     by = dplyr::join_by(x >= x)))
    want <- dplyr::inner_join(df, ki, by = dplyr::join_by(x >= x))
    expect_identical(x_canon(got), x_canon(want))
  }
})

test_that("INT64 is integer64 only: a schema type decides the R class", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(i = c(1L, 2L, 3L, NA), b = c(TRUE, FALSE, NA, TRUE), d = c(1, 2, 3, 4))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    out <- collect(dplyr::mutate(g, r = roll_sum(i, 2L), rb = roll_sum(b, 2L), rd = roll_sum(d, 2L)))
    expect_type(out$r, "double")
    expect_type(out$rb, "double")
    expect_false(inherits(out$r, "integer64"))
    expect_identical(unname(unclass(g)$schema$types[1]), "INT32")
  }
  expect_identical(
    unname(unclass(dplyr::mutate(tbl_gpu(df, lazy = FALSE), r = roll_sum(i, 2L)))$schema$types[4]),
    "FLOAT64"
  )
})

test_that("integer64 with a double in a set operation is an error, with an integer it widens", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  a <- tibble::tibble(k = x_i64(c("9007199254740993", NA)))
  b <- tibble::tibble(k = c(2.5, 3))
  expect_error(dplyr::union(a, b), "not compatible", class = "rlang_error")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(dplyr::union(tbl_gpu(a, lazy = lazy), tbl_gpu(b, lazy = lazy))),
      "not compatible", class = "rlang_error"
    )
  }
})

test_that("a long glimpse line is cut like pillar's", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    a = 1:9,
    b = x_i64(c("9007199254740993", "9007199254740992", "-9007199254740993", "1", "2", "3", "4", "5", NA)),
    c = x_i64(c("9007199254740993", "9007199254740992", "-9007199254740993", "1", "2", "3", "4", "5", NA))
  )
  gpu <- capture.output(dplyr::glimpse(tbl_gpu(df, lazy = FALSE)))
  cpu <- capture.output(dplyr::glimpse(df))
  expect_identical(gpu, cpu)
})

# ---------------------------------------------------------------------------
# Round 2: an integer64 value is handled by its bits or goes to the CPU, never
# read as a double (the audit behind B1-B3)
# ---------------------------------------------------------------------------

# dplyr and the GPU table (eager and lazy) all fail on `pipeline`, with
# dplyr's condition class and message
x_both_error <- function(df, pipeline) {
  want <- tryCatch(pipeline(df), error = identity)
  expect_s3_class(want, "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(collect(pipeline(tbl_gpu(df, lazy = lazy))), error = identity)
    expect_s3_class(got, "error")
    expect_identical(class(got), class(want))
    expect_identical(conditionMessage(got), conditionMessage(want))
  }
}

test_that("integer64 literals whose bits are NaN doubles compare exactly everywhere they can sit", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    x = x_i64(c("-5", "-1", "0", "3", NA, "9223372036854775807", "9223372036854775806", "-4503599627370495")),
    i = c(-5L, -1L, 0L, 3L, NA, 7L, 1L, 2L),
    v = c(1.5, -1, 2, NA, 4, 5, 6, 7)
  )
  kn <- x_i64(-1)
  km <- x_i64("9223372036854775807")
  kq <- x_i64("-4503599627370495")
  x_oracle(df, function(t) dplyr::filter(t, x >= kq))
  x_oracle(df, function(t) dplyr::filter(t, x < km))
  x_oracle(df, function(t) dplyr::filter(t, x == km))
  x_oracle(df, function(t) dplyr::filter(t, i != kn))
  x_oracle(df, function(t) dplyr::filter(t, kn < x))
  x_oracle(df, function(t) dplyr::filter(t, x > !!x_i64(-1)))
  x_oracle(df, function(t) dplyr::filter(t, x > bit64::as.integer64(-1)))
  x_oracle(df, function(t) dplyr::filter(t, x > bit64::as.integer64("-4503599627370494")))
  x_oracle(df, function(t) dplyr::filter(t, dplyr::between(x, kn, x_i64(3))))
  x_oracle(df, function(t) dplyr::filter(t, dplyr::between(i, kn, x_i64(3))))
  x_oracle(df, function(t) dplyr::mutate(t, z = x > kn, w = i == kn, u = kn == kn, q = kn < km, r = km == km))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::if_else(x > kn, 1L, 0L)))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::case_when(x > kn ~ "a", i == kn ~ "b", .default = "c")))
  x_oracle(df, function(t) dplyr::mutate(t, z = ifelse(x <= kq, TRUE, FALSE)))
  x_oracle(df, function(t) dplyr::mutate(dplyr::group_by(t, i > 0L), z = sum(x > kn, na.rm = TRUE)))
  x_oracle(df, function(t) dplyr::summarise(t, n = sum(x > kn, na.rm = TRUE), m = sum(i < km, na.rm = TRUE)))
  x_oracle(df, function(t) dplyr::mutate(t, z = is.na(x) | x > kq))
  x_native(df, function(t) dplyr::filter(t, x > kn))
  x_native(df, function(t) dplyr::filter(t, i > kn))
})

test_that("an integer64 lag()/lead() default is cast to x's type, or taken as bits when x is integer64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    i = 1:4, v = c(1.5, 2.5, 3.5, NA), b = c(TRUE, FALSE, NA, TRUE),
    d = as.Date("2020-01-01") + 0:3, x = x_i64(c("1", "-1", NA, "9223372036854775807"))
  )
  k5 <- x_i64(5)
  kn <- x_i64(-1)
  kna <- bit64::NA_integer64_
  x_oracle(df, function(t) dplyr::mutate(t, a = dplyr::lag(i, default = kn), b2 = dplyr::lead(i, 2, k5)))
  x_oracle(df, function(t) dplyr::mutate(t, a = dplyr::lag(v, default = kn), b2 = dplyr::lead(v, default = kna)))
  x_oracle(df, function(t) dplyr::mutate(t, a = dplyr::lag(i, default = kna)))
  x_oracle(df, function(t) dplyr::mutate(t, a = dplyr::lag(x, default = kna), b2 = dplyr::lead(x, default = NA)))
  x_oracle(df, function(t) dplyr::mutate(t, a = dplyr::lag(x, default = x_i64("9223372036854775807"))))
  x_oracle(df, function(t) dplyr::mutate(t, a = dplyr::lag(x, default = 0L)))
  x_both_error(df, function(t) dplyr::mutate(t, a = dplyr::lag(d, default = k5)))
})

test_that("an integer64 where dplyr wants a plain number is dplyr's call (error or value), not a read of the bits", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(i = c(-1L, 2L, NA, 4L), v = c(1.5, -1, 2, NA), s = c("ab", "cd", NA, "ef"))
  k2 <- x_i64(2)
  kn <- x_i64(-1)
  x_both_error(df, function(t) dplyr::mutate(t, z = dplyr::lag(i, n = k2)))
  x_both_error(df, function(t) dplyr::mutate(t, z = dplyr::lead(v, k2)))
  x_both_error(df, function(t) dplyr::mutate(t, z = dplyr::ntile(i, k2)))
  x_both_error(df, function(t) dplyr::slice(t, k2))
  # a vctrs error on both sides: x and a bound have no common type
  x_both_error(df, function(t) dplyr::filter(t, dplyr::between(i, k2, 5)))
  x_both_error(df, function(t) dplyr::filter(t, dplyr::between(v, kn, 5)))
  x_both_error(df, function(t) dplyr::filter(t, dplyr::between(i, 1, k2)))
  # integer64 bounds and an integer x have one: integer64
  x_oracle(df, function(t) dplyr::filter(t, dplyr::between(i, kn, k2)))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::between(i, kn, x_i64("9223372036854775807"))))
  # the same values that are numbers for dplyr
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::lag(i, n = 2L), w = dplyr::ntile(i, 2L)))
})

test_that("integer64 in the other literal positions is dplyr's route", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    g = c(1L, 1L, 2L), i = c(-1L, 2L, NA), x = x_i64(c("-1", "2", NA)),
    d = as.Date("2020-01-01") + 0:2, p = as.POSIXct("2020-01-01", tz = "UTC") + c(0, 60, NA)
  )
  k2 <- x_i64(2)
  kn <- x_i64(-1)
  kv <- x_i64(c(-1, 2))
  x_oracle(df, function(t) dplyr::filter(t, i %in% kv))
  x_oracle(df, function(t) dplyr::filter(t, x %in% kv))
  x_oracle(df, function(t) dplyr::mutate(t, z = kn, w = c(kn, k2, kn)))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::coalesce(i, kn), w = dplyr::if_else(is.na(i), k2, kn)))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::na_if(i, kn), w = dplyr::na_if(x, kn)))
  x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::case_when(i > 0L ~ kn, .default = k2)))
  suppressWarnings(x_oracle(df, function(t) dplyr::mutate(t, z = dplyr::recode(i, `-1` = k2))))
  x_oracle(df, function(t) dplyr::mutate(t, z = i + kn, w = i * k2, u = -kn))
  x_oracle(df, function(t) dplyr::summarise(t, z = kn, w = dplyr::nth(i, 2L, default = k2)))
  suppressWarnings(x_oracle(df, function(t) dplyr::mutate(t, z = d + k2, w = p - kn)))
  suppressWarnings(x_oracle(df, function(t) dplyr::filter(t, d > kn)))
})

test_that("tidyselect predicates see integer64 as integer64, and a double when bit64 is missing", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = x_i64(c(1, 2, 3)), y = x_i64(c(3, 2, 1)), v = c(1.5, 2.5, 3.5))
  x_oracle(df, function(t) dplyr::rename_with(t, toupper, where(bit64::is.integer64)))
  x_oracle(df, function(t) dplyr::mutate(t, dplyr::across(where(bit64::is.integer64), ~ .x > 1L)))
  x_oracle(df, function(t) dplyr::filter(t, dplyr::if_all(where(bit64::is.integer64), ~ .x > 1L)))
  x_oracle(df, function(t) dplyr::summarise(dplyr::group_by(t, dplyr::across(where(bit64::is.integer64))), n = dplyr::n()))
  x_oracle(df, function(t) dplyr::summarise(t, n = dplyr::n(), .by = where(bit64::is.integer64)))
  x_oracle(df, function(t) dplyr::select(t, !where(bit64::is.integer64)))
  x_oracle(df, function(t) dplyr::select(t, where(is.numeric) & !where(bit64::is.integer64)))
  local_mocked_bindings(bit64_available = function() FALSE)
  expect_identical(
    names(collect(dplyr::select(tbl_gpu(df), where(is.double)))),
    names(dplyr::select(tibble::tibble(g = 1L, x = 1, y = 1, v = 1), where(is.double)))
  )
  # without bit64 the columns are doubles, so nothing is an integer64
  expect_identical(names(collect(dplyr::select(tbl_gpu(df), where(~ inherits(.x, "integer64"))))),
                   character())
})

test_that("an integer64 key joined with a logical key matches dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(x = x_i64(c("0", "1", NA, "9007199254740993")), i = 1:4)
  kl <- tibble::tibble(x = c(TRUE, FALSE, NA), w = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    # vctrs' common type of integer64 and logical is integer64
    got <- collect(dplyr::left_join(tbl_gpu(df, lazy = lazy), tbl_gpu(kl, lazy = lazy), by = "x"))
    want <- dplyr::left_join(df, kl, by = "x")
    expect_identical(x_canon(got), x_canon(want))
  }
})

# ---------------------------------------------------------------------------
# Round 3: desc() of an integer64 is dplyr's -xtfrm() of the bits (B4), and a
# classed NA literal against a timestamp keeps its type (B5)
# ---------------------------------------------------------------------------

test_that("desc() of an integer64 goes to dplyr wherever cuplyr would read it as a sort direction", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L), i = 1:6,
    x = x_i64(c("5", NA, "3", "-1", "9007199254740993", "2"))
  )
  x_oracle(df, function(t) dplyr::mutate(t, dplyr::across(x, ~ dplyr::min_rank(dplyr::desc(.x)), .names = "r")))
  x_oracle(df, function(t) dplyr::mutate(dplyr::group_by(t, g), r = dplyr::row_number(dplyr::desc(x)),
                                          d = dplyr::dense_rank(dplyr::desc(x))))
  x_oracle(df, function(t) dplyr::filter(dplyr::group_by(t, g), dplyr::row_number(dplyr::desc(x)) == 2L))
  x_oracle(df, function(t) dplyr::mutate(t, l = dplyr::lead(i, order_by = dplyr::desc(x)),
                                          l2 = dplyr::lag(i, 2L, order_by = dplyr::desc(x))))
  x_oracle(df, function(t) dplyr::mutate(t, s = dplyr::with_order(dplyr::desc(x), cumsum, i)))
  x_oracle(df, function(t) dplyr::mutate(dplyr::group_by(t, g), s = dplyr::order_by(dplyr::desc(x), cumsum(i))))
  x_oracle(df, function(t) dplyr::mutate(t, f = dplyr::first(i, order_by = dplyr::desc(x)),
                                          n2 = dplyr::nth(i, 2L, order_by = dplyr::desc(x)),
                                          l = dplyr::last(i, order_by = dplyr::desc(x))))
  x_oracle(df, function(t) dplyr::summarise(dplyr::group_by(t, g), f = dplyr::first(i, order_by = dplyr::desc(x))))
  x_oracle(df, function(t) dplyr::top_n(dplyr::group_by(t, g), 2, x))
  x_oracle(df, function(t) dplyr::slice_max(t, dplyr::desc(x), n = 2))
  x_oracle(df, function(t) dplyr::arrange(t, dplyr::across(x, dplyr::desc)))
})

test_that("an integer64 sort key that is not desc() stays on the GPU", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L), i = 1:6,
    x = x_i64(c("5", NA, "3", "-1", "9007199254740993", "2"))
  )
  x_native(df, function(t) dplyr::arrange(t, dplyr::desc(x)))
  x_native(df, function(t) dplyr::mutate(t, r = dplyr::min_rank(x), l = dplyr::lag(i, order_by = x)))
  x_native(df, function(t) dplyr::mutate(t, l = dplyr::lag(i, order_by = -x)))
  x_native(df, function(t) dplyr::mutate(t, r = dplyr::min_rank(dplyr::desc(i))))
  x_native(df, function(t) dplyr::mutate(t, s = dplyr::order_by(x, cumsum(i))))
})

test_that("a Date/POSIXct against NA_integer64_ is dplyr's in every comparison shape", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(
    d = as.Date("2020-01-01") + 0:4,
    p = as.POSIXct("2020-01-01", tz = "UTC") + c(0, 60, NA, 120, 180),
    i = 1:5
  )
  na64 <- bit64::NA_integer64_
  x_oracle(df, function(t) dplyr::mutate(t, z = bit64::NA_integer64_ <= d, w = p != na64))
  x_oracle(df, function(t) dplyr::filter(t, na64 == p | i > 3L))
  x_both_error(df, function(t) dplyr::mutate(t, z = dplyr::between(d, bit64::NA_integer64_, d)))
  # a plain numeric NA and a same-type timestamp NA still compare on the GPU
  x_native(df, function(t) dplyr::mutate(t, z = d == NA_real_, w = p != NA_integer_,
                                          u = d < as.Date(NA)))
})
