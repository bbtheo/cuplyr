# F18 acceptance tests: type fidelity and factor correctness.
# A1: a factor compared with a number compares labels (base R's Ops.factor /
#     Ops.ordered), never the integer codes.
# A2: eager select(x = y) dropping a factor `x` must not leave x's levels on
#     the new `x`.
# A3: integer-backed Date / POSIXct upload as timestamps, not INT32.
# B:  classed atomics with no GPU representation fail loudly at upload.
# C:  NA_character_ patterns in grepl()/sub()/startsWith() give NA.
# See scratchpad/parity/features/F18.md "## Design".

same_both_f18 <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Labels look numeric and the codes disagree with them: code 1 is "10",
# code 2 is "1", code 3 is "2". `o` is ordered with "2" < "10" < "1".
f18_fct_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")),
    o = factor(c("1", "2", "10", NA, "2", "1"), levels = c("2", "10", "1"), ordered = TRUE),
    n = c(1L, 2L, 3L, 4L, 2L, 10L),
    g = c("a", "a", "b", "b", "b", "a")
  )
}

# ---------------------------------------------------------------------------
# A1: factor compared with a number
# ---------------------------------------------------------------------------

test_that("F18 A1: mutate() factor ==/!= number compares labels", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::mutate(d,
    a = f == 1, b = f == 10, c = f != 2, e = 1 == f, h = f == 1L, k = f == 1.5
  ))
  same_both_f18(df, function(d) dplyr::mutate(d, a = f == 1))
  same_both_f18(df, function(d) dplyr::mutate(d, a = f != 10))
})

test_that("F18 A1: filter() factor ==/!= number compares labels", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::filter(d, f == 1))
  same_both_f18(df, function(d) dplyr::filter(d, f == 10))
  same_both_f18(df, function(d) dplyr::filter(d, f != 2))
  same_both_f18(df, function(d) dplyr::filter(d, 2 == f))
  same_both_f18(df, function(d) dplyr::filter(d, f == 7))
  same_both_f18(df, function(d) dplyr::filter(d, f == 1 | n > 3))
})

test_that("F18 A1: factor %in% numbers compares labels", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::filter(d, f %in% c(1, 10)))
  same_both_f18(df, function(d) dplyr::filter(d, !f %in% 2))
  same_both_f18(df, function(d) dplyr::mutate(d, a = f %in% c(1, 10), b = f %in% 1.5, c = f %in% 2L))
})

test_that("F18 A1: ordered factor </<=/>/>=/== number uses level order of the label", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::mutate(d,
    a = o < 2, b = o >= 10, c = o > 1, e = o == 2, h = o <= 1, k = o > 10
  ))
  same_both_f18(df, function(d) dplyr::filter(d, o >= 10))
  same_both_f18(df, function(d) dplyr::filter(d, o < 1))
  # a number that is not a level compares as NA
  same_both_f18(df, function(d) dplyr::mutate(d, a = o < 1.5))
})

test_that("F18 A1: unordered factor < number warns and gives NA, like base R", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  expected <- suppressWarnings(dplyr::mutate(df, y = f < 1))
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(
      out <- dplyr::collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f < 1)),
      "not meaningful for factors"
    )
    expect_equal(tibble::as_tibble(out), expected)
  }
})

test_that("F18 A1: grouped, nested and summarised factor-number comparisons", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::mutate(dplyr::group_by(d, g), y = f == 1))
  same_both_f18(df, function(d) dplyr::filter(dplyr::group_by(d, g), f == 2))
  same_both_f18(df, function(d) dplyr::mutate(d, y = dplyr::if_else(f == 1, "one", "other")))
  same_both_f18(df, function(d) dplyr::mutate(d, y = dplyr::case_when(f == 10 ~ 1L, .default = 0L)))
  same_both_f18(df, function(d) dplyr::summarise(dplyr::group_by(d, g),
    m = sum(f == 2, na.rm = TRUE)), arrange_by = "g")
  same_both_f18(df, function(d) dplyr::summarise(d, m = sum(f == 1, na.rm = TRUE)))
})

test_that("F18 A1: factor compared with an integer column compares labels", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::mutate(d, y = f == n))
  same_both_f18(df, function(d) dplyr::filter(d, f == n))
})

test_that("F18 A1: count()/add_count()/distinct()/slice_max() expressions compare labels", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  same_both_f18(df, function(d) dplyr::count(d, y = f == 1))
  same_both_f18(df, function(d) dplyr::add_count(d, y = f == 1, name = "k"))
  same_both_f18(df, function(d) dplyr::distinct(d, f, y = f == 1), arrange_by = c("f", "y"))
  same_both_f18(df, function(d) dplyr::slice_max(d, f == 1, n = 1), arrange_by = "n")
  same_both_f18(df, function(d) dplyr::slice_min(d, f %in% c(2, 10), n = 1), arrange_by = "n")
  same_both_f18(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), f == 1, n = 1),
                arrange_by = c("g", "n"))
})

test_that("F18 A1: na_if(factor, number) inside an expression errors like dplyr", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  expect_error(dplyr::filter(df, !is.na(dplyr::na_if(f, 1))), "Can't convert")
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_error(dplyr::collect(dplyr::filter(g, !is.na(dplyr::na_if(f, 1)))), "Can't convert")
    expect_error(dplyr::collect(dplyr::mutate(g, y = is.na(dplyr::na_if(f, 1)))), "Can't convert")
    expect_error(dplyr::collect(dplyr::summarise(g, m = sum(is.na(dplyr::na_if(f, 1))))), "Can't convert")
  }
})

test_that("F18 A1: comparisons on values computed from a factor never use codes", {
  skip_if_no_gpu()
  df <- f18_fct_df()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    # dplyr warns "not meaningful for factors" and the comparison is NA -> 0 rows
    for (pipe in list(
      function(d) dplyr::filter(d, dplyr::near(f, 1)),
      function(d) dplyr::filter(d, f - 1 == 0),
      function(d) dplyr::filter(d, f / 2 == 0.5)
    )) {
      expected <- suppressWarnings(pipe(df))
      expect_warning(out <- dplyr::collect(pipe(g)), "not meaningful for factors")
      expect_equal(tibble::as_tibble(out), expected)
    }
    expect_error(dplyr::collect(dplyr::filter(g, dplyr::coalesce(f, 1) == 1)), "Can't combine")
    expect_error(dplyr::collect(dplyr::mutate(g, y = round(f) == 1)), "not meaningful for factors")
    expect_error(dplyr::collect(dplyr::summarise(g, m = sum(round(f) == 1, na.rm = TRUE))),
                 "not meaningful for factors")
    expect_error(dplyr::collect(dplyr::count(g, y = round(f) == 1)), "not meaningful for factors")
  }
})

test_that("F18 A1: zero-row factor-number comparisons", {
  skip_if_no_gpu()
  df <- f18_fct_df()[0, ]
  same_both_f18(df, function(d) dplyr::filter(d, f == 1))
  same_both_f18(df, function(d) dplyr::mutate(d, y = f == 1, z = f %in% c(1, 2)))
})

# ---------------------------------------------------------------------------
# A2: select() renaming onto a dropped factor's name
# ---------------------------------------------------------------------------

test_that("F18 A2: select(x = y) does not keep the dropped factor x's levels", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    x = factor(c("a", "b", "a")),
    y = c(1.5, 2, 3),
    z = 1:3
  )
  same_both_f18(df, function(d) dplyr::select(d, x = y))
  same_both_f18(df, function(d) dplyr::select(d, x = y, y = x))
  same_both_f18(df, function(d) dplyr::select(d, z, x = y))
  same_both_f18(df, function(d) dplyr::select(d, y = x))
  same_both_f18(df, function(d) dplyr::select(d, x = z, w = x))
  same_both_f18(df, function(d) dplyr::rename(d, w = x, x = y))

  out <- tibble::as_tibble(dplyr::collect(dplyr::select(tbl_gpu(df), x = y)))
  expect_type(out$x, "double")
  expect_false(is.factor(out$x))
  expect_equal(out$x, c(1.5, 2, 3))
})

# ---------------------------------------------------------------------------
# A3: integer-backed Date and POSIXct
# ---------------------------------------------------------------------------

f18_date_df <- function() {
  tibble::tibble(
    d = seq.Date(as.Date("2020-01-01"), by = "day", length.out = 4),
    d2 = structure(c(18262L, NA, 18264L, 18265L), class = "Date"),
    n = 1:4
  )
}

test_that("F18 A3: integer-backed Date round-trips as a Date", {
  skip_if_no_gpu()
  df <- f18_date_df()
  expect_type(df$d, "integer")

  for (lazy in c(FALSE, TRUE)) {
    out <- tibble::as_tibble(dplyr::collect(tbl_gpu(df, lazy = lazy)))
    expect_s3_class(out$d, "Date")
    expect_s3_class(out$d2, "Date")
    expect_equal(as.double(out$d), as.double(df$d))
    expect_equal(as.double(out$d2), as.double(df$d2))
  }
  same_both_f18(df, function(d) d)
})

test_that("F18 A3: integer-backed Date works through verbs", {
  skip_if_no_gpu()
  df <- f18_date_df()
  same_both_f18(df, function(d) dplyr::filter(d, d > as.Date("2020-01-02")))
  same_both_f18(df, function(d) dplyr::filter(d, d2 >= structure(18263L, class = "Date")))
  same_both_f18(df, function(d) dplyr::arrange(d, dplyr::desc(d)))
  same_both_f18(df, function(d) dplyr::summarise(dplyr::group_by(d, d), k = dplyr::n()),
                arrange_by = "d")
})

test_that("F18 A3: integer-backed POSIXct round-trips as a POSIXct", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    t = structure(c(0L, 3600L, NA, 7200L), class = c("POSIXct", "POSIXt"), tzone = "UTC"),
    n = 1:4
  )
  expect_type(df$t, "integer")
  for (lazy in c(FALSE, TRUE)) {
    out <- tibble::as_tibble(dplyr::collect(tbl_gpu(df, lazy = lazy)))
    expect_s3_class(out$t, "POSIXct")
    expect_equal(as.double(out$t), as.double(df$t))
  }
  same_both_f18(df, function(d) dplyr::filter(d, n > 1))
})

# ---------------------------------------------------------------------------
# B: classed atomics without a GPU representation fail loudly
# ---------------------------------------------------------------------------

f18_expect_unsupported <- function(col, nm, cls_pattern) {
  df <- tibble::tibble(n = 1:3)
  df[[nm]] <- col
  for (lazy in c(FALSE, TRUE)) {
    err <- expect_error(tbl_gpu(df, lazy = lazy), class = "cuplyr_error_unsupported_class")
    expect_match(conditionMessage(err), nm, fixed = TRUE)
    expect_match(conditionMessage(err), cls_pattern)
  }
}

test_that("F18 B: S3-classed atomics with no GPU type error at upload", {
  skip_if_no_gpu()
  f18_expect_unsupported(ts(1:3), "tscol", "ts")
  f18_expect_unsupported(as.difftime(c(1, 2, 3), units = "hours"), "dtcol", "difftime")
  f18_expect_unsupported(structure(c(1, 2, 3), class = "alien"), "aliencol", "alien")
  f18_expect_unsupported(I(1:3), "asiscol", "AsIs")
})

test_that("F18 B: hms and IDate error at upload", {
  skip_if_no_gpu()
  skip_if_not_installed("hms")
  f18_expect_unsupported(hms::hms(1:3), "hmscol", "hms")
  skip_if_not_installed("data.table")
  f18_expect_unsupported(
    data.table::as.IDate(c("2020-01-01", "2020-01-02", "2020-01-03")), "idatecol", "IDate"
  )
})

test_that("F18 B: an S4 class extending integer errors at upload", {
  skip_if_no_gpu()
  F18S4 <- suppressWarnings(methods::setClass("F18S4", contains = "integer"))
  on.exit(methods::removeClass("F18S4"), add = TRUE)
  f18_expect_unsupported(F18S4(1:3), "s4col", "F18S4")
})

test_that("F18 B: POSIXlt errors clearly at upload", {
  skip_if_no_gpu()
  lt <- as.POSIXlt(as.POSIXct(c("2020-01-01", "2020-01-02", "2020-01-03"), tz = "UTC"))
  f18_expect_unsupported(lt, "ltcol", "POSIXlt")
  # the message must not blame device memory
  err <- tryCatch(tbl_gpu(tibble::tibble(ltcol = lt)), error = identity)
  expect_false(grepl("device memory", conditionMessage(err)))
})

test_that("F18 B: bind_cols() of an unsupported class errors the same way", {
  skip_if_no_gpu()
  expect_error(
    cuplyr::bind_cols(tbl_gpu(tibble::tibble(n = 1:3)), dtcol = as.difftime(c(1, 2, 3), units = "mins")),
    class = "cuplyr_error_unsupported_class"
  )
})

test_that("F18 B: supported classes still upload silently; integer64 still warns", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    f = factor(c("a", "b", "a")),
    o = factor(c("a", "b", "a"), ordered = TRUE),
    d = as.Date(c("2020-01-01", "2020-01-02", NA)),
    t = as.POSIXct(c("2020-01-01 10:00:00", NA, "2020-01-03 00:00:00"), tz = "UTC"),
    x = c(1.5, NA, 3),
    i = c(1L, 2L, NA),
    s = c("a", NA, "c"),
    l = c(TRUE, NA, FALSE)
  )
  expect_no_condition(g <- tbl_gpu(df))
  out <- tibble::as_tibble(dplyr::collect(g))
  expect_equal(out, df)

  skip_if_not_installed("bit64")
  expect_warning(tbl_gpu(tibble::tibble(k = bit64::as.integer64(1:3))), "integer64")
})

# ---------------------------------------------------------------------------
# C: NA_character_ patterns
# ---------------------------------------------------------------------------

test_that("F18 C: NA_character_ pattern in base string functions gives NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c("a", "NA", NA, "b"), n = 1:4)
  p <- NA_character_
  same_both_f18(df, function(d) dplyr::mutate(d,
    a = grepl(NA_character_, x),
    b = grepl(p, x, fixed = TRUE),
    c = sub(NA_character_, "z", x),
    e = gsub(p, "z", x),
    h = startsWith(x, NA_character_),
    k = endsWith(x, p),
    m = sub("a", NA_character_, x)
  ))
  same_both_f18(df, function(d) dplyr::filter(d, grepl(NA_character_, x)))
  same_both_f18(df, function(d) dplyr::filter(d, !grepl(p, x)))
})

test_that("F18 C: stringr's NA pattern error is kept", {
  skip_if_no_gpu()
  skip_if_not_installed("stringr")
  df <- tibble::tibble(x = c("a", "NA", NA))
  expect_error(dplyr::mutate(df, y = stringr::str_detect(x, NA_character_)), "can not contain NA")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      dplyr::collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = stringr::str_detect(x, NA_character_))),
      "can not contain NA"
    )
    expect_error(
      dplyr::collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = stringr::str_replace(x, NA_character_, "z"))),
      "can not contain NA"
    )
  }
})
