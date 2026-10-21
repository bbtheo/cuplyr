# F49 acceptance tests, tier B: horizontal aggregates on a rowwise tbl_gpu run
# natively on the GPU (sum/mean/min/max/prod of c_across()/c()/several
# columns, single-column aggregates, n(), cur_group_id(), length()), with
# dplyr 1.2.1's exact values, NA semantics and result types, in mutate(),
# filter()/filter_out() and summarise(). Shapes without a native form fall
# back and still match dplyr (value or error).
#
# Contract: scratchpad/parity/features/F49.md, ## Design (tier B).

f49b_df <- function() {
  tibble::tibble(
    id = 1:6,
    g = c("a", "a", "b", "b", "c", "c"),
    i1 = c(1L, NA, 3L, -4L, 5L, 7L),
    i2 = c(2L, 5L, NA, 4L, 0L, 1L),
    d1 = c(1.5, NA, 2.25, -0.5, 1e10, 3),
    d2 = c(0.5, 1, NA, 3, -2, 0.25),
    l1 = c(TRUE, NA, FALSE, TRUE, FALSE, TRUE),
    l2 = c(TRUE, FALSE, NA, NA, TRUE, FALSE),
    s = c("p", "q", "r", "s", "t", "u"),
    f = factor(c("x", "y", "x", "z", "y", "x")),
    dt = as.Date("2020-01-01") + 0:5
  )
}

# Rows that are entirely NA in the numeric/logical columns (row 1), or
# partly NA.
f49b_na_df <- function() {
  tibble::tibble(
    id = 1:3,
    i1 = c(NA, 1L, NA),
    i2 = c(NA, NA, 2L),
    d1 = c(NA, 1.5, NA),
    d2 = c(NA, NA, -1),
    l1 = c(NA, TRUE, NA)
  )
}

f49b_col_types <- function(x) {
  vapply(x, function(col) paste(c(class(col), typeof(col)), collapse = "/"), character(1))
}

# Same shape as f49_check() in test-parity-F49-rowwise.R: dplyr oracle vs
# tbl_gpu, class + group_vars + column types + data. `native = TRUE` forbids
# any CPU fallback (verbs and collect()).
f49b_check <- function(df, pipeline, lazy = FALSE, native = TRUE) {
  withr::local_options(dplyr.summarise.inform = FALSE)
  oracle <- pipeline(tibble::as_tibble(df))
  mode <- if (native) "error" else getOption("cuplyr.fallback", "silent")
  got <- withr::with_options(list(cuplyr.fallback = mode), pipeline(tbl_gpu(df, lazy = lazy)))
  if (native) {
    expect_true(is_tbl_gpu(got))
  }
  if (is_tbl_gpu(got)) {
    got <- withr::with_options(list(cuplyr.fallback = mode), collect(got))
  }
  expect_identical(class(got), class(oracle))
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(oracle))
  expect_identical(f49b_col_types(got), f49b_col_types(oracle))
  expect_equal(got, oracle)
  invisible(got)
}

# ---------------------------------------------------------------------------
# B1. sum/mean/min/max/prod over several columns, natively
# ---------------------------------------------------------------------------

test_that("aggregates of c_across() match dplyr natively (values, NAs, types)", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (vars in list(character(), "id")) {
    for (lazy in c(FALSE, TRUE)) {
      f49b_check(df, function(x) {
        dplyr::rowwise(x, dplyr::all_of(vars)) |>
          dplyr::mutate(
            s_int = sum(dplyr::c_across(i1:i2)),
            s_mix = sum(dplyr::c_across(c(i1, d1, l1))),
            s_rm = sum(dplyr::c_across(c(i1, i2, d2)), na.rm = TRUE),
            s_int_rm = sum(dplyr::c_across(c(i1, i2)), na.rm = TRUE),
            m_int = mean(dplyr::c_across(c(i1, i2))),
            m_rm = mean(dplyr::c_across(c(d1, d2, i1)), na.rm = TRUE),
            mn_int = min(dplyr::c_across(c(i1, i2))),
            mx_dbl = max(dplyr::c_across(c(d1, d2))),
            mn_dbl_rm = min(dplyr::c_across(c(d1, d2, i2)), na.rm = TRUE),
            mx_mix = max(dplyr::c_across(c(i1, d2))),
            p = prod(dplyr::c_across(c(i1, d2))),
            p_int = prod(dplyr::c_across(c(i1, i2)))
          )
      }, lazy = lazy)
    }
  }
})

test_that("aggregates of c() and of several column arguments match dplyr natively", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (lazy in c(FALSE, TRUE)) {
    f49b_check(df, function(x) {
      dplyr::rowwise(x) |>
        dplyr::mutate(
          a = sum(c(i1, i2)),
          b = mean(c(i1, d1, d2)),
          c = max(c(i1, l1)),
          d = min(i1, i2),
          e = max(d1, d2, na.rm = TRUE),
          f2 = sum(i1, i2, d1, na.rm = TRUE),
          h = prod(i1, i2),
          k = sum(c(l1, l2)),
          m = mean(c(l1, l2), na.rm = TRUE),
          n2 = max(c(l1, l2))
        )
    }, lazy = lazy)
  }
})

test_that("integer min()/max(na.rm = TRUE) without an all-NA row stay integer", {
  skip_if_no_gpu()
  df <- f49b_df()
  # rows of (i1, i2) are never both NA here, so dplyr returns integers
  for (lazy in c(FALSE, TRUE)) {
    f49b_check(df, function(x) {
      dplyr::rowwise(x, id) |>
        dplyr::mutate(
          mn = min(dplyr::c_across(c(i1, i2)), na.rm = TRUE),
          mx = max(i1, i2, l1, na.rm = TRUE)
        )
    }, lazy = lazy, native = !lazy)
  }
})

test_that("aggregates of one column per row are the column with dplyr's type", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (lazy in c(FALSE, TRUE)) {
    f49b_check(df, function(x) {
      dplyr::rowwise(x, g) |>
        dplyr::mutate(
          s_i = sum(i1),
          s_l = sum(l1),
          s_d_rm = sum(d1, na.rm = TRUE),
          m_i = mean(i1),
          m_d_rm = mean(d1, na.rm = TRUE),
          mn_d = min(d1),
          mx_l = max(l1),
          mx_i = max(i1),
          p_i = prod(i1)
        )
    }, lazy = lazy)
  }
  # mean(<NA>, na.rm = TRUE) of one row is NaN, not NA
  got <- collect(dplyr::mutate(dplyr::rowwise(tbl_gpu(df)), m = mean(d1, na.rm = TRUE)))
  expect_true(is.nan(got$m[2]))
})

test_that("c_across() resolves tidyselect like dplyr, never selecting the rowwise vars", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (lazy in c(FALSE, TRUE)) {
    f49b_check(df, function(x) {
      dplyr::rowwise(x, id) |>
        dplyr::mutate(
          a = sum(dplyr::c_across(dplyr::where(is.numeric))),
          b = max(dplyr::c_across(dplyr::num_range("d", 1:2)), na.rm = TRUE),
          c = mean(dplyr::c_across(i1:d2), na.rm = TRUE),
          d = sum(dplyr::c_across(c(-g, -s, -f, -dt, -l1, -l2)))
        )
    }, lazy = lazy)
    # without vars, where(is.numeric) includes id
    f49b_check(df, function(x) {
      dplyr::rowwise(x) |> dplyr::mutate(a = sum(dplyr::c_across(dplyr::where(is.numeric))))
    }, lazy = lazy)
  }
})

test_that("n(), cur_group_id() and length() of a row are constants/row numbers natively", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (vars in list(character(), "g")) {
    for (lazy in c(FALSE, TRUE)) {
      f49b_check(df, function(x) {
        dplyr::rowwise(x, dplyr::all_of(vars)) |>
          dplyr::mutate(
            n = dplyr::n(),
            k = dplyr::cur_group_id(),
            len = length(dplyr::c_across(i1:d2)),
            len2 = length(c(i1, i2)),
            r = d1 / dplyr::n()
          )
      }, lazy = lazy)
    }
  }
})

test_that("rewritten aggregates compose with elementwise expressions and later dots", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (lazy in c(FALSE, TRUE)) {
    f49b_check(df, function(x) {
      dplyr::rowwise(x, g) |>
        dplyr::mutate(
          z = sum(dplyr::c_across(i1:i2)) * 2L + id,
          big = dplyr::if_else(max(c(d1, d2)) > 1, "big", "small"),
          s = sum(c(i1, i2), na.rm = TRUE),
          t = s / 2,
          above = mean(c(i1, i2)) > 2,
          w = abs(min(d1, d2, na.rm = TRUE))
        )
    }, lazy = lazy)
  }
})

# ---------------------------------------------------------------------------
# B2. filter() / summarise() with rewrites
# ---------------------------------------------------------------------------

test_that("filter()/filter_out() with row aggregates match dplyr natively", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (vars in list(character(), "g")) {
    for (lazy in c(FALSE, TRUE)) {
      r <- function(x) dplyr::rowwise(x, dplyr::all_of(vars))
      f49b_check(df, function(x) dplyr::filter(r(x), sum(dplyr::c_across(i1:i2), na.rm = TRUE) > 3),
                 lazy = lazy)
      f49b_check(df, function(x) dplyr::filter(r(x), max(d1, d2) > 1), lazy = lazy)
      f49b_check(df, function(x) dplyr::filter(r(x), dplyr::n() == 1, mean(c(d1, d2)) < 3), lazy = lazy)
      f49b_check(df, function(x) dplyr::filter_out(r(x), mean(c(i1, i2)) < 3), lazy = lazy)
    }
  }
})

test_that("summarise() with row aggregates matches dplyr natively", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (lazy in c(FALSE, TRUE)) {
    for (groups in list(NULL, "keep", "drop", "rowwise")) {
      f49b_check(df, function(x) {
        dplyr::summarise(
          dplyr::rowwise(x, id),
          t = sum(dplyr::c_across(c(i1, d1))),
          m = mean(c(i1, i2), na.rm = TRUE),
          mx = max(d1, d2),
          .groups = groups
        )
      }, lazy = lazy)
    }
    f49b_check(df, function(x) {
      dplyr::summarise(dplyr::rowwise(x), t = sum(dplyr::c_across(i1:d2), na.rm = TRUE), n = dplyr::n())
    }, lazy = lazy)
  }
})

# ---------------------------------------------------------------------------
# B3. NA edge cases: all-NA rows, data-dependent types, overflow
# ---------------------------------------------------------------------------

test_that("sum/mean(na.rm = TRUE) of an all-NA row give 0 and NaN, natively", {
  skip_if_no_gpu()
  df <- f49b_na_df()
  for (lazy in c(FALSE, TRUE)) {
    got <- f49b_check(df, function(x) {
      dplyr::rowwise(x) |>
        dplyr::mutate(
          s_int = sum(c(i1, i2), na.rm = TRUE),
          s_dbl = sum(dplyr::c_across(c(d1, d2)), na.rm = TRUE),
          s_lgl = sum(c(l1, i1), na.rm = TRUE),
          s_na = sum(c(i1, i2)),
          m = mean(c(d1, d2), na.rm = TRUE),
          m_int = mean(dplyr::c_across(c(i1, i2)), na.rm = TRUE)
        )
    }, lazy = lazy)
    expect_identical(got$s_int[1], 0L)
    expect_true(is.nan(got$m[1]))
    expect_true(is.nan(got$m_int[1]))
  }
})

test_that("double min()/max(na.rm = TRUE) of an all-NA row give -Inf/Inf with dplyr's warning", {
  skip_if_no_gpu()
  df <- f49b_na_df()
  pipe <- function(x) {
    dplyr::rowwise(x) |> dplyr::mutate(mn = min(c(d1, d2), na.rm = TRUE))
  }
  oracle <- suppressWarnings(pipe(df))
  for (lazy in c(FALSE, TRUE)) {
    withr::local_options(cuplyr.fallback = "error")
    expect_warning(got <- collect(pipe(tbl_gpu(df, lazy = lazy))), "no non-missing arguments to min")
    expect_identical(f49b_col_types(got), f49b_col_types(oracle))
    expect_equal(got, oracle)
    expect_identical(got$mn[1], Inf)
  }
  pipe_max <- function(x) dplyr::rowwise(x) |> dplyr::mutate(mx = max(d1, d2, na.rm = TRUE))
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(got <- collect(pipe_max(tbl_gpu(df, lazy = lazy))), "no non-missing arguments to max")
    expect_identical(got$mx[1], -Inf)
    expect_equal(got, suppressWarnings(pipe_max(df)))
  }
})

test_that("integer min()/max(na.rm = TRUE) with an all-NA row become double like dplyr", {
  skip_if_no_gpu()
  df <- f49b_na_df()
  pipe <- function(x) {
    dplyr::rowwise(x) |>
      dplyr::mutate(mn = min(c(i1, i2), na.rm = TRUE), mx = max(dplyr::c_across(c(i1, l1)), na.rm = TRUE))
  }
  oracle <- suppressWarnings(pipe(df))
  expect_type(oracle$mn, "double")
  for (lazy in c(FALSE, TRUE)) {
    # eager must stay native; lazy may take the CPU route, but must match dplyr
    mode <- if (lazy) "silent" else "error"
    withr::local_options(cuplyr.fallback = mode)
    expect_warning(got <- collect(pipe(tbl_gpu(df, lazy = lazy))), "no non-missing arguments")
    expect_identical(f49b_col_types(got), f49b_col_types(oracle))
    expect_equal(got, oracle)
  }
})

test_that("an integer row sum that leaves the integer range is double, like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(i1 = c(.Machine$integer.max, 1L, NA), i2 = c(1L, 2L, 3L))
  pipe <- function(x) dplyr::rowwise(x) |> dplyr::mutate(s = sum(dplyr::c_across(c(i1, i2))))
  oracle <- pipe(df)
  expect_type(oracle$s, "double")
  withr::local_options(cuplyr.fallback = "error")
  expect_no_warning(got <- collect(pipe(tbl_gpu(df))))
  expect_identical(f49b_col_types(got), f49b_col_types(oracle))
  expect_equal(got, oracle)
  # lazy: dplyr's result, or the F42 loud failure for a fixed lazy schema
  res <- tryCatch(collect(pipe(tbl_gpu(df, lazy = TRUE))),
                  cuplyr_error_sum_overflow = function(e) "overflow")
  if (!identical(res, "overflow")) {
    expect_identical(f49b_col_types(res), f49b_col_types(oracle))
    expect_equal(res, oracle)
  }
  # no overflow: stays integer
  ok <- tibble::tibble(i1 = c(.Machine$integer.max - 1L, -5L), i2 = c(1L, 2L))
  f49b_check(ok, pipe)
  f49b_check(ok, pipe, lazy = TRUE)
})

# ---------------------------------------------------------------------------
# B4. Shapes without a native form: dplyr's result or dplyr's error
# ---------------------------------------------------------------------------

test_that("non-numeric selections and other aggregates fall back to dplyr's result", {
  skip_if_no_gpu()
  df <- f49b_df()
  for (lazy in c(FALSE, TRUE)) {
    # Date min/max keep the Date class
    f49b_check(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = max(c(dt, dt + 1))),
               lazy = lazy, native = FALSE)
    # mean(a, b) is mean(a, trim = b), not a row mean
    f49b_check(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = mean(id, id)),
               lazy = lazy, native = FALSE)
    # aggregates with no rewrite
    f49b_check(df, function(x) {
      dplyr::mutate(dplyr::rowwise(x), md = stats::median(dplyr::c_across(c(i1, i2))), sdv = stats::sd(c(d1, d2)))
    }, lazy = lazy, native = FALSE)
  }
})

test_that("invalid selections raise dplyr's errors", {
  skip_if_no_gpu()
  df <- f49b_df()
  same_error <- function(pipeline, lazy) {
    want <- tryCatch(pipeline(df), error = identity)
    got <- tryCatch({
      out <- pipeline(tbl_gpu(df, lazy = lazy))
      if (is_tbl_gpu(out)) collect(out) else out
    }, error = identity)
    expect_s3_class(want, "error")
    expect_s3_class(got, "error")
    expect_identical(class(got), class(want))
  }
  for (lazy in c(FALSE, TRUE)) {
    same_error(function(x) dplyr::mutate(dplyr::rowwise(x), z = sum(dplyr::c_across(c(i1, s)))), lazy)
    same_error(function(x) dplyr::mutate(dplyr::rowwise(x), z = sum(dplyr::c_across(c(i1, f)))), lazy)
    same_error(function(x) dplyr::mutate(dplyr::rowwise(x), z = sum(c(i1, s))), lazy)
    same_error(function(x) dplyr::mutate(dplyr::rowwise(x, g), z = sum(dplyr::c_across(c(g, i1)))), lazy)
    same_error(function(x) dplyr::mutate(dplyr::rowwise(x), z = sum(dplyr::c_across(nope))), lazy)
  }
})

# ---------------------------------------------------------------------------
# B5. It is actually fast: no per-row R evaluation
# ---------------------------------------------------------------------------

test_that("a 1e5-row rowwise sum/mean runs natively and in well under the CPU time", {
  skip_if_no_gpu()
  set.seed(49)
  n <- 1e5
  df <- tibble::tibble(a = stats::runif(n), b = stats::runif(n), c = stats::runif(n),
                       d = stats::runif(n), e = stats::runif(n))
  df$c[seq(1, n, by = 7)] <- NA
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(df)
  # dplyr's CPU route takes ~20 s for this; the GPU route must not loop in R
  elapsed <- system.time({
    out <- gt |>
      dplyr::rowwise() |>
      dplyr::mutate(s = sum(dplyr::c_across(a:e)), m = mean(c(a, b, c), na.rm = TRUE)) |>
      collect()
  })[["elapsed"]]
  expect_lt(elapsed, 5)
  expect_s3_class(out, "rowwise_df")
  expect_equal(out$s, df$a + df$b + df$c + df$d + df$e)
  cnt <- (!is.na(df$a)) + (!is.na(df$b)) + (!is.na(df$c))
  expect_equal(out$m, (df$a + df$b + ifelse(is.na(df$c), 0, df$c)) / cnt)
})
