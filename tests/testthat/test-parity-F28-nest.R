# F28 acceptance tests: group_nest() / nest_by() / nest_join() on tbl_gpu.
#
# Contract (tier A, dplyr 1.2.1):
# - group_nest() and nest_by() forward `...` (including `.drop`) to
#   group_by() exactly as dplyr does, so `.drop = TRUE` drops empty factor
#   levels (conformance: test-group-nest.R "group_nest() respects .drop").
# - nest_join() keeps accepting a host `y` (with or without `copy = TRUE`);
#   the F05 copy rule for nest_join() was cut from F28 and moved to F49
#   (it would cost conformance test-join.R #6295 until rowwise() is GPU-backed).
# - Everything else (name default, keep, na_matches, unmatched, the `...`
#   check, by = NULL / character() / join_by()) keeps matching dplyr.
#
# All three verbs return host objects (tibble / rowwise_df / x's class with a
# list-column), never a tbl_gpu.

f28_fct <- function() {
  tibble::tibble(
    f = factor(c("b", "b", "d", NA), levels = c("a", "b", "c", "d")),
    g = c(1L, 2L, 1L, 1L),
    x = c(1.5, NA, 3.5, 4.5),
    s = c("p", "q", NA, "s")
  )
}

f28_types <- function() {
  tibble::tibble(
    k   = c(2L, 1L, NA, 2L, 3L, 1L),
    chr = c("a", "b", "a", NA, "c", "b"),
    lg  = c(TRUE, NA, FALSE, TRUE, FALSE, TRUE),
    dbl = c(1.5, 2.5, NA, 4.5, 5.5, 6.5),
    dt  = as.Date("2024-01-01") + c(0, 1, NA, 3, 4, 5),
    ts  = as.POSIXct("2024-01-01 12:00:00", tz = "UTC") + c(0, 60, 120, NA, 240, 300),
    fct = factor(c("u", "v", NA, "u", "w", "v"), levels = c("u", "v", "w", "z"))
  )
}

f28_x <- function() {
  tibble::tibble(
    k = c(1L, 2L, NA, 4L, 1L),
    a = c("p", "q", "r", "s", "t"),
    g = c(1L, 1L, 2L, 2L, 2L)
  )
}

f28_y <- function() {
  tibble::tibble(
    k  = c(1L, 1L, NA, 5L, 2L),
    b  = c(1.5, 2.5, 3.5, 4.5, NA),
    d  = as.Date("2024-01-01") + 0:4,
    ff = factor(c("m", "n", "m", NA, "n"), levels = c("m", "n", "o"))
  )
}

# Run both the eager and the lazy oracle comparison.
f28_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# A pipeline that nest_joins its input with `y`: a tbl_gpu copy of `y` on the
# GPU side (eager or lazy per `lazy_y`), the plain tibble on the oracle side.
# The variable is called `yy` on both sides, so the default `name` matches.
f28_nest_join_pipe <- function(y, ..., lazy_y = FALSE) {
  args <- list(...)
  function(t) {
    yy <- if (is_tbl_gpu(t)) tbl_gpu(y, lazy = lazy_y) else y
    do.call(dplyr::nest_join, c(list(t, quote(yy)), args))
  }
}

# --- group_nest(): .drop --------------------------------------------------------

test_that("group_nest() respects .drop = TRUE (conformance case)", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor("b", levels = c("a", "b", "c")), x = 1, y = 2)
  for (lazy in c(FALSE, TRUE)) {
    nested <- dplyr::group_nest(tbl_gpu(df, lazy = lazy), f, .drop = TRUE)
    expect_equal(nrow(nested), 1L)
  }
  f28_both(df, function(t) dplyr::group_nest(t, f, .drop = TRUE))
})

test_that("group_nest() .drop matches dplyr on factor, NA and multi-column keys", {
  skip_if_no_gpu()
  df <- f28_fct()
  f28_both(df, function(t) dplyr::group_nest(t, f, .drop = TRUE))
  f28_both(df, function(t) dplyr::group_nest(t, f, .drop = FALSE))
  f28_both(df, function(t) dplyr::group_nest(t, f))
  f28_both(df, function(t) dplyr::group_nest(t, f, g, .drop = TRUE))
  f28_both(df, function(t) dplyr::group_nest(t, f, g, .drop = FALSE))
  f28_both(df, function(t) dplyr::group_nest(t, f, .drop = TRUE, keep = TRUE))
})

test_that("group_nest() evaluates .drop in the caller's environment", {
  skip_if_no_gpu()
  df <- f28_fct()
  drop_it <- TRUE
  f28_both(df, function(t) dplyr::group_nest(t, f, .drop = drop_it))
  res <- dplyr::group_nest(tbl_gpu(df), f, .drop = drop_it)
  expect_equal(as.character(res$f), c("b", "d", NA))
})

test_that("group_nest() on grouped input keeps the grouping's .drop", {
  skip_if_no_gpu()
  df <- f28_fct()
  f28_both(df, function(t) dplyr::group_nest(dplyr::group_by(t, f, .drop = FALSE)))
  f28_both(df, function(t) dplyr::group_nest(dplyr::group_by(t, f, .drop = TRUE)))
  f28_both(df, function(t) dplyr::group_nest(dplyr::group_by(t, f, .drop = FALSE), keep = TRUE))
})

test_that("group_nest() options, computed keys, empty and ungrouped input match dplyr", {
  skip_if_no_gpu()
  df <- f28_types()
  f28_both(df, function(t) dplyr::group_nest(t, k))
  f28_both(df, function(t) dplyr::group_nest(t, chr, lg, .key = "rows"))
  f28_both(df, function(t) dplyr::group_nest(t, dt, keep = TRUE))
  f28_both(df, function(t) dplyr::group_nest(t, fct))
  f28_both(df, function(t) dplyr::group_nest(t, odd = k %% 2L))
  f28_both(df, function(t) dplyr::group_nest(t))
  f28_both(df[0, ], function(t) dplyr::group_nest(t, k))
  f28_both(df[0, ], function(t) dplyr::group_nest(t, fct, .drop = FALSE))
})

test_that("group_nest() returns a tibble whose list-column holds tibbles", {
  skip_if_no_gpu()
  res <- dplyr::group_nest(tbl_gpu(f28_fct()), f, .drop = TRUE)
  expect_false(is_tbl_gpu(res))
  expect_s3_class(res, "tbl_df")
  expect_s3_class(res$data, "vctrs_list_of")
  expect_s3_class(res$data[[1]], "tbl_df")
})

test_that("group_nest() on grouped input warns about `...` like dplyr", {
  skip_if_no_gpu()
  df <- f28_fct()
  want <- testthat::capture_warnings(
    want_res <- dplyr::group_nest(dplyr::group_by(df, f), g)
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- dplyr::group_by(tbl_gpu(df, lazy = lazy), f)
    got <- testthat::capture_warnings(res <- dplyr::group_nest(gt, g))
    expect_identical(got, want)
    expect_equal(res, want_res)
  }
})

# --- nest_by(): .drop -----------------------------------------------------------

test_that("nest_by() respects .drop = TRUE", {
  skip_if_no_gpu()
  df <- f28_fct()
  f28_both(df, function(t) dplyr::nest_by(t, f, .drop = TRUE))
  f28_both(df, function(t) dplyr::nest_by(t, f, g, .drop = TRUE))
  f28_both(df, function(t) dplyr::nest_by(t, f, .drop = FALSE))
  f28_both(df, function(t) dplyr::nest_by(t, f))
  res <- dplyr::nest_by(tbl_gpu(df), f, .drop = TRUE)
  expect_equal(as.character(res$f), c("b", "d", NA))
})

test_that("nest_by() options and grouped input match dplyr and return a rowwise_df", {
  skip_if_no_gpu()
  df <- f28_types()
  f28_both(df, function(t) dplyr::nest_by(t, k, .key = "rows", .keep = TRUE))
  f28_both(df, function(t) dplyr::nest_by(t, odd = k %% 2L))
  f28_both(df, function(t) dplyr::nest_by(dplyr::group_by(t, fct, .drop = FALSE)))
  f28_both(df[0, ], function(t) dplyr::nest_by(t, chr))
  for (lazy in c(FALSE, TRUE)) {
    res <- dplyr::nest_by(tbl_gpu(df, lazy = lazy), chr)
    expect_false(is_tbl_gpu(res))
    expect_s3_class(res, "rowwise_df")
    expect_identical(dplyr::group_vars(res), "chr")
  }
})

test_that("nest_by() on grouped input with `...` errors like dplyr", {
  skip_if_no_gpu()
  df <- f28_fct()
  want <- tryCatch(dplyr::nest_by(dplyr::group_by(df, f), g), error = identity)
  for (lazy in c(FALSE, TRUE)) {
    gt <- dplyr::group_by(tbl_gpu(df, lazy = lazy), f)
    got <- tryCatch(dplyr::nest_by(gt, g), error = identity)
    expect_s3_class(got, "rlang_error")
    expect_identical(conditionMessage(got), conditionMessage(want))
  }
})

# --- nest_join(): host y with copy = TRUE ----------------------------------------

test_that("nest_join() with copy = TRUE matches dplyr (eager and lazy x)", {
  skip_if_no_gpu()
  y <- f28_y()
  y_df <- as.data.frame(y)
  y0 <- y[0, ]
  f28_both(f28_x(), function(t) dplyr::nest_join(t, y, by = "k", copy = TRUE))
  # a base data.frame y nests base data frames, exactly as on host data
  f28_both(f28_x(), function(t) dplyr::nest_join(t, y_df, by = "k", copy = TRUE))
  f28_both(f28_x(), function(t) dplyr::nest_join(t, y0, by = "k", copy = TRUE))
  f28_both(f28_x(), function(t) {
    dplyr::nest_join(dplyr::group_by(t, g), y, by = "k", copy = TRUE)
  })
  res <- dplyr::nest_join(tbl_gpu(f28_x()), y, by = "k", copy = TRUE)
  expect_named(res, c("k", "a", "g", "y"))
  expect_false(is_tbl_gpu(res))
})

# --- nest_join(): tbl_gpu y keeps matching dplyr -------------------------------------

test_that("nest_join() with a tbl_gpu y matches dplyr (eager and lazy, both sides)", {
  skip_if_no_gpu()
  y <- f28_y()
  for (lazy_y in c(FALSE, TRUE)) {
    f28_both(f28_x(), f28_nest_join_pipe(y, by = "k", lazy_y = lazy_y))
    f28_both(f28_x(), f28_nest_join_pipe(y, by = "k", keep = TRUE, lazy_y = lazy_y))
    f28_both(f28_x(), f28_nest_join_pipe(y, by = "k", na_matches = "never", lazy_y = lazy_y))
    f28_both(f28_x(), f28_nest_join_pipe(y, by = "k", name = "m", lazy_y = lazy_y))
    f28_both(f28_x(), f28_nest_join_pipe(y, by = dplyr::join_by(k >= k), lazy_y = lazy_y))
    f28_both(f28_x(), f28_nest_join_pipe(y[0, ], by = "k", lazy_y = lazy_y))
  }
})

test_that("nest_join() default name comes from the y expression", {
  skip_if_no_gpu()
  gx <- tbl_gpu(f28_x())
  my_lookup <- tbl_gpu(f28_y())
  expect_named(dplyr::nest_join(gx, my_lookup, by = "k"), c("k", "a", "g", "my_lookup"))
  expect_named(
    dplyr::nest_join(gx, dplyr::filter(my_lookup, b > 2), by = "k"),
    c("k", "a", "g", "dplyr::filter(my_lookup, b > 2)")
  )
})

test_that("nest_join() keeps grouped x grouped and handles pending lazy ops on both sides", {
  skip_if_no_gpu()
  y <- f28_y()
  f28_both(f28_x(), function(t) {
    yy <- if (is_tbl_gpu(t)) dplyr::filter(tbl_gpu(y, lazy = TRUE), !is.na(b)) else dplyr::filter(y, !is.na(b))
    dplyr::nest_join(dplyr::group_by(dplyr::mutate(t, k2 = k), g), yy, by = "k")
  })
  res <- dplyr::nest_join(dplyr::group_by(tbl_gpu(f28_x()), g), tbl_gpu(y), by = "k")
  expect_s3_class(res, "grouped_df")
  expect_identical(dplyr::group_vars(res), "g")
})

test_that("nest_join() unmatched = 'error' and the `...` check raise dplyr's conditions", {
  skip_if_no_gpu()
  gy <- tbl_gpu(f28_y())
  for (lazy in c(FALSE, TRUE)) {
    gx <- tbl_gpu(f28_x(), lazy = lazy)
    expect_error(dplyr::nest_join(gx, gy, by = "k", unmatched = "error"),
                 class = "dplyr_error_join_matches_remaining")
    expect_error(dplyr::nest_join(gx, gy, by = "k", multiple = "first"),
                 class = "rlib_error_dots_nonempty")
  }
})

test_that("nest_join() by = NULL messages once; by = character() warns once", {
  skip_if_no_gpu()
  x <- f28_x()
  y <- f28_y()
  gx <- tbl_gpu(x)
  gy <- tbl_gpu(y)
  want <- testthat::capture_messages(dplyr::nest_join(x, y))
  got <- testthat::capture_messages(res <- dplyr::nest_join(gx, gy))
  expect_identical(got, want)
  expect_equal(unname(as.list(res)), unname(as.list(dplyr::nest_join(x, y))))

  got_w <- testthat::capture_warnings(res <- dplyr::nest_join(gx, gy, by = character()))
  expect_length(got_w, 1L)
  expect_match(got_w, "by = character()", fixed = TRUE)
  expect_equal(vapply(res$gy, nrow, integer(1)), rep(nrow(y), nrow(x)))
})

test_that("nest_join() nests every column type of y", {
  skip_if_no_gpu()
  ty <- f28_types()
  tx <- tibble::tibble(k = c(1L, 2L, 3L, NA, 7L), z = 1:5)
  for (lazy_y in c(FALSE, TRUE)) {
    f28_both(tx, f28_nest_join_pipe(ty, by = "k", lazy_y = lazy_y))
    f28_both(tx, f28_nest_join_pipe(ty, by = "k", keep = TRUE, lazy_y = lazy_y))
  }
  # keys of other types
  f28_both(ty, f28_nest_join_pipe(ty, by = c("chr", "dt")))
  f28_both(ty, f28_nest_join_pipe(ty, by = "fct"))
})
