# F83 attempt 2, acceptance: the three single-site fixes.
#  1. `%in%` whose set is not a literal runs in dplyr (it crashed the evaluator).
#  2. xor() of a non-logical operand runs in dplyr (a raw cudf error or a bitwise
#     result).
#  3. A literal-only op that warns in a screened verb call goes to dplyr, which
#     raises its own warning (ir_eval_pure_op() dropped it).

f83a2_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L),
    x = c(1, 2, 3, NA, 5, 6),
    i = c(1L, 2L, 3L, 4L, NA, 6L),
    b = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE),
    s = c("a", "b", "c", "a", NA, "b")
  )
}

f83a2_both <- function(df, p) {
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
}

f83a2_native <- function(df, p) {
  withr::local_options(cuplyr.fallback = "error")
  f83a2_both(df, p)
}

# --- 1. %in% ------------------------------------------------------------------

test_that("F83-a2: a number folded by a new name, left of %in% a column", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  pipes <- list(
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = NROW(1:3) %in% i),
    function(d) dplyr::filter(dplyr::group_by(d, g), gamma(3) %in% x),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(NROW(1:3) %in% i)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = sign(1) %in% i),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = choose(3, 1) %in% i),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = identical(1, 1) %in% b),
    function(d) dplyr::mutate(d, v = NROW(1:3) %in% i, .by = g),
    function(d) dplyr::filter(d, gamma(3) %in% x, .by = g)
  )
  for (p in pipes) {
    f83a2_both(df, p)
  }
})

test_that("F83-a2: a column set in the shapes that crash on dev", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  pipes <- list(
    function(d) dplyr::mutate(d, v = x %in% i),
    function(d) dplyr::mutate(d, v = 3L %in% i),
    function(d) dplyr::mutate(d, v = abs(3L) %in% i),
    function(d) dplyr::mutate(d, v = length(1:3) %in% i),
    function(d) dplyr::mutate(d, v = qnorm(0.5) %in% x),
    function(d) dplyr::filter(d, x %in% i),
    function(d) dplyr::filter(d, 3L %in% i),
    function(d) dplyr::summarise(d, v = sum(x %in% i)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = x %in% i),
    function(d) dplyr::filter(dplyr::group_by(d, g), x %in% i),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x %in% i)),
    # a set computed from a column
    function(d) dplyr::mutate(d, v = x %in% (i + 1L)),
    function(d) dplyr::filter(d, i %in% abs(i)),
    # strings, a logical, a mixed-type pair
    function(d) dplyr::mutate(d, v = s %in% s),
    function(d) dplyr::mutate(d, v = "a" %in% s),
    function(d) dplyr::mutate(d, v = b %in% b),
    function(d) dplyr::mutate(d, v = s %in% i)
  )
  for (p in pipes) {
    f83a2_both(df, p)
  }
})

test_that("F83-a2: a factor column on either side of a column set", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L),
    f = factor(c("a", "b", NA, "a")),
    h = factor(c("b", "b", "a", NA)),
    s = c("a", "z", "b", NA)
  )
  pipes <- list(
    function(d) dplyr::mutate(d, v = f %in% h),
    function(d) dplyr::mutate(d, v = f %in% s),
    function(d) dplyr::mutate(d, v = s %in% f),
    function(d) dplyr::filter(dplyr::group_by(d, g), f %in% h)
  )
  for (p in pipes) {
    f83a2_both(df, p)
  }
})

test_that("F83-a2: a column set on NA and zero-row inputs", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  all_na <- tibble::tibble(g = c(1L, 2L), x = c(NA_real_, NA_real_), i = c(NA_integer_, NA_integer_))
  empty <- all_na[0L, ]
  pipes <- list(
    function(d) dplyr::mutate(d, v = NA %in% i),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = x %in% i),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = NROW(1:3) %in% i),
    function(d) dplyr::filter(d, x %in% i),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x %in% i))
  )
  for (p in pipes) {
    f83a2_both(all_na, p)
    f83a2_both(empty, p)
  }
  f83a2_both(df, function(d) dplyr::mutate(d, v = NA %in% i))
})

test_that("F83-a2: a column set inside a longer pipeline", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  p <- function(d) {
    d <- dplyr::mutate(d, y = x + 1)
    d <- dplyr::filter(d, !(y %in% i))
    dplyr::arrange(dplyr::mutate(d, w = i * 2L), g, i)
  }
  f83a2_both(df, p)
})

test_that("F83-a2: a literal set stays native (guard)", {
  skip_if_no_gpu()
  df <- f83a2_df()
  pipes <- list(
    function(d) dplyr::mutate(d, v = x %in% c(1, 3)),
    function(d) dplyr::mutate(d, v = x %in% 3),
    function(d) dplyr::mutate(d, v = x %in% NA),
    function(d) dplyr::mutate(d, v = x %in% numeric(0)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = i %in% c(1L, 3L)),
    function(d) dplyr::filter(dplyr::group_by(d, g), s %in% c("a", "b")),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x %in% c(1, 3))),
    # a folded literal on the left of a literal set
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = NROW(1:3) %in% c(1L, 3L)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = gamma(3) %in% c(2, 3))
  )
  for (p in pipes) {
    f83a2_native(df, p)
  }
  fdf <- tibble::tibble(f = factor(c("a", "b", NA, "a")))
  f83a2_native(fdf, function(d) dplyr::mutate(d, v = f %in% c("a", "z")))
})

test_that("F83-a2: a column set records a reason for the fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  expect_error(
    dplyr::mutate(tbl_gpu(f83a2_df()), v = x %in% i),
    class = "cuplyr_error_fallback"
  )
})

# --- 2. xor -------------------------------------------------------------------

test_that("F83-a2: xor() with a non-logical operand equals dplyr", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  pipes <- list(
    # the review's crash
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = xor(x > 0, sign(1))),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = xor(sign(0), x > 0)),
    function(d) dplyr::filter(dplyr::group_by(d, g), xor(x > 2, sign(1))),
    function(d) dplyr::mutate(d, v = xor(x > 0, sign(1)), .by = g),
    # columns: bitwise on dev
    function(d) dplyr::mutate(d, v = xor(i, b)),
    function(d) dplyr::mutate(d, v = xor(b, i)),
    function(d) dplyr::mutate(d, v = xor(i, i)),
    function(d) dplyr::mutate(d, v = xor(x, b)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = xor(i, b)),
    function(d) dplyr::filter(d, xor(i, b)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(xor(i, b), na.rm = TRUE)),
    # literals of another type
    function(d) dplyr::mutate(d, v = xor(b, 1L)),
    function(d) dplyr::mutate(d, v = xor(b, 0)),
    function(d) dplyr::mutate(d, v = xor(b, NA_integer_)),
    # an arithmetic operand
    function(d) dplyr::mutate(d, v = xor(i - i, b))
  )
  for (p in pipes) {
    f83a2_both(df, p)
  }
})

test_that("F83-a2: xor() on NA and zero-row inputs", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  all_na <- tibble::tibble(g = c(1L, 2L), i = c(NA_integer_, NA_integer_), b = c(NA, NA))
  pipes <- list(
    function(d) dplyr::mutate(d, v = xor(i, b)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = xor(b, sign(1)))
  )
  for (p in pipes) {
    f83a2_both(all_na, p)
    f83a2_both(all_na[0L, ], p)
  }
})

test_that("F83-a2: xor() of a string keeps dplyr's error", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  expect_error(dplyr::mutate(df, v = xor(s, b)), "numeric, logical or complex")
  expect_error(dplyr::mutate(tbl_gpu(df, lazy = FALSE), v = xor(s, b)), "numeric, logical or complex")
  expect_error(
    dplyr::collect(dplyr::mutate(tbl_gpu(df, lazy = TRUE), v = xor(s, b))),
    "numeric, logical or complex"
  )
})

test_that("F83-a2: xor() of logicals stays native (guard)", {
  skip_if_no_gpu()
  df <- f83a2_df()
  pipes <- list(
    function(d) dplyr::mutate(d, v = xor(b, b)),
    function(d) dplyr::mutate(d, v = xor(b, TRUE)),
    function(d) dplyr::mutate(d, v = xor(x > 0, NA)),
    function(d) dplyr::mutate(d, v = xor(x > 0, i > 2L)),
    function(d) dplyr::mutate(d, v = xor(is.na(x), b)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = xor(x > 0, b)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = xor(x > 0, identical(1, 1))),
    function(d) dplyr::filter(dplyr::group_by(d, g), xor(x > 2, b)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(xor(x > 2, b), na.rm = TRUE))
  )
  for (p in pipes) {
    f83a2_native(df, p)
  }
})

# --- 3. a literal op that warns -----------------------------------------------

f83a2_warned <- function(df, p, pattern, nm) {
  want <- suppressWarnings(p(df))
  for (lazy in c(FALSE, TRUE)) {
    got <- NULL
    expect_warning(
      got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy))),
      pattern,
      class = "rlang_warning",
      label = nm
    )
    expect_equal(dplyr::ungroup(got), dplyr::ungroup(want), ignore_attr = TRUE, info = nm)
  }
}

test_that("F83-a2: a clean fold feeding an op that warns keeps dplyr's warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  pipes <- list(
    mutate = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * sqrt(qnorm(0.1))),
    filter = function(d) dplyr::filter(dplyr::group_by(d, g), x > sqrt(sign(-1))),
    summarise = function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * sqrt(qnorm(0.1)))),
    by = function(d) dplyr::mutate(d, v = x * log(sign(-1)), .by = g),
    # A written arithmetic literal takes the same path. (`sqrt(-1)` doesn't:
    # `-1` is a unary-minus call the fold leaves alone, and it stays a follow-up.)
    literal = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * sqrt(0 - 1))
  )
  for (nm in names(pipes)) {
    f83a2_warned(df, pipes[[nm]], "There were 3 warnings", nm)
  }
})

test_that("F83-a2: integer overflow between literals keeps dplyr's warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- f83a2_df()
  p <- function(d) dplyr::mutate(dplyr::group_by(d, g), v = i + (2147483647L + 1L))
  f83a2_warned(df, p, "There were 3 warnings", "overflow")
})

test_that("F83-a2: a clean fold into a registry op stays native (guard)", {
  skip_if_no_gpu()
  df <- f83a2_df()
  pipes <- list(
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * sqrt(qnorm(0.9))),
    function(d) dplyr::filter(dplyr::group_by(d, g), x > sqrt(sign(1))),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * sqrt(4), na.rm = TRUE)),
    function(d) dplyr::mutate(d, v = x * sqrt(4), .by = g)
  )
  for (p in pipes) {
    f83a2_native(df, p)
  }
})

test_that("F83-a2: an unscreened call parses as before (guard)", {
  skip_if_no_gpu()
  df <- f83a2_df()
  # Ungrouped mutate()/filter() are not screened: values equal dplyr's, and the
  # call stays on the GPU.
  withr::local_options(cuplyr.fallback = "error")
  got <- suppressWarnings(dplyr::collect(dplyr::mutate(tbl_gpu(df, lazy = FALSE), v = x * sqrt(-1))))
  want <- suppressWarnings(dplyr::mutate(df, v = x * sqrt(-1)))
  expect_equal(got, want, ignore_attr = TRUE)
})

test_that("F83-a2: ir_eval_pure_op() declines a warning only under the mark", {
  prev <- cuplyr:::fallback_reasons_open()
  on.exit(cuplyr:::fallback_reasons_close(prev), add = TRUE)
  # not marked: no value, no warning, not a decline
  expect_no_warning(got <- cuplyr:::ir_eval_pure_op("sqrt", list(-1)))
  expect_false(got$ok)
  expect_null(got$declined)
  cuplyr:::fallback_mark_screened()
  expect_no_warning(got <- cuplyr:::ir_eval_pure_op("sqrt", list(-1)))
  expect_false(got$ok)
  expect_true(isTRUE(got$declined))
  # a clean value and an error are as before, under the mark too
  expect_identical(cuplyr:::ir_eval_pure_op("sqrt", list(4))$value, 2)
  got <- cuplyr:::ir_eval_pure_op("+", list("a", 1))
  expect_false(got$ok)
  expect_null(got$declined)
})
