# F83 round 2: the screened-verb mark, the true decline reasons, and the shapes
# the acceptance file leaves out.

f83x_df <- function() {
  tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4), s = c("a", "b", "c", "d"))
}

# The CPU-fallback reasons `expr` signals (warnings muffled)
f83x_reasons <- function(expr) {
  got <- list()
  tryCatch(
    withCallingHandlers(
      expr,
      cuplyr_cpu_fallback = function(c) got[[length(got) + 1L]] <<- c,
      warning = function(w) invokeRestart("muffleWarning")
    ),
    error = function(e) NULL # (dplyr's own error: the reason was signalled before it)
  )
  vapply(got, function(c) c$reason, "")
}

test_that("F83: an ungrouped fold that warns keeps one bare warning, as on dev", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    t <- tbl_gpu(f83x_df(), lazy = lazy)
    expect_warning(
      out <- dplyr::collect(dplyr::mutate(t, v = x * qnorm(2))),
      "NaNs produced",
      class = "simpleWarning"
    )
    expect_equal(out$v, rep(NA_real_, 4L))
  }
})

test_that("F83: the screened mark ends with the call", {
  skip_if_no_gpu()
  expect_false(cuplyr:::fallback_is_screened())
  t <- tbl_gpu(f83x_df())
  invisible(suppressWarnings(dplyr::mutate(dplyr::group_by(t, g), v = x * qnorm(2))))
  expect_false(cuplyr:::fallback_is_screened())
  invisible(dplyr::summarise(t, v = sum(x)))
  expect_false(cuplyr:::fallback_is_screened())
})

test_that("F83: a declined fold says why", {
  skip_if_no_gpu()
  t <- dplyr::group_by(tbl_gpu(f83x_df()), g)
  declined <- "raised a warning"
  expect_match(f83x_reasons(dplyr::collect(dplyr::mutate(t, v = x * qnorm(2)))), declined)
  expect_match(f83x_reasons(dplyr::collect(dplyr::mutate(t, v = qnorm(2)))), declined)
  expect_match(f83x_reasons(dplyr::collect(dplyr::filter(t, x > qnorm(2)))), declined)
  expect_match(f83x_reasons(dplyr::collect(dplyr::filter(t, qnorm(2) > 0))), declined)
  expect_match(
    f83x_reasons(dplyr::collect(dplyr::summarise(t, v = sum(x * dnorm(c(0, 1)))))),
    "vector constant"
  )
  expect_match(
    f83x_reasons(dplyr::collect(dplyr::mutate(t, v = x + class(1)))),
    "`class(1)` is not known to be pure",
    fixed = TRUE
  )
})

test_that("F83: class() is off the list: the parsers with their own dispatch give dplyr's error", {
  skip_if_no_gpu()
  df <- f83x_df()
  withr::local_options(cuplyr.fallback = "silent")
  g <- function(d) dplyr::group_by(d, g)
  pipes <- list(
    function(d) dplyr::mutate(g(d), v = dplyr::case_match(x, 1 ~ class(1), .default = x)),
    function(d) dplyr::mutate(g(d), v = dplyr::case_match(x, class(1) ~ 1, .default = x)),
    function(d) dplyr::mutate(g(d), v = dplyr::replace_when(x, x > 1 ~ class(1))),
    function(d) dplyr::mutate(g(d), v = dplyr::replace_values(x, 1 ~ class(1))),
    function(d) dplyr::mutate(g(d), v = dplyr::when_any(x > 0, class(1))),
    function(d) dplyr::mutate(g(d), v = dplyr::lag(x, default = class(1)))
  )
  for (p in pipes) {
    want <- tryCatch(p(df), error = function(e) e)
    got <- tryCatch(dplyr::collect(p(tbl_gpu(df))), error = function(e) e)
    expect_s3_class(want, "error")
    expect_identical(class(got)[[1L]], class(want)[[1L]])
  }
})

test_that("F83: a string through {{ }} or a variable stays native", {
  skip_if_no_gpu()
  df <- f83x_df()
  wrap <- function(d, a) dplyr::mutate(d, v = dplyr::if_else(x > 0, {{ a }}, "b"))
  lab <- "lit"
  withr::local_options(cuplyr.fallback = "error")
  expect_no_error(dplyr::collect(wrap(tbl_gpu(df), "lit")))
  expect_no_error(dplyr::collect(dplyr::mutate(tbl_gpu(df), v = dplyr::if_else(x > 0, lab, "b"))))
})

test_that("F83: an ungrouped top_n() with a column-free wt that warns keeps dplyr's warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  expect_warning(
    dplyr::collect(dplyr::top_n(tbl_gpu(f83x_df()), 2, wt = qnorm(2))),
    "There was 1 warning",
    class = "rlang_warning"
  )
})

test_that("F83: only a top_n_rank() call marks an ungrouped filter()", {
  skip_if_no_gpu()
  t <- tbl_gpu(f83x_df())
  prev <- cuplyr:::fallback_reasons_open()
  on.exit(cuplyr:::fallback_reasons_close(prev), add = TRUE)
  cuplyr:::top_n_rank_mark_screened(rlang::quos(base::is.na(x * qnorm(0.5))), t)
  expect_false(cuplyr:::fallback_is_screened())
})

test_that("F83: an impure n leaves an ungrouped top_n() evaluated once", {
  skip_if_no_gpu()
  count <- 0L
  n_fn <- function() {
    count <<- count + 1L
    2
  }
  withr::local_options(cuplyr.fallback = "silent")
  res <- dplyr::collect(dplyr::top_n(tbl_gpu(f83x_df()), n_fn(), x))
  expect_identical(count, 1L)
  expect_equal(nrow(res), 2L)
})

test_that("F83: string arithmetic: a column or a written string keeps the GPU error", {
  skip_if_no_gpu()
  t <- tbl_gpu(f83x_df())
  expect_error(dplyr::mutate(t, v = s + 1), "does not support arithmetic on STRING")
  expect_error(dplyr::mutate(t, v = x + "ten"), "does not support arithmetic on STRING")
})

test_that("F83: ir_fold_eval() declines a plain warning only under the mark", {
  prev <- cuplyr:::fallback_reasons_open()
  on.exit(cuplyr:::fallback_reasons_close(prev), add = TRUE)
  plain <- function() {
    warning("plain")
    1
  }
  classed <- function() {
    rlang::warn("classed", class = "f83x_warning")
    1
  }
  # not marked: the fold keeps its value, the warning goes on
  expect_warning(got <- cuplyr:::ir_fold_eval(plain()), "plain")
  expect_identical(got$value, 1)
  cuplyr:::fallback_mark_screened()
  expect_true(cuplyr:::fallback_is_screened())
  expect_identical(cuplyr:::ir_fold_eval(plain()), list(declined = TRUE))
  # a classed warning passes through, an error is no value
  expect_warning(got <- cuplyr:::ir_fold_eval(classed()), class = "f83x_warning")
  expect_identical(got$value, 1)
  expect_null(cuplyr:::ir_fold_eval(stop("boom")))
  expect_identical(names(cuplyr:::ir_fold_eval(stop("boom"), keep_error = TRUE)), "error")
})
