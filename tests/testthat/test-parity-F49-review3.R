# F49 acceptance tests added after review round 3 (blocking findings B6-B7,
# scratchpad/parity/features/F49/reviews/round-3.md). dplyr 1.2.1 behaviour
# pinned with tools/r on plain tibbles (runs/review-3/probe1.log, probe3.log).
#
# Lazy contract, as in test-parity-F49-rowwise-rewrites.R: a lazy plan has a
# fixed schema, so where dplyr's result type depends on the data, cuplyr may
# fail loudly (cuplyr_error_sum_overflow / cuplyr_error_rowwise_retype)
# instead. A wrong type, or a silently different result, is never allowed.

f49r3_col_types <- function(x) {
  vapply(x, function(col) paste(c(class(col), typeof(col)), collapse = "/"), character(1))
}

f49r3_same <- function(got, oracle, label) {
  expect_identical(class(got), class(oracle), info = label)
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(oracle), info = label)
  expect_identical(f49r3_col_types(got), f49r3_col_types(oracle), info = label)
  expect_equal(got, oracle, info = label)
}

# Runs a lazy pipeline: dplyr's result (by any route), or one of the designed
# loud failures. Returns TRUE when it produced a result.
f49r3_lazy_ok <- function(df, pipeline, oracle, label) {
  res <- tryCatch(
    suppressWarnings(collect(pipeline(tbl_gpu(df, lazy = TRUE)))),
    cuplyr_error_sum_overflow = function(e) "loud",
    cuplyr_error_rowwise_retype = function(e) "loud"
  )
  if (identical(res, "loud")) {
    expect_true(TRUE)
    return(invisible(FALSE))
  }
  f49r3_same(res, oracle, label)
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# B6. A filter after a rowwise integer aggregate must not change the rows the
# aggregate sees (dplyr computes every row first).
# ---------------------------------------------------------------------------

test_that("an integer row sum that overflows stays dplyr's double (or fails loudly) when a filter follows", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  withr::local_options(cuplyr.fallback = "silent")
  df <- tibble::tibble(a = c(1L, .Machine$integer.max, 3L), b = c(2L, 1L, 4L))
  pipes <- list(
    sum_c = function(d) d |> dplyr::rowwise() |> dplyr::mutate(s = sum(c(a, b))) |> dplyr::filter(a < 5L),
    sum_c_across = function(d) {
      d |> dplyr::rowwise() |> dplyr::mutate(s = sum(dplyr::c_across(a:b))) |> dplyr::filter(a < 5L)
    },
    sum_args_filter_out = function(d) {
      d |> dplyr::rowwise() |> dplyr::mutate(s = sum(a, b)) |> dplyr::filter_out(a > 5L)
    }
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    oracle <- p(df)
    expect_type(oracle$s, "double")
    f49r3_same(collect(p(tbl_gpu(df))), oracle, paste(nm, "eager"))
    f49r3_lazy_ok(df, p, oracle, paste(nm, "lazy"))
  }
})

test_that("a lazy integer row aggregate followed by a filter stays integer when nothing overflows", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, 7L, 3L, NA), b = c(2L, 1L, 4L, 5L))
  pipes <- list(
    sum = function(d) d |> dplyr::rowwise() |> dplyr::mutate(s = sum(c(a, b))) |> dplyr::filter(b < 5L),
    max = function(d) d |> dplyr::rowwise() |> dplyr::mutate(s = max(dplyr::c_across(a:b))) |> dplyr::filter(b < 5L),
    # the filter removes every row: dplyr still computed the max on 4 rows,
    # so the column is integer; no zero-row rule may fire here
    max_empty = function(d) d |> dplyr::rowwise() |> dplyr::mutate(s = max(c(a, b))) |> dplyr::filter(b > 100L)
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    oracle <- p(df)
    expect_type(oracle$s, "integer")
    withr::with_options(list(cuplyr.fallback = "silent"), {
      f49r3_same(collect(p(tbl_gpu(df, lazy = FALSE))), oracle, paste(nm, "eager"))
      f49r3_same(collect(p(tbl_gpu(df, lazy = TRUE))), oracle, paste(nm, "lazy"))
    })
  }
  # the plain integer case stays native in lazy mode
  withr::with_options(list(cuplyr.fallback = "error"), {
    got <- collect(pipes$sum(tbl_gpu(df, lazy = TRUE)))
  })
  f49r3_same(got, pipes$sum(df), "sum lazy native")
})

# ---------------------------------------------------------------------------
# B7. Zero-row rowwise min()/max(): dplyr evaluates the expression once on an
# empty group, so an integer/logical min/max is double (+-Inf) with base R's
# warning.
# ---------------------------------------------------------------------------

test_that("rowwise min()/max() on a zero-row table have dplyr's type and warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  withr::local_options(cuplyr.fallback = "silent", dplyr.summarise.inform = FALSE)
  df0 <- tibble::tibble(a = integer(), b = integer(), l = logical(), d = double())
  pipes <- list(
    max_c_int = function(x) dplyr::mutate(dplyr::rowwise(x), m = max(c(a, b))),
    min_c_int_narm = function(x) dplyr::mutate(dplyr::rowwise(x), m = min(c(a, b), na.rm = TRUE)),
    max_single_int = function(x) dplyr::mutate(dplyr::rowwise(x), m = max(a)),
    min_single_lgl = function(x) dplyr::mutate(dplyr::rowwise(x), m = min(l)),
    max_cacross_int = function(x) dplyr::mutate(dplyr::rowwise(x), m = max(dplyr::c_across(a:b))),
    max_args_lgl_narm = function(x) dplyr::mutate(dplyr::rowwise(x), m = max(l, a, na.rm = TRUE)),
    max_c_int_dbl = function(x) dplyr::mutate(dplyr::rowwise(x), m = max(c(a, d))),
    summarise_max_c_int = function(x) dplyr::summarise(dplyr::rowwise(x), m = max(c(a, b)), .groups = "drop"),
    summarise_min_cacross = function(x) dplyr::summarise(dplyr::rowwise(x), m = min(dplyr::c_across(a:l)), .groups = "drop")
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    oracle <- suppressWarnings(p(df0))
    expect_type(oracle$m, "double")
    expect_warning(got <- collect(p(tbl_gpu(df0))), "no non-missing arguments to m(in|ax)", info = nm)
    f49r3_same(got, oracle, paste(nm, "eager"))
    f49r3_lazy_ok(df0, p, oracle, paste(nm, "lazy"))
  }
  # sum() of a zero-row integer table stays integer (dplyr: sum(integer()) is 0L)
  ps <- function(x) dplyr::mutate(dplyr::rowwise(x), s = sum(c(a, b)))
  f49r3_same(collect(ps(tbl_gpu(df0))), ps(df0), "sum eager")
  f49r3_same(collect(ps(tbl_gpu(df0, lazy = TRUE))), ps(df0), "sum lazy")
})
