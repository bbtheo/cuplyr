# F03 acceptance tests: 2-d `[` indexing on tbl_gpu with tibble semantics.
#
# Each case runs the same function on a plain tibble (the tibble/dplyr oracle)
# and on a tbl_gpu, eager and lazy. A data-frame result must come back as a
# tbl_gpu with the oracle's group_vars() and match the oracle after collect();
# a vector result (drop = TRUE, one column) must be identical to the oracle's.

f03_df <- function() {
  data.frame(
    g = c("a", "b", "a", "c"),
    x = c(1L, NA, 3L, 4L),
    y = c("p", "q", NA, "s"),
    z = c(1.5, 2, NA, 4.5),
    f = factor(c("u", "v", "u", NA), levels = c("u", "v", "w")),
    d = as.Date("2024-01-01") + c(0L, 1L, NA, 3L),
    l = c(TRUE, NA, FALSE, TRUE),
    stringsAsFactors = FALSE
  )
}

f03_up <- function(df, lazy = FALSE) {
  g <- tbl_gpu(df)
  if (lazy) as_lazy(g) else g
}

# `prep` turns the uploaded/oracle table into the real input (e.g. group_by()).
f03_expect_same <- function(fn, df = f03_df(), lazy = FALSE, prep = identity) {
  label <- paste(if (lazy) "lazy" else "eager", ":", paste(deparse(body(fn)), collapse = " "))
  oracle <- fn(prep(tibble::as_tibble(df)))
  gpu <- fn(prep(f03_up(df, lazy)))

  if (is.data.frame(oracle)) {
    expect_true(is_tbl_gpu(gpu), info = paste(label, "- result should be a tbl_gpu"))
    if (!is_tbl_gpu(gpu)) {
      return(invisible())
    }
    expect_identical(dplyr::group_vars(gpu), dplyr::group_vars(oracle), info = label)
    expect_identical(
      tibble::as_tibble(collect(gpu)),
      tibble::as_tibble(dplyr::ungroup(oracle)),
      info = label
    )
  } else {
    expect_identical(gpu, oracle, info = label)
  }
}

f03_both <- function(fn, df = f03_df(), prep = identity) {
  f03_expect_same(fn, df, lazy = FALSE, prep = prep)
  f03_expect_same(fn, df, lazy = TRUE, prep = prep)
}

# The error tibble raises must be raised by tbl_gpu too: same class, and when
# raised at call time also the same message and call (tibble reports the
# user's subscript and the `[` call). Eager raises it at call time. Lazy
# raises it at call time when `lazy_at_call` is TRUE, and may defer it to
# collect() otherwise (row errors on a lazy table with pending ops, where
# nrow isn't known yet); a deferred error only has to match the class.
f03_expect_same_error <- function(fn, df = f03_df(), lazy_at_call = TRUE, prep = identity) {
  label <- paste(deparse(body(fn)), collapse = " ")
  oracle <- tryCatch(fn(prep(tibble::as_tibble(df))), error = function(e) e)
  expect_true(inherits(oracle, "error"), info = paste("oracle should error:", label))
  cls <- class(oracle)[[1]]

  same_as_oracle <- function(err, mode) {
    expect_s3_class(err, cls)
    expect_identical(conditionMessage(err), conditionMessage(oracle), info = paste(mode, label))
    expect_identical(deparse(conditionCall(err)), deparse(conditionCall(oracle)), info = paste(mode, label))
  }

  eager <- tryCatch(fn(prep(f03_up(df))), error = function(e) e)
  expect_true(inherits(eager, "error"), info = paste("eager should error at call time:", label))
  if (inherits(eager, "error")) same_as_oracle(eager, "eager:")

  if (lazy_at_call) {
    lazy <- tryCatch(fn(prep(f03_up(df, lazy = TRUE))), error = function(e) e)
    expect_true(inherits(lazy, "error"), info = paste("lazy should error at call time:", label))
    if (inherits(lazy, "error")) same_as_oracle(lazy, "lazy:")
  } else {
    expect_error(collect(fn(prep(f03_up(df, lazy = TRUE)))), class = cls, info = paste("lazy:", label))
  }
}

f03_no_transfer <- function(expr) {
  testthat::local_mocked_bindings(
    gpu_collect = function(...) stop("F03: unexpected gpu_collect()"),
    gpu_head = function(...) stop("F03: unexpected gpu_head()"),
    df_to_gpu = function(...) stop("F03: unexpected df_to_gpu()"),
    .package = "cuplyr"
  )
  force(expr)
}

# ---------------------------------------------------------------------------
# rows

test_that("x[i, ] with positive, duplicated and reordered positions", {
  skip_if_no_gpu()
  f03_both(function(t) t[1:2, ])
  f03_both(function(t) t[c(4, 1, 1), ])
  f03_both(function(t) t[c(2L, 1L), ])
  f03_both(function(t) t[c(0, 1), ])
  f03_both(function(t) t[3, ])
})

test_that("x[i, ] with negative positions drops rows, ignoring out-of-range ones", {
  skip_if_no_gpu()
  f03_both(function(t) t[-1, ])
  f03_both(function(t) t[-c(1, 3), ])
  f03_both(function(t) t[c(-4, -4), ])
  f03_both(function(t) t[-9, ])
  f03_both(function(t) t[-(1:4), ])
})

test_that("x[i, ] with an empty, zero or NULL index gives zero rows", {
  skip_if_no_gpu()
  f03_both(function(t) t[0, ])
  f03_both(function(t) t[integer(), ])
  f03_both(function(t) t[NULL, ])
  f03_both(function(t) t[FALSE, ])
})

test_that("x[i, ] with logical masks, recycled scalars and NA", {
  skip_if_no_gpu()
  f03_both(function(t) t[c(TRUE, FALSE, TRUE, FALSE), ])
  f03_both(function(t) t[TRUE, ])
  f03_both(function(t) t[NA, ])
  f03_both(function(t) t[c(TRUE, NA, FALSE, TRUE), ])
})

test_that("x[i, ] gives all-NA rows for NA and out-of-bounds positions, every column type", {
  skip_if_no_gpu()
  f03_both(function(t) t[c(1, NA), ])
  f03_both(function(t) t[NA_integer_, ])
  f03_both(function(t) t[7, ])
  f03_both(function(t) t[c(2, 9, 0, 3), ])
  f03_both(function(t) t[c(NA, 4, NA), ])
  f03_expect_same(function(t) t[2^31, ])
})

test_that("x[i, ] with a character index follows tibble (no row names)", {
  skip_if_no_gpu()
  f03_both(function(t) t[c("2", "a"), ])
  f03_both(function(t) t[c("0", " 2", "02", "-1", "1.5", "4", "9"), ])
})

test_that("x[i, ] on a zero-row table", {
  skip_if_no_gpu()
  df0 <- f03_df()[0, ]
  f03_both(function(t) t[1, ], df = df0)
  f03_both(function(t) t[c(1, NA), ], df = df0)
  f03_both(function(t) t[integer(), ], df = df0)
  f03_both(function(t) t[-1, ], df = df0)
  f03_both(function(t) t[, "y"], df = df0)
})

test_that("out-of-range non-whole or negative-with-NA rows follow tibble's fix_oob()", {
  skip_if_no_gpu()
  # tibble turns out-of-range elements into NA before validating, so these
  # are NA rows, not errors; validation must use the real row count
  f03_both(function(t) t[5.5, ])
  f03_both(function(t) t[c(-5, NA), ])
  f03_both(function(t) t[1.5, ], df = f03_df()[1, ])
  f03_expect_same_error(function(t) t[c(-5, 2), ])
  f03_expect_same_error(function(t) t[c(2.5, 9.5), ])
  f03_expect_same_error(function(t) t[c(5, -1), ])
})

test_that("lazy row subscripts after pending ops resolve against the post-op row count", {
  skip_if_no_gpu()
  # filter keeps 3 of the 4 rows
  after_filter <- function(t) dplyr::filter(t, x > 1L | is.na(x))
  f03_both(function(t) after_filter(t)[4.5, ])
  f03_both(function(t) after_filter(t)[c(-4, NA), ])
  f03_both(function(t) after_filter(t)[c(-3, -3), ])
  f03_both(function(t) after_filter(t)[NA, ])
  f03_expect_same_error(function(t) after_filter(t)[c(-4, 2), ], lazy_at_call = FALSE)
  f03_expect_same_error(function(t) after_filter(t)[c(TRUE, FALSE, TRUE, FALSE), ], lazy_at_call = FALSE)
})

# ---------------------------------------------------------------------------
# columns and combined

test_that("x[, j] and 1-d x[j] select columns by name, position, negation and mask", {
  skip_if_no_gpu()
  f03_both(function(t) t[, 1])
  f03_both(function(t) t[, "y"])
  f03_both(function(t) t[, c("z", "x")])
  f03_both(function(t) t[, -1])
  f03_both(function(t) t[, c(-1, -2)])
  f03_both(function(t) t[, c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE)])
  f03_both(function(t) t[, TRUE])
  f03_both(function(t) t[c("y", "x")])
  f03_both(function(t) t[2])
  f03_both(function(t) t[-1])
})

test_that("x[], x[, ] return the whole table", {
  skip_if_no_gpu()
  f03_both(function(t) t[])
  f03_both(function(t) t[, ])
})

test_that("x[i, j] combines row and column selection", {
  skip_if_no_gpu()
  f03_both(function(t) t[1:2, "y"])
  f03_both(function(t) t[c(3, 1), c("d", "f")])
  f03_both(function(t) t[-1, -1])
  f03_both(function(t) t[c(TRUE, FALSE, TRUE, FALSE), 2:3])
  f03_both(function(t) t[c(1, NA), c("f", "d", "l", "y")])
  f03_both(function(t) t[c(1, 1), "x"])
})

# ---------------------------------------------------------------------------
# drop

test_that("drop = TRUE with one column returns the column vector", {
  skip_if_no_gpu()
  f03_both(function(t) t[, 1, drop = TRUE])
  f03_both(function(t) t[1:2, "z", drop = TRUE])
  f03_both(function(t) t[2, "f", drop = TRUE])
  f03_both(function(t) t[c(1, NA), "d", drop = TRUE])
  f03_both(function(t) t[c(4, 2), "y", drop = TRUE])
  f03_both(function(t) t[-1, "l", drop = TRUE])
})

test_that("drop = TRUE with several columns is ignored, drop = FALSE keeps a table", {
  skip_if_no_gpu()
  f03_both(function(t) t[, 1:2, drop = TRUE])
  f03_both(function(t) t[1, , drop = TRUE])
  f03_both(function(t) t[, "y", drop = FALSE])
  f03_both(function(t) t[c(1, 2), 1, drop = FALSE])
})

test_that("drop= in the 1-d form x[j] warns like tibble and is ignored", {
  skip_if_no_gpu()
  df <- f03_df()
  w_oracle <- NULL
  oracle <- withCallingHandlers(
    tibble::as_tibble(df)[c(1, 2), drop = TRUE],
    warning = function(w) { w_oracle <<- w; invokeRestart("muffleWarning") }
  )
  expect_s3_class(w_oracle, "rlang_warning")
  for (lazy in c(FALSE, TRUE)) {
    w_gpu <- NULL
    gpu <- withCallingHandlers(
      f03_up(df, lazy)[c(1, 2), drop = TRUE],
      warning = function(w) { w_gpu <<- w; invokeRestart("muffleWarning") }
    )
    expect_s3_class(w_gpu, "rlang_warning")
    expect_identical(conditionMessage(w_gpu), conditionMessage(w_oracle))
    expect_identical(tibble::as_tibble(collect(gpu)), oracle)
  }
})

# ---------------------------------------------------------------------------
# grouped input

test_that("grouped x[i, ] keeps the groups and regroups the surviving rows", {
  skip_if_no_gpu()
  by_g <- function(t) dplyr::group_by(t, g)
  f03_both(function(t) t[c(1, 3), ], prep = by_g)
  f03_both(function(t) t[c(4, 1), ], prep = by_g)
  f03_both(function(t) t[-1, ], prep = by_g)
  f03_both(function(t) t[c(1, NA), ], prep = by_g)
  f03_both(function(t) t[0, ], prep = by_g)
  f03_both(function(t) t[c(TRUE, FALSE, TRUE, TRUE), ], prep = by_g)

  gpu <- dplyr::group_by(f03_up(f03_df()), g)[c(1, 3), ]
  oracle <- dplyr::group_by(tibble::as_tibble(f03_df()), g)[c(1, 3), ]
  expect_identical(dplyr::n_groups(gpu), dplyr::n_groups(oracle))
})

test_that("grouped x[, j] keeps only the group columns still present", {
  skip_if_no_gpu()
  by_g <- function(t) dplyr::group_by(t, g)
  f03_both(function(t) t[, "z"], prep = by_g)
  f03_both(function(t) t[, c("g", "z")], prep = by_g)
  f03_both(function(t) t[1:2, c("z", "g")], prep = by_g)
  f03_both(function(t) t["z"], prep = by_g)

  by_gl <- function(t) dplyr::group_by(t, g, l)
  f03_both(function(t) t[, c("l", "z")], prep = by_gl)
  f03_both(function(t) t[, c("l", "g")], prep = by_gl)
  f03_both(function(t) t[2:3, c("z", "l", "g")], prep = by_gl)
})

test_that("grouped x[i, j, drop = TRUE] ungroups a multi-column result, pulls a single one", {
  skip_if_no_gpu()
  by_g <- function(t) dplyr::group_by(t, g)
  f03_both(function(t) t[, c("g", "z"), drop = TRUE], prep = by_g)
  f03_both(function(t) t[1:2, c("g", "x"), drop = TRUE], prep = by_g)
  f03_both(function(t) t[, "z", drop = TRUE], prep = by_g)
})

test_that("grouped 1-d x[j, drop = TRUE] warns and ungroups, like dplyr", {
  skip_if_no_gpu()
  by_g <- function(t) dplyr::group_by(t, g)
  quiet <- function(fn) function(t) suppressWarnings(fn(t))
  f03_both(quiet(function(t) t[c("g", "z"), drop = TRUE]), prep = by_g)
  f03_both(quiet(function(t) t["g", drop = TRUE]), prep = by_g)
  f03_both(quiet(function(t) t[drop = TRUE]), prep = by_g)
  f03_both(quiet(function(t) t[c("g", "z"), drop = FALSE]), prep = by_g)
})

test_that("grouped x[i, ] preserves .drop = FALSE", {
  skip_if_no_gpu()
  df <- f03_df()
  oracle <- dplyr::group_by(tibble::as_tibble(df), g, .drop = FALSE)[1:2, ]
  for (lazy in c(FALSE, TRUE)) {
    gpu <- dplyr::group_by(f03_up(df, lazy), g, .drop = FALSE)[1:2, ]
    expect_identical(dplyr::group_by_drop_default(gpu), dplyr::group_by_drop_default(oracle))
    expect_identical(dplyr::group_vars(gpu), dplyr::group_vars(oracle))
  }
})

# ---------------------------------------------------------------------------
# errors

test_that("invalid row subscripts raise tibble's error classes", {
  skip_if_no_gpu()
  f03_expect_same_error(function(t) t[c(-1, 2), ])
  f03_expect_same_error(function(t) t[1.5, ])
  f03_expect_same_error(function(t) t[c(-1, NA), ])
  # The logical-length check needs nrow, so lazy may raise it at collect().
  f03_expect_same_error(function(t) t[c(TRUE, FALSE), ], lazy_at_call = FALSE)
  f03_expect_same_error(function(t) t[c(TRUE, FALSE, TRUE, FALSE, TRUE), ], lazy_at_call = FALSE)
})

test_that("invalid column subscripts raise tibble's error classes", {
  skip_if_no_gpu()
  f03_expect_same_error(function(t) t[, "w"])
  f03_expect_same_error(function(t) t[, 99])
  f03_expect_same_error(function(t) t[, -99])
  f03_expect_same_error(function(t) t[, NA])
  f03_expect_same_error(function(t) t[, c(1, NA)])
  f03_expect_same_error(function(t) t[, c(-1, 1)])
  f03_expect_same_error(function(t) t[, 1.5])
  f03_expect_same_error(function(t) t[, c(TRUE, FALSE)])
  f03_expect_same_error(function(t) t["w"])
  f03_expect_same_error(function(t) t[1:2, "w"])
  # tibble resolves j before i
  f03_expect_same_error(function(t) t[c(TRUE, FALSE), "w"])
})

test_that("unsupported shapes error loudly instead of returning something wrong", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    gt <- f03_up(f03_df(), lazy)
    # zero-column results: tbl_gpu can't hold a 0-column table
    expect_error(collect(gt[, 0]))
    expect_error(collect(gt[character()]))
    # duplicated columns: tbl_gpu needs unique names
    expect_error(collect(gt[, c("x", "x")]))
    # matrix subscript
    expect_error(collect(gt[matrix(TRUE, 4, 7)]))
  }
  gt <- f03_up(f03_df())
  expect_error(gt[1, "x"] <- 5L)
})

# ---------------------------------------------------------------------------
# laziness and GPU residency

test_that("x[i, j] on a lazy table stays lazy and resolves rows after pending ops", {
  skip_if_no_gpu()
  df <- f03_df()
  lz <- f03_up(df, lazy = TRUE)
  out <- lz[c(1, NA), c("y", "x")]
  expect_true(is_tbl_gpu(out))
  expect_identical(out$exec_mode, "lazy")
  expect_true(has_pending_ops(out))

  after_filter <- function(t) dplyr::filter(t, x > 1L | is.na(x))
  f03_both(function(t) after_filter(t)[-1, ])
  f03_both(function(t) after_filter(t)[c(TRUE, FALSE, TRUE), ])
  f03_both(function(t) after_filter(t)[c(3, 5), "z"])
  f03_both(function(t) dplyr::mutate(t[c(4, 2, NA), ], w = z * 2))
  f03_both(function(t) dplyr::arrange(t, z)[1:2, c("z", "g")])
  # grouped summarise row order is unspecified on the GPU, so sort it
  f03_both(function(t) dplyr::arrange(dplyr::summarise(dplyr::group_by(t[-1, ], g), n = dplyr::n()), g))
})

test_that("eager row gathers stay on the GPU", {
  skip_if_no_gpu()
  gt <- f03_up(f03_df())
  out <- f03_no_transfer(gt[c(2, NA, 1), c("y", "f")])
  expect_true(is_tbl_gpu(out))
  expect_identical(
    tibble::as_tibble(collect(out)),
    tibble::as_tibble(f03_df())[c(2, NA, 1), c("y", "f")]
  )
})
