# F12 acceptance tests: bind_rows()/bind_cols() contract (dplyr 1.2.1).
# Splicing with !!!, zero-column inputs, the first input's grouping, vctrs
# size rules in bind_cols(), and dplyr's error classes and calls.

f12_df <- function() {
  tibble::tibble(x = c(1L, NA, 3L), y = c("a", NA, "c"))
}

f12_typed <- function() {
  tibble::tibble(
    f = factor(c("u", NA, "v"), levels = c("v", "u", "w")),
    dt = as.Date(c("2020-01-01", NA, "2021-06-30")),
    l = c(TRUE, NA, FALSE),
    n = c(1.5, NA, 3)
  )
}

# A zero-column input of the same kind and height as `d` (a tibble for the
# dplyr oracle, an eager or lazy tbl_gpu otherwise). select() of no columns
# is not part of this feature, so the table is built directly.
f12_zero_cols <- function(d) {
  if (!is_tbl_gpu(d)) {
    return(tibble::tibble(.rows = nrow(d)))
  }
  out <- tbl_gpu(tibble::tibble(.rows = nrow(d)))
  if (identical(d$exec_mode, "lazy")) as_lazy(out) else out
}

# Both sides must fail with `class`; the messages must be identical unless
# `same_message = FALSE`. Returns the GPU-side condition.
f12_expect_same_error <- function(gpu, cpu, class, same_message = TRUE) {
  g <- rlang::catch_cnd(gpu, "error")
  c <- rlang::catch_cnd(cpu, "error")
  expect_s3_class(c, class)
  expect_s3_class(g, class)
  if (same_message && !is.null(g) && !is.null(c)) {
    expect_identical(conditionMessage(g), conditionMessage(c))
  }
  invisible(g)
}

# ---------------------------------------------------------------------------
# !!! splicing (gap A9): the generic must not force `!!!list` for dispatch
# ---------------------------------------------------------------------------

test_that("bind_rows(!!!list) splices tables (eager and lazy)", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(!!!list(d, d))
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
})

test_that("bind_rows(!!!named list, .id =) names the sources", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(!!!list(a = d, b = d), .id = "src")
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
})

test_that("bind_rows(!!!list) with a tibble first still returns a tbl_gpu", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f12_df())
  out <- bind_rows(!!!list(f12_df(), gt))
  expect_true(is_tbl_gpu(out))
  expect_identical(collect(out), dplyr::bind_rows(f12_df(), f12_df()))
})

test_that("bind_cols(!!!list) splices tables and vectors", {
  skip_if_no_gpu()
  pipe <- function(d) bind_cols(!!!list(d, tibble::tibble(z = c(1.5, 2.5, NA))))
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
  pipe2 <- function(d) bind_cols(!!!list(d, w = 7:9))
  expect_same_as_dplyr(f12_df(), pipe2)
})

test_that("bind_rows(!!!list) of plain tibbles still goes to dplyr", {
  skip_if_no_gpu()
  out <- bind_rows(!!!list(f12_df(), f12_df()))
  expect_identical(out, dplyr::bind_rows(f12_df(), f12_df()))
})

# ---------------------------------------------------------------------------
# Zero-column inputs in bind_rows(): their rows are NA-padded
# ---------------------------------------------------------------------------

test_that("bind_rows() pads a zero-column tibble's rows with NA", {
  skip_if_no_gpu()
  before <- function(d) bind_rows(tibble::tibble(.rows = 2), d)
  after <- function(d) bind_rows(d, tibble::tibble(.rows = 2))
  expect_same_as_dplyr(f12_df(), before)
  expect_same_as_dplyr_lazy(f12_df(), before)
  expect_same_as_dplyr(f12_df(), after)
  expect_same_as_dplyr_lazy(f12_df(), after)
})

test_that("bind_rows() pads a zero-column tbl_gpu's rows with NA", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(f12_zero_cols(d), d)
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
})

test_that("zero-column padding keeps factor levels, dates, logicals, doubles", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(tibble::tibble(.rows = 1), d, tibble::tibble(.rows = 2))
  expect_same_as_dplyr(f12_typed(), pipe)
  expect_same_as_dplyr_lazy(f12_typed(), pipe)
})

test_that("zero-column inputs with .id", {
  skip_if_no_gpu()
  mixed <- function(d) bind_rows(a = tibble::tibble(.rows = 2), b = d, .id = "id")
  expect_same_as_dplyr(f12_df(), mixed)
  expect_same_as_dplyr_lazy(f12_df(), mixed)
  only <- function(d) {
    bind_rows(f12_zero_cols(d), tibble::tibble(.rows = 1),
              .id = "id")
  }
  expect_same_as_dplyr(f12_df(), only)
  expect_same_as_dplyr_lazy(f12_df(), only)
})

test_that("zero-column, zero-row inputs and empty tables", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(tibble::tibble(.rows = 0), d)
  expect_same_as_dplyr(f12_df(), pipe)
  pad_empty <- function(d) bind_rows(tibble::tibble(.rows = 2), d)
  expect_same_as_dplyr(f12_df()[0, ], pad_empty)
  expect_same_as_dplyr_lazy(f12_df()[0, ], pad_empty)
})

# ---------------------------------------------------------------------------
# Grouping: the result keeps the FIRST input's groups (and .drop)
# ---------------------------------------------------------------------------

test_that("bind_rows() keeps the first input's groups", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(dplyr::group_by(d, y), d)
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
  single <- function(d) bind_rows(dplyr::group_by(d, y))
  expect_same_as_dplyr(f12_df(), single)
  expect_same_as_dplyr_lazy(f12_df(), single)
  in_list <- function(d) bind_rows(list(dplyr::group_by(d, y), d))
  expect_same_as_dplyr(f12_df(), in_list)
})

test_that("bind_rows() is ungrouped when only a later input is grouped", {
  skip_if_no_gpu()
  pipe <- function(d) bind_rows(d, dplyr::group_by(d, y))
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
})

test_that("bind_rows() keeps groups with .id and with zero-column inputs", {
  skip_if_no_gpu()
  with_id <- function(d) bind_rows(dplyr::group_by(d, y), d, .id = "src")
  expect_same_as_dplyr(f12_df(), with_id)
  expect_same_as_dplyr_lazy(f12_df(), with_id)
  pad <- function(d) bind_rows(dplyr::group_by(d, x, y), tibble::tibble(.rows = 1))
  expect_same_as_dplyr(f12_df(), pad)
})

test_that("bind_rows() keeps a grouped tibble's groups when a tbl_gpu follows", {
  skip_if_no_gpu()
  gd <- dplyr::group_by(f12_df(), y)
  pipe <- function(d) bind_rows(gd, d)
  expect_same_as_dplyr(f12_df(), pipe)
  expect_same_as_dplyr_lazy(f12_df(), pipe)
  out <- bind_rows(gd, tbl_gpu(f12_df()))
  expect_true(is_tbl_gpu(out))
})

test_that("bind_rows() keeps .drop = FALSE of the first input", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    e = 1,
    f = factor(c(1, 1, 2, 2), levels = 1:3),
    g = c(1, 1, 2, 2),
    x = c(1, 2, 1, 4)
  )
  cpu <- dplyr::bind_rows(
    dplyr::group_by(df, e, f, g, .drop = FALSE),
    dplyr::group_by(df, e, f, g, .drop = FALSE)
  )
  gg <- dplyr::group_by(tbl_gpu(df), e, f, g, .drop = FALSE)
  out <- bind_rows(gg, gg)
  expect_identical(dplyr::group_vars(out), c("e", "f", "g"))
  expect_false(dplyr::group_by_drop_default(out))
  expect_identical(dplyr::group_size(out), dplyr::group_size(cpu))

  lazy <- dplyr::group_by(as_lazy(tbl_gpu(df)), e, f, g, .drop = FALSE)
  out_lazy <- bind_rows(lazy, tbl_gpu(df))
  expect_false(dplyr::group_by_drop_default(out_lazy))
  expect_identical(dplyr::group_vars(out_lazy), c("e", "f", "g"))
})

# ---------------------------------------------------------------------------
# bind_cols(): vctrs size rules (size-1 inputs recycle, others must match)
# ---------------------------------------------------------------------------

test_that("bind_cols() recycles size-1 data frames and tables", {
  skip_if_no_gpu()
  df1_after <- function(d) bind_cols(d, tibble::tibble(z = 1))
  expect_same_as_dplyr(f12_df(), df1_after)
  expect_same_as_dplyr_lazy(f12_df(), df1_after)
  gpu1_first <- function(d) {
    bind_cols(dplyr::select(dplyr::slice_head(d, n = 1), z = x), d)
  }
  expect_same_as_dplyr(f12_df(), gpu1_first)
  expect_same_as_dplyr_lazy(f12_df(), gpu1_first)
  typed1 <- function(d) bind_cols(dplyr::slice_head(d, n = 1), w = 1:4)
  expect_same_as_dplyr(f12_typed(), typed1)
})

test_that("bind_cols() takes the size of a zero-column input", {
  skip_if_no_gpu()
  tib0 <- function(d) bind_cols(tibble::tibble(.rows = 3), d)
  expect_same_as_dplyr(f12_df(), tib0)
  expect_same_as_dplyr_lazy(f12_df(), tib0)
  gpu0 <- function(d) bind_cols(f12_zero_cols(d), d)
  expect_same_as_dplyr(f12_df(), gpu0)
  vec0 <- function(d) bind_cols(f12_zero_cols(d), z = 4:6)
  expect_same_as_dplyr(f12_df(), vec0)
})

test_that("bind_cols() recycles a length-1 vector to zero rows", {
  skip_if_no_gpu()
  pipe <- function(d) bind_cols(d, z = 1)
  expect_same_as_dplyr(f12_df()[0, ], pipe)
  expect_same_as_dplyr_lazy(f12_df()[0, ], pipe)
})

test_that("bind_cols() size mismatches raise vctrs' error", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f12_df())
  df <- f12_df()
  cls <- "vctrs_error_incompatible_size"

  f12_expect_same_error(
    bind_cols(gt, tibble::tibble(z = 1:2)),
    dplyr::bind_cols(df, tibble::tibble(z = 1:2)), cls
  )
  f12_expect_same_error(
    bind_cols(gt, tbl_gpu(tibble::tibble(z = 1:2))),
    dplyr::bind_cols(df, tibble::tibble(z = 1:2)), cls
  )
  f12_expect_same_error(
    bind_cols(gt, z = 1:2),
    dplyr::bind_cols(df, z = 1:2), cls
  )
  f12_expect_same_error(
    bind_cols(gt, gt, tibble::tibble(q = 1:2)),
    dplyr::bind_cols(df, df, tibble::tibble(q = 1:2)), cls
  )
  f12_expect_same_error(
    bind_cols(tibble::tibble(z = 1), gt, w = 1:2),
    dplyr::bind_cols(tibble::tibble(z = 1), df, w = 1:2), cls
  )
  f12_expect_same_error(
    bind_cols(tibble::tibble(.rows = 2), gt),
    dplyr::bind_cols(tibble::tibble(.rows = 2), df), cls
  )
  f12_expect_same_error(
    bind_cols(gt, tibble::tibble(z = integer())),
    dplyr::bind_cols(df, tibble::tibble(z = integer())), cls
  )
})

test_that("bind_cols() size check uses a lazy table's post-op row count", {
  skip_if_no_gpu()
  gl <- dplyr::filter(as_lazy(tbl_gpu(f12_df())), !is.na(x))
  cpu <- dplyr::filter(f12_df(), !is.na(x))
  f12_expect_same_error(
    bind_cols(gl, tibble::tibble(z = 1:3)),
    dplyr::bind_cols(cpu, tibble::tibble(z = 1:3)),
    "vctrs_error_incompatible_size"
  )
  out <- bind_cols(gl, tibble::tibble(z = 1:2))
  expect_identical(collect(out), dplyr::bind_cols(cpu, tibble::tibble(z = 1:2)))
})

# ---------------------------------------------------------------------------
# Errors: dplyr's classes, and the user's call (not an internal helper)
# ---------------------------------------------------------------------------

test_that("bind_rows(.id = 5) errors like dplyr, citing bind_rows()", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f12_df())
  g <- f12_expect_same_error(
    bind_rows(gt, gt, .id = 5),
    dplyr::bind_rows(f12_df(), f12_df(), .id = 5), "rlang_error"
  )
  expect_identical(g$call, quote(bind_rows(gt, gt, .id = 5)))
})

test_that("bind_rows() invalid inputs cite bind_rows()", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f12_df())
  g <- f12_expect_same_error(
    bind_rows(gt, 1:3),
    dplyr::bind_rows(f12_df(), 1:3), "rlang_error"
  )
  expect_identical(g$call, quote(bind_rows(gt, 1:3)))
  ll <- list(gt, rlang::env(a = 1))
  g2 <- f12_expect_same_error(
    bind_rows(ll),
    dplyr::bind_rows(list(f12_df(), rlang::env(a = 1))), "rlang_error"
  )
  expect_identical(g2$call, quote(bind_rows(ll)))
})

test_that("bind_rows() incompatible column types raise vctrs' error class", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f12_df())
  g <- f12_expect_same_error(
    bind_rows(gt, tibble::tibble(x = "a")),
    dplyr::bind_rows(f12_df(), tibble::tibble(x = "a")),
    "vctrs_error_incompatible_type"
  )
  expect_identical(g$call, quote(bind_rows(gt, tibble::tibble(x = "a"))))
  # The factor label carries a hash in vctrs, so only the class is compared.
  gf <- tbl_gpu(tibble::tibble(a = factor("a")))
  f12_expect_same_error(
    bind_rows(gf, tibble::tibble(a = 1L)),
    dplyr::bind_rows(tibble::tibble(a = factor("a")), tibble::tibble(a = 1L)),
    "vctrs_error_incompatible_type", same_message = FALSE
  )
})

test_that("bind_cols() size errors cite bind_cols()", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f12_df())
  g <- rlang::catch_cnd(bind_cols(gt, z = 1:2), "error")
  expect_s3_class(g, "vctrs_error_incompatible_size")
  expect_identical(g$call, quote(bind_cols(gt, z = 1:2)))
})
