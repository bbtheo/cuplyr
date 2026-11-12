# F47 acceptance tests: dplyr::bind_rows() / dplyr::bind_cols() /
# purrr::map_dfr() over tbl_gpu inputs (e.g. group_split() pieces) bind on the
# GPU and give exactly what cuplyr::bind_rows() / cuplyr::bind_cols() give.
# purrr::list_rbind() rejects a non-data.frame element before any cuplyr code
# runs (is.data.frame() does not dispatch), so it stays unsupported, as do
# direct vctrs::vec_rbind() / vec_cbind() calls.

f47_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 3L, 3L, 3L),
    x = c(1.5, NA, 3, 4, -2, 0),
    s = c("a", NA, "c", "d", "e", "a"),
    f = factor(c("u", "v", NA, "u", "w", "v"), levels = c("w", "v", "u")),
    d = as.Date(c("2020-01-01", NA, "2021-06-30", "1999-12-31", "2000-02-29", "2024-01-01")),
    t = as.POSIXct(c(0, 3600, NA, 86400, 1e9, 1.5e9), origin = "1970-01-01", tz = "UTC"),
    b = c(TRUE, NA, FALSE, TRUE, FALSE, NA)
  )
}

f47_mode <- function(x) .subset2(x, "exec_mode") %||% "eager"

# `via` (a dplyr/purrr binder over tbl_gpu inputs) must give the same tbl_gpu
# as `ref` (the cuplyr binder): class, rows, column types, groups, mode.
f47_expect_same_as_cuplyr <- function(via, ref) {
  expect_true(is_tbl_gpu(via))
  expect_true(is_tbl_gpu(ref))
  expect_identical(class(via), class(ref))
  expect_identical(dplyr::group_vars(via), dplyr::group_vars(ref))
  expect_identical(f47_mode(via), f47_mode(ref))
  expect_identical(collect(via), collect(ref))
}

# Both must fail with the same class and message (the call may differ).
f47_expect_same_error_as_cuplyr <- function(via, ref, class) {
  v <- rlang::catch_cnd(via, "error")
  r <- rlang::catch_cnd(ref, "error")
  expect_s3_class(r, class)
  expect_s3_class(v, class)
  if (!is.null(v) && !is.null(r)) {
    expect_identical(conditionMessage(v), conditionMessage(r))
  }
  invisible(v)
}

# Run `expr` with every host download path disabled.
f47_no_download <- function(expr) {
  testthat::local_mocked_bindings(
    gpu_collect = function(...) stop("F47: a tbl_gpu was downloaded"),
    gpu_head = function(...) stop("F47: a tbl_gpu was downloaded"),
    .package = "cuplyr"
  )
  force(expr)
}

# ---------------------------------------------------------------------------
# group_split() -> bind round trip against real dplyr
# ---------------------------------------------------------------------------

test_that("dplyr::bind_rows(group_split()) matches dplyr (eager and lazy)", {
  skip_if_no_gpu()
  pipe <- function(d) dplyr::bind_rows(dplyr::group_split(dplyr::group_by(d, g)))
  expect_same_as_dplyr(f47_df(), pipe)
  expect_same_as_dplyr_lazy(f47_df(), pipe)
})

test_that("dplyr::bind_rows(group_split(), .id =) matches dplyr", {
  skip_if_no_gpu()
  pipe <- function(d) dplyr::bind_rows(dplyr::group_split(d, g), .id = "piece")
  expect_same_as_dplyr(f47_df(), pipe)
  expect_same_as_dplyr_lazy(f47_df(), pipe)
  named <- function(d) {
    p <- dplyr::group_split(d, g)
    dplyr::bind_rows(stats::setNames(as.list(p), c("one", "two", "three")), .id = "k")
  }
  expect_same_as_dplyr(f47_df(), named)
})

test_that("purrr::map_dfr() over group_split() pieces matches dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("purrr")
  pipe <- function(d) {
    purrr::map_dfr(dplyr::group_split(d, g), function(p) dplyr::mutate(p, y = x * 2))
  }
  expect_same_as_dplyr(f47_df(), pipe)
  expect_same_as_dplyr_lazy(f47_df(), pipe)
  with_id <- function(d) {
    p <- dplyr::group_split(d, g)
    purrr::map_dfr(stats::setNames(as.list(p), c("a", "b", "c")),
                   function(q) dplyr::select(q, g, x), .id = "src")
  }
  expect_same_as_dplyr(f47_df(), with_id)
  expect_same_as_dplyr_lazy(f47_df(), with_id)
})

test_that("dplyr::bind_rows(a, b) of two tables matches dplyr and keeps the first's groups", {
  skip_if_no_gpu()
  pipe <- function(d) {
    dplyr::bind_rows(dplyr::group_by(dplyr::filter(d, g < 3), g), dplyr::filter(d, g == 3))
  }
  expect_same_as_dplyr(f47_df(), pipe)
  expect_same_as_dplyr_lazy(f47_df(), pipe)
  # different column sets: union of columns, NA padding, first's column order
  pad <- function(d) dplyr::bind_rows(dplyr::select(d, x, s), dplyr::select(d, b, x))
  expect_same_as_dplyr(f47_df(), pad)
  expect_same_as_dplyr_lazy(f47_df(), pad)
})

# ---------------------------------------------------------------------------
# Same result as cuplyr::bind_rows(): classes, types, groups, mode
# ---------------------------------------------------------------------------

test_that("dplyr::bind_rows() returns what cuplyr::bind_rows() returns (eager and lazy pieces)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    pieces <- dplyr::group_split(tbl_gpu(f47_df(), lazy = lazy), g)
    f47_expect_same_as_cuplyr(dplyr::bind_rows(pieces), cuplyr::bind_rows(pieces))
    f47_expect_same_as_cuplyr(
      dplyr::bind_rows(pieces, .id = "id"), cuplyr::bind_rows(pieces, .id = "id")
    )
    f47_expect_same_as_cuplyr(
      dplyr::bind_rows(pieces[[3]], pieces[[1]]), cuplyr::bind_rows(pieces[[3]], pieces[[1]])
    )
    gb <- dplyr::group_by(tbl_gpu(f47_df(), lazy = lazy), g, .drop = FALSE)
    f47_expect_same_as_cuplyr(dplyr::bind_rows(gb, gb), cuplyr::bind_rows(gb, gb))
  }
})

test_that("dplyr::bind_rows() unifies types exactly like cuplyr::bind_rows() and dplyr", {
  skip_if_no_gpu()
  a <- tibble::tibble(
    f = factor(c("x", "y"), levels = c("y", "x")),
    n = c(1L, NA),
    z = c(NA, NA),
    d = as.Date(c("2020-01-01", NA))
  )
  b <- tibble::tibble(
    f = factor(c("q", NA, "x"), levels = c("x", "q")),
    n = c(2.5, NA, 4),
    z = c(1.5, NA, 2),
    t = as.POSIXct(c(0, NA, 60), origin = "1970-01-01", tz = "UTC")
  )
  ga <- tbl_gpu(a)
  gb <- tbl_gpu(b)
  out <- dplyr::bind_rows(ga, gb)
  f47_expect_same_as_cuplyr(out, cuplyr::bind_rows(ga, gb))
  expect_identical(collect(out), dplyr::bind_rows(a, b))
  out2 <- dplyr::bind_rows(list(gb, ga), .id = "src")
  f47_expect_same_as_cuplyr(out2, cuplyr::bind_rows(list(gb, ga), .id = "src"))
  expect_identical(collect(out2), dplyr::bind_rows(list(b, a), .id = "src"))
})

test_that("zero-row pieces and a single piece bind like cuplyr::bind_rows()", {
  skip_if_no_gpu()
  g <- tbl_gpu(f47_df())
  empty <- dplyr::filter(g, g > 100)
  f47_expect_same_as_cuplyr(dplyr::bind_rows(empty, g), cuplyr::bind_rows(empty, g))
  f47_expect_same_as_cuplyr(dplyr::bind_rows(empty, empty), cuplyr::bind_rows(empty, empty))
  f47_expect_same_as_cuplyr(dplyr::bind_rows(g), cuplyr::bind_rows(g))
  f47_expect_same_as_cuplyr(dplyr::bind_rows(list(g)), cuplyr::bind_rows(list(g)))
  expect_identical(collect(dplyr::bind_rows(empty, g)), dplyr::bind_rows(f47_df()[0, ], f47_df()))
})

test_that("dplyr::bind_rows() flattens and drops NULLs exactly like cuplyr::bind_rows()", {
  skip_if_no_gpu()
  g1 <- tbl_gpu(f47_df()[1:2, ])
  g2 <- tbl_gpu(f47_df()[3:6, ])
  f47_expect_same_as_cuplyr(
    dplyr::bind_rows(list(g1, NULL), NULL, list(g2)),
    cuplyr::bind_rows(list(g1, NULL), NULL, list(g2))
  )
  # a nested unnamed list is flattened only one level by dplyr: still an error
  f47_expect_same_error_as_cuplyr(
    dplyr::bind_rows(g1, list(list(g2))),
    cuplyr::bind_rows(g1, list(list(g2))),
    "rlang_error"
  )
  # a spliced argument is evaluated once
  n <- 0L
  mk <- function() { n <<- n + 1L; list(g1, g2) }
  out <- dplyr::bind_rows(!!!mk())
  expect_identical(n, 1L)
  f47_expect_same_as_cuplyr(out, cuplyr::bind_rows(g1, g2))
})

test_that("a vctrs call on a tbl_gpu inside dplyr::bind_rows()'s arguments still works", {
  skip_if_no_gpu()
  g <- tbl_gpu(f47_df())
  out <- dplyr::bind_rows(vctrs::vec_slice(g, c(2L, 1L)), g)
  f47_expect_same_as_cuplyr(out, cuplyr::bind_rows(vctrs::vec_slice(g, c(2L, 1L)), g))
  expect_identical(collect(out), dplyr::bind_rows(f47_df()[c(2, 1), ], f47_df()))
  # nested binders: the inner map_dfr() result feeds the outer bind
  skip_if_not_installed("purrr")
  pieces <- dplyr::group_split(g, g)
  nested <- dplyr::bind_rows(purrr::map_dfr(pieces, identity), pieces[[1]])
  f47_expect_same_as_cuplyr(nested, cuplyr::bind_rows(cuplyr::bind_rows(pieces), pieces[[1]]))
})

# ---------------------------------------------------------------------------
# Mixed inputs: the tibble is uploaded, the tbl_gpu is never downloaded
# ---------------------------------------------------------------------------

test_that("dplyr::bind_rows() of a tbl_gpu and a tibble returns cuplyr's tbl_gpu without downloading", {
  skip_if_no_gpu()
  tb <- f47_df()[1:2, ]
  df <- as.data.frame(f47_df()[3:4, ])
  g <- tbl_gpu(f47_df()[5:6, ])
  out1 <- f47_no_download(dplyr::bind_rows(tb, g))
  out2 <- f47_no_download(dplyr::bind_rows(g, tb))
  out3 <- f47_no_download(dplyr::bind_rows(df, g, tb))
  f47_expect_same_as_cuplyr(out1, cuplyr::bind_rows(tb, g))
  f47_expect_same_as_cuplyr(out2, cuplyr::bind_rows(g, tb))
  f47_expect_same_as_cuplyr(out3, cuplyr::bind_rows(df, g, tb))
  expect_identical(collect(out1), dplyr::bind_rows(tb, f47_df()[5:6, ]))
})

test_that("binding 100 lazy and eager pieces through dplyr and purrr never downloads", {
  skip_if_no_gpu()
  n <- 20000L
  big <- tibble::tibble(k = rep_len(1:100, n), v = as.double(seq_len(n)), w = rep_len(c("p", "q", NA), n))
  for (lazy in c(FALSE, TRUE)) {
    pieces <- dplyr::group_split(tbl_gpu(big, lazy = lazy), k)
    expect_length(pieces, 100L)
    out <- f47_no_download(dplyr::bind_rows(pieces))
    expect_true(is_tbl_gpu(out))
    f47_expect_same_as_cuplyr(out, cuplyr::bind_rows(pieces))
    if (requireNamespace("purrr", quietly = TRUE)) {
      out_p <- f47_no_download(purrr::map_dfr(pieces, identity, .id = "i"))
      f47_expect_same_as_cuplyr(out_p, cuplyr::bind_rows(pieces, .id = "i"))
    }
  }
})

# ---------------------------------------------------------------------------
# Errors: the same class and message as cuplyr::bind_rows()
# ---------------------------------------------------------------------------

test_that("dplyr::bind_rows() errors like cuplyr::bind_rows()", {
  skip_if_no_gpu()
  gx <- tbl_gpu(tibble::tibble(x = c(1, 2)))
  gs <- tbl_gpu(tibble::tibble(x = c("a", "b")))
  f47_expect_same_error_as_cuplyr(
    dplyr::bind_rows(gx, gs), cuplyr::bind_rows(gx, gs), "vctrs_error_incompatible_type"
  )
  f47_expect_same_error_as_cuplyr(
    dplyr::bind_rows(gx, tibble::tibble(x = "a")),
    cuplyr::bind_rows(gx, tibble::tibble(x = "a")),
    "vctrs_error_incompatible_type"
  )
  f47_expect_same_error_as_cuplyr(
    dplyr::bind_rows(gx, 1:3), cuplyr::bind_rows(gx, 1:3), "rlang_error"
  )
  f47_expect_same_error_as_cuplyr(
    dplyr::bind_rows(gx, gx, .id = 5), cuplyr::bind_rows(gx, gx, .id = 5), "rlang_error"
  )
  skip_if_not_installed("purrr")
  f47_expect_same_error_as_cuplyr(
    purrr::map_dfr(list(gx, gs), identity), cuplyr::bind_rows(gx, gs),
    "vctrs_error_incompatible_type"
  )
})

# ---------------------------------------------------------------------------
# bind_cols(): the same hook
# ---------------------------------------------------------------------------

test_that("dplyr::bind_cols() over tbl_gpu inputs matches dplyr and cuplyr::bind_cols()", {
  skip_if_no_gpu()
  pipe <- function(d) dplyr::bind_cols(dplyr::select(d, g, x), dplyr::select(d, s, f))
  expect_same_as_dplyr(f47_df(), pipe)
  expect_same_as_dplyr_lazy(f47_df(), pipe)
  g <- tbl_gpu(f47_df())
  a <- dplyr::select(g, g, x)
  b <- dplyr::select(g, x, s)
  f47_expect_same_as_cuplyr(
    suppressMessages(dplyr::bind_cols(a, b)), suppressMessages(cuplyr::bind_cols(a, b))
  )
  f47_expect_same_as_cuplyr(
    dplyr::bind_cols(a, b, .name_repair = "minimal"),
    cuplyr::bind_cols(a, b, .name_repair = "minimal")
  )
  tb <- tibble::tibble(z = 1:6)
  f47_expect_same_as_cuplyr(
    f47_no_download(dplyr::bind_cols(tb, a)), cuplyr::bind_cols(tb, a)
  )
  f47_expect_same_error_as_cuplyr(
    dplyr::bind_cols(a, tibble::tibble(z = 1:2)),
    cuplyr::bind_cols(a, tibble::tibble(z = 1:2)),
    "vctrs_error_incompatible_size"
  )
})

# ---------------------------------------------------------------------------
# What stays unsupported, loudly
# ---------------------------------------------------------------------------

test_that("direct vctrs::vec_rbind()/vec_cbind() on a tbl_gpu is still a clear error naming the alternatives", {
  skip_if_no_gpu()
  g1 <- tbl_gpu(tibble::tibble(x = 1:3))
  g2 <- tbl_gpu(tibble::tibble(x = 4:6))
  e <- rlang::catch_cnd(vctrs::vec_rbind(g1, g2), "error")
  expect_s3_class(e, "cuplyr_error_vctrs_bind_unsupported")
  msg <- conditionMessage(e)
  expect_match(msg, "cuplyr::bind_rows()", fixed = TRUE)
  expect_match(msg, "dplyr::bind_rows()", fixed = TRUE)
  expect_match(msg, "list_rbind()", fixed = TRUE)
  expect_error(vctrs::vec_cbind(g1, g2), class = "cuplyr_error_vctrs_bind_unsupported")
})

test_that("purrr::list_rbind() of tbl_gpu pieces errors (purrr rejects them before dispatch)", {
  skip_if_no_gpu()
  skip_if_not_installed("purrr")
  pieces <- dplyr::group_split(tbl_gpu(f47_df()), g)
  expect_error(purrr::list_rbind(pieces), "data frame")
})
