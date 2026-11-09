# F98: top-k for lazy arrange() |> head() and slice_min()/slice_max().
#
# `cuplyr.topk_path` ("auto", "topk", "sort") picks gpu_slice_rank()'s route, so
# every parity case below runs through the top-k route AND the sort route.

first <- dplyr::first

f98_df <- function(n = 40L, seed = 98L) {
  set.seed(seed)
  tibble::tibble(
    id = seq_len(n),
    g = rep_len(c(2L, 1L, 3L), n),
    # heavy ties, so the order among ties is observable through `id`
    x = sample(c(1.5, 2.5, 3.5, -1, 0, 7), n, replace = TRUE),
    xna = replace(sample(c(1.5, 2.5, 3.5, -1, 0, 7), n, replace = TRUE),
                  sample.int(n, n %/% 4L), NA),
    i = sample(c(5L, 3L, 9L, 1L), n, replace = TRUE),
    ina = replace(sample(c(5L, 3L, 9L, 1L), n, replace = TRUE), sample.int(n, n %/% 4L), NA),
    d = as.Date("2024-01-01") + sample(0:5, n, replace = TRUE),
    dna = replace(as.Date("2024-01-01") + sample(0:5, n, replace = TRUE),
                  sample.int(n, n %/% 4L), NA),
    t = as.POSIXct("2024-01-01 00:00:00", tz = "UTC") + 3600 * sample(0:5, n, replace = TRUE),
    s = sample(c("b", "a", "c"), n, replace = TRUE),
    f = factor(sample(c("b", "a", "c"), n, replace = TRUE), levels = c("c", "a", "b")),
    l = sample(c(TRUE, FALSE), n, replace = TRUE)
  )
}

# GPU-computed NaN next to NA and infinities
f98_nan_df <- function() {
  tibble::tibble(
    id = 1:10,
    a = c(0, 1, -1, 0, 2, NA, 0, 3, 1, -5),
    b = c(0, 0, 0, 0, 1, 1, 0, 1, 1, 1)
  )
}

f98_paths <- c("auto", "topk", "sort")

# Run one pipeline eager and lazy, on every route.
f98_expect <- function(df, pipeline) {
  for (p in f98_paths) {
    withr::local_options(cuplyr.topk_path = p)
    expect_same_as_dplyr(df, pipeline)
    expect_same_as_dplyr_lazy(df, pipeline)
  }
}

# The node types of a plan, top first (no joins here)
f98_chain <- function(node) {
  out <- character()
  while (!is.null(node)) {
    out <- c(out, node$type)
    node <- node$input
  }
  out
}

f98_opt <- function(pipeline, df = f98_df()) {
  optimize_ast(.subset2(pipeline(tbl_gpu(df, lazy = TRUE)), "lazy_ops"))
}

# ---- arrange() |> head() / slice_head() -------------------------------------

test_that("arrange |> head matches dplyr: keys, direction, ties, NA", {
  skip_if_no_gpu()
  df <- f98_df()
  pipes <- list(
    function(d) head(dplyr::arrange(d, x), 5),
    function(d) head(dplyr::arrange(d, dplyr::desc(x)), 5),
    function(d) head(dplyr::arrange(d, -x), 7),
    function(d) head(dplyr::arrange(d, xna), 8),
    function(d) head(dplyr::arrange(d, dplyr::desc(xna)), 8),
    function(d) head(dplyr::arrange(d, ina), 6),
    function(d) head(dplyr::arrange(d, dplyr::desc(ina)), 6),
    function(d) head(dplyr::arrange(d, i), 6),
    function(d) head(dplyr::arrange(d, d), 6),
    function(d) head(dplyr::arrange(d, dplyr::desc(dna)), 6),
    function(d) head(dplyr::arrange(d, t), 6),
    function(d) head(dplyr::arrange(d, dplyr::desc(t)), 6),
    function(d) dplyr::slice_head(dplyr::arrange(d, x), n = 5),
    function(d) dplyr::slice_head(dplyr::arrange(d, dplyr::desc(ina)), n = 5)
  )
  for (p in pipes) f98_expect(df, p)
})

test_that("arrange |> head size rules match dplyr: n, prop, negative, oversized, zero", {
  skip_if_no_gpu()
  df <- f98_df()
  pipes <- list(
    function(d) head(dplyr::arrange(d, xna), 0),
    function(d) head(dplyr::arrange(d, xna), 1),
    function(d) head(dplyr::arrange(d, xna), 39),
    function(d) head(dplyr::arrange(d, xna), 40),
    function(d) head(dplyr::arrange(d, xna), 500),
    function(d) head(dplyr::arrange(d, xna), -3),
    function(d) head(dplyr::arrange(d, xna), -100),
    function(d) head(dplyr::arrange(d, xna), 2.9),
    function(d) dplyr::slice_head(dplyr::arrange(d, x), prop = 0.25),
    function(d) dplyr::slice_head(dplyr::arrange(d, xna), prop = 0.9),
    function(d) dplyr::slice_head(dplyr::arrange(d, xna), prop = -0.25),
    function(d) dplyr::slice_head(dplyr::arrange(d, xna), n = -3),
    function(d) dplyr::slice_head(dplyr::arrange(d, xna), prop = 2)
  )
  for (p in pipes) f98_expect(df, p)
})

test_that("arrange |> head matches dplyr on NaN, infinities and signed zero", {
  skip_if_no_gpu()
  nan <- f98_nan_df()
  zinf <- tibble::tibble(
    id = 1:10,
    x = c(0, -0, Inf, -Inf, 1, -0, 0, Inf, -Inf, 1),
    y = 1 / c(0, -0, Inf, -Inf, 1, -0, 0, Inf, -Inf, 1)
  )
  pipes_nan <- list(
    function(d) head(dplyr::arrange(dplyr::mutate(d, r = a / b), r), 3),
    function(d) head(dplyr::arrange(dplyr::mutate(d, r = a / b), dplyr::desc(r)), 3),
    function(d) head(dplyr::arrange(dplyr::mutate(d, r = a / b), r), 8),
    function(d) head(dplyr::arrange(dplyr::mutate(d, r = a / b), dplyr::desc(r)), 8)
  )
  for (p in pipes_nan) f98_expect(nan, p)
  pipes_z <- list(
    function(d) head(dplyr::arrange(d, x), 4),
    function(d) head(dplyr::arrange(d, x), 6),
    function(d) head(dplyr::arrange(d, dplyr::desc(x)), 4),
    function(d) head(dplyr::arrange(d, dplyr::desc(x)), 6)
  )
  for (p in pipes_z) f98_expect(zinf, p)
})

test_that("arrange |> head matches dplyr on empty, all-NA and groups", {
  skip_if_no_gpu()
  df <- f98_df()
  empty <- df[0, ]
  all_na <- tibble::tibble(id = 1:6, x = rep(NA_real_, 6), i = rep(NA_integer_, 6))
  f98_expect(empty, function(d) head(dplyr::arrange(d, x), 3))
  f98_expect(all_na, function(d) head(dplyr::arrange(d, x), 3))
  f98_expect(all_na, function(d) head(dplyr::arrange(d, dplyr::desc(i)), 3))
  # head() ignores groups and keeps them; slice_head() is per group
  f98_expect(df, function(d) head(dplyr::arrange(dplyr::group_by(d, g), x), 3))
  f98_expect(df, function(d) dplyr::slice_head(dplyr::arrange(dplyr::group_by(d, g), x), n = 1))
  f98_expect(df, function(d) head(dplyr::arrange(dplyr::group_by(d, g), x, .by_group = TRUE), 5))
  f98_expect(df, function(d) dplyr::slice_head(dplyr::arrange(d, x), n = 2, by = g))
})

test_that("verbs around the top-k node see the same rows", {
  skip_if_no_gpu()
  df <- f98_df()
  pipes <- list(
    function(d) dplyr::select(head(dplyr::arrange(d, x), 4), id),
    function(d) dplyr::filter(head(dplyr::arrange(d, x), 8), id > 10L),
    function(d) dplyr::mutate(head(dplyr::arrange(d, xna), 5), z = x + 1),
    function(d) head(dplyr::arrange(dplyr::filter(d, x > 0), dplyr::desc(x)), 3),
    function(d) head(dplyr::arrange(dplyr::arrange(d, id), x), 6),
    function(d) head(dplyr::arrange(dplyr::arrange(d, dplyr::desc(id)), x), 6),
    function(d) dplyr::summarise(head(dplyr::arrange(d, x), 5), m = sum(x)),
    function(d) head(dplyr::arrange(dplyr::select(d, id, x), x), 5)
  )
  for (p in pipes) f98_expect(df, p)
})

# ---- slice_min() / slice_max() -------------------------------------------------

test_that("slice_min/max match dplyr: keys, ties, with_ties, na_rm", {
  skip_if_no_gpu()
  df <- f98_df()
  for (key in c("x", "xna", "i", "ina", "d", "dna", "t")) {
    k <- as.symbol(key)
    pipes <- list(
      function(d) dplyr::slice_min(d, !!k, n = 2),
      function(d) dplyr::slice_max(d, !!k, n = 2),
      function(d) dplyr::slice_min(d, !!k, n = 2, with_ties = FALSE),
      function(d) dplyr::slice_max(d, !!k, n = 2, with_ties = FALSE),
      function(d) dplyr::slice_min(d, !!k, n = 2, na_rm = TRUE),
      function(d) dplyr::slice_max(d, !!k, n = 3, na_rm = TRUE, with_ties = FALSE)
    )
    for (p in pipes) f98_expect(df, p)
  }
})

test_that("slice_min/max size rules match dplyr", {
  skip_if_no_gpu()
  df <- f98_df()
  pipes <- list(
    function(d) dplyr::slice_min(d, xna, n = 0),
    function(d) dplyr::slice_min(d, xna, n = 500),
    function(d) dplyr::slice_max(d, xna, n = 500, with_ties = FALSE),
    function(d) dplyr::slice_min(d, xna, n = 35),
    function(d) dplyr::slice_max(d, xna, n = 35, with_ties = FALSE),
    function(d) dplyr::slice_min(d, xna, n = -2),
    function(d) dplyr::slice_max(d, xna, n = -2, with_ties = FALSE),
    function(d) dplyr::slice_min(d, x, prop = 0.1),
    function(d) dplyr::slice_max(d, xna, prop = 0.5, with_ties = FALSE),
    function(d) dplyr::slice_min(d, xna, prop = -0.25),
    function(d) dplyr::slice_max(d, xna, prop = 2),
    function(d) dplyr::slice_min(d, xna, n = 500, na_rm = TRUE),
    function(d) dplyr::slice_max(d, xna, n = 35, na_rm = TRUE, with_ties = FALSE)
  )
  for (p in pipes) f98_expect(df, p)
})

test_that("slice_min/max match dplyr on NaN, infinities, signed zero and expressions", {
  skip_if_no_gpu()
  nan <- f98_nan_df()
  zinf <- tibble::tibble(id = 1:8, x = c(0, -0, Inf, -Inf, 1, -0, 0, Inf))
  df <- f98_df()
  pipes_nan <- list(
    function(d) dplyr::slice_min(dplyr::mutate(d, r = a / b), r, n = 3),
    function(d) dplyr::slice_max(dplyr::mutate(d, r = a / b), r, n = 3),
    function(d) dplyr::slice_max(dplyr::mutate(d, r = a / b), r, n = 9),
    function(d) dplyr::slice_min(dplyr::mutate(d, r = a / b), r, n = 9, with_ties = FALSE),
    function(d) dplyr::slice_max(dplyr::mutate(d, r = a / b), r, n = 2, na_rm = TRUE)
  )
  for (p in pipes_nan) f98_expect(nan, p)
  for (p in list(
    function(d) dplyr::slice_min(d, x, n = 2),
    function(d) dplyr::slice_min(d, x, n = 4, with_ties = FALSE),
    function(d) dplyr::slice_max(d, x, n = 5),
    function(d) dplyr::slice_max(d, x, n = 5, with_ties = FALSE)
  )) f98_expect(zinf, p)
  f98_expect(df, function(d) dplyr::slice_min(d, x * 2 + i, n = 3))
  f98_expect(df, function(d) dplyr::slice_max(d, x + i, n = 3, with_ties = FALSE))
})

test_that("slice_min/max match dplyr on empty, all-NA, grouped and non-numeric keys", {
  skip_if_no_gpu()
  df <- f98_df()
  empty <- df[0, ]
  all_na <- tibble::tibble(id = 1:6, x = rep(NA_real_, 6))
  f98_expect(empty, function(d) dplyr::slice_min(d, x, n = 2))
  f98_expect(all_na, function(d) dplyr::slice_min(d, x, n = 2))
  f98_expect(all_na, function(d) dplyr::slice_max(d, x, n = 2, with_ties = FALSE))
  f98_expect(all_na, function(d) dplyr::slice_max(d, x, n = 2, na_rm = TRUE))
  f98_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 1))
  f98_expect(df, function(d) dplyr::slice_max(d, xna, n = 2, by = g, with_ties = FALSE))
  f98_expect(df, function(d) dplyr::slice_min(d, l, n = 2))
  f98_expect(df, function(d) dplyr::slice_max(d, f, n = 2, with_ties = FALSE))
  f98_expect(df, function(d) dplyr::slice_min(dplyr::mutate(d, ss = s), ss, n = 2))
})

test_that("slice_min/max errors match dplyr", {
  skip_if_no_gpu()
  df <- f98_df()
  gpu <- tbl_gpu(df)
  expect_error(dplyr::slice_min(df, x, n = 1, prop = 0.5))
  expect_error(dplyr::slice_min(gpu, x, n = 1, prop = 0.5))
  expect_error(dplyr::slice_max(df, x, n = 1:2))
  expect_error(dplyr::slice_max(gpu, x, n = 1:2))
  expect_error(dplyr::slice_min(df, n = 1))
  expect_error(dplyr::slice_min(gpu, n = 1))
})

# ---- plan shape --------------------------------------------------------------

f98_expect_topk <- function(node, order_col, descending = FALSE) {
  expect_identical(node$type, "slice")
  expect_identical(node$mode, "rank")
  expect_identical(node$order_col, order_col)
  expect_identical(isTRUE(node$descending), descending)
  expect_false(isTRUE(node$with_ties))
  expect_false(isTRUE(node$na_rm))
  expect_true(isTRUE(node$top_k))
  expect_false(identical(node$input$type, "arrange"))
}

test_that("the rewrite fires on a single numeric, Date or POSIXct key", {
  skip_if_no_gpu()
  f98_expect_topk(f98_opt(function(d) head(dplyr::arrange(d, x), 3)), "x")
  f98_expect_topk(f98_opt(function(d) head(dplyr::arrange(d, dplyr::desc(x)), 3)), "x", TRUE)
  f98_expect_topk(f98_opt(function(d) dplyr::slice_head(dplyr::arrange(d, i), n = 3)), "i")
  f98_expect_topk(f98_opt(function(d) dplyr::slice_head(dplyr::arrange(d, ina), prop = 0.1)), "ina")
  f98_expect_topk(f98_opt(function(d) head(dplyr::arrange(d, d), -3)), "d")
  f98_expect_topk(f98_opt(function(d) head(dplyr::arrange(d, t), 3)), "t")
  # below a filter (the barrier found by find_next_barrier()) and over one
  over <- f98_opt(function(d) dplyr::filter(head(dplyr::arrange(d, x), 8), id > 3L))
  expect_identical(over$type, "filter")
  f98_expect_topk(over$input, "x")
  under <- f98_opt(function(d) head(dplyr::arrange(dplyr::filter(d, id > 3L), x), 3))
  f98_expect_topk(under, "x")
  expect_identical(under$input$type, "filter")
  # the slice node reports its key, even when nothing above reads it (F99)
  sel <- f98_opt(function(d) dplyr::select(head(dplyr::arrange(d, x), 3), id))
  chain <- f98_chain(sel)
  expect_false("arrange" %in% chain)
  expect_true("slice" %in% chain)
})

test_that("the rewrite declines strings, factors, logicals, multi-key and a step in between", {
  skip_if_no_gpu()
  declines <- list(
    function(d) head(dplyr::arrange(d, s), 3),
    function(d) head(dplyr::arrange(d, f), 3),
    function(d) head(dplyr::arrange(d, l), 3),
    function(d) head(dplyr::arrange(d, x, id), 3),
    function(d) head(dplyr::arrange(dplyr::group_by(d, g), x, .by_group = TRUE), 3),
    function(d) head(dplyr::mutate(dplyr::arrange(d, x), z = x + 1), 3),
    function(d) head(dplyr::select(dplyr::arrange(d, x), id, x), 3),
    function(d) head(dplyr::filter(dplyr::arrange(d, x), id > 3L), 3),
    function(d) dplyr::slice_tail(dplyr::arrange(d, x), n = 3),
    function(d) dplyr::slice_head(dplyr::arrange(d, x), n = 1, by = g),
    function(d) dplyr::slice_head(dplyr::arrange(dplyr::group_by(d, g), x), n = 1)
  )
  for (p in declines) {
    chain <- f98_chain(f98_opt(p))
    expect_true("arrange" %in% chain, info = paste(deparse(body(p)), collapse = ""))
  }
})

test_that("filters do not cross the top-k node", {
  skip_if_no_gpu()
  node <- f98_opt(function(d) dplyr::filter(head(dplyr::arrange(d, x), 8), id > 10L, x > 0))
  expect_identical(node$type, "filter")
  expect_identical(node$input$type, "slice")
  expect_false("filter" %in% f98_chain(node$input$input))
})

# ---- the route gpu_slice_rank() takes ----------------------------------------

f98_route <- function(df, col, n, descending = FALSE, with_ties = TRUE, na_rm = FALSE,
                      path = "auto", is_prop = FALSE) {
  ptr <- .subset2(tbl_gpu(df), "ptr")
  idx <- match(col, names(df)) - 1L
  attr(gpu_slice_rank(ptr, idx, descending, n, is_prop, with_ties, na_rm, path), "topk")
}

test_that("gpu_slice_rank takes the top-k route for small n on supported keys", {
  skip_if_no_gpu()
  df <- f98_df(2000L)
  for (col in c("x", "xna", "i", "ina", "d", "dna", "t")) {
    expect_true(isTRUE(f98_route(df, col, 5)), info = col)
    expect_true(isTRUE(f98_route(df, col, 5, descending = TRUE, with_ties = FALSE)), info = col)
  }
})

test_that("gpu_slice_rank declines when it must", {
  skip_if_no_gpu()
  df <- f98_df(2000L)
  # non-numeric keys, even when forced
  for (col in c("s", "l")) {
    expect_false(isTRUE(f98_route(df, col, 5, path = "topk")), info = col)
  }
  # n close to the row count: a plain sort is as fast
  expect_false(isTRUE(f98_route(df, "x", 1999)))
  expect_false(isTRUE(f98_route(df, "x", -1)))
  # forced off
  expect_false(isTRUE(f98_route(df, "x", 5, path = "sort")))
  # n larger than the non-NA rows: NA rows are in the result
  all_na <- tibble::tibble(x = c(NA, 1, 2, NA, NA, NA))
  expect_false(isTRUE(f98_route(all_na, "x", 3, path = "topk")))
  expect_false(isTRUE(f98_route(all_na, "x", 0, path = "topk")))
  # empty table
  expect_false(isTRUE(f98_route(df[0, ], "x", 3, path = "topk")))
  # a forced top-k still gives the right rows at n close to the row count
  expect_true(isTRUE(f98_route(df, "x", 1999, path = "topk")))
})
