# F115: native route for grouped slice_min()/slice_max().
#
# `cuplyr.topk_path` picks the route as for F98: "auto" takes the native route
# when the shape is supported, "sort" keeps the window desugar, so each parity
# case below runs through both.

f115_df <- function(n = 120L, seed = 115L) {
  set.seed(seed)
  main <- tibble::tibble(
    id = seq_len(n),
    # first appearance is 3, 1, 2, NA: not the sorted order
    g = rep_len(c(3L, 1L, 2L, NA), n),
    g2 = sample(1:2, n, replace = TRUE),
    s = rep_len(c("b", "a", NA, "c"), n),
    f = factor(rep_len(c("y", "x"), n), levels = c("z", "x", "y")),
    # heavy ties, so tie order shows through `id`
    x = replace(sample(c(1.5, 2.5, 0, -0, -1, 7, Inf, -Inf), n, replace = TRUE),
                sample.int(n, n %/% 5L), NA),
    xnn = sample(c(1.5, 2.5, 0, -1, 7), n, replace = TRUE),
    i = replace(sample(c(5L, 3L, 9L, 1L), n, replace = TRUE), sample.int(n, n %/% 5L), NA),
    inn = sample(c(5L, 3L, 9L, 1L), n, replace = TRUE),
    d = replace(as.Date("2024-01-01") + sample(0:5, n, replace = TRUE), sample.int(n, n %/% 5L), NA),
    t = replace(as.POSIXct("2024-01-01 00:00:00", tz = "UTC") + 3600 * sample(0:5, n, replace = TRUE),
                sample.int(n, n %/% 5L), NA),
    b = sample(c(TRUE, FALSE), n, replace = TRUE)
  )
  # a one-row group and a group whose keys are all missing
  extra <- main[c(1L, 2L, 3L), ]
  extra$id <- n + 1:3
  extra$g <- c(50L, 60L, 60L)
  extra$x <- c(NA, NA, NA)
  extra$i <- c(NA, NA, NA)
  extra$d <- as.Date(c(NA, NA, NA))
  extra$t <- as.POSIXct(c(NA, NA, NA), tz = "UTC")
  dplyr::bind_rows(main, extra)
}

f115_paths <- c("auto", "sort")

# One pipeline, eager and lazy, through both routes.
f115_expect <- function(df, pipeline) {
  for (p in f115_paths) {
    withr::local_options(cuplyr.topk_path = p)
    expect_same_as_dplyr(df, pipeline)
    expect_same_as_dplyr_lazy(df, pipeline)
  }
}

# Modes of every slice node in a lazy plan.
f115_plan <- function(node) {
  out <- list(type = character(), mode = character())
  while (!is.null(node)) {
    out$type <- c(out$type, node$type)
    if (identical(node$type, "slice")) out$mode <- c(out$mode, node$mode)
    node <- node$input
  }
  out
}

test_that("group_by(): n, direction, ties and missing keys match dplyr", {
  skip_if_no_gpu()
  df <- f115_df()
  cases <- list(
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 3),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 3),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 3, with_ties = FALSE),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 3, with_ties = FALSE),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 3, na_rm = TRUE),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 3, with_ties = FALSE, na_rm = TRUE),
    # n past the group size, so the missing rows come in
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 1000),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 1000, with_ties = FALSE),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = Inf),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 0),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 0, with_ties = FALSE),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = -2),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = -2, with_ties = FALSE),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = -1000),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), xnn, n = 2),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), xnn, n = 2, with_ties = FALSE)
  )
  for (pipeline in cases) f115_expect(df, pipeline)
})

test_that("prop sizes match dplyr", {
  skip_if_no_gpu()
  df <- f115_df()
  for (p in c(0.1, 0.25, 0.5, 1, 1.5, -0.1, -0.5, 0, Inf, -Inf)) {
    f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), x, prop = p))
    f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), x, prop = p, with_ties = FALSE))
    f115_expect(df, function(d) dplyr::slice_max(d, i, prop = p, by = g2, na_rm = TRUE))
  }
})

test_that("every supported key type matches dplyr", {
  skip_if_no_gpu()
  df <- f115_df()
  for (key in c("x", "xnn", "i", "inn", "d", "t")) {
    k <- rlang::sym(key)
    f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), !!k, n = 2))
    f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), !!k, n = 2))
    f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), !!k, n = 2, with_ties = FALSE))
    f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), !!k, n = 2, with_ties = FALSE, na_rm = TRUE))
  }
  # a computed key is not an output column
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), xnn + inn, n = 2))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), x * 2, n = 2, with_ties = FALSE))
})

test_that("grouping shapes and group order match dplyr", {
  skip_if_no_gpu()
  df <- f115_df()
  # strings and a missing group key
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, s), x, n = 2))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, s), i, n = 2, with_ties = FALSE))
  # several keys
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g, g2), x, n = 2))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, s, g2), x, n = 2, with_ties = FALSE))
  # other group column types
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, f), x, n = 2))
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, b), i, n = 2))
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, d), xnn, n = 1))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, xnn), i, n = 1))
})

test_that(".by and by = group by first appearance", {
  skip_if_no_gpu()
  df <- f115_df()
  f115_expect(df, function(d) dplyr::slice_min(d, x, n = 2, by = g))
  f115_expect(df, function(d) dplyr::slice_max(d, x, n = 2, by = g))
  f115_expect(df, function(d) dplyr::slice_min(d, x, n = 2, with_ties = FALSE, by = g))
  f115_expect(df, function(d) dplyr::slice_max(d, x, n = 2, with_ties = FALSE, na_rm = TRUE, by = g))
  f115_expect(df, function(d) dplyr::slice_min(d, i, n = 2, by = c(g, g2)))
  f115_expect(df, function(d) dplyr::slice_max(d, xnn, n = 1, by = s))
  f115_expect(df, function(d) dplyr::slice_min(d, x, prop = 0.5, by = s))
})

test_that("NaN ranks as missing and -0 ties with 0", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    id = 1:20,
    g = rep(c(2L, 1L), 10L),
    a = c(0, 1, -1, 0, 2, NA, 0, 3, 1, -5, 0, 0, 1, -1, 2, 0, 0, NA, 3, 1),
    b = c(0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0),
    z = rep(c(0, -0, -0, 0), 5L)
  )
  nan_key <- function(d) dplyr::mutate(d, y = a / b)
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(nan_key(d), g), y, n = 3))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(nan_key(d), g), y, n = 3))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(nan_key(d), g), y, n = 8, with_ties = FALSE))
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(nan_key(d), g), y, n = 3, na_rm = TRUE))
  f115_expect(df, function(d) dplyr::slice_max(nan_key(d), y, n = 20, by = g))
  # a NaN group key is its own group, apart from NA
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(nan_key(d), y), z, n = 1, with_ties = FALSE))
  f115_expect(df, function(d) dplyr::slice_max(nan_key(d), id, n = 1, by = y))
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), z, n = 1))
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), z, n = 2, with_ties = FALSE))
  f115_expect(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), z, n = 2, with_ties = FALSE))
})

test_that("empty tables and empty groups match dplyr", {
  skip_if_no_gpu()
  df <- f115_df()
  empty <- df[0L, ]
  f115_expect(empty, function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 2))
  f115_expect(empty, function(d) dplyr::slice_max(d, x, n = 2, by = g))
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(dplyr::filter(d, id < 0L), g), x))

  # .drop = FALSE keeps a level with no rows
  f115_expect(df, function(d) dplyr::slice_min(dplyr::group_by(d, f, .drop = FALSE), x, n = 2))
  f115_expect(df, function(d) {
    dplyr::summarise(dplyr::slice_min(dplyr::group_by(d, f, .drop = FALSE), x, n = 2), n = dplyr::n())
  })
  # .drop = FALSE where the slice itself empties a group
  f115_expect(df, function(d) {
    dplyr::summarise(dplyr::slice_min(dplyr::group_by(d, f, .drop = FALSE), x, n = 0), n = dplyr::n())
  })
  # the slice empties a level under the default .drop
  f115_expect(df, function(d) {
    dplyr::summarise(dplyr::slice_min(dplyr::group_by(d, f), x, n = 0), n = dplyr::n())
  })
})

test_that("the rest of a lazy plan sees the keys it needs", {
  skip_if_no_gpu()
  df <- f115_df()
  pipe <- function(d) {
    d |>
      dplyr::mutate(z = xnn + inn) |>
      dplyr::group_by(g) |>
      dplyr::slice_max(z, n = 1, with_ties = FALSE) |>
      dplyr::ungroup() |>
      dplyr::select(id)
  }
  f115_expect(df, pipe)
  # compute() runs the optimizer; pruning must keep the group and key columns
  for (p in f115_paths) {
    withr::local_options(cuplyr.topk_path = p)
    lz <- tbl_gpu(df, lazy = TRUE) |>
      dplyr::group_by(g) |>
      dplyr::slice_min(x, n = 2) |>
      dplyr::ungroup() |>
      dplyr::select(id)
    expect_equal(dplyr::collect(lz), dplyr::select(dplyr::ungroup(dplyr::slice_min(dplyr::group_by(df, g), x, n = 2)), id))
    expect_equal(dplyr::collect(dplyr::compute(lz)), dplyr::collect(lz))
  }
  f115_expect(df, function(d) {
    dplyr::arrange(dplyr::filter(dplyr::slice_min(dplyr::group_by(d, g), xnn, n = 5), inn > 3L), id)
  })
})

test_that("errors match dplyr", {
  skip_if_no_gpu()
  df <- f115_df()
  errs <- function(f) {
    list(
      cpu = tryCatch({ f(dplyr::group_by(df, g)); NULL }, error = conditionMessage),
      gpu = tryCatch({ f(dplyr::group_by(tbl_gpu(df, lazy = FALSE), g)); NULL }, error = conditionMessage)
    )
  }
  for (f in list(
    function(d) dplyr::slice_min(d, x, n = 1, prop = 0.5),
    function(d) dplyr::slice_max(d, x, n = c(1, 2)),
    function(d) dplyr::slice_min(d, x, by = g2)
  )) {
    e <- errs(f)
    expect_type(e$cpu, "character")
    expect_identical(e$gpu, e$cpu)
  }
})

test_that("large random tables match dplyr through the native route", {
  skip_if_no_gpu()
  set.seed(1150L)
  n <- 2e5L
  df <- tibble::tibble(
    id = seq_len(n),
    g = sample.int(1000L, n, replace = TRUE),
    x = replace(sample(seq(-50, 50), n, replace = TRUE) * 1.0, sample.int(n, n %/% 20L), NA),
    i = sample.int(500L, n, replace = TRUE)
  )
  for (p in c("auto", "sort")) {
    withr::local_options(cuplyr.topk_path = p)
    expect_same_as_dplyr(df, function(d) dplyr::slice_min(dplyr::group_by(d, g), x, n = 3))
    expect_same_as_dplyr(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), i, n = 1, with_ties = FALSE))
    expect_same_as_dplyr_lazy(df, function(d) dplyr::slice_max(d, x, prop = 0.01, by = g))
  }
})

test_that("the native route is chosen for supported keys", {
  skip_if_no_gpu()
  df <- f115_df()
  lz <- dplyr::group_by(tbl_gpu(df, lazy = TRUE), g)
  plans <- list(
    dplyr::slice_min(lz, x, n = 2),
    dplyr::slice_max(lz, i, n = 2, with_ties = FALSE),
    dplyr::slice_min(lz, d, prop = 0.5),
    dplyr::slice_max(lz, t, n = 1),
    dplyr::slice_min(lz, xnn + inn, n = 2),
    dplyr::slice_min(dplyr::group_by(tbl_gpu(df, lazy = TRUE), s, g2), x, n = 1),
    dplyr::slice_min(tbl_gpu(df, lazy = TRUE), x, n = 2, by = g)
  )
  for (p in plans) {
    plan <- f115_plan(.subset2(p, "lazy_ops"))
    expect_true("rank_grouped" %in% plan$mode, info = paste(plan$type, collapse = " < "))
    expect_false("window" %in% plan$type, info = paste(plan$type, collapse = " < "))
  }
})

test_that("the native route runs on an eager table too", {
  skip_if_no_gpu()
  df <- f115_df()
  calls <- 0L
  real <- gpu_slice_rank_grouped
  local_mocked_bindings(gpu_slice_rank_grouped = function(...) {
    calls <<- calls + 1L
    real(...)
  })
  invisible(dplyr::collect(dplyr::slice_min(dplyr::group_by(tbl_gpu(df, lazy = FALSE), g), x, n = 2)))
  expect_equal(calls, 1L)
  invisible(dplyr::collect(dplyr::slice_max(tbl_gpu(df, lazy = FALSE), i, n = 1, by = g)))
  expect_equal(calls, 2L)
})

test_that("unsupported keys and the sort option decline the route", {
  skip_if_no_gpu()
  df <- f115_df()
  lz <- dplyr::group_by(tbl_gpu(df, lazy = TRUE), g)
  declined <- list(
    dplyr::slice_min(lz, s, n = 2),
    dplyr::slice_min(lz, b, n = 2),
    dplyr::slice_min(lz, f, n = 2),
    dplyr::slice_min(lz, tibble::tibble(x, i), n = 2)
  )
  for (p in declined) {
    expect_false("rank_grouped" %in% f115_plan(.subset2(p, "lazy_ops"))$mode)
  }
  for (pipeline in list(
    function(d) dplyr::slice_min(dplyr::group_by(d, g), s, n = 2),
    function(d) dplyr::slice_max(dplyr::group_by(d, g), b, n = 2),
    function(d) dplyr::slice_min(dplyr::group_by(d, g), f, n = 2)
  )) {
    expect_same_as_dplyr(df, pipeline)
    expect_same_as_dplyr_lazy(df, pipeline)
  }

  withr::local_options(cuplyr.topk_path = "sort")
  plan <- f115_plan(.subset2(dplyr::slice_min(lz, x, n = 2), "lazy_ops"))
  expect_false("rank_grouped" %in% plan$mode)
  expect_true("window" %in% plan$type)
})
