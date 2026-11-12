# F04 acceptance tests: dim()/nrow()/ncol() on lazy tables, coercion
# (as.data.frame, as_tibble, as.list, as.matrix, collect() keeping groups),
# rbind/cbind, and `[` resetting .drop when no group column survives.

f04d_df <- function() {
  data.frame(
    g = c("a", "b", "a", "c", "b"),
    x = c(1.5, 2, NA, 4.5, 0),
    i = c(1L, NA, 3L, 4L, 5L),
    s = c("p", "q", NA, "s", "t"),
    f = factor(c("u", "v", "u", NA, "v"), levels = c("u", "v", "w")),
    d = as.Date("2024-01-01") + c(0L, 1L, NA, 3L, 4L),
    l = c(TRUE, NA, FALSE, TRUE, FALSE),
    stringsAsFactors = FALSE
  )
}

f04d_up <- function(df = f04d_df(), lazy = FALSE) {
  tbl_gpu(df, lazy = lazy)
}

f04d_no_exec <- function(expr) {
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) stop("F04: unexpected plan execution"),
    compute = function(...) stop("F04: unexpected compute()"),
    gpu_collect = function(...) stop("F04: unexpected gpu_collect()"),
    .package = "cuplyr"
  )
  force(expr)
}

# ---------------------------------------------------------------------------
# dim / nrow / ncol

test_that("dim(), nrow(), NROW() of a lazy table reflect its pending ops", {
  skip_if_no_gpu()
  df <- f04d_df()
  pipelines <- list(
    function(t) dplyr::filter(t, i > 2L),
    function(t) dplyr::filter(t, i > 100L),
    function(t) dplyr::mutate(dplyr::filter(t, !is.na(x)), y = x * 2),
    function(t) dplyr::summarise(dplyr::group_by(t, g), n = dplyr::n()),
    function(t) dplyr::select(dplyr::slice_head(t, n = 2), g, x),
    function(t) dplyr::distinct(t, g)
  )
  for (p in pipelines) {
    label <- paste(deparse(body(p)), collapse = " ")
    oracle <- p(tibble::as_tibble(df))
    lz <- p(f04d_up(df, lazy = TRUE))
    eg <- p(f04d_up(df, lazy = FALSE))
    expect_identical(dim(lz), dim(oracle), info = label)
    expect_identical(nrow(lz), nrow(oracle), info = label)
    expect_identical(NROW(lz), NROW(oracle), info = label)
    expect_identical(ncol(lz), ncol(oracle), info = label)
    expect_identical(dim(eg), dim(oracle), info = label)
  }
})

test_that("nrow() on a lazy table does not change the caller's object", {
  skip_if_no_gpu()
  lzf <- dplyr::filter(f04d_up(lazy = TRUE), i > 2L)
  before <- unclass(lzf)
  n <- nrow(lzf)
  expect_identical(n, 3L)
  after <- unclass(lzf)
  expect_identical(after$lazy_ops, before$lazy_ops)
  expect_identical(after$schema, before$schema)
  expect_identical(after$exec_mode, "lazy")
  expect_identical(nrow(lzf), 3L)
  expect_identical(nrow(collect(lzf)), 3L)
})

test_that("ncol() and length() of a lazy table come from the schema without executing", {
  skip_if_no_gpu()
  lz <- dplyr::mutate(dplyr::filter(f04d_up(lazy = TRUE), i > 2L), y = x + 1, z = i * 2L)
  expect_identical(f04d_no_exec(ncol(lz)), 9L)
  expect_identical(f04d_no_exec(NCOL(lz)), 9L)
  expect_identical(f04d_no_exec(length(lz)), 9L)
  lz2 <- dplyr::select(dplyr::filter(f04d_up(lazy = TRUE), i > 2L), g, x)
  expect_identical(f04d_no_exec(ncol(lz2)), 2L)
})

test_that("dim() of an eager table or a lazy table without pending ops does not execute", {
  skip_if_no_gpu()
  expect_identical(f04d_no_exec(dim(f04d_up())), c(5L, 7L))
  expect_identical(f04d_no_exec(dim(f04d_up(lazy = TRUE))), c(5L, 7L))
  expect_identical(f04d_no_exec(nrow(f04d_up(lazy = TRUE))), 5L)
})

test_that("x[nrow(x), ] on a lazy table with pending ops picks the right row", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- dplyr::filter(tibble::as_tibble(df), i > 2L)
  lzf <- dplyr::filter(f04d_up(df, lazy = TRUE), i > 2L)
  expect_identical(tibble::as_tibble(collect(lzf[nrow(lzf), ])), oracle[nrow(oracle), ])
  expect_identical(
    tibble::as_tibble(collect(lzf[seq_len(nrow(lzf) - 1L), "s"])),
    oracle[seq_len(nrow(oracle) - 1L), "s"]
  )
})

# ---------------------------------------------------------------------------
# coercion

test_that("as.data.frame() returns the base data.frame tibble would", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    expect_identical(as.data.frame(gt), as.data.frame(oracle))
    expect_identical(
      as.data.frame(dplyr::group_by(gt, g, .drop = FALSE)),
      as.data.frame(dplyr::group_by(oracle, g, .drop = FALSE))
    )
    p <- function(t) dplyr::mutate(dplyr::filter(t, i > 2L), y = i * 2L)
    expect_identical(as.data.frame(p(gt)), as.data.frame(p(oracle)))
    expect_identical(as.data.frame(p(gt)[0, ]), as.data.frame(p(oracle)[0, ]))
  }
})

test_that("as_tibble() works on grouped and lazy tables and drops the groups", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    expect_identical(tibble::as_tibble(gt), oracle)
    expect_identical(dplyr::as_tibble(gt), oracle)
    expect_identical(
      tibble::as_tibble(dplyr::group_by(gt, g, l)),
      tibble::as_tibble(dplyr::group_by(oracle, g, l))
    )
    p <- function(t) dplyr::arrange(dplyr::filter(t, i > 2L), dplyr::desc(i))
    expect_identical(tibble::as_tibble(p(gt)), p(oracle))
  }
})

test_that("as.list(), lapply() and sapply() see the columns", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    expect_identical(as.list(gt), as.list(oracle))
    expect_identical(lapply(gt, class), lapply(oracle, class))
    expect_identical(sapply(gt, function(v) sum(is.na(v))), sapply(oracle, function(v) sum(is.na(v))))
    lzf <- dplyr::filter(gt, i > 2L)
    expect_identical(as.list(lzf), as.list(dplyr::filter(oracle, i > 2L)))
  }
})

test_that("as.matrix() matches tibble", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    expect_identical(as.matrix(dplyr::select(gt, x, i)), as.matrix(dplyr::select(oracle, x, i)))
    expect_identical(as.matrix(gt), as.matrix(oracle))
  }
})

test_that("collect() keeps the groups of a grouped tbl_gpu, as dplyr/dbplyr collect() do", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle_in <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    pipelines <- list(
      function(t) dplyr::group_by(t, g),
      function(t) dplyr::group_by(t, g, .drop = FALSE),
      function(t) dplyr::group_by(t, f, .drop = FALSE),
      function(t) dplyr::filter(dplyr::group_by(t, g, l), i > 1L),
      function(t) dplyr::arrange(dplyr::summarise(dplyr::group_by(t, g, l), n = dplyr::n()), g, l),
      function(t) dplyr::mutate(dplyr::group_by(t, g), m = mean(x, na.rm = TRUE)),
      function(t) dplyr::filter(dplyr::group_by(t, g), i > 100L)
    )
    for (p in pipelines) {
      label <- paste(if (lazy) "lazy" else "eager", paste(deparse(body(p)), collapse = " "))
      out <- collect(p(gt))
      expected <- p(oracle_in)
      expect_s3_class(out, "grouped_df")
      expect_identical(out, expected, info = label)
    }
    # ungrouped stays a plain tibble
    expect_identical(collect(dplyr::ungroup(dplyr::group_by(gt, g))), oracle_in)
    # (the input is a plain data.frame, which stays one: F31)
    expect_identical(class(collect(gt)), class(df))
  }
})

# ---------------------------------------------------------------------------
# rbind / cbind

test_that("rbind() of tbl_gpus row-binds on the GPU like rbind() of tibbles", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    out <- rbind(gt, gt)
    expect_true(is_tbl_gpu(out))
    expect_identical(tibble::as_tibble(collect(out)), rbind(oracle, oracle))
    out3 <- rbind(gt, dplyr::filter(gt, i > 2L), gt)
    expect_identical(
      tibble::as_tibble(collect(out3)),
      rbind(oracle, dplyr::filter(oracle, i > 2L), oracle)
    )
    out_df <- rbind(gt, df[1:2, ])
    expect_true(is_tbl_gpu(out_df))
    expect_identical(tibble::as_tibble(collect(out_df)), rbind(oracle, df[1:2, ]))
    gd <- dplyr::group_by(gt, g)
    ogd <- dplyr::group_by(oracle, g)
    out_g <- rbind(gd, gd)
    expect_identical(dplyr::group_vars(out_g), dplyr::group_vars(rbind(ogd, ogd)))
    expect_identical(collect(out_g), rbind(ogd, ogd))
  }
})

test_that("cbind() of tbl_gpus column-binds on the GPU", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- tibble::as_tibble(df)
  other <- data.frame(y = 5:1, w = letters[1:5], stringsAsFactors = FALSE)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    out <- cbind(gt, f04d_up(other, lazy))
    expect_true(is_tbl_gpu(out))
    expect_identical(
      tibble::as_tibble(collect(out)),
      tibble::as_tibble(cbind(oracle, tibble::as_tibble(other)))
    )
    out2 <- cbind(dplyr::select(gt, g, x), other)
    expect_true(is_tbl_gpu(out2))
    expect_identical(
      tibble::as_tibble(collect(out2)),
      tibble::as_tibble(cbind(dplyr::select(oracle, g, x), other))
    )
  }
})

test_that("rbind()/cbind() never return a table built from internal fields", {
  skip_if_no_gpu()
  gt <- f04d_up()
  for (out in list(rbind(gt, gt), cbind(gt, dplyr::rename(gt, g2 = g, x2 = x, i2 = i, s2 = s, f2 = f, d2 = d, l2 = l)))) {
    expect_true(is_tbl_gpu(out))
    expect_false(any(c("ptr", "schema", "lazy_ops") %in% names(out)))
  }
})

# ---------------------------------------------------------------------------
# `[` dropping every group column (F03 reviewer N1)

test_that("x[, j] that drops every group column resets .drop like dplyr", {
  skip_if_no_gpu()
  df <- f04d_df()
  oracle <- dplyr::group_by(tibble::as_tibble(df), g, .drop = FALSE)
  for (lazy in c(FALSE, TRUE)) {
    gd <- dplyr::group_by(f04d_up(df, lazy), g, .drop = FALSE)
    for (fn in list(function(t) t[, "x"], function(t) t["x"], function(t) t[2:3, c("x", "s")])) {
      out <- fn(gd)
      o <- fn(oracle)
      expect_identical(dplyr::group_vars(out), dplyr::group_vars(o))
      expect_identical(dplyr::group_by_drop_default(out), dplyr::group_by_drop_default(o))
      expect_identical(dplyr::group_by_drop_default(dplyr::group_by(out, x)),
                       dplyr::group_by_drop_default(dplyr::group_by(o, x)))
    }
    kept <- gd[, c("g", "x")]
    expect_identical(dplyr::group_by_drop_default(kept), FALSE)
  }
})

# ---------------------------------------------------------------------------
# Added after round 1 review (B3)

test_that("collect() keeps .drop = FALSE through verbs after group_by()", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("a", "b", "a", "b"), levels = c("a", "b", "c")),
    x = c(3L, 1L, 2L, 5L),
    s = c("p", "q", "r", "s"),
    stringsAsFactors = FALSE
  )
  oracle_in <- tibble::as_tibble(df)
  pipelines <- list(
    function(t) dplyr::arrange(dplyr::group_by(t, f, .drop = FALSE), x),
    function(t) dplyr::mutate(dplyr::group_by(t, f, .drop = FALSE), y = x * 2L),
    function(t) dplyr::filter(dplyr::group_by(t, f, .drop = FALSE), x > 1L),
    function(t) dplyr::select(dplyr::group_by(t, f, .drop = FALSE), f, x),
    function(t) dplyr::rename(dplyr::group_by(t, f, .drop = FALSE), ff = f),
    function(t) dplyr::arrange(dplyr::mutate(dplyr::group_by(t, f, .drop = FALSE), y = -x), y)
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04d_up(df, lazy)
    for (p in pipelines) {
      label <- paste(if (lazy) "lazy" else "eager", paste(deparse(body(p)), collapse = " "))
      out <- p(gt)
      expected <- p(oracle_in)
      expect_identical(dplyr::group_by_drop_default(out), FALSE, info = label)
      expect_identical(collect(out), expected, info = label)
      expect_identical(
        dplyr::summarise(collect(out), n = dplyr::n()),
        dplyr::summarise(expected, n = dplyr::n()),
        info = label
      )
    }
  }
})

# ---------------------------------------------------------------------------
# Added after round 2 review (N12): .drop = FALSE survives the verbs that
# build their result without push_op(), so collect() hands dplyr the right flag.

test_that("count(), add_count(), slice_sample() and regrouping keep .drop = FALSE", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("a", "b", "a", "b"), levels = c("a", "b", "c")),
    g = c("p", "q", "p", "p"),
    x = c(3L, 1L, 2L, 5L),
    stringsAsFactors = FALSE
  )
  oracle_in <- dplyr::group_by(tibble::as_tibble(df), f, .drop = FALSE)
  pipelines <- list(
    function(t) dplyr::count(t, g),
    function(t) dplyr::add_count(t, g),
    function(t) dplyr::slice_sample(t, n = 1),
    function(t) dplyr::group_by(t, f, g),
    function(t) dplyr::group_by(t, g, .add = TRUE)
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- dplyr::group_by(f04d_up(df, lazy), f, .drop = FALSE)
    for (p in pipelines) {
      label <- paste(if (lazy) "lazy" else "eager", paste(deparse(body(p)), collapse = " "))
      out <- p(gt)
      expected <- p(oracle_in)
      expect_identical(dplyr::group_vars(out), dplyr::group_vars(expected), info = label)
      expect_identical(dplyr::group_by_drop_default(out), dplyr::group_by_drop_default(expected), info = label)
      expect_identical(dplyr::n_groups(collect(out)), dplyr::n_groups(expected), info = label)
    }
  }
})

# ---------------------------------------------------------------------------
# Added after round 3 review (B5): an ungrouped table reports .drop = TRUE,
# so regrouping it does not inherit a stale flag.

test_that("a join that removes every group column leaves .drop = TRUE for the next group_by()", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "b"), levels = c("a", "b", "c")), k = 1:2)
  y <- data.frame(k = 1:2, f = c("u", "v"), stringsAsFactors = FALSE)
  p_join <- function(t, u) dplyr::left_join(dplyr::group_by(t, f, .drop = FALSE), u, by = "k")
  oj <- p_join(tibble::as_tibble(df), tibble::as_tibble(y))
  for (lazy in c(FALSE, TRUE)) {
    j <- p_join(f04d_up(df, lazy), f04d_up(y, lazy))
    expect_identical(dplyr::group_vars(j), dplyr::group_vars(oj))
    expect_identical(dplyr::group_by_drop_default(j), dplyr::group_by_drop_default(oj))
    o <- dplyr::group_by(j, f.x)
    oo <- dplyr::group_by(oj, f.x)
    expect_identical(dplyr::group_by_drop_default(o), dplyr::group_by_drop_default(oo))
    expect_identical(dplyr::n_groups(o), dplyr::n_groups(oo))
    expect_identical(collect(o), oo)
    expect_identical(dplyr::summarise(collect(o), n = dplyr::n()), dplyr::summarise(oo, n = dplyr::n()))
    # ungrouping explicitly also reports TRUE
    expect_identical(
      dplyr::group_by_drop_default(dplyr::ungroup(dplyr::group_by(f04d_up(df, lazy), f, .drop = FALSE))),
      TRUE
    )
  }
})
