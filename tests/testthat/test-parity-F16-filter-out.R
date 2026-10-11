# F16 acceptance tests: filter_out().
#
# dplyr 1.2.1's filter_out() drops the rows where the AND of all its dots is
# TRUE and keeps the rows where it is FALSE or NA (keep iff
# !((c1 & c2 & ...) %in% TRUE)). Zero dots drop every row. Same engine as
# filter(): eager and lazy, grouped and `.by` window predicates, the CPU
# fallback routes, and dplyr's errors and warnings.
#
# Also covers filter()'s matrix/array/named-input checks (shared with
# filter_out(), see the F16 brief, goal 3).

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

f16_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3, NA),
    s = c("a", "b", NA, "a", "c", "b"),
    x = c(1, NA, 3, 4, 5, 2),
    i = c(10L, 20L, NA, 40L, 50L, 60L),
    y = c(TRUE, NA, FALSE, TRUE, TRUE, FALSE),
    f = factor(c("u", "v", "u", NA, "v", "u"), levels = c("u", "v", "w")),
    d = as.Date("2024-01-01") + c(0, 1, NA, 3, 4, 5)
  )
}

f16_parity <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f16_inputs <- function(df) {
  list(eager = tbl_gpu(df), lazy = as_lazy(tbl_gpu(df)))
}

# Both sides fail with the same condition message. `gpu` and `cpu` are
# unevaluated promises. The GPU side counts as failing if it errors at the
# verb call or when collected (a lazy table may defer the error).
f16_expect_same_error <- function(gpu, cpu, info = NULL) {
  c <- rlang::catch_cnd(cpu, "error")
  g <- rlang::catch_cnd(gpu, "error")
  expect_false(is.null(c), info = paste(info, "(dplyr must error)"))
  expect_false(is.null(g), info = paste(info, "(cuplyr must error)"))
  expect_s3_class(g, "rlang_error")
  if (!is.null(g) && !is.null(c)) {
    expect_identical(conditionMessage(g), conditionMessage(c), info = info)
  }
  invisible(g)
}

my_gt2 <- function(v) v > 2

# ---------------------------------------------------------------------------
# NA semantics, multiple dots, expression surface
# ---------------------------------------------------------------------------

test_that("filter_out() keeps rows where the condition is FALSE or NA", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(d, x > 2))
  f16_parity(df, function(d) dplyr::filter_out(d, y))
  f16_parity(df, function(d) dplyr::filter_out(d, !y))
  f16_parity(df, function(d) dplyr::filter_out(d, is.na(x)))
  f16_parity(df, function(d) dplyr::filter_out(d, i >= 40L))
  f16_parity(df, function(d) dplyr::filter_out(d, s == "a"))
  f16_parity(df, function(d) dplyr::filter_out(d, s %in% c("a", NA)))
  f16_parity(df, function(d) dplyr::filter_out(d, f == "u"))
  f16_parity(df, function(d) dplyr::filter_out(d, d > as.Date("2024-01-02")))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::between(x, 2, 4)))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::if_else(x > 2, y, NA)))
})

test_that("filter_out() ANDs all dots before taking the complement", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1, y))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1 & y))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1 | y))
  f16_parity(df, function(d) dplyr::filter_out(d, xor(x > 2, y)))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1, y, s != "c"))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1, !is.na(s)))
})

test_that("filter() and filter_out() are complements on all 3-valued combinations", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    x = c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, NA, NA, NA),
    y = c(TRUE, FALSE, NA, TRUE, FALSE, NA, TRUE, FALSE, NA),
    id = 1:9
  )
  f16_parity(df, function(d) dplyr::filter_out(d, x, y))
  f16_parity(df, function(d) dplyr::filter_out(d, x | y))
  f16_parity(df, function(d) dplyr::filter_out(d, !x))
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    kept <- c(collect(filter(t, x, y))$id, collect(dplyr::filter_out(t, x, y))$id)
    expect_identical(sort(kept), 1:9, info = mode)
  }
})

test_that("filter_out() with constant dots and no dots", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(d))
  f16_parity(df, function(d) dplyr::filter_out(d, TRUE))
  f16_parity(df, function(d) dplyr::filter_out(d, FALSE))
  f16_parity(df, function(d) dplyr::filter_out(d, NA))
  f16_parity(df, function(d) dplyr::filter_out(d, TRUE, TRUE))
  f16_parity(df, function(d) dplyr::filter_out(d, TRUE, x > 2))
  f16_parity(df, function(d) dplyr::filter_out(d, FALSE, x > 2))
  f16_parity(df, function(d) dplyr::filter_out(d, NA, x > 2))
  f16_parity(df, function(d) dplyr::filter_out(d, !!!list()))
})

test_that("filter_out() on zero-row, all-NA and zero-column inputs", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df[0, ], function(d) dplyr::filter_out(d, x > 2))
  f16_parity(df[0, ], function(d) dplyr::filter_out(group_by(d, g), x > mean(x)))
  df_na <- tibble::tibble(x = c(NA_real_, NA_real_), y = c(NA, NA))
  f16_parity(df_na, function(d) dplyr::filter_out(d, x > 0))
  f16_parity(df_na, function(d) dplyr::filter_out(d, y))

  zero_col <- function(d) select(d, -dplyr::everything())
  f16_parity(df, function(d) dplyr::filter_out(zero_col(d), TRUE))
  f16_parity(df, function(d) dplyr::filter_out(zero_col(d), FALSE))
  f16_parity(tibble::tibble(), function(d) dplyr::filter_out(d, TRUE))
  f16_parity(tibble::tibble(), function(d) dplyr::filter_out(d, FALSE))
})

# ---------------------------------------------------------------------------
# Grouped, .by, window predicates, across-family, pick, cur_group_id
# ---------------------------------------------------------------------------

test_that("grouped filter_out() evaluates window predicates per group", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g), x > mean(x, na.rm = TRUE)))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g), dplyr::n() > 1))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g), dplyr::row_number() == 1L))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, s), x > 1, cumsum(i) > 20L))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g, s), y))
  # .by: same per-group semantics, ungrouped result
  f16_parity(df, function(d) dplyr::filter_out(d, x > mean(x, na.rm = TRUE), .by = g))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::row_number() == 1L, .by = c(g, s)))
  # the conformance example
  df2 <- tibble::tibble(g = c(1, 1, 2, 1, 2), x = c(5, 10, 1, 2, 3))
  f16_parity(df2, function(d) dplyr::filter_out(d, x > mean(x), .by = g))
})

test_that("grouped filter_out() keeps the original row order", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 2, 1, 2, 1), time = 5:1, x = 5:1)
  f16_parity(df, function(d) arrange(dplyr::filter_out(group_by(d, g), x <= 2), time))
  f16_parity(df, function(d) dplyr::filter_out(arrange(group_by(d, g), time), x <= 2))
})

test_that("if_any(), if_all(), cur_group_id() and pick() inside filter_out()", {
  skip_if_no_gpu()
  df <- tibble::tibble(x1 = 1:10, x2 = c(1:5, 10:6))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::if_all(dplyr::starts_with("x"), ~ . > 6)))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::if_any(dplyr::starts_with("x"), ~ . > 6)))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::if_any(c(x1, x2), ~ .x > 8), x1 > 2))

  dfg <- f16_df()
  f16_parity(dfg, function(d) dplyr::filter_out(group_by(d, g), dplyr::cur_group_id() == 2L))

  dfp <- tibble::tibble(x = c(1, 2, NA, 3), y = c(2, NA, 5, 3))
  f16_parity(dfp, function(d) dplyr::filter_out(d, vctrs::vec_detect_complete(dplyr::pick(x, y))))
})

# ---------------------------------------------------------------------------
# CPU fallback routes
# ---------------------------------------------------------------------------

test_that("filter_out() with an expression the IR can't parse falls back correctly", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(d, my_gt2(x)))
  # mixed with a GPU-parseable dot: still one AND, then the complement
  f16_parity(df, function(d) dplyr::filter_out(d, my_gt2(x), y))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g), my_gt2(x)))
  f16_parity(df, function(d) dplyr::filter_out(d, my_gt2(x), .by = g))
})

test_that("filter_out() with a standalone logical vector", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(d, c(TRUE, FALSE, NA, TRUE, FALSE, NA)))
  f16_parity(df, function(d) dplyr::filter_out(d, c(TRUE, FALSE, NA, TRUE, FALSE, NA), x > 1))
})

test_that("a short logical vector is recycled per group (filter and filter_out)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 1, 2, 2), y = c(1, 2, 3, 4))
  f16_parity(df, function(d) filter(group_by(d, x), c(TRUE, FALSE)))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, x), c(TRUE, FALSE)))
})

# ---------------------------------------------------------------------------
# Lazy pipelines (the optimizer sees the filter_out node)
# ---------------------------------------------------------------------------

test_that("filter_out() composes with other verbs in lazy pipelines", {
  skip_if_no_gpu()
  df <- f16_df()
  f16_parity(df, function(d) select(dplyr::filter_out(mutate(d, z = x * 2), z > 4), z, s))
  f16_parity(df, function(d) dplyr::filter_out(filter(d, !is.na(s)), x > 2))
  f16_parity(df, function(d) filter(dplyr::filter_out(d, x > 4), y))
  f16_parity(df, function(d) dplyr::filter_out(dplyr::filter_out(d, x > 4), s == "a"))
  f16_parity(df, function(d) arrange(dplyr::filter_out(d, y), dplyr::desc(x)))
  f16_parity(df, function(d) summarise(dplyr::filter_out(group_by(d, g), x > 2), n = dplyr::n()),
             arrange_by = "g")
})

test_that("filter_out() after a join keeps the join's NA rows correctly", {
  skip_if_no_gpu()
  df <- f16_df()
  rhs <- tibble::tibble(s = c("a", "b", "z"), w = c(1, NA, 3))
  # same src on each side: a tbl_gpu joins a tbl_gpu, a tibble joins a tibble
  side <- function(d) if (is_tbl_gpu(d)) tbl_gpu(rhs) else rhs
  # predicate on the right-hand column: unmatched rows have w = NA and are kept
  f16_parity(df, function(d) dplyr::filter_out(left_join(d, side(d), by = "s"), w > 0),
             arrange_by = c("x", "i"))
  f16_parity(df, function(d) dplyr::filter_out(inner_join(d, side(d), by = "s"), w > 0),
             arrange_by = c("x", "i"))
  # predicate on the left-hand column
  f16_parity(df, function(d) dplyr::filter_out(left_join(d, side(d), by = "s"), x > 2),
             arrange_by = c("x", "i"))
})

# ---------------------------------------------------------------------------
# Errors and warnings (filter and filter_out)
# ---------------------------------------------------------------------------

test_that("filter() and filter_out() reject matrices with >1 column and >2-dim arrays", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    f16_expect_same_error(
      collect(filter(t, matrix(TRUE, nrow = 3, ncol = 2))),
      filter(df, matrix(TRUE, nrow = 3, ncol = 2)),
      info = paste(mode, "filter matrix")
    )
    f16_expect_same_error(
      collect(dplyr::filter_out(t, matrix(TRUE, nrow = 3, ncol = 2))),
      dplyr::filter_out(df, matrix(TRUE, nrow = 3, ncol = 2)),
      info = paste(mode, "filter_out matrix")
    )
    f16_expect_same_error(
      collect(filter(t, array(TRUE, dim = c(3, 1, 1)))),
      filter(df, array(TRUE, dim = c(3, 1, 1))),
      info = paste(mode, "filter array")
    )
    f16_expect_same_error(
      collect(dplyr::filter_out(t, array(TRUE, dim = c(3, 1, 1)))),
      dplyr::filter_out(df, array(TRUE, dim = c(3, 1, 1))),
      info = paste(mode, "filter_out array")
    )
  }
})

test_that("a 1-column matrix warns (deprecated) and works, ungrouped and grouped", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  m <- matrix(c(TRUE, FALSE), nrow = 2)
  df <- tibble::tibble(x = 1:2)
  dfg <- tibble::tibble(x = c(1, 1, 2, 2), y = c(1, 2, 3, 4))
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    expect_warning(out <- collect(filter(t, m)), class = "lifecycle_warning_deprecated")
    expect_equal(tibble::as_tibble(out), tibble::tibble(x = 1L), info = mode)
    expect_warning(out <- collect(dplyr::filter_out(t, m)), class = "lifecycle_warning_deprecated")
    expect_equal(tibble::as_tibble(out), tibble::tibble(x = 2L), info = mode)

    tg <- group_by(f16_inputs(dfg)[[mode]], x)
    expect_warning(out <- collect(filter(tg, m)), class = "lifecycle_warning_deprecated")
    expect_equal(tibble::as_tibble(dplyr::ungroup(out)), tibble::tibble(x = c(1, 2), y = c(1, 3)),
                 info = mode)
    expect_warning(out <- collect(dplyr::filter_out(tg, m)), class = "lifecycle_warning_deprecated")
    expect_equal(tibble::as_tibble(dplyr::ungroup(out)), tibble::tibble(x = c(1, 2), y = c(2, 4)),
                 info = mode)
  }
})

test_that("filter() and filter_out() reject named inputs and the `by` typo", {
  skip_if_no_gpu()
  df <- f16_df()
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    f16_expect_same_error(collect(filter(t, a = x > 1)), filter(df, a = x > 1),
                          info = paste(mode, "filter named"))
    f16_expect_same_error(collect(dplyr::filter_out(t, a = x > 1)),
                          dplyr::filter_out(df, a = x > 1),
                          info = paste(mode, "filter_out named"))
    f16_expect_same_error(collect(dplyr::filter_out(t, by = g)),
                          dplyr::filter_out(df, by = g),
                          info = paste(mode, "filter_out by typo"))
  }
})

test_that("filter_out() `.by` errors match dplyr", {
  skip_if_no_gpu()
  df <- f16_df()
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    f16_expect_same_error(collect(dplyr::filter_out(group_by(t, g), x > 1, .by = g)),
                          dplyr::filter_out(group_by(df, g), x > 1, .by = g),
                          info = paste(mode, ".by on grouped"))
    f16_expect_same_error(collect(dplyr::filter_out(t, x > 1, .by = g, .preserve = TRUE)),
                          dplyr::filter_out(df, x > 1, .by = g, .preserve = TRUE),
                          info = paste(mode, ".by with .preserve"))
  }
})

test_that("filter_out() with a non-logical or wrong-size condition errors", {
  skip_if_no_gpu()
  df <- f16_df()
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    expect_error(collect(dplyr::filter_out(t, x)), info = mode)
    expect_error(collect(dplyr::filter_out(t, c(TRUE, FALSE))), info = mode)
  }
})

# ---------------------------------------------------------------------------
# Added after round 1 review (B1, B2)
# ---------------------------------------------------------------------------

test_that("IR-parseable filter_out() calls run on the GPU, not the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f16_df()
  f16_parity(df, function(d) dplyr::filter_out(d, x > 2))
  f16_parity(df, function(d) dplyr::filter_out(d, y))
  f16_parity(df, function(d) dplyr::filter_out(d, !y))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1, y))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1 | y))
  f16_parity(df, function(d) dplyr::filter_out(d, x > 1, y, s != "c"))
  f16_parity(df, function(d) dplyr::filter_out(d, s %in% c("a", NA)))
  f16_parity(df, function(d) dplyr::filter_out(d, d > as.Date("2024-01-02")))
  f16_parity(df, function(d) dplyr::filter_out(d))
  f16_parity(df, function(d) dplyr::filter_out(d, TRUE))
  f16_parity(df, function(d) dplyr::filter_out(d, NA, x > 2))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g, s), y))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g), dplyr::n() > 1))
  f16_parity(df, function(d) dplyr::filter_out(d, dplyr::row_number() == 1L, .by = g))
  f16_parity(df, function(d) dplyr::filter_out(group_by(d, g), x > 2, .preserve = TRUE))
  f16_parity(df, function(d) filter(group_by(d, g), x > 2, .preserve = TRUE))
  df3 <- tibble::tibble(
    x = c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, NA, NA, NA),
    y = c(TRUE, FALSE, NA, TRUE, FALSE, NA, TRUE, FALSE, NA)
  )
  f16_parity(df3, function(d) dplyr::filter_out(d, x, y))
  f16_parity(df3, function(d) dplyr::filter_out(d, x | y))
  dfi <- tibble::tibble(x1 = 1:10, x2 = c(1:5, 10:6))
  f16_parity(dfi, function(d) dplyr::filter_out(d, dplyr::if_any(c(x1, x2), ~ .x > 6)))
})

test_that("a non-logical window condition errors in filter_out() and filter()", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(1, 0, -1, NA))
  for (mode in c("eager", "lazy")) {
    t <- f16_inputs(df)[[mode]]
    expect_error(collect(dplyr::filter_out(t, cumsum(x))), info = mode)
    expect_error(collect(dplyr::filter_out(group_by(t, g), mean(x))), info = mode)
    expect_error(collect(dplyr::filter_out(group_by(t, g), dplyr::n())), info = mode)
    expect_error(collect(filter(t, cumsum(x))), info = mode)
    expect_error(collect(filter(group_by(t, g), dplyr::n())), info = mode)
  }
})
