# F14 acceptance tests: verbs that end with zero columns return a valid
# zero-column tbl_gpu with dplyr's row count, eager and lazy.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - select(df, -everything()), select(df) -> n x 0
# - transmute(df) -> n x 0; transmute(gdf) -> the grouping columns
# - distinct(<n x 0>) -> 1 x 0, or 0 x 0 when n == 0
# - distinct(df, NULL) -> 1 x 0 (NULL dots are dropped, leaving no key);
#   grouped: the distinct grouping columns
# - df[, 0], df[0], df[character()], df[integer()] -> n x 0; df[1:2, 0] -> 2 x 0;
#   gdf[integer()] -> ungrouped n x 0
# - select()/relocate()/distinct()/transmute() on an n x 0 input work
# Verbs that still can't take a zero-column input with rows must raise a
# clear error (F07's contract), never return a wrong row count.

f14z_df <- function() {
  data.frame(
    x = c(1L, 2L, 2L, NA, 5L),
    s = c("a", "b", "b", NA, "a"),
    v = c(1.5, NA, 2.5, 2.5, 0),
    stringsAsFactors = FALSE
  )
}

# Parity on values plus dim(): expect_equal() alone is lax about 0-column
# tibbles' row counts on some paths, so check dim() explicitly.
f14z_parity <- function(df, pipeline, lazy = TRUE) {
  r <- expect_same_as_dplyr(df, pipeline)
  expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs (eager)")
  if (lazy) {
    r <- expect_same_as_dplyr_lazy(df, pipeline)
    expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs (lazy)")
  }
}

# A zero-column tbl_gpu reports its row count without collecting.
f14z_expect_shape <- function(x, n) {
  expect_true(is_tbl_gpu(x))
  expect_identical(dim(x), c(as.integer(n), 0L))
  expect_identical(nrow(x), as.integer(n))
  out <- collect(x)
  expect_identical(dim(out), c(as.integer(n), 0L))
}

# =============================================================================
# select()
# =============================================================================

test_that("select() can drop every column and keeps the row count", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) select(d, -everything()))
  f14z_parity(df, function(d) select(d))
  f14z_parity(df, function(d) select(d, -x, -s, -v))
  f14z_parity(df, function(d) select(d, any_of(character())))
  f14z_parity(df, function(d) select(d, where(is.list)))
  f14z_parity(df[0, ], function(d) select(d, -everything()))
})

test_that("select() to zero columns after pending lazy ops counts their rows", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) d |> filter(x > 1L) |> select(-everything()))
  f14z_parity(df, function(d) d |> mutate(w = v * 2) |> filter(w > 1) |> select())
  f14z_parity(df, function(d) d |> arrange(v) |> slice(1:2) |> select(-everything()))
})

test_that("zero-column select() keeps the exec mode and reports its size", {
  skip_if_no_gpu()
  df <- f14z_df()
  e <- select(tbl_gpu(df), -everything())
  f14z_expect_shape(e, 5)
  expect_identical(e$exec_mode, "eager")
  l <- tbl_gpu(df, lazy = TRUE) |> filter(x > 1L) |> select(-everything())
  expect_identical(l$exec_mode, "lazy")
  f14z_expect_shape(l, 3)
  expect_identical(group_vars(select(tbl_gpu(df), -everything())), character())
})

test_that("select()/relocate() on a zero-column table keep its rows", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) d |> select(-everything()) |> select())
  f14z_parity(df, function(d) d |> select(-everything()) |> select(everything()))
  f14z_parity(df, function(d) d |> select(-everything()) |> relocate())
  f14z_parity(data.frame(row.names = 1:4), function(d) select(d, everything()))
})

# =============================================================================
# transmute()
# =============================================================================

test_that("transmute() with no expressions keeps the row count", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) transmute(d))
  f14z_parity(df[0, ], function(d) transmute(d))
  f14z_parity(df, function(d) d |> filter(!is.na(x)) |> transmute())
  f14z_parity(df, function(d) d |> select(-everything()) |> transmute())
})

test_that("transmute() with no expressions on grouped data keeps the groups", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) d |> group_by(s) |> transmute())
  f14z_parity(df, function(d) d |> group_by(s, x) |> transmute())
})

test_that("the conformance transmute() no-args case works", {
  skip_if_no_gpu()
  df <- data.frame(x = 1, y = 2)
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_equal(collect(transmute(g)), collect(g[integer()]))
    f14z_expect_shape(transmute(g), 1)
    gf <- group_by(g, x)
    expect_equal(collect(transmute(gf)), collect(gf[1L]))
  }
})

# =============================================================================
# distinct()
# =============================================================================

test_that("distinct() on a zero-column table gives one row (none if empty)", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) d |> select(-everything()) |> distinct())
  f14z_parity(df[0, ], function(d) d |> select(-everything()) |> distinct())
  f14z_parity(data.frame(row.names = 1:3), function(d) distinct(d))
  f14z_parity(data.frame(), function(d) distinct(d))
})

test_that("distinct() with only NULL keys gives one row", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) distinct(d, NULL))
  f14z_parity(df[0, ], function(d) distinct(d, NULL))
  f14z_parity(df, function(d) d |> group_by(s) |> distinct(NULL))
  f14z_parity(df, function(d) distinct(d, s, NULL))
  f14z_parity(df, function(d) distinct(d, NULL, x, .keep_all = TRUE))
})

test_that("the conformance distinct() zero-column cases work", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(data.frame(x = 1:10), lazy = lazy) |> select(-x)
    dd <- distinct(g)
    expect_identical(ncol(dd), 0L)
    f14z_expect_shape(dd, 1)

    d2 <- select(tbl_gpu(data.frame(x = c(1, 1)), lazy = lazy), any_of(character(0)))
    expect_identical(nrow(distinct(d2)), 1L)
    expect_identical(nrow(distinct(tbl_gpu(tibble::tibble(), lazy = lazy))), 0L)
  }
})

test_that("unique() of a tbl_gpu is distinct() over all columns", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 1, 1, 1), y = c(1, 1, 2, 2), z = c(1, 2, 1, 1))
  f14z_parity(df, function(d) unique(d))
  f14z_parity(df, function(d) unique(d[, "x", drop = FALSE]))
  f14z_parity(df, function(d) d |> group_by(y) |> unique())
})

test_that("unique() arguments it doesn't support error instead of being ignored", {
  # Added after round-1 review B2: base unique.data.frame() honours fromLast
  # and rejects incomparables; matching it or erroring are both fine.
  skip_if_no_gpu()
  u <- data.frame(k = c(1, 2, 1), w = c(5, 6, 5))
  expected <- unique(tibble::as_tibble(u), fromLast = TRUE)
  for (lazy in c(FALSE, TRUE)) {
    res <- tryCatch(collect(unique(tbl_gpu(u, lazy = lazy), fromLast = TRUE)),
                    error = function(e) e)
    if (!inherits(res, "error")) {
      expect_equal(tibble::as_tibble(res), expected)
    }
    expect_error(unique(tbl_gpu(u, lazy = lazy), incomparables = 5))
  }
})

# =============================================================================
# `[` with an empty column subscript
# =============================================================================

test_that("`[` can return zero columns", {
  skip_if_no_gpu()
  df <- f14z_df()
  f14z_parity(df, function(d) d[, 0])
  f14z_parity(df, function(d) d[0])
  f14z_parity(df, function(d) d[character()])
  f14z_parity(df, function(d) d[integer()])
  f14z_parity(df, function(d) d[, FALSE])
  f14z_parity(df, function(d) d[1:2, 0])
  f14z_parity(df, function(d) d[d$x > 1 & !is.na(d$x), integer()], lazy = FALSE)
  f14z_parity(df, function(d) group_by(d, s)[integer()])
  f14z_parity(df, function(d) d |> filter(v > 1) |> (\(t) t[, 0])())
})

test_that("`[` with a row subscript on a zero-column table counts the right rows", {
  # Added after round-2 review B3: must match tibble or error, never a
  # wrong row count.
  skip_if_no_gpu()
  z_tbl <- tibble::tibble(.rows = 5)
  cases <- list(
    function(t) t[-1, 0],
    function(t) t[-(1:2), character()],
    function(t) t[c(TRUE, FALSE, TRUE, FALSE, TRUE), 0],
    function(t) t[2:3, ]
  )
  for (p in cases) {
    expected <- p(z_tbl)
    for (lazy in c(FALSE, TRUE)) {
      z <- select(tbl_gpu(data.frame(a = 1:5), lazy = lazy), -everything())
      res <- tryCatch(collect(p(z)), error = function(e) e)
      if (inherits(res, "error")) {
        expect_match(conditionMessage(res), "no columns")
      } else {
        expect_identical(dim(res), dim(expected))
      }
    }
  }
})

# =============================================================================
# Verbs that don't support a zero-column input: error or correct, never wrong
# =============================================================================

test_that("other verbs on a zero-column table with rows error clearly or match dplyr", {
  skip_if_no_gpu()
  df <- f14z_df()
  pipelines <- list(
    function(d) mutate(d, k = 1),
    function(d) arrange(d),
    function(d) count(d),
    function(d) filter(d, dplyr::row_number() > 2L)
  )
  for (p in pipelines) {
    expected <- p(select(tibble::as_tibble(df), -everything()))
    for (lazy in c(FALSE, TRUE)) {
      z <- select(tbl_gpu(df, lazy = lazy), -everything())
      res <- tryCatch(collect(p(z)), error = function(e) e)
      if (inherits(res, "error")) {
        expect_match(conditionMessage(res), "no columns")
      } else {
        expect_identical(dim(res), dim(expected))
        expect_equal(tibble::as_tibble(dplyr::ungroup(res)), expected)
      }
    }
  }
})
