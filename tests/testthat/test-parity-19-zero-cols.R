# Issue #19 acceptance tests: a table with rows but no columns (N x 0) keeps
# its row count through verbs, collect() and print(), eager and lazy.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - mutate() that leaves no column (`col = NULL` for every column, `.keep =
#   "none"`/"used"` with nothing left, with or without `.by`) -> n x 0,
#   ungrouped; dropping a group_by() column with NULL is an error.
# - filter()/filter_out() on n x 0: any mask (literal, env vector,
#   row_number(), n()) selects rows of the n.
# - cross_join(): n x 0 by m x 0 -> (n * m) x 0; n x 0 by y -> y repeated n
#   times (vec_rep(y, n)); x by m x 0 -> each x row m times (vec_rep_each),
#   x's groups kept; a 0-row side gives 0 rows.
# - `by = character()` joins (deprecated cross spelling) with an n x 0 side:
#   mutating joins are cross joins; semi_join() keeps all of x when y has rows,
#   none otherwise; anti_join() the reverse.
# - Set operations of two zero-column tables: intersect() -> 1 x 0 when both
#   have rows; setdiff() -> 1 x 0 when only x has rows; symdiff() -> 1 x 0 when
#   exactly one has rows; union() -> 1 x 0 unless both are empty; union_all()
#   adds the counts; setequal() compares "has rows". A zero-column and a
#   non-zero-column table are incompatible (error).
# - vctrs: vec_size()/vec_slice()/vec_rep()/vec_ptype() of an n x 0 table.
# - reframe() with a named zero-column data frame result recycles it like any
#   other result (dplyr #6509), also when that data frame is a tbl_gpu.
# - print(): the header shows the n x 0 dimensions.

p19_df <- function() {
  tibble::tibble(
    x = c(1L, 2L, NA, 4L),
    g = c("a", "a", "b", NA),
    v = c(1.5, NA, 2.5, 0)
  )
}

p19_zero <- function(n) tibble::new_tibble(list(), nrow = n)

# A second table for a two-table pipeline, in the same form as `d` (a tibble
# for the dplyr oracle, a tbl_gpu in d's exec mode on the GPU side).
p19_like <- function(d, y) {
  if (is_tbl_gpu(d)) {
    tbl_gpu(y, lazy = identical(.subset2(d, "exec_mode"), "lazy"))
  } else {
    tibble::as_tibble(y)
  }
}

# Parity eager and lazy, plus dim(): expect_equal() is lax about the row count
# of zero-column tibbles, so it is checked on its own.
p19_parity <- function(df, pipeline, ...) {
  r <- expect_same_as_dplyr(df, pipeline, ...)
  expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs (eager)")
  r <- expect_same_as_dplyr_lazy(df, pipeline, ...)
  expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs (lazy)")
}

# A result as plain data: a tbl_gpu is collected, and so is a tbl_gpu
# data-frame column of a plain tibble.
p19_plain <- function(x) {
  if (is_tbl_gpu(x)) {
    x <- collect(x)
  }
  for (nm in names(x)) {
    if (is_tbl_gpu(x[[nm]])) x[[nm]] <- collect(x[[nm]])
  }
  x
}

# A zero-column tbl_gpu that reports `n` rows without collecting, and collects
# to n x 0.
p19_expect_shape <- function(x, n) {
  expect_true(is_tbl_gpu(x))
  expect_identical(dim(x), c(as.integer(n), 0L))
  expect_identical(nrow(x), as.integer(n))
  expect_identical(dim(collect(x)), c(as.integer(n), 0L))
}

# =============================================================================
# mutate() / transmute() that leave no column
# =============================================================================

test_that("mutate() can drop every column and keeps the row count", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL))
  p19_parity(df, function(d) mutate(d, .keep = "none"))
  p19_parity(df, function(d) mutate(d, .keep = "used"))
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL, .keep = "used"))
  p19_parity(df, function(d) mutate(d, y = x + 1L, .keep = "none") |> mutate(y = NULL))
  p19_parity(df, function(d) mutate(d, s = x + v, .keep = "unused") |> mutate(g = NULL, s = NULL))
  p19_parity(df[0, ], function(d) mutate(d, x = NULL, g = NULL, v = NULL))
  p19_parity(df[0, ], function(d) mutate(d, .keep = "none"))
})

test_that("mutate() can NULL out the `.by` columns", {
  skip_if_no_gpu()
  # dplyr's own test (test-mutate.R, "can `NULL` out the `.by` column")
  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(tibble::tibble(x = 1:3), lazy = lazy)
    out <- mutate(gdf, x = NULL, .by = x)
    p19_expect_shape(out, 3)
    expect_identical(collect(out), tibble::new_tibble(list(), nrow = 3))
    expect_identical(group_vars(out), character())
  }
  df <- p19_df()
  p19_parity(tibble::tibble(x = 1:3), function(d) mutate(d, x = NULL, .by = x))
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL, .by = g))
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL, .by = c(g, x)))
  p19_parity(df, function(d) mutate(d, .keep = "none", .by = g) |> mutate(g = NULL))
})

test_that("mutate() to zero columns after pending lazy ops counts their rows", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) d |> filter(!is.na(x)) |> mutate(x = NULL, g = NULL, v = NULL))
  p19_parity(df, function(d) d |> arrange(v) |> slice(1:2) |> mutate(.keep = "none"))
  p19_parity(df, function(d) d |> mutate(w = v * 2) |> filter(w > 1) |>
    mutate(x = NULL, g = NULL, v = NULL, w = NULL))
})

test_that("a zero-column mutate() result keeps the exec mode and works with later verbs", {
  skip_if_no_gpu()
  df <- p19_df()
  e <- mutate(tbl_gpu(df, lazy = FALSE), x = NULL, g = NULL, v = NULL)
  p19_expect_shape(e, 4)
  expect_identical(.subset2(e, "exec_mode"), "eager")
  l <- tbl_gpu(df, lazy = TRUE) |> filter(!is.na(v)) |> mutate(.keep = "none")
  expect_identical(.subset2(l, "exec_mode"), "lazy")
  p19_expect_shape(l, 3)

  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL) |> mutate(b = 2L))
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL) |> summarise(n = n()))
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL) |> count())
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL) |> distinct())
  p19_parity(df, function(d) mutate(d, x = NULL, g = NULL, v = NULL) |> slice_head(n = 2))
})

test_that("mutate() keeps group_by() columns, and dropping one is an error as in dplyr", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) group_by(d, g) |> mutate(x = NULL, v = NULL))
  p19_parity(df, function(d) group_by(d, g) |> mutate(.keep = "none"))
  expect_error(mutate(group_by(df, g), g = NULL))
  for (lazy in c(FALSE, TRUE)) {
    expect_error(mutate(group_by(tbl_gpu(df, lazy = lazy), g), g = NULL, x = NULL, v = NULL))
  }
})

test_that("transmute() that leaves no column keeps the row count", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) transmute(d, x = NULL))
  p19_parity(df, function(d) transmute(d, x = NULL, g = NULL))
  p19_parity(df, function(d) d |> filter(!is.na(g)) |> transmute(v = NULL))
})

# =============================================================================
# filter() / filter_out() on a zero-column table
# =============================================================================

test_that("filter() on a zero-column table selects rows of its count", {
  skip_if_no_gpu()
  z <- p19_zero(4)
  keep <- c(TRUE, FALSE, NA, TRUE)
  p19_parity(z, function(d) filter(d, TRUE))
  p19_parity(z, function(d) filter(d, FALSE))
  p19_parity(z, function(d) filter(d, NA))
  p19_parity(z, function(d) filter(d, keep))
  p19_parity(z, function(d) filter(d, dplyr::row_number() > 1L))
  p19_parity(z, function(d) filter(d, dplyr::n() > 2L))
  p19_parity(z, function(d) filter(d, dplyr::row_number() %% 2L == 0L, TRUE))
  p19_parity(z, function(d) filter(d, dplyr::row_number() > 1L) |> filter(dplyr::row_number() > 1L))
  p19_parity(z, function(d) filter_out(d, dplyr::row_number() == 1L))
  p19_parity(p19_zero(0), function(d) filter(d, dplyr::row_number() > 1L))
  p19_parity(p19_df(), function(d) select(d, -everything()) |> filter(dplyr::row_number() <= 2L))
})

test_that("filter() on a zero-column table rejects a mask of the wrong size", {
  skip_if_no_gpu()
  z <- p19_zero(4)
  expect_error(filter(z, c(TRUE, FALSE)), class = "rlang_error")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(filter(tbl_gpu(z, lazy = lazy), c(TRUE, FALSE))), class = "rlang_error")
  }
})

# =============================================================================
# cross_join()
# =============================================================================

test_that("cross_join() of zero-column tables multiplies their row counts", {
  skip_if_no_gpu()
  z <- p19_zero(3)
  p19_parity(z, function(d) cross_join(d, d))
  p19_parity(z, function(d) cross_join(d, p19_like(d, p19_zero(2))))
  p19_parity(z, function(d) cross_join(d, p19_like(d, p19_zero(0))))
  p19_parity(p19_zero(0), function(d) cross_join(d, p19_like(d, p19_zero(3))))
})

test_that("cross_join() of a zero-column x repeats y once per row of x", {
  skip_if_no_gpu()
  y <- tibble::tibble(
    k = c(2L, NA, 7L),
    f = factor(c("u", "w", "u"), levels = c("w", "u")),
    dt = as.Date(c("2024-01-02", NA, "1999-12-31")),
    s = c("p", "q", NA)
  )
  p19_parity(p19_zero(3), function(d) cross_join(d, p19_like(d, y)))
  p19_parity(p19_zero(1), function(d) cross_join(d, p19_like(d, y)))
  p19_parity(p19_zero(0), function(d) cross_join(d, p19_like(d, y)))
  p19_parity(p19_zero(2), function(d) cross_join(d, p19_like(d, y[0, ])))
})

test_that("cross_join() with a zero-column y repeats each row of x", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) cross_join(d, p19_like(d, p19_zero(3))))
  p19_parity(df, function(d) cross_join(d, p19_like(d, p19_zero(1))))
  p19_parity(df, function(d) cross_join(d, p19_like(d, p19_zero(0))))
  p19_parity(df[0, ], function(d) cross_join(d, p19_like(d, p19_zero(3))))
  p19_parity(df, function(d) cross_join(d, p19_like(d, p19_zero(2)), suffix = c("_l", "_r")))
  # x's groups survive, as in dplyr
  p19_parity(df, function(d) cross_join(group_by(d, g), p19_like(d, p19_zero(2))))
  # pending lazy ops on x, and a zero-column side made by select()
  p19_parity(df, function(d) d |> filter(!is.na(x)) |> cross_join(p19_like(d, p19_zero(2))))
  p19_parity(df, function(d) cross_join(select(d, -everything()), d))
  p19_parity(df, function(d) cross_join(d, select(d, -everything())))
})

test_that("cross_join() matches dplyr's own zero-column test", {
  skip_if_no_gpu()
  # test-join-cross.R, "cross join works with 0 column, >0 row tibbles"
  for (lazy in c(FALSE, TRUE)) {
    df1 <- tbl_gpu(tibble::new_tibble(list(), nrow = 3), lazy = lazy)
    df2 <- tbl_gpu(tibble::tibble(x = 1:2), lazy = lazy)
    out <- cross_join(df1, df1)
    p19_expect_shape(out, 9)
    expect_identical(collect(out), tibble::new_tibble(list(), nrow = 9))
    expect_identical(collect(cross_join(df1, df2)), vctrs::vec_rep(tibble::tibble(x = 1:2), times = 3))
    expect_identical(.subset2(cross_join(df1, df2), "exec_mode"), if (lazy) "lazy" else "eager")
  }
  # a plain data frame y with copy = TRUE
  z <- tbl_gpu(p19_zero(2))
  expect_identical(
    collect(cross_join(z, tibble::tibble(x = 1:2), copy = TRUE)),
    vctrs::vec_rep(tibble::tibble(x = 1:2), times = 2)
  )
})

test_that("cross_join() with a zero-column side still validates its arguments", {
  skip_if_no_gpu()
  z <- p19_zero(3)
  d <- tibble::tibble(x = 1:2)
  expect_error(cross_join(z, d, suffix = "a"), class = "rlang_error")
  expect_error(cross_join(tbl_gpu(z), tbl_gpu(d), suffix = "a"), class = "rlang_error")
  expect_error(cross_join(tbl_gpu(z), tbl_gpu(d), 1), class = "rlib_error_dots_nonempty")
})

# =============================================================================
# `by = character()` joins with a zero-column side (dplyr's deprecated cross
# spelling; the deprecation warning is not under test)
# =============================================================================

test_that("mutating joins with `by = character()` and a zero-column side are cross joins", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) suppressWarnings(left_join(d, p19_like(d, p19_zero(3)), by = character())))
  p19_parity(df, function(d) suppressWarnings(inner_join(d, p19_like(d, p19_zero(2)), by = character())))
  p19_parity(df, function(d) suppressWarnings(full_join(d, p19_like(d, p19_zero(2)), by = character())))
  p19_parity(df, function(d) suppressWarnings(right_join(d, p19_like(d, p19_zero(2)), by = character())))
  z <- p19_zero(3)
  p19_parity(z, function(d) suppressWarnings(inner_join(d, p19_like(d, p19_df()), by = character())))
  p19_parity(z, function(d) suppressWarnings(left_join(d, d, by = character())))
})

test_that("filtering joins with `by = character()` and a zero-column side keep all rows or none", {
  skip_if_no_gpu()
  df <- p19_df()
  p19_parity(df, function(d) suppressWarnings(semi_join(d, p19_like(d, p19_zero(3)), by = character())))
  p19_parity(df, function(d) suppressWarnings(semi_join(d, p19_like(d, p19_zero(0)), by = character())))
  p19_parity(df, function(d) suppressWarnings(anti_join(d, p19_like(d, p19_zero(3)), by = character())))
  p19_parity(df, function(d) suppressWarnings(anti_join(d, p19_like(d, p19_zero(0)), by = character())))
  z <- p19_zero(3)
  p19_parity(z, function(d) suppressWarnings(semi_join(d, p19_like(d, p19_df()), by = character())))
  p19_parity(z, function(d) suppressWarnings(anti_join(d, p19_like(d, p19_zero(0)), by = character())))
  p19_parity(z, function(d) suppressWarnings(anti_join(d, d, by = character())))
})

# =============================================================================
# Set operations of zero-column tables
# =============================================================================

test_that("set operations of zero-column tables follow dplyr", {
  skip_if_no_gpu()
  for (nx in c(3L, 0L)) {
    for (ny in c(2L, 0L)) {
      x <- p19_zero(nx)
      yy <- p19_zero(ny)
      p19_parity(x, function(d) intersect(d, p19_like(d, yy)))
      p19_parity(x, function(d) setdiff(d, p19_like(d, yy)))
      p19_parity(x, function(d) symdiff(d, p19_like(d, yy)))
      p19_parity(x, function(d) union(d, p19_like(d, yy)))
      p19_parity(x, function(d) union_all(d, p19_like(d, yy)))
      for (lazy in c(FALSE, TRUE)) {
        expect_identical(
          setequal(tbl_gpu(x, lazy = lazy), tbl_gpu(yy, lazy = lazy)),
          dplyr::setequal(x, yy)
        )
      }
    }
  }
})

test_that("set operations between a zero-column and a non-zero-column table error", {
  skip_if_no_gpu()
  z <- p19_zero(3)
  d <- tibble::tibble(x = 1:2)
  expect_error(intersect(z, d), class = "rlang_error")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(intersect(tbl_gpu(z, lazy = lazy), tbl_gpu(d, lazy = lazy)), class = "rlang_error")
    expect_error(setdiff(tbl_gpu(z, lazy = lazy), tbl_gpu(d, lazy = lazy)), class = "rlang_error")
  }
})

# =============================================================================
# vctrs on a zero-column tbl_gpu
# =============================================================================

test_that("vctrs sizes, slices and repeats a zero-column tbl_gpu by its row count", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    z <- tbl_gpu(p19_zero(3), lazy = lazy)
    expect_identical(vctrs::vec_size(z), 3L)
    s <- vctrs::vec_slice(z, c(1L, 3L))
    expect_true(is_tbl_gpu(s))
    expect_identical(collect(s), p19_zero(2))
    expect_identical(collect(vctrs::vec_slice(z, integer())), p19_zero(0))
    expect_identical(collect(vctrs::vec_rep(z, times = 2)), p19_zero(6))
    expect_identical(collect(vctrs::vec_rep_each(z, times = 2)), p19_zero(6))
    expect_identical(collect(vctrs::vec_ptype(z)), p19_zero(0))

    z0 <- tbl_gpu(tibble::tibble(), lazy = lazy)
    expect_identical(vctrs::vec_size(z0), 0L)
    expect_identical(collect(vctrs::vec_ptype(z0)), tibble::tibble())
  }
})

# =============================================================================
# reframe() with zero-column data frame results (dplyr #6509)
# =============================================================================

test_that("reframe() recycles a named zero-column data frame result", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  p19_parity(df, function(d) reframe(d, empty = tibble::tibble()))
  p19_parity(df, function(d) reframe(d, x = sum(x), empty = tibble::new_tibble(list(), nrow = 3L)))
  p19_parity(df, function(d) reframe(d, empty = tibble::new_tibble(list(), nrow = 3L), x = sum(x)))
})

test_that("reframe() takes a zero-column tbl_gpu as a data frame result", {
  skip_if_no_gpu()
  # dplyr's own test (test-reframe.R, #6509), with the data frames on the GPU
  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(tibble::tibble(x = 1:3), lazy = lazy)
    empty <- tbl_gpu(tibble::tibble(), lazy = lazy)
    empty3 <- tbl_gpu(p19_zero(3), lazy = lazy)
    expect_identical(
      p19_plain(reframe(gdf, empty = empty)),
      tibble::tibble(empty = tibble::tibble())
    )
    expect_identical(
      p19_plain(reframe(gdf, x = sum(x), empty = empty)),
      tibble::tibble(x = integer(), empty = tibble::tibble())
    )
    expect_identical(
      p19_plain(reframe(gdf, empty = empty3)),
      tibble::tibble(empty = p19_zero(3))
    )
    expect_identical(
      p19_plain(reframe(gdf, x = sum(x), empty = empty3)),
      tibble::tibble(x = c(6L, 6L, 6L), empty = p19_zero(3))
    )
    expect_identical(
      p19_plain(reframe(gdf, empty = empty3, x = sum(x))),
      tibble::tibble(empty = p19_zero(3), x = c(6L, 6L, 6L))
    )
  }
})

# =============================================================================
# print(), collect() and the row count through other verbs
# =============================================================================

test_that("print() of a zero-column tbl_gpu shows its n x 0 dimensions", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    for (n in c(3L, 0L)) {
      z <- tbl_gpu(p19_zero(n), lazy = lazy)
      out <- capture.output(print(z))
      expect_match(out[[1]], pillar::dim_desc(p19_zero(n)), fixed = TRUE)
      expect_identical(collect(z), p19_zero(n))
    }
    m <- mutate(tbl_gpu(p19_df(), lazy = lazy), x = NULL, g = NULL, v = NULL)
    out <- capture.output(print(m))
    expect_match(out[[1]], pillar::dim_desc(p19_zero(4)), fixed = TRUE)
    j <- cross_join(tbl_gpu(p19_zero(2), lazy = lazy), tbl_gpu(p19_zero(3), lazy = lazy))
    out <- capture.output(print(j))
    expect_match(out[[1]], pillar::dim_desc(p19_zero(6)), fixed = TRUE)
  }
})

test_that("a zero-column table keeps its row count through the verbs that take it", {
  skip_if_no_gpu()
  z <- p19_zero(5)
  p19_parity(z, function(d) arrange(d))
  p19_parity(z, function(d) relocate(d))
  p19_parity(z, function(d) rename(d))
  p19_parity(z, function(d) group_by(d))
  p19_parity(z, function(d) ungroup(d))
  p19_parity(z, function(d) slice(d, 2:4))
  p19_parity(z, function(d) slice_tail(d, n = 2))
  p19_parity(z, function(d) head(d, 3))
  p19_parity(z, function(d) distinct(d))
  p19_parity(z, function(d) summarise(d, n = n()))
  p19_parity(z, function(d) count(d))
  p19_parity(z, function(d) add_count(d))
  p19_parity(z, function(d) mutate(d, r = dplyr::row_number()))
  p19_parity(z, function(d) bind_rows(d, d))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(z, lazy = lazy)
    p19_expect_shape(compute(g), 5)
    p19_expect_shape(dplyr::collapse(g), 5)
    expect_identical(tibble::as_tibble(g), z)
    expect_identical(dim(as.data.frame(g)), c(5L, 0L))
  }
})

test_that("distinct() of a zero-column table has no columns (dplyr #1437)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    df <- tbl_gpu(tibble::tibble(x = 1:10), lazy = lazy) |> select(-x)
    ddf <- distinct(df)
    expect_identical(ncol(ddf), 0L)
    p19_expect_shape(ddf, 1)
  }
})
