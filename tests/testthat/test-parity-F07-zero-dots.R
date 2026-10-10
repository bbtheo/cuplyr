# F07 acceptance tests: summarise() with zero dots, and the zero-column
# tbl_gpu it needs to return for the ungrouped case.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - summarise(df) -> 1 x 0 tibble (also for 0-row and 0-column input)
# - summarise(group_by(df, ...)) -> one row per group, grouping columns only,
#   in group_by()'s canonical order (ascending, NA last, C-locale strings,
#   FALSE < TRUE, factor level order), peeling one grouping layer / honouring
#   .groups
# - summarise(df, .by = ...) -> distinct .by combinations, first-appearance
#   order, ungrouped
#
# expect_equal() (waldo with tolerance) treats integer and double as equal,
# so every parity check here also compares column classes explicitly.

f07z_parity <- function(df, pipeline, arrange_by = NULL, lazy = TRUE) {
  r <- expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                   info = "column classes differ (eager)")
  expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs (eager)")
  if (lazy) {
    r <- expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ (lazy)")
    expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs (lazy)")
  }
}

f07z_df <- function() {
  data.frame(
    s = c("b", "B", "a", NA, "b", "a"),
    l = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE),
    f = factor(c("z", "y", "z", NA, "y", "z"), levels = c("z", "y", "x")),
    i = c(3L, NA, 1L, 3L, 2L, 1L),
    v = c(1.5, 2.5, NA, 4, 5, 6),
    stringsAsFactors = FALSE
  )
}

# -----------------------------------------------------------------------------
# Ungrouped zero dots -> 1 x 0
# -----------------------------------------------------------------------------

test_that("F07: ungrouped summarise() with zero dots returns one row, zero columns", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) dplyr::summarise(d))
  f07z_parity(df, function(d) dplyr::summarise(d, !!!list()))

  res <- dplyr::summarise(tbl_gpu(df))
  expect_true(is_tbl_gpu(res))
  expect_identical(dim(res), c(1L, 0L))
  expect_identical(collect(res), tibble::tibble(.rows = 1L))
  expect_identical(dplyr::group_vars(res), character())
  expect_no_error(utils::capture.output(print(res)))
})

test_that("F07: ungrouped zero-dot summarise() on 0-row input still returns one row", {
  skip_if_no_gpu()
  df <- f07z_df()[0, ]

  f07z_parity(df, function(d) dplyr::summarise(d))
})

test_that("F07: lazy zero-dot summarise() after pending ops returns 1 x 0", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) {
    d |> dplyr::filter(v > 2) |> dplyr::mutate(e = v * 2) |> dplyr::summarise()
  })
})

test_that("F07: ungrouped .groups= is a no-op with zero dots, like dplyr", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) dplyr::summarise(d, .groups = "drop"))
  f07z_parity(df, function(d) dplyr::summarise(d, .groups = "keep"))
})

# -----------------------------------------------------------------------------
# Grouped zero dots -> sorted distinct group keys
# -----------------------------------------------------------------------------

test_that("F07: grouped zero-dot summarise() returns sorted keys (character, C locale, NA last)", {
  skip_if_no_gpu()
  df <- f07z_df()

  # No arrange_by: dplyr defines this row order.
  f07z_parity(df, function(d) d |> dplyr::group_by(s) |> dplyr::summarise())
})

test_that("F07: grouped zero-dot summarise() sorts integer keys with NA last", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) d |> dplyr::group_by(i) |> dplyr::summarise())
})

test_that("F07: grouped zero-dot summarise() sorts logical keys FALSE, TRUE, NA", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) d |> dplyr::group_by(l) |> dplyr::summarise())
})

test_that("F07: grouped zero-dot summarise() on a factor key keeps level order and all levels", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) d |> dplyr::group_by(f) |> dplyr::summarise())

  res <- tbl_gpu(df) |> dplyr::group_by(f) |> dplyr::summarise() |> collect()
  expect_identical(levels(res$f), c("z", "y", "x"))
})

test_that("F07: grouped zero-dot summarise() with two keys sorts lexicographically and peels one layer", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) d |> dplyr::group_by(i, s) |> dplyr::summarise())
  f07z_parity(df, function(d) d |> dplyr::group_by(f, l) |> dplyr::summarise())

  res <- tbl_gpu(df) |> dplyr::group_by(i, s) |> dplyr::summarise()
  expect_identical(dplyr::group_vars(res), "i")
})

test_that("F07: grouped zero-dot summarise() honours every .groups value", {
  skip_if_no_gpu()
  df <- f07z_df()

  for (g in c("drop_last", "drop", "keep")) {
    f07z_parity(df, function(d) {
      d |> dplyr::group_by(i, s) |> dplyr::summarise(.groups = g)
    })
  }

  gpu <- tbl_gpu(df) |> dplyr::group_by(i, s)
  expect_identical(dplyr::group_vars(dplyr::summarise(gpu, .groups = "keep")), c("i", "s"))
  expect_identical(dplyr::group_vars(dplyr::summarise(gpu, .groups = "drop")), character())
  expect_identical(dplyr::group_vars(dplyr::summarise(gpu, .groups = "drop_last")), "i")
})

test_that("F07: repeated zero-dot summarise() peels groups one layer at a time", {
  skip_if_no_gpu()
  df <- data.frame(x = rep(1:4, each = 4), y = rep(1:2, each = 8), z = seq_len(16) / 3)

  f07z_parity(df, function(d) d |> dplyr::group_by(x, y) |> dplyr::summarise())
  f07z_parity(df, function(d) {
    d |> dplyr::group_by(x, y) |> dplyr::summarise() |> dplyr::summarise()
  })

  gpu <- tbl_gpu(df) |> dplyr::group_by(x, y)
  expect_identical(dplyr::group_vars(dplyr::summarise(gpu)), "x")
  expect_identical(dplyr::group_vars(dplyr::summarise(dplyr::summarise(gpu))), character())
})

test_that("F07: grouped zero-dot summarise() validates .groups like dplyr", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(f07z_df()) |> dplyr::group_by(s)

  expect_error(dplyr::summarise(gpu, .groups = "bogus"))
})

test_that("F07: grouped zero-dot summarise() emits the regroup message when asked", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = TRUE)
  gpu <- tbl_gpu(f07z_df()) |> dplyr::group_by(i, s)

  expect_message(dplyr::summarise(gpu), "regrouped")
  expect_no_message(dplyr::summarise(gpu, .groups = "drop_last"))
})

test_that("F07: grouped zero-dot summarise() on 0-row input keeps the key columns and types", {
  skip_if_no_gpu()
  df <- f07z_df()[0, ]

  f07z_parity(df, function(d) d |> dplyr::group_by(s) |> dplyr::summarise())
  f07z_parity(df, function(d) d |> dplyr::group_by(f, i) |> dplyr::summarise())
})

test_that("F07: lazy grouped zero-dot summarise() after pending ops", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) {
    d |>
      dplyr::filter(!is.na(v)) |>
      dplyr::mutate(k = i * 2L) |>
      dplyr::group_by(k, s) |>
      dplyr::summarise(.groups = "drop")
  })
})

test_that("F07: grouped zero-dot summarise() result composes with later verbs", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) {
    d |>
      dplyr::group_by(i) |>
      dplyr::summarise() |>
      dplyr::mutate(j = i + 1L)
  })
})

# -----------------------------------------------------------------------------
# .by zero dots -> first-appearance order
# -----------------------------------------------------------------------------

test_that("F07: summarise(.by =) with zero dots returns distinct keys in first-appearance order", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) dplyr::summarise(d, .by = s))
  f07z_parity(df, function(d) dplyr::summarise(d, .by = c(l, f)))
  f07z_parity(df, function(d) dplyr::summarise(d, .by = c(i, s)))
})

test_that("F07: summarise(.by =) with zero dots on 0-row input", {
  skip_if_no_gpu()
  df <- f07z_df()[0, ]

  f07z_parity(df, function(d) dplyr::summarise(d, .by = s))
})

# -----------------------------------------------------------------------------
# Zero-column tbl_gpu
# -----------------------------------------------------------------------------

test_that("F07: a zero-column data frame uploads with its row count", {
  skip_if_no_gpu()
  df <- tibble::tibble(.rows = 10L)

  gpu <- tbl_gpu(df)
  expect_identical(dim(gpu), c(10L, 0L))
  expect_identical(nrow(gpu), 10L)
  expect_identical(collect(gpu), tibble::tibble(.rows = 10L))
  expect_no_error(utils::capture.output(print(gpu)))
})

test_that("F07: summarise() on a zero-column tbl_gpu matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(.rows = 10L)

  f07z_parity(df, function(d) dplyr::summarise(d))
  f07z_parity(df, function(d) dplyr::summarise(d, n = dplyr::n()))
})

test_that("F07: summarise() result of zero columns feeds summarise(n = n())", {
  skip_if_no_gpu()
  df <- f07z_df()

  f07z_parity(df, function(d) dplyr::summarise(dplyr::summarise(d), n = dplyr::n()))
})

test_that("F07: other verbs on a zero-column tbl_gpu never return a silently wrong row count", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(tibble::tibble(.rows = 10L))

  # Full support for verbs on zero-column tables is F14's scope. Here the
  # only requirement is: either a loud error, or the dplyr-correct result.
  res <- tryCatch(collect(dplyr::mutate(gpu, z = 1L)), error = function(e) NULL)
  if (!is.null(res)) {
    expect_identical(res, dplyr::mutate(tibble::tibble(.rows = 10L), z = 1L))
  } else {
    succeed()
  }
})

# -----------------------------------------------------------------------------
# Round-1 review additions (B1-B3): the zero-column row count must survive
# every path, and be either right or a loud error.
# -----------------------------------------------------------------------------

f07z_error_or_equal <- function(expr, expected) {
  res <- tryCatch({
    r <- expr
    if (is_tbl_gpu(r)) collect(r) else r
  }, error = function(e) NULL)
  if (is.null(res)) {
    succeed()
  } else {
    expect_identical(dim(res), dim(expected))
    expect_identical(tibble::as_tibble(res), tibble::as_tibble(expected))
  }
}

test_that("F07: ungroup() and empty group_by() keep a zero-column table's row count", {
  skip_if_no_gpu()
  df <- tibble::tibble(.rows = 5L)

  f07z_parity(df, function(d) dplyr::summarise(dplyr::ungroup(d), n = dplyr::n()))
  f07z_parity(df, function(d) dplyr::summarise(dplyr::group_by(d), n = dplyr::n()))
  f07z_parity(df, function(d) dplyr::ungroup(d))

  res <- tbl_gpu(f07z_df()) |> dplyr::summarise() |> dplyr::ungroup()
  expect_identical(dim(res), c(1L, 0L))
  expect_identical(collect(res), tibble::tibble(.rows = 1L))
})

test_that("F07: binds and literal-mask filter on zero-column tables are right or a loud error", {
  skip_if_no_gpu()
  t2 <- tibble::tibble(.rows = 2L)
  t3 <- tibble::tibble(.rows = 3L)
  t5 <- tibble::tibble(.rows = 5L)

  f07z_error_or_equal(
    cuplyr::bind_rows(tbl_gpu(t2), tbl_gpu(t3)),
    dplyr::bind_rows(t2, t3)
  )
  f07z_error_or_equal(
    cuplyr::bind_rows(dplyr::summarise(tbl_gpu(f07z_df())), dplyr::summarise(tbl_gpu(f07z_df()))),
    tibble::tibble(.rows = 2L)
  )
  f07z_error_or_equal(
    cuplyr::bind_cols(tbl_gpu(t3), tbl_gpu(t3)),
    dplyr::bind_cols(t3, t3)
  )
  f07z_error_or_equal(
    dplyr::filter(tbl_gpu(t5), rep(TRUE, 5)),
    dplyr::filter(t5, rep(TRUE, 5))
  )
})

test_that("F07: zero-dot ungrouped summarise() after a lazy join returns 1 x 0", {
  skip_if_no_gpu()
  jx <- data.frame(k = 1:3, a = c(4, 5, 6))
  jy <- data.frame(k = 1:3, b = c(7, 8, 9))

  for (lazy in c(FALSE, TRUE)) {
    res <- dplyr::left_join(tbl_gpu(jx, lazy = lazy), tbl_gpu(jy, lazy = lazy), by = "k") |>
      dplyr::summarise() |>
      collect()
    expect_identical(res, tibble::tibble(.rows = 1L))
  }
})

# -----------------------------------------------------------------------------
# Known-failure handover (conformance test-slice.R "slice handles zero-row and
# zero-column inputs"): slice() on a zero-column table keeps a row count.
# -----------------------------------------------------------------------------

test_that("F07: slice() family on a zero-column tbl_gpu returns the right row count", {
  skip_if_no_gpu()
  df <- tibble::tibble(.rows = 10L)

  f07z_parity(df, function(d) dplyr::slice(d, 1))
  f07z_parity(df, function(d) dplyr::slice(d, c(1, 3)))
  f07z_parity(df, function(d) dplyr::slice(d, -1))
  f07z_parity(df, function(d) dplyr::slice(d, 20))
  f07z_parity(df, function(d) dplyr::slice_head(d, n = 3))
  f07z_parity(df, function(d) dplyr::slice_tail(d, n = 2))
  f07z_parity(data.frame(x = numeric()), function(d) dplyr::slice(d, 1))
})

# -----------------------------------------------------------------------------
# Round-2 review additions: joins and vctrs on zero-column tables (right or a
# loud error), and lazy zero-dot summarise() must still run pending ops.
# -----------------------------------------------------------------------------

test_that("F07: by = character() joins with a zero-column side are right or a loud error", {
  skip_if_no_gpu()
  x2 <- tibble::tibble(a = 1:2)
  z5 <- tibble::tibble(.rows = 5L)

  for (lazy in c(FALSE, TRUE)) {
    gx <- tbl_gpu(x2, lazy = lazy)
    gz <- tbl_gpu(z5, lazy = lazy)
    for (j in list(dplyr::left_join, dplyr::full_join, dplyr::inner_join,
                   dplyr::semi_join, dplyr::anti_join)) {
      f07z_error_or_equal(
        suppressWarnings(j(gx, gz, by = character())),
        suppressWarnings(j(x2, z5, by = character()))
      )
    }
    f07z_error_or_equal(
      suppressWarnings(dplyr::right_join(gz, gx, by = character())),
      suppressWarnings(dplyr::right_join(z5, x2, by = character()))
    )
  }
})

test_that("F07: vctrs sizes a zero-column tbl_gpu correctly or errors loudly", {
  skip_if_no_gpu()

  for (lazy in c(FALSE, TRUE)) {
    g5 <- tbl_gpu(tibble::tibble(.rows = 5L), lazy = lazy)
    sz <- tryCatch(vctrs::vec_size(g5), error = function(e) NULL)
    if (!is.null(sz)) expect_identical(sz, 5L) else succeed()
    ls <- tryCatch(vctrs::list_sizes(list(g5)), error = function(e) NULL)
    if (!is.null(ls)) expect_identical(ls, 5L) else succeed()
  }
})

test_that("F07: lazy zero-dot summarise() still runs pending ops that error", {
  skip_if_no_gpu()
  kx <- data.frame(k = 1:3)
  ky <- data.frame(k = 1:2)

  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      dplyr::inner_join(tbl_gpu(kx, lazy = lazy), tbl_gpu(ky, lazy = lazy),
                        by = "k", unmatched = "error") |>
        dplyr::summarise() |>
        collect()
    )
  }
})

# -----------------------------------------------------------------------------
# Round-3 review addition: `$<-` / `[[<-` on a zero-column tbl_gpu (F04 entry
# points) must keep the row count, or error loudly.
# -----------------------------------------------------------------------------

test_that("F07: column assignment on a zero-column tbl_gpu is right or a loud error", {
  skip_if_no_gpu()
  z5 <- tibble::tibble(.rows = 5L)

  assign_dollar <- function(x, value) { x$a <- value; x }
  assign_brackets <- function(x, value) { x[["a"]] <- value; x }

  for (lazy in c(FALSE, TRUE)) {
    for (assign in list(assign_dollar, assign_brackets)) {
      for (value in list(NA, NA_real_, as.Date("2024-01-01"), 1:5)) {
        f07z_error_or_equal(assign(tbl_gpu(z5, lazy = lazy), value), assign(z5, value))
      }
      # tibble rejects a length-0 value for a 5-row table: cuplyr must not
      # silently produce a 0-row result.
      res <- tryCatch({
        r <- assign(tbl_gpu(z5, lazy = lazy), integer())
        collect(r)
      }, error = function(e) NULL)
      expect_null(res)
    }
  }
})
