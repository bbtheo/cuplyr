# F20: rowwise() semantics on tbl_gpu.
#
# Contract (see scratchpad/parity/features/F20.md, ## Design):
# - rowwise() of a tbl_gpu runs on the CPU and returns dplyr's own rowwise_df;
#   every verb after it is dplyr's, so results match dplyr exactly.
# - No path silently turns a rowwise_df into a GPU table that is grouped by
#   the rowwise vars (the F31 side effect) or that quietly drops rowwise-ness:
#   * tbl_gpu()/as_tbl_gpu() of a rowwise_df upload it UNGROUPED and raise a
#     classed warning `cuplyr_warning_rowwise_upload`;
#   * a rowwise_df on the "other side" of a two-table verb (join y, set-op y,
#     a non-first bind input) is uploaded silently and ungrouped, since dplyr
#     ignores y's grouping there;
#   * bind_rows()/bind_cols() whose FIRST input is a rowwise_df return what
#     dplyr returns: a rowwise_df with its rowwise vars.

f20_df <- function() {
  tibble::tibble(
    id = 1:5,
    g = c("a", "a", "b", "b", "b"),
    a = c(1L, NA, 3L, 4L, 5L),
    b = c(2.5, 1, NA, 0, -1),
    c = c(10L, 20L, 30L, 40L, 50L),
    f = factor(c("x", "y", "x", "z", "y")),
    d = as.Date("2020-01-01") + 0:4
  )
}

# Run `pipeline` on the plain tibble (dplyr oracle) and on a tbl_gpu (eager or
# lazy, optionally with a pending op first), and require the same class, the
# same group_vars() and equal data INCLUDING the grouping structure.
expect_same_rowwise <- function(df, pipeline, lazy = FALSE) {
  oracle <- pipeline(tibble::as_tibble(df))
  gin <- tbl_gpu(df, lazy = lazy)
  got <- pipeline(gin)
  if (is_tbl_gpu(got)) {
    got <- collect(got)
  }
  expect_identical(class(got), class(oracle))
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(oracle))
  expect_equal(got, oracle)
}

# ---------------------------------------------------------------------------
# A1. Upload of a rowwise_df
# ---------------------------------------------------------------------------

test_that("tbl_gpu(<rowwise_df>) uploads ungrouped with a classed warning", {
  skip_if_no_gpu()
  df <- f20_df()

  for (rdf in list(dplyr::rowwise(df, g), dplyr::rowwise(df), dplyr::rowwise(df, id, g))) {
    expect_warning(gt <- tbl_gpu(rdf), class = "cuplyr_warning_rowwise_upload")
    expect_true(is_tbl_gpu(gt))
    expect_identical(dplyr::group_vars(gt), character())
    expect_equal(tibble::as_tibble(collect(gt)), tibble::as_tibble(dplyr::ungroup(rdf)))
  }
})

test_that("tbl_gpu(<rowwise_df>, lazy = TRUE) and as_tbl_gpu() follow the same rule", {
  skip_if_no_gpu()
  rdf <- dplyr::rowwise(f20_df(), g)

  expect_warning(gl <- tbl_gpu(rdf, lazy = TRUE), class = "cuplyr_warning_rowwise_upload")
  expect_identical(gl$exec_mode, "lazy")
  expect_identical(dplyr::group_vars(gl), character())

  expect_warning(ga <- as_tbl_gpu(rdf), class = "cuplyr_warning_rowwise_upload")
  expect_identical(dplyr::group_vars(ga), character())
})

test_that("an uploaded rowwise_df is not secretly grouped by its rowwise vars", {
  skip_if_no_gpu()
  rdf <- dplyr::rowwise(f20_df(), g)
  gt <- suppressWarnings(tbl_gpu(rdf))

  # Whole-table aggregates, exactly as on dplyr::ungroup(rdf).
  expected <- dplyr::ungroup(rdf) |>
    dplyr::mutate(m = mean(c), n = dplyr::n())
  got <- gt |> dplyr::mutate(m = mean(c), n = dplyr::n()) |> collect()
  expect_equal(tibble::as_tibble(got), tibble::as_tibble(expected))

  s <- gt |> dplyr::summarise(total = sum(c))
  expect_identical(dplyr::group_vars(s), character())
  expect_equal(tibble::as_tibble(collect(s)), tibble::tibble(total = 150L))
})

test_that("a grouped (non-rowwise) upload still keeps its groups, without the warning", {
  skip_if_no_gpu()
  gdf <- dplyr::group_by(f20_df(), g)
  expect_no_warning(gt <- tbl_gpu(gdf))
  expect_identical(dplyr::group_vars(gt), "g")
})

# ---------------------------------------------------------------------------
# A1b. rowwise_df on the other side of a two-table verb: silent, ungrouped
# ---------------------------------------------------------------------------

test_that("a rowwise_df join y is uploaded silently and the result matches dplyr", {
  skip_if_no_gpu()
  df <- f20_df()
  y <- dplyr::rowwise(tibble::tibble(g = c("a", "b"), w = c(1.5, 2.5)), g)

  expected <- dplyr::left_join(df, y, by = "g")
  expect_no_warning(got <- dplyr::left_join(tbl_gpu(df), y, by = "g", copy = TRUE))
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(expected))
  expect_equal(tibble::as_tibble(collect(got)), tibble::as_tibble(expected))

  expected_lazy <- dplyr::inner_join(df, y, by = "g")
  expect_no_warning(
    got_lazy <- dplyr::inner_join(tbl_gpu(df, lazy = TRUE), y, by = "g", copy = TRUE)
  )
  expect_identical(dplyr::group_vars(got_lazy), character())
  expect_equal(tibble::as_tibble(collect(got_lazy)), tibble::as_tibble(expected_lazy))
})

test_that("a rowwise_df set-op y is uploaded silently and the result matches dplyr", {
  skip_if_no_gpu()
  df <- f20_df() |> dplyr::select(id, g, c)
  y <- dplyr::rowwise(tibble::tibble(id = 6L, g = "c", c = 60L), g)

  expected <- dplyr::union(df, y)
  expect_no_warning(got <- dplyr::union(tbl_gpu(df), y))
  expect_identical(dplyr::group_vars(got), character())
  expect_equal(tibble::as_tibble(collect(got)), tibble::as_tibble(expected))
})

test_that("a rowwise_df that is not the first bind input is bound silently, like dplyr", {
  skip_if_no_gpu()
  df <- f20_df() |> dplyr::select(id, g, c)
  y <- dplyr::rowwise(tibble::tibble(id = 6L, g = "c", c = 60L), g)

  expected <- dplyr::bind_rows(df, y)
  expect_no_warning(got <- cuplyr::bind_rows(tbl_gpu(df), y))
  expect_true(is_tbl_gpu(got))
  expect_identical(dplyr::group_vars(got), character())
  expect_equal(tibble::as_tibble(collect(got)), tibble::as_tibble(expected))

  z <- dplyr::rowwise(tibble::tibble(k = 5:1))
  expected_cols <- dplyr::bind_cols(df, z)
  expect_no_warning(got_cols <- cuplyr::bind_cols(tbl_gpu(df), z))
  expect_identical(dplyr::group_vars(got_cols), character())
  expect_equal(tibble::as_tibble(collect(got_cols)), tibble::as_tibble(expected_cols))
})

# ---------------------------------------------------------------------------
# A1c. rowwise_df as the FIRST bind input: dplyr's rowwise_df result
# ---------------------------------------------------------------------------

test_that("bind_rows() with a rowwise_df first input returns dplyr's rowwise_df", {
  skip_if_no_gpu()
  df <- f20_df() |> dplyr::select(id, g, c)
  x <- dplyr::rowwise(df, g)
  more <- tibble::tibble(id = 6:7, g = c("c", "d"), c = c(60L, 70L))

  expected <- dplyr::bind_rows(x, more)
  for (other in list(tbl_gpu(more), tbl_gpu(more, lazy = TRUE))) {
    got <- cuplyr::bind_rows(x, other)
    if (is_tbl_gpu(got)) got <- collect(got)
    expect_identical(class(got), class(expected))
    expect_identical(dplyr::group_vars(got), "g")
    expect_equal(got, expected)
  }
})

test_that("bind_cols() with a rowwise_df first input returns dplyr's rowwise_df", {
  skip_if_no_gpu()
  df <- f20_df() |> dplyr::select(id, g)
  x <- dplyr::rowwise(df, g)
  more <- tibble::tibble(k = 5:1, w = c(0.5, 1.5, NA, 3.5, 4.5))

  expected <- dplyr::bind_cols(x, more)
  got <- cuplyr::bind_cols(x, tbl_gpu(more))
  if (is_tbl_gpu(got)) got <- collect(got)
  expect_identical(class(got), class(expected))
  expect_identical(dplyr::group_vars(got), "g")
  expect_equal(got, expected)
})

# ---------------------------------------------------------------------------
# A2. rowwise() of a tbl_gpu: dplyr's rowwise_df and dplyr's verbs after it
# ---------------------------------------------------------------------------

test_that("rowwise() of a tbl_gpu returns dplyr's rowwise_df (eager and lazy)", {
  skip_if_no_gpu()
  df <- f20_df()
  for (lazy in c(FALSE, TRUE)) {
    expect_same_rowwise(df, function(x) dplyr::rowwise(x), lazy = lazy)
    expect_same_rowwise(df, function(x) dplyr::rowwise(x, g), lazy = lazy)
    expect_same_rowwise(df, function(x) dplyr::rowwise(x, id, f), lazy = lazy)
    # a pending op before rowwise()
    expect_same_rowwise(df, function(x) {
      x |> dplyr::filter(id > 1) |> dplyr::mutate(e = c * 2L) |> dplyr::rowwise(g)
    }, lazy = lazy)
  }
})

test_that("rowwise mutate with dplyr::c_across() matches dplyr, NAs included", {
  skip_if_no_gpu()
  df <- f20_df()
  for (lazy in c(FALSE, TRUE)) {
    expect_same_rowwise(df, function(x) {
      x |> dplyr::rowwise() |> dplyr::mutate(s = sum(dplyr::c_across(c(a, c))))
    }, lazy = lazy)
    expect_same_rowwise(df, function(x) {
      x |> dplyr::rowwise(id) |>
        dplyr::mutate(
          s = sum(dplyr::c_across(a:c), na.rm = TRUE),
          m = mean(dplyr::c_across(c(a, b))),
          mx = max(a, c, na.rm = TRUE),
          mn = min(c(a, b, c))
        )
    }, lazy = lazy)
  }
})

test_that("rowwise summarise matches dplyr's grouped/rowwise result and message", {
  skip_if_no_gpu()
  df <- f20_df()
  expect_same_rowwise(df, function(x) {
    x |> dplyr::rowwise(id) |> dplyr::summarise(t = sum(dplyr::c_across(c(a, c))), .groups = "keep")
  })
  expect_same_rowwise(df, function(x) {
    x |> dplyr::rowwise(id, g) |> dplyr::summarise(t = mean(c(a, b)), .groups = "drop")
  })
  expect_same_rowwise(df, function(x) {
    x |> dplyr::rowwise() |> dplyr::summarise(t = a + c)
  })
  # rowwise -> grouped_df conversion message, as dplyr emits it.
  expect_message(
    tbl_gpu(df) |> dplyr::rowwise(g) |> dplyr::summarise(t = a + c),
    "converted the output from a rowwise data frame to a grouped"
  )
  expect_same_rowwise(df, function(x) {
    suppressMessages(x |> dplyr::rowwise(f) |> dplyr::summarise(dd = d + 1))
  }, lazy = TRUE)
})

test_that("rowwise filter, n() and cur_group_id() match dplyr", {
  skip_if_no_gpu()
  df <- f20_df()
  expect_same_rowwise(df, function(x) {
    x |> dplyr::rowwise() |> dplyr::filter(sum(dplyr::c_across(a:b), na.rm = TRUE) > 3)
  })
  expect_same_rowwise(df, function(x) {
    x |> dplyr::rowwise(g) |> dplyr::mutate(n = dplyr::n(), k = dplyr::cur_group_id())
  }, lazy = TRUE)
})

test_that("group_by() then rowwise() keeps the group vars; re-grouping errors like dplyr", {
  skip_if_no_gpu()
  df <- f20_df()
  expect_same_rowwise(df, function(x) {
    x |> dplyr::group_by(g) |> dplyr::rowwise() |> dplyr::mutate(s = a + c)
  })
  expect_error(
    tbl_gpu(df) |> dplyr::group_by(g) |> dplyr::rowwise(id),
    "Can't re-group when creating rowwise data"
  )
})

test_that("rowwise() on zero rows matches dplyr", {
  skip_if_no_gpu()
  df <- f20_df()[0, ]
  for (lazy in c(FALSE, TRUE)) {
    expect_same_rowwise(df, function(x) {
      x |> dplyr::rowwise(g) |> dplyr::mutate(s = sum(dplyr::c_across(c(a, c))))
    }, lazy = lazy)
  }
  expect_same_rowwise(f20_df(), function(x) {
    x |> dplyr::filter(id > 99) |> dplyr::rowwise() |> dplyr::mutate(s = a + c)
  }, lazy = TRUE)
})

test_that("ungroup() after rowwise() gives dplyr's plain tibble", {
  skip_if_no_gpu()
  expect_same_rowwise(f20_df(), function(x) {
    x |> dplyr::rowwise() |> dplyr::mutate(s = sum(dplyr::c_across(c(a, c)))) |> dplyr::ungroup()
  })
})

test_that("summarise(.groups = \"rowwise\") on a tbl_gpu returns dplyr's rowwise_df", {
  skip_if_no_gpu()
  df <- f20_df()
  for (lazy in c(FALSE, TRUE)) {
    expect_same_rowwise(df, function(x) {
      x |> dplyr::summarise(t = sum(c), .groups = "rowwise")
    }, lazy = lazy)
    expect_same_rowwise(df, function(x) {
      x |> dplyr::group_by(g) |> dplyr::summarise(t = sum(c), .groups = "rowwise")
    }, lazy = lazy)
  }
})

test_that("the oracle helper agrees on the data of a rowwise pipeline", {
  skip_if_no_gpu()
  df <- as.data.frame(f20_df())
  pipe <- function(x) {
    x |> dplyr::rowwise(id) |> dplyr::mutate(s = sum(dplyr::c_across(c(a, c)), na.rm = TRUE))
  }
  expect_same_as_dplyr(df, pipe)
  expect_same_as_dplyr_lazy(df, pipe)
})
