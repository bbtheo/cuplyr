# F07 acceptance tests: min()/max() empty-group identity over INTEGER
# columns, and dplyr's warning.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - min()/max() over an integer column stays integer iff every group still
#   has at least one value after na.rm. If ANY group is empty after NA
#   removal, the WHOLE result column becomes double, with Inf / -Inf for the
#   empty group(s).
# - na.rm = FALSE with NAs present gives NA_integer_ (integer, no warning).
# - Whenever a group hits the identity, dplyr warns (wrapping base R's
#   "no non-missing arguments to min; returning Inf" / "... max; returning
#   -Inf"), for double input too.
#
# Grouped summarise row order in cuplyr is cudf hash order (separate,
# out-of-scope gap), hence arrange_by. expect_equal() does not tell integer
# from double, so column classes are compared explicitly.

f07m_parity <- function(df, pipeline, arrange_by = NULL, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline, arrange_by = arrange_by))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
}

f07m_df <- function() {
  data.frame(
    g = c("a", "a", "b", "b", "c"),
    x = c(1L, 2L, NA, NA, 5L),      # group b entirely NA
    z = c(7L, NA, 8L, NA, 6L),      # every group keeps a value
    y = c(1, 2, NA, NA, 5),         # double, group b entirely NA
    stringsAsFactors = FALSE
  )
}

# -----------------------------------------------------------------------------
# Values and types
# -----------------------------------------------------------------------------

test_that("F07: grouped min()/max(na.rm = TRUE) on an all-NA integer group promote to double Inf/-Inf", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE))
  }, arrange_by = "g")

  res <- suppressWarnings(
    tbl_gpu(df) |>
      dplyr::group_by(g) |>
      dplyr::summarise(mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE)) |>
      collect() |>
      dplyr::arrange(g)
  )
  expect_type(res$mn, "double")
  expect_identical(res$mn, c(1, Inf, 5))
  expect_identical(res$mx, c(2, -Inf, 5))
})

test_that("F07: integer min()/max(na.rm = TRUE) stay integer when no group is empty", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(mn = min(z, na.rm = TRUE), mx = max(z, na.rm = TRUE))
  }, arrange_by = "g")

  res <- tbl_gpu(df) |>
    dplyr::group_by(g) |>
    dplyr::summarise(mn = min(z, na.rm = TRUE)) |>
    collect()
  expect_type(res$mn, "integer")
})

test_that("F07: only the aggregation that needs the identity is promoted", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(
        mnx = min(x, na.rm = TRUE), mxz = max(z, na.rm = TRUE),
        mnz = min(z), n = dplyr::n()
      )
  }, arrange_by = "g")
})

test_that("F07: integer min()/max() without na.rm give NA_integer_ for groups with NA", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(mn = min(x), mx = max(z))
  }, arrange_by = "g")
})

test_that("F07: ungrouped integer min()/max(na.rm = TRUE) over all-NA input give double Inf/-Inf", {
  skip_if_no_gpu()
  df <- data.frame(x = c(NA_integer_, NA_integer_, NA_integer_), w = c(1L, 2L, 3L))

  f07m_parity(df, function(d) {
    dplyr::summarise(d, mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE), mw = max(w))
  })
})

test_that("F07: summarise(.by =) integer min()/max(na.rm = TRUE) on an all-NA group", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    dplyr::summarise(d, mn = min(x, na.rm = TRUE), mx = max(z, na.rm = TRUE), .by = g)
  })
})

test_that("F07: integer min()/max() on an all-NA group after pending lazy ops", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    d |>
      dplyr::filter(g != "c") |>
      dplyr::mutate(x2 = x * 2L) |>
      dplyr::group_by(g) |>
      dplyr::summarise(mn = min(x2, na.rm = TRUE))
  }, arrange_by = "g")
})

test_that("F07: verbs after a promoted min()/max() see a double column (lazy and eager)", {
  skip_if_no_gpu()
  df <- f07m_df()

  f07m_parity(df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(mn = min(x, na.rm = TRUE), .groups = "drop") |>
      dplyr::mutate(k = mn + 1L) |>
      dplyr::arrange(dplyr::desc(mn))
  })
  f07m_parity(df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(mx = max(x, na.rm = TRUE), .groups = "drop") |>
      dplyr::filter(mx < 3)
  }, arrange_by = "g")
})

test_that("F07: the promoted column's schema type is FLOAT64", {
  skip_if_no_gpu()
  df <- f07m_df()

  for (lazy in c(FALSE, TRUE)) {
    gpu <- tbl_gpu(df, lazy = lazy)
    res <- suppressWarnings(
      gpu |> dplyr::group_by(g) |> dplyr::summarise(mn = min(x, na.rm = TRUE))
    )
    out <- suppressWarnings(
      res |> dplyr::mutate(k = mn * 1L) |> collect()
    )
    expect_type(out$k, "double")
  }
})

# -----------------------------------------------------------------------------
# Warnings
# -----------------------------------------------------------------------------

test_that("F07: min()/max(na.rm = TRUE) on an empty group warn like dplyr (eager)", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(f07m_df()) |> dplyr::group_by(g)

  expect_warning(
    collect(dplyr::summarise(gpu, mn = min(x, na.rm = TRUE))),
    "no non-missing arguments to min; returning Inf"
  )
  expect_warning(
    collect(dplyr::summarise(gpu, mx = max(x, na.rm = TRUE))),
    "no non-missing arguments to max; returning -Inf"
  )
})

test_that("F07: double-column min()/max(na.rm = TRUE) on an empty group also warn", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(f07m_df()) |> dplyr::group_by(g)

  expect_warning(
    collect(dplyr::summarise(gpu, mn = min(y, na.rm = TRUE))),
    "no non-missing arguments to min; returning Inf"
  )
})

test_that("F07: lazy min()/max(na.rm = TRUE) on an empty group warn by collect()", {
  skip_if_no_gpu()
  gpu <- as_lazy(tbl_gpu(f07m_df())) |> dplyr::group_by(g)

  expect_warning(
    collect(dplyr::summarise(gpu, mn = min(x, na.rm = TRUE))),
    "no non-missing arguments to min; returning Inf"
  )
})

test_that("F07: no warning when no group needs the identity", {
  skip_if_no_gpu()
  df <- f07m_df()

  for (lazy in c(FALSE, TRUE)) {
    gpu <- tbl_gpu(df, lazy = lazy)
    gpu <- dplyr::group_by(gpu, g)
    expect_no_warning(collect(dplyr::summarise(gpu, mn = min(z, na.rm = TRUE), mx = max(y))))
    expect_no_warning(collect(dplyr::summarise(gpu, mn = min(x), m = mean(y, na.rm = TRUE))))
  }
})
