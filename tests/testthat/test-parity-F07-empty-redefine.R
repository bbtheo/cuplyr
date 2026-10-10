# F07 acceptance tests: 0-row summarise() and summarise() redefining an
# existing (grouping or non-grouping) column.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - 0-row ungrouped summarise() returns ONE row of base R's empty-vector
#   reductions (n() = 0L, mean = NaN, sum = 0, min = Inf, max = -Inf as
#   double + warning, median/sd = NA_real_, n_distinct = 0L,
#   first = typed NA, any = FALSE, all = TRUE).
# - 0-row grouped summarise() keeps every grouping column with its type;
#   min()/max() over an integer column is typed double (dplyr evaluates once
#   on an empty chunk, where min(integer(0)) is Inf).
# - An aggregate named like a grouping column replaces that key column IN
#   PLACE (same position, aggregate's type, group order unchanged); group_vars
#   keep the name.
# - A later dot that names an earlier dot's output sees the NEW value.
#
# cuplyr's grouped summarise row order is cudf hash order (a separate,
# out-of-scope gap), so grouped cases with aggregations use arrange_by.
# expect_equal() does not distinguish integer from double, so column classes
# are compared explicitly.

f07e_parity <- function(df, pipeline, arrange_by = NULL, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline, arrange_by = arrange_by))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
    expect_identical(dim(r$gpu), dim(r$oracle), info = "dim differs")
    for (nm in names(r$oracle)) {
      if (is.factor(r$oracle[[nm]])) {
        expect_identical(levels(r$gpu[[nm]]), levels(r$oracle[[nm]]),
                         info = paste("factor levels differ for", nm))
      }
    }
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
}

# -----------------------------------------------------------------------------
# 0-row input
# -----------------------------------------------------------------------------

f07e_empty <- function() {
  data.frame(
    g = character(), h = integer(), x = integer(), y = double(), l = logical(),
    f = factor(character(), levels = c("u", "v")),
    stringsAsFactors = FALSE
  )
}

test_that("F07: 0-row ungrouped summarise() returns one row of empty-vector reductions", {
  skip_if_no_gpu()
  df <- f07e_empty()

  f07e_parity(df, function(d) {
    dplyr::summarise(
      d,
      n = dplyr::n(), m = mean(y), s = sum(y), md = median(y), sdv = sd(y),
      nd = dplyr::n_distinct(x), fx = dplyr::first(x), a = any(l), al = all(l)
    )
  })
})

test_that("F07: 0-row ungrouped min()/max() return Inf/-Inf as double (integer and double input)", {
  skip_if_no_gpu()
  df <- f07e_empty()

  f07e_parity(df, function(d) {
    dplyr::summarise(d, mnx = min(x), mxx = max(x), mny = min(y), mxy = max(y))
  })
  f07e_parity(df, function(d) {
    dplyr::summarise(d, mnx = min(x, na.rm = TRUE), mxy = max(y, na.rm = TRUE))
  })
})

test_that("F07: 0-row ungrouped min() warns like dplyr", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(f07e_empty())

  expect_warning(
    collect(dplyr::summarise(gpu, mn = min(x))),
    "no non-missing arguments to min; returning Inf"
  )
  expect_warning(
    collect(dplyr::summarise(as_lazy(gpu), mx = max(y))),
    "no non-missing arguments to max; returning -Inf"
  )
})

test_that("F07: 0-row grouped summarise() keeps grouping columns with their types", {
  skip_if_no_gpu()
  df <- f07e_empty()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(n = dplyr::n(), m = mean(y))
  })
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(f) |> dplyr::summarise(n = dplyr::n(), s = sum(y))
  })
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g, h) |> dplyr::summarise(n = dplyr::n())
  })
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(h, f) |> dplyr::summarise(m = mean(y), .groups = "keep")
  })
})

test_that("F07: 0-row grouped and .by min()/max() over an integer column are typed double", {
  skip_if_no_gpu()
  df <- f07e_empty()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(mn = min(x), mx = max(x, na.rm = TRUE))
  })
  f07e_parity(df, function(d) {
    dplyr::summarise(d, mn = min(x), .by = g)
  })
})

test_that("F07: 0-row after a lazy filter keeps grouping columns", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), y = c(1, 2), stringsAsFactors = FALSE)

  f07e_parity(df, function(d) {
    d |>
      dplyr::filter(y > 100) |>
      dplyr::group_by(g) |>
      dplyr::summarise(m = mean(y), n = dplyr::n())
  })
})

# -----------------------------------------------------------------------------
# Redefining a grouping column
# -----------------------------------------------------------------------------

f07e_df <- function() {
  data.frame(
    g = c("a", "a", "b", "b", "c"),
    h = c(1L, 1L, 2L, 2L, 3L),
    x = c(10L, 20L, 30L, NA, 50L),
    y = c(1.5, 2.5, 3.5, 4.5, 5.5),
    f = factor(c("u", "u", "v", "w", "w"), levels = c("w", "v", "u")),
    stringsAsFactors = FALSE
  )
}

test_that("F07: summarise() redefining the only grouping column replaces it in place", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(h) |> dplyr::summarise(h = max(x, na.rm = TRUE))
  }, arrange_by = "h")
})

test_that("F07: a redefined grouping column keeps its position before the other aggregates", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(h) |> dplyr::summarise(n = dplyr::n(), h = max(y))
  }, arrange_by = "h")
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(m = mean(y), g = min(y), n = dplyr::n())
  }, arrange_by = "g")
})

test_that("F07: redefining a non-last grouping column changes its type and keeps group_vars", {
  skip_if_no_gpu()
  df <- f07e_df()

  # g is character; max(y) is double.
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g, h) |> dplyr::summarise(g = max(y), s = sum(y))
  }, arrange_by = "g")
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g, h) |> dplyr::summarise(g = max(y), .groups = "keep")
  }, arrange_by = "g")
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g, h) |> dplyr::summarise(h = mean(y), .groups = "drop")
  }, arrange_by = "h")

  gpu <- tbl_gpu(df) |> dplyr::group_by(g, h)
  expect_identical(dplyr::group_vars(dplyr::summarise(gpu, g = max(y))), "g")
  expect_identical(
    dplyr::group_vars(dplyr::summarise(gpu, g = max(y), .groups = "keep")),
    c("g", "h")
  )
})

test_that("F07: redefining a factor grouping column drops its factor levels", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(f) |> dplyr::summarise(f = dplyr::n(), m = mean(y))
  }, arrange_by = "m")
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(f) |> dplyr::summarise(n = dplyr::n(), f = max(y))
  }, arrange_by = "f")

  res <- tbl_gpu(df) |> dplyr::group_by(f) |> dplyr::summarise(f = max(y)) |> collect()
  expect_false(is.factor(res$f))
  expect_type(res$f, "double")
})

test_that("F07: summarise(.by =) redefining a .by column replaces it in place", {
  skip_if_no_gpu()
  df <- f07e_df()

  # .by order is first-appearance, which dplyr defines: no arrange_by.
  f07e_parity(df, function(d) dplyr::summarise(d, h = max(y), n = dplyr::n(), .by = h))
  f07e_parity(df, function(d) dplyr::summarise(d, n = dplyr::n(), g = min(y), .by = c(g, h)))
})

test_that("F07: redefining a grouping column after pending lazy ops", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) {
    d |>
      dplyr::filter(y > 2) |>
      dplyr::mutate(z = y * 2) |>
      dplyr::group_by(h) |>
      dplyr::summarise(h = sum(z), n = dplyr::n())
  }, arrange_by = "h")
})

# -----------------------------------------------------------------------------
# Later dots see redefined columns
# -----------------------------------------------------------------------------

test_that("F07: a later dot referencing a redefined grouping column sees the new value", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(h) |> dplyr::summarise(h = max(y), m = mean(h))
  }, arrange_by = "h")
})

test_that("F07: a later dot referencing a redefined non-grouping column sees the new value", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) dplyr::summarise(d, y = mean(x, na.rm = TRUE), z = max(y)))
  f07e_parity(df, function(d) dplyr::summarise(d, y = max(y), w = min(y)))
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(y = max(y), w = sum(y))
  }, arrange_by = "g")
})

test_that("F07: two dots with the same name: the later one wins, like dplyr", {
  skip_if_no_gpu()
  df <- f07e_df()

  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(m = mean(y), m = max(y))
  }, arrange_by = "g")
})

test_that("F07: a later dot reading a redefined column through the .data pronoun sees the new value", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b"), x = c(1, 2, 4), stringsAsFactors = FALSE)

  f07e_parity(df, function(d) dplyr::summarise(d, x = max(x), y = min(.data[["x"]])))
  f07e_parity(df, function(d) dplyr::summarise(d, x = max(x), y = min(.data$x)))
  f07e_parity(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(x = max(x), y = min(.data[["x"]]))
  }, arrange_by = "g")
})
