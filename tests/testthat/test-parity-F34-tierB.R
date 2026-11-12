# F34 acceptance tests, tier B: summarise() type parity and polish.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - B4: min()/max() of a logical column is integer. A group that is empty
#   after na.rm, or a 0-row input, makes the column double with Inf/-Inf and
#   a warning (base R's "no non-missing arguments to min; returning Inf"),
#   the same rule as F07's integer min()/max().
# - B5: non-syntactic column names work inside aggregates.
# - B7: an unnamed data-frame result is spliced into columns, and across()
#   evaluates every column from the input (not sequentially).

f34b_parity <- function(df, pipeline, arrange_by = NULL, lazy = TRUE) {
  run <- function(fun) {
    r <- suppressWarnings(fun(df, pipeline, arrange_by = arrange_by))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class),
                     info = "column classes differ")
  }
  run(expect_same_as_dplyr)
  if (lazy) run(expect_same_as_dplyr_lazy)
}

f34b_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 2, 3),
    l = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE),
    l2 = c(TRUE, FALSE, NA, NA, NA, TRUE),     # group g == 2 all NA
    n = c(1L, 2L, 3L, 4L, 5L, 6L),
    `my x` = c(1, 2, 3, 4, 5, 6),
    check.names = FALSE
  )
}

# -----------------------------------------------------------------------------
# B4: min()/max() of logical
# -----------------------------------------------------------------------------

test_that("B4: min()/max() of logical is integer", {
  skip_if_no_gpu()
  f34b_parity(f34b_df(), function(d) {
    d |> group_by(g) |> summarise(mn = min(l), mx = max(l, na.rm = TRUE))
  }, arrange_by = "g")
  f34b_parity(f34b_df(), function(d) d |> summarise(mn = min(l, na.rm = TRUE), mx = max(l2)))
})

test_that("B4: logical min()/max() with an empty group is double Inf/-Inf and warns", {
  skip_if_no_gpu()
  f34b_parity(f34b_df(), function(d) {
    d |> group_by(g) |> summarise(mn = min(l2, na.rm = TRUE), mx = max(l2, na.rm = TRUE))
  }, arrange_by = "g")
  expect_warning(
    tbl_gpu(f34b_df()) |> group_by(g) |> summarise(mx = max(l2, na.rm = TRUE)) |> collect(),
    "no non-missing arguments to max; returning -Inf"
  )
})

test_that("B4: logical min()/max() over 0 rows is double Inf/-Inf and warns", {
  skip_if_no_gpu()
  f34b_parity(f34b_df(), function(d) {
    d |> filter(n > 100) |> summarise(mn = min(l), mx = max(l))
  })
  expect_warning(
    tbl_gpu(f34b_df()) |> filter(n > 100) |> summarise(mn = min(l)) |> collect(),
    "no non-missing arguments to min; returning Inf"
  )
})

# -----------------------------------------------------------------------------
# B5: non-syntactic names
# -----------------------------------------------------------------------------

test_that("B5: backticked column names work in summarise aggregates", {
  skip_if_no_gpu()
  f34b_parity(f34b_df(), function(d) {
    d |> summarise(s = sum(`my x`), m = mean(`my x` * 2), k = n_distinct(`my x`))
  })
  f34b_parity(f34b_df(), function(d) {
    d |> group_by(g) |> summarise(s = max(`my x`), m = mean(`my x` + n))
  }, arrange_by = "g")
})

test_that("B5: backticked column names stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  out <- tbl_gpu(f34b_df()) |> group_by(g) |>
    summarise(s = sum(`my x`), m = mean(`my x` * 2)) |> collect()
  expect_equal(nrow(out), 3L)
})

# -----------------------------------------------------------------------------
# B7: the fallback runs the user's original dots
# -----------------------------------------------------------------------------

test_that("B7: an unnamed data-frame result is spliced into columns", {
  skip_if_no_gpu()
  df <- data.frame(x = c(2, 3))
  f34b_parity(df, function(d) d |> summarise(tibble::tibble(y = sum(x) * 2, z = 3)))
})

test_that("B7: across() lambdas in summarise() see the input columns", {
  skip_if_no_gpu()
  df <- data.frame(x = 2, y = 4, z = 8)
  f34b_parity(df, function(d) d |> summarise(across(everything(), ~ .x / y)))
  f34b_parity(df, function(d) d |> summarise(a = 2, across(c(x, y, z), ~ .x / .data$y)))
})

test_that("B7: grouped across() lambdas in summarise() see the input columns", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), x = c(2, 4, 6), y = c(4, 4, 2))
  f34b_parity(df, function(d) {
    d |> group_by(g) |> summarise(across(c(y, x), ~ sum(.x / y)))
  }, arrange_by = "g")
})
