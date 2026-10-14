# F22 round 3: summarise()'s median() of a float column is R's type-7 value at
# +-Inf (it shares the quantile path), NaN computed on the GPU counts as
# missing (na.rm = TRUE drops it, quantile() errors without it), aggregations
# of the same window argument share one window stage, and reframe() leaves a
# string min()/max()/range() to dplyr outside a C collation.

f22r3_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("summarise() median() of a float column at +-Inf matches R", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5),
    x = c(1, Inf, Inf, -Inf, -Inf, 3, 2, 5, 9, -Inf, Inf, Inf, NA)
  )
  f22r3_both(df, function(d) {
    d |> group_by(g) |> summarise(m = median(x))
  })
  f22r3_both(df[1:11, ], function(d) {
    d |> group_by(g) |> summarise(m = median(x), q = quantile(x, 0.5, names = FALSE))
  })
  f22r3_both(df, function(d) {
    d |> group_by(g) |> summarise(m = median(x, na.rm = TRUE), mn = mean(x, na.rm = TRUE))
  })
  f22r3_both(df[1:9, ], function(d) summarise(d, m = median(x)))
  f22r3_both(df[4:6, ], function(d) summarise(d, m = median(x), s = sum(x)))
  f22r3_both(df, function(d) summarise(d, m = median(x, na.rm = TRUE)))
  f22r3_both(df, function(d) summarise(d, m = median(x)))
})

test_that("summarise() median() of an integer column is unchanged", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1L, 1L, 2L, 2L, 2L, 3L), i = c(1L, 4L, 2L, NA, 8L, 5L))
  f22r3_both(df, function(d) d |> group_by(g) |> summarise(m = median(i), r = median(i, na.rm = TRUE)))
  f22r3_both(df, function(d) summarise(d, m = median(i, na.rm = TRUE)))
})

# Row 1 of each group is 0 / 0 = NaN on the GPU (an upload turns R's NaN into
# NA, so only a computed NaN reaches gpu_summarise() as NaN); group 3 is NaN
# only; group 4 has a real NA as well.
f22r3_nan_df <- function() {
  data.frame(
    g = c(1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4),
    x = c(0, 1, 2, 0, 5, 3, 0, 0, 0, NA, 6),
    y = c(0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1)
  )
}

test_that("summarise() with na.rm = TRUE drops NaN computed on the GPU", {
  skip_if_no_gpu()
  df <- f22r3_nan_df()
  f22r3_both(df, function(d) {
    d |> mutate(z = x / y) |> group_by(g) |>
      summarise(s = sum(z, na.rm = TRUE), mn = mean(z, na.rm = TRUE),
                md = median(z, na.rm = TRUE), sdv = sd(z, na.rm = TRUE),
                v = var(z, na.rm = TRUE), q = quantile(z, 0.5, na.rm = TRUE, names = FALSE),
                nd = dplyr::n_distinct(z, na.rm = TRUE), f = dplyr::first(z, na_rm = TRUE),
                l = dplyr::last(z, na_rm = TRUE), n2 = dplyr::nth(z, 2, na_rm = TRUE),
                n = dplyr::n())
  })
  # min()/max() without the NaN-only group (its identity warns)
  f22r3_both(df[df$g != 3, ], function(d) {
    d |> mutate(z = x / y) |> group_by(g) |>
      summarise(lo = min(z, na.rm = TRUE), hi = max(z, na.rm = TRUE))
  })
  # a computed argument inside the aggregation, grouped and ungrouped
  f22r3_both(df, function(d) {
    d |> group_by(g) |> summarise(mn = mean(x / y, na.rm = TRUE), md = median(x / y, na.rm = TRUE))
  })
  f22r3_both(df, function(d) {
    summarise(d, mn = mean(x / y, na.rm = TRUE), lo = min(x / y, na.rm = TRUE),
              hi = max(x / y, na.rm = TRUE), md = median(x / y, na.rm = TRUE),
              q = quantile(x / y, 0.25, na.rm = TRUE, names = FALSE),
              s = sum(x / y, na.rm = TRUE), sdv = sd(x / y, na.rm = TRUE))
  })
})

test_that("summarise() with na.rm = FALSE never returns a number that skipped a GPU NaN", {
  skip_if_no_gpu()
  df <- f22r3_nan_df()[1:6, ]
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy) |> mutate(z = x / y)
    expect_error(collect(summarise(group_by(gt, g), q = quantile(z, 0.5))),
                 "missing values and NaN's not allowed")
    expect_error(collect(summarise(gt, q = quantile(z, 0.5))),
                 "missing values and NaN's not allowed")
    out <- collect(summarise(group_by(gt, g), lo = min(z), hi = max(z), md = median(z),
                             mn = mean(z), s = sum(z), sdv = sd(z), v = var(z),
                             .groups = "drop"))
    # R gives NaN (NA for median()); NA is the accepted NaN-as-NA divergence
    for (col in c("lo", "hi", "md", "mn", "s", "sdv", "v")) {
      expect_true(all(is.na(out[[col]])), info = col)
    }
    ung <- collect(summarise(gt, lo = min(z), hi = max(z), md = median(z), mn = mean(z)))
    expect_true(all(is.na(unlist(ung))))
  }
})

test_that("summarise() with na.rm = FALSE keeps R's NaN for sum()/mean() of a GPU NaN", {
  skip_if_no_gpu()
  df <- f22r3_nan_df()[1:6, ]
  f22r3_both(df, function(d) {
    d |> mutate(z = x / y) |> group_by(g) |> summarise(s = sum(z), mn = mean(z))
  })
})

test_that("aggregations of one window argument share its window stage", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), x = c(1, 2, 3, 4, 5))
  gt <- group_by(tbl_gpu(df), g)
  dots <- rlang::quos(
    a = max(cumsum(x)), b = min(cumsum(x)), c = sum(cumsum(x)),
    d = min(x - mean(x)), e = max(x - mean(x)), f = any(cumsum(x) > 3)
  )
  # (summarise_core() has stripped quantile()'s probs before this step)
  res <- preprocess_agg_expressions(gt, dots)
  arg <- function(nm) as.character(rlang::quo_get_expr(res$dots[[nm]])[[2]])
  expect_identical(arg("a"), arg("b"))
  expect_identical(arg("a"), arg("c"))
  expect_identical(arg("d"), arg("e"))
  expect_false(identical(arg("a"), arg("d")))
  # cumsum(x) once; mean(x) once plus x - mean(x) once; any()'s own column
  expect_lte(length(res$data$schema$names) - length(gt$schema$names), 5L)
  expect_same_as_dplyr(df, function(d) {
    d |> group_by(g) |>
      summarise(a = quantile(cumsum(x), 0.1, names = FALSE), b = quantile(cumsum(x), 0.9, names = FALSE),
                c = sum(cumsum(x)), d = quantile(x - mean(x), 0.25, names = FALSE),
                e = max(x - mean(x)), f = any(cumsum(x) > 3))
  })
})

test_that("reframe() plan leaves string min()/max()/range() to dplyr outside a C collation", {
  skip_if_no_gpu()
  schema <- list(names = c("g", "s", "x"), types = c("FLOAT64", "STRING", "FLOAT64"))
  withr::local_collate("C")
  expect_false(is.null(reframe_native_plan(rlang::quos(r = range(s)), schema)))
  expect_false(is.null(reframe_native_plan(rlang::quos(lo = min(s), q = quantile(x, c(0, 1))), schema)))
  withr::local_collate("en_US.UTF-8")
  expect_null(reframe_native_plan(rlang::quos(r = range(s)), schema))
  expect_null(reframe_native_plan(rlang::quos(lo = min(s), q = quantile(x, c(0, 1))), schema))
  expect_null(reframe_native_plan(rlang::quos(hi = max(s, na.rm = TRUE), r = range(x)), schema))
  expect_null(reframe_native_plan(rlang::quos(r = range(toupper(s))), schema))
  # numbers keep the native path whatever the collation
  expect_false(is.null(reframe_native_plan(rlang::quos(r = range(x), lo = min(x)), schema)))
})
