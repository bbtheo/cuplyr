# summarise() with arithmetic around reducers (#22): cases beyond the acceptance
# file test-parity-22-post-agg.R. They use f124_check() (helper-parity-F124.R):
# values, types, warnings, group vars and no CPU fallback, eager and lazy.

test_that("a column named like the split's temp aggregate is not overwritten", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), .post_agg1 = c(1, 2, 3), x = c(1, 2, 3))
  f124_check(df, function(d) summarise(group_by(d, g), m = mean(.post_agg1) + 1, s = sum(x) / n()))
})

test_that("each min()/max() of a dot warns once, as dplyr evaluates each call", {
  skip_if_no_gpu()
  fdf <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c", "d")),
                        x = c(1, 2, 3))
  f124_check(fdf, function(d) {
    summarise(group_by(d, f, .drop = FALSE), r = max(x) - min(x), t = max(x) * 2)
  })
})

test_that("the same reducer in two dots, and a post dot over a plain one", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 5), y = c(2, 4, 6, 8))
  f124_check(df, function(d) {
    summarise(group_by(d, g), a = mean(x) + 1, b = mean(x) * 2, m = mean(y), r = m / mean(x))
  })
})

test_that("a composite summarise followed by select() keeps what the plan needs", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 5), y = c(2, 4, 6, 8))
  f124_check(df, function(d) {
    d |> group_by(g) |> summarise(a = mean(x) + 1, b = sum(y) / n()) |> select(b, g)
  })
  f124_check(df, function(d) {
    d |> summarise(a = mean(x) + 1, b = sum(y) / n(), .by = g) |> filter(b > 4)
  })
})

test_that("a split call that dplyr runs is not given a post step on the CPU result", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  # a column beside an aggregate: dplyr's error, on every route
  withr::local_options(cuplyr.fallback = "silent")
  expect_error(collect(summarise(group_by(tbl_gpu(df), g), m = mean(x) + x)), "must be size 1")
  expect_error(collect(summarise(group_by(tbl_gpu(df, lazy = TRUE), g), m = mean(x) + x)), "must be size 1")
})
