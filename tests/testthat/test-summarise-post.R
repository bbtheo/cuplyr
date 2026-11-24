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

# The notices a split call raises: the conditions `expr` signals, warnings muffled
sp_notices <- function(expr, class) {
  got <- list()
  withCallingHandlers(
    expr,
    condition = function(c) {
      if (inherits(c, class)) got[[length(got) + 1L]] <<- c
    },
    warning = function(w) invokeRestart("muffleWarning")
  )
  got
}

test_that("the plan-break notice of a split call names the user's dot, not a temp aggregate", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1L, 2L, 3L))
  got <- sp_notices(
    tbl_gpu(df, lazy = TRUE) |> filter(x > 0L) |> group_by(g) |> summarise(s = sum(x) + 1L) |> collect(),
    "cuplyr_plan_break"
  )
  expect_length(got, 1L)
  expect_match(got[[1L]]$reason, "(result type depends on the data): s", fixed = TRUE)
  expect_no_match(got[[1L]]$reason, ".post_agg", fixed = TRUE)
})

test_that("a declined post step claims the dot it stopped at, not an earlier native one", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3), y = c(4, 5, 6))
  for (lazy in c(FALSE, TRUE)) {
    got <- sp_notices(
      summarise(tbl_gpu(df, lazy = lazy), a = mean(x) + 1, b = dplyr::lag(mean(y))) |> collect(),
      "cuplyr_cpu_fallback"
    )
    expect_length(got, 1L)
    expect_identical(got[[1L]]$expr, "dplyr::lag(mean(y))")
    expect_match(got[[1L]]$reason, "function of the group's rows around a reducer", fixed = TRUE)
  }
})

test_that("a quosure embedded in a dot keeps its own environment for what is around its reducer", {
  skip_if_no_gpu()
  k <- 2
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  f <- function(d, stat) {
    k <- 100
    summarise(group_by(d, g), m = {{ stat }} * k, s = sum(x) / n())
  }
  f124_check(df, function(d) f(d, mean(x) + k))
  # the same reducer twice, once in each environment, is two leaves
  f2 <- function(d, stat) {
    w <- 5
    summarise(d, m = {{ stat }} + mean(x * w))
  }
  w <- 3
  f124_check(df, function(d) f2(d, mean(x * w)))
})

test_that("many repeated reducers in one dot share one leaf each", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3), y = c(2, 4, 9))
  f124_check(df, function(d) {
    summarise(group_by(d, g), r = (sum(x) + sum(y)) / (sum(x) - sum(y)) + sum(x) * sum(y))
  })
})
