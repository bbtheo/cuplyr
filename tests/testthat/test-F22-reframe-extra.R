# F22 extras: reframe() shapes beyond the acceptance file -- optimizer
# composition (join sides, projection through the node), time zones, a literal
# next to a single group, wide probs vectors, and shapes that must stay on the
# CPU. dplyr keeps quantile()'s element names; a tbl_gpu column can't (see
# test-parity-F22-reframe.R), so the oracle side is unnamed.

f22x_unname <- function(df) {
  for (nm in names(df)) df[[nm]] <- unname(df[[nm]])
  tibble::as_tibble(df)
}

f22x_check <- function(df, pipeline, native = TRUE) {
  expected <- f22x_unname(pipeline(tibble::as_tibble(df)))
  for (lazy in c(FALSE, TRUE)) {
    got <- withr::with_options(
      list(cuplyr.fallback = if (native) "error" else "silent"),
      collect(pipeline(tbl_gpu(df, lazy = lazy)))
    )
    expect_equal(tibble::as_tibble(got), expected,
                 info = if (lazy) "lazy" else "eager")
  }
}

f22x_df <- function() {
  data.frame(
    g = c(2, 1, 2, 1, 3, 2, 1, 3),
    x = c(1.5, 2, 3.25, 4, 8, -2, 6.5, 0),
    i = c(7L, 3L, 1L, 9L, 4L, 2L, 5L, 8L),
    t = as.POSIXct("2024-01-01", tz = "America/New_York") + c(5, 2, 9, 0, 3, 7, 1, 4) * 3600
  )
}

test_that("a reframe() on a join side survives the optimizer", {
  skip_if_no_gpu()
  other <- data.frame(g = c(1, 2, 3, 4), w = c(10, 20, 30, 40))
  df <- f22x_df()
  for (lazy in c(FALSE, TRUE)) {
    run <- function(a, b) {
      b |>
        reframe(q = quantile(x, c(0.25, 0.75)), .by = g) |>
        (\(r) left_join(a, r, by = "g"))() |>
        select(w, q)
    }
    expected <- f22x_unname(run(tibble::as_tibble(other), tibble::as_tibble(df)))
    got <- withr::with_options(
      list(cuplyr.fallback = "error"),
      collect(run(tbl_gpu(other, lazy = lazy), tbl_gpu(df, lazy = lazy)))
    )
    expect_equal(tibble::as_tibble(got), expected)
  }
})

test_that("reframe() is a barrier: filters and mutates below/above keep their meaning", {
  skip_if_no_gpu()
  f22x_check(f22x_df(), function(d) {
    d |>
      filter(i > 1L) |>
      mutate(y = x * 2) |>
      group_by(g) |>
      reframe(q = quantile(y, c(0.2, 0.8)), n = n()) |>
      mutate(z = q + n) |>
      filter(z > 5)
  })
})

test_that("range() of a POSIXct keeps its time zone", {
  skip_if_no_gpu()
  df <- f22x_df()
  for (lazy in c(FALSE, TRUE)) {
    got <- withr::with_options(
      list(cuplyr.fallback = "error"),
      collect(reframe(group_by(tbl_gpu(df, lazy = lazy), g), r = range(t)))
    )
    want <- reframe(group_by(tibble::as_tibble(df), g), r = range(t))
    expect_equal(got, want)
    expect_identical(attr(got$r, "tzone"), "America/New_York")
  }
})

test_that("wide probs vectors and a literal beside a single group", {
  skip_if_no_gpu()
  df <- f22x_df()
  f22x_check(df, function(d) reframe(d, q = quantile(x, seq(0, 1, 0.1))))
  f22x_check(df, function(d) {
    reframe(d, k = 1:3, p = c("a", "b", "c"), q = quantile(i, c(0, 0.5, 1)), b = c(TRUE, NA, FALSE))
  })
  f22x_check(df, function(d) reframe(d, tag = "all", m = mean(x)))
})

test_that("reframe() with size-1 results and a literal of length 1 recycles", {
  skip_if_no_gpu()
  f22x_check(f22x_df(), function(d) reframe(group_by(d, g), one = 1L, m = max(i)))
})

test_that("shapes the plan can't prove equal stay on the CPU and match dplyr", {
  skip_if_no_gpu()
  df <- f22x_df()
  fb <- function(pipeline) f22x_check(df, pipeline, native = FALSE)
  # a probs vector that reads a column, a random literal, partial arg names
  fb(function(d) reframe(d, q = quantile(i, abs(x) / 10), .by = g))
  fb(function(d) reframe(d, r = runif(2) > 2, m = mean(i), .by = g))
  fb(function(d) reframe(d, q = quantile(i, prob = c(0.5, 0.9)), .by = g))
  # another namespace's quantile() is not stats' (and not ours to rewrite)
  fb(function(d) reframe(d, q = quantile(i, 0.5), r = range(x, finite = TRUE), .by = g))
})

test_that("a native reframe() result is ungrouped and chains into group_by()", {
  skip_if_no_gpu()
  res <- reframe(group_by(tbl_gpu(f22x_df(), lazy = TRUE), g), q = quantile(x, c(0.5, 0.9)))
  expect_identical(dplyr::group_vars(res), character(0))
  out <- collect(summarise(group_by(res, g), m = mean(q)))
  want <- summarise(group_by(reframe(group_by(tibble::as_tibble(f22x_df()), g),
                                     q = unname(quantile(x, c(0.5, 0.9)))), g), m = mean(q))
  expect_equal(out, want)
})
