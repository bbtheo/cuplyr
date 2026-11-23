# Snapshot tests for #13: the missing-column errors read as dplyr's.
#
# The snapshots in _snaps/parity-13-snapshots.md were recorded from dplyr 1.2.1
# on plain tibbles running this same code, so they are dplyr's text: a cuplyr
# table (eager and lazy) must print exactly the same error. They run with
# NOT_CRAN set, so tools/test and the merge gate run them (issue #86).

p13s_snapshots_on <- function(envir = parent.frame()) {
  withr::local_envvar(NOT_CRAN = "true", .local_envir = envir)
}

p13s_tbl <- function(df, lazy) tbl_gpu(df, lazy = lazy)

p13s_df <- function() {
  tibble::tibble(x = c(2L, 1L, 3L), y = c(1.5, NA, 3), g = c("a", "a", "b"))
}

p13s_df2 <- function() {
  tibble::tibble(x = c(1L, 2L, 4L), z = c(4L, 5L, 6L))
}

test_that("filter() missing-column errors (snapshot)", {
  skip_if_no_gpu()
  p13s_snapshots_on()
  d <- p13s_tbl(p13s_df(), lazy = FALSE)
  l <- p13s_tbl(p13s_df(), lazy = TRUE)
  expect_snapshot(error = TRUE, filter(d, xx > 1))
  expect_snapshot(error = TRUE, filter(l, xx > 1))
  expect_snapshot(error = TRUE, filter(group_by(d, g), xx > 1))
  expect_snapshot(error = TRUE, filter(l, x > 1, .data$xx > 1))
})

test_that("arrange() missing-column errors (snapshot)", {
  skip_if_no_gpu()
  p13s_snapshots_on()
  d <- p13s_tbl(p13s_df(), lazy = FALSE)
  l <- p13s_tbl(p13s_df(), lazy = TRUE)
  expect_snapshot(error = TRUE, arrange(d, xx))
  expect_snapshot(error = TRUE, arrange(l, x, desc(xx)))
})

test_that("summarise() missing-column errors (snapshot)", {
  skip_if_no_gpu()
  p13s_snapshots_on()
  d <- p13s_tbl(p13s_df(), lazy = FALSE)
  l <- p13s_tbl(p13s_df(), lazy = TRUE)
  expect_snapshot(error = TRUE, summarise(d, m = mean(xx)))
  expect_snapshot(error = TRUE, summarise(group_by(l, g), m = mean(xx)))
  expect_snapshot(error = TRUE, summarise(d, across(xx, mean)))
  expect_snapshot(error = TRUE, count(l, g, wt = xx))
})

test_that("mutate() and group_by() missing-column errors (snapshot)", {
  skip_if_no_gpu()
  p13s_snapshots_on()
  d <- p13s_tbl(p13s_df(), lazy = FALSE)
  l <- p13s_tbl(p13s_df(), lazy = TRUE)
  expect_snapshot(error = TRUE, mutate(d, z = xx + 1))
  expect_snapshot(error = TRUE, mutate(l, z = .data$xx))
  expect_snapshot(error = TRUE, group_by(d, xx))
  expect_snapshot(error = TRUE, group_by(l, xx, yy))
})

test_that("tidyselect missing-column errors (snapshot)", {
  skip_if_no_gpu()
  p13s_snapshots_on()
  d <- p13s_tbl(p13s_df(), lazy = FALSE)
  l <- p13s_tbl(p13s_df(), lazy = TRUE)
  expect_snapshot(error = TRUE, select(d, xx))
  expect_snapshot(error = TRUE, relocate(l, xx))
  expect_snapshot(error = TRUE, rename(d, a = xx))
})

test_that("join missing-column errors (snapshot)", {
  skip_if_no_gpu()
  p13s_snapshots_on()
  d <- p13s_tbl(p13s_df(), lazy = FALSE)
  l <- p13s_tbl(p13s_df(), lazy = TRUE)
  d2 <- p13s_tbl(p13s_df2(), lazy = FALSE)
  l2 <- p13s_tbl(p13s_df2(), lazy = TRUE)
  expect_snapshot(error = TRUE, left_join(d, d2, by = "xx"))
  expect_snapshot(error = TRUE, inner_join(l, l2, by = c(x = "zz")))
  expect_snapshot(error = TRUE, left_join(d, d2, by = c("xx", "yy")))
})
