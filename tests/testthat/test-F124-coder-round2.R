# F124 round 2: the plan breaks and costs of a window min()/max() (reviews/round-1.md
# N1, N2, N4, N5, B3). Helper: helper-parity-F124.R.

test_that("F124: a double min()/max(), or a column called max, over unknown rows breaks no plan", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3, y = c(1.5, 2.5, 3.5), max = c(1, 2, 3))
  f124_check(df, function(d) mutate(filter(d, a > 10L), m = max(y)), lazy_warns_at = "collect")
  f124_check(df, function(d) mutate(filter(d, a > 10L), z = max + 1), lazy_warns_at = "collect")
  f124_check(df, function(d) filter(filter(d, a > 10L), y == max(y)), lazy_warns_at = "collect")
  # (a column named max inside a call that is not max())
  f124_check(df, function(d) mutate(filter(d, a > 10L), z = abs(max)), lazy_warns_at = "collect")
})

test_that("F124: an integer min()/max() over unknown rows runs the plan once and keeps dplyr's type", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3, y = c(1.5, 2.5, 3.5))
  f124_check(df, function(d) mutate(filter(d, a > 10L), m = max(a)))
  f124_check(df, function(d) mutate(filter(d, a > 10L), m = max(a), n = min(y)))
  got <- f124_capture(mutate(filter(tbl_gpu(df, lazy = TRUE), a > 10L), m = max(a)))
  expect_length(got$plan_break, 1L)
  # nothing is empty: the type stays integer, the plan still runs once
  keep <- f124_capture(collect(mutate(filter(tbl_gpu(df, lazy = TRUE), a > 1L), m = max(a))))
  expect_length(keep$plan_break, 1L)
  expect_identical(keep$value$m, c(3L, 3L))
  expect_length(keep$warnings, 0L)
})

test_that("F124: the helper join of cur_group_id() leaves no integer min()/max() typed wrongly", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), a = 1:3)
  f124_check(df, function(d) mutate(group_by(filter(d, a > 10L), g), id = dplyr::cur_group_id(), m = max(a)))
  f124_check(df, function(d) mutate(group_by(filter(d, a > 1L), g), id = dplyr::cur_group_id(), m = max(a)),
             lazy_warns_at = "any")
})

test_that("F124: a grouping that keeps empty groups after pending steps runs them once, for the warning", {
  skip_if_no_gpu()
  fdf <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), x = c(1, 2, 3), i = 1:3)
  f124_check(fdf, function(d) mutate(mutate(group_by(d, f, .drop = FALSE), z = i + 1L), m = max(x)))
  f124_check(fdf, function(d) filter(mutate(group_by(d, f, .drop = FALSE), z = i + 1L), x == max(x)))
  got <- f124_capture(mutate(mutate(group_by(tbl_gpu(fdf, lazy = TRUE), f, .drop = FALSE), z = i + 1L), m = max(x)))
  expect_length(got$plan_break, 1L)
  expect_length(got$warnings, 1L)
  # nothing pending: read at the verb, no break
  none <- f124_capture(mutate(group_by(tbl_gpu(fdf, lazy = TRUE), f, .drop = FALSE), m = max(x)))
  expect_length(none$plan_break, 0L)
  expect_length(none$warnings, 1L)
})

test_that("F124: a column-free call next to a lazy min()/max() is evaluated once", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3)
  n <- 0L
  bump <- function() {
    n <<- n + 1L
    1
  }
  res <- suppressWarnings(collect(mutate(filter(tbl_gpu(df, lazy = TRUE), a > 10L), m = max(a + bump()))))
  expect_identical(n, 1L)
  expect_identical(nrow(res), 0L)
  expect_type(res$m, "double")
})

test_that("F124: a window min()/max() with nothing empty builds no group index at the verb", {
  skip_if_no_gpu()
  fdf <- tibble::tibble(f = factor(c("a", "b", "b"), levels = c("a", "b")), x = c(1, 2, 3))
  local_mocked_bindings(compute_group_data = function(...) stop("the group index was built"))
  for (lazy in c(TRUE, FALSE)) {
    g <- group_by(tbl_gpu(fdf, lazy = lazy), f, .drop = FALSE)
    expect_no_warning(m <- mutate(g, m = max(x)))
    expect_equal(collect(m)$m, c(1, 3, 3))
    expect_no_warning(k <- filter(g, x == max(x)))
    expect_equal(collect(k)$x, c(1, 3))
  }
})
