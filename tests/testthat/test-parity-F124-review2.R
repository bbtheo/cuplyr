# F124: cases from the round-2 review (reviews/round-2.md B7, B8, B9, N17).
# Helper: helper-parity-F124.R.

test_that("F124: min()/max() of a one-value-per-group argument neither warns nor turns double", {
  skip_if_no_gpu()
  d0 <- tibble::tibble(x = integer(0))
  f124_check(d0, function(d) mutate(d, m = max(n())))
  f124_check(d0, function(d) filter(d, n() == max(n())))
  # the inner max() sees the empty group, the outer one its single value
  f124_check(d0, function(d) mutate(d, m = max(max(x))))
  fd <- tibble::tibble(f = factor(c("a", "a"), levels = c("a", "b")), x = 1:2)
  f124_check(fd, function(d) mutate(group_by(d, f, .drop = FALSE), m = max(n())))
  f124_check(fd, function(d) mutate(group_by(d, f, .drop = FALSE), m = max(sum(x))))
  f124_check(fd, function(d) mutate(group_by(d, f, .drop = FALSE), m = min(x), k = max(n())))
})

test_that("F124: a Date shifted by an integer min()/max() over an empty group stays on the GPU", {
  skip_if_no_gpu()
  dd <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), a = 1:3,
                       b = c(TRUE, FALSE, TRUE), d = as.Date("2020-01-01") + 0:2)
  f124_check(dd, function(d) mutate(filter(d, a > 10L), e = d + max(a)))
  f124_check(dd, function(d) mutate(mutate(group_by(d, f, .drop = FALSE), z = a + 1L), e = d + max(a)))
  f124_check(dd, function(d) mutate(group_by(d, f, .drop = FALSE), e = d + min(b)))
  # control: no empty group
  f124_check(dd, function(d) mutate(filter(d, a > 1L), e = d + max(a)), lazy_warns_at = "collect")
})

test_that("F124: a lazy mutate() stays lazy when it ran its pending steps for min()/max()", {
  skip_if_no_gpu()
  da <- tibble::tibble(a = 1:3)
  r <- suppressWarnings(mutate(filter(tbl_gpu(da, lazy = TRUE), a > 10L), m = max(a), r = runif(1), z = rev(a)))
  expect_identical(unclass(r)$exec_mode, "lazy")
  r2 <- suppressWarnings(mutate(filter(tbl_gpu(da, lazy = TRUE), a > 10L), m = max(a)))
  expect_identical(unclass(r2)$exec_mode, "lazy")
  dd <- tibble::tibble(a = 1:3, d = as.Date("2020-01-01") + 0:2)
  r3 <- suppressWarnings(mutate(filter(tbl_gpu(dd, lazy = TRUE), a > 10L), e = d + max(a)))
  expect_identical(unclass(r3)$exec_mode, "lazy")
})
