# F124: cases from the round-1 review (reviews/round-1.md B1, B2, B4, B5).
# Helper: helper-parity-F124.R.

test_that("F124: a summarise over zero groups of a fixed-keys table warns once (fuzz seed 3864)", {
  skip_if_no_gpu()
  withr::local_collate("C")
  x0 <- tibble::tibble(b = character(0), c = numeric(0))
  f124_check(x0, function(d) summarise(filter(group_by(d, b, .drop = FALSE), b %in% "c"), s1 = max(c)))
  y0 <- tibble::tibble(c = numeric(0))
  f124_check(x0, function(d) {
    yy <- if (is_tbl_gpu(d)) tbl_gpu(y0, lazy = identical(d$exec_mode, "lazy")) else y0
    summarise(filter(inner_join(group_by(d, b, .drop = FALSE), yy, by = "c"), b %in% "c"), s1 = max(c))
  })
  g0 <- tibble::tibble(g = integer(0), x = numeric(0))
  f124_check(g0, function(d) summarise(filter(group_by(d, g), x > 0, .preserve = TRUE), m = max(x)))
})

test_that("F124: the integer type is decided from the parsed dots (across() function values, earlier dots)", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3, b = 4:6, y = c(1.5, 2.5, 3.5))
  f124_check(df, function(d) mutate(filter(d, a > 10L), across(c(a, b), max)))
  f124_check(df, function(d) mutate(filter(d, y > 10), y = y > 2, m = max(y)))
  f124_check(df, function(d) mutate(filter(d, y > 10), z = a + 1L, m = min(z)))
})

test_that("F124: across()/if_any() dots are labelled with the user's dot", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 2L, 2L), x = c(NA, 1, 2), y = c(NA, 3, 4))
  f124_check(df, function(d) summarise(group_by(d, g), across(c(x, y), ~ max(.x, na.rm = TRUE))))
  d2 <- tibble::tibble(a = integer(0), b = integer(0))
  f124_check(d2, function(d) mutate(d, across(c(a, b), max)))
  f124_check(d2, function(d) mutate(d, across(c(a, b), ~ min(.x))))
  f124_check(d2, function(d) filter(d, if_any(c(a, b), ~ .x == max(.x))))
})

test_that("F124: the warning names the verb as it was called", {
  skip_if_no_gpu()
  d0 <- tibble::tibble(x = integer(0), y = double(0))
  f124_check(d0, function(d) summarize(d, m = min(x)))
  f124_check(d0, function(d) dplyr::mutate(d, m = max(x)))
  f124_check(d0, function(d) dplyr::filter(d, y < max(y)))
  f124_check(d0, function(d) dplyr::summarise(d, m = max(y)))
})
