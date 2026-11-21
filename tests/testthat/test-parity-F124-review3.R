# F124: cases from the round-3 review (reviews/round-3.md B10).
# Helper: helper-parity-F124.R.

test_that("F124: an empty argument before a min()/max() call next to an impure call (lazy filter)", {
  skip_if_no_gpu()
  da <- tibble::tibble(a = 1:3)
  m <- matrix(1:4, 2)
  mode_val <- "low"
  # (runif(1) * 0 is 0 whatever the draw, so dplyr and the GPU agree)
  f124_check(da, function(d) mutate(filter(d, a > 1L), z = a + m[, 1][1], r = runif(1) * 0, k = max(a)))
  f124_check(da, function(d) {
    mutate(filter(d, a > 1L), z = a + switch(mode_val, low = , mid = 1L, 2L), r = runif(1) * 0, k = max(a))
  })
  f124_check(da, function(d) mutate(filter(d, a > 1L), r = runif(1) * 0, k = m[, 1][1] + max(a)))
  # the same over a table the filter empties: dplyr's double and warning
  f124_check(da, function(d) mutate(filter(d, a > 10L), z = a + m[, 1][1], r = runif(1) * 0, k = max(a)))
})
