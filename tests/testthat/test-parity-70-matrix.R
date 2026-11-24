# Acceptance tests for #70, item 2: a matrix column with more than one column
# has no GPU form. Wherever a CPU value would go to the GPU (tbl_gpu() itself,
# or the result of a verb that ran on the CPU), cuplyr raises a classed error
# that names the column, never a raw cuDF error and never a result with the
# wrong number of rows. List and data frame columns get the same classed error
# at tbl_gpu(). (A one-column matrix such as scale()'s still uploads as a plain
# column; keeping its dim is #21.)

p70m_df <- tibble::tibble(a = rep(1:3, each = 2), b = as.double(1:6))

# `f(tbl)` must fail with `class` and a message naming `col`, eager and lazy.
p70m_expect_error <- function(f, class, col, df = p70m_df) {
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(f(tbl_gpu(df, lazy = lazy))), regexp = paste0("`", col, "`"),
                 class = class, fixed = TRUE, info = paste("lazy =", lazy))
  }
}

test_that("tbl_gpu() names a matrix, list or data frame column it can't upload", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    expect_error(tbl_gpu(tibble::tibble(a = 1:2, m = matrix(1:4, ncol = 2)), lazy = lazy),
                 "`m`", fixed = TRUE, class = "cuplyr_error_unsupported_class")
    expect_error(tbl_gpu(tibble::tibble(a = 1:2, m = array(1:8, c(2, 2, 2))), lazy = lazy),
                 "`m`", fixed = TRUE, class = "cuplyr_error_unsupported_class")
    expect_error(tbl_gpu(tibble::tibble(a = 1:2, l = list(1, "x")), lazy = lazy),
                 "`l`", fixed = TRUE, class = "cuplyr_error_unsupported_class")
    expect_error(tbl_gpu(tibble::tibble(a = 1:2, d = tibble::tibble(x = 1:2)), lazy = lazy),
                 "`d`", fixed = TRUE, class = "cuplyr_error_unsupported_class")
  }
  # the error is not a GPU failure
  err <- tryCatch(tbl_gpu(tibble::tibble(a = 1:2, m = matrix(1:4, ncol = 2))),
                  error = identity)
  expect_false(inherits(err, "cuplyr_error_gpu"))
})

test_that("mutate() with a matrix result raises a classed error naming the column", {
  skip_if_no_gpu()
  cls <- "cuplyr_error_unsupported_result_class"
  p70m_expect_error(function(d) mutate(d, m = cbind(a, b)), cls, "m")
  p70m_expect_error(function(d) mutate(group_by(d, a), m = cbind(a, b)), cls, "m")
  p70m_expect_error(function(d) mutate(rowwise(d), m = cbind(a, b)), cls, "m")
  p70m_expect_error(function(d) mutate(d, m = as.matrix(data.frame(a, b))), cls, "m")
  p70m_expect_error(function(d) mutate(d, m = array(1:12, c(6, 1, 2))), cls, "m")
  p70m_expect_error(function(d) mutate(d, m = matrix(1:12, ncol = 2)), cls, "m")
})

test_that("transmute(), summarise() and reframe() never return the matrix's cells as rows", {
  skip_if_no_gpu()
  cls <- "cuplyr_error_unsupported_result_class"
  p70m_expect_error(function(d) transmute(d, m = cbind(a, b)), cls, "m")
  p70m_expect_error(function(d) summarise(d, m = matrix(1:2, ncol = 2)), cls, "m")
  p70m_expect_error(function(d) summarise(group_by(d, a), m = matrix(c(min(b), max(b)), ncol = 2)),
                    cls, "m")
  p70m_expect_error(function(d) reframe(d, m = matrix(1:4, ncol = 2)), cls, "m")
})

test_that("one-column matrices still work as before", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    out <- collect(mutate(tbl_gpu(p70m_df, lazy = lazy), b = scale(b)))
    expect_equal(as.vector(out$b), as.vector(scale(p70m_df$b)))
    out <- collect(mutate(group_by(tbl_gpu(p70m_df, lazy = lazy), a), b = scale(b)))
    expected <- mutate(group_by(p70m_df, a), b = scale(b))
    expect_equal(as.vector(out$b), as.vector(expected$b))
    out <- collect(tbl_gpu(tibble::tibble(a = 1:2, m = matrix(1:2, ncol = 1)), lazy = lazy))
    expect_equal(as.vector(out$m), 1:2)
  }
})
