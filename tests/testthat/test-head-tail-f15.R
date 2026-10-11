# head()/tail() on tbl_gpu follow utils::head.data.frame()/tail.data.frame():
# rows kept for every real `n`, and grouping ignored for the row selection
# (base subsets with `[`, which regroups the result).

test_that("head()/tail() keep base's row count for every real n", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5)
  ns <- c(0, -0, 0.5, 1, 2.5, 4.99, 5, 7, Inf,
          -0.25, -0.5, -0.99, -1, -1.5, -4.5, -5, -7, -Inf)
  for (n in ns) {
    expect_same_as_dplyr(df, function(d) head(d, n))
    expect_same_as_dplyr(df, function(d) tail(d, n))
    expect_same_as_dplyr_lazy(df, function(d) head(d, n))
    expect_same_as_dplyr_lazy(df, function(d) tail(d, n))
  }
})

test_that("head()/tail() on a grouped table take rows overall and keep the groups", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:6, g = c(1L, 1L, 1L, 2L, 2L, 2L))
  pipes <- list(
    function(d) head(group_by(d, g), 2),
    function(d) tail(group_by(d, g), 2),
    function(d) head(group_by(d, g), -4),
    function(d) tail(group_by(d, g), -0.5)
  )
  for (p in pipes) {
    cpu <- p(df)
    for (gpu in list(p(tbl_gpu(df)), p(as_lazy(tbl_gpu(df))))) {
      col <- collect(gpu)
      expect_equal(dplyr::group_vars(gpu), dplyr::group_vars(cpu))
      expect_s3_class(col, "grouped_df")
      expect_equal(dplyr::group_keys(col), dplyr::group_keys(cpu))
      expect_equal(tibble::as_tibble(dplyr::ungroup(col)),
                   tibble::as_tibble(dplyr::ungroup(cpu)))
    }
  }
})
