# A lazy mutate()/transmute() whose expressions read no column must not lose
# the row count when projection pruning drops every source column.

test_that("lazy constant mutate keeps the row count when all inputs are pruned", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, NA, 3), y = c(TRUE, FALSE, NA))
  pipelines <- list(
    function(d) dplyr::transmute(d, a = TRUE),
    function(d) dplyr::mutate(d, a = TRUE, .keep = "none"),
    function(d) dplyr::mutate(d, a = FALSE, .keep = "used"),
    function(d) dplyr::mutate(d, a = 1L) |> dplyr::select(a)
  )
  for (p in pipelines) {
    expect_same_as_dplyr_lazy(df, p)
  }
})
