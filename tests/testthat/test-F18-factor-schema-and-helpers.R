# F18-a2 round 2: running-schema factor levels inside one call (R1-B1) and
# parse-time type checks that must not hard-error on a factor x (R1-N1).

f18s_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")),
    f2 = factor(c("1", "2", "10", NA, "2", "10"), levels = c("10", "1", "2")),
    f3 = factor(c("1", "2", "10", NA, "2", "10"), levels = c("1", "2", "10")),
    n = 1:6
  )
}

f18s_same <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("a column dropped with NULL loses its levels for later dots", {
  skip_if_no_gpu()
  df <- f18s_df()
  f18s_same(df, function(d) dplyr::mutate(d, f2 = NULL, f2 = n, y = f == f2))
  f18s_same(df, function(d) dplyr::mutate(d, f2 = f3, f3 = NULL, y = f == f2))
  f18s_same(df, function(d) dplyr::mutate(d, y = f == f2, f2 = f3))
})

test_that("string values against a factor x in the other value helpers fall back", {
  skip_if_no_gpu()
  df <- f18s_df()
  f18s_same(df, function(d) dplyr::mutate(d, y = dplyr::recode_values(f, "1" ~ "one", default = "o")))
  f18s_same(df, function(d) dplyr::mutate(d, y = dplyr::replace_when(f, f == "1" ~ "2")))
  f18s_same(df, function(d) dplyr::filter(d, dplyr::na_if(f, "1") == "2"))
})

test_that("update_schema_for_expr() maintains factor_levels", {
  sch <- list(names = c("f", "n"), types = c(f = "DICTIONARY32", n = "INT32"),
              factor_levels = list(f = c("a", "b")))
  copy <- update_schema_for_expr(sch, list(output_col = "h", output_type = c(h = "DICTIONARY32"), ir = ir_col("f")))
  expect_identical(copy$factor_levels, list(f = c("a", "b"), h = c("a", "b")))
  repl <- update_schema_for_expr(sch, list(output_col = "f", output_type = c(f = "INT32"), ir = ir_col("n")))
  expect_null(repl$factor_levels)
})
