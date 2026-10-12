# F18 round 3: a parse-time type check that fails only because a factor
# operand is involved signals cuplyr_factor_needs_cpu, and ir_parse_quo()
# turns that into NULL (CPU fallback). Type errors without a factor keep
# their plain errors.

f18n_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")),
    n = 1:6,
    s = c("a", "b", NA, "d", "e", "f")
  )
}

f18n_same <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("type checks raise cuplyr_factor_needs_cpu only for a factor operand", {
  expect_error(ir_unify_types(c("DICTIONARY32", "STRING"), "coalesce"),
               "coalesce\\(\\): can't combine <DICTIONARY32> and <STRING>",
               class = "cuplyr_factor_needs_cpu")
  expect_error(ir_unify_types(c("DICTIONARY32", "INT32"), "if_else"),
               "if_else\\(\\): can't unify types DICTIONARY32, INT32",
               class = "cuplyr_factor_needs_cpu")
  expect_error(infer_mutate_output_type("-", c("DICTIONARY32", "FLOAT64"), NULL),
               "does not support arithmetic on DICTIONARY32 columns",
               class = "cuplyr_factor_needs_cpu")

  # No factor and no timestamp (F37 routes those to the CPU too): the same
  # plain errors as before, not the fallback class.
  for (thunk in list(
    function() ir_unify_types(c("FLOAT64", "STRING"), "coalesce"),
    function() ir_unify_types(c("BOOL8", "STRING"), "if_else"),
    function() infer_mutate_output_type("+", c("STRING", "INT32"), NULL)
  )) {
    err <- tryCatch(thunk(), error = function(e) e)
    expect_s3_class(err, "error")
    expect_false(inherits(err, "cuplyr_factor_needs_cpu"))
  }
  expect_error(ir_unify_types(c("FLOAT64", "STRING"), "coalesce"),
               "coalesce\\(\\): can't combine <FLOAT64> and <STRING>")
})

test_that("ir_parse_quo() returns NULL when a nested type check trips on a factor", {
  schema <- list(names = c("f", "n"), types = c("DICTIONARY32", "INT32"))
  expect_null(ir_parse_quo(rlang::quo(paste0(coalesce(f, "none"), "!")), schema))
  expect_null(ir_parse_quo(rlang::quo(if_else(n > 2, f - 1, NA)), schema))

  # Without a factor the nested type error still propagates.
  schema2 <- list(names = c("x", "n"), types = c("FLOAT64", "INT32"))
  expect_error(ir_parse_quo(rlang::quo(paste0(coalesce(x, "none"), "!")), schema2),
               "can't combine <FLOAT64> and <STRING>")
})

test_that("factor-in-helper dots fall back in every verb that parses expressions", {
  skip_if_no_gpu()
  df <- f18n_df()
  f18n_same(df, function(d) dplyr::filter_out(d, grepl("^1", dplyr::coalesce(f, "none"))))
  f18n_same(df, function(d) dplyr::transmute(d, y = nchar(dplyr::coalesce(f, s))))
  f18n_same(df, function(d) dplyr::distinct(d, y = toupper(dplyr::coalesce(f, "z"))), arrange_by = "y")
  f18n_same(df, function(d) {
    d |>
      dplyr::group_by(y = tolower(dplyr::coalesce(f, "z"))) |>
      dplyr::summarise(m = sum(n))
  }, arrange_by = "y")  # grouped GPU summarise emits cuDF's group order (pre-existing)
  f18n_same(df, function(d) dplyr::summarise(d, k = sum(nchar(dplyr::coalesce(f, "none")))))
  f18n_same(df, function(d) dplyr::slice_min(d, nchar(dplyr::coalesce(f, "zzzz")), n = 2))
  f18n_same(df, function(d) dplyr::arrange(d, nchar(dplyr::coalesce(f, "zzzz")), n))
  f18n_same(df, function(d) dplyr::mutate(d, dplyr::across(f, ~ paste0(dplyr::coalesce(.x, "q"), "!"))))
  f18n_same(df, function(d) dplyr::mutate(d, y = dplyr::lag(paste0(dplyr::coalesce(f, "q")))))
})

test_that("a type mismatch without a factor still errors at parse time", {
  skip_if_no_gpu()
  df <- f18n_df()
  expect_error(dplyr::mutate(tbl_gpu(df), y = paste0(dplyr::coalesce(n, "a"), "!")),
               "can't combine <INT32> and <STRING>")
})

test_that("a mutate factor copy whose source has no recorded levels drops the old entry (R2-N3)", {
  input_schema <- list(
    names = c("f", "g"), types = c("DICTIONARY32", "DICTIONARY32"),
    factor_levels = list(f = c("a", "b"))
  )
  node <- list(type = "mutate", expressions = list(
    list(output_col = "f", output_type = "DICTIONARY32", ir = ir_col("g"))
  ))
  expect_null(propagate_factor_levels(node, input_schema, c("f", "g")))
  expect_null(update_schema_for_expr(input_schema, node$expressions[[1]])$factor_levels)
})
