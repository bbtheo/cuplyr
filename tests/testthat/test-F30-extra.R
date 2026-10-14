# F30 round 2: extra coverage for the NaN clause, the clause-folding handler
# shared by case_when()/replace_when(), the parser's empty-argument handling,
# and the verbs that reach recode() through other routes.

recode <- dplyr::recode

f30x_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("the parser returns NULL (not an error) for empty arguments", {
  schema <- list(names = c("x", "n"), types = c(x = "STRING", n = "INT32"))
  env <- environment()
  expect_null(ir_parse_recode(quote(recode(, a = "A")), env, schema))
  expect_null(ir_parse_recode(quote(recode(x, a = )), env, schema))
  expect_null(ir_parse_recode(quote(recode(x, a = "A", .default = )), env, schema))
  # a well-formed call still lowers
  expect_false(is.null(ir_parse_recode(quote(recode(x, a = "A")), env, schema)))
})

test_that("a computed NaN is missing for every FLOAT64 recode shape", {
  skip_if_no_gpu()
  df <- data.frame(d = c(0, 1, 2, NA, 3), g = c(1L, 1L, 2L, 2L, 2L))
  # NaN made in the same mutate, in an earlier mutate, then used by filter,
  # grouped mutate.
  f30x_native(df, function(d) {
    d |> dplyr::mutate(z = d / d, b = recode(z, `1` = "one", .default = "o"),
                       w = recode(z, `1` = 10, .missing = -1), v = recode(z, `1` = 10),
                       u = recode(z, 10, 20))
  })
  f30x_native(df, function(d) {
    d |> dplyr::mutate(z = d / d) |> dplyr::mutate(w = recode(z, `1` = 10, .missing = -1))
  })
  f30x_native(df, function(d) {
    d |> dplyr::mutate(z = d / d) |> dplyr::filter(recode(z, `1` = 10, .missing = -1) == -1)
  })
  f30x_native(df, function(d) {
    d |> dplyr::mutate(z = d / d) |> dplyr::group_by(g) |>
      dplyr::mutate(m = recode(z, `1` = 1, .missing = 0))
  })
})

test_that("case_when() and replace_when() fold column values, casts and NA conditions", {
  skip_if_no_gpu()
  df <- data.frame(a = c(1L, 2L, NA, 4L, 5L), b = c(1.5, 2.5, 3.5, NA, 5.5),
                   s = c("x", "y", NA, "z", "x"), stringsAsFactors = FALSE)
  # column values needing a cast, literal values, and a later expression that
  # reads the earlier outputs (indices must stay valid after scratch is freed)
  f30x_native(df, function(d) {
    d |> dplyr::mutate(
      y = dplyr::case_when(a == 1L ~ b, a == 2L ~ 9L, a == 4L ~ b * 2, .default = -1),
      z = y + a,
      w = dplyr::case_when(s == "x" ~ 1, s %in% c("y", "z") ~ 2),
      v = dplyr::case_when(is.na(a) ~ "na", a > 2L ~ s, .default = "low"),
      r = dplyr::replace_when(b, a == 1L ~ 0, a == 2L ~ b * 2, is.na(a) ~ -9)
    )
  })
  # nested handlers: each frame frees only what its own clauses pushed
  f30x_native(df, function(d) {
    d |> dplyr::mutate(y = dplyr::case_when(
      dplyr::case_when(a > 3L ~ TRUE, .default = FALSE) ~ b,
      dplyr::coalesce(a, 0L) == 0L ~ 100,
      .default = 0))
  })
})

test_that("recode() runs natively under count(), distinct(), slice and across()", {
  skip_if_no_gpu()
  df <- data.frame(x = c("a", "b", "a", NA, "c", "b"), n = c(3L, 1L, 2L, 5L, 4L, 1L),
                   stringsAsFactors = FALSE)
  f30x_native(df, function(d) d |> dplyr::count(y = recode(x, a = "A", b = "B")),
              arrange_by = "y")
  f30x_native(df, function(d) d |> dplyr::distinct(y = recode(n, `1` = 10L, `2` = 20L)),
              arrange_by = "y")
  f30x_native(df, function(d) {
    d |> dplyr::slice_max(order_by = recode(n, `1` = 10L, .default = 0L), n = 3, with_ties = FALSE)
  })
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, function(d) {
    d |> dplyr::mutate(dplyr::across(x, ~ recode(.x, a = "A", .default = "?")))
  })
})
