# F67 extras: shapes the acceptance tests don't reach (logical digits, folded
# -0 constants in value positions, three-argument round()).

test_that("round() with a logical digits value and a folded constant digits", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.05, 7.55, 0.15, -1.05, 1250, NA))
  f67_check(df, function(d) dplyr::mutate(d, a = round(x, TRUE), b = round(x, 2 - 3), c = round(x, 3 %/% 2)))
})

test_that("a product that evaluates to -0 keeps its sign in value positions", {
  skip_if_no_gpu()
  k <- -1
  df <- data.frame(x = c(1, -1, 0, NA), y = c(NA, 2, NA, 3))
  f67_check(df, function(d) {
    dplyr::mutate(d,
      a = dplyr::if_else(x > 5, 1, -1 * 0), b = dplyr::coalesce(y, -1 * 0),
      c = dplyr::case_when(x > 0 ~ -1 * 0, .default = 1), e = dplyr::na_if(y, 2) + (-1 * 0)
    )
  })
  # k * 0 is folded by the IR (ir_fold_literal_call()) to a -0 double
  f67_check(df, function(d) {
    dplyr::mutate(d,
      a = dplyr::if_else(x > 5, 1, k * 0), b = dplyr::coalesce(y, k * 0),
      c = dplyr::case_when(x > 0 ~ k * 0, .default = 1), e = x / (k * 0)
    )
  })
})

test_that("round() with three arguments is dplyr's error", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2.5))
  # dplyr's error, caused by R's own error from round()
  want <- tryCatch(dplyr::mutate(tibble::as_tibble(df), r = round(x, 1, 2)), error = identity)
  expect_s3_class(want, "dplyr:::mutate_error")
  msg <- conditionMessage(want$parent)
  expect_error(dplyr::mutate(tbl_gpu(df), r = round(x, 1, 2)) |> dplyr::collect(), msg, fixed = TRUE)
  expect_error(dplyr::mutate(tbl_gpu(df, lazy = TRUE), r = round(x, 1, 2)) |> dplyr::collect(),
               msg, fixed = TRUE)
})
