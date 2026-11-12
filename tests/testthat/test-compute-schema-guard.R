# compute() names the result with the unoptimized schema; it must never put
# values under the wrong names if an optimizer pass changes the column order
# (F38 round-1 B1 was this failure mode), and must fail loudly on a bug that
# changes the column set.

test_that("compute() realigns a reordered optimized plan by name", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6, z = 7:9)
  lz <- tbl_gpu(df, lazy = TRUE) |> mutate(a = x + y)
  local_mocked_bindings(optimize_ast = function(ast) {
    ast_select(ast, rev(infer_schema(ast)$names))
  })
  res <- tibble::as_tibble(collect(lz))
  expect_identical(res, dplyr::mutate(df, a = x + y))
})

test_that("compute() fails loudly when the optimized plan has other columns", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  lz <- tbl_gpu(df, lazy = TRUE) |> mutate(a = x + y)
  local_mocked_bindings(optimize_ast = function(ast) ast_select(ast, c("x", "a")))
  expect_error(compute(lz), "Internal error")
})
