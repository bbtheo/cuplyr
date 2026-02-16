test_that("lower_head() returns a GPU table pointer", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  node <- ast_head(ast_source(gpu_df$schema), 5L)

  ptr <- lower_head(node, gpu_df$ptr)

  expect_true(inherits(ptr, "externalptr"))
  expect_equal(gpu_dim(ptr)[1], 5L)
  expect_equal(gpu_dim(ptr)[2], ncol(mtcars))
})
