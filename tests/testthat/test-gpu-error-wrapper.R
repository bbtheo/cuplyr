test_that("wrap_gpu_call adds a readable GPU error message", {
  expect_error(
    wrap_gpu_call("test_op", stop("low-level error")),
    "GPU operation 'test_op' failed"
  )
  expect_error(
    wrap_gpu_call("test_op", stop("low-level error")),
    "Original error: low-level error"
  )
})
