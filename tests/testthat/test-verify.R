test_that("verify_installation() returns expected structure", {
  skip_if_no_gpu()

  result <- verify_installation()

  expect_type(result, "list")
  expect_true("ok" %in% names(result))
  expect_true("package_loaded" %in% names(result))
  expect_true("gpu_available" %in% names(result))
  expect_true("gpu_name" %in% names(result))
  expect_true("roundtrip_ok" %in% names(result))
  expect_true("time_seconds" %in% names(result))
})

test_that("verify_installation() passes on working system", {
  skip_if_no_gpu()

  result <- verify_installation()

  expect_true(result$ok)
  expect_true(result$package_loaded)
  expect_true(result$gpu_available)
  expect_true(result$roundtrip_ok)
  expect_true(is.numeric(result$time_seconds))
  expect_true(result$time_seconds >= 0)
  expect_true(nzchar(result$gpu_name))
})
