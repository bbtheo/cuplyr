# Memory regression tests for lazy execution

test_that("lazy mutate chain uses near-constant GPU memory", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(256 * 1024 * 1024)

  df <- data.frame(x = runif(1e6), y = runif(1e6))

  gpu_gc(verbose = FALSE)
  before <- gpu_memory_state()$used_bytes

  result <- tbl_gpu(df, lazy = TRUE) |>
    mutate(a = x + y) |>
    mutate(b = a * 2) |>
    mutate(c = b - x) |>
    mutate(d = c / y)

  during <- gpu_memory_state()$used_bytes

  source_size <- nrow(df) * 2 * 8
  expect_lt(during - before, source_size * 1.5)

  collected <- result |> collect()
  expected_d <- with(df, ((x + y) * 2 - x) / y)
  expect_equal(collected$d, expected_d, tolerance = 1e-10)

  rm(result, collected)
  gpu_gc(verbose = FALSE)

  final <- gpu_memory_state()$used_bytes
  expect_lt(final - before, source_size * 0.2)
})
