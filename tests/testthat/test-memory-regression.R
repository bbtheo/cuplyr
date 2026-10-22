# Memory regression tests for lazy execution

test_that("lazy mutate chain uses near-constant GPU memory", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(256 * 1024 * 1024)

  df <- data.frame(x = runif(1e6), y = runif(1e6))

  # The first lazy mutate pipeline in a process allocates ~18 MiB once (lazy
  # CUDA module loading / cudf first-use state) and never returns it. Pay that
  # here, on a small table, so it isn't counted against the pipeline below.
  # Without it this test fails 5/5 when run alone and only passes in the suite
  # because earlier files have already paid.
  tbl_gpu(data.frame(x = runif(1e3), y = runif(1e3)), lazy = TRUE) |>
    mutate(a = x + y) |>
    mutate(b = a * 2) |>
    mutate(c = b - x) |>
    mutate(d = c / y) |>
    collect()

  # The deltas below are bytes allocated from cuplyr's resource
  # (pool_used_bytes; device-level `used_bytes` when there is none): a memory
  # pool keeps its pages, so device-level deltas would read 0.
  gpu_gc(verbose = FALSE)
  before <- gpu_alloc_used_bytes()

  result <- tbl_gpu(df, lazy = TRUE) |>
    mutate(a = x + y) |>
    mutate(b = a * 2) |>
    mutate(c = b - x) |>
    mutate(d = c / y)

  during <- gpu_alloc_used_bytes()

  source_size <- nrow(df) * 2 * 8
  expect_lt(during - before, source_size * 1.5)

  collected <- result |> collect()
  expected_d <- with(df, ((x + y) * 2 - x) / y)
  expect_equal(collected$d, expected_d, tolerance = 1e-10)

  rm(result, collected)
  gpu_gc(verbose = FALSE)

  final <- gpu_alloc_used_bytes()
  expect_lt(final - before, source_size * 0.2)
})
