# F61 round 2 (Theo, 2026-10-23): every group_split() chunk owns only its own
# rows' device memory. A kept chunk must not keep the gathered table, the
# input, or any other chunk's rows alive, whether or not it was ever used.
#
# Measured at the allocator (`pool_used_bytes`, see helper-memory.R): bytes
# currently allocated from cuplyr's memory resource to live objects. All
# columns are fixed-width so gpu_memory_usage()'s estimate is close to the
# real size.

f61mem_used <- function() {
  gc(full = TRUE)
  gpu_sync_device()
  gpu_alloc_used_bytes()
}

f61mem_df <- function(n) {
  set.seed(6161)
  data.frame(
    k = sample.int(2000L, n, replace = TRUE),
    x = runif(n),
    y = rnorm(n),
    i = sample(c(1:100, NA), n, replace = TRUE)
  )
}

test_that("one kept, never-used chunk of a 5e6-row split holds only its own rows", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(1.5e9)
  withr::local_options(cuplyr.fallback = "error")
  base <- f61mem_used()

  gt <- tbl_gpu(f61mem_df(5e6))
  input_bytes <- f61mem_used() - base
  expect_gt(input_bytes, 100e6)

  parts <- dplyr::group_split(gt, k)
  first <- parts[[1]]
  rm(parts, gt)
  held <- f61mem_used() - base

  own <- gpu_memory_usage(first)
  expect_lt(own, 1e6)
  # The chunk's own bytes plus alignment slack, nowhere near the input's size.
  expect_lt(held, own + 2e6)

  # gpu_memory_usage() of a chunk reports the chunk, sized like a fresh copy.
  copy <- tbl_gpu(dplyr::collect(first))
  expect_equal(own, gpu_memory_usage(copy))
  rm(copy)

  # Using the chunk changes nothing; dropping it frees everything.
  expect_identical(nrow(dplyr::collect(first)), as.integer(dim(first)[1]))
  expect_lt(f61mem_used() - base, own + 2e6)
  rm(first)
  expect_lt(f61mem_used() - base, 2e6)
})

test_that("keeping every chunk holds about one copy of the input, not two", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(1.5e9)
  withr::local_options(cuplyr.fallback = "error")
  base <- f61mem_used()

  gt <- tbl_gpu(f61mem_df(2e6))
  input_bytes <- f61mem_used() - base
  parts <- dplyr::group_split(gt, k)
  rm(gt)
  held <- f61mem_used() - base
  # The gathered intermediate is gone; the chunks together are ~ the input
  # (plus per-chunk alignment padding for 2000 chunks).
  expect_lt(held, 1.25 * input_bytes)
  expect_gt(held, 0.9 * input_bytes)

  # Dropping all but a few chunks frees the rest.
  keep <- parts[c(1L, 1000L, 2000L)]
  rm(parts)
  kept <- f61mem_used() - base
  own <- sum(vapply(keep, gpu_memory_usage, numeric(1)))
  expect_lt(kept, own + 3e6)
  rm(keep)
})
