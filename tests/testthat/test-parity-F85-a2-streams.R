# F85 attempt 2 (B7): libcudf forks CUDA streams for parquet reads and writes,
# and under the default async memory resource that leaves later arrange(),
# filter() and summarise() calls in the process 1-8% slower. With every forked
# stream being the default one the cost is gone, and the reads and writes
# measured were no slower (scratchpad/parity/features/F85-a2.md). The
# variable has to be in the environment before libcudf first builds its stream
# pool, so the package sets it at load, unless the user already has.

test_that("libcudf's stream pool is told to use the default stream from package load", {
  expect_false(is.na(Sys.getenv("LIBCUDF_USE_DEBUG_STREAM_POOL", NA_character_)))
})
