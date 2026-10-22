# F56: measure GPU allocations at the RMM allocator.
#
# Device-level free memory (cudaMemGetInfo) cannot see allocations the CUDA
# driver serves from pages the process already holds: allocations of up to
# 1 MiB are packed into 2 MiB pages, and a freed one leaves a hole that the
# next small allocation reuses without any change in device usage. That made
# "tbl_gpu() allocates GPU memory" (test-gpu-memory.R) measure 0 B whenever
# earlier tests left enough holes behind.
#
# Contract of the internal `gpu_alloc_tracking(expr)`: evaluate `expr` with an
# RMM statistics adaptor as the current device resource, restore the previous
# resource on exit (also on error), and return
#   list(value, allocated_bytes, held_bytes, peak_bytes)
# where `allocated_bytes` is every byte allocated through RMM while `expr` ran,
# `held_bytes` the part of it not yet freed when `expr` returned, and
# `peak_bytes` the maximum of the held amount during `expr`. Windows nest; an
# outer window also counts the inner window's allocations.
# `gpu_alloc_tracking_active()` is the number of open windows.

# Leave the driver's 2 MiB pages half used: 781 KiB tables, two per page,
# every other one freed. The survivors are returned and must be kept alive.
f56_fragment_pages <- function(n = 60L) {
  tabs <- lapply(seq_len(n), function(i) tbl_gpu(data.frame(x = runif(1e5))))
  tabs[seq(1L, n, by = 2L)] <- list(NULL)
  gc_gpu()
  tabs
}

f56_used <- function() {
  info <- gpu_details()
  info$total_memory - info$free_memory
}

test_that("tracking sees the upload when driver pages hide it from device usage", {
  skip_if_no_gpu()
  survivors <- f56_fragment_pages()

  df <- create_large_test_data(nrow = 100000, ncol = 5)
  before <- f56_used()
  tracked <- gpu_alloc_tracking(tbl_gpu(df))
  device_diff <- f56_used() - before

  expected <- 100000 * 5 * 8
  expect_true(tracked$held_bytes >= expected,
              info = sprintf("held %s, device-level diff %s",
                             format_bytes(tracked$held_bytes),
                             format_bytes(device_diff)))
  expect_true(tracked$held_bytes < 2 * expected)
  expect_true(tracked$allocated_bytes >= tracked$held_bytes)
  expect_true(tracked$peak_bytes >= tracked$held_bytes)
  expect_equal(as.data.frame(collect(tracked$value)), df)
  rm(survivors)
})

test_that("large columns raise device usage even with half-used pages", {
  skip_if_no_gpu()
  survivors <- f56_fragment_pages()

  df <- create_large_test_data(nrow = 1000000, ncol = 5)
  gc_gpu()
  before <- f56_used()
  g <- tbl_gpu(df)
  diff <- f56_used() - before

  expect_true(diff >= 0.5 * 1000000 * 5 * 8,
              info = sprintf("device-level diff %s", format_bytes(diff)))
  rm(survivors, g)
})

test_that("gpu_alloc_tracking() returns the value and zero counts for host-only work", {
  skip_if_no_gpu()
  tracked <- gpu_alloc_tracking(1 + 1)
  expect_identical(tracked$value, 2)
  expect_identical(tracked$allocated_bytes, 0)
  expect_identical(tracked$held_bytes, 0)
  expect_identical(tracked$peak_bytes, 0)
  expect_type(tracked$allocated_bytes, "double")

  tracked_null <- gpu_alloc_tracking(NULL)
  expect_null(tracked_null$value)
  expect_named(tracked_null, c("value", "allocated_bytes", "held_bytes", "peak_bytes"),
               ignore.order = TRUE)
})

test_that("memory freed inside the window counts as allocated but not held", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 100000, ncol = 5)
  gc_gpu()
  tracked <- gpu_alloc_tracking({
    g <- tbl_gpu(df)
    rm(g)
    gc(full = TRUE)
    gc(full = TRUE)
    NULL
  })
  expect_true(tracked$allocated_bytes >= 100000 * 5 * 8)
  expect_true(tracked$peak_bytes >= 100000 * 5 * 8)
  expect_identical(tracked$held_bytes, 0)
})

test_that("frees of memory allocated before the window are not counted", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 100000, ncol = 5)
  gc_gpu()
  first <- gpu_alloc_tracking(tbl_gpu(df))
  expect_true(first$held_bytes >= 100000 * 5 * 8)

  g <- first$value
  first$value <- NULL
  second <- gpu_alloc_tracking({
    rm(g)
    gc(full = TRUE)
    gc(full = TRUE)
    NULL
  })
  expect_identical(second$allocated_bytes, 0)
  expect_identical(second$held_bytes, 0)

  # The buffers allocated in `first`'s window were freed after it closed;
  # the process must still allocate and free normally.
  third <- gpu_alloc_tracking(tbl_gpu(df))
  expect_true(third$held_bytes >= 100000 * 5 * 8)
})

test_that("tracking covers cudf-internal allocations of verbs, eager and lazy", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 100000, ncol = 2)
  g <- tbl_gpu(df)

  m <- gpu_alloc_tracking(mutate(g, z = col1 + col2))
  expect_true(m$held_bytes >= 100000 * 8)

  s <- gpu_alloc_tracking(slice(g, 1:50000))
  expect_true(s$held_bytes >= 50000 * 2 * 8)

  lz <- tbl_gpu(df, lazy = TRUE) |> mutate(z = col1 * 2)
  cm <- gpu_alloc_tracking(compute(lz))
  expect_true(cm$held_bytes >= 100000 * 8)
  expect_equal(
    tibble::as_tibble(collect(cm$value)),
    tibble::as_tibble(dplyr::mutate(df, z = col1 * 2))
  )
})

test_that("windows nest and the outer window counts the inner one", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 100000, ncol = 5)
  expect_identical(gpu_alloc_tracking_active(), 0L)

  depth_inside <- NULL
  outer <- gpu_alloc_tracking({
    a <- tbl_gpu(df)
    inner <- gpu_alloc_tracking({
      depth_inside <- gpu_alloc_tracking_active()
      tbl_gpu(df)
    })
    list(a = a, inner = inner)
  })
  inner <- outer$value$inner

  expect_identical(depth_inside, 2L)
  expect_identical(gpu_alloc_tracking_active(), 0L)
  expect_true(inner$held_bytes >= 100000 * 5 * 8)
  expect_true(outer$held_bytes >= inner$held_bytes + 100000 * 5 * 8)
  expect_true(outer$allocated_bytes >= inner$allocated_bytes)
})

test_that("an error inside the window propagates and closes the window", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 100000, ncol = 5)

  expect_error(
    gpu_alloc_tracking(rlang::abort("boom", class = "f56_test_error")),
    class = "f56_test_error"
  )
  expect_identical(gpu_alloc_tracking_active(), 0L)

  # An error after an allocation: the window still closes, and the table
  # allocated in it is freed later without trouble.
  expect_error(
    gpu_alloc_tracking({
      g <- tbl_gpu(df)
      stop("after allocation")
    }),
    "after allocation"
  )
  expect_identical(gpu_alloc_tracking_active(), 0L)
  gc_gpu()

  # Nested: the inner error leaves the outer window open and counting.
  outer <- gpu_alloc_tracking({
    try(gpu_alloc_tracking(stop("inner")), silent = TRUE)
    d <- gpu_alloc_tracking_active()
    list(d = d, t = tbl_gpu(df))
  })
  expect_identical(outer$value$d, 1L)
  expect_true(outer$held_bytes >= 100000 * 5 * 8)
  expect_identical(gpu_alloc_tracking_active(), 0L)
})

test_that("many windows whose memory outlives them stay safe", {
  skip_if_no_gpu()
  df <- data.frame(x = runif(1e4), y = runif(1e4))
  kept <- lapply(1:50, function(i) gpu_alloc_tracking(tbl_gpu(df))$value)
  expect_identical(gpu_alloc_tracking_active(), 0L)
  rm(kept)
  gc_gpu()
  # All of those buffers were freed after their windows closed; allocating
  # and freeing must keep working, inside and outside new windows.
  again <- gpu_alloc_tracking(tbl_gpu(df))
  expect_true(again$held_bytes >= 1e4 * 2 * 8)
  expect_equal(as.data.frame(collect(again$value)), df)
})
