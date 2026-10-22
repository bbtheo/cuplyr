# F57: gpu_sync(), a device-wide wait for benchmarking.
#
# Contract (scratchpad/parity/features/F57.md, Design F): an exported,
# documented gpu_sync() that returns invisible(NULL) once all GPU work queued
# by cuplyr/libcudf has finished, with or without the memory pool.

test_that("gpu_sync() is exported and documented", {
  skip_if_no_gpu()
  expect_true("gpu_sync" %in% getNamespaceExports("cuplyr"))
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  expect_true(file.exists(file.path(root, "man", "gpu_sync.Rd")))
})

test_that("gpu_sync() returns invisible NULL when idle and after GPU work", {
  skip_if_no_gpu()
  expect_invisible(gpu_sync())
  expect_null(gpu_sync())

  df <- data.frame(g = sample.int(100L, 1e6, replace = TRUE), x = runif(1e6))
  g <- tbl_gpu(df, lazy = FALSE)
  r <- g |> group_by(g) |> mutate(c = cumsum(x)) |> arrange(x)
  expect_null(gpu_sync())
  res <- collect(r)
  expect_identical(nrow(res), 1000000L)
  expect_equal(res$x, sort(df$x))

  lz <- tbl_gpu(df, lazy = TRUE) |> mutate(y = x * 2)
  cm <- compute(lz)
  expect_invisible(gpu_sync())
  expect_equal(collect(cm)$y, df$x * 2)
})

test_that("gpu_sync() works with the memory pool off", {
  skip_if_no_gpu()
  skip_if_not_installed("callr")
  skip_if_not_installed("pkgload")
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  out <- callr::r(
    function(root) {
      suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                         export_all = TRUE, helpers = FALSE,
                                         attach_testthat = FALSE))
      g <- tbl_gpu(data.frame(x = runif(1e5)), lazy = FALSE)
      m <- dplyr::mutate(g, r = cumsum(x))
      v <- withVisible(gpu_sync())
      list(resource = gpu_memory_state()$resource, value = v$value,
           visible = v$visible, n = nrow(dplyr::collect(m)))
    },
    args = list(root = root),
    env = c(callr::rcmd_safe_env(), CUPLYR_MEMORY_POOL = "0"),
    timeout = 600
  )
  expect_identical(out$resource, "cuda")
  expect_null(out$value)
  expect_false(out$visible)
  expect_identical(out$n, 100000L)
})
