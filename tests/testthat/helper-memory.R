# Memory measurement helpers (F57).
#
# cuplyr installs an RMM pool at load time, which hides device-level free
# memory (cudaMemGetInfo): the pool keeps its pages, so freeing a table does not
# move `used_bytes`, and an allocation is served from pages already reserved.
# Tests that check how much a table occupies, or that memory comes back, measure
# the allocator instead: `pool_used_bytes`, the bytes currently allocated from
# cuplyr's resource. With no resource (`resource == "cuda"`, the pool switched
# off) the same helpers fall back to the device-level number, which is then the
# allocator's own.

# Bytes in use: allocator level under a resource, device level without one.
gpu_alloc_used_bytes <- function() {
  st <- gpu_memory_state()
  if (is.na(st$pool_used_bytes)) st$used_bytes else st$pool_used_bytes
}

# gpu_memory_snapshot() with `used_memory` measured by gpu_alloc_used_bytes().
gpu_alloc_snapshot <- function() {
  if (!has_gpu()) {
    return(list(used_memory = NA_real_, available = FALSE))
  }
  list(used_memory = gpu_alloc_used_bytes(), available = TRUE)
}

# Run `fn(...)` in a fresh R process with cuplyr's memory pool switched off,
# which loads this worktree (no compile, the library is already built). For the
# tests whose claim is about the CUDA driver (device-level memory), which a pool
# in the test process would hide.
run_with_pool_off <- function(fn, ...) {
  skip_if_not_installed("callr")
  skip_if_not_installed("pkgload")
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  callr::r(
    function(root, fn, args) {
      suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                         export_all = TRUE, helpers = FALSE,
                                         attach_testthat = FALSE))
      stopifnot(identical(gpu_memory_state()$resource, "cuda"))
      do.call(fn, args)
    },
    args = list(root = root, fn = fn, args = list(...)),
    env = c(callr::rcmd_safe_env(), CUPLYR_MEMORY_POOL = "0"),
    timeout = 600
  )
}
