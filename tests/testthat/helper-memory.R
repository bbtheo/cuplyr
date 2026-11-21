# Memory measurement helpers (F57).
#
# cuplyr installs an RMM resource at load time (async by default, since F65;
# or the pool), which hides device-level free memory (cudaMemGetInfo): the
# resource keeps freed pages for reuse (the pool until the process exits; async
# until gpu_gc() or an opt-in release threshold hands them back), so freeing a
# table need not move `used_bytes`, and an allocation is served from pages
# already reserved.
# Tests that check how much a table occupies, or that memory comes back, measure
# the allocator instead: `pool_used_bytes`, the bytes currently allocated from
# cuplyr's resource. With no resource (`resource == "cuda"`, the resource switched
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

# This process's own device memory in bytes, with attribute `level` "process".
# Claims about the device itself (driver pages, a second pool, memory handed back
# to the device) can't read the allocator's counters, and the device-wide reading
# (`used_bytes`) also moves when any other process uses the GPU (a compositor, a
# browser). nvidia-smi reports the memory of each compute process, so this
# process's figure is not moved by the others. It is in MiB: coarse, but the
# bounds that use it are tens of MiB or more. When nvidia-smi is missing, fails,
# or doesn't list this PID, the device-wide reading is returned instead, with
# level "device". Self-contained on purpose: it also runs in child processes,
# where only base/utils and gpu_memory_state() exist.
gpu_own_device_used_bytes <- function() {
  smi <- Sys.which("nvidia-smi")
  if (nzchar(smi)) {
    lines <- tryCatch(
      suppressWarnings(system2(
        smi, c("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"),
        stdout = TRUE, stderr = FALSE
      )),
      error = function(e) character()
    )
    if (is.character(lines) && length(lines)) {
      parts <- strsplit(lines, ",", fixed = TRUE)
      pid <- vapply(parts, function(p) trimws(p[[1]]), character(1))
      mib <- vapply(parts, function(p) {
        if (length(p) < 2L) NA_real_ else suppressWarnings(as.numeric(trimws(p[[2]])))
      }, numeric(1))
      mine <- pid == as.character(Sys.getpid()) & !is.na(mib)
      if (any(mine)) {
        out <- sum(mib[mine]) * 2^20
        attr(out, "level") <- "process"
        return(out)
      }
    }
  }
  out <- as.numeric(gpu_memory_state()$used_bytes)
  attr(out, "level") <- "device"
  out
}

# Hand this process's cached GPU memory back to the device before a child R
# process starts. The async resource keeps freed pages for reuse (a pool never
# gives them back), and a test that ran out of memory can leave it holding most
# of the card, leaving the child no room for its CUDA context. Returns, invisibly,
# the device's free bytes afterwards (NA without a GPU), for failure messages.
gpu_free_for_child <- function() {
  if (!has_gpu()) return(invisible(NA_real_))
  gpu_gc(aggressive = FALSE)
  invisible(gpu_memory_state()$free_bytes)
}

# Run `func(...)` (the elements of `args`) in a fresh R process and return its
# value. Every child R process in the tests starts here: this process first
# frees its cached GPU memory (gpu_free_for_child()), and the child's global
# environment has gpu_own_device_used_bytes() before `func` runs. `func` must
# not rely on its defining environment (it is shipped with globalenv() as
# parent); `env` adds environment variables for the child.
gpu_child_r <- function(func, args = list(), env = character(), timeout = 600) {
  skip_if_not_installed("callr")
  gpu_free_for_child()
  environment(func) <- globalenv()
  reader <- gpu_own_device_used_bytes
  environment(reader) <- globalenv()
  callr::r(
    function(func, args, reader) {
      assign("gpu_own_device_used_bytes", reader, envir = globalenv())
      do.call(func, args)
    },
    args = list(func = func, args = args, reader = reader),
    env = c(callr::rcmd_safe_env(), env),
    timeout = timeout
  )
}

# Run `fn(...)` in a fresh R process with cuplyr's memory pool switched off,
# which loads this worktree (no compile, the library is already built). For the
# tests whose claim is about the CUDA driver (device-level memory), which a pool
# in the test process would hide.
run_with_pool_off <- function(fn, ...) {
  skip_if_not_installed("pkgload")
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  gpu_child_r(
    function(root, fn, args) {
      suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                         export_all = TRUE, helpers = FALSE,
                                         attach_testthat = FALSE))
      stopifnot(identical(gpu_memory_state()$resource, "cuda"))
      do.call(fn, args)
    },
    args = list(root = root, fn = fn, args = list(...)),
    env = c(CUPLYR_MEMORY_POOL = "0")
  )
}
