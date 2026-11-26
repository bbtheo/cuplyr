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

# A snapshot list of the shape the memory tests compare (`used_memory`,
# `available`), with `used_memory` measured by gpu_alloc_used_bytes(). NA and
# `available = FALSE` without a GPU.
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
# process's figure is not moved by the others. It has MiB resolution: coarse, but
# fine for the bounds that use it (the smallest, 4 MiB in the F57 probe, checks a
# delta that should be 0). When nvidia-smi is missing, fails, or doesn't list
# this PID, the device-wide reading is returned instead, with level "device".
# Two readings may differ in level (one nvidia-smi call can fail), so take their
# difference with gpu_own_delta(). Self-contained on purpose: it also runs in
# child processes, where only base/utils and gpu_memory_state() exist.
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

# `b - a` for two readings of gpu_own_device_used_bytes(), as a plain number.
# A process-level and a device-level reading can't be subtracted (the device
# figure includes other processes and the driver's own pages), so readings at
# different levels are an error, not a wrong delta. Self-contained, like the
# reader: it is shipped to child processes with it.
gpu_own_delta <- function(a, b) {
  level <- function(x) if (is.null(attr(x, "level"))) "unlabelled" else attr(x, "level")
  if (!identical(level(a), level(b))) {
    stop("own-memory readings at different levels: ", level(a), " vs ", level(b),
         call. = FALSE)
  }
  as.numeric(b) - as.numeric(a)
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

# What the last gpu_child_r() saw before it started its child: `free_before`, the
# device's free bytes after this process handed its cache back (NA without a
# GPU). A record in an environment, so the child's value is returned as is; a
# test reads it for its failure messages.
gpu_child_state <- new.env(parent = emptyenv())

# Run `func(...)` (the elements of `args`) in a fresh R process and return its
# value. Every callr child in tests/testthat starts here (children that bench/
# code or quarto start call gpu_free_for_child() themselves): this process first
# frees its cached GPU memory (once, gpu_free_for_child(); the bytes free
# afterwards are kept in gpu_child_state$free_before), and the child's global
# environment has gpu_own_device_used_bytes() and gpu_own_delta() before `func`
# runs. `func` is re-homed to the child's global environment, so it can't rely
# on its defining environment (functions inside `args` travel with a serialized
# copy of theirs); `env` adds environment variables for the child.
gpu_child_r <- function(func, args = list(), env = character(), timeout = 600) {
  skip_if_not_installed("callr")
  gpu_child_state$free_before <- gpu_free_for_child()
  environment(func) <- globalenv()
  readers <- list(gpu_own_device_used_bytes = gpu_own_device_used_bytes,
                  gpu_own_delta = gpu_own_delta)
  readers <- lapply(readers, function(f) {
    environment(f) <- globalenv()
    f
  })
  callr::r(
    function(func, args, readers) {
      list2env(readers, envir = globalenv())
      do.call(func, args)
    },
    args = list(func = func, args = args, readers = readers),
    env = c(callr::rcmd_safe_env(), env),
    timeout = timeout
  )
}

# The root of the package source tree these tests run in, or NULL when they run
# elsewhere (the check runs them from <pkg>.Rcheck/tests/testthat, with the
# package installed and no R/, man/, src/ or vignettes/ around).
cuplyr_source_root <- function() {
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
  if (file.exists(file.path(root, "DESCRIPTION")) && dir.exists(file.path(root, "R")) &&
      dir.exists(file.path(root, "man")) && dir.exists(file.path(root, "src"))) {
    root
  } else {
    NULL
  }
}

# Skip a test that reads the source tree (man/, vignettes/, NEWS.md) or reloads
# the package with pkgload, which only exist in a checkout.
skip_if_no_source_tree <- function() {
  skip_if(is.null(cuplyr_source_root()), "package source tree not available in this test context")
}

# The loader a callr child runs first: load_all() of the source tree when
# `root` is one (no compile: the library is already built), else the installed
# package (callr children get this process's library paths) with its internals
# attached, so the child's code calls them unqualified like under load_all().
# Re-homed to the child's global environment like every function that travels
# to a child.
cuplyr_child_loader <- function() {
  load <- function(root) {
    if (file.exists(file.path(root, "DESCRIPTION")) && dir.exists(file.path(root, "src"))) {
      suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                         export_all = TRUE, helpers = FALSE,
                                         attach_testthat = FALSE))
    } else {
      suppressMessages(library(cuplyr))
      if (!"cuplyr-internals" %in% search()) {
        attach(as.list(asNamespace("cuplyr"), all.names = TRUE), name = "cuplyr-internals",
               warn.conflicts = FALSE)
      }
    }
    invisible(NULL)
  }
  environment(load) <- globalenv()
  load
}

# Run `fn(...)` in a fresh R process with cuplyr's memory pool switched off,
# which loads this worktree (no compile, the library is already built). For the
# tests whose claim is about the CUDA driver (device-level memory), which a pool
# in the test process would hide.
run_with_pool_off <- function(fn, ...) {
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  gpu_child_r(
    function(root, load, fn, args) {
      load(root)
      stopifnot(identical(gpu_memory_state()$resource, "cuda"))
      do.call(fn, args)
    },
    args = list(root = root, load = cuplyr_child_loader(), fn = fn, args = list(...)),
    env = c(CUPLYR_MEMORY_POOL = "0")
  )
}
