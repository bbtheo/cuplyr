# F65: the async memory resource (cudaMallocAsync) is the default, and it gives
# memory back to the device.
#
# Contract (scratchpad/parity/features/F65.md, Design):
# - With nothing set, the resource is "async". "pool" and the off switch
#   ("cuda") still work; option > env var. An unknown kind warns, naming the
#   option or env var it came from, and falls back to async. A zero-length
#   cuplyr.memory_pool option counts as unset instead of breaking the load.
# - Design revision 1 (after round 1): cuplyr.pool_initial only primes the
#   pool (both kinds). By default the async pool keeps freed memory cached like
#   the pool does (no automatic release, so it is as fast as the pool), and
#   gpu_gc() gives back all unused memory it holds. An opt-in
#   cuplyr.release_threshold / CUPLYR_RELEASE_THRESHOLD (bytes) sets CUDA's
#   release threshold: while the async pool's whole reservation is above it,
#   unused memory goes back at the next synchronisation (0 = keep nothing).
#   gpu_gc() reports what went back as `released_bytes` (0 under "pool", which
#   keeps its pages; NA under "cuda"), next to the allocator-level
#   `freed_bytes`. cuplyr.memory_resource = "cuda"/"none" means no resource.
# - gpu_memory_state(), gpu_allocatable_bytes(), gpu_alloc_tracking() and the
#   OOM retry (cuplyr_error_gpu_oom) stay meaningful under async.
# - vignettes/memory.Rmd documents the resources; ?gpu_memory_state links it;
#   NEWS says the default changed and how to get the pool back.
#
# Anything that depends on the process's resource runs in a callr child that
# loads this worktree (pkgload::load_all(compile = FALSE)) with the three
# switches set explicitly ("" = unset), so it depends neither on the test
# process's own resource nor on the caller's environment.

f65_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
}

f65_env <- function(pool = "", kind = "", initial = "", threshold = "") {
  c(CUPLYR_MEMORY_POOL = pool, CUPLYR_MEMORY_RESOURCE = kind,
    CUPLYR_POOL_INITIAL = initial, CUPLYR_RELEASE_THRESHOLD = threshold)
}

# Clear every switch in the test process (options and env vars).
f65_local_clean <- function(env = parent.frame()) {
  withr::local_options(cuplyr.memory_resource = NULL, cuplyr.memory_pool = NULL,
                       cuplyr.pool_initial = NULL, cuplyr.release_threshold = NULL,
                       .local_envir = env)
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = NA, CUPLYR_MEMORY_POOL = NA,
                      CUPLYR_POOL_INITIAL = NA, CUPLYR_RELEASE_THRESHOLD = NA,
                      .local_envir = env)
}


# Run `fn()` in a fresh R process that loads this worktree with env `env` and
# options `opts`. `fn` may only use package functions (callr ships it without
# its environment); the load's warnings are in `.f65_load_warnings` in the
# child's global environment.
#
# The child needs device memory of its own (a CUDA context, its resource), but
# this test process caches what earlier tests freed: under async (or a pool)
# that can be nearly the whole card after a test that ran out of memory, and
# then the child can't even create its context (merge gate 2026-10-23: the
# child reported total memory 0 and resource "cuda"). So hand the cache back
# first with gpu_gc(), which trims an async resource. What the child saw is
# kept for f65_info(), so a failing expectation explains itself.
f65_state <- new.env(parent = emptyenv())

f65_child <- function(fn, env = f65_env(), opts = list()) {
  skip_if_not_installed("callr")
  skip_if_not_installed("pkgload")
  parent_free <- NA_real_
  if (has_gpu()) {
    gpu_gc(aggressive = FALSE)
    parent_free <- gpu_memory_state()$free_bytes
  }
  res <- callr::r(
    function(root, fn, opts) {
      if (length(opts)) options(opts)
      # Warnings raised while loading, for fn() to inspect.
      w <- character()
      withCallingHandlers(
        suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                           export_all = TRUE, helpers = FALSE,
                                           attach_testthat = FALSE)),
        warning = function(cnd) {
          w <<- c(w, conditionMessage(cnd))
          invokeRestart("muffleWarning")
        }
      )
      assign(".f65_load_warnings", w, envir = globalenv())
      st <- gpu_memory_state()
      list(value = fn(), warnings = w, resource = st$resource,
           free = st$free_bytes, total = st$total_bytes)
    },
    args = list(root = f65_root(), fn = fn, opts = opts),
    env = c(callr::rcmd_safe_env(), env),
    timeout = 600
  )
  f65_state$last <- c(res[c("warnings", "resource", "free", "total")],
                      parent_free = parent_free)
  res$value
}

# What the last child saw at load, for `info =` of resource expectations.
f65_info <- function() {
  d <- f65_state$last
  if (is.null(d)) return("no child run yet")
  sprintf("child: resource %s, device free %.3f GB of %.3f GB; parent free before spawning %.3f GB; child load warnings: %s",
          d$resource, d$free / 1e9, d$total / 1e9, d$parent_free / 1e9,
          if (length(d$warnings)) paste(d$warnings, collapse = " | ") else "none")
}

# The children allocate from the device, not from this process's memory
# resource: memory this process's resource holds (it caches what earlier tests
# freed) counts in gpu_allocatable_bytes() here but is not available to a
# child. So give it back first (gpu_gc() trims an async resource; a pool can't
# shrink) and check the device's free memory.
f65_skip_unless_device_free <- function(bytes) {
  skip_if_no_gpu()
  gpu_gc(aggressive = FALSE)
  free <- gpu_memory_state()$free_bytes
  if (is.na(free) || free < bytes) {
    skip(sprintf("needs %.1f GB free on the device, %.1f GB free", bytes / 1e9, free / 1e9))
  }
}

f65_mib <- 2^20
f65_tol <- 64 * f65_mib

# ---------------------------------------------------------------------------
# Choosing the resource
# ---------------------------------------------------------------------------

test_that("with nothing set the async resource is installed", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    st <- gpu_memory_state()
    list(resource = st$resource, pool_bytes = st$pool_bytes,
         pool_used_bytes = st$pool_used_bytes, total = st$total_bytes,
         kind = memory_resource_config()$kind)
  })
  expect_identical(out$resource, "async", info = f65_info())
  expect_identical(out$kind, "async")
  expect_true(is.numeric(out$pool_bytes) && !is.na(out$pool_bytes))
  expect_true(is.numeric(out$pool_used_bytes) && !is.na(out$pool_used_bytes))
  expect_lte(out$pool_bytes, out$total)
})

test_that("memory_resource_config() defaults to async and keeps option > env", {
  f65_local_clean()
  expect_identical(memory_resource_config()$kind, "async")
  expect_true(memory_resource_config()$enabled)

  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = ""), {
    expect_no_warning(cfg <- memory_resource_config())
    expect_identical(cfg$kind, "async")
  })
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "pool"),
                     expect_identical(memory_resource_config()$kind, "pool"))
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = " Pool "),
                     expect_identical(memory_resource_config()$kind, "pool"))
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "ASYNC"),
                     expect_identical(memory_resource_config()$kind, "async"))
  withr::with_options(list(cuplyr.memory_resource = "pool"),
                      expect_identical(memory_resource_config()$kind, "pool"))
  # The option wins over the env var, both ways.
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "async"),
    withr::with_options(list(cuplyr.memory_resource = "pool"),
                        expect_identical(memory_resource_config()$kind, "pool")))
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "pool"),
    withr::with_options(list(cuplyr.memory_resource = "async"),
                        expect_identical(memory_resource_config()$kind, "async")))
})

test_that("an unknown resource kind warns, names its source, and means async", {
  f65_local_clean()

  withr::with_options(list(cuplyr.memory_resource = "bogus"), {
    expect_warning(cfg <- memory_resource_config(), "cuplyr.memory_resource")
    expect_identical(cfg$kind, "async")
  })
  withr::with_options(list(cuplyr.memory_resource = "bogus"),
    expect_warning(memory_resource_config(), "bogus"))

  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "managed"), {
    expect_warning(cfg <- memory_resource_config(), "CUPLYR_MEMORY_RESOURCE")
    expect_identical(cfg$kind, "async")
  })
  # The warning quotes the value as the user typed it.
  withr::with_options(list(cuplyr.memory_resource = "Managed"),
    expect_warning(memory_resource_config(), "Managed", fixed = TRUE))
})

test_that("cuplyr.memory_resource = \"cuda\" or \"none\" means no resource, without a warning", {
  f65_local_clean()
  withr::with_options(list(cuplyr.memory_resource = "cuda"), {
    expect_no_warning(cfg <- memory_resource_config())
    expect_false(cfg$enabled)
  })
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "none"), {
    expect_no_warning(cfg <- memory_resource_config())
    expect_false(cfg$enabled)
  })
  withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "CUDA"),
                     expect_false(memory_resource_config()$enabled))

  skip_if_no_gpu()
  out <- f65_child(function() gpu_memory_state()$resource, env = f65_env(kind = "cuda"))
  expect_identical(out, "cuda", info = f65_info())
})

test_that("cuplyr.release_threshold: unset by default, option > env, bad values warn", {
  f65_local_clean()
  cfg <- memory_resource_config()
  expect_true(is.na(cfg$release_threshold))
  # pool_initial is not the threshold.
  withr::with_options(list(cuplyr.pool_initial = 2^28),
                      expect_true(is.na(memory_resource_config()$release_threshold)))

  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = ""),
                     expect_true(is.na(memory_resource_config()$release_threshold)))
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "0"),
                     expect_identical(memory_resource_config()$release_threshold, 0))
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "268435456"),
                     expect_identical(memory_resource_config()$release_threshold, 268435456))
  withr::with_options(list(cuplyr.release_threshold = 1e9),
                      expect_identical(memory_resource_config()$release_threshold, 1e9))
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "0"),
    withr::with_options(list(cuplyr.release_threshold = 1e9),
                        expect_identical(memory_resource_config()$release_threshold, 1e9)))

  withr::with_options(list(cuplyr.release_threshold = "1GB"), {
    expect_warning(cfg <- memory_resource_config(), "cuplyr.release_threshold", fixed = TRUE)
    expect_true(is.na(cfg$release_threshold))
  })
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "-5"), {
    expect_warning(cfg <- memory_resource_config(), "CUPLYR_RELEASE_THRESHOLD", fixed = TRUE)
    expect_true(is.na(cfg$release_threshold))
  })
})

test_that("an unknown kind at load warns and installs async", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    list(resource = gpu_memory_state()$resource,
         warnings = get(".f65_load_warnings", envir = globalenv()))
  }, env = f65_env(kind = "bogus"))
  expect_identical(out$resource, "async", info = f65_info())
  expect_true(any(grepl("CUPLYR_MEMORY_RESOURCE", out$warnings, fixed = TRUE)),
              info = paste(out$warnings, collapse = " | "))

  out_opt <- f65_child(function() {
    list(resource = gpu_memory_state()$resource,
         warnings = get(".f65_load_warnings", envir = globalenv()))
  }, opts = list(cuplyr.memory_resource = "bogus"))
  expect_identical(out_opt$resource, "async", info = f65_info())
  expect_true(any(grepl("cuplyr.memory_resource", out_opt$warnings, fixed = TRUE)),
              info = paste(out_opt$warnings, collapse = " | "))

  # A valid or unset kind loads without warnings from the resource switches.
  quiet <- f65_child(function() get(".f65_load_warnings", envir = globalenv()))
  expect_false(any(grepl("memory_resource|MEMORY_RESOURCE", quiet)),
               info = paste(quiet, collapse = " | "))
})

test_that("a zero-length or NA cuplyr.memory_pool option counts as unset", {
  f65_local_clean()
  withr::with_options(list(cuplyr.memory_pool = logical(0)), {
    expect_no_error(cfg <- memory_resource_config())
    expect_true(cfg$enabled)
    expect_identical(cfg$kind, "async")
  })
  withr::with_options(list(cuplyr.memory_pool = character(0)),
                      expect_true(memory_resource_config()$enabled))
  withr::with_options(list(cuplyr.memory_pool = NA),
                      expect_true(memory_resource_config()$enabled))
  # Unset means the env var decides.
  withr::with_envvar(c(CUPLYR_MEMORY_POOL = "0"),
    withr::with_options(list(cuplyr.memory_pool = logical(0)),
                        expect_false(memory_resource_config()$enabled)))

  skip_if_no_gpu()
  out <- f65_child(function() gpu_memory_state()$resource,
                   opts = list(cuplyr.memory_pool = logical(0)))
  expect_identical(out, "async", info = f65_info())
})

test_that("the pool and the off switch are still available", {
  skip_if_no_gpu()
  pool <- f65_child(function() gpu_memory_state()$resource, env = f65_env(kind = "pool"))
  expect_identical(pool, "pool", info = f65_info())
  pool_opt <- f65_child(function() gpu_memory_state()$resource,
                        opts = list(cuplyr.memory_resource = "pool"))
  expect_identical(pool_opt, "pool", info = f65_info())
  off <- f65_child(function() {
    st <- gpu_memory_state()
    list(resource = st$resource, pool_bytes = st$pool_bytes,
         released = gpu_gc(aggressive = FALSE)$released_bytes)
  }, env = f65_env(pool = "0"))
  expect_identical(off$resource, "cuda", info = f65_info())
  expect_true(is.na(off$pool_bytes))
  expect_true(is.na(off$released))
})

# ---------------------------------------------------------------------------
# Async gives memory back
# ---------------------------------------------------------------------------

test_that("under async, a dropped 1 GB table goes back to the device after gpu_gc()", {
  f65_skip_unless_device_free(3e9)
  out <- f65_child(function() {
    host <- data.frame(a = runif(1.25e8))           # 1e9 bytes on the GPU
    gpu_gc()
    s0 <- gpu_memory_state()
    g <- tbl_gpu(host, lazy = FALSE)
    gpu_sync()
    s1 <- gpu_memory_state()
    rm(g)
    msgs <- character()
    res <- withCallingHandlers(gpu_gc(verbose = TRUE), message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
    s2 <- gpu_memory_state()
    list(resource = s0$resource,
         free0 = s0$free_bytes, free1 = s1$free_bytes, free2 = s2$free_bytes,
         pool0 = s0$pool_bytes, pool1 = s1$pool_bytes, pool2 = s2$pool_bytes,
         used1 = s1$pool_used_bytes, used2 = s2$pool_used_bytes,
         freed = res$freed_bytes, released = res$released_bytes, msgs = msgs)
  })
  expect_identical(out$resource, "async", info = f65_info())
  # The table really came from the device ...
  expect_gte(out$free0 - out$free1, 0.9e9)
  expect_gte(out$used1 - out$used2, 1e9)
  # ... and gpu_gc() gave it back: device free memory and the async pool's
  # reservation are where they were before the table existed.
  expect_gte(out$free2, out$free0 - 64 * 2^20)
  expect_lte(out$pool2, out$pool0 + 64 * 2^20)
  expect_gte(out$freed, 1e9)
  expect_gte(out$released, 0.9e9)
  expect_lte(out$released, out$pool1)
  # The verbose message says memory went back to the device, not that a pool
  # kept it.
  expect_true(any(grepl("device", out$msgs, ignore.case = TRUE)),
              info = paste(out$msgs, collapse = " | "))
  expect_false(any(grepl("keeps the pages", out$msgs, fixed = TRUE)),
               info = paste(out$msgs, collapse = " | "))
})

test_that("by default async keeps freed memory cached across syncs, like the pool, until gpu_gc()", {
  f65_skip_unless_device_free(3e9)
  out <- f65_child(function() {
    big <- tbl_gpu(data.frame(a = runif(2e8)), lazy = FALSE)   # 1.6 GB, kept
    tmp <- tbl_gpu(data.frame(a = runif(5e7)), lazy = FALSE)   # 400 MB, dropped
    gpu_sync()
    rm(tmp)
    gc(); gc()
    gpu_sync()
    s1 <- gpu_memory_state()
    res <- gpu_gc()
    s2 <- gpu_memory_state()
    list(resource = s1$resource,
         cached1 = s1$pool_bytes - s1$pool_used_bytes,
         cached2 = s2$pool_bytes - s2$pool_used_bytes,
         used2 = s2$pool_used_bytes, released = res$released_bytes,
         n = nrow(big))
  })
  expect_identical(out$resource, "async", info = f65_info())
  # The dropped 400 MB stays cached for reuse through the synchronisation ...
  expect_gte(out$cached1, 0.35e9)
  # ... and gpu_gc() gives the cache back while the 1.6 GB table stays.
  expect_gte(out$released, 0.35e9)
  expect_lte(out$cached2, 64 * 2^20)
  expect_gte(out$used2, 1.6e9)
  expect_identical(out$n, 200000000L)
})

test_that("with cuplyr.release_threshold set, async gives memory back at the next sync without gpu_gc()", {
  f65_skip_unless_device_free(3e9)
  out <- f65_child(function() {
    host <- data.frame(a = runif(1.25e8))
    gc()
    gpu_sync()
    s0 <- gpu_memory_state()
    g <- tbl_gpu(host, lazy = FALSE)
    gpu_sync()
    s1 <- gpu_memory_state()
    rm(g)
    gc(); gc()
    gpu_sync()
    s2 <- gpu_memory_state()
    list(resource = s0$resource, free0 = s0$free_bytes, free1 = s1$free_bytes,
         free2 = s2$free_bytes, pool1 = s1$pool_bytes, pool2 = s2$pool_bytes)
  }, env = f65_env(threshold = format(256 * 2^20, scientific = FALSE)))
  expect_identical(out$resource, "async", info = f65_info())
  expect_gte(out$free0 - out$free1, 0.6e9)
  expect_gte(out$pool1, 1e9)
  # After the sync the reservation is back at or under the threshold.
  expect_lte(out$pool2, 256 * 2^20 + 64 * 2^20)
  expect_gte(out$free2, out$free0 - 64 * 2^20)

  # The threshold caps the whole reservation, in-use memory included: with a
  # 1.6 GB table held, a dropped 400 MB table is not kept.
  held <- f65_child(function() {
    big <- tbl_gpu(data.frame(a = runif(2e8)), lazy = FALSE)
    tmp <- tbl_gpu(data.frame(a = runif(5e7)), lazy = FALSE)
    gpu_sync()
    rm(tmp)
    gc(); gc()
    gpu_sync()
    s <- gpu_memory_state()
    list(cached = s$pool_bytes - s$pool_used_bytes, n = nrow(big))
  }, env = f65_env(threshold = format(256 * 2^20, scientific = FALSE)))
  expect_lte(held$cached, 64 * 2^20)
})

test_that("cuplyr.pool_initial primes the async pool and is not a release threshold", {
  f65_skip_unless_device_free(3e9)
  out <- f65_child(function() {
    s0 <- gpu_memory_state()
    host <- data.frame(a = runif(1.25e8))
    g <- tbl_gpu(host, lazy = FALSE)
    gpu_sync()
    rm(g)
    gc(); gc()
    gpu_sync()
    s1 <- gpu_memory_state()
    list(resource = s0$resource, pool0 = s0$pool_bytes, pool1 = s1$pool_bytes)
  }, env = f65_env(initial = format(512 * 2^20, scientific = FALSE)))
  expect_identical(out$resource, "async", info = f65_info())
  expect_gte(out$pool0, 512 * 2^20)
  # The dropped 1 GB stays cached: 512 MiB is not a threshold.
  expect_gte(out$pool1, 1e9)
})

test_that("under async, cuplyr.pool_initial = 0 reserves nothing; with release_threshold = 0 nothing is kept", {
  skip_if_no_gpu()
  run <- function() {
    s0 <- gpu_memory_state()
    g <- tbl_gpu(data.frame(x = runif(1e6), k = rep(1:10, 1e5)), lazy = FALSE)
    r <- dplyr::collect(dplyr::summarise(dplyr::group_by(g, k), s = sum(x)))
    rm(g)
    gc(); gc()
    gpu_sync()
    s1 <- gpu_memory_state()
    list(resource = s0$resource, pool0 = s0$pool_bytes, pool1 = s1$pool_bytes,
         n = nrow(r))
  }
  lazy_res <- f65_child(run, opts = list(cuplyr.pool_initial = 0))
  expect_identical(lazy_res$resource, "async", info = f65_info())
  expect_identical(lazy_res$n, 10L)
  expect_lte(lazy_res$pool0, 64 * 2^20)

  lean <- f65_child(run, opts = list(cuplyr.pool_initial = 0, cuplyr.release_threshold = 0))
  expect_identical(lean$resource, "async", info = f65_info())
  expect_identical(lean$n, 10L)
  expect_lte(lean$pool0, 64 * 2^20)
  expect_lte(lean$pool1, 64 * 2^20)
})

test_that("under pool, gpu_gc() frees at the allocator but the pool keeps its pages", {
  f65_skip_unless_device_free(3e9)
  out <- f65_child(function() {
    host <- data.frame(a = runif(1.25e8))
    g <- tbl_gpu(host, lazy = FALSE)
    gpu_sync()
    s1 <- gpu_memory_state()
    rm(g)
    msgs <- character()
    res <- withCallingHandlers(gpu_gc(verbose = TRUE), message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
    s2 <- gpu_memory_state()
    list(resource = s1$resource, free1 = s1$free_bytes, free2 = s2$free_bytes,
         pool1 = s1$pool_bytes, pool2 = s2$pool_bytes,
         freed = res$freed_bytes, released = res$released_bytes, msgs = msgs)
  }, env = f65_env(kind = "pool", initial = format(256 * 2^20, scientific = FALSE)))
  expect_identical(out$resource, "pool", info = f65_info())
  expect_gte(out$pool1, 1e9)
  expect_identical(out$pool2, out$pool1)
  expect_gte(out$freed, 1e9)
  expect_identical(out$released, 0)
  expect_lte(abs(out$free2 - out$free1), 64 * 2^20)
  expect_true(any(grepl("the pool keeps the pages", out$msgs, fixed = TRUE)),
              info = paste(out$msgs, collapse = " | "))
})

# ---------------------------------------------------------------------------
# State readings, tracking and OOM under async
# ---------------------------------------------------------------------------

test_that("state readings, allocation tracking and gpu_gc() are meaningful under async", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    gpu_gc()
    s0 <- gpu_memory_state()
    a0 <- gpu_allocatable_bytes(s0)
    g <- tbl_gpu(data.frame(x = runif(1e6)), lazy = FALSE)   # 8e6 bytes
    s1 <- gpu_memory_state()
    tr <- gpu_alloc_tracking(tbl_gpu(data.frame(y = runif(1e6)), lazy = FALSE))
    tracked <- c(tr$allocated_bytes, tr$held_bytes, tr$peak_bytes)
    rm(g, tr)
    res <- gpu_gc()
    list(resource = s0$resource, s0 = s0, s1 = s1, a0 = a0,
         gc = res[c("freed_bytes", "released_bytes")], tracked = tracked)
  })
  expect_identical(out$resource, "async", info = f65_info())
  s0 <- out$s0
  s1 <- out$s1
  for (f in c("free_bytes", "used_bytes", "total_bytes", "pool_bytes", "pool_used_bytes")) {
    expect_true(is.numeric(s0[[f]]) && !is.na(s0[[f]]), label = f)
  }
  expect_gte(s0$pool_bytes, s0$pool_used_bytes)
  expect_gte(s1$pool_bytes, s1$pool_used_bytes)
  # pool_used_bytes is exact at the allocator: one 1e6-row FLOAT64 column.
  expect_gte(s1$pool_used_bytes - s0$pool_used_bytes, 8e6)
  expect_lte(s1$pool_used_bytes - s0$pool_used_bytes, 8e6 + 1e6)
  # Allocatable = device free + the async pool's unused reservation.
  expect_equal(out$a0, s0$free_bytes + max(s0$pool_bytes - s0$pool_used_bytes, 0))
  expect_gte(out$a0, s0$free_bytes)
  # F56 tracking counts what the window allocated and still holds.
  expect_true(is.numeric(out$tracked), info = paste(out$tracked))
  expect_gte(out$tracked[[1]], 8e6)
  expect_gte(out$tracked[[2]], 8e6)
  expect_gte(out$tracked[[3]], 8e6)
  # gpu_gc() reports both levels.
  expect_gte(out$gc$freed_bytes, 16e6)
  expect_true(is.numeric(out$gc$released_bytes) && !is.na(out$gc$released_bytes))
  expect_gte(out$gc$released_bytes, 0)
})

test_that("a real device OOM under async raises cuplyr_error_gpu_oom and the session recovers", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    a <- tbl_gpu(data.frame(x = seq_len(4e4)), lazy = FALSE)
    b <- tbl_gpu(data.frame(y = seq_len(4e4)), lazy = FALSE)
    e <- tryCatch(suppressWarnings(dplyr::cross_join(a, b)), error = function(e) e)
    gpu_gc()
    g <- tbl_gpu(data.frame(x = c(1, NA, 3)), lazy = FALSE)
    list(resource = gpu_memory_state()$resource, cls = class(e),
         msg = conditionMessage(e),
         after = dplyr::collect(dplyr::mutate(g, y = x * 2))$y)
  })
  expect_identical(out$resource, "async", info = f65_info())
  expect_true("cuplyr_error_gpu_oom" %in% out$cls, info = out$msg)
  expect_equal(out$after, c(2, NA, 6))
})

test_that("an injected OOM under async is retried once and succeeds", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    g <- tbl_gpu(data.frame(x = 1:20, k = rep(1:2, 10)), lazy = FALSE)
    gpu_oom_inject(1L)
    r <- dplyr::collect(dplyr::mutate(g, y = x * 2L))
    gpu_oom_inject(0L)
    list(resource = gpu_memory_state()$resource, y = r$y)
  })
  expect_identical(out$resource, "async", info = f65_info())
  expect_identical(out$y, (1:20) * 2L)
})

# ---------------------------------------------------------------------------
# Round 3 additions (round-2 review N3, N5, N7)
# ---------------------------------------------------------------------------

test_that("the async pool's release threshold is the device total by default and the configured value otherwise", {
  skip_if_no_gpu()
  thr <- function() list(resource = gpu_memory_state()$resource,
                         total = gpu_memory_state()$total_bytes,
                         thr = gpu_mr_release_threshold())
  def <- f65_child(thr)
  expect_identical(def$resource, "async", info = f65_info())
  # No automatic release: the threshold is the whole card (RMM's default).
  expect_gte(def$thr, 0.9 * def$total)

  set <- f65_child(thr, env = f65_env(threshold = format(256 * 2^20, scientific = FALSE)))
  expect_identical(set$thr, 256 * 2^20)
  zero <- f65_child(thr, opts = list(cuplyr.release_threshold = 0))
  expect_identical(zero$thr, 0)

  # Not an async pool: no threshold.
  pool <- f65_child(thr, env = f65_env(kind = "pool", threshold = "0"))
  expect_identical(pool$resource, "pool", info = f65_info())
  expect_true(is.na(pool$thr))
  off <- f65_child(thr, env = f65_env(pool = "0"))
  expect_identical(off$resource, "cuda", info = f65_info())
  expect_true(is.na(off$thr))
})

test_that("under pool, cuplyr.release_threshold is ignored: the pool keeps its pages", {
  f65_skip_unless_device_free(3e9)
  out <- f65_child(function() {
    g <- tbl_gpu(data.frame(a = runif(1.25e8)), lazy = FALSE)
    gpu_sync()
    rm(g)
    gc(); gc()
    gpu_sync()
    s <- gpu_memory_state()
    list(resource = s$resource, pool = s$pool_bytes)
  }, env = f65_env(kind = "pool", threshold = "0",
                   initial = format(256 * 2^20, scientific = FALSE)))
  expect_identical(out$resource, "pool", info = f65_info())
  expect_gte(out$pool, 1e9)
})

test_that("gpu_allocatable_bytes() counts the async pool's cached memory", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    gpu_gc()
    g <- tbl_gpu(data.frame(x = runif(1e7)), lazy = FALSE)   # 80 MB
    gpu_sync()
    rm(g)
    gc(); gc()
    gpu_sync()
    s <- gpu_memory_state()
    list(resource = s$resource, free = s$free_bytes,
         cached = s$pool_bytes - s$pool_used_bytes,
         alloc = gpu_allocatable_bytes(s))
  })
  expect_identical(out$resource, "async", info = f65_info())
  expect_gte(out$cached, 70e6)
  expect_gte(out$alloc, out$free + 70e6)
})

test_that("with the off switch set, a bad cuplyr.memory_resource does not claim async is used", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    list(resource = gpu_memory_state()$resource,
         warnings = get(".f65_load_warnings", envir = globalenv()))
  }, env = f65_env(pool = "0", kind = "bogus"))
  expect_identical(out$resource, "cuda", info = f65_info())
  w <- grep("MEMORY_RESOURCE|memory_resource", out$warnings, value = TRUE)
  # The bad value is reported, once, without claiming async.
  expect_length(w, 1)
  expect_false(any(grepl("using \"async\"", w, fixed = TRUE)),
               info = paste(w, collapse = " | "))
  expect_true(all(grepl("cuda|ignored", w)), info = paste(w, collapse = " | "))
})

test_that("a reload that reuses the existing resource does not validate the switches", {
  skip_if_no_gpu()
  out <- f65_child(function() {
    options(cuplyr.memory_resource = "bogus", cuplyr.release_threshold = "1GB")
    w <- character()
    withCallingHandlers(
      suppressMessages(pkgload::load_all(Sys.getenv("F65_ROOT"), quiet = TRUE,
                                         compile = FALSE, export_all = TRUE,
                                         helpers = FALSE, attach_testthat = FALSE)),
      warning = function(cnd) {
        w <<- c(w, conditionMessage(cnd))
        invokeRestart("muffleWarning")
      }
    )
    list(resource = gpu_memory_state()$resource, warnings = w)
  }, env = c(f65_env(kind = "pool"), F65_ROOT = f65_root()))
  expect_identical(out$resource, "pool", info = f65_info())
  expect_false(any(grepl("memory_resource|release_threshold", out$warnings)),
               info = paste(out$warnings, collapse = " | "))
})

test_that("an NA or empty cuplyr.release_threshold option counts as unset", {
  f65_local_clean()
  withr::with_options(list(cuplyr.release_threshold = NA), {
    expect_no_warning(cfg <- memory_resource_config())
    expect_true(is.na(cfg$release_threshold))
  })
  withr::with_options(list(cuplyr.release_threshold = ""), {
    expect_no_warning(cfg <- memory_resource_config())
    expect_true(is.na(cfg$release_threshold))
  })
  # Unset means the env var decides.
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "0"),
    withr::with_options(list(cuplyr.release_threshold = NA),
                        expect_identical(memory_resource_config()$release_threshold, 0)))
})

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

test_that("the memory vignette exists and covers the resources", {
  path <- file.path(f65_root(), "vignettes", "memory.Rmd")
  expect_true(file.exists(path))
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  expect_match(txt, "VignetteIndexEntry", fixed = TRUE)
  for (s in c("async", "pool", "cuda", "cuplyr.memory_resource",
              "CUPLYR_MEMORY_RESOURCE", "cuplyr.memory_pool", "cuplyr.pool_initial",
              "cuplyr.release_threshold", "CUPLYR_RELEASE_THRESHOLD",
              "gpu_memory_state()", "pool_used_bytes", "free_bytes",
              "gpu_memory_usage()", "gpu_gc()", "released_bytes",
              "cuplyr_error_gpu_oom", "RTX 5070")) {
    expect_match(txt, s, fixed = TRUE, label = s)
  }
  # GPU chunks must not run on a machine without a GPU.
  expect_match(txt, "has_gpu()", fixed = TRUE)
})

test_that("?gpu_memory_state links the vignette and NEWS documents the change", {
  rd <- file.path(f65_root(), "man", "gpu_memory_state.Rd")
  rd_txt <- paste(readLines(rd, warn = FALSE), collapse = "\n")
  expect_match(rd_txt, "vignette(\"memory\"", fixed = TRUE)
  expect_match(rd_txt, "async", fixed = TRUE)

  news <- paste(readLines(file.path(f65_root(), "NEWS.md"), warn = FALSE), collapse = "\n")
  first <- sub("(?s)\n# cuplyr [0-9].*$", "", news, perl = TRUE)  # development section only
  expect_match(first, "cuplyr.memory_resource", fixed = TRUE)
  expect_match(first, "cuplyr.release_threshold", fixed = TRUE)
  expect_match(first, "async", fixed = TRUE)
  expect_match(first, "\"pool\"", fixed = TRUE)
})
