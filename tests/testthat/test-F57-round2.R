# F57 round 2: what the reviewer's blocking findings and notes pin.
# - the join-size warning compares against what cuplyr can allocate (device free
#   memory plus the unused part of its memory resource), not device free memory
# - reloading the package with live tables neither stacks pools on the device
#   nor hides the resource from gpu_memory_state()
# - `CUPLYR_MEMORY_RESOURCE=async` works end to end
# - mutate()'s `.by` / `.before` / `.after` shortcut keeps a forwarded missing
#   argument an error
# Round 3 (one resource per process, never freed):
# - after any reload, new tables are allocated from that one resource, so
#   pool_used_bytes, gpu_gc() and gpu_alloc_tracking() see them
# - no value of the `.cuplyr_mr_handle` option (round 2's hand-over) can crash
#   a load or make it create a second resource
# - pkgload::unload() followed by load_all() leaves old tables and libcudf working

# Run `fn(load)` in a fresh R process that has loaded this worktree (no compile,
# the library is already built) with the environment variables `env` set.
# `load()` loads it again (a reload). callr ships `fn` without its environment,
# so `fn` only uses package functions (export_all = TRUE) and `load`.
f57r2_subprocess <- function(fn, env = character(), globals = list()) {
  skip_if_not_installed("callr")
  skip_if_not_installed("pkgload")
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  # Functions in `globals` are re-homed to the subprocess's global environment.
  globals <- lapply(globals, function(f) {
    environment(f) <- globalenv()
    f
  })
  callr::r(
    function(root, fn, globals) {
      list2env(globals, envir = globalenv())
      load <- function() {
        suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                           export_all = TRUE, helpers = FALSE,
                                           attach_testthat = FALSE))
      }
      load()
      fn(load)
    },
    args = list(root = root, fn = fn, globals = globals),
    env = c(callr::rcmd_safe_env(), env),
    timeout = 600
  )
}

test_that("gpu_allocatable_bytes() adds the unused part of the memory resource", {
  skip_if_no_gpu()
  st <- function(...) c(list(available = TRUE, free_bytes = 100), list(...))
  expect_equal(gpu_allocatable_bytes(st(resource = "pool", pool_bytes = 1000, pool_used_bytes = 300)), 800)
  expect_equal(gpu_allocatable_bytes(st(resource = "async", pool_bytes = 50, pool_used_bytes = 50)), 100)
  # No resource: plain cudaMalloc, device free memory is the answer. A state
  # without the fields (older mocks) is the same.
  expect_equal(gpu_allocatable_bytes(st(resource = "cuda", pool_bytes = NA_real_, pool_used_bytes = NA_real_)), 100)
  expect_equal(gpu_allocatable_bytes(st()), 100)
  expect_true(is.na(gpu_allocatable_bytes(list(available = FALSE, free_bytes = NA_real_))))
  # More allocated than held can't happen; it never lowers the answer.
  expect_equal(gpu_allocatable_bytes(st(resource = "pool", pool_bytes = 10, pool_used_bytes = 20)), 100)

  # The real state: never below the device's free memory.
  real <- gpu_memory_state()
  expect_gte(gpu_allocatable_bytes(real), real$free_bytes)
  if (!identical(real$resource, "cuda")) {
    expect_equal(gpu_allocatable_bytes(real),
                 real$free_bytes + real$pool_bytes - real$pool_used_bytes)
  }
})

test_that("the join-size warning reads the memory resource's free pages", {
  skip_if_no_gpu()
  x <- tbl_gpu(data.frame(id = 1:100), lazy = FALSE)
  y <- tbl_gpu(data.frame(k = 1:100), lazy = FALSE)
  # est = 100 * 100 * 8 bytes * 2 columns = 160 kB; the device has 1 byte free
  # but the pool holds 1 GB of unused pages: no warning.
  testthat::local_mocked_bindings(
    gpu_memory_state = function() {
      list(available = TRUE, free_bytes = 1, resource = "pool",
           pool_bytes = 1e9, pool_used_bytes = 0)
    },
    .package = "cuplyr"
  )
  expect_no_warning(dplyr::cross_join(x, y))
})

test_that("a join does not warn about free memory the pool holds (real pool)", {
  skip_if_no_gpu()
  # CUPLYR_POOL_INITIAL is clamped to 90% of the free memory: the pool takes
  # almost all of what the device has, the way a pool that grew in a long
  # session does, and device-level free memory is the rest. A cross join sized
  # at 1.2 x that rest is estimated above 0.8 x the device's free memory and far
  # below what the pool can still serve.
  out <- f57r2_subprocess(function(load) {
    st <- gpu_memory_state()
    n <- ceiling(sqrt(1.2 * st$free_bytes / 16))
    a <- tbl_gpu(data.frame(p = runif(n)), lazy = FALSE)
    b <- tbl_gpu(data.frame(q = runif(n)), lazy = FALSE)
    warnings <- character()
    res <- withCallingHandlers(
      dplyr::cross_join(a, b),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    list(resource = st$resource, free = st$free_bytes, pool = st$pool_bytes,
         est = 16 * n * n, rows = nrow(res), warnings = warnings,
         allocatable = gpu_allocatable_bytes(st))
  }, env = c(CUPLYR_MEMORY_POOL = "true", CUPLYR_MEMORY_RESOURCE = "pool",
             CUPLYR_POOL_INITIAL = "1e15"))
  expect_identical(out$resource, "pool")
  # The set-up must really be the bug's: more estimated than 0.8 x device free.
  skip_if(out$est <= 0.8 * out$free, "the pool did not take enough of the card")
  expect_lt(out$est, 0.8 * out$allocatable)
  expect_identical(out$warnings, character())
  expect_equal(out$rows, ceiling(sqrt(1.2 * out$free / 16))^2)
})

test_that("reloading with live tables reuses the memory resource instead of orphaning a pool", {
  skip_if_no_gpu()
  out <- f57r2_subprocess(function(load) {
    s0 <- gpu_memory_state()
    # No live table: the one resource is reused, nothing is freed or added.
    load()
    load()
    s_empty <- gpu_memory_state()

    keep <- list()
    for (i in 1:3) {
      keep[[i]] <- tbl_gpu(data.frame(x = runif(1e5)), lazy = FALSE)
      load()
    }
    s_live <- gpu_memory_state()
    readable <- nrow(dplyr::collect(keep[[1]]))
    last <- dplyr::collect(dplyr::mutate(keep[[3]], y = x * 2))
    rm(keep, last)
    invisible(gc(full = TRUE))
    invisible(gc(full = TRUE))
    s1 <- gpu_memory_state()
    list(s0 = s0, s_empty = s_empty, s_live = s_live, s1 = s1, readable = readable)
  }, env = c(CUPLYR_MEMORY_POOL = "true", CUPLYR_MEMORY_RESOURCE = "pool"))

  mib <- 2^20
  expect_identical(out$readable, 100000L)
  expect_identical(out$s1$resource, "pool")
  # What the device holds does not grow with each reload...
  expect_lt(out$s_empty$used_bytes - out$s0$used_bytes, 128 * mib)
  expect_lt(out$s_live$used_bytes - out$s0$used_bytes, 128 * mib)
  expect_lt(out$s1$used_bytes - out$s0$used_bytes, 128 * mib)
  # ... and gpu_memory_state() reports the one resource, with the tables freed.
  expect_gte(out$s1$pool_bytes, 1)
  expect_lt(out$s1$pool_used_bytes - out$s0$pool_used_bytes, 16 * mib)
})

test_that("CUPLYR_MEMORY_RESOURCE=async runs a pipeline and reports its allocations", {
  skip_if_no_gpu()
  out <- f57r2_subprocess(function(load) {
    df <- data.frame(g = c(1L, 2L, 1L, 2L), x = c(1, 2, 3, 4))
    s0 <- gpu_memory_state()
    g <- tbl_gpu(data.frame(x = runif(1e6)), lazy = FALSE)
    s_live <- gpu_memory_state()
    rm(g)
    invisible(gc(full = TRUE))
    s_freed <- gpu_memory_state()
    res <- tbl_gpu(df, lazy = FALSE) |>
      dplyr::group_by(g) |>
      dplyr::mutate(r = cumsum(x), m = mean(x)) |>
      dplyr::arrange(dplyr::desc(x)) |>
      dplyr::collect()
    load()
    list(resource = s0$resource, s0 = s0, s_live = s_live, s_freed = s_freed,
         res = as.data.frame(res), after_reload = gpu_memory_state()$resource,
         allocatable = gpu_allocatable_bytes(s_live))
  }, env = c(CUPLYR_MEMORY_POOL = "true", CUPLYR_MEMORY_RESOURCE = "async"))
  expect_identical(out$resource, "async")
  df <- data.frame(g = c(1L, 2L, 1L, 2L), x = c(1, 2, 3, 4))
  expected <- as.data.frame(
    dplyr::arrange(dplyr::mutate(dplyr::group_by(df, g), r = cumsum(x), m = mean(x)),
                   dplyr::desc(x))
  )
  expect_equal(out$res, expected, ignore_attr = TRUE)
  expect_gte(out$s_live$pool_used_bytes - out$s0$pool_used_bytes, 8e6)
  expect_lt(out$s_freed$pool_used_bytes - out$s0$pool_used_bytes, 1e6)
  expect_gte(out$allocatable, out$s_live$free_bytes)
  expect_identical(out$after_reload, "async")
})

test_that("mutate() treats a forwarded missing .by/.before/.after as an error, an absent one as absent", {
  skip_if_no_gpu()
  df <- data.frame(x = 1:3)
  gt <- tbl_gpu(df, lazy = FALSE)

  by_fwd <- function(d, by) mutate(d, n = 1, .by = by)
  after_fwd <- function(d, a) mutate(d, n = 1, .after = a)
  before_fwd <- function(d, a) mutate(d, n = 1, .before = a)
  # `.by`'s message differs from dplyr's (resolve_by() in R/select.R predates
  # F57), so only "it is an error" is pinned for it.
  expect_error(by_fwd(df))
  expect_error(by_fwd(gt))
  expect_error(by_fwd(tbl_gpu(df, lazy = TRUE)))
  # `.before`/`.after`: dplyr's own error, class and message.
  for (f in list(after_fwd, before_fwd)) {
    expect_error(f(df), class = "vctrs_error_subscript_oob")
    expect_error(f(gt), class = "vctrs_error_subscript_oob")
    expect_error(f(tbl_gpu(df, lazy = TRUE)), class = "vctrs_error_subscript_oob")
    expect_identical(
      conditionMessage(tryCatch(f(gt), error = identity)),
      conditionMessage(tryCatch(f(df), error = identity))
    )
  }

  # Absent and explicit NULL are both "not given".
  expect_same_as_dplyr(df, function(d) mutate(d, n = 1))
  expect_same_as_dplyr(df, function(d) mutate(d, n = 1, .by = NULL, .before = NULL, .after = NULL))
})

# Upload 1e6 doubles (8,000,000 bytes) and report where they went. Shipped to the
# subprocess as an argument of `fn`, so it only uses package functions.
f57r3_probe <- function() {
  invisible(gc(full = TRUE))
  s0 <- gpu_memory_state()
  g <- tbl_gpu(data.frame(v = runif(1e6)), lazy = FALSE)
  s1 <- gpu_memory_state()
  tr <- gpu_alloc_tracking(tbl_gpu(data.frame(v = runif(1e6)), lazy = FALSE))
  held <- tr$held_bytes
  rm(g)
  freed <- gpu_gc(aggressive = FALSE)$freed_bytes
  rm(tr)
  invisible(gc(full = TRUE))
  list(resource = s1$resource, pool_bytes = s1$pool_bytes,
       pool_used = s1$pool_used_bytes - s0$pool_used_bytes,
       device_used = s1$used_bytes - s0$used_bytes,
       held = held, freed = freed)
}

expect_f57r3_pooled <- function(st, label) {
  expect_identical(st$resource, "pool", label = paste(label, "resource"))
  # The upload is carved out of the pool, not cudaMalloc'ed next to it.
  expect_gte(st$pool_used, 8e6, label = paste(label, "pool_used delta"))
  expect_lt(st$pool_used, 9e6, label = paste(label, "pool_used delta"))
  expect_lt(st$device_used, 4 * 2^20, label = paste(label, "device-level used delta"))
  expect_gte(st$held, 8e6, label = paste(label, "gpu_alloc_tracking() held"))
  expect_gte(st$freed, 8e6, label = paste(label, "gpu_gc() freed_bytes"))
  expect_lt(st$freed, 9e6, label = paste(label, "gpu_gc() freed_bytes"))
}

test_that("after any reload, new tables are allocated from the one memory resource", {
  skip_if_no_gpu()
  out <- f57r2_subprocess(function(load) {
    probe <- get("f57r3_probe_shipped", envir = globalenv())
    a <- probe()                               # first load
    load()
    b <- probe()                               # reload, no live table
    x <- tbl_gpu(data.frame(a = runif(1e5)), lazy = FALSE)
    load()
    c <- probe()                               # reload with a live table
    options(cuplyr.memory_pool = FALSE)
    load()
    d <- probe()                               # reload with the switch off
    list(a = a, b = b, c = c, d = d, x = nrow(dplyr::collect(x)))
  }, env = c(CUPLYR_MEMORY_POOL = "true", CUPLYR_MEMORY_RESOURCE = "pool"),
  globals = list(f57r3_probe_shipped = f57r3_probe))
  expect_identical(out$x, 100000L)
  for (stage in c("a", "b", "c", "d")) expect_f57r3_pooled(out[[stage]], stage)
  # One resource: every stage reports the same pool.
  expect_identical(out$d$pool_bytes, out$a$pool_bytes)
})

test_that("no value of the old hand-over option crashes a load or adds a resource", {
  skip_if_no_gpu()
  out <- f57r2_subprocess(function(load) {
    probe <- get("f57r3_probe_shipped", envir = globalenv())
    df <- data.frame(g = c(1L, 2L, 1L, 2L, 3L), x = c(5, 1, 4, 2, 3))
    keep <- tbl_gpu(df, lazy = FALSE)
    start <- gpu_memory_state()
    h <- getOption(".cuplyr_mr_handle")
    junk <- list(
      stale_double = c(1, 4096), na = NA, minus_one = -1, nan = c(NaN, NaN),
      list = list(1, 4096), string = "0x1000",
      serialized = unserialize(serialize(h, NULL)),
      null_extptr = methods::new("externalptr"),
      removed = NULL
    )
    res <- list()
    for (nm in names(junk)) {
      options(.cuplyr_mr_handle = junk[[nm]])
      load()
      p <- probe()
      p$res <- as.data.frame(tbl_gpu(df, lazy = FALSE) |>
        dplyr::group_by(g) |>
        dplyr::summarise(s = sum(x), m = max(x)) |>
        dplyr::arrange(g) |>
        dplyr::collect())
      res[[nm]] <- p
    }
    res$kept <- nrow(dplyr::collect(keep))
    res$grew <- gpu_memory_state()$used_bytes - start$used_bytes
    res$start_pool <- start$pool_bytes
    res
  }, env = c(CUPLYR_MEMORY_POOL = "true", CUPLYR_MEMORY_RESOURCE = "pool",
             CUPLYR_POOL_INITIAL = "67108864"),
  globals = list(f57r3_probe_shipped = f57r3_probe))

  df <- data.frame(g = c(1L, 2L, 1L, 2L, 3L), x = c(5, 1, 4, 2, 3))
  expected <- as.data.frame(dplyr::arrange(
    dplyr::summarise(dplyr::group_by(df, g), s = sum(x), m = max(x)), g))
  expect_identical(out$kept, 5L)
  # Still the one resource: no 64 MiB pool per reload.
  expect_lt(out$grew, 32 * 2^20)
  for (nm in setdiff(names(out), c("kept", "grew", "start_pool"))) {
    expect_f57r3_pooled(out[[nm]], nm)
    expect_identical(out[[nm]]$pool_bytes, out$start_pool, label = paste(nm, "pool_bytes"))
    expect_equal(out[[nm]]$res, expected, ignore_attr = TRUE, label = nm)
  }
})

test_that("pkgload::unload() then load_all() keeps old tables and libcudf working", {
  skip_if_no_gpu()
  # `first_pool`: whether the first load in the process has the pool. With it
  # off, the second load (a fresh copy of the library) creates the resource, and
  # it is that copy pkgload::unload() later dyn.unload()s.
  run <- function(first_pool) {
    f57r2_subprocess(function(load) {
      df <- data.frame(g = c(1L, 2L, 1L, 2L, 3L), x = c(5, 1, 4, 2, 3))
      options(cuplyr.memory_pool = TRUE)
      load()
      # Columns allocated by, and finalized through, the copy about to go.
      old <- tbl_gpu(data.frame(v = runif(1e5)), lazy = FALSE)
      old_sorted <- dplyr::arrange(old, v)
      pkgload::unload("cuplyr")
      # Map another library into the address range the unloaded copy left, so
      # the next copy can't land on it and run in its place by accident.
      filler <- file.path(tempdir(), "f57filler.so")
      file.copy(file.path(R.home("library"), "stats", "libs", "stats.so"), filler)
      dyn.load(filler)
      load()
      new <- as.data.frame(tbl_gpu(df, lazy = FALSE) |>
        dplyr::group_by(g) |>
        dplyr::summarise(s = sum(x)) |>
        dplyr::arrange(g) |>
        dplyr::collect())
      n_old <- nrow(dplyr::collect(old_sorted))
      rm(old, old_sorted)
      invisible(gc(full = TRUE))
      invisible(gc(full = TRUE))
      z <- dplyr::collect(dplyr::arrange(tbl_gpu(data.frame(v = runif(1e6)), lazy = FALSE), v))
      list(new = new, n_old = n_old, sorted = !is.unsorted(z$v), n = nrow(z),
           resource = gpu_memory_state()$resource)
    }, env = c(CUPLYR_MEMORY_POOL = if (first_pool) "true" else "0",
               CUPLYR_MEMORY_RESOURCE = "pool"))
  }
  df <- data.frame(g = c(1L, 2L, 1L, 2L, 3L), x = c(5, 1, 4, 2, 3))
  expected <- as.data.frame(dplyr::arrange(
    dplyr::summarise(dplyr::group_by(df, g), s = sum(x)), g))
  for (first_pool in c(TRUE, FALSE)) {
    out <- run(first_pool)
    expect_equal(out$new, expected, ignore_attr = TRUE)
    expect_identical(out$n_old, 100000L)
    expect_true(out$sorted)
    expect_identical(out$n, 1000000L)
    expect_identical(out$resource, "pool")
  }
})
