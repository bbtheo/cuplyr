# F57: an RMM memory resource installed at load time, with truthful reporting.
#
# Contract (scratchpad/parity/features/F57.md, Design C):
# - At load, unless disabled, cuplyr installs a pool ("pool":
#   rmm pool_memory_resource, or "async": cuda_async_memory_resource) as the
#   current RMM device resource.
# - Off switch, read once at load: options(cuplyr.memory_pool = FALSE) or env
#   CUPLYR_MEMORY_POOL = "0"/"false"/"no"/"off". The option wins when set. Off
#   installs nothing (resource "cuda", today's behaviour).
# - Initial size: options(cuplyr.pool_initial = <bytes>) or env
#   CUPLYR_POOL_INITIAL; never more than the card.
# - gpu_memory_state() keeps its device-level fields and adds `resource`,
#   `pool_bytes` (held by the pool from the device) and `pool_used_bytes`
#   (allocated from the pool right now); both NA when resource == "cuda".
# - gpu_gc()$freed_bytes is allocator-level under a pool.
# - gpu_alloc_tracking() (F56) keeps working on top of the pool.
#
# Load-time behaviour runs in callr subprocesses that load this worktree with
# pkgload::load_all(compile = FALSE): they reuse the already built library.

f57_pkg_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
}

# Run `fn(state)` in a fresh R process that loads this worktree after setting
# `opts` and env vars `env` (twice when `reload`). callr ships `fn` without its
# environment, so `fn` only uses package functions and its `state` argument,
# which is f57_state_fn() re-homed to the global environment.
f57_subprocess <- function(fn = function(state) state(), env = character(),
                           opts = list(), reload = FALSE) {
  skip_if_not_installed("callr")
  skip_if_not_installed("pkgload")
  state <- f57_state_fn
  environment(state) <- globalenv()
  callr::r(
    function(root, opts, fn, reload, state) {
      if (length(opts)) options(opts)
      load <- function() {
        suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE,
                                           export_all = TRUE, helpers = FALSE,
                                           attach_testthat = FALSE))
      }
      load()
      if (reload) load()
      fn(state)
    },
    args = list(root = f57_pkg_root(), opts = opts, fn = fn, reload = reload,
                state = state),
    env = c(callr::rcmd_safe_env(), env),
    timeout = 600
  )
}

f57_state_fn <- function() {
  st <- gpu_memory_state()
  g <- tbl_gpu(data.frame(x = c(1, NA, 3)), lazy = FALSE)
  y <- dplyr::collect(dplyr::mutate(g, y = x * 2))$y
  tr <- gpu_alloc_tracking(tbl_gpu(data.frame(x = runif(1e5)), lazy = FALSE))
  list(
    resource = st$resource,
    pool_bytes = st$pool_bytes,
    pool_used_bytes = st$pool_used_bytes,
    total_bytes = st$total_bytes,
    used_bytes = st$used_bytes,
    y = y,
    held = tr$held_bytes
  )
}

f57_mb40 <- 1e6 * 5 * 8

test_that("a pool resource is installed by default and reported by gpu_memory_state()", {
  skip_if_no_gpu()
  st <- gpu_memory_state()

  # Today's fields are still there.
  expect_true(all(c("available", "total_bytes", "free_bytes", "used_bytes",
                    "total_gb", "free_gb", "used_gb") %in% names(st)))
  expect_true(all(c("resource", "pool_bytes", "pool_used_bytes") %in% names(st)))

  expect_true(st$resource %in% c("pool", "async"), info = st$resource)
  expect_type(st$pool_bytes, "double")
  expect_type(st$pool_used_bytes, "double")
  expect_false(is.na(st$pool_bytes))
  expect_false(is.na(st$pool_used_bytes))

  expect_gte(st$pool_used_bytes, 0)
  expect_lte(st$pool_used_bytes, st$pool_bytes)
  expect_lte(st$pool_bytes, st$total_bytes)
  # Device-level "used" includes everything the pool holds from the device.
  expect_gte(st$used_bytes, st$pool_bytes)
  expect_equal(st$used_bytes, st$total_bytes - st$free_bytes)
})

test_that("pool_used_bytes follows allocations and frees", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 1e6, ncol = 5)

  gc_gpu()
  before <- gpu_memory_state()$pool_used_bytes
  g <- tbl_gpu(df, lazy = FALSE)
  during <- gpu_memory_state()$pool_used_bytes
  expect_gte(during - before, f57_mb40)
  expect_lt(during - before, 2 * f57_mb40)

  rm(g)
  gc_gpu()
  after <- gpu_memory_state()$pool_used_bytes
  expect_lt(after - before, 0.1 * f57_mb40)
})

test_that("gpu_gc() reports allocator-level freed bytes under the pool", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 1e6, ncol = 5)
  gc_gpu()

  local({
    g <- tbl_gpu(df, lazy = FALSE)
    expect_data_on_gpu(g)
  })
  res <- gpu_gc(verbose = FALSE)
  # The 40 MB table was garbage when gpu_gc() started and freed by it. Under a
  # pool the device-level free memory doesn't move, so a device-level number
  # would say ~0 here.
  expect_gte(res$freed_bytes, 0.9 * f57_mb40)
  expect_true(res$before$resource %in% c("pool", "async"))

  local({
    g <- tbl_gpu(df, lazy = FALSE)
    NULL
  })
  expect_message(gpu_gc(verbose = TRUE), "freed")
})

test_that("allocation tracking works on top of the pool and restores it", {
  skip_if_no_gpu()
  df <- create_large_test_data(nrow = 1e6, ncol = 5)
  resource <- gpu_memory_state()$resource

  gc_gpu()
  before <- gpu_memory_state()$pool_used_bytes
  tr <- gpu_alloc_tracking(tbl_gpu(df, lazy = FALSE))
  delta <- gpu_memory_state()$pool_used_bytes - before

  expect_gte(tr$held_bytes, f57_mb40)
  # What the tracking window saw went to the pool.
  expect_gte(delta, 0.95 * tr$held_bytes)
  expect_lte(delta, 1.3 * tr$held_bytes)
  expect_identical(gpu_alloc_tracking_active(), 0L)
  expect_identical(gpu_memory_state()$resource, resource)
  expect_equal(as.data.frame(collect(tr$value)), df)
})

test_that("CUPLYR_MEMORY_POOL=0 installs nothing (today's behaviour)", {
  skip_if_no_gpu()
  out <- f57_subprocess(env = c(CUPLYR_MEMORY_POOL = "0"))
  expect_identical(out$resource, "cuda")
  expect_true(is.na(out$pool_bytes))
  expect_true(is.na(out$pool_used_bytes))
  expect_equal(out$y, c(2, NA, 6))
  expect_gte(out$held, 1e5 * 8)

  out_off <- f57_subprocess(env = c(CUPLYR_MEMORY_POOL = "off"))
  expect_identical(out_off$resource, "cuda")
})

test_that("options(cuplyr.memory_pool = FALSE) disables the pool and wins over the env var", {
  skip_if_no_gpu()
  out <- f57_subprocess(env = c(CUPLYR_MEMORY_POOL = "true"),
                        opts = list(cuplyr.memory_pool = FALSE))
  expect_identical(out$resource, "cuda")
  expect_true(is.na(out$pool_bytes))
  expect_equal(out$y, c(2, NA, 6))

  on <- f57_subprocess(env = c(CUPLYR_MEMORY_POOL = "0"),
                       opts = list(cuplyr.memory_pool = TRUE))
  expect_true(on$resource %in% c("pool", "async"))
})

test_that("the initial pool size is honoured and never exceeds the card", {
  skip_if_no_gpu()
  mib256 <- 256 * 2^20
  out <- f57_subprocess(env = c(CUPLYR_MEMORY_POOL = "true",
                                CUPLYR_POOL_INITIAL = format(mib256, scientific = FALSE)))
  expect_true(out$resource %in% c("pool", "async"))
  if (identical(out$resource, "pool")) {
    expect_gte(out$pool_bytes, mib256)
  }
  expect_lte(out$pool_bytes, out$total_bytes)
  expect_equal(out$y, c(2, NA, 6))

  # A request larger than the card is clamped; loading still works.
  huge <- f57_subprocess(function(state) suppressWarnings(state()),
                         env = c(CUPLYR_MEMORY_POOL = "true"),
                         opts = list(cuplyr.pool_initial = 1e15))
  expect_true(huge$resource %in% c("pool", "async"))
  expect_lte(huge$pool_bytes, huge$total_bytes)
  expect_equal(huge$y, c(2, NA, 6))
})

test_that("loading the package twice in one process leaves a working resource", {
  skip_if_no_gpu()
  out <- f57_subprocess(function(state) {
    res <- state()
    g <- tbl_gpu(data.frame(x = runif(1e5)), lazy = FALSE)
    m <- dplyr::mutate(g, r = cumsum(x))
    rm(g)
    gpu_gc(aggressive = FALSE)
    res$n <- nrow(dplyr::collect(m))
    res
  }, env = c(CUPLYR_MEMORY_POOL = "true"), reload = TRUE)
  expect_true(out$resource %in% c("pool", "async"))
  expect_equal(out$y, c(2, NA, 6))
  expect_identical(out$n, 100000L)
})
