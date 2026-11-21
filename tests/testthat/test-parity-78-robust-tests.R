# Issue 78: timing, memory and child-process tests that don't flake on a busy
# machine.
#
# Three rules, checked over every tests/testthat/{test,helper}-*.R file:
#
#   1. A timing assertion with little headroom times through the shared helper
#      (timing_medians(), helper-timing.R). Raw timers (system.time(),
#      proc.time(), Sys.time()) stay only where listed below: order-of-magnitude
#      bounds with ten times or more headroom, and code that isn't a speed check.
#   2. A memory assertion reads the allocator's counters (gpu_alloc_used_bytes(),
#      gpu_alloc_snapshot(), pool_used_bytes, gpu_alloc_tracking()). A claim
#      about the device itself (driver pages, a second pool, memory handed back
#      to the device) reads this process's own device memory,
#      gpu_own_device_used_bytes(), which other processes on the GPU (a
#      compositor, a browser) don't move. Device-wide readings stay only where
#      listed below.
#   3. A child R process is started through gpu_child_r() (helper-memory.R),
#      which first hands this process's cached GPU memory back to the device
#      (gpu_free_for_child()). Code that starts a GPU child some other way
#      (quarto, the benchmark drivers) calls gpu_free_for_child() first.
#
# The checks are on the parsed code: a "unit" is one top-level expression of a
# file, named by its test_that() description or by the name it assigns.

p78_self <- "test-parity-78-robust-tests.R"

p78_files <- function() {
  f <- list.files(testthat::test_path(), pattern = "^(test|helper)-.*[.]R$")
  setdiff(f, p78_self)
}

# list of list(file, unit, expr), one per top-level expression of `file`
p78_units <- function(file) {
  exprs <- parse(testthat::test_path(file), keep.source = FALSE)
  lapply(exprs, function(e) {
    nm <- NULL
    if (is.call(e) && identical(e[[1]], as.name("test_that")) && length(e) >= 2L &&
        is.character(e[[2]])) {
      nm <- e[[2]]
    } else if (is.call(e) && length(e) == 3L &&
               (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) &&
               is.name(e[[2]])) {
      nm <- as.character(e[[2]])
    } else {
      nm <- deparse(e, width.cutoff = 60L)[[1]]
    }
    list(file = file, unit = nm, expr = e)
  })
}

# every call inside `e`, `e` included
p78_calls <- function(e) {
  acc <- list()
  walk <- function(x) {
    if (is.call(x)) {
      acc[[length(acc) + 1L]] <<- x
      for (a in as.list(x)) if (!missing(a)) walk(a)
    }
  }
  walk(e)
  acc
}

# What a call is: its function name (bare, and "pkg::name" when qualified),
# "pkg::name" for a `::` node itself, and "$field" for x$field / x[["field"]].
p78_tags <- function(cl) {
  h <- cl[[1]]
  tags <- character()
  if (is.name(h)) {
    tags <- as.character(h)
  } else if (is.call(h) && length(h) == 3L &&
             (identical(h[[1]], as.name("::")) || identical(h[[1]], as.name(":::")))) {
    tags <- c(paste0(as.character(h[[2]]), "::", as.character(h[[3]])), as.character(h[[3]]))
  }
  if (length(cl) == 3L && is.name(h) && as.character(h) %in% c("::", ":::")) {
    tags <- c(tags, paste0(as.character(cl[[2]]), "::", as.character(cl[[3]])))
  }
  if (length(cl) == 3L && is.name(h) && as.character(h) %in% c("$", "[[")) {
    f <- cl[[3]]
    if (is.name(f)) tags <- c(tags, paste0("$", as.character(f)))
    if (is.character(f) && length(f) == 1L) tags <- c(tags, paste0("$", f))
  }
  tags
}

p78_unit_tags <- function(u) unique(unlist(lapply(p78_calls(u$expr), p78_tags)))

p78_timer_tags <- c("system.time", "proc.time", "Sys.time")
p78_device_tags <- c("$used_bytes", "$free_bytes", "$free_memory", "gpu_memory_snapshot",
                     "gpu_mem_info")
p78_process_tags <- c("callr::r", "callr::r_bg", "callr::r_session", "callr::rscript",
                      "callr::rcmd", "callr::rcmd_bg", "callr::r_process", "processx::run",
                      "processx::process")
p78_alloc_tags <- c("gpu_alloc_used_bytes", "gpu_alloc_snapshot", "gpu_alloc_tracking",
                    "$pool_used_bytes")

# "file :: unit" of every unit whose calls include one of `tags`
p78_scan <- function(tags, skip = character()) {
  out <- character()
  for (f in setdiff(p78_files(), skip)) {
    for (u in p78_units(f)) {
      if (any(tags %in% p78_unit_tags(u))) out <- c(out, paste(f, u$unit, sep = " :: "))
    }
  }
  unique(out)
}

# Does `unit` of `file` call one of `tags`, directly or through a function
# defined at the top level of the same file? NA when there is no such unit.
p78_uses <- function(file, unit, tags) {
  units <- p78_units(file)
  nms <- vapply(units, function(u) u$unit, character(1))
  target <- units[nms == unit]
  if (!length(target)) return(NA)
  direct <- vapply(units, function(u) any(tags %in% p78_unit_tags(u)), logical(1))
  is_fun <- vapply(units, function(u) {
    is.call(u$expr) && length(u$expr) == 3L && is.call(u$expr[[3]]) &&
      identical(u$expr[[3]][[1]], as.name("function"))
  }, logical(1))
  wrappers <- nms[direct & is_fun]
  any(vapply(target, function(u) any(c(tags, wrappers) %in% p78_unit_tags(u)), logical(1)))
}

p78_key <- function(file, unit) paste(file, unit, sep = " :: ")

# --- rule 1: timing ----------------------------------------------------------

# Raw timers that stay, with why. Order-of-magnitude bounds: ten times or more
# between the bound and what the code takes, so machine load doesn't reach them.
p78_timer_allow <- c(
  # order of magnitude
  "test-F58-scope.R :: a subtree shared many times in a returned value is walked once" =
    "1 s bound; seconds without the visited set, microseconds with it",
  "test-dplyr-glimpse.R :: glimpse() stays fast on a large lazy table (no full materialize of preview data)" =
    "5 s bound for a preview of a 2e6-row lazy table",
  "test-gpu-info.R :: GPU detection is fast" =
    "1 s bound for has_gpu()",
  "test-parity-F49-rowwise-rewrites.R :: a 1e5-row rowwise sum/mean runs natively and in well under the CPU time" =
    "5 s bound; dplyr's per-row route takes ~20 s",
  "test-parity-F59-median.R :: F59: roll_median() at 1e6 rows takes well under a second; the sorted engine a few seconds" =
    "frollmedian ~20 ms vs 1 s; the sorted engine ~0.2 s vs 5 s",
  "test-parity-F66-median-exact.R :: F66: the exact default stays fast at 1e6 rows for w = 11/101/1001" =
    "~0.2 s vs 2 s",
  "test-parity-F61-recode.R :: f61r_time" =
    "recode 32-35 ms vs bounds of 0.5 s or dplyr's own 0.4-0.6 s",
  # not speed checks
  "helper-parity-F63.R :: f63_render" =
    "reports how long a render took; no bound",
  "test-F75-extra.R :: F75: the classifier allows pure shapes and flags the rest" =
    "Sys.time() is an expression under test",
  "test-F82-round4-extra.R :: F82: an if_any() lambda with a clock read stays native; a user today() goes to dplyr" =
    "a clock read as data",
  "test-bench-F73-lib-extra.R :: rw_polars_child() does not busy-wait while the child has an unfinished output line" =
    "this process's CPU time while a child sleeps (spinning vs waiting)",
  "test-parity-F106-timing.R :: timing_spin() busy-waits for the time it is given" =
    "tests the timing helper itself",
  "test-parity-F121-extras.R :: pp_time_cell() with cap = Inf sets no time limit and gives the same rows as without the argument" =
    "a spin loop for the benchmark harness's time limit",
  "test-parity-F121-extras.R :: a capped call is recorded as capped, with its error kept apart from other errors" =
    "a spin loop for the benchmark harness's time limit",
  "test-parity-F121-lib.R :: pp_busy() counts other processes' user-space load, not this process or kernel work" =
    "a spin loop for the benchmark harness's CPU-load probe",
  "test-parity-F121-rowwise.R :: pp_time_cell() caps a call at `cap` seconds and records it as capped" =
    "a 60 s spin capped well under 20 s",
  "test-parity-F63-qmd-render.R :: the report renders at smoke size and its knitr cache behaves" =
    "file timestamps",
  "test-parity-F63-qmd-static.R :: rw_cache_key() changes with file contents, versions and config only" =
    "file timestamps"
)

# Timing tests with little headroom that time through timing_medians().
p78_timer_converted <- list(
  c("test-parity-F61-group-split.R", "native group_split() at 10,000 groups on 1e6 rows reaches CPU parity"),
  c("test-parity-F59-rolling.R", "F59: narrow windows are no slower than dev, and wide windows cost about the same (ungrouped 1e7, grouped 2e6)"),
  c("test-F93-options-extra.R", "F93x: the unset lookup of the per-group readers stays cheap"),
  c("test-parity-F18-factor-rule.R", "F18 rule: the factor check is linear in expression size (wide if_any)")
)

test_that("raw timers are left only in order-of-magnitude bounds and in code that isn't a speed check", {
  found <- p78_scan(p78_timer_tags, skip = c("helper-timing.R", "helper-perf.R"))
  expect_identical(sort(setdiff(found, names(p78_timer_allow))), character(0))
  # every entry still exists and still uses a raw timer (a stale entry hides a change)
  expect_identical(sort(setdiff(names(p78_timer_allow), found)), character(0))
})

test_that("the timing tests with little headroom time through timing_medians()", {
  for (x in p78_timer_converted) {
    expect_identical(p78_uses(x[[1]], x[[2]], "timing_medians"), TRUE, label = p78_key(x[[1]], x[[2]]))
  }
})

# --- rule 2: memory ------------------------------------------------------------

# Device-wide readings that stay, with why. helper-memory.R holds the readers
# themselves and isn't scanned.
p78_device_allow <- c(
  "test-F57-round2.R :: gpu_allocatable_bytes() adds the unused part of the memory resource" =
    "the real state is compared with itself",
  "test-F57-round2.R :: a join does not warn about free memory the pool holds (real pool)" =
    "device free memory sizes the set-up, which is skipped when it isn't the bug's",
  "test-F57-round2.R :: CUPLYR_MEMORY_RESOURCE=async runs a pipeline and reports its allocations" =
    "allocatable vs free of one reading",
  "test-gpu-helpers.R :: gpu_memory_state() returns expected structure with GPU" =
    "tests the device-level fields themselves",
  "test-gpu-helpers.R :: gpu_memory_state() values are consistent" =
    "identities within one reading",
  "test-gpu-helpers.R :: gpu_memory_state() returns NA values without GPU" =
    "no GPU",
  "test-gpu-info.R :: gpu_details() returns expected fields when GPU available" =
    "tests gpu_details()'s fields themselves",
  "test-gpu-info.R :: GPU memory values are sensible" =
    "range checks of one reading",
  "test-gpu-info.R :: Multiple calls to gpu_details() return consistent info" =
    "two back-to-back readings within 10% of the card (1.2 GB here)",
  "test-parity-F56-alloc-tracking.R :: f56_used" =
    "only in a failure message of the tracking test",
  "test-parity-F57-pool.R :: f57_state_fn" =
    "reports the fields; the tests read the allocator's",
  "test-parity-F57-pool.R :: a pool resource is installed by default and reported by gpu_memory_state()" =
    "field checks; device usage >= the pool holds whatever else runs",
  "test-parity-F65-async.R :: f65_child" =
    "what the child saw, for failure messages",
  "test-parity-F65-async.R :: f65_skip_unless_device_free" =
    "decides whether to skip",
  "test-parity-F65-async.R :: state readings, allocation tracking and gpu_gc() are meaningful under async" =
    "field checks and identities within one reading",
  "test-parity-F65-async.R :: gpu_allocatable_bytes() counts the async pool's cached memory" =
    "allocatable vs free of one reading",
  "test-parity-F80-round3.R :: gpu_memory_state() reads the device's memory without gpu_info()" =
    "field checks within one reading"
)

# Claims about what a table occupies, read at the allocator.
p78_alloc_converted <- list(
  c("test-gpu-helpers.R", "gpu_memory_state() reflects allocations"),
  c("test-tbl-gpu.R", "tbl_gpu() allocates GPU memory"),
  c("test-tbl-gpu.R", "tbl_gpu() GPU memory is freed on garbage collection")
)

# Claims about the device, read as this process's own device memory.
p78_own_converted <- list(
  c("test-gpu-memory.R", "tbl_gpu() raises device-level GPU memory use"),
  c("test-parity-F56-alloc-tracking.R", "large columns raise device usage even with half-used pages"),
  c("test-F57-round2.R", "reloading with live tables reuses the memory resource instead of orphaning a pool"),
  c("test-F57-round2.R", "f57r3_probe"),
  c("test-F57-round2.R", "no value of the old hand-over option crashes a load or adds a resource"),
  c("test-parity-F65-async.R", "under async, a dropped 1 GB table goes back to the device after gpu_gc()"),
  c("test-parity-F65-async.R", "with cuplyr.release_threshold set, async gives memory back at the next sync without gpu_gc()"),
  c("test-parity-F65-async.R", "under pool, gpu_gc() frees at the allocator but the pool keeps its pages")
)

test_that("device-wide memory readings are left only where they can't be moved by another process", {
  found <- p78_scan(p78_device_tags, skip = "helper-memory.R")
  expect_identical(sort(setdiff(found, names(p78_device_allow))), character(0))
  expect_identical(sort(setdiff(names(p78_device_allow), found)), character(0))
})

test_that("memory tests read the allocator, or this process's own device memory for claims about the device", {
  for (x in p78_alloc_converted) {
    expect_identical(p78_uses(x[[1]], x[[2]], p78_alloc_tags), TRUE, label = p78_key(x[[1]], x[[2]]))
  }
  for (x in p78_own_converted) {
    expect_identical(p78_uses(x[[1]], x[[2]], "gpu_own_device_used_bytes"), TRUE,
                     label = p78_key(x[[1]], x[[2]]))
  }
})

# --- rule 3: child processes ---------------------------------------------------

# GPU children started by other code than gpu_child_r().
p78_other_launchers <- list(
  c("helper-parity-F63.R", "f63_render"),
  c("test-parity-F121-smoke.R", "f121_run_driver"),
  c("test-parity-F121-extras.R", "the driver's rowwise steps check, probe, time and cap at a small scale"),
  c("test-parity-F121-lib.R", "pp_first_call() loads this checkout's cuplyr in the child")
)

test_that("every child R process in the tests is started by gpu_child_r()", {
  found <- p78_scan(p78_process_tags)
  expect_identical(found, "helper-memory.R :: gpu_child_r")
})

test_that("GPU children started by quarto or the benchmark drivers free this process's memory first", {
  for (x in p78_other_launchers) {
    expect_identical(p78_uses(x[[1]], x[[2]], "gpu_free_for_child"), TRUE, label = p78_key(x[[1]], x[[2]]))
  }
})

test_that("gpu_child_r() hands the GPU memory back before it starts the child", {
  skip_if_not_installed("callr")
  calls <- character()
  local_mocked_bindings(
    has_gpu = function(...) TRUE,
    gpu_gc = function(...) {
      calls <<- c(calls, "gpu_gc")
      invisible(NULL)
    },
    .package = "cuplyr"
  )
  local_mocked_bindings(
    r = function(...) {
      calls <<- c(calls, "child")
      "from the child"
    },
    .package = "callr"
  )
  expect_identical(gpu_child_r(function() 1), "from the child")
  expect_identical(calls, c("gpu_gc", "child"))
})

test_that("memory the async resource cached is back on the device when the child starts", {
  skip_if_no_gpu()
  skip_if_not_installed("callr")
  skip_if(!identical(gpu_memory_state()$resource, "async"), "needs the async resource")
  gpu_gc(aggressive = FALSE)
  skip_if(gpu_allocatable_bytes() < 3e9, "needs 3 GB free on the device")
  local({
    g <- tbl_gpu(data.frame(a = runif(1.25e8)), lazy = FALSE)   # 1e9 bytes
    gpu_sync()
  })
  gc(full = TRUE)
  gc(full = TRUE)
  gpu_sync()
  st <- gpu_memory_state()
  cached <- st$pool_bytes - st$pool_used_bytes
  skip_if(cached < 0.9e9, "the async resource did not keep the dropped table cached")
  at_start <- NA_real_
  local_mocked_bindings(
    r = function(...) {
      s <- gpu_memory_state()
      at_start <<- s$pool_bytes - s$pool_used_bytes
      NULL
    },
    .package = "callr"
  )
  gpu_child_r(function() NULL)
  expect_lte(at_start, 64 * 2^20)
})

test_that("a child from gpu_child_r() gets its own async resource and the own-memory reader", {
  skip_if_no_gpu()
  skip_if_not_installed("pkgload")
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  out <- gpu_child_r(function(root) {
    has_reader <- exists("gpu_own_device_used_bytes", envir = globalenv(), inherits = FALSE)
    suppressMessages(pkgload::load_all(root, quiet = TRUE, compile = FALSE, export_all = TRUE,
                                       helpers = FALSE, attach_testthat = FALSE))
    g <- tbl_gpu(data.frame(x = c(1, 2, NA)), lazy = FALSE)
    list(has_reader = has_reader, resource = gpu_memory_state()$resource,
         x = dplyr::collect(dplyr::mutate(g, y = x * 2))$y)
  }, args = list(root = root))
  expect_true(out$has_reader)
  expect_identical(out$resource, "async")
  expect_identical(out$x, c(2, 4, NA))
})

test_that("gpu_own_device_used_bytes() follows this process's allocations up and back down", {
  skip_if_no_gpu()
  out <- run_with_pool_off(function() {
    pid_listed <- function() {
      smi <- Sys.which("nvidia-smi")
      if (!nzchar(smi)) return(FALSE)
      p <- tryCatch(suppressWarnings(system2(smi, c("--query-compute-apps=pid", "--format=csv,noheader"),
                                             stdout = TRUE, stderr = FALSE)),
                    error = function(e) character())
      as.character(Sys.getpid()) %in% trimws(p)
    }
    gc(full = TRUE)
    u0 <- gpu_own_device_used_bytes()
    g <- tbl_gpu(data.frame(x = runif(1e7)), lazy = FALSE)   # 8e7 bytes
    gpu_sync()
    u1 <- gpu_own_device_used_bytes()
    rm(g)
    gc(full = TRUE)
    gc(full = TRUE)
    gpu_sync()
    u2 <- gpu_own_device_used_bytes()
    list(u0 = as.numeric(u0), u1 = as.numeric(u1), u2 = as.numeric(u2),
         level = attr(u1, "level"), listed = pid_listed())
  })
  expect_true(out$level %in% c("process", "device"))
  # nvidia-smi lists this process: the reading is its own, not the device's
  if (isTRUE(out$listed)) expect_identical(out$level, "process")
  expect_gte(out$u1 - out$u0, 0.9 * 8e7)
  expect_lt(out$u1 - out$u0, 2 * 8e7)
  expect_lt(out$u2 - out$u0, 0.1 * 8e7)
})
