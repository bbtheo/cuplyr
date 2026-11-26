# #94: cuplyr runs on RAPIDS 26.02 and later, with RMM's resource_ref model.
#
# Contract (runtime):
# - cuplyr is built against, and loads, libcudf and RMM >= 26.02; the loaded
#   libcudf is the one it was built against. verify_installation() passes and
#   reports that version.
# - The memory resource cuplyr installs (async by default, or the pool) keeps
#   its contract after the port: gpu_memory_state() reports it, its
#   `pool_used_bytes` counts bytes allocated through it exactly (a column of n
#   doubles adds at least 8n bytes; freeing it gives them back), gpu_gc()
#   still works, and a second load_all() in the same process reuses it.
# - gpu_alloc_tracking() windows count allocations, nest (an outer window
#   counts the inner one), and leave the resource current as before.
# - gpu_oom_inject(n) still makes the next n allocations fail: one failure is
#   retried away, a persistent one raises cuplyr_error_gpu_oom; disarming
#   leaves allocations working.
# - Results of the verbs whose C++ had to change for libcudf 26.x (joins
#   including semi/anti joins, n_distinct(), slice_*(), filter() literals,
#   bind_rows(), every column type's transfer, parquet with a pushed-down
#   filter) match dplyr, eager and lazy.

p94_version_ok <- function(v) {
  !is.na(v) && numeric_version(v) >= numeric_version("26.02")
}

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

test_that("cuplyr is built against and loads libcudf and RMM 26.02 or later", {
  skip_if_no_gpu()
  built <- cuplyr_build_versions()
  expect_true(p94_version_ok(built[["libcudf"]]), info = built[["libcudf"]])
  expect_true(p94_version_ok(built[["rmm"]]), info = built[["rmm"]])

  utils::capture.output(info <- diagnostics(redact = FALSE))
  expect_identical(info$libcudf_loaded_version, info$libcudf_version)
})

test_that("verify_installation() passes and reports the libcudf version", {
  skip_if_no_gpu()
  utils::capture.output(res <- verify_installation())
  expect_true(res$ok)
  expect_true(p94_version_ok(res$libcudf_version), info = res$libcudf_version)
})

# ---------------------------------------------------------------------------
# The installed memory resource
# ---------------------------------------------------------------------------

p94_n <- 2e6  # 16 MB of doubles

test_that("the installed resource counts what is allocated through it and gives it back", {
  skip_if_no_gpu()
  st <- gpu_memory_state()
  skip_if(identical(st$resource, "cuda"), "no memory resource in this session")
  expect_true(st$resource %in% c("async", "pool"), info = st$resource)

  gc()
  before <- gpu_memory_state()$pool_used_bytes
  g <- tbl_gpu(data.frame(x = as.double(seq_len(p94_n))), lazy = FALSE)
  held <- gpu_memory_state()$pool_used_bytes
  expect_gte(held - before, 8 * p94_n)
  expect_lt(held - before, 8 * p94_n + 2^20)

  rm(g)
  gc()
  after <- gpu_memory_state()$pool_used_bytes
  expect_lt(after - before, 2^20)

  res <- gpu_gc()
  expect_type(res$freed_bytes, "double")
  st2 <- gpu_memory_state()
  expect_identical(st2$resource, st$resource)
  expect_false(is.na(st2$pool_bytes))
})

# Each kind in a fresh process: the kind is reported, the counter is exact, a
# table survives a second load_all() (the resource is reused, not replaced),
# and memory comes back. The child loads this worktree, or the installed
# package outside a source tree (cuplyr_child_loader()). The second load is
# pkgload's, so it runs only when `reload` (a source tree): without it, `kind2`
# and `after_reload` are NULL and the rest of the contract is still checked.
p94_kind_probe <- function(kind, reload = !is.null(cuplyr_source_root())) {
  if (reload) skip_if_not_installed("pkgload")
  root <- normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
  gpu_child_r(
    function(root, load, reload, n) {
      load(root)
      st0 <- gpu_memory_state()
      g <- tbl_gpu(data.frame(x = as.double(seq_len(n))), lazy = FALSE)
      st1 <- gpu_memory_state()
      st2 <- NULL
      if (reload) {
        load(root)
        st2 <- gpu_memory_state()
      }
      y <- dplyr::collect(dplyr::mutate(g, y = x * 2))$y
      tr <- gpu_alloc_tracking(tbl_gpu(data.frame(z = as.double(seq_len(n))), lazy = FALSE))
      tracked <- tr$allocated_bytes
      rm(g, tr)
      gc()
      st3 <- gpu_memory_state()
      list(kind0 = st0$resource, reloaded = reload, kind2 = st2$resource,
           delta = st1$pool_used_bytes - st0$pool_used_bytes,
           after_reload = if (reload) st2$pool_used_bytes - st0$pool_used_bytes,
           back = st3$pool_used_bytes - st0$pool_used_bytes,
           y_ok = identical(y, 2 * as.double(seq_len(n))),
           tracked = tracked)
    },
    args = list(root = root, load = cuplyr_child_loader(), reload = reload, n = p94_n),
    env = c(CUPLYR_MEMORY_RESOURCE = kind)
  )
}

test_that("the async resource keeps its contract in a fresh process", {
  skip_if_no_gpu()
  r <- p94_kind_probe("async")
  expect_identical(r$kind0, "async")
  expect_gte(r$delta, 8 * p94_n)
  expect_true(r$y_ok)
  expect_gte(r$tracked, 8 * p94_n)
  expect_lt(r$back, 2^20)
  if (r$reloaded) {
    expect_identical(r$kind2, "async")
    expect_gte(r$after_reload, 8 * p94_n)
  }
})

test_that("the pool resource keeps its contract in a fresh process", {
  skip_if_no_gpu()
  r <- p94_kind_probe("pool")
  expect_identical(r$kind0, "pool")
  expect_gte(r$delta, 8 * p94_n)
  expect_true(r$y_ok)
  expect_gte(r$tracked, 8 * p94_n)
  expect_lt(r$back, 2^20)
  if (r$reloaded) {
    expect_identical(r$kind2, "pool")
    expect_gte(r$after_reload, 8 * p94_n)
  }
})

# ---------------------------------------------------------------------------
# Allocation tracking windows
# ---------------------------------------------------------------------------

test_that("allocation tracking windows count allocations and nest", {
  skip_if_no_gpu()
  expect_identical(gpu_alloc_tracking_active(), 0L)
  kind <- gpu_memory_state()$resource

  inner <- NULL
  outer <- gpu_alloc_tracking({
    a <- tbl_gpu(data.frame(x = as.double(seq_len(p94_n))), lazy = FALSE)
    inner <- gpu_alloc_tracking(tbl_gpu(data.frame(y = as.double(seq_len(p94_n))), lazy = FALSE))
    expect_identical(gpu_alloc_tracking_active(), 1L)
    a
  })
  expect_identical(gpu_alloc_tracking_active(), 0L)

  expect_gte(inner$allocated_bytes, 8 * p94_n)
  expect_gte(inner$held_bytes, 8 * p94_n)
  expect_gte(inner$peak_bytes, inner$held_bytes)
  # the outer window counts its own table and the inner one
  expect_gte(outer$allocated_bytes, inner$allocated_bytes + 8 * p94_n)
  expect_gte(outer$peak_bytes, outer$held_bytes)

  # the same resource is current afterwards, and still counts
  expect_identical(gpu_memory_state()$resource, kind)
  rm(outer, inner)
  gc()
  tr <- gpu_alloc_tracking(NULL)
  expect_identical(tr$allocated_bytes, 0)
})

test_that("a window freed after it closed does not break later allocations", {
  skip_if_no_gpu()
  for (i in 1:3) {
    tr <- gpu_alloc_tracking(tbl_gpu(data.frame(x = as.double(seq_len(1e5))), lazy = FALSE))
    expect_gte(tr$allocated_bytes, 8e5)
    rm(tr)
    gc()
  }
  df <- data.frame(x = c(3, 1, 2))
  expect_identical(collect(arrange(tbl_gpu(df, lazy = FALSE), x))$x, c(1, 2, 3))
})

# ---------------------------------------------------------------------------
# OOM injection
# ---------------------------------------------------------------------------

test_that("gpu_oom_inject() still makes allocations fail, and disarming restores them", {
  skip_if_no_gpu()
  withr::defer(gpu_oom_inject(0L))
  df <- data.frame(g = c("a", "b", "a"), x = c(1, 2, 3))
  a <- tbl_gpu(df, lazy = FALSE)

  # one failure: retried away
  gpu_oom_inject(1L)
  res <- collect(summarise(group_by(a, g), s = sum(x)))
  expect_identical(gpu_oom_inject_remaining(), 0L)
  expect_equal(res[order(res$g), ]$s, c(4, 2))

  # a persistent one: cuplyr's OOM error after one retry
  gpu_oom_inject(3L)
  expect_error(collect(mutate(a, y = x * 2)), class = "cuplyr_error_gpu_oom")
  expect_identical(gpu_oom_inject_remaining(), 1L)

  gpu_oom_inject(0L)
  expect_identical(gpu_oom_inject_remaining(), 0L)
  expect_identical(collect(mutate(a, y = x * 2))$y, c(2, 4, 6))

  # injection on top of a tracking window, and a window on top of injection
  tr <- gpu_alloc_tracking({
    gpu_oom_inject(1L)
    collect(mutate(a, y = x + 1))
  })
  expect_identical(tr$value$y, c(2, 3, 4))
  gpu_oom_inject(0L)
  expect_identical(collect(mutate(a, y = x - 1))$y, c(0, 1, 2))
})

# ---------------------------------------------------------------------------
# dplyr parity on the code paths libcudf 26.x changed
# ---------------------------------------------------------------------------

p94_like <- function(d, other) {
  if (inherits(d, "tbl_gpu")) tbl_gpu(other, lazy = identical(d$exec_mode, "lazy")) else other
}

p94_both <- function(df, pipeline, arrange_by = NULL) {
  expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
}

p94_x <- function() {
  data.frame(k = c(1L, 2L, NA, 3L, 2L, 5L),
             s = c("a", "b", NA, "c", "b", "e"),
             v = c(1.5, NA, 3, 4, 5, 6))
}

p94_y <- function() {
  data.frame(k = c(2L, NA, 3L, 4L, 2L),
             s = c("b", NA, "c", "d", "b"),
             w = c(10, 20, 30, 40, 50))
}

test_that("semi_join() and anti_join() match dplyr, NA keys included", {
  skip_if_no_gpu()
  y <- p94_y()
  for (na in c("na", "never")) {
    p94_both(p94_x(), function(d) semi_join(d, p94_like(d, y), by = "k", na_matches = na))
    p94_both(p94_x(), function(d) anti_join(d, p94_like(d, y), by = "k", na_matches = na))
    p94_both(p94_x(), function(d) semi_join(d, p94_like(d, y), by = c("k", "s"), na_matches = na))
    p94_both(p94_x(), function(d) anti_join(d, p94_like(d, y), by = "s", na_matches = na))
  }
  p94_both(p94_x()[0, ], function(d) semi_join(d, p94_like(d, y), by = "k"))
  p94_both(p94_x(), function(d) anti_join(d, p94_like(d, y[0, ]), by = "k"))
})

test_that("mutating joins match dplyr, NA keys and multiple matches included", {
  skip_if_no_gpu()
  y <- p94_y()
  p94_both(p94_x(), function(d) inner_join(d, p94_like(d, y), by = "k", relationship = "many-to-many"),
           arrange_by = c("k", "s.x", "v", "w"))
  p94_both(p94_x(), function(d) left_join(d, p94_like(d, y), by = "k", relationship = "many-to-many"),
           arrange_by = c("k", "s.x", "v", "w"))
  p94_both(p94_x(), function(d) full_join(d, p94_like(d, y), by = "s", na_matches = "never",
                                          relationship = "many-to-many"),
           arrange_by = c("s", "k.x", "v", "w"))
  p94_both(p94_x(), function(d) right_join(d, p94_like(d, y), by = c("k", "s")),
           arrange_by = c("k", "s", "w"))
})

test_that("n_distinct() matches dplyr, grouped and ungrouped, with NA", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1L, 1L, 2L, 2L, 2L, 3L),
                   x = c(1, NA, 2, 2, NA, NA),
                   s = c("a", "a", NA, "b", "b", NA))
  p94_both(df, function(d) summarise(d, n = n_distinct(x), m = n_distinct(s),
                                     k = n_distinct(x, na.rm = TRUE)))
  p94_both(df, function(d) summarise(group_by(d, g), n = n_distinct(x), m = n_distinct(s),
                                     k = n_distinct(s, na.rm = TRUE)))
  p94_both(df[0, ], function(d) summarise(d, n = n_distinct(x)))
  p94_both(df, function(d) distinct(d, s))
  p94_both(df, function(d) count(d, s))
})

test_that("slice_*() verbs match dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b", "b", "c"),
                   x = c(3, 1, NA, 5, 2, 4))
  p94_both(df, function(d) slice_head(d, n = 2))
  p94_both(df, function(d) slice_tail(group_by(d, g), n = 1))
  p94_both(df, function(d) slice_min(d, x, n = 2))
  p94_both(df, function(d) slice_max(group_by(d, g), x, n = 1, with_ties = FALSE))
  p94_both(df, function(d) slice(group_by(d, g), 2))
  p94_both(df, function(d) slice(d, -1))
  p94_both(df[0, ], function(d) slice_head(d, n = 1))
})

test_that("filter() literals of every type match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, 2L, NA, 4L), dbl = c(0.5, NA, 2.5, -1),
                   s = c("a", "b", NA, "d"), l = c(TRUE, FALSE, NA, TRUE),
                   dt = as.Date(c("2024-01-01", NA, "2024-03-01", "2023-12-31")))
  p94_both(df, function(d) filter(d, i > 1L))
  p94_both(df, function(d) filter(d, dbl <= 2.5, !is.na(s)))
  p94_both(df, function(d) filter(d, s == "b" | s %in% c("d", "z")))
  p94_both(df, function(d) filter(d, l))
  p94_both(df, function(d) filter(d, dt >= as.Date("2024-01-01")))
  p94_both(df, function(d) filter(d, i + 1L == 3L | dbl * 2 > 4))
})

test_that("bind_rows() matches dplyr, with missing columns and empty inputs", {
  skip_if_no_gpu()
  a <- data.frame(x = c(1, 2), s = c("a", NA))
  b <- data.frame(x = 3, t = 1L)
  p94_both(a, function(d) bind_rows(d, p94_like(d, b)))
  p94_both(a, function(d) bind_rows(d, p94_like(d, a[0, ])))
  p94_both(a[0, ], function(d) bind_rows(d, p94_like(d, b)))
})

test_that("every column type round-trips and survives a verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    l = c(TRUE, NA, FALSE),
    i = c(1L, NA, -3L),
    d = c(1.5, NA, -Inf),
    s = c("a", NA, "é"),
    dt = as.Date(c("2024-01-01", NA, "1969-12-31")),
    ts = as.POSIXct(c("2024-01-01 12:00:00", NA, "1969-12-31 23:59:59"), tz = "UTC"),
    f = factor(c("b", NA, "a"), levels = c("a", "b", "z"))
  )
  p94_both(df, function(d) d)
  p94_both(df, function(d) arrange(d, i))
  p94_both(df, function(t) mutate(t, i2 = i * 2L, d2 = d + 1))
  if (requireNamespace("bit64", quietly = TRUE)) {
    df64 <- tibble::tibble(x = bit64::as.integer64(c("9007199254740993", NA, "-5")))
    expect_identical(collect(tbl_gpu(df64, lazy = FALSE))$x, df64$x)
  }
})

test_that("parquet files read with a pushed-down filter match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, 5L, NA, 9L, 2L), dbl = c(1, 2, 3, NA, 5),
                   s = c("a", "b", "c", NA, "e"),
                   dt = as.Date("2024-01-01") + c(0L, 10L, NA, 30L, 40L))
  path <- withr::local_tempfile(fileext = ".parquet")
  write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path)

  cases <- list(
    function(d) filter(d, i > 2L),
    function(d) filter(d, dbl <= 3),
    function(d) filter(d, dt >= as.Date("2024-01-20")),
    function(d) filter(d, s == "b"),
    function(d) filter(d, i > 1L, dbl < 5)
  )
  for (p in cases) {
    expected <- tibble::as_tibble(p(df))
    for (lazy in c(FALSE, TRUE)) {
      got <- tibble::as_tibble(collect(p(read_parquet_gpu(path, lazy = lazy))))
      expect_equal(got, expected, ignore_attr = TRUE)
    }
  }
})
