# F121 helpers: load bench/parquet-lib.R (the parquet-to-summary pipeline
# benchmark) with the benchmark library first on the path, make small fixture
# files on a real disk (page-cache eviction does nothing on tmpfs), and an
# independent base-R reference for every pipeline variant.
#
# bench/ is .Rbuildignore'd, so the tests skip outside a source checkout.

f121_bench_lib <- function() {
  Sys.getenv("CUPLYR_BENCH_LIB", "/home/theo/cuplr/scratchpad/benchmarks/rlib")
}

# Where the benchmark keeps its data; the tests write their files under
# <data root>/test-tmp (a disk, not /tmp, which is tmpfs here).
f121_data_root <- function() {
  Sys.getenv("CUPLYR_BENCH_DATA", "/home/theo/cuplr/scratchpad/benchmarks/parquet-pipeline")
}

f121_file <- function(rel) {
  root <- testthat::test_path("..", "..")
  if (!file.exists(file.path(root, "DESCRIPTION"))) {
    testthat::skip("not in a source checkout; bench/ is not shipped")
  }
  path <- file.path(root, rel)
  testthat::expect_true(file.exists(path), label = paste(rel, "exists"))
  if (!file.exists(path)) testthat::skip(paste(rel, "missing (failure recorded)"))
  normalizePath(path)
}

# Source parquet-lib.R into a fresh environment, with the benchmark library
# prefixed to .libPaths() for the caller's frame. A package missing from the
# benchmark library (when it exists) is a failure, not a skip.
f121_load <- function(pkgs = c("arrow", "duckdb", "DBI", "data.table"), env = parent.frame()) {
  path <- f121_file("bench/parquet-lib.R")
  lib <- f121_bench_lib()
  if (dir.exists(lib)) {
    withr::local_libpaths(lib, action = "prefix", .local_envir = env)
    for (p in pkgs) {
      ok <- requireNamespace(p, quietly = TRUE)
      testthat::expect_true(ok, label = paste(p, "is installed (benchmark or user library)"))
      if (!ok) testthat::skip(paste(p, "missing (failure recorded)"))
    }
  } else {
    for (p in pkgs) testthat::skip_if_not_installed(p)
  }
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

# A fresh directory on disk, removed when `env` exits.
f121_tmpdir <- function(env = parent.frame()) {
  base <- file.path(f121_data_root(), "test-tmp")
  if (!dir.exists(dirname(base)) && !dir.create(dirname(base), recursive = TRUE, showWarnings = FALSE)) {
    testthat::skip("benchmark data root not writable")
  }
  dir.create(base, showWarnings = FALSE)
  d <- tempfile("f121-", tmpdir = base)
  dir.create(d)
  withr::defer(unlink(d, recursive = TRUE), envir = env)
  normalizePath(d)
}

f121_is_tmpfs <- function(path) {
  fs <- tryCatch(system2("stat", c("-f", "-c", "%T", shQuote(path)), stdout = TRUE),
                 error = function(e) "", warning = function(w) "")
  any(grepl("tmpfs|ramfs", fs))
}

# Small fixture files (60000 rows, row groups of 5000 rows: 12 groups, so the
# sorted day column lets the pushed date filter skip row groups), made once
# per test process.
f121_fixture_env <- new.env()
f121_fixture <- function(pp) {
  if (!is.null(f121_fixture_env$files)) return(f121_fixture_env$files)
  base <- file.path(f121_data_root(), "test-tmp")
  dir.create(base, recursive = TRUE, showWarnings = FALSE)
  d <- tempfile("f121-fix-", tmpdir = base)
  dir.create(d)
  withr::defer(unlink(d, recursive = TRUE), envir = testthat::teardown_env())
  files <- list(
    narrow = file.path(d, "fact-narrow.parquet"),
    wide = file.path(d, "fact-wide.parquet"),
    dim = file.path(d, "stores.parquet"),
    n = 60000L, rg = 5000L
  )
  pp$pp_gen_fact(files$narrow, n = files$n, wide = FALSE, seed = 121L, row_group_rows = files$rg)
  pp$pp_gen_fact(files$wide, n = files$n, wide = TRUE, seed = 121L, row_group_rows = files$rg)
  pp$pp_gen_dim(files$dim, seed = 121L)
  f121_fixture_env$files <- files
  files
}

f121_variants <- c("p1_push", "p1_nopush", "p2_push", "p2_nopush", "p3_push", "p3_nopush")

f121_cols <- list(
  p1 = c("store", "channel", "n", "revenue", "avg_price", "max_qty"),
  p2 = c("region", "channel", "n", "revenue", "avg_price", "max_qty"),
  p3 = c("product", "revenue", "n")
)

# Independent reference in base R (no dplyr, no data.table, no SQL), in the
# canonical form pp_normalize() must produce: a plain data.frame, columns as
# in f121_cols, keys integer (store, product) or character (channel, region),
# measures double, sorted by the keys with method = "radix" (p1, p2) or by
# revenue descending then product (p3: the top `top_n`, ties at the boundary
# kept, as slice_max() does).
f121_base_ref <- function(variant, f, d, k) {
  push <- grepl("_push$", variant)
  keep <- if (push) {
    f$day >= k$day_from & f$qty >= k$qty_min
  } else {
    f$price >= k$price_min & f$channel != k$channel_drop
  }
  keep <- keep & !is.na(keep)
  x <- f[keep, , drop = FALSE]
  x$revenue <- x$price * x$qty * (1 - x$discount)
  pipe <- sub("_.*$", "", variant)
  if (pipe == "p2") x$region <- d$region[match(x$store, d$store)]
  keys <- switch(pipe, p1 = c("store", "channel"), p2 = c("region", "channel"), p3 = "product")
  key <- do.call(paste, c(unname(as.list(x[keys])), sep = "\r"))
  ix <- split(seq_len(nrow(x)), key)
  first <- vapply(ix, function(i) i[[1L]], 1L)
  out <- x[first, keys, drop = FALSE]
  out$n <- as.double(lengths(ix))
  out$revenue <- vapply(ix, function(i) sum(x$revenue[i], na.rm = TRUE), 0)
  if (pipe == "p3") {
    out <- out[, f121_cols$p3, drop = FALSE]
    o <- order(-out$revenue, out$product, method = "radix")
    out <- out[o, , drop = FALSE]
    cut <- out$revenue[[min(k$top_n, nrow(out))]]
    out <- out[out$revenue >= cut, , drop = FALSE]
  } else {
    out$avg_price <- vapply(ix, function(i) mean(x$price[i]), 0)
    out$max_qty <- vapply(ix, function(i) as.double(max(x$qty[i])), 0)
    out <- out[, f121_cols[[pipe]], drop = FALSE]
    o <- do.call(order, c(unname(as.list(out[keys])), list(method = "radix")))
    out <- out[o, , drop = FALSE]
  }
  rownames(out) <- NULL
  out
}
