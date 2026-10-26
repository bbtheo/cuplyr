# F68a acceptance tests: lazy verb calls are cheap.
#
# Before F68a, recording a verb on a lazy table re-inferred the schema of the
# whole pending AST (current_schema() -> infer_schema_full(), itself calling
# the recursive infer_schema() at every level), and lowering/optimizing
# re-inferred each node's input schema the same way: both grow with the square
# of the plan's depth. On F62's 5-verb pipeline that made lazy ~1 ms slower
# than eager at 1e3 and 1e6 rows (F62/measure/summary.txt).
#
# Timings are medians of interleaved, gpu_sync()-fenced batches (the method of
# F62's measure.R), and the checks are ratios on the same machine, so they
# don't depend on the GPU's absolute speed.

f68s_time <- function(fns, samples = 11L, target = 0.03) {
  # One median per closure, sampled in interleaved rounds so drift (clocks,
  # other load) hits every closure alike. Each sample is a batch of calls
  # lasting >= `target` seconds; gc() runs untimed before each batch.
  for (fn in fns) for (i in 1:3) fn()
  reps <- vapply(fns, function(fn) {
    gpu_sync(); t0 <- proc.time()[["elapsed"]]; fn(); gpu_sync()
    one <- max(proc.time()[["elapsed"]] - t0, 1e-4)
    max(1L, min(500L, as.integer(ceiling(target / one))))
  }, integer(1))
  t <- matrix(NA_real_, samples, length(fns))
  for (s in seq_len(samples)) {
    for (k in seq_along(fns)) {
      gc(verbose = FALSE)
      gpu_sync(); t0 <- proc.time()[["elapsed"]]
      for (j in seq_len(reps[[k]])) fns[[k]]()
      gpu_sync()
      t[s, k] <- (proc.time()[["elapsed"]] - t0) / reps[[k]]
    }
  }
  stats::setNames(apply(t, 2, stats::median), names(fns))
}

f68s_df <- function(n) {
  set.seed(1)
  data.frame(g = sample.int(100L, n, TRUE), x = runif(n), y = runif(n), w = rnorm(n))
}

f68s_pipeline <- function(d) {
  d |>
    filter(x > 0.1) |>
    mutate(z = x * y + w) |>
    group_by(g) |>
    summarise(m = mean(z), n = n()) |>
    arrange(desc(m))
}

f68s_chain <- function(d, k) {
  for (i in seq_len(k)) {
    d <- d |> mutate(x = x + 1) |> filter(x > -1e9) |> select(g, x, y, w)
  }
  d
}

test_that("recording a verb costs the same on a deep pending plan as on a shallow one", {
  skip_if_no_gpu()
  lz <- tbl_gpu(f68s_df(1000L), lazy = TRUE)
  shallow <- f68s_chain(lz, 1L)   # 3 pending nodes
  deep <- f68s_chain(lz, 40L)     # 120 pending nodes
  t <- f68s_time(list(
    filter_shallow = function() filter(shallow, y > 0.5),
    filter_deep = function() filter(deep, y > 0.5),
    mutate_shallow = function() mutate(shallow, q = x * y),
    mutate_deep = function() mutate(deep, q = x * y),
    summarise_shallow = function() summarise(group_by(shallow, g), m = mean(x)),
    summarise_deep = function() summarise(group_by(deep, g), m = mean(x))
  ), samples = 7L)
  # Before F68a the deep plan made each call ~40-100x slower.
  expect_lt(t[["filter_deep"]] / t[["filter_shallow"]], 1.5)
  expect_lt(t[["mutate_deep"]] / t[["mutate_shallow"]], 1.5)
  expect_lt(t[["summarise_deep"]] / t[["summarise_shallow"]], 1.5)
})

test_that("collecting a plan costs time linear in its depth", {
  skip_if_no_gpu()
  lz <- tbl_gpu(f68s_df(1000L), lazy = TRUE)
  p10 <- f68s_chain(lz, 10L)   # 30 pending nodes
  p40 <- f68s_chain(lz, 40L)   # 120 pending nodes
  expect_equal(collect(p40), collect(f68s_chain(tbl_gpu(f68s_df(1000L), lazy = FALSE), 40L)))
  t <- f68s_time(list(c10 = function() collect(p10), c40 = function() collect(p40)), samples = 5L)
  # 4x the nodes: linear work is ~4x; the quadratic schema re-inference was far more.
  expect_lt(t[["c40"]] / t[["c10"]], 6)
})

test_that("F62's 5-verb pipeline is no slower lazy than eager at 1e3 and 1e6 rows", {
  skip_if_no_gpu()
  for (n in c(1e3, 1e6)) {
    df <- f68s_df(n)
    eg <- tbl_gpu(df, lazy = FALSE)
    lz <- tbl_gpu(df, lazy = TRUE)
    expect_equal(collect(f68s_pipeline(lz)), collect(f68s_pipeline(eg)))
    t <- f68s_time(list(
      eager = function() collect(f68s_pipeline(eg)),
      lazy = function() collect(f68s_pipeline(lz))
    ), target = 0.15)  # long batches: proc.time()'s 1 ms clock is < 1% of a sample
    expect_lte(t[["lazy"]], t[["eager"]],
               label = sprintf("lazy %.3f ms at n = %g (eager %.3f ms)", 1e3 * t[["lazy"]], n,
                               1e3 * t[["eager"]]))
    rm(eg, lz)
    gc()
  }
})

test_that("the perf gate times lazy recording + collect of a 5-verb pipeline at 1e3 rows", {
  path <- test_path("test-perf-regression.R")
  found <- list()
  walk <- function(e) {
    if (is.call(e)) {
      if (identical(e[[1]], as.name("expect_no_perf_regression"))) {
        found[[length(found) + 1L]] <<- e
      }
      for (a in as.list(e)[-1]) if (!missing(a)) walk(a)
    }
  }
  for (e in parse(path, keep.source = FALSE)) walk(e)
  ids <- vapply(found, function(cl) as.character(cl[[2]]), character(1))
  expect_true("lazy_pipeline5_1e3" %in% ids)
  expect_false(anyDuplicated(ids) > 0L)
  cl <- found[[match("lazy_pipeline5_1e3", ids)]]
  expect_identical(cl$calibration, "kernel")
  # the timed closure records the plan inside the call (not a prebuilt plan)
  body_txt <- paste(deparse(cl[[3]]), collapse = " ")
  expect_match(body_txt, "collect")
  expect_match(body_txt, "summarise")
})
