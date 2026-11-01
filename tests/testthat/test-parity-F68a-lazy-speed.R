# F68a acceptance tests: lazy verb calls are cheap.
#
# Before F68a, recording a verb on a lazy table re-inferred the schema of the
# whole pending AST (current_schema() -> infer_schema_full(), itself calling
# the recursive infer_schema() at every level), and lowering/optimizing
# re-inferred each node's input schema the same way: both grow with the square
# of the plan's depth. On F62's 5-verb pipeline that made lazy ~1 ms slower
# than eager at 1e3 and 1e6 rows (F62/measure/summary.txt).
#
# Timings are medians of interleaved, gpu_sync()-fenced batches
# (timing_medians(), helper-timing.R), and the checks are ratios on the same
# machine, so they don't depend on the GPU's absolute speed. Batches are timed
# with the wall clock (proc.time() reads whole ms) and sized from a probe that
# lasts long enough to read. A ratio over its bound is measured again only
# when it is under 2x that bound (3 for 1.5, 12 for 6; up to 3 attempts,
# timing_retry()); a real regression reads the same all three times and fails.
# The bounds are unchanged, and the failure label lists every attempt.

# every attempt's ratio num/den, e.g. "1.62 / 1.12"
f68s_ratios <- function(runs, num, den) {
  paste(sprintf("%.2f", vapply(runs, function(r) r[[num]] / r[[den]], numeric(1))), collapse = " / ")
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
  ratios <- function(t) {
    c(filter = t[["filter_deep"]] / t[["filter_shallow"]],
      mutate = t[["mutate_deep"]] / t[["mutate_shallow"]],
      summarise = t[["summarise_deep"]] / t[["summarise_shallow"]])
  }
  # Before F68a the deep plan made each call ~40-100x slower.
  runs <- timing_retry(
    function() {
      timing_medians(list(
        filter_shallow = function() filter(shallow, y > 0.5),
        filter_deep = function() filter(deep, y > 0.5),
        mutate_shallow = function() mutate(shallow, q = x * y),
        mutate_deep = function() mutate(deep, q = x * y),
        summarise_shallow = function() summarise(group_by(shallow, g), m = mean(x)),
        summarise_deep = function() summarise(group_by(deep, g), m = mean(x))
      ), samples = 11L, target = 0.03, label = "F68a recording deep/shallow")
    },
    ok = function(t) all(ratios(t) < 1.5),
    retry_if = function(t) all(ratios(t) < 3)
  )
  t <- runs[[length(runs)]]
  for (v in c("filter", "mutate", "summarise")) {
    expect_lt(ratios(t)[[v]], 1.5,
              label = sprintf("%s() deep/shallow: %s (attempts)", v,
                              f68s_ratios(runs, paste0(v, "_deep"), paste0(v, "_shallow"))))
  }
})

test_that("collecting a plan costs time linear in its depth", {
  skip_if_no_gpu()
  lz <- tbl_gpu(f68s_df(1000L), lazy = TRUE)
  p10 <- f68s_chain(lz, 10L)   # 30 pending nodes
  p40 <- f68s_chain(lz, 40L)   # 120 pending nodes
  expect_equal(collect(p40), collect(f68s_chain(tbl_gpu(f68s_df(1000L), lazy = FALSE), 40L)))
  # 4x the nodes: linear work is ~4x; the quadratic schema re-inference was far more.
  runs <- timing_retry(
    function() {
      timing_medians(list(c10 = function() collect(p10), c40 = function() collect(p40)),
                     samples = 11L, target = 0.03, label = "F68a collect c40/c10")
    },
    ok = function(t) t[["c40"]] / t[["c10"]] < 6,
    retry_if = function(t) t[["c40"]] / t[["c10"]] < 12
  )
  t <- runs[[length(runs)]]
  expect_lt(t[["c40"]] / t[["c10"]], 6,
            label = sprintf("collect c40/c10: %s (attempts)", f68s_ratios(runs, "c40", "c10")))
})

test_that("F62's 5-verb pipeline is no slower lazy than eager at 1e3 and 1e6 rows", {
  skip_if_no_gpu()
  for (n in c(1e3, 1e6)) {
    df <- f68s_df(n)
    eg <- tbl_gpu(df, lazy = FALSE)
    lz <- tbl_gpu(df, lazy = TRUE)
    expect_equal(collect(f68s_pipeline(lz)), collect(f68s_pipeline(eg)))
    runs <- timing_retry(
      function() {
        timing_medians(list(
          eager = function() collect(f68s_pipeline(eg)),
          lazy = function() collect(f68s_pipeline(lz))
        ), samples = 11L, target = 0.15, label = sprintf("F68a lazy vs eager n=%g", n))
      },
      ok = function(t) t[["lazy"]] <= t[["eager"]],
      retry_if = function(t) t[["lazy"]] / t[["eager"]] < 1.1
    )
    t <- runs[[length(runs)]]
    expect_lte(t[["lazy"]], t[["eager"]],
               label = sprintf("lazy %.3f ms at n = %g (eager %.3f ms); lazy/eager over attempts: %s",
                               1e3 * t[["lazy"]], n, 1e3 * t[["eager"]],
                               f68s_ratios(runs, "lazy", "eager")))
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
