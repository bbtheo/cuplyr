# F68b acceptance tests: the per-call R cost of one eager verb.
#
# At 1e3 rows a cuplyr verb is almost all R (parsing, validation, schema and
# metadata bookkeeping, wrapping): the GPU part of a filter()/mutate() is under
# 0.1 ms. These tests hold the eager verbs to the 0.5 ms target and F62's
# 5-verb pipeline below the times it had before (3.1 ms eager, 3.0 ms lazy at
# 1e3 rows; 3.25 / 3.13 ms at 1e6 on the reference machine).
#
# Timings are medians of interleaved, gpu_sync()-fenced batches (the method of
# F62's measure.R): each sample is a batch of calls lasting >= `target`
# seconds, with an untimed gc() before it.

f68b_time <- function(fns, samples = 15L, target = 0.05) {
  for (fn in fns) for (i in 1:5) fn()
  reps <- vapply(fns, function(fn) {
    gpu_sync(); t0 <- proc.time()[["elapsed"]]; fn(); gpu_sync()
    one <- max(proc.time()[["elapsed"]] - t0, 1e-4)
    max(1L, min(1000L, as.integer(ceiling(target / one))))
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

f68b_df <- function(n) {
  set.seed(1)
  data.frame(g = sample.int(100L, n, TRUE), x = runif(n), y = runif(n), w = rnorm(n))
}

f68b_pipeline <- function(d) {
  d |>
    filter(x > 0.1) |>
    mutate(z = x * y + w) |>
    group_by(g) |>
    summarise(m = mean(z), n = n()) |>
    arrange(desc(m))
}

test_that("one eager verb on 1e3 rows takes under 0.5 ms", {
  skip_if_no_gpu()
  df <- f68b_df(1000L)
  eg <- tbl_gpu(df, lazy = FALSE)
  geg <- group_by(eg, g)
  tb <- tibble::as_tibble(df)
  same <- function(a, b) expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(b))
  # the timed calls give dplyr's results
  same(filter(eg, x > 0.5), filter(tb, x > 0.5))
  same(mutate(eg, z = x * 2 + y), mutate(tb, z = x * 2 + y))
  same(select(eg, g, x), select(tb, g, x))
  same(arrange(eg, x), arrange(tb, x))
  same(summarise(geg, m = mean(x)), summarise(group_by(tb, g), m = mean(x)))
  t <- f68b_time(list(
    filter = function() filter(eg, x > 0.5),
    mutate = function() mutate(eg, z = x * 2 + y),
    select = function() select(eg, g, x),
    arrange = function() arrange(eg, x),
    summarise = function() summarise(geg, m = mean(x))
  ))
  for (v in names(t)) {
    expect_lt(t[[v]], 0.5e-3, label = sprintf("eager %s() at 1e3 rows: %.3f ms", v, 1e3 * t[[v]]))
  }
})

test_that("F62's 5-verb pipeline is faster than before in both modes", {
  skip_if_no_gpu()
  for (n in c(1e3, 1e6)) {
    df <- f68b_df(n)
    eg <- tbl_gpu(df, lazy = FALSE)
    lz <- tbl_gpu(df, lazy = TRUE)
    ref <- f68b_pipeline(tibble::as_tibble(df))
    expect_equal(tibble::as_tibble(collect(f68b_pipeline(eg))), ref)
    expect_equal(tibble::as_tibble(collect(f68b_pipeline(lz))), ref)
    t <- f68b_time(list(
      eager = function() collect(f68b_pipeline(eg)),
      lazy = function() collect(f68b_pipeline(lz))
    ), samples = 11L, target = 0.15)
    # before: eager 3.10 / 3.25 ms, lazy 3.00 / 3.13 ms (1e3 / 1e6 rows)
    for (m in names(t)) {
      expect_lt(t[[m]], 2.8e-3,
                label = sprintf("%s pipeline at n = %g: %.3f ms", m, n, 1e3 * t[[m]]))
    }
    rm(eg, lz)
    gc()
  }
})

test_that("the perf gate times single eager verbs at 1e3 rows", {
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
  expect_true("eager_verb_floor_1e3" %in% ids)
  expect_false(anyDuplicated(ids) > 0L)
  skip_if_not("eager_verb_floor_1e3" %in% ids)
  cl <- found[[match("eager_verb_floor_1e3", ids)]]
  expect_identical(cl$calibration, "kernel")
  body_txt <- paste(deparse(cl[[3]]), collapse = " ")
  for (verb in c("filter", "mutate", "select", "arrange", "summarise")) {
    expect_match(body_txt, paste0("\\b", verb, "\\("), info = verb)
  }
  # the R-side floor: the closure does not copy results back to R
  expect_no_match(body_txt, "collect")
})
