# F68b acceptance tests: the per-call R cost of one eager verb.
#
# At 1e3 rows a cuplyr verb is almost all R (parsing, validation, schema and
# metadata bookkeeping, wrapping): the GPU part of a filter()/mutate() is under
# 0.1 ms. These tests hold the eager verbs to the 0.5 ms target and F62's
# 5-verb pipeline below the times it had before (3.1 ms eager, 3.0 ms lazy at
# 1e3 rows; 3.25 / 3.13 ms at 1e6 on the reference machine).
#
# Timings are medians of interleaved, gpu_sync()-fenced batches
# (timing_medians(), helper-timing.R): each sample is a batch of calls lasting
# about `target` seconds, timed with the wall clock (proc.time() reads whole
# ms) and sized from a probe that lasts long enough to read. The bounds are
# unchanged. Batches are now ~50 ms as this method always documented (the old
# timer gave ~25 or ~250 ms); on the same code the 50 ms reading is ~2% below
# the old 250 ms mode (F106 round-1 review probe), so the 0.5 ms bound is read
# slightly more leniently than in the old 500-rep runs. A run that misses a bound by under 10% is measured again (up to 3
# attempts, 1 s apart; timing_retry()); a larger miss fails at once, and the
# failure label lists every attempt.

# every attempt's values, e.g. "0.512 / 0.498 ms"
f68b_attempts <- function(runs, v) {
  paste(sprintf("%.3f", vapply(runs, function(r) 1e3 * r[[v]], numeric(1))), collapse = " / ")
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
  fns <- list(
    filter = function() filter(eg, x > 0.5),
    mutate = function() mutate(eg, z = x * 2 + y),
    select = function() select(eg, g, x),
    arrange = function() arrange(eg, x),
    summarise = function() summarise(geg, m = mean(x))
  )
  runs <- timing_retry(
    function() timing_medians(fns, label = "F68b eager verbs 1e3"),
    ok = function(t) all(t < 0.5e-3),
    retry_if = function(t) all(t < 0.55e-3),
    pause = 1
  )
  t <- runs[[length(runs)]]
  for (v in names(t)) {
    expect_lt(t[[v]], 0.5e-3,
              label = sprintf("eager %s() at 1e3 rows: %s ms (attempts)", v, f68b_attempts(runs, v)))
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
    runs <- timing_retry(
      function() {
        timing_medians(list(
          eager = function() collect(f68b_pipeline(eg)),
          lazy = function() collect(f68b_pipeline(lz))
        ), samples = 11L, target = 0.15, label = sprintf("F68b pipeline n=%g", n))
      },
      ok = function(t) all(t < 2.8e-3),
      retry_if = function(t) all(t < 2.8e-3 * 1.1),
      pause = 1
    )
    t <- runs[[length(runs)]]
    # before: eager 3.10 / 3.25 ms, lazy 3.00 / 3.13 ms (1e3 / 1e6 rows)
    for (m in names(t)) {
      expect_lt(t[[m]], 2.8e-3,
                label = sprintf("%s pipeline at n = %g: %s ms (attempts)", m, n,
                                f68b_attempts(runs, m)))
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
