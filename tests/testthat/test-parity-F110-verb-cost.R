# F110 acceptance: the per-call R cost of one verb at 1e3 rows, back near the
# level F68b left it at.
#
# Between F68b (80c3a81) and dev c56ffb6 one eager verb at 1e3 rows got ~20%
# slower (filter 0.34 -> 0.41 ms, mutate 0.38 -> 0.45, grouped summarise 0.38
# -> 0.45 on the reference machine), a few microseconds per merged
# correctness check (the bisect table is in the F110 brief). Recording a lazy
# verb got slower by the same amounts.
#
# Absolute floors fail under CPU load (F106), so each verb is timed against a
# plain dplyr call on a tibble of the same data, in the same interleaved
# rounds (timing_medians(), helper-timing.R). That cancels a slower machine
# but not a busy one: with other threads running (the F110 PM and reviewer
# measured 8, 24 and 48 spinning processes on 24 threads) a ratio moves +-10%
# or more between attempts, in both directions and stably within an attempt,
# so no statistic of one attempt's samples recovers the quiet value. Only a
# quiet attempt can judge a bound, so:
#   - an attempt is quiet when under `busy_limit` of all CPUs were busy during
#     it (/proc/stat; a quiet run reads about 5%). Where /proc/stat can't be
#     read every attempt counts as quiet;
#   - a verb is judged by its best ratio over the quiet attempts only;
#   - a miss is measured again (up to 5 attempts, 2 s apart), except that a
#     miss of over 10% on a quiet attempt fails at once;
#   - with no quiet attempt the test skips and says so: that run judged nothing.
#
# Bounds come from the context that judges them: a full tools/test, which runs
# every file in one R process. The verb is dearer there than in a fresh process
# under tools/test-file (summarise reads 1.09 fresh, 1.11-1.13 in the suite;
# filter 1.10 and 1.12-1.14). The F110 gate failure showed it; a top-level
# probe puts grouped summarise's ratio up 3-5% from data.table and lubridate
# being loaded or a heap 1.5x larger (the suite's is 2.3x, with both loaded).
# Measured ratios, the suite column over two full runs:
#   verb            reference   F68b (file)  dev (file)  feature file / suite   bound
#   filter          ref_filter  1.04         1.19-1.22   1.10 / 1.12-1.14       1.16
#   mutate          ref_mutate  1.02         1.19-1.21   1.09 / 1.08-1.10       1.12
#   summarise (g)   ref_mutate  1.03         1.20-1.22   1.09 / 1.11-1.13       1.16
#   select          ref_filter  0.31         0.32        0.32 / 0.31-0.32       0.345
#   arrange         ref_filter  0.45         0.45        0.46 / 0.46            0.49
#   floor5          ref_filter  4.72         5.22-5.30   4.91 / 4.87-4.97       5.10
#   lazy filter     ref_filter  0.47         0.63        0.54 / 0.52-0.53       0.645
#   lazy mutate     ref_mutate  0.56         0.72        0.62 / 0.61            0.745
#   lazy summ. (g)  ref_mutate  0.29         0.43        0.32 / 0.33            0.44
#   lazy5           ref_filter  5.13         5.70        5.55 / 5.52-5.61       5.87
# (dplyr's filter() and mutate() took 0.35 and 0.39 ms.) "file" is one fresh
# tools/test-file process; "suite" is a full tools/test. The filter, mutate,
# summarise and floor5 bounds stay below what dev reads in a fresh process.
# select and arrange did not regress; their bounds keep them there. `floor5` is the
# perf gate's eager_verb_floor_1e3 workload. The lazy rows (recording a verb;
# lazy5 also runs it) are a no-regression guard.

f110_df <- function(n) {
  set.seed(1)
  data.frame(g = sample.int(100L, n, TRUE), x = runif(n), y = runif(n), w = rnorm(n))
}

# every attempt's ratio for `v`, e.g. "1.091 / 1.074"
f110_attempts <- function(runs, v) {
  paste(sprintf("%.3f", vapply(runs, function(r) r$ratios[[v]], numeric(1))), collapse = " / ")
}

# Total and idle jiffies of all CPUs so far; NULL where /proc/stat can't be read.
f110_cpu_times <- function() {
  line <- tryCatch(readLines("/proc/stat", n = 1L), error = function(e) NULL, warning = function(w) NULL)
  if (is.null(line)) {
    return(NULL)
  }
  v <- as.numeric(strsplit(trimws(sub("^cpu", "", line)), " +")[[1L]])
  c(total = sum(v[1:8]), idle = v[[4L]] + v[[5L]])
}

# The fraction of all CPUs busy since `from` (a f110_cpu_times()); 0 when it
# can't be read, so such a machine counts as quiet and a miss fails.
f110_cpu_busy <- function(from) {
  now <- f110_cpu_times()
  if (is.null(from) || is.null(now) || now[["total"]] <= from[["total"]]) {
    return(0)
  }
  1 - (now[["idle"]] - from[["idle"]]) / (now[["total"]] - from[["total"]])
}

test_that("one verb at 1e3 rows costs about what it did after F68b, relative to dplyr", {
  skip_if_no_gpu()
  df <- f110_df(1000L)
  eg <- tbl_gpu(df, lazy = FALSE)
  geg <- group_by(eg, g)
  lz <- tbl_gpu(df, lazy = TRUE)
  glz <- group_by(lz, g)
  tb <- tibble::as_tibble(df)
  gtb <- group_by(tb, g)
  pipe5 <- function(d) {
    d |> filter(x > 0.1) |> mutate(z = x * y + w) |> group_by(g) |>
      summarise(m = mean(z), n = n()) |> arrange(desc(m))
  }

  # the timed calls give dplyr's results
  same <- function(a, b) expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(b))
  same(filter(eg, x > 0.5), filter(tb, x > 0.5))
  same(mutate(eg, z = x * 2 + y), mutate(tb, z = x * 2 + y))
  same(select(eg, g, x), select(tb, g, x))
  same(arrange(eg, x), arrange(tb, x))
  same(summarise(geg, m = mean(x)), summarise(gtb, m = mean(x)))
  same(filter(lz, x > 0.5), filter(tb, x > 0.5))
  same(mutate(lz, z = x * 2 + y), mutate(tb, z = x * 2 + y))
  same(summarise(glz, m = mean(x)), summarise(gtb, m = mean(x)))
  same(pipe5(lz), pipe5(tb))

  fns <- list(
    filter = function() filter(eg, x > 0.5),
    mutate = function() mutate(eg, z = x * 2 + y),
    select = function() select(eg, g, x),
    arrange = function() arrange(eg, x),
    summarise = function() summarise(geg, m = mean(x)),
    floor5 = function() {
      f <- filter(eg, x > 0.5)
      m <- mutate(eg, z = x * 2 + y)
      s <- select(eg, g, x)
      a <- arrange(eg, x)
      u <- summarise(geg, m = mean(x))
      gpu_sync()
      u
    },
    lazy_filter = function() filter(lz, x > 0.5),
    lazy_mutate = function() mutate(lz, z = x * 2 + y),
    lazy_summarise = function() summarise(glz, m = mean(x)),
    lazy5 = function() collect(pipe5(lz)),
    ref_filter = function() dplyr::filter(tb, x > 0.5),
    ref_mutate = function() dplyr::mutate(tb, z = x * 2 + y)
  )
  # verb -> c(reference closure, bound on the ratio)
  bounds <- list(
    filter = list("ref_filter", 1.16),
    mutate = list("ref_mutate", 1.12),
    summarise = list("ref_mutate", 1.16),
    select = list("ref_filter", 0.345),
    arrange = list("ref_filter", 0.49),
    floor5 = list("ref_filter", 5.10),
    lazy_filter = list("ref_filter", 0.645),
    lazy_mutate = list("ref_mutate", 0.745),
    lazy_summarise = list("ref_mutate", 0.44),
    lazy5 = list("ref_filter", 5.87)
  )
  lim <- vapply(bounds, function(b) b[[2]], numeric(1))
  ratios <- function(t) vapply(names(bounds), function(v) t[[v]] / t[[bounds[[v]][[1]]]], numeric(1))
  best <- lim * Inf
  busy_limit <- 0.15
  runs <- timing_retry(
    function() {
      from <- f110_cpu_times()
      r <- ratios(timing_medians(fns, samples = 21L, label = "F110 verb cost 1e3"))
      busy <- f110_cpu_busy(from)
      if (busy < busy_limit) best <<- pmin(best, r)
      list(ratios = r, busy = busy)
    },
    ok = function(run) all(best < lim),
    retry_if = function(run) run$busy >= busy_limit || all(best < lim * 1.1),
    attempts = 5L,
    pause = 2
  )
  busy <- vapply(runs, function(run) run$busy, numeric(1))
  busy_text <- paste(sprintf("%.0f%%", 100 * busy), collapse = " / ")
  if (all(busy >= busy_limit)) {
    skip(sprintf("no quiet attempt to judge the bounds: %s of all CPUs busy (attempts)", busy_text))
  }
  for (v in names(bounds)) {
    expect_lt(best[[v]], lim[[v]],
              label = sprintf("%s at 1e3 rows / dplyr's %s: best quiet of %s, CPUs busy %s (attempts)", v,
                              sub("ref_", "", bounds[[v]][[1]]), f110_attempts(runs, v), busy_text))
  }
})
