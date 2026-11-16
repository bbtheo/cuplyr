# #50: summarise(.by =) on inputs of 2^17 rows and more takes the path that
# runs the groupby as for group_by() and then looks up each group's first row
# (src/ops_groupby.cpp, first_appearance_keys()). The result must still be
# dplyr's first-appearance order, whichever way the first rows are found:
#  - shuffled keys: a prefix of the rows shows every group;
#  - a group that only appears at the very end: the prefix misses it;
#  - sorted or clustered keys: the first rows are the run starts;
#  - nearly all keys distinct: the result is aligned by sorting and a lookup,
#    not by a hash join of the first-appearance keys.
# test-parity-50-by-order.R covers every key type under 2^17 rows.

by50l_n <- 131072L + 4001L

by50l_expect <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

# a large test leaves R's heap and string cache bigger than it found them, which
# the timing tests that run after it (test-parity-F110-verb-cost.R) notice
by50l_release <- function(...) {
  rm(list = ls(envir = parent.frame()), envir = parent.frame())
  invisible(gc())
}

by50l_pick <- function(v, n = by50l_n) sample(v, n, replace = TRUE)

test_that("a large shuffled summarise(.by) keeps first-appearance order for every key type", {
  skip_if_no_gpu()
  set.seed(5011)
  n <- by50l_n
  df <- data.frame(
    g = by50l_pick(c("m", "c", "x", "a", NA)),
    k = by50l_pick(c(7L, -2L, 3L, NA, 11L)),
    d = by50l_pick(c(2.5, NaN, -NaN, NA, -0, 0, -1.25, Inf)),
    f = factor(by50l_pick(c("lo", "hi", "mid", NA)), levels = c("mid", "unused", "hi", "lo")),
    dt = as.Date("2024-03-01") + by50l_pick(c(9L, 0L, NA, 3L)),
    ts = as.POSIXct("2024-03-01 12:00:00", tz = "UTC") + by50l_pick(c(3600, 0, NA, 60)),
    b = by50l_pick(c(TRUE, FALSE, NA)),
    x = ifelse(runif(n) < 0.1, NA, round(rnorm(n), 3)),
    y = sample.int(50L, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  for (key in c("g", "k", "d", "f", "dt", "ts", "b")) {
    by50l_expect(df, function(d) {
      dplyr::summarise(d, n = dplyr::n(), s = sum(y), m = mean(x, na.rm = TRUE),
                       .by = dplyr::all_of(key))
    })
  }
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(y), .by = c(g, d)))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(f, k, b)))
  by50l_release()
})

test_that("a large summarise(.by) still reads min/max/mean of every column type", {
  skip_if_no_gpu()
  set.seed(5012)
  n <- by50l_n
  df <- data.frame(
    g = by50l_pick(c("p", "q", "r", NA)),
    flag = by50l_pick(c(TRUE, FALSE, NA)),
    z = by50l_pick(c(1.5, NaN, NA, -2, 8)),
    dt = as.Date("2024-03-01") + by50l_pick(c(9L, 0L, NA, 3L)),
    i = by50l_pick(c(4L, NA, -9L, 12L)),
    stringsAsFactors = FALSE
  )
  by50l_expect(df, function(d) {
    dplyr::summarise(d, lo = min(flag, na.rm = TRUE), hi = max(z, na.rm = TRUE),
                     md = max(dt, na.rm = TRUE), mz = mean(z, na.rm = TRUE),
                     mi = median(i, na.rm = TRUE), n = dplyr::n(), .by = g)
  })
  by50l_release()
})

test_that("a large summarise(.by) finds groups that first appear late", {
  skip_if_no_gpu()
  set.seed(5013)
  n <- by50l_n
  # the first rows hold two groups; the others show up at the very end, the NA
  # group in the last row
  s <- by50l_pick(c("early1", "early2"))
  s[n - 3L] <- "late1"
  s[n - 2L] <- "late2"
  s[n - 1L] <- "late1"
  s[n] <- NA
  k <- by50l_pick(c(1L, 2L))
  k[(n - 3L):n] <- c(30L, NA, 20L, 10L)
  df <- data.frame(s = s, k = k, x = runif(n), stringsAsFactors = FALSE)
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = s))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = k))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(s, k)))
  by50l_release()
})

test_that("a large summarise(.by) on sorted or clustered keys keeps first-appearance order", {
  skip_if_no_gpu()
  set.seed(5014)
  n <- by50l_n
  # sorted ascending (NA last, NaN before it) and descending
  d_keys <- c(-1.5, 0, 2.5, Inf, NaN, NA)
  sorted <- data.frame(
    s = sort(by50l_pick(c("b", "a", "d", "c", NA)), na.last = TRUE),
    k = sort(by50l_pick(c(5L, 3L, NA, 9L, 1L)), decreasing = TRUE, na.last = TRUE),
    d = sort(by50l_pick(d_keys), na.last = TRUE),
    x = runif(n),
    stringsAsFactors = FALSE
  )
  for (key in c("s", "k", "d")) {
    by50l_expect(sorted, function(d) {
      dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = dplyr::all_of(key))
    })
  }
  by50l_expect(sorted, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(s, k)))

  # runs of a few thousand rows that revisit earlier groups: the first rows are
  # only some of the run starts
  runs <- rep(rep(c("r3", "r1", "r3", "r2", NA, "r1", "r4", "r2"), each = 2000L), length.out = n)
  df <- data.frame(s = runs, k = rep(c(2L, 2L, 1L, NA, 1L, 3L, 3L, 2L), each = 1999L, length.out = n),
                   x = runif(n), stringsAsFactors = FALSE)
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s2 = sum(x), .by = s))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = k))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(s, k)))
  by50l_release()
})

test_that("a large summarise(.by) with many groups keeps first-appearance order", {
  skip_if_no_gpu()
  set.seed(5015)
  n <- by50l_n
  pool <- sprintf("key-%09d", sample.int(n))                    # the only strings made here
  df <- data.frame(
    u = sample.int(n),                                           # every key distinct
    su = sample(pool),                                           # every string distinct
    q = sample.int(n %/% 4L, n, replace = TRUE),                 # about n/4 groups
    sq = pool[sample.int(n %/% 4L, n, replace = TRUE)],
    h = sample.int(40L, n, replace = TRUE),
    x = runif(n),
    stringsAsFactors = FALSE
  )
  for (key in c("u", "su", "q", "sq")) {
    by50l_expect(df, function(d) {
      dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = dplyr::all_of(key))
    })
  }
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(sq, h)))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(h, q)))
  by50l_release()
})

test_that("summarise(.by) around the 2^17-row switch keeps first-appearance order", {
  skip_if_no_gpu()
  set.seed(5016)
  for (n in c(131071L, 131072L, 131073L)) {
    df <- data.frame(
      g = sample(c("z", "y", NA, "x", "w"), n, replace = TRUE),
      k = sample.int(300L, n, replace = TRUE),
      x = runif(n),
      stringsAsFactors = FALSE
    )
    by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = g))
    by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(g, k)))
  }
  by50l_release()
})

test_that("a large summarise(.by) with a single group, or only NA keys, is one row", {
  skip_if_no_gpu()
  n <- by50l_n
  df <- data.frame(g = rep("only", n), na = rep(NA_character_, n), x = seq_len(n) * 1.5,
                   stringsAsFactors = FALSE)
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = g))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = na))
  by50l_release()
})
