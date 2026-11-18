# #50: summarise(.by =) puts each group's first row in the one groupby as a MIN
# over the row position and sorts the k-row result by it
# (src/ops_groupby.cpp, group_order = "first"). These inputs of 2^17 rows and
# more must still give dplyr's first-appearance order:
#  - shuffled keys of every type;
#  - a group that first appears late (in a later part of the table);
#  - sorted or clustered keys that revisit groups;
#  - many groups, up to n/2.
# test-parity-50-by-order.R covers the small cases.

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

test_that("a large summarise(.by) on clustered keys that revisit groups keeps first-appearance order", {
  skip_if_no_gpu()
  # 20 runs of 50000 rows, groups revisited between runs. NA, NaN, -NaN, -0
  # and 0 keys among them.
  n <- 1e6
  each <- 50000L
  lab <- c("r3", "r1", "r3", "r2", NA, "r1", "r4", "r2", "r5", "r3",
           "r6", "r1", NA, "r7", "r2", "r8", "r3", "r9", "r1", "r10")
  dv <- c(0, -0, NaN, -NaN, 2.5, NA, 2.5, 0, Inf, NaN,
          -Inf, NA, -0, 7, NaN, 0, 7, -Inf, 2.5, -NaN)
  iv <- c(5L, 9L, 5L, NA, 1L, 9L, 3L, 1L, NA, 5L,
          8L, 3L, 2L, 9L, 6L, 1L, 4L, 8L, 7L, 5L)
  set.seed(5017)
  df <- data.frame(
    cs = rep(lab, each = each),
    cd = rep(dv, each = each),
    ci = rep(iv, each = each),
    cj = rep(rep(c(2L, 4L, 2L, 6L, NA), length.out = 34L), each = 30000L, length.out = n),
    x = runif(n),
    stringsAsFactors = FALSE
  )
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = cs))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = cd))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = ci))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = c(cs, cj)))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(cd, ci, cs)))
  by50l_release()
})

test_that("a large summarise(.by) finds a one-row group in the middle and one in the last row", {
  skip_if_no_gpu()
  # 4e6 shuffled rows of two groups: "mid" at row 300000 and "end" in the last
  # row are the only rows of their groups
  n <- 4e6
  set.seed(5018)
  g <- sample(c("a", "b"), n, replace = TRUE)
  mid <- g
  mid[300000] <- "mid"
  both <- mid
  both[n] <- "end"
  df <- data.frame(mid = mid, both = both, x = runif(n), stringsAsFactors = FALSE)
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = mid))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), m = mean(x), .by = both))
  by50l_release()
})

test_that("a large summarise(.by) with n/2 groups keeps first-appearance order", {
  skip_if_no_gpu()
  # n/2 groups; doubles with NaN, -NaN, -0, 0 and NA among many distinct values
  n <- 1e6
  set.seed(5019)
  special <- c(NaN, -NaN, -0, 0, NA)
  d <- round(runif(n) * 3e5) / 7
  d[sample.int(n, 5000L)] <- sample(special, 5000L, replace = TRUE)
  df <- data.frame(
    s2 = sprintf("key-%09d", sample(rep(seq_len(n / 2), 2))),
    d = d,
    x = runif(n),
    stringsAsFactors = FALSE
  )
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = s2))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = d))
  by50l_release()
})

test_that("a large summarise(.by) on short runs finds groups that first appear late", {
  skip_if_no_gpu()
  # 2e5 runs of 10 rows over 999 labels (and NA); "mid" first appears in run
  # 150000, "late" only in the last run. Crossed with a shuffled INT32 the keys
  # change almost every row.
  n <- 2e6
  set.seed(5305)
  nr <- n / 10
  rl <- sprintf("g%03d", sample.int(999L, nr, replace = TRUE))
  rl[sample.int(nr, 50L)] <- NA
  rl[150000L] <- "mid"
  rl[nr] <- "late"
  ip <- sample.int(2L, n, replace = TRUE)
  ip[sample.int(n, n / 100)] <- NA
  df <- data.frame(rs = rep(rl, each = 10L), ip = ip, x = runif(n), stringsAsFactors = FALSE)
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(x), .by = rs))
  by50l_expect(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(rs, ip)))
  by50l_release()
})
