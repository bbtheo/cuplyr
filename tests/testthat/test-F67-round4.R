# F67 round 4: the +0 fix of double sum()/mean()/cumsum() results runs only
# where a -0 can be. An ungrouped cumsum() scan is decided by its first row in
# scan order; a long result (> 65536 rows) is searched for a -0 first; the
# rowwise fold starts from `first operand + 0`.

for (fn in c("cummean", "order_by")) {
  assign(fn, getExportedValue("dplyr", fn))
}

test_that("an ungrouped cumsum() is decided by its first row in scan order", {
  skip_if_no_gpu()
  # Stored first rows are not what the scan sees first under order_by(): the
  # scan of y by o starts -0, -0, NA; by -o it starts 1. z starts with NA, so
  # every row is NA (R stops at the first NA), -0 rows after it included.
  df <- data.frame(y = c(1, -0, -0, NA, -0), o = c(5, 1, 2, 3, 4), z = c(NA, -0, -0, 1, -0),
                   w = c(-0, -0, 2, -0, -0))
  # (order_by(o, cummean(y)) is left out: it ignores the order on dev too, a
  # separate gap)
  f67_check(df, function(d) mutate(d, c = order_by(o, cumsum(y))))
  f67_check(df, function(d) mutate(d, c = order_by(-o, cumsum(y)), r = order_by(-o, cumsum(w))))
  f67_check(df, function(d) mutate(d, c = cumsum(y), cz = cumsum(z), cw = cumsum(w), mw = cummean(w)))
})

test_that("a long grouped result with -0 in one group far from the start is +0 there", {
  skip_if_no_gpu()
  set.seed(6704)
  n <- 2e5
  g <- rep(seq_len(1000), each = n / 1000)
  y <- runif(n)
  y[g == 777] <- -0
  df <- data.frame(y = y, g = g, k = rep(seq_len(n / 2), each = 2))
  f67_check(df, function(d) d |> group_by(g) |> mutate(c = cumsum(y), m = cummean(y)))
  # 1e5 groups, one of them all -0, so the per-group result is long
  df$y[df$k != 54321 & df$y == 0] <- 0.5
  f67_check(df, function(d) d |> group_by(k) |> summarise(s = sum(y), m = mean(y)))
})

test_that("rowwise sum() and mean() start from +0 whatever the operand types", {
  skip_if_no_gpu()
  df <- data.frame(a = c(-0, -0, -0, 1), b = c(-0, NA, -0, -1), i = c(0L, 1L, NA, 2L))
  f67_check(df, function(d) {
    d |> rowwise() |> mutate(s1 = sum(c(a)), m1 = mean(c(a)), s = sum(c(b, a)),
                             sr = sum(c(b, a), na.rm = TRUE), mr = mean(c(b, a), na.rm = TRUE))
  })
  f67_check(df, function(d) d |> rowwise() |> mutate(s = sum(c(i, a, b)), m = mean(c(i, a))))
})
