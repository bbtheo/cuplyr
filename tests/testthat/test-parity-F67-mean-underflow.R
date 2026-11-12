# F67 attempt 2: R's mean() divides its sum by n, so a tiny negative sum can
# round to -0 (mean(c(-5e-324, 0, 0, 0)) is -0 in R), while a group of only
# -0 values has a +0 sum and so a +0 mean. Both signs must come through, in
# summarise() and the window mean() of mutate(), grouped and not, eager and
# lazy. f67_check() (helper-parity-F67.R) compares every zero's sign. (A
# window mean(x, na.rm = TRUE) runs on the CPU today, so mutate() cases use
# mean(x) only.)

# Tests see the package namespace; bind dplyr's first() as library(dplyr) would.
first <- getExportedValue("dplyr", "first")

test_that("a mean() that underflows to zero keeps R's -0 (summarise)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(-5e-324, 0, 0, 0), y = c(-5e-324, NA, 0, 0))
  f67_check(df, function(d) {
    summarise(d, m = mean(x), r = mean(y, na.rm = TRUE), s = sum(x))
  })
  f67_check(df, function(d) summarise(d, m = mean(x)))
  # Half of the smallest subnormal: a tie that rounds to even, -0.
  f67_check(data.frame(x = c(-5e-324, 0)), function(d) summarise(d, m = mean(x)))
  # Positive underflow is +0; a mean that doesn't underflow is unchanged.
  f67_check(data.frame(x = c(5e-324, 0, 0, 0)), function(d) summarise(d, m = mean(x)))
  f67_check(data.frame(x = c(-1e-323, 0, 0)), function(d) summarise(d, m = mean(x)))
})

test_that("a mean() that underflows to zero keeps R's -0 (ungrouped mutate)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(-5e-324, 0, 0, 0), y = c(-5e-324, NA, 0, 0))
  f67_check(df, function(d) mutate(d, m = mean(x), my = mean(y)))
})

test_that("grouped mean() gives -0 for an underflow and +0 for all -0 values", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c(1L, 1L, 1L, 1L, 2L, 2L, 3L, 4L, 4L, 5L, 5L),
    x = c(-5e-324, 0, 0, 0, 1, 2, -0, -5e-324, NA, -0, NA)
  )
  f67_check(df, function(d) d |> group_by(g) |> summarise(m = mean(x)))
  f67_check(df, function(d) d |> group_by(g) |> summarise(m = mean(x, na.rm = TRUE), s = sum(x, na.rm = TRUE)))
  f67_check(df, function(d) d |> group_by(g) |> mutate(m = mean(x)))
  # first() puts the whole summarise() on cudf's sort-based groupby, whose
  # SUM of an all -0 group is -0 (the hash groupby's starts from +0).
  f67_check(df, function(d) d |> group_by(g) |> summarise(m = mean(x), s = sum(x), f = first(x)))
  f67_check(df, function(d) {
    d |> group_by(g) |> summarise(m = mean(x, na.rm = TRUE), s = sum(x, na.rm = TRUE), f = first(x))
  })
})

test_that("many groups: underflowing means stay -0, all -0 groups become +0", {
  skip_if_no_gpu()
  # More than 65536 groups, so a per-group result this long takes the
  # searched path of the zero-sign fix as well (the window mutate, and the
  # summarise with first(), which runs on the sort-based groupby).
  n_g <- 70000L
  g <- rep(seq_len(n_g), each = 2L)
  x <- rep(c(-5e-324, 0), n_g)
  x[g %% 3L == 0L] <- -0
  x[g %% 7L == 0L] <- c(1, 2)
  df <- data.frame(g = g, x = x)
  f67_check(df, function(d) d |> group_by(g) |> summarise(m = mean(x)))
  f67_check(df, function(d) d |> group_by(g) |> summarise(m = mean(x), s = sum(x), f = first(x)))
  f67_check(df, function(d) d |> group_by(g) |> mutate(m = mean(x)))
})
