# Issue 166: reading a double column with NaN as missing (the comparison,
# is.na() and coalesce() operands, arrange keys, the rowwise na.rm fold, ...)
# builds only the NaN-as-null mask. libcudf 26.10 deprecates the mask-only
# nans_to_nulls(); its replacement column_nans_to_nulls() copies the whole column
# (8 B per row, thrown away) on every such read. The perf gate's shapes can't see
# that, so these tests count the bytes allocated at the allocator.

# Bytes allocated by `expr`, after one run to warm the kernels and caches.
n166_alloc <- function(expr) {
  expr <- substitute(expr)
  env <- parent.frame()
  gpu_alloc_tracking(eval(expr, env))
  gpu_alloc_tracking(eval(expr, env))$allocated_bytes
}

test_that("a comparison on a NaN-free double column allocates no more than on an INT32 column, per row", {
  skip_if_no_gpu()
  n <- 4e6
  set.seed(166)
  g <- tbl_gpu(data.frame(x = runif(n), y = runif(n),
                          i = sample.int(1000L, n, TRUE), j = sample.int(1000L, n, TRUE)),
               lazy = FALSE)
  # 2 B/row of slack: one 1-byte flag column and the mask are all the double
  # route may add. The copy it replaced added 8 B per operand.
  slack <- 2 * n

  one_d <- n166_alloc(filter(g, x > 2))
  one_i <- n166_alloc(filter(g, i > 5000L))
  expect_lte(one_d - one_i, slack, label = sprintf("filter(x > 2) %.1f MB vs INT32 %.1f MB", one_d / 1e6, one_i / 1e6))

  two_d <- n166_alloc(filter(g, x > 0.5, y < 0.5))
  two_i <- n166_alloc(filter(g, i > 500L, j < 500L))
  expect_lte(two_d - two_i, 2 * slack,
             label = sprintf("two-column filter %.1f MB vs INT32 %.1f MB", two_d / 1e6, two_i / 1e6))

  mut_d <- n166_alloc(mutate(g, b = x > 0.5))
  mut_i <- n166_alloc(mutate(g, b = i > 500L))
  expect_lte(mut_d - mut_i, slack, label = sprintf("mutate(b = x > 0.5) %.1f MB vs INT32 %.1f MB", mut_d / 1e6, mut_i / 1e6))
})

# x: neither / NaN only / NA only / both; the columns of one frame, so one
# pipeline checks them all, with a second column to filter on.
n166_df <- function(n = 40L) {
  set.seed(1661)
  x <- runif(n, -2, 2)
  list(
    neither = data.frame(x = x, k = seq_len(n)),
    nan = data.frame(x = replace(x, c(2, 9, 30), NaN), k = seq_len(n)),
    na = data.frame(x = replace(x, c(1, 10, 31), NA), k = seq_len(n)),
    both = data.frame(x = replace(replace(x, c(2, 9, 30), NaN), c(1, 10, 31, 40), NA), k = seq_len(n)),
    all_nan = data.frame(x = rep(NaN, 5L), k = 1:5),
    zero_rows = data.frame(x = double(), k = integer())
  )
}

test_that("a double column read as NaN-is-missing keeps nulls null and turns NaN to null, in every mix", {
  skip_if_no_gpu()
  for (nm in names(n166_df())) {
    df <- n166_df()[[nm]]
    info <- paste("x:", nm)
    expect_same_as_dplyr(df, function(d) dplyr::filter(d, x > 0), info = info)
    expect_same_as_dplyr_lazy(df, function(d) dplyr::filter(d, x > 0), info = info)
    expect_same_as_dplyr(df, function(d) dplyr::filter(d, x > -1, k > 3L), info = info)
    expect_same_as_dplyr(df, function(d) dplyr::mutate(d, b = x > 0, na = is.na(x), nn = is.nan(x), co = dplyr::coalesce(x, -9)),
                         info = info)
    expect_same_as_dplyr_lazy(df, function(d) dplyr::mutate(d, b = x > 0, na = is.na(x), co = dplyr::coalesce(x, -9)),
                              info = info)
    expect_same_as_dplyr(df, function(d) dplyr::filter(d, is.na(x)), info = info)
    expect_same_as_dplyr(df, function(d) dplyr::filter(d, !is.na(x), x < 1), info = info)
  }
})

# slice_head()/slice_tail() copy the rows they keep into a new table, so this
# reads columns that start at row 0; no verb hands nan_null_mask() a column view
# with an offset.
test_that("a double column cut by slice_head()/slice_tail() is read the same way", {
  skip_if_no_gpu()
  for (nm in c("nan", "na", "both", "neither")) {
    df <- n166_df()[[nm]]
    info <- paste("x:", nm)
    expect_same_as_dplyr(df, function(d) dplyr::filter(dplyr::slice_tail(d, n = 25L), x > 0), info = info)
    expect_same_as_dplyr(df, function(d) dplyr::mutate(dplyr::slice_tail(d, n = 25L), b = x > 0, na = is.na(x)), info = info)
    expect_same_as_dplyr_lazy(df, function(d) dplyr::mutate(dplyr::slice_tail(d, n = 25L), b = x > 0, na = is.na(x)), info = info)
    expect_same_as_dplyr(df, function(d) dplyr::filter(dplyr::slice_head(dplyr::slice_tail(d, n = 30L), n = 20L), x < 1), info = info)
  }
})

test_that("the NaN-as-null reads of slice_max()/slice_min(), arrange() and the rowwise na.rm fold agree with dplyr", {
  skip_if_no_gpu()
  for (nm in c("nan", "na", "both", "neither")) {
    df <- n166_df()[[nm]]
    info <- paste("x:", nm)
    expect_same_as_dplyr(df, function(d) dplyr::slice_max(d, x, n = 5L, with_ties = FALSE, na_rm = TRUE), info = info)
    expect_same_as_dplyr(df, function(d) dplyr::slice_min(d, x, n = 5L, with_ties = FALSE, na_rm = TRUE), info = info)
    d2 <- transform(df, y = rev(x))
    expect_same_as_dplyr(d2, function(d) dplyr::rowwise(d) |> dplyr::mutate(m = mean(c(x, y), na.rm = TRUE)) |> dplyr::ungroup(),
                         info = info)
    expect_same_as_dplyr(d2, function(d) dplyr::rowwise(d) |> dplyr::mutate(s = sum(x, y, na.rm = TRUE)) |> dplyr::ungroup(),
                         info = info)
  }
})
