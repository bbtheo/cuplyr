# F61: group_split() chunks are views into packed buffers that one
# contiguous_split allocated, one buffer per chunk (src/gpu_table.hpp,
# src/ops_slice.cpp). Every reader must see an ordinary table, whatever
# happens first to a chunk.

f61x_df <- function(n = 6000L, G = 700L) {
  set.seed(619)
  tibble::tibble(
    k = sample(seq_len(G), n, replace = TRUE),
    x = runif(n),
    s = sample(c(letters, NA), n, replace = TRUE)
  )
}

test_that("a chunk works as input to every kind of verb, before any other access", {
  skip_if_no_gpu()
  df <- f61x_df()
  oracle <- dplyr::group_split(df, k)
  res <- dplyr::group_split(tbl_gpu(df), k)
  # a different first access per chunk
  for (i in seq_along(res)) {
    ch <- res[[i]]
    got <- switch(i %% 6L + 1L,
      dplyr::collect(ch),
      dplyr::collect(dplyr::mutate(ch, y = x * 2)),
      dplyr::collect(dplyr::arrange(ch, x)),
      dplyr::collect(dplyr::summarise(ch, m = max(x))),
      dplyr::collect(dplyr::inner_join(ch, ch[0, ], by = "k")),
      dplyr::collect(dplyr::bind_rows(ch, ch)))
    exp <- oracle[[i]]
    got_exp <- switch(i %% 6L + 1L,
      exp,
      dplyr::mutate(exp, y = x * 2),
      dplyr::arrange(exp, x),
      dplyr::summarise(exp, m = max(x)),
      dplyr::inner_join(exp, exp[0, ], by = "k"),
      dplyr::bind_rows(exp, exp))
    expect_identical(got, got_exp)
  }
})

test_that("dim() of a chunk and a repeated collect agree", {
  skip_if_no_gpu()
  df <- f61x_df()
  oracle <- dplyr::group_split(df, k)
  res <- dplyr::group_split(tbl_gpu(df), k)
  expect_identical(lapply(res, dim), lapply(oracle, dim))
  expect_identical(dplyr::collect(res[[3]]), oracle[[3]])
  expect_identical(dplyr::collect(res[[3]]), oracle[[3]])
  expect_identical(lapply(res, nrow), lapply(oracle, nrow))
})

test_that("chunks are independent of the source and of each other once those are dropped", {
  skip_if_no_gpu()
  df <- f61x_df()
  oracle <- dplyr::group_split(df, k)
  gt <- tbl_gpu(df)
  res <- dplyr::group_split(gt, k)
  first <- dplyr::collect(res[[1]])
  rm(gt)
  gpu_gc()
  res[c(1, 5, 9)] <- list(NULL)
  gpu_gc()
  rest <- lapply(res[-c(1, 5, 9)], dplyr::collect)
  expect_identical(first, oracle[[1]])
  expect_identical(rest, as.list(oracle)[-c(1, 5, 9)])
})

test_that("an ungrouped table splits into one chunk equal to itself", {
  skip_if_no_gpu()
  df <- f61x_df(n = 50L)
  res <- dplyr::group_split(tbl_gpu(df))
  expect_length(res, 1L)
  expect_identical(dplyr::collect(res[[1]]), df)
})
