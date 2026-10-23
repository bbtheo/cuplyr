# F61 item 2: native group_split() at many groups.
#
# Contract (F21 / F47): chunks are independent tbl_gpu tables, one per group in
# dplyr's group order, identical to dplyr's chunks once collected; the ptype
# is a 0-row tbl_gpu. No CPU fallback (pinned with cuplyr.fallback = "error").
# These tests use thousands of groups, so they exercise whatever batched split
# replaces the per-group work.

f61s_df <- function(n = 30000L, G = 4000L, seed = 612L) {
  set.seed(seed)
  tibble::tibble(
    k = sample(c(seq_len(G), NA), n, replace = TRUE),
    x = runif(n),
    i = sample(c(1:100, NA), n, replace = TRUE),
    s = sample(c(letters, NA, ""), n, replace = TRUE),
    f = factor(sample(c("lo", "mid", "hi", NA), n, replace = TRUE),
               levels = c("lo", "mid", "hi", "unused")),
    b = sample(c(TRUE, FALSE, NA), n, replace = TRUE),
    dt = as.Date("2020-01-01") + sample(0:400, n, replace = TRUE),
    ts = as.POSIXct("2024-03-01 00:00:00", tz = "Europe/Helsinki") +
      sample(0:1e6, n, replace = TRUE)
  )
}

f61s_check <- function(res, oracle) {
  expect_s3_class(res, "vctrs_list_of")
  expect_identical(length(res), length(oracle))
  expect_true(all(vapply(res, is_tbl_gpu, logical(1))))
  got <- lapply(res, dplyr::collect)
  expect_identical(got, as.list(oracle))
  expect_identical(dplyr::collect(attr(res, "ptype")), attr(oracle, "ptype"))
}

test_that("group_split() with thousands of groups matches dplyr chunk for chunk", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f61s_df()
  gt <- tbl_gpu(df)
  f61s_check(dplyr::group_split(gt, k), dplyr::group_split(df, k))
  f61s_check(dplyr::group_split(gt, k, .keep = FALSE), dplyr::group_split(df, k, .keep = FALSE))
  f61s_check(dplyr::group_split(gt, k, s), dplyr::group_split(df, k, s))
  f61s_check(dplyr::group_split(dplyr::group_by(gt, s, f)),
             dplyr::group_split(dplyr::group_by(df, s, f)))
})

test_that("group_split() at many groups: .drop = FALSE, computed keys, lazy input", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f61s_df(n = 12000L, G = 1500L)
  gt <- tbl_gpu(df)
  f61s_check(dplyr::group_split(gt, f, k, .drop = FALSE),
             dplyr::group_split(df, f, k, .drop = FALSE))
  f61s_check(dplyr::group_split(gt, kk = k %/% 3L), dplyr::group_split(df, kk = k %/% 3L))
  lz <- dplyr::filter(as_lazy(gt), !is.na(i))
  res <- dplyr::group_split(lz, k)
  f61s_check(res, dplyr::group_split(dplyr::filter(df, !is.na(i)), k))
  expect_true(all(vapply(res, function(ch) identical(ch$exec_mode, "lazy"), logical(1))))
})

test_that("group_split() with one row per group and with a single group", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(k = sample(5000L), v = as.double(seq_len(5000L)),
                       s = paste0("r", seq_len(5000L)))
  gt <- tbl_gpu(df)
  f61s_check(dplyr::group_split(gt, k), dplyr::group_split(df, k))
  f61s_check(dplyr::group_split(gt, s, .keep = FALSE), dplyr::group_split(df, s, .keep = FALSE))
  one <- dplyr::mutate(df, one = 1L)
  f61s_check(dplyr::group_split(tbl_gpu(one), one), dplyr::group_split(one, one))
  f61s_check(dplyr::group_split(gt), dplyr::group_split(df))
  f61s_check(dplyr::group_split(tbl_gpu(df[0, ]), k), dplyr::group_split(df[0, ], k))
})

test_that("many-group chunks own their data: they outlive the input and each other", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f61s_df(n = 20000L, G = 3000L)
  oracle <- dplyr::group_split(df, k)
  gt <- tbl_gpu(df)
  res <- dplyr::group_split(gt, k)
  rm(gt)
  gc(full = TRUE)
  keep <- res[c(1L, 500L, length(res))]
  rm(res)
  gc(full = TRUE)
  expect_identical(lapply(keep, dplyr::collect),
                   as.list(oracle)[c(1L, 500L, length(oracle))])
  # usable by verbs and by bind_rows()
  sums <- vapply(keep, function(ch) dplyr::pull(dplyr::summarise(ch, s = sum(x))), numeric(1))
  expect_equal(sums, vapply(as.list(oracle)[c(1L, 500L, length(oracle))],
                            function(ch) sum(ch$x), numeric(1)))
  both <- dplyr::collect(cuplyr::bind_rows(keep[[1]], keep[[3]]))
  expect_identical(both, dplyr::bind_rows(oracle[[1]], oracle[[length(oracle)]]))
})

# --- timing -----------------------------------------------------------------

f61s_time <- function(f, reps = 5L) {
  f()
  cuplyr:::gpu_sync_device()
  t <- vapply(seq_len(reps), function(i) {
    gc(FALSE)
    s <- proc.time()[["elapsed"]]
    f()
    cuplyr:::gpu_sync_device()
    proc.time()[["elapsed"]] - s
  }, numeric(1))
  stats::median(t)
}

test_that("native group_split() at 10,000 groups on 1e6 rows reaches CPU parity", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  set.seed(613)
  n <- 1e6
  df <- tibble::tibble(g = sample.int(10000L, n, TRUE), x = runif(n), y = rnorm(n),
                       s = sample(letters, n, TRUE))
  gt <- tbl_gpu(df)
  gpu <- f61s_time(function() dplyr::group_split(gt, g))
  cpu <- f61s_time(function() dplyr::group_split(dplyr::collect(gt), g))
  expect_lte(gpu, cpu)
})
