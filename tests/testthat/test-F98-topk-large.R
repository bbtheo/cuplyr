# F98: the top-k route on tables big enough to have real cuts, ties across the
# cut, -0/+0, NaN and Inf, against dplyr.

test_that("top-k route matches dplyr at a few thousand rows", {
  skip_if_no_gpu()
  set.seed(9801)
  n <- 5000L
  x <- sample(c(-Inf, Inf, -0, 0, NaN, NA, round(rnorm(40), 1)), n, replace = TRUE)
  df <- tibble::tibble(
    id = seq_len(n), x = x,
    i = sample(c(NA, -5:5), n, replace = TRUE),
    t = as.POSIXct("2024-01-01", tz = "UTC") + sample(c(NA, 0:30), n, replace = TRUE)
  )
  pipes <- list(
    function(d) dplyr::slice_min(d, x, n = 25),
    function(d) dplyr::slice_max(d, x, n = 25, with_ties = FALSE),
    function(d) dplyr::slice_min(d, x, n = 300, na_rm = TRUE),
    function(d) dplyr::slice_max(d, i, n = 100),
    function(d) dplyr::slice_min(d, t, n = 50, with_ties = FALSE),
    function(d) dplyr::slice_max(d, t, prop = 0.05),
    function(d) head(dplyr::arrange(d, x), 40),
    function(d) head(dplyr::arrange(d, dplyr::desc(x)), 40),
    function(d) head(dplyr::arrange(d, dplyr::desc(i)), 300)
  )
  for (p in c("auto", "topk", "sort")) {
    withr::local_options(cuplyr.topk_path = p)
    for (pipe in pipes) {
      expect_same_as_dplyr(df, pipe)
      expect_same_as_dplyr_lazy(df, pipe)
    }
  }
})

test_that("a FLOAT32 key takes the top-k route and orders like its doubles", {
  skip_if_no_gpu()
  set.seed(9802)
  # exactly representable in single precision
  df <- tibble::tibble(x = sample(c(-0, 0, NA, -2:2 / 4, -Inf, Inf), 3000, replace = TRUE))
  ptr <- gpu_cast_column(.subset2(tbl_gpu(df), "ptr"), 0L, "FLOAT32")
  for (desc in c(FALSE, TRUE)) {
    for (ties in c(FALSE, TRUE)) {
      top <- gpu_slice_rank(ptr, 0L, desc, 20, FALSE, ties, FALSE, "topk")
      expect_true(isTRUE(attr(top, "topk")))
      srt <- gpu_slice_rank(ptr, 0L, desc, 20, FALSE, ties, FALSE, "sort")
      expect_identical(gpu_dim(top), gpu_dim(srt))
      expect_equal(gpu_collect(top, "x"), gpu_collect(srt, "x"))
    }
  }
})

test_that("a cut at exactly the number of valid rows matches dplyr", {
  skip_if_no_gpu()
  set.seed(9803)
  df <- tibble::tibble(i = 1:4000,
                       x = ifelse(runif(4000) < 0.3, NA, sample(1:50, 4000, replace = TRUE)))
  n_valid <- sum(!is.na(df$x))
  for (p in c("auto", "topk", "sort")) {
    withr::local_options(cuplyr.topk_path = p)
    for (ties in c(FALSE, TRUE)) {
      expect_same_as_dplyr(df, function(d) dplyr::slice_min(d, x, n = n_valid, with_ties = ties))
      expect_same_as_dplyr(df, function(d) dplyr::slice_max(d, x, n = n_valid, with_ties = ties))
    }
  }
})

test_that("a cut inside a block of -0/+0 values matches dplyr on every route", {
  skip_if_no_gpu()
  set.seed(9804)
  n <- 6000
  df <- tibble::tibble(i = seq_len(n),
                       x = sample(c(-0, 0, -1, 1, 2, NA), n, replace = TRUE,
                                  prob = c(0.3, 0.3, 0.1, 0.1, 0.1, 0.1)))
  for (p in c("topk", "sort")) {
    withr::local_options(cuplyr.topk_path = p)
    for (k in c(1, 50, 900, 1500)) {
      for (ties in c(FALSE, TRUE)) {
        expect_same_as_dplyr(df, function(d) dplyr::slice_min(d, x, n = k, with_ties = ties))
        expect_same_as_dplyr(df, function(d) dplyr::slice_max(d, x, n = k, with_ties = ties))
      }
    }
  }
})

test_that("the route marker does not outlive a lazy pipeline", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.topk_path = "topk")
  df <- tibble::tibble(x = c(3, 1, 2, 5, 4, 9, 8, 7))
  res <- tbl_gpu(df, lazy = TRUE) |> dplyr::arrange(x) |> dplyr::slice_min(x, n = 2) |> compute()
  expect_null(attr(.subset2(res, "ptr"), "topk"))
})
