# F21 tier B: native group_split(). Chunks are tbl_gpu tables (dplyr's
# dplyr_chop() keeps the class of the ungrouped input), built on the GPU
# without a CPU fallback.

f21n_df <- function() {
  tibble::tibble(
    x = 1:8,
    g = factor(c("b", "a", NA, "b", "a", "a", NA, "b"), levels = c("a", "b", "c")),
    s = c("B", "a", "a", NA, "B", "a", NA, "a"),
    h = c(2, 1, 2, 1, 2, 2, 1, 1),
    t = as.POSIXct("2024-01-01 10:00:00", tz = "America/New_York") + 0:7
  )
}

f21n_schema <- function(x) {
  s <- x$schema
  list(names = s$names, types = unname(s$types), levels = s$factor_levels, tz = s$tz)
}

test_that("group_split() runs on the GPU and returns tbl_gpu chunks and a tbl_gpu ptype", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f21n_df()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = FALSE)
    if (lazy) gt <- dplyr::mutate(as_lazy(gt), y = x * 2L)
    cases <- list(
      dplyr::group_split(gt, g),
      dplyr::group_split(gt, s, .keep = FALSE),
      dplyr::group_split(gt, g, .drop = FALSE),
      dplyr::group_split(dplyr::group_by(gt, h)),
      dplyr::group_split(dplyr::group_by(gt, g, .drop = FALSE), .keep = FALSE),
      dplyr::group_split(gt)
    )
    for (res in cases) {
      expect_s3_class(res, "vctrs_list_of")
      ptype <- attr(res, "ptype")
      expect_true(is_tbl_gpu(ptype))
      expect_identical(dim(ptype)[1], 0L)
      for (chunk in res) {
        expect_true(is_tbl_gpu(chunk))
        expect_identical(dplyr::group_vars(chunk), character())
        expect_identical(chunk$exec_mode, if (lazy) "lazy" else "eager")
        expect_identical(f21n_schema(chunk), f21n_schema(ptype))
      }
    }
  }
})

test_that("native group_split() chunks match dplyr's, empty groups and time zones included", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f21n_df()
  res <- dplyr::group_split(tbl_gpu(df), g, .drop = FALSE)
  oracle <- dplyr::group_split(df, g, .drop = FALSE)
  expect_identical(length(res), length(oracle))
  for (i in seq_along(oracle)) {
    expect_identical(dplyr::collect(res[[i]]), oracle[[i]])
  }
  expect_identical(dplyr::collect(attr(res, "ptype")), attr(oracle, "ptype"))
})

test_that("native group_split() chunks are independent GPU tables usable by verbs", {
  skip_if_no_gpu()
  set.seed(21)
  n <- 20000L
  df <- tibble::tibble(
    k = sample(200L, n, replace = TRUE),
    v = runif(n),
    w = sample(c(letters, NA), n, replace = TRUE)
  )
  res <- dplyr::group_split(tbl_gpu(df), k)
  oracle <- dplyr::group_split(df, k)
  expect_identical(length(res), length(oracle))
  expect_identical(
    vapply(res, function(ch) dim(ch)[1], integer(1)),
    vapply(oracle, nrow, integer(1))
  )
  sums <- vapply(res, function(ch) dplyr::pull(dplyr::summarise(ch, s = sum(v))), numeric(1))
  expect_equal(sums, vapply(oracle, function(ch) sum(ch$v), numeric(1)))
  expect_identical(
    dplyr::collect(cuplyr::bind_rows(res)),
    dplyr::bind_rows(oracle)
  )
  rm(res)
  gc()
})
