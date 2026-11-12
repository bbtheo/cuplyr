# F79: a logical vector in filter() needs the row count of a lazy input, so its
# pending steps run early; that is a plan break and is reported, once.

test_that("a vector next to pending steps reports one plan break and stays lazy", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:6, y = 6:1)
  m <- c(TRUE, FALSE, TRUE, TRUE, TRUE)
  breaks <- 0L
  res <- withCallingHandlers(
    dplyr::filter(dplyr::filter(tbl_gpu(df, lazy = TRUE), x > 1L), y > 2L, m),
    cuplyr_plan_break = function(cnd) {
      breaks <<- breaks + 1L
      expect_match(cnd$reason, "row count")
    }
  )
  expect_identical(breaks, 1L)
  expect_identical(.subset2(res, "exec_mode"), "lazy")
  expect_identical(collect(res), dplyr::filter(dplyr::filter(df, x > 1L), y > 2L, m))

  # nothing pending: no break
  breaks <- 0L
  withCallingHandlers(
    dplyr::filter(tbl_gpu(df, lazy = TRUE), c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE)),
    cuplyr_plan_break = function(cnd) breaks <<- breaks + 1L
  )
  expect_identical(breaks, 0L)
})

# How many plan breaks one call signals, and its result.
f79_breaks <- function(run) {
  breaks <- 0L
  res <- withCallingHandlers(run(), cuplyr_plan_break = function(cnd) breaks <<- breaks + 1L)
  list(breaks = breaks, res = res)
}

test_that("a vector, an integer sum() and cur_group_id() report one plan break at most", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:6, y = 6:1)
  m <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)
  m2 <- c(TRUE, FALSE, TRUE, TRUE, FALSE)
  pipe <- function(d) dplyr::filter(d, m, sum(x) > 0L, dplyr::cur_group_id() == 1L)
  pipe_nocg <- function(d) dplyr::filter(d, m, sum(x) > 0L)
  pending_cg <- function(d) dplyr::filter(dplyr::filter(d, x > 1L), m2, sum(x) > 0L, dplyr::cur_group_id() == 1L)
  pending_sum <- function(d) dplyr::filter(dplyr::filter(d, x > 1L), m2, sum(x) > 0L)

  # nothing pending: the helper column is the verb's own, so no break
  got <- f79_breaks(function() pipe(tbl_gpu(df, lazy = TRUE)))
  expect_identical(got$breaks, 0L)
  expect_identical(collect(got$res), pipe(df))
  got <- f79_breaks(function() pipe_nocg(tbl_gpu(df, lazy = TRUE)))
  expect_identical(got$breaks, 0L)
  expect_identical(collect(got$res), pipe_nocg(df))

  # one pending step: exactly one break, with or without the helper
  got <- f79_breaks(function() pending_cg(tbl_gpu(df, lazy = TRUE)))
  expect_identical(got$breaks, 1L)
  expect_identical(collect(got$res), pending_cg(df))
  got <- f79_breaks(function() pending_sum(tbl_gpu(df, lazy = TRUE)))
  expect_identical(got$breaks, 1L)
  expect_identical(collect(got$res), pending_sum(df))
})

test_that("both logical-vector routes signal the same fallback notice, once", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4)
  m <- c(TRUE, FALSE, TRUE, TRUE)
  m2 <- c(TRUE, TRUE, FALSE, TRUE)
  all_true <- rep(TRUE, 4L)
  for (lazy in c(FALSE, TRUE)) {
    notices <- function(run) {
      seen <- list()
      res <- withCallingHandlers(run(), cuplyr_cpu_fallback = function(cnd) seen[[length(seen) + 1L]] <<- cnd)
      collect(res)
      seen
    }
    gt <- tbl_gpu(df, lazy = lazy)
    # (mask only, then mixed, then a mixed call whose mask keeps every row)
    only <- notices(function() dplyr::filter(gt, m))
    expect_length(only, 1L)
    mixed <- notices(function() dplyr::filter(gt, x > 0L, m))
    expect_length(mixed, 1L)
    expect_identical(mixed[[1L]]$reason, only[[1L]]$reason)
    expect_match(conditionMessage(mixed[[1L]]), "Nothing was collected", fixed = TRUE)
    kept <- notices(function() dplyr::filter(gt, x > 0L, all_true))
    expect_length(kept, 1L)
    expect_identical(kept[[1L]]$reason, only[[1L]]$reason)
    expect_length(notices(function() dplyr::filter(gt, x > 0L, m, m2)), 1L)
    expect_length(notices(function() dplyr::filter(gt, m, x > mean(x))), 1L)
    # a native call and a scalar constant say nothing
    expect_length(notices(function() dplyr::filter(gt, x > 0L, TRUE)), 0L)

    withr::local_options(cuplyr.fallback = "error")
    expect_error(dplyr::filter(gt, m), class = "cuplyr_error_fallback")
    expect_error(dplyr::filter(gt, x > 0L, m), class = "cuplyr_error_fallback")
    expect_error(dplyr::filter(gt, x > 0L, all_true), class = "cuplyr_error_fallback")
    withr::local_options(cuplyr.fallback = "silent")
  }
})
