# F61 item 3: faster checked integer arithmetic must keep F42's contract.
#
# Base R (?Arithmetic, verified with dplyr 1.2.1 on tibbles): integer +, -, *
# whose exact result leaves +-.Machine$integer.max is NA with one warning
# "NAs produced by integer overflow"; NA operands give NA without a warning.
# These tests target shortcuts a faster check could take: deciding "no
# overflow possible" from column ranges, from literals, or from the final
# result only. Each case has a row-level answer that a range-based or
# result-only check gets wrong if it is too coarse.

M <- .Machine$integer.max

f61i_parity <- function(df, pipeline, warn = NA) {
  for (fun in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    r <- suppressWarnings(fun(df, pipeline))
    expect_identical(lapply(r$gpu, class), lapply(r$oracle, class))
  }
  gt <- tbl_gpu(df)
  if (isTRUE(warn)) {
    expect_warning(dplyr::collect(pipeline(gt)), "NAs produced by integer overflow")
    expect_warning(dplyr::collect(pipeline(as_lazy(gt))), "NAs produced by integer overflow")
  } else if (identical(warn, FALSE)) {
    expect_no_warning(dplyr::collect(pipeline(gt)))
    expect_no_warning(dplyr::collect(pipeline(as_lazy(gt))))
  }
}

gpu_only <- function() withr::local_options(cuplyr.fallback = "error", .local_envir = parent.frame())

test_that("column ranges that could overflow, rows that don't: no NA, no warning", {
  skip_if_no_gpu()
  gpu_only()
  # max(x) + max(y) and min(x) * max(y) leave the range, but no single row does.
  df <- data.frame(x = c(M, -M, 5L, 0L), y = c(-M, M, -5L, 0L))
  f61i_parity(df, function(d) dplyr::mutate(d, a = x + y, s = x - x, m = (x + y) * 1000L),
              warn = FALSE)
  big <- data.frame(x = c(65536L, 1L, -65536L), y = c(1L, 32767L, 32767L))
  f61i_parity(big, function(d) dplyr::mutate(d, m = x * y), warn = FALSE)
})

test_that("an inner node overflows while the final result is back in range", {
  skip_if_no_gpu()
  gpu_only()
  df <- data.frame(x = c(M, 3L, -M, NA), y = c(2L, 4L, 2L, 1L))
  # (x * y) %/% y and x + x - x are in range only if the inner node wraps.
  f61i_parity(df, function(d) dplyr::mutate(d, a = x * y - x * y, b = x + x - x,
                                            c = (x + 1L) - 1L), warn = TRUE)
})

test_that("only one row of many overflows, at either end of the table", {
  skip_if_no_gpu()
  gpu_only()
  n <- 200000L
  x <- rep(7L, n)
  x[1L] <- M
  x[n] <- -M
  df <- data.frame(x = x, y = rep(1L, n))
  f61i_parity(df, function(d) dplyr::mutate(d, a = x + y, b = x - y, c = x * 2L), warn = TRUE)
  f61i_parity(df, function(d) dplyr::summarise(d, n_na = sum(is.na(x + y)), s = sum(x %/% 7L)))
})

test_that("rows nulled by an earlier overflow don't leak wrapped values", {
  skip_if_no_gpu()
  gpu_only()
  df <- data.frame(x = c(M, 2L, -M, 4L), y = c(M, 3L, M, NA))
  f61i_parity(df, function(d) {
    d |>
      dplyr::mutate(p = x * y) |>
      dplyr::mutate(q = p + 1L, r = p * 0L, s = dplyr::coalesce(p, -1L) + 1L)
  }, warn = TRUE)
})

test_that("all-NA and zero-row integer columns: no warning, integer type", {
  skip_if_no_gpu()
  gpu_only()
  na <- data.frame(x = rep(NA_integer_, 5), y = c(1L, M, -M, 0L, 2L))
  f61i_parity(na, function(d) dplyr::mutate(d, a = x + y, b = x * y, c = y - x), warn = FALSE)
  f61i_parity(na[0, ], function(d) dplyr::mutate(d, a = x + y, b = x * 2L), warn = FALSE)
})

test_that("literal-only bounds: boundary literals in range and just out of range", {
  skip_if_no_gpu()
  gpu_only()
  df <- data.frame(x = c(0L, 1L, -1L, NA))
  f61i_parity(df, function(d) dplyr::mutate(d, a = x + (M - 1L), b = x - (M - 1L)),
              warn = FALSE)
  f61i_parity(df, function(d) dplyr::mutate(d, a = x + M), warn = TRUE)
  f61i_parity(df, function(d) dplyr::mutate(d, a = x * M, b = x * -M), warn = FALSE)
})

test_that("grouped mutate, filter and logical operands keep the checked semantics", {
  skip_if_no_gpu()
  gpu_only()
  df <- data.frame(g = c(1L, 1L, 2L, 2L, 3L), x = c(M, 1L, -M, 5L, NA),
                   l = c(TRUE, FALSE, TRUE, NA, TRUE))
  f61i_parity(df, function(d) dplyr::mutate(dplyr::group_by(d, g), a = x + l, b = x * 2L),
              warn = TRUE)
  f61i_parity(df, function(d) dplyr::filter(d, x + l > 0L))
})
