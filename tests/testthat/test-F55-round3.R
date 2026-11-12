# F55 round 3: roll_*() shapes beside the protected B7 acceptance test.
#
# The rolling family reads a floating value column through its NaN-nulled copy
# (one per value column per frame), so R/roll.R's is.na()/anyNA() treatment of
# NaN holds for every function and both na.rm settings.

f55r3_same <- function(df, pipeline) {
  for (lazy in c(FALSE, TRUE)) {
    res <- if (lazy) {
      expect_same_as_dplyr_lazy(df, pipeline)
    } else {
      expect_same_as_dplyr(df, pipeline)
    }
    for (nm in names(res$oracle)) {
      if (is.double(res$oracle[[nm]])) {
        expect_identical(is.nan(res$gpu[[nm]]), is.nan(res$oracle[[nm]]),
                         info = sprintf("is.nan(%s), %s", nm, if (lazy) "lazy" else "eager"))
      }
    }
  }
}

test_that("F55 r3: every roll_*() over NaN, Inf and NA, both na.rm settings", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(x = c(NaN, NaN, 2, Inf, NaN, -Inf, NA, 5, NaN, 1))
  f55r3_same(df, function(d) {
    dplyr::mutate(d,
      m0 = roll_mean(x, 2), m1 = roll_mean(x, 3, na.rm = TRUE),
      s0 = roll_sum(x, 2), s1 = roll_sum(x, 3, na.rm = TRUE),
      lo0 = roll_min(x, 2), lo1 = roll_min(x, 2, na.rm = TRUE),
      hi0 = roll_max(x, 3), hi1 = roll_max(x, 3, na.rm = TRUE),
      sd0 = roll_sd(x, 2), sd1 = roll_sd(x, 3, na.rm = TRUE)
    )
  })
})

test_that("F55 r3: roll_*() leave a NaN-free double and an integer column as they were", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L),
    x = c(1.5, NA, 3, 4, 5, NA),
    i = c(1L, 2L, NA, 4L, 5L, 6L)
  )
  f55r3_same(df, function(d) {
    dplyr::mutate(d,
      a = roll_mean(x, 2, na.rm = TRUE), b = roll_max(x, 2),
      c = roll_sum(i, 2), e = roll_min(i, 2, na.rm = TRUE),
      .by = g
    )
  })
})

test_that("F55 r3: roll_sd() of a window with fewer than two values is NA, not NaN", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  # No NaN in the input: R's sd() of one value is NA, where cudf's STD
  # (ddof = 1) gives NaN; is.nan() tells them apart.
  df <- tibble::tibble(
    g = c(1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L),
    x = c(NA, NA, 2, 7, NA, 3, NA, 5, NA, 1)
  )
  f55r3_same(df, function(d) {
    dplyr::mutate(d,
      a = roll_sd(x, 3, na.rm = TRUE), b = roll_sd(x, 2, na.rm = TRUE),
      c = roll_sd(x, 1), e = roll_sd(x, 1, na.rm = TRUE)
    )
  })
  f55r3_same(df, function(d) dplyr::mutate(d, a = roll_sd(x, 2, na.rm = TRUE), .by = g))
})
