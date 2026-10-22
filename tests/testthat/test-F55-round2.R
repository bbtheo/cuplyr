# F55 round 2: extra checks beside the protected acceptance tests.
#
# - Computed FLOAT64 operands of comparisons/is.na() that can only be NaN where
#   their columns are NaN stay inline, reading the columns with NaN as null
#   (nan_only_from_leaves()); anything that can make a NaN out of finite or
#   infinite inputs (x / y, Inf - Inf, 0 * Inf, sqrt(-1), ...) is materialized.
# - na_if() / window aggregates / consecutive_id() with Inf and NaN together.

f55r2_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L, 3L, 3L),
    x = c(1, NaN, 2, Inf, -Inf, NA, 0, 5),
    y = c(Inf, 1, -Inf, 2, Inf, 3, 0, NaN),
    i = c(1L, 2L, NA, 4L, 5L, 6L, 7L, 8L)
  )
}

f55r2_same <- function(df, pipeline, nan_skip = character()) {
  for (lazy in c(FALSE, TRUE)) {
    res <- if (lazy) {
      expect_same_as_dplyr_lazy(df, pipeline)
    } else {
      expect_same_as_dplyr(df, pipeline)
    }
    for (nm in setdiff(names(res$oracle), nan_skip)) {
      if (is.double(res$oracle[[nm]])) {
        expect_identical(is.nan(res$gpu[[nm]]), is.nan(res$oracle[[nm]]),
                         info = sprintf("is.nan(%s), %s", nm, if (lazy) "lazy" else "eager"))
      }
    }
  }
}

test_that("F55 r2: comparisons of NaN-safe computed operands read the columns as missing", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f55r2_df()
  f55r2_same(df, function(d) {
    dplyr::mutate(d,
      a = abs(x) > 1, b = -x < 0, c = x + 1 >= 3, d = 2 - x <= 0, e = x * 2 != 4,
      f = floor(x) == 1, g2 = ceiling(x) > 0, h = exp(x) > 1, k = abs(x - 1) > 0.5,
      m = is.na(abs(x)), n = is.na(x * 3), o = is.na(-x + 1)
    )
  })
  f55r2_same(df, function(d) dplyr::filter(d, abs(x) > 1))
  f55r2_same(df, function(d) dplyr::filter(d, !(x + 1 > 2)))
})

test_that("F55 r2: aggregates over NaN-safe computed operands", {
  skip_if_no_gpu()
  # No native pin: a lazy summarise() with a temp column reports its early
  # compute() through the fallback notice (see the acceptance file).
  f55r2_same(f55r2_df(), function(d) {
    dplyr::summarise(d, n = sum(is.na(abs(x))), m = sum(x * 2 > 1, na.rm = TRUE), .by = g)
  })
})

test_that("F55 r2: operands that can make a NaN from non-NaN inputs are still checked", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f55r2_df()
  # Inf - Inf, 0 * Inf, Inf / Inf, sqrt(-1), 0 / 0 are NaN made of non-NaN values.
  suppressWarnings(f55r2_same(df, function(d) {
    dplyr::mutate(d,
      a = x - y > 0, b = x * y > 0, c = x / y > 1, d = is.na(x - y), e = is.na(x * y),
      f = is.na(x / y), g2 = sqrt(x - 3) > 0, h = is.na(sqrt(x - 3)), k = x * 0 > 0, m = is.na(x * 0),
      n = (x + y) * 2 > 1, o = is.na(sin(x))
    )
  }))
  f55r2_same(df, function(d) dplyr::filter(d, x - y > -Inf))
})

test_that("F55 r2: a NaN-free column and a mixed expression keep working", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(a = c(1, 2, 3), b = c(0.5, NA, 4), i = 1:3)
  f55r2_same(df, function(d) dplyr::mutate(d, p = abs(a) > 1 & b < 3, q = i * 2.5 > 3, r = -a + 1 > b))
  f55r2_same(df, function(d) dplyr::filter(d, abs(a) > 1, b * 2 > 1))
})

test_that("F55 r2: na_if() keeps NaN apart from NA and Inf", {
  skip_if_no_gpu()
  df <- tibble::tibble(w = c(NaN, Inf, NA, -Inf, 2), y = c(Inf, Inf, NaN, NaN, NA))
  f55r2_same(df, function(d) dplyr::mutate(d, a = dplyr::na_if(w, Inf), b = dplyr::na_if(w, y), c = dplyr::na_if(w, NA)))
  f55r2_same(df, function(d) dplyr::mutate(d, a = dplyr::na_if(w, NaN), b = dplyr::na_if(y, w)))
})

test_that("F55 r2: window aggregates and consecutive_id() with NaN, NA and Inf", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L, 3L, 3L), w = c(Inf, NaN, -Inf, NA, 1, 2, 3))
  f55r2_same(df, function(d) {
    dplyr::mutate(d, mx = max(w), md = median(w), s = sd(w), mn = mean(w), sm = sum(w), .by = g)
  }, nan_skip = "mx")  # max() of a NaN group: NA here, NaN in R (F22's divergence)
  f55r2_same(df, function(d) dplyr::mutate(d, id = dplyr::consecutive_id(w)))
  f55r2_same(
    tibble::tibble(w = c(Inf, Inf, NaN, NaN, NA, NA, NaN, 1, 1)),
    function(d) dplyr::mutate(d, id = dplyr::consecutive_id(w))
  )
})
