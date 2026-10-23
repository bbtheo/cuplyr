# F61 item 4: the ungrouped summarise() builds its constant group key once per
# call on the device instead of a host vector per aggregation. Results must
# not change: every aggregation kind of the ungrouped path, NA handling,
# na.rm, one-row and zero-row tables, eager and lazy.

f61m_both <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

f61m_df <- function(n = 5000L, seed = 614L) {
  set.seed(seed)
  data.frame(
    x = c(runif(n - 3L), NA, 0.5, -1),
    i = c(sample.int(1000L, n - 1L, TRUE), NA),
    l = sample(c(TRUE, FALSE, NA), n, TRUE),
    s = sample(c(letters, NA), n, TRUE),
    stringsAsFactors = FALSE
  )
}

test_that("ungrouped summarise() with many aggregations of every kind matches dplyr", {
  skip_if_no_gpu()
  f61m_both(f61m_df(), function(t) {
    dplyr::summarise(t,
      n = dplyr::n(),
      sx = sum(x), sxr = sum(x, na.rm = TRUE), mx = mean(x, na.rm = TRUE),
      mn = min(i, na.rm = TRUE), mxi = max(i, na.rm = TRUE), si = sum(i, na.rm = TRUE),
      sd = sd(x, na.rm = TRUE), v = var(x, na.rm = TRUE),
      med = median(x, na.rm = TRUE), q = quantile(x, 0.25, na.rm = TRUE, names = FALSE),
      a = any(l), al = all(l, na.rm = TRUE),
      nd = dplyr::n_distinct(s), f = dplyr::first(s), la = dplyr::last(x)
    )
  })
})

test_that("ungrouped summarise() NA propagation and identities without na.rm", {
  skip_if_no_gpu()
  f61m_both(f61m_df(), function(t) {
    dplyr::summarise(t, sx = sum(x), mi = min(i), ma = max(x), me = mean(i),
                     a = any(l), al = all(l), med = median(x))
  })
  all_na <- data.frame(x = c(NA_real_, NA_real_), i = c(NA_integer_, NA_integer_))
  f61m_both(all_na, function(t) {
    dplyr::summarise(t, sx = sum(x, na.rm = TRUE), mx = mean(x, na.rm = TRUE),
                     si = sum(i, na.rm = TRUE), n = dplyr::n(), s2 = sum(x))
  })
})

test_that("ungrouped summarise() on one row and on zero rows", {
  skip_if_no_gpu()
  one <- f61m_df()[2, ]
  f61m_both(one, function(t) {
    dplyr::summarise(t, n = dplyr::n(), sx = sum(x), sd = sd(x), med = median(x),
                     q = quantile(x, 0.5, names = FALSE), mi = min(i), a = any(l, na.rm = TRUE))
  })
  zero <- f61m_df()[0, ]
  f61m_both(zero, function(t) {
    dplyr::summarise(t, n = dplyr::n(), sx = sum(x), si = sum(i), a = any(l), al = all(l))
  })
})

test_that("ungrouped summarise() over expressions and after a lazy filter", {
  skip_if_no_gpu()
  # Not pinned native: a lazy summarise() over a temp expression materializes
  # first (an existing, notified route).
  pipeline <- function(t) {
    t |>
      dplyr::filter(!is.na(i), i > 10L) |>
      dplyr::summarise(n = dplyr::n(), big = sum(x > 0.5, na.rm = TRUE),
                       m = mean(x * 2, na.rm = TRUE), s = sum(i * 2L),
                       q = quantile(i, 0.9, names = FALSE))
  }
  expect_same_as_dplyr(f61m_df(), pipeline)
  expect_same_as_dplyr_lazy(f61m_df(), pipeline)
})
