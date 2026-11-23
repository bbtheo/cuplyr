# F124 x #50: summarise(.by =) orders its groups by first appearance inside
# gpu_summarise(), after the hit masks of min()/max() are computed. The masks
# travel with the rows through that reorder, so the warning's "In group k:"
# names dplyr's first hit group in first-appearance order: neither the sorted
# key order nor the groupby's own (hash) order.

test_that("summarise(.by) names the first hit group in first-appearance order", {
  skip_if_no_gpu()
  withr::local_collate("C")
  set.seed(124)
  n_groups <- 5000L
  n <- 3e5
  # first appearances in a shuffled key order, with the smallest key late (the
  # 4000th group); every later row is random
  first <- sample(n_groups)
  first[c(which(first == 1L), 4000L)] <- first[c(4000L, which(first == 1L))]
  g <- c(first, sample(n_groups, n - n_groups, replace = TRUE))
  x <- sample(1000L, n, replace = TRUE)
  d <- runif(n)
  s <- sample(letters, n, replace = TRUE)
  # all-NA groups: the smallest key (late in first-appearance order), the 7th
  # group to appear and the 2500th
  na_keys <- c(1L, first[[7L]], first[[2500L]])
  hit <- g %in% na_keys
  x[hit] <- NA
  d[hit] <- NA
  s[hit] <- NA
  df <- tibble::tibble(g = g, x = x, d = d, s = s)

  f124_check(df, function(t) summarise(t, a = min(x, na.rm = TRUE), .by = g))
  f124_check(df, function(t) {
    summarise(t, n = n(), b = max(d, na.rm = TRUE), c = min(s, na.rm = TRUE), .by = g)
  })
})

test_that("summarise(.by) with several keys names the first hit group in first-appearance order", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(
    g = c("z", "a", "m", "a", "z", "b", "m"),
    h = c(2L, 1L, 3L, 1L, 2L, 1L, 3L),
    x = c(1, NA, NA, NA, 2, 5, NA)
  )
  f124_check(df, function(t) summarise(t, lo = min(x, na.rm = TRUE), hi = max(x, na.rm = TRUE), .by = c(g, h)))
})
