# #21, third round: the cases around the acceptance tests of
# test-parity-21-round3.R (rebuilt `.drop = FALSE` keys deeper than two keys,
# constant names on rebuilt keys, quantile() names over empty groups, the join
# sides a constant name stays native on).

test_that("rebuilt .drop = FALSE keys take the names of the first group with their value", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor(c("u", "v", "u", "u"), levels = c("u", "v", "w")),
                      b = stats::setNames(c(2L, 1L, 1L, 2L), c("x", "y", "z", "t")),
                      g = factor(c("a", "a", "b", "a"), levels = c("a", "b", "c")),
                      v = 1:4)
  # every row of one key named alike, under a factor key with an unused level
  dc <- tibble::tibble(f = factor(c("u", "v"), levels = c("u", "v", "w")),
                       b = stats::setNames(c(1L, 1L), c("k", "k")), v = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    for (keys in list(c("f", "b", "g"), c("b", "f"), c("f", "b"))) {
      expect_equal(group_keys(group_by(tbl_gpu(d, lazy = lazy), dplyr::across(dplyr::all_of(keys)), .drop = FALSE)),
                   group_keys(group_by(d, dplyr::across(dplyr::all_of(keys)), .drop = FALSE)),
                   info = paste(info, paste(keys, collapse = ",")))
    }
    expect_equal(group_keys(group_by(tbl_gpu(dc, lazy = lazy), f, b, .drop = FALSE)),
                 group_keys(group_by(dc, f, b, .drop = FALSE)), info = info)
    expect_equal(collect(summarise(group_by(tbl_gpu(dc, lazy = lazy), f, b, .drop = FALSE), n = n(), .groups = "drop")),
                 summarise(group_by(dc, f, b, .drop = FALSE), n = n(), .groups = "drop"), info = info)
  }
})

test_that("quantile() keeps its name over the empty groups of a .drop = FALSE summarise", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = factor(c("a", "a", "b"), levels = c("a", "b", "c")),
                      s = structure(c("p", "q", "p"), label = "S"), x = c(1, 2, 3))
  pipe <- function(t) summarise(group_by(t, g, .drop = FALSE), q = quantile(x, 0.5))
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_same_as_dplyr(d, pipe)
    expect_same_as_dplyr_lazy(d, pipe)
    expect_same_as_dplyr(d, function(t) summarise(group_by(t, g, s, .drop = FALSE), n = n(), .groups = "drop"))
  })
})

test_that("a constant name stays on the GPU in right and cross joins where every row is kept", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1, 2), a = c(3, 4))
  d <- tibble::tibble(k = c(1, 1, 3), x = c(1, 2, 3))
  s <- function(t) summarise(group_by(t, k), q = quantile(x, 0.5))
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    gx <- tbl_gpu(x, lazy = lazy)
    gd <- tbl_gpu(d, lazy = lazy)
    withr::with_options(list(cuplyr.fallback = "error"), {
      expect_equal(collect(arrange(right_join(gx, s(gd), by = "k"), k)),
                   arrange(right_join(x, s(d), by = "k"), k), info = info)
      expect_equal(collect(arrange(right_join(gx, s(gd), by = "k", keep = TRUE), k.y)),
                   arrange(right_join(x, s(d), by = "k", keep = TRUE), k.y), info = info)
      expect_equal(collect(cross_join(s(gd), gx)), cross_join(s(d), x), info = info)
      expect_equal(collect(cross_join(gx, s(gd))), cross_join(x, s(d)), info = info)
      expect_equal(collect(inner_join(s(gd), gx, by = "k", keep = TRUE)),
                   inner_join(s(d), x, by = "k", keep = TRUE), info = info)
    })
    # x's constant name on rows a right join adds: dplyr's names
    expect_equal(collect(arrange(right_join(s(gd), gx, by = "k"), k)),
                 arrange(right_join(s(d), x, by = "k"), k), info = info)
  }
})

test_that("a column named quantile is no quantile() call", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2), quantile = c(1, 2, 3))
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_same_as_dplyr(d, function(t) summarise(t, m = mean(quantile) + 1, .by = g))
  })
})

test_that("summarise() of a plain data.frame keeps a key's other attributes and drops its constant name", {
  skip_if_no_gpu()
  df <- structure(list(h = structure(c(1, 1, 2), label = "H", names = c("n", "n", "n")), x = c(1, 2, 3)),
                  class = "data.frame", row.names = c(NA, -3L))
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(summarise(tbl_gpu(df, lazy = lazy), m = mean(x), .by = h))
    expect_equal(as.list(got), as.list(summarise(df, m = mean(x), .by = h)), info = paste("lazy =", lazy))
    expect_s3_class(got, "data.frame", exact = TRUE)
  }
})
