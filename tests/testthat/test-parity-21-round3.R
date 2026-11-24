# Acceptance tests for #21, added after the second review:
# - a plain data.frame: dplyr's summarise() and reframe() turn their result into
#   a data.frame with as.data.frame(), which drops element names (and keeps the
#   other attributes); the oracle here is dplyr run on the data.frame itself;
# - `.drop = FALSE` keys: dplyr's compute_groups() rebuilds the keys only when a
#   key is a factor; a rebuilt factor key has levels and class only, a rebuilt
#   other key keeps its attributes and first-occurrence names;
# - a constant name (quantile()'s "50%") goes through joins on the GPU where
#   every output row of the column comes from a matched row.

# dplyr on `df` as given (a plain data.frame stays one) against tbl_gpu(df),
# eager and lazy, column by column with their attributes.
p21r3_same_df <- function(df, pipeline) {
  expected <- as.list(dplyr::ungroup(pipeline(df)))
  for (lazy in c(FALSE, TRUE)) {
    got <- as.list(dplyr::ungroup(collect(pipeline(tbl_gpu(df, lazy = lazy)))))
    expect_equal(got, expected, info = paste("lazy =", lazy))
  }
}

p21r3_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

p21r3_native <- function(code) {
  withr::with_options(list(cuplyr.fallback = "error"), code)
}

test_that("summarise() and reframe() on a plain data.frame drop element names, as dplyr does", {
  skip_if_no_gpu()
  df <- data.frame(h = c(1, 1, 2), x = c(1, 2, 3))
  df$lab <- structure(c(5, 6, 7), label = "L")
  p21r3_same_df(df, function(t) summarise(t, q = quantile(x, 0.5), .by = h))
  p21r3_same_df(df, function(t) summarise(t, q = quantile(x, 0.5)))
  p21r3_same_df(df, function(t) summarise(t, q = quantile(x, 0.5) + 0, .by = h))
  p21r3_same_df(df, function(t) summarise(t, m = mean(x), .by = lab))
  p21r3_same_df(df, function(t) reframe(t, r = range(x), .by = h))
  # grouped: group_by() makes a tibble, which keeps them
  p21r3_same_df(df, function(t) summarise(group_by(t, h), q = quantile(x, 0.5)))
  # a named column of a plain data.frame
  # (built directly: data.frame() and `$<-` drop a column's element names)
  dn <- structure(list(g = c(1, 1, 2), b = stats::setNames(c(3L, 1L, 2L), c("x", "y", "z"))),
                  class = "data.frame", row.names = c(NA, -3L))
  expect_named(dn$b, c("x", "y", "z"))
  p21r3_same_df(dn, function(t) t)
  p21r3_same_df(dn, function(t) filter(t, b > 1L))
  p21r3_same_df(dn, function(t) arrange(t, b))
  p21r3_same_df(dn, function(t) mutate(t, k = 1))
  p21r3_same_df(dn, function(t) summarise(t, n = n(), .by = b))
  p21r3_same_df(dn, function(t) summarise(group_by(t, b), n = n()))
  p21r3_same_df(dn, function(t) reframe(t, n = n(), .by = b))
})

test_that(".drop = FALSE keys follow dplyr's compute_groups()", {
  skip_if_no_gpu()
  # every level observed, and one unused
  df <- tibble::tibble(f = structure(factor(c("u", "v", "u")), label = "F"), x = 1:3)
  df2 <- tibble::tibble(f = structure(factor(c("u", "v", "u"), levels = c("u", "v", "w")), label = "F"),
                        x = 1:3)
  # no factor key: .drop = FALSE changes nothing
  kn <- tibble::tibble(b = stats::setNames(c(1L, 2L, 1L), c("x", "y", "z")),
                       s = structure(c("p", "q", "p"), label = "S"), v = 1:3)
  # a factor key next to a non-factor key
  mix <- tibble::tibble(f = structure(factor(c("u", "v", "u"), levels = c("u", "v", "w")), label = "F"),
                        s = structure(c("p", "q", "p"), label = "S"),
                        b = stats::setNames(c(1L, 2L, 1L), c("x", "y", "z")), v = 1:3)
  for (d in list(df, df2)) {
    p21r3_both(d, function(t) summarise(group_by(t, f, .drop = FALSE), n = n()))
    p21r3_both(d, function(t) summarise(group_by(t, f, .drop = FALSE)))
    p21r3_both(d, function(t) count(t, f, .drop = FALSE))
  }
  p21r3_both(kn, function(t) summarise(group_by(t, b, .drop = FALSE), n = n()))
  p21r3_both(kn, function(t) summarise(group_by(t, s, .drop = FALSE), n = n()))
  p21r3_both(mix, function(t) summarise(group_by(t, f, s, .drop = FALSE), n = n(), .groups = "drop"))
  p21r3_both(mix, function(t) summarise(group_by(t, s, f, .drop = FALSE), n = n(), .groups = "drop"))
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    for (d in list(df, df2)) {
      expect_equal(group_keys(group_by(tbl_gpu(d, lazy = lazy), f, .drop = FALSE)),
                   group_keys(group_by(d, f, .drop = FALSE)), info = info)
    }
    expect_equal(group_keys(group_by(tbl_gpu(kn, lazy = lazy), b, .drop = FALSE)),
                 group_keys(group_by(kn, b, .drop = FALSE)), info = info)
    expect_equal(group_data(group_by(tbl_gpu(kn, lazy = lazy), b, .drop = FALSE))$b,
                 group_data(group_by(kn, b, .drop = FALSE))$b, info = info)
    expect_equal(group_keys(group_by(tbl_gpu(mix, lazy = lazy), f, s, .drop = FALSE)),
                 group_keys(group_by(mix, f, s, .drop = FALSE)), info = info)
    expect_equal(group_keys(group_by(tbl_gpu(mix, lazy = lazy), f, b, .drop = FALSE)),
                 group_keys(group_by(mix, f, b, .drop = FALSE)), info = info)
  }
})

test_that("a quantile() summary keeps its name through joins and binds", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2, 2, 3), x = c(1, 2, 3, 4, 5))
  y <- tibble::tibble(g = c(1, 2, 4), w = c(10, 20, 40))
  s <- function(t) summarise(group_by(t, g), q = quantile(x, 0.9))
  for (lazy in c(FALSE, TRUE)) {
    gd <- tbl_gpu(d, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    info <- paste("lazy =", lazy)
    # every row of q comes from a matched (or kept) row of the summary: on the GPU
    p21r3_native({
      expect_equal(collect(inner_join(s(gd), gy, by = "g")), inner_join(s(d), y, by = "g"), info = info)
      expect_equal(collect(left_join(s(gd), gy, by = "g")), left_join(s(d), y, by = "g"), info = info)
      expect_equal(collect(semi_join(s(gd), gy, by = "g")), semi_join(s(d), y, by = "g"), info = info)
      expect_equal(collect(anti_join(s(gd), gy, by = "g")), anti_join(s(d), y, by = "g"), info = info)
      expect_equal(collect(arrange(inner_join(gy, s(gd), by = "g"), g)),
                   arrange(inner_join(y, s(d), by = "g"), g), info = info)
    })
    # null-extended rows of q: dplyr's names
    expect_equal(collect(arrange(left_join(gy, s(gd), by = "g"), g)),
                 arrange(left_join(y, s(d), by = "g"), g), info = info)
    expect_equal(collect(arrange(full_join(s(gd), gy, by = "g"), g)),
                 arrange(full_join(s(d), y, by = "g"), g), info = info)
    expect_equal(collect(bind_rows(s(gd), s(gd))), bind_rows(s(d), s(d)), info = info)
    m <- function(t) summarise(group_by(t, g), q = mean(x))
    expect_equal(collect(bind_rows(s(gd), m(gd))), bind_rows(s(d), m(d)), info = info)
  }
})
