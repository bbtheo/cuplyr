# Acceptance tests for #21, added after the third review: the paths of
# summarise() and reframe() that skipped the round-3 checks (`.groups =
# "rowwise"`, reframe() over a `.drop = FALSE` grouping), fixed keys of a
# `.drop = FALSE` table, and inputs the reviewer probed that no test pinned
# (NA in a rebuilt key next to an ordered factor, a constant name on a join key).
# dplyr 1.2.1 is the oracle.

p21r4_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# dplyr on `df` as given (a plain data.frame stays one) against tbl_gpu(df).
p21r4_same_df <- function(df, pipeline) {
  expected <- as.list(dplyr::ungroup(pipeline(df)))
  for (lazy in c(FALSE, TRUE)) {
    got <- as.list(dplyr::ungroup(collect(pipeline(tbl_gpu(df, lazy = lazy)))))
    expect_equal(got, expected, info = paste("lazy =", lazy))
  }
}

test_that("rows under an empty parent key never take a constant name", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor(c("u", "u"), levels = c("u", "v")),
                      b = stats::setNames(c(1L, 2L), c("k", "k")), x = c(1, 2))
  p21r4_both(d, function(t) summarise(group_by(t, f, b, .drop = FALSE), n = n(), .groups = "rowwise"))
  p21r4_both(d, function(t) summarise(group_by(t, f, b, .drop = FALSE), n = n(), .groups = "drop"))
  p21r4_both(d, function(t) summarise(group_by(t, f, b, .drop = FALSE), n = n(), .groups = "keep"))
  p21r4_both(d, function(t) reframe(group_by(t, f, b, .drop = FALSE), n = n()))
  p21r4_both(d, function(t) count(t, f, b, .drop = FALSE))
  for (lazy in c(FALSE, TRUE)) {
    expect_equal(group_keys(group_by(tbl_gpu(d, lazy = lazy), f, b, .drop = FALSE)),
                 group_keys(group_by(d, f, b, .drop = FALSE)), info = paste("lazy =", lazy))
  }
})

test_that("fixed keys of a .drop = FALSE table follow dplyr's rebuilt keys", {
  skip_if_no_gpu()
  mix <- tibble::tibble(f = structure(factor(c("u", "v", "u"), levels = c("u", "v", "w")), label = "F"),
                        s = structure(c("p", "q", "p"), label = "S"), v = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    g <- filter(group_by(tbl_gpu(mix, lazy = lazy), f, s, .drop = FALSE), v == 2L)
    e <- filter(group_by(mix, f, s, .drop = FALSE), v == 2L)
    expect_equal(group_keys(g), group_keys(e), info = info)
    expect_equal(group_data(g)$f, group_data(e)$f, info = info)
    expect_equal(group_data(g)$s, group_data(e)$s, info = info)
    expect_equal(collect(summarise(g, n = n(), .groups = "drop")), summarise(e, n = n(), .groups = "drop"),
                 info = info)
  }
})

test_that("summarise(.groups = \"rowwise\") on a plain data.frame drops element names", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3), h = c(1, 1, 2))
  p21r4_same_df(df, function(t) summarise(t, q = quantile(x, 0.5), .groups = "rowwise"))
  p21r4_same_df(df, function(t) summarise(t, q = quantile(x, 0.5)))
  # a grouped input is a tibble: names stay
  p21r4_same_df(df, function(t) summarise(group_by(t, h), q = quantile(x, 0.5), .groups = "rowwise"))
})

test_that("an NA in a rebuilt key next to an ordered factor gets dplyr's names", {
  skip_if_no_gpu()
  d <- tibble::tibble(
    f = factor(c("u", "u", "v"), levels = c("u", "v", "w"), ordered = TRUE),
    b = structure(stats::setNames(c(2L, NA, NA), c("q", "r", "s")), label = "B"),
    x = c(1, 2, 3)
  )
  p21r4_both(d, function(t) summarise(group_by(t, f, b, .drop = FALSE), n = n(), .groups = "drop"))
  p21r4_both(d, function(t) count(t, f, b, .drop = FALSE))
  for (lazy in c(FALSE, TRUE)) {
    expect_equal(group_keys(group_by(tbl_gpu(d, lazy = lazy), f, b, .drop = FALSE)),
                 group_keys(group_by(d, f, b, .drop = FALSE)), info = paste("lazy =", lazy))
  }
})

test_that("a constant name on a join key follows dplyr", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2, 2, 3), x = c(1, 2, 3, 4, 5))
  s <- function(t) summarise(group_by(t, g), q = quantile(x, 0.5))
  y_same <- tibble::tibble(q = c(1.5, 3.5, 9), w = 1:3)
  y_cast <- tibble::tibble(q = c(5L, 7L), w = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    gd <- tbl_gpu(d, lazy = lazy)
    expect_equal(collect(inner_join(s(gd), tbl_gpu(y_same, lazy = lazy), by = "q")),
                 inner_join(s(d), y_same, by = "q"), info = info)
    expect_equal(collect(left_join(s(gd), tbl_gpu(y_same, lazy = lazy), by = "q")),
                 left_join(s(d), y_same, by = "q"), info = info)
    expect_equal(collect(arrange(full_join(s(gd), tbl_gpu(y_same, lazy = lazy), by = "q"), q)),
                 arrange(full_join(s(d), y_same, by = "q"), q), info = info)
    expect_equal(collect(arrange(right_join(s(gd), tbl_gpu(y_same, lazy = lazy), by = "q"), q)),
                 arrange(right_join(s(d), y_same, by = "q"), q), info = info)
    expect_equal(collect(left_join(s(gd), tbl_gpu(y_cast, lazy = lazy), by = "q")),
                 left_join(s(d), y_cast, by = "q"), info = info)
    expect_equal(collect(inner_join(s(gd), tbl_gpu(y_same, lazy = lazy), by = "q", keep = TRUE)),
                 inner_join(s(d), y_same, by = "q", keep = TRUE), info = info)
    # x's key is the one cast (integer to double)
    xk <- tibble::tibble(k = stats::setNames(1:3, c("n", "n", "n")), v = 1:3)
    yk <- tibble::tibble(k = c(1, 3, 5), w = 1:3)
    expect_equal(collect(inner_join(tbl_gpu(xk, lazy = lazy), tbl_gpu(yk, lazy = lazy), by = "k")),
                 inner_join(xk, yk, by = "k"), info = info)
    expect_equal(collect(left_join(tbl_gpu(xk, lazy = lazy), tbl_gpu(yk, lazy = lazy), by = "k")),
                 left_join(xk, yk, by = "k"), info = info)
  }
})
