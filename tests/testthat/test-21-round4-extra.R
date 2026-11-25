# #21, fourth round: the cases around the acceptance tests of
# test-parity-21-round4.R (a merged join key that dplyr casts, the keys of a
# zero-dot summarise() and of a CPU route over fixed keys, `.groups =
# "rowwise"` through the key check, reframe() over rebuilt keys).

p21x4_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("a merged factor key has no attributes, as vctrs' cast leaves it", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = structure(factor(c("p", "q")), label = "K"), v = 1:2)
  y <- tibble::tibble(k = factor(c("p", "q")), w = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    gx <- tbl_gpu(x, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    expect_equal(collect(left_join(gx, gy, by = "k")), left_join(x, y, by = "k"), info = info)
    expect_equal(collect(inner_join(gx, gy, by = "k")), inner_join(x, y, by = "k"), info = info)
    expect_equal(collect(arrange(full_join(gx, gy, by = "k"), k)), arrange(full_join(x, y, by = "k"), k),
                 info = info)
    # both keys stay as they are
    expect_equal(collect(left_join(gx, gy, by = "k", keep = TRUE)), left_join(x, y, by = "k", keep = TRUE),
                 info = info)
    # a filtering join returns x
    expect_equal(collect(semi_join(gx, gy, by = "k")), semi_join(x, y, by = "k"), info = info)
  }
})

test_that("a merged key keeps x's entry when only y's key is cast", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = structure(stats::setNames(c("p", "q"), c("a", "a")), label = "K"), v = 1:2)
  y <- tibble::tibble(k = factor(c("p", "z")), w = 1:2)
  xd <- tibble::tibble(k = structure(c(1, 2), label = "K"), v = 1:2)
  yi <- tibble::tibble(k = c(1L, 5L), w = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy =", lazy)
    expect_equal(collect(left_join(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy), by = "k")),
                 left_join(x, y, by = "k"), info = info)
    expect_equal(collect(inner_join(tbl_gpu(xd, lazy = lazy), tbl_gpu(yi, lazy = lazy), by = "k")),
                 inner_join(xd, yi, by = "k"), info = info)
    expect_equal(collect(arrange(full_join(tbl_gpu(xd, lazy = lazy), tbl_gpu(yi, lazy = lazy), by = "k"), k)),
                 arrange(full_join(xd, yi, by = "k"), k), info = info)
    # x's key is the one cast: no attributes
    expect_equal(collect(left_join(tbl_gpu(yi, lazy = lazy), tbl_gpu(xd, lazy = lazy), by = "k")),
                 left_join(yi, xd, by = "k"), info = info)
  }
})

test_that("a zero-dot summarise() over fixed keys keeps the keys' attributes", {
  skip_if_no_gpu()
  mix <- tibble::tibble(f = structure(factor(c("u", "v", "u"), levels = c("u", "v", "w")), label = "F"),
                        s = structure(c("p", "q", "p"), label = "S"), v = 1:3)
  p21x4_both(mix, function(t) summarise(filter(group_by(t, f, s, .drop = FALSE), v == 2L), .groups = "drop"))
  p21x4_both(mix, function(t) summarise(group_by(t, f, s, .drop = FALSE), .groups = "drop"))
  # slice(.preserve = TRUE) of a .drop = TRUE grouping: nothing is rebuilt
  p21x4_both(mix, function(t) summarise(slice(group_by(t, f, s), 0L, .preserve = TRUE), .groups = "drop"))
})

test_that("a CPU route over fixed keys gives the keys dplyr's attributes", {
  skip_if_no_gpu()
  mix <- tibble::tibble(f = structure(factor(c("u", "v", "u"), levels = c("u", "v", "w")), label = "F"),
                        s = structure(c("p", "q", "p"), label = "S"), v = 1:3)
  # (paste(collapse =) has no native form: the whole call runs in dplyr)
  pipe <- function(t) {
    summarise(filter(group_by(t, f, s, .drop = FALSE), v == 2L), p = paste(v, collapse = "-"), .groups = "drop")
  }
  p21x4_both(mix, pipe)
  pipe_keep <- function(t) {
    summarise(slice(group_by(t, f, s), 0L, .preserve = TRUE), p = paste(v, collapse = "-"), .groups = "drop")
  }
  p21x4_both(mix, pipe_keep)
})

test_that("summarise(.groups = \"rowwise\") takes the key check of every summarise()", {
  skip_if_no_gpu()
  d <- tibble::tibble(b = stats::setNames(c(2L, 1L, 2L), c("x", "y", "z")), v = c(1, 2, 3))
  p21x4_both(d, function(t) summarise(group_by(t, b), n = n(), .groups = "rowwise"))
  p21x4_both(d, function(t) summarise(t, n = n(), .by = b, .groups = NULL))
  # a rowwise result of a plain data.frame: no names
  df <- data.frame(b = stats::setNames(c(2L, 1L, 2L), c("x", "y", "z")), v = c(1, 2, 3))
  expected <- as.list(dplyr::ungroup(summarise(df, q = quantile(v, 0.5), .groups = "rowwise")))
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(summarise(tbl_gpu(df, lazy = lazy), q = quantile(v, 0.5), .groups = "rowwise"))
    expect_equal(as.list(dplyr::ungroup(got)), expected, info = paste("lazy =", lazy))
  }
})

test_that("reframe() over rebuilt keys gives a constant key name dplyr's rows", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor(c("u", "u", "v"), levels = c("u", "v", "w")),
                      b = structure(stats::setNames(c(1L, 2L, 1L), c("k", "k", "k")), label = "B"),
                      x = c(1, 2, 3))
  p21x4_both(d, function(t) reframe(group_by(t, f, b, .drop = FALSE), q = quantile(x, c(0.25, 0.75))))
  p21x4_both(d, function(t) reframe(group_by(t, b, f, .drop = FALSE), m = mean(x)))
  # no factor key: nothing is rebuilt, the name stays native
  withr::with_options(list(cuplyr.fallback = "error"), {
    p21x4_both(d, function(t) reframe(group_by(t, b, .drop = FALSE), m = mean(x)))
  })
})

test_that("the column-attribute helpers leave a host result and attribute-free joins alone", {
  skip_if_no_gpu()
  host <- data.frame(col_attrs = 1:2)
  expect_identical(col_attrs_drop_names(host), host)
  a <- tbl_gpu(tibble::tibble(k = 1:2), lazy = FALSE)
  expect_false(join_names_need_cpu(a, a, "full"))
  expect_false(join_names_need_cpu(a, data.frame(k = 1L), "left"))
})
