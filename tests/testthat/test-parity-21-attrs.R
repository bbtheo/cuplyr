# Acceptance tests for #21: per-column attributes (a `label`, a `foo`, scale()'s
# `scaled:center`), one-column matrix columns and quantile()'s names, eager and
# lazy, against dplyr 1.2.1 on plain tibbles. dplyr's rules:
# - an attribute is a property of the column, not of its rows: filter(),
#   arrange(), slice(), distinct(), group_by(), the group keys of summarise()
#   and the key columns of a join (x's attributes) keep it;
# - an expression decides for itself (`f + 1` keeps it, `f > 1` and reducers
#   drop it); a bare copy keeps it;
# - bind_rows() and union() drop it;
# - a one-column matrix stays a matrix with as many rows as the table;
# - quantile() names its result ("50%"), so summarise() keeps that name.

p21_labelled <- function() {
  tibble::tibble(
    g = structure(c(1L, 1L, 2L, 2L, 3L), label = "group"),
    num = structure(c(1.5, 2.5, NA, 4.5, 5.5), label = "X label", foo = "bar"),
    s = structure(c("a", "b", "c", "d", "e"), label = "S"),
    l = structure(c(TRUE, FALSE, NA, TRUE, TRUE), label = "L"),
    fct = structure(factor(c("u", "v", "u", "w", "v")), label = "F"),
    dt = structure(as.Date("2020-01-01") + 0:4, label = "D"),
    ts = structure(as.POSIXct("2020-01-01 10:00:00", tz = "America/New_York") + 3600 * 0:4,
                   label = "T"),
    v = c(3, 1, 4, 1, 5)
  )
}

p21_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

p21_native <- function(code) {
  withr::with_options(list(cuplyr.fallback = "error"), code)
}

test_that("column attributes survive the round trip, for every GPU type", {
  skip_if_no_gpu()
  df <- p21_labelled()
  p21_both(df, function(x) x)
  for (lazy in c(FALSE, TRUE)) {
    out <- collect(tbl_gpu(df, lazy = lazy))
    for (nm in names(df)) {
      expect_identical(out[[nm]], df[[nm]], info = paste(nm, lazy))
    }
    expect_identical(pull(tbl_gpu(df, lazy = lazy), num), df$num)
    expect_identical(tbl_gpu(df, lazy = lazy)$fct, df$fct)
  }
  p21_both(df[0, ], function(x) x)
})

test_that("group_by() keeps attributes (dplyr's test)", {
  skip_if_no_gpu()
  d <- data.frame(x = structure(1:10, foo = "bar"))
  for (lazy in c(FALSE, TRUE)) {
    gd <- group_by(tbl_gpu(d, lazy = lazy))
    expect_equal(attr(gd$x, "foo"), "bar")
    expect_equal(attr(collect(group_by(tbl_gpu(d, lazy = lazy), x))$x, "foo"), "bar")
  }
})

test_that("row verbs keep attributes on the GPU", {
  skip_if_no_gpu()
  df <- p21_labelled()
  p21_native({
    p21_both(df, function(x) filter(x, v > 1))
    p21_both(df, function(x) filter(x, num > 2))
    p21_both(df, function(x) filter(x, v > 100))
    p21_both(df, function(x) filter(group_by(x, g), v == max(v)))
    p21_both(df, function(x) arrange(x, desc(v)))
    p21_both(df, function(x) arrange(x, num))
    p21_both(df, function(x) slice(x, c(5, 1, 3)))
    p21_both(df, function(x) slice_head(x, n = 2))
    p21_both(df, function(x) slice_max(x, v, n = 2))
    p21_both(df, function(x) distinct(x, g, .keep_all = TRUE))
    p21_both(df, function(x) distinct(x, s, fct))
    p21_both(df, function(x) group_by(x, g, fct))
    p21_both(df, function(x) ungroup(group_by(x, g)))
    p21_both(df, function(x) select(x, num, lab = s))
    p21_both(df, function(x) rename(x, xx = num))
    p21_both(df, function(x) relocate(x, v))
  })
})

test_that("mutate() keeps attributes of untouched and copied columns on the GPU", {
  skip_if_no_gpu()
  df <- p21_labelled()
  p21_native({
    p21_both(df, function(x) mutate(x, k = v * 2))
    p21_both(df, function(x) mutate(group_by(x, g), k = v + 1))
    p21_both(df, function(x) mutate(x, n2 = num, d2 = dt))
    p21_both(df, function(x) mutate(x, num = v))
    p21_both(df, function(x) transmute(x, num, k = v))
  })
})

test_that("mutate() expressions over an attributed column follow R", {
  skip_if_no_gpu()
  df <- p21_labelled()
  p21_both(df, function(x) mutate(x, n2 = num + 1))
  p21_both(df, function(x) mutate(x, n2 = -num, g2 = g * 2L))
  p21_both(df, function(x) mutate(x, big = num > 2))
  p21_both(df, function(x) mutate(x, miss = is.na(num)))
  p21_both(df, function(x) mutate(x, m = mean(num, na.rm = TRUE)))
  p21_both(df, function(x) mutate(group_by(x, g), n2 = num * 2))
  p21_both(df, function(x) mutate(x, z = structure(1:5, foo = "baz")))
})

test_that("summarise() keeps the attributes of its group keys only", {
  skip_if_no_gpu()
  df <- p21_labelled()
  p21_native({
    p21_both(df, function(x) summarise(group_by(x, g), n = n(), m = mean(num, na.rm = TRUE), s = max(v)))
    p21_both(df, function(x) summarise(group_by(x, fct), n = n()))
    p21_both(df, function(x) summarise(group_by(x, s, dt), v = sum(v), .groups = "drop"))
    p21_both(df, function(x) summarise(x, n = n(), m = min(v), .by = c(g, l)))
    p21_both(df, function(x) summarise(x, m = mean(num, na.rm = TRUE), n = n_distinct(g)))
    p21_both(df, function(x) count(x, g))
  })
})

test_that("joins give the key x's attributes and keep each side's columns", {
  skip_if_no_gpu()
  df <- p21_labelled()
  y <- tibble::tibble(g = c(1L, 3L, 9L), w = structure(c(10, 30, 90), label = "W"))
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    info <- paste("lazy =", lazy)
    expect_equal(collect(arrange(left_join(gt, gy, by = "g"), g, v)),
                 arrange(left_join(df, y, by = "g"), g, v), info = info)
    expect_equal(collect(arrange(inner_join(gt, gy, by = "g"), g, v)),
                 arrange(inner_join(df, y, by = "g"), g, v), info = info)
    expect_equal(collect(arrange(right_join(gt, gy, by = "g"), g, v)),
                 arrange(right_join(df, y, by = "g"), g, v), info = info)
    expect_equal(collect(arrange(full_join(gt, gy, by = "g"), g, v)),
                 arrange(full_join(df, y, by = "g"), g, v), info = info)
    expect_equal(collect(semi_join(gt, gy, by = "g")), semi_join(df, y, by = "g"), info = info)
    expect_equal(collect(anti_join(gt, gy, by = "g")), anti_join(df, y, by = "g"), info = info)
    # bind_rows() and union() drop them
    expect_equal(collect(bind_rows(gt, gt)), bind_rows(df, df), info = info)
    expect_equal(collect(arrange(union(gt, gt), v, s)), arrange(union(df, df), v, s), info = info)
  }
})

test_that("a one-column matrix column stays a matrix", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:4, m = matrix(c(4, 3, 2, 1), ncol = 1))
  p21_both(df, function(x) x)
  p21_native({
    p21_both(df, function(x) filter(x, a > 1))
    p21_both(df, function(x) arrange(x, m))
    p21_both(df, function(x) mutate(x, z = a * 2))
    p21_both(df, function(x) select(x, m))
    p21_both(df, function(x) slice(x, 2))
  })
  p21_both(df, function(x) mutate(x, z = m + 1))
  # colnames are the matrix's; rownames follow the rows
  dn <- tibble::tibble(a = 1:3, m = matrix(c(7, 8, 9), ncol = 1, dimnames = list(c("r1", "r2", "r3"), "c1")))
  p21_both(dn, function(x) x)
  p21_both(dn, function(x) filter(x, a != 2L))
  p21_both(dn, function(x) arrange(x, desc(a)))
  # a matrix of more than one column still has no GPU form (#70)
  expect_error(tbl_gpu(tibble::tibble(a = 1:2, m = matrix(1:4, ncol = 2))),
               "`m`", fixed = TRUE, class = "cuplyr_error_unsupported_class")
})

test_that("mutate() handles matrix columns (dplyr's test)", {
  skip_if_no_gpu()
  df <- data.frame(a = rep(1:3, each = 2), b = 1:6)
  p21_both(df, function(x) mutate(x, b = scale(b)))
  p21_both(df, function(x) mutate(group_by(x, a), b = scale(b)))
  p21_both(df, function(x) mutate(rowwise(x), b = scale(b)))
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_equal(dim(collect(mutate(gt, b = scale(b)))$b), c(6, 1))
    expect_equal(dim(collect(mutate(group_by(gt, a), b = scale(b)))$b), c(6, 1))
    expect_equal(dim(collect(mutate(rowwise(gt), b = scale(b)))$b), c(6, 1))
    # and keeps its shape through later verbs
    expect_equal(dim(collect(filter(mutate(gt, b = scale(b)), a > 1))$b), c(4, 1))
  }
  p21_both(df, function(x) filter(mutate(x, b = scale(b)), a > 1))
  p21_both(df, function(x) summarise(mutate(x, b = scale(b)), n = n(), .by = a))
})

test_that("summarise() keeps quantile()'s names, on the GPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 3), y = c("a", "a", "b", "b", "c"), x = c(1, 2, 3, 4, 5))
  p21_native({
    p21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.5)))
    p21_both(df, function(t) summarise(t, q = quantile(x, 0.25)))
    p21_both(df, function(t) summarise(t, q = quantile(x, 0.25), .by = y))
    p21_both(df, function(t) summarise(t, q = quantile(x, 1 / 3)))
    p21_both(df, function(t) summarise(group_by(t, g), lo = quantile(x, 0.1), hi = quantile(x, 0.9), n = n()))
    # empty input: one named row ungrouped, no row grouped
    p21_both(df[0, ], function(t) summarise(t, q = quantile(x, 0.25)))
    p21_both(df[0, ], function(t) summarise(group_by(t, g), q = quantile(x, 0.25)))
  })
  p21_both(df, function(t) summarise(t, q = quantile(x, 0.9, names = FALSE)))
  # around an aggregate, R keeps quantile()'s name
  p21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.5) + 1))
  # and the name follows later verbs
  p21_both(df, function(t) filter(summarise(group_by(t, g), q = quantile(x, 0.5)), q > 1))
  p21_both(df, function(t) arrange(summarise(group_by(t, g), q = quantile(x, 0.5)), desc(q)))
  # dplyr's test
  d <- data.frame(x = c(1:3), y = letters[1:3])
  p21_both(d, function(t) summarise(group_by(t, y), a = length(x), b = quantile(x, 0.5)))
  for (lazy in c(FALSE, TRUE)) {
    res <- collect(summarise(group_by(tbl_gpu(d, lazy = lazy), y), a = length(x), b = quantile(x, 0.5)))
    expect_equal(res$b, c("50%" = 1, "50%" = 2, "50%" = 3))
  }
})
