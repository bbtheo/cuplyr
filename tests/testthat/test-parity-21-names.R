# Acceptance tests for #21: the names of a named vector column follow their
# rows the way dplyr's do (vctrs slices them with the values), eager and lazy.
# dplyr's rules, pinned with dplyr 1.2.1 on plain tibbles:
# - filter()/arrange()/slice*() slice the names with the rows; a zero-row
#   result keeps a zero-length names attribute;
# - mutate() leaves untouched columns alone, a bare copy keeps them, R's own
#   rules apply inside an expression (`b + 1L` keeps them, `mean(b)` does not);
# - a named constant in mutate() is recycled with its names;
# - summarise() reducers drop them; a group key keeps the names of the first
#   row of its group; a summary that returns names (`c("1" = x)`) keeps them;
# - select()/rename()/relocate()/group_by()/ungroup()/pull()/`$` keep them;
# - joins slice them; bind_rows() concatenates them ("" for an unnamed input).

p21_named <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    b = stats::setNames(c(5L, 3L, 4L, 1L, 2L), c("v", "w", "x", "y", "z")),
    d = stats::setNames(c(0.5, NA, 2.5, 3.5, 4.5), c("a", NA, "c", "c", "")),
    s = c("p", "q", "r", "s", "t")
  )
}

# The same pipeline, eager and lazy, against dplyr.
p21_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Run `code` with every CPU fallback turned into an error.
p21_native <- function(code) {
  withr::with_options(list(cuplyr.fallback = "error"), code)
}

test_that("names survive the round trip through the GPU", {
  skip_if_no_gpu()
  df <- p21_named()
  p21_both(df, function(x) x)
  for (lazy in c(FALSE, TRUE)) {
    out <- collect(tbl_gpu(df, lazy = lazy))
    expect_identical(names(out$b), c("v", "w", "x", "y", "z"))
    expect_identical(names(out$d), c("a", NA, "c", "c", ""))
    expect_null(names(out$g))
    expect_identical(as_tibble(tbl_gpu(df, lazy = lazy))$b, df$b)
    expect_identical(as.data.frame(tbl_gpu(df, lazy = lazy))$b, as.data.frame(df)$b)
    expect_identical(collect(compute(tbl_gpu(df, lazy = lazy)))$b, df$b)
  }
  # a zero-row named column
  p21_both(df[0, ], function(x) x)
})

test_that("pull(), `$` and [[ keep the names", {
  skip_if_no_gpu()
  df <- p21_named()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_identical(pull(gt, b), pull(df, b))
    expect_identical(pull(gt, d), pull(df, d))
    expect_identical(gt$b, df$b)
    expect_identical(gt[["d"]], df[["d"]])
    expect_identical(pull(mutate(gt, k = 1), b), df$b)
  }
})

test_that("mutate() keeps the names of the columns it does not touch, on the GPU", {
  skip_if_no_gpu()
  df <- p21_named()
  p21_native({
    p21_both(df, function(x) mutate(x, k = 1))
    p21_both(df, function(x) mutate(x, k = g * 2))
    p21_both(df, function(x) mutate(group_by(x, g), k = g + 1))
    p21_both(df, function(x) mutate(x, b2 = b, d2 = d))
    p21_both(df, function(x) mutate(x, b = g * 2))
    p21_both(df, function(x) mutate(x, k = 1, .keep = "unused"))
    p21_both(df, function(x) mutate(x, k = 1, .before = 1))
  })
  # the issue's example
  issue <- tibble::tibble(a = 1:3, b = stats::setNames(1:3, letters[1:3]))
  p21_both(issue, function(x) mutate(x, c = 1))
})

test_that("mutate() expressions that read a named column keep R's names", {
  skip_if_no_gpu()
  df <- p21_named()
  p21_both(df, function(x) mutate(x, b2 = b + 1L))
  p21_both(df, function(x) mutate(x, b2 = b * 2L, d2 = sqrt(d)))
  p21_both(df, function(x) mutate(x, gt = b > 2L))
  p21_both(df, function(x) mutate(x, m = mean(b)))
  p21_both(df, function(x) mutate(x, l = dplyr::lag(b)))
  p21_both(df, function(x) mutate(x, z = dplyr::if_else(b > 2L, b, 0L)))
  p21_both(df, function(x) mutate(group_by(x, g), b2 = b + 1L))
  p21_both(df, function(x) mutate(group_by(x, g), m = mean(b)))
  p21_both(df, function(x) mutate(x, b2 = b - 1L, .by = g))
  p21_both(df, function(x) transmute(x, b, b2 = b))
  p21_both(df, function(x) transmute(x, b2 = b + 1L))
})

test_that("a named constant in mutate() keeps its names", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3)
  p21_both(df, function(x) mutate(x, b = stats::setNames(1:3, letters[1:3])))
  p21_both(df, function(x) mutate(x, b = c(p = 1)))
  p21_both(df, function(x) mutate(group_by(x, a), b = c(p = 1)))
  p21_both(df, function(x) mutate(x, b = stats::setNames(c("u", "v", "w"), c("x", "y", "z"))))
  # dplyr's own test: a named list keeps its names too
  for (lazy in c(FALSE, TRUE)) {
    out <- collect(mutate(tbl_gpu(df, lazy = lazy), b = stats::setNames(as.list(1:3), letters[1:3])))
    expect_named(out$b, letters[1:3])
    out <- collect(mutate(tbl_gpu(df, lazy = lazy), b = stats::setNames(1:3, letters[1:3])))
    expect_named(out$b, letters[1:3])
  }
})

test_that("filter() slices the names with the rows", {
  skip_if_no_gpu()
  df <- p21_named()
  p21_both(df, function(x) filter(x, b > 2L))
  p21_both(df, function(x) filter(x, s %in% c("q", "t")))
  p21_both(df, function(x) filter(x, !is.na(d)))
  p21_both(df, function(x) filter(group_by(x, g), b == max(b)))
  p21_both(df, function(x) filter(x, b > 3L, .by = g))
  # no row left: a zero-length names attribute, as in dplyr
  p21_both(df, function(x) filter(x, b > 100L))
  # after a native step
  p21_both(df, function(x) filter(mutate(x, k = g + 1), k > 2))
})

test_that("arrange() and slice() reorder the names with the rows", {
  skip_if_no_gpu()
  df <- p21_named()
  p21_both(df, function(x) arrange(x, b))
  p21_both(df, function(x) arrange(x, desc(d)))
  p21_both(df, function(x) arrange(x, s))
  p21_both(df, function(x) slice(x, c(4, 2)))
  p21_both(df, function(x) slice_head(x, n = 2))
  p21_both(df, function(x) slice_tail(x, n = 2))
  p21_both(df, function(x) slice_min(x, b, n = 2))
  p21_both(df, function(x) slice_max(group_by(x, g), b, n = 1))
  p21_both(df, function(x) distinct(x, g, .keep_all = TRUE))
  p21_both(df, function(x) distinct(x, b))
})

test_that("summarise() keeps the names dplyr keeps", {
  skip_if_no_gpu()
  df <- p21_named()
  # reducers drop the names of their argument
  p21_both(df, function(x) summarise(group_by(x, g), b = sum(b), d = mean(d, na.rm = TRUE)))
  p21_both(df, function(x) summarise(x, b = max(b), n = n()))
  # a group key keeps the names of its group's first row
  p21_both(df, function(x) summarise(group_by(x, b), n = n()))
  p21_both(df, function(x) count(x, b))
  p21_both(df, function(x) summarise(x, n = n(), .by = d))
  # a summary that returns names (dplyr's tests)
  p21_both(tibble::tibble(a = 1:3), function(x) summarise(x, b = c("1" = a[[1]])))
  p21_both(tibble::tibble(a = c(1, 1, 2)),
           function(x) summarise(group_by(x, a), b = stats::setNames(nm = a[[1]])))
  p21_both(tibble::tibble(a = 1:3),
           function(x) summarise(rowwise(x), b = stats::setNames(nm = a)))
})

test_that("group_by(), ungroup(), select(), rename() and relocate() keep the names", {
  skip_if_no_gpu()
  df <- p21_named()
  p21_native({
    p21_both(df, function(x) group_by(x, g))
    p21_both(df, function(x) group_by(x, b))
    p21_both(df, function(x) ungroup(group_by(x, g)))
    p21_both(df, function(x) select(x, b, s))
    p21_both(df, function(x) select(x, bb = b))
    p21_both(df, function(x) rename(x, bb = b, dd = d))
    p21_both(df, function(x) relocate(x, d, .before = g))
  })
})

test_that("joins and binds follow dplyr's names", {
  skip_if_no_gpu()
  df <- p21_named()
  y <- tibble::tibble(g = c(1, 3, 4), h = stats::setNames(c(10, 30, 40), c("p", "q", "r")))
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    info <- paste("lazy =", lazy)
    expect_equal(collect(left_join(gt, gy, by = "g")), left_join(df, y, by = "g"), info = info)
    expect_equal(collect(inner_join(gt, gy, by = "g")), inner_join(df, y, by = "g"), info = info)
    expect_equal(collect(semi_join(gt, gy, by = "g")), semi_join(df, y, by = "g"), info = info)
    expect_equal(collect(anti_join(gt, gy, by = "g")), anti_join(df, y, by = "g"), info = info)
    expect_equal(collect(arrange(full_join(gt, gy, by = "g"), g)),
                 arrange(full_join(df, y, by = "g"), g), info = info)
    # a named key column of x
    k <- tibble::tibble(b = stats::setNames(c(1L, 9L), c("P", "Q")), z = 1)
    expect_equal(collect(inner_join(gt, tbl_gpu(k, lazy = lazy), by = "b")),
                 inner_join(df, k, by = "b"), info = info)
    expect_equal(collect(bind_rows(gt, gt)), bind_rows(df, df), info = info)
    plain <- tibble::tibble(b = 7L)
    expect_equal(collect(bind_rows(gt, tbl_gpu(plain, lazy = lazy))), bind_rows(df, plain), info = info)
    expect_equal(collect(bind_cols(tbl_gpu(tibble::tibble(z = 1:5), lazy = lazy), gt)),
                 bind_cols(tibble::tibble(z = 1:5), df), info = info)
  }
})

test_that("a named table still prints and glimpses", {
  skip_if_no_gpu()
  gt <- tbl_gpu(p21_named())
  expect_no_error(utils::capture.output(print(gt)))
  expect_no_error(utils::capture.output(glimpse(gt)))
  expect_no_error(utils::capture.output(print(tbl_gpu(p21_named(), lazy = TRUE) |> filter(b > 1L))))
})
