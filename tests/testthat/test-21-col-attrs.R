# Beyond the acceptance tests of #21: the routes that keep names with their
# rows (head/tail, `[`, cross joins, set operations), the shape of `$col_attrs`
# itself, and that a table without names or attributes pays for none of it.

c21_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    b = stats::setNames(c(5L, 3L, 4L, 1L, 2L), c("v", "w", "x", "y", "z")),
    lab = structure(c(1.5, 2.5, NA, 4.5, 5.5), label = "L")
  )
}

c21_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("a table without names or attributes has no $col_attrs, through any verb", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:5, f = factor(letters[1:5]), d = as.Date("2020-01-01") + 0:4,
                   t = as.POSIXct("2020-01-01", tz = "Europe/Helsinki") + 0:4)
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_null(.subset2(gt, "col_attrs"))
    out <- gt |> filter(a > 1) |> mutate(k = a * 2) |> arrange(desc(a)) |>
      group_by(f) |> summarise(n = n())
    expect_null(.subset2(out, "col_attrs"))
  }
})

test_that("$col_attrs holds only what a column has", {
  skip_if_no_gpu()
  gt <- tbl_gpu(c21_df(), lazy = FALSE)
  ca <- .subset2(gt, "col_attrs")
  expect_named(ca, c("b", "lab"))
  expect_identical(ca$b$names, c("v", "w", "x", "y", "z"))
  expect_identical(ca$lab$attrs, list(label = "L"))
  # equal names are one constant
  same <- tbl_gpu(tibble::tibble(a = stats::setNames(1:3, rep("n", 3))), lazy = FALSE)
  expect_identical(.subset2(same, "col_attrs")$a$names_const, "n")
  # the entry follows a rename and goes with a dropped column
  expect_named(.subset2(rename(gt, bb = b), "col_attrs"), c("bb", "lab"))
  expect_named(.subset2(select(gt, g, lab), "col_attrs"), "lab")
  gt2 <- gt
  names(gt2) <- c("g", "b2", "lab")
  expect_named(.subset2(gt2, "col_attrs"), c("b2", "lab"))
  expect_identical(collect(gt2)$b2, c21_df()$b)
})

test_that("head(), tail() and `[` move the names with the rows", {
  skip_if_no_gpu()
  df <- c21_df()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_identical(collect(head(gt, 2)), head(df, 2))
    expect_identical(collect(tail(gt, 2)), tail(df, 2))
    expect_identical(collect(gt[c(4, 2), ]), df[c(4, 2), ])
  }
})

test_that("cross joins, set operations and row verbs follow dplyr's names", {
  skip_if_no_gpu()
  df <- c21_df()
  y <- tibble::tibble(h = c("p", "q"))
  other <- tibble::tibble(g = c(1, 9), b = stats::setNames(c(5L, 7L), c("v", "k")),
                          lab = c(0, 1))
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_equal(collect(cross_join(gt, tbl_gpu(y, lazy = lazy))), cross_join(df, y))
    go <- tbl_gpu(other, lazy = lazy)
    expect_equal(collect(union_all(gt, go)), union_all(df, other))
    expect_equal(collect(arrange(union(gt, go), g, lab)), arrange(union(df, other), g, lab))
    expect_equal(collect(intersect(gt, go)), intersect(df, other))
    expect_equal(collect(setdiff(gt, go)), setdiff(df, other))
  }
})

test_that("a constant attribute survives where dplyr keeps it, and a named column does not mislead", {
  skip_if_no_gpu()
  df <- c21_df()
  # constants stay native through row verbs even next to a named column ... which routes it
  c21_both(df, function(x) filter(select(x, g, lab), lab > 2))
  c21_both(df, function(x) arrange(select(x, g, lab), desc(lab)))
  # a named column that a verb drops no longer routes
  withr::with_options(list(cuplyr.fallback = "error"), {
    c21_both(df, function(x) filter(select(x, g, lab), lab > 2))
  })
  # names are on the right rows after a chain that moves them twice
  c21_both(df, function(x) slice(arrange(filter(x, g > 1), b), 2:3))
})

test_that("a CPU route reports the column whose names it moves", {
  skip_if_no_gpu()
  gt <- tbl_gpu(c21_df())
  expect_error(
    withr::with_options(list(cuplyr.fallback = "error"), filter(gt, g > 1)),
    "b", class = "cuplyr_error_fallback"
  )
})

test_that("mutate() constants with names or attributes keep them", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3)
  c21_both(df, function(x) mutate(x, z = structure(7, foo = "bar")))
})

test_that("quantile() names: names = FALSE, digits and probabilities", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 3), x = c(1, 2, 3, 4, 5))
  withr::with_options(list(cuplyr.fallback = "error"), {
    c21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.123456789)))
    c21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.123456789, digits = 3)))
    c21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.5, names = FALSE)))
    c21_both(df, function(t) summarise(t, q = quantile(x, 0.5), m = mean(x)))
  })
  # other shapes around it are dplyr's
  c21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.5)[[1]]))
  c21_both(df, function(t) summarise(group_by(t, g), q = quantile(x, 0.5), r = q + 1))
})
