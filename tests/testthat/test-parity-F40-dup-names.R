# F40 acceptance tests: select()/relocate() by position on a table with
# repeated column names.
#
# dplyr 1.2.1 contract (verified with real dplyr on tibbles built with
# .name_repair = "minimal"):
# - select(dd, 2) returns the SECOND column; select(dd, -1), select(dd, 2:3),
#   select(dd, 2, y) and select(dd, a = 2, b = 1) all address by position.
# - relocate() keeps both duplicated columns and moves them by position.
# - Selecting the duplicated name itself (select(dd, x), everything(),
#   where(...) covering both) raises vctrs_error_names_must_be_unique.
# The oracle helpers can't take duplicated names (as_tibble() repairs them),
# so these compare against hand-checked dplyr results.

f40d_df <- function() tibble::tibble(x = 1, x = 2, y = 3, .name_repair = "minimal")

f40d_check <- function(res, names, values) {
  expect_named(res, names)
  out <- collect(res)
  expect_named(out, names)
  expect_identical(unname(unlist(as.list(out))), values)
}

test_that("select() by position picks the right duplicated column (eager and lazy)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(f40d_df(), lazy = lazy)
    f40d_check(select(g, 2), "x", 2)
    f40d_check(select(g, 1), "x", 1)
    f40d_check(select(g, -1), c("x", "y"), c(2, 3))
    f40d_check(select(g, 2:3), c("x", "y"), c(2, 3))
    f40d_check(select(g, 2, y), c("x", "y"), c(2, 3))
    f40d_check(select(g, 3, 2), c("y", "x"), c(3, 2))
    f40d_check(select(g, a = 2, b = 1), c("a", "b"), c(2, 1))
    f40d_check(select(g, b = 1, y), c("b", "y"), c(1, 3))
    f40d_check(select(g, last_col(1)), "x", 2)
  }
})

test_that("relocate() by position keeps both duplicated columns", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(f40d_df(), lazy = lazy)
    f40d_check(relocate(g, 2), c("x", "x", "y"), c(2, 1, 3))
    f40d_check(relocate(g, y), c("y", "x", "x"), c(3, 1, 2))
    f40d_check(relocate(g, y, .before = 2), c("x", "y", "x"), c(1, 3, 2))
    f40d_check(relocate(g, 1, .after = last_col()), c("x", "y", "x"), c(2, 3, 1))
    f40d_check(relocate(g, z = 2), c("z", "x", "y"), c(2, 1, 3))
  }
})

test_that("positional select() composes with later verbs and compute()", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = c(1, 5), x = c(2, 6), y = c(3, 7), .name_repair = "minimal")
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(dd, lazy = lazy)
    out <- g |> select(2, 3) |> mutate(z = x + y) |> filter(z > 5) |> collect()
    expect_identical(out, tibble::tibble(x = 6, y = 7, z = 13))
    out2 <- g |> select(2, 3) |> filter(y > 0) |> compute() |> collect()
    expect_identical(out2, tibble::tibble(x = c(2, 6), y = c(3, 7)))
    out3 <- g |> relocate(y, .before = 2) |> select(3) |> collect()
    expect_identical(out3, tibble::tibble(x = c(2, 6)))
    out4 <- g |> select(1, 3) |> arrange(desc(x)) |> compute() |> collect()
    expect_identical(out4, tibble::tibble(x = c(5, 1), y = c(7, 3)))
  }
})

test_that("selecting a duplicated name is dplyr's names-must-be-unique error", {
  skip_if_no_gpu()
  g <- tbl_gpu(f40d_df())
  expect_error(select(g, x), class = "vctrs_error_names_must_be_unique")
  expect_error(select(g, everything()), class = "vctrs_error_names_must_be_unique")
  expect_error(select(g, 4), class = "vctrs_error_subscript_oob")
})

# Added after round-1 review B1/B2. Oracle for duplicated names: the same
# pipeline on the minimal-name tibble (dplyr) and on tbl_gpu (eager, lazy).
f40d_same <- function(df, pipeline) {
  expected <- pipeline(df)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(pipeline(tbl_gpu(df, lazy = lazy)))
    expect_identical(got, expected, info = paste("lazy =", lazy))
  }
}

test_that("positional select()/relocate()/rename() keep factor columns right (review B1)", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = factor("a"), x = 2.5, y = 1, .name_repair = "minimal")
  ff <- tibble::tibble(x = factor("a"), x = factor(c("b"), levels = c("z", "b")),
                       .name_repair = "minimal")
  df2 <- tibble::tibble(x = 1.5, x = factor("a"), .name_repair = "minimal")
  f40d_same(dd, function(d) select(d, 2))
  f40d_same(dd, function(d) select(d, 1, y))
  f40d_same(dd, function(d) select(d, a = 2, b = 1))
  f40d_same(dd, function(d) relocate(d, 2))
  f40d_same(ff, function(d) select(d, 2))
  f40d_same(ff, function(d) relocate(d, 2))
  f40d_same(ff, function(d) rename(d, z = 2))
  f40d_same(df2, function(d) d)
  f40d_same(df2, function(d) select(d, b = 2, a = 1))
  # both columns under their shared name: dplyr's names-must-be-unique error
  expect_error(select(tbl_gpu(df2), 2, 1), class = "vctrs_error_names_must_be_unique")
})

test_that("one column selected twice under two names is positional too", {
  skip_if_no_gpu()
  d <- tibble::tibble(x = 1, y = factor("u"))
  f40d_same(d, function(d) select(d, a = x, b = x))
  f40d_same(d, function(d) select(d, a = y, y, b = y))
  f40d_same(d, function(d) select(d, b = 1, a = 1, y))
})

test_that("verbs that read columns error on duplicated names like dplyr (review B2)", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = 1, x = 2, y = 3, .name_repair = "minimal")
  msg <- "Can't transform a data frame with duplicate names."
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(dd, lazy = lazy)
    expect_error(collect(filter(g, y > 0)), msg, fixed = TRUE)
    expect_error(collect(filter_out(g, y > 5)), msg, fixed = TRUE)
    expect_error(collect(mutate(g, z = x * 10)), msg, fixed = TRUE)
    expect_error(collect(transmute(g, z = y)), msg, fixed = TRUE)
    expect_error(collect(summarise(g, n = dplyr::n())), msg, fixed = TRUE)
    expect_error(collect(arrange(g, y)), msg, fixed = TRUE)
  }
  # after a positional select() that makes the names unique, verbs work
  f40d_same(dd, function(d) d |> select(2, 3) |> mutate(z = x * 10) |> filter(y > 0) |> arrange(z))
})

# Added after round-2 review B3/B4 (expectations checked with tools/r on the
# minimal-name tibble).
test_that("distinct()/unique() and `[` address duplicated columns by position (review B3)", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = c(1, 1), x = c(1, 2), y = c(3, 3), .name_repair = "minimal")
  d2 <- tibble::tibble(x = c(1, 5), x = c(2, 6), y = c(3, 7), .name_repair = "minimal")
  f40d_same(dd, function(d) distinct(d))
  f40d_same(dd, function(d) unique(d))
  f40d_same(dd, function(d) distinct(d, y))
  f40d_same(d2, function(d) d[, 1:2])
  f40d_same(d2, function(d) d[2])
  f40d_same(d2, function(d) d[c(3, 2)])
  f40d_same(d2, function(d) d[2, ])
  f40d_same(d2, function(d) head(d, 1))
  f40d_same(d2, function(d) tail(d, 1))
})

test_that("slice*() and joins raise dplyr's duplicated-name errors (review B4)", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = c(1, 5), x = c(2, 6), y = c(3, 7), .name_repair = "minimal")
  msg <- "Can't transform a data frame with duplicate names."
  jmsg <- "Input columns in `x` must be unique."
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(dd, lazy = lazy)
    h <- tbl_gpu(tibble::tibble(y = 3), lazy = lazy)
    expect_error(collect(slice(g, 1)), msg, fixed = TRUE)
    expect_error(collect(slice(g)), msg, fixed = TRUE)
    expect_error(collect(slice_head(g, n = 1)), msg, fixed = TRUE)
    expect_error(collect(slice_tail(g, n = 1)), msg, fixed = TRUE)
    expect_error(collect(slice_min(g, y, n = 1)), msg, fixed = TRUE)
    expect_error(collect(slice_max(g, y, n = 1)), msg, fixed = TRUE)
    expect_error(collect(slice_sample(g, n = 1)), msg, fixed = TRUE)
    expect_error(collect(semi_join(g, h, by = "y")), jmsg, fixed = TRUE)
    expect_error(collect(anti_join(g, h, by = "y")), jmsg, fixed = TRUE)
    expect_error(collect(left_join(g, h, by = "y")), jmsg, fixed = TRUE)
    expect_error(collect(inner_join(g, h, by = "y")), jmsg, fixed = TRUE)
  }
})

# Added after round-3 review B5: `names<-` creating duplicated names after a
# reorder must keep each factor column's own levels.
f40d_same_names <- function(df, pipeline, nm) {
  expected <- pipeline(df)
  names(expected) <- nm
  for (lazy in c(FALSE, TRUE)) {
    g <- pipeline(tbl_gpu(df, lazy = lazy))
    names(g) <- nm
    expect_identical(collect(g), expected, info = paste("lazy =", lazy))
  }
}

test_that("names<- that duplicates names keeps factor levels with their columns (review B5)", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor("a", levels = c("a", "b")),
                      h = factor("z", levels = c("y", "z")), v = 1)
  f40d_same_names(d, function(d) relocate(d, h), c("x", "x", "v"))
  f40d_same_names(d, function(d) select(d, h, f), c("x", "x"))
  f40d_same_names(d, function(d) select(d, v, h, f), c("x", "x", "x"))
  dd <- tibble::tibble(x = factor("a"), x = factor("z", levels = c("y", "z")), y = 1,
                       .name_repair = "minimal")
  f40d_same_names(dd, function(d) d, c("a", "b", "y"))
  f40d_same_names(dd, function(d) relocate(d, 2), c("p", "q", "y"))
})
