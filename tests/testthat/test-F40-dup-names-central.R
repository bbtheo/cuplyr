# Duplicated column names beyond the F40 acceptance tests: push_op()'s
# fail-closed rule (node_is_position_safe()) and the verbs made
# position-correct alongside it. Expectations are dplyr/tibble on the same
# minimal-name tibble.

f40c_dd <- function() {
  tibble::tibble(x = c(1, 1), x = c(1, 2), y = c(3, 3), .name_repair = "minimal")
}

f40c_same <- function(df, pipeline) {
  expected <- suppressMessages(pipeline(df))
  for (lazy in c(FALSE, TRUE)) {
    got <- suppressMessages(collect(pipeline(tbl_gpu(df, lazy = lazy))))
    expect_identical(got, expected, info = paste("lazy =", lazy))
  }
}

test_that("push_op() refuses a by-name node on a duplicated-name table", {
  skip_if_no_gpu()
  msg <- "Can't transform a data frame with duplicate names."
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(f40c_dd(), lazy = lazy)
    expect_error(push_op(g, ast_filter(input_node(g), list())), msg, fixed = TRUE)
    expect_error(push_op(g, ast_select(input_node(g), c("x", "y"))), msg, fixed = TRUE)
    expect_error(push_op(g, ast_distinct(input_node(g), "y")), msg, fixed = TRUE)
    expect_error(push_op(g, ast_rename(input_node(g), "y", "z")), msg, fixed = TRUE)
    out <- push_op(g, ast_select(input_node(g), c("x", "y"), positions = c(2L, 3L)))
    expect_identical(collect(out), tibble::tibble(x = c(1, 2), y = c(3, 3)))
  }
})

test_that("distinct() keys a bare duplicated name on its first column, like dplyr", {
  skip_if_no_gpu()
  dd <- f40c_dd()
  f40c_same(dd, function(d) distinct(d, x))
  f40c_same(dd, function(d) distinct(d, x, .keep_all = TRUE))
  f40c_same(dd, function(d) distinct(d, y, .keep_all = TRUE))
  f40c_same(dd, function(d) d |> distinct() |> select(2, 3))
  ff <- tibble::tibble(x = 1, x = factor(c("b", "a"), levels = c("b", "a")),
                       .name_repair = "minimal")
  f40c_same(ff, function(d) distinct(d))
  msg <- "Can't transform a data frame with duplicate names."
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(dd, lazy = lazy)
    # dplyr computes these keys with mutate(), which refuses the table
    expect_error(collect(distinct(g, z = y)), msg, fixed = TRUE)
    expect_error(collect(distinct(g, NULL)), msg, fixed = TRUE)
  }
})

test_that("$ and [[ address duplicated columns like tibble", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = c(1, 5), x = c(2, 6), y = c(3, 7), .name_repair = "minimal")
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(dd, lazy = lazy)
    expect_identical(g$x, dd$x)
    expect_identical(g[[2]], dd[[2]])
    expect_identical(g[["y"]], dd[["y"]])
  }
})

test_that("group_by(), pull(), count() and the other joins refuse duplicated names", {
  skip_if_no_gpu()
  dd <- f40c_dd()
  msg <- "Can't transform a data frame with duplicate names."
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(dd, lazy = lazy)
    h <- tbl_gpu(tibble::tibble(y = 3), lazy = lazy)
    expect_error(group_by(g, y), "Column name `x` must not be duplicated.", fixed = TRUE)
    expect_error(group_by(g), "Column name `x` must not be duplicated.", fixed = TRUE)
    expect_error(pull(g, y), "must be uniquely named", fixed = TRUE)
    expect_error(collect(count(g)), msg, fixed = TRUE)
    expect_error(collect(add_count(g)), msg, fixed = TRUE)
    expect_error(collect(cross_join(g, h)), "Input columns in `x` must be unique.", fixed = TRUE)
    expect_error(collect(full_join(g, h, by = "y")), "Input columns in `x` must be unique.",
                 fixed = TRUE)
    expect_error(collect(right_join(h, g, by = "y")), "Input columns in `y` must be unique.",
                 fixed = TRUE)
  }
})

test_that("set operations and bind_rows() follow dplyr on duplicated names", {
  skip_if_no_gpu()
  dd <- tibble::tibble(x = c(1, 5), x = c(2, 6), y = c(3, 7), .name_repair = "minimal")
  f40c_same(dd, function(d) union(d, d))
  f40c_same(dd, function(d) intersect(d, d))
  f40c_same(dd, function(d) setdiff(d, d))
  f40c_same(dd, function(d) union_all(d, d))
  f40c_same(dd, function(d) bind_rows(d, d))
  expect_message(bind_rows(tbl_gpu(dd), tbl_gpu(dd)), "New names")
})
