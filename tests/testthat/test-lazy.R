# Tests for Lazy Evaluation

# Execution Mode Control Tests

test_that("resolve_exec_mode handles TRUE/FALSE", {
  expect_equal(resolve_exec_mode(TRUE), "lazy")
  expect_equal(resolve_exec_mode(FALSE), "eager")
})

test_that("resolve_exec_mode handles string values", {
  expect_equal(resolve_exec_mode("lazy"), "lazy")
  expect_equal(resolve_exec_mode("eager"), "eager")
})
test_that("resolve_exec_mode rejects invalid values", {
  expect_error(resolve_exec_mode(NA))
  expect_error(resolve_exec_mode("invalid"))
  expect_error(resolve_exec_mode(123))
})

test_that("resolve_exec_mode uses option when explicit is NULL", {
  old_opt <- getOption("cuplyr.exec_mode")
  on.exit(options(cuplyr.exec_mode = old_opt))

  options(cuplyr.exec_mode = "lazy")
  expect_equal(resolve_exec_mode(NULL), "lazy")

  options(cuplyr.exec_mode = "eager")
  expect_equal(resolve_exec_mode(NULL), "eager")
})

test_that("grouping metadata verbs preserve exec_mode", {
  old_opt <- getOption("cuplyr.exec_mode")
  on.exit(options(cuplyr.exec_mode = old_opt))
  options(cuplyr.exec_mode = "lazy")

  tbl <- new_tbl_gpu(
    ptr = NULL,
    schema = list(names = "a", types = "INT32"),
    lazy_ops = NULL,
    groups = character(),
    exec_mode = "eager"
  )

  grouped <- group_by(tbl, a)
  expect_identical(grouped$exec_mode, "eager")

  ungrouped <- ungroup(grouped)
  expect_identical(ungrouped$exec_mode, "eager")
})

test_that("has_pending_ops treats empty list lazy_ops as no pending ops", {
  tbl <- new_tbl_gpu(
    ptr = NULL,
    schema = list(names = "a", types = "INT32"),
    lazy_ops = list(),
    groups = character(),
    exec_mode = "eager"
  )

  expect_false(has_pending_ops(tbl))
})

test_that("compute accepts legacy empty-list lazy_ops", {
  tbl <- new_tbl_gpu(
    ptr = NULL,
    schema = list(names = "a", types = "INT32"),
    lazy_ops = list(),
    groups = character(),
    exec_mode = "lazy"
  )

  out <- compute(tbl)
  expect_null(out$lazy_ops)
})

test_that("compute validates non-empty invalid lazy_ops shape", {
  tbl <- new_tbl_gpu(
    ptr = NULL,
    schema = list(names = "a", types = "INT32"),
    lazy_ops = list(invalid = TRUE),
    groups = character(),
    exec_mode = "lazy"
  )

  expect_error(compute(tbl), "lazy_ops.*AST|Invalid lazy_ops")
})

test_that("tbl_gpu respects lazy parameter", {
  skip_if_no_gpu()

  eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)
  expect_equal(eager_tbl$exec_mode, "eager")

  lazy_tbl <- tbl_gpu(mtcars, lazy = TRUE)
  expect_equal(lazy_tbl$exec_mode, "lazy")
})

test_that("tbl_gpu switching to eager computes pending ops", {
  skip_if_no_gpu()

  lazy_tbl <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20)

  expect_true(has_pending_ops(lazy_tbl))

  eager_tbl <- tbl_gpu(lazy_tbl, lazy = FALSE)
  expect_false(is_lazy(eager_tbl))
  expect_false(has_pending_ops(eager_tbl))

  result <- collect(eager_tbl)
  expect_true(all(result$mpg > 20))
  expect_true(nrow(result) > 0)
})

test_that("is_lazy returns correct values", {
  skip_if_no_gpu()

  eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)
  lazy_tbl <- tbl_gpu(mtcars, lazy = TRUE)

  expect_false(is_lazy(eager_tbl))
  expect_true(is_lazy(lazy_tbl))
  expect_false(is_lazy(mtcars))
})

test_that("as_lazy switches to lazy mode", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = FALSE)
  expect_false(is_lazy(tbl))

  lazy_tbl <- as_lazy(tbl)
  expect_true(is_lazy(lazy_tbl))
})

test_that("as_eager switches to eager mode and computes", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20)

  expect_true(has_pending_ops(tbl))

  eager_tbl <- as_eager(tbl)
  expect_false(is_lazy(eager_tbl))
  expect_false(has_pending_ops(eager_tbl))
})

# Lazy Pipeline Tests

test_that("lazy filter builds AST without executing", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE)
  result <- tbl |> filter(mpg > 20)

  expect_true(has_pending_ops(result))
  expect_s3_class(result$lazy_ops, "ast_filter")
})

test_that("lazy filter ignores name collisions in calling environment", {
  skip_if_no_gpu()

  mpg <- 1
  on.exit(rm(mpg), add = TRUE)
  result <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    collect()

  expect_true(all(result$mpg > 20))
  expect_true(nrow(result) > 0)
})

test_that("lazy mutate builds AST without executing", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE)
  result <- tbl |> mutate(kpl = mpg * 0.425)

  expect_true(has_pending_ops(result))
  expect_s3_class(result$lazy_ops, "ast_mutate")
  expect_true("kpl" %in% result$schema$names)
})

test_that("lazy select builds AST without executing", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE)
  result <- tbl |> select(mpg, cyl, hp)

  expect_true(has_pending_ops(result))
  expect_s3_class(result$lazy_ops, "ast_select")
  expect_equal(result$schema$names, c("mpg", "cyl", "hp"))
})

test_that("lazy arrange builds AST (barrier)", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE)
  result <- tbl |> arrange(desc(mpg))

  expect_true(has_pending_ops(result))
  expect_s3_class(result$lazy_ops, "ast_arrange")
})

test_that("lazy pipeline chains correctly", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE)
  result <- tbl |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425) |>
    select(mpg, kpl, hp)

  expect_true(has_pending_ops(result))
  expect_equal(ast_depth(result$lazy_ops), 4)  # source -> filter -> mutate -> select
})

# Lazy vs Eager Equivalence Tests

test_that("lazy and eager filter produce same results", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10, y = 10:1)

  eager_result <- tbl_gpu(df, lazy = FALSE) |>
    filter(x > 5) |>
    collect()

  lazy_result <- tbl_gpu(df, lazy = TRUE) |>
    filter(x > 5) |>
    collect()

  expect_equal(eager_result, lazy_result)
})

test_that("lazy and eager mutate produce same results", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10, y = 10:1)

  eager_result <- tbl_gpu(df, lazy = FALSE) |>
    mutate(z = x + y) |>
    collect()

  lazy_result <- tbl_gpu(df, lazy = TRUE) |>
    mutate(z = x + y) |>
    collect()

  expect_equal(eager_result, lazy_result)
})

test_that("lazy and eager select produce same results", {
  skip_if_no_gpu()

  eager_result <- tbl_gpu(mtcars, lazy = FALSE) |>
    select(mpg, cyl) |>
    collect()

  lazy_result <- tbl_gpu(mtcars, lazy = TRUE) |>
    select(mpg, cyl) |>
    collect()

  expect_equal(eager_result, lazy_result)
})

test_that("lazy and eager arrange produce same results", {
  skip_if_no_gpu()

  df <- data.frame(x = c(3, 1, 2), y = c("c", "a", "b"))

  eager_result <- tbl_gpu(df, lazy = FALSE) |>
    arrange(x) |>
    collect()

  lazy_result <- tbl_gpu(df, lazy = TRUE) |>
    arrange(x) |>
    collect()

  expect_equal(eager_result, lazy_result)
})

test_that("lazy and eager complex pipeline produces same results", {
  skip_if_no_gpu()

  df <- data.frame(
    x = runif(100),
    y = runif(100),
    g = rep(1:10, each = 10)
  )

  eager_result <- tbl_gpu(df, lazy = FALSE) |>
    filter(x > 0.3) |>
    mutate(z = x + y) |>
    mutate(w = z * 2) |>
    select(g, z, w) |>
    arrange(desc(w)) |>
    collect()

  lazy_result <- tbl_gpu(df, lazy = TRUE) |>
    filter(x > 0.3) |>
    mutate(z = x + y) |>
    mutate(w = z * 2) |>
    select(g, z, w) |>
    arrange(desc(w)) |>
    collect()

  expect_equal(eager_result, lazy_result)
})

# Compute and Collapse Tests

test_that("compute materializes lazy operations", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425)

  expect_true(has_pending_ops(tbl))

  computed <- compute(tbl)
  expect_false(has_pending_ops(computed))
  expect_true(is_lazy(computed))  # still in lazy mode
})

test_that("compute is no-op in eager mode", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = FALSE)
  computed <- compute(tbl)

  expect_identical(tbl$ptr, computed$ptr)
})

test_that("collapse adds barrier node", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    collapse()

  expect_true(has_pending_ops(tbl))
  expect_s3_class(tbl$lazy_ops, "ast_barrier")
})

test_that("compute allows branching", {
  skip_if_no_gpu()

  base <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    compute()

  branch1 <- base |> filter(cyl == 4) |> collect()
  branch2 <- base |> filter(cyl == 6) |> collect()

  expect_true(nrow(branch1) > 0 || nrow(branch2) > 0)
  expect_true(all(branch1$cyl == 4) || nrow(branch1) == 0)
  expect_true(all(branch2$cyl == 6) || nrow(branch2) == 0)
})

# show_query Tests

test_that("show_query prints AST info", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425)

  expect_output(show_query(tbl), "Pending operations")
  expect_output(show_query(tbl), "AST depth")
})

test_that("show_query handles eager tables", {
  skip_if_no_gpu()

  tbl <- tbl_gpu(mtcars, lazy = FALSE)
  expect_output(show_query(tbl), "No pending operations")
})

# Edge Cases

test_that("empty filter returns all rows in lazy mode", {
  skip_if_no_gpu()

  result <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter() |>
    collect()

  expect_equal(nrow(result), nrow(mtcars))
})

test_that("empty mutate returns unchanged data in lazy mode", {
  skip_if_no_gpu()

  result <- tbl_gpu(mtcars, lazy = TRUE) |>
    mutate() |>
    collect()

  expect_equal(ncol(result), ncol(mtcars))
})

test_that("column replacement works in lazy mode", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 6:10)

  eager_result <- tbl_gpu(df, lazy = FALSE) |>
    mutate(x = x * 2) |>
    collect()

  lazy_result <- tbl_gpu(df, lazy = TRUE) |>
    mutate(x = x * 2) |>
    collect()

  expect_equal(eager_result, lazy_result)
  expect_equal(lazy_result$x, c(2, 4, 6, 8, 10))
})

test_that("chained mutations with dependencies work", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5)

  result <- tbl_gpu(df, lazy = TRUE) |>
    mutate(y = x + 1) |>
    mutate(z = y * 2) |>
    collect()

  expect_equal(result$y, 2:6)
  expect_equal(result$z, (2:6) * 2)
})
