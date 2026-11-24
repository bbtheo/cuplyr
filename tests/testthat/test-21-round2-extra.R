# Round 2 of #21: what the review found beyond the acceptance tests. The
# column-origin map under rename/select, the reasons a route gives, a `[` that
# adds rows, attributes dplyr's own classes keep, and constants that read as
# attributed only when their class is one a column can hold.

c21r2_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# The value of `expr` and the CPU-fallback conditions it signalled
c21r2_cpu <- function(expr) {
  conds <- list()
  value <- withr::with_options(
    list(cuplyr.fallback = NULL, cuplyr.plan_break = NULL),
    withCallingHandlers(
      expr,
      cuplyr_cpu_fallback = function(c) conds[[length(conds) + 1L]] <<- c,
      warning = function(w) invokeRestart("muffleWarning")
    )
  )
  list(value = value, conds = conds)
}

test_that("`[` that adds rows gives the added rows no name, as tibble does", {
  skip_if_no_gpu()
  df <- tibble::tibble(b = c(a = 1, a = 2), k = 1:2)
  expect_identical(.subset2(tbl_gpu(df), "col_attrs")$b$names_const, "a")
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_equal(tibble::as_tibble(collect(gt[c(1, 5), ])), df[c(1, 5), ], info = lazy)
    expect_equal(tibble::as_tibble(collect(gt[c(2, NA), ])), df[c(2, NA), ], info = lazy)
    expect_equal(tibble::as_tibble(collect(gt[2:1, ])), df[2:1, ], info = lazy)
  }
})

test_that("a constant whose class holds extra state is the type check's to report", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  # (the difftime has no GPU type, so the CPU run's result is refused: the
  # reason of the notice is what counts)
  got <- c21r2_cpu(try(collect(mutate(tbl_gpu(df), p = as.difftime(2, units = "hours"), q = x + 1L)),
                       silent = TRUE))
  expect_gt(length(got$conds), 0L)
  expect_match(got$conds[[1L]]$reason, "^unsupported type")
  expect_no_match(got$conds[[1L]]$reason, "names or attributes")
  expect_true(col_value_has_attrs(structure(1, label = "L")))
  expect_false(col_value_has_attrs(as.difftime(2, units = "hours")))
  expect_false(col_value_has_attrs(c(a = 1)))
})

test_that("a rename or select(new = old) moves the metadata with its column", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    a = structure(1:3, label = "A"),
    b = structure(c("x", "y", "z"), label = "B"),
    k = 1:3
  )
  c21r2_both(df, function(d) select(d, a = b))
  c21r2_both(df, function(d) rename(d, b = a, a = b))
  c21r2_both(df, function(d) select(d, b, k, a = k))
  gt <- tbl_gpu(df)
  expect_identical(.subset2(select(gt, a = b), "col_attrs")$a$attrs, list(label = "B"))
  # the unspecified marker follows the same map
  u <- tibble::tibble(a = vctrs::unspecified(2), b = 1:2)
  expect_null(.subset2(select(tbl_gpu(u), a = b), "unspecified"))
  expect_identical(.subset2(select(tbl_gpu(u), c = a), "unspecified"), "c")
})

test_that("the reason names the columns that caused the route", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1, 2, 1),
    b = stats::setNames(1:3, c("p", "q", "r")),
    d = stats::setNames(4:6, c("p", "q", "r")),
    lab = structure(1:3, label = "L")
  )
  got <- c21r2_cpu(collect(filter(tbl_gpu(df), b > 1L)))
  expect_match(got$conds[[1L]]$reason, "columns `b`, `d` have element names", fixed = TRUE)
  got <- c21r2_cpu(collect(summarise(group_by(tbl_gpu(df), b), n = n())))
  expect_match(got$conds[[1L]]$reason, "column `b` has element names", fixed = TRUE)
  expect_no_match(got$conds[[1L]]$reason, "`d`", fixed = TRUE)
})

test_that("count() and summarise() by a named column give dplyr's result", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 2, 1), b = stats::setNames(1:3, c("p", "q", "r")))
  got <- c21r2_cpu(collect(count(tbl_gpu(df), b)))
  expect_identical(got$conds[[1L]]$verb, "count")
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_equal(collect(count(gt, b)), count(df, b))
    expect_equal(collect(summarise(group_by(gt, b), n = n())), summarise(group_by(df, b), n = n()))
  }
})

test_that("collect() never fails on a column attribute R validates", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 4, 9), y = 1:3)
  lagged <- mutate(tbl_gpu(df), x = stats::lag(x))
  for (res in list(
    collect(lagged),
    collect(filter(lagged, y > 1L)),
    collect(slice(lagged, 2:3)),
    collect(distinct(lagged)),
    collect(group_by(lagged, y))
  )) {
    expect_true(is.numeric(res$x))
  }
  expect_equal(attr(collect(filter(lagged, y > 1L))$x, "tsp"), c(0, 2, 1))
})
