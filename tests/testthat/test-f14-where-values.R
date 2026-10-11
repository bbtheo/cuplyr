# select()/relocate()/rename()/rename_with() answer a where() predicate from
# the column types only when a static check proves it a base type test;
# every other predicate (anyNA, n_distinct, mean, a user function) is
# resolved on the real data with a notified CPU fallback.

f14v_df <- function() {
  data.frame(
    x = c(1L, NA, 3L), s = c("a", "b", "c"), v = c(NA, 2.5, 1), k = c(1, 1, 1),
    d = as.Date(c("2024-01-01", "2024-01-02", NA)),
    stringsAsFactors = FALSE
  )
}

f14v_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("value predicates select the same columns as dplyr in every verb", {
  skip_if_no_gpu()
  df <- f14v_df()
  f14v_both(df, function(d) select(d, where(~ anyNA(.x))))
  f14v_both(df, function(d) select(d, where(~ !anyNA(.x))))
  f14v_both(df, function(d) select(d, where(~ dplyr::n_distinct(.x) > 1)))
  f14v_both(df, function(d) select(d, where(~ is.numeric(.x) && mean(.x, na.rm = TRUE) > 1.9)))
  f14v_both(df, function(d) select(d, where(function(col) all(col > 0, na.rm = TRUE))))
  f14v_both(df, function(d) select(d, -where(~ anyNA(.x))))
  f14v_both(df, function(d) select(d, s, where(~ anyNA(.x))))
  f14v_both(df, function(d) relocate(d, where(~ !anyNA(.x))))
  f14v_both(df, function(d) relocate(d, where(~ anyNA(.x)), .after = k))
  f14v_both(df, function(d) relocate(d, k, .before = where(~ anyNA(.x))))
  f14v_both(df, function(d) rename_with(d, toupper, .cols = where(~ anyNA(.x))))
  f14v_both(df, function(d) rename_with(d, toupper, .cols = where(~ dplyr::n_distinct(.x) == 1)))
})

test_that("value predicates see the rows of a lazy pipeline and keep grouping", {
  skip_if_no_gpu()
  df <- f14v_df()
  f14v_both(df, function(d) d |> filter(!is.na(x)) |> select(where(~ !anyNA(.x))))
  f14v_both(df, function(d) d |> mutate(z = x * 2L) |> select(where(~ anyNA(.x))))
  f14v_both(df, function(d) {
    d |> group_by(k) |> suppressMessages() |> select(where(~ anyNA(.x))) |> suppressMessages()
  })
})

test_that("a value predicate that would error on a proxy is resolved on the data", {
  skip_if_no_gpu()
  df <- f14v_df()
  # would fail on a zero-row column, fine on the real one
  f14v_both(df, function(d) select(d, where(~ is.numeric(.x) && isTRUE(.x[[1]] > 0))))
})

test_that("a value predicate is a notified fallback; type predicates are not", {
  skip_if_no_gpu()
  df <- f14v_df()
  g <- tbl_gpu(df)
  withr::local_options(cuplyr.fallback = "error")
  expect_error(select(g, where(~ anyNA(.x))), "fell back to CPU")
  expect_error(relocate(g, where(~ !anyNA(.x))), "fell back to CPU")
  expect_error(rename_with(g, toupper, .cols = where(~ anyNA(.x))), "fell back to CPU")
  expect_no_error(select(g, where(is.numeric)))
  expect_no_error(select(g, where(~ is.double(.x))))
  expect_no_error(select(g, starts_with("x"), -s))
})

test_that("a column that doesn't exist is tidyselect's own error", {
  skip_if_no_gpu()
  g <- tbl_gpu(f14v_df())
  withr::local_options(cuplyr.fallback = "error")
  expect_error(select(g, where(is.numeric), nope), class = "vctrs_error_subscript_oob")
  expect_error(relocate(g, nope), class = "vctrs_error_subscript_oob")
})

test_that("relocate() on a zero-column table still validates its selection", {
  skip_if_no_gpu()
  z <- tbl_gpu(data.frame(a = 1:3))[, 0]
  expect_error(relocate(z, nope), class = "vctrs_error_subscript_oob")
  expect_equal(nrow(collect(relocate(z))), 3L)
})

test_that("x[i, 0] counts the selected rows", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(a = 1:5, b = letters[1:5]))
  expect_equal(dim(collect(g[c(1, 3, 3, 9), 0])), dim(tibble::tibble(a = 1:5)[c(1, 3, 3, 9), 0]))
  expect_equal(nrow(collect(g[c(TRUE, FALSE, TRUE, FALSE, TRUE), 0])), 3L)
})

test_that("unique() rejects the arguments it can't honour", {
  skip_if_no_gpu()
  u <- data.frame(k = c(1, 2, 1), w = c(5, 6, 5))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(u, lazy = lazy)
    expect_error(unique(g, fromLast = TRUE), "fromLast")
    expect_error(unique(g, incomparables = 5), "incomparables")
    expect_error(unique(g, bogus = 1), "only supports")
    expect_equal(collect(unique(g, fromLast = FALSE)), unique(tibble::as_tibble(u)))
  }
})

f14v_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f14v_both(df, pipeline, ...)
}

test_that("value predicates outside any proxy's range match dplyr (round-2 B1)", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:6, b = c(1, 1, 1, 2, 2, 2),
                   s = c("x@y.com", "a", "b", "c", "d", "e"),
                   m = c(NA, NA, 1, 2, 3, 4), stringsAsFactors = FALSE)
  f14v_both(df, function(d) select(d, where(~ dplyr::n_distinct(.x) > 2)))
  f14v_both(df, function(d) select(d, where(~ dplyr::n_distinct(.x) < 4)))
  f14v_both(df, function(d) select(d, where(~ sum(is.na(.x)) > 1)))
  f14v_both(df, function(d) select(d, where(~ any(grepl("@", .x)))))
  f14v_both(df, function(d) select(d, where(~ length(.x) < 3)))
  f14v_both(df, function(d) select(d, where(~ all(.x > -10, na.rm = TRUE))))
  f14v_both(df, function(d) relocate(d, where(~ dplyr::n_distinct(.x) > 2), .after = last_col()))
  f14v_both(df, function(d) rename_with(d, toupper, .cols = where(~ dplyr::n_distinct(.x) > 2)))
  f14v_both(df, function(d) {
    suppressMessages(select(group_by(d, b), where(~ dplyr::n_distinct(.x) > 2)))
  })
})

test_that("type-test lambdas and closures stay native", {
  skip_if_no_gpu()
  df <- data.frame(
    x = 1:3, v = c(1.5, 2.5, NA), s = c("a", "b", "c"),
    d = as.Date(c("2024-01-01", NA, "2024-01-03")),
    o = factor(c("lo", "hi", "lo"), levels = c("lo", "hi"), ordered = TRUE),
    f = factor(c("u", "v", "u")), stringsAsFactors = FALSE
  )
  is_num_or_fct <- function(col) is.numeric(col) || is.factor(col)
  not_date <- function(x) {
    !inherits(x, "Date")
  }
  lam <- ~ is.character(.x) | is.ordered(.x)
  f14v_native(df, function(d) select(d, where(is.ordered)))
  f14v_native(df, function(d) select(d, where(~ is.numeric(.x) && !is.integer(.x))))
  f14v_native(df, function(d) select(d, where(function(col) inherits(col, "factor") && !is.ordered(col))))
  f14v_native(df, function(d) select(d, where(\(z) (is.logical(z) || base::is.double(z)))))
  f14v_native(df, function(d) select(d, where(is_num_or_fct)))
  f14v_native(df, function(d) select(d, where(not_date)))
  f14v_native(df, function(d) select(d, where(lam)))
  asf <- rlang::as_function(~ inherits(.x, "Date"))
  f14v_native(df, function(d) select(d, where(asf)))
  f14v_native(df, function(d) relocate(d, where(is.ordered), .after = x))
  f14v_native(df, function(d) rename_with(d, toupper, .cols = where(~ is.factor(.))))
})

test_that("predicates the static check can't prove type-only fall back", {
  skip_if_no_gpu()
  g <- tbl_gpu(f14v_df())
  withr::local_options(cuplyr.fallback = "error")
  # a user function named like a base type test
  is.numeric <- function(x) TRUE
  expect_error(select(g, where(~ is.numeric(.x))), "fell back to CPU")
  rm(is.numeric)
  # a type test applied to something other than the column
  expect_error(select(g, where(~ is.numeric(.x[1]))), "fell back to CPU")
  expect_error(select(g, where(function(x) is.numeric(1))), "fell back to CPU")
  # inherits() with a class that is not a constant
  cls <- "Date"
  expect_error(select(g, where(~ inherits(.x, cls))), "fell back to CPU")
  # a closure that does more than test the type
  f <- function(x) {
    y <- x
    is.numeric(y)
  }
  expect_error(select(g, where(f)), "fell back to CPU")
  # a call tidyselect evaluates that is not a known helper
  expect_error(select(g, identity(c("x", "s"))), "fell back to CPU")
})

test_that("predicate fallbacks give dplyr's answers", {
  skip_if_no_gpu()
  df <- f14v_df()
  is.numeric <- function(x) TRUE
  f14v_both(df, function(d) select(d, where(~ is.numeric(.x))))
  rm(is.numeric)
  f14v_both(df, function(d) select(d, identity(c("x", "s"))))
})

test_that("group_cols() sees the table's groups", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:3, b = c(1, 1, 2), c = c("x", "y", "z"), stringsAsFactors = FALSE)
  f14v_native(df, function(d) relocate(group_by(d, b), group_cols()))
  f14v_native(df, function(d) relocate(group_by(d, c, b), group_cols(), .after = a))
  f14v_native(df, function(d) rename_with(group_by(d, b), toupper, .cols = group_cols()))
  f14v_native(df, function(d) suppressMessages(select(group_by(d, b), -group_cols())))
  f14v_native(df, function(d) select(group_by(d, b), group_cols(), a))
  # and on the real data, when the selection also needs a fallback
  f14v_both(df, function(d) relocate(group_by(d, b), group_cols(), where(~ dplyr::n_distinct(.x) > 2)))
})

test_that("x[i, ] on a zero-column table keeps the selected row count", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    z <- select(tbl_gpu(data.frame(a = 1:5), lazy = lazy), -everything())
    expect_identical(dim(collect(z[-1, 0])), c(4L, 0L))
    expect_identical(dim(collect(z[-(1:2), character()])), c(3L, 0L))
    expect_identical(dim(collect(z[c(TRUE, FALSE, TRUE, FALSE, TRUE), 0])), c(3L, 0L))
    expect_identical(dim(collect(z[2:3, ])), c(2L, 0L))
    expect_identical(dim(collect(z[TRUE, ])), c(5L, 0L))
  }
})

test_that("the static check classifies predicates (no GPU)", {
  env <- rlang::current_env()
  expect_true(where_predicate_is_type_only(quote(is.numeric), env))
  expect_true(where_predicate_is_type_only(quote(base::is.ordered), env))
  expect_true(where_predicate_is_type_only(quote(~ !is.character(.x)), env))
  expect_true(where_predicate_is_type_only(quote(function(x) { inherits(x, "Date") }), env))
  expect_false(where_predicate_is_type_only(quote(~ anyNA(.x)), env))
  expect_false(where_predicate_is_type_only(quote(~ is.numeric(.x) && mean(.x) > 1), env))
  expect_false(where_predicate_is_type_only(quote(function(x, y) is.numeric(y)), env))
  expect_false(where_predicate_is_type_only(quote(~ inherits(.x, "Date", which = TRUE)), env))
  expect_false(where_predicate_is_type_only(quote(lubridate_is_date), env))
  expect_true(selection_is_type_only(rlang::quos(a, -b, starts_with("x"), 1:2, all_of(v)), NULL, c("a", "b")))
  expect_false(selection_is_type_only(rlang::quos(a, foo(b)), NULL, c("a", "b")))
})
