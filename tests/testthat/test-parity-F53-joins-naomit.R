# F53 items 6-8 acceptance tests.
#
# 6. A lazy mutating join followed by filter() must raise dplyr's advisory
#    many-to-many warning exactly as dplyr does (dplyr checks the FULL join
#    inputs, before the filter), with identical rows.
# 7. stats::na.omit(<tbl_gpu>) drops rows with any NA, like
#    na.omit.data.frame(). A tbl_gpu does not carry the `na.action` attribute.
# 8. nest_join() with a plain data frame `y` and `copy = FALSE` errors like the
#    other GPU joins; `copy = TRUE` works.

f53j_x <- function() tibble::tibble(k = c(1, 1, 2, 3), a = 1:4)
f53j_y <- function() tibble::tibble(k = c(1, 1, 2, 4), b = 5:8)

f53j_warnings <- function(expr) {
  ws <- list()
  value <- withCallingHandlers(expr, warning = function(w) {
    ws[[length(ws) + 1L]] <<- w
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = ws)
}

# Run `pipeline(x, y)` on tibbles and on lazy/eager tbl_gpus; compare rows and
# the many-to-many warning (class, message, count).
f53j_expect_join_parity <- function(pipeline) {
  want <- f53j_warnings(pipeline(f53j_x(), f53j_y()))
  for (lazy in c(FALSE, TRUE)) {
    got <- f53j_warnings(collect(pipeline(tbl_gpu(f53j_x(), lazy = lazy),
                                          tbl_gpu(f53j_y(), lazy = lazy))))
    expect_equal(tibble::as_tibble(got$value), tibble::as_tibble(want$value),
                 info = paste("lazy =", lazy))
    cls <- function(ws) vapply(ws, function(w) class(w)[1], character(1))
    expect_identical(cls(got$warnings), cls(want$warnings), info = paste("lazy =", lazy))
    expect_identical(vapply(got$warnings, conditionMessage, character(1)),
                     vapply(want$warnings, conditionMessage, character(1)),
                     info = paste("lazy =", lazy))
  }
}

# ---- item 6: relationship warning survives a later filter -----------------------

test_that("inner_join() + filter keeps dplyr's many-to-many warning", {
  skip_if_no_gpu()
  # x-side filter that removes the duplicated x key's first row
  f53j_expect_join_parity(function(x, y) filter(inner_join(x, y, by = "k"), a > 1))
  # y-side filter
  f53j_expect_join_parity(function(x, y) filter(inner_join(x, y, by = "k"), b > 5))
  # both sides and a mixed predicate
  f53j_expect_join_parity(function(x, y) filter(inner_join(x, y, by = "k"), a > 1, b > 5, a + b > 7))
})

test_that("left_join() / right_join() + filter keep dplyr's many-to-many warning", {
  skip_if_no_gpu()
  f53j_expect_join_parity(function(x, y) filter(left_join(x, y, by = "k"), a > 1))
  f53j_expect_join_parity(function(x, y) filter(right_join(x, y, by = "k"), b > 5))
  f53j_expect_join_parity(function(x, y) {
    x |> left_join(y, by = "k") |> filter(a != 2L) |> select(k, b)
  })
})

test_that("joins that can't warn keep their results under a later filter", {
  skip_if_no_gpu()
  f53j_expect_join_parity(function(x, y) {
    filter(inner_join(x, y, by = "k", relationship = "many-to-many"), a > 1, b > 5)
  })
  f53j_expect_join_parity(function(x, y) {
    filter(left_join(x, y, by = "k", multiple = "first"), a > 1)
  })
  f53j_expect_join_parity(function(x, y) {
    filter(inner_join(x, y, by = dplyr::join_by(k >= k)), a > 1)
  })
  f53j_expect_join_parity(function(x, y) {
    filter(semi_join(x, y, by = "k"), a > 1)
  })
})

# ---- item 7: na.omit() ------------------------------------------------------------

f53j_strip_na_action <- function(df) {
  attr(df, "na.action") <- NULL
  df
}

test_that("na.omit() drops rows with any NA, like na.omit.data.frame()", {
  skip_if_no_gpu()
  d <- tibble::tibble(
    a = c(1, NA, 3, NaN, 5, 6),
    b = c("x", "y", NA, "w", "v", "u"),
    i = c(1L, 2L, 3L, 4L, NA, 6L),
    l = c(TRUE, FALSE, TRUE, NA, TRUE, FALSE),
    dt = as.Date(c("2020-01-01", NA, "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06")),
    f = factor(c("p", "q", NA, "p", "q", "p"))
  )
  want <- f53j_strip_na_action(stats::na.omit(d))
  for (lazy in c(FALSE, TRUE)) {
    g <- stats::na.omit(tbl_gpu(d, lazy = lazy))
    expect_true(is_tbl_gpu(g))
    expect_identical(.subset2(g, "exec_mode"), if (lazy) "lazy" else "eager")
    got <- collect(g)
    expect_equal(got, want)
    expect_null(attr(got, "na.action"))
  }
  # through the generic, as users call it
  expect_equal(collect(na.omit(tbl_gpu(d))), want)
})

test_that("na.omit() on a table without NAs, an empty table, and a grouped table", {
  skip_if_no_gpu()
  clean <- tibble::tibble(a = c(1, 2), b = c("x", "y"))
  expect_equal(collect(na.omit(tbl_gpu(clean))), clean)
  empty <- clean[0, ]
  expect_equal(collect(na.omit(tbl_gpu(empty))), empty)
  all_na <- tibble::tibble(a = c(NA_real_, NA_real_), b = c("x", "y"))
  expect_equal(collect(na.omit(tbl_gpu(all_na))), f53j_strip_na_action(na.omit(all_na)))

  gd <- dplyr::group_by(tibble::tibble(a = c(1, NA, 3), g = c(1, 1, 2)), g)
  want <- f53j_strip_na_action(na.omit(gd))
  for (lazy in c(FALSE, TRUE)) {
    g <- na.omit(dplyr::group_by(tbl_gpu(dplyr::ungroup(gd), lazy = lazy), g))
    expect_identical(group_vars(g), "g")
    got <- collect(g)
    expect_equal(dplyr::ungroup(got), dplyr::ungroup(want))
    expect_identical(group_vars(got), group_vars(want))
  }
})

test_that("na.omit() keeps an addNA() level element (it is not NA)", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = addNA(factor(c("a", NA, "b"))), x = c(1, 2, NA))
  want <- f53j_strip_na_action(na.omit(d))
  expect_equal(collect(na.omit(tbl_gpu(d))), want)
})

# ---- item 8: nest_join() copy rule --------------------------------------------------

test_that("nest_join() with a host y and copy = FALSE errors like the other joins", {
  skip_if_no_gpu()
  x <- f53j_x()
  y <- f53j_y()
  for (lazy in c(FALSE, TRUE)) {
    gx <- tbl_gpu(x, lazy = lazy)
    want <- tryCatch(inner_join(gx, y, by = "k"), error = identity)
    expect_s3_class(want, "error")
    got <- tryCatch(nest_join(gx, y, by = "k"), error = identity)
    expect_s3_class(got, class(want)[1])
    expect_identical(conditionMessage(got), conditionMessage(want))
    # copy = TRUE and a tbl_gpu y keep working
    ok <- nest_join(gx, y, by = "k", copy = TRUE)
    ref <- nest_join(x, y, by = "k")
    expect_identical(names(ok), names(ref))
    expect_equal(lapply(ok$y, tibble::as_tibble), lapply(ref$y, tibble::as_tibble))
    ok2 <- nest_join(gx, tbl_gpu(y, lazy = lazy), by = "k")
    expect_equal(lapply(ok2[[3]], tibble::as_tibble), lapply(ref$y, tibble::as_tibble))
  }
})
