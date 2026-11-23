# More shapes of #30 (mutate()/transmute() errors): the call a CPU route
# reports, the group of a call that dplyr doesn't expand, `&&`, and the staged
# across() column. Same comparison as test-parity-30-error-classes.R.

p30b_df <- function() {
  tibble::tibble(
    g = c(2, 1, 2),
    s = c("b", "a", "b"),
    a = c(1L, 2L, 3L),
    b = c(0.5, NA, 2)
  )
}

p30b_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    }),
    error = function(e) e
  )
}

p30b_same_error <- function(df, pipeline, parent = TRUE, message = TRUE) {
  cpu <- p30b_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"), info = "dplyr itself errors")
  if (!inherits(cpu, "error")) {
    return(invisible())
  }
  for (lazy in c(FALSE, TRUE)) {
    gpu <- p30b_err(tbl_gpu(df, lazy = lazy), pipeline)
    mode <- if (lazy) "lazy" else "eager"
    expect_true(inherits(gpu, "error"), info = sprintf("GPU side errors (%s)", mode))
    if (!inherits(gpu, "error")) next
    expect_identical(class(gpu), class(cpu), info = sprintf("class (%s): %s", mode, conditionMessage(gpu)))
    expect_identical(rlang::format_error_call(gpu$call), rlang::format_error_call(cpu$call),
                     info = sprintf("call (%s)", mode))
    if (parent) {
      expect_identical(class(gpu$parent), class(cpu$parent), info = sprintf("parent class (%s)", mode))
    }
    if (message) {
      expect_identical(conditionMessage(gpu), conditionMessage(cpu), info = sprintf("message (%s)", mode))
    }
  }
  invisible(cpu)
}

test_that("a CPU route reports the verb's call, not dplyr::mutate()", {
  skip_if_no_gpu()
  df <- p30b_df()
  cpu <- p30b_same_error(df, function(d) mutate(d, m = stop("boom")))
  expect_identical(rlang::format_error_call(cpu$call), "`mutate()`")
  p30b_same_error(df, function(d) transmute(d, m = stop("boom")))
  p30b_same_error(df, function(d) mutate(d, m = no_such_fn(a)))
  p30b_same_error(df, function(d) d |> rowwise() |> mutate(m = no_such_col))
  p30b_same_error(df, function(d) d |> group_by(g) |> mutate(m = stop("boom")))
  # a vector constant of the wrong size is checked at lowering, also for transmute()
  p30b_same_error(df, function(d) mutate(d, m = 1:2))
  p30b_same_error(df, function(d) transmute(d, m = 1:2))
})

test_that("a mutate() inside a dot keeps its own call in the chain", {
  skip_if_no_gpu()
  df <- p30b_df()
  other <- tibble::tibble(x = 1:2)
  p30b_same_error(df, function(d) mutate(d, m = nrow(mutate(other, y = no_such_col))))
})

test_that("a named or argument-taking across() is evaluated per group", {
  skip_if_no_gpu()
  df <- p30b_df()
  # (the call is qualified: the test env doesn't attach dplyr, and dplyr looks
  # an across() it doesn't expand up in it)
  p30b_same_error(df, function(d) {
    d |> group_by(g) |> mutate(dplyr::across(no_such_col, mean, na.rm = TRUE))
  })
  p30b_same_error(df, function(d) mutate(d, dplyr::across(c(a, b), mean, na.rm = TRUE, .names = "{.col}_m"), .by = s) |>
                    mutate(z = no_such_col))
})

test_that("a staged across() and if_all() name the column in 'Can't compute'", {
  skip_if_no_gpu()
  df <- p30b_df()
  p30b_same_error(df, function(d) mutate(d, across(c(a, b), ~ .x + a + no_such_col)))
  p30b_same_error(df, function(d) transmute(d, across(c(a, b), ~ .x + a + no_such_col)))
  p30b_same_error(df, function(d) d |> group_by(g) |> mutate(m = dplyr::if_all(c(a, b), ~ .x > no_such_col)),
                  message = FALSE)
})

test_that("a column error in across() is read before .cols", {
  skip_if_no_gpu()
  df <- p30b_df()
  p30b_same_error(df, function(d) mutate(d, across(no_such_col, .x, .names = "{.col}_n")))
  p30b_same_error(df, function(d) mutate(d, m = dplyr::if_any(no_such_col, .x)))
  p30b_same_error(df, function(d) transmute(d, across(no_such_col, .x)))
})

test_that("&& in a mutate() of a column is dplyr's", {
  skip_if_no_gpu()
  df <- p30b_df()
  p30b_same_error(df, function(d) mutate(d, m = a > 1L && b > 0))
  p30b_same_error(df, function(d) d |> group_by(g) |> mutate(m = a > 1L || b > 0))
  # a one-row table: dplyr succeeds
  one <- tibble::tibble(a = 2L, b = 1)
  expect_equal(
    tibble::as_tibble(collect(mutate(tbl_gpu(one), m = a > 1L && b > 0))),
    mutate(one, m = a > 1L && b > 0)
  )
  # filter() keeps its message
  expect_error(filter(tbl_gpu(df), a > 1L && b > 0), "not supported in filter")
})

test_that("a nested loop of mutate() doesn't see an outer call's state", {
  skip_if_no_gpu()
  df <- p30b_df()
  other <- tbl_gpu(tibble::tibble(x = 1:2))
  # the nested native mutate() fails inside the outer one's dot: dplyr wraps it twice
  e <- tryCatch(mutate(tbl_gpu(df), m = nrow(collect(mutate(other, y = no_such_col)))),
                error = function(e) e)
  expect_s3_class(e, "dplyr:::mutate_error")
  # afterwards the outer verb is unaffected
  expect_equal(tibble::as_tibble(collect(mutate(tbl_gpu(df), m = a + 1L))), mutate(df, m = a + 1L))
})

test_that("summarise(.by =) errors report summarise()", {
  skip_if_no_gpu()
  df <- p30b_df()
  p30b_same_error(df, function(d) d |> group_by(g) |> summarise(n = dplyr::n(), .by = s, .groups = "rowwise"),
                  parent = FALSE, message = FALSE)
})
