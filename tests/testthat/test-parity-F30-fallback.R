# F30 acceptance tests, part 2: recode() shapes that stay on (or may use) the
# CPU route. Nothing here is pinned native; each must give dplyr's exact
# result, warnings and errors, eager and lazy.
#
# Covered: the data-dependent "Unreplaced values treated as NA" warning
# (incompatible type, no .default), factor input and recode_factor(),
# duplicate keys (character: first wins; numeric: last wins), non-numeric
# names for a numeric .x, replacements read from columns or vectors, a user's
# own recode() masking dplyr's, across() lambdas, and dplyr's error cases.

# Tests see the package namespace, which doesn't import recode(); bind dplyr's
# own functions here, as library(dplyr) would for a user.
recode <- dplyr::recode
recode_factor <- dplyr::recode_factor

f30f_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f30f_df <- function() {
  data.frame(
    x = c("a", "b", "c", NA, "a"),
    n = c(1L, 2L, 3L, NA, 1L),
    d = c(1, 2, 3.5, NA, 1),
    y = c("p", "q", "r", "s", "t"),
    f = factor(c("a", "b", "c", NA, "a"), levels = c("c", "b", "a")),
    stringsAsFactors = FALSE
  )
}

# Run `pipeline` on the dplyr oracle and on eager/lazy tbl_gpu, collecting
# every warning message. Returns list(cpu = chr, eager = chr, lazy = chr).
f30f_warnings <- function(df, pipeline) {
  capture <- function(make) {
    msgs <- character()
    withCallingHandlers(
      collect(pipeline(make())),
      warning = function(w) {
        msgs <<- c(msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    msgs
  }
  list(
    cpu = capture(function() tibble::as_tibble(df)),
    eager = capture(function() tbl_gpu(df, lazy = FALSE)),
    lazy = capture(function() tbl_gpu(df, lazy = TRUE))
  )
}

f30f_expect_unreplaced_warning <- function(df, pipeline, expected) {
  w <- f30f_warnings(df, pipeline)
  pat <- "Unreplaced values treated as NA"
  expect_identical(any(grepl(pat, w$cpu, fixed = TRUE)), expected)
  expect_identical(any(grepl(pat, w$eager, fixed = TRUE)), expected)
  expect_identical(any(grepl(pat, w$lazy, fixed = TRUE)), expected)
}

# dplyr's error (the condition recode() raised, i.e. the mutate error's
# parent) must be reproduced on eager and lazy tbl_gpu.
f30f_expect_same_error <- function(df, pipeline) {
  inner <- function(e) if (!is.null(e$parent)) e$parent else e
  e_cpu <- tryCatch({ pipeline(tibble::as_tibble(df)); NULL }, error = identity)
  expect_s3_class(e_cpu, "error")
  for (lazy in c(FALSE, TRUE)) {
    e_gpu <- tryCatch({ collect(pipeline(tbl_gpu(df, lazy = lazy))); NULL },
                      error = identity)
    expect_s3_class(e_gpu, "error")
    if (inherits(e_gpu, "error")) {
      expect_identical(conditionMessage(inner(e_gpu)), conditionMessage(inner(e_cpu)))
      expect_identical(class(inner(e_gpu)), class(inner(e_cpu)))
    }
  }
}

test_that("numeric to character without .default: NA plus dplyr's warning", {
  skip_if_no_gpu()
  df <- f30f_df()
  p_warn <- function(t) dplyr::mutate(t, z = recode(n, `1` = "one"))
  suppressWarnings(f30f_both(df, p_warn))
  f30f_expect_unreplaced_warning(df, p_warn, TRUE)

  # Every non-NA value matched: same result type, no warning.
  p_all <- function(t) dplyr::mutate(t, z = recode(n, `1` = "one", `2` = "two", `3` = "three"))
  f30f_both(df, p_all)
  f30f_expect_unreplaced_warning(df, p_all, FALSE)

  # Positional keys, character to integer, integer to double: same rule.
  p_more <- function(t) {
    dplyr::mutate(t,
      z1 = recode(n, "first", "second"),
      z2 = recode(x, a = 1L),
      z3 = recode(n, `1` = 10, `2` = 20)
    )
  }
  suppressWarnings(f30f_both(df, p_more))
  f30f_expect_unreplaced_warning(df, p_more, TRUE)
})

test_that("factor input and recode_factor() match dplyr", {
  skip_if_no_gpu()
  f30f_both(f30f_df(), function(t) {
    dplyr::mutate(t,
      r1 = recode(f, a = "A"),
      r2 = recode(f, a = "A", b = "A"),
      r3 = recode(f, a = 1L, .default = 0L),
      r4 = recode(f, a = "A", .default = "other"),
      r5 = recode_factor(x, b = "B", a = "A"),
      r6 = recode_factor(n, `2` = "two", `1` = "one", .default = "other", .ordered = TRUE)
    )
  })
})

test_that("duplicate keys: character first wins, numeric last wins", {
  skip_if_no_gpu()
  f30f_both(f30f_df(), function(t) {
    dplyr::mutate(t,
      c_dup = recode(x, a = "first", a = "second"),
      n_dup = recode(n, `1` = 10L, `1` = 11L),
      n_dup2 = recode(d, `1` = 5, `1.0` = 6)
    )
  })
})

test_that("non-numeric names for numeric x match nothing (with R's warning)", {
  skip_if_no_gpu()
  df <- f30f_df()
  p <- function(t) dplyr::mutate(t, z = recode(n, abc = 10L, `2` = 20L))
  suppressWarnings(f30f_both(df, p))
  w <- f30f_warnings(df, p)
  pat <- "NAs introduced by coercion"
  expect_true(any(grepl(pat, w$cpu, fixed = TRUE)))
  expect_true(any(grepl(pat, w$eager, fixed = TRUE)))
  expect_true(any(grepl(pat, w$lazy, fixed = TRUE)))
  # A non-whole key for an integer x matches nothing either.
  f30f_both(df, function(t) dplyr::mutate(t, z = recode(n, `1.5` = 10L, `2` = 20L)))
})

test_that("replacements taken from columns or full-length vectors", {
  skip_if_no_gpu()
  f30f_both(f30f_df(), function(t) {
    dplyr::mutate(t,
      z1 = recode(x, a = y),
      z2 = recode(x, a = "A", .default = y),
      z3 = recode(x, b = "B", .missing = y)
    )
  })
})

test_that("a user's own recode() is called, not dplyr's", {
  skip_if_no_gpu()
  recode <- function(.x, ...) paste0(.x, "!")
  f30f_both(f30f_df(), function(t) dplyr::mutate(t, z = recode(x, a = "A")))
})

test_that("recode() in an across() lambda", {
  skip_if_no_gpu()
  df <- data.frame(p = c("a", "b", NA), q = c("b", "a", "c"), stringsAsFactors = FALSE)
  f30f_both(df, function(t) {
    dplyr::mutate(t, dplyr::across(c(p, q), ~ recode(.x, a = "A", b = "B")))
  })
  f30f_both(df, function(t) {
    dplyr::mutate(t, dplyr::across(c(p, q), ~ recode(.x, a = 1L, .default = 0L), .names = "{.col}_i"))
  })
})

test_that("non-column .x expressions match dplyr", {
  skip_if_no_gpu()
  f30f_both(f30f_df(), function(t) {
    dplyr::mutate(t,
      z1 = recode(toupper(x), A = "upper-a"),
      z2 = recode(n + 1L, `2` = 0L)
    )
  })
})

test_that("dplyr's recode() errors are reproduced", {
  skip_if_no_gpu()
  df <- f30f_df()
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x, a = "A", b = 1)))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x, a = "A", b = NA)))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x, a = 1L, b = 2.5)))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x, "A")))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(n, `1` = "x", "y")))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x)))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(d, `1` = 0, .missing = -1L)))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x, a = "A", .default = 1L)))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(f, a = "A", .missing = "M")))
  f30f_expect_same_error(df, function(t) dplyr::mutate(t, z = recode(x == "a", `TRUE` = "t")))
})
