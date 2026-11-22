# Acceptance tests for #30: mutate()/transmute() errors have dplyr's classes.
#
# Contract (dplyr 1.2.1, R/mutate.R mutate_cols() + R/conditions.R
# dplyr_error_handler()):
# - An error raised while a mutate()/transmute() dot is expanded or evaluated
#   is re-raised as a `dplyr:::mutate_error` whose parent is the original error,
#   with an "In argument: `<label>`." bullet and, when the error happened while a
#   group was being evaluated (grouped or `.by`, at least one row), an
#   "In group 1: `<keys>`." bullet. The call is the mutate()/transmute() call.
# - An error evaluating an across() expansion has an extra "Can't compute column"
#   layer; a tidyselect error of across()/if_any() has no group bullet.
# - `.before`/`.after` are always resolved (also with no new columns); their
#   errors are tidyselect's, unwrapped, from relocate().
# - Errors dplyr doesn't wrap stay unwrapped, and errors that already are dplyr's
#   (CPU fallback, rowwise rerun) are not wrapped twice.

p30_df <- function() {
  tibble::tibble(
    g = c(2, 1, 2),
    s = c("b", "a", "b"),
    f = factor(c("y", "x", "y")),
    dt = as.Date(c("2024-03-02", "2024-01-05", "2024-03-02")),
    a = c(1L, 2L, 3L),
    b = c(0.5, NA, 2)
  )
}

# The condition `pipeline(x)` raises (collecting a tbl_gpu result), or NULL.
p30_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    }),
    error = function(e) e
  )
}

# The bullets of an rlang error's own message (no parent), as plain strings:
# rlang keeps the first in `message` and the rest (the group bullet) in `body`.
p30_bullets <- function(e) {
  m <- c(e$message, if (is.character(e$body)) e$body)
  stats::setNames(as.character(m), names(m))
}

p30_call <- function(e) {
  call <- e$call
  if (is.null(call)) NA_character_ else rlang::format_error_call(call)
}

# dplyr and cuplyr (eager and lazy) raise the same condition: same class, same
# call, same message bullets and, unless switched off, the same parent class and
# the same full message (rlang's, which includes the parent chain).
p30_same_error <- function(df, pipeline, parent = TRUE, message = TRUE) {
  cpu <- p30_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"), info = "dplyr itself errors")
  if (!inherits(cpu, "error")) {
    return(invisible())
  }
  for (lazy in c(FALSE, TRUE)) {
    gpu <- p30_err(tbl_gpu(df, lazy = lazy), pipeline)
    mode <- if (lazy) "lazy" else "eager"
    expect_true(inherits(gpu, "error"), info = sprintf("GPU side errors (%s)", mode))
    if (!inherits(gpu, "error")) next
    expect_identical(class(gpu), class(cpu),
                     info = sprintf("error class (%s): %s", mode, conditionMessage(gpu)))
    expect_identical(p30_call(gpu), p30_call(cpu), info = sprintf("error call (%s)", mode))
    expect_identical(p30_bullets(gpu), p30_bullets(cpu), info = sprintf("bullets (%s)", mode))
    if (parent) {
      expect_identical(class(gpu$parent), class(cpu$parent),
                       info = sprintf("parent class (%s)", mode))
    }
    if (message) {
      expect_identical(conditionMessage(gpu), conditionMessage(cpu),
                       info = sprintf("full message (%s)", mode))
    }
  }
  invisible(cpu)
}

p30_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# =============================================================================
# 1. Evaluation errors inside mutate()/transmute()
# =============================================================================

test_that("an unknown object in an ungrouped mutate() is a dplyr:::mutate_error", {
  skip_if_no_gpu()
  df <- p30_df()
  cpu <- p30_same_error(df, function(d) mutate(d, m1 = no_such_col))
  # the contract itself, independent of the comparison
  expect_s3_class(cpu, "dplyr:::mutate_error")
  p30_same_error(df, function(d) mutate(d, m1 = no_such_col + 1))
  p30_same_error(df, function(d) mutate(d, m1 = a + no_such_col))
  p30_same_error(df, function(d) mutate(d, no_such_col * 2))
  p30_same_error(df, function(d) mutate(d, a2 = a, m1 = no_such_col))
  p30_same_error(df, function(d) mutate(d, m1 = dplyr::if_else(a > 1L, no_such_col, 0)))
})

test_that("a grouped mutate() error names group 1 (sorted groups)", {
  skip_if_no_gpu()
  df <- p30_df()
  cpu <- p30_same_error(df, function(d) d |> group_by(g) |> mutate(m1 = no_such_col))
  expect_length(p30_bullets(cpu), 2L)
  p30_same_error(df, function(d) d |> group_by(s) |> mutate(m1 = no_such_col + 1))
  p30_same_error(df, function(d) d |> group_by(f) |> mutate(m1 = no_such_col))
  p30_same_error(df, function(d) d |> group_by(dt) |> mutate(m1 = no_such_col))
  p30_same_error(df, function(d) d |> group_by(g, s) |> mutate(m1 = no_such_col))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(a2 = a * 2L, m1 = no_such_col))
})

test_that("a mutate(.by =) error names group 1 (first appearance)", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, m1 = no_such_col, .by = g))
  p30_same_error(df, function(d) mutate(d, m1 = no_such_col * a, .by = c(s, g)))
})

test_that("a grouped error after pending steps names the group of the filtered data", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) {
    d |> filter(a > 1L) |> group_by(g) |> mutate(m1 = no_such_col)
  })
  p30_same_error(df, function(d) {
    d |> mutate(h = a %% 2L) |> group_by(h) |> mutate(m1 = no_such_col)
  })
})

test_that("a grouped table with no rows has no group bullet", {
  skip_if_no_gpu()
  df <- p30_df()[0, ]
  cpu <- p30_same_error(df, function(d) d |> group_by(g) |> mutate(m1 = no_such_col))
  expect_length(p30_bullets(cpu), 1L)
  p30_same_error(df, function(d) mutate(d, m1 = no_such_col, .by = g))
  p30_same_error(df, function(d) mutate(d, m1 = no_such_col))
})

test_that("transmute() errors are dplyr:::mutate_error with the transmute() call", {
  skip_if_no_gpu()
  df <- p30_df()
  cpu <- p30_same_error(df, function(d) transmute(d, m1 = no_such_col))
  expect_s3_class(cpu, "dplyr:::mutate_error")
  p30_same_error(df, function(d) transmute(d, a, m1 = no_such_col + a))
  p30_same_error(df, function(d) d |> group_by(g) |> transmute(m1 = no_such_col))
  p30_same_error(df, function(d) d |> group_by(s) |> transmute(a2 = a, m1 = no_such_col * 2))
})

test_that("a missing .data column is wrapped with the argument and group", {
  skip_if_no_gpu()
  df <- p30_df()
  # (the parent's own message is cuplyr's; class, call and bullets are dplyr's)
  p30_same_error(df, function(d) mutate(d, m1 = .data$no_such_col),
                 parent = FALSE, message = FALSE)
  p30_same_error(df, function(d) mutate(d, m1 = .data[["no_such_col"]] + 1),
                 parent = FALSE, message = FALSE)
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(m1 = .data$no_such_col),
                 parent = FALSE, message = FALSE)
  p30_same_error(df, function(d) transmute(d, m1 = .data$no_such_col),
                 parent = FALSE, message = FALSE)
})

test_that("an unknown object inside a window call is wrapped", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, m1 = cumsum(no_such_col)))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(m1 = dplyr::lag(no_such_col)))
  p30_same_error(df, function(d) mutate(d, m1 = mean(no_such_col), .by = s))
})

test_that("an error evaluating an across() expansion has dplyr's chain", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, across(a, ~ .x + no_such_col)))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(across(c(a, b), ~ .x * no_such_col)))
  p30_same_error(df, function(d) transmute(d, across(a, ~ .x - no_such_col)))
})

# =============================================================================
# 2. across() selection errors
# =============================================================================

test_that("across(no_such_col) is wrapped as dplyr wraps it", {
  skip_if_no_gpu()
  df <- p30_df()
  cpu <- p30_same_error(df, function(d) mutate(d, across(no_such_col, ~ .x + 1)))
  expect_s3_class(cpu, "dplyr:::mutate_error")
  expect_s3_class(cpu$parent, "vctrs_error_subscript_oob")
  p30_same_error(df, function(d) mutate(d, across(c(a, no_such_col), ~ .x * 2)))
  p30_same_error(df, function(d) mutate(d, across(no_such_col, .x, .names = "{.col}_n")))
  p30_same_error(df, function(d) mutate(d, a2 = a, across(no_such_col, ~ .x + 1)))
  p30_same_error(df, function(d) transmute(d, across(no_such_col, ~ .x + 1)))
})

test_that("a grouped across() selection error has no group bullet", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(across(no_such_col, ~ .x + 1)))
  p30_same_error(df, function(d) mutate(d, across(no_such_col, ~ .x + 1), .by = s))
  p30_same_error(df, function(d) d |> group_by(g) |> transmute(across(no_such_col, ~ .x + 1)))
})

test_that("if_any()/if_all() selection errors inside mutate() are wrapped", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, m = dplyr::if_any(no_such_col, is.na)))
  p30_same_error(df, function(d) mutate(d, m = dplyr::if_all(c(a, no_such_col), ~ .x > 0)))
})

# =============================================================================
# 3. .before / .after
# =============================================================================

test_that("mutate(.after/.before = no_such_col) errors like dplyr with no new column", {
  skip_if_no_gpu()
  df <- p30_df()
  cpu <- p30_same_error(df, function(d) mutate(d, .after = no_such_col))
  expect_s3_class(cpu, "vctrs_error_subscript_oob")
  p30_same_error(df, function(d) mutate(d, .before = no_such_col))
  p30_same_error(df, function(d) mutate(d, a = a + 1L, .after = no_such_col))
  p30_same_error(df, function(d) mutate(d, a = a + 1L, .before = c(b, no_such_col)))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(.after = no_such_col))
  p30_same_error(df, function(d) mutate(d, .after = no_such_col, .keep = "none"))
})

test_that("mutate(.after/.before = no_such_col) errors like dplyr with new columns", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, n1 = a + 1L, .after = no_such_col))
  p30_same_error(df, function(d) mutate(d, n1 = a + 1L, .before = no_such_col))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(n1 = cumsum(a), .after = no_such_col))
  p30_same_error(df, function(d) mutate(d, n1 = a * 2L, .by = g, .before = no_such_col))
})

test_that("both .before and .after is dplyr's relocate() error", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, n1 = a, .before = a, .after = b))
  p30_same_error(df, function(d) mutate(d, .before = a, .after = b))
})

test_that("valid .before/.after still match dplyr and stay on the GPU", {
  skip_if_no_gpu()
  df <- p30_df()
  withr::local_options(cuplyr.fallback = "error")
  p30_both(df, function(d) mutate(d, .after = a))
  p30_both(df, function(d) mutate(d, .before = b))
  p30_both(df, function(d) mutate(d, a = a + 1L, .before = s))
  p30_both(df, function(d) mutate(d, n1 = a + 1L, .after = a))
  p30_both(df, function(d) d |> group_by(g) |> mutate(n1 = a * 2L, .before = 1))
  p30_both(df, function(d) mutate(d, .after = c(a, b)))
})

# =============================================================================
# Guards: what dplyr doesn't wrap, and errors that already are dplyr's
# =============================================================================

test_that("errors dplyr doesn't wrap stay unwrapped", {
  skip_if_no_gpu()
  df <- p30_df()
  cpu <- p30_same_error(df, function(d) d |> group_by(g) |> mutate(m = a, .by = s))
  expect_false(inherits(cpu, "dplyr:::mutate_error"))
  p30_same_error(df, function(d) transmute(d, m = a, .keep = "all"))
  p30_same_error(df, function(d) transmute(d, m = a, .after = b))
  p30_same_error(df, function(d) mutate(d, m = a, .by = no_such_col))
})

test_that("errors dplyr raises on a CPU route are not wrapped twice", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, m = stop("boom")))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(m = stop("boom")))
  p30_same_error(df, function(d) transmute(d, m = stop("boom")))
  p30_same_error(df, function(d) d |> rowwise() |> mutate(m = no_such_col))
  p30_same_error(df, function(d) d |> rowwise(g) |> mutate(m = a + no_such_col))
  p30_same_error(df, function(d) mutate(d, m = 1:2))
})

test_that("a caught error inside a dot does not leave mutate() wrapping later errors", {
  skip_if_no_gpu()
  df <- p30_df()
  # an error raised and handled by the user's own code inside a dot is not the
  # verb's error; the call succeeds as in dplyr
  safe <- function(x) tryCatch(stop("inner"), error = function(e) x)
  p30_both(df, function(d) mutate(d, m = safe(a)))
  # after a wrapped error, the next valid call is unaffected
  gpu <- tbl_gpu(df)
  expect_error(mutate(gpu, m = no_such_col), class = "dplyr:::mutate_error")
  expect_equal(
    tibble::as_tibble(collect(mutate(gpu, m = a + 1L))),
    mutate(df, m = a + 1L)
  )
  expect_error(relocate(gpu, no_such_col), class = "vctrs_error_subscript_oob")
})

test_that("valid mutate()/transmute() calls stay on the GPU", {
  skip_if_no_gpu()
  df <- p30_df()
  withr::local_options(cuplyr.fallback = "error")
  p30_both(df, function(d) mutate(d, m1 = a + 1L, m2 = b * 2))
  p30_both(df, function(d) d |> group_by(g) |> mutate(m1 = cumsum(a)))
  p30_both(df, function(d) mutate(d, m1 = a * 2L, .by = s))
  p30_both(df, function(d) transmute(d, m1 = a + 1L, b))
  p30_both(df, function(d) mutate(d, across(c(a, b), ~ .x * 2)))
})

# =============================================================================
# Added after review: shapes round 1 got wrong
# =============================================================================

test_that("a grouped table with no rows has no group bullet, also with empty groups", {
  skip_if_no_gpu()
  df0 <- tibble::tibble(f = factor(character(), levels = c("x", "y")), a = integer())
  p30_same_error(df0, function(d) d |> group_by(f, .drop = FALSE) |> mutate(m = no_such_col))
  df <- tibble::tibble(g = c(1, 2), a = 1:2)
  p30_same_error(df, function(d) {
    d |> group_by(g) |> filter(a > 5L, .preserve = TRUE) |> mutate(m = no_such_col)
  })
})

test_that("a staged across() names the column, not a temporary", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, across(c(a, b), ~ .x + a + no_such_col)))
})

test_that("if_any()/if_all() errors are evaluated per group, as in dplyr", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(m = dplyr::if_any(no_such_col, is.na)))
  p30_same_error(df, function(d) mutate(d, m = dplyr::if_all(no_such_col, is.na), .by = s))
  p30_same_error(df, function(d) d |> group_by(g) |> mutate(dplyr::across(no_such_col, mean, na.rm = TRUE)))
})

test_that("an if_any() evaluation error has dplyr's class chain", {
  skip_if_no_gpu()
  df <- p30_df()
  # (the innermost call text of dplyr's chain is not compared)
  p30_same_error(df, function(d) mutate(d, m = dplyr::if_any(c(a, b), ~ .x > no_such_col)),
                 message = FALSE)
})

test_that("&& / || in mutate() behave as in dplyr", {
  skip_if_no_gpu()
  df <- p30_df()
  p30_same_error(df, function(d) mutate(d, m = a > 1L && b > 0))
  p30_same_error(df, function(d) transmute(d, m = a > 1L || b > 0))
  one <- tibble::tibble(a = 2L, b = 1)
  p30_both(one, function(d) mutate(d, m = a > 1L && b > 0))
})
