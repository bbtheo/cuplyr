# F124 (attempt 2, round-4 review N6): when a group is empty, dplyr computes the
# groups that have rows with integer arithmetic (NA on integer overflow, NA for
# `%/%`/`%%` by 0L) and only then makes the column a double (vctrs combines the
# empty group's double with the integer chunks). So an integer window min()/max()
# expression keeps its integer VALUES and gets dplyr's TYPE.
# Helper: helper-parity-F124.R.

f124r_values <- function(df, pipeline, info) {
  want <- suppressWarnings(oracle_normalize(pipeline(tibble::as_tibble(df))))
  for (lazy in c(FALSE, TRUE)) {
    lbl <- paste(info, if (lazy) "[lazy]" else "[eager]")
    fb <- character()
    got <- withCallingHandlers(
      suppressWarnings(oracle_normalize(collect(pipeline(tbl_gpu(df, lazy = lazy))))),
      cuplyr_cpu_fallback = function(c) fb <<- c(fb, conditionMessage(c))
    )
    expect_identical(f124_col_types(got), f124_col_types(want), info = paste(lbl, "types"))
    expect_identical(got, want, info = paste(lbl, "values"))
    expect_identical(fb, character(), info = paste(lbl, "CPU fallback"))
  }
}

test_that("F124 N6: integer overflow in a group with rows stays NA when another group is empty", {
  skip_if_no_gpu()
  big <- .Machine$integer.max
  d1 <- tibble::tibble(f = factor(c("a", "a"), levels = c("a", "b")), x = c(1L, big))
  grp <- function(d) group_by(d, f, .drop = FALSE)
  f124r_values(d1, function(d) mutate(grp(d), m = max(x) + 1L), "max(x) + 1L")
  f124r_values(d1, function(d) mutate(grp(d), m = min(x) - 2L - big), "min(x) - 2L - big")
  d2 <- tibble::tibble(f = factor(c("a", "a", "a"), levels = c("a", "b")), x = c(1L, 2L, 1000000000L))
  f124r_values(d2, function(d) mutate(grp(d), m = max(x) * n()), "max(x) * n()")
  f124r_values(d2, function(d) mutate(grp(d), m = max(x) * 3L), "max(x) * 3L")
})

test_that("F124 N6: %/% and %% by 0L give NA in a group with rows when another group is empty", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), x = c(1L, 4L, 7L))
  grp <- function(d) group_by(d, f, .drop = FALSE)
  f124r_values(d, function(d) mutate(grp(d), m = max(x) %/% 0L), "max(x) %/% 0L")
  f124r_values(d, function(d) mutate(grp(d), m = min(x) %% 0L), "min(x) %% 0L")
  f124r_values(d, function(d) mutate(grp(d), m = max(x) %/% 2L, r = min(x) %% 3L), "%/% and %% by non-zero")
})

test_that("F124 N6: a later dot reads the double column with double semantics", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), x = c(1L, 4L, 7L))
  grp <- function(d) group_by(d, f, .drop = FALSE)
  # (m is a double in every group once the column is combined, so m %/% 0L is
  # Inf in dplyr, and m + 1L cannot overflow)
  f124r_values(d, function(d) mutate(grp(d), m = max(x), k = m %/% 0L), "k = m %/% 0L")
  f124r_values(d, function(d) mutate(grp(d), m = max(x), k = m + 1L), "k = m + 1L")
})

test_that("F124 N6: dplyr's type and warning for each root kind, with an empty group", {
  skip_if_no_gpu()
  d <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")),
                      x = c(1L, 4L, 7L), l = c(TRUE, FALSE, TRUE))
  grp <- function(d) group_by(d, f, .drop = FALSE)
  f124_check(d, function(d) mutate(grp(d), m = max(x) + 1L))
  f124_check(d, function(d) mutate(grp(d), m = -min(x)))
  f124_check(d, function(d) mutate(grp(d), m = abs(max(x) - 10L)))
  f124_check(d, function(d) mutate(grp(d), m = max(l) + 0L))
  f124_check(d, function(d) mutate(grp(d), m = x - min(x)))
  # roots whose type does not depend on the group: no double
  f124_check(d, function(d) mutate(grp(d), m = x == max(x)))
  f124_check(d, function(d) mutate(grp(d), m = max(x) > 2L))
  f124_check(d, function(d) mutate(grp(d), m = is.na(max(x))))
  # zero-row table, eager and lazy
  d0 <- d[0, ]
  f124_check(d0, function(d) mutate(d, m = max(x) %/% 0L))
  f124_check(d0, function(d) mutate(d, m = min(x) + 1L, k = m * 2L))
})
