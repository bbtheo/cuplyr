# Issue #17: common subexpression elimination in fused mutate() and filter().
#
# A subtree that the expression evaluator would materialize more than once in
# one evaluation (a handler such as round() or %in%, a checked integer chain,
# a whole dot) is computed once and its column reused. Cheap arithmetic that
# libcudf's AST inlines is deliberately left alone (hoisting it measured
# slower). Results, warnings and errors are unchanged, so every case compares
# with dplyr; `gpu_cse_reuse_count()` (a cumulative session counter of reuses)
# shows whether the sharing happened.

cse_df <- function() {
  # (-0 is made at run time: R's byte compiler pools constants with
  # identical(), which treats -0 and 0 as equal)
  zeros <- c(0, 0)
  zeros[1] <- zeros[1] * -1
  data.frame(
    x = c(zeros, 1.234, NA, NaN, -2.5, 0.75, 0.31, 0.5, 0.06),
    y = c(1, 2, 3, 4, 5, 6, 7, 8, NA, 10),
    i = c(1L, -3L, NA, 7L, 100L, 5L, 1L, 77L, -2L, 0L),
    j = c(2L, 3L, 4L, NA, -5L, 6L, 7L, 8L, 9L, 10L),
    g = c("a", "b", "a", "b", "a", "b", "a", "b", "a", "b"),
    s = c("ab", "xyz", NA, "", "hello", "ab", "q", "zz", "abc", "x"),
    stringsAsFactors = FALSE
  )
}

big_df <- function() {
  data.frame(
    bi = c(2000000000L, 1L, NA, -2000000000L, 5L),
    bj = c(200000000L, 2L, 3L, -200000000L, NA)
  )
}

# Reuses counted while `expr` runs (lazy pipelines evaluate at collect()).
cse_reuses <- function(expr) {
  n0 <- gpu_cse_reuse_count()
  force(expr)
  gpu_cse_reuse_count() - n0
}

test_that("a repeated round() across dots is computed once (eager and lazy)", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(x, 2) + 1, b = round(x, 2) * 2)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)

  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = TRUE)))), 1)
})

test_that("three uses reuse twice", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(x, 1) + 1, b = round(x, 1) * y, c = round(x, 1) - y)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 2)
})

test_that("a repeat inside one dot is computed once", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(x, 2) * round(x, 2) + round(x, 2))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 2)
})

test_that("two identical dots share one computation", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(x, 2), b = round(x, 2))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeated %in% inside if_else() is computed once", {
  skip_if_no_gpu()
  df <- cse_df()
  st <- c(1L, 5L, 77L)
  p <- function(d) mutate(d, a = dplyr::if_else(i %in% st, x, y), b = dplyr::if_else(i %in% st, 1L, 0L))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeated string handler is computed once", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = nchar(s) + 1L, b = nchar(s) * 2L)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeated checked integer subtree is computed once", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = (i + j) * 2L, b = (i + j) / 3)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("an overflowing repeated integer subtree warns exactly once", {
  skip_if_no_gpu()
  df <- big_df()
  p <- function(d) mutate(d, a = (bi + bj) * 1L, b = (bi + bj) - 0L)
  want <- suppressWarnings(p(tibble::as_tibble(df)))

  for (lazy in c(FALSE, TRUE)) {
    msgs <- character()
    n <- cse_reuses(
      got <- withCallingHandlers(
        collect(p(tbl_gpu(df, lazy = lazy))),
        warning = function(w) {
          msgs <<- c(msgs, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      )
    )
    expect_equal(tibble::as_tibble(got), want, info = paste("lazy =", lazy))
    expect_length(msgs, 1)
    expect_match(msgs, "NAs produced by integer overflow", fixed = TRUE)
    expect_equal(n, 1, info = paste("lazy =", lazy))
  }
})

test_that("a repeated checked chain past the fusion cap is shared at its root", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = (i + j + i + j + i) * 2L, b = (i + j + i + j + i) - 1L)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("filter() predicates share a repeated subtree", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) filter(d, round(x, 1) > 0.2, round(x, 1) < 0.9)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = TRUE)))), 1)
})

test_that("an integer subtree that every use fuses and that can't overflow stays fused", {
  skip_if_no_gpu()
  df <- cse_df()
  # (each use is one fused kernel on the proven-range path; a shared column
  # would add a write and a read)
  p <- function(d) filter(d, (i + j) * 2L > 0L, (i + j) - 5L < 100L)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
  pm <- function(d) mutate(d, s = (i + j) * 2L, t = (i + j) - i)
  expect_same_as_dplyr(df, pm)
  expect_same_as_dplyr_lazy(df, pm)
  expect_equal(cse_reuses(collect(pm(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("filter() shares an integer subtree whose range can overflow", {
  skip_if_no_gpu()
  df <- big_df()
  p <- function(d) filter(d, (bi + bj) * 1L > 0L, (bi + bj) - 5L < 100L)
  want <- suppressWarnings(p(tibble::as_tibble(df)))
  for (lazy in c(FALSE, TRUE)) {
    msgs <- character()
    n <- cse_reuses(
      got <- withCallingHandlers(
        collect(p(tbl_gpu(df, lazy = lazy))),
        warning = function(w) {
          msgs <<- c(msgs, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      )
    )
    expect_equal(tibble::as_tibble(got), want, info = paste("lazy =", lazy))
    expect_length(msgs, 1)
    expect_match(msgs, "NAs produced by integer overflow", fixed = TRUE)
    expect_equal(n, 1, info = paste("lazy =", lazy))
  }
})

test_that("transmute() shares a repeated subtree", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) transmute(d, a = round(x, 2) + 1, b = round(x, 2) * 2)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("lazy mutate() calls fused into one block share a repeat", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) d |> mutate(a = round(x, 2) + 1) |> mutate(b = round(x, 2) * 2)
  expect_same_as_dplyr_lazy(df, p)
  expect_same_as_dplyr(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = TRUE)))), 1)
})

test_that("grouped mutate() with a window shares the repeat", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) {
    d |>
      group_by(g) |>
      mutate(
        a = round(x - sum(y, na.rm = TRUE), 2) + 1,
        b = round(x - sum(y, na.rm = TRUE), 2) * 2
      )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_gte(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("zero-row inputs give dplyr's result", {
  skip_if_no_gpu()
  df <- cse_df()[0, ]
  pm <- function(d) mutate(d, a = round(x, 2) + 1, b = round(x, 2) * 2, c = (i + j) * 2L, e = (i + j) - 1L)
  pf <- function(d) filter(d, round(x, 1) > 0.2, round(x, 1) < 0.9)
  expect_same_as_dplyr(df, pm)
  expect_same_as_dplyr_lazy(df, pm)
  expect_same_as_dplyr(df, pf)
  expect_same_as_dplyr_lazy(df, pf)
})

test_that("cheap arithmetic that the AST inlines is not hoisted", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = (x + y) * 2, b = (x + y) / 3)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("0 and -0 literals are not merged", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = 1 / round(x + 0, 2), b = 1 / round(x + -0, 2))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  got <- collect(p(tbl_gpu(df, lazy = FALSE)))
  expect_identical(got$a[1:2], c(Inf, Inf))
  expect_identical(got$b[1:2], c(-Inf, Inf))
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("NA and NaN literals are not merged", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(y + NA_real_, 2), b = round(y + NaN, 2))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  got <- collect(p(tbl_gpu(df, lazy = FALSE)))
  expect_true(all(is.na(got$a) & !is.nan(got$a)))
  expect_true(all(is.nan(got$b[!is.na(df$y)])))
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("a column overwritten between two uses is not merged", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(x, 2), x = x + 1, b = round(x, 2))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("different digits are different subtrees", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) mutate(d, a = round(x, 1) + 1, b = round(x, 2) + 1)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("a -0 literal from the environment is not merged with 0", {
  skip_if_no_gpu()
  df <- cse_df()
  mz <- 0
  mz <- mz * -1
  p <- function(d) mutate(d, a = 1 / round(x + 0, 2), b = 1 / round(x + mz, 2))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  got <- collect(p(tbl_gpu(df, lazy = FALSE)))
  expect_identical(got$b[1:2], c(-Inf, Inf))
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
})

test_that("calls that differ only in their metadata are not merged", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 5, NA, 2), y = c(3, 2, 4, NA))
  p <- function(d) {
    d |>
      rowwise() |>
      mutate(s = sum(x, y, na.rm = TRUE), m = max(x, y, na.rm = TRUE)) |>
      ungroup()
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("case_when() conditions and values share a repeated round()", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) {
    mutate(d, a = dplyr::case_when(
      round(x, 1) > 0.5 ~ round(x, 1),
      round(x, 1) < 0 ~ -round(x, 1),
      .default = round(x, 1) * 2
    ))
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 4)
})

test_that("between() of a handler shares it", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) filter(d, dplyr::between(round(x, 1), 0.2, 0.8))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_reuses(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("summarise() temp columns over a shared subtree give dplyr's result", {
  skip_if_no_gpu()
  df <- cse_df()
  p <- function(d) {
    summarise(d, s = sum(round(x, 1) > 0.3, na.rm = TRUE), a = any(round(x, 1) > 0.3, na.rm = TRUE))
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})
