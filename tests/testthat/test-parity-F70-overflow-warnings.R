# F70 acceptance tests: the integer-overflow warning in lazy plans.
#
# dplyr (1.2.1, verified on tibbles) evaluates every expression of a verb on
# the rows that verb receives: R's integer `+`, `-`, `*` give NA with the
# warning "NAs produced by integer overflow" (dplyr wraps it in "There was 1
# warning in `mutate()`") for any row whose result leaves
# +-.Machine$integer.max, whether or not a later verb drops that row or column.
# All predicates of ONE filter() call are evaluated on all of its input rows;
# a later filter() call sees only the rows the earlier ones kept. A join's
# output holds only matched rows (inner/semi/anti/cross), so a predicate after
# it never sees the dropped ones.
#
# Eager cuplyr already follows this (F42). A lazy plan goes through the
# optimizer, whose rewrites changed which rows or columns a checked integer
# expression was evaluated on, so the warning vanished or appeared in lazy mode
# only (values were always the same):
# - filter pushdown below a mutate (warning lost),
# - dead-column pruning of a mutate expression (warning lost),
# - merging filter() calls (warning gained),
# - pushdown of a predicate into a join input whose rows don't all reach the
#   join output (warning gained).
# Every check below asserts dplyr's own behaviour first, then lazy (and eager)
# cuplyr's warning, values and column classes.

F70_M <- .Machine$integer.max

f70_df <- function() {
  data.frame(
    a = c(F70_M, 1L, 5L, 2L),
    b = c(1L, 2L, 3L, 4L),
    keep = c(FALSE, TRUE, TRUE, TRUE),
    x = c(1, 2, 3, 4),
    g = c("p", "p", "q", "q")
  )
}

# Run `f()` (collecting a tbl_gpu result inside the handler, so a lazy plan's
# warnings are caught) and record whether it warned about integer overflow.
f70_run <- function(f) {
  msgs <- character()
  val <- withCallingHandlers(
    {
      r <- f()
      if (is_tbl_gpu(r)) r <- collect(r)
      r
    },
    warning = function(w) {
      msgs <<- c(msgs, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(val = val, overflow = any(grepl("integer overflow", msgs, fixed = TRUE)))
}

# `pipeline(d, src)`: `src(df)` turns a second data.frame into the same kind of
# table as `d` (tibble, eager tbl_gpu or lazy tbl_gpu), for joins.
f70_check <- function(df, pipeline, warns, arrange_by = NULL) {
  as_tbl <- function(x) tibble::as_tibble(x)
  as_eager <- function(x) tbl_gpu(x)
  as_lz <- function(x) as_lazy(tbl_gpu(x))

  oracle <- f70_run(function() pipeline(as_tbl(df), as_tbl))
  expect_identical(oracle$overflow, warns,
                   label = "dplyr's integer-overflow warning (test premise)")
  want <- oracle_normalize(oracle$val, arrange_by)

  for (mode in c("lazy", "eager")) {
    src <- if (mode == "lazy") as_lz else as_eager
    got <- f70_run(function() pipeline(src(df), src))
    expect_identical(got$overflow, warns,
                     label = sprintf("%s cuplyr integer-overflow warning", mode))
    got_tbl <- oracle_normalize(got$val, arrange_by)
    expect_equal(got_tbl, want, label = sprintf("%s cuplyr values", mode))
    expect_identical(lapply(got_tbl, class), lapply(want, class),
                     label = sprintf("%s cuplyr column classes", mode))
  }
}

# ---------------------------------------------------------------------------
# Filter pushdown below a mutate: dplyr computes the mutate on every row, so
# the overflowing row warns even though the filter drops it.
# ---------------------------------------------------------------------------

test_that("pushdown: a filter after an overflowing mutate keeps the warning (settled chain)", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) d |> mutate(y = a + b) |> filter(keep),
            warns = TRUE)
})

test_that("pushdown: a filter after an overflowing mutate keeps the warning (with a select)", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> mutate(y = a + b) |> filter(keep) |> select(b, y)
  }, warns = TRUE)
})

test_that("pushdown: explicit compute() of the plan keeps the warning", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    r <- d |> mutate(y = a * 2L) |> filter(keep)
    if (is_tbl_gpu(r)) compute(r) else r
  }, warns = TRUE)
})

test_that("pushdown: checked arithmetic next to a double expression, or nested in if_else()", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> mutate(w = x * 2, y = a - (-1L)) |> filter(keep)
  }, warns = TRUE)
  f70_check(f70_df(), function(d, src) {
    d |> mutate(y = dplyr::if_else(keep, a * 2L, 0L)) |> filter(keep)
  }, warns = TRUE)
})

test_that("pushdown: a logical operand of integer arithmetic", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> mutate(y = a + !keep) |> filter(b > 1L)
  }, warns = TRUE)
})

test_that("pushdown: two fused mutate() calls", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> mutate(y = a + b) |> mutate(z = x * 2) |> filter(keep)
  }, warns = TRUE)
})

test_that("pushdown: grouped mutate and filter", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> group_by(g) |> mutate(y = a + b) |> filter(b > 1L) |> ungroup()
  }, warns = TRUE)
})

# ---------------------------------------------------------------------------
# Dead-column pruning: dplyr computes a mutate column even when no later verb
# reads it.
# ---------------------------------------------------------------------------

test_that("pruning: an overflowing column dropped by select() still warns", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) d |> mutate(y = a + b) |> select(a, b),
            warns = TRUE)
  f70_check(f70_df(), function(d, src) {
    d |> mutate(y = a + b, w = x * 2) |> select(w)
  }, warns = TRUE)
})

test_that("pruning: transmute() and fused mutate() calls", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> transmute(y = a + b, w = x) |> select(w)
  }, warns = TRUE)
  f70_check(f70_df(), function(d, src) {
    d |> mutate(y = a + b) |> mutate(w = x) |> select(w)
  }, warns = TRUE)
})

test_that("pruning: an overflowing column dropped after a join still warns", {
  skip_if_no_gpu()
  yy <- data.frame(b = c(2L, 3L, 4L), z = c(10, 20, 30))
  f70_check(f70_df(), function(d, src) {
    d |>
      mutate(y = a + b) |>
      left_join(src(yy), by = "b", relationship = "many-to-many") |>
      select(a, z)
  }, warns = TRUE)
})

# ---------------------------------------------------------------------------
# Filter merging: a later filter() call sees only the rows the earlier ones
# kept, so arithmetic that would overflow on a dropped row does not warn.
# ---------------------------------------------------------------------------

test_that("merging: a later filter() does not see rows an earlier one dropped (settled chain)", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) d |> filter(a < 100L) |> filter(a + b > 0L),
            warns = FALSE)
})

test_that("merging: a filter pushed next to another filter is not merged with it", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> filter(a < 100L) |> mutate(w = x * 2) |> filter(a + b > 0L)
  }, warns = FALSE)
})

test_that("merging: three filters, the checked one in the middle", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> filter(b > 1L) |> filter(a + b > 0L) |> filter(a < 100L)
  }, warns = FALSE)
})

test_that("merging: an earlier filter that keeps no rows", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) d |> filter(b > 100L) |> filter(a + b > 0L),
            warns = FALSE)
})

test_that("merging: grouped filters", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) {
    d |> group_by(g) |> filter(a < 100L) |> filter(a + b > 0L) |> ungroup()
  }, warns = FALSE)
})

test_that("merging: the checked predicate in the first filter, or in one call, still warns", {
  skip_if_no_gpu()
  f70_check(f70_df(), function(d, src) d |> filter(a + b > 0L) |> filter(a < 100L),
            warns = TRUE)
  f70_check(f70_df(), function(d, src) d |> filter(a < 100L, a + b > 0L),
            warns = TRUE)
})

# ---------------------------------------------------------------------------
# Joins: a predicate after a join sees only the rows the join outputs.
# ---------------------------------------------------------------------------

test_that("joins: semi_join() and anti_join() drop the overflowing row first", {
  skip_if_no_gpu()
  yy <- data.frame(b = c(2L, 3L, 4L), z = c(10, 20, 30))
  y1 <- data.frame(b = 1L, z = 0)
  f70_check(f70_df(), function(d, src) {
    d |> semi_join(src(yy), by = "b") |> filter(a + b > 0L)
  }, warns = FALSE)
  f70_check(f70_df(), function(d, src) {
    d |> anti_join(src(y1), by = "b") |> filter(a + b > 0L)
  }, warns = FALSE)
})

test_that("joins: inner_join() drops the unmatched overflowing row (left and right inputs)", {
  skip_if_no_gpu()
  yy <- data.frame(b = c(2L, 3L, 4L), z = c(10, 20, 30))
  f70_check(f70_df(), function(d, src) {
    d |>
      inner_join(src(yy), by = "b", relationship = "many-to-many") |>
      filter(a + b > 0L)
  }, warns = FALSE, arrange_by = "b")
  f70_check(f70_df(), function(d, src) {
    src(yy) |>
      inner_join(d, by = "b", relationship = "many-to-many") |>
      filter(a * 2L > 0L)
  }, warns = FALSE, arrange_by = "b")
})

test_that("joins: cross_join() with an empty table", {
  skip_if_no_gpu()
  y0 <- data.frame(q = numeric())
  f70_check(f70_df(), function(d, src) {
    d |> cross_join(src(y0)) |> filter(a + b > 0L)
  }, warns = FALSE)
})

test_that("joins: every left row of a left join, and right row of a right join, is seen", {
  skip_if_no_gpu()
  yy <- data.frame(b = c(2L, 3L, 4L), z = c(10, 20, 30))
  f70_check(f70_df(), function(d, src) {
    d |>
      left_join(src(yy), by = "b", relationship = "many-to-many") |>
      filter(a + b > 0L)
  }, warns = TRUE, arrange_by = "b")
  yr <- data.frame(b = c(2L, 99L), c = c(5L, F70_M))
  f70_check(f70_df(), function(d, src) {
    d |>
      right_join(src(yr), by = "b", relationship = "many-to-many") |>
      filter(c + 1L > 0L)
  }, warns = TRUE, arrange_by = "b")
})

# ---------------------------------------------------------------------------
# Plans. Pipelines without checked integer arithmetic (doubles, integer
# comparisons, %/%, %%, unary minus) are optimized exactly as before F70; the
# expected text is the optimized plan explain() printed on dev 3fd9d62. A
# checked predicate may still move into a left join's left input (every left
# row reaches the output).
# ---------------------------------------------------------------------------

f70_plan <- function(q) {
  out <- capture.output(explain(q))
  out[seq(match("Optimized plan:", out) + 1L, length(out))]
}

test_that("plans without checked integer arithmetic are unchanged", {
  skip_if_no_gpu()
  d <- data.frame(a = 1:4, b = 4:1, x = c(1, 2, 3, 4), keep = c(TRUE, FALSE, TRUE, TRUE))
  lz <- tbl_gpu(d, lazy = TRUE)
  y <- tbl_gpu(data.frame(b = 1:3, z = c(1, 2, 3)), lazy = TRUE)

  expect_identical(f70_plan(lz |> mutate(y = x + 1) |> filter(keep)), c(
    "ast_mutate [1 expressions]",
    "| ast_filter [1 predicates]",
    "| | ast_source [4 cols]"
  ))
  expect_identical(f70_plan(lz |> mutate(y = a %/% 2L, m = a %% 3L, n = -a) |> filter(b > 1L)), c(
    "ast_mutate [3 expressions]",
    "| ast_filter [1 predicates]",
    "| | ast_source [4 cols]"
  ))
  expect_identical(f70_plan(lz |> mutate(y = x * 2, w = x + 1) |> select(a, w)), c(
    "ast_select [2 cols]",
    "| ast_mutate [1 expressions]",
    "| | ast_select [2 cols]",
    "| | | ast_source [4 cols]"
  ))
  expect_identical(f70_plan(lz |> filter(a > 1L) |> filter(b < 4L, x > 1)), c(
    "ast_filter [3 predicates]",
    "| ast_source [4 cols]"
  ))
  expect_identical(f70_plan(lz |> filter(x > 1) |> filter(x * 2 > 3)), c(
    "ast_filter [2 predicates]",
    "| ast_source [4 cols]"
  ))
  expect_identical(f70_plan(lz |> inner_join(y, by = "b", relationship = "many-to-many") |>
                              filter(x > 1, z > 1)), c(
    "ast_join [inner join; on b == b]",
    "| ast_filter [1 predicates]",
    "| | ast_source [4 cols]",
    "| ast_filter [1 predicates]",
    "| | ast_source [2 cols]"
  ))
  expect_identical(f70_plan(lz |> semi_join(y, by = "b") |> filter(x > 2)), c(
    "ast_join [semi join; on b == b]",
    "| ast_filter [1 predicates]",
    "| | ast_source [4 cols]",
    "| ast_select [1 cols]",
    "| | ast_source [2 cols]"
  ))
})

test_that("a checked predicate still moves into a left join's left input", {
  skip_if_no_gpu()
  d <- data.frame(a = 1:4, b = 4:1, x = c(1, 2, 3, 4))
  lz <- tbl_gpu(d, lazy = TRUE)
  y <- tbl_gpu(data.frame(b = 1:3, z = c(1, 2, 3)), lazy = TRUE)
  expect_identical(f70_plan(lz |> left_join(y, by = "b", relationship = "many-to-many") |>
                              filter(a + b > 0L)), c(
    "ast_join [left join; on b == b]",
    "| ast_filter [1 predicates]",
    "| | ast_source [3 cols]",
    "| ast_source [2 cols]"
  ))
})

# ---------------------------------------------------------------------------
# Added after round 1 (review B1/B2): all predicates of ONE filter() call are
# evaluated on the same rows, so when a checked predicate cannot move into a
# join input, its sibling predicates must not move either (they would drop
# rows before the checked one is evaluated).
# ---------------------------------------------------------------------------

test_that("joins: a checked predicate's siblings stay with it (semi/anti)", {
  skip_if_no_gpu()
  y1 <- data.frame(b = c(1L, 2L, 3L), z = c(10, 20, 30))
  y2 <- data.frame(b = c(2L, 3L), z = c(20, 30))
  f70_check(f70_df(), function(d, src) {
    d |> semi_join(src(y1), by = "b") |> filter(keep, a + b > 0L)
  }, warns = TRUE)
  f70_check(f70_df(), function(d, src) {
    d |> anti_join(src(y2), by = "b") |> filter(keep, a + b > 0L)
  }, warns = TRUE)
})

test_that("joins: a checked predicate's siblings stay with it (inner/cross/left)", {
  skip_if_no_gpu()
  y1 <- data.frame(b = c(1L, 2L, 3L), z = c(10, 20, 30))
  yq <- data.frame(q = c(1, 2))
  f70_check(f70_df(), function(d, src) {
    d |>
      inner_join(src(y1), by = "b", relationship = "many-to-many") |>
      filter(z > 10, a + b > 0L)
  }, warns = TRUE, arrange_by = "b")
  f70_check(f70_df(), function(d, src) {
    d |>
      inner_join(src(y1), by = "b", relationship = "many-to-many") |>
      filter(keep, a + b > 0L)
  }, warns = TRUE, arrange_by = "b")
  f70_check(f70_df(), function(d, src) {
    d |> cross_join(src(yq)) |> filter(q > 100, a + b > 0L)
  }, warns = TRUE)
  f70_check(f70_df(), function(d, src) {
    d |>
      left_join(src(y1), by = "b", relationship = "many-to-many") |>
      filter(keep, (a + b) > z)
  }, warns = TRUE, arrange_by = "b")
})

test_that("joins: a checked predicate that moves into a left join's left input", {
  skip_if_no_gpu()
  y1 <- data.frame(b = c(1L, 2L, 3L), z = c(10, 20, 30))
  # its right-only sibling stays above; every left row is still seen
  f70_check(f70_df(), function(d, src) {
    d |>
      left_join(src(y1), by = "b", relationship = "many-to-many") |>
      filter(a + b > 0L, z > 10)
  }, warns = TRUE, arrange_by = "b")
  # it lands on a filter already in the left input and is not merged with it
  f70_check(f70_df(), function(d, src) {
    d |>
      filter(a < 100L) |>
      left_join(src(y1), by = "b", relationship = "many-to-many") |>
      filter(a + b > 0L)
  }, warns = FALSE, arrange_by = "b")
})

test_that("a filter node whose checked predicate stays above a join stays whole", {
  skip_if_no_gpu()
  d <- data.frame(a = 1:4, b = 4:1, x = c(1, 2, 3, 4), keep = c(TRUE, FALSE, TRUE, TRUE))
  lz <- tbl_gpu(d, lazy = TRUE)
  y <- tbl_gpu(data.frame(b = 1:3, z = c(1, 2, 3)), lazy = TRUE)
  expect_identical(f70_plan(lz |> inner_join(y, by = "b", relationship = "many-to-many") |>
                              filter(x > 1, a + b > 0L)), c(
    "ast_filter [2 predicates]",
    "| ast_join [inner join; on b == b]",
    "| | ast_source [4 cols]",
    "| | ast_source [2 cols]"
  ))
  expect_identical(f70_plan(lz |> semi_join(y, by = "b") |> filter(x > 2, a + b > 0L)), c(
    "ast_filter [2 predicates]",
    "| ast_join [semi join; on b == b]",
    "| | ast_source [4 cols]",
    "| | ast_select [1 cols]",
    "| | | ast_source [2 cols]"
  ))
})
