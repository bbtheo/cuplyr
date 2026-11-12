# F99: column pruning across optimizer barriers.
#
# Each barrier type gets a lazy pipeline on a wide table, with two contracts:
# the optimized plan drops the unused columns below the barrier (read from the
# plan the optimizer returns), and the result still equals dplyr, eager and
# lazy. The negative block lists the cases that must keep their columns.

f99_df <- function(n = 40L) {
  withr::with_seed(99L, {
    df <- tibble::tibble(
      g = sample(1:4, n, TRUE),
      h = sample(c("p", "q"), n, TRUE),
      f = factor(sample(c("a", "b", "c"), n, TRUE), levels = c("a", "b", "c", "zz")),
      d = as.Date("2024-01-01") + sample.int(30L, n, TRUE),
      t = as.POSIXct("2024-01-01 00:00:00", tz = "UTC") + sample.int(10000L, n, TRUE),
      i = sample.int(10L, n, TRUE),
      x1 = rnorm(n),
      x2 = rnorm(n),
      x3 = rnorm(n),
      x4 = rnorm(n),
      w1 = runif(n),
      w2 = runif(n),
      w3 = sample(letters, n, TRUE),
      w4 = sample(c(TRUE, FALSE), n, TRUE)
    )
    df$x2[c(3L, 11L)] <- NA_real_
    df
  })
}

f99_lazy <- function(df = f99_df()) tbl_gpu(df, lazy = TRUE)
f99_plan <- function(x) optimize_ast(unclass(x)$lazy_ops)

# The first node of a given type, walking down the plan (left side of a join).
f99_node <- function(plan, type) {
  node <- plan
  while (!is.null(node)) {
    if (identical(node$type, type)) return(node)
    node <- if (identical(node$type, "join")) node$left else node$input
  }
  NULL
}

# The columns that reach the first node of `type`.
f99_reach <- function(plan, type) {
  node <- f99_node(plan, type)
  expect_false(is.null(node), label = paste("plan holds a", type, "node"))
  infer_schema(node$input)$names
}

f99_wide_cols <- c("w1", "w2", "w3", "w4")
f99_expect_pruned <- function(reach) {
  expect_length(intersect(reach, f99_wide_cols), 0L)
}

f99_expect_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---------------------------------------------------------------------------
# Plan shape, one barrier type at a time
# ---------------------------------------------------------------------------

test_that("summarise: only its groups and aggregation inputs reach the filter", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> summarise(m = mean(x2, na.rm = TRUE)))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "x2"))

  plan <- f99_plan(f99_lazy() |> group_by(g) |> filter(x1 > 0) |>
                     summarise(m = mean(x2, na.rm = TRUE), s = sum(x3)))
  expect_setequal(f99_reach(plan, "filter"), c("g", "x1", "x2", "x3"))
})

test_that("summarise(n()) keeps one column to count rows on", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> summarise(n = n()))
  expect_setequal(f99_reach(plan, "filter"), "x1")
  f99_expect_both(f99_df(), function(t) t |> filter(x1 > 0) |> summarise(n = n()))

  # no predicate column either: the row count still needs one column
  f99_expect_both(f99_df(), function(t) t |> arrange(x1) |> summarise(n = n()))
  f99_expect_both(f99_df(), function(t) t |> slice_head(n = 7) |> summarise(n = n()))
})

test_that("arrange: the sort keys plus what is read above it", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> arrange(x2) |> select(x3))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "x2", "x3"))

  plan <- f99_plan(f99_lazy() |> arrange(x2) |> select(x3, g))
  expect_setequal(f99_reach(plan, "arrange"), c("x2", "x3", "g"))

  plan <- f99_plan(f99_lazy() |> group_by(g) |> arrange(x2, .by_group = TRUE) |> select(g, x3))
  expect_setequal(f99_reach(plan, "arrange"), c("g", "x2", "x3"))
})

test_that("arrange at the root keeps every column", {
  skip_if_no_gpu()
  df <- f99_df()
  plan <- f99_plan(f99_lazy(df) |> filter(x1 > 0) |> arrange(x2))
  expect_setequal(f99_reach(plan, "filter"), names(df))
})

test_that("distinct: its keys", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> distinct(g, h))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "g", "h"))
})

test_that("window: partition, order and argument columns plus what is read above", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> group_by(g) |>
                     mutate(r = cumsum(x2)) |> ungroup() |> select(r, x3))
  reach <- f99_reach(plan, "filter")
  expect_true(all(c("x1", "x2", "g", "x3") %in% reach))
  f99_expect_pruned(reach)
  expect_false("x4" %in% reach)
})

test_that("rename: names are mapped through", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> rename(a = x2) |> summarise(m = mean(a, na.rm = TRUE)))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "x2"))

  plan <- f99_plan(f99_lazy() |> rename(a = x2) |> filter(a > 0) |> summarise(m = mean(a)))
  expect_false(any(f99_wide_cols %in% f99_reach(plan, "rename")))
})

test_that("slice: what is read above it plus the ranking column", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> slice_head(n = 5) |> select(x3))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "x3"))

  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> slice_max(x2, n = 3) |> select(x3))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "x2", "x3"))
})

test_that("a chain of barriers prunes all the way down", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> filter(x1 > 0) |> arrange(x2) |> filter(x3 > -1) |>
                     summarise(m = mean(x4)))
  # the bottom filter serves the arrange's key and everything above the arrange
  bottom <- plan
  while (!identical(bottom$type, "source")) {
    if (identical(bottom$type, "filter")) last_filter <- bottom
    bottom <- bottom$input
  }
  expect_setequal(infer_schema(last_filter$input)$names, c("x1", "x2", "x3", "x4"))
  expect_false(any(f99_wide_cols %in% f99_reach(plan, "arrange")))
})

test_that("a join below a barrier prunes through the existing side rules", {
  skip_if_no_gpu()
  y <- tibble::tibble(g = 1:4, v = c(2.5, 3.5, 4.5, 5.5), unused_y = letters[1:4])
  plan <- f99_plan(f99_lazy() |> left_join(tbl_gpu(y, lazy = TRUE), by = "g") |>
                     summarise(m = mean(x2 + v, na.rm = TRUE)))
  join <- f99_node(plan, "join")
  expect_false(any(f99_wide_cols %in% infer_schema(join$left)$names))
  expect_false("unused_y" %in% infer_schema(join$right)$names)
})

test_that("a select the user wrote below a barrier is narrowed too", {
  skip_if_no_gpu()
  plan <- f99_plan(f99_lazy() |> select(g, x1, x2, x3, w1) |> filter(x1 > 0) |>
                     summarise(m = mean(x2, na.rm = TRUE)))
  expect_setequal(f99_reach(plan, "filter"), c("x1", "x2"))
})

test_that("the filter is not pushed back below the pruning select", {
  skip_if_no_gpu()
  df <- f99_df()
  plan <- f99_plan(f99_lazy(df) |> filter(x1 > 0) |> summarise(m = mean(x2, na.rm = TRUE)))
  filter_node <- f99_node(plan, "filter")
  expect_lt(length(infer_schema(filter_node$input)$names), length(names(df)))
})

test_that("a filter above a barrier keeps its predicate column when no column is spare", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(3, 1, 2), b = c(1, -1, 2))
  plan <- f99_plan(tbl_gpu(df, lazy = TRUE) |> arrange(a) |> filter(b > 0) |> select(a))
  expect_setequal(f99_reach(plan, "arrange"), c("a", "b"))
  f99_expect_both(df, function(t) t |> arrange(a) |> filter(b > 0) |> select(a))
  f99_expect_both(df, function(t) t |> arrange(a) |> filter(b > 0) |> summarise(m = mean(a)))

  df2 <- tibble::tibble(a = c(1, -1, 2), b = c(10, 20, 30))
  f99_expect_both(df2, function(t) t |> rename(c2 = b) |> filter(a > 0) |> select(c2))
  f99_expect_both(df2, function(t) t |> slice_head(n = 3) |> filter(a > 0) |> select(b))

  df3 <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 3, 2, 4), y = c(1, -1, 1, 1))
  f99_expect_both(df3, function(t) t |> group_by(g) |> filter(x > mean(x), y > 0) |> select(g, x))
})

test_that("narrow plans: stacked barriers, two filters, a reordering select", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(3, 1, 2, 4), b = c(1, -1, 2, 5), c = c(9, 8, 7, 6))
  f99_expect_both(df, function(t) t |> arrange(a) |> slice_head(n = 3) |> filter(b > 0) |> select(a))
  f99_expect_both(df, function(t) t |> arrange(a) |> filter(b > 0) |> arrange(c) |> select(a))
  f99_expect_both(df, function(t) t |> arrange(a) |> filter(b > 0) |> filter(c > 6.5) |> select(a))
  f99_expect_both(df, function(t) t |> arrange(a) |> filter(b > 0, c > 6.5) |> summarise(m = mean(a)))
  f99_expect_both(df, function(t) t |> select(c, a, b) |> arrange(a) |> filter(b > 0) |> select(c))
  f99_expect_both(df, function(t) t |> select(c, a) |> filter(a > 1) |> arrange(c) |> select(a))
  f99_expect_both(df, function(t) t |> arrange(a) |> mutate(d = b * 2) |> filter(d > 0) |> select(a))
})

# F99 removes columns only: an expression below a barrier that nothing reads is
# still computed, as it is on dev, because it can warn.
test_that("a mutate below a barrier keeps its dead expressions and their warnings", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(NA_real_, 1), y = c(NA_real_, 2), z = c(2, 1), i = c(.Machine$integer.max, 1L))
  rw <- function(t) t |> rowwise() |> mutate(m = min(c(x, y), na.rm = TRUE)) |> ungroup()
  for (lazy in c(TRUE, FALSE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_warning(res <- rw(gt) |> arrange(z) |> select(z) |> collect(), "no non-missing arguments to min")
    expect_identical(res$z, c(1, 2))
    expect_warning(res <- rw(gt) |> arrange(z) |> summarise(n = n()) |> collect(), "no non-missing arguments to min")
    expect_identical(res$n, 2L)
    expect_warning(res <- rw(gt) |> filter(z > 0) |> summarise(s = sum(z)) |> collect(), "no non-missing arguments to min")
    expect_identical(res$s, 3)
    expect_warning(res <- gt |> mutate(w = i * 2L) |> arrange(z) |> select(z) |> collect(), "integer overflow")
    expect_identical(res$z, c(1, 2))
  }

  wide <- f99_df()
  plan <- f99_plan(f99_lazy(wide) |> mutate(z = x1 * 2) |> summarise(m = mean(x2, na.rm = TRUE)))
  node <- f99_node(plan, "mutate")
  expect_false(is.null(node))
  expect_identical(vapply(node$expressions, `[[`, "", "output_col"), "z")
  reach <- infer_schema(node$input)$names
  expect_setequal(reach, c("x1", "x2"))
})

test_that("a rename under a join whose old names nothing reads", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = 1:3, a = c(1.5, 2.5, 3.5), c = c(2, 3, 4), z = 7)
  y <- tibble::tibble(k = 1:3, v = 4:6)
  f99_expect_both(df, function(t) {
    other <- if (inherits(t, "tbl_gpu")) tbl_gpu(y, lazy = identical(t$exec_mode, "lazy")) else y
    t |> rename(b = a, d = c) |> left_join(other, by = "k") |> select(k, v)
  })
})

# ---------------------------------------------------------------------------
# Results: every barrier equals dplyr, eager and lazy
# ---------------------------------------------------------------------------

test_that("barrier pipelines on a wide table equal dplyr", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> summarise(m = mean(x2, na.rm = TRUE)))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> group_by(g) |>
                    summarise(m = mean(x2, na.rm = TRUE), s = sum(x3), .groups = "drop"))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |>
                    summarise(m = mean(x2, na.rm = TRUE), .by = g), arrange_by = "g")
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> arrange(x2, g) |> select(x3))
  f99_expect_both(df, function(t) t |> group_by(g) |> arrange(x3, .by_group = TRUE) |> select(g, x4))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> distinct(g, h))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> group_by(g) |>
                    mutate(r = cumsum(x3)) |> ungroup() |> select(r, x4))
  f99_expect_both(df, function(t) t |> rename(a = x2, b = x3) |> summarise(m = mean(a, na.rm = TRUE)))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> slice_head(n = 5) |> select(x3))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> slice_max(x3, n = 3) |> select(x4))
  f99_expect_both(df, function(t) t |> slice_min(x1 + x3, n = 2) |> select(g))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> arrange(x2) |> filter(x3 > -1) |>
                    summarise(m = mean(x4), n = n()))
})

test_that("a rename whose old name nothing reads keeps working", {
  skip_if_no_gpu()
  df <- f99_df()
  # (reading neither `aa` nor `bb` above the rename)
  f99_expect_both(df, function(t) t |> rename(aa = x2, bb = x3) |> summarise(m = mean(x1)))
  f99_expect_both(df, function(t) t |> rename(aa = x2, bb = x3) |> arrange(x1) |> select(g))
  f99_expect_both(df, function(t) t |> rename(aa = x2) |> filter(x1 > 0) |> summarise(n = n()))
})

test_that("mutate above a barrier: intermediates, replacements and creation order", {
  skip_if_no_gpu()
  df <- f99_df()
  # z is an intermediate read by w within one mutate; neither exists below
  f99_expect_both(df, function(t) t |> arrange(x1) |> filter(x2 > 0) |>
                    mutate(z = x3 + 1, w = z * 2) |> select(w))
  # a replace-in-place above a barrier keeps its position
  f99_expect_both(df, function(t) t |> arrange(x1) |> mutate(x2 = x2 * 2) |> select(x2, x3))
  f99_expect_both(df, function(t) t |> arrange(x1) |> mutate(x2 = x2 * 2) |> relocate(x3, .before = 1))
  f99_expect_both(df, function(t) t |> arrange(x1) |> filter(x3 > 0) |> mutate(z = 1L) |> select(z))
})

test_that("a join below a barrier equals dplyr, suffixes included", {
  skip_if_no_gpu()
  df <- f99_df()
  y <- tibble::tibble(g = 1:4, v = c(2.5, 3.5, 4.5, 5.5), x1 = c(9, 8, 7, 6), unused_y = letters[1:4])
  pipeline <- function(t) {
    other <- if (inherits(t, "tbl_gpu")) tbl_gpu(y, lazy = identical(t$exec_mode, "lazy")) else y
    t |> left_join(other, by = "g") |> summarise(m = mean(x2 + v, na.rm = TRUE), s = sum(x1.y))
  }
  f99_expect_both(df, pipeline)
  pipeline2 <- function(t) {
    other <- if (inherits(t, "tbl_gpu")) tbl_gpu(y, lazy = identical(t$exec_mode, "lazy")) else y
    t |> left_join(other, by = "g") |> arrange(v, x1.x) |> select(x1.x, g)
  }
  f99_expect_both(df, pipeline2)
})

test_that("group_by, factor, Date, POSIXct and logical columns come through typed", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> arrange(d) |> select(d, t, f, w4))
  f99_expect_both(df, function(t) t |> group_by(f, .drop = FALSE) |> filter(x1 > 0) |>
                    summarise(n = n(), m = mean(x3)))
  f99_expect_both(df, function(t) t |> filter(f == "a") |> summarise(n = n(), d0 = min(d)))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> mutate(s = as.character(f)) |>
                    arrange(s, x3) |> select(s, t))
})

test_that("the group vars of the result survive the pruning", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> group_by(g, h) |> filter(x1 > 0) |> arrange(x2) |> select(g, x3))
  f99_expect_both(df, function(t) t |> group_by(g, h) |> filter(x1 > 0) |>
                    summarise(m = mean(x3)))
  f99_expect_both(df, function(t) t |> group_by(g) |> filter(x1 > 0) |>
                    slice_head(n = 10) |> select(g, x3) |> ungroup())
})

test_that("zero rows and all-NA columns", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> filter(x1 > 1e9) |> summarise(m = mean(x2), n = n()))
  f99_expect_both(df, function(t) t |> group_by(g) |> filter(x1 > 1e9) |> summarise(m = mean(x2)))
  f99_expect_both(df, function(t) t |> filter(x1 > 1e9) |> arrange(x2) |> select(x3))
  empty <- df[0L, ]
  f99_expect_both(empty, function(t) t |> filter(x1 > 0) |> summarise(m = mean(x2), n = n()))
  f99_expect_both(empty, function(t) t |> arrange(x2) |> select(x3))
  na_df <- df
  na_df$x2 <- NA_real_
  f99_expect_both(na_df, function(t) t |> filter(x1 > 0) |> summarise(m = mean(x2), n = n()))
})

test_that("a column read by a dropped name is still dplyr's error", {
  skip_if_no_gpu()
  df <- f99_df()
  for (t in list(df, tbl_gpu(df, lazy = TRUE))) {
    expect_error(t |> rename(aa = x2) |> summarise(m = mean(x2)) |> collect())
  }
})

# ---------------------------------------------------------------------------
# Negative cases: these must keep their columns
# ---------------------------------------------------------------------------

test_that("distinct(.keep_all = TRUE) keeps every column", {
  skip_if_no_gpu()
  df <- f99_df()
  plan <- f99_plan(f99_lazy(df) |> filter(x1 > 0) |> distinct(g, .keep_all = TRUE) |> select(x3))
  expect_setequal(f99_reach(plan, "filter"), names(df))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> distinct(g, .keep_all = TRUE) |> select(x3))
})

test_that("collapse() is a fence: nothing is pruned below it", {
  skip_if_no_gpu()
  df <- f99_df()
  plan <- f99_plan(f99_lazy(df) |> filter(x1 > 0) |> collapse() |> summarise(m = mean(x2, na.rm = TRUE)))
  expect_setequal(f99_reach(plan, "filter"), names(df))
})

test_that("reframe keeps its wide input", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |>
                    reframe(q = range(x3)))
})

test_that("across() and where() expansions read their columns after the barrier", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> arrange(x2) |>
                    mutate(dplyr::across(dplyr::where(is.numeric), ~ .x * 2)) |> select(x3, w1, i))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> arrange(x2) |>
                    summarise(dplyr::across(dplyr::everything(), ~ n_distinct(.x))))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> arrange(x2) |>
                    mutate(k = rowSums(dplyr::across(c(x3, x4, w1)))) |> select(k))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |>
                    summarise(dplyr::across(dplyr::where(is.numeric), ~ sum(is.na(.x)))))
  plan <- f99_plan(f99_lazy(df) |> filter(x1 > 0) |> arrange(x2) |>
                     mutate(dplyr::across(dplyr::where(is.numeric), ~ .x * 2)) |> select(x3, w1))
  # every numeric column is rewritten, so a dead one is the mutate's to prune,
  # but the columns read above must have survived
  reach <- f99_reach(plan, "arrange")
  expect_true(all(c("x2", "x3", "w1") %in% reach))
})

test_that("a CPU-fallback step in the middle reads whole-table columns", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_cpu <- function(x) x + 1
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> mutate(y = f99_cpu(x2)) |>
                    arrange(y) |> summarise(m = mean(y, na.rm = TRUE), s = sum(x3)))
  f99_expect_both(df, function(t) t |> arrange(x1) |> mutate(y = f99_cpu(w1)) |> select(y, x4))
})

test_that("a plan-break in the middle keeps the columns later steps read", {
  skip_if_no_gpu()
  df <- f99_df()
  # (an integer sum() is computed at the verb on a lazy table, F80)
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> group_by(g) |>
                    mutate(s = sum(i)) |> ungroup() |> select(w1, s, x3) |> arrange(w1))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> summarise(s = sum(i), m = mean(x3)))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> group_by(g) |>
                    summarise(s = sum(i), .groups = "drop") |> arrange(s))
})

test_that("rowwise tables keep their result", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) t |> rowwise() |>
                    mutate(s = sum(dplyr::c_across(c(x1, x3)))) |> ungroup() |>
                    filter(x4 > 0) |> summarise(m = mean(s)))
  f99_expect_both(df, function(t) t |> rowwise(g) |> mutate(s = x1 + x3) |>
                    filter(x4 > 0) |> ungroup() |> arrange(s) |> select(g, s))
  f99_expect_both(df, function(t) t |> filter(x1 > 0) |> rowwise() |>
                    summarise(m = mean(dplyr::c_across(c(x3, x4)))))
})

test_that("set operations and n()/cur_group_id() over a barrier", {
  skip_if_no_gpu()
  df <- f99_df()
  f99_expect_both(df, function(t) {
    a <- t |> filter(x1 > 0) |> select(g, h)
    b <- t |> filter(x1 < 0) |> select(g, h)
    union(a, b) |> arrange(g, h)
  })
  f99_expect_both(df, function(t) t |> group_by(g) |> filter(x1 > 0) |>
                    mutate(k = n(), id = dplyr::cur_group_id()) |> ungroup() |>
                    arrange(x2, k) |> select(k, id))
})

test_that("duplicated column names disable pruning instead of breaking", {
  skip_if_no_gpu()
  gdf <- tbl_gpu(tibble::tibble(a = 1:6, b = c(2.5, 1.5, 3.5, 0.5, 5.5, 4.5)), lazy = TRUE)
  dup <- gdf |> select(x = a, b, x2 = a)
  names(dup)[3] <- "x"
  res <- dup |> distinct(b) |> arrange(b) |> collect()
  expect_identical(res$b, c(0.5, 1.5, 2.5, 3.5, 4.5, 5.5))
  res2 <- dup |> select(2, 1) |> slice_head(n = 2) |> collect()
  expect_identical(res2$x, 1:2)
  expect_identical(res2$b, c(2.5, 1.5))
})

# ---------------------------------------------------------------------------
# What must not change: no-barrier plans, eager mode
# ---------------------------------------------------------------------------

test_that("a plan with no barrier is unchanged and needs no required-set walk", {
  skip_if_no_gpu()
  df <- f99_df()
  x <- f99_lazy(df) |> filter(x1 > 0) |> mutate(z = x1 * 2)
  plan <- f99_plan(x)
  expect_setequal(infer_schema(f99_node(plan, "mutate")$input)$names, names(df))
  expect_null(f99_node(plan, "select"))

  # optimized plan of a no-barrier chain is the input node itself
  ops <- unclass(f99_lazy(df) |> filter(x1 > 0))$lazy_ops
  expect_identical(optimize_ast(ops), ops)
})

test_that("the eager path does not go through the optimizer", {
  skip_if_no_gpu()
  df <- f99_df()
  eager <- tbl_gpu(df, lazy = FALSE) |> filter(x1 > 0) |> summarise(m = mean(x2, na.rm = TRUE))
  expect_null(unclass(eager)$lazy_ops)
  expect_equal(collect(eager)$m, df |> filter(x1 > 0) |> summarise(m = mean(x2, na.rm = TRUE)) |> dplyr::pull(m))
})

# ---------------------------------------------------------------------------
# 1e6 rows x 50 columns: the plan, and the time against hand-written pruning
# ---------------------------------------------------------------------------

test_that("a wide filter |> summarise costs about what the hand-pruned plan does", {
  skip_if_no_gpu()
  n <- 1e6L
  wide <- withr::with_seed(1L, {
    m <- matrix(runif(n * 42L), ncol = 42L)
    d <- as.data.frame(m)
    names(d) <- paste0("c", seq_len(42L))
    for (k in 43:50) d[[paste0("c", k)]] <- sample(c("alpha", "beta", "gamma", "delta"), n, TRUE)
    d$g <- sample.int(10L, n, TRUE)
    d
  })
  gt <- tbl_gpu(wide, lazy = TRUE)

  auto <- function() gt |> filter(c1 > 0.5) |> summarise(m = mean(c2)) |> collect()
  hand <- function() gt |> select(c1, c2) |> filter(c1 > 0.5) |> summarise(m = mean(c2)) |> collect()
  expect_equal(auto()$m, wide |> filter(c1 > 0.5) |> summarise(m = mean(c2)) |> dplyr::pull(m))

  plan <- f99_plan(gt |> filter(c1 > 0.5) |> summarise(m = mean(c2)))
  expect_setequal(f99_reach(plan, "filter"), c("c1", "c2"))

  runs <- timing_retry(
    function() timing_medians(list(auto = auto, hand = hand), samples = 11L, label = "F99 wide"),
    ok = function(v) v[["auto"]] <= 1.5 * v[["hand"]],
    retry_if = function(v) v[["auto"]] <= 2.5 * v[["hand"]],
    attempts = 3L
  )
  last <- runs[[length(runs)]]
  expect_lte(last[["auto"]], 1.5 * last[["hand"]],
             label = paste(vapply(runs, function(r) sprintf("%.2f/%.2f ms", r[["auto"]] * 1e3, r[["hand"]] * 1e3), ""), collapse = "; "))
})
