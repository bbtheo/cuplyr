# Mirrored dplyr-oracle tests for on-the-fly `.by=` grouping in mutate()/
# filter() (Phase 5, task W6).
#
# `resolve_by()` (R/execute.R) is the shared resolution helper: `.by=`
# behaves EXACTLY like `group_by(<cols>) |> <verb>(...) |> ungroup()` --
# same window/aggregate machinery `group_by()`-driven mutate()/filter()
# already use (Phase 5 W1-W5), threaded through via `verb_groups` --
# except the result is always ungrouped (`.by` never leaves grouping
# metadata behind), and supplying `.by=` on an already-`group_by()`-grouped
# table is a hard error, matching dplyr's own message verbatim.
#
# The `slice()` family's own `by=`/`.by=` (Phase 5, task W7) is GPU-native
# too (R/slice.R::slice_grouped_native()) -- see test-dplyr-slice.R for the
# bulk of its oracle coverage (including the group-ordering semantics,
# which differ from mutate()/filter()'s `.by=` -- slice's own `by=` orders
# output by each group's FIRST-APPEARANCE position, not literal original
# row order). This file adds the `by=`-on-a-grouped-`.data` error check for
# slice_head()/slice_min(), for parity with the mutate()/filter() checks
# above.

# =============================================================================
# mutate(.by=): window aggregates, scan/shift, ranks
# =============================================================================

test_that("mutate(.by=) aggregate broadcast (mean) matches group_by()|>mutate()|>ungroup()", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5))

  pipeline <- function(d) dplyr::mutate(d, y = mean(x), .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) scan family (cumsum) matches dplyr, row order preserved on shuffled data", {
  skip_if_no_gpu()
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 20, 30, 40, 50))

  pipeline <- function(d) dplyr::mutate(d, cs = cumsum(x), .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) shift family (lag/lead) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(1, 2, 3, 4, 5))

  pipeline <- function(d) dplyr::mutate(d, l = dplyr::lag(x), .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) rank family (min_rank/row_number) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(3, 1, 2, 20, 10))

  pipeline <- function(d) {
    dplyr::mutate(d, r = dplyr::min_rank(x), rn = dplyr::row_number(), .by = g)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) with multi-column tidyselect (c(g, h)) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "b", "b"), h = c(1, 1, 2, 2), x = c(1, 2, 3, 4)
  )

  pipeline <- function(d) dplyr::mutate(d, y = mean(x), .by = c(g, h))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) with tidyselect selector (starts_with) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    grp = c("a", "a", "b", "b"), other = c(1, 1, 2, 2), x = c(1, 2, 3, 4)
  )

  pipeline <- function(d) dplyr::mutate(d, y = mean(x), .by = dplyr::starts_with("g"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) with a window-free expression is accepted and grouping is irrelevant", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  pipeline <- function(d) dplyr::mutate(d, y = x + 1, .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) always keeps the by-column even under .keep='none'", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  pipeline <- function(d) dplyr::mutate(d, y = x + 1, .by = g, .keep = "none")
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true("g" %in% names(result$gpu))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate(.by=) result is always ungrouped (group_vars empty)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  gdf <- tbl_gpu(df)

  result <- dplyr::mutate(gdf, y = mean(x), .by = g)
  expect_identical(result$groups, character())
  expect_identical(dplyr::group_vars(result), character())
})

test_that("mutate(.by=) narrow n()/sum() single-dot shape still works via the general window path", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5))

  pipeline_n <- function(d) dplyr::mutate(d, n = dplyr::n(), .by = g)
  expect_same_as_dplyr(df, pipeline_n)
  expect_same_as_dplyr_lazy(df, pipeline_n)

  pipeline_sum <- function(d) dplyr::mutate(d, s = sum(x), .by = g)
  expect_same_as_dplyr(df, pipeline_sum)
  expect_same_as_dplyr_lazy(df, pipeline_sum)
})

# =============================================================================
# filter(.by=): window/aggregate predicates
# =============================================================================

test_that("filter(.by=) aggregate predicate (x > mean(x)) matches group_by()|>filter()|>ungroup()", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(1, 5, 3, 4, 100))

  pipeline <- function(d) dplyr::filter(d, x > mean(x), .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(.by=) row_number()/n() predicates match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(1, 2, 3, 4, 5))

  pipeline <- function(d) dplyr::filter(d, dplyr::row_number() == 1, .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  pipeline_n <- function(d) dplyr::filter(d, dplyr::n() > 2, .by = g)
  expect_same_as_dplyr(df, pipeline_n)
  expect_same_as_dplyr_lazy(df, pipeline_n)
})

test_that("filter(.by=) preserves original row order on shuffled data", {
  skip_if_no_gpu()
  df <- data.frame(
    id = 1:6,
    g = c("b", "a", "b", "a", "b", "a"),
    x = c(10, 1, 20, 2, 30, 3)
  )

  pipeline <- function(d) dplyr::filter(d, x > mean(x), .by = g)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$id, result$oracle$id)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(.by=) with a window-free predicate is accepted and grouping is irrelevant", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  pipeline <- function(d) dplyr::filter(d, x > 1, .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(.by=) result is always ungrouped (group_vars empty)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  gdf <- tbl_gpu(df)

  result <- dplyr::filter(gdf, x > mean(x), .by = g)
  expect_identical(result$groups, character())
  expect_identical(dplyr::group_vars(result), character())
})

test_that("filter(.by=) with multi-column tidyselect matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "b", "b"), h = c(1, 1, 2, 2), x = c(1, 2, 3, 4)
  )

  pipeline <- function(d) dplyr::filter(d, x > mean(x), .by = c(g, h))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# summarise(.by=): first-appearance group order (Phase 5, task W9)
# =============================================================================
#
# Unlike mutate()/filter()'s `.by=` (which behave EXACTLY like group_by()),
# summarise(.by=) emits rows in each group's FIRST-APPEARANCE order in
# `.data` -- verified empirically against dplyr 1.2.1 (see
# summarise_by_desugar(), R/summarise.R). No `arrange_by=` is needed in
# these oracle comparisons (unlike group_by()|>summarise(), whose group
# order is cuDF's own unspecified hash-groupby order and always needs
# arrange_by= to compare) -- .by='s order is exactly reproducible.

test_that("summarise(.by=) emits groups in first-appearance order, not sorted key order", {
  skip_if_no_gpu()
  # "b" appears before "a" in the data -- group_by()|>summarise() would sort
  # ("a" before "b"); .by= must NOT.
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 20, 30, 40, 50))

  pipeline <- function(d) dplyr::summarise(d, s = sum(x), .by = g)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$g, c("b", "a"))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise(.by=) with multiple aggregations matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 20, 30, 40, 50))

  pipeline <- function(d) {
    dplyr::summarise(d, m = mean(x), s = sum(x), n = dplyr::n(), .by = g)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise(.by=) with multi-column tidyselect matches dplyr, first-appearance order", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("b", "a", "b", "a"), h = c(1, 2, 1, 2), x = c(10, 20, 30, 40)
  )

  pipeline <- function(d) dplyr::summarise(d, s = sum(x), .by = c(g, h))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$g, c("b", "a"))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise(.by=) result is always ungrouped (group_vars empty)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  gdf <- tbl_gpu(df)

  result <- dplyr::summarise(gdf, s = sum(x), .by = g)
  expect_identical(result$groups, character())
  expect_identical(dplyr::group_vars(result), character())
})

test_that("summarise(.by=) on a grouped tbl_gpu errors, matching dplyr's exact message", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  gdf <- dplyr::group_by(tbl_gpu(df), g)
  odf <- dplyr::group_by(dplyr::as_tibble(df), g)

  oracle_msg <- tryCatch(
    dplyr::summarise(odf, s = sum(x), .by = g),
    error = function(e) conditionMessage(e)
  )
  expect_match(oracle_msg, "Can't supply `.by` when `.data` is a grouped data frame.", fixed = TRUE)

  expect_error(
    dplyr::summarise(gdf, s = sum(x), .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("summarise(.by=) grouped-error fires even with zero dots", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = c(1, 2))
  gdf <- dplyr::group_by(tbl_gpu(df), g)

  expect_error(
    dplyr::summarise(gdf, .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("summarise(.by=) with a nonexistent column raises a tidyselect error", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = c(1, 2))
  gdf <- tbl_gpu(df)

  expect_error(dplyr::summarise(gdf, s = sum(x), .by = zzz), "doesn't exist")
})

test_that("summarise(.by=) with a computed aggregation sub-expression matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 20, 30, 40, 50))

  pipeline <- function(d) dplyr::summarise(d, s = sum(x > 25), .by = g)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$g, c("b", "a"))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise(.by=) never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 20, 30, 40, 50))
  gdf <- tbl_gpu(df)

  expect_no_error(gdf |> dplyr::summarise(s = sum(x), .by = g) |> collect())
})

test_that("summarise(.by=) on a 0-row table matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = character(0), x = numeric(0))

  pipeline <- function(d) dplyr::summarise(d, s = sum(x), .by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .by = NULL (explicit) is identical to omitting .by
# =============================================================================

test_that("mutate(.by = NULL) is identical to omitting .by (whole-table aggregate)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  pipeline <- function(d) dplyr::mutate(d, y = mean(x), .by = NULL)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(.by = NULL) is identical to omitting .by (whole-table aggregate)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  pipeline <- function(d) dplyr::filter(d, x > mean(x), .by = NULL)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise(.by = NULL) is identical to omitting .by (group_by()-driven)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::summarise(s = sum(x), .by = NULL)
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

# =============================================================================
# Errors: `.by=` on an already-grouped table (exact dplyr message)
# =============================================================================

test_that("mutate(.by=) on a grouped tbl_gpu errors, matching dplyr's exact message", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  gdf <- dplyr::group_by(tbl_gpu(df), g)
  odf <- dplyr::group_by(dplyr::as_tibble(df), g)

  oracle_msg <- tryCatch(
    dplyr::mutate(odf, y = mean(x), .by = g),
    error = function(e) conditionMessage(e)
  )
  expect_match(oracle_msg, "Can't supply `.by` when `.data` is a grouped data frame.", fixed = TRUE)

  expect_error(
    dplyr::mutate(gdf, y = mean(x), .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("filter(.by=) on a grouped tbl_gpu errors, matching dplyr's exact message", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  gdf <- dplyr::group_by(tbl_gpu(df), g)

  expect_error(
    dplyr::filter(gdf, x > mean(x), .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("mutate(.by=) grouped-error fires even with zero dots", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = c(1, 2))
  gdf <- dplyr::group_by(tbl_gpu(df), g)

  expect_error(
    dplyr::mutate(gdf, .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("filter(.by=) grouped-error fires even with zero dots", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = c(1, 2))
  gdf <- dplyr::group_by(tbl_gpu(df), g)

  expect_error(
    dplyr::filter(gdf, .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("slice_head(by=) on a grouped tbl_gpu errors, matching dplyr's exact message", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  gdf <- dplyr::group_by(tbl_gpu(df), g)
  odf <- dplyr::group_by(dplyr::as_tibble(df), g)

  oracle_msg <- tryCatch(
    dplyr::slice_head(odf, n = 1, by = g),
    error = function(e) conditionMessage(e)
  )
  expect_match(oracle_msg, "Can't supply `by` when `.data` is a grouped data frame.", fixed = TRUE)

  expect_error(
    dplyr::slice_head(gdf, n = 1, by = g),
    "Can't supply `by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("slice_min(by=) on a grouped tbl_gpu errors, matching dplyr's exact message", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  gdf <- dplyr::group_by(tbl_gpu(df), g)

  expect_error(
    dplyr::slice_min(gdf, x, n = 1, by = g),
    "Can't supply `by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

test_that("slice(.by=) on a grouped tbl_gpu errors, matching dplyr's exact message", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  gdf <- dplyr::group_by(tbl_gpu(df), g)

  expect_error(
    dplyr::slice(gdf, 1, .by = g),
    "Can't supply `.by` when `.data` is a grouped data frame.",
    fixed = TRUE
  )
})

# =============================================================================
# Errors: `.by=` referencing a nonexistent column (tidyselect error)
# =============================================================================

test_that("mutate(.by=) with a nonexistent column raises a tidyselect error", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = c(1, 2))
  gdf <- tbl_gpu(df)

  expect_error(dplyr::mutate(gdf, y = mean(x), .by = zzz), "doesn't exist")
})

test_that("filter(.by=) with a nonexistent column raises a tidyselect error", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = c(1, 2))
  gdf <- tbl_gpu(df)

  expect_error(dplyr::filter(gdf, x > mean(x), .by = zzz), "doesn't exist")
})

# =============================================================================
# No-fallback assertions: `.by=` is fully GPU-native (never CPU fallback)
# =============================================================================

test_that("mutate(.by=) with window/aggregate expressions never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5))
  gdf <- tbl_gpu(df)

  expect_no_error(gdf |> dplyr::mutate(y = mean(x), .by = g) |> collect())
  expect_no_error(gdf |> dplyr::mutate(cs = cumsum(x), .by = g) |> collect())
  expect_no_error(gdf |> dplyr::mutate(rn = dplyr::row_number(), .by = g) |> collect())
  expect_no_error(gdf |> dplyr::mutate(l = dplyr::lag(x), .by = g) |> collect())
  expect_no_error(gdf |> dplyr::mutate(y = x + 1, .by = g) |> collect())
  expect_no_error(gdf |> dplyr::mutate(n = dplyr::n(), .by = g) |> collect())
})

test_that("filter(.by=) with window/aggregate predicates never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5))
  gdf <- tbl_gpu(df)

  expect_no_error(gdf |> dplyr::filter(x > mean(x), .by = g) |> collect())
  expect_no_error(gdf |> dplyr::filter(dplyr::row_number() == 1, .by = g) |> collect())
  expect_no_error(gdf |> dplyr::filter(x > 1, .by = g) |> collect())
})

test_that("slice_head(by=)/slice_min(by=) never trigger a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5))
  gdf <- tbl_gpu(df)

  expect_no_error(gdf |> dplyr::slice_head(n = 1, by = g) |> collect())
  expect_no_error(gdf |> dplyr::slice_min(x, n = 1, by = g) |> collect())
})

# =============================================================================
# slice_head()/slice_min()(by=): oracle parity, ungrouped result
# (bulk of slice()-family by= coverage lives in test-dplyr-slice.R; these
# two mirror this file's own mutate()/filter() layout for easy comparison)
# =============================================================================

test_that("slice_head(by=) matches dplyr and comes back ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 20, 30, 40, 50))

  pipeline <- function(d) dplyr::slice_head(d, n = 1, by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("slice_min(by=) matches dplyr and comes back ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("b", "a", "b", "a", "b"), x = c(10, 5, 30, 2, 1))

  pipeline <- function(d) dplyr::slice_min(d, x, n = 1, by = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# 0-row table
# =============================================================================

test_that("mutate(.by=)/filter(.by=) on a 0-row table match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = character(0), x = numeric(0))

  pipeline_mutate <- function(d) dplyr::mutate(d, y = mean(x), .by = g)
  expect_same_as_dplyr(df, pipeline_mutate)
  expect_same_as_dplyr_lazy(df, pipeline_mutate)

  pipeline_filter <- function(d) dplyr::filter(d, x > mean(x), .by = g)
  expect_same_as_dplyr(df, pipeline_filter)
  expect_same_as_dplyr_lazy(df, pipeline_filter)
})
