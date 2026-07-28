# Mirrored dplyr-oracle tests for window/aggregate functions inside
# filter() predicates (Phase 5, task W5).
#
# Before this task, filter() had NO ir_has_window() guard at all: a
# window-bearing predicate (e.g. filter(x > mean(x))) reached
# ir_bind()'s "internal: window op '<op>' reached lowering" guard and
# surfaced as a confusing wrapped GPU error ("GPU operation 'filter'
# failed... Original error: internal: window op 'mean' reached lowering")
# instead of either working natively or falling back cleanly. See the
# first test below for the regression pin.
#
# Implementation shape (R/filter.R): filter.tbl_gpu() now does a two-pass
# plan-then-emit exactly like mutate_window() (R/mutate.R): pass 1 parses
# every dot and classifies which ones contain a window call
# (ir_has_window()); if any do, ALL of them are planned jointly via
# plan_window_stages() (R/window.R) -- the exact machinery mutate_window()
# uses -- and validated for lowerability (window_spec_lowerable(),
# R/lower.R) BEFORE anything is pushed onto the table (never half-lower).
# Only then are the window stage nodes (pre-mutate + ast_window) pushed,
# followed by ONE combined ast_filter folding every dot's (possibly
# window-rewritten) predicate together, and a trailing ast_select
# restoring the original column set.
#
# Multi-dot semantics (the main trap this task calls out, verified
# empirically against real dplyr -- see below): `filter(cumsum(x) < 10, x
# > 2)`-style multi-dot calls evaluate EVERY dot's predicate against the
# SAME, ORIGINAL, pre-filter() data, then AND the results together -- NOT
# sequential/progressive narrowing, where a later dot's window aggregate
# would "see" an earlier dot's already-filtered rows. Empirical proof (see
# "multi-dot AND semantics" test below): `filter(x > 0, cumsum(x) < 10)`
# on `x = c(-5, 3, -5, 3, -5, 3, -5, 3)` keeps all four `x == 3` rows (the
# single-pass/AND-against-original-data reading), NOT just the first three
# (what recomputing cumsum() only over the x > 0-surviving subset would
# give). cuplyr reproduces this by materializing every window-bearing
# dot's temp columns up front, against the table exactly as it stood at
# filter() call time, before any dot's predicate has removed a row.

# =============================================================================
# Shared fixtures
# =============================================================================

oracle_filter_grouped_df <- function() {
  data.frame(
    g = c("a", "a", "a", "b", "b", "b", "b"),
    x = c(1, 2, 3, 4, 5, 6, 7),
    stringsAsFactors = FALSE
  )
}

# Rows NOT sorted by group -- grouped filter() must preserve ORIGINAL row
# order in the kept rows (unlike grouped slice(), which reorders into
# group-key order -- see scratchpad/phase5_window_design.md section 0).
oracle_filter_shuffled_df <- function() {
  data.frame(
    id = 1:9,
    g = c("b", "a", "c", "a", "b", "c", "a", "b", "c"),
    x = c(10, 1, 100, 2, 20, 200, 3, 30, 300),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Crash regression: filter(x > mean(x)) used to reach ir_bind()'s window
# guard and error with a wrapped GPU internal error instead of working
# =============================================================================

test_that("filter(x > mean(x)) no longer crashes with a wrapped GPU internal error (W4 regression)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4))
  gdf <- tbl_gpu(df)

  # Before this task's fix, this call errored with:
  #   "GPU operation 'filter' failed... Original error: internal: window
  #   op 'mean' reached lowering"
  # It must now either succeed with the correct result, or (documented,
  # not exercised by this test) cleanly fall back -- never crash.
  result <- gdf |> dplyr::filter(x > mean(x)) |> collect()
  expect_equal(sort(result$x), c(3, 4))
})

test_that("filter(x > mean(x)) matches dplyr, ungrouped and grouped", {
  skip_if_no_gpu()
  df_ungrouped <- data.frame(x = c(1, 2, 3, 4))
  pipeline <- function(d) dplyr::filter(d, x > mean(x))
  expect_same_as_dplyr(df_ungrouped, pipeline)
  expect_same_as_dplyr_lazy(df_ungrouped, pipeline)

  df_grouped <- oracle_filter_grouped_df()
  grouped_pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x))
  expect_same_as_dplyr(df_grouped, grouped_pipeline)
  expect_same_as_dplyr_lazy(df_grouped, grouped_pipeline)
})

# =============================================================================
# n() / row_number()
# =============================================================================

test_that("filter(n() > 2) matches dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- oracle_filter_grouped_df()

  pipeline_ungrouped <- function(d) dplyr::filter(d, dplyr::n() > 2)
  expect_same_as_dplyr(df, pipeline_ungrouped)
  expect_same_as_dplyr_lazy(df, pipeline_ungrouped)

  pipeline_grouped <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(dplyr::n() > 2)
  expect_same_as_dplyr(df, pipeline_grouped)
  expect_same_as_dplyr_lazy(df, pipeline_grouped)
})

test_that("filter(row_number() == 1) matches dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- oracle_filter_grouped_df()

  pipeline_ungrouped <- function(d) dplyr::filter(d, dplyr::row_number() == 1)
  expect_same_as_dplyr(df, pipeline_ungrouped)
  expect_same_as_dplyr_lazy(df, pipeline_ungrouped)

  pipeline_grouped <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(dplyr::row_number() == 1)
  expect_same_as_dplyr(df, pipeline_grouped)
  expect_same_as_dplyr_lazy(df, pipeline_grouped)
})

test_that("filter(row_number() <= 3) per group matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_filter_grouped_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(dplyr::row_number() <= 3)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# min_rank(desc(x)) -- documented gap: desc() inside a rank-family call's
# own argument doesn't parse at all (a pre-existing ir.R limitation shared
# with mutate()/summarise(), not something W5 introduces or fixes), so the
# predicate falls to the opaque-expression path and, since `x` isn't
# resolvable without a data mask, errors with the documented message
# rather than silently misbehaving or crashing.
# =============================================================================

test_that("filter(min_rank(desc(x)) <= 2): desc() inside rank's own arg does not parse (documented gap)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(3, 1, 2))
  gdf <- tbl_gpu(df)

  expect_error(
    gdf |> dplyr::filter(dplyr::min_rank(dplyr::desc(x)) <= 2) |> collect(),
    "filter\\(\\) only supports comparisons"
  )
})

# =============================================================================
# lag()/lead() -- NA-first-row semantics
# =============================================================================

test_that("filter(lag(x) > 0) drops rows where lag() is NA, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, -1, 2, -2, 3))
  pipeline <- function(d) dplyr::filter(d, dplyr::lag(x) > 0)
  result <- expect_same_as_dplyr(df, pipeline)
  # The very first row's lag() is NA -- NA > 0 is NA, dplyr drops it (not
  # kept), so row 1 (x = 1) must be ABSENT from the result.
  expect_false(1 %in% result$gpu$x[result$gpu$x == 1 & seq_len(nrow(result$gpu)) == 1])
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(lag(x) > 0) grouped, with NA-first-row-per-group semantics", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b"),
    x = c(1, -1, 2, 3, -3, 4)
  )
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(dplyr::lag(x) > 0)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Multi-dot AND semantics -- the main semantic trap (task requirement #3)
# =============================================================================

test_that("filter(x > 0, cumsum(x) < 10) evaluates every dot against the ORIGINAL data, then ANDs (not sequential)", {
  skip_if_no_gpu()
  # Designed so the two readings disagree: recomputing cumsum() only over
  # the x > 0-surviving subset (x = 3,3,3,3) would give cumsum 3,6,9,12,
  # keeping only the first three under a `< 10` threshold. Evaluating
  # cumsum() over the ORIGINAL x (all 8 rows) gives a cumsum that never
  # reaches 10 (values are -5,-2,-7,-4,-9,-6,-11,-8), so ALL FOUR x==3
  # rows survive -- this is dplyr's real, verified behavior.
  df <- data.frame(x = c(-5, 3, -5, 3, -5, 3, -5, 3))

  oracle <- dplyr::filter(df, x > 0, cumsum(x) < 10)
  expect_equal(nrow(oracle), 4)
  expect_true(all(oracle$x == 3))

  pipeline <- function(d) dplyr::filter(d, x > 0, cumsum(x) < 10)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter(cumsum(x) < 10, x > 2) (window dot first) matches dplyr's AND-against-original-data semantics", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 5, 2, 8, 3, 9, 1, 2))
  pipeline <- function(d) dplyr::filter(d, cumsum(x) < 10, x > 2)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("multiple window dots share CSE and both AND against original data", {
  skip_if_no_gpu()
  df <- oracle_filter_grouped_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x) - 10, dplyr::n() > 2)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Window + plain compound in ONE dot
# =============================================================================

test_that("filter(x > mean(x) & y < 5) (window + plain compound in one dot) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b"),
    x = c(1, 5, 3, 10, 2, 8),
    y = c(1, 10, 1, 10, 1, 10)
  )
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x) & y < 5)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Row-order preservation on shuffled data (task requirement #2)
# =============================================================================

test_that("grouped window filter() preserves ORIGINAL row order on shuffled data", {
  skip_if_no_gpu()
  df <- oracle_filter_shuffled_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x))

  result <- expect_same_as_dplyr(df, pipeline)
  # expect_same_as_dplyr() already checks values match; this additionally
  # pins that the SURVIVING ids come back in ascending (original) order,
  # not group-key order (which slice()'s window path uses instead).
  expect_true(all(diff(result$gpu$id) > 0))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("ungrouped window filter() preserves ORIGINAL row order on shuffled data", {
  skip_if_no_gpu()
  df <- oracle_filter_shuffled_df()
  pipeline <- function(d) dplyr::filter(d, x > mean(x))

  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(all(diff(result$gpu$id) > 0))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Edge cases: 0-row, single-row groups, all-filtered-out groups
# =============================================================================

test_that("window filter() on a 0-row table matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = character(0), x = numeric(0))

  pipeline_ungrouped <- function(d) dplyr::filter(d, x > mean(x))
  expect_same_as_dplyr(df, pipeline_ungrouped)
  expect_same_as_dplyr_lazy(df, pipeline_ungrouped)

  pipeline_grouped <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x))
  expect_same_as_dplyr(df, pipeline_grouped)
  expect_same_as_dplyr_lazy(df, pipeline_grouped)
})

test_that("window filter() with single-row groups matches dplyr (mean(x) == x, always FALSE for '>')", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b", "c"), x = c(5, 10, 15))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("window filter() where an entire group is filtered out matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(100, 200, 1, 2))
  # Group "a"'s values are both far above their own (much larger) group
  # mean scaled by 10, so every row is dropped for "a"; group "b" keeps
  # its row(s) above its own scaled mean.
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x > mean(x) * 10)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_false("a" %in% result$gpu$g)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Fallback: genuinely unplannable window shapes (task requirement #1)
#
# As of Phase 5 task W4, every window kind currently in the registry
# (shift/rank/scan/agg) is lowerable (window_spec_lowerable(), R/lower.R),
# so there is no REAL dplyr syntax that reaches the "unlowerable spec"
# fallback branch today. This test exercises that branch directly by
# temporarily registering a fake window-class op with an unsupported
# `window$kind` (mirroring how the registry is structured for genuine
# window ops) -- proving filter() falls back to CPU evaluation (with a
# notification) instead of crashing, exactly like mutate_window_fallback()
# does for the same, currently-unreachable-via-real-syntax case.
# =============================================================================

test_that("a window spec gpu_window() can't lower falls back to CPU evaluation with a notification, not a crash", {
  skip_if_no_gpu()

  # A real, CPU-side identity function -- so the fallback's re-run of the
  # actual `dplyr::filter()` call (on the collected tibble) evaluates to
  # a correct, independently-verifiable oracle result.
  bogus_win <- function(x) x

  testthat::local_mocked_bindings(
    ir_call_registry = local({
      r <- ir_call_registry
      r[["bogus_win"]] <- list(
        arity = 1L, parse = NULL, type = type_arg1, lower = NULL,
        window = list(kind = "unsupported_kind_for_test")
      )
      r
    }),
    .package = "cuplyr"
  )

  df <- data.frame(x = c(1, 2, 3, 4, 5))
  gdf <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- gdf |> dplyr::filter(bogus_win(x) > 2) |> collect(),
    "fell back to CPU evaluation"
  )

  expect_equal(sort(result$x), c(3, 4, 5))
})
