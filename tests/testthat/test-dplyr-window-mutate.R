# Mirrored dplyr-oracle tests for window functions wired into mutate()
# (Phase 5, tasks W3 + W4)
#
# Every test compares a window-function-using mutate() pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a tbl_gpu, in
# both eager and lazy exec modes, via expect_same_as_dplyr()/
# expect_same_as_dplyr_lazy() (see helper-oracle.R).
#
# Scope: as of Phase 5 task W4, every window kind (shift/rank/scan/agg --
# see R/lower.R's validate_window_spec()/window_spec_lowerable()) is
# GPU-native: row_number() (bare and ranked), lag()/lead(),
# cumsum()/cummax()/cummin()/cumprod() and their R desugars
# (cummean()/cumall()/cumany()/consecutive_id()) (all W3), plus
# min_rank()/dense_rank()/percent_rank()/cume_dist(),
# mean()/sum()/min()/max()/n()/sd()/var()/first()/last()/nth(), and
# ntile() (which desugars into both a rank-kind and an agg-kind spec) --
# all W4. The rank family gets its own dedicated oracle file
# (test-dplyr-rank.R); order_by()/with_order()/lag()'s own order_by= get
# test-dplyr-order-by.R. This file covers the agg family (mean()/sum()/
# min()/max()/n()/sd()/var()/first()/last()/nth()) and ntile(), plus the
# W3 scan/shift coverage above.
#
# mean()/sum()/min()/max()/sd()/var() match R's own na.rm = FALSE default
# (ANY NA in a group makes the WHOLE group's result NA/NaN as appropriate)
# -- a real, deliberate divergence from src/ops_groupby.cpp's summarise()
# aggregations, which do NOT propagate NA this way (a separate,
# out-of-scope, pre-existing gap; see src/ops_window.cpp's own comment on
# the fix applied only to the window-function path). Test fixtures below
# use double-typed (not integer) value columns for sum()/mean()/etc. to
# avoid a second, unrelated pre-existing divergence: cuplyr's sum() of an
# INT32 column returns FLOAT64 in both summarise() and this window path
# (reusing R/ast.R's make_aggregation() type table verbatim, per design),
# where dplyr itself keeps a plain integer sum when there's no overflow --
# tracked separately, not a window-function-specific gap.

# =============================================================================
# Shared fixtures
# =============================================================================

# Grouped, with NAs positioned to exercise sticky-NA propagation within each
# group (group "a" ends in NA -> every later cumulative value in "a" is NA
# too; group "b" has an NA in the middle).
oracle_window_grouped_df <- function() {
  data.frame(
    id = 1:10,
    g = c("a", "a", "a", "b", "b", "b", "b", "c", "c", "c"),
    x = c(1, 2, NA, 4, 5, 6, NA, 8, 9, 10),
    stringsAsFactors = FALSE
  )
}

oracle_window_ungrouped_df <- function() {
  data.frame(x = c(1, 2, NA, 4, 5), stringsAsFactors = FALSE)
}

# Rows are NOT sorted by group, so lag()/lead()/row_number()/cumsum() must
# each preserve ORIGINAL row order in the output while still computing
# within-group values correctly -- this is the row-order-preservation
# contract grouped mutate() has (unlike grouped slice()).
oracle_window_shuffled_df <- function() {
  data.frame(
    id = 1:9,
    g = c("b", "a", "c", "a", "b", "c", "a", "b", "c"),
    x = c(10, 1, 100, 2, 20, 200, 3, 30, 300),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# cumsum()/cummax()/cummin()/cumprod() -- sticky-NA semantics
# =============================================================================

test_that("cumsum()/cummax()/cummin()/cumprod() with NAs match dplyr's sticky-NA semantics, ungrouped", {
  skip_if_no_gpu()
  df <- oracle_window_ungrouped_df()
  pipeline <- function(d) {
    dplyr::mutate(d, cs = cumsum(x), mx = cummax(x), mn = cummin(x), pr = cumprod(x))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("cumsum()/cummax()/cummin()/cumprod() with NAs match dplyr's sticky-NA semantics, grouped", {
  skip_if_no_gpu()
  df <- oracle_window_grouped_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(cs = cumsum(x), mx = cummax(x), mn = cummin(x), pr = cumprod(x))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# cummean()
# =============================================================================

test_that("cummean() matches dplyr, grouped and ungrouped, with NAs", {
  skip_if_no_gpu()
  df_ungrouped <- oracle_window_ungrouped_df()
  pipeline <- function(d) dplyr::mutate(d, cm = dplyr::cummean(x))
  expect_same_as_dplyr(df_ungrouped, pipeline)
  expect_same_as_dplyr_lazy(df_ungrouped, pipeline)

  df_grouped <- oracle_window_grouped_df()
  grouped_pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(cm = dplyr::cummean(x))
  expect_same_as_dplyr(df_grouped, grouped_pipeline)
  expect_same_as_dplyr_lazy(df_grouped, grouped_pipeline)
})

# =============================================================================
# cumall()/cumany() -- logical columns with NAs
# =============================================================================

test_that("cumall()/cumany() with NAs match dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "a", "b", "b", "b", "b"),
    x = c(TRUE, TRUE, NA, TRUE, FALSE, NA, TRUE, FALSE)
  )
  pipeline <- function(d) dplyr::mutate(d, ca = dplyr::cumall(x), cy = dplyr::cumany(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  grouped_pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(ca = dplyr::cumall(x), cy = dplyr::cumany(x))
  }
  expect_same_as_dplyr(df, grouped_pipeline)
  expect_same_as_dplyr_lazy(df, grouped_pipeline)
})

# =============================================================================
# lag()/lead() -- n=, default=, row-order preservation on shuffled groups
# =============================================================================

test_that("lag()/lead() with n=/default= preserve row order on shuffled grouped data", {
  skip_if_no_gpu()
  df <- oracle_window_shuffled_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(
        lg1 = dplyr::lag(x),
        lg2 = dplyr::lag(x, n = 2, default = -1),
        ld1 = dplyr::lead(x),
        ld2 = dplyr::lead(x, n = 2, default = -1)
      )
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(result$gpu$id, df$id)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("lag()/lead() default= (untyped NA) matches dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- oracle_window_ungrouped_df()
  pipeline <- function(d) dplyr::mutate(d, lg = dplyr::lag(x), ld = dplyr::lead(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# row_number() -- bare form
# =============================================================================

test_that("row_number() bare matches dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b", "b"), x = c(10, 20, 30, 40, 50))

  pipeline <- function(d) dplyr::mutate(d, rn = dplyr::row_number())
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  grouped_pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(rn = dplyr::row_number())
  expect_same_as_dplyr(df, grouped_pipeline)
  expect_same_as_dplyr_lazy(df, grouped_pipeline)
})

# =============================================================================
# consecutive_id() -- including NA runs
# =============================================================================

test_that("consecutive_id() with NA runs matches dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "a", "a", "b", "b", "b", "b"),
    x = c(1, NA, NA, 2, 2, 1, 2, 2, NA)
  )

  pipeline <- function(d) dplyr::mutate(d, cid = dplyr::consecutive_id(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  grouped_pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(cid = dplyr::consecutive_id(x))
  expect_same_as_dplyr(df, grouped_pipeline)
  expect_same_as_dplyr_lazy(df, grouped_pipeline)
})

# =============================================================================
# Window function combined with arithmetic / nested (window-of-window)
# =============================================================================

test_that("window function combined with arithmetic (x - lag(x)) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b", "b"), x = c(1, 3, 6, 10, 20, 25))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(delta = x - dplyr::lag(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("nested window functions (lag(cumsum(x))) resolve across decomposition stages", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5, 6))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(z = dplyr::lag(cumsum(x)))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("a later dot referencing an earlier window dot's own output matches dplyr (a = cumsum(x), b = lag(a))", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5, 6))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(a = cumsum(x), b = dplyr::lag(a))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("multiple window dots in one mutate() call, some cross-referencing, match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5, 6))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(
        rn = dplyr::row_number(),
        cs = cumsum(x),
        lg = dplyr::lag(cs),
        delta = cs - lg
      )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .keep= interplay with a window dot
# =============================================================================

test_that(".keep='used' with a window dot keeps only columns actually referenced (plus groups/new)", {
  skip_if_no_gpu()
  df <- data.frame(id = 1:6, g = c("a", "a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5, 6))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(cs = cumsum(x), .keep = "used")

  result <- expect_same_as_dplyr(df, pipeline)
  expect_setequal(names(result$gpu), c("g", "x", "cs"))
  expect_false("id" %in% names(result$gpu))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that(".keep='unused' with a window dot drops columns not referenced by any dot", {
  skip_if_no_gpu()
  df <- data.frame(id = 1:6, g = c("a", "a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5, 6))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(cs = cumsum(x), .keep = "unused")

  result <- expect_same_as_dplyr(df, pipeline)
  expect_setequal(names(result$gpu), c("g", "id", "cs"))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Mixed grouped table sizes: single-row groups, 0-row tables
# =============================================================================

test_that("window functions on a grouped table with single-row groups matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "b", "b", "c", "c", "c"),
    x = c(1, 2, 3, 4, 5, 6)
  )
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(cs = cumsum(x), lg = dplyr::lag(x), ld = dplyr::lead(x), rn = dplyr::row_number())
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("window functions on a 0-row table match dplyr, ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(x = numeric(0))
  pipeline <- function(d) {
    dplyr::mutate(d, cs = cumsum(x), lg = dplyr::lag(x), rn = dplyr::row_number())
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("window functions on a 0-row table match dplyr, grouped", {
  skip_if_no_gpu()
  df <- data.frame(g = character(0), x = numeric(0))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(cs = cumsum(x), lg = dplyr::lag(x), rn = dplyr::row_number())
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# mean()/sum()/min()/max()/n() -- now GPU-native (Phase 5, task W4)
#
# Previously (W3) these fell back to CPU evaluation; flipped to native
# parity tests here per the W4 task ("mean(x)/ntile/min_rank now NATIVE:
# flip those fallback tests to no-fallback parity tests").
# =============================================================================

test_that("mutate() with mean(x) inside an expression is GPU-native and matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  pipeline <- function(d) dplyr::mutate(d, y = x - mean(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("x - mean(x) grouped centering matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "b", "b", "b", "b"),
    x = c(1, 2, 3, 10, 20, 30, 40)
  )
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(centered = x - mean(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() with ntile() is GPU-native and matches dplyr (values only -- see file header re: type)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  pipeline <- function(d) dplyr::mutate(d, y = dplyr::ntile(x, 2))
  # ignore_col_types = TRUE: ntile()'s R desugar (R/ir.R::ir_parse_ntile())
  # infers FLOAT64 (its formula's "/" true-division step forces this),
  # while dplyr's own ntile() is genuinely integer -- a pre-existing (W1),
  # values-only-affects-formatting divergence, not something this task
  # introduced or is in scope to fix.
  expect_same_as_dplyr(df, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = TRUE)
})

test_that("mutate() with ntile() matches dplyr, grouped, with NAs and a non-multiple group size", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c("a", "a", "a", "a", "a", "b", "b", "b"),
    x = c(5, 3, NA, 1, 4, 10, 20, 30)
  )
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(nt = dplyr::ntile(x, 3), nt2 = dplyr::ntile(n = 2))
  expect_same_as_dplyr(df, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = TRUE)
})

test_that("mutate() with min_rank() is GPU-native and matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(3, 1, 4, 2))
  pipeline <- function(d) dplyr::mutate(d, r = dplyr::min_rank(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() mixing a native cumsum() with a native mean() in the same call matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  pipeline <- function(d) dplyr::mutate(d, cs = cumsum(x), y = x - mean(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# sum()/min()/max()/n() -- agg family, ungrouped and grouped
# =============================================================================

test_that("sum()/min()/max()/n() inside mutate() match dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(3, 1, 2, 6, 4))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(s = sum(x), mn = min(x), mx = max(x), cnt = dplyr::n())
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  ungrouped_pipeline <- function(d) dplyr::mutate(d, s = sum(x), mn = min(x), mx = max(x), cnt = dplyr::n())
  expect_same_as_dplyr(df, ungrouped_pipeline)
  expect_same_as_dplyr_lazy(df, ungrouped_pipeline)
})

test_that("mean()/sum()/min()/max() propagate NA per group (na.rm = FALSE default), matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(1, 2, NA, 4, 5))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(m = mean(x), s = sum(x), mn = min(x), mx = max(x))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(all(is.na(result$gpu$m[result$gpu$g == "a"])))
  expect_false(any(is.na(result$gpu$m[result$gpu$g == "b"])))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("sum(x, na.rm = TRUE) excludes NA per group, matching dplyr (Phase 5, task W9)", {
  # sum()'s ordinary registry entry only accepts the bare 1-arg form; the
  # 2-arg `sum(x, na.rm = TRUE)` shape is parsed by ir_parse_sum_call()
  # (R/ir.R), which desugars `x` to `coalesce(x, 0)` before the window sum
  # -- found while deleting the narrow grouped_aggregate_mutate() path
  # (Phase 5 task W9), which real, unmodified dplyr::add_tally(wt = )
  # depends on (dplyr:::tally_n() unconditionally emits
  # `base::sum(wt, na.rm = TRUE)`) -- see test-dplyr-count.R's own
  # "add_count(wt = ) with NA in the weight column" test for the
  # user-facing regression this was written to catch.
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(1, 2, NA, 4, 5))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(s = sum(x, na.rm = TRUE))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$s[result$gpu$g == "a"], rep(3, 3))
  expect_same_as_dplyr_lazy(df, pipeline)

  ungrouped_pipeline <- function(d) dplyr::mutate(d, s = sum(x, na.rm = TRUE))
  expect_same_as_dplyr(df, ungrouped_pipeline)
  expect_same_as_dplyr_lazy(df, ungrouped_pipeline)
})

test_that("sum(x, na.rm = FALSE) is a no-op, matching dplyr's own default", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a"), x = c(1, 2, NA))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(s = sum(x, na.rm = FALSE))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(all(is.na(result$gpu$s)))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("sum(x, na.rm = TRUE) with all-NA group gives 0, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b"), x = c(NA_real_, NA_real_, 5))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(s = sum(x, na.rm = TRUE))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# sd()/var() -- agg family, incl. single-row-group NA
# =============================================================================

test_that("sd()/var() windows match dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(1, 2, 3, 10, 20))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(sdv = sd(x), vr = var(x))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  ungrouped_pipeline <- function(d) dplyr::mutate(d, sdv = sd(x), vr = var(x))
  expect_same_as_dplyr(df, ungrouped_pipeline)
  expect_same_as_dplyr_lazy(df, ungrouped_pipeline)
})

test_that("sd()/var() give NA for a single-row group, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b"), x = c(5, 6, 7))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(sdv = sd(x), vr = var(x))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(is.na(result$gpu$sdv[result$gpu$g == "b"]))
  expect_true(is.na(result$gpu$vr[result$gpu$g == "b"]))
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# first()/last()/nth() -- agg family, incl. negative k and out-of-range
# =============================================================================

test_that("first()/last()/nth() windows match dplyr, grouped and ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "b", "b"), x = c(30, 10, 20, 5, 1))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(f = dplyr::first(x), l = dplyr::last(x), n2 = dplyr::nth(x, 2))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  ungrouped_pipeline <- function(d) {
    dplyr::mutate(d, f = dplyr::first(x), l = dplyr::last(x), n2 = dplyr::nth(x, 2))
  }
  expect_same_as_dplyr(df, ungrouped_pipeline)
  expect_same_as_dplyr_lazy(df, ungrouped_pipeline)
})

test_that("nth() with negative k (counts from the end) matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "a", "a", "b", "b", "b"), x = c(10, 20, 30, 40, 1, 2, 3))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(n_m1 = dplyr::nth(x, -1), n_m2 = dplyr::nth(x, -2))
  }
  result <- expect_same_as_dplyr(df, pipeline)
  expect_equal(result$gpu$n_m1[result$gpu$g == "a"], rep(40, 4))
  expect_equal(result$gpu$n_m2[result$gpu$g == "a"], rep(30, 4))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("nth() out of range (positive or negative) gives NA, matching dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = "a", x = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, n_big = dplyr::nth(x, 10), n_negbig = dplyr::nth(x, -10))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(all(is.na(result$gpu$n_big)))
  expect_true(all(is.na(result$gpu$n_negbig)))
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("first()/last() with NA at the picked position match dplyr (na_rm = FALSE default)", {
  skip_if_no_gpu()
  df <- data.frame(g = "a", x = c(NA, 2, 3, NA, 5))
  pipeline <- function(d) dplyr::mutate(d, f = dplyr::first(x), l = dplyr::last(x))
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(all(is.na(result$gpu$f)))
  expect_same_as_dplyr_lazy(df, pipeline)
})
