# Mirrored dplyr-oracle tests for window functions wired into mutate()
# (Phase 5, task W3)
#
# Every test compares a window-function-using mutate() pipeline run against a
# plain tibble (the oracle) with the same pipeline run against a tbl_gpu, in
# both eager and lazy exec modes, via expect_same_as_dplyr()/
# expect_same_as_dplyr_lazy() (see helper-oracle.R).
#
# Scope (see scratchpad/phase5_window_design.md section 8's W3 row, and
# R/lower.R's validate_window_spec_w2()/window_spec_w3_lowerable()): only
# row_number() (bare), lag()/lead(), cumsum()/cummax()/cummin()/cumprod()
# and their R desugars (cummean()/cumall()/cumany()/consecutive_id()) are
# GPU-native this wave. Anything needing a rank-with-value spec
# (row_number(x), min_rank(), ...) or the aggregate family
# (mean()/sum()/min()/max()/n()/sd()/var()/first()/last()/nth(), and
# ntile(), which desugars into both) still falls back to CPU evaluation --
# see the "fallback" section at the bottom of this file.

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
# Fallback: not-yet-lowerable window specs (W4 work) fall back cleanly
# =============================================================================

test_that("mutate() with mean(x) inside an expression falls back to CPU with notification", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  withr::local_options(cuplyr.fallback = "warn")
  gt <- tbl_gpu(df)

  expect_warning(
    result <- dplyr::mutate(gt, y = x - mean(x)),
    "mutate.*fell back to CPU evaluation"
  )
  oracle <- dplyr::mutate(df, y = x - mean(x))
  expect_equal(tibble::as_tibble(collect(result)), tibble::as_tibble(oracle))
})

test_that("mutate() with ntile() falls back to CPU with notification", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  withr::local_options(cuplyr.fallback = "warn")
  gt <- tbl_gpu(df)

  expect_warning(
    result <- dplyr::mutate(gt, y = dplyr::ntile(x, 2)),
    "mutate.*fell back to CPU evaluation"
  )
  oracle <- dplyr::mutate(df, y = dplyr::ntile(x, 2))
  expect_equal(tibble::as_tibble(collect(result)), tibble::as_tibble(oracle))
})

test_that("mutate() with min_rank() falls back to CPU with notification", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(3, 1, 4, 2))
  withr::local_options(cuplyr.fallback = "warn")
  gt <- tbl_gpu(df)

  expect_warning(
    result <- dplyr::mutate(gt, r = dplyr::min_rank(x)),
    "mutate.*fell back to CPU evaluation"
  )
  oracle <- dplyr::mutate(df, r = dplyr::min_rank(x))
  expect_equal(tibble::as_tibble(collect(result)), tibble::as_tibble(oracle))
})

test_that("mutate() mixing a native cumsum() with a not-yet-lowerable mean() falls back entirely", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))
  withr::local_options(cuplyr.fallback = "warn")
  gt <- tbl_gpu(df)

  expect_warning(
    result <- dplyr::mutate(gt, cs = cumsum(x), y = x - mean(x)),
    "mutate.*fell back to CPU evaluation"
  )
  oracle <- dplyr::mutate(df, cs = cumsum(x), y = x - mean(x))
  expect_equal(tibble::as_tibble(collect(result)), tibble::as_tibble(oracle))
})
