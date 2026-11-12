# Mirrored dplyr-oracle tests for relocate() semantics (Phase 3, task 3)
#
# relocate() is GPU-native: the reorder itself reuses ast_select (a plain
# gpu_select() gather, R/relocate.R); a rename (`relocate(df, a = x)`) is
# schema-only, stacking an ast_rename node on top (R/ast.R, R/rename.R) --
# see R/relocate.R's file header comment for the tidyselect::eval_relocate()
# contract this mirrors from dplyr::relocate.data.frame() itself.

oracle_relocate_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3, 4, 5),
    y = c(10, 20, 30, 40, 50),
    z = c(100, 200, 300, 400, 500),
    g = c(1, 1, 2, 2, 3)
  )
}

# =============================================================================
# Default (no .before/.after): moves selected columns to the front
# =============================================================================

test_that("relocate() with no .before/.after moves columns to the front", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, z)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate() of multiple columns preserves their relative order", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, z, x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# .before= / .after=
# =============================================================================

test_that("relocate(.before=) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, z, .before = y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate(.after=) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, x, .after = y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate(.before= and .after=) errors", {
  skip_if_no_gpu()
  gt <- tbl_gpu(oracle_relocate_df())
  expect_error(dplyr::relocate(gt, z, .before = y, .after = x), "before.*after|after.*before")
})

# =============================================================================
# Tidyselect helpers in all three slots
# =============================================================================

test_that("relocate() with everything() in the selection slot matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, dplyr::everything(), .after = x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate() with a tidyselect helper in .before matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, z, .before = dplyr::starts_with("y"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate() with a tidyselect helper in .after matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, x, .after = dplyr::starts_with("y"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate() with a tidyselect helper in the selection slot matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, dplyr::starts_with("z"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# relocate() can rename while relocating
# =============================================================================

test_that("relocate(new = old) renames while relocating", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) dplyr::relocate(d, renamed = x, .after = y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Relocating group columns
# =============================================================================

test_that("relocate() of a group column preserves group_vars() at the new position", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::relocate(g, .after = z)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("relocate(new = old) of a group column updates group_vars() to the new name", {
  skip_if_no_gpu()
  df <- oracle_relocate_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::relocate(grp = g, .after = z)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# No-fallback assertion: relocate() is GPU-native now
# =============================================================================

test_that("relocate() never triggers the CPU fallback notifier", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(oracle_relocate_df())

  expect_no_error(gt |> dplyr::relocate(z))
  expect_no_error(gt |> dplyr::relocate(z, .before = y))
  expect_no_error(gt |> dplyr::relocate(x, .after = y))
  expect_no_error(gt |> dplyr::relocate(renamed = x))
  expect_no_error(gt |> dplyr::group_by(g) |> dplyr::relocate(g, .after = z))
})
