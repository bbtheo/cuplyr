# Mirrored dplyr-oracle tests for rename()/rename_with() semantics, and the
# select(new = old) rename fix (Phase 3, task 3)
#
# rename()/rename_with() are GPU-native (schema-only -- no GPU work at all,
# see R/rename.R, R/ast.R::ast_rename(), R/lower.R::lower_rename()).
# select()'s `new = old` rename-on-select was previously silently dropped
# (R/select.R discarded `names(tidyselect::eval_select(...))`); this file
# also covers that fix.
#
# Every test compares a dplyr pipeline run against a plain tibble (the
# oracle) with the same pipeline run against a `tbl_gpu`, in both eager and
# lazy exec modes, via `expect_same_as_dplyr()`/`expect_same_as_dplyr_lazy()`
# (see helper-oracle.R).

oracle_rename_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3, 4, 5),
    y = c(10, 20, 30, 40, 50),
    g = c(1, 1, 2, 2, 3),
    s = c("a", "b", "c", "d", "e")
  )
}

# =============================================================================
# rename(): basic new = old, position/data untouched
# =============================================================================

test_that("rename(new = old) only changes names, matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename(d, a = x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename() of multiple columns matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename(d, a = x, b = y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename() with no dots is a no-op, matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename(d)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename() swap (x = y, y = x) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename(d, x = y, y = x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename(all_of(c(new = old))) tidyselect form matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  new_map <- c(a = "x")
  pipeline <- function(d) dplyr::rename(d, dplyr::all_of(new_map))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename() to an existing (un-renamed) name errors", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  gt <- tbl_gpu(df)

  expect_error(dplyr::rename(gt, y = x), "[Nn]ames must be unique")
})

# =============================================================================
# rename(): renaming a group column updates group_vars()
# =============================================================================

test_that("rename() of a grouped column updates group_vars()", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::rename(grp = g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# rename(): factor levels survive a rename
# =============================================================================

test_that("rename() preserves factor levels on the renamed column", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1, 1, 2),
    f = factor(c("lo", "hi", "lo"), levels = c("lo", "hi"))
  )
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::rename(fac = f) |> collect()

  expect_true(is.factor(result$fac))
  expect_equal(levels(result$fac), c("lo", "hi"))
  expect_equal(as.character(result$fac), as.character(df$f))
})

# =============================================================================
# rename_with(): plain, .cols selection, lambda
# =============================================================================

test_that("rename_with(toupper) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename_with(d, toupper)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename_with(toupper, .cols=) matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename_with(d, toupper, .cols = c(x, y))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename_with() with a lambda matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename_with(d, ~ paste0(.x, "_2"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename_with() with tidyselect .cols helper matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::rename_with(d, toupper, .cols = dplyr::starts_with("s"))
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("rename_with() producing duplicate names errors", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  gt <- tbl_gpu(df)

  expect_error(
    dplyr::rename_with(gt, function(x) rep("dup", length(x)), .cols = c(x, y)),
    "[Nn]ames must be unique"
  )
})

# =============================================================================
# select(new = old): the rename-on-select fix
# =============================================================================

test_that("select(new = old) renames while selecting", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::select(d, a = x, y)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("select(everything()) keeps original names", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::select(d, dplyr::everything())
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("select() mixing a partial rename with plain selection matches dplyr", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) dplyr::select(d, renamed_x = x, y, g)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("select(new = old) on a grouped column updates group_vars()", {
  skip_if_no_gpu()
  df <- oracle_rename_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::select(grp = g, x)
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("select(new = old) preserves factor levels", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1, 1, 2),
    f = factor(c("lo", "hi", "lo"), levels = c("lo", "hi"))
  )
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::select(fac = f) |> collect()

  expect_true(is.factor(result$fac))
  expect_equal(levels(result$fac), c("lo", "hi"))
})

# =============================================================================
# No-fallback assertions: rename()/rename_with()/select(rename) are
# GPU-native now, never routed through gpu_fallback()
# =============================================================================

test_that("rename()/rename_with()/select(rename) never trigger the CPU fallback notifier", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(oracle_rename_df())

  expect_no_error(gt |> dplyr::rename(a = x))
  expect_no_error(gt |> dplyr::rename())
  expect_no_error(gt |> dplyr::rename(x = y, y = x))
  expect_no_error(gt |> dplyr::rename_with(toupper))
  expect_no_error(gt |> dplyr::rename_with(toupper, .cols = c(x, y)))
  expect_no_error(gt |> dplyr::group_by(g) |> dplyr::rename(grp = g))
  expect_no_error(gt |> dplyr::select(a = x, y))
  expect_no_error(gt |> dplyr::select(dplyr::everything()))
})
