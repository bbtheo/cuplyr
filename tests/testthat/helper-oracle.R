# Mirrored oracle test infrastructure
#
# This file provides the shared harness that every future dplyr-parity test
# (`test-dplyr-<file>.R`, per scratchpad/todo.md Phase 0) will build on:
#
#   expect_same_as_dplyr(df, pipeline, ...)       - eager GPU vs dplyr-on-tibble
#   expect_same_as_dplyr_lazy(df, pipeline, ...)  - lazy GPU vs dplyr-on-tibble
#
# The idea: write ONE `pipeline` function (a plain function of one table
# argument, using dplyr verbs) and run it against a tibble (the "oracle") and
# against a tbl_gpu (eager or lazy). Both results are normalized and compared.
#
# Design decisions (documented here since they affect every parity test):
#
# 1. Column types: by default we compare with `ignore_col_types = FALSE`,
#    i.e. types must match exactly. Known, tracked divergences (e.g. mutate()
#    always tagging FLOAT64 even for INT32+INT32, per scratchpad/todo.md
#    Phase 0 "Fix mutate output-type inference divergence") mean some
#    pipelines can only be compared with `ignore_col_types = TRUE`, which
#    coerces all numeric-family columns (integer/double/logical) to double
#    on BOTH sides before comparing, so values are still checked exactly but
#    integer-vs-double distinctions are ignored.
#
# 2. Row order: dplyr guarantees row order is preserved (filter/mutate/select)
#    or well-defined (arrange). Some GPU operations (group-by aggregation,
#    joins) do not guarantee a particular row order even though the current
#    implementation happens to produce one deterministically. Callers pass
#    `arrange_by = c("col1", "col2")` to sort both sides before comparing
#    when order is not part of the contract being tested.
#
# 3. Grouping metadata: `collect.tbl_gpu()` always returns a plain tibble
#    (see R/collect.R) -- it never restores `group_by()` structure, unlike
#    `dplyr::collect()` on a database backend. So there is nothing to compare
#    on the *post-collect* object. Instead we capture `dplyr::group_vars()`
#    on the pipeline's return value BEFORE collecting on both sides: the
#    oracle's tibble/grouped-df result, and the GPU's tbl_gpu result (via
#    `group_vars.tbl_gpu()`, i.e. `$groups`). This catches real divergences,
#    e.g. cuplyr's `summarise()` always drops all grouping (`.groups` is
#    effectively always "drop", see R/summarise.R:123) where dplyr's default
#    is "drop_last". Phase 6 of the roadmap is expected to fix this; until
#    then, oracle pipelines that call summarise() should pass
#    `.groups = "drop"` explicitly so both sides agree (see
#    test-oracle-smoke.R).

# =============================================================================
# Internal helpers
# =============================================================================

# Coerce numeric-family columns (integer/double/logical) to double, on both
# sides, so type mismatches (e.g. INT32 vs FLOAT64) don't fail the comparison
# when only *values* are under test.
# @keywords internal
oracle_coerce_numeric <- function(tbl) {
  # NB: is.numeric() is FALSE for logicals, so test both to match the
  # documented integer/double/logical contract.
  numeric_family <- function(x) is.numeric(x) || is.logical(x)
  dplyr::mutate(tbl, dplyr::across(dplyr::where(numeric_family), as.double))
}

# Normalize a pipeline result (tibble or data.frame) for comparison:
# drop grouping structure (compared separately) and coerce to a tibble.
# @keywords internal
oracle_normalize <- function(x, arrange_by = NULL, ignore_col_types = FALSE) {
  tbl <- dplyr::ungroup(tibble::as_tibble(x))

  if (ignore_col_types) {
    tbl <- oracle_coerce_numeric(tbl)
  }

  if (!is.null(arrange_by)) {
    tbl <- dplyr::arrange(tbl, dplyr::across(dplyr::all_of(arrange_by)))
  }

  tbl
}

# Get group_vars() of a pre-collect pipeline result, whether it's a
# tbl_gpu (use the package's own group_vars.tbl_gpu / $groups) or a
# tibble/grouped_df/data.frame (use dplyr::group_vars()).
# @keywords internal
oracle_group_vars <- function(x) {
  if (is_tbl_gpu(x)) {
    return(x$groups)
  }
  dplyr::group_vars(x)
}

# Shared comparison body used by both expect_same_as_dplyr() and
# expect_same_as_dplyr_lazy(). `label` distinguishes error messages
# between the eager and lazy variants.
# @keywords internal
oracle_compare <- function(df, pipeline, arrange_by, ignore_col_types, label) {
  stopifnot(is.data.frame(df), is.function(pipeline))

  # --- CPU oracle ---
  oracle_input <- dplyr::as_tibble(df)
  oracle_result <- pipeline(oracle_input)
  oracle_groups <- oracle_group_vars(oracle_result)

  # --- GPU side ---
  gpu_input <- tbl_gpu(df)
  if (identical(label, "lazy")) {
    gpu_input <- as_lazy(gpu_input)
  }
  gpu_result <- pipeline(gpu_input)
  gpu_groups <- oracle_group_vars(gpu_result)

  testthat::expect_equal(
    gpu_groups, oracle_groups,
    info = sprintf(
      "group_vars() mismatch between GPU (%s) and dplyr oracle result", label
    )
  )

  gpu_collected <- collect(gpu_result)

  oracle_tbl <- oracle_normalize(oracle_result, arrange_by, ignore_col_types)
  gpu_tbl <- oracle_normalize(gpu_collected, arrange_by, ignore_col_types)

  testthat::expect_equal(
    names(gpu_tbl), names(oracle_tbl),
    info = sprintf("column names mismatch between GPU (%s) and dplyr oracle", label)
  )

  testthat::expect_equal(
    gpu_tbl, oracle_tbl,
    info = sprintf("GPU (%s) result differs from dplyr oracle", label)
  )

  invisible(list(oracle = oracle_tbl, gpu = gpu_tbl))
}

# =============================================================================
# Public helpers
# =============================================================================

#' Compare an eager tbl_gpu pipeline against dplyr-on-tibble
#'
#' Runs `pipeline(dplyr::as_tibble(df))` as the CPU oracle and
#' `pipeline(tbl_gpu(df)) |> collect()` on GPU, then compares.
#'
#' @param df A plain data.frame.
#' @param pipeline A function taking one table argument (tibble or tbl_gpu)
#'   and returning a table (may be grouped/ungrouped; may be a tbl_gpu that
#'   has not yet been collected).
#' @param arrange_by Optional character vector of column names. When given,
#'   both the oracle and GPU results are arranged by these columns before
#'   comparing, for operations that don't guarantee row order (e.g. joins,
#'   group-by aggregation).
#' @param ignore_col_types Default `FALSE`. When `TRUE`, numeric-family
#'   columns (integer/double/logical) are coerced to double on both sides
#'   before comparing, to paper over known type divergences (see file header
#'   comment).
#' @return Invisibly, a list with the normalized `oracle` and `gpu` tibbles.
expect_same_as_dplyr <- function(df, pipeline, ...,
                                  arrange_by = NULL,
                                  ignore_col_types = FALSE) {
  oracle_compare(df, pipeline, arrange_by, ignore_col_types, label = "eager")
}

#' Compare a lazy tbl_gpu pipeline against dplyr-on-tibble
#'
#' Same contract as [expect_same_as_dplyr()], but runs the GPU side through
#' `tbl_gpu(df) |> as_lazy()`, verifying that the lazy execution path
#' (AST build -> optimizer -> lower_and_execute on `collect()`) also matches
#' the dplyr oracle.
#'
#' @inheritParams expect_same_as_dplyr
#' @return Invisibly, a list with the normalized `oracle` and `gpu` tibbles.
expect_same_as_dplyr_lazy <- function(df, pipeline, ...,
                                      arrange_by = NULL,
                                      ignore_col_types = FALSE) {
  oracle_compare(df, pipeline, arrange_by, ignore_col_types, label = "lazy")
}
