# Per-file classification for the dplyr conformance harness (Phase 13).
#
# `runnable_files()` / `out_of_scope_files()` below classify every file
# vendored into inst/dplyr-conformance/tests/ (test-*.R only; helper-*.R are
# always sourced by `load_helpers = TRUE`, never run as a test file
# themselves). `tools/run-dplyr-conformance.R` processes only the runnable
# ("CORE") set unless explicitly asked to also attempt the out-of-scope set.
#
# Classification policy (scratchpad/todo.md Phase 13's own out-of-scope
# bullet, verbatim categories: "db/SQL plumbing", "defunct/deprecation-warning
# files", "locale legacy" -- extended here with one more category, INTERNAL,
# per this task's own instructions, for files whose tests are ENTIRELY about
# dplyr's internal S3/generic contract for `grouped_df`/its own condition
# system rather than about verb *behavior* a GPU backend could ever satisfy
# or fail meaningfully -- e.g. `new_grouped_df()`, `dplyr_row_slice()`
# defaults, exact internal error-call plumbing). A file is only placed in
# INTERNAL if essentially ALL of its tests are of this shape; a file that
# mixes a few `dplyr:::`-internal assertions among otherwise-ordinary verb
# tests stays RUNNABLE, and any such individual failures are categorized in
# the generated report instead (that's what the report's "genuine gap vs.
# shim artifact vs. dplyr-internal usage" breakdown is for -- the MANIFEST
# is a coarser, file-level cut).
#
# NOTE ON THE "~55 CORE FILES" FIGURE: no enumerated 55-file list was found
# anywhere under scratchpad/ at the time this manifest was written (searched
# both todo.md and workflow_state.md for an explicit list) -- despite being
# referenced from workflow_state.md's Phase 13 kickoff notes as coming from
# "Phase-0 research", no such artifact exists in the repo. This manifest was
# instead built from scratch by applying the documented out-of-scope
# criteria to the full 81-file vendored corpus, which yields 65 runnable
# files, not ~55. Recorded here rather than silently reconciled.

out_of_scope <- list(
  # -- db/SQL plumbing --------------------------------------------------
  "test-src-dbi.R" = "DB: tests dplyr's DBI/SQL backend source() plumbing; cuplyr is a local in-process GPU backend, not a database backend -- N/A.",
  "test-copy-to.R" = "DB: tests copy_to() against DBI/SQL destinations; N/A for a GPU backend.",

  # -- defunct/deprecation-warning files ---------------------------------
  "test-defunct.R" = "DEPRECATED: tests dplyr functions that are fully defunct (hard-error on use); nothing for a backend to implement.",
  "test-defunct-each.R" = "DEPRECATED: tests the defunct `each()`-style verb suffixes removed from dplyr long ago.",
  "test-defunct-lazyeval.R" = "DEPRECATED: tests the defunct lazyeval (`_()`/`.dots=`) NSE interface dplyr replaced with rlang/tidy-eval.",
  "test-deprec-context.R" = "DEPRECATED: tests deprecated context functions (superseded by n()/cur_group() etc., already covered by test-context.R).",
  "test-deprec-do.R" = "DEPRECATED: tests deprecated do() call forms; do() itself is in-scope (CPU-fallback, see test-fallback.R) but this file is specifically about superseded argument forms.",
  "test-deprec-funs.R" = "DEPRECATED: tests deprecated funs()/fun_list() scoped-verb helper machinery, superseded by across().",
  "test-all-equal.R" = "DEPRECATED: `all_equal()` itself is deprecated in dplyr 1.2 (file's own first test asserts this); testing a deprecated comparison helper's own deprecation warning is not a GPU-backend verb concern.",

  # -- locale legacy ------------------------------------------------------
  "test-locale.R" = "LOCALE: tests string-collation/locale-dependent sorting behavior; out of scope, documented dependency on the C library's locale tables, not a cuplyr concern.",

  # -- INTERNAL: dplyr's own S3/generic contract for grouped_df, not verb behavior --
  "test-generics.R" = "INTERNAL: tests default methods of dplyr's own extension-point generics (dplyr_row_slice()/dplyr_col_modify()/dplyr_reconstruct() defaults, vec_restore() dispatch) operating on plain data.frame/grouped_df -- these are the CONTRACT other backends implement, not user-facing verb behavior; cuplyr's tbl_gpu implements this contract already (R/generics.R) but via its own internal representation, so poking dplyr's DEFAULT method implementations here doesn't exercise cuplyr code at all.",
  "test-grouped-df.R" = "INTERNAL: tests `new_grouped_df()`/`validate_grouped_df()` and grouped_df's own internal `.rows`-list-column structure directly; tbl_gpu's grouping representation is a character vector of column names (see CLAUDE.md), not a grouped_df object at all -- structurally inapplicable, not a gap.",
  "test-tbl.R" = "INTERNAL: tests `tbl_vars()`/`tbl_nongroup_vars()` against grouped_df's internal group-attribute representation directly (`attr(gdf, \"groups\")`); same grouped_df-internals mismatch as test-grouped-df.R.",
  "test-conditions.R" = "INTERNAL: tests dplyr's own internal error-condition class hierarchy and `dplyr_local_error_call()` call-context plumbing via exact `expect_snapshot(error = TRUE)` text on dplyr's own condition classes -- about dplyr's internal error machinery, not about whether cuplyr computes the right answer.",
  "test-utils.R" = "INTERNAL: tests non-exported dplyr:::-internal helper functions (e.g. `quo_is_variable_reference()`) directly, called via `:::` in the test file itself -- these are implementation details of dplyr's OWN NSE parsing, never invoked by cuplyr code at all.",
  "test-data-mask.R" = "INTERNAL: single test of dplyr's internal data-mask coercion for a degenerate 0x0 matrix-to-data.frame edge case; about dplyr's internal data-mask construction, not verb semantics."
)

# Everything else vendored is treated as CORE / runnable.
all_test_files <- function(tests_dir) {
  sort(basename(Sys.glob(file.path(tests_dir, "test-*.R"))))
}

runnable_files <- function(tests_dir) {
  all <- all_test_files(tests_dir)
  setdiff(all, names(out_of_scope))
}

out_of_scope_files <- function(tests_dir) {
  all <- all_test_files(tests_dir)
  intersect(all, names(out_of_scope))
}

out_of_scope_reason <- function(file) {
  out_of_scope[[file]] %||% NA_character_
}

`%||%` <- function(x, y) if (is.null(x)) y else x
