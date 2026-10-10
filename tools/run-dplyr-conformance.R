#!/usr/bin/env Rscript
# Run dplyr's own vendored test suite (inst/dplyr-conformance/tests/) against
# cuplyr's tbl_gpu backend and produce a per-file ledger plus a per-test TSV.
#
# Phase 13 (dplyr-parity roadmap): the goal is "the whole dplyr test suite runs
# on cuplyr", measured here rather than by the hand-mirrored oracle suite
# (tests/testthat/test-dplyr-*.R). F01 reworked the harness so that the number
# reflects cuplyr and not the harness (items C1-C8 of
# scratchpad/dplyr_feature_gaps.md section 4).
#
# This file is the thin CLI (load the package, parse args, loop over files,
# write the report). All logic lives in tools/conformance-lib.R, which the
# unit tests in tests/testthat/test-parity-F01-harness.R drive on fake files.
# Read the header of that file for the mechanism and why it is built this way:
#   - construction-site shim scoping (only fixtures that enter a pipeline are
#     uploaded; everything else stays a plain tibble/data.frame);
#   - strict comparisons (grouping restored on collect, recursion into lists
#     and list-columns, bare data.frame fixtures compared as bare, including
#     expect_s3_class(x, "data.frame", exact = TRUE), a harness relaxation);
#   - pinned CRAN-mode snapshot semantics and testthat edition 3;
#   - per-test instrumentation and the `not_exercised` status;
#   - the exclusion manifest tools/conformance-exclusions.tsv and `excluded`.
#
# Honest limitations (the ledger is only as meaningful as these):
#   - Anything routed to dplyr's CPU fallback in cuplyr (do()/rows_*()/colwise/
#     rowwise/...) round-trips through real tibbles internally, so a green
#     result there is a weaker signal than one on a GPU-native verb.
#   - `error = TRUE` snapshots are credited as a success only when, statement
#     by statement, cuplyr errors exactly where dplyr's reference in _snaps
#     errors and no credited error is an infrastructure/missing-feature error
#     that the reference does not also show for that statement; the wording is
#     never compared (the report lists every relaxed check with dplyr's
#     reference text, and marks AUDIT where the final "!" lines differ). A
#     snapshot without a usable reference block credits nothing and records a
#     skip expectation without ending the test. `error = FALSE` snapshots run the code
#     but compare nothing, so a test whose only assertion is one is a skip.
#   - Only bare `tibble()`/`data.frame()`/`tribble()` calls are shimmed;
#     `tibble::tibble()`-qualified calls, `as_tibble()`/`as.data.frame()` and
#     matrix/list inputs run on real tibbles. A passing test that built a table
#     with a shimmed constructor but never created a tbl_gpu is `not_exercised`;
#     a passing test with no constructor call at all (vector-only) stays `pass`
#     and is counted separately as exercised = FALSE.
#   - A HARNESS-ONLY `$`/`[[` method for tbl_gpu (defined in the shim env, not
#     in the package) collects on every column access; see the findings
#     appendix of the report for why that is not a package API yet.
#
# Usage:
#   Rscript tools/run-dplyr-conformance.R                # run all runnable files
#   Rscript tools/run-dplyr-conformance.R test-distinct.R # run just one file
#   Rscript tools/run-dplyr-conformance.R --all           # also attempt out-of-scope files
#   ... --ids ids.tsv        # also write one row per test (file, test, status first)
#   ... --report path.md     # report location (default scratchpad/dplyr_conformance_report.md)
#   ... --compare old.tsv    # add a per-file before/after table and the list of
#                            # lost passes (id, new status, message) against an
#                            # earlier ids TSV, e.g. the ratcheted baseline
#   ... --exclusions path    # manifest (default tools/conformance-exclusions.tsv;
#                            # "none" disables exclusions)
#
# Run it through tools/conformance (GPU lock) rather than directly. The
# package is pkgload::load_all()-ed from this checkout, never library()-ed, so
# each worktree measures its own code.
#
# Output: the markdown report plus a compact summary table printed to stdout.

suppressPackageStartupMessages({
  library(testthat)
  library(dplyr)
  library(tibble)
  # dplyr's own test files call plenty of unqualified helper functions that
  # are only visible because `test_check("dplyr")`'s real execution
  # environment is `env_clone(asNamespace("dplyr"))` (see cf_make_shim_env()
  # in the lib) -- that clone's parent chain is dplyr's own `import(rlang)`/
  # `import(vctrs, except = data_frame)`. Two more, `local_options()` (withr)
  # and `map_chr()` (purrr), are Suggests-only in dplyr and resolve in a real
  # run through the search path, so they are attached here to reproduce that.
  library(withr)
  library(purrr)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

find_pkg_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1) {
    script_path <- normalizePath(sub("^--file=", "", file_arg))
    return(normalizePath(file.path(dirname(script_path), "..")))
  }
  normalizePath(".")
}
pkg_root <- find_pkg_root()
source(file.path(pkg_root, "tools", "ensure-optimized.R"))
cuplyr_ensure_optimized_dll(pkg_root)
pkgload::load_all(pkg_root, quiet = TRUE, debug = FALSE)
source(file.path(pkg_root, "tools", "conformance-lib.R"))
tests_dir <- file.path(pkg_root, "inst", "dplyr-conformance", "tests")
source(file.path(pkg_root, "inst", "dplyr-conformance", "manifest.R"))

if (!has_gpu()) {
  stop(
    "No GPU detected (has_gpu() is FALSE). The conformance harness needs a ",
    "real GPU to construct tbl_gpu objects, same as pixi run test's ",
    "skip_if_no_gpu() gate."
  )
}

# ---- CLI args ---------------------------------------------------------------

cli_args <- commandArgs(trailingOnly = TRUE)
take_flag_value <- function(flag) {
  i <- match(flag, cli_args)
  if (is.na(i)) return(NULL)
  val <- cli_args[[i + 1L]]
  cli_args <<- cli_args[-c(i, i + 1L)]
  val
}
ids_path <- take_flag_value("--ids")
compare_path <- take_flag_value("--compare")
exclusions_path <- take_flag_value("--exclusions") %||%
  file.path(pkg_root, "tools", "conformance-exclusions.tsv")
report_path <- take_flag_value("--report") %||%
  file.path(pkg_root, "scratchpad", "dplyr_conformance_report.md")
run_all_incl_out_of_scope <- "--all" %in% cli_args
explicit_files <- setdiff(cli_args, "--all")

target_files <- if (length(explicit_files) > 0) {
  explicit_files
} else if (run_all_incl_out_of_scope) {
  all_test_files(tests_dir)
} else {
  runnable_files(tests_dir)
}

exclusions <- if (identical(exclusions_path, "none")) NULL else conformance_load_exclusions(exclusions_path)

# ---- Run ---------------------------------------------------------------------

message("Running dplyr conformance harness on ", length(target_files), " file(s)...")
all_results <- vector("list", length(target_files))
names(all_results) <- target_files
for (i in seq_along(target_files)) {
  fname <- target_files[[i]]
  message(sprintf("[%d/%d] %s", i, length(target_files), fname))
  all_results[[fname]] <- tryCatch(
    conformance_run_file(file.path(tests_dir, fname), exclusions),
    error = function(e) {
      list(
        file = fname, n_tests = 0L, passed = 0L, failed = 0L, errored = 0L, skipped = 0L,
        not_exercised = 0L, excluded = 0L, file_error = conditionMessage(e),
        reasons = character(0), ids = cf_empty_ids(),
        unmatched_exclusions = NULL, relaxed_snapshots = NULL
      )
    }
  )
}

# ---- Aggregate + report ------------------------------------------------------

summary_df <- do.call(rbind, lapply(all_results, function(r) {
  data.frame(
    file = r$file, n_tests = r$n_tests, passed = r$passed, failed = r$failed,
    errored = r$errored, skipped = r$skipped, not_exercised = r$not_exercised,
    excluded = r$excluded, file_error = !is.na(r$file_error),
    stringsAsFactors = FALSE
  )
}))
rownames(summary_df) <- NULL

message(sprintf(
  paste0("\nTOTAL: %d tests across %d files -- %d passed, %d failed, %d errored, %d skipped, ",
    "%d not_exercised, %d excluded. %d file(s) errored before any test ran."),
  sum(summary_df$n_tests), nrow(summary_df), sum(summary_df$passed), sum(summary_df$failed),
  sum(summary_df$errored), sum(summary_df$skipped), sum(summary_df$not_exercised),
  sum(summary_df$excluded), sum(summary_df$file_error)
))
print(summary_df)

id_rows <- do.call(rbind, lapply(all_results, function(r) r$ids))
if (is.null(id_rows)) id_rows <- cf_empty_ids()

cmp <- NULL
if (!is.null(compare_path)) {
  cmp <- conformance_compare(conformance_read_ids(compare_path), id_rows)
  message(sprintf("Compared against %s: %d lost pass(es), %d new pass(es).",
    compare_path, nrow(cmp$lost_passes), nrow(cmp$new_passes)))
}

vendored_version <- {
  pv <- file.path(pkg_root, "inst", "dplyr-conformance", "PROVENANCE.txt")
  if (file.exists(pv)) {
    vline <- grep("^dplyr version:", readLines(pv), value = TRUE)
    if (length(vline) > 0) sub("^dplyr version: ", "", vline[1]) else "unknown"
  } else "unknown"
}
lines <- conformance_report(
  all_results, exclusions = exclusions, cmp = cmp,
  header = c(
    "# dplyr conformance report (Phase 13)",
    "",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    sprintf("Vendored dplyr version: see `inst/dplyr-conformance/PROVENANCE.txt` (%s)", vendored_version),
    sprintf("Files run this pass: %d (%s)", length(target_files),
      if (run_all_incl_out_of_scope) "all vendored, including out-of-scope" else "runnable/CORE set only"),
    sprintf("Exclusion manifest: %s", if (is.null(exclusions)) "none" else
      sprintf("%s (%d entries)", basename(exclusions_path), nrow(exclusions)))
  )
)

out_of_scope_tbl <- out_of_scope_files(tests_dir)
lines <- c(lines, "", "## Out-of-scope files (not run by default)", "",
  "| File | Reason |", "|---|---|")
for (fname in out_of_scope_tbl) {
  lines <- c(lines, sprintf("| %s | %s |", fname, gsub("\\|", "\\\\|", out_of_scope_reason(fname))))
}

# ---- Manually-triaged notable findings ---------------------------------
# The categories in the report are produced by grepping first-failure
# messages, which is enough to sort "harness plumbing" from "actual cuplyr
# behavior" in bulk, but a handful of specific, high-value findings only
# surfaced from reading individual failures by hand. Recorded here as a
# fixed, dated appendix -- NOT regenerated automatically (the classifier can't
# produce prose like this), but the counts above are. Update this block by
# hand if a future run's investigation turns up something new.
lines <- c(lines, "", "## Notable findings (manually triaged, 2026-07-31; predates F01)", "", "
### 1. Genuine bug: integer-backed `Date` columns silently lose Date-ness on upload

`R/utils.R::gpu_type_from_r()` checks `is.integer(x)` (line 6) BEFORE it ever
checks `is.double(x)` -> `inherits(x, \"Date\")` (line 7-8). R's `Date` class
can be backed by EITHER storage mode (`as.Date(\"2013-01-01\") + 0:1` is
double-backed; `seq.Date(as.Date(\"2013-01-01\"), by = \"1 days\", length.out = 2)`
is integer-backed -- confirmed empirically, `typeof(seq.Date(...))` is
`\"integer\"`). For an integer-backed Date column, `is.integer(x)` is TRUE and
returns `\"INT32\"` immediately, so the `inherits(x, \"Date\")` branch is
**never reached** -- the column silently uploads as a plain integer, with no
warning, and every downstream comparison/format/collect() sees raw day
counts instead of dates. Found via `test-filter.R`'s vendored
`\"date class remains on filter (#273)\"` test (which happens to construct
its fixture with `seq.Date()`); reproduced directly against `tbl_gpu()`
outside the harness (no shim involved):
```r
x1 <- data.frame(date = seq.Date(as.Date(\"2013-01-01\"), by = \"1 days\", length.out = 2))
tbl_gpu(x1)$schema$types  # \"INT32\" -- should be \"TIMESTAMP_DAYS\"
```
Likely fix: check `inherits(x, c(\"Date\", \"POSIXct\"))` before the
`is.integer()`/`is.double()` storage-mode branches, not after. Test-first
per CLAUDE.md's STRICT bugfix mandate before patching.

### 2. Genuine gap: `tbl_gpu` has no vctrs `vec_size()`/`vec_slice()`/`vec_ptype2()` contract

The single largest concrete category (51 first-failures this run). Any
dplyr/vctrs-based helper that operates on a WHOLE table argument rather than
a column -- `coalesce()`, `if_else()`, `between()`, `consecutive_id()`,
`n_distinct()`, `lead()`/`lag()`, `rank()`, `nth_value()`, `with_order()`,
`vec_assign()`-backed replacement forms -- rejects a `tbl_gpu` outright with
`\"..1 must be a vector, not a <tbl_gpu> object\"`, because none of these
call a dplyr S3 verb method at all; they go straight to `vctrs::vec_*()`,
which requires a registered `vec_ptype2.tbl_gpu()`/`vec_size.tbl_gpu()` (or
for `tbl_gpu` to inherit a recognized vctrs prototype) to accept it as an
argument. This is orthogonal to `[.tbl_gpu`/`dplyr_row_slice.tbl_gpu` (the
dplyr-specific extension points already implemented, R/generics.R) --
registering the vctrs contract is a separate, real, and fairly contained
unit of work that would unlock all of the functions above at once.

### 3. Documented design choice, not a bug: no 2-d `[` (row+col) indexing

28 first-failures. `[.tbl_gpu` (R/generics.R) deliberately implements only
the 1-d, column-selecting form (`df[cols]`), matching a plain list's `[`
semantics -- dplyr's own tests routinely do `result[1:2, ]` to check a row
subset directly (bypassing any verb). This is already a conscious,
documented scope cut (R/generics.R's own module header), not new
information -- included here for completeness since it's the second-largest
bucket, not because it's actionable in the same way as (1)/(2).

### 4. Shim limitation, not a cuplyr fact: `tbl_gpu` has no `$`/`[[` method in the real package

The harness itself defines HARNESS-ONLY `$.tbl_gpu`/`[[.tbl_gpu` methods
(see tools/conformance-lib.R's shim environment) because dplyr's tests
overwhelmingly extract columns via `result$col`/`result[[\"col\"]]` directly
in plain test-body code. Worth a real design discussion (not resolved here):
would adding a real, exported `$.tbl_gpu`/`[[.tbl_gpu` (auto-collecting)
fight the rest of the package's explicit `collect()` philosophy, or is
`$`/`[[` common/read-only enough (unlike a full auto-materializing accessor
pattern) to be worth the ergonomics win? Flagged, not decided.
")

writeLines(lines, report_path)
message("\nWrote report to ", report_path)

if (!is.null(ids_path)) {
  conformance_write_ids(id_rows, ids_path)
  message("Wrote per-test IDs to ", ids_path)
}
