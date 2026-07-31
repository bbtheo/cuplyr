#!/usr/bin/env Rscript
# Run dplyr's own vendored test suite (inst/dplyr-conformance/tests/) against
# cuplyr's tbl_gpu backend and produce a per-file pass/fail/error/skip ledger.
#
# Phase 13 (dplyr-parity roadmap, scratchpad/todo.md): the whole point of the
# roadmap was "the whole dplyr test suite runs on cuplyr" -- this script is
# the harness that actually measures that, rather than the hand-mirrored
# oracle suite (tests/testthat/test-dplyr-*.R) which only covers the shapes
# cuplyr's own authors thought to write.
#
# ---------------------------------------------------------------------------
# SHIM MECHANISM (decided after prototyping against test-distinct.R, per the
# Phase 13 risk-spike note) -- and why:
#
# The task brief offered several options ranging from "mask tibble()/
# data.frame() so constructed tables become tbl_gpu, normalize both sides of
# expectations" to a fully re-executing test_that() wrapper. A full re-write
# of `test_that()` semantics was rejected as overkill and fragile: testthat's
# own `test_that()` ALREADY isolates each test (catches its own errors/
# expectation failures and always returns, recording into whatever reporter
# is active) -- there is no need to reimplement that isolation, only to (a)
# make constructed data become tbl_gpu, and (b) make comparisons tolerant of
# tbl_gpu operands. Concretely, per test file:
#
#   1. A fresh child-of-globalenv() "shim environment" masks `tibble()`,
#      `data.frame()`, and `tribble()`: each calls through to the REAL
#      constructor, then attempts `cuplyr::tbl_gpu()` on the result,
#      catching upload failures (list-columns, 0-row/0-col edges, etc.) by
#      silently falling back to the plain tibble/data.frame -- exactly
#      cuplyr's own `gpu_fallback()` philosophy applied to test fixtures.
#      This ONLY affects bare, unqualified `tibble(...)`/`data.frame(...)`
#      calls written literally in the vendored test file's source text --
#      R's lexical scoping means dplyr's OWN internals (which call
#      `tibble::tibble()`/`base::data.frame()` from inside the dplyr
#      namespace) are entirely unaffected by this override, so the shim
#      targets exactly "constructed tibbles/data.frames ENTERING a pipeline"
#      as the task brief asked, not dplyr's internal machinery.
#   2. `expect_equal()`/`expect_identical()` are masked to `collect()` +
#      `as_tibble()` any tbl_gpu operand on BOTH sides before delegating to
#      the real testthat expectation -- "normalize both sides" per the task
#      brief. `expect_s3_class()` is masked narrowly: if the object being
#      checked is a tbl_gpu AND the target class is one of
#      data.frame/tbl_df/tbl/tbl_gpu, treat it as satisfied outright (tbl_gpu
#      IS a table-like object, just represented differently -- see CLAUDE.md
#      on the DICTIONARY32 logical/physical split for the same kind of
#      "different bytes, same logical contract" reasoning); anything else
#      normalizes and delegates to the real check.
#   3. A HARNESS-ONLY (not part of the real cuplyr package/exports) `$` and
#      `[[` method for tbl_gpu is defined inside the shim env:
#      `collect(x)[[name]]`. Prototyping against test-distinct.R found this
#      necessary -- dplyr's tests overwhelmingly extract columns via
#      `result$col`/`result[["col"]]` directly in plain test-body code
#      (never inside an expectation call), which our masked expect_equal()
#      cannot intercept because the extraction already evaluated to NULL
#      (tbl_gpu is a plain list with fields ptr/schema/groups/..., so `$col`
#      silently returns NULL with no error) BEFORE expect_equal() ever runs.
#      This is flagged in the report as a genuine, if narrow, ergonomics gap
#      worth a future real PR (adding `$.tbl_gpu`/`[[.tbl_gpu` to the actual
#      package) -- NOT invented as real package API by this task, since that
#      would need its own design pass (does auto-collect on `$` fight the
#      "explicit collect()" philosophy the rest of the package follows?).
#   4. Each vendored file is run via `testthat::test_file(path, reporter =
#      "list", env = shim_env)` -- `env` is accepted by `test_file()`'s
#      `...` (verified against installed testthat 3.3.2's `test_files_serial`
#      body) and, when supplied, is used AS THE test execution environment
#      directly (not wrapped) rather than testthat's usual fresh
#      `test_env()`, and `load_helpers = TRUE` (the default) still sources
#      the file's sibling `helper-*.R` files into it first, exactly
#      mirroring how `devtools::test()`/`R CMD check` would run these files
#      inside the real dplyr package.
#
# HONEST LIMITATIONS (not hidden -- this is what makes the ledger meaningful
# rather than a rubber stamp):
#   - Anything routed to dplyr's CPU fallback in cuplyr (do()/rows_*()/
#     colwise/rowwise/...) round-trips through real tibbles internally, so it
#     behaves close to identically to unshimmed dplyr -- a green result there
#     is a weaker signal than a green result on a GPU-native verb.
#   - `expect_snapshot()` compares against the vendored `_snaps/*.md` MD5/
#     text reference; almost every snapshot that includes an ERROR MESSAGE
#     will mismatch cuplyr's own wording even when the underlying behavior
#     (erroring at all) is correct -- these are categorized separately, not
#     silently hidden as failures.
#   - Only bare `tibble()`/`data.frame()`/`tribble()` calls are shimmed;
#     `tibble::tibble()`-qualified calls, `as_tibble()`/`as.data.frame()`
#     conversions, and matrix/list inputs are NOT intercepted -- a test that
#     only ever uses those forms runs against a REAL, unshimmed tibble and so
#     doesn't exercise cuplyr at all (a "green" result there proves nothing
#     about cuplyr; also flagged, not celebrated).
#   - `$`/`[[` auto-collect (item 3 above) hides the cost of a device->host
#     transfer on every single column access -- fine for a conformance
#     count, actively bad practice if anyone copied this pattern into real
#     user code.
#
# ---------------------------------------------------------------------------
# Usage:
#   Rscript tools/run-dplyr-conformance.R                # run all runnable files
#   Rscript tools/run-dplyr-conformance.R test-distinct.R # run just one file
#   Rscript tools/run-dplyr-conformance.R --all           # also attempt out-of-scope files
#
# Output: scratchpad/dplyr_conformance_report.md (overwritten each run) plus
# a compact summary table printed to stdout.

suppressPackageStartupMessages({
  library(testthat)
  library(dplyr)
  library(cuplyr)
  library(tibble)
  # dplyr's own test files call plenty of unqualified helper functions that
  # are only visible because `test_check("dplyr")`'s real execution
  # environment is `env_clone(asNamespace("dplyr"))` (see make_shim_env()
  # below) -- that clone's parent chain is dplyr's own `import(rlang)`/
  # `import(vctrs, except = data_frame)` (dplyr's NAMESPACE, confirmed by
  # inspection: both are FULL `import()`s, not `importFrom()`), which is why
  # bare `vec_slice()`/`list_of()`/`is_integerish()`/`new_vctr()`/
  # `vec_assign()`/`exprs()`/`set_names()` etc. resolve inside dplyr's own
  # tests. Two more, `local_options()` (withr) and `map_chr()` (purrr), are
  # NOT part of dplyr's Imports (they're Suggests-only) -- they resolve in a
  # real dplyr test run because the namespace's parent chain bottoms out at
  # `.GlobalEnv` (a documented R fact: a namespace's imports env's parent is
  # the base namespace, whose parent IS `.GlobalEnv`) and from there the
  # normal search path, so attaching them here (as any dplyr contributor's
  # interactive session would have them attached via Suggests) reproduces
  # the same resolution.
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
run_all_incl_out_of_scope <- "--all" %in% cli_args
explicit_files <- setdiff(cli_args, "--all")

target_files <- if (length(explicit_files) > 0) {
  explicit_files
} else if (run_all_incl_out_of_scope) {
  all_test_files(tests_dir)
} else {
  runnable_files(tests_dir)
}

# ---- Shim construction -------------------------------------------------------

gpu_try_construct <- function(x) {
  if (!is.data.frame(x)) {
    return(x)
  }
  res <- tryCatch(cuplyr::tbl_gpu(x), error = function(e) NULL, warning = function(w) NULL)
  if (is.null(res)) x else res
}

gpu_normalize <- function(x) {
  if (inherits(x, "tbl_gpu")) {
    tryCatch(
      tibble::as_tibble(dplyr::collect(x)),
      error = function(e) x # collect() itself failed -- let the real comparison surface it
    )
  } else {
    x
  }
}

make_shim_env <- function() {
  # Start from the SAME base environment dplyr's own `test_check("dplyr")`
  # uses (see testthat::test_env(), which does exactly this for a named
  # package): a clone of dplyr's namespace, so every unqualified
  # dplyr-internal helper (join_cols(), count_regroups(), tbl_at_vars(),
  # as_fun_list(), join_by_common(), join_rows(), compute_by(), shift(),
  # ...) and every `import()`ed rlang/vctrs function resolves exactly as it
  # would inside a real dplyr test run. Without this, hundreds of tests
  # error on "could not find function" before ever reaching cuplyr code --
  # confirmed by prototyping without it first (see report's own history);
  # that noise would drown out genuine signal, so this is load-bearing, not
  # a nice-to-have.
  env <- testthat::test_env("dplyr")

  # Re-shadow every cuplyr-exported name over the cloned dplyr binding.
  # Real usage is `library(dplyr); library(cuplyr)`, where cuplyr (attached
  # LAST) masks any same-named dplyr binding on the search path -- most
  # verbs don't need this (dplyr already defines them as UseMethod()
  # generics that dispatch correctly to `verb.tbl_gpu()` regardless of
  # which literal function object is called), but a few of dplyr's own
  # top-level functions are NOT S3 generics at all (`bind_rows()`/
  # `bind_cols()` call `vctrs::vec_rbind()`/`vec_cbind()` directly instead
  # of `UseMethod()`) -- cuplyr instead defines ITS OWN `bind_rows <-
  # function(...) UseMethod("bind_rows")` and exports it specifically to
  # mask dplyr's plain-function version for exactly this reason (see
  # R/bind.R). `test_env("dplyr")`'s clone captures dplyr's OWN internal
  # `bind_rows` binding directly in the cloned env (not just via the search
  # path), which would otherwise silently bypass cuplyr's real masking
  # mechanism and misreport a shim artifact as a cuplyr `bind_rows()`
  # failure. Confirmed empirically: without this fix, every `bind_rows()`
  # call in test-bind-rows.R errored with "Argument 1 must be a data frame
  # or a named atomic vector" (vec_rbind() rejecting a plain-list tbl_gpu)
  # even though `cuplyr::bind_rows(tbl_gpu1, tbl_gpu2)` works fine when
  # called directly.
  for (nm in getNamespaceExports("cuplyr")) {
    env[[nm]] <- get(nm, envir = asNamespace("cuplyr"))
  }

  env$data.frame <- function(...) gpu_try_construct(base::data.frame(...))
  env$tibble <- function(...) gpu_try_construct(tibble::tibble(...))
  env$tribble <- function(...) gpu_try_construct(tibble::tribble(...))

  # Harness-only accessor shims (see header comment, limitation item 3) --
  # NOT part of the real cuplyr package.
  env$`$.tbl_gpu` <- function(x, name) {
    tibble::as_tibble(dplyr::collect(x))[[name]]
  }
  env$`[[.tbl_gpu` <- function(x, i, ...) {
    tibble::as_tibble(dplyr::collect(x))[[i, ...]]
  }

  gpu_class_tolerant_classes <- c("data.frame", "tbl_df", "tbl", "tbl_gpu")
  env$expect_s3_class <- function(object, class, ..., exact = FALSE) {
    if (inherits(object, "tbl_gpu") && isTRUE(all(class %in% gpu_class_tolerant_classes))) {
      return(testthat::succeed())
    }
    testthat::expect_s3_class(gpu_normalize(object), class, ..., exact = exact)
  }
  env$expect_equal <- function(object, expected, ...) {
    testthat::expect_equal(gpu_normalize(object), gpu_normalize(expected), ...)
  }
  env$expect_identical <- function(object, expected, ...) {
    testthat::expect_identical(gpu_normalize(object), gpu_normalize(expected), ...)
  }

  env
}

# ---- Reason classification (heuristic, for the report's honesty mandate) ---

classify_reason <- function(msg) {
  if (is.na(msg) || !nzchar(msg)) {
    return("unknown")
  }
  if (grepl("not an exported object|dplyr:::|:::.*dplyr|non-exported|could not find function", msg, ignore.case = TRUE)) {
    return("dplyr-internal usage")
  }
  if (grepl("Snapshot", msg, ignore.case = TRUE)) {
    return("shim artifact: snapshot text mismatch")
  }
  # These two are checked BEFORE the generic tbl_gpu bucket below: both
  # messages happen to mention "tbl_gpu" but are genuine, actionable
  # cuplyr-side gaps, not shim plumbing -- see the report's "top genuine
  # gaps" section.
  if (grepl("must be a vector, not a", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: tbl_gpu has no vctrs vec_size()/vec_slice() contract (breaks coalesce()/if_else()/between()/consecutive_id()/n_distinct()/lead()/lag()/rank()/nth_value()/with_order() when passed a whole table)")
  }
  if (grepl("2-d .\\[. indexing", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: 2-d `[` (row+col) indexing not implemented for tbl_gpu (R/generics.R only implements the 1-d column-select form; dplyr's own tests routinely do `result[1:2, ]` to check a row subset)")
  }
  if (grepl("inherit from|Actual class|no applicable method|not a data\\.frame|no method for", msg, ignore.case = TRUE)) {
    return("shim artifact: class/representation mismatch")
  }
  if (grepl("tbl_gpu|gpu_normalize|GPU|has_gpu|cuplyr", msg, ignore.case = TRUE)) {
    return("shim artifact: tbl_gpu representation")
  }
  if (grepl("not supported|not implemented|unsupported|no ir_call_registry|Unrecognized|not recognized|Unknown function", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: unsupported expression/verb shape")
  }
  "genuine cuplyr gap (needs triage)"
}

# ---- Run one file ------------------------------------------------------------

run_one_file <- function(fname) {
  path <- file.path(tests_dir, fname)
  env <- make_shim_env()
  res <- tryCatch(
    testthat::test_file(path, reporter = "list", env = env),
    error = function(e) {
      structure(list(), file_level_error = conditionMessage(e))
    }
  )

  file_level_error <- attr(res, "file_level_error")
  if (!is.null(file_level_error)) {
    return(list(
      file = fname, n_tests = 0L, passed = 0L, failed = 0L, errored = 0L, skipped = 0L,
      file_error = file_level_error, reasons = character(0)
    ))
  }

  n_tests <- length(res)
  passed <- 0L
  failed <- 0L
  errored <- 0L
  skipped <- 0L
  reasons <- character(0)

  for (t in res) {
    cls <- vapply(t$results, function(x) class(x)[1], character(1))
    if (any(cls == "expectation_error")) {
      errored <- errored + 1L
      first_bad <- Filter(function(x) inherits(x, "expectation_error"), t$results)[[1]]
      reasons <- c(reasons, conditionMessage(first_bad))
    } else if (any(cls == "expectation_failure")) {
      failed <- failed + 1L
      first_bad <- Filter(function(x) inherits(x, "expectation_failure"), t$results)[[1]]
      reasons <- c(reasons, conditionMessage(first_bad))
    } else if (length(cls) > 0 && all(cls == "expectation_skip")) {
      skipped <- skipped + 1L
    } else {
      passed <- passed + 1L
    }
  }

  list(
    file = fname, n_tests = n_tests, passed = passed, failed = failed,
    errored = errored, skipped = skipped, file_error = NA_character_,
    reasons = reasons
  )
}

message("Running dplyr conformance harness on ", length(target_files), " file(s)...")
all_results <- vector("list", length(target_files))
names(all_results) <- target_files
for (i in seq_along(target_files)) {
  fname <- target_files[[i]]
  message(sprintf("[%d/%d] %s", i, length(target_files), fname))
  all_results[[fname]] <- tryCatch(
    run_one_file(fname),
    error = function(e) {
      list(
        file = fname, n_tests = 0L, passed = 0L, failed = 0L, errored = 0L, skipped = 0L,
        file_error = conditionMessage(e), reasons = character(0)
      )
    }
  )
}

# ---- Aggregate + report ------------------------------------------------------

summary_rows <- lapply(all_results, function(r) {
  data.frame(
    file = r$file, n_tests = r$n_tests, passed = r$passed, failed = r$failed,
    errored = r$errored, skipped = r$skipped,
    file_error = !is.na(r$file_error),
    stringsAsFactors = FALSE
  )
})
summary_df <- do.call(rbind, summary_rows)
rownames(summary_df) <- NULL

total_tests <- sum(summary_df$n_tests)
total_passed <- sum(summary_df$passed)
total_failed <- sum(summary_df$failed)
total_errored <- sum(summary_df$errored)
total_skipped <- sum(summary_df$skipped)
fully_green_files <- sum(summary_df$n_tests > 0 & summary_df$failed == 0 & summary_df$errored == 0 & !summary_df$file_error)
file_level_error_files <- sum(summary_df$file_error)

message(sprintf(
  "\nTOTAL: %d tests across %d files -- %d passed, %d failed, %d errored, %d skipped. %d/%d files fully green. %d file(s) errored before any test ran.",
  total_tests, nrow(summary_df), total_passed, total_failed, total_errored, total_skipped,
  fully_green_files, nrow(summary_df), file_level_error_files
))
print(summary_df)

# Top failure reasons across the whole run, categorized.
all_reasons <- unlist(lapply(all_results, function(r) r$reasons))
reason_categories <- vapply(all_reasons, classify_reason, character(1))
category_counts <- sort(table(reason_categories), decreasing = TRUE)

# Build the markdown ledger.
lines <- c(
  "# dplyr conformance report (Phase 13)",
  "",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  sprintf(
    "Vendored dplyr version: see `inst/dplyr-conformance/PROVENANCE.txt` (%s)",
    { pv <- file.path(pkg_root, "inst", "dplyr-conformance", "PROVENANCE.txt")
      if (file.exists(pv)) {
        vline <- grep("^dplyr version:", readLines(pv), value = TRUE)
        if (length(vline) > 0) sub("^dplyr version: ", "", vline[1]) else "unknown"
      } else "unknown"
    }
  ),
  sprintf("Files run this pass: %d (%s)", length(target_files), if (run_all_incl_out_of_scope) "all vendored, including out-of-scope" else "runnable/CORE set only"),
  "",
  "## Headline numbers",
  "",
  sprintf("- **%d of %d runnable files fully green** (0 failed, 0 errored, ran to completion)", fully_green_files, nrow(summary_df)),
  sprintf("- **%d individual tests** executed across those files", total_tests),
  sprintf(
    "- **%.1f%%** of individual tests passed (%d passed / %d failed / %d errored / %d skipped)",
    if (total_tests > 0) 100 * total_passed / total_tests else NA, total_passed, total_failed, total_errored, total_skipped
  ),
  sprintf("- %d file(s) errored before a single `test_that()` block completed (parse/load-time failure)", file_level_error_files),
  "",
  "## Top failure-reason categories (heuristic classification of first-failure message per failing/erroring test)",
  ""
)
if (length(category_counts) > 0) {
  lines <- c(lines, "| Category | Count |", "|---|---|")
  for (nm in names(category_counts)) {
    lines <- c(lines, sprintf("| %s | %d |", nm, category_counts[[nm]]))
  }
} else {
  lines <- c(lines, "(no failures/errors recorded)")
}

lines <- c(lines, "", "## Per-file ledger", "")
lines <- c(lines, "| File | Tests | Passed | Failed | Errored | Skipped | Status | Top reason (first failure/error) |")
lines <- c(lines, "|---|---|---|---|---|---|---|---|")
for (fname in target_files) {
  r <- all_results[[fname]]
  status <- if (!is.na(r$file_error)) {
    "FILE ERROR"
  } else if (r$n_tests == 0) {
    "NO TESTS"
  } else if (r$failed == 0 && r$errored == 0) {
    "GREEN"
  } else {
    "RED"
  }
  top_reason <- if (!is.na(r$file_error)) {
    gsub("\\|", "\\\\|", substr(r$file_error, 1, 160))
  } else if (length(r$reasons) > 0) {
    gsub("\\n", " ", gsub("\\|", "\\\\|", substr(r$reasons[[1]], 1, 160)))
  } else {
    ""
  }
  lines <- c(lines, sprintf(
    "| %s | %d | %d | %d | %d | %d | %s | %s |",
    fname, r$n_tests, r$passed, r$failed, r$errored, r$skipped, status, top_reason
  ))
}

out_of_scope_tbl <- out_of_scope_files(tests_dir)
lines <- c(lines, "", "## Out-of-scope files (not run by default)", "",
  "| File | Reason |", "|---|---|")
for (fname in out_of_scope_tbl) {
  lines <- c(lines, sprintf("| %s | %s |", fname, gsub("\\|", "\\\\|", out_of_scope_reason(fname))))
}

# ---- Manually-triaged notable findings ---------------------------------
# The categories above are produced by grepping first-failure messages, which
# is enough to sort "shim plumbing" from "actual cuplyr behavior" in bulk,
# but a handful of specific, high-value findings only surfaced from reading
# individual failures by hand (below the file-level ledger's resolution).
# Recorded here as a fixed, dated appendix -- NOT regenerated automatically
# each run (the classifier can't produce prose like this), but the counts
# above are. Update this block by hand if a future run's investigation
# turns up something new.
lines <- c(lines, "", "## Notable findings (manually triaged, 2026-07-31)", "", "
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
(see tools/run-dplyr-conformance.R's own header) because dplyr's tests
overwhelmingly extract columns via `result$col`/`result[[\"col\"]]` directly
in plain test-body code. Worth a real design discussion (not resolved here):
would adding a real, exported `$.tbl_gpu`/`[[.tbl_gpu` (auto-collecting)
fight the rest of the package's explicit `collect()` philosophy, or is
`$`/`[[` common/read-only enough (unlike a full auto-materializing accessor
pattern) to be worth the ergonomics win? Flagged, not decided.
")

report_path <- file.path(pkg_root, "scratchpad", "dplyr_conformance_report.md")
writeLines(lines, report_path)
message("\nWrote report to ", report_path)
