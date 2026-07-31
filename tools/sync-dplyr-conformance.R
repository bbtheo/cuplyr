#!/usr/bin/env Rscript
# Vendor dplyr's own testthat suite into inst/dplyr-conformance/tests/.
#
# Phase 13 (dplyr-parity roadmap, scratchpad/todo.md) needs dplyr's *actual*
# test-*.R files, not hand-copied excerpts, so the conformance harness
# (tools/run-dplyr-conformance.R) is running the real oracle. Installed CRAN
# binaries normally strip the tests/ directory entirely, but this machine
# happens to have a source/full install of dplyr that still ships it
# (see below) -- this script prefers that copy and only falls back to a
# network clone when no local copy is available, so a re-sync never
# silently requires network access if it doesn't have to.
#
# Usage:
#   Rscript tools/sync-dplyr-conformance.R
#
# What it does:
#   1. Locate a source of dplyr's tests/testthat/*.R files:
#      a. Any installed dplyr under .libPaths() (or common ~/R/*/... library
#         roots) that still ships tests/testthat/ (source installs, and some
#         non-CRAN-binary installs, do) -- preferred, no network needed.
#      b. Else, shallow git clone of tidyverse/dplyr at a pinned tag to a
#         temp dir and read tests/testthat/ from there.
#   2. Copy every test-*.R and helper-*.R file into
#      inst/dplyr-conformance/tests/ (overwriting any previous vendor drop).
#   3. Regenerate inst/dplyr-conformance/PROVENANCE.txt recording the exact
#      version/source/commit and date, so `git diff` shows what changed on
#      a re-sync.
#
# dplyr is MIT licensed (2026 dplyr authors) -- see
# inst/dplyr-conformance/README.md for the attribution note required by
# vendoring MIT-licensed source files verbatim.

pinned_tag <- "v1.2.1" # dplyr version cuplyr's DESCRIPTION currently targets

# Robustly locate the package root regardless of how the script is invoked
# (Rscript tools/sync-dplyr-conformance.R from repo root is the documented
# usage, but don't assume a working directory).
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
dest_dir <- file.path(pkg_root, "inst", "dplyr-conformance", "tests")
dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)

message("cuplyr conformance vendoring: pkg_root = ", pkg_root)

# ---- 1. Locate a source of dplyr's tests/testthat/ -------------------------

candidate_test_dirs <- function() {
  candidates <- character(0)
  for (lib in .libPaths()) {
    d <- file.path(lib, "dplyr", "tests", "testthat")
    if (dir.exists(d)) candidates <- c(candidates, d)
  }
  # Also probe common ~/R/*/... library roots directly, in case dplyr is
  # installed for a different R minor version than the one currently
  # running this script (exactly the situation on this machine: R 4.6 is
  # the active version but only the R 4.5 library still ships dplyr's
  # tests/ directory).
  home_r_glob <- Sys.glob(file.path(Sys.getenv("HOME"), "R", "*", "*"))
  for (lib in home_r_glob) {
    d <- file.path(lib, "dplyr", "tests", "testthat")
    if (dir.exists(d) && !(d %in% candidates)) candidates <- c(candidates, d)
  }
  unique(candidates)
}

dplyr_version_at <- function(lib_dplyr_dir) {
  desc <- file.path(lib_dplyr_dir, "DESCRIPTION")
  if (!file.exists(desc)) {
    return(NA_character_)
  }
  read.dcf(desc, fields = "Version")[1, 1]
}

installed_candidates <- candidate_test_dirs()
# Prefer a candidate whose tests/testthat actually has test-*.R files (an
# empty/placeholder tests dir shouldn't count).
installed_candidates <- Filter(function(d) {
  length(Sys.glob(file.path(d, "test-*.R"))) > 0
}, installed_candidates)

source_kind <- NULL
source_dir <- NULL
source_version <- NA_character_
source_note <- NULL

if (length(installed_candidates) > 0) {
  source_dir <- installed_candidates[[1]]
  lib_dplyr_dir <- dirname(dirname(source_dir))
  source_version <- dplyr_version_at(lib_dplyr_dir)
  source_kind <- "installed-library"
  source_note <- sprintf(
    "Copied from a local installed dplyr package that still ships tests/testthat/ (%s), dplyr %s.",
    source_dir, source_version
  )
  message(
    "Found installed dplyr tests at ", source_dir,
    " (dplyr ", source_version, ")"
  )
} else {
  message(
    "No installed dplyr copy ships tests/testthat/ -- falling back to a ",
    "shallow git clone of tidyverse/dplyr@", pinned_tag
  )
  tmp <- tempfile("dplyr-conformance-clone-")
  dir.create(tmp)
  clone_cmd <- sprintf(
    "git clone --depth 1 --branch %s https://github.com/tidyverse/dplyr.git %s",
    shQuote(pinned_tag), shQuote(tmp)
  )
  status <- system(clone_cmd)
  if (status != 0) {
    stop(
      "Could not find an installed dplyr copy with tests/testthat/, and ",
      "the fallback git clone of tidyverse/dplyr@", pinned_tag, " failed ",
      "(network access may be unavailable in this environment). ",
      "Install a source copy of dplyr (install.packages('dplyr', type = 'source')) ",
      "or provide network access and re-run this script."
    )
  }
  source_dir <- file.path(tmp, "tests", "testthat")
  source_version <- pinned_tag
  source_kind <- "git-clone"
  commit <- tryCatch(
    system2("git", c("-C", shQuote(tmp), "rev-parse", "HEAD"), stdout = TRUE),
    error = function(e) NA_character_
  )
  source_note <- sprintf(
    "Cloned from https://github.com/tidyverse/dplyr.git @ %s (commit %s).",
    pinned_tag, commit
  )
}

# ---- 2. Copy test-*.R and helper-*.R into inst/dplyr-conformance/tests/ ----

src_files <- sort(c(
  Sys.glob(file.path(source_dir, "test-*.R")),
  Sys.glob(file.path(source_dir, "helper-*.R"))
))
if (length(src_files) == 0) {
  stop("No test-*.R/helper-*.R files found at ", source_dir)
}

# Wipe previous vendor drop so deletions upstream are reflected on re-sync.
old_files <- Sys.glob(file.path(dest_dir, "*.R"))
if (length(old_files) > 0) file.remove(old_files)
old_snaps <- file.path(dest_dir, "_snaps")
if (dir.exists(old_snaps)) unlink(old_snaps, recursive = TRUE)

invisible(file.copy(src_files, dest_dir, overwrite = TRUE))
message("Vendored ", length(src_files), " files into ", dest_dir)

# Also vendor _snaps/ (expect_snapshot() reference output) and any fixture
# data files (e.g. utf-8.txt used by helper-encoding.R) so expect_snapshot()
# comparisons are meaningful (mismatches are a real, informative conformance
# signal) rather than silently auto-recording "new" snapshots every run.
snaps_src <- file.path(source_dir, "_snaps")
if (dir.exists(snaps_src)) {
  invisible(file.copy(snaps_src, dest_dir, recursive = TRUE, overwrite = TRUE))
  message("Vendored _snaps/ reference output into ", dest_dir)
}
fixture_files <- setdiff(
  list.files(source_dir, full.names = TRUE),
  c(src_files, snaps_src, file.path(source_dir, "testthat.R"))
)
fixture_files <- fixture_files[!file.info(fixture_files)$isdir %in% TRUE]
if (length(fixture_files) > 0) {
  invisible(file.copy(fixture_files, dest_dir, overwrite = TRUE))
  message("Vendored ", length(fixture_files), " fixture file(s): ", paste(basename(fixture_files), collapse = ", "))
}

# ---- 3. Provenance record ---------------------------------------------------

provenance <- c(
  "# Provenance for inst/dplyr-conformance/tests/",
  "# Generated by tools/sync-dplyr-conformance.R -- do not hand-edit.",
  "",
  paste0("Sync date: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  paste0("Source kind: ", source_kind),
  paste0("dplyr version: ", source_version),
  paste0("Source detail: ", source_note),
  paste0("File count: ", length(src_files)),
  "",
  "License: dplyr is released under the MIT license (2026 dplyr authors).",
  "These files are vendored verbatim (copyright/license notice preserved",
  "in inst/dplyr-conformance/README.md) solely to run as an opt-in,",
  "non-CRAN-shipped conformance harness against cuplyr's tbl_gpu backend",
  "(see tools/run-dplyr-conformance.R). They are not part of cuplyr's own",
  "test suite and are not exercised by `pixi run test`/R CMD check.",
  "",
  "Files:",
  paste0("  - ", basename(src_files))
)
writeLines(provenance, file.path(dirname(dest_dir), "PROVENANCE.txt"))
message("Wrote provenance record to ", file.path(dirname(dest_dir), "PROVENANCE.txt"))
