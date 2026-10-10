#!/usr/bin/env Rscript
# Conformance step of the merge gate (called by tools/merge-gate; not for
# agents). Pure R on two TSVs plus git: no GPU, no package load.
#
#   Rscript tools/conformance-gate.R --baseline <tsv> --new <tsv> --out <gate dir>
#                                    [--base-ref dev]
#
# Run from the integration worktree, detached at the candidate (dev + the
# squashed feature); <base-ref> is still the pre-merge dev. The rules are
# conformance_gate_check() in tools/conformance-lib.R. Writes into <gate dir>:
#   conformance_regressions.txt, conformance_new_passes.txt,
#   conformance_reclassified.txt, conformance_known_failures_accepted.txt
# (one "file :: test" per line) and prints the summary lines.
# Exit 0 = PASS, 3 = FAIL; anything else is a script error (also a FAIL).

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default = NULL) {
  i <- match(name, args)
  if (is.na(i)) {
    if (is.null(default)) stop("missing ", name, call. = FALSE)
    return(default)
  }
  if (i == length(args)) stop(name, " needs a value", call. = FALSE)
  args[[i + 1L]]
}
baseline_path <- opt("--baseline")
new_path <- opt("--new")
out <- opt("--out")
base_ref <- opt("--base-ref", "dev")

script_dir <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
lib <- new.env(parent = globalenv())
sys.source(file.path(script_dir, "conformance-lib.R"), envir = lib)

git <- function(...) suppressWarnings(system2("git", c(...), stdout = TRUE, stderr = TRUE))
# stdout only (stderr to the console), for content that is written out.
git_out <- function(...) suppressWarnings(system2("git", c(...), stdout = TRUE, stderr = ""))
git_status <- function(x) if (is.null(attr(x, "status"))) 0L else attr(x, "status")

feature <- lib$conformance_gate_feature_name(out)

if (git_status(git("rev-parse", "--verify", "-q", paste0(base_ref, "^{commit}"))) != 0L) {
  stop("base ref '", base_ref, "' is not a commit", call. = FALSE)
}

# harness_changed: does the candidate touch the harness relative to dev?
harness_paths <- c("tools/conformance-lib.R", "tools/run-dplyr-conformance.R",
  "tools/conformance-exclusions.tsv", "inst/dplyr-conformance/")
d <- git("diff", "--quiet", base_ref, "HEAD", "--", harness_paths)
harness_changed <- switch(as.character(git_status(d)), "0" = FALSE, "1" = TRUE,
  stop("git diff against ", base_ref, " failed: ", paste(d, collapse = "\n"), call. = FALSE))

# dev's copy of a file as a temp path; absent at dev = NULL (empty list).
base_copy <- function(path) {
  if (git_status(git("cat-file", "-e", paste0(base_ref, ":", path))) != 0L) return(NULL)
  tmp <- tempfile(fileext = ".tsv")
  txt <- git_out("show", paste0(base_ref, ":", path))
  if (git_status(txt) != 0L) stop("git show ", base_ref, ":", path, " failed", call. = FALSE)
  writeLines(txt, tmp)
  tmp
}
kf <- "tools/conformance-known-failures.tsv"
base_kf <- base_copy(kf)
ex <- "tools/conformance-exclusions.tsv"
base_ex <- base_copy(ex)

res <- lib$conformance_gate_check(
  baseline = lib$conformance_read_gate_ids(baseline_path),
  new = lib$conformance_read_gate_ids(new_path),
  known_candidate = lib$conformance_load_known_failures(kf),
  known_base = lib$conformance_load_known_failures(base_kf),
  feature = feature, harness_changed = harness_changed
)

ids <- function(d) if (nrow(d) > 0) paste(d$file, d$test, sep = " :: ") else character()
writeLines(ids(res$regressions), file.path(out, "conformance_regressions.txt"))
writeLines(ids(res$new_passes), file.path(out, "conformance_new_passes.txt"))
writeLines(ids(res$reclassified), file.path(out, "conformance_reclassified.txt"))
writeLines(ids(res$accepted), file.path(out, "conformance_known_failures_accepted.txt"))

added <- lib$conformance_gate_added(
  known_candidate = lib$conformance_load_known_failures(kf),
  known_base = lib$conformance_load_known_failures(base_kf),
  excl_candidate = if (file.exists(ex)) lib$conformance_load_exclusions(ex) else NULL,
  excl_base = if (is.null(base_ex)) NULL else lib$conformance_load_exclusions(base_ex))
writeLines(c(lib$conformance_gate_summary(res), added))
cat(sprintf("conformance: feature %s, harness %s vs %s\n", feature,
  if (harness_changed) "changed" else "unchanged", base_ref))
quit(status = if (res$ok) 0L else 3L)
