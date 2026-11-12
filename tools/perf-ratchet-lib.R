# The adaptive perf baseline: rules for lowering scratchpad/perf_baseline.json
# after a fully passing merge gate. Pure R + jsonlite: no package load, no GPU.
# Sourced by tools/perf-ratchet.R (the CLI behind tools/perf-ratchet and
# tools/perf-baseline-set) and by the F64 tests.
#
# An observation is the gate's calibrated per-benchmark median (observed_s in
# the file tests/testthat/helper-perf.R writes when CUPLYR_PERF_OBSERVATIONS is
# set), the value the 1.5x check compares against the baseline. Each entry
# keeps last_observed_s, its observation from the previous passing gate. A
# gate's confirmed value is max(last_observed_s, observed_s), a speed reached
# in two consecutive passing gates, so one lucky session cannot drag the
# baseline down. Then median_s <- min(median_s, confirmed). Entries without
# last_observed_s are only seeded. Ids missing from the baseline enter with
# their first observation. _calibration and _calibration_transfer define the
# reference session and are never touched here.

perf_ratchet_reserved <- c("_calibration", "_calibration_transfer")

`%||%` <- function(x, y) if (is.null(x)) y else x

perf_ratchet_now <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC")

perf_ratchet_valid_s <- function(x) {
  is.numeric(x) && length(x) == 1 && is.finite(x) && x > 0
}

# Read the baseline as a named list (id -> record). Unlike the harness's
# reader, a missing or corrupt file is an error: treating it as empty would
# make the next write wipe the baseline.
perf_ratchet_read_baseline <- function(path) {
  if (!file.exists(path)) stop("baseline not found: ", path, call. = FALSE)
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  if (!nzchar(trimws(txt))) stop("baseline is empty: ", path, call. = FALSE)
  parsed <- jsonlite::fromJSON(txt, simplifyVector = FALSE)
  if (!is.list(parsed)) stop("baseline is not a JSON object: ", path, call. = FALSE)
  parsed
}

# Temp file in the target's directory, then rename (same filesystem, so the
# swap is atomic). On any error the temp file is removed and the target is
# untouched. Doubles go out at full precision (digits = NA).
perf_ratchet_write_baseline <- function(path, data) {
  dir <- dirname(path)
  if (!dir.exists(dir)) stop("directory does not exist: ", dir, call. = FALSE)
  tmp <- tempfile(pattern = ".perf-ratchet-", tmpdir = dir)
  ok <- FALSE
  on.exit(if (!ok) unlink(tmp), add = TRUE)
  txt <- jsonlite::toJSON(data, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)
  # An unwritable directory warns and then errors; the error is what matters.
  suppressWarnings(writeLines(as.character(txt), tmp))
  if (!file.rename(tmp, path)) stop("could not rename ", tmp, " to ", path, call. = FALSE)
  ok <- TRUE
  invisible(path)
}

perf_ratchet_read_observations <- function(path) {
  if (!file.exists(path)) stop("observations file not found: ", path, call. = FALSE)
  obs <- utils::read.delim(path, sep = "\t", quote = "", comment.char = "", stringsAsFactors = FALSE,
    colClasses = c(id = "character", calibration = "character"))
  need <- c("id", "observed_s", "iterations")
  if (!all(need %in% names(obs))) {
    stop("observations file lacks columns: ", paste(setdiff(need, names(obs)), collapse = ", "), call. = FALSE)
  }
  obs$observed_s <- suppressWarnings(as.numeric(obs$observed_s))
  obs
}

# Apply one gate's observations. Returns list(baseline, changes); `changes` has
# one row per lowered or new entry (old_s is NA for new ids).
perf_ratchet_apply <- function(baseline, observations, commit, recorded_at = perf_ratchet_now()) {
  ids <- as.character(observations$id)
  dup <- unique(ids[duplicated(ids)])
  if (length(dup) > 0) stop("duplicate ids in the observations: ", paste(dup, collapse = ", "), call. = FALSE)

  changes <- data.frame(id = character(), old_s = numeric(), new_s = numeric(), kind = character(),
    stringsAsFactors = FALSE)
  add_change <- function(id, old_s, new_s, kind) {
    changes[nrow(changes) + 1L, ] <<- list(id, old_s, new_s, kind)
  }

  for (i in seq_along(ids)) {
    id <- ids[i]
    obs_s <- observations$observed_s[i]
    if (is.na(id) || !nzchar(id) || id %in% perf_ratchet_reserved || !perf_ratchet_valid_s(obs_s)) next
    iters <- observations$iterations[i]
    entry <- baseline[[id]]

    if (is.null(entry) || !perf_ratchet_valid_s(entry$median_s)) {
      baseline[[id]] <- list(median_s = obs_s, recorded_at = recorded_at, commit = commit,
        iterations = iters, last_observed_s = obs_s)
      add_change(id, NA_real_, obs_s, "new")
      next
    }

    old_s <- as.numeric(entry$median_s)
    last_s <- entry$last_observed_s
    if (perf_ratchet_valid_s(last_s)) {
      confirmed <- max(as.numeric(last_s), obs_s)
      if (confirmed < old_s) {
        entry$median_s <- confirmed
        entry$recorded_at <- recorded_at
        entry$commit <- commit
        entry$iterations <- iters
        add_change(id, old_s, confirmed, "lowered")
      }
    }
    entry$last_observed_s <- obs_s
    baseline[[id]] <- entry
  }

  list(baseline = baseline, changes = changes)
}

# TRUE only if the suite, perf and conformance parts all passed: each has its
# PASS line and none of the part status lines says FAIL. Only a status line
# counts ("suite: FAIL", "conformance: OWNER FAIL: ..."); a NOTE or WARNING that
# merely has FAIL in a test name does not.
perf_ratchet_gate_passed <- function(summary_lines) {
  has <- function(part) any(startsWith(summary_lines, paste0(part, ": PASS")))
  failed <- any(grepl("^(suite|perf|conformance): (OWNER )?FAIL", summary_lines))
  has("suite") && has("perf") && has("conformance") && !failed
}

perf_ratchet_format_changes <- function(changes) {
  fmt <- function(x) format(x, digits = 6)
  vapply(seq_len(nrow(changes)), function(i) {
    old <- if (is.na(changes$old_s[i])) "new" else fmt(changes$old_s[i])
    paste(changes$id[i], old, "->", fmt(changes$new_s[i]))
  }, character(1))
}

# The merge-queue step. A gate that did not pass changes nothing (returns NULL).
# Otherwise: apply, write out_dir/perf_ratchet.txt, write the baseline
# atomically, then append the changes to the gate summary. The rename is the
# last step that can fail, so an error anywhere leaves the baseline untouched.
perf_ratchet_run <- function(baseline_path, observations_path, summary_path, out_dir, commit,
                             recorded_at = perf_ratchet_now()) {
  summary_lines <- readLines(summary_path, warn = FALSE)
  if (!perf_ratchet_gate_passed(summary_lines)) return(NULL)

  baseline <- perf_ratchet_read_baseline(baseline_path)
  obs <- perf_ratchet_read_observations(observations_path)
  res <- perf_ratchet_apply(baseline, obs, commit, recorded_at)

  lines <- if (nrow(res$changes) > 0) perf_ratchet_format_changes(res$changes) else "unchanged"
  if (!dir.exists(out_dir)) stop("directory does not exist: ", out_dir, call. = FALSE)
  writeLines(lines, file.path(out_dir, "perf_ratchet.txt"))
  perf_ratchet_write_baseline(baseline_path, res$baseline)
  # The baseline is already written: a failure here must not look like one.
  tryCatch(
    cat(paste0("perf baseline: ", lines), file = summary_path, sep = "\n", append = TRUE),
    error = function(e) warning("could not append to ", summary_path, ": ", conditionMessage(e), call. = FALSE)
  )
  res$changes
}

# Theo's deliberate set (raise or lower) of named entries. `values` is a named
# numeric vector of seconds. last_observed_s is dropped, so the next gate only
# seeds the entry again. All checks run before anything is written.
perf_baseline_set <- function(baseline_path, values, commit = "manual", recorded_at = perf_ratchet_now()) {
  ids <- names(values)
  if (length(values) == 0 || is.null(ids) || anyNA(ids) || any(!nzchar(ids))) {
    stop("values must be a named numeric vector (id = seconds)", call. = FALSE)
  }
  if (anyDuplicated(ids)) stop("duplicate ids: ", paste(unique(ids[duplicated(ids)]), collapse = ", "), call. = FALSE)
  bad <- intersect(ids, perf_ratchet_reserved)
  if (length(bad) > 0) {
    stop("reserved ids cannot be set (re-record instead): ", paste(bad, collapse = ", "), call. = FALSE)
  }
  ok <- vapply(values, perf_ratchet_valid_s, logical(1))
  if (!all(ok)) stop("values must be finite and positive: ", paste(ids[!ok], collapse = ", "), call. = FALSE)

  baseline <- perf_ratchet_read_baseline(baseline_path)
  for (id in ids) {
    old <- baseline[[id]]
    baseline[[id]] <- list(median_s = as.numeric(values[[id]]), recorded_at = recorded_at, commit = commit,
      iterations = old$iterations %||% 5)
  }
  perf_ratchet_write_baseline(baseline_path, baseline)
  invisible(baseline)
}
