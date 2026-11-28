# Planning helpers for tools/run-tests.R (sharded suite runs). No side effects
# when sourced: functions only, nothing is loaded, read or run.

# The shard count `tools/test` uses when neither --shards nor CUPLYR_TEST_SHARDS
# says otherwise. Chosen from measured peaks of GPU and host memory, see
# .github/CONTRIBUTING.md; capped by a third of the cores so the machine keeps
# headroom. (An explicit count is not capped.)
cuplyr_default_test_shards <- 3L

# A shard count from user input: a positive whole number written in digits.
# `what` names where it came from, for the error.
check_shard_count <- function(x, what) {
  x <- trimws(as.character(x))
  n <- suppressWarnings(as.numeric(x))
  if (length(x) != 1L || !grepl("^[0-9]+$", x) || is.na(n) || n < 1 || n > .Machine$integer.max) {
    stop(what, " must be a positive whole number, not '", paste(x, collapse = " "), "'",
         call. = FALSE)
  }
  as.integer(n)
}

# Shard count: CUPLYR_TEST_SHARDS if set (a positive whole number), else the
# built-in default capped at a third of the cores (1 when the core count is
# unknown).
default_shards <- function() {
  cores <- parallel::detectCores()
  cap <- if (is.na(cores)) 1L else max(1L, cores %/% 3L)
  env <- trimws(Sys.getenv("CUPLYR_TEST_SHARDS", ""))
  if (!nzchar(env)) {
    return(as.integer(min(cuplyr_default_test_shards, cap)))
  }
  check_shard_count(env, "CUPLYR_TEST_SHARDS")
}

# One file name per line; '#' starts a comment; blank lines are ignored. A
# missing file is an empty list.
read_serial_list <- function(path) {
  if (!file.exists(path)) return(character())
  lines <- trimws(sub("#.*$", "", readLines(path, warn = FALSE)))
  lines[nzchar(lines)]
}

# `file<TAB>seconds` with a header row, as a named numeric vector; a missing
# file is an empty one.
read_durations <- function(path) {
  empty <- stats::setNames(numeric(), character())
  if (!file.exists(path) || file.size(path) == 0L) return(empty)
  d <- utils::read.delim(path, sep = "\t", quote = "", stringsAsFactors = FALSE,
                         colClasses = c(file = "character", seconds = "numeric"))
  d <- d[is.finite(d$seconds), , drop = FALSE]
  if (nrow(d) == 0L) return(empty)
  stats::setNames(d$seconds, d$file)
}

# Byte (C locale) order, whatever the session's locale.
sort_bytes <- function(x) sort(x, method = "radix")

# Which files run in which process. Serial-listed files go to `serial`; the
# rest are dealt out longest-processing-time first: by recorded seconds,
# longest first (ties by name), each to the least loaded shard (equal loads:
# the one with fewer files, then the lowest index). A file with no recorded
# duration counts as the median of the recorded ones of the files being planned
# (1 s when none is recorded). There are never more shards than parallel files,
# and none is left empty, even when every duration is zero. Each shard and
# `serial` list their files in byte order of the name. The result does not
# depend on the order of `files`, `durations` or `serial`.
suite_plan <- function(files, durations, serial, shards) {
  files <- unique(as.character(files))
  serial <- sort_bytes(intersect(as.character(serial), files))
  par <- setdiff(files, serial)
  n <- min(as.integer(shards), length(par))
  if (n < 1L) {
    return(list(shards = list(), serial = serial))
  }

  secs <- as.numeric(durations[par])
  known <- is.finite(secs)
  guess <- if (any(known)) stats::median(secs[known]) else 1
  secs[!known] <- guess
  secs <- pmax(secs, 0)

  ord <- order(-secs, factor(par, levels = sort_bytes(par)))
  load <- numeric(n)
  count <- integer(n)
  bins <- vector("list", n)
  for (i in ord) {
    # least loaded; among equals the one with fewest files (so zero-second
    # files still spread out and no shard stays empty), then the lowest index
    k <- order(load, count)[[1L]]
    bins[[k]] <- c(bins[[k]], par[[i]])
    load[[k]] <- load[[k]] + secs[[i]]
    count[[k]] <- count[[k]] + 1L
  }
  list(shards = lapply(bins, sort_bytes), serial = serial)
}
