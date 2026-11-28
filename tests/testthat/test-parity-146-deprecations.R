# #146: calls to deprecated declarations (libcudf, RMM or any other non-system header) fail a check.
#
# The ordinary build never fails on a deprecation (a new RAPIDS release must
# not break an install). The opt-in switch CUPLYR_WERROR_DEPRECATED=1, read by
# configure, compiles with -Werror=deprecated-declarations, and
# tools/check-deprecations builds a fresh copy of the sources with it (no GPU,
# no lock, the source tree untouched). The merge gate runs that check and
# tools/check (the release check) builds with the switch on.
#
# The compile tests build a one-file package from the real configure and
# src/Makevars.in, so they exercise the same flags as the package build in a
# few seconds instead of a full rebuild. They need the source tree and the
# build environment, so they skip under R CMD check (tests run from the
# installed package there).

p146_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p146_skip_if_no_source_tree <- function() {
  root <- p146_root()
  skip_if_not(
    file.exists(file.path(root, "configure")) &&
      file.exists(file.path(root, "src", "Makevars.in")) &&
      file.exists(file.path(root, "src", "cuplyr_compat.hpp")) &&
      dir.exists(file.path(root, "tools")),
    "source tree not available in this test context"
  )
}

# A one-file package: the real configure, src/Makevars.in and
# src/cuplyr_compat.hpp, plus `files` (name -> lines) in src/.
p146_mini_pkg <- function(files = list(), env = parent.frame()) {
  root <- p146_root()
  dir <- withr::local_tempdir("cuplyr146-", .local_envir = env)
  dir.create(file.path(dir, "src"))
  stopifnot(file.copy(file.path(root, "configure"), dir))
  Sys.chmod(file.path(dir, "configure"), "755")
  stopifnot(all(file.copy(file.path(root, "src", c("Makevars.in", "cuplyr_compat.hpp")),
                          file.path(dir, "src"))))
  writeLines(c(
    "Package: cuplyrprobe146",
    "Version: 0.0.1",
    "Title: Deprecation Check Probe",
    "Description: Compiles one file with the cuplyr build flags.",
    "License: GPL-3",
    "Encoding: UTF-8"
  ), file.path(dir, "DESCRIPTION"))
  writeLines("useDynLib(cuplyrprobe146)", file.path(dir, "NAMESPACE"))
  for (nm in names(files)) writeLines(files[[nm]], file.path(dir, "src", nm))
  dir
}

# Runs configure in `dir` with CUPLYR_WERROR_DEPRECATED set to `value` (NA:
# unset); returns the generated src/Makevars lines.
p146_configure <- function(dir, value) {
  withr::with_envvar(c(CUPLYR_WERROR_DEPRECATED = value), {
    out <- suppressWarnings(system2(
      "bash", c("-c", shQuote(paste("cd", shQuote(dir), "&& ./configure"))),
      stdout = TRUE, stderr = TRUE
    ))
  })
  status <- attr(out, "status") %||% 0L
  expect_identical(as.integer(status), 0L,
                   label = paste("configure exit status\n", paste(out, collapse = "\n")))
  readLines(file.path(dir, "src", "Makevars"), warn = FALSE)
}

p146_werror_lines <- function(makevars) {
  code <- makevars[!grepl("^\\s*#", makevars)]
  grep("-Werror=deprecated-declarations", code, fixed = TRUE, value = TRUE)
}

p146_check_script <- function() {
  file.path(p146_root(), "tools", "check-deprecations")
}

# Runs tools/check-deprecations --src `dir`; list(status, output).
p146_run_check <- function(dir) {
  script <- p146_check_script()
  expect_true(file.exists(script), label = "tools/check-deprecations exists")
  if (!file.exists(script)) {
    return(list(status = NA_integer_, output = character()))
  }
  out <- suppressWarnings(system2(script, c("--src", shQuote(dir)), stdout = TRUE, stderr = TRUE))
  list(status = as.integer(attr(out, "status") %||% 0L), output = out)
}

p146_cudf_has_new_api <- function() {
  prefix <- Sys.getenv("CONDA_PREFIX")
  nzchar(prefix) && file.exists(file.path(prefix, "include", "cudf", "reduction", "distinct_count.hpp"))
}

test_that("#146: configure adds -Werror=deprecated-declarations only when asked", {
  p146_skip_if_no_source_tree()
  dir <- p146_mini_pkg()

  for (on in c("1", "true", "yes")) {
    lines <- p146_werror_lines(p146_configure(dir, on))
    expect_true(length(lines) >= 1L, label = paste0("Makevars has the flag with CUPLYR_WERROR_DEPRECATED=", on))
  }
  # Off by default: an ordinary build (and a user's install) never fails on a
  # deprecation.
  for (off in c(NA, "", "0", "false")) {
    expect_identical(p146_werror_lines(p146_configure(dir, off)), character(),
                     label = paste0("Makevars -Werror lines with CUPLYR_WERROR_DEPRECATED=", off))
  }
})

test_that("#146: deprecations are detected, never suppressed", {
  p146_skip_if_no_source_tree()
  root <- p146_root()
  makevars_in <- readLines(file.path(root, "src", "Makevars.in"), warn = FALSE)
  expect_false(any(grepl("-Wno-deprecated", makevars_in, fixed = TRUE)))
  expect_false(any(grepl("-Wno-error", makevars_in, fixed = TRUE)))
  dir <- p146_mini_pkg()
  makevars <- p146_configure(dir, "1")
  expect_false(any(grepl("-Wno-deprecated", makevars, fixed = TRUE)))
  expect_false(any(grepl("-Wno-error", makevars, fixed = TRUE)))
})

# The check's summary: the lines between its "--- deprecation ..." marker and
# its "deprecations: PASS|FAIL" verdict.
p146_summary <- function(output) {
  first <- grep("^--- deprecation", output)
  last <- grep("^deprecations: (PASS|FAIL)", output)
  if (length(first) == 0L || length(last) == 0L || max(last) <= first[1L]) {
    return(NA_character_)
  }
  output[seq.int(first[1L] + 1L, length.out = max(last) - first[1L] - 1L)]
}

test_that("#146: the check fails on a call to a deprecated API and names the call site", {
  p146_skip_if_no_source_tree()
  # One deprecated call in a header the .cpp includes, one in the .cpp.
  calls <- c(
    '#pragma once',
    '[[deprecated("probe146: use cuplyr_probe_new_api()")]] inline int cuplyr_probe_old_api() { return 1; }',
    'inline int cuplyr_probe_from_header() { return cuplyr_probe_old_api(); }'
  )
  probe <- c(
    '#include "cuplyr_compat.hpp"',
    '#include "probe_calls.hpp"',
    '',
    'extern "C" int cuplyr_probe_self() { return cuplyr_probe_old_api() + cuplyr_probe_from_header(); }',
    '',
    '#if CUPLYR_CUDF_NEW_API',
    '// Deprecated in libcudf 26.10 in favour of apply_retention_mask().',
    'std::unique_ptr<cudf::table> cuplyr_probe_cudf(cudf::table_view const& t,',
    '                                               cudf::column_view const& m) {',
    '  return cudf::apply_boolean_mask(t, m);',
    '}',
    '#endif'
  )
  dir <- p146_mini_pkg(list(probe.cpp = probe, probe_calls.hpp = calls))
  res <- p146_run_check(dir)
  out <- paste(res$output, collapse = "\n")

  expect_false(identical(res$status, 0L), label = "check exit status is non-zero")
  expect_match(out, "deprecations: FAIL", fixed = TRUE)
  # The summary lists the compiler's diagnostics with file:line, and only them
  # (no compile command lines, which all carry the flag).
  summary <- p146_summary(res$output)
  expect_false(anyNA(summary), label = paste("the check prints a summary section\n", out))
  diag <- "\\[-Werror=deprecated-declarations\\]"
  expect_true(any(grepl(paste0("probe\\.cpp:[0-9]+:[0-9]+: error: .*cuplyr_probe_old_api.*", diag), summary)),
              label = paste("summary names the call in probe.cpp\n", paste(summary, collapse = "\n")))
  expect_true(any(grepl(paste0("probe_calls\\.hpp:[0-9]+:[0-9]+: error: .*cuplyr_probe_old_api.*", diag), summary)),
              label = paste("summary names the call in probe_calls.hpp\n", paste(summary, collapse = "\n")))
  if (p146_cudf_has_new_api()) {
    expect_true(any(grepl(paste0("probe\\.cpp:[0-9]+:[0-9]+: error: .*apply_boolean_mask.*", diag), summary)),
                label = paste("summary names the libcudf call\n", paste(summary, collapse = "\n")))
  }
  expect_false(any(grepl("-fpic|-c probe\\.cpp", summary)),
               label = "summary holds no compile command lines")
  # The sources are compiled in a copy: nothing is written next to them.
  expect_identical(
    list.files(file.path(dir, "src"), pattern = "(\\.o|\\.so)$|^Makevars$"),
    character()
  )
})

test_that("#146: code using only current APIs passes the check, header deprecations only warn", {
  p146_skip_if_no_source_tree()
  # A deprecated *header* (as RMM 26.02 includes CCCL's <cuda/stream_ref>)
  # warns with #pragma GCC warning, not a deprecated declaration: it is
  # RAPIDS's own include, not a cuplyr call, and must not fail the check.
  header <- c(
    '#pragma once',
    '#pragma GCC warning "probe146: this header is deprecated"'
  )
  probe <- c(
    '#include "cuplyr_compat.hpp"',
    '#include "probe_old_header.hpp"',
    '',
    'std::unique_ptr<cudf::table> cuplyr_probe_filter(cudf::table_view const& t,',
    '                                                 cudf::column_view const& m) {',
    '  return cuplyr::apply_retention_mask(t, m);',
    '}'
  )
  dir <- p146_mini_pkg(list(probe.cpp = probe, probe_old_header.hpp = header))
  res <- p146_run_check(dir)
  out <- paste(res$output, collapse = "\n")

  expect_identical(res$status, 0L, label = paste("check exit status\n", out))
  expect_match(out, "deprecations: PASS", fixed = TRUE)
  # Shown in the summary, not hidden.
  summary <- p146_summary(res$output)
  expect_true(any(grepl("probe_old_header\\.hpp:[0-9]+.*warning: .*probe146: this header is deprecated", summary)),
              label = paste("summary shows the header deprecation\n", paste(summary, collapse = "\n")))
  expect_false(any(grepl("-fpic|-c probe\\.cpp", summary)),
               label = "summary holds no compile command lines")
})

test_that("#146: cuplyr's sources never silence a deprecation", {
  p146_skip_if_no_source_tree()
  # The check is only as good as the sources it compiles: a
  # `#pragma GCC diagnostic ignored "-Wdeprecated-declarations"` around a call
  # makes it pass silently (GCC's `ignored` beats -Werror=). No exceptions.
  src <- file.path(p146_root(), "src")
  files <- list.files(src, pattern = "\\.(c|cc|cpp|h|hpp|cu|cuh)$|^Makevars", full.names = TRUE)
  files <- files[basename(files) != "Makevars"]  # generated; Makevars.in is checked
  pattern <- paste(
    c('diagnostic\\s+(ignored|warning)\\s+\\\\?"-Wdeprecated',  # #pragma and _Pragma("...\\"...")
      "-Wno-deprecated", "-Wno-error", "IGNORE_DEPRECATED", "DISABLE_DEPRECATION",
      "_SILENCE_[A-Z_]*DEPRECAT",
      "pragma\\s+GCC\\s+system_header",    # silences every diagnostic below it
      "(^|[\\s=\"'])-w([\\s\"']|$)",         # drops all warnings
      # CCCL's and libcudf's own wrappers around the same pragma
      # (_CCCL_SUPPRESS_DEPRECATED_PUSH, _CCCL_DIAG_SUPPRESS_GCC("-W..."), ...)
      # and any other string literal naming a deprecation warning.
      "SUPPRESS_DEPRECATED|DIAG_SUPPRESS",
      '\\\\?"-W(no-)?(error=)?deprecated'),
    collapse = "|"
  )
  hits <- unlist(lapply(files, function(f) {
    lines <- readLines(f, warn = FALSE)
    i <- grep(pattern, lines, perl = TRUE)
    if (length(i)) paste0(basename(f), ":", i, ": ", trimws(lines[i])) else character()
  }))
  expect_identical(hits, character())
})

test_that("#146: a POSIXct mean still rounds half away from zero, eager, lazy, grouped and windowed", {
  skip_if_no_gpu()
  # Half-microsecond means either side of the epoch, and at +-2e15 microseconds
  # (2033 / 1906), where the double mean is still exact.
  ties <- tibble::tibble(
    g = rep(1:6, each = 2),
    t = .POSIXct(c(0, 1e-6, -1e-6, 0, 2e-6, 3e-6, -3e-6, -2e-6,
                   2e9, 2e9 + 1e-6, -2e9 - 1e-6, -2e9), tz = "UTC")
  )
  want_us <- c(1, -1, 3, -3, 2e15 + 1, -2e15 - 1)
  # What collect() gives for those microsecond counts.
  want <- as.numeric(collect(tbl_gpu(tibble::tibble(t = .POSIXct(want_us / 1e6, tz = "UTC"))))$t)
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(ties, lazy = lazy)
    got <- collect(summarise(gt, m = mean(t), .by = g))
    expect_identical(as.numeric(got$m), want, label = paste("summarise, lazy =", lazy))
    got <- collect(mutate(gt, m = mean(t), .by = g))
    expect_identical(as.numeric(got$m), rep(want, each = 2), label = paste("mutate, lazy =", lazy))
    got <- collect(summarise(group_by(gt, g), m = median(t)))
    expect_identical(as.numeric(got$m), want, label = paste("median, lazy =", lazy))
  }
  # Equal values in [2^52, 2^53) microseconds (year 2112 on): every mean is an
  # integer and must come back unchanged. floor(|f| + 0.5) would tie an odd
  # count to the even one above it.
  big_us <- 2^52 + c(1, 3, 5, 7, 1001, 123457, 2^51 - 1)
  big <- tibble::tibble(g = rep(seq_along(big_us), each = 2),
                        t = .POSIXct(rep(big_us / 1e6, each = 2), tz = "UTC"))
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(big, lazy = lazy)
    stored <- as.numeric(collect(gt)$t)[c(TRUE, FALSE)]
    got <- collect(summarise(gt, m = mean(t), .by = g))
    expect_identical(as.numeric(got$m), stored, label = paste("mean past 2^52, lazy =", lazy))
    got <- collect(mutate(gt, m = mean(t), .by = g))
    expect_identical(as.numeric(got$m), rep(stored, each = 2),
                     label = paste("windowed mean past 2^52, lazy =", lazy))
    got <- collect(summarise(gt, m = median(t), .by = g))
    expect_identical(as.numeric(got$m), stored, label = paste("median past 2^52, lazy =", lazy))
  }
  # NA and an all-NA group stay NA.
  na <- tibble::tibble(g = c(1L, 1L, 2L), t = .POSIXct(c(1e-6, NA, NA), tz = "UTC"))
  got <- collect(summarise(tbl_gpu(na), m = mean(t, na.rm = TRUE), n = mean(t), .by = g))
  expect_identical(is.na(got$m), c(FALSE, TRUE))
  expect_identical(is.na(got$n), c(TRUE, TRUE))
})

test_that("#146: src/Makevars.in stays portable make (no GNU extensions)", {
  p146_skip_if_no_source_tree()
  # The package check warns on these in src/Makevars(.in) ("checking for GNU
  # extensions in Makefiles"); main passes that check.
  lines <- readLines(file.path(p146_root(), "src", "Makevars.in"), warn = FALSE)
  code <- lines[!grepl("^\\s*#", lines)]
  gnu <- "\\+=|:=|\\?=|!=|^\\s*(ifeq|ifneq|ifdef|ifndef|else|endif)\\b|\\$\\(shell|\\$\\(wildcard|\\.NOTPARALLEL"
  expect_identical(grep(gnu, code, value = TRUE, perl = TRUE), character())
  # Every placeholder sits inside an assignment, never on a line of its own
  # (make is run on Makevars.in for some checks).
  expect_false(any(grepl("^\\s*@[A-Z_]+@\\s*$", code)))
})

test_that("#146: the check fails when the switch does not reach the compiler", {
  p146_skip_if_no_source_tree()
  # A configure or Makevars.in that lost the switch must not pass silently:
  # strip the deprecation placeholder from the probe's Makevars.in.
  probe <- c('#include "cuplyr_compat.hpp"', 'extern "C" int cuplyr_probe_ok() { return 0; }')
  dir <- p146_mini_pkg(list(probe.cpp = probe))
  mi <- file.path(dir, "src", "Makevars.in")
  lines <- readLines(mi, warn = FALSE)
  stripped <- gsub("@[A-Z_]*DEPRECAT[A-Z_]*@", "", lines)
  expect_false(identical(stripped, lines), label = "Makevars.in has a deprecation placeholder")
  writeLines(stripped, mi)
  res <- p146_run_check(dir)
  out <- paste(res$output, collapse = "\n")
  expect_false(identical(res$status, 0L), label = paste("check exit status\n", out))
  expect_match(out, "deprecations: FAIL[^\n]*-Werror=deprecated-declarations", perl = TRUE)

  # Not a package: FAIL with a reason, no PASS.
  empty <- withr::local_tempdir("cuplyr146-empty-")
  res <- p146_run_check(empty)
  out <- paste(res$output, collapse = "\n")
  expect_false(identical(res$status, 0L))
  expect_match(out, "deprecations: FAIL (", fixed = TRUE)
  expect_false(grepl("deprecations: PASS", out, fixed = TRUE))
})

test_that("#146: one check run lists the call sites of every file", {
  p146_skip_if_no_source_tree()
  # make -k: porting to a new RAPIDS needs every call site, not the first file's.
  dep <- '[[deprecated("probe146")]] inline int cuplyr_probe_old_api() { return 1; }'
  dir <- p146_mini_pkg(list(
    probe_a.cpp = c(dep, 'extern "C" int cuplyr_probe_a() { return cuplyr_probe_old_api(); }'),
    probe_b.cpp = c(dep, 'extern "C" int cuplyr_probe_b() { return cuplyr_probe_old_api(); }')
  ))
  res <- p146_run_check(dir)
  summary <- p146_summary(res$output)
  expect_false(identical(res$status, 0L))
  expect_true(any(grepl("probe_a\\.cpp:[0-9]+:[0-9]+: error: ", summary)),
              label = paste("summary lists probe_a.cpp\n", paste(summary, collapse = "\n")))
  expect_true(any(grepl("probe_b\\.cpp:[0-9]+:[0-9]+: error: ", summary)),
              label = paste("summary lists probe_b.cpp\n", paste(summary, collapse = "\n")))
})

test_that("#146: the merge gate and the release check run with deprecations as errors", {
  p146_skip_if_no_source_tree()
  root <- p146_root()
  gate <- readLines(file.path(root, "tools", "merge-gate"), warn = FALSE)
  gate_code <- gate[!grepl("^\\s*#", gate)]
  expect_true(any(grepl("tools/check-deprecations", gate_code, fixed = TRUE)),
              label = "tools/merge-gate runs tools/check-deprecations")
  expect_true(any(grepl("deprecations: FAIL", gate_code, fixed = TRUE)),
              label = "tools/merge-gate reports a failed deprecation check")
  check <- readLines(file.path(root, "tools", "check"), warn = FALSE)
  check_code <- check[!grepl("^\\s*#", check)]
  expect_true(any(grepl("CUPLYR_WERROR_DEPRECATED", check_code, fixed = TRUE)),
              label = "tools/check builds with CUPLYR_WERROR_DEPRECATED")
})
