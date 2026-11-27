# #146: calls to deprecated libcudf/RMM APIs fail a check.
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

test_that("#146: the check fails on a call to a deprecated API and names the call site", {
  p146_skip_if_no_source_tree()
  probe <- c(
    '#include "cuplyr_compat.hpp"',
    '',
    '[[deprecated("probe146: use cuplyr_probe_new_api()")]] inline int cuplyr_probe_old_api() { return 1; }',
    '',
    'extern "C" int cuplyr_probe_self() { return cuplyr_probe_old_api(); }',
    '',
    '#if CUPLYR_CUDF_NEW_API',
    '// Deprecated in libcudf 26.10 in favour of apply_retention_mask().',
    'std::unique_ptr<cudf::table> cuplyr_probe_cudf(cudf::table_view const& t,',
    '                                               cudf::column_view const& m) {',
    '  return cudf::apply_boolean_mask(t, m);',
    '}',
    '#endif'
  )
  dir <- p146_mini_pkg(list(probe.cpp = probe))
  res <- p146_run_check(dir)
  out <- paste(res$output, collapse = "\n")

  expect_false(identical(res$status, 0L), label = "check exit status is non-zero")
  expect_match(out, "probe\\.cpp:[0-9]+", label = "check output names the file and line")
  expect_match(out, "cuplyr_probe_old_api", fixed = TRUE)
  expect_match(out, "deprecated-declarations", fixed = TRUE)
  if (p146_cudf_has_new_api()) {
    expect_match(out, "apply_boolean_mask", fixed = TRUE)
  }
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
  # Shown, not hidden.
  expect_match(out, "probe146: this header is deprecated", fixed = TRUE)
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
