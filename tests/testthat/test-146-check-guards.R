# #146: tools/check-deprecations only passes when the check really ran with
# deprecations as errors, against the active environment's libcudf. These
# build a one-file package (the real configure and src/Makevars.in), like
# test-parity-146-deprecations.R, and skip without the source tree.

g146_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

g146_skip_if_no_source_tree <- function() {
  root <- g146_root()
  skip_if_not(
    file.exists(file.path(root, "configure")) &&
      file.exists(file.path(root, "src", "Makevars.in")) &&
      file.exists(file.path(root, "tools", "check-deprecations")),
    "source tree not available in this test context"
  )
}

# `deprecated`: probe.cpp calls a deprecated declaration (only
# -Werror=deprecated-declarations fails it); otherwise it calls nothing.
g146_mini_pkg <- function(deprecated, env = parent.frame()) {
  root <- g146_root()
  dir <- withr::local_tempdir("cuplyr146g-", .local_envir = env)
  dir.create(file.path(dir, "src"))
  stopifnot(file.copy(file.path(root, "configure"), dir))
  Sys.chmod(file.path(dir, "configure"), "755")
  stopifnot(all(file.copy(file.path(root, "src", c("Makevars.in", "cuplyr_compat.hpp")),
                          file.path(dir, "src"))))
  writeLines(c(
    "Package: cuplyrprobe146g",
    "Version: 0.0.1",
    "Title: Deprecation Check Probe",
    "Description: Compiles one file with the cuplyr build flags.",
    "License: GPL-3"
  ), file.path(dir, "DESCRIPTION"))
  writeLines("useDynLib(cuplyrprobe146g)", file.path(dir, "NAMESPACE"))
  writeLines(c(
    '#include "cuplyr_compat.hpp"',
    '[[deprecated("probe146")]] inline int cuplyr_probe_old_api() { return 1; }',
    if (deprecated) {
      'extern "C" int cuplyr_probe_g() { return cuplyr_probe_old_api(); }'
    } else {
      'extern "C" int cuplyr_probe_g() { return 0; }'
    }
  ), file.path(dir, "src", "probe.cpp"))
  dir
}

g146_run_check <- function(dir, env) {
  script <- file.path(g146_root(), "tools", "check-deprecations")
  out <- withr::with_envvar(env, suppressWarnings(
    system2(script, c("--src", shQuote(dir)), stdout = TRUE, stderr = TRUE)
  ))
  list(status = as.integer(attr(out, "status") %||% 0L), output = paste(out, collapse = "\n"))
}

test_that("#146: a user flag that cancels the deprecation error fails the check, named", {
  g146_skip_if_no_source_tree()
  dir <- g146_mini_pkg(deprecated = TRUE)
  # As a ~/.R/Makevars would: these come after PKG_CXXFLAGS on the compile line.
  makevars <- withr::local_tempfile(lines = "CXX20FLAGS += -Wno-error=deprecated-declarations -w")
  res <- g146_run_check(dir, c(R_MAKEVARS_USER = makevars))
  expect_false(identical(res$status, 0L), label = paste("check exit status\n", res$output))
  expect_match(res$output, "deprecations: FAIL[^\n]*cancels[^\n]*-Wno-error=deprecated-declarations", perl = TRUE)
  expect_match(res$output, "deprecations: FAIL[^\n]*cancels[^\n]* -w", perl = TRUE)
  expect_false(grepl("deprecations: PASS", res$output, fixed = TRUE))
})

test_that("#146: the check fails when configure picks a libcudf outside the active environment", {
  g146_skip_if_no_source_tree()
  prefix <- Sys.getenv("CONDA_PREFIX")
  skip_if_not(nzchar(prefix) && file.exists(file.path(prefix, "include", "cudf", "types.hpp")),
              "no libcudf in the active environment")
  # A valid CUDF_HOME that is not CONDA_PREFIX (here a link farm onto the same
  # files, so the build itself works); configure prefers it. An empty user
  # Renviron so a CUDF_HOME set there doesn't replace this one.
  other <- withr::local_tempdir("cuplyr146-cudf-home-")
  for (d in c("include", "lib")) file.symlink(file.path(prefix, d), file.path(other, d))
  renviron <- withr::local_tempfile(lines = character())
  dir <- g146_mini_pkg(deprecated = FALSE)
  res <- g146_run_check(dir, c(CUDF_HOME = other, R_ENVIRON_USER = renviron))
  expect_false(identical(res$status, 0L), label = paste("check exit status\n", res$output))
  expect_match(res$output, "deprecations: FAIL[^\n]*outside the active environment[^\n]*CUDF_HOME", perl = TRUE)
  expect_false(grepl("deprecations: PASS", res$output, fixed = TRUE))
})
