# Shared by tools/build, tools/document, tools/r, tools/run-tests.R and
# tools/run-dplyr-conformance.R: make sure src/cuplyr.so is compiled with
# optimization before anything is loaded or timed.
#
# pkgbuild::compile_dll() and pkgload::load_all() default to `debug = TRUE`,
# which appends `-UNDEBUG -Wall -pedantic -g -O0` and makes the R<->cudf
# conversion loops 2.5-5x slower than the -O2 build the perf baseline was
# recorded with. They only recompile when a source is newer than the .so, so
# flipping the flags does not rebuild a worktree that already holds an -O0
# .so; this does. Detection reads the compiler flags recorded in the .so's
# DWARF producer strings (" -O0 "), so it needs neither a load nor a GPU.

# "missing" (no .so yet), "unoptimized" (some unit records -O0), "unverifiable"
# (no GCC producer string, so the flags were not recorded: no -g, compressed
# debug sections, or clang) or "optimized". The same scan as
# perf_build_is_optimized() in tests/testthat/helper-perf.R.
cuplyr_so_build_state <- function(root) {
  so <- file.path(root, "src", "cuplyr.so")
  if (!file.exists(so)) {
    return("missing")
  }
  bytes <- readBin(so, "raw", file.size(so))
  if (length(grepRaw(" -O0 ", bytes, fixed = TRUE)) > 0) {
    return("unoptimized")
  }
  if (length(grepRaw("GNU C", bytes, fixed = TRUE)) == 0) {
    return("unverifiable")
  }
  "optimized"
}

# The build-env stamp. The object files in src/ are compiled against one
# libcudf/RMM and are not rebuilt when the active environment changes (make
# only compares timestamps; configure re-runs but only rewrites Makevars), so
# switching a worktree between RAPIDS series (CUPLYR_PIXI_ENV=rapids2602, or a
# `pixi install` that moves the default environment to a new series) would link
# stale objects into a broken .so. After every build the prefix and libcudf
# version it was built against are written next to the objects; when the
# current environment differs (or an .so predates the stamp), the objects are
# cleaned first.
cuplyr_build_stamp_file <- function(root) {
  file.path(root, "src", ".cuplyr-build-env")
}

# The libcudf version in `<prefix>/include/cudf/version_config.hpp`, as
# "MAJOR.MINOR.PATCH"; "unknown" when it can't be read.
cuplyr_prefix_cudf_version <- function(prefix) {
  header <- file.path(prefix, "include", "cudf", "version_config.hpp")
  if (!nzchar(prefix) || !file.exists(header)) {
    return("unknown")
  }
  lines <- readLines(header, warn = FALSE)
  field <- function(what) {
    hit <- regmatches(lines, regexec(
      paste0("^\\s*#\\s*define\\s+CUDF_VERSION_", what, "\\s+([0-9]+)"), lines))
    hit <- Filter(length, hit)
    if (length(hit) == 0L) NA_character_ else hit[[1L]][2L]
  }
  v <- c(field("MAJOR"), field("MINOR"), field("PATCH"))
  if (anyNA(v[1:2])) "unknown" else paste(v[!is.na(v)], collapse = ".")
}

# The environment the package would be built against now: the prefix configure
# picks (it skips a CUDF_HOME that holds no headers, so ask it rather than
# guess from the variables) and that prefix's libcudf version. Running
# configure only rewrites src/Makevars, which every compile rewrites anyway.
cuplyr_current_build_env <- function(root) {
  prefix <- ""
  status <- suppressWarnings(system2(
    "bash", c("-c", shQuote(paste("cd", shQuote(root), "&& bash ./configure"))),
    stdout = FALSE, stderr = FALSE))
  makevars <- file.path(root, "src", "Makevars")
  if (identical(as.integer(status), 0L) && file.exists(makevars)) {
    inc <- sub("^CUDF_INCLUDE=", "", grep("^CUDF_INCLUDE=", readLines(makevars), value = TRUE))
    if (length(inc) > 0L) prefix <- dirname(inc[[1L]])
  }
  c(prefix = prefix, libcudf = cuplyr_prefix_cudf_version(prefix))
}

cuplyr_read_build_stamp <- function(root) {
  f <- cuplyr_build_stamp_file(root)
  if (!file.exists(f)) {
    return(NULL)
  }
  lines <- readLines(f, warn = FALSE)
  kv <- regmatches(lines, regexec("^([a-z]+)=(.*)$", lines))
  kv <- Filter(length, kv)
  stats::setNames(vapply(kv, `[`, "", 3L), vapply(kv, `[`, "", 2L))
}

cuplyr_write_build_stamp <- function(root, env) {
  writeLines(paste0(names(env), "=", env), cuplyr_build_stamp_file(root))
}

# TRUE when the objects in src/ were not built against the environment `now`
# (from cuplyr_current_build_env()), or can't be shown to have been.
cuplyr_build_env_changed <- function(root, now) {
  if (!any(file.exists(file.path(root, "src", c("cuplyr.so", list.files(file.path(root, "src"), "\\.o$")))))) {
    return(FALSE)  # nothing built yet
  }
  stamp <- cuplyr_read_build_stamp(root)
  is.null(stamp) || !identical(unname(stamp[names(now)]), unname(now))
}

# Only positive evidence of -O0 triggers a clean rebuild. An unverifiable .so
# is not rebuilt and is not an error: a rebuild with the same toolchain records
# no flags either, so treating it as unoptimized would rebuild (or abort) on
# every tool call, and correctness runs don't need -O2. The build that just ran
# used debug = FALSE, so its flags are right by construction; the message says
# that this can't be checked. perf_check_build() refuses to time it, and the
# suite's "the loaded cuplyr build is optimized" test fails, so every full
# test run reports it.
cuplyr_ensure_optimized_dll <- function(root = ".", quiet = TRUE) {
  root <- normalizePath(root)
  Sys.setenv(PKG_BUILD_EXTRA_FLAGS = "false")

  # configure runs once per call: the environment is what the build below uses,
  # and what the stamp records once it has compiled.
  env <- cuplyr_current_build_env(root)
  if (cuplyr_build_env_changed(root, env)) {
    message("cuplyr: src/ was built against another libcudf/RMM than the active environment (",
            Sys.getenv("CONDA_PREFIX", ""), "); rebuilding")
    pkgbuild::clean_dll(root)
  }
  if (identical(cuplyr_so_build_state(root), "unoptimized")) {
    message("cuplyr: src/cuplyr.so was built with -O0; rebuilding optimized")
    pkgbuild::clean_dll(root)
  }
  pkgbuild::compile_dll(root, debug = FALSE, quiet = quiet)
  cuplyr_write_build_stamp(root, env)

  state <- cuplyr_so_build_state(root)
  if (identical(state, "unoptimized")) {
    stop("src/cuplyr.so is still compiled with -O0 after a rebuild; check src/Makevars", call. = FALSE)
  }
  if (identical(state, "unverifiable")) {
    message(
      "cuplyr: src/cuplyr.so records no GCC compiler flags (no -g, compressed debug info, or not GCC), ",
      "so -O0 objects can't be detected; perf checks will refuse this build and the suite's ",
      "\"the loaded cuplyr build is optimized\" test will fail"
    )
  }
  invisible(TRUE)
}
