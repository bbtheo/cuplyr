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

  if (identical(cuplyr_so_build_state(root), "unoptimized")) {
    message("cuplyr: src/cuplyr.so was built with -O0; rebuilding optimized")
    pkgbuild::clean_dll(root)
  }
  pkgbuild::compile_dll(root, debug = FALSE, quiet = quiet)

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
