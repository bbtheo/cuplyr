# #135: how the conda package links CUDA, and what conda/build.sh feeds
# configure.
#
# Contract:
# - CUPLYR_FORCE_CUDA_STUBS=1 makes configure link libcuda against the CUDA
#   stub of the toolkit it uses, even when the build machine has a real
#   driver, and put no driver directory (and no stub directory) on the
#   library's RUNPATH. libcuda is linked --as-needed: cuplyr calls only the
#   CUDA runtime API, so the built library does not list libcuda.so.1 as
#   needed and loads on a machine without a driver (a conda build or test
#   machine); at run time libcudart finds the user's driver through the
#   system loader. With no stub to link, configure fails rather than fall
#   back to the build machine's driver.
# - Without the switch a source install behaves as before: a driver found on
#   the machine is linked and its directory is on RUNPATH.
# - conda/build.sh (run by rattler-build and conda-build in the source
#   directory, with PREFIX = the host environment) configures against PREFIX
#   only, whatever CONDA_PREFIX, CUDA_HOME or CUDF_HOME the caller's shell
#   has; removes compiled objects copied from a development checkout; forces
#   the stub; installs with $R CMD INSTALL; and stops when configure fails.
#
# These tests need neither a GPU nor a CUDA install: configure and build.sh
# run against fake prefixes in a scratch copy of the package (as
# test-configure.R does).

p135b_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p135b_file <- function(...) file.path(p135b_root(), ...)

p135b_skip_if_no_sources <- function(...) {
  skip_on_os("windows")
  skip_if_not(
    file.exists(p135b_file("configure")) && dir.exists(p135b_file("conda")),
    message = "configure / conda/ not available in this test context"
  )
}

p135b_touch <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  file.create(path)
}

p135b_script <- function(path, lines) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(c("#!/usr/bin/env bash", lines), path)
  Sys.chmod(path, "0755")
}

# A conda host environment with RAPIDS `cudf_version` and a CUDA 13 toolkit
# (cuda-nvcc, cuda-cudart-dev, cuda-driver-dev), and a fake R that records
# how it was called.
p135b_fake_prefix <- function(root, stub = TRUE, cudf_version = c(26, 10, 0)) {
  root <- normalizePath(root)
  p135b_touch(file.path(root, "targets", "x86_64-linux", "include", "cuda.h"))
  if (stub) {
    p135b_touch(file.path(root, "targets", "x86_64-linux", "lib", "stubs", "libcuda.so"))
  }
  p135b_script(file.path(root, "bin", "nvcc"),
               "echo 'Cuda compilation tools, release 13.0, V13.0.0'")
  p135b_touch(file.path(root, "lib", "libcudart.so"))
  p135b_touch(file.path(root, "include", "cudf", "types.hpp"))
  p135b_touch(file.path(root, "include", "rmm", "rmm.hpp"))
  writeLines(
    c("#pragma once",
      sprintf("#define CUDF_VERSION_MAJOR %s", cudf_version[[1]]),
      sprintf("#define CUDF_VERSION_MINOR %02d", cudf_version[[2]]),
      sprintf("#define CUDF_VERSION_PATCH %02d", cudf_version[[3]])),
    file.path(root, "include", "cudf", "version_config.hpp")
  )
  for (lib in c("libcudf.so", "librmm.so", "libkvikio.so")) {
    p135b_touch(file.path(root, "lib", lib))
  }
  p135b_script(file.path(root, "bin", "R"),
               "printf '%s\\n' \"$*\" >> \"$(dirname \"$0\")/../r-calls.txt\"")
  root
}

# A system toolkit: everything under one CUDA_HOME.
p135b_fake_system_cuda <- function(root) {
  root <- normalizePath(root)
  p135b_touch(file.path(root, "include", "cuda.h"))
  p135b_touch(file.path(root, "lib64", "stubs", "libcuda.so"))
  p135b_touch(file.path(root, "lib64", "libcudart.so"))
  p135b_script(file.path(root, "bin", "nvcc"),
               "echo 'Cuda compilation tools, release 13.0, V13.0.0'")
  root
}

# A scratch copy of what configure and build.sh read.
p135b_scratch_pkg <- function(envir = parent.frame()) {
  pkg <- normalizePath(withr::local_tempdir("cuplyr-conda-", .local_envir = envir))
  dir.create(file.path(pkg, "src"))
  dir.create(file.path(pkg, "conda"))
  file.copy(p135b_file("configure"), pkg)
  file.copy(p135b_file("DESCRIPTION"), pkg)
  file.copy(p135b_file("src", "Makevars.in"), file.path(pkg, "src"))
  if (file.exists(p135b_file("conda", "build.sh"))) {
    file.copy(p135b_file("conda", "build.sh"), file.path(pkg, "conda"))
  }
  pkg
}

p135b_run <- function(pkg, cmd, env = character(), unset = character()) {
  args <- c(as.vector(rbind("-u", unset)), sprintf("%s=%s", names(env), env), "bash", cmd)
  # a missing script makes bash exit 127, which system2() raises as an error
  output <- withr::with_dir(pkg, suppressWarnings(tryCatch(
    system2("env", args, stdout = TRUE, stderr = TRUE),
    error = function(e) structure(conditionMessage(e), status = 127L)
  )))
  status <- attr(output, "status")
  makevars <- file.path(pkg, "src", "Makevars")
  vars <- character()
  text <- character()
  if (file.exists(makevars)) {
    text <- readLines(makevars, warn = FALSE)
    lines <- grep("^[A-Za-z_]+\\s*=", text, value = TRUE)
    vars <- stats::setNames(trimws(sub("^[^=]*=", "", lines)), trimws(sub("=.*", "", lines)))
  }
  list(
    status = if (is.null(status)) 0L else as.integer(status),
    output = paste(output, collapse = "\n"),
    vars = vars,
    makevars = paste(text, collapse = "\n")
  )
}

p135b_configure <- function(prefix, force = NULL) {
  pkg <- p135b_scratch_pkg(parent.frame())
  env <- c(CONDA_PREFIX = prefix)
  if (!is.null(force)) env <- c(env, CUPLYR_FORCE_CUDA_STUBS = force)
  p135b_run(pkg, "configure", env = env,
            unset = c("CUDA_HOME", "CUDF_HOME", "CUPLYR_ALLOW_CUDA_STUBS", "CUPLYR_FORCE_CUDA_STUBS"))
}

# Expand $(NAME) references with the Makevars' own variables.
p135b_expand <- function(vars, name) {
  value <- vars[[name]]
  for (i in 1:10) {
    refs <- unique(regmatches(value, gregexpr("\\$\\(([A-Za-z_]+)\\)", value))[[1]])
    if (!length(refs)) break
    for (ref in refs) {
      key <- gsub("^\\$\\(|\\)$", "", ref)
      value <- gsub(ref, if (key %in% names(vars)) vars[[key]] else "", value, fixed = TRUE)
    }
  }
  value
}

# Every directory a link line searches (-L) or records (-rpath).
p135b_link_dirs <- function(pkg_libs) {
  tokens <- strsplit(trimws(pkg_libs), "\\s+")[[1]]
  l_dirs <- sub("^-L", "", grep("^-L.", tokens, value = TRUE))
  rpath_tokens <- grep("-rpath", tokens, value = TRUE)
  rpaths <- sub("^.*-rpath[,=]", "", rpath_tokens)
  rpaths <- unlist(strsplit(rpaths, "[:,]"))
  list(L = l_dirs, rpath = rpaths[nzchar(rpaths)])
}

# Is --as-needed in effect for -lcuda on this link line?
p135b_cuda_as_needed <- function(pkg_libs) {
  tokens <- strsplit(trimws(pkg_libs), "\\s+")[[1]]
  pos <- which(tokens == "-lcuda")
  if (length(pos) != 1L) return(NA)
  before <- unlist(strsplit(tokens[seq_len(pos - 1L)], ","))
  toggles <- grep("^--(no-)?as-needed$", before, value = TRUE)
  length(toggles) > 0L && toggles[[length(toggles)]] == "--as-needed"
}

p135b_real_driver_dir <- function() {
  out <- suppressWarnings(tryCatch(
    system2("sh", c("-c", shQuote("ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null")),
            stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  ))
  hits <- sub("^.*=>\\s*", "", grep("libcuda\\.so\\.1\\b", out, value = TRUE))
  hits <- hits[file.exists(hits) & !grepl("/stubs/|/compat/|/targets/", hits)]
  hits <- hits[file.size(hits) > 1e6]
  if (length(hits)) dirname(hits[[1]]) else NULL
}

# ---------------------------------------------------------------------------
# configure: CUPLYR_FORCE_CUDA_STUBS
# ---------------------------------------------------------------------------

test_that("CUPLYR_FORCE_CUDA_STUBS=1 links the environment's stub, even with a driver installed", {
  p135b_skip_if_no_sources()
  prefix <- p135b_fake_prefix(withr::local_tempdir("host-env-"))

  res <- p135b_configure(prefix, force = "1")

  expect_identical(res$status, 0L, info = res$output)
  expect_identical(res$vars[["CUDA_DRIVER_LIB"]],
                   file.path(prefix, "targets", "x86_64-linux", "lib", "stubs"),
                   info = res$output)
  expect_identical(res$vars[["CUDA_DRIVER_RPATH_FLAG"]], "", info = res$makevars)
  expect_match(res$output, "stub", ignore.case = TRUE)

  dirs <- p135b_link_dirs(p135b_expand(res$vars, "PKG_LIBS"))
  expect_gt(length(dirs$rpath), 0L)
  expect_true(all(startsWith(dirs$rpath, prefix)), info = paste(dirs$rpath, collapse = "\n"))
  expect_false(any(grepl("stubs", dirs$rpath)), info = paste(dirs$rpath, collapse = "\n"))
  expect_true(all(startsWith(dirs$L, prefix)), info = paste(dirs$L, collapse = "\n"))
})

test_that("CUPLYR_FORCE_CUDA_STUBS=1 links libcuda --as-needed", {
  p135b_skip_if_no_sources()
  prefix <- p135b_fake_prefix(withr::local_tempdir("host-env-"))

  res <- p135b_configure(prefix, force = "1")

  expect_identical(res$status, 0L, info = res$output)
  pkg_libs <- p135b_expand(res$vars, "PKG_LIBS")
  expect_true(isTRUE(p135b_cuda_as_needed(pkg_libs)), info = pkg_libs)
  # the rest of the link line is as before: libcudf and libcudart are linked
  expect_match(pkg_libs, "(^|\\s)-lcudf(\\s|$)")
  expect_match(pkg_libs, "(^|\\s)-lcudart(\\s|$)")
})

test_that("CUPLYR_FORCE_CUDA_STUBS=1 fails when there is no stub to link", {
  p135b_skip_if_no_sources()
  prefix <- p135b_fake_prefix(withr::local_tempdir("host-env-"), stub = FALSE)

  res <- p135b_configure(prefix, force = "1")

  expect_false(identical(res$status, 0L), info = res$output)
  expect_match(res$output, "stub", ignore.case = TRUE)
  expect_match(res$output, "CUPLYR_FORCE_CUDA_STUBS", fixed = TRUE)
})

test_that("without CUPLYR_FORCE_CUDA_STUBS, a source install still links the installed driver", {
  p135b_skip_if_no_sources()
  driver <- p135b_real_driver_dir()
  skip_if(is.null(driver), "no NVIDIA driver on this machine")
  prefix <- p135b_fake_prefix(withr::local_tempdir("host-env-"))

  for (force in list(NULL, "0")) {
    res <- p135b_configure(prefix, force = force)
    expect_identical(res$status, 0L, info = res$output)
    expect_false(grepl("stubs", res$vars[["CUDA_DRIVER_LIB"]]), info = res$output)
    expect_match(res$vars[["CUDA_DRIVER_RPATH_FLAG"]], "-rpath", fixed = TRUE, info = res$makevars)
  }
})

# ---------------------------------------------------------------------------
# conda/build.sh
# ---------------------------------------------------------------------------

p135b_build_env <- function(pkg, host, build) {
  c(
    PREFIX = host,
    BUILD_PREFIX = build,
    SRC_DIR = pkg,
    RECIPE_DIR = file.path(pkg, "conda"),
    PKG_NAME = "r-cuplyr",
    PKG_VERSION = read.dcf(file.path(pkg, "DESCRIPTION"), fields = "Version")[[1]],
    CONDA_BUILD = "1",
    R = file.path(host, "bin", "R"),
    PATH = paste(file.path(host, "bin"), file.path(build, "bin"), Sys.getenv("PATH"), sep = ":")
  )
}

test_that("conda/build.sh configures against the host prefix only and installs with $R", {
  p135b_skip_if_no_sources()
  host <- p135b_fake_prefix(withr::local_tempdir("host-env-"))
  build <- normalizePath(withr::local_tempdir("build-env-"))
  dir.create(file.path(build, "bin"))
  # what a maintainer's shell may hold: an active pixi/conda env with another
  # RAPIDS, and a system toolkit
  other_env <- p135b_fake_prefix(withr::local_tempdir("active-env-"), cudf_version = c(26, 2, 0))
  system_cuda <- p135b_fake_system_cuda(withr::local_tempdir("system-cuda-"))

  pkg <- p135b_scratch_pkg()
  # objects and a Makevars left by a development build of the checkout
  for (f in c("stale.o", "cuplyr.so")) p135b_touch(file.path(pkg, "src", f))
  writeLines("CUDF_INCLUDE=/somewhere/else", file.path(pkg, "src", "Makevars"))

  env <- c(p135b_build_env(pkg, host, build),
           CONDA_PREFIX = other_env, CUDF_HOME = other_env, CUDA_HOME = system_cuda)
  res <- p135b_run(pkg, file.path("conda", "build.sh"), env = env,
                   unset = c("CUPLYR_ALLOW_CUDA_STUBS", "CUPLYR_FORCE_CUDA_STUBS"))

  expect_identical(res$status, 0L, info = res$output)
  expect_false(file.exists(file.path(pkg, "src", "stale.o")))
  expect_false(file.exists(file.path(pkg, "src", "cuplyr.so")))

  expect_identical(res$vars[["CUDF_INCLUDE"]], file.path(host, "include"), info = res$makevars)
  expect_identical(res$vars[["CUDA_INCLUDE"]],
                   file.path(host, "targets", "x86_64-linux", "include"), info = res$makevars)
  expect_identical(res$vars[["CUDA_DRIVER_LIB"]],
                   file.path(host, "targets", "x86_64-linux", "lib", "stubs"), info = res$makevars)
  expect_identical(res$vars[["CUDA_DRIVER_RPATH_FLAG"]], "", info = res$makevars)

  pkg_libs <- p135b_expand(res$vars, "PKG_LIBS")
  dirs <- p135b_link_dirs(pkg_libs)
  expect_true(all(startsWith(c(dirs$L, dirs$rpath), host)), info = pkg_libs)
  expect_true(isTRUE(p135b_cuda_as_needed(pkg_libs)), info = pkg_libs)
  expect_false(grepl(other_env, res$makevars, fixed = TRUE), info = res$makevars)
  expect_false(grepl(system_cuda, res$makevars, fixed = TRUE), info = res$makevars)

  calls <- file.path(host, "r-calls.txt")
  expect_true(file.exists(calls), info = res$output)
  if (file.exists(calls)) {
    expect_match(paste(readLines(calls), collapse = "\n"), "CMD INSTALL", fixed = TRUE)
  }
})

test_that("conda/build.sh stops when configure fails", {
  p135b_skip_if_no_sources()
  host <- p135b_fake_prefix(withr::local_tempdir("host-env-"), stub = FALSE)
  build <- normalizePath(withr::local_tempdir("build-env-"))
  dir.create(file.path(build, "bin"))
  pkg <- p135b_scratch_pkg()

  res <- p135b_run(pkg, file.path("conda", "build.sh"),
                   env = p135b_build_env(pkg, host, build),
                   unset = c("CONDA_PREFIX", "CUDA_HOME", "CUDF_HOME",
                             "CUPLYR_ALLOW_CUDA_STUBS", "CUPLYR_FORCE_CUDA_STUBS"))

  expect_false(identical(res$status, 0L), info = res$output)
  expect_false(file.exists(file.path(host, "r-calls.txt")), info = res$output)
})
