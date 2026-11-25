# configure's CUDF_HOME note on an unsupported libcudf (#61): it appears only
# when CUDF_HOME supplied the rejected prefix and that prefix is not the active
# conda env. These cover the CUDF_HOME layouts test-parity-61-version-bounds.R
# doesn't: a pip-wheel path, and the active env spelled with a trailing slash.

pkg_file <- function(...) {
  file.path(normalizePath(testthat::test_path("..", ".."), mustWork = FALSE), ...)
}

skip_if_no_configure <- function() {
  skip_on_os("windows")
  skip_if_not(
    file.exists(pkg_file("configure")) && file.exists(pkg_file("src", "Makevars.in")),
    message = "configure not available in this test context"
  )
  skip_if(Sys.which("bash") == "", message = "bash not available")
}

touch <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  file.create(path)
}

# A conda env with CUDA and a libcudf of the given c(major, minor, patch).
fake_prefix <- function(root, cudf_version) {
  touch(file.path(root, "targets", "x86_64-linux", "include", "cuda.h"))
  touch(file.path(root, "targets", "x86_64-linux", "lib", "stubs", "libcuda.so"))
  nvcc <- file.path(root, "bin", "nvcc")
  dir.create(dirname(nvcc), recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c("#!/usr/bin/env bash", "echo 'Cuda compilation tools, release 12.9, V12.9.0'"),
    nvcc
  )
  Sys.chmod(nvcc, "0755")
  touch(file.path(root, "lib", "libcudart.so"))
  touch(file.path(root, "include", "cudf", "types.hpp"))
  touch(file.path(root, "lib", "libcudf.so"))
  writeLines(
    sprintf("#define CUDF_VERSION_%s %s", c("MAJOR", "MINOR", "PATCH"), cudf_version),
    file.path(root, "include", "cudf", "version_config.hpp")
  )
  root
}

run_configure <- function(conda_prefix, cudf_home) {
  pkg <- withr::local_tempdir("cuplyr-configure-", .local_envir = parent.frame())
  dir.create(file.path(pkg, "src"))
  file.copy(pkg_file("configure"), pkg)
  file.copy(pkg_file("src", "Makevars.in"), file.path(pkg, "src"))
  args <- c(
    "-u", "CUDA_HOME",
    sprintf("CUDF_HOME=%s", cudf_home),
    sprintf("CONDA_PREFIX=%s", conda_prefix),
    "CUPLYR_ALLOW_CUDA_STUBS=1",
    "bash", "configure"
  )
  output <- withr::with_dir(pkg, suppressWarnings(
    system2("env", args, stdout = TRUE, stderr = TRUE)
  ))
  status <- attr(output, "status")
  list(
    status = if (is.null(status)) 0L else as.integer(status),
    output = paste(output, collapse = "\n")
  )
}

# The output after the first ERROR line.
after_error <- function(output) {
  lines <- strsplit(output, "\n", fixed = TRUE)[[1]]
  at <- grep("^ERROR", lines)
  if (length(at) == 0L) {
    return("")
  }
  paste(lines[(at[[1]] + 1L):length(lines)], collapse = "\n")
}

test_that("a pip-layout CUDF_HOME that supplied the rejected libcudf gets the note", {
  skip_if_no_configure()
  env <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "10", "2"))
  pip_env <- fake_prefix(withr::local_tempdir("pip-env-"), c("25", "12", "0"))
  cudf_home <- file.path(pip_env, "lib", "python3.12", "site-packages", "libcudf")
  dir.create(cudf_home, recursive = TRUE)

  res <- run_configure(env, cudf_home)

  expect_false(res$status == 0L, info = res$output)
  expect_match(res$output, paste0("Found libcudf at: ", pip_env), fixed = TRUE)
  expect_match(after_error(res$output), paste0("CUDF_HOME=", cudf_home), fixed = TRUE)
})

test_that("CUDF_HOME naming the active env with a trailing slash gets no note", {
  skip_if_no_configure()
  env <- fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0"))
  cudf_home <- paste0(env, "/")

  res <- run_configure(env, cudf_home)

  expect_false(res$status == 0L, info = res$output)
  expect_match(res$output, "25.12", fixed = TRUE)
  expect_no_match(after_error(res$output), paste0("CUDF_HOME=", cudf_home), fixed = TRUE)
})
