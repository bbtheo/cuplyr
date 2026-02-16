test_that("get_source_dir() finds cloned child cuplyr directory", {
  tmp <- tempfile("cuplyr-src-root-")
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  src_dir <- file.path(tmp, "cuplyr")
  dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c(
      "Package: cuplyr",
      "Version: 0.0.0.9000"
    ),
    file.path(src_dir, "DESCRIPTION")
  )
  file.create(file.path(src_dir, "configure"))

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(tmp)

  detected <- get_source_dir(
    repo = NULL,
    ref = "master",
    dry_run = FALSE,
    verbose = FALSE
  )

  expect_equal(detected, normalizePath(src_dir))
})

test_that("detect_environment() prioritizes Colab signals over container marker", {
  withr::local_envvar(c(
    COLAB_RELEASE_TAG = "test-colab",
    COLAB_GPU = "1",
    KUBERNETES_SERVICE_HOST = ""
  ))

  expect_equal(detect_environment(), "colab")
})

test_that("install_via_conda() retries with relaxed package constraints", {
  src_dir <- tempfile("cuplyr-src-")
  dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(src_dir, recursive = TRUE, force = TRUE), add = TRUE)

  conda_prefix <- tempfile("cuplyr-conda-")
  on.exit(unlink(conda_prefix, recursive = TRUE, force = TRUE), add = TRUE)

  fake_bin <- tempfile("cuplyr-bin-")
  dir.create(fake_bin, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(fake_bin, recursive = TRUE, force = TRUE), add = TRUE)

  log_file <- file.path(fake_bin, "mamba.log")
  state_file <- file.path(fake_bin, "mamba.state")

  mamba_path <- file.path(fake_bin, "mamba")
  writeLines(
    c(
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      sprintf("echo \"$@\" >> %s", shQuote(log_file)),
      sprintf("if [ ! -f %s ]; then", shQuote(state_file)),
      sprintf("  touch %s", shQuote(state_file)),
      "  exit 2",
      "fi",
      "mkdir -p \"$CUPLYR_TEST_PREFIX/include/cudf\"",
      "mkdir -p \"$CUPLYR_TEST_PREFIX/lib\"",
      "touch \"$CUPLYR_TEST_PREFIX/include/cudf/types.hpp\"",
      "touch \"$CUPLYR_TEST_PREFIX/lib/libcudf.so\"",
      "exit 0"
    ),
    mamba_path
  )
  Sys.chmod(mamba_path, mode = "0755")

  withr::local_envvar(c(
    PATH = paste(fake_bin, Sys.getenv("PATH"), sep = ":"),
    CUPLYR_TEST_PREFIX = conda_prefix
  ))

  testthat::local_mocked_bindings(
    detect_environment = function(override = "") "container",
    run_in_dir = function(dir, cmd, args = character(), verbose = FALSE) 0L
  )

  expect_no_error(
    install_via_conda(
      src_dir = src_dir,
      conda_prefix = conda_prefix,
      configure_args = character(),
      dry_run = FALSE,
      verbose = FALSE
    )
  )

  calls <- readLines(log_file, warn = FALSE)
  expect_length(calls, 2)
  expect_match(calls[1], "libcudf=25\\.12")
  expect_false(grepl("cuda-toolkit", calls[1]))
  expect_match(calls[2], "(^| )libcudf( |$)")
  expect_match(calls[2], "(^| )librmm( |$)")
  expect_match(calls[2], "(^| )libkvikio( |$)")
})

test_that("install_via_conda() errors clearly when libcudf artifacts are missing", {
  src_dir <- tempfile("cuplyr-src-")
  dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(src_dir, recursive = TRUE, force = TRUE), add = TRUE)

  conda_prefix <- tempfile("cuplyr-conda-")
  on.exit(unlink(conda_prefix, recursive = TRUE, force = TRUE), add = TRUE)

  fake_bin <- tempfile("cuplyr-bin-")
  dir.create(fake_bin, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(fake_bin, recursive = TRUE, force = TRUE), add = TRUE)

  mamba_path <- file.path(fake_bin, "mamba")
  writeLines(
    c(
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "exit 0"
    ),
    mamba_path
  )
  Sys.chmod(mamba_path, mode = "0755")

  withr::local_envvar(c(PATH = paste(fake_bin, Sys.getenv("PATH"), sep = ":")))

  testthat::local_mocked_bindings(
    detect_environment = function(override = "") "container",
    run_in_dir = function(dir, cmd, args = character(), verbose = FALSE) 0L
  )

  expect_error(
    install_via_conda(
      src_dir = src_dir,
      conda_prefix = conda_prefix,
      configure_args = character(),
      dry_run = FALSE,
      verbose = FALSE
    ),
    "libcudf headers/libraries"
  )
})
