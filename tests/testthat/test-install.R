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

test_that("find_real_driver_lib() finds large libcuda.so.1 files", {
  tmp <- tempfile("cuplyr-driver-")
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  # Create a large file (real driver)
  real_driver <- file.path(tmp, "libcuda.so.1")
  # Write 2MB of data to simulate real driver
  writeBin(raw(2000000), real_driver)

  testthat::local_mocked_bindings(
    find_real_driver_lib = function() {
      for (p in c(tmp, "/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu")) {
        so <- file.path(p, "libcuda.so.1")
        if (file.exists(so)) {
          size <- file.info(so)$size
          if (!is.na(size) && size >= 1000000) return(p)
        }
      }
      NULL
    }
  )

  result <- find_real_driver_lib()
  expect_equal(result, tmp)
})

test_that("find_real_driver_lib() ignores stub libraries", {
  tmp <- tempfile("cuplyr-stub-")
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  # Create a small file (stub)
  stub <- file.path(tmp, "libcuda.so.1")
  writeBin(raw(1000), stub)

  testthat::local_mocked_bindings(
    find_real_driver_lib = function() {
      for (p in c(tmp)) {
        so <- file.path(p, "libcuda.so.1")
        if (file.exists(so)) {
          size <- file.info(so)$size
          if (!is.na(size) && size >= 1000000) return(p)
        }
      }
      NULL
    }
  )

  result <- find_real_driver_lib()
  expect_null(result)
})

test_that("disable_cuda_stubs() renames small stub files", {
  tmp <- tempfile("cuplyr-stubs-")
  lib_dir <- file.path(tmp, "lib")
  stubs_dir <- file.path(lib_dir, "stubs")
  dir.create(stubs_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  # Create stub files
  stub1 <- file.path(stubs_dir, "libcuda.so")
  stub2 <- file.path(lib_dir, "libcuda.so.1")
  writeBin(raw(1000), stub1)
  writeBin(raw(2000), stub2)

  disabled <- disable_cuda_stubs(lib_dir)

  expect_true(file.exists(paste0(stub1, ".disabled")))
  expect_true(file.exists(paste0(stub2, ".disabled")))
  expect_false(file.exists(stub1))
  expect_false(file.exists(stub2))
  expect_length(disabled, 2)
})

test_that("disable_cuda_stubs() preserves large driver files", {
  tmp <- tempfile("cuplyr-driver-")
  lib_dir <- file.path(tmp, "lib")
  dir.create(lib_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  # Create a large file (real driver)
  real_driver <- file.path(lib_dir, "libcuda.so.1")
  writeBin(raw(2000000), real_driver)

  disabled <- disable_cuda_stubs(lib_dir)

  expect_true(file.exists(real_driver))
  expect_false(file.exists(paste0(real_driver, ".disabled")))
  expect_length(disabled, 0)
})

test_that("detect_environment() detects local by default", {
  withr::local_envvar(c(
    COLAB_RELEASE_TAG = "",
    COLAB_GPU = "",
    KUBERNETES_SERVICE_HOST = "",
    CUPLYR_ENV = ""
  ))

  testthat::local_mocked_bindings(
    has_command = function(cmd) FALSE
  )

  expect_equal(detect_environment(), "local")
})

test_that("detect_environment() detects container from Kubernetes env var", {
  withr::local_envvar(c(
    COLAB_RELEASE_TAG = "",
    COLAB_GPU = "",
    KUBERNETES_SERVICE_HOST = "kubernetes.default.svc.cluster.local",
    CUPLYR_ENV = ""
  ))

  expect_equal(detect_environment(), "container")
})
