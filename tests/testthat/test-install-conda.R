# The R installer's conda and pixi paths, driven by a fake mamba on PATH.

# A fake mamba that logs its arguments. With `populate = TRUE` it creates
# libcudf, nvcc and conda-meta in the `-p` prefix, as a successful solve would.
local_fake_mamba <- function(populate = TRUE, status = 0L, env = parent.frame()) {
  fake_bin <- withr::local_tempdir("cuplyr-bin-", .local_envir = env)
  log_file <- file.path(fake_bin, "mamba.log")
  body <- c(
    "#!/usr/bin/env bash",
    sprintf('echo "$*" >> %s', shQuote(log_file)),
    'prefix=""; prev=""',
    'for a in "$@"; do [ "$prev" = "-p" ] && prefix="$a"; prev="$a"; done'
  )
  if (populate) {
    body <- c(
      body,
      'mkdir -p "$prefix/include/cudf" "$prefix/lib" "$prefix/bin" "$prefix/conda-meta"',
      'touch "$prefix/include/cudf/types.hpp" "$prefix/lib/libcudf.so"',
      'printf "#!/bin/sh\\n" > "$prefix/bin/nvcc"; chmod +x "$prefix/bin/nvcc"'
    )
  }
  body <- c(body, sprintf("exit %d", status))
  writeLines(body, file.path(fake_bin, "mamba"))
  Sys.chmod(file.path(fake_bin, "mamba"), mode = "0755")
  withr::local_envvar(
    PATH = paste(fake_bin, Sys.getenv("PATH"), sep = ":"),
    .local_envir = env
  )
  log_file
}

local_installer_mocks <- function(env = parent.frame()) {
  testthat::local_mocked_bindings(
    detect_environment = function(override = "") "container",
    run_in_dir = function(dir, cmd, args = character(), verbose = FALSE) 0L,
    .env = env
  )
}

run_install_via_conda <- function(conda_prefix = NULL, dry_run = FALSE) {
  install_via_conda(
    src_dir = withr::local_tempdir("cuplyr-src-"),
    conda_prefix = conda_prefix,
    configure_args = character(),
    dry_run = dry_run,
    verbose = FALSE
  )
}

test_that("install_via_conda() pins RAPIDS 25.12 on CUDA 12 and adds nvcc", {
  skip_on_os("windows")
  log_file <- local_fake_mamba()
  local_installer_mocks()

  expect_no_error(run_install_via_conda(withr::local_tempdir("cuplyr-conda-")))

  calls <- readLines(log_file, warn = FALSE)
  expect_length(calls, 1)
  expect_match(calls, "--no-channel-priority", fixed = TRUE)
  expect_match(calls, "libcudf=25.12", fixed = TRUE)
  expect_match(calls, "cuda-version>=12.2,<13", fixed = TRUE)
  expect_match(calls, "cuda-nvcc", fixed = TRUE)
  expect_match(calls, "cuda-cudart-dev", fixed = TRUE)
})

test_that("install_via_conda() does not fall back to an unpinned RAPIDS", {
  skip_on_os("windows")
  log_file <- local_fake_mamba(populate = FALSE, status = 1L)
  local_installer_mocks()

  expect_error(run_install_via_conda(withr::local_tempdir("cuplyr-conda-")))

  calls <- readLines(log_file, warn = FALSE)
  expect_true(
    all(grepl("libcudf=25.12", calls, fixed = TRUE)),
    info = paste(calls, collapse = "\n")
  )
})

test_that("install_via_conda() defaults to a lasting prefix, not tempdir()", {
  data_home <- withr::local_tempdir("xdg-data-")
  withr::local_envvar(XDG_DATA_HOME = data_home, CONDA_PREFIX = NA, CONDA_DEFAULT_ENV = NA)
  local_installer_mocks()

  expect_message(
    run_install_via_conda(dry_run = TRUE),
    file.path(data_home, "cuplyr", "rapids"),
    fixed = TRUE
  )
})

test_that("install_via_conda() adds to an active environment instead of replacing it", {
  skip_on_os("windows")
  env_dir <- withr::local_tempdir("user-env-")
  dir.create(file.path(env_dir, "conda-meta"))
  withr::local_envvar(CONDA_PREFIX = env_dir, CONDA_DEFAULT_ENV = "myenv")
  log_file <- local_fake_mamba()
  local_installer_mocks()

  expect_no_error(run_install_via_conda())

  calls <- readLines(log_file, warn = FALSE)
  expect_match(calls[1], paste("^install -y -p", env_dir))
})

test_that("install_via_conda() leaves the base environment alone", {
  skip_on_os("windows")
  base_dir <- withr::local_tempdir("conda-base-")
  dir.create(file.path(base_dir, "conda-meta"))
  data_home <- withr::local_tempdir("xdg-data-")
  withr::local_envvar(
    CONDA_PREFIX = base_dir, CONDA_DEFAULT_ENV = "base", XDG_DATA_HOME = data_home
  )
  log_file <- local_fake_mamba()
  local_installer_mocks()

  expect_no_error(run_install_via_conda())

  calls <- readLines(log_file, warn = FALSE)
  expect_false(any(grepl(base_dir, calls, fixed = TRUE)))
  expect_match(calls[1], file.path(data_home, "cuplyr", "rapids"), fixed = TRUE)
})

test_that("install_via_conda() adds nvcc to a prefix that already has libcudf", {
  skip_on_os("windows")
  prefix <- withr::local_tempdir("cuplyr-conda-")
  dir.create(file.path(prefix, "include", "cudf"), recursive = TRUE)
  dir.create(file.path(prefix, "lib"))
  file.create(file.path(prefix, c("include/cudf/types.hpp", "lib/libcudf.so")))
  log_file <- local_fake_mamba()
  local_installer_mocks()

  expect_no_error(run_install_via_conda(prefix))

  calls <- readLines(log_file, warn = FALSE)
  expect_length(calls, 1)
  expect_match(calls, "^install ")
  expect_match(calls, "cuda-nvcc", fixed = TRUE)
  expect_false(grepl("libcudf", calls, fixed = TRUE))
})

test_that("install_cuplyr() looks for pixi.toml in the source tree", {
  src_dir <- withr::local_tempdir("cuplyr-src-")
  writeLines(c("Package: cuplyr", "Version: 0.0.1"), file.path(src_dir, "DESCRIPTION"))
  file.create(file.path(src_dir, c("configure", "pixi.toml")))
  withr::local_dir(withr::local_tempdir("elsewhere-"))
  testthat::local_mocked_bindings(has_command = function(cmd) cmd %in% c("pixi", "mamba"))

  expect_message(
    install_cuplyr(repo = src_dir, dry_run = TRUE),
    "Selected install method: pixi",
    fixed = TRUE
  )
})

test_that("check_cuda_toolkit() finds nvcc in a conda environment", {
  skip_on_os("windows")
  prefix <- withr::local_tempdir("conda-env-")
  dir.create(file.path(prefix, "bin"))
  nvcc <- file.path(prefix, "bin", "nvcc")
  writeLines(
    c("#!/bin/sh", "echo 'Cuda compilation tools, release 12.9, V12.9.86'"),
    nvcc
  )
  Sys.chmod(nvcc, "0755")
  withr::local_envvar(CUDA_HOME = NA, CONDA_PREFIX = prefix, PATH = "/usr/bin:/bin")

  res <- check_cuda_toolkit()

  expect_true(res$ok, info = res$message)
  expect_equal(res$value, "12.9")
  expect_match(res$message, prefix, fixed = TRUE)
})
