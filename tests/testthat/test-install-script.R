test_that("install.sh dry-run works outside source tree", {
  script <- normalizePath(
    file.path(testthat::test_path("..", ".."), "install.sh"),
    mustWork = FALSE
  )
  skip_if_not(
    file.exists(script),
    message = "install.sh not available in this test context"
  )

  tmp <- tempfile("cuplyr-install-test-")
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(tmp)

  output <- system2(
    "bash",
    c(script, "--dry-run", "--method=pixi"),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }

  expect_equal(
    as.integer(status),
    0L,
    info = paste(output, collapse = "\n")
  )
})

install_script <- function() {
  normalizePath(
    file.path(testthat::test_path("..", ".."), "install.sh"),
    mustWork = FALSE
  )
}

# Runs install.sh with `args` from a scratch directory. `env` is set and
# `unset` removed from the environment; `bin` holds fake commands placed
# first on PATH, ahead of only the system directories.
run_install_script <- function(args, env = character(), unset = character(),
                               bin = NULL, cwd = NULL) {
  skip_on_os("windows")
  skip_if_not(
    file.exists(install_script()),
    message = "install.sh not available in this test context"
  )
  if (is.null(cwd)) {
    cwd <- withr::local_tempdir("cuplyr-install-cwd-", .local_envir = parent.frame())
  }
  if (!is.null(bin)) {
    env[["PATH"]] <- paste(c(bin, "/usr/bin", "/bin"), collapse = ":")
  }
  unset <- unique(c(unset, "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CUDA_HOME"))
  cmd <- c(
    as.vector(rbind("-u", setdiff(unset, names(env)))),
    if (length(env)) sprintf("%s=%s", names(env), env),
    "bash", install_script(), args
  )
  output <- withr::with_dir(cwd, system2("env", cmd, stdout = TRUE, stderr = TRUE))
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = paste(output, collapse = "\n"))
}

fake_command <- function(bin, name, body) {
  dir.create(bin, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(bin, name)
  writeLines(c("#!/usr/bin/env bash", body), path)
  Sys.chmod(path, "0755")
  path
}

test_that("auto-detection looks for pixi.toml in the cloned source", {
  bin <- withr::local_tempdir("fake-bin-")
  log <- file.path(bin, "pixi.log")
  fake_command(bin, "R", "exit 0")
  # `git clone ... <dir>`: the last argument is the target directory.
  fake_command(bin, "git", c(
    'dir="${@: -1}"',
    'mkdir -p "$dir"',
    'touch "$dir/DESCRIPTION" "$dir/configure" "$dir/pixi.toml"'
  ))
  fake_command(bin, "pixi", sprintf('echo "$PWD $*" >> %s', shQuote(log)))

  res <- run_install_script(character(), env = c(HOME = bin), bin = bin)

  expect_equal(res$status, 0L, info = res$output)
  expect_match(res$output, "Auto-detected method: pixi", fixed = TRUE)
  expect_match(readLines(log), "/cuplyr run install$")
})

test_that("conda install defaults to a lasting prefix, not a temp dir", {
  home <- withr::local_tempdir("fake-home-")

  res <- run_install_script(
    c("--method=conda", "--dry-run"),
    env = c(HOME = home, XDG_DATA_HOME = file.path(home, "data"))
  )

  expect_equal(res$status, 0L, info = res$output)
  expect_match(
    res$output,
    sprintf("create -y -p \"%s\"", file.path(home, "data", "cuplyr", "rapids")),
    fixed = TRUE
  )
})

test_that("conda install also installs the CUDA compiler and headers", {
  res <- run_install_script(
    c("--method=conda", "--dry-run"),
    env = c(HOME = withr::local_tempdir("fake-home-"))
  )

  expect_equal(res$status, 0L, info = res$output)
  expect_match(res$output, "cuda-nvcc", fixed = TRUE)
  expect_match(res$output, "cuda-cudart-dev", fixed = TRUE)
})

test_that("conda install adds to an active environment instead of replacing it", {
  env_dir <- withr::local_tempdir("user-env-")
  dir.create(file.path(env_dir, "conda-meta"))

  res <- run_install_script(
    c("--method=conda", "--dry-run"),
    env = c(
      HOME = withr::local_tempdir("fake-home-"),
      CONDA_PREFIX = env_dir,
      CONDA_DEFAULT_ENV = "myenv"
    )
  )

  expect_equal(res$status, 0L, info = res$output)
  expect_match(res$output, sprintf("install -y -p \"%s\"", env_dir), fixed = TRUE)
  expect_no_match(res$output, "create -y", fixed = TRUE)
})

test_that("conda install leaves the base environment alone", {
  base_dir <- withr::local_tempdir("conda-base-")
  dir.create(file.path(base_dir, "conda-meta"))

  res <- run_install_script(
    c("--method=conda", "--dry-run"),
    env = c(
      HOME = withr::local_tempdir("fake-home-"),
      CONDA_PREFIX = base_dir,
      CONDA_DEFAULT_ENV = "base"
    )
  )

  expect_equal(res$status, 0L, info = res$output)
  expect_no_match(res$output, base_dir, fixed = TRUE)
})

test_that("system install leaves the CUDA search to configure", {
  bin <- withr::local_tempdir("fake-bin-")
  fake_command(bin, "R", "exit 0")
  src <- withr::local_tempdir("cuplyr-src-")
  file.create(file.path(src, "DESCRIPTION"))
  fake_command(src, "configure", "echo configure-ran >&2; exit 0")

  # A CUDA_HOME without include/cuda.h, as on a machine whose toolkit is
  # somewhere configure searches but install.sh used not to.
  res <- run_install_script(
    "--method=system",
    env = c(HOME = bin, CUDA_HOME = "/nonexistent-cuda"),
    bin = bin,
    cwd = src
  )

  expect_equal(res$status, 0L, info = res$output)
  expect_match(res$output, "configure-ran", fixed = TRUE)
})

test_that("conda install pins RAPIDS 25.12 on CUDA 12 whatever the channel priority", {
  res <- run_install_script(
    c("--method=conda", "--dry-run"),
    env = c(HOME = withr::local_tempdir("fake-home-"))
  )

  expect_equal(res$status, 0L, info = res$output)
  expect_match(res$output, "--no-channel-priority", fixed = TRUE)
  expect_match(res$output, "libcudf=25.12", fixed = TRUE)
  expect_match(res$output, "cuda-version>=12.2,<13", fixed = TRUE)
})

test_that("a failed conda solve stops instead of installing an unpinned RAPIDS", {
  bin <- withr::local_tempdir("fake-bin-")
  log <- file.path(bin, "mamba.log")
  fake_command(bin, "R", "exit 0")
  fake_command(bin, "mamba", c(sprintf('echo "$*" >> %s', shQuote(log)), "exit 1"))
  src <- withr::local_tempdir("cuplyr-src-")
  file.create(file.path(src, c("DESCRIPTION", "configure")))

  res <- run_install_script(
    "--method=conda",
    env = c(HOME = withr::local_tempdir("fake-home-")),
    bin = bin,
    cwd = src
  )

  expect_false(res$status == 0L, info = res$output)
  calls <- readLines(log)
  expect_true(all(grepl("libcudf=25.12", calls, fixed = TRUE)), info = paste(calls, collapse = "\n"))
})
