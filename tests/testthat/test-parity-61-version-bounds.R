# cuplyr builds against RAPIDS 25.12 only: 26.02 removed the
# rmm/mr/device/*.hpp forwarding headers and 26.06 removed
# rmm::mr::device_memory_resource. The declared ranges must say so, and
# configure must reject an unsupported libcudf before the compiler does.
#
# None of these tests need a GPU. The configure tests run configure against
# fake prefixes in a scratch copy of the package (as test-configure.R does).

pkg_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

pkg_file <- function(...) {
  file.path(pkg_root(), ...)
}

skip_if_no_pkg_file <- function(...) {
  skip_if_not(
    file.exists(pkg_file(...)),
    message = paste(file.path(...), "not available in this test context")
  )
}

supported_range <- ">=25.12,<26.02"

# ---------------------------------------------------------------------------
# Declared ranges
# ---------------------------------------------------------------------------

test_that("pixi.toml bounds libcudf, librmm and libkvikio to >=25.12,<26.02", {
  skip_if_no_pkg_file("pixi.toml")
  lines <- readLines(pkg_file("pixi.toml"))

  for (pkg in c("libcudf", "librmm", "libkvikio")) {
    hit <- grep(sprintf("^\\s*%s\\s*=", pkg), lines, value = TRUE)
    expect_length(hit, 1L)
    spec <- gsub("\\s|\"|'", "", sub("^[^=]*=", "", hit))
    expect_identical(spec, supported_range, info = paste("pixi.toml:", pkg))
  }
})

test_that("conda/meta.yaml bounds libcudf, librmm and libkvikio in host and run", {
  skip_if_no_pkg_file("conda", "meta.yaml")
  lines <- readLines(pkg_file("conda", "meta.yaml"))

  for (pkg in c("libcudf", "librmm", "libkvikio")) {
    hit <- grep(sprintf("^\\s*-\\s*%s(\\s|$)", pkg), lines, value = TRUE)
    # one entry under requirements/host, one under requirements/run
    expect_length(hit, 2L)
    spec <- gsub("\\s", "", sub(sprintf("^\\s*-\\s*%s", pkg), "", hit))
    expect_identical(spec, rep(supported_range, 2L), info = paste("meta.yaml:", pkg))
  }
})

test_that("DESCRIPTION SystemRequirements gives libcudf an upper bound", {
  skip_if_no_pkg_file("DESCRIPTION")
  sysreq <- read.dcf(pkg_file("DESCRIPTION"), fields = "SystemRequirements")[[1]]
  sysreq <- gsub("\\s+", " ", sysreq)

  expect_match(sysreq, "libcudf[^,]*>=\\s*25\\.12")
  expect_match(sysreq, "libcudf[^)]*<\\s*26\\.02")
})

test_that("the README requirements table states both bounds for libcudf", {
  skip_if_no_pkg_file("README.md")
  lines <- readLines(pkg_file("README.md"), warn = FALSE)
  row <- grep("^\\|\\s*RAPIDS libcudf\\s*\\|", lines, value = TRUE)

  expect_length(row, 1L)
  expect_match(row, "25\\.12")
  expect_match(row, "26\\.02")
  expect_no_match(row, "or newer")
})

test_that("configure never states an open-ended libcudf requirement", {
  skip_if_no_pkg_file("configure")
  lines <- readLines(pkg_file("configure"))
  open_ended <- grep(">=\\s*25\\.12", lines, value = TRUE)
  open_ended <- open_ended[!grepl("26\\.02", open_ended)]

  expect_identical(open_ended, character())
})

# ---------------------------------------------------------------------------
# Deprecated RMM headers
# ---------------------------------------------------------------------------

test_that("no C++ source includes a deprecated rmm/mr/device/ forwarding header", {
  skip_if_not(dir.exists(pkg_file("src")), message = "src/ not available in this test context")
  files <- list.files(
    pkg_file("src"),
    pattern = "\\.(c|cc|cpp|cu|cuh|h|hpp)$",
    recursive = TRUE,
    full.names = TRUE
  )
  expect_gt(length(files), 0L)

  offenders <- character()
  for (f in files) {
    lines <- readLines(f, warn = FALSE)
    hit <- grep("#\\s*include\\s*[<\"]rmm/mr/device/", lines)
    if (length(hit) > 0L) {
      offenders <- c(offenders, sprintf("%s:%d", basename(f), hit))
    }
  }

  expect_identical(offenders, character())
})

# ---------------------------------------------------------------------------
# configure's libcudf version check
# ---------------------------------------------------------------------------

skip_if_no_configure <- function() {
  skip_on_os("windows")
  skip_if_no_pkg_file("configure")
  skip_if_no_pkg_file("src", "Makevars.in")
  skip_if(Sys.which("bash") == "", message = "bash not available")
}

touch <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  file.create(path)
}

fake_nvcc <- function(path, release = "12.9") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c(
      "#!/usr/bin/env bash",
      sprintf("echo 'Cuda compilation tools, release %s, V%s.0'", release, release)
    ),
    path
  )
  Sys.chmod(path, "0755")
}

# A conda/pixi environment holding CUDA and libcudf. `cudf_version` is
# c(major, minor, patch) as strings, written to include/cudf/version_config.hpp
# in the layout libcudf ships; NULL leaves the header out.
fake_prefix <- function(root, cudf_version = c("25", "12", "0")) {
  touch(file.path(root, "targets", "x86_64-linux", "include", "cuda.h"))
  touch(file.path(root, "targets", "x86_64-linux", "lib", "stubs", "libcuda.so"))
  fake_nvcc(file.path(root, "bin", "nvcc"))
  touch(file.path(root, "lib", "libcudart.so"))
  touch(file.path(root, "include", "cudf", "types.hpp"))
  touch(file.path(root, "lib", "libcudf.so"))
  if (!is.null(cudf_version)) {
    writeLines(
      c(
        "/*",
        " * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.",
        " * SPDX-License-Identifier: Apache-2.0",
        " */",
        "#pragma once",
        "",
        sprintf("#define CUDF_VERSION_MAJOR %s", cudf_version[[1]]),
        sprintf("#define CUDF_VERSION_MINOR %s", cudf_version[[2]]),
        sprintf("#define CUDF_VERSION_PATCH %s", cudf_version[[3]])
      ),
      file.path(root, "include", "cudf", "version_config.hpp")
    )
  }
  root
}

# Runs configure in a scratch package copy with the fake prefix as the conda
# environment. Returns the exit status, the combined output and whether
# src/Makevars was generated.
run_configure_with <- function(prefix) {
  pkg <- withr::local_tempdir("cuplyr-configure-", .local_envir = parent.frame())
  dir.create(file.path(pkg, "src"))
  file.copy(pkg_file("configure"), pkg)
  file.copy(pkg_file("src", "Makevars.in"), file.path(pkg, "src"))

  args <- c(
    "-u", "CUDF_HOME", "-u", "CUDA_HOME",
    sprintf("CONDA_PREFIX=%s", prefix),
    "CUPLYR_ALLOW_CUDA_STUBS=1",
    "bash", "configure"
  )
  output <- withr::with_dir(pkg, suppressWarnings(
    system2("env", args, stdout = TRUE, stderr = TRUE)
  ))
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(
    status = as.integer(status),
    output = paste(output, collapse = "\n"),
    makevars = file.exists(file.path(pkg, "src", "Makevars"))
  )
}

# Shell errors that would mean the check crashed instead of reporting.
shell_error <- "syntax error|value too great for base|integer expression expected|unbound variable|command not found"

expect_rejected <- function(res, found) {
  expect_false(res$status == 0L, info = res$output)
  expect_false(res$makevars, info = res$output)
  expect_match(res$output, "libcudf", info = res$output)
  expect_match(res$output, found, info = res$output)
  expect_match(res$output, "25\\.12", info = res$output)
  expect_match(res$output, "26\\.02", info = res$output)
  expect_no_match(res$output, shell_error)
}

expect_accepted <- function(res) {
  expect_identical(res$status, 0L, info = res$output)
  expect_true(res$makevars, info = res$output)
  expect_no_match(res$output, shell_error)
}

test_that("configure accepts libcudf 25.12", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0"))

  res <- run_configure_with(prefix)

  expect_accepted(res)
  # configure reports the version it found (the generated Makevars it echoes
  # also mentions 25.12, but never on a line naming libcudf)
  expect_match(res$output, "libcudf[^\n]*25\\.12", info = res$output)
  expect_no_match(res$output, "WARNING[^\n]*version", ignore.case = TRUE)
})

test_that("configure accepts a 25.12 patch release", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "1"))

  res <- run_configure_with(prefix)

  expect_accepted(res)
})

test_that("configure rejects libcudf 26.06 early with a clear message", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "06", "0"))

  res <- run_configure_with(prefix)

  expect_rejected(res, "26\\.0?6")
})

test_that("configure rejects libcudf 26.02, the first release without the old RMM headers", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "2", "0"))

  res <- run_configure_with(prefix)

  expect_rejected(res, "26\\.0?2")
})

test_that("configure reads zero-padded minor versions without shell arithmetic errors", {
  skip_if_no_configure()
  # "08" and "09" are invalid octal in shell arithmetic.
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "08", "0"))

  res <- run_configure_with(prefix)

  expect_rejected(res, "26\\.0?8")
})

test_that("configure rejects a libcudf older than 25.12", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("25", "10", "0"))

  res <- run_configure_with(prefix)

  expect_rejected(res, "25\\.10")
})

test_that("configure rejects a far newer libcudf (major bump)", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("27", "0", "0"))

  res <- run_configure_with(prefix)

  expect_rejected(res, "27\\.0")
})

test_that("configure warns and continues when the libcudf version header is missing", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), NULL)

  res <- run_configure_with(prefix)

  expect_accepted(res)
  expect_match(res$output, "WARNING", info = res$output)
  expect_match(res$output, "version", ignore.case = TRUE, info = res$output)
  expect_match(res$output, "version_config\\.hpp", info = res$output)
})

test_that("configure warns and continues when the version header is unparseable", {
  skip_if_no_configure()
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), NULL)
  writeLines("#pragma once", file.path(prefix, "include", "cudf", "version_config.hpp"))

  res <- run_configure_with(prefix)

  expect_accepted(res)
  expect_match(res$output, "WARNING", info = res$output)
  expect_match(res$output, "version", ignore.case = TRUE, info = res$output)
})
