# cuplyr builds against RAPIDS 25.12 only: 26.02 removed the
# rmm/mr/device/*.hpp forwarding headers and RMM's legacy allocate(bytes,
# stream) interface, and 26.06 removed rmm::mr::device_memory_resource. The declared ranges must say so, and
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

test_that("README and the articles never state an open-ended RAPIDS requirement", {
  docs <- c(
    pkg_file("README.md"),
    list.files(pkg_file("vignettes"), pattern = "\\.(Rmd|qmd|md)$", full.names = TRUE)
  )
  docs <- docs[file.exists(docs)]
  skip_if(length(docs) == 0L, message = "README/vignettes not available in this test context")

  open_ended <- character()
  for (f in docs) {
    lines <- readLines(f, warn = FALSE)
    hit <- grep(
      "(cudf|rapids)[^\n]*25\\.12[^\n]*(or later|or newer|and later|and newer|\\+)",
      lines,
      ignore.case = TRUE
    )
    hit <- hit[!grepl("26\\.02", lines[hit])]
    if (length(hit) > 0L) {
      open_ended <- c(open_ended, sprintf("%s:%d: %s", basename(f), hit, lines[hit]))
    }
  }

  expect_identical(open_ended, character())
})

test_that("the getting-started article's RAPIDS requirement names both bounds", {
  skip_if_no_pkg_file("vignettes", "getting-started.Rmd")
  lines <- readLines(pkg_file("vignettes", "getting-started.Rmd"), warn = FALSE)
  row <- grep("^\\s*[-*]\\s*RAPIDS", lines, value = TRUE)

  expect_length(row, 1L)
  expect_match(row, "25\\.12")
  expect_match(row, "26\\.02")
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
# environment. Returns the exit status, the combined output (only stderr with
# `stderr_only = TRUE`) and whether src/Makevars was generated.
run_configure_with <- function(prefix, cudf_home = NULL, stderr_only = FALSE) {
  pkg <- withr::local_tempdir("cuplyr-configure-", .local_envir = parent.frame())
  dir.create(file.path(pkg, "src"))
  file.copy(pkg_file("configure"), pkg)
  file.copy(pkg_file("src", "Makevars.in"), file.path(pkg, "src"))

  # env's -u options must come before its NAME=value assignments
  args <- c(
    if (is.null(cudf_home)) c("-u", "CUDF_HOME"),
    "-u", "CUDA_HOME",
    if (!is.null(cudf_home)) sprintf("CUDF_HOME=%s", cudf_home),
    sprintf("CONDA_PREFIX=%s", prefix),
    "CUPLYR_ALLOW_CUDA_STUBS=1",
    # system2() pastes its args into a shell command line, so quote the script
    if (stderr_only) c("bash", "-c", shQuote("bash configure 2>&1 1>/dev/null")) else c("bash", "configure")
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

# Any conda/mamba command configure prints must install the supported series
# with the recipe install.sh and install_cuplyr() use: pinned to 25.12, CUDA 12
# builds, and no channel priority (the pinned solve fails under strict channel
# priority, see #47). The cuda-version spec must be quoted for the shell.
expect_pinned_install_hint <- function(output) {
  lines <- strsplit(output, "\n", fixed = TRUE)[[1]]
  hint <- grep("(mamba|conda)\\s+install", lines, value = TRUE)
  expect_gt(length(hint), 0L)
  for (h in hint) {
    expect_match(h, "libcudf=25\\.12", info = h)
    expect_match(h, "librmm=25\\.12", info = h)
    expect_match(h, "libkvikio=25\\.12", info = h)
    expect_match(h, "--no-channel-priority", fixed = TRUE, info = h)
    expect_match(h, "-c rapidsai", fixed = TRUE, info = h)
    expect_match(h, "-c conda-forge", fixed = TRUE, info = h)
    expect_match(h, "-c nvidia", fixed = TRUE, info = h)
    expect_match(h, "[\"']cuda-version>=12\\.2,<13[\"']", info = h)
  }
}

expect_rejected <- function(res, found) {
  expect_false(res$status == 0L, info = res$output)
  expect_false(res$makevars, info = res$output)
  expect_match(res$output, "libcudf", info = res$output)
  expect_match(res$output, found, info = res$output)
  expect_match(res$output, "25\\.12", info = res$output)
  expect_match(res$output, "26\\.02", info = res$output)
  expect_no_match(res$output, shell_error)
  expect_pinned_install_hint(res$output)
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

test_that("configure rejects an older libcudf whose minor has one digit", {
  skip_if_no_configure()
  # CMake writes 25.08 as MINOR 8; a lexical comparison would accept "25.8".
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("25", "8", "0"))

  res <- run_configure_with(prefix)

  expect_rejected(res, "25\\.0?8")
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
  expect_match(res$output, "version_config\\.hpp", info = res$output)
})

test_that("configure's libcudf-not-found hint installs the supported series", {
  skip_if_no_configure()
  # CUDA but no libcudf in the environment
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"))
  unlink(file.path(prefix, "include", "cudf"), recursive = TRUE)

  res <- run_configure_with(prefix)
  skip_if(
    grepl("Found libcudf at", res$output),
    message = "this machine has a system libcudf configure finds anyway"
  )

  expect_false(res$status == 0L, info = res$output)
  expect_match(res$output, "libcudf not found", info = res$output)
  expect_pinned_install_hint(res$output)
})

# ---------------------------------------------------------------------------
# The rejection's CUDF_HOME note and other remedies
# ---------------------------------------------------------------------------

# The rejection message after its ERROR line.
after_error <- function(output) {
  lines <- strsplit(output, "\n", fixed = TRUE)[[1]]
  at <- grep("^ERROR", lines)
  if (length(at) == 0L) {
    return("")
  }
  paste(lines[(at[[1]] + 1L):length(lines)], collapse = "\n")
}

# When CUDF_HOME supplied the rejected prefix (and it isn't the active env),
# installing into the active env won't help, so the message says so and names
# the variable with its value ("CUDF_HOME=<value>"). In every other case that
# note would be false, so the value must not appear after the ERROR line.
expect_cudf_home_note <- function(res, cudf_home, present) {
  note <- paste0("CUDF_HOME=", cudf_home)
  tail <- after_error(res$output)
  if (present) {
    expect_true(grepl(note, tail, fixed = TRUE), info = res$output)
  } else {
    expect_false(grepl(note, tail, fixed = TRUE), info = res$output)
  }
}

test_that("a rejected libcudf from CUDF_HOME tells the user CUDF_HOME wins", {
  skip_if_no_configure()
  # the active env has a supported libcudf; CUDF_HOME points elsewhere at 26.06
  env <- fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0"))
  cudf_home <- fake_prefix(withr::local_tempdir("cudf-home-"), c("26", "06", "0"))

  res <- run_configure_with(env, cudf_home = cudf_home)

  expect_rejected(res, "26\\.0?6")
  expect_match(res$output, paste0("Found libcudf at: ", cudf_home), fixed = TRUE)
  expect_cudf_home_note(res, cudf_home, present = TRUE)
})

test_that("a stale pip CUDF_HOME that configure skipped gets no CUDF_HOME note", {
  skip_if_no_configure()
  env <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "06", "0"))
  stale <- file.path(
    withr::local_tempdir("gone-"),
    "lib", "python3.12", "site-packages", "libcudf"
  )

  res <- run_configure_with(env, cudf_home = stale)

  expect_rejected(res, "26\\.0?6")
  expect_match(res$output, paste0("Found libcudf at: ", env), fixed = TRUE)
  expect_cudf_home_note(res, stale, present = FALSE)
})

test_that("CUDF_HOME equal to the active env gets no CUDF_HOME note", {
  skip_if_no_configure()
  # as inst/docker/Dockerfile does: CUDF_HOME=/opt/conda, the active env
  env <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "06", "0"))

  res <- run_configure_with(env, cudf_home = env)

  expect_rejected(res, "26\\.0?6")
  expect_cudf_home_note(res, env, present = FALSE)
})

test_that("the rejection never sends the user to an install.sh run that would skip RAPIDS", {
  skip_if_no_configure()
  # install.sh reuses an active env that already has libcudf, so it only helps
  # with a fresh --conda-prefix
  env <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "06", "0"))

  res <- run_configure_with(env)

  expect_rejected(res, "26\\.0?6")
  lines <- strsplit(res$output, "\n", fixed = TRUE)[[1]]
  for (l in grep("install\\.sh", lines, value = TRUE)) {
    expect_match(l, "--conda-prefix", fixed = TRUE, info = l)
  }
})

# ---------------------------------------------------------------------------
# The rejection reaches users whose tooling hides configure's stdout
# ---------------------------------------------------------------------------

test_that("configure prints the rejection on stderr", {
  skip_if_no_configure()
  # install.sh without --verbose discards configure's stdout and keeps stderr
  prefix <- fake_prefix(withr::local_tempdir("conda-env-"), c("26", "06", "0"))

  res <- run_configure_with(prefix, stderr_only = TRUE)

  expect_rejected(res, "26\\.0?6")
  expect_match(res$output, "ERROR", info = res$output)
})

test_that("install.sh without --verbose shows why configure rejected libcudf", {
  skip_if_no_configure()
  skip_if_no_pkg_file("install.sh")

  # the issue's case: an active conda env that resolved a newer RAPIDS
  env_dir <- fake_prefix(withr::local_tempdir("user-env-"), c("26", "06", "0"))
  dir.create(file.path(env_dir, "conda-meta"))

  # fake mamba records calls; fake R passes install.sh's dependency check
  bin <- withr::local_tempdir("fake-bin-")
  mamba_log <- file.path(bin, "mamba.log")
  writeLines(
    c("#!/usr/bin/env bash", sprintf("echo \"$*\" >> %s", shQuote(mamba_log)), "exit 0"),
    file.path(bin, "mamba")
  )
  writeLines(
    c(
      "#!/usr/bin/env bash",
      "if [ \"$1\" = CMD ]; then echo 'FAKE-BUILD-RAN' >&2; exit 1; fi",
      "exit 0"
    ),
    file.path(bin, "R")
  )
  Sys.chmod(file.path(bin, c("mamba", "R")), "0755")

  # a scratch source tree holding this configure
  src <- withr::local_tempdir("cuplyr-src-")
  dir.create(file.path(src, "src"))
  file.copy(pkg_file("configure"), src)
  file.copy(pkg_file("src", "Makevars.in"), file.path(src, "src"))
  file.create(file.path(src, "DESCRIPTION"))

  args <- c(
    "-u", "CUDA_HOME", "-u", "CUDF_HOME",
    paste0("HOME=", withr::local_tempdir("fake-home-")),
    paste0("PATH=", paste(c(bin, "/usr/bin", "/bin"), collapse = ":")),
    paste0("CONDA_PREFIX=", env_dir),
    "CONDA_DEFAULT_ENV=myenv",
    "CUPLYR_ALLOW_CUDA_STUBS=1",
    "bash", pkg_file("install.sh")
  )
  output <- withr::with_dir(src, suppressWarnings(
    system2("env", args, stdout = TRUE, stderr = TRUE)
  ))
  status <- attr(output, "status")
  output <- paste(output, collapse = "\n")

  skip_if_not(
    grepl("Configuring cuplyr", output, fixed = TRUE),
    message = "install.sh did not reach configure with this fake setup"
  )
  expect_false(is.null(status) || status == 0L, info = output)
  expect_no_match(output, "FAKE-BUILD-RAN", fixed = TRUE)
  expect_match(output, "libcudf[^\n]*26\\.0?6", info = output)
  expect_match(output, "26\\.02", info = output)
  expect_pinned_install_hint(output)
})
