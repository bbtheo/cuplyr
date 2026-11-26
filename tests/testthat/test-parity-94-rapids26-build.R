# #94: cuplyr builds against RAPIDS 26.02 and later, and no longer against
# 25.12.
#
# Contract:
# - Supported libcudf range everywhere it is declared (configure, DESCRIPTION,
#   the install docs): >= 26.02, no upper bound. (A conda binary is pinned to
#   the series it was built against; test-parity-135-conda-recipe.R.) 25.12 and
#   older are rejected by configure before the compiler runs.
# - Development and the gates run on RAPIDS 26.10: pixi.toml's default
#   environment pins the 26.10 series (CUDA 13), and a second environment,
#   `rapids2602`, pins the declared minimum 26.02.
# - The conda installers (install.sh, install_cuplyr()) and every install hint
#   configure prints install the tested series, 26.10 on CUDA 13, with no
#   channel priority.
# - The C++ uses RMM's resource_ref model only: no legacy MR interface
#   (-DRMM_ENABLE_LEGACY_MR_INTERFACE), no rmm::mr::device_memory_resource, no
#   pointer-returning get_current_device_resource().
# - check_deps() reads the libcudf version it finds and fails below 26.02;
#   diagnostics() warns when the loaded libcudf is older than 26.02.
# - The tools run in a selectable pixi environment: CUPLYR_PIXI_MANIFEST /
#   CUPLYR_PIXI_ENV, or a worktree's untracked .cuplyr-env file; variables in
#   the caller's environment win over the file; the default is the main
#   checkout's pixi.toml and its default environment.
#
# Only the diagnostics test needs a GPU. The configure tests run configure
# against fake prefixes in a scratch copy of the package.

p94_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p94_file <- function(...) file.path(p94_root(), ...)

p94_skip_if_no_file <- function(...) {
  skip_if_not(file.exists(p94_file(...)),
              message = paste(file.path(...), "not available in this test context"))
}

# The lines of a C/C++ source with comments removed (good enough for grepping
# identifiers: // comments, and block-comment lines starting with /* or *).
p94_code_lines <- function(f) {
  lines <- readLines(f, warn = FALSE)
  lines <- lines[!grepl("^\\s*(/\\*|\\*)", lines)]
  sub("//.*$", "", lines)
}

# The same, after `filter(f, lines)` blanked exempt regions (it sees the raw
# lines, comments included, so it can find marker comments).
p94_code_lines_keep <- function(f, filter) {
  lines <- filter(f, readLines(f, warn = FALSE))
  lines[grepl("^\\s*(/\\*|\\*)", lines)] <- ""
  sub("//.*$", "", lines)
}

# ---------------------------------------------------------------------------
# Declared ranges
# ---------------------------------------------------------------------------

test_that("DESCRIPTION requires libcudf >= 26.02 with no upper bound", {
  p94_skip_if_no_file("DESCRIPTION")
  sysreq <- read.dcf(p94_file("DESCRIPTION"), fields = "SystemRequirements")[[1]]
  sysreq <- gsub("\\s+", " ", sysreq)

  expect_match(sysreq, "libcudf \\(>= ?26\\.02\\)")
  expect_no_match(sysreq, "25\\.12")
  expect_no_match(sysreq, "libcudf[^)]*<")
})

# conda/recipe.yaml: see test-parity-135-conda-recipe.R
# (each binary is pinned to the series it is built against, and the recipe
# builds nothing older than configure's minimum).

test_that("pixi.toml pins 26.10 for the default environment and 26.02 for rapids2602", {
  p94_skip_if_no_file("pixi.toml")
  lines <- readLines(p94_file("pixi.toml"))
  specs <- function(pkg) {
    hit <- grep(sprintf("^\\s*%s\\s*=", pkg), lines, value = TRUE)
    gsub("\\s|\"|'", "", sub("^[^=]*=", "", hit))
  }

  for (pkg in c("libcudf", "librmm", "libkvikio")) {
    expect_setequal(specs(pkg), c("26.10.*", "26.02.*"))
  }
  expect_true(all(specs("cuda-version") == "13.*"), info = paste(specs("cuda-version"), collapse = ", "))
  expect_gt(length(specs("cuda-version")), 0L)

  # an environment named rapids2602 is declared
  env_start <- grep("^\\s*\\[environments\\]", lines)
  expect_length(env_start, 1L)
  expect_true(any(grepl("^\\s*rapids2602\\s*=", lines[seq(env_start, length(lines))])))
  # nothing still names 25.12
  expect_no_match(paste(lines, collapse = "\n"), "25\\.12")
})

test_that("the README requirements table names 26.02 as the minimum", {
  p94_skip_if_no_file("README.md")
  lines <- readLines(p94_file("README.md"), warn = FALSE)
  row <- grep("^\\|\\s*RAPIDS libcudf\\s*\\|", lines, value = TRUE)

  expect_length(row, 1L)
  expect_match(row, "26\\.02")
  expect_no_match(row, "25\\.12")
  expect_no_match(row, "not supported", ignore.case = TRUE)
})

test_that("the getting-started article's RAPIDS requirement names 26.02 as the minimum", {
  p94_skip_if_no_file("vignettes", "getting-started.Rmd")
  lines <- readLines(p94_file("vignettes", "getting-started.Rmd"), warn = FALSE)
  row <- grep("^\\s*[-*]\\s*RAPIDS", lines, value = TRUE)

  expect_length(row, 1L)
  expect_match(row, "26\\.02")
  expect_no_match(row, "25\\.12")
})

test_that("NEWS records that RAPIDS 25.12 is no longer supported", {
  p94_skip_if_no_file("NEWS.md")
  lines <- readLines(p94_file("NEWS.md"), warn = FALSE)
  heads <- grep("^# ", lines)
  expect_gt(length(heads), 0L)
  dev <- lines[seq(heads[1], if (length(heads) > 1L) heads[2] - 1L else length(lines))]
  expect_match(dev[1], "development version", ignore.case = TRUE)

  hit <- grep("25\\.12", dev, value = TRUE)
  expect_gt(length(hit), 0L)
  expect_true(any(grepl("26\\.02", hit) & grepl("#94", hit, fixed = TRUE)),
              info = paste(hit, collapse = "\n"))
})

# ---------------------------------------------------------------------------
# The C++ uses the resource_ref model only
# ---------------------------------------------------------------------------

test_that("no build file turns on RMM's legacy memory-resource interface", {
  for (f in c("configure", file.path("src", "Makevars.in"))) {
    p94_skip_if_no_file(f)
    lines <- readLines(p94_file(f), warn = FALSE)
    lines <- lines[!grepl("^\\s*#", lines)]
    expect_false(any(grepl("RMM_ENABLE_LEGACY_MR_INTERFACE", lines, fixed = TRUE)), info = f)
  }
})

test_that("no C++ source uses rmm::mr::device_memory_resource or get_current_device_resource()", {
  skip_if_not(dir.exists(p94_file("src")), message = "src/ not available in this test context")
  files <- list.files(p94_file("src"), pattern = "\\.(c|cc|cpp|cu|cuh|h|hpp)$",
                      recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("RcppExports", files)]
  expect_gt(length(files), 0L)

  # One exemption: libcudf 26.02's cudf::get_current_device_resource_ref()
  # returns RMM's legacy pointer-map resource, so on 26.02 cuplyr's resource must
  # also be installed there. That bridge lives in src/cuplyr_mr.hpp between the
  # marker comments `RMM 26.02 legacy-map bridge: begin` / `: end`, inside
  # `#if !CUPLYR_RMM_NEW_API` (the 26.10 build, which has no
  # device_memory_resource at all, would fail if it leaked out).
  drop_bridge <- function(f, lines) {
    if (basename(f) != "cuplyr_mr.hpp") return(lines)
    b <- grep("RMM 26.02 legacy-map bridge: begin", lines, fixed = TRUE)
    e <- grep("RMM 26.02 legacy-map bridge: end", lines, fixed = TRUE)
    if (length(b) != length(e) || length(b) == 0L) return(lines)
    keep <- rep(TRUE, length(lines))
    for (i in seq_along(b)) {
      expect_lt(b[[i]], e[[i]])
      keep[b[[i]]:e[[i]]] <- FALSE
    }
    lines[!keep] <- ""
    lines
  }

  offenders <- character()
  for (f in files) {
    code <- p94_code_lines_keep(f, drop_bridge)
    hit <- grep(paste0("device_memory_resource|",
                       "get_current_device_resource\\s*\\(|",
                       "RMM_ENABLE_LEGACY_MR_INTERFACE"), code)
    if (length(hit) > 0L) {
      offenders <- c(offenders, sprintf("%s:%d: %s", basename(f), hit, trimws(code[hit])))
    }
  }
  expect_identical(offenders, character())
})

# ---------------------------------------------------------------------------
# configure's libcudf version check
# ---------------------------------------------------------------------------

p94_skip_if_no_configure <- function() {
  skip_on_os("windows")
  p94_skip_if_no_file("configure")
  p94_skip_if_no_file("src", "Makevars.in")
  skip_if(Sys.which("bash") == "", message = "bash not available")
}

p94_touch <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  file.create(path)
}

# A conda environment holding CUDA (nvcc `release`) and libcudf
# `cudf_version` = c(major, minor, patch) as strings, in the layout libcudf
# ships include/cudf/version_config.hpp.
p94_fake_prefix <- function(root, cudf_version, release = "13.4") {
  p94_touch(file.path(root, "targets", "x86_64-linux", "include", "cuda.h"))
  p94_touch(file.path(root, "targets", "x86_64-linux", "lib", "stubs", "libcuda.so"))
  nvcc <- file.path(root, "bin", "nvcc")
  dir.create(dirname(nvcc), recursive = TRUE, showWarnings = FALSE)
  writeLines(c("#!/usr/bin/env bash",
               sprintf("echo 'Cuda compilation tools, release %s, V%s.0'", release, release)),
             nvcc)
  Sys.chmod(nvcc, "0755")
  p94_touch(file.path(root, "lib", "libcudart.so"))
  p94_touch(file.path(root, "include", "cudf", "types.hpp"))
  p94_touch(file.path(root, "lib", "libcudf.so"))
  writeLines(
    c("#pragma once",
      sprintf("#define CUDF_VERSION_MAJOR %s", cudf_version[[1]]),
      sprintf("#define CUDF_VERSION_MINOR %s", cudf_version[[2]]),
      sprintf("#define CUDF_VERSION_PATCH %s", cudf_version[[3]])),
    file.path(root, "include", "cudf", "version_config.hpp")
  )
  root
}

# configure in a scratch package copy, with `prefix` as the active conda
# environment and no CUDF_HOME / CUDA_HOME.
p94_run_configure <- function(prefix) {
  pkg <- withr::local_tempdir("cuplyr-configure-", .local_envir = parent.frame())
  dir.create(file.path(pkg, "src"))
  file.copy(p94_file("configure"), pkg)
  file.copy(p94_file("src", "Makevars.in"), file.path(pkg, "src"))
  args <- c("-u", "CUDF_HOME", "-u", "CUDA_HOME",
            sprintf("CONDA_PREFIX=%s", prefix), "CUPLYR_ALLOW_CUDA_STUBS=1",
            "bash", "configure")
  output <- withr::with_dir(pkg, suppressWarnings(
    system2("env", args, stdout = TRUE, stderr = TRUE)
  ))
  status <- attr(output, "status")
  makevars <- file.path(pkg, "src", "Makevars")
  list(
    status = if (is.null(status)) 0L else as.integer(status),
    output = paste(output, collapse = "\n"),
    makevars = if (file.exists(makevars)) readLines(makevars) else NULL
  )
}

p94_shell_error <- "syntax error|value too great for base|integer expression expected|unbound variable|command not found"

p94_expect_accepted <- function(res) {
  expect_identical(res$status, 0L, info = res$output)
  expect_false(is.null(res$makevars), info = res$output)
  expect_no_match(res$output, p94_shell_error)
  expect_no_match(res$output, "WARNING[^\n]*version", ignore.case = TRUE)
  # the generated Makevars never turns the legacy interface on
  expect_false(any(grepl("RMM_ENABLE_LEGACY_MR_INTERFACE", res$makevars, fixed = TRUE)))
}

# Every conda/mamba command configure prints installs the tested series: 26.10
# on CUDA 13, with no channel priority, the spec quoted for the shell.
p94_expect_install_hint <- function(output) {
  lines <- strsplit(output, "\n", fixed = TRUE)[[1]]
  hint <- grep("(mamba|conda)\\s+install", lines, value = TRUE)
  expect_gt(length(hint), 0L)
  for (h in hint) {
    expect_match(h, "libcudf=26\\.10", info = h)
    expect_match(h, "librmm=26\\.10", info = h)
    expect_match(h, "libkvikio=26\\.10", info = h)
    expect_match(h, "--no-channel-priority", fixed = TRUE, info = h)
    expect_match(h, "[\"']cuda-version>=13,<14[\"']", info = h)
    expect_no_match(h, "25\\.12", info = h)
  }
}

p94_expect_rejected <- function(res, found) {
  expect_false(res$status == 0L, info = res$output)
  expect_null(res$makevars)
  expect_match(res$output, "libcudf", info = res$output)
  expect_match(res$output, found, info = res$output)
  expect_match(res$output, "26\\.02", info = res$output)
  expect_no_match(res$output, p94_shell_error)
  p94_expect_install_hint(res$output)
}

test_that("configure accepts libcudf 26.02, the declared minimum", {
  p94_skip_if_no_configure()
  for (v in list(c("26", "02", "0"), c("26", "2", "1"))) {
    res <- p94_run_configure(p94_fake_prefix(withr::local_tempdir("conda-env-"), v))
    p94_expect_accepted(res)
    expect_match(res$output, "libcudf[^\n]*26\\.02", info = res$output)
  }
})

test_that("configure accepts libcudf 26.10 and later releases (no upper bound)", {
  p94_skip_if_no_configure()
  for (v in list(c("26", "10", "2"), c("26", "08", "0"), c("27", "02", "0"))) {
    res <- p94_run_configure(p94_fake_prefix(withr::local_tempdir("conda-env-"), v))
    p94_expect_accepted(res)
  }
})

test_that("configure rejects libcudf 25.12 and older with a message naming 26.02", {
  p94_skip_if_no_configure()
  res <- p94_run_configure(p94_fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0")))
  p94_expect_rejected(res, "25\\.12")

  res <- p94_run_configure(p94_fake_prefix(withr::local_tempdir("conda-env-"), c("25", "8", "0")))
  p94_expect_rejected(res, "25\\.0?8")
})

# ---------------------------------------------------------------------------
# The conda installers install the tested series
# ---------------------------------------------------------------------------

test_that("install.sh installs RAPIDS 26.10 on CUDA 13", {
  p94_skip_if_no_file("install.sh")
  txt <- paste(readLines(p94_file("install.sh"), warn = FALSE), collapse = "\n")
  expect_match(txt, "libcudf=26.10", fixed = TRUE)
  expect_match(txt, "cuda-version>=13,<14", fixed = TRUE)
  expect_no_match(txt, "libcudf=25\\.12")
  expect_no_match(txt, "cuda-version>=12\\.2,<13")
})

test_that("install_cuplyr()'s conda route installs RAPIDS 26.10 on CUDA 13", {
  body <- paste(deparse(install_via_conda), collapse = "\n")
  expect_match(body, "libcudf=26.10", fixed = TRUE)
  expect_match(body, "cuda-version>=13,<14", fixed = TRUE)
  expect_no_match(body, "25\\.12")
})

# ---------------------------------------------------------------------------
# check_deps() and diagnostics() know the minimum
# ---------------------------------------------------------------------------

test_that("check_deps()'s libcudf check fails below 26.02 and reports the version", {
  old <- p94_fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0"))
  res <- find_libcudf_in(old)
  expect_false(res$ok)
  expect_identical(res$version, "25.12.0")
  expect_match(res$message, "25.12.0", fixed = TRUE)
  expect_match(res$message, "26.02", fixed = TRUE)

  for (v in list(c("26", "02", "1"), c("26", "10", "2"))) {
    ok <- p94_fake_prefix(withr::local_tempdir("conda-env-"), v)
    res <- find_libcudf_in(ok)
    expect_true(res$ok, info = res$message)
    want <- sprintf("%s.%02d.%s", v[[1]], as.integer(v[[2]]), v[[3]])
    expect_identical(res$version, want)
    expect_match(res$message, want, fixed = TRUE)
  }
})

test_that("check_deps()'s libcudf check of the loaded library uses the same minimum", {
  old <- p94_fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0"))
  local_mocked_bindings(loaded_libcudf_path = function(...) file.path(old, "lib", "libcudf.so"))
  res <- check_libcudf()
  expect_false(res$ok)
  expect_identical(res$version, "25.12.0")
  expect_match(res$message, "26.02", fixed = TRUE)

  new <- p94_fake_prefix(withr::local_tempdir("conda-env-"), c("26", "10", "2"))
  local_mocked_bindings(loaded_libcudf_path = function(...) file.path(new, "lib", "libcudf.so"))
  res <- check_libcudf()
  expect_true(res$ok, info = res$message)
  expect_identical(res$version, "26.10.2")
})

test_that("diagnostics() warns when the loaded libcudf is older than 26.02", {
  skip_if_no_gpu()
  old <- p94_fake_prefix(withr::local_tempdir("conda-env-"), c("25", "12", "0"))
  local_mocked_bindings(loaded_libcudf_path = function(...) file.path(old, "lib", "libcudf.so"))
  txt <- paste(utils::capture.output(diagnostics(redact = FALSE)), collapse = "\n")
  expect_match(txt, "WARNING[^\n]*26\\.02")

  new <- p94_fake_prefix(withr::local_tempdir("conda-env-"), c("26", "10", "2"))
  local_mocked_bindings(loaded_libcudf_path = function(...) file.path(new, "lib", "libcudf.so"))
  txt <- paste(utils::capture.output(diagnostics(redact = FALSE)), collapse = "\n")
  expect_no_match(txt, "WARNING[^\n]*26\\.02")
})

# ---------------------------------------------------------------------------
# The tools' environment selector (tools/lib.sh)
# ---------------------------------------------------------------------------

# A scratch git checkout holding a copy of tools/lib.sh, and a stub `pixi` on
# PATH that records how `pixi shell-hook` was called. Returns a function that
# sources lib.sh in a clean bash with extra environment `env`, runs
# cuplyr_activate_env and returns the recorded arguments.
p94_selector <- function() {
  skip_on_os("windows")
  p94_skip_if_no_file("tools", "lib.sh")
  skip_if(Sys.which("bash") == "" || Sys.which("git") == "", message = "bash/git not available")
  root <- withr::local_tempdir("cuplyr-libsh-", .local_envir = parent.frame())
  dir.create(file.path(root, "tools"))
  file.copy(p94_file("tools", "lib.sh"), file.path(root, "tools", "lib.sh"))
  system2("git", c("init", "-q", shQuote(root)))
  bin <- file.path(root, "fakebin")
  dir.create(bin)
  writeLines(c("#!/usr/bin/env bash",
               "printf 'export P94_ARGS=%q\\n' \"$*\""),
             file.path(bin, "pixi"))
  Sys.chmod(file.path(bin, "pixi"), "0755")
  function(env = character(), file = NULL) {
    if (!is.null(file)) writeLines(file, file.path(root, ".cuplyr-env"))
    else unlink(file.path(root, ".cuplyr-env"))
    script <- sprintf("source %s; cuplyr_activate_env; echo \"$P94_ARGS\"",
                      shQuote(file.path(root, "tools", "lib.sh")))
    args <- c("-u", "CUPLYR_ENV_ACTIVE", "-u", "CUPLYR_PIXI_ENV", "-u", "CUPLYR_PIXI_MANIFEST",
              "-u", "PIXI_PROJECT_MANIFEST", "-u", "PIXI_ENVIRONMENT_NAME",
              sprintf("PATH=%s:%s", bin, Sys.getenv("PATH")), env,
              "bash", "-c", shQuote(script))
    out <- suppressWarnings(system2("env", args, stdout = TRUE, stderr = TRUE))
    list(root = root, out = paste(out, collapse = "\n"), status = attr(out, "status"))
  }
}

p94_env_arg <- function(out) {
  m <- regmatches(out, regexpr("(-e|--environment)[ =][^ ]+", out))
  if (length(m) == 0L) "default" else sub("^(-e|--environment)[ =]", "", m)
}

p94_manifest_arg <- function(out) {
  m <- regmatches(out, regexpr("--manifest-path[ =][^ ]+", out))
  if (length(m) == 0L) NA_character_ else sub("^--manifest-path[ =]", "", m)
}

test_that("the tools default to the main checkout's pixi.toml and default environment", {
  run <- p94_selector()
  res <- run()
  expect_null(res$status)
  expect_identical(normalizePath(p94_manifest_arg(res$out), mustWork = FALSE),
                   normalizePath(file.path(res$root, "pixi.toml"), mustWork = FALSE))
  expect_identical(p94_env_arg(res$out), "default")
})

test_that("CUPLYR_PIXI_ENV and CUPLYR_PIXI_MANIFEST select the environment", {
  run <- p94_selector()
  res <- run(c("CUPLYR_PIXI_ENV=rapids2602"))
  expect_identical(p94_env_arg(res$out), "rapids2602", info = res$out)

  res <- run(c("CUPLYR_PIXI_MANIFEST=/elsewhere/pixi.toml", "CUPLYR_PIXI_ENV=rapids2610"))
  expect_identical(p94_manifest_arg(res$out), "/elsewhere/pixi.toml", info = res$out)
  expect_identical(p94_env_arg(res$out), "rapids2610", info = res$out)
})

test_that("a worktree's .cuplyr-env file selects the environment; the caller's variables win", {
  run <- p94_selector()
  file <- c("CUPLYR_PIXI_MANIFEST=/staging/pixi.toml", "CUPLYR_PIXI_ENV=rapids2610")
  res <- run(file = file)
  expect_identical(p94_manifest_arg(res$out), "/staging/pixi.toml", info = res$out)
  expect_identical(p94_env_arg(res$out), "rapids2610", info = res$out)

  res <- run(c("CUPLYR_PIXI_ENV=rapids2602"), file = file)
  expect_identical(p94_manifest_arg(res$out), "/staging/pixi.toml", info = res$out)
  expect_identical(p94_env_arg(res$out), "rapids2602", info = res$out)
})
