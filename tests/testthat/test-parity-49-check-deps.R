# check_deps(): driver version from both nvidia-smi header formats, libcudf
# found through the loaded library and the pixi env, stale CUDF_HOME /
# CONDA_PREFIX reported (#49).

old_header <- c(
  "Fri Nov 21 10:00:00 2026",
  "+-----------------------------------------------------------------------------------------+",
  "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.14      CUDA Version: 12.4     |",
  "|-----------------------------------------+------------------------+----------------------+"
)

new_header <- c(
  "Fri Nov 21 10:00:00 2026",
  "+-----------------------------------------------------------------------------------------+",
  "| NVIDIA-SMI 615.78.09    KMD Version: 615.78.08    CUDA UMD Version: 13.4 |",
  "|-----------------------------------------+------------------------+----------------------+"
)

# A cuDF prefix with headers and the shared library under `libdir`.
make_cudf_prefix <- function(prefix, libdir = "lib") {
  dir.create(file.path(prefix, "include", "cudf"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(prefix, libdir), recursive = TRUE, showWarnings = FALSE)
  file.create(file.path(prefix, "include", "cudf", "types.hpp"))
  file.create(file.path(prefix, libdir, "libcudf.so"))
  invisible(prefix)
}

same_path <- function(a, b) {
  identical(normalizePath(a, mustWork = FALSE), normalizePath(b, mustWork = FALSE))
}

# Environment with no cuDF location configured.
local_no_cudf_env <- function(.local_envir = parent.frame()) {
  withr::local_envvar(
    CUDF_HOME = NA, CONDA_PREFIX = NA,
    PIXI_PROJECT_ROOT = NA, PIXI_ENVIRONMENT_NAME = NA,
    .local_envir = .local_envir
  )
  empty <- withr::local_tempdir(.local_envir = .local_envir)
  withr::local_dir(empty, .local_envir = .local_envir)
  invisible(empty)
}

# --- driver version parsing ---------------------------------------------------

test_that("parse_nvidia_smi_driver_version() reads the old header format", {
  expect_identical(parse_nvidia_smi_driver_version(old_header), "550.54.14")
})

test_that("parse_nvidia_smi_driver_version() reads the KMD Version header format", {
  expect_identical(parse_nvidia_smi_driver_version(new_header), "615.78.08")
})

test_that("parse_nvidia_smi_driver_version() reads --query-gpu csv output", {
  expect_identical(parse_nvidia_smi_driver_version("615.78.08"), "615.78.08")
  expect_identical(parse_nvidia_smi_driver_version("  615.78.08  "), "615.78.08")
  # one line per GPU: the first is taken
  expect_identical(
    parse_nvidia_smi_driver_version(c("615.78.08", "615.78.08")),
    "615.78.08"
  )
  expect_identical(parse_nvidia_smi_driver_version("550.54"), "550.54")
})

test_that("parse_nvidia_smi_driver_version() returns NA when nothing parses", {
  expect_identical(parse_nvidia_smi_driver_version(character()), NA_character_)
  expect_identical(
    parse_nvidia_smi_driver_version(c("No devices were found", "")),
    NA_character_
  )
  expect_identical(
    parse_nvidia_smi_driver_version("Field \"driver_version\" is not a valid field to query."),
    NA_character_
  )
})

# --- check_nvidia_driver() with nvidia-smi mocked -----------------------------

test_that("check_nvidia_driver() uses the --query-gpu interface", {
  seen <- list()
  testthat::local_mocked_bindings(
    run_nvidia_smi = function(args = character()) {
      seen[[length(seen) + 1]] <<- args
      if (any(grepl("--query-gpu=driver_version", args, fixed = TRUE))) {
        "620.10.01"
      } else {
        new_header
      }
    }
  )
  res <- check_nvidia_driver()
  expect_true(res$ok)
  # the query answer is used, not the header
  expect_identical(res$value, "620.10.01")
  expect_identical(res$name, "NVIDIA Driver")
  expect_match(res$message, "620.10.01", fixed = TRUE)
  expect_true(any(vapply(seen, function(a) {
    any(grepl("--query-gpu=driver_version", a, fixed = TRUE))
  }, logical(1))))
})

test_that("check_nvidia_driver() falls back to the new header when the query fails", {
  testthat::local_mocked_bindings(
    run_nvidia_smi = function(args = character()) {
      if (length(args) > 0) NULL else new_header
    }
  )
  res <- check_nvidia_driver()
  expect_true(res$ok)
  expect_identical(res$value, "615.78.08")
  expect_no_match(res$message, "unknown")
})

test_that("check_nvidia_driver() falls back to the old header when the query output is unusable", {
  testthat::local_mocked_bindings(
    run_nvidia_smi = function(args = character()) {
      if (length(args) > 0) {
        "Field \"driver_version\" is not a valid field to query."
      } else {
        old_header
      }
    }
  )
  res <- check_nvidia_driver()
  expect_true(res$ok)
  expect_identical(res$value, "550.54.14")
})

test_that("check_nvidia_driver() fails when nvidia-smi is unavailable", {
  testthat::local_mocked_bindings(
    run_nvidia_smi = function(args = character()) NULL
  )
  res <- check_nvidia_driver()
  expect_false(res$ok)
  expect_identical(res$value, NA_character_)
  expect_match(res$message, "nvidia-smi")
})

test_that("check_nvidia_driver() still passes with an unparsable header", {
  testthat::local_mocked_bindings(
    run_nvidia_smi = function(args = character()) {
      if (length(args) > 0) NULL else c("| NVIDIA-SMI something new |")
    }
  )
  res <- check_nvidia_driver()
  expect_true(res$ok)
  expect_identical(res$value, NA_character_)
})

test_that("run_nvidia_smi() returns NULL when the command is missing", {
  withr::local_envvar(PATH = withr::local_tempdir())
  # quiet: a missing command is a check result, not a warning
  expect_no_warning(expect_null(run_nvidia_smi()))
  expect_no_warning(expect_null(
    run_nvidia_smi(c("--query-gpu=driver_version", "--format=csv,noheader"))
  ))
})

# --- loaded_libcudf_path() ----------------------------------------------------

test_that("loaded_libcudf_path() finds libcudf.so in a maps file", {
  maps <- withr::local_tempfile()
  writeLines(c(
    "55d0c0000000-55d0c0001000 r--p 00000000 103:02 1001                      /usr/lib/R/bin/exec/R",
    "7f00a0000000-7f00a1000000 r-xp 00000000 103:02 2002                      /home/u/repo/.pixi/envs/default/lib/libcudf_kafka.so",
    "7f00b0000000-7f00b1000000 r-xp 00000000 103:02 2003                      /home/u/repo/.pixi/envs/default/lib/libcudf.so",
    "7f00b1000000-7f00b2000000 r--p 01000000 103:02 2003                      /home/u/repo/.pixi/envs/default/lib/libcudf.so",
    "7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                          [stack]",
    "7ffd00100000-7ffd00102000 r-xp 00000000 00:00 0                          [vdso]"
  ), maps)
  expect_identical(
    loaded_libcudf_path(maps),
    "/home/u/repo/.pixi/envs/default/lib/libcudf.so"
  )
})

test_that("loaded_libcudf_path() accepts a versioned soname and paths with spaces", {
  maps <- withr::local_tempfile()
  writeLines(c(
    "7f00a0000000-7f00a1000000 r-xp 00000000 103:02 2002                      /opt/my envs/rapids/lib64/libcudf.so.25.12"
  ), maps)
  expect_identical(loaded_libcudf_path(maps), "/opt/my envs/rapids/lib64/libcudf.so.25.12")
})

test_that("loaded_libcudf_path() returns NA when libcudf is not mapped", {
  maps <- withr::local_tempfile()
  writeLines(c(
    "55d0c0000000-55d0c0001000 r--p 00000000 103:02 1001                      /usr/lib/R/bin/exec/R",
    "7f00a0000000-7f00a1000000 r-xp 00000000 103:02 2002                      /usr/lib/libcudf_kafka.so",
    "7f00a0000000-7f00a1000000 r-xp 00000000 103:02 2002                      /usr/lib/libcudftestutil.so"
  ), maps)
  expect_identical(loaded_libcudf_path(maps), NA_character_)
  expect_identical(
    loaded_libcudf_path(file.path(withr::local_tempdir(), "no-such-maps")),
    NA_character_
  )
})

# --- default search paths include the pixi env --------------------------------

test_that("default_cudf_search_paths() puts CUDF_HOME, CONDA_PREFIX, then pixi before system paths", {
  root <- withr::local_tempdir()
  withr::local_envvar(
    CUDF_HOME = "/x/cudf-home", CONDA_PREFIX = "/x/conda",
    PIXI_PROJECT_ROOT = root, PIXI_ENVIRONMENT_NAME = NA
  )
  paths <- default_cudf_search_paths()
  pixi <- file.path(root, ".pixi", "envs", "default")
  expect_identical(paths[1:2], c("/x/cudf-home", "/x/conda"))
  i_pixi <- which(vapply(paths, same_path, logical(1), b = pixi))
  expect_length(i_pixi, 1)
  expect_lt(i_pixi, match("/usr/local", paths))
  expect_lt(i_pixi, match("/usr", paths))
})

test_that("default_cudf_search_paths() uses PIXI_ENVIRONMENT_NAME", {
  root <- withr::local_tempdir()
  withr::local_envvar(
    CUDF_HOME = NA, CONDA_PREFIX = NA,
    PIXI_PROJECT_ROOT = root, PIXI_ENVIRONMENT_NAME = "cuda12"
  )
  paths <- default_cudf_search_paths()
  expect_true(any(vapply(paths, same_path, logical(1),
                         b = file.path(root, ".pixi", "envs", "cuda12"))))
})

test_that("default_cudf_search_paths() drops unset variables", {
  local_no_cudf_env()
  paths <- default_cudf_search_paths()
  expect_false(any(!nzchar(paths)))
  expect_false(anyNA(paths))
})

# --- check_libcudf() ----------------------------------------------------------

test_that("check_libcudf() finds a pixi env in the working directory without CONDA_PREFIX", {
  local_no_cudf_env()
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  pixi <- make_cudf_prefix(file.path(getwd(), ".pixi", "envs", "default"))

  res <- check_libcudf()
  expect_true(res$ok)
  expect_true(same_path(res$value, pixi))
})

test_that("check_libcudf() finds the pixi env named by PIXI_PROJECT_ROOT", {
  local_no_cudf_env()
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  root <- withr::local_tempdir()
  withr::local_envvar(PIXI_PROJECT_ROOT = root)
  pixi <- make_cudf_prefix(file.path(root, ".pixi", "envs", "default"), libdir = "lib64")

  res <- check_libcudf()
  expect_true(res$ok)
  expect_true(same_path(res$value, pixi))
})

test_that("check_libcudf() reports the libcudf cuplyr has loaded", {
  local_no_cudf_env()
  prefix <- withr::local_tempdir()
  lib <- file.path(prefix, "lib", "libcudf.so")
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) lib,
    default_cudf_search_paths = function() character()
  )

  res <- check_libcudf()
  expect_true(res$ok)
  expect_identical(res$name, "libcudf")
  expect_true(same_path(res$value, prefix))
  expect_match(res$message, lib, fixed = TRUE)
})

test_that("check_libcudf() reports a CUDF_HOME that does not exist, but stays ok when libcudf is loaded", {
  local_no_cudf_env()
  stale <- file.path(withr::local_tempdir(), "old-env")
  withr::local_envvar(CUDF_HOME = stale)
  prefix <- withr::local_tempdir()
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) file.path(prefix, "lib", "libcudf.so")
  )

  res <- check_libcudf()
  expect_true(res$ok)
  expect_match(res$message, "CUDF_HOME", fixed = TRUE)
  expect_match(res$message, stale, fixed = TRUE)
  expect_match(res$message, "does not exist", fixed = TRUE)
})

test_that("check_libcudf() reports a CUDF_HOME that does not exist when nothing is found", {
  local_no_cudf_env()
  stale <- file.path(withr::local_tempdir(), "old-env")
  withr::local_envvar(CUDF_HOME = stale)
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) NA_character_,
    default_cudf_search_paths = function() stale
  )

  res <- check_libcudf()
  expect_false(res$ok)
  expect_match(res$message, "CUDF_HOME", fixed = TRUE)
  expect_match(res$message, stale, fixed = TRUE)
  expect_match(res$message, "does not exist", fixed = TRUE)
})

test_that("check_libcudf() reports a stale CUDF_HOME when libcudf is found elsewhere", {
  local_no_cudf_env()
  stale <- file.path(withr::local_tempdir(), "old-env")
  withr::local_envvar(CUDF_HOME = stale)
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  pixi <- make_cudf_prefix(file.path(getwd(), ".pixi", "envs", "default"))

  res <- check_libcudf()
  expect_true(res$ok)
  expect_true(same_path(res$value, pixi))
  expect_match(res$message, "CUDF_HOME", fixed = TRUE)
  expect_match(res$message, stale, fixed = TRUE)
  expect_match(res$message, "does not exist", fixed = TRUE)
})

test_that("check_libcudf() reports a CONDA_PREFIX that does not exist", {
  local_no_cudf_env()
  stale <- file.path(withr::local_tempdir(), "gone-conda")
  withr::local_envvar(CONDA_PREFIX = stale)
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) NA_character_,
    default_cudf_search_paths = function() stale
  )

  res <- check_libcudf()
  expect_false(res$ok)
  expect_match(res$message, "CONDA_PREFIX", fixed = TRUE)
  expect_match(res$message, stale, fixed = TRUE)
  expect_match(res$message, "does not exist", fixed = TRUE)
})

test_that("check_libcudf() says nothing about CUDF_HOME when it exists", {
  local_no_cudf_env()
  home <- make_cudf_prefix(withr::local_tempdir())
  withr::local_envvar(CUDF_HOME = home)
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)

  res <- check_libcudf()
  expect_true(res$ok)
  expect_true(same_path(res$value, home))
  expect_no_match(res$message, "does not exist", fixed = TRUE)
})

test_that("check_libcudf() reports the loaded libcudf over a valid CUDF_HOME", {
  local_no_cudf_env()
  home <- make_cudf_prefix(withr::local_tempdir())
  withr::local_envvar(CUDF_HOME = home)
  loaded <- withr::local_tempdir()
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) file.path(loaded, "lib", "libcudf.so")
  )

  res <- check_libcudf()
  expect_true(res$ok)
  expect_true(same_path(res$value, loaded))
  expect_no_match(res$message, "does not exist", fixed = TRUE)
})

test_that("check_libcudf(search_paths =) searches only the given paths", {
  local_no_cudf_env()
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) "/somewhere/lib/libcudf.so"
  )
  empty <- withr::local_tempdir()
  res <- check_libcudf(search_paths = empty)
  expect_false(res$ok)

  home <- make_cudf_prefix(withr::local_tempdir())
  res <- check_libcudf(search_paths = home)
  expect_true(res$ok)
  expect_true(same_path(res$value, home))
})

test_that("check_deps() JSON output carries the stale CUDF_HOME note", {
  local_no_cudf_env()
  stale <- file.path(withr::local_tempdir(), "old-env")
  withr::local_envvar(CUDF_HOME = stale)
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) NA_character_,
    default_cudf_search_paths = function() stale,
    run_nvidia_smi = function(args = character()) "615.78.08"
  )
  out <- paste(capture.output(res <- check_deps(format = "json")), collapse = "\n")
  expect_false(res$checks$cudf$ok)
  expect_identical(res$checks$driver$value, "615.78.08")
  expect_match(out, stale, fixed = TRUE)
  expect_match(out, '"value": "615.78.08"', fixed = TRUE)
})

# --- check_cuda_toolkit() finds nvcc in the pixi env --------------------------

# An executable nvcc stub under `prefix/bin` that prints a release line.
make_nvcc_stub <- function(prefix, release = "12.9") {
  dir.create(file.path(prefix, "bin"), recursive = TRUE, showWarnings = FALSE)
  nvcc <- file.path(prefix, "bin", "nvcc")
  writeLines(c(
    "#!/bin/sh",
    sprintf("echo 'Cuda compilation tools, release %s, V%s.86'", release, release)
  ), nvcc)
  Sys.chmod(nvcc, "0755")
  invisible(prefix)
}

test_that("check_cuda_toolkit() finds nvcc in a pixi env in the working directory", {
  skip_on_os("windows")
  local_no_cudf_env()
  withr::local_envvar(CUDA_HOME = NA)
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  make_nvcc_stub(file.path(getwd(), ".pixi", "envs", "default"), release = "12.9")

  res <- check_cuda_toolkit()
  expect_true(res$ok)
  expect_identical(res$value, "12.9")
  expect_match(res$message, file.path(".pixi", "envs", "default"), fixed = TRUE)
})

test_that("check_cuda_toolkit() finds nvcc in the pixi env named by PIXI_PROJECT_ROOT", {
  skip_on_os("windows")
  local_no_cudf_env()
  root <- withr::local_tempdir()
  withr::local_envvar(CUDA_HOME = NA, PIXI_PROJECT_ROOT = root, PIXI_ENVIRONMENT_NAME = "cuda12")
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  make_nvcc_stub(file.path(root, ".pixi", "envs", "cuda12"), release = "12.8")

  res <- check_cuda_toolkit()
  expect_true(res$ok)
  expect_identical(res$value, "12.8")
  expect_match(res$message, file.path(".pixi", "envs", "cuda12"), fixed = TRUE)
})

test_that("check_cuda_toolkit() prefers CUDA_HOME and CONDA_PREFIX over the pixi env", {
  skip_on_os("windows")
  local_no_cudf_env()
  testthat::local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  make_nvcc_stub(file.path(getwd(), ".pixi", "envs", "default"), release = "12.9")
  conda <- make_nvcc_stub(withr::local_tempdir(), release = "12.6")
  withr::local_envvar(CUDA_HOME = NA, CONDA_PREFIX = conda)
  expect_identical(check_cuda_toolkit()$value, "12.6")

  home <- make_nvcc_stub(withr::local_tempdir(), release = "12.4")
  withr::local_envvar(CUDA_HOME = home)
  expect_identical(check_cuda_toolkit()$value, "12.4")
})

test_that("check_cuda_toolkit() finds nvcc in the env whose libcudf cuplyr loaded", {
  skip_on_os("windows")
  local_no_cudf_env()
  withr::local_envvar(CUDA_HOME = NA)
  env <- make_nvcc_stub(withr::local_tempdir(), release = "12.7")
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) file.path(env, "lib", "libcudf.so")
  )

  res <- check_cuda_toolkit()
  expect_true(res$ok)
  expect_identical(res$value, "12.7")
  expect_match(res$message, env, fixed = TRUE)
})

test_that("check_cuda_toolkit() puts the loaded env after CUDA_HOME/CONDA_PREFIX and before a pixi guess", {
  skip_on_os("windows")
  local_no_cudf_env()
  withr::local_envvar(CUDA_HOME = NA)
  env <- make_nvcc_stub(withr::local_tempdir(), release = "12.7")
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) file.path(env, "lib", "libcudf.so")
  )
  make_nvcc_stub(file.path(getwd(), ".pixi", "envs", "default"), release = "12.9")
  expect_identical(check_cuda_toolkit()$value, "12.7")

  conda <- make_nvcc_stub(withr::local_tempdir(), release = "12.6")
  withr::local_envvar(CONDA_PREFIX = conda)
  expect_identical(check_cuda_toolkit()$value, "12.6")
})

test_that("check_cuda_toolkit() ignores a loaded env without nvcc", {
  skip_on_os("windows")
  local_no_cudf_env()
  withr::local_envvar(CUDA_HOME = NA)
  env <- withr::local_tempdir()
  testthat::local_mocked_bindings(
    loaded_libcudf_path = function(...) file.path(env, "lib", "libcudf.so")
  )
  make_nvcc_stub(file.path(getwd(), ".pixi", "envs", "default"), release = "12.9")
  expect_identical(check_cuda_toolkit()$value, "12.9")
})

# --- end to end on this machine -----------------------------------------------

test_that("check_nvidia_driver() reads the real driver version", {
  skip_if_no_gpu()
  skip_if(is.null(run_nvidia_smi()), "nvidia-smi not available")
  res <- check_nvidia_driver()
  expect_true(res$ok)
  expect_match(res$value, "^[0-9]+(\\.[0-9]+)+$")
})

test_that("loaded_libcudf_path() finds the libcudf this session uses", {
  skip_if_no_gpu()
  skip_if_not(file.exists("/proc/self/maps"), "no /proc/self/maps")
  lib <- loaded_libcudf_path()
  expect_false(is.na(lib))
  expect_true(file.exists(lib))
  expect_match(basename(lib), "^libcudf\\.so")
})

test_that("check_deps() is ok on a working install without CONDA_PREFIX and with a stale CUDF_HOME", {
  skip_if_no_gpu()
  skip_if_not(file.exists("/proc/self/maps"), "no /proc/self/maps")
  skip_if(is.null(run_nvidia_smi()), "nvidia-smi not available")
  local_no_cudf_env()
  stale <- file.path(withr::local_tempdir(), "old-env")
  withr::local_envvar(CUDF_HOME = stale)
  # The CUDA toolkit check is not part of this issue; keep it out of the result.
  testthat::local_mocked_bindings(
    check_cuda_toolkit = function() {
      list(ok = TRUE, name = "CUDA Toolkit", value = "13.0", message = "mocked")
    }
  )

  res <- NULL
  capture.output(res <- check_deps())
  expect_true(res$checks$cudf$ok)
  # value is the prefix of the libcudf actually loaded
  expect_gt(length(Sys.glob(file.path(res$checks$cudf$value, "lib*", "libcudf.so*"))), 0)
  expect_match(res$checks$cudf$message, stale, fixed = TRUE)
  expect_match(res$checks$cudf$message, "does not exist", fixed = TRUE)
  expect_true(res$checks$driver$ok)
  expect_false(is.na(res$checks$driver$value))
  expect_true(res$ok)
})
