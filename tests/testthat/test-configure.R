# configure is run against fake toolkit prefixes in a scratch copy of the
# package, so these tests need neither a GPU nor a real CUDA install.

configure_script <- function() {
  normalizePath(
    file.path(testthat::test_path("..", ".."), "configure"),
    mustWork = FALSE
  )
}

skip_if_no_configure <- function() {
  skip_on_os("windows")
  skip_if_not(
    file.exists(configure_script()),
    message = "configure not available in this test context"
  )
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

# A conda/pixi environment: headers under targets/, nvcc and libs at the top.
fake_conda_prefix <- function(root) {
  touch(file.path(root, "targets", "x86_64-linux", "include", "cuda.h"))
  touch(file.path(root, "targets", "x86_64-linux", "lib", "stubs", "libcuda.so"))
  fake_nvcc(file.path(root, "bin", "nvcc"))
  touch(file.path(root, "lib", "libcudart.so"))
  touch(file.path(root, "include", "cudf", "types.hpp"))
  touch(file.path(root, "lib", "libcudf.so"))
  root
}

# A system toolkit: everything under one CUDA_HOME.
fake_system_cuda <- function(root) {
  touch(file.path(root, "include", "cuda.h"))
  touch(file.path(root, "lib64", "stubs", "libcuda.so"))
  touch(file.path(root, "lib64", "libcudart.so"))
  fake_nvcc(file.path(root, "bin", "nvcc"))
  root
}

# Runs configure in a scratch package copy and returns its output and the
# generated Makevars as name -> value.
run_configure <- function(env, unset = character()) {
  pkg <- withr::local_tempdir("cuplyr-configure-", .local_envir = parent.frame())
  dir.create(file.path(pkg, "src"))
  file.copy(configure_script(), pkg)
  file.copy(
    file.path(dirname(configure_script()), "src", "Makevars.in"),
    file.path(pkg, "src")
  )

  unset <- unique(c(unset, "CUDF_HOME"))
  args <- c(
    as.vector(rbind("-u", unset)),
    sprintf("%s=%s", names(env), env),
    "CUPLYR_ALLOW_CUDA_STUBS=1",
    "bash", "configure"
  )
  output <- withr::with_dir(pkg, system2("env", args, stdout = TRUE, stderr = TRUE))
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }

  makevars <- file.path(pkg, "src", "Makevars")
  vars <- character()
  if (file.exists(makevars)) {
    lines <- grep("^[A-Z_]+=", readLines(makevars), value = TRUE)
    vars <- stats::setNames(sub("^[^=]*=", "", lines), sub("=.*", "", lines))
  }
  list(status = as.integer(status), output = paste(output, collapse = "\n"), vars = vars)
}

test_that("configure finds CUDA split across a conda environment", {
  skip_if_no_configure()
  prefix <- fake_conda_prefix(withr::local_tempdir("conda-env-"))

  res <- run_configure(c(CONDA_PREFIX = prefix), unset = "CUDA_HOME")

  expect_equal(res$status, 0L, info = res$output)
  expect_equal(
    res$vars[["CUDA_INCLUDE"]],
    file.path(prefix, "targets", "x86_64-linux", "include"),
    info = res$output
  )
  expect_equal(res$vars[["CUDA_RUNTIME_LIB"]], file.path(prefix, "lib"))
  expect_match(res$vars[["PKG_CPPFLAGS"]], "-I$(CUDA_INCLUDE)", fixed = TRUE)
})

test_that("configure prefers the conda toolkit over a system one", {
  skip_if_no_configure()
  skip_if_not(
    any(file.exists(file.path(c("/usr/local/cuda", "/opt/cuda"), "include", "cuda.h"))),
    message = "no system CUDA toolkit to compete with"
  )
  prefix <- fake_conda_prefix(withr::local_tempdir("conda-env-"))

  res <- run_configure(c(CONDA_PREFIX = prefix), unset = "CUDA_HOME")

  expect_equal(res$status, 0L, info = res$output)
  expect_equal(res$vars[["CUDA_HOME"]], prefix, info = res$output)
})

test_that("an explicit CUDA_HOME wins over the conda toolkit", {
  skip_if_no_configure()
  prefix <- fake_conda_prefix(withr::local_tempdir("conda-env-"))
  system_cuda <- fake_system_cuda(withr::local_tempdir("system-cuda-"))

  res <- run_configure(c(CONDA_PREFIX = prefix, CUDA_HOME = system_cuda))

  expect_equal(res$status, 0L, info = res$output)
  expect_equal(res$vars[["CUDA_HOME"]], system_cuda, info = res$output)
  expect_equal(res$vars[["CUDA_INCLUDE"]], file.path(system_cuda, "include"))
})
