test_that("check_deps() returns expected structure", {
  result <- check_deps(verbose = FALSE)

  expect_type(result, "list")
  expect_true("ok" %in% names(result))
  expect_true("checks" %in% names(result))
  expect_type(result$ok, "logical")

  # Each check has required fields
  for (check in result$checks) {
    expect_true(all(c("ok", "name", "value", "message") %in% names(check)))
    expect_type(check$name, "character")
    expect_type(check$message, "character")
  }
})

test_that("check_deps() has all expected checks", {
  result <- check_deps()

  expected_checks <- c("driver", "cuda", "cudf", "r_version", "r_packages", "gpu")
  expect_true(all(expected_checks %in% names(result$checks)))
})

test_that("check_deps(format = 'json') produces valid JSON structure", {
  json_output <- capture.output(check_deps(format = "json"))
  json_str <- paste(json_output, collapse = "\n")

  # Basic JSON structure checks

  expect_true(grepl("^\\{", json_str))
  expect_true(grepl("\"ok\":", json_str))
  expect_true(grepl("\"checks\":", json_str))
})

test_that("R version check works correctly", {
  result <- check_deps()
  r_check <- result$checks$r_version

  expect_true(r_check$ok)
  expect_equal(r_check$value, as.character(getRversion()))
})

test_that("R packages check finds installed packages", {
  result <- check_deps()
  pkg_check <- result$checks$r_packages

  # Since we're running tests, all required packages must be installed
  expect_true(pkg_check$ok)
})

test_that("check_deps() GPU access works when cuplyr is loaded", {
  skip_if_no_gpu()

  result <- check_deps()
  expect_true(result$checks$gpu$ok)
})

test_that("check_deps() treats skipped GPU check as non-fatal", {
  make_check <- function(ok, name, message = "ok") {
    list(ok = ok, name = name, value = "test", message = message)
  }

  testthat::local_mocked_bindings(
    check_nvidia_driver = function() make_check(TRUE, "NVIDIA Driver"),
    check_cuda_toolkit = function() make_check(TRUE, "CUDA Toolkit"),
    check_libcudf = function() make_check(TRUE, "libcudf"),
    check_r_version = function() make_check(TRUE, "R Version"),
    check_r_packages = function() make_check(TRUE, "R Packages"),
    check_gpu_access = function() make_check(NA, "GPU Access", "cuplyr not yet installed (skipped)")
  )

  result <- suppressMessages(check_deps(verbose = FALSE))
  expect_true(result$ok)
  expect_true(is.na(result$checks$gpu$ok))
})

test_that("check_libcudf() requires shared library presence", {
  tmp <- tempfile("cuplyr-cudf-")
  dir.create(file.path(tmp, "include", "cudf"), recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  file.create(file.path(tmp, "include", "cudf", "types.hpp"))

  result <- check_libcudf(search_paths = tmp)
  expect_false(result$ok)
  expect_match(result$message, "libcudf", ignore.case = TRUE)
})

test_that("check_libcudf() succeeds when headers and library exist", {
  tmp <- tempfile("cuplyr-cudf-")
  dir.create(file.path(tmp, "include", "cudf"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(tmp, "lib"), recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  file.create(file.path(tmp, "include", "cudf", "types.hpp"))
  file.create(file.path(tmp, "lib", "libcudf.so"))

  result <- check_libcudf(search_paths = tmp)
  expect_true(result$ok)
})
