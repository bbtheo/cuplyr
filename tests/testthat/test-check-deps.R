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
