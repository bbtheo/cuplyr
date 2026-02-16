test_that("diagnostics() returns expected structure", {
  result <- diagnostics(redact = FALSE)

  expect_type(result, "list")
  expect_true("cuplyr_version" %in% names(result))
  expect_true("r_version" %in% names(result))
  expect_true("os" %in% names(result))
  expect_true("cuda_home" %in% names(result))
  expect_true("gpu" %in% names(result))
  expect_true("ld_library_path" %in% names(result))
})

test_that("diagnostics() redaction works", {
  home <- Sys.getenv("HOME", "~")

  redacted <- diagnostics(redact = TRUE)
  unredacted <- diagnostics(redact = FALSE)

  # Redacted version should not contain home directory
  if (nzchar(home) && nzchar(redacted$cuda_home)) {
    expect_false(grepl(home, redacted$cuda_home, fixed = TRUE))
  }
})

test_that("diagnostics() reports correct cuplyr version", {
  result <- diagnostics(redact = FALSE)
  expect_equal(result$cuplyr_version, as.character(packageVersion("cuplyr")))
})

test_that("diagnostics() reports correct R version", {
  result <- diagnostics(redact = FALSE)
  expected <- paste0(R.version$major, ".", R.version$minor)
  expect_equal(result$r_version, expected)
})

test_that("diagnostics() detects GPU when available", {
  skip_if_no_gpu()

  result <- diagnostics(redact = FALSE)
  expect_true(isTRUE(result$gpu$available))
  expect_true(nzchar(result$gpu$name))
})
