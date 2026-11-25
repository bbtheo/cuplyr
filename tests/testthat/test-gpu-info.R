# Tests for GPU detection and information functions
#
# These tests verify:
# - GPU availability detection
# - GPU information retrieval
# - GPU info display

test_that("has_gpu() returns a logical value", {
  result <- has_gpu()
  expect_type(result, "logical")
  expect_length(result, 1)
  expect_false(is.na(result))
})

test_that("has_gpu() returns TRUE on systems with GPU", {
  skip_if_no_gpu()
  expect_true(has_gpu())
})

test_that("has_gpu() handles errors gracefully", {
  # Should never error, only return TRUE or FALSE
  expect_no_error(has_gpu())
})

test_that("gpu_details() returns a list", {
  result <- gpu_details()
  expect_type(result, "list")
})

test_that("gpu_details() returns expected fields when GPU available", {
  skip_if_no_gpu()

  info <- gpu_details()

  # Required fields
  expect_true("available" %in% names(info))
  expect_true(info$available)

  expect_true("device_count" %in% names(info))
  expect_type(info$device_count, "integer")
  expect_true(info$device_count >= 1)

  expect_true("device_id" %in% names(info))
  expect_type(info$device_id, "integer")
  expect_true(info$device_id >= 0)

  expect_true("name" %in% names(info))
  expect_type(info$name, "character")
  expect_true(nchar(info$name) > 0)

  expect_true("compute_capability" %in% names(info))
  expect_type(info$compute_capability, "character")
  # Should match pattern like "7.5" or "8.9"
  expect_match(info$compute_capability, "^\\d+\\.\\d+$")

  expect_true("total_memory" %in% names(info))
  expect_type(info$total_memory, "double")
  expect_true(info$total_memory > 0)

  expect_true("free_memory" %in% names(info))
  expect_type(info$free_memory, "double")
  expect_true(info$free_memory >= 0)
  expect_true(info$free_memory <= info$total_memory)

  expect_true("multiprocessors" %in% names(info))
  expect_type(info$multiprocessors, "integer")
  expect_true(info$multiprocessors > 0)
})

test_that("gpu_details() returns minimal info when no GPU", {
  # This test only runs when there's no GPU
  skip_if(has_gpu(), "GPU is available, skipping no-GPU test")

  info <- gpu_details()
  expect_type(info, "list")
  expect_false(isTRUE(info$available))
})

test_that("gpu_details() handles errors gracefully", {
  # Should never throw an unhandled error
  expect_no_error(gpu_details())
})

test_that("show_gpu() returns invisibly and produces output", {
  skip_if_no_gpu()

  # Capture output
  output <- capture.output(result <- show_gpu())

  # Should return the info invisibly
  expect_type(result, "list")
  expect_true(result$available)

  # Should produce console output
  expect_true(length(output) > 0)

  # Output should contain key information
  output_text <- paste(output, collapse = "\n")
  expect_match(output_text, "GPU Information|Device:", ignore.case = TRUE)
})

test_that("show_gpu() displays informative message when no GPU", {
  skip_if(has_gpu(), "GPU is available, skipping no-GPU test")

  output <- capture.output(result <- show_gpu())
  output_text <- paste(output, collapse = "\n")

  expect_match(output_text, "No GPU available", ignore.case = TRUE)
})

test_that("GPU compute capability meets minimum requirements", {
  skip_if_no_gpu()

  info <- gpu_details()
  cc <- info$compute_capability

  # Parse compute capability
  parts <- as.numeric(strsplit(cc, "\\.")[[1]])
  major <- parts[1]
  minor <- parts[2]

  # Package requires compute capability >= 7.5 (Turing)
  expect_true(major > 7 || (major == 7 && minor >= 5),
              info = paste("Compute capability", cc, "is below 7.5"))
})

test_that("GPU memory values are sensible", {
  skip_if_no_gpu()

  info <- gpu_details()

  # Memory should be at least 1 GB for any modern GPU
  min_memory <- 1e9  # 1 GB
  expect_true(
    info$total_memory >= min_memory,
    info = sprintf("Total memory %s seems too low",
                   format_bytes(info$total_memory))
  )

  # Free memory should be positive
  expect_true(info$free_memory > 0)

  # Free memory should not exceed total
  expect_true(info$free_memory <= info$total_memory)

  # Used memory calculation should be consistent
  used <- info$total_memory - info$free_memory
  expect_true(used >= 0)
})

test_that("Multiple calls to gpu_details() return consistent info",
{
  skip_if_no_gpu()

  info1 <- gpu_details()
  info2 <- gpu_details()

  # Static properties should be identical
  expect_identical(info1$device_count, info2$device_count)
  expect_identical(info1$device_id, info2$device_id)
  expect_identical(info1$name, info2$name)
  expect_identical(info1$compute_capability, info2$compute_capability)
  expect_identical(info1$total_memory, info2$total_memory)
  expect_identical(info1$multiprocessors, info2$multiprocessors)

  # Free memory may vary slightly but should be close
  diff_pct <- abs(info1$free_memory - info2$free_memory) / info1$total_memory
  expect_true(diff_pct < 0.1, info = "Free memory varied by more than 10%")
})

test_that("GPU detection is fast", {
  # has_gpu() should be quick (under 1 second)
  time <- system.time(result <- has_gpu())
  expect_true(time["elapsed"] < 1,
              info = sprintf("has_gpu() took %.2f seconds", time["elapsed"]))
})
