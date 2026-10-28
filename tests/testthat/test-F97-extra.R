# F97 (extra): the large-strings-disabled refusal reaches every upload path
# with its class, before any column is uploaded, and its message tells the
# user the variable's actual value and the fix.

local_strings_disabled <- function(enabled = "0", .env = parent.frame()) {
  withr::local_envvar(
    LIBCUDF_LARGE_STRINGS_ENABLED = enabled,
    LIBCUDF_LARGE_STRINGS_THRESHOLD = "16",
    .local_envir = .env
  )
}

test_that("a `$<-` upload over the limit is the classed error naming the column", {
  skip_if_no_gpu()
  local_strings_disabled()
  g <- tbl_gpu(tibble::tibble(s = c("abc", "def")))
  err <- expect_error(
    { g$wide <- c("abcdefghijklm", "nopqrstuv") },
    class = "cuplyr_error_string_too_large"
  )
  expect_match(conditionMessage(err), "wide", fixed = TRUE)
  expect_match(conditionMessage(err), "22", fixed = TRUE)
  expect_identical(collect(g)$s, c("abc", "def"))
})

test_that("a CPU-fallback re-upload over the limit is the classed error", {
  skip_if_no_gpu()
  local_strings_disabled()
  g <- tbl_gpu(tibble::tibble(s = c("abcdefghij", "klmno")))
  err <- expect_error(
    suppressMessages(mutate(g, t = vapply(s, function(z) paste0(z, z, z), ""))),
    class = "cuplyr_error_string_too_large"
  )
  expect_match(conditionMessage(err), "`t`", fixed = TRUE)
})

test_that("the refusal happens before df_to_gpu() uploads any column", {
  skip_if_no_gpu()
  local_strings_disabled()
  local_mocked_bindings(df_to_gpu = function(...) stop("df_to_gpu was reached"))
  df <- tibble::tibble(a = as.numeric(1:10), big_col = rep(c("abcdefghij", "klmnopqrstu"), 5))
  expect_error(tbl_gpu(df), class = "cuplyr_error_string_too_large")
})

test_that("with large strings enabled the pre-pass is skipped", {
  skip_if_no_gpu()
  withr::local_envvar(LIBCUDF_LARGE_STRINGS_ENABLED = "1", LIBCUDF_LARGE_STRINGS_THRESHOLD = "16")
  local_mocked_bindings(first_string_column_over = function(...) stop("pre-pass ran"))
  expect_no_error(tbl_gpu(tibble::tibble(s = c("abcdefghij", "klmnopqrstu"))))
})

test_that("the message prints the variable's actual value, the threshold and the fix", {
  skip_if_no_gpu()
  local_strings_disabled("true")
  df <- tibble::tibble(s = c("abcdefghij", "klmnopqrstu"))
  err <- expect_error(tbl_gpu(df), class = "cuplyr_error_string_too_large")
  msg <- conditionMessage(err)
  expect_match(msg, 'LIBCUDF_LARGE_STRINGS_ENABLED="true"', fixed = TRUE)
  expect_match(msg, 'LIBCUDF_LARGE_STRINGS_THRESHOLD="16"', fixed = TRUE)
  expect_match(msg, "Unset LIBCUDF_LARGE_STRINGS_ENABLED or set it to \"1\"", fixed = TRUE)
})

test_that("with large strings enabled the same upload goes through at 64-bit offsets", {
  skip_if_no_gpu()
  local_strings_disabled("1")
  df <- tibble::tibble(s = c("abcdefghij", "klmnopqrstu", NA))
  g <- tbl_gpu(df)
  expect_identical(gpu_string_offsets_width(g$ptr, 0L), 64L)
  expect_equal(tibble::as_tibble(collect(g)), df)
})
