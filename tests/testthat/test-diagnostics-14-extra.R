# Issue #14, beyond the acceptance tests: diagnostics() and the once-per-session
# invalid-option warnings, HOME = "/" in redaction, the registry's
# computed-default flag, and JSON for strings that aren't valid UTF-8.
#
# The invalid-option warnings are remembered for the whole session, per value,
# so every test below uses a value of its own.

d14x_quiet <- function(expr) {
  utils::capture.output(res <- expr)
  res
}

test_that("under warn = 0, diagnostics() gives an invalid-option warning once", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "bogus-14x-warn0", warn = 0)
  expect_warning(d14x_quiet(diagnostics(redact = FALSE)),
                 class = "cuplyr_warning_invalid_option")
  # the user saw it: later reads stay quiet, as after any other read
  expect_no_warning(cuplyr_options(internal = TRUE))
})

test_that("under warn = 2, a fallback still raises the invalid option after diagnostics()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "bogus-14x-fallback", warn = 2)
  expect_no_error(suppressWarnings(d14x_quiet(diagnostics(redact = FALSE))))
  df <- data.frame(x = 1:5)
  for (i in 1:2) {
    expect_error(
      collect(mutate(tbl_gpu(df), w = qnorm(x / 10))),
      "bogus-14x-fallback", fixed = TRUE
    )
  }
})

test_that("under warn = 2, diagnostics() keeps a warning that was already given", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "bogus-14x-given", warn = 0)
  expect_warning(cuplyr_options(), class = "cuplyr_warning_invalid_option")
  withr::local_options(warn = 2)
  # given before diagnostics(), so it stays given: no error, now or later
  expect_no_error(d14x_quiet(diagnostics(redact = FALSE)))
  expect_no_error(cuplyr_options(internal = TRUE))
})

test_that("HOME = '/' doesn't turn every '/' into '~'", {
  skip_if_no_gpu()
  withr::local_envvar(HOME = "/")
  lib <- "/opt/rapids-14x/lib/libcudf.so"
  user <- Sys.info()[["user"]]
  skip_if(nzchar(user) && grepl(user, lib, fixed = TRUE))
  local_mocked_bindings(loaded_libcudf_path = function(...) lib)
  info <- d14x_quiet(diagnostics(redact = TRUE))
  expect_identical(info$libcudf_path, lib)
})

test_that("the text report takes 'computed' from the registry", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.pool_initial = NULL, cuplyr.release_threshold = NULL)
  withr::local_envvar(CUPLYR_POOL_INITIAL = NA, CUPLYR_RELEASE_THRESHOLD = NA)
  line <- function(out, o) grep(paste0(o, " = "), out, fixed = TRUE, value = TRUE)
  out <- utils::capture.output(diagnostics(redact = FALSE))
  expect_match(line(out, "cuplyr.pool_initial"), "default computed", fixed = TRUE)
  expect_match(line(out, "cuplyr.release_threshold"), "default unset", fixed = TRUE)
  # flip the registry's flags: the labels follow, no option name is hard-coded
  reg <- cuplyr_option_registry
  reg$pool_initial$computed <- FALSE
  reg$release_threshold$computed <- TRUE
  local_mocked_bindings(cuplyr_option_registry = reg)
  out <- utils::capture.output(diagnostics(redact = FALSE))
  expect_match(line(out, "cuplyr.pool_initial"), "default unset", fixed = TRUE)
  expect_match(line(out, "cuplyr.release_threshold"), "default computed", fixed = TRUE)
  # the list keeps NA for both
  opts <- d14x_quiet(diagnostics(redact = FALSE))$options
  expect_identical(opts$default[opts$option == "cuplyr.pool_initial"], NA_character_)
  expect_identical(opts$default[opts$option == "cuplyr.release_threshold"], NA_character_)
})

test_that("json_escape() makes valid UTF-8 of a string with a stray byte", {
  # pure R: runs without a GPU too
  skip_if_not_installed("jsonlite")
  skip_if(!l10n_info()[["UTF-8"]])
  bad <- "/opt/caf\xff-14x"
  txt <- to_json(list(path = bad, ok = "/opt/café"))
  expect_true(validUTF8(txt))
  expect_true(jsonlite::validate(txt))
  parsed <- jsonlite::fromJSON(txt, simplifyVector = FALSE)
  expect_identical(parsed$path, "/opt/caf<ff>-14x")
  expect_identical(parsed$ok, "/opt/café")
  # valid strings are written exactly as before
  expect_identical(json_escape("a \"q\" b\\c\nd café"), "a \\\"q\\\" b\\\\c\\nd café")
})
