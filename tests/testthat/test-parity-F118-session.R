# F118: the harness's own guarantees that the acceptance file doesn't pin.

test_that("generated calls resolve base functions whatever the session attached", {
  # bit64 masks `%in%` on the search path; the lookup env must not see that
  withr::local_package("bit64")
  env <- fz_env(fuzz_make_case(tibble::tibble(a = 1L), list()))
  expect_identical(get("%in%", envir = env), base::`%in%`)
  expect_identical(get("lag", envir = env), dplyr::lag)
  expect_identical(topenv(env), globalenv())
})

test_that("a session's dplyr.legacy_locale can't add a warning or change the result", {
  cc <- fuzz_make_case(
    tibble::tibble(k = c("b", "A"), a = 1:2),
    list(quote(arrange(k)))
  )
  plain <- fuzz_oracle(cc)
  withr::local_options(dplyr.legacy_locale = TRUE)
  set <- fuzz_oracle(cc)
  expect_length(set$warnings, 0L)
  expect_identical(set$value, plain$value)
  expect_true(getOption("dplyr.legacy_locale"))
})

test_that("the first shrink pass can't spend a later pass's budget", {
  # every candidate "still fails", so the steps pass would drop steps forever
  cc <- fuzz_make_case(
    x = tibble::tibble(a = 1:16),
    steps = rep(list(quote(filter(a > 0L))), 40L)
  )
  rows_tried <- FALSE
  fails <- function(k) {
    if (nrow(k$x) < 16L) rows_tried <<- TRUE
    TRUE
  }
  s <- fuzz_shrink(cc, fails = fails, max_attempts = 20L)
  expect_true(rows_tried)
  expect_true("steps" %in% attr(s, "budget_exhausted"))
})

test_that("signatures don't split on how a name is quoted, and keep apostrophes", {
  expect_identical(fz_norm_msg("no 'abc' here"), fz_norm_msg("no `xyz` here"))
  expect_identical(fz_norm_msg("no \"abc\" here"), fz_norm_msg("no 'xyz' here"))
  expect_identical(fz_norm_msg("no ‘abc’ here"), fz_norm_msg("no `xyz` here"))
  expect_match(fz_norm_msg("can't find 'a'"), "can't find", fixed = TRUE)
  expect_identical(
    fz_norm_msg("GPU operation 'mutate' failed.\nOriginal error: internal: bad 'lag'"),
    fz_norm_msg("internal: bad 'lead'")
  )
})

test_that("a NaN and an NA timestamp are different the other way round", {
  d <- function(v) fuzz_outcome(value = tibble::tibble(d = structure(v, class = "Date")))
  expect_equal(fuzz_compare(d(NA_real_), d(NaN))$kind, "value")
})
