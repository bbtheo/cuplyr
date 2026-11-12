# Phase 11 L5: base-R string functions native in filter()/mutate() (R/ir.R's
# dedicated dispatch functions, src/expr_eval.hpp's apply_handler() cases).
#
# GOVERNING FINDING (scratchpad/phase11_design.md): dplyr's own test suite
# barely exercises string functions -- this file's job is oracle parity for
# the base-R functions cuplyr now implements natively, INCLUDING every NA
# rule verified empirically against real R first (see each test's own
# comment), not to be an exhaustive stringr/regex conformance suite. Every
# test runs eager AND lazy. Deliberate scope cuts (paste's non-STRING args,
# collapse=, ignore.case=/perl=/useBytes=, non-literal patterns/positions,
# nchar(type != "chars")) are pinned as "falls back, doesn't crash" via
# test-dplyr-fallback-expr.R-style checks at the bottom of this file.

test_that("toupper()/tolower() match dplyr, including NA propagation", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("Hello World", "foo BAR", NA, "  trim me  "))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate(tbl, u = toupper(s), l = tolower(s))
  })
  expect_same_as_dplyr_lazy(df, function(tbl) {
    dplyr::mutate(tbl, u = toupper(s), l = tolower(s))
  })
})

test_that("nchar() matches dplyr, including NA -> NA (not a length-2 placeholder)", {
  skip_if_no_gpu()
  # Verified empirically (plain Rscript): nchar(NA) is NA in this R version
  # (not the historical "2" some older R versions returned) -- cuDF's
  # count_characters() null-propagates already, so no special-casing needed.
  df <- tibble::tibble(s = c("Hello World", "foo BAR", NA, ""))

  expect_same_as_dplyr(df, function(tbl) dplyr::mutate(tbl, n = nchar(s)))
  expect_same_as_dplyr_lazy(df, function(tbl) dplyr::mutate(tbl, n = nchar(s)))
})

test_that("substr() matches dplyr's exact clamping rules", {
  skip_if_no_gpu()
  # Verified empirically against real substr(): start < 1 clamps UP to 1
  # before 0-based conversion; stop past the string end clamps to the
  # string's own length; start beyond the string's length is empty; stop <
  # start is empty; NA propagates.
  df <- tibble::tibble(s = c("hello", "goodbye", NA, ""))

  pipeline <- function(tbl) {
    dplyr::mutate(
      tbl,
      normal = substr(s, 1, 3),
      neg_start = substr(s, -2, 3),
      zero_start = substr(s, 0, 3),
      past_end = substr(s, 2, 100),
      start_past_len = substr(s, 10, 20),
      stop_before_start = substr(s, 5, 2),
      neg_stop = substr(s, 1, -1)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("substr() with a non-constant start/stop falls back to CPU and matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("hello", "goodbye"), k = c(1, 2))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, sub = substr(s, k, 3)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, sub = substr(s, k, 3)))
})

test_that("trimws() matches dplyr for which = both/left/right", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("  hi  ", "no_space", NA, "\tleading\t"))

  pipeline <- function(tbl) {
    dplyr::mutate(
      tbl,
      both = trimws(s),
      left = trimws(s, which = "left"),
      right = trimws(s, which = "right")
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("trimws(whitespace = ) falls back to CPU and matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(s = c("--hi--", "--bye--"))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, t = trimws(s, whitespace = "-")),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, t = trimws(s, whitespace = "-")))
})

test_that("paste0()/paste() match dplyr, including the NA -> literal 'NA' rule", {
  skip_if_no_gpu()
  # Verified empirically: paste0("a", NA, "b") is "aNAb", NOT NA -- a
  # genuine NA argument becomes the two-character string "NA".
  df <- tibble::tibble(a = c("foo", "BAR", NA, "banana"), b = c("x", "y", "z", NA))

  pipeline <- function(tbl) {
    dplyr::mutate(
      tbl,
      p0 = paste0(a, "-", b),
      p1 = paste(a, b),
      p2 = paste(a, b, sep = "_")
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("paste0() with a non-STRING argument falls back (the 'x4' vs '4.0' trap)", {
  skip_if_no_gpu()
  # Design doc's own trap: paste0("x", 4) is "x4", NOT "x4.0" -- cuplyr
  # deliberately does not replicate as.character()'s numeric formatting, so
  # a non-STRING dot always falls back to real R's own paste0() instead.
  df <- tibble::tibble(a = c("x", "y"))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, p = paste0(a, 4)),
    "fell back to CPU evaluation"
  )
  oracle <- dplyr::mutate(df, p = paste0(a, 4))
  expect_equal(oracle$p, c("x4", "y4"))
  expect_equal(collect(result), oracle)
})

test_that("paste0(collapse = ) falls back to CPU and matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("x", "y", "z"))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, p = paste0(a, collapse = ",")),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, p = paste0(a, collapse = ",")))
})

test_that("grepl() matches dplyr, including NA -> FALSE (not NA)", {
  skip_if_no_gpu()
  # Verified empirically: grepl("a", c("cat", NA, "dog")) is
  # TRUE, FALSE, FALSE -- NOT TRUE, NA, FALSE.
  df <- tibble::tibble(a = c("foo", "BAR", NA, "banana"))

  pipeline <- function(tbl) {
    dplyr::mutate(tbl, hit = grepl("an", a), hit_fixed = grepl("an", a, fixed = TRUE))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("grepl() as a filter() predicate matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("foo", "BAR", NA, "banana"), y = c(1, 2, 3, 4))

  expect_same_as_dplyr(df, function(tbl) dplyr::filter(tbl, grepl("an", a) & y > 0))
  expect_same_as_dplyr_lazy(df, function(tbl) dplyr::filter(tbl, grepl("an", a) & y > 0))
})

test_that("grepl(ignore.case = TRUE) falls back to CPU and matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("foo", "BAR", "banana"))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, h = grepl("BAN", a, ignore.case = TRUE)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, h = grepl("BAN", a, ignore.case = TRUE)))
})

test_that("grepl() with a non-literal (column) pattern falls back to CPU and matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("foo", "BAR", "banana"), pat = c("f", "B", "ban"))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  # Two independent warnings fire here: cuplyr's own fallback notification,
  # and (bubbled up from the real dplyr call the fallback re-runs) base R's
  # own "argument 'pattern' has length > 1" warning -- suppress the latter
  # specifically so expect_warning() below only has to match the former.
  withCallingHandlers(
    expect_warning(
      result <- dplyr::mutate(g, h = grepl(pat, a)),
      "fell back to CPU evaluation"
    ),
    warning = function(w) {
      if (grepl("length > 1", conditionMessage(w))) invokeRestart("muffleWarning")
    }
  )
  expect_equal(collect(result), suppressWarnings(dplyr::mutate(df, h = grepl(pat, a))))
})

test_that("sub()/gsub() match dplyr, including NA propagation and fixed=TRUE", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("banana", NA, "aardvark"))

  pipeline <- function(tbl) {
    dplyr::mutate(
      tbl,
      s1 = sub("a", "X", a),
      s2 = gsub("a", "X", a),
      s3 = sub("a", "X", a, fixed = TRUE),
      s4 = gsub("a", "X", a, fixed = TRUE)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("startsWith()/endsWith() match dplyr, including NA propagation", {
  skip_if_no_gpu()
  # Verified empirically: startsWith(c("abc", NA), "a") is TRUE, NA -- NOT
  # TRUE, FALSE (unlike grepl()'s NA -> FALSE rule above).
  df <- tibble::tibble(a = c("foo", "BAR", NA, "banana"))

  pipeline <- function(tbl) {
    dplyr::mutate(tbl, sw = startsWith(a, "b"), ew = endsWith(a, "a"))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# -----------------------------------------------------------------------------
# Nested usage inside a larger AST-native expression (build_ast()'s own
# operand-type-matching logic, src/expr_eval.hpp) -- a real correctness risk
# this task found: a STRING-producing "call" node (toupper()/nchar()/etc.)
# used as an operand of a comparison/arithmetic op needs resolve_static_type()
# to report its REAL type, or libcudf throws "non-matching operand types".
# -----------------------------------------------------------------------------

test_that("a string function's result nests correctly inside a comparison/arithmetic AST", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("foo", "BAR", "banana"), y = c(1, 2, 3))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, toupper(a) == "BAR" | y > 2)
  })
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate(tbl, n2 = nchar(a) + 1L)
  })
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, nchar(a) > 3.5)
  })
})

# -----------------------------------------------------------------------------
# cuDF regex != R's TRE/PCRE dialect -- simple ASCII patterns only (per the
# design doc's own scope note). Anchors/character classes/quantifiers that
# are common to both dialects are fair game for oracle testing; anything
# dialect-specific is out of scope.
# -----------------------------------------------------------------------------

test_that("simple ASCII regex patterns in grepl()/sub()/gsub() match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c("foo123", "bar", "123baz", "BAZ99"))

  pipeline <- function(tbl) {
    dplyr::mutate(
      tbl,
      has_digit = grepl("[0-9]+", a),
      starts_digit = grepl("^[0-9]", a),
      ends_digit = grepl("[0-9]$", a),
      no_digits = gsub("[0-9]", "", a)
    )
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})
