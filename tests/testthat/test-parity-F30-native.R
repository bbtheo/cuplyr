# F30 acceptance tests, part 1: recode() shapes that run natively on the GPU.
#
# dplyr 1.2.1 R/recode.R (verified with tools/r):
#   - character .x: replacements named, key = the name verbatim.
#   - numeric .x: all named (key = as.double(name)) or all unnamed (key =
#     position).
#   - the result type is typeof() of the first replacement; all values must
#     share it exactly.
#   - NA rows never match a key and never take .default; only .missing
#     replaces them.
#   - without .default, unmatched values keep .x when the types agree.
#
# "Native" is pinned with options(cuplyr.fallback = "error"): any CPU
# fallback inside these pipelines raises.

# Tests see the package namespace, which doesn't import recode(); bind dplyr's
# own functions here, as library(dplyr) would for a user.
recode <- dplyr::recode

f30_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f30_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f30_both(df, pipeline, ...)
}

f30_df <- function() {
  data.frame(
    x = c("a", "b", "c", NA, "a", "NA", "b c", "", "d"),
    n = c(1L, 2L, 3L, NA, 1L, 4L, -1L, 0L, 2L),
    d = c(1, 2, 3.5, NA, 0.1, -1, 2.5, 0, 1e6),
    g = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 2L, 3L),
    stringsAsFactors = FALSE
  )
}

test_that("character keys by name; unmatched keep x, NA stays NA", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::mutate(t,
      y1 = recode(x, a = "A", b = "B"),
      y2 = recode(x, `b c` = "BC", "NA" = "na-string", c = "C"),
      y3 = recode(.x = x, d = "D")
    )
  })
})

test_that("!!! splicing a named character lookup is native (small and 20 keys)", {
  skip_if_no_gpu()
  small <- c(a = "Apple", b = "Banana", c = "Cherry")
  big <- stats::setNames(paste0("V", 1:20), c(letters[1:19], "b c"))
  f30_native(f30_df(), function(t) {
    dplyr::mutate(t,
      y1 = recode(x, !!!small),
      y2 = recode(x, !!!big),
      y3 = recode(x, !!!big, .default = "other")
    )
  })
  val <- "injected"
  f30_native(f30_df(), function(t) dplyr::mutate(t, y = recode(x, a = !!val)))
})

test_that(".default and .missing on character input", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::mutate(t,
      y1 = recode(x, a = "A", .default = "Z"),
      y2 = recode(x, a = "A", .missing = "M"),
      y3 = recode(x, a = "A", .default = "Z", .missing = "M"),
      y4 = recode(x, a = "A", .default = NA_character_),
      y5 = recode(x, a = NA_character_, b = "B")
    )
  })
})

test_that("character input recoded to another type with .default", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::mutate(t,
      i = recode(x, a = 1L, b = 2L, .default = 0L),
      i_m = recode(x, a = 1L, .default = 0L, .missing = -1L),
      dbl = recode(x, a = 1.5, c = -2, .default = 0),
      lgl = recode(x, a = TRUE, b = FALSE, .default = NA),
      all_na = recode(x, a = NA, .default = NA),
      dbl_na = recode(x, a = 1, .default = NA_real_)
    )
  })
})

test_that("integer input: named and positional keys", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::mutate(t,
      named = recode(n, `1` = 10L, `3` = 30L),
      neg = recode(n, `-1` = 99L, `0` = 100L),
      pos = recode(n, 100L, 200L),
      chr = recode(n, `1` = "one", `2` = "two", .default = "many"),
      chr_m = recode(n, `1` = "one", .default = "other", .missing = "none"),
      pos_chr = recode(n, "first", "second", .default = "rest"),
      miss = recode(n, `1` = 10L, .missing = -1L),
      dbl = recode(n, `2` = 0.5, .default = 1)
    )
  })
})

test_that("double input: named and positional keys, -0 matches 0", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::mutate(t,
      named = recode(d, `3.5` = 100, `0.1` = 0, `-1` = 1),
      zero = recode(d, `0` = 42),
      pos = recode(d, 10, 20),
      chr = recode(d, `1` = "one", `1e6` = "million", .default = "?"),
      miss = recode(d, `1` = 0, .missing = -1),
      both = recode(d, `2` = 20, .default = -5, .missing = -9)
    )
  })
  neg_zero <- data.frame(d = c(-0, 0, 1, NA))
  f30_native(neg_zero, function(t) {
    dplyr::mutate(t, y = recode(d, `0` = 100, .default = 7))
  })
})

test_that("NaN input behaves like NA (no .default, with .default, with .missing)", {
  skip_if_no_gpu()
  nan_df <- data.frame(d = c(1, NaN, NA, 2))
  f30_native(nan_df, function(t) {
    dplyr::select(
      dplyr::mutate(t,
        y1 = recode(d, `1` = 0),
        y2 = recode(d, `1` = 0, .default = 5),
        y3 = recode(d, `1` = 0, .missing = -1)
      ),
      y1, y2, y3
    )
  })
})

test_that("grouped mutate, namespaced call, reuse of an earlier output", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::mutate(dplyr::group_by(t, g),
      y = dplyr::recode(x, a = "A", .default = "other"),
      z = toupper(x),
      w = recode(z, A = "a-again", B = "b-again")
    )
  })
})

test_that("recode() inside filter()", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::filter(t, recode(x, a = "A", b = "A") == "A")
  })
  f30_native(f30_df(), function(t) {
    dplyr::filter(dplyr::group_by(t, g), recode(n, `1` = 0L, .default = 1L) == 1L)
  })
})

test_that("recode() inside summarise() aggregations", {
  skip_if_no_gpu()
  f30_native(f30_df(), function(t) {
    dplyr::summarise(dplyr::group_by(t, g),
      s = sum(recode(n, `1` = 10L, .default = 0L, .missing = 0L)),
      m = mean(recode(x, a = 1, .default = 0, .missing = 0))
    )
  }, arrange_by = "g")
  f30_native(f30_df(), function(t) {
    dplyr::summarise(t, s = sum(recode(x, a = 1L, b = 2L, .default = 0L), na.rm = TRUE))
  })
})

test_that("zero-row input keeps the result type", {
  skip_if_no_gpu()
  empty <- f30_df()[0, ]
  f30_native(empty, function(t) {
    dplyr::mutate(t,
      y = recode(x, a = "A"),
      i = recode(x, a = 1L, .default = 0L),
      k = recode(n, `1` = "one", .default = "other"),
      e = recode(d, `1` = 2)
    )
  })
})

test_that("lazy pipeline around a native recode()", {
  skip_if_no_gpu()
  lookup <- c(a = "A", b = "B", c = "C")
  f30_native(f30_df(), function(t) {
    t |>
      dplyr::filter(!is.na(n)) |>
      dplyr::mutate(y = recode(x, !!!lookup, .default = "other")) |>
      dplyr::filter(y != "C") |>
      dplyr::arrange(dplyr::desc(n), y) |>
      dplyr::select(y, n)
  })
})

# Round 1 review B1: a NaN produced on the GPU is a value, not a null, but
# dplyr's recode() treats it as missing (never matched, never .default).
test_that("a NaN computed on the GPU counts as missing", {
  skip_if_no_gpu()
  df <- data.frame(d = c(0, 1, 2, NA, -1))
  f30_native(df, function(t) {
    t |>
      dplyr::mutate(z = d / d) |>
      dplyr::mutate(
        y = recode(z, `1` = "one", .default = "other"),
        w = recode(z, `1` = 10, .missing = -1),
        v = recode(z, `1` = 10),
        u = recode(z, `1` = 10, .default = 0),
        s = recode(z, `2` = 20, .default = 0, .missing = -9)
      ) |>
      dplyr::select(y, w, v, u, s)
  })
  f30_native(df, function(t) {
    dplyr::select(
      dplyr::mutate(t, z = d / d, y = recode(z, `1` = "one", .default = "other")),
      y
    )
  })
})

# Round 1 review B2: a lookup with many keys must not need GPU memory
# proportional to keys x rows (the pre-F30 CPU route handled this).
test_that("a 300-key lookup with long labels does not exhaust GPU memory", {
  skip_if_no_gpu()
  set.seed(30)
  keys <- sprintf("code_%04d", 1:300)
  lookup <- stats::setNames(paste0(strrep("L", 200), sprintf("_%04d", 1:300)), keys)
  df <- data.frame(x = sample(c(keys, "other", NA), 2e5, TRUE), stringsAsFactors = FALSE)
  f30_native(df, function(t) dplyr::mutate(t, y = recode(x, !!!lookup)))
  f30_native(df, function(t) dplyr::mutate(t, y = recode(x, !!!lookup, .default = "none", .missing = "na")))
})
