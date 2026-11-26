# F61 item 1: recode() with many literal keys runs as one lookup.
#
# Semantics are dplyr 1.2.1's R/recode.R, unchanged from F30 (see
# test-parity-F30-native.R for the small-key shapes):
#   - character .x: key = name; numeric .x: key = as.double(name) or position;
#   - NA (and NaN) rows never match a key and never take .default; only
#     .missing replaces them;
#   - without .default, unmatched values keep .x.
# These tests use thousands of keys, so they exercise whatever single-pass
# lookup replaces the per-key clause chain. Native is pinned with
# options(cuplyr.fallback = "error").

recode <- dplyr::recode

f61r_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f61r_K <- 3000L

f61r_df <- function(n = 400L, seed = 61L) {
  set.seed(seed)
  K <- f61r_K
  s <- sample(c(paste0("k", seq_len(K + 50L)), NA, "", "NA"), n, replace = TRUE)
  i <- sample(c(seq_len(K + 50L), -3L, 0L, NA), n, replace = TRUE)
  d <- sample(c(as.double(seq_len(K + 50L)) / 4, -0, 0, NA, NaN, -2.5), n, replace = TRUE)
  data.frame(s = s, i = i, d = d, g = rep_len(1:7, n), stringsAsFactors = FALSE)
}

f61r_chr_lut <- function() {
  K <- f61r_K
  # dplyr requires every character key to be a non-empty name.
  keys <- c(paste0("k", seq_len(K)), "NA")
  vals <- c(paste0("v", seq_len(K)), "na-string")
  vals[c(5L, 17L)] <- NA_character_
  stats::setNames(vals, keys)
}

test_that("many character keys: unmatched keep x, NA stays NA, NA values allowed", {
  skip_if_no_gpu()
  lut <- f61r_chr_lut()
  f61r_native(f61r_df(), function(t) dplyr::mutate(t, y = recode(s, !!!lut)))
})

test_that("many character keys with .default and .missing", {
  skip_if_no_gpu()
  lut <- f61r_chr_lut()
  f61r_native(f61r_df(), function(t) {
    dplyr::mutate(t,
      y1 = recode(s, !!!lut, .default = "other"),
      y2 = recode(s, !!!lut, .missing = "M"),
      y3 = recode(s, !!!lut, .default = "other", .missing = "M"),
      y4 = recode(s, !!!lut, .default = NA_character_)
    )
  })
})

test_that("many character keys recoded to integer, double and logical", {
  skip_if_no_gpu()
  K <- f61r_K
  keys <- paste0("k", seq_len(K))
  ints <- stats::setNames(seq_len(K) * 3L, keys)
  dbls <- stats::setNames(seq_len(K) / 8, keys)
  lgls <- stats::setNames(rep(c(TRUE, FALSE, NA), length.out = K), keys)
  f61r_native(f61r_df(), function(t) {
    dplyr::mutate(t,
      a = recode(s, !!!ints, .default = -1L),
      b = recode(s, !!!ints, .default = 0L, .missing = -9L),
      c = recode(s, !!!dbls, .default = NA_real_),
      e = recode(s, !!!lgls, .default = NA)
    )
  })
})

test_that("many integer keys: named (incl. negative and zero) and positional", {
  skip_if_no_gpu()
  K <- f61r_K
  named <- stats::setNames(c(seq_len(K) * 2L, 777L, 555L), c(seq_len(K), -3L, 0L))
  pos <- seq_len(K) + 100000L
  to_chr <- stats::setNames(paste0("n", seq_len(K)), seq_len(K))
  f61r_native(f61r_df(), function(t) {
    dplyr::mutate(t,
      a = recode(i, !!!named),
      b = recode(i, !!!pos),
      c = recode(i, !!!to_chr, .default = "other", .missing = "none"),
      e = recode(i, !!!named, .missing = -1L)
    )
  })
})

test_that("many double keys: -0 matches 0, NaN behaves like NA", {
  skip_if_no_gpu()
  K <- f61r_K
  keys <- c(as.double(seq_len(K)) / 4, 0, -2.5)
  named <- stats::setNames(c(seq_len(K) + 0.5, 1000, 2000), as.character(keys))
  pos <- as.double(seq_len(K)) * -1
  f61r_native(f61r_df(), function(t) {
    dplyr::mutate(t,
      a = recode(d, !!!named),
      b = recode(d, !!!named, .default = 7),
      c = recode(d, !!!named, .missing = -1),
      e = recode(d, !!!named, .default = 7, .missing = -1),
      f = recode(d, !!!pos),
      h = recode(d, !!!stats::setNames(paste0("d", seq_len(K)), as.character(keys[seq_len(K)])),
                 .default = "?")
    )
  })
})

test_that("many keys in grouped mutate, filter() and summarise()", {
  skip_if_no_gpu()
  lut <- f61r_chr_lut()
  ints <- stats::setNames(seq_len(f61r_K), paste0("k", seq_len(f61r_K)))
  f61r_native(f61r_df(), function(t) {
    dplyr::mutate(dplyr::group_by(t, g),
      y = recode(s, !!!lut, .default = "other"),
      z = dplyr::n()
    )
  })
  f61r_native(f61r_df(), function(t) {
    dplyr::filter(t, recode(s, !!!lut, .default = "other") != "other")
  })
  f61r_native(f61r_df(), function(t) {
    dplyr::summarise(dplyr::group_by(t, g),
      tot = sum(recode(s, !!!ints, .default = 0L, .missing = 0L))
    )
  }, arrange_by = "g")
})

test_that("many keys on zero rows and one row keep the result type", {
  skip_if_no_gpu()
  lut <- f61r_chr_lut()
  ints <- stats::setNames(seq_len(f61r_K), paste0("k", seq_len(f61r_K)))
  dkeys <- stats::setNames(as.double(seq_len(f61r_K)), seq_len(f61r_K))
  for (df in list(f61r_df()[0, ], f61r_df()[1, ], f61r_df(n = 1L, seed = 3L))) {
    f61r_native(df, function(t) {
      dplyr::mutate(t,
        a = recode(s, !!!lut),
        b = recode(s, !!!ints, .default = 0L),
        c = recode(d, !!!dkeys, .missing = 0)
      )
    })
  }
})

test_that("many keys where every row is NA or unmatched", {
  skip_if_no_gpu()
  lut <- f61r_chr_lut()
  df <- data.frame(s = c(NA, "zzz", NA, "k999999"), d = c(NA, NaN, 1e9, -1e9),
                   stringsAsFactors = FALSE)
  dkeys <- stats::setNames(as.double(seq_len(f61r_K)), seq_len(f61r_K))
  f61r_native(df, function(t) {
    dplyr::mutate(t,
      a = recode(s, !!!lut),
      b = recode(s, !!!lut, .default = "D"),
      c = recode(d, !!!dkeys),
      e = recode(d, !!!dkeys, .default = 0, .missing = 1)
    )
  })
})

test_that("a lazy chain around a many-key recode()", {
  skip_if_no_gpu()
  lut <- f61r_chr_lut()
  back <- stats::setNames(names(lut), lut)
  back <- back[!is.na(names(back))]
  f61r_native(f61r_df(), function(t) {
    t |>
      dplyr::filter(!is.na(i)) |>
      dplyr::mutate(y = recode(s, !!!lut, .default = "other")) |>
      dplyr::filter(y != "other") |>
      dplyr::mutate(y2 = recode(y, !!!back)) |>
      dplyr::arrange(i, y) |>
      dplyr::select(y, y2, i)
  })
})

test_that("shapes F30 sends to the CPU stay on the CPU with many keys", {
  skip_if_no_gpu()
  K <- f61r_K
  # Data-dependent "Unreplaced values treated as NA" warning: no .default and
  # a value type different from x.
  ints <- stats::setNames(seq_len(K), paste0("k", seq_len(K)))
  df <- f61r_df()
  suppressWarnings(
    expect_same_as_dplyr(df, function(t) dplyr::mutate(t, y = recode(s, !!!ints)))
  )
  # Duplicate keys: dplyr keeps the first match.
  dup <- c(stats::setNames(paste0("v", seq_len(K)), paste0("k", seq_len(K))), k1 = "second")
  expect_same_as_dplyr(df, function(t) dplyr::mutate(t, y = recode(s, !!!dup)))
  # Factor .x.
  fdf <- data.frame(f = factor(df$s))
  expect_same_as_dplyr(fdf, function(t) dplyr::mutate(t, y = recode(f, !!!f61r_chr_lut())))
})

# --- timing -----------------------------------------------------------------
# timing_medians(): median of 5 runs after 1 warmup, each run timed whole with
# a device fence around it. A reading within 1.25x of its bound is re-measured
# (up to 3 times) and every attempt goes in the failure label.

f61r_attempts <- function(runs, v) {
  paste(sprintf("%.3f", vapply(runs, function(r) r[[v]], numeric(1))), collapse = " / ")
}

test_that("5000-key recode() on 1e3 rows is no slower than dplyr on a tibble", {
  skip_timing_under_check()
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  K <- 5000L
  set.seed(615)
  n <- 1000L
  df <- data.frame(s = sample(c(paste0("k", seq_len(K + 10L)), NA), n, TRUE),
                   d = sample(c(seq_len(K + 10L), NA), n, TRUE) + 0,
                   stringsAsFactors = FALSE)
  lut_s <- stats::setNames(paste0("v", seq_len(K)), paste0("k", seq_len(K)))
  lut_d <- stats::setNames(as.double(seq_len(K)) / 2, seq_len(K))
  gt <- tbl_gpu(df)
  tb <- tibble::as_tibble(df)

  fns <- list(
    gpu_s = function() dplyr::mutate(gt, z = recode(s, !!!lut_s)),
    cpu_s = function() dplyr::mutate(tb, z = recode(s, !!!lut_s)),
    gpu_d = function() dplyr::mutate(gt, z = recode(d, !!!lut_d, .default = -1)),
    cpu_d = function() dplyr::mutate(tb, z = recode(d, !!!lut_d, .default = -1))
  )
  # worst ratio of gpu to its bound max(cpu, 0.05); <= 1 passes
  rel <- function(m) c(s = m[["gpu_s"]] / max(m[["cpu_s"]], 0.05),
                       d = m[["gpu_d"]] / max(m[["cpu_d"]], 0.05))
  runs <- timing_retry(
    function() {
      m <- timing_medians(fns, samples = 5L, warmup = 1L, label = "F61 recode 5000 keys 1e3 rows")
      list(m = m, rel = rel(m))
    },
    ok = function(r) all(r$rel <= 1),
    retry_if = function(r) all(r$rel <= 1.25),
    attempts = 3L
  )
  last <- runs[[length(runs)]]
  for (v in c("s", "d")) {
    expect_lte(last$m[[paste0("gpu_", v)]], max(last$m[[paste0("cpu_", v)]], 0.05),
               label = sprintf("recode (%s) on gpu vs max(dplyr tibble, 0.05 s); gpu/bound over attempts: %s",
                               v, paste(sprintf("%.2f", vapply(runs, function(r) r$rel[[v]], numeric(1))),
                                        collapse = " / ")))
  }
})

test_that("5000-key recode() on 1e6 rows takes well under a second", {
  skip_timing_under_check()
  # dplyr itself needs 24-60 s here (one pass per key), so the reference is
  # absolute: dev took 2.1 s (STRING) and 4.6 s (FLOAT64 with .default).
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  K <- 5000L
  set.seed(616)
  n <- 1e6
  df <- data.frame(s = sample(c(paste0("k", seq_len(K + 10L)), NA), n, TRUE),
                   d = sample(c(seq_len(K + 10L), NA), n, TRUE) + 0,
                   stringsAsFactors = FALSE)
  lut_s <- stats::setNames(paste0("v", seq_len(K)), paste0("k", seq_len(K)))
  lut_d <- stats::setNames(as.double(seq_len(K)) / 2, seq_len(K))
  gt <- tbl_gpu(df)
  fns <- list(
    string = function() dplyr::mutate(gt, z = recode(s, !!!lut_s)),
    double = function() dplyr::mutate(gt, z = recode(d, !!!lut_d, .default = -1))
  )
  runs <- timing_retry(
    function() timing_medians(fns, samples = 5L, warmup = 1L, label = "F61 recode 5000 keys 1e6 rows"),
    ok = function(m) all(m < 0.5),
    retry_if = function(m) all(m < 0.5 * 1.25),
    attempts = 3L
  )
  m <- runs[[length(runs)]]
  for (v in names(fns)) {
    expect_lt(m[[v]], 0.5, label = sprintf("recode (%s) median at 1e6 rows (attempts: %s s)", v,
                                           f61r_attempts(runs, v)))
  }
})
