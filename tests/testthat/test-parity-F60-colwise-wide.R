# F60 acceptance tests: the colwise verbs (`_at`/`_all`/`_if`) on wide tables.
#
# dplyr builds one dot per (column, function), `<fn>(col)` with a function
# object as the head; colwise_rewrite_dots() (R/colwise.R) turns each into a
# plain call. Warm, that rewrite was ~0.1 ms per dot (it re-resolved the same
# function for every column), so at 200 columns `summarise_all(mean)` was
# ~1.5x `summarise(across(everything(), mean))`. The rewrite must cost about
# what across() pays to resolve `.fns` once, and still give dplyr's results
# for every dot: sequential dots, function lists, extra args, per-column
# checks (desc() on strings), group columns.

for (f60_fn in c("vars", "all_vars", "any_vars", "desc", "starts_with",
                 "mutate_at", "mutate_all", "mutate_if", "transmute_at", "transmute_all",
                 "summarise_at", "summarise_all", "summarise_if",
                 "filter_all", "filter_if", "arrange_at", "arrange_all")) {
  assign(f60_fn, getExportedValue("dplyr", f60_fn))
}
rm(f60_fn)

# w FLOAT64 columns v1..vw with some NAs, plus an INT32 group column g
f60_wide <- function(n = 40L, w = 60L, na = TRUE, seed = 60L) {
  set.seed(seed)
  m <- matrix(round(runif(n * w, -10, 10), 3), n, w)
  if (na) m[sample(length(m), max(1L, length(m) %/% 20L))] <- NA
  df <- as.data.frame(m)
  names(df) <- paste0("v", seq_len(w))
  cbind(data.frame(g = rep_len(1:4, n)), df)
}

f60_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f60_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f60_both(df, pipeline, ...)
}

test_that("wide summarise_all/_at/_if keep dplyr's results, names and order natively", {
  skip_if_no_gpu()
  df <- f60_wide()
  num <- df[-1L]
  f60_native(num, function(d) summarise_all(d, mean))
  f60_native(num, function(d) summarise_all(d, mean, na.rm = TRUE))
  f60_native(num, function(d) summarise_all(d, list(m = mean, s = sd), na.rm = TRUE))
  f60_native(num, function(d) summarise_all(d, list(mean, "max"), na.rm = TRUE))
  f60_native(num, function(d) summarise_at(d, vars(starts_with("v1")), list(lo = min, hi = max), na.rm = TRUE))
  f60_native(num, function(d) summarise_if(d, is.numeric, ~ sum(.x, na.rm = TRUE)))
  f60_native(df, function(d) summarise_all(dplyr::group_by(d, g), mean, na.rm = TRUE),
             arrange_by = "g")
  f60_native(df, function(d) summarise_at(dplyr::group_by(d, g), vars(v1:v40), list(n = ~ sum(!is.na(.x)), m = max), na.rm = FALSE),
             arrange_by = "g")
})

test_that("wide mutate/transmute colwise verbs keep dplyr's results natively", {
  skip_if_no_gpu()
  df <- f60_wide()
  num <- df[-1L]
  f60_native(num, function(d) mutate_all(d, ~ .x + 1))
  f60_native(df, function(d) mutate_if(d, is.double, ~ .x * 2))
  f60_native(num, function(d) mutate_at(d, vars(v2:v50), list(d = ~ .x * 2, h = ~ .x / 2)))
  f60_native(num, function(d) transmute_at(d, vars(v10:v59), abs))
  f60_native(num, function(d) transmute_all(d, list(a = ~ .x - 1, b = round)))
  # grouped: the group column is left out by _all/_if
  f60_native(df, function(d) mutate_all(dplyr::group_by(d, g), ~ .x * 10))
})

test_that("wide colwise dots stay sequential and see the data mask", {
  skip_if_no_gpu()
  num <- f60_wide(na = FALSE)[-1L]
  # each dot sees the columns earlier dots replaced: v2 = v2 + new v1, ...
  f60_native(num, function(d) mutate_all(d, ~ .x + v1))
  f60_native(num, function(d) mutate_at(d, vars(v1:v30), ~ .x - v31 * 2))
  # the same function object for every column, extra args per call
  f60_native(abs(num) + 1, function(d) mutate_at(d, vars(v5:v55), log, base = 2))
})

test_that("two different functions in one list are not confused with each other", {
  skip_if_no_gpu()
  num <- f60_wide(na = FALSE)[-1L]
  k1 <- 2
  k2 <- 3
  f1 <- function(x) x * k1
  f2 <- function(x) x * k2
  f60_both(num, function(d) mutate_all(d, list(a = f1, b = f2)))
  f60_both(num, function(d) summarise_all(d, list(a = min, b = max, c = min)))
  f60_native(num, function(d) mutate_all(d, list(p = ~ .x + 1, q = ~ .x + 2, r = ~ .x + 1)))
})

test_that("wide filter and arrange colwise verbs keep dplyr's results", {
  skip_if_no_gpu()
  df <- f60_wide(n = 60L)
  num <- df[-1L]
  f60_native(num, function(d) filter_all(d, any_vars(. > 9.5)))
  f60_native(num, function(d) filter_if(d, is.numeric, all_vars(!is.na(.) | . > 100)))
  f60_native(num, function(d) arrange_at(d, vars(v1, v2, v3), desc))
  # desc() is checked per column: the STRING column still sorts in the
  # session collation like dplyr
  mixed <- cbind(data.frame(s = sample(c("b", "A", "a", "B"), 60L, TRUE)), num[1:3])
  f60_both(mixed, function(d) arrange_all(d, desc))
})

test_that("0-row wide inputs give dplyr's results", {
  skip_if_no_gpu()
  num <- f60_wide(na = FALSE)[0L, -1L]
  f60_native(num, function(d) mutate_all(d, ~ .x + 1))
  f60_native(num, function(d) summarise_all(d, list(m = max, s = sum)))
})

# --- speed: warm colwise vs its across() equivalent ----------------------------

# Each pair is timed with timing_medians() (interleaved batches, an untimed gc()
# before each one, wall clock) and a near miss is measured again (up to 3
# attempts, 1 s apart; timing_retry()); a miss of 10% or more fails at once.
#
# Batches last about 0.5 s because the earlier method (system.time() around 3
# calls, which runs a gc() first) read one code base two ways: a process read
# mutate_if / across as either about 1.33 or about 1.45, the same mode for every
# repeat in that process. Full-suite readings, 2026-11-09, one per suite: dev
# fb7e7c5 1.440, 1.463, 1.342, 1.333; F98 1.359, 1.319, 1.435, 1.316; F98 without
# test-F98-topk-large.R 1.345, 1.454 (the 1.463 and the 1.454 failed the 1.45
# limit). F98's merge gate read 1.48 and 1.47. What sets the mode was not
# established. One call right after a gc() is 1.36-1.38 in the six suites that
# recorded it (the last two of dev and of F98, both of F98 without that file) and
# in lone processes. Those six suites also timed batches at the settings below:
# 1.354-1.399, dev 1.375 and 1.373. Shorter batches stay split: 0.25 s read
# 1.375-1.412 with the two modes apart, and timing_medians()'s default 0.05 s (one
# or two calls per batch) read 1.37-1.53 over all ten suites. The limit stays 1.45.
test_that("warm colwise verbs on 200 columns cost about what across() costs", {
  skip_if_no_gpu()
  num <- f60_wide(n = 1000L, w = 200L, na = FALSE)[-1L]
  g <- tbl_gpu(num, lazy = FALSE)
  withr::local_options(cuplyr.fallback = "error")

  pairs <- list(
    "summarise_all(mean)" = list(
      limit = 1.25,
      colwise = function() summarise_all(g, mean),
      across = function() dplyr::summarise(g, dplyr::across(dplyr::everything(), mean))
    ),
    "summarise_all(list(m, s))" = list(
      limit = 1.25,
      colwise = function() summarise_all(g, list(m = mean, s = sd)),
      across = function() dplyr::summarise(g, dplyr::across(dplyr::everything(), list(m = mean, s = sd)))
    ),
    "mutate_all(~ .x + 1)" = list(
      limit = 1.25,
      colwise = function() mutate_all(g, ~ .x + 1),
      across = function() dplyr::mutate(g, dplyr::across(dplyr::everything(), ~ .x + 1))
    ),
    # mutate_if also pays dplyr's own per-column predicate loop (tbl_if_vars()
    # pulls every column of the tbl_ptype() proxy), which across(where()) does not
    "mutate_if(is.numeric, ~ .x * 2)" = list(
      limit = 1.45,
      colwise = function() mutate_if(g, is.numeric, ~ .x * 2),
      across = function() dplyr::mutate(g, dplyr::across(dplyr::where(is.numeric), ~ .x * 2))
    )
  )

  for (nm in names(pairs)) {
    p <- pairs[[nm]]
    runs <- timing_retry(
      function() {
        t <- timing_medians(p[c("colwise", "across")], samples = 7L, target = 0.5,
                            label = paste("F60", nm))
        t[["colwise"]] / t[["across"]]
      },
      ok = function(r) r < p$limit,
      retry_if = function(r) r < p$limit * 1.1,
      pause = 1
    )
    expect_lt(runs[[length(runs)]], p$limit,
              label = sprintf("%s / across ratio: %s (attempts)", nm,
                              paste(sprintf("%.3f", unlist(runs)), collapse = " / ")))
  }
})
