# The IR reads a registry call's arguments by position, so a call whose named
# arguments R would bind differently (reordered, partial, unknown, or after
# `...`) goes to dplyr, and so does a positional argument that R would put in
# `...` or reject. In-order names stay native. (F19-a2 round-3 review B1: the
# colwise rewrite emits `between(x, right = 3, left = 1)`; so can a user.)
#
# dplyr 1.2.1 / base R facts these tests rely on (scratchpad/coder-r4):
# - between(x, left, right, ..., ptype); if_else(condition, true, false,
#   missing, ..., ptype, size); near(x, y, tol); coalesce(..., .ptype, .size);
#   na_if(x, y); round(x, digits, ...); log(x, base); xor(x, y).
# - Arithmetic primitives ignore names (`-`(e2 = 1, e1 = 5) is -4);
#   sqrt()/abs()/exp()/is.na() reject a name other than `x`; mean(foo = x)
#   errors; round(x, 1, 2) and log(x, 2, 3) are "unused argument" errors;
#   n_distinct(x = x) errors (dots must be unnamed); desc(foo = x) errors,
#   except as an arrange() dot, whose desc() head dplyr strips by position.

for (ina_fn in c("between", "if_else", "near", "coalesce", "na_if", "lag", "lead",
                 "case_when", "first", "nth", "ntile", "min_rank", "cummean",
                 "n_distinct", "desc", "vars", "mutate_at")) {
  assign(ina_fn, getExportedValue("dplyr", ina_fn))
}
rm(ina_fn)

ina_df <- function() {
  data.frame(g = c(1L, 1L, 2L, 2L, 3L, 3L), x = c(0.5, 1, 2, 3, 4, NA),
             i = c(5L, NA, 2L, 7L, 1L, 3L), b = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE),
             s = c("a", "b", "ab", NA, "ca", "bb"), stringsAsFactors = FALSE)
}

# `pipeline` on a tibble and on a tbl_gpu, eager and lazy: same names, column
# classes and values (the oracle helpers let integer vs double through).
ina_check <- function(df, pipeline) {
  run <- function(input) {
    out <- suppressWarnings(suppressMessages(pipeline(input)))
    dplyr::ungroup(tibble::as_tibble(collect(out)))
  }
  oracle <- run(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    gpu <- run(tbl_gpu(df, lazy = lazy))
    expect_identical(names(gpu), names(oracle), info = label)
    expect_identical(lapply(gpu, class), lapply(oracle, class), info = label)
    expect_equal(gpu, oracle, info = label)
  }
}

# dplyr errors, and so does cuplyr, eager and lazy (with dplyr's message).
ina_both_error <- function(df, pipeline, pattern) {
  expect_error(pipeline(tibble::as_tibble(df)), pattern, info = "oracle")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(suppressWarnings(suppressMessages(collect(pipeline(tbl_gpu(df, lazy = lazy))))),
                 pattern, info = if (lazy) "lazy" else "eager")
  }
}

# Same as ina_check(), and the GPU side may not fall back.
ina_native <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  ina_check(df, pipeline)
}

test_that("reordered named arguments are matched by name in plain mutate()", {
  skip_if_no_gpu()
  df <- ina_df()
  ina_check(df, function(d) dplyr::mutate(d, y = between(x, right = 3, left = 1)))
  ina_check(df, function(d) dplyr::mutate(d, y = if_else(b, false = 0, true = 1)))
  ina_check(df, function(d) dplyr::mutate(d, y = if_else(false = "no", true = "yes", condition = b)))
  ina_check(df, function(d) dplyr::mutate(d, y = near(x, tol = 0.6, y = 3)))
  ina_check(df, function(d) dplyr::mutate(d, y = round(digits = 1, x = x)))
  ina_check(df, function(d) dplyr::mutate(d, y = log(base = 2, x = x)))
  ina_check(df, function(d) dplyr::mutate(d, y = na_if(y = 2, x = x)))
  ina_check(df, function(d) dplyr::mutate(d, y = `%in%`(table = c(1, 2), x = x)))
  ina_check(df, function(d) dplyr::mutate(d, y = xor(y = b, x = !b)))
})

test_that("partial and post-dots names are dplyr's, not read by position", {
  skip_if_no_gpu()
  df <- ina_df()
  # partial names R completes
  ina_check(df, function(d) dplyr::mutate(d, y = round(x, d = 1)))
  ina_check(df, function(d) dplyr::mutate(d, y = log(x, b = 2)))
  ina_check(df, function(d) dplyr::mutate(d, y = near(x, 3, t = 0.6)))
  ina_check(df, function(d) dplyr::mutate(d, y = between(x, l = 1, r = 3)))
  # arguments after `...` change the type or are not data
  ina_check(df, function(d) dplyr::mutate(d, y = coalesce(x, .ptype = 0)))
  ina_check(df, function(d) dplyr::mutate(d, y = coalesce(i, 0L, .ptype = double())))
  ina_check(df, function(d) dplyr::mutate(d, y = if_else(b, 1, 0, ptype = integer())))
  ina_check(df, function(d) dplyr::mutate(d, y = between(x, 1, 3, ptype = double())))
})

test_that("unknown names and extra positional arguments give dplyr's errors", {
  skip_if_no_gpu()
  df <- ina_df()
  ina_both_error(df, function(d) dplyr::mutate(d, y = sqrt(foo = x)), "does not match")
  ina_both_error(df, function(d) dplyr::mutate(d, y = is.na(foo = x)), "does not match")
  ina_both_error(df, function(d) dplyr::mutate(d, y = mean(foo = x)), "missing")
  ina_both_error(df, function(d) dplyr::mutate(d, y = round(x, foo = 1)), "unused argument")
  ina_both_error(df, function(d) dplyr::mutate(d, y = round(x, 1, 2)), "unused argument")
  ina_both_error(df, function(d) dplyr::mutate(d, y = log(x, 2, 3)), "unused argument")
  ina_both_error(df, function(d) dplyr::summarise(d, n = n_distinct(x = x)), "by position")
  ina_both_error(df, function(d) dplyr::mutate(d, y = lag(x, order_by = desc(foo = i))),
                 "unused argument")
  ina_both_error(df, function(d) mutate_at(d, vars(x), round, digits = 1, foo = 2), "unused argument")
})

test_that("reordered names are matched by name in filter(), summarise() and across()", {
  skip_if_no_gpu()
  df <- ina_df()
  ina_check(df, function(d) dplyr::filter(d, between(x, right = 3, left = 1)))
  ina_check(df, function(d) dplyr::filter(d, if_else(b, false = TRUE, true = FALSE)))
  ina_check(df, function(d) dplyr::summarise(d, y = sum(between(x, right = 3, left = 1), na.rm = TRUE)))
  ina_check(df, function(d) dplyr::summarise(dplyr::group_by(d, g),
                                             y = sum(near(x, tol = 0.6, y = 3), na.rm = TRUE)))
  ina_check(df, function(d) dplyr::mutate(d, dplyr::across(x, ~ between(.x, right = 3, left = 1))))
  ina_check(df, function(d) dplyr::mutate(dplyr::group_by(d, g),
                                          dplyr::across(c(x, i), ~ round(digits = 1, x = .x))))
})

test_that("in-order named arguments stay on the GPU in mutate()", {
  skip_if_no_gpu()
  df <- ina_df()
  ina_native(df, function(d) dplyr::mutate(d, y = round(x, digits = 1)))
  ina_native(df, function(d) dplyr::mutate(d, y = round(x = x, digits = 1)))
  ina_native(df, function(d) dplyr::mutate(d, y = log(x, base = 2)))
  ina_native(df, function(d) dplyr::mutate(d, y = if_else(b, 1, 0, missing = 2)))
  ina_native(df, function(d) dplyr::mutate(d, y = if_else(b, true = 1, false = 0)))
  ina_native(df, function(d) dplyr::mutate(d, y = if_else(condition = b, true = 1, false = 0)))
  ina_native(df, function(d) dplyr::mutate(d, y = between(x, left = 1, right = 3)))
  ina_native(df, function(d) dplyr::mutate(d, y = between(x, 1, right = 3)))
  ina_native(df, function(d) dplyr::mutate(d, y = near(x, 3, tol = 0.6)))
  ina_native(df, function(d) dplyr::mutate(d, y = na_if(x, y = 2)))
  ina_native(df, function(d) dplyr::mutate(d, y = xor(x = b, y = b)))
  ina_native(df, function(d) dplyr::mutate(d, y = coalesce(x, 0)))
  ina_native(df, function(d) dplyr::mutate(d, y = x %in% c(1, 2)))
  ina_native(df, function(d) dplyr::mutate(d, y = x - 1, z = -x))
  ina_native(df, function(d) dplyr::mutate(d, y = case_when(x > 1 ~ 1, .default = 0)))
  ina_native(df, function(d) dplyr::mutate(d, y = lag(x, n = 2, default = 0), z = lead(x, default = 0)))
  ina_native(df, function(d) dplyr::mutate(d, y = first(x, order_by = i), z = nth(x, n = 2)))
  ina_native(df, function(d) dplyr::mutate(d, y = ntile(x, n = 2), z = ntile(n = 2)))
  ina_native(df, function(d) dplyr::mutate(d, y = mean(x = x), z = max(x = i)))
  ina_native(df, function(d) dplyr::mutate(d, y = sqrt(x = x), z = abs(x = x), w = is.na(x = x)))
  ina_native(df, function(d) dplyr::mutate(d, y = cumsum(x = i), z = min_rank(x = x)))
  ina_native(df, function(d) dplyr::mutate(d, y = cummean(x = x)))
  ina_native(df, function(d) dplyr::mutate(d, y = sum(x, na.rm = TRUE)))
  ina_native(df, function(d) dplyr::mutate(d, y = grepl(pattern = "a", x = s),
                                           z = substr(s, start = 1, stop = 1)))
  ina_native(df, function(d) dplyr::mutate(d, y = nchar(s, type = "chars"),
                                           z = trimws(s, which = "left")))
  ina_native(df, function(d) dplyr::mutate(d, y = paste(s, s, sep = "-"), z = toupper(x = s)))
})

test_that("in-order named arguments stay on the GPU in filter(), summarise(), across() and colwise", {
  skip_if_no_gpu()
  df <- ina_df()
  ina_native(df, function(d) dplyr::filter(d, between(x, left = 1, right = 3)))
  ina_native(df, function(d) dplyr::filter(d, if_else(b, true = TRUE, false = FALSE)))
  ina_native(df, function(d) dplyr::summarise(d, y = sum(x, na.rm = TRUE),
                                              q = quantile(x, probs = 0.5, na.rm = TRUE, names = FALSE),
                                              n2 = nth(x, n = 2), m = mean(x = x),
                                              nd = n_distinct(x)))
  ina_native(df, function(d) dplyr::summarise(dplyr::group_by(d, g),
                                              y = sum(round(x, digits = 1), na.rm = TRUE),
                                              z = sum(between(x, left = 1, right = 3), na.rm = TRUE)))
  ina_native(df, function(d) dplyr::mutate(d, dplyr::across(c(x, i), ~ round(.x, digits = 1))))
  ina_native(df, function(d) mutate_at(d, vars(x), log, base = 2))
  ina_native(df, function(d) mutate_at(d, vars(b), if_else, 1, 0, missing = 2))
  ina_native(df, function(d) mutate_at(d, vars(x), lag, n = 2, default = 0))
  ina_native(df, function(d) dplyr::arrange(d, desc(x), i))
  ina_native(df, function(d) dplyr::arrange(d, desc(foo = x), i))
})
