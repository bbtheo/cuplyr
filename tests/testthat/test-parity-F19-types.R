# F19 attempt 2 acceptance tests: colwise verbs over columns whose type the
# GPU can't reduce or scan the way R does keep dplyr's results, types and
# warnings (attempt 1 round-4 review B1/B2). The colwise rewrite emits plain
# calls; the plain verbs' type gate (F48, ir_type_gate()) decides native vs
# CPU, so these shapes go to the CPU and the all-numeric ones stay native.
#
# dplyr 1.2.1, verified on plain tibbles (scratchpad F19-a2/probe1.R):
# - summarise_all(<x dbl, s chr>, mean): x = 2, s = NA_real_ + warning
#   "argument is not numeric or logical: returning NA"; sd: s = NA_real_ +
#   "NAs introduced by coercion"; median: s = "b" (character), no warning.
# - mutate_all(<x dbl, s chr>, mean): s = NA_real_ x3 + the mean() warning.
# - summarise_all(<b lgl, i int>, list(mn = min, mx = max, sm = sum,
#   md = median)): min/max/sum are integer; b_md is logical, i_md integer.
# - mutate_at(<s chr>, vars(s), cummax): s = NA_real_ + "NAs introduced by
#   coercion"; cummax() of a logical is integer.

for (f19t_fn in c("vars", "mutate_at", "mutate_all", "summarise_all", "summarise_at")) {
  assign(f19t_fn, getExportedValue("dplyr", f19t_fn))
}
rm(f19t_fn)

f19t_mixed <- function() {
  data.frame(g = c(1, 1, 2), x = c(1, 2, 3), s = c("a", "b", "c"), stringsAsFactors = FALSE)
}

f19t_lgl_int <- function() {
  data.frame(b = c(TRUE, FALSE, TRUE), i = c(1L, 5L, 3L))
}

# Run `pipeline` on a tibble and on a tbl_gpu (eager and lazy). Values,
# column names, column classes and grouping must match exactly (the oracle
# helpers let integer vs double through), and so must the warnings' text
# when `warn` is given (each side must warn with a message matching it).
f19t_check <- function(df, pipeline, warn = NULL) {
  run <- function(input) {
    msgs <- character()
    res <- withCallingHandlers(
      {
        out <- pipeline(input)
        groups <- if (is_tbl_gpu(out)) out$groups else dplyr::group_vars(out)
        list(data = dplyr::ungroup(tibble::as_tibble(collect(out))), groups = groups)
      },
      warning = function(w) {
        msgs <<- c(msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) invokeRestart("muffleMessage")
    )
    res$warnings <- msgs
    res
  }
  oracle <- run(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    gpu <- run(tbl_gpu(df, lazy = lazy))
    expect_identical(gpu$groups, oracle$groups, info = label)
    expect_identical(names(gpu$data), names(oracle$data), info = label)
    expect_identical(lapply(gpu$data, class), lapply(oracle$data, class), info = label)
    expect_equal(gpu$data, oracle$data, info = label)
    if (is.null(warn)) {
      expect_length(gpu$warnings, length(oracle$warnings))
    } else {
      expect_match(paste(oracle$warnings, collapse = "\n"), warn, fixed = TRUE, info = "oracle")
      expect_match(paste(gpu$warnings, collapse = "\n"), warn, fixed = TRUE, info = label)
    }
  }
}

f19t_not_numeric <- "argument is not numeric or logical: returning NA"
f19t_coercion <- "NAs introduced by coercion"

test_that("summarise_all() of numeric reducers over a character column gives dplyr's NA and warning", {
  skip_if_no_gpu()
  df <- f19t_mixed()
  f19t_check(df[c("x", "s")], function(d) summarise_all(d, mean), warn = f19t_not_numeric)
  f19t_check(df[c("x", "s")], function(d) summarise_all(d, sd), warn = f19t_coercion)
  f19t_check(df[c("x", "s")], function(d) summarise_all(d, ~ mean(.x)), warn = f19t_not_numeric)
  f19t_check(df, function(d) summarise_at(d, vars(x, s), mean), warn = f19t_not_numeric)
})

test_that("summarise_all(median) of a character column is its middle string", {
  skip_if_no_gpu()
  df <- f19t_mixed()
  f19t_check(df[c("x", "s")], function(d) summarise_all(d, median))
  f19t_check(df[c("x", "s")], function(d) summarise_at(d, vars(s), median))
})

test_that("grouped summarise_all() over a character column matches dplyr", {
  skip_if_no_gpu()
  df <- f19t_mixed()
  f19t_check(df, function(d) summarise_all(dplyr::group_by(d, g), ~ mean(.)), warn = f19t_not_numeric)
  f19t_check(df, function(d) summarise_all(dplyr::group_by(d, g), mean), warn = f19t_not_numeric)
  # (a grouped median of strings is dplyr's own error when group sizes differ)
})

test_that("mutate_all(mean) over a character column gives dplyr's NA and warning", {
  skip_if_no_gpu()
  dm <- data.frame(x = c(1, 2, 3), s = c("b", "a", "c"), stringsAsFactors = FALSE)
  f19t_check(dm, function(d) mutate_all(d, mean), warn = f19t_not_numeric)
  f19t_check(dm, function(d) mutate_at(d, vars(x, s), list(m = mean)), warn = f19t_not_numeric)
  dg <- data.frame(g = c(1, 1, 2), x = c(1, 2, 3), s = c("b", "a", "c"), stringsAsFactors = FALSE)
  f19t_check(dg, function(d) suppressMessages(mutate_all(dplyr::group_by(d, g), mean)),
             warn = f19t_not_numeric)
})

test_that("min/max/sum/median of logical and integer columns keep dplyr's types", {
  skip_if_no_gpu()
  d3 <- f19t_lgl_int()
  f19t_check(d3, function(d) summarise_all(d, list(mn = min, mx = max, sm = sum, md = median)))
  f19t_check(d3, function(d) summarise_all(d, median))
  d3na <- data.frame(b = c(TRUE, FALSE, TRUE, NA), i = c(1L, 5L, 3L, NA))
  f19t_check(d3na, function(d) summarise_all(d, list(md = median), na.rm = TRUE))
  # a grouped integer median is integer or double depending on group sizes
  dgi <- data.frame(g = c(1, 1, 2), i = c(1L, 2L, 3L))
  f19t_check(dgi, function(d) summarise_all(dplyr::group_by(d, g), median))
})

test_that("cummax()/cummin() of a character or logical column keep dplyr's result", {
  skip_if_no_gpu()
  dm <- data.frame(x = c(1, 2, 3), s = c("b", "a", "c"), stringsAsFactors = FALSE)
  f19t_check(dm, function(d) mutate_at(d, vars(s), cummax), warn = f19t_coercion)
  f19t_check(dm, function(d) mutate_at(d, vars(s), cummin), warn = f19t_coercion)
  f19t_check(f19t_lgl_int(), function(d) mutate_at(d, vars(b), cummax))
})

test_that("the same reducers over numeric columns stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  num <- data.frame(x = c(1.5, 2, NA, 4, -1), z = c(3, 1, 2, 5, 4))
  f19t_check(num, function(d) summarise_all(d, list(mean, median, sd), na.rm = TRUE))
  f19t_check(num, function(d) summarise_all(d, list(mn = min, mx = max, sm = sum), na.rm = TRUE))
  f19t_check(num["z"], function(d) mutate_all(d, cummax))
  f19t_check(num["z"], function(d) mutate_all(d, mean))
  dg <- data.frame(g = c(1L, 1L, 2L, 2L, 3L), x = c(1.5, 2, NA, 4, -1), y = c(10L, 20L, NA, 40L, 5L))
  f19t_check(dg, function(d) summarise_all(dplyr::group_by(d, g), list(m = mean, s = sum), na.rm = TRUE))
})

# Round 1 review (F19-a2/reviews/round-1.md) B1-B4. In R, max()/min() and
# unary minus of a logical are integer, round() of a logical is double.

test_that("window max()/min() and unary minus of a logical column are integer like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1L, 1L, 2L), b = c(FALSE, FALSE, TRUE), bn = c(TRUE, NA, FALSE))
  f19t_check(df, function(d) mutate_at(d, vars(b), max))
  f19t_check(df, function(d) dplyr::mutate_if(d, is.logical, max))
  f19t_check(df, function(d) mutate_at(d, vars(b, bn), list(mx = max, mn = min), na.rm = TRUE))
  f19t_check(df, function(d) mutate_at(dplyr::group_by(d, g), vars(b), ~ max(.)))
  f19t_check(df, function(d) mutate_at(d, vars(b), ~ -.))
  f19t_check(df, function(d) mutate_at(dplyr::group_by(d, g), vars(b, bn), function(v) -v))
})

test_that("round() of a logical column is double like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(b = c(FALSE, FALSE, TRUE), x = c(1.5, 2, 3))
  f19t_check(df, function(d) mutate_all(d, round, 1))
  f19t_check(df, function(d) mutate_at(d, vars(b), round))
})

test_that("quantile() with a non-default type keeps dplyr's result", {
  skip_if_no_gpu()
  # names = FALSE: a GPU column carries no names attribute (dplyr's would be
  # "50%"), a separate, pre-existing gap
  dq <- data.frame(x = c(1, 2, 3, 4, 10), y = c(5, 1, 4, 2, 3))
  f19t_check(dq, function(d) summarise_all(d, quantile, probs = 0.5, type = 6, names = FALSE))
  f19t_check(dq, function(d) summarise_at(d, vars(x), quantile, probs = 0.9, type = 1, names = FALSE))
})

test_that("an earlier dot named like a formula placeholder shadows it, as in dplyr's data mask", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4))
  f19t_check(df, function(d) mutate_at(d, vars(x), list(.x = ~ 1, y = ~ .x + 1)))
  f19t_check(df, function(d) mutate_at(d, vars(x), list(. = ~ 2, y = ~ . * 3)))
})

# Round 2 review (F19-a2/reviews/round-2.md) B1 and N1: each reducer takes
# only its own spelling of the NA flag (na.rm for base R's, na_rm for
# dplyr's first()/last()/nth()). The other spelling is one more `...`
# value for sum()/max()/min() (TRUE counts as 1), swallowed by mean(), and
# an "unused argument" error for sd() and first().

f19t_both_error <- function(df, pipeline, pattern) {
  expect_error(pipeline(tibble::as_tibble(df)), pattern, info = "oracle")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(pipeline(tbl_gpu(df, lazy = lazy))), pattern,
                 info = if (lazy) "lazy" else "eager")
  }
}

test_that("a reducer given the other spelling of the NA flag computes what dplyr computes", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1L, 1L, 2L), x = c(1, 2, NA), y = c(1L, 2L, 3L), z = c(0.1, 0.5, 0.2))
  f19t_check(df[-1], function(d) summarise_all(d, sum, na_rm = TRUE))
  f19t_check(df, function(d) summarise_at(d, vars(x, z), mean, na_rm = TRUE))
  f19t_check(df, function(d) summarise_at(d, vars(z), list(mx = max, mn = min), na_rm = TRUE))
  f19t_check(df, function(d) summarise_all(dplyr::group_by(d, g), mean, na_rm = TRUE))
  f19t_check(df, function(d) dplyr::summarise(d, s = sum(y, na_rm = TRUE), m = mean(x, na_rm = TRUE)))
  # dplyr's own first()/last() take na_rm
  f19t_check(df, function(d) summarise_at(d, vars(x), dplyr::last, na_rm = TRUE))
})

test_that("na.rm given as a number is base R's, not an error", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, NA), y = c(1L, 2L, 3L), z = c(0.1, 0.5, 0.2))
  f19t_check(df, function(d) summarise_all(d, sum, na.rm = 1))
})

test_that("an NA flag a reducer doesn't take is dplyr's error", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, NA), y = c(1L, 2L, 3L))
  f19t_both_error(df, function(d) dplyr::summarise_if(d, is.numeric, sd, na_rm = TRUE), "unused argument")
  f19t_both_error(df, function(d) summarise_at(d, vars(x), dplyr::first, na.rm = TRUE), "unused argument")
})

# Round 3 review (F19-a2/reviews/round-3.md) B1: R matches named arguments
# by name, so extra arguments given out of formal order (or a name that
# belongs after `...`, like coalesce()'s .ptype) must not be read by
# position.

# dplyr is imported, not attached, in the test environment; colwise verbs
# know a function value by its name in the caller's environment
for (f19t_fn in c("between", "if_else", "near", "coalesce", "mutate_if")) {
  assign(f19t_fn, getExportedValue("dplyr", f19t_fn))
}
rm(f19t_fn)

f19t_args_df <- function() {
  data.frame(g = c(1L, 1L, 2L, 2L, 3L, 3L), x = c(0.5, 1, 2, 3, 4, NA),
             i = c(5L, NA, 2L, 7L, 1L, 3L), b = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE))
}

test_that("named extra arguments out of formal order are matched by name like dplyr", {
  skip_if_no_gpu()
  df <- f19t_args_df()
  f19t_check(df, function(d) mutate_at(d, vars(x), between, right = 3, left = 1))
  f19t_check(df, function(d) mutate_at(d, vars(b), if_else, false = 0, true = 1))
  f19t_check(df, function(d) mutate_if(d, is.logical, if_else, false = "no", true = "yes"))
  f19t_check(df, function(d) mutate_at(d, vars(x), near, tol = 0.6, y = 3))
  f19t_check(df, function(d) summarise_at(d, vars(x), ~ sum(between(., right = 3, left = 1), na.rm = TRUE)))
  f19t_check(df, function(d) mutate_at(d, vars(x), coalesce, .ptype = 0))
  f19t_check(df, function(d) dplyr::mutate(d, y = between(x, right = 3, left = 1)))
  f19t_check(df, function(d) mutate_at(dplyr::group_by(d, g), vars(x), between, right = 3, left = 1))
})

test_that("named extra arguments in formal order stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f19t_args_df()
  f19t_check(df, function(d) mutate_at(d, vars(x), between, left = 1, right = 3))
  f19t_check(df, function(d) mutate_all(dplyr::select(d, x, i), round, digits = 1))
  f19t_check(df, function(d) mutate_at(d, vars(b), if_else, true = 1, false = 0))
  f19t_check(df, function(d) mutate_at(d, vars(x), log, base = 2))
})
