# F18 attempt 2 acceptance tests: THE RULE for factors on the GPU.
#
# A factor operand (a DICTIONARY32 column, or a factor value from the
# environment) may stay on the GPU only
#   1. as a bare column reference: plain copy, key, or sort key on its own;
#   2. inside is.na(<factor column>) / n_distinct(<factor column>);
#   3. in == or != between two factor columns with identical level vectors
#      (and <, <=, >, >= only when both are also ordered). %in% between two
#      factor columns falls back (the GPU %in% takes only a literal set).
# Everything else that touches a factor goes to the CPU fallback, so dplyr's
# own labels, warnings and errors come out. See
# scratchpad/parity/features/F18-a2.md "## Design".

# Labels look numeric and disagree with the codes: in `f`, code 1 is "10".
# f3 has the same level SET as f in another order; f4 has a different set.
f18r_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")),
    f2 = factor(c("1", "2", "10", NA, "2", "10"), levels = c("10", "1", "2")),
    f3 = factor(c("1", "2", "10", NA, "2", "10"), levels = c("1", "2", "10")),
    f4 = factor(c("1", "2", "10", NA, "2", "7"), levels = c("1", "2", "10", "7")),
    o = factor(c("lo", "hi", "mid", NA, "hi", "lo"), levels = c("lo", "mid", "hi"), ordered = TRUE),
    o2 = factor(c("hi", "hi", "lo", "mid", NA, "lo"), levels = c("lo", "mid", "hi"), ordered = TRUE),
    n = c(1L, 2L, 3L, 4L, 2L, 10L),
    s = c("1", "2", "10", "x", NA, "1"),
    g = c("a", "a", "b", "b", "b", "a")
  )
}

f18r_same <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# dplyr warns (pattern) and returns a result: the GPU must warn the same way
# and return the same result, eager and lazy.
f18r_warns <- function(df, pipeline, pattern, arrange_by = NULL) {
  expect_warning(expected <- pipeline(df), pattern)
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_warning(out <- dplyr::collect(pipeline(g)), pattern)
    expect_equal(oracle_normalize(out, arrange_by), oracle_normalize(expected, arrange_by),
                 info = paste("lazy =", lazy))
  }
}

# dplyr errors (pattern): so must the GPU, eager and lazy.
f18r_errors <- function(df, pipeline, pattern) {
  expect_error(pipeline(df), pattern)
  for (lazy in c(FALSE, TRUE)) {
    expect_error(dplyr::collect(pipeline(tbl_gpu(df, lazy = lazy))), pattern)
  }
}

# ---------------------------------------------------------------------------
# Rule 3: factor vs factor
# ---------------------------------------------------------------------------

test_that("F18 rule: factors with the same level set in another order compare labels (N8)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::mutate(d, y = f == f3, z = f != f3))
  f18r_same(df, function(d) dplyr::filter(d, f == f3))
  f18r_same(df, function(d) dplyr::filter(d, f %in% f3))
  f18r_same(df, function(d) dplyr::filter(d, f %in% f4))
  f18r_same(df, function(d) dplyr::summarise(dplyr::group_by(d, g), k = sum(f == f3, na.rm = TRUE)),
            arrange_by = "g")
  f18r_same(df, function(d) dplyr::mutate(dplyr::group_by(d, g), y = f != f3))
})

test_that("F18 rule: factors with different level sets error like base R", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_errors(df, function(d) dplyr::mutate(d, y = f == f4), "level sets of factors are different")
  f18r_errors(df, function(d) dplyr::filter(d, f != f4), "level sets of factors are different")
})

test_that("F18 rule: ordering comparisons of unordered factors warn and give NA", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_warns(df, function(d) dplyr::mutate(d, y = f < f2), "not meaningful for factors")
  f18r_warns(df, function(d) dplyr::filter(d, f >= f2), "not meaningful for factors")
})

test_that("F18 rule: ordered factors with identical levels compare on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::filter(d, o < o2))
  f18r_same(df, function(d) dplyr::mutate(d, y = o >= o2, z = o == o2))
})

# ---------------------------------------------------------------------------
# Arithmetic and is.na() of a computed factor (N12, N15)
# ---------------------------------------------------------------------------

test_that("F18 rule: factor arithmetic warns and gives NA, never codes (N12)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_warns(df, function(d) dplyr::mutate(d, y = f / 2), "not meaningful for factors")
  f18r_warns(df, function(d) dplyr::mutate(d, y = -f), "not meaningful for factors")
  f18r_warns(df, function(d) dplyr::arrange(d, -f), "not meaningful for factors")
  f18r_warns(df, function(d) dplyr::arrange(d, -o, n), "not meaningful for ordered factors")
  f18r_warns(df, function(d) dplyr::mutate(dplyr::group_by(d, g), y = f * 2),
             "not meaningful for factors")
})

test_that("F18 rule: is.na() of a value computed from a factor is not exempt (N15)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_warns(df, function(d) dplyr::filter(d, is.na(f - 1) == TRUE), "not meaningful for factors")
  f18r_warns(df, function(d) dplyr::mutate(d, y = is.na(f + 0)), "not meaningful for factors")
  f18r_warns(df, function(d) dplyr::summarise(d, k = sum(is.na(f * 2))), "not meaningful for factors")
})

# ---------------------------------------------------------------------------
# Factor values from the environment (B4)
# ---------------------------------------------------------------------------

test_that("F18 rule: factor values from the environment use their labels (B4)", {
  skip_if_no_gpu()
  df <- f18r_df()
  fx <- factor("2", levels = c("10", "1", "2"))
  keep <- factor(c("1", "x"))
  f18r_same(df, function(d) dplyr::filter(d, s %in% keep))
  f18r_same(df, function(d) dplyr::filter(d, n == fx))
  f18r_same(df, function(d) dplyr::filter(d, f == fx))
  f18r_same(df, function(d) dplyr::filter(d, f %in% keep))
  f18r_same(df, function(d) dplyr::mutate(d, y = fx))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::lag(f, default = fx)))
  f18r_same(df, function(d) dplyr::mutate(d, y = s %in% keep, z = n != fx))
})

test_that("F18 rule: na_if() with a factor value errors like dplyr (B4)", {
  skip_if_no_gpu()
  df <- f18r_df()
  fx <- factor("2", levels = c("10", "1", "2"))
  f18r_errors(df, function(d) dplyr::mutate(d, y = dplyr::na_if(n, fx)), "Can't convert")
  f18r_errors(df, function(d) dplyr::filter(d, dplyr::na_if(n, fx) > 1), "Can't convert")
})

# ---------------------------------------------------------------------------
# Aggregates and helpers over a factor
# ---------------------------------------------------------------------------

test_that("F18 rule: summarise of a bare factor keeps the factor or errors like dplyr", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::summarise(d, a = dplyr::first(f), b = dplyr::last(f)))
  f18r_same(df, function(d) dplyr::summarise(d, a = max(o), b = min(o, na.rm = TRUE)))
  f18r_same(df, function(d) dplyr::summarise(dplyr::group_by(d, g),
    a = dplyr::first(f), b = max(o, na.rm = TRUE), k = dplyr::n_distinct(f)
  ), arrange_by = "g")
  f18r_errors(df, function(d) dplyr::summarise(d, m = max(f)), "not meaningful for factors")
  f18r_errors(df, function(d) dplyr::count(d, g, wt = f), "not meaningful for factors")
  f18r_errors(df, function(d) dplyr::add_count(d, g, wt = f), "not meaningful for factors")
})

test_that("F18 rule: helpers and windows over a factor match dplyr", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::coalesce(f, f2)))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::if_else(n > 2, f, f2)))
  f18r_same(df, function(d) dplyr::mutate(d, y = as.integer(f)))
  f18r_same(df, function(d) dplyr::mutate(d, y = paste0(f, "x")))
  f18r_same(df, function(d) dplyr::filter(d, paste0(f) == "10"))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::case_when(f == f3 ~ "same", .default = "diff")))
  f18r_same(df, function(d) dplyr::mutate(dplyr::group_by(d, g), y = dplyr::lag(f)))
  f18r_errors(df, function(d) dplyr::mutate(d, y = nchar(f)), "requires a character vector")
})

test_that("F18 rule: a bare factor as a filter predicate errors like dplyr", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_errors(df, function(d) dplyr::filter(d, f), "must be a logical vector")
})

test_that("F18 rule: zero-row factor expressions match dplyr", {
  skip_if_no_gpu()
  df <- f18r_df()[0, ]
  f18r_same(df, function(d) dplyr::filter(d, f == f3))
  f18r_same(df, function(d) dplyr::mutate(d, y = f != f3, z = dplyr::coalesce(f, f2)))
  f18r_same(df, function(d) dplyr::summarise(d, a = dplyr::first(f)))
})

# ---------------------------------------------------------------------------
# Rules 1-3: the allowed places stay on the GPU
# ---------------------------------------------------------------------------

test_that("F18 rule: bare factor references, keys and allowed calls stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::filter(d, f == f2))
  f18r_same(df, function(d) dplyr::filter(d, f != f2 | is.na(f)))
  f18r_same(df, function(d) dplyr::filter(d, !is.na(f) & n > 1))
  f18r_same(df, function(d) dplyr::mutate(d, h = f, z = is.na(o)))
  f18r_same(df, function(d) dplyr::transmute(d, h = f, n))
  f18r_same(df, function(d) dplyr::arrange(d, f, n))
  f18r_same(df, function(d) dplyr::arrange(d, dplyr::desc(f), n))
  f18r_same(df, function(d) dplyr::arrange(d, dplyr::desc(o), n))
  f18r_same(df, function(d) dplyr::count(d, f), arrange_by = "f")
  f18r_same(df, function(d) dplyr::distinct(d, f), arrange_by = "f")
  f18r_same(df, function(d) dplyr::summarise(dplyr::group_by(d, f), k = dplyr::n()), arrange_by = "f")
  f18r_same(df, function(d) dplyr::summarise(dplyr::group_by(d, h = f), k = dplyr::n()), arrange_by = "h")
  f18r_same(df, function(d) dplyr::summarise(d, k = dplyr::n_distinct(f), m = sum(is.na(f))))
  f18r_same(df, function(d) dplyr::slice_min(d, f, n = 2))
})

# ---------------------------------------------------------------------------
# The rule is one linear pass (B5)
# ---------------------------------------------------------------------------

test_that("F18 rule: ir_factor_rule() classifies the root of an expression", {
  skip_if_no_gpu()
  g <- tbl_gpu(f18r_df(), lazy = TRUE)
  sch <- current_schema(g)
  rule <- function(expr) {
    ir_factor_rule(ir_parse_quo(rlang::new_quosure(expr, globalenv()), sch), sch)
  }
  expect_identical(rule(quote(f)), "bare")
  expect_identical(rule(quote(n + 1)), "none")
  expect_identical(rule(quote(is.na(f))), "none")
  expect_identical(rule(quote(f == f2 & n > 1)), "none")
  expect_identical(rule(quote(o < o2)), "none")
  expect_identical(rule(quote(f == f3)), "cpu")
  expect_identical(rule(quote(f < f2)), "cpu")
  expect_identical(rule(quote(f == 1)), "cpu")
  expect_identical(rule(quote(is.na(f - 1))), "cpu")
  expect_identical(rule(quote(n > 1 | -f > 0)), "cpu")
})

test_that("F18 rule: the factor check is linear in expression size (wide if_any)", {
  skip_if_no_gpu()
  k <- 400L
  d <- as.data.frame(stats::setNames(replicate(k, c(1, 2, -1), simplify = FALSE), paste0("x", seq_len(k))))
  d$f <- factor(c("a", "b", "a"))
  d$f2 <- factor(c("a", "a", "b"), levels = c("a", "b"))
  g <- tbl_gpu(d, lazy = TRUE)
  sch <- current_schema(g)
  chain <- function(first_term) {
    terms <- c(first_term, paste0("x", seq_len(k), " > 5"))
    ir_parse_quo(rlang::new_quosure(str2lang(paste(terms, collapse = " | ")), globalenv()), sch)
  }
  ir_none <- chain("f == f2")
  ir_cpu <- chain("f == 1")
  expect_identical(ir_factor_rule(ir_none, sch), "none")
  expect_identical(ir_factor_rule(ir_cpu, sch), "cpu")
  t_none <- system.time(for (i in 1:3) ir_factor_rule(ir_none, sch))[["elapsed"]] / 3
  t_cpu <- system.time(for (i in 1:3) ir_factor_rule(ir_cpu, sch))[["elapsed"]] / 3
  expect_lt(t_none, 0.1)
  expect_lt(t_cpu, 0.1)

  # End to end: building the lazy filter must not be dominated by the check
  t_verb <- system.time(
    out <- dplyr::filter(g, dplyr::if_any(dplyr::starts_with("x"), ~ .x > 1.5))
  )[["elapsed"]]
  expect_lt(t_verb, 1.5)
  expect_equal(
    tibble::as_tibble(dplyr::collect(out)),
    dplyr::filter(tibble::as_tibble(d), dplyr::if_any(dplyr::starts_with("x"), ~ .x > 1.5))
  )
})

# ---------------------------------------------------------------------------
# Added after round 1 review (B1, B2, N1)
# ---------------------------------------------------------------------------

test_that("F18 rule: a factor redefined earlier in the same call is checked with its new levels (R1-B1)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::mutate(d, f2 = f3, y = f == f2))
  f18r_same(df, function(d) dplyr::transmute(d, f2 = f3, y = f == f2))
  f18r_same(df, function(d) dplyr::distinct(d, f2 = f3, y = f == f2), arrange_by = c("f2", "y"))
  f18r_same(df, function(d) dplyr::count(d, f2 = f3, y = f == f2), arrange_by = c("f2", "y"))
  f18r_same(df, function(d) dplyr::summarise(dplyr::group_by(d, f2 = f3, y = f == f2), k = dplyr::n(),
                                             .groups = "drop"), arrange_by = c("f2", "y"))
  f18r_same(df, function(d) dplyr::mutate(d, f2 = n, y = f == f2))
  f18r_errors(df, function(d) dplyr::mutate(d, f2 = f4, y = f == f2), "level sets of factors are different")
  # a plain copy carries its source's levels, so the copy compares
  # correctly, and on the GPU
  withr::with_options(list(cuplyr.fallback = "error"), {
    f18r_same(df, function(d) dplyr::mutate(d, f3 = f, y = f == f3))
  })
})

test_that("F18 rule: %in% between two factor columns matches dplyr (R1-B2)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::filter(d, f %in% f2))
  f18r_same(df, function(d) dplyr::mutate(d, y = f %in% f2))
})

test_that("F18 rule: string values against a factor in value helpers match dplyr (R1-N1)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::na_if(f, "1")))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::recode_values(f, "1" ~ "one", default = "other")))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::replace_values(f, "1" ~ "2")))
})

test_that("F18 rule: a factor value as a window default falls back (R1-N3)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_errors(df, function(d) dplyr::mutate(d, y = dplyr::lag(n, default = factor("2"))), "Can't convert")
})

test_that("F18 rule: helpers that combine a factor with strings fall back, never hard-error (R2-B1)", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")), n = 1:6)
  f18r_same(df, function(d) dplyr::mutate(d, y = paste0(dplyr::coalesce(f, "none"), "!")))
  f18r_same(df, function(d) dplyr::mutate(d, y = toupper(dplyr::if_else(is.na(f), "missing", f))))
  f18r_same(df, function(d) dplyr::filter(d, grepl("^1", dplyr::coalesce(f, "none"))))
  f18r_same(df, function(d) dplyr::count(d, y = paste0(dplyr::coalesce(f, "none"), "!")), arrange_by = "y")
  f18r_same(df, function(d) dplyr::summarise(d, k = dplyr::n_distinct(paste0(dplyr::coalesce(f, "none"), "!"))))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::na_if(dplyr::if_else(n > 2, f, "other"), "other")))
  f18r_same(df, function(d) dplyr::mutate(d, y = dplyr::case_when(n > 4 ~ dplyr::if_else(n > 5, f, "x"))))
  f18r_warns(df, function(d) dplyr::mutate(d, y = dplyr::if_else(n > 2, f - 1, NA)), "not meaningful for factors")
})

test_that("F18 rule: n_distinct() of a factor through the pronoun or parentheses works on the GPU (R3-B1)", {
  skip_if_no_gpu()
  df <- f18r_df()
  f18r_same(df, function(d) dplyr::summarise(d, k = dplyr::n_distinct((f))))
  f18r_same(df, function(d) dplyr::summarise(d, k = dplyr::n_distinct(.data$o, na.rm = TRUE)))
  f18r_same(df, function(d) dplyr::summarise(dplyr::group_by(d, g), k = dplyr::n_distinct(.data[["f"]])),
            arrange_by = "g")
  withr::with_options(list(cuplyr.fallback = "error"), {
    f18r_same(df, function(d) dplyr::summarise(d, k = dplyr::n_distinct(.data$f)))
  })
})
