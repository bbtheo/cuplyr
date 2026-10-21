# F50 guards: the factor rewrites touch only their exact shapes.

f50g_schema <- function(levels, ordered = FALSE) {
  lv <- levels
  attr(lv, "ordered") <- ordered
  list(names = c("f", "x"), types = c(f = "DICTIONARY32", x = "FLOAT64"),
       factor_levels = list(f = lv))
}

test_that("F50: an NA level, duplicated names or unknown levels decline the rewrite", {
  sch <- f50g_schema(c("a", NA))
  ir <- ir_parse_quo(rlang::quo(f == "a"), sch)
  expect_identical(ir$args[[2]]$type, "STRING")
  expect_null(ir_parse_quo(rlang::quo(as.character(f)), sch))
  sch2 <- f50g_schema(c("a", "b"))
  sch2$factor_levels <- NULL
  expect_identical(ir_parse_quo(rlang::quo(f %in% "a"), sch2)$args[[2]]$type, "STRING")
  sch3 <- f50g_schema(c("a", "b"))
  sch3$names <- c("f", "f")
  expect_identical(ir_parse_quo(rlang::quo(f == "a"), sch3)$args[[2]]$type, "STRING")
})

test_that("F50: as.integer()/as.character() of anything but a factor column are unchanged", {
  sch <- f50g_schema(c("a", "b"))
  lim <- 3.7
  ir <- ir_parse_quo(rlang::quo(as.integer(lim)), sch)
  expect_identical(ir$kind, "lit")
  expect_identical(ir$value, 3L)
  expect_null(ir_parse_quo(rlang::quo(as.integer(x)), sch))
  expect_null(ir_parse_quo(rlang::quo(as.character(x)), sch))
  expect_null(ir_parse_quo(rlang::quo(as.character(f, 1)), sch))
})

test_that("F50: shapes outside the allowlist still go through F18's rule", {
  sch <- f50g_schema(c("a", "b"))
  for (e in list(quote(f < "a"), quote("a" %in% f), quote(f == 1), quote(f %in% c(1, 2)),
                 quote(f == x))) {
    ir <- ir_parse_quo(rlang::new_quosure(e), sch)
    expect_false(is.null(ir), info = deparse(e))
    expect_identical(ir_factor_rule(ir, sch), "cpu", info = deparse(e))
  }
})

test_that("F50: a column named like a called function does not block constant folding", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "b", NA)), c = 1:3)
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), y = f %in% c("a", "b")))
  })
  expect_identical(res$y, c(TRUE, TRUE, FALSE))
})

test_that("F50: as.integer(lim) with a constant still folds in a filter", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "b", NA)), x = c(1, 5, 9))
  lim <- 4.9
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(dplyr::filter(tbl_gpu(df), x > as.integer(lim)))
  })
  expect_identical(res$x, c(5, 9))
})

test_that("F50: a later dot sees the as.character() result as a string, not the factor", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("b", "a", NA)), x = 1:3)
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), f = as.character(f), z = f == "a"))
  })
  expect_identical(res$z, c(FALSE, TRUE, NA))
})

# ---------------------------------------------------------------------------
# expr_has_column_ref() and embedded quosures
# ---------------------------------------------------------------------------

test_that("F50: expr_has_column_ref() looks through an embedded quosure", {
  sch <- f50g_schema(c("a", "b"))
  # an unsupplied `{{ arg }}` wraps the missing-argument sentinel: no crash, no column
  missing_quo <- rlang::new_quosure(rlang::missing_arg(), globalenv())
  e <- rlang::call2("f13_ignore", missing_quo)
  expect_false(expr_has_column_ref(e, sch))
  expect_false(expr_has_column_ref(missing_quo, sch))
  # `{{ col }}` carries a column reference, as dev's walker reported for it
  col_quo <- rlang::new_quosure(quote(x), globalenv())
  expect_true(expr_has_column_ref(rlang::call2("f13_ignore", col_quo), sch))
  expect_true(expr_has_column_ref(col_quo, sch))
  # a quosure of a non-column is not one
  other_quo <- rlang::new_quosure(quote(zz + 1), globalenv())
  expect_false(expr_has_column_ref(rlang::call2("f13_ignore", other_quo), sch))
  # nested: a call inside the quosure
  nested <- rlang::new_quosure(quote(log(x)), globalenv())
  expect_true(expr_has_column_ref(rlang::call2("g", nested), sch))
})

test_that("F50: an unsupplied {{ arg }} inside a mutate/group_by call does not crash", {
  skip_if_no_gpu()
  df <- data.frame(g1 = c(2L, 1L, 2L, 1L), a = c(1, 2, 3, 2))
  ignore <- function(x) 1
  named <- function(d, var) dplyr::group_by(d, g = ignore({{ var }}))
  res <- dplyr::collect(named(tbl_gpu(df)))
  expect_identical(res$g, rep(1, 4))
  via_mutate <- function(d, var) dplyr::mutate(d, g = ignore({{ var }}))
  res <- dplyr::collect(via_mutate(tbl_gpu(df)))
  expect_identical(res$g, rep(1, 4))
})

# ---------------------------------------------------------------------------
# Codes-vs-labels traps next to the native rewrites: these run through the CPU
# (or mix native and CPU) and must still see labels, not codes.
# ---------------------------------------------------------------------------

f50g_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1", "10", "2"),
               levels = c("10", "1", "2", "unused")),
    o = factor(c("lo", "mid", "hi", "mid", NA, "lo", "hi", "mid"),
               levels = c("lo", "mid", "hi"), ordered = TRUE),
    x = c(1.5, 2, 3, 4, 5, NA, 7, 8)
  )
}

f50g_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("F50: as.numeric(as.character(f)) is the label's number, not the code", {
  skip_if_no_gpu()
  f50g_both(f50g_df(), function(d) dplyr::mutate(d, n = as.numeric(as.character(f))))
})

test_that("F50: between() on an ordered factor compares labels as character, like dplyr (vctrs)", {
  skip_if_no_gpu()
  f50g_both(f50g_df(), function(d) dplyr::mutate(d, b = dplyr::between(o, "hi", "mid")))
})

test_that("F50: case_match() on a factor matches labels", {
  skip_if_no_gpu()
  f50g_both(f50g_df(), function(d) suppressWarnings(dplyr::mutate(
    d, z = dplyr::case_match(f, "1" ~ "one", c("2", "10") ~ "many", .default = "other")
  )))
})

# ---------------------------------------------------------------------------
# Non-UTF-8 levels: the GPU compares bytes, so as.character(f) of a factor
# whose levels aren't UTF-8 must not decode natively (round-2 B1).
# ---------------------------------------------------------------------------

f50g_latin1_df <- function() {
  lv <- iconv(c("\u00e9t\u00e9", "b"), "UTF-8", "latin1")
  tibble::tibble(f = factor(c(lv[1], lv[2], NA, lv[1]), levels = lv), x = 1:4)
}

test_that("F50: as.character() of latin1 levels compared with a UTF-8 or latin1 literal matches dplyr", {
  skip_if_no_gpu()
  df <- f50g_latin1_df()
  lv <- levels(df$f)
  expect_identical(Encoding(lv[1]), "latin1")
  f50g_both(df, function(d) dplyr::mutate(
    d, e = as.character(f) == "\u00e9t\u00e9", e2 = as.character(f) == lv[1]
  ))
  f50g_both(df, function(d) dplyr::filter(d, as.character(f) == "\u00e9t\u00e9"))
  f50g_both(df, function(d) dplyr::filter(d, as.character(f) == lv[1]))
})

test_that("F50: ==/%in% on latin1 levels with a UTF-8 literal stay native and match dplyr", {
  skip_if_no_gpu()
  df <- f50g_latin1_df()
  pipe <- function(d) dplyr::mutate(
    d, q = f == "\u00e9t\u00e9", r = f %in% c("\u00e9t\u00e9", "zz"), n = f != "b"
  )
  f50g_both(df, pipe)
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(pipe(tbl_gpu(df)))
  })
  expect_identical(res$q, c(TRUE, FALSE, NA, TRUE))
  expect_identical(res$r, c(TRUE, FALSE, FALSE, TRUE))
})

test_that("F50: as.character() of UTF-8 levels compared with a latin1 literal matches dplyr", {
  skip_if_no_gpu()
  utf8 <- enc2utf8("\u00e9t\u00e9")
  df <- tibble::tibble(f = factor(c(utf8, "b", NA, utf8), levels = c(utf8, "b")), x = 1:4)
  lit <- iconv(utf8, "UTF-8", "latin1")
  expect_identical(Encoding(lit), "latin1")
  f50g_both(df, function(d) dplyr::mutate(
    d, e = as.character(f) == lit, i = as.character(f) %in% c(lit, "zz")
  ))
  f50g_both(df, function(d) dplyr::filter(d, as.character(f) == lit))
  # the UTF-8 literal still decodes natively
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), e = as.character(f) == utf8))
  })
  expect_identical(res$e, c(TRUE, FALSE, NA, TRUE))
})

test_that("F50: as.character() decodes natively only for ASCII or UTF-8 levels", {
  utf8 <- enc2utf8("\u00e9t\u00e9")
  expect_false(is.null(ir_parse_quo(rlang::quo(as.character(f)), f50g_schema(c(utf8, "b")))))
  expect_false(is.null(ir_parse_quo(rlang::quo(as.character(f)), f50g_schema(c("a", "b")))))
  latin1 <- iconv(utf8, "UTF-8", "latin1")
  expect_null(ir_parse_quo(rlang::quo(as.character(f)), f50g_schema(c(latin1, "b"))))
  raw_bytes <- utf8
  Encoding(raw_bytes) <- "bytes"
  expect_null(ir_parse_quo(rlang::quo(as.character(f)), f50g_schema(c("a", raw_bytes))))
  # a non-UTF-8 string literal anywhere next to the decode declines it too
  sch <- f50g_schema(c(utf8, "b"))
  expect_null(ir_parse_quo(rlang::quo(as.character(f) == !!latin1), sch))
  expect_null(ir_parse_quo(rlang::quo(as.character(f) %in% c("b", !!latin1)), sch))
  expect_false(is.null(ir_parse_quo(rlang::quo(as.character(f) %in% c("b", !!utf8)), sch)))
  expect_false(is.null(ir_parse_quo(rlang::quo(f == !!latin1), sch)))
  # as.integer()/as.numeric() never look at the labels
  expect_false(is.null(ir_parse_quo(rlang::quo(as.integer(f)), f50g_schema(c(latin1, "b")))))
})

# ---------------------------------------------------------------------------
# Internal op names are not user-callable (round-2 N3)
# ---------------------------------------------------------------------------

test_that("F50: user functions named like internal ops run as the user's functions", {
  sch <- list(names = c("i", "x"), types = c(i = "INT32", x = "FLOAT64"))
  for (e in list(quote(factor_labels(i, "a")), quote(factor_codes(i)),
                 quote(cast_float64(i)), quote(cast_int32(x)))) {
    expect_null(ir_parse_quo(rlang::new_quosure(e), sch), label = deparse(e))
  }
  skip_if_no_gpu()
  df <- tibble::tibble(i = c(1L, 5L, 100L), x = c(1.5, 2, 3))
  factor_labels <- function(i, s) paste0(s, i)
  factor_codes <- function(i) i * 2L
  cast_float64 <- function(i) -i
  cast_int32 <- function(x) x + 0.25
  pipe <- function(d) dplyr::mutate(
    d, a = factor_labels(i, "a"), b = factor_codes(i), c = cast_float64(i),
    d = cast_int32(x), k = factor_codes(3L)
  )
  f50g_both(df, pipe)
  res <- dplyr::collect(pipe(tbl_gpu(df)))
  expect_identical(res$a, c("a1", "a5", "a100"))
  expect_identical(res$k, rep(6L, 3))
})

test_that("F50: as.character({{ v }}) of a factor column decodes natively", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("b", "a", NA)), x = 1:3)
  decode <- function(d, v) dplyr::mutate(d, s = as.character({{ v }}), e = as.character({{ v }}) == "a")
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(decode(tbl_gpu(df), f))
    lazy <- dplyr::collect(decode(as_lazy(tbl_gpu(df)), f))
  })
  expect_identical(res$s, c("b", "a", NA))
  expect_identical(res$e, c(FALSE, TRUE, NA))
  expect_identical(tibble::as_tibble(lazy), tibble::as_tibble(res))
})

test_that("F50: factor_labels decodes out-of-range codes to NA instead of reading past the levels", {
  skip_if_no_gpu()
  tbl <- tbl_gpu(data.frame(i = c(1L, 2L, 0L, 3L, NA, -1L, -2L, 1000L, .Machine$integer.max)))
  ir <- ir_call("factor_labels", list(ir_col("i"), ir_lit(c("a", "b"), "STRING")))
  ptr <- gpu_compute_column(tbl$ptr, ir_bind(ir, tbl$schema))
  expect_identical(gpu_collect(ptr, "v")$v, c("a", "b", NA, NA, NA, NA, NA, NA, NA))
})

# ---------------------------------------------------------------------------
# An unsupplied {{ v }} inside as.character() (round-3 B1): dplyr evaluates
# as.character() of the missing argument to "", whatever the verb.
# ---------------------------------------------------------------------------

test_that("F50: as.character() of an unsupplied {{ v }} folds to \"\" in every verb, like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("a", "b", NA)), x = 1:3)
  w_filter <- function(d, v) dplyr::filter(d, as.character({{ v }}) == "a")
  w_count <- function(d, v) dplyr::count(d, k = as.character({{ v }}))
  # `x + nchar()`: a column-free aggregation argument (`sum(1L)`) and a
  # comparison of two string literals next to a column (`k == "" & x > 0`)
  # both fail on dev, unrelated to factors; this argument reaches the factor
  # conversion and runs natively.
  w_summ <- function(d, v) dplyr::summarise(d, n = sum(x + nchar(as.character({{ v }}))))
  w_dist <- function(d, v) dplyr::distinct(d, k = as.character({{ v }}))
  w_mut <- function(d, v) dplyr::mutate(d, s = as.character({{ v }}))
  w_paren <- function(d, v) dplyr::filter(d, as.character(({{ v }})) == "")
  w_slice <- function(d, v) dplyr::slice_min(d, nchar(as.character({{ v }})) + x, n = 1)
  f50g_both(df, function(d) w_filter(d))
  f50g_both(df, function(d) w_count(d))
  f50g_both(df, function(d) w_summ(d))
  f50g_both(df, function(d) w_dist(d))
  f50g_both(df, function(d) w_mut(d))
  f50g_both(df, function(d) w_paren(d))
  f50g_both(df, function(d) w_slice(d))
})

test_that("F50: ir_parse_factor_conversion() declines a missing argument however it is wrapped", {
  sch <- f50g_schema(c("a", "b"))
  missing_quo <- rlang::new_quosure(rlang::missing_arg(), globalenv())
  shapes <- list(
    rlang::call2("as.character", missing_quo),
    rlang::call2("as.character", rlang::call2("(", missing_quo)),
    rlang::call2("as.character", rlang::new_quosure(missing_quo, globalenv())),
    rlang::call2("as.integer", missing_quo),
    rlang::call2("as.character", rlang::missing_arg()),
    rlang::call2("as.character", rlang::call2("(", rlang::missing_arg()))
  )
  for (e in shapes) {
    expect_null(ir_parse_factor_conversion(e, sch, as.character(e[[1]])), label = deparse(e))
  }
  ir <- ir_parse_quo(rlang::new_quosure(shapes[[1]], globalenv()), sch)
  expect_identical(ir$kind, "lit")
  expect_identical(ir$value, "")
})

test_that("F50: a latin1 lag() default next to the decode falls back and matches dplyr", {
  skip_if_no_gpu()
  utf8 <- enc2utf8("été")
  df <- tibble::tibble(f = factor(c(utf8, "b", NA, utf8), levels = c(utf8, "b")), x = 1:4)
  lit <- iconv(utf8, "UTF-8", "latin1")
  expect_identical(Encoding(lit), "latin1")
  f50g_both(df, function(d) dplyr::mutate(
    d, y = dplyr::lag(as.character(f), default = lit) == "été"
  ))
})
