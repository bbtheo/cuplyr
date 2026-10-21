# F19: a colwise dot comparing a STRING with a non-STRING operand keeps
# dplyr's result (the plain verbs' type gate sends it to the CPU).

# dplyr is imported, not attached, in the test environment
for (fn in c("vars", "mutate_at", "mutate_all", "filter_all")) {
  assign(fn, getExportedValue("dplyr", fn))
}
rm(fn)

both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("a known function with a string argument compares like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4), s = c("a", "b", NA, "d"))
  both(df, function(d) mutate_at(d, vars(x), "==", "a"))
  both(df, function(d) mutate_at(d, vars(x, s), `!=`, "b"))
})

test_that("%in% with a string set on a numeric column matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4), s = c("a", "b", NA, "d"))
  both(df, function(d) filter_all(d, function(v) v %in% c("a", "2", "b")))
  both(df, function(d) mutate_all(d, function(v) v %in% c("a", "b")))
})

test_that("a string held in a variable is seen through a closure", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4), s = c("a", "b", NA, "d"))
  lbl <- "b"
  both(df, function(d) filter_all(d, function(v) v != lbl))
  both(df, function(d) mutate_at(d, vars(x, s), function(v) v == lbl))
})

test_that("string-vs-string colwise comparisons are still rewritten", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4), s = c("a", "b", NA, "d"))
  q <- rlang::new_quosure(as.call(list(function(v) v == "a", quote(s))), globalenv())
  dots <- colwise_rewrite_dots(list(s = q), current_schema(tbl_gpu(df)))
  expect_false(colwise_has_fn_head(rlang::quo_get_expr(dots[[1]])))
  # the rewrite checks no types: the plain verbs' gate decides native vs CPU
  q <- rlang::new_quosure(as.call(list(function(v) v == "a", quote(x))), globalenv())
  dots <- colwise_rewrite_dots(list(x = q), current_schema(tbl_gpu(df)))
  expect_false(colwise_has_fn_head(rlang::quo_get_expr(dots[[1]])))
})

test_that("plain filter() of a number against a string matches dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4))
  both(df, function(d) filter(d, x != "a"))
})

test_that("cur_ prefixed variables do not force a fallback (call heads only)", {
  skip_if_no_gpu()
  expect_false(colwise_calls_cur(quote(.x * cur_rate)))
  expect_true(colwise_calls_cur(quote(.x + nchar(cur_column()))))
  expect_true(colwise_calls_cur(quote(dplyr::cur_column())))
  expect_false(colwise_calls_cur(quote(case_when(x > 1 ~ 1, ))))
})

# F19 round 4: aggregates over comparisons stay native in any collation.

test_that("a comparison inside an aggregate stays native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  summarise_all <- getExportedValue("dplyr", "summarise_all")
  df <- data.frame(x = c(1, 3, NA, 5), y = c(0, 4, 2, 1))
  for (coll in c("C", "en_US.UTF-8")) {
    ok <- tryCatch({ withr::local_collate(coll); TRUE }, warning = function(w) FALSE)
    if (!ok) next
    withr::local_collate(coll)
    both(df, function(d) summarise_all(d, ~ any(. > 2, na.rm = TRUE)))
    both(df, function(d) summarise_all(d, ~ mean(. > 2, na.rm = TRUE)))
    both(df, function(d) summarise_all(d, ~ max(. > 2, na.rm = TRUE)))
  }
})

test_that("only C and POSIX count as byte-order collations", {
  skip_if_no_gpu()
  withr::local_collate("C")
  expect_true(session_collation_is_c())
  ok <- tryCatch({ withr::local_collate("C.UTF-8"); TRUE }, warning = function(w) FALSE)
  skip_if_not(ok, "C.UTF-8 unavailable")
  withr::local_collate("C.UTF-8")
  expect_identical(session_collation_is_c(), identical(sort(c("a", "B")), c("B", "a")))
})
