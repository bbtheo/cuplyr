# F121 addition (Theo 2026-11-11): a rowwise pipeline. A per-row max and sum
# over four numeric columns of the wide table, then a small summary by
# category, in each engine's natural form: cuplyr rowwise() + c_across(),
# dplyr rowwise() + c_across(), dplyr vectorised (pmax, +), data.table
# vectorised (pmax, rowSums) and DuckDB (greatest, arithmetic). Pins the forms,
# their results against a base-R reference, cuplyr's native/fallback verdict
# from F88's condition (cuplyr_cpu_fallback), and the per-call time cap.
# Helpers in helper-parity-F121.R.

f121_rowwise_forms <- c("cuplyr_rowwise", "dplyr_rowwise", "dplyr_vectorised",
                        "datatable_vectorised", "duckdb")

# base-R reference: canonical form (category, n, m_mean, s_sum), sorted by category
f121_rowwise_ref <- function(f, cols) {
  m <- do.call(pmax, unname(as.list(f[cols])))
  s <- Reduce(`+`, f[cols])
  ix <- split(seq_len(nrow(f)), f$category)
  out <- data.frame(category = names(ix), n = as.double(lengths(ix)),
                    m_mean = vapply(ix, function(i) mean(m[i]), 0),
                    s_sum = vapply(ix, function(i) sum(s[i]), 0), stringsAsFactors = FALSE)
  out <- out[order(out$category, method = "radix"), , drop = FALSE]
  rownames(out) <- NULL
  out
}

test_that("the rowwise forms, columns and constants", {
  pp <- f121_load()
  expect_identical(pp$pp_rowwise_forms(), f121_rowwise_forms)
  cols <- pp$pp_constants()$rowwise_cols
  expect_identical(cols, c("x17", "x18", "x19", "x20"))
  expect_setequal(pp$pp_columns("rowwise"), c("category", cols))
  for (form in f121_rowwise_forms) expect_true(is.function(pp$pp_rowwise_fn(form)), label = form)
  expect_error(pp$pp_rowwise_fn("nope"))
})

test_that("each rowwise form is written in its engine's natural form", {
  pp <- f121_load()
  code <- function(form) paste(deparse(pp$pp_rowwise_fn(form)), collapse = "\n")
  for (form in c("cuplyr_rowwise", "dplyr_rowwise")) {
    expect_match(code(form), "rowwise(", fixed = TRUE, label = form)
    expect_match(code(form), "c_across(", fixed = TRUE, label = form)
  }
  expect_match(code("cuplyr_rowwise"), "read_parquet_gpu(", fixed = TRUE)
  expect_match(code("dplyr_vectorised"), "pmax(", fixed = TRUE)
  expect_false(grepl("rowwise(", code("dplyr_vectorised"), fixed = TRUE))
  expect_match(code("datatable_vectorised"), "pmax(", fixed = TRUE)
  expect_match(code("datatable_vectorised"), "rowSums(", fixed = TRUE)
  sql <- pp$pp_rowwise_sql("/some/fact.parquet")
  expect_match(sql, "greatest\\(", ignore.case = TRUE)
  expect_match(sql, "read_parquet('/some/fact.parquet')", fixed = TRUE)
  expect_false(grepl("SELECT\\s+\\*", sql, ignore.case = TRUE))
})

test_that("every CPU rowwise form matches the base-R reference", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  cols <- pp$pp_constants()$rowwise_cols
  ref <- f121_rowwise_ref(as.data.frame(arrow::read_parquet(fx$wide)), cols)
  for (form in setdiff(f121_rowwise_forms, "cuplyr_rowwise")) {
    engine <- switch(form, duckdb = "duckdb", datatable_vectorised = "data.table", "dplyr")
    ctx <- pp$pp_context(engine)
    res <- pp$pp_run_rowwise(form, fx$wide, ctx)
    expect_true(is.data.frame(res), label = form)
    nz <- pp$pp_normalize(res, "rowwise")
    expect_identical(names(nz), c("category", "n", "m_mean", "s_sum"))
    expect_equal(nz, ref, tolerance = 1e-9, label = form)
    expect_true(pp$pp_compare(nz, ref)$ok, label = paste(form, "pp_compare"))
    pp$pp_close(ctx)
  }
})

test_that("the dplyr and data.table rowwise forms read only the needed columns", {
  pp <- f121_load()
  fx <- f121_fixture(pp)
  seen <- list()
  orig <- pp$pp_read_arrow
  pp$pp_read_arrow <- function(path, cols) { seen[[length(seen) + 1L]] <<- cols; orig(path, cols) }
  for (form in c("dplyr_rowwise", "dplyr_vectorised", "datatable_vectorised")) {
    seen <- list()
    pp$pp_run_rowwise(form, fx$wide, NULL)
    expect_length(seen, 1L)
    expect_setequal(seen[[1]], pp$pp_columns("rowwise"))
  }
})

test_that("cuplyr's rowwise form matches the reference, reads only its columns, and its fallback verdict comes from F88's condition", {
  skip_if_no_gpu()
  pp <- f121_load()
  fx <- f121_fixture(pp)
  cols <- pp$pp_constants()$rowwise_cols
  ref <- f121_rowwise_ref(as.data.frame(arrow::read_parquet(fx$wide)), cols)
  res <- pp$pp_run_rowwise("cuplyr_rowwise", fx$wide, pp$pp_context("cuplyr"))
  expect_s3_class(res, "tbl_df")
  expect_equal(pp$pp_normalize(res, "rowwise"), ref, tolerance = 1e-9)
  expect_setequal(parquet_last_read()$columns, pp$pp_columns("rowwise"))

  # pp_fallback_probe(): runs fn once, untimed, and reports any cuplyr_cpu_fallback
  # condition (F88) it signals; it does not change cuplyr.fallback.
  old <- getOption("cuplyr.fallback")
  v <- pp$pp_fallback_probe(function() pp$pp_run_rowwise("cuplyr_rowwise", fx$wide, NULL))
  expect_identical(getOption("cuplyr.fallback"), old)
  expect_true(all(c("native", "reason") %in% names(v)))
  expect_type(v$native, "logical")
  expect_length(v$native, 1L)
  # the verdict agrees with an independent watch on the same condition
  seen <- FALSE
  withCallingHandlers(pp$pp_run_rowwise("cuplyr_rowwise", fx$wide, NULL),
                      cuplyr_cpu_fallback = function(cnd) seen <<- TRUE)
  expect_identical(v$native, !seen)
  # and it sees a real fallback (signif() runs on the CPU)
  g <- tbl_gpu(data.frame(a = c(1.234, 5.678)))
  fb <- pp$pp_fallback_probe(function() collect(mutate(g, b = signif(a, 2))))
  expect_false(fb$native)
  expect_true(nzchar(fb$reason))
})

test_that("pp_time_cell() caps a call at `cap` seconds and records it as capped", {
  pp <- f121_load(character())
  # the slow call would run 60 s; the untimed gc() around each call can take
  # seconds in a large test process, so the bound is generous but far below 60
  slow <- function() { t0 <- proc.time()[["elapsed"]]; while (proc.time()[["elapsed"]] - t0 < 60) NULL; 1 }
  fast <- function() 2
  t0 <- proc.time()[["elapsed"]]
  out <- pp$pp_time_cell(list(slow = slow, fast = fast), reps = 2L, cap = 0.3,
                         busy_fn = function(c0, c1) c(total = 0, own = 0, kernel = 0, foreign = 0),
                         wait = function(...) 0)
  expect_lt(proc.time()[["elapsed"]] - t0, 20)
  s <- out[out$engine == "slow", ]
  expect_identical(nrow(s), 1L)
  expect_identical(s$status, "capped")
  expect_false(s$ok)
  f <- out[out$engine == "fast", ]
  expect_identical(sort(f$rep), 0:2)
  expect_true(all(f$status == "ok"))
  # no cap by default
  cap_default <- eval(formals(pp$pp_time_cell)$cap)
  expect_true(is.null(cap_default) || is.infinite(cap_default))
})
