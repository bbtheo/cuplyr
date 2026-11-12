# F118 acceptance: the randomized differential harness itself.
#
# The comparator tests hand-build outcomes and need no GPU. They exist because
# a comparator that is too lenient is worse than none: every kind of difference
# the harness claims to catch has a planted case here.

tb <- function(...) tibble::tibble(...)
ok <- function(value, ...) fuzz_outcome(value = value, ...)
err <- function(class, msg = "boom") {
  fuzz_outcome(error = structure(
    class = c(class, "error", "condition"),
    list(message = msg, call = NULL)
  ))
}
kind_of <- function(oracle, gpu) {
  m <- fuzz_compare(oracle, gpu)
  if (is.null(m)) "none" else m$kind
}
category_of <- function(oracle, gpu) fuzz_compare(oracle, gpu)$category

# ---- comparator: equal things are equal ------------------------------------

test_that("identical results of every column type compare equal", {
  v <- tb(
    i = c(1L, NA), d = c(1.5, NaN), l = c(TRUE, NA), s = c("a", NA),
    f = factor(c("u", NA), levels = c("u", "v")),
    o = factor(c("u", "v"), ordered = TRUE),
    dt = as.Date(c("2020-01-01", NA)),
    ts = as.POSIXct(c(0, NA), tz = "UTC", origin = "1970-01-01"),
    b = bit64::as.integer64(c("9007199254740993", NA))
  )
  expect_null(fuzz_compare(ok(v), ok(v)))
})

test_that("row names are not a difference", {
  a <- as.data.frame(tb(x = 1:2))
  rownames(a) <- c("r1", "r2")
  expect_null(fuzz_compare(ok(a), ok(tb(x = 1:2))))
})

# ---- comparator: types -------------------------------------------------------

test_that("type differences are caught even where the values agree", {
  expect_equal(kind_of(ok(tb(x = 1L)), ok(tb(x = 1))), "type")
  expect_equal(kind_of(ok(tb(x = TRUE)), ok(tb(x = 1L))), "type")
  expect_equal(kind_of(ok(tb(x = "a")), ok(tb(x = factor("a")))), "type")
  expect_equal(
    kind_of(ok(tb(x = bit64::as.integer64(1))), ok(tb(x = 1))),
    "type"
  )
  expect_equal(
    kind_of(ok(tb(x = as.Date("2020-01-01"))),
            ok(tb(x = as.POSIXct("2020-01-01", tz = "UTC")))),
    "type"
  )
})

test_that("type differences are caught on zero-row results", {
  expect_equal(kind_of(ok(tb(x = integer())), ok(tb(x = double()))), "type")
})

test_that("factor levels, ordered-ness and time zones are part of the type", {
  expect_equal(
    kind_of(ok(tb(f = factor("a", levels = c("a", "b")))),
            ok(tb(f = factor("a", levels = c("b", "a"))))),
    "type"
  )
  expect_equal(
    kind_of(ok(tb(f = factor("a", ordered = TRUE))), ok(tb(f = factor("a")))),
    "type"
  )
  expect_equal(
    kind_of(ok(tb(t = as.POSIXct(0, tz = "UTC", origin = "1970-01-01"))),
            ok(tb(t = as.POSIXct(0, tz = "Asia/Tokyo", origin = "1970-01-01")))),
    "type"
  )
})

# ---- comparator: values ------------------------------------------------------

test_that("value differences waldo's tolerance hides are caught", {
  expect_equal(kind_of(ok(tb(x = NA_real_)), ok(tb(x = NaN))), "value")
  expect_equal(kind_of(ok(tb(x = NaN)), ok(tb(x = NA_real_))), "value")
  # F96: expect_equal() compares integer64 loosely
  expect_equal(
    kind_of(ok(tb(x = bit64::as.integer64("9007199254740993"))),
            ok(tb(x = bit64::as.integer64("9007199254740992")))),
    "value"
  )
  expect_equal(
    kind_of(ok(tb(x = bit64::as.integer64(c("1", NA)))),
            ok(tb(x = bit64::as.integer64(c("1", "2"))))),
    "value"
  )
})

test_that("plain value differences of every type are caught", {
  expect_equal(kind_of(ok(tb(x = 1L)), ok(tb(x = 2L))), "value")
  expect_equal(kind_of(ok(tb(x = c(1L, NA))), ok(tb(x = c(1L, 2L)))), "value")
  expect_equal(kind_of(ok(tb(x = "a")), ok(tb(x = "b"))), "value")
  expect_equal(kind_of(ok(tb(x = NA_character_)), ok(tb(x = "NA"))), "value")
  expect_equal(kind_of(ok(tb(x = NA_character_)), ok(tb(x = ""))), "value")
  expect_equal(kind_of(ok(tb(x = TRUE)), ok(tb(x = FALSE))), "value")
  expect_equal(
    kind_of(ok(tb(x = factor(c("a", "b"), levels = c("a", "b")))),
            ok(tb(x = factor(c("b", "b"), levels = c("a", "b"))))),
    "value"
  )
  expect_equal(
    kind_of(ok(tb(x = as.Date("2020-01-01"))), ok(tb(x = as.Date("2020-01-02")))),
    "value"
  )
  expect_equal(
    kind_of(ok(tb(x = as.POSIXct(1000, tz = "UTC", origin = "1970-01-01"))),
            ok(tb(x = as.POSIXct(1001, tz = "UTC", origin = "1970-01-01")))),
    "value"
  )
  expect_equal(kind_of(ok(tb(x = Inf)), ok(tb(x = -Inf))), "value")
})

test_that("the float tolerance is tight", {
  expect_null(fuzz_compare(ok(tb(x = 1)), ok(tb(x = 1 + 1e-13))))
  expect_equal(kind_of(ok(tb(x = 1)), ok(tb(x = 1 + 1e-6))), "value")
  expect_equal(kind_of(ok(tb(x = 1e12)), ok(tb(x = 1e12 + 10))), "value")
})

test_that("column names, order and row counts are compared", {
  expect_equal(kind_of(ok(tb(a = 1L)), ok(tb(b = 1L))), "names")
  expect_equal(kind_of(ok(tb(a = 1L, b = 2L)), ok(tb(b = 2L, a = 1L))), "names")
  expect_equal(kind_of(ok(tb(a = 1:2)), ok(tb(a = 1L))), "value")
})

test_that("row order is compared, and reported apart from wrong rows", {
  expect_equal(kind_of(ok(tb(a = 1:2)), ok(tb(a = 2:1))), "order")
  expect_equal(kind_of(ok(tb(a = c(1L, 1L, 2L))), ok(tb(a = c(1L, 2L, 2L)))), "value")
})

test_that("grouping is compared", {
  expect_equal(
    kind_of(ok(tb(a = 1L), groups = "a"), ok(tb(a = 1L), groups = character())),
    "groups"
  )
  expect_equal(category_of(ok(tb(a = 1L), groups = "a"), ok(tb(a = 1L))), "silent_wrong")
})

# ---- comparator: errors ------------------------------------------------------

test_that("one side erroring is caught and categorized", {
  expect_equal(kind_of(ok(tb(a = 1L)), err("rlang_error")), "error_presence")
  expect_equal(category_of(ok(tb(a = 1L)), err("rlang_error")), "crash_or_raw")
  expect_equal(kind_of(err("rlang_error"), ok(tb(a = 1L))), "error_presence")
  expect_equal(category_of(err("rlang_error"), ok(tb(a = 1L))), "silent_wrong")
})

test_that("error classes follow dplyr where dplyr has one", {
  ptype <- c("vctrs_error_ptype2", "vctrs_error_incompatible_type", "rlang_error")
  expect_null(fuzz_compare(err(ptype), err(ptype)))
  expect_null(fuzz_compare(err("dplyr:::mutate_error"), err(c("cuplyr_x", "dplyr:::mutate_error"))))
  expect_equal(kind_of(err(ptype), err("rlang_error")), "error_class")
  expect_equal(category_of(err(ptype), err("cuplyr_error_unsupported")), "error_class")
  # dplyr has no specific class: any error matches
  expect_null(fuzz_compare(err("rlang_error"), err("cuplyr_error_unsupported")))
  expect_null(fuzz_compare(err("simpleError"), err("rlang_error")))
})

test_that("a raw C++ error is always a crash_or_raw finding", {
  raw <- err(c("Rcpp::exception", "C++Error"))
  std <- err(c("std::runtime_error", "C++Error"))
  expect_equal(category_of(ok(tb(a = 1L)), raw), "crash_or_raw")
  expect_equal(category_of(err("rlang_error"), raw), "crash_or_raw")
  expect_equal(category_of(err("simpleError"), std), "crash_or_raw")
  # wrap_gpu_call() rewords a C++ failure into this class
  gpu <- err(c("cuplyr_error_gpu", "rlang_error"))
  expect_equal(category_of(ok(tb(a = 1L)), gpu), "crash_or_raw")
  expect_equal(category_of(err("rlang_error"), gpu), "crash_or_raw")
  expect_equal(category_of(err("dplyr:::mutate_error"), gpu), "crash_or_raw")
  # lazy mode lets cuplyr's own internal stop() through unwrapped
  internal <- err("simpleError", "internal: window op 'lag' reached lowering")
  expect_equal(category_of(err("rlang_error"), internal), "crash_or_raw")
  expect_equal(category_of(ok(tb(a = 1L)), internal), "crash_or_raw")
})

# ---- comparator: warnings ----------------------------------------------------

w <- function(msg, class = "simpleWarning") {
  structure(class = c(class, "warning", "condition"),
            list(message = msg, call = NULL))
}

test_that("whether each side warns is compared", {
  expect_equal(kind_of(ok(tb(a = 1L), warnings = list(w("x"))), ok(tb(a = 1L))), "warning")
  expect_equal(kind_of(ok(tb(a = 1L)), ok(tb(a = 1L), warnings = list(w("x")))), "warning")
  expect_equal(
    category_of(ok(tb(a = 1L), warnings = list(w("x"))), ok(tb(a = 1L))),
    "warning_only"
  )
  expect_null(fuzz_compare(
    ok(tb(a = 1L), warnings = list(w("x"))),
    ok(tb(a = 1L), warnings = list(w("different text")))
  ))
})

test_that("a NaN-math warning is the only warning masked for being NaN-math", {
  expect_null(fuzz_compare(
    ok(tb(a = NaN), warnings = list(w("NaNs produced"))),
    ok(tb(a = NaN))
  ))
  expect_equal(
    kind_of(ok(tb(a = NaN), warnings = list(w("something else"))), ok(tb(a = NaN))),
    "warning"
  )
})

test_that("a real dplyr NaN-math warning is masked, a second warning in the verb is not", {
  # dplyr folds a verb's warnings into one condition quoting only the first
  nan <- fuzz_make_case(tb(a = c(-1, 4)), list(quote(mutate(m = sqrt(a)))))
  one <- fuzz_oracle(nan)
  expect_length(one$warnings, 1L)
  expect_null(fuzz_compare(one, ok(one$value)))

  # a second warning that doesn't depend on the session: POSIXct comparison's
  # tzone warning disappears once lubridate is loaded (earlier test files do)
  two <- fuzz_make_case(
    tb(a = c(-1, 4)),
    list(quote(mutate(m1 = sqrt(a), m2 = as.integer(c("x", "1")))))
  )
  both <- fuzz_oracle(two)
  expect_match(conditionMessage(both$warnings[[1]]), "There were 2 warnings", fixed = TRUE)
  expect_equal(kind_of(both, ok(both$value)), "warning")
})

test_that("dplyr's many-to-many join warning reaches the comparator", {
  x <- tb(k = c(1L, 1L, 2L), a = 1:3)
  y <- tb(k = c(1L, 1L, 3L), b = 4:6)
  cc <- fuzz_make_case(x, list(quote(left_join(y, by = "k"))), y = y)
  o <- fuzz_oracle(cc)
  expect_null(o$error)
  expect_gte(length(o$warnings), 1L)
  expect_equal(kind_of(o, ok(o$value)), "warning")
})

test_that("the timestamp masks are per element and need the matching value", {
  d <- function(v) tb(d = structure(v, class = "Date"))
  # +-Inf that the GPU holds as NA, and nothing else
  expect_null(fuzz_compare(ok(d(Inf)), ok(d(NA_real_))))
  expect_equal(kind_of(ok(d(Inf)), ok(d(5))), "value")
  expect_equal(kind_of(ok(d(c(Inf, 1))), ok(d(c(NA, 2)))), "value")
  expect_equal(kind_of(ok(d(1)), ok(d(NA_real_))), "value")
  # a fractional Date floored on the GPU, and nothing else
  expect_null(fuzz_compare(ok(d(18262.5)), ok(d(18262))))
  expect_equal(kind_of(ok(d(18262.5)), ok(d(18263))), "value")
  expect_equal(kind_of(ok(d(18262)), ok(d(18263))), "value")
  # F81 documents mean() of an empty group: a NaN Date in dplyr, NA on the GPU
  expect_null(fuzz_compare(ok(d(NaN)), ok(d(NA_real_))))
  expect_equal(kind_of(ok(d(NaN)), ok(d(1))), "value")
  expect_true("nan_timestamp" %in% vapply(fuzz_known_differences(), `[[`, "", "id"))
})

test_that("the upload warnings are masked only when the value explains them", {
  wi <- w("inf", c("cuplyr_warning_infinite_timestamp", "cuplyr_warning_timestamp_unrepresentable"))
  wf <- w("frac", "cuplyr_warning_fractional_date")
  dt <- function(v) tb(d = structure(v, class = "Date"))
  expect_null(fuzz_compare(ok(dt(Inf)), ok(dt(NA_real_), warnings = list(wi))))
  expect_equal(kind_of(ok(dt(1)), ok(dt(1), warnings = list(wi))), "warning")
  expect_null(fuzz_compare(ok(dt(18262.5)), ok(dt(18262), warnings = list(wf))))
  expect_equal(kind_of(ok(dt(18262)), ok(dt(18262), warnings = list(wf))), "warning")
})

test_that("a value difference wins over a warning difference", {
  expect_equal(
    kind_of(ok(tb(a = 1L), warnings = list(w("x"))), ok(tb(a = 2L))),
    "value"
  )
})

# ---- known differences -------------------------------------------------------

test_that("every known difference has an id, a reason and a ledger reference", {
  kd <- fuzz_known_differences()
  expect_gt(length(kd), 0L)
  for (e in kd) {
    expect_true(is.character(e$id) && length(e$id) == 1L && nzchar(e$id))
    expect_true(is.character(e$reason) && nchar(e$reason) >= 20L, info = e$id)
    expect_true(is.character(e$ledger) && nzchar(e$ledger), info = e$id)
  }
  ids <- vapply(kd, `[[`, "", "id")
  expect_false(anyDuplicated(ids) > 0L)
})

# ---- generator ---------------------------------------------------------------

gen_seeds <- 1:600
cases <- NULL
gen <- function() {
  if (is.null(cases)) cases <<- lapply(gen_seeds, fuzz_case)
  cases
}
step_names <- function(cs) {
  unique(unlist(lapply(cs, function(cc) all.names(as.call(c(quote(list), cc$steps))))))
}

test_that("the same seed gives the same case and code, whatever the RNG state", {
  set.seed(1)
  a <- fuzz_case(42L)
  set.seed(2)
  runif(3)
  b <- fuzz_case(42L)
  expect_identical(a$x, b$x)
  expect_identical(a$y, b$y)
  expect_identical(a$steps, b$steps)
  expect_identical(a$vars, b$vars)
  expect_identical(fuzz_code(a), fuzz_code(b))
})

test_that("generating a case leaves the caller's RNG alone", {
  set.seed(7)
  before <- .Random.seed
  fuzz_case(3L)
  expect_identical(.Random.seed, before)
})

test_that("different seeds give different cases", {
  codes <- vapply(gen()[1:50], fuzz_code, "")
  expect_gt(length(unique(codes)), 45L)
})

test_that("the table generator covers every column type and shape", {
  cs <- gen()
  classes <- unique(unlist(lapply(cs, function(cc) lapply(cc$x, function(col) class(col)[1]))))
  for (cl in c("integer", "numeric", "logical", "character", "factor", "Date", "POSIXct", "integer64")) {
    expect_true(cl %in% classes, info = cl)
  }
  rows <- vapply(cs, function(cc) nrow(cc$x), 1L)
  expect_true(any(rows == 0L))
  expect_true(any(rows == 1L))
  expect_true(any(rows > 1L))
  has_nan <- vapply(cs, function(cc) any(vapply(cc$x, function(col) is.double(col) && !inherits(col, c("Date", "POSIXct", "integer64")) && any(is.nan(col)), NA)), NA)
  has_na <- vapply(cs, function(cc) anyNA(cc$x), NA)
  expect_true(any(has_nan))
  expect_true(any(has_na))
  dup_key <- vapply(cs, function(cc) nrow(cc$x) > 1L && anyDuplicated(cc$x[[1]]) > 0L, NA)
  expect_true(any(dup_key))
})

test_that("the grammar reaches every verb and expression family in the brief", {
  nm <- step_names(gen())
  verbs <- c("filter", "mutate", "select", "rename", "relocate", "arrange",
             "distinct", "group_by", "ungroup", "summarise", "count",
             "slice", "slice_head", "slice_tail", "slice_min", "slice_max",
             "left_join", "inner_join", "bind_rows")
  for (v in verbs) expect_true(v %in% nm, info = v)
  exprs <- c("if_else", "case_when", "coalesce", "is.na", "%in%", "sum", "mean",
             "min", "max", "lag", "lead", "cumsum", "row_number",
             "across", "if_any", ".env", ".data", "[[", "==", "&", "|", "!", "+")
  for (e in exprs) expect_true(e %in% nm, info = e)
  any_by <- vapply(gen(), function(cc) any(vapply(cc$steps, function(s) ".by" %in% names(as.list(s)), NA)), NA)
  expect_true(any(any_by))
  any_drop <- vapply(gen(), function(cc) any(vapply(cc$steps, function(s) ".drop" %in% names(as.list(s)), NA)), NA)
  expect_true(any(any_drop))
})

test_that("steps after a summarise still use window and aggregate calls", {
  # one case's expression budget must not be spent by an earlier verb
  win <- c("lag", "lead", "cumsum", "cummax", "cummin", "dense_rank", "min_rank", "ntile")
  agg <- c("sum", "mean", "min", "max", "median")
  later <- c(win = 0L, agg = 0L)
  for (cc in gen()) {
    heads <- vapply(cc$steps, function(s) as.character(s[[1]]), "")
    i <- which(heads %in% c("summarise", "summarize"))
    i <- i[!vapply(cc$steps[i], function(s) "across" %in% all.names(s), NA)]
    if (length(i) == 0L) next
    for (s in cc$steps[-seq_len(i[[1]])]) {
      if (as.character(s[[1]]) %in% c("mutate", "transmute", "filter") &&
          !("across" %in% all.names(s))) {
        later[["win"]] <- later[["win"]] + sum(all.names(s) %in% win)
        later[["agg"]] <- later[["agg"]] + sum(all.names(s) %in% agg)
      }
    }
  }
  expect_gt(later[["win"]], 0L)
  expect_gt(later[["agg"]], 0L)
})

test_that("generated pipelines are mostly valid but keep invalid calls in", {
  cs <- gen()[1:150]
  outs <- lapply(cs, fuzz_oracle)
  err_share <- mean(vapply(outs, function(o) !is.null(o$error), NA))
  expect_gt(err_share, 0.005)
  expect_lt(err_share, 0.40)
})

test_that("fuzz_code() is runnable and rebuilds the case", {
  for (s in c(1L, 7L, 11L, 23L, 31L)) {
    cc <- fuzz_case(s)
    env <- new.env(parent = asNamespace("cuplyr"))
    eval(parse(text = fuzz_code(cc)), env)
    expect_true(is.function(env$pipeline), info = paste("seed", s))
    expect_null(
      fuzz_compare(ok(tibble::as_tibble(cc$x)), ok(tibble::as_tibble(env$x))),
      info = s
    )
    from_code <- tryCatch(env$pipeline(env$x), error = function(e) e)
    oracle <- fuzz_oracle(cc)
    if (inherits(from_code, "error")) {
      expect_false(is.null(oracle$error), info = paste("seed", s))
    } else {
      expect_null(oracle$error, info = paste("seed", s))
      expect_null(fuzz_compare(ok(dplyr::ungroup(from_code)), ok(dplyr::ungroup(oracle$value))), info = paste("seed", s))
    }
  }
})

# ---- shrinker: no GPU needed with a custom predicate --------------------------

test_that("the shrinker reduces rows, columns and steps under a predicate", {
  cc <- fuzz_make_case(
    x = tb(a = 1:20, b = as.double(1:20), c = letters[1:20], d = 20:1),
    steps = list(quote(mutate(e = a + 1L)), quote(filter(a > 0L)),
                 quote(mutate(f = b * 2)), quote(select(a, b, e, f)))
  )
  fails <- function(k) "b" %in% names(k$x) && nrow(k$x) >= 2L && length(k$steps) >= 1L
  s <- fuzz_shrink(cc, fails = fails)
  expect_s3_class(s, "fuzz_case")
  expect_true(fails(s))
  expect_equal(nrow(s$x), 2L)
  expect_equal(names(s$x), "b")
  expect_equal(length(s$steps), 1L)
})

test_that("columns a removed dot read are dropped by the shrinker", {
  cc <- fuzz_make_case(
    x = tb(a = 1:3, b = 4:6, c = 7:9),
    steps = list(quote(mutate(m1 = a + b, m2 = c * 2L)))
  )
  fails <- function(k) {
    o <- fuzz_oracle(k)
    is.null(o$error) && "m2" %in% names(o$value) && nrow(o$value) >= 1L
  }
  s <- fuzz_shrink(cc, fails = fails)
  expect_equal(names(s$x), "c")
  expect_equal(nrow(s$x), 1L)
})

test_that("the shrinker returns the case unchanged when it cannot shrink", {
  cc <- fuzz_make_case(x = tb(a = 1L), steps = list(quote(filter(a > 0L))))
  s <- fuzz_shrink(cc, fails = function(k) identical(k, cc))
  expect_identical(s$x, cc$x)
  expect_identical(s$steps, cc$steps)
})

# ---- end to end on the GPU ----------------------------------------------------

plain <- tb(
  id = c(1L, 2L, 2L, 3L),
  v = c(1.5, NA, 3.5, NaN),
  s = c("a", "b", NA, "a")
)

test_that("a matching pipeline has no findings on either mode", {
  skip_if_no_gpu()
  cc <- fuzz_make_case(plain, list(quote(filter(id > 1L)), quote(mutate(w = id + 1L))))
  res <- fuzz_check(cc)
  expect_s3_class(res, "data.frame")
  expect_equal(nrow(res), 0L)
})

test_that("a planted corruption is found in eager and lazy and shrinks", {
  skip_if_no_gpu()
  cc <- fuzz_make_case(
    x = tb(a = 1:12, b = as.double(12:1), c = rep(c("u", "v"), 6L)),
    steps = list(quote(mutate(flag = a > 0L)), quote(mutate(z = b * 2)), quote(filter(a > 0L)))
  )
  inject <- function(v) {
    if ("flag" %in% names(v)) v$flag <- !v$flag
    v
  }
  res <- fuzz_check(cc, inject = inject)
  expect_setequal(res$mode, c("eager", "lazy"))
  expect_true(all(res$kind == "value"))
  expect_true(all(res$category == "silent_wrong"))

  s <- fuzz_shrink(cc, fails = function(k) nrow(fuzz_check(k, inject = inject)) > 0L)
  expect_lte(length(s$steps), 1L)
  expect_lte(nrow(s$x), 2L)
  expect_lte(ncol(s$x), 2L)
  expect_gt(nrow(fuzz_check(s, inject = inject)), 0L)
  expect_match(fuzz_code(s), "flag", fixed = TRUE)
})

test_that("the default shrink predicate keeps the same finding", {
  skip_if_no_gpu()
  inject <- function(v) {
    if ("flag" %in% names(v)) v$flag <- !v$flag
    v
  }
  cc <- fuzz_make_case(
    x = tb(a = 1:6, b = 6:1),
    steps = list(quote(mutate(flag = a > 0L)), quote(filter(b > 0L)))
  )
  s <- fuzz_shrink(cc, fails = NULL, inject = inject)
  before <- fuzz_check(cc, inject = inject)
  after <- fuzz_check(s, inject = inject)
  expect_gt(nrow(after), 0L)
  expect_true(all(after$signature %in% before$signature))
})

test_that("a planted integer64 corruption is found", {
  skip_if_no_gpu()
  x <- tb(a = bit64::as.integer64(c("9007199254740993", "2")))
  cc <- fuzz_make_case(x, list(quote(filter(!is.na(a)))))
  inject <- function(v) {
    v$a[1] <- bit64::as.integer64("9007199254740992")
    v
  }
  res <- fuzz_check(cc, inject = inject)
  expect_gt(nrow(res), 0L)
  expect_true(all(res$kind == "value"))
})

test_that("verdicts are deterministic across two runs", {
  skip_if_no_gpu()
  for (s in c(2L, 5L, 9L, 14L, 21L, 33L, 400L, 2306L, 3105L)) {
    cc <- fuzz_case(s)
    a <- fuzz_check(cc)
    b <- fuzz_check(cc)
    expect_identical(a[c("mode", "kind", "category", "signature")],
                     b[c("mode", "kind", "category", "signature")], info = paste("seed", s))
  }
})

test_that("the pipeline does not depend on the session's collation", {
  # these seeds generated different pipelines under C and en_US.UTF-8
  available <- withr::with_collate(
    Sys.getlocale("LC_COLLATE"),
    nzchar(suppressWarnings(Sys.setlocale("LC_COLLATE", "en_US.UTF-8")))
  )
  skip_if_not(available, "en_US.UTF-8 is not available")
  for (s in c(2306L, 3105L, 3406L, 3452L)) {
    a <- withr::with_collate("C", fuzz_code(fuzz_case(s)))
    b <- withr::with_collate("en_US.UTF-8", fuzz_code(fuzz_case(s)))
    expect_identical(a, b, info = paste("seed", s))
  }
})

test_that("a run pins the session state and leaves the session as it found it", {
  before <- list(
    collate = Sys.getlocale("LC_COLLATE"),
    tz = Sys.getenv("TZ", NA),
    language = Sys.getenv("LANGUAGE", NA)
  )
  cc <- fuzz_make_case(
    tb(t = .POSIXct(1578000000, tz = "Asia/Tokyo")),
    list(quote(mutate(same = t == as.POSIXct("2020-01-03 12:00:00", tz = "UTC"))))
  )
  o <- fuzz_oracle(cc)
  # the harness runs with lubridate loaded, whatever the session had loaded
  expect_true(isNamespaceLoaded("lubridate"))
  expect_length(o$warnings, 0L)
  expect_identical(Sys.getlocale("LC_COLLATE"), before$collate)
  expect_identical(Sys.getenv("TZ", NA), before$tz)
  expect_identical(Sys.getenv("LANGUAGE", NA), before$language)
})

test_that("campaigns report findings as a TSV and stats", {
  skip_if_no_gpu()
  out <- withr::local_tempfile(fileext = ".tsv")
  inject <- function(v) {
    if (nrow(v) > 0L && is.integer(v[[1]])) v[[1]][1] <- v[[1]][1] + 1L
    v
  }
  res <- fuzz_campaign(1:12, out = out, inject = inject)
  expect_true(all(c("seed", "mode", "kind", "category", "signature", "detail", "code") %in% names(res)))
  expect_gt(nrow(res), 0L)
  read <- utils::read.delim(out, quote = "", stringsAsFactors = FALSE)
  expect_equal(nrow(read), nrow(res))
  expect_true(all(c("seed", "mode", "kind", "category", "signature", "detail", "code") %in% names(read)))
  stats <- attr(res, "stats")
  expect_equal(stats$cases, 12L)
  expect_true(is.numeric(stats$native_share) && stats$native_share >= 0 && stats$native_share <= 1)
  expect_true(is.numeric(stats$plan_break_share))
  # native_share counts pipelines that errored; this one does not
  expect_true(is.numeric(stats$native_ok_share))
  expect_lte(stats$native_ok_share, stats$native_share)
  expect_true(length(stats$masked) == 0L || !is.null(names(stats$masked)))
})

test_that("most generated pipelines reach the GPU without a CPU fallback", {
  skip_if_no_gpu()
  stats <- attr(fuzz_campaign(1:40), "stats")
  expect_gte(stats$native_share, 0.30)
})

test_that("fallbacks are counted from F88's condition class", {
  skip_if_no_gpu()
  native <- fuzz_make_case(plain, list(quote(filter(id > 1L))))
  expect_equal(fuzz_gpu(native, "eager")$fallbacks, 0L)
  expect_equal(fuzz_gpu(native, "lazy")$fallbacks, 0L)
  # a user-defined function has no GPU translation
  cpu <- fuzz_make_case(plain, list(quote(mutate(w = (function(z) z)(id)))))
  expect_gt(fuzz_gpu(cpu, "eager")$fallbacks, 0L)
})

test_that("a warning raised at collect() on a lazy plan counts as that side warning", {
  skip_if_no_gpu()
  # F80's documented move: the lazy plan raises this at collect(), the eager
  # verb and dplyr at the verb. Same verdict either way.
  x <- tb(a = c(1L, 2L))
  cc <- fuzz_make_case(x, list(quote(mutate(b = a + 2147483647L))))
  for (mode in c("eager", "lazy")) {
    out <- fuzz_gpu(cc, mode)
    expect_null(out$error, info = mode)
    expect_gte(length(out$warnings), 1L)
  }
  expect_equal(nrow(fuzz_check(cc)), 0L)
})

test_that("the quick subset is seeded, bounded and its skips are findings", {
  seeds <- fuzz_quick_seeds()
  expect_true(is.integer(seeds))
  expect_gte(length(seeds), 40L)
  expect_false(anyDuplicated(seeds) > 0L)
  skips <- fuzz_quick_skips()
  if (length(skips) > 0L) {
    expect_true(is.character(skips))
    expect_false(is.null(names(skips)))
    expect_true(all(nzchar(unname(skips))))
    expect_true(all(as.integer(names(skips)) %in% seeds))
  }
})

test_that("the quick subset keeps error and empty-result seeds", {
  run <- setdiff(fuzz_quick_seeds(), as.integer(names(fuzz_quick_skips())))
  outs <- lapply(run, function(s) fuzz_oracle(fuzz_case(s)))
  errors <- sum(vapply(outs, function(o) !is.null(o$error), NA))
  empty <- sum(vapply(outs, function(o) is.null(o$error) && nrow(o$value) == 0L, NA))
  expect_gte(errors, 2L)
  expect_gte(empty, 2L)
})
