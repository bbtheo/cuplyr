# #69: calls that dplyr rejects must not return a result on a tbl_gpu. Each
# call below raises an error in dplyr 1.2.1 (pinned with tools/r on plain
# tibbles); cuplyr must raise the same error (class, call, parent class,
# message), eager and lazy. The ordinary neighbours of each call stay native
# (`cuplyr.fallback = "error"`), so the checks cost the common path nothing.
#
# dplyr 1.2.1 facts these tests rely on:
# - if_else() is vctrs::vec_if_else(): `true`/`false`/`missing` must recycle
#   to the size of `condition`, so a size-1 condition with a column value is
#   "Can't recycle `true` (size 3) to size 1." (vctrs_error_incompatible_size)
#   unless the group/table has one row. An all-constant if_else() is a scalar.
# - lag()/lead() call check_number_whole(n): `n = NULL` is "`n` must be a whole
#   number, not `NULL`."
# - sum(na.rm = x) is base sum() of nothing with a length-n na.rm: "'length =
#   3' in coercion to 'logical(1)'".
# - arrange() evaluates `-key` as `..i = -key` in a mutate(): `-` of a Date,
#   POSIXct or string column is an error; of a factor, a warning and NA keys.
# - arrange() of an environment variable sorts by its value (size n), ignores
#   a size-1 value or NULL, and errors on any other size.

for (p69_fn in c("if_else", "lag", "lead", "desc", "case_when", "between", "first", "last", "nth")) {
  assign(p69_fn, getExportedValue("dplyr", p69_fn))
}
rm(p69_fn)

p69_df <- function() {
  data.frame(
    g = c(1, 1, 2),
    x = c(3, 1, 2),
    y = 4:6,
    i = c(3L, NA, 1L),
    b = c(TRUE, FALSE, NA),
    dt = as.Date("2020-01-01") + c(2, 0, 1),
    ts = as.POSIXct("2020-01-01", tz = "UTC") + c(2, 0, 1),
    s = c("b", "a", "c"),
    f = factor(c("b", "a", "c")),
    stringsAsFactors = FALSE
  )
}

p69_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings(suppressMessages({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    })),
    error = function(e) e
  )
}

# dplyr errors on `df`; the GPU side raises the same error, eager and lazy.
p69_same_error <- function(df, pipeline) {
  cpu <- p69_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"), info = "dplyr itself errors")
  if (!inherits(cpu, "error")) {
    return(invisible())
  }
  for (lazy in c(FALSE, TRUE)) {
    mode <- if (lazy) "lazy" else "eager"
    gpu <- p69_err(tbl_gpu(df, lazy = lazy), pipeline)
    expect_true(inherits(gpu, "error"), info = sprintf("GPU side errors (%s)", mode))
    if (!inherits(gpu, "error")) next
    expect_identical(class(gpu), class(cpu), info = sprintf("class (%s): %s", mode, conditionMessage(gpu)))
    expect_identical(rlang::format_error_call(gpu$call), rlang::format_error_call(cpu$call),
                     info = sprintf("call (%s)", mode))
    expect_identical(class(gpu$parent), class(cpu$parent), info = sprintf("parent class (%s)", mode))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu), info = sprintf("message (%s)", mode))
  }
  invisible(cpu)
}

# The same result as dplyr, eager and lazy, with no CPU fallback.
p69_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# The same result as dplyr, eager and lazy (any route).
p69_same <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---------------------------------------------------------------------------
# 1. if_else() with a size-1 condition
# ---------------------------------------------------------------------------

test_that("if_else() with a size-1 condition and a column value is dplyr's recycling error", {
  skip_if_no_gpu()
  df <- p69_df()
  th <- NaN
  cpu <- p69_same_error(df, function(d) mutate(d, z = if_else(is.nan(th), x, y)))
  expect_s3_class(cpu$parent, "vctrs_error_incompatible_size")
  p69_same_error(df, function(d) mutate(d, z = if_else(TRUE, x, 0)))
  p69_same_error(df, function(d) mutate(d, z = if_else(FALSE, 0, x)))
  # the column in `missing` only
  p69_same_error(df, function(d) mutate(d, z = if_else(TRUE, 1, 0, missing = x)))
  # an NA condition (a logical NA literal)
  p69_same_error(df, function(d) mutate(d, z = if_else(NA, x, 0)))
  # inside a larger expression
  p69_same_error(df, function(d) mutate(d, z = if_else(th > 0, x, y) + 1))
  # an injected scalar
  flag <- TRUE
  p69_same_error(df, function(d) mutate(d, z = if_else(!!flag, x, y)))
  # transmute() too
  p69_same_error(df, function(d) transmute(d, z = if_else(TRUE, x, y)))
})

test_that("if_else() with a size-1 condition errors per group, in filter() and in summarise()", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_same_error(df, function(d) d |> group_by(g) |> mutate(z = if_else(TRUE, x, 0)))
  p69_same_error(df, function(d) mutate(d, z = if_else(TRUE, x, 0), .by = g))
  p69_same_error(df, function(d) filter(d, if_else(TRUE, x > 1, FALSE)))
  p69_same_error(df, function(d) summarise(d, m = sum(if_else(TRUE, x, 0))))
  p69_same_error(df, function(d) d |> group_by(g) |> summarise(m = sum(if_else(TRUE, x, 0))))
})

test_that("if_else() with a size-1 condition works where dplyr's sizes agree", {
  skip_if_no_gpu()
  df <- p69_df()
  # a one-row table: every argument has size 1
  p69_same(df[1, , drop = FALSE], function(d) mutate(d, z = if_else(TRUE, x, 0)))
  # a grouped table whose every group has one row
  p69_same(df, function(d) d |> group_by(x) |> mutate(z = if_else(TRUE, x, 0)), arrange_by = "x")
  # zero rows: dplyr errors (size 0 can't recycle to size 1)
  p69_same_error(df[0, , drop = FALSE], function(d) mutate(d, z = if_else(TRUE, x, 0)))
})

test_that("if_else() with a column condition or all-constant arguments stays native", {
  skip_if_no_gpu()
  df <- p69_df()
  th <- NaN
  p69_native(df, function(d) mutate(d, z = if_else(x > 1, x, 0)))
  p69_native(df, function(d) mutate(d, z = if_else(b, x, 0, missing = -1)))
  # all arguments constant: a scalar, recycled by mutate()
  p69_native(df, function(d) mutate(d, z = if_else(is.nan(th), 1, 0)))
  p69_native(df, function(d) d |> group_by(g) |> mutate(z = if_else(x > 1, y, 0L)))
  p69_native(df, function(d) filter(d, if_else(x > 1, TRUE, FALSE)))
  # case_when()'s catch-all `TRUE ~` is a valid size-1 condition
  p69_native(df, function(d) mutate(d, z = case_when(x > 2 ~ 1, TRUE ~ x)))
})

test_that("if_else() with a one-value condition that is not a literal is dplyr's error", {
  skip_if_no_gpu()
  df <- p69_df()
  th <- 0
  # constant conditions the parser does not fold to a literal
  p69_same_error(df, function(d) mutate(d, z = if_else(th > -1, x, y)))
  p69_same_error(df, function(d) mutate(d, z = if_else(th %in% c(0, 1), x, y)))
  p69_same_error(df, function(d) mutate(d, z = if_else(between(th, -1, 1), x, y)))
  # one value per group: n(), an aggregate
  p69_same_error(df, function(d) mutate(d, z = if_else(n() > 2, x, y)))
  p69_same_error(df, function(d) mutate(d, z = if_else(sum(x) > 0, x, y)))
  p69_same_error(df, function(d) d |> group_by(g) |> mutate(z = if_else(n() > 1, x, y)))
  p69_same_error(df, function(d) filter(d, if_else(n() > 2, x > 1, FALSE)))
  p69_same_error(df, function(d) summarise(d, m = sum(if_else(n() > 2, x, 0))))
})

test_that("if_else() with a one-value condition and one-value values stays native", {
  skip_if_no_gpu()
  df <- p69_df()
  flag <- TRUE
  th <- 0
  # a literal-condition if_else() inside an expression that reads a column
  p69_native(df, function(d) mutate(d, z = x + if_else(flag, 1, 0)))
  p69_native(df, function(d) mutate(d, z = x + if_else(th > -1, 1, 0)))
  # size-1 values: aggregates
  p69_native(df, function(d) mutate(d, z = if_else(flag, mean(x), median(x))))
  p69_native(df, function(d) d |> group_by(g) |> mutate(z = if_else(n() > 1, mean(x), 0)))
  p69_native(df, function(d) d |> group_by(g) |> summarise(m = if_else(flag, mean(x), median(x))),
             arrange_by = "g")
  # a rowwise table: every row is its own group of size 1, so dplyr recycles
  # (native on the GPU before the size-1 check existed)
  p69_native(df, function(d) d |> rowwise() |> mutate(z = if_else(flag, x, y)))
  p69_native(df, function(d) d |> rowwise() |> mutate(z = if_else(NA, x, y)))
  p69_native(df, function(d) d |> rowwise() |> mutate(z = if_else(NA, x, y, missing = y)))
  # rowwise after a pending step that keeps rows (any route)
  p69_same(df, function(d) d |> filter(x > 1) |> rowwise() |> mutate(z = if_else(flag, x, y)))
})

test_that("on a zero-row rowwise table a one-value if_else() condition is dplyr's error", {
  skip_if_no_gpu()
  df <- p69_df()
  flag <- TRUE
  # dplyr evaluates the expression once on the empty columns: size 0 can't
  # recycle to the condition's size 1
  p69_same_error(df, function(d) d |> filter(x > 100) |> rowwise() |> mutate(z = if_else(NA, x, y)))
  p69_same_error(df, function(d) d |> filter(x > 100) |> rowwise() |> mutate(z = if_else(flag, x, y)))
  p69_same_error(df[0, , drop = FALSE], function(d) d |> rowwise() |> mutate(z = if_else(NA, x, y)))
  p69_same_error(df[0, , drop = FALSE], function(d) d |> rowwise() |> mutate(z = if_else(flag, x, y)))
  p69_same_error(df[0, , drop = FALSE], function(d) d |> rowwise() |> transmute(z = if_else(flag, x, y)))
})

test_that("more one-value if_else() conditions are dplyr's error", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_same_error(df, function(d) d |> group_by(g) |> mutate(z = if_else(cur_group_id() == 1L, x, y)))
  p69_same_error(df, function(d) mutate(d, z = if_else(n() > 1, x, y), .by = g))
  p69_same_error(df, function(d) mutate(d, dplyr::across(c(x, y), ~ if_else(n() > 2, .x, 0))))
  p69_same_error(df, function(d) mutate(d, z = if_else(first(x) > 1, x, y)))
})

test_that("an if_else() condition that is not logical is dplyr's error", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_same_error(df, function(d) mutate(d, z = if_else(1, mean(x), 0)))
  p69_same_error(df, function(d) mutate(d, z = if_else("a", mean(x), 0)))
  p69_same_error(df, function(d) filter(d, if_else(1, n() > 2, TRUE)))
  # a double column as the condition
  p69_same_error(df, function(d) mutate(d, z = if_else(x, x, 0)))
})

test_that("an NA if_else() condition with one-value values gives dplyr's NA or `missing`", {
  skip_if_no_gpu()
  df <- p69_df()
  flag <- NA
  p69_same(df, function(d) mutate(d, z = if_else(NA, mean(x), 0)))
  p69_same(df, function(d) mutate(d, z = x + if_else(NA, 1, 0)))
  p69_same(df, function(d) d |> group_by(g) |> mutate(z = if_else(NA, n(), 0L)))
  p69_same(df, function(d) filter(d, if_else(NA, n() > 2, TRUE)))
  p69_same(df, function(d) mutate(d, z = if_else(NA, mean(x), 0, missing = -1)))
  p69_same(df, function(d) mutate(d, z = if_else(flag, mean(x), median(x))))
  p69_same(df, function(d) d |> group_by(g) |> summarise(m = if_else(flag, mean(x), median(x))),
           arrange_by = "g")
})

# ---------------------------------------------------------------------------
# 2. lag()/lead() with `n = NULL`
# ---------------------------------------------------------------------------

test_that("lag()/lead() with `n = NULL` is dplyr's error", {
  skip_if_no_gpu()
  df <- p69_df()
  cpu <- p69_same_error(df, function(d) mutate(d, z = lag(x, n = NULL)))
  expect_match(conditionMessage(cpu$parent), "`n` must be a whole number, not `NULL`", fixed = TRUE)
  p69_same_error(df, function(d) mutate(d, z = lead(x, n = NULL)))
  p69_same_error(df, function(d) mutate(d, z = lag(x, NULL)))
  p69_same_error(df, function(d) mutate(d, z = lag(x, n = NULL, default = 0)))
  p69_same_error(df, function(d) d |> group_by(g) |> mutate(z = lag(x, NULL)))
  p69_same_error(df, function(d) filter(d, !is.na(lag(x, n = NULL))))
  p69_same_error(df, function(d) mutate(d, z = lead(x, n = NULL) + 1))
})

test_that("lag()/lead() with a valid or absent `n`, and NULL defaults, stay native", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_native(df, function(d) mutate(d, z = lag(x)))
  p69_native(df, function(d) mutate(d, z = lead(x, n = 2)))
  p69_native(df, function(d) mutate(d, z = lag(x, 1L, default = NULL)))
  p69_native(df, function(d) mutate(d, z = lag(x, order_by = NULL)))
  p69_native(df, function(d) d |> group_by(g) |> mutate(z = lead(x, n = 1L)))
})

test_that("an argument written `= NULL` whose default is not NULL is R's error", {
  skip_if_no_gpu()
  df <- p69_df()
  # first()/last()/nth() check_bool(na_rm)
  p69_same_error(df, function(d) mutate(d, z = first(x, na_rm = NULL)))
  p69_same_error(df, function(d) mutate(d, z = last(x, na_rm = NULL)))
  p69_same_error(df, function(d) mutate(d, z = nth(x, 2, na_rm = NULL)))
  # base R: paste()'s sep, nchar()'s type
  p69_same_error(df, function(d) mutate(d, z = paste(s, s, sep = NULL)))
  p69_same_error(df, function(d) mutate(d, z = nchar(s, type = NULL)))
  p69_same_error(df, function(d) mutate(d, z = trimws(s, whitespace = NULL)))
  p69_same_error(df, function(d) mutate(d, z = grepl("a", s, ignore.case = NULL)))
  p69_same_error(df, function(d) mutate(d, z = sub("a", "b", s, perl = NULL)))
})

test_that("arguments written `= NULL` that are NULL by default stay native", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_native(df, function(d) mutate(d, z = first(x, default = NULL)))
  p69_native(df, function(d) mutate(d, z = nth(x, 2, order_by = NULL)))
  p69_native(df, function(d) mutate(d, z = paste(s, s, collapse = NULL)))
  p69_native(df, function(d) mutate(d, z = paste0(s, "_", collapse = NULL)))
  # base nchar() accepts `allowNA = NULL` (any route)
  p69_same(df, function(d) mutate(d, z = nchar(s, allowNA = NULL)))
})

# ---------------------------------------------------------------------------
# 3. sum(na.rm = x): the lone argument is named
# ---------------------------------------------------------------------------

test_that("sum(na.rm = <column>) is dplyr's error in mutate() and filter()", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_same_error(df, function(d) mutate(d, s = sum(na.rm = x)))
  p69_same_error(df, function(d) mutate(d, s = sum(na.rm = b)))
  p69_same_error(df, function(d) d |> group_by(g) |> mutate(s = sum(na.rm = x)))
  p69_same_error(df, function(d) mutate(d, s = sum(na.rm = x) + 1))
  p69_same_error(df, function(d) filter(d, sum(na.rm = x) > 1))
  # already right: summarise(), and mean()/max() in mutate()
  p69_same_error(df, function(d) summarise(d, s = sum(na.rm = x)))
  p69_same_error(df, function(d) d |> group_by(g) |> summarise(s = sum(na.rm = x)))
  p69_same_error(df, function(d) mutate(d, s = mean(na.rm = x)))
  p69_same_error(df, function(d) mutate(d, s = max(na.rm = x)))
})

test_that("sum() with an unnamed column, with or without na.rm, stays native", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_native(df, function(d) mutate(d, s = sum(x)))
  p69_native(df, function(d) mutate(d, s = sum(i, na.rm = TRUE)))
  p69_native(df, function(d) mutate(d, s = sum(i, na.rm = FALSE)))
  p69_native(df, function(d) d |> group_by(g) |> mutate(s = sum(x)))
  # the shape dplyr's add_tally(wt = ) emits
  p69_native(df, function(d) add_tally(d, wt = y))
  # a name that sum() puts in `...` is data: same value as dplyr (any route)
  p69_same(df, function(d) mutate(d, s = sum(x = x)))
})

# ---------------------------------------------------------------------------
# 4/6. arrange() with `-` of a column R can't negate
# ---------------------------------------------------------------------------

test_that("arrange(-<Date/POSIXct/string>) is dplyr's error", {
  skip_if_no_gpu()
  df <- p69_df()
  cpu <- p69_same_error(df, function(d) arrange(d, -dt))
  expect_match(conditionMessage(cpu$parent), "unary - is not defined for \"Date\" objects", fixed = TRUE)
  p69_same_error(df, function(d) arrange(d, -ts))
  p69_same_error(df, function(d) arrange(d, -s))
  # later key, after a valid one
  p69_same_error(df, function(d) arrange(d, x, -dt))
  # grouped, with and without .by_group (the key's index counts the groups)
  p69_same_error(df, function(d) d |> group_by(g) |> arrange(-dt))
  p69_same_error(df, function(d) d |> group_by(g) |> arrange(-dt, .by_group = TRUE))
  # zero rows: still an error in dplyr
  p69_same_error(df[0, , drop = FALSE], function(d) arrange(d, -s))
  p69_same_error(df[0, , drop = FALSE], function(d) arrange(d, -dt))
  # after a lazy step
  p69_same_error(df, function(d) d |> mutate(e = dt + 1) |> arrange(-e))
})

test_that("arrange(-<factor>) warns like dplyr and keeps dplyr's order", {
  skip_if_no_gpu()
  df <- p69_df()
  pipe <- function(d) arrange(d, -f)
  expect_warning(expected <- pipe(tibble::as_tibble(df)), "not meaningful for factors")
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(out <- dplyr::collect(pipe(tbl_gpu(df, lazy = lazy))), "not meaningful for factors")
    expect_equal(oracle_normalize(out), oracle_normalize(expected), info = if (lazy) "lazy" else "eager")
  }
})

test_that("arrange(-x) and desc() of a negatable or any column stay native", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_native(df, function(d) arrange(d, -x))
  p69_native(df, function(d) arrange(d, -i))
  p69_native(df, function(d) arrange(d, -y, x))
  p69_native(df, function(d) arrange(d, -b, x))
  p69_native(df, function(d) arrange(d, desc(dt)))
  p69_native(df, function(d) arrange(d, desc(ts)))
  p69_native(df, function(d) arrange(d, desc(s)))
  p69_native(df, function(d) d |> group_by(g) |> arrange(-x, .by_group = TRUE))
})

# ---------------------------------------------------------------------------
# 5. cumsum(x) + NULL (already right: regression)
# ---------------------------------------------------------------------------

test_that("a NULL operand is dplyr's size error", {
  skip_if_no_gpu()
  df <- p69_df()
  p69_same_error(df, function(d) mutate(d, z = cumsum(x) + NULL))
  p69_same_error(df, function(d) mutate(d, z = x * NULL))
  nl <- NULL
  p69_same_error(df, function(d) mutate(d, z = x + nl))
  p69_same_error(df, function(d) filter(d, cumsum(x) + NULL > 1))
})

# ---------------------------------------------------------------------------
# 7. arrange() of an environment variable (already right: regression)
# ---------------------------------------------------------------------------

test_that("arrange() of an environment variable sorts, ignores or errors like dplyr", {
  skip_if_no_gpu()
  df <- p69_df()
  v <- c(2, 1, 3)
  k <- 5
  nl <- NULL
  v2 <- c(1, 2)
  p69_same(df, function(d) arrange(d, v))
  p69_same(df, function(d) arrange(d, -v))
  p69_same(df, function(d) arrange(d, desc(v)))
  p69_same(df, function(d) arrange(d, k))
  p69_same(df, function(d) arrange(d, nl))
  p69_same(df, function(d) arrange(d, k, x))
  p69_same_error(df, function(d) arrange(d, v2))
  # a column wins over a variable of the same name
  x <- c(9, 8, 7)
  p69_native(df, function(d) arrange(d, x))
})
