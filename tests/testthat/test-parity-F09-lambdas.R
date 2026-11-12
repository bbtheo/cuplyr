# F09 acceptance tests: across() lambda semantics (gaps A6f, A6g) and the
# sequential-expansion hazard F08 left on the CPU fallback.
#
# A6f: `.fns` given as a VALUE (a symbol bound to a formula/function/list,
#      or a call such as local(~ . + 1)) is evaluated in the caller's env and
#      dispatched on its runtime type. dplyr never inlines such values: they
#      are called with their own environment, so their free variables do NOT
#      see data columns. Literal list() of lambdas names outputs
#      "{.col}_{.fn}" with positional labels.
# A6g: non-inlinable literal lambdas (return(), ..1, several formals) behave
#      like dplyr: real function frame, data-mask environment, arguments
#      shadow columns.
# Hazard: one across() dot is evaluated against the data as it was before
#      that dot. mutate()/transmute() should do this on the GPU (staging);
#      summarise() may keep the CPU fallback.
#
# Tests marked "native" run with options(cuplyr.fallback = "error").

f09_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    x = c(1, NA, 3, 4, 5),
    y = c(10, 20, 30, 40, 50),
    z = c(2, 4, 8, 16, 32)
  )
}

f09_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f09_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f09_both(df, pipeline, ...)
}

# =============================================================================
# A6f: formula values
# =============================================================================

test_that("A6f native: a symbol bound to a formula is inlined (mutate, summarise)", {
  skip_if_no_gpu()
  f <- ~ .x * 2 + 1
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), f)))
  f09_native(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, z), f))
  })
  m <- ~ max(.x, na.rm = TRUE)
  f09_native(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(c(x, y), m))
  })
  f09_native(f09_df(), function(d) dplyr::summarise(d, dplyr::across(c(x, y), m)))
})

test_that("A6f native: a formula value works in transmute() and on 0 rows", {
  skip_if_no_gpu()
  f <- ~ .x - 1
  f09_native(f09_df(), function(d) dplyr::transmute(d, dplyr::across(c(x, y), f)))
  f09_native(f09_df()[0, ], function(d) dplyr::mutate(d, dplyr::across(c(x, y), f)))
})

test_that("A6f native: .fns given as a call that returns a formula", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), local(~ . + 1))))
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(x, z), list(f = local(~ . + 1))))
  })
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(x, local({
      `_local_var` <- 3
      ~ . + `_local_var`
    })))
  })
})

test_that("A6f: a formula value resolves free variables in its own env, not the data", {
  skip_if_no_gpu()
  # `y` is both a column and a variable in the formula's env: dplyr uses
  # the formula env's value (100), not the column.
  f <- local({
    y <- 100
    ~ .x / y
  })
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), f)))
  fm <- local({
    y <- 100
    ~ max(.x, na.rm = TRUE) / y
  })
  f09_both(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(c(x, z), list(f = fm)))
  })

  # Local formula env (#5881) with a string column
  fp <- local({
    prefix <- "foo"
    ~ paste(prefix, .x)
  })
  s <- tibble::tibble(x = c("a", "b", NA))
  f09_both(s, function(d) dplyr::mutate(d, dplyr::across(x, fp)))
  f09_both(s, function(d) dplyr::mutate(d, dplyr::across(x, list(f = fp))))
})

test_that("A6f: a formula value can't see data columns (error like dplyr)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2), w_f09_col = c(3, 4))
  f <- ~ .x + w_f09_col
  expect_error(dplyr::mutate(df, dplyr::across(x, f)), "w_f09_col")
  expect_error(collect(dplyr::mutate(tbl_gpu(df), dplyr::across(x, f))), "w_f09_col")
  expect_error(
    collect(dplyr::summarise(tbl_gpu(df), dplyr::across(x, list(a = f)))),
    "w_f09_col"
  )
})

test_that("A6f: cur_column() inside a formula value", {
  skip_if_no_gpu()
  fc <- ~ .x + nchar(dplyr::cur_column())
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), fc)))
})

# =============================================================================
# A6f: function values and lists
# =============================================================================

test_that("A6f native: a symbol bound to a closure or \\(x) lambda", {
  skip_if_no_gpu()
  fv <- function(v) v * 10
  bs <- \(v) v - 1
  k <- 3
  fk <- function(v) v + k
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), fv)))
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), bs)))
  f09_native(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, z), fk))
  })
})

test_that("A6f: a closure value with a free variable named like a column uses its env", {
  skip_if_no_gpu()
  fy <- local({
    y <- 1000
    function(v) v + y
  })
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), fy)))
})

test_that("A6f native: a symbol bound to a list of formulas and functions", {
  skip_if_no_gpu()
  fns <- list(a = ~ max(.x, na.rm = TRUE), b = mean)
  f09_native(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(c(x, y), fns))
  })
  f09_native(f09_df(), function(d) dplyr::summarise(d, dplyr::across(c(x, y), fns)))
  f09_native(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(c(x, y), fns, .names = "{.fn}_{.col}"))
  })
  unnamed <- list(~ .x + 1, ~ .x * 2)
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), unnamed)))
})

test_that("A6f: a list value in mutate() (fallback allowed)", {
  skip_if_no_gpu()
  fl <- list(a = ~ .x + 1, b = mean)
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(x, fl)))
  f09_both(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, z), fl, .names = "{.col}.{.fn}"))
  })
})

test_that("A6f native: literal list() of lambdas is named {.col}_{.fn}", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(x, list(~ .x + 1))))
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(x, z), list(a = ~ .x, ~ .x * 2)))
  })
  f09_native(f09_df(), function(d) {
    dplyr::summarise(d, dplyr::across(c(x, y), list(function(v) max(v, na.rm = TRUE))))
  })
  # Literal lambdas are maskable: they can use columns.
  f09_both(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(x, z), list(function(v) v + mean(y))))
  })
})

test_that("A6f: list(mean) on a column named mean (#6545)", {
  skip_if_no_gpu()
  df <- tibble::tibble(mean = c(1, 2, 3, 4, 5))
  f09_both(df, function(d) dplyr::summarise(d, dplyr::across(dplyr::everything(), list(mean))))
  f09_both(df, function(d) dplyr::summarise(d, dplyr::across(dplyr::everything(), mean)))
})

test_that("A6f native: integer columns keep dplyr's types through a list value", {
  skip_if_no_gpu()
  # A 3-row, 2-group table hits a separate, pre-existing grouped-summarise
  # row-order gap (groups come back 2, 1), unrelated to across().
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), n = c(1L, 2L, 3L, 4L, 5L))
  fns <- list(mx = ~ max(.x), avg = mean)
  f09_native(df, function(d) dplyr::summarise(dplyr::group_by(d, g), dplyr::across(n, fns)))
})

test_that("A6f native: if_any()/if_all() with a formula value or a list of lambdas", {
  skip_if_no_gpu()
  pf <- ~ .x > 5
  f09_native(f09_df(), function(d) dplyr::filter(d, dplyr::if_any(c(x, z), pf)))
  f09_native(f09_df(), function(d) dplyr::filter(d, dplyr::if_all(c(x, z), pf)))
  f09_native(f09_df(), function(d) dplyr::mutate(d, k = dplyr::if_any(c(x, z), pf)))
  two <- list(~ .x > 20, ~ .x < 2)
  f09_native(f09_df(), function(d) dplyr::filter(d, dplyr::if_any(c(x, z), two)))
  f09_native(f09_df(), function(d) dplyr::filter(d, dplyr::if_all(c(x, z), list(~ .x > 0, ~ .x < 20))))
})

test_that("A6f: invalid .fns values error like dplyr", {
  skip_if_no_gpu()
  gdf <- tbl_gpu(f09_df())
  msg <- "must be a function, a formula, or a list of functions/formulas"
  expect_error(collect(dplyr::mutate(gdf, dplyr::across(x, 42))), msg)
  expect_error(collect(dplyr::summarise(gdf, dplyr::across(x, "mean"))), msg)
  expect_error(collect(dplyr::mutate(gdf, dplyr::across(x, quote(~ .x)))))
  expect_error(
    collect(dplyr::summarise(gdf, dplyr::across(x, list(f = mean, f = mean)))),
    "Names must be unique"
  )
  boom <- function() stop("oh no f09")
  expect_error(collect(dplyr::summarise(gdf, dplyr::across(x, boom()))), "oh no f09")
})

# =============================================================================
# A6g: non-inlinable literal lambdas
# =============================================================================

test_that("A6g: return() inside a literal lambda or formula (mutate, transmute)", {
  skip_if_no_gpu()
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), function(v) return(v + 1))))
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), ~ return(.x * 2))))
  f09_both(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, z), \(v) {
      if (TRUE) return(v - 1)
      v
    }))
  })
  f09_both(f09_df(), function(d) dplyr::transmute(d, dplyr::across(c(x, y), function(v) return(v / 2))))
})

test_that("A6g: return() inside a lambda in summarise()", {
  skip_if_no_gpu()
  f09_both(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
                     dplyr::across(c(x, y), function(v) return(max(v, na.rm = TRUE))))
  })
})

test_that("A6g: non-inlinable lambdas see .data, their lexical env, and shadow columns", {
  skip_if_no_gpu()
  f09_both(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(z, function(col) {
      return(.data[["y"]])
    }))
  })
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), function(y) return(y))))
  off <- 7
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), ~ return(.x + off))))
})

test_that("A6g: ..1 formulas and multi-argument lambdas with extra args", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, z), ~ ..1 + mean(y))))
  f09_both(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(x, z), function(a, b) a + b, 5))
  })
  f09_both(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(x, function(a, b) a + b, b = y))
  })
  f09_both(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(c(x, y), function(a, b) max(a) + b, b = 1))
  })
})

# =============================================================================
# Sequential-expansion hazard: all expansions of one across() read the
# pre-across() data
# =============================================================================

test_that("hazard native: mutate() lambdas that read an overwritten column", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, y, z), ~ .x / y)))
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, y, z), ~ .x / .data$y)))
  f09_native(f09_df(), function(d) dplyr::mutate(d, a = 2, dplyr::across(c(x, y, z), ~ .x / y)))
  f09_native(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, y, z), ~ .x / y))
  })
  # Each expansion reads the other's source: a cycle
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, y), ~ .x + x + y)))
  # Renaming onto a later source column
  f09_native(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(y = x, z = y), ~ .x)))
})

test_that("hazard native: .keep and .before with a staged across()", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y, z), ~ .x / y), .keep = "used")
  })
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, w = 1, dplyr::across(c(x, z), ~ .x / y), .keep = "unused")
  })
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(y, x), list(n = ~ .x / y, y = ~ .x)), .before = x)
  })
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y, z), list(r = ~ .x / y, y = ~ .x + 1)), .keep = "none")
  })
})

test_that("hazard native: transmute() and chained lazy mutates", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) dplyr::transmute(d, dplyr::across(c(x, y, z), ~ .x / y)))
  f09_native(f09_df(), function(d) {
    d |>
      dplyr::mutate(y = y + 1) |>
      dplyr::mutate(dplyr::across(c(x, y, z), ~ .x / y)) |>
      dplyr::mutate(q = x + y)
  })
  f09_native(f09_df()[0, ], function(d) dplyr::mutate(d, dplyr::across(c(x, y, z), ~ .x / y)))
})

test_that("hazard: string access to an overwritten column still reads the old value", {
  skip_if_no_gpu()
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(x, y, z), ~ .x / .data[["y"]])))
  f09_both(f09_df(), function(d) dplyr::mutate(d, dplyr::across(c(y, x), ~ .x + get("y"))))
})

test_that("hazard: summarise() and factor columns (fallback allowed)", {
  skip_if_no_gpu()
  f09_both(f09_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(c(x, y, z), ~ max(.x) / max(y)))
  })
  fdf <- tibble::tibble(
    f = factor(c("b", "a", "b"), levels = c("b", "a")),
    y = c(1, 2, 3)
  )
  f09_both(fdf, function(d) dplyr::mutate(d, dplyr::across(c(y = f, z = y), ~ .x)))
})

# =============================================================================
# Round 1 review findings (B1-B4, N5)
# =============================================================================

test_that("B1: guards see columns created by earlier dots of the same call", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2))
  w <- 5
  f <- ~ .x + w
  f09_both(df, function(d) dplyr::mutate(d, w = 1, dplyr::across(x, f)))
  fm <- ~ max(.x) + w
  f09_both(df, function(d) dplyr::summarise(d, w = 1, dplyr::across(x, list(m = fm))))
  withr::local_options(lifecycle_verbosity = "quiet")
  k <- 10
  f09_both(df, function(d) dplyr::mutate(d, k = 2, dplyr::across(x, `*`, k)))
})

test_that("B2: pkg::fn values are called, not stripped to a bare name", {
  skip_if_no_gpu()
  s <- tibble::tibble(x = c("hello world", "foo bar"))
  f09_both(s, function(d) dplyr::mutate(d, dplyr::across(x, tools::toTitleCase)))
  # stats::lag() is not dplyr::lag(): it doesn't shift a plain vector. dplyr's
  # column also carries a `tsp` attribute, which a GPU column can't hold, so
  # compare the values only.
  num <- tibble::tibble(x = c(1, 2, 3))
  expect_equal(as.numeric(dplyr::mutate(num, dplyr::across(x, stats::lag))$x), c(1, 2, 3))
  for (lazy in c(FALSE, TRUE)) {
    out <- collect(dplyr::mutate(tbl_gpu(num, lazy = lazy), dplyr::across(x, stats::lag)))
    expect_equal(as.numeric(out$x), c(1, 2, 3))
    out <- collect(dplyr::mutate(tbl_gpu(num, lazy = lazy), dplyr::across(x, list(l = stats::lag))))
    expect_equal(names(out), c("x", "x_l"))
    expect_equal(as.numeric(out$x_l), c(1, 2, 3))
  }
})

test_that("B3: function values whose body needs a call frame are called", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  fns <- list(r = rev)
  f09_both(df, function(d) dplyr::mutate(d, dplyr::across(x, fns)))
  dbl <- function(v) UseMethod("dbl")
  dbl.default <- function(v) v * 2
  registerS3method("dbl", "default", dbl.default, envir = environment())
  f09_both(df, function(d) dplyr::mutate(d, dplyr::across(x, dbl)))
  inner <- function(v) sapply(v, function(v) v * 2)
  f09_both(df, function(d) dplyr::mutate(d, dplyr::across(x, inner)))
  mis <- function(v) if (missing(v)) NA else v + 1
  f09_both(df, function(d) dplyr::mutate(d, dplyr::across(x, mis)))
})

test_that("B4: an embraced .fns ({{ fn }}) behaves like the value", {
  skip_if_no_gpu()
  wrap <- function(d, fn) dplyr::mutate(d, dplyr::across(x, {{ fn }}))
  wrap_s <- function(d, fn) dplyr::summarise(d, dplyr::across(x, {{ fn }}))
  df <- tibble::tibble(x = c(1, 4, 9))
  f09_both(df, function(d) wrap(d, sqrt))
  f09_both(df, function(d) wrap_s(d, mean))
  f09_both(df, function(d) wrap(d, ~ .x + 1))
  f09_both(df, function(d) wrap(d, list(a = ~ .x * 2, b = sqrt)))
})

test_that("N5: staging with .before, and lazy fusion with a staged rename", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(y = x, n = y), ~ .x), .before = x)
  })
  f09_native(f09_df(), function(d) {
    d |>
      dplyr::mutate(a = 1) |>
      dplyr::mutate(dplyr::across(c(y = x, z = y), ~ .x + a))
  })
})

# =============================================================================
# Round 2 review findings (B5, N7)
# =============================================================================

test_that("B5: guards see columns spliced in by an earlier unnamed data-frame dot", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2))
  w <- 5
  f <- ~ .x + w
  g <- function(v) v + w
  f09_both(df, function(d) dplyr::mutate(d, tibble::tibble(w = x * 0 + 1), dplyr::across(x, f)))
  f09_both(df, function(d) dplyr::mutate(d, tibble::tibble(w = x * 0 + 1), dplyr::across(x, g)))
  withr::local_options(lifecycle_verbosity = "quiet")
  k <- 10
  f09_both(df, function(d) {
    dplyr::mutate(d, tibble::tibble(k = x * 0 + 2), dplyr::across(x, `*`, k))
  })
})

test_that("N7 native: staging combined with window functions", {
  skip_if_no_gpu()
  f09_native(f09_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(x, y, z), ~ dplyr::lag(.x) / y))
  })
  f09_native(f09_df(), function(d) {
    dplyr::mutate(d, dplyr::across(c(y, z), ~ cumsum(.x) + y), .keep = "unused")
  })
  f09_native(f09_df(), function(d) {
    d |>
      dplyr::mutate(dplyr::across(c(y = x, x = y), ~ dplyr::lag(.x))) |>
      dplyr::mutate(q = x + y)
  })
})
