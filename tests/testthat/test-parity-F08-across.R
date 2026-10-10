# F08 acceptance tests: across()/if_any()/if_all() small fixes (gaps A6a-A6e).
#
# A6a: a named top-level across() dot packs a data frame column. tbl_gpu
#      can't hold one, so the call goes to the whole-call CPU fallback and
#      hands back dplyr's own result.
# A6b: if_any()/if_all() over an empty selection give FALSE/TRUE.
# A6c: zero-argument if_any()/if_all() use the deprecated everything()
#      default (with dplyr's lifecycle warning) and identity `.fns`.
# A6d: across() without `.cols` does the same.
# A6e: renaming inside `.cols` names the outputs by the new names.
#
# A6b-A6e must stay on the GPU (`cuplyr.fallback = "error"`).

f08_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    x = c(1, NA, 3, 4, 5),
    y = c(10L, 20L, 30L, NA, 50L)
  )
}

f08_lgl <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    p = c(TRUE, FALSE, NA, TRUE, FALSE),
    q = c(FALSE, FALSE, TRUE, NA, TRUE)
  )
}

f08_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# =============================================================================
# A6a: named across() dot
# =============================================================================

test_that("A6a: mutate() with a named across() dot packs a df column like dplyr", {
  skip_if_no_gpu()
  f08_both(f08_df(), function(d) dplyr::mutate(d, s = dplyr::across(c(x, y), ~ .x + 1)))
  f08_both(f08_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), s = dplyr::across(dplyr::everything(), ~ .x * 2))
  })
})

test_that("A6a: summarise() with a named across() dot (grouped and ungrouped)", {
  skip_if_no_gpu()
  f08_both(f08_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), s = dplyr::across(dplyr::everything(), ~ max(.x)))
  })
  f08_both(f08_df(), function(d) {
    dplyr::summarise(d, m = dplyr::across(c(x, y), ~ mean(.x, na.rm = TRUE)))
  })
})

test_that("A6a: named across() mixed with ordinary dots keeps dplyr's column order", {
  skip_if_no_gpu()
  f08_both(f08_df(), function(d) {
    dplyr::mutate(d, a = 2, s = dplyr::across(dplyr::everything(), identity), b = x + 1)
  })
})

test_that("A6a: transmute() with a named across() dot", {
  skip_if_no_gpu()
  f08_both(f08_df(), function(d) dplyr::transmute(d, s = dplyr::across(x)))
})

test_that("A6a: named across() without .fns and with an empty selection", {
  skip_if_no_gpu()
  f08_both(f08_df(), function(d) dplyr::mutate(d, z = dplyr::across(dplyr::everything())))
  f08_both(f08_df(), function(d) dplyr::mutate(d, z = dplyr::across(c(), as.character)))
})

test_that("A6a: named across() on a 0-row table", {
  skip_if_no_gpu()
  f08_both(f08_df()[0, ], function(d) dplyr::mutate(d, s = dplyr::across(c(x, y), ~ .x + 1)))
})

test_that("A6a: filter() with a named across() dot raises dplyr's named-input error", {
  skip_if_no_gpu()
  expect_error(dplyr::filter(f08_df(), x = dplyr::across(x)), "named input")
  expect_error(dplyr::filter(tbl_gpu(f08_df()), x = dplyr::across(x)), "named input")
})

# =============================================================================
# A6b: if_any()/if_all() over an empty selection
# =============================================================================

test_that("A6b: filter() with if_any()/if_all() over an empty selection", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_any(c(), ~ TRUE)))
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_all(c(), ~ FALSE)))
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_any(dplyr::starts_with("zz"), ~ .x > 0)))
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_all(dplyr::starts_with("zz"), ~ .x > 0)))
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_any(c())))
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_all(c(), identity)))
})

test_that("A6b: grouped filter() where the only selected column is the group column", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_any(dplyr::starts_with("g"), ~ .x > 100))
  })
  f08_both(df, function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_all(dplyr::starts_with("g"), ~ .x > 100))
  })
})

test_that("A6b: mutate()/transmute() with an empty if_any()/if_all() give FALSE/TRUE columns", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) {
    dplyr::mutate(d, a = dplyr::if_any(c(), ~ TRUE), b = dplyr::if_all(c(), ~ FALSE))
  })
  f08_both(df, function(d) {
    dplyr::mutate(dplyr::group_by(d, g), a = dplyr::if_any(c()), b = dplyr::if_all(c()))
  })
  f08_both(df, function(d) dplyr::transmute(d, a = dplyr::if_all(c(), ~ FALSE)))
})

test_that("A6b: summarise() with an empty if_any()/if_all()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g), a = dplyr::if_any(c(), ~ TRUE),
                     b = dplyr::if_all(c(), ~ FALSE))
  })
  f08_both(df, function(d) dplyr::summarise(d, a = dplyr::if_any(c(), ~ TRUE)))
})

test_that("A6b: empty if_any()/if_all() on a 0-row table", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()[0, ]
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_all(c(), ~ TRUE)))
  f08_both(df, function(d) dplyr::mutate(d, a = dplyr::if_any(c(), ~ TRUE)))
})

test_that("A6b: an unnamed if_any()/if_all() dot is auto-named like dplyr", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) dplyr::mutate(d, dplyr::if_any(c(), ~ TRUE)))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::if_any(c(x, y), ~ .x > 4)))
  f08_both(df, function(d) dplyr::transmute(d, dplyr::if_all(c(x, y), ~ .x > 0)))
})

# =============================================================================
# A6c: zero-argument if_any()/if_all()
# =============================================================================

test_that("A6c: zero-argument if_any()/if_all() in filter() (ungrouped and grouped)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_lgl()
  quiet <- function(f) function(d) suppressWarnings(f(d))
  f08_both(dplyr::select(df, -g), quiet(function(d) dplyr::filter(d, dplyr::if_any())))
  f08_both(dplyr::select(df, -g), quiet(function(d) dplyr::filter(d, dplyr::if_all())))
  f08_both(df, quiet(function(d) dplyr::filter(dplyr::group_by(d, g), dplyr::if_any())))
  f08_both(df, quiet(function(d) dplyr::filter(dplyr::group_by(d, g), dplyr::if_all())))
})

test_that("A6c: zero-argument if_any()/if_all() in mutate()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- dplyr::select(f08_lgl(), -g)
  quiet <- function(f) function(d) suppressWarnings(f(d))
  f08_both(df, quiet(function(d) dplyr::mutate(d, a = dplyr::if_any(), b = dplyr::if_all())))
})

test_that("A6c: zero-argument if_any()/if_all() warn with dplyr's deprecation", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  df <- dplyr::select(f08_lgl(), -g)
  msg_any <- "Using `if_any\\(\\)` without supplying `.cols` was deprecated in dplyr 1.1.0"
  msg_all <- "Using `if_all\\(\\)` without supplying `.cols` was deprecated in dplyr 1.1.0"

  expect_warning(dplyr::filter(df, dplyr::if_any()), msg_any)
  expect_warning(dplyr::filter(tbl_gpu(df), dplyr::if_any()), msg_any,
                 class = "lifecycle_warning_deprecated")
  expect_warning(dplyr::filter(tbl_gpu(df, lazy = TRUE), dplyr::if_all()), msg_all,
                 class = "lifecycle_warning_deprecated")
  expect_warning(dplyr::mutate(tbl_gpu(df), a = dplyr::if_all()), msg_all)
})

test_that("A6c: identity if_any()/if_all() over non-logical columns is an error", {
  skip_if_no_gpu()
  df <- f08_df()
  quiet <- function(e) suppressWarnings(e)
  expect_error(quiet(dplyr::filter(df, dplyr::if_any())), "must be a logical vector")
  expect_error(quiet(dplyr::filter(tbl_gpu(df), dplyr::if_any())), "must be a logical vector")
  expect_error(dplyr::filter(df, dplyr::if_all(c(x, y))), "must be a logical vector")
  expect_error(dplyr::filter(tbl_gpu(df), dplyr::if_all(c(x, y))), "must be a logical vector")
  expect_error(dplyr::mutate(df, a = dplyr::if_any(c(x, y))), "must be a logical vector")
  expect_error(dplyr::mutate(tbl_gpu(df), a = dplyr::if_any(c(x, y))), "must be a logical vector")
})

test_that("A6c: if_any() with .fns but no .cols uses everything()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  quiet <- function(f) function(d) suppressWarnings(f(d))
  f08_both(df, quiet(function(d) dplyr::filter(d, dplyr::if_any(.fns = ~ .x > 4))))
  f08_both(df, quiet(function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_all(.fns = ~ !is.na(.x)))
  }))
})

# =============================================================================
# A6d: across() without .cols
# =============================================================================

test_that("A6d: across() without .cols uses everything() (mutate, ungrouped and grouped)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  quiet <- function(f) function(d) suppressWarnings(f(d))
  f08_both(df, quiet(function(d) dplyr::mutate(d, dplyr::across(.fns = ~ .x * 2))))
  f08_both(df, quiet(function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(.fns = ~ .x * 2))
  }))
  f08_both(df, quiet(function(d) dplyr::mutate(d, dplyr::across())))
})

test_that("A6d: across() without .cols in summarise() and transmute()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  quiet <- function(f) function(d) suppressWarnings(f(d))
  f08_both(df, quiet(function(d) {
    dplyr::summarise(dplyr::group_by(d, g), dplyr::across(.fns = ~ sum(.x, na.rm = TRUE)))
  }))
  f08_both(df, quiet(function(d) {
    dplyr::transmute(d, dplyr::across(.fns = ~ .x + 1, .names = "{.col}_p1"))
  }))
})

test_that("A6d: across() without .cols warns with dplyr's deprecation", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  df <- f08_df()
  msg <- "Using `across\\(\\)` without supplying `.cols` was deprecated in dplyr 1.1.0"

  expect_warning(dplyr::mutate(df, dplyr::across(.fns = ~ .x * 2)), msg)
  expect_warning(dplyr::mutate(tbl_gpu(df), dplyr::across(.fns = ~ .x * 2)), msg)
  expect_warning(dplyr::mutate(tbl_gpu(df, lazy = TRUE), dplyr::across()), msg)
  expect_warning(
    dplyr::summarise(dplyr::group_by(tbl_gpu(df), g), dplyr::across(.fns = ~ sum(.x))),
    msg
  )
})

test_that("A6d: across() with .cols supplied does not warn", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  expect_no_warning(dplyr::mutate(tbl_gpu(f08_df()), dplyr::across(c(x, y), ~ .x + 1)))
  expect_no_warning(dplyr::filter(tbl_gpu(f08_lgl()), dplyr::if_any(c(p, q))))
})

# =============================================================================
# A6e: renaming inside .cols
# =============================================================================

test_that("A6e: mutate() with all_of() renames in .cols", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  cols <- c(a = "x", b = "y")
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(dplyr::all_of(cols), identity)))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(dplyr::all_of(cols), ~ .x + 1)))
  f08_both(df, function(d) {
    dplyr::mutate(d, dplyr::across(dplyr::all_of(cols), identity, .names = "{.col}_name"))
  })
})

test_that("A6e: c(new = old) renames, partial renames and renaming onto an existing column", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(c(a = x, y), ~ .x * 2)))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(c(x = y), ~ .x + 1L)))
})

test_that("A6e: renamed .cols with a list of functions and {.fn}/{.col} templates", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) {
    dplyr::summarise(d, dplyr::across(c(a = x, b = y), list(mean = mean, sum = sum)))
  })
  f08_both(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
                     dplyr::across(c(a = x, b = y), list(m = ~ max(.x), n = ~ min(.x)),
                                   .names = "{.fn}.{.col}"))
  })
})

test_that("A6e: renamed .cols without .fns (identity) in summarise()", {
  skip_if_no_gpu()
  # A bare column in summarise() is not an aggregation, so this may use the
  # CPU fallback; only the renamed output is under test here.
  df <- tibble::tibble(x = 1, y = 2)
  f08_both(df, function(d) dplyr::summarise(d, dplyr::across(c(a = x, b = y))))
  f08_both(df, function(d) dplyr::summarise(d, dplyr::across(dplyr::all_of(c(a = "x", b = "y")))))
})

test_that("A6e: renamed .cols in a grouped mutate()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  f08_both(f08_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), dplyr::across(c(a = x, b = y), ~ .x - mean(.x)))
  })
})

test_that("A6e: cur_column() still reports the original column name", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  f08_both(f08_df(), function(d) dplyr::mutate(d, dplyr::across(c(a = x, b = y), ~ dplyr::cur_column())))
})

test_that("A6e: renamed .cols in if_any()/if_all() and transmute()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f08_df()
  f08_both(df, function(d) dplyr::filter(d, dplyr::if_any(c(a = x, b = y), ~ .x > 4)))
  f08_both(df, function(d) dplyr::mutate(d, k = dplyr::if_all(c(a = x, b = y), ~ .x > 2)))
  f08_both(df, function(d) dplyr::transmute(d, dplyr::across(c(a = x), ~ .x + 1)))
})

test_that("A6e: duplicate new names and renamed group columns are errors", {
  skip_if_no_gpu()
  df <- f08_df()
  expect_error(dplyr::mutate(df, dplyr::across(c(a = x, a = y), ~ .x)), "unique")
  expect_error(dplyr::mutate(tbl_gpu(df), dplyr::across(c(a = x, a = y), ~ .x)), "unique")
  gdf <- dplyr::group_by(df, g)
  expect_error(dplyr::mutate(gdf, dplyr::across(c(a = g), ~ .x)), "doesn't exist")
  expect_error(
    dplyr::mutate(dplyr::group_by(tbl_gpu(df), g), dplyr::across(c(a = g), ~ .x)),
    "doesn't exist"
  )
})

# =============================================================================
# Added after round 1 review (reviews/round-1.md, N11)
# =============================================================================

test_that("A6b: the empty-selection constant doesn't count as a used column for .keep", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, NA, 3), y = c(TRUE, FALSE, NA))
  f08_both(df, function(d) dplyr::mutate(d, a = dplyr::if_all(c(), ~ TRUE), .keep = "unused"))
  f08_both(df, function(d) dplyr::mutate(d, a = dplyr::if_any(c(), ~ TRUE), .keep = "used"))
  f08_both(df, function(d) {
    dplyr::mutate(dplyr::group_by(d, g), a = dplyr::if_any(c(), ~ TRUE), .keep = "used")
  })
  f08_both(df, function(d) dplyr::mutate(d, a = dplyr::if_all(c()), .keep = "none"))
})

test_that("A6c: explicit identity in if_any()/if_all() over non-logical columns is an error", {
  skip_if_no_gpu()
  df <- data.frame(w = c(TRUE, FALSE, NA), y = 1:3, z = 4:6)
  for (d in list(df, tbl_gpu(df), tbl_gpu(df, lazy = TRUE))) {
    expect_error(dplyr::filter(d, dplyr::if_any(y, identity)), "must be a logical vector")
    expect_error(dplyr::filter(d, dplyr::if_all(c(w, y), identity)), "must be a logical vector")
    expect_error(dplyr::mutate(d, a = dplyr::if_any(y, identity)), "must be a logical vector")
    expect_error(dplyr::mutate(d, a = dplyr::if_all(c(y, z), identity)), "must be a logical vector")
    expect_error(dplyr::mutate(d, a = dplyr::if_any(c(y, z), ~ .x)), "must be a logical vector")
  }
})

test_that("A6d/A6e: factor columns through identity, rename and zero-argument across()", {
  skip_if_no_gpu()
  fd <- tibble::tibble(f = factor(c("a", "b", "a")), x = 1:3)
  quiet <- function(f) function(d) suppressWarnings(f(d))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(f, identity)))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(dplyr::everything(), identity)))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(f, identity, .names = "{.col}_copy")))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(c(h = f), identity)))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(c(h = f))))
  f08_both(fd, quiet(function(d) dplyr::mutate(d, dplyr::across())))
})

test_that("A6c/A6d: an explicit `.cols = NULL` selects nothing (not the deprecated default)", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  lg <- tibble::tibble(p = c(TRUE, FALSE), q = c(FALSE, FALSE))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(NULL, ~ .x * 2)))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(.cols = NULL, ~ .x * 2)))
  f08_both(lg, function(d) dplyr::filter(d, dplyr::if_any(NULL, ~ .x)))
  f08_both(lg, function(d) dplyr::filter(d, dplyr::if_all(.cols = NULL)))
  expect_no_warning(dplyr::mutate(tbl_gpu(df), dplyr::across(NULL, ~ .x * 2)))
})

test_that("A6e: renaming onto a column that a later expansion reads matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(c(x = y, y = x), identity)))
  f08_both(df, function(d) dplyr::mutate(d, dplyr::across(c(y = x, z = y), ~ .x + 1)))
  f08_both(df, function(d) dplyr::transmute(d, dplyr::across(c(y = x, x = y), ~ .x * 2)))
})

# =============================================================================
# Added after round 2 review (reviews/round-2.md, B6, N7)
# =============================================================================

test_that("factor-valued if_else()/case_when()/coalesce() keep dplyr's factor results", {
  skip_if_no_gpu()
  fd <- tibble::tibble(f = factor(c("a", NA, "b")), f2 = factor(c("x", "y", "z")), x = 1:3)
  f08_both(fd, function(d) dplyr::mutate(d, h = dplyr::if_else(x > 1L, f, f)))
  f08_both(fd, function(d) dplyr::mutate(d, f = dplyr::coalesce(f, f2)))
  f08_both(fd, function(d) dplyr::mutate(d, h = dplyr::case_when(x > 1L ~ f2, .default = f)))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(f, ~ dplyr::coalesce(.x, f2))))
})

test_that("bare factor copies through across() stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  fd <- tibble::tibble(f = factor(c("a", "b", "a")), x = 1:3)
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(f, identity, .names = "{.col}_copy")))
  f08_both(fd, function(d) dplyr::mutate(d, dplyr::across(c(h = f))))
})

# =============================================================================
# Added after round 3 review (reviews/round-3.md, B7-B9)
# =============================================================================

test_that("cur_group_id()'s helper column never leaks through the CPU fallback", {
  skip_if_no_gpu()
  fd <- tibble::tibble(g = c(1, 1, 2), f = factor(c("a", NA, "b")),
                       f2 = factor(c("x", "y", "z")), x = 1:3)
  f08_both(fd, function(d) dplyr::mutate(dplyr::group_by(d, g), id = dplyr::cur_group_id(), h = dplyr::lag(f)))
  f08_both(fd, function(d) dplyr::mutate(d, id = dplyr::cur_group_id(), h = dplyr::first(f), .by = g))
  f08_both(fd, function(d) dplyr::mutate(d, id = dplyr::cur_group_id(), h = dplyr::coalesce(f, f2)))
  f08_both(fd, function(d) {
    dplyr::mutate(d, id = dplyr::cur_group_id(), h = dplyr::coalesce(f, f2), .keep = "used")
  })
  f08_both(fd, function(d) dplyr::mutate(d, id = dplyr::cur_group_id(), y = identity(x)))
  f08_both(fd, function(d) dplyr::transmute(d, id = dplyr::cur_group_id(), h = dplyr::coalesce(f, f2)))
})

test_that("distinct() with a renamed factor key keeps the factor", {
  skip_if_no_gpu()
  fd <- tibble::tibble(f = factor(c("b", "a", "b"), levels = c("b", "a", "z")), x = 1:3)
  f08_both(fd, function(d) dplyr::distinct(d, h = f), arrange_by = "h")
  f08_both(fd, function(d) dplyr::distinct(d, x, h = f))
  f08_both(fd, function(d) dplyr::distinct(d, h = f, .keep_all = TRUE), arrange_by = "h")
  f08_both(fd, function(d) dplyr::mutate(dplyr::distinct(d, x, h = f), k = h))
})

test_that("grouped filter() if_any()/if_all() never select cur_group_id()'s helper column", {
  skip_if_no_gpu()
  d2 <- tibble::tibble(g = c(1, 1, 2, 3), x = c(0, 1, 0, 0))
  quiet <- function(f) function(d) suppressWarnings(f(d))
  expect_same_as_dplyr(d2, quiet(function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_any(.fns = ~ .x > 2), dplyr::cur_group_id() >= 1)
  }))
  expect_same_as_dplyr(d2, function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_all(dplyr::everything(), ~ .x < 2),
                  dplyr::cur_group_id() <= 3)
  })
  expect_same_as_dplyr(d2, function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_any(dplyr::where(is.numeric), ~ .x > 2),
                  dplyr::cur_group_id() >= 1)
  })
  lg <- tibble::tibble(g = c(1, 1, 2), p = c(TRUE, FALSE, TRUE))
  expect_same_as_dplyr(lg, function(d) {
    dplyr::filter(dplyr::group_by(d, g), dplyr::if_all(dplyr::everything()), dplyr::cur_group_id() == 1)
  })
})
