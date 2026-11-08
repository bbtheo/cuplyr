# F111 acceptance, items 3 and 5: how often a user's expression runs.
#
# Item 5 matches dplyr exactly: an active binding read through a plain symbol
# in an if_any()/if_all() lambda runs once per column.
#
# Item 3 is covered as far as it can be made true. A column-free call that
# errors is evaluated by cuplyr once at most, then once by dplyr: dplyr has to
# raise it (only it can wrap the error), and an error outside the mask is no
# proof that dplyr fails (a user function may call pick() or n()), so cuplyr
# can't report it itself. That makes 2 the bound for the error path (3 on dev
# for mutate()/transmute()/group_by()). The brief's value path (a call folded
# into a column-reading dot that a type rule then sends to the CPU) was dropped
# with numbers: see the Design section of the F111 brief.

# Tests see the package namespace, which doesn't import these; bind the dplyr
# functions these pipelines call bare, as library(dplyr) would for a user.
for (.f111c in c("if_any", "if_all", "if_else", "n", "ntile", "pick")) {
  if (!exists(.f111c, mode = "function")) assign(.f111c, getExportedValue("dplyr", .f111c))
}
rm(.f111c)

f111c_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3),
    y2 = c(3, 2, 1),
    g = c("a", "a", "b"),
    flag = c(TRUE, FALSE, TRUE)
  )
}

# Runs `verb(d, counter)` on dplyr and on the GPU, eager and lazy, with a fresh
# counter each time. Returns list(dplyr = n, eager = n, lazy = n, errors = classes).
f111c_counts <- function(verb, df = f111c_df()) {
  out <- list()
  classes <- list()
  for (m in c("dplyr", "eager", "lazy")) {
    k <- 0L
    fail <- function(...) {
      k <<- k + 1L
      stop("nope")
    }
    t <- switch(m,
      dplyr = df,
      eager = tbl_gpu(df, lazy = FALSE),
      lazy = tbl_gpu(df, lazy = TRUE)
    )
    res <- tryCatch(
      {
        collect(verb(t, fail))
        NULL
      },
      error = function(e) e
    )
    out[[m]] <- k
    classes[[m]] <- if (is.null(res)) "no error" else class(res)
  }
  list(counts = out, classes = classes)
}

f111c_expect_error_counts <- function(verb, bound = 2L, info = NULL) {
  r <- f111c_counts(verb)
  expect_identical(r$counts$dplyr, 1L, info = paste("dplyr count:", info))
  expect_lte(r$counts$eager, bound, label = paste("eager count:", info))
  expect_lte(r$counts$lazy, bound, label = paste("lazy count:", info))
  expect_identical(r$classes$eager, r$classes$dplyr, info = paste("eager class:", info))
  expect_identical(r$classes$lazy, r$classes$dplyr, info = paste("lazy class:", info))
}

test_that("F111 item 3: a failing column-free dot runs at most twice, and the error is dplyr's", {
  skip_if_no_gpu()
  cfg2 <- list(a = 1)
  cases <- list(
    mutate = function(t, fail) mutate(t, y = fail()),
    mutate_later = function(t, fail) mutate(t, y = x + 1, z = fail()),
    mutate_first = function(t, fail) mutate(t, z = fail(), y = x + 1),
    transmute = function(t, fail) transmute(t, y = fail()),
    group_by = function(t, fail) group_by(t, y = fail()),
    filter = function(t, fail) filter(t, fail()),
    filter_read = function(t, fail) filter(t, x > fail()),
    mutate_read = function(t, fail) mutate(t, y = x + fail()),
    slice = function(t, fail) slice(t, fail()),
    dollar_arg = function(t, fail) mutate(t, y = fail(cfg2$a)),
    dollar_transmute = function(t, fail) transmute(t, y = fail(cfg2$a)),
    dollar_group_by = function(t, fail) group_by(t, y = fail(cfg2$a)),
    ns_arg = function(t, fail) mutate(t, y = fail(base::pi))
  )
  for (nm in names(cases)) f111c_expect_error_counts(cases[[nm]], info = nm)
})

test_that("F111 item 3: a column-free dot that fails only outside the mask still works", {
  skip_if_no_gpu()
  df <- f111c_df()
  needs_mask <- function() n()
  pick_it <- function() nrow(pick(x))
  for (lazy in c(FALSE, TRUE)) {
    expect_equal(
      collect(mutate(tbl_gpu(df, lazy = lazy), z = needs_mask()))$z,
      mutate(df, z = needs_mask())$z
    )
    expect_equal(
      collect(transmute(tbl_gpu(df, lazy = lazy), z = pick_it()))$z,
      transmute(df, z = pick_it())$z
    )
    expect_equal(
      collect(filter(tbl_gpu(df, lazy = lazy), needs_mask() > 0))$x,
      filter(df, needs_mask() > 0)$x
    )
  }
})

test_that("F111 item 3: a window call that needs the mask is still parsed natively", {
  skip_if_no_gpu()
  df <- f111c_df()
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(t) mutate(t, z = ntile(n = 2)),
    function(t) mutate(t, z = ntile(n = 2L) + 1L),
    function(t) mutate(t, z = if_else(ntile(n = 2) > 1L, 1L, 0L)),
    function(t) mutate(group_by(t, g), z = ntile(n = 2)),
    function(t) group_by(t, z = ntile(n = 2))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F111 item 3: a grouped failing dot still runs once, as in dplyr", {
  skip_if_no_gpu()
  grouped <- list(
    mutate = function(t, fail) mutate(group_by(t, g), y = fail()),
    filter = function(t, fail) filter(group_by(t, g), fail())
  )
  for (nm in names(grouped)) {
    r <- f111c_counts(grouped[[nm]])
    expect_identical(r$counts$eager, r$counts$dplyr, info = nm)
    expect_identical(r$counts$lazy, r$counts$dplyr, info = nm)
  }
})

# ---------------------------------------------------------------------------
# Item 5: active bindings
# ---------------------------------------------------------------------------

# Builds the verb in an environment holding an active binding `ab` that counts
# its reads; returns list(fn = verb, reads = function() n, reset = function()).
f111c_active <- function(make_verb) {
  e <- new.env(parent = parent.frame())
  n <- 0L
  makeActiveBinding("ab", function() {
    n <<- n + 1L
    2
  }, e)
  makeActiveBinding("abl", function() {
    n <<- n + 1L
    TRUE
  }, e)
  environment(make_verb) <- e
  list(fn = make_verb(), reads = function() n, reset = function() n <<- 0L)
}

f111c_expect_active <- function(make_verb, info = NULL) {
  df <- f111c_df()
  a <- f111c_active(make_verb)
  counts <- integer()
  pipeline <- function(d) {
    a$reset()
    out <- collect(a$fn(d))
    counts <<- c(counts, a$reads())
    out
  }
  expect_same_as_dplyr(df, pipeline)
  expect_length(counts, 2L)
  expect_identical(counts[[2]], counts[[1]],
                   info = sprintf("%s (eager): dplyr %d, GPU %d", info, counts[[1]], counts[[2]]))
  counts <- integer()
  expect_same_as_dplyr_lazy(df, pipeline)
  expect_length(counts, 2L)
  expect_identical(counts[[2]], counts[[1]],
                   info = sprintf("%s (lazy): dplyr %d, GPU %d", info, counts[[1]], counts[[2]]))
}

test_that("F111 item 5: an active binding in an if_any()/if_all() lambda is read as in dplyr", {
  skip_if_no_gpu()
  makers <- list(
    if_any_one = function() function(t) filter(t, if_any(x, ~ .x > ab)),
    if_any_two = function() function(t) filter(t, if_any(c(x, y2), ~ .x > ab)),
    if_all_two = function() function(t) filter(t, if_all(c(x, y2), ~ .x > ab)),
    fn_lambda = function() function(t) filter(t, if_any(c(x, y2), function(v) v > ab)),
    in_mutate = function() function(t) mutate(t, z = if_any(c(x, y2), ~ .x > ab)),
    bare_body = function() function(t) filter(t, if_any(c(x, y2), ~ abl)),
    bare_fn_body = function() function(t) filter(t, if_all(c(x, y2), function(v) abl))
  )
  for (nm in names(makers)) f111c_expect_active(makers[[nm]], info = nm)
})

test_that("F111 item 5: a plain variable in the lambda stays on the GPU", {
  skip_if_no_gpu()
  df <- f111c_df()
  v <- 2
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, function(t) filter(t, if_any(c(x, y2), ~ .x > v)))
  expect_same_as_dplyr_lazy(df, function(t) filter(t, if_all(c(x, y2), ~ .x > v)))
  expect_same_as_dplyr(df, function(t) mutate(t, z = if_any(c(x, y2), ~ .x > v)))
})

test_that("F111 item 5: a binding that is active but not read in the piece changes nothing", {
  skip_if_no_gpu()
  df <- f111c_df()
  e <- new.env(parent = globalenv())
  n <- 0L
  makeActiveBinding("unused_ab", function() {
    n <<- n + 1L
    1
  }, e)
  v <- 2
  fn <- function(t) filter(t, if_any(c(x, y2), ~ .x > v))
  environment(fn) <- list2env(list(v = 2), parent = e)
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, fn)
  expect_identical(n, 0L)
})
