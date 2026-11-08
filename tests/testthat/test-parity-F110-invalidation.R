# F110 acceptance: whatever the per-call checks remember between calls, a
# change between two calls is seen by the second one.
#
# F110 makes the checks every verb runs (F77's shadow screen, F75's purity
# walk, F70's checked-integer flag, F93's option reads) cheaper. Anything it
# memoises across calls must notice what makes the old answer wrong: a
# function defined (or removed) after a first call in the quosure's env, an
# enclosing env, the global env or an env attached to the search path; the
# same expression over a schema with other types or other columns; an option
# set between two calls. Every scenario first runs the call natively (so a
# memo is warm), then changes the world and runs the SAME call again, eager
# and lazy, against dplyr. Quosures are built once with a fixed env so a memo
# keyed on (expression, env) would see the same key twice.

f110i_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L),
    x = c(-1.5, 2, 3, -4, 0.25, 6),
    i = c(3L, -2L, 5L, 1L, NA, 7L)
  )
}

# eager and lazy against dplyr; `native`: every CPU fallback is an error
f110i_both <- function(df, pipeline, native = FALSE) {
  if (native) withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

# The pipelines over one quosure `q` whose head is `abs` (or `>` for `cmp`).
f110i_pipes <- function(q, cmp) {
  list(
    mutate = function(d) dplyr::mutate(d, y = !!q),
    filter = function(d) dplyr::filter(d, !!q > 1),
    summarise = function(d) dplyr::summarise(dplyr::group_by(d, g), m = sum(!!q)),
    gt = function(d) dplyr::filter(d, !!cmp)
  )
}

# native -> shadowed in `where` -> removed again, for every pipeline
f110i_cycle <- function(df, q, cmp, where) {
  pipes <- f110i_pipes(q, cmp)
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]], native = TRUE)
  assign("abs", function(x) -x * 10, envir = where)
  assign(">", function(e1, e2) base::`<`(e1, e2), envir = where)
  for (nm in names(pipes)) {
    f110i_both(df, pipes[[nm]])
    # and it is the shadow that sends the call to dplyr
    withr::with_options(list(cuplyr.fallback = "error"), {
      expect_error(collect(pipes[[nm]](tbl_gpu(df, lazy = FALSE))), if (nm == "gt") ">" else "abs",
                   label = nm)
    })
  }
  rm(list = c("abs", ">"), envir = where)
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]], native = TRUE)
}

test_that("F110: a shadow defined after a call in the quosure's own env is seen", {
  skip_if_no_gpu()
  e <- new.env(parent = environment())
  q <- rlang::new_quosure(quote(abs(x)), e)
  cmp <- rlang::new_quosure(quote(x > 1), e)
  f110i_cycle(f110i_df(), q, cmp, e)
})

test_that("F110: a shadow defined after a call in an enclosing env is seen", {
  skip_if_no_gpu()
  outer <- new.env(parent = environment())
  e <- new.env(parent = outer)
  q <- rlang::new_quosure(quote(abs(x)), e)
  cmp <- rlang::new_quosure(quote(x > 1), e)
  f110i_cycle(f110i_df(), q, cmp, outer)
})

test_that("F110: a global shadow defined after a call is seen", {
  skip_if_no_gpu()
  ge <- globalenv()
  withr::defer(suppressWarnings(rm(list = c("abs", ">"), envir = ge)))
  q <- rlang::new_quosure(quote(abs(x)), ge)
  cmp <- rlang::new_quosure(quote(x > 1), ge)
  f110i_cycle(f110i_df(), q, cmp, ge)
})

test_that("F110: a shadow attached to the search path after a call is seen", {
  skip_if_no_gpu()
  df <- f110i_df()
  q <- rlang::new_quosure(quote(abs(x)), globalenv())
  cmp <- rlang::new_quosure(quote(x > 1), globalenv())
  pipes <- f110i_pipes(q, cmp)
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]], native = TRUE)
  shadow <- list(abs = function(x) -x * 10, `>` = function(e1, e2) base::`<`(e1, e2))
  attach_fn <- get("attach", envir = baseenv())
  attach_fn(shadow, name = "f110_shadow", warn.conflicts = FALSE)
  withr::defer(if ("f110_shadow" %in% search()) detach("f110_shadow", character.only = TRUE))
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]])
  detach("f110_shadow", character.only = TRUE)
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]], native = TRUE)
})

test_that("F110: a pure name redefined after a call is evaluated as often as dplyr does", {
  skip_if_no_gpu()
  df <- f110i_df()
  e <- new.env(parent = environment())
  counter <- new.env()
  counter$n <- 0L
  qs <- rlang::new_quosure(quote(mean(x + round(1.5))), e)
  qm <- rlang::new_quosure(quote(x + round(1.5)), e)
  qf <- rlang::new_quosure(quote(x > round(1.5)), e)
  pipes <- list(
    summarise = function(d) dplyr::summarise(dplyr::group_by(d, g), m = !!qs),
    mutate = function(d) dplyr::mutate(dplyr::group_by(d, g), y = !!qm),
    filter = function(d) dplyr::filter(dplyr::group_by(d, g), !!qf)
  )
  # base::round() is pure: native, folded once
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]], native = TRUE)
  assign("round", function(x, digits = 0) {
    counter$n <- counter$n + 1L
    100
  }, envir = e)
  for (nm in names(pipes)) {
    counter$n <- 0L
    ref <- pipes[[nm]](df)
    n_dplyr <- counter$n
    for (lazy in c(FALSE, TRUE)) {
      counter$n <- 0L
      out <- collect(pipes[[nm]](tbl_gpu(df, lazy = lazy)))
      expect_equal(counter$n, n_dplyr, label = sprintf("%s (lazy = %s) calls of round()", nm, lazy))
      expect_equal(tibble::as_tibble(dplyr::ungroup(out)), tibble::as_tibble(dplyr::ungroup(ref)))
    }
  }
  rm("round", envir = e)
  for (nm in names(pipes)) f110i_both(df, pipes[[nm]], native = TRUE)
})

test_that("F110: the same expression over other types or other columns gets its own overflow check", {
  skip_if_no_gpu()
  e <- new.env(parent = environment())
  q <- rlang::new_quosure(quote(a + b), e)
  pm <- function(d) dplyr::mutate(d, y = !!q)
  pf <- function(d) dplyr::filter(d, !!q > 0)
  big <- .Machine$integer.max
  d_dbl <- tibble::tibble(a = c(2e9, 1), b = c(2e9, 1))
  d_int <- tibble::tibble(a = c(big, 1L), b = c(1L, 1L))
  d_b <- tibble::tibble(b = c(big, 1L))
  overflow <- "NAs produced by integer overflow"
  gpu <- function(p, d, lazy) collect(p(tbl_gpu(d, lazy = lazy)))
  for (p in list(pm, pf)) {
    for (lazy in c(FALSE, TRUE)) {
      # doubles first: nothing to check, no warning
      expect_no_warning(gpu(p, d_dbl, lazy))
      # the same quosure over integers overflows, as in dplyr
      expect_warning(p(d_int), overflow)
      expect_warning(gpu(p, d_int, lazy), overflow)
      # `a` is now an integer variable, not a column
      assign("a", 1L, envir = e)
      expect_warning(p(d_b), overflow)
      expect_warning(gpu(p, d_b, lazy), overflow)
      # and a double variable again checks nothing
      assign("a", 1, envir = e)
      expect_no_warning(p(d_b))
      expect_no_warning(gpu(p, d_b, lazy))
      rm("a", envir = e)
      expect_no_warning(gpu(p, d_dbl, lazy))
    }
  }
  expect_equal(suppressWarnings(gpu(pm, d_int, FALSE)), suppressWarnings(pm(d_int)))
})

test_that("F110: an option set between two calls applies to the second", {
  skip_if_no_gpu()
  df <- f110i_df()
  f110i_unknown <- function(x) x + 1
  p <- function(d) dplyr::mutate(d, y = f110i_unknown(x))
  for (lazy in c(FALSE, TRUE)) {
    t <- tbl_gpu(df, lazy = lazy)
    withr::local_options(cuplyr.fallback = "silent")
    expect_no_warning(collect(p(t)))
    options(cuplyr.fallback = "warn")
    expect_warning(collect(p(t)), "f110i_unknown")
    options(cuplyr.fallback = "error")
    expect_error(collect(p(t)), "f110i_unknown")
    options(cuplyr.fallback = "silent")
    expect_no_warning(collect(p(t)))
    expect_equal(tibble::as_tibble(collect(p(t))), p(df))
  }
})
