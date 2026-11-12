# F76 acceptance tests: first()/last()/nth() inside summarise() keep the
# logical type of a computed logical argument.
#
# dplyr 1.2.1 (R/nth-value.R, nth()/first()/last()): the result is
# `vec_slice(x, n)` (or `default`, cast to `x`'s type with vec_cast()), so it
# always has the argument's type: a logical argument gives a logical result,
# also `NA` for an out-of-range position or an empty group.
#
# Before F76, summarise() routed a computed argument (anything that is not a
# bare column: `cumall(b)`, `cumany(b)`, `i > 2`, `!b`, `is.na(x)`, ...)
# through a temp column that declares a BOOL8 expression INT32 (the
# sum()-oriented promotion; only any()/all() opted out), so first()/last()/
# nth() of it returned an integer. mutate() was already right.
#
# Other reducers over the same logical arguments keep dplyr's types (sum()/
# min()/max()/n_distinct() integer, mean() double, any()/all() logical); the
# regression block at the end pins them.

f76_df <- data.frame(
  g = c(1, 1, 1, 2, 2, 2, 3),
  b = c(TRUE, NA, FALSE, TRUE, TRUE, FALSE, NA),
  i = c(3L, NA, 1L, 5L, 4L, 2L, 7L),
  x = c(3.5, 1, NA, 5, 4, 2, 0),
  f = factor(c("u", "v", "u", "v", "u", "v", "u"), levels = c("u", "v", "w")),
  stringsAsFactors = FALSE
)

f76_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

# Runs `pipeline` on the GPU (eager and lazy) with CPU fallbacks turned into
# errors, and checks the schema the plan reports agrees with what collect()
# returns: a lazy summarise must not claim INT32 for a logical column.
f76_on_gpu <- function(df, pipeline, logical_cols) {
  for (lazy in c(FALSE, TRUE)) {
    withr::with_options(list(cuplyr.fallback = "error"), {
      res <- pipeline(tbl_gpu(df, lazy = lazy))
      sch <- .subset2(res, "schema")
      types <- stats::setNames(sch$types, sch$names)
      expect_equal(unname(types[logical_cols]), rep("BOOL8", length(logical_cols)),
                   info = sprintf("schema types (lazy = %s)", lazy))
      out <- collect(res)
    })
    for (col in logical_cols) {
      expect_type(out[[col]], "logical")
    }
  }
}

# --- the reported bug: cumall()/cumany() under first()/last()/nth() ---------

test_that("ungrouped summarise first/last/nth of cumall/cumany is logical", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(dplyr::cumany(b)), n2 = dplyr::nth(dplyr::cumall(b), 2),
      nm1 = dplyr::nth(dplyr::cumany(b), -1), fa = dplyr::first(dplyr::cumall(b)), la = dplyr::last(dplyr::cumany(b))
    )
  })
})

test_that("grouped summarise first/last/nth of cumall/cumany is logical", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(
        l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(dplyr::cumany(b)), n2 = dplyr::nth(dplyr::cumall(b), 2),
        n2L = dplyr::nth(dplyr::cumany(b), 2L), nm2 = dplyr::nth(dplyr::cumall(b), -2), n9 = dplyr::nth(dplyr::cumany(b), 9)
      )
  })
})

test_that("summarise(.by=) first/last/nth of cumall/cumany is logical", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d, l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(dplyr::cumany(b)), n2 = dplyr::nth(dplyr::cumall(b), 2),
              n = dplyr::n(), .by = g)
  })
})

test_that("na_rm = TRUE keeps the logical type", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      l = dplyr::last(dplyr::cumall(b), na_rm = TRUE), f = dplyr::first(dplyr::cumany(b), na_rm = TRUE),
      n2 = dplyr::nth(dplyr::cumall(b), 2, na_rm = TRUE), li = dplyr::last(i > 2, na_rm = TRUE),
      .by = g
    )
  })
})

test_that("a logical expression over a window result keeps the logical type", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      a = dplyr::last(!dplyr::cumany(b)), o = dplyr::first(dplyr::cumall(b) | FALSE),
      c = dplyr::nth(dplyr::lag(i) > 2, 2), n = dplyr::last(is.na(dplyr::cumall(b))),
      .by = g
    )
  })
})

# --- the same cause without a window: any computed logical argument --------

test_that("first/last/nth of a computed logical argument is logical (ungrouped)", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      gt = dplyr::first(i > 2), nb = dplyr::last(!b), na = dplyr::nth(is.na(x), 2),
      an = dplyr::last(b & TRUE), co = dplyr::first(dplyr::coalesce(b, FALSE)),
      ie = dplyr::nth(dplyr::if_else(b, TRUE, FALSE), -1)
    )
  })
})

test_that("first/last/nth of a computed logical argument is logical (grouped)", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(gt = dplyr::first(i > 2), nb = dplyr::last(!b), na = dplyr::nth(is.na(x), 2),
                out = dplyr::nth(i > 2, 5))
  })
})

# --- empty inputs and empty groups -------------------------------------------

test_that("zero-row input gives a logical NA (ungrouped) / zero logical rows (grouped)", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    d |>
      dplyr::filter(x > 100) |>
      dplyr::summarise(l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(i > 2), n2 = dplyr::nth(dplyr::cumany(b), 2))
  })
  f76_both(f76_df, function(d) {
    d |>
      dplyr::filter(x > 100) |>
      dplyr::group_by(g) |>
      dplyr::summarise(l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(i > 2), n2 = dplyr::nth(dplyr::cumany(b), 2))
  })
})

test_that("empty groups of a .drop = FALSE factor key give logical NA", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    d |>
      dplyr::group_by(f, .drop = FALSE) |>
      dplyr::summarise(l = dplyr::last(dplyr::cumall(b)), f2 = dplyr::first(i > 2), n2 = dplyr::nth(dplyr::cumany(b), 2))
  })
})

# --- the result is computed on the GPU and the plan's schema is right -------

test_that("the logical result stays on the GPU and the schema reports BOOL8", {
  skip_if_no_gpu()
  f76_on_gpu(f76_df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(dplyr::cumany(b)), n2 = dplyr::nth(dplyr::cumall(b), 2),
                gt = dplyr::first(i > 2), nb = dplyr::last(!b, na_rm = TRUE))
  }, c("l", "f", "n2", "gt", "nb"))
  f76_on_gpu(f76_df, function(d) {
    dplyr::summarise(d, l = dplyr::last(dplyr::cumall(b)), gt = dplyr::nth(i > 2, -1))
  }, c("l", "gt"))
})

test_that("downstream verbs see a logical column", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    d |>
      dplyr::summarise(l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(i > 2), .by = g) |>
      dplyr::filter(l | f) |>
      dplyr::mutate(nl = !l)
  })
})

# --- mutate() (already right; pinned so the fix doesn't break it) -----------

test_that("mutate first/last/nth of cumall/cumany and logical expressions is logical", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::mutate(d, l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(dplyr::cumany(b)), n2 = dplyr::nth(dplyr::cumall(b), 2),
           gt = dplyr::first(i > 2), .by = g)
  })
  f76_both(f76_df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(l = dplyr::last(dplyr::cumall(b)), nb = dplyr::nth(!b, -1))
  })
  f76_both(f76_df, function(d) dplyr::mutate(d, l = dplyr::last(dplyr::cumall(b)), f = dplyr::first(dplyr::cumany(b))))
})

# --- other reducers over the same arguments keep dplyr's types --------------

test_that("other reducers over logical arguments keep dplyr's types", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      s = sum(dplyr::cumall(b)), mx = max(dplyr::cumany(b)), mn = min(i > 2),
      me = mean(dplyr::cumany(b)), nd = dplyr::n_distinct(dplyr::cumall(b)), an = any(dplyr::cumall(b)),
      al = all(i > 2), .by = g
    )
  })
  # sd()/var() of a computed logical are doubles, as in dplyr (review round
  # 1, N3). A results-based test can't tell whether the temp column was
  # BOOL8 or INT32 here: these reducers give the same answer for both.
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      sd1 = sd(i > 2), va = var(dplyr::cumall(b)), sd2 = sd(!b, na.rm = TRUE),
      .by = g
    )
  })
})

test_that("one argument shared by sum() and first() keeps each reducer's type", {
  skip_if_no_gpu()
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      s = sum(i > 2), f = dplyr::first(i > 2), l = dplyr::last(i > 2), m = max(i > 2),
      sc = sum(dplyr::cumall(b)), fc = dplyr::first(dplyr::cumall(b)), lc = dplyr::last(dplyr::cumall(b)),
      .by = g
    )
  })
})

test_that("a window argument shared by first() and other reducers, across() and fixed keys", {
  skip_if_no_gpu()
  # review round 1, probes 1 and 3: sharing of window stages / temp columns
  # between reducers that do and don't keep the logical type, in both orders.
  # All of it runs on the GPU: a CPU fallback fails the test.
  withr::local_options(cuplyr.fallback = "error")
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      dplyr::across(c(i, x), ~ dplyr::first(.x > 2)),
      fc = dplyr::first(dplyr::cumall(b)), sc = sum(dplyr::cumall(b)),
      lc = dplyr::last(dplyr::cumall(b)), fl = dplyr::first(dplyr::lag(b)),
      sl = sum(dplyr::lag(b)), nl = dplyr::nth(dplyr::lag(b), 2),
      .by = g
    )
  })
  f76_both(f76_df, function(d) {
    dplyr::summarise(d,
      sl = sum(dplyr::lag(b)), fl = dplyr::first(dplyr::lag(b)),
      sc = sum(dplyr::cumany(b)), lc = dplyr::last(dplyr::cumany(b)),
      .by = g
    )
  })
  f76_both(f76_df, function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::filter(i > 3, .preserve = TRUE) |>
      dplyr::summarise(s = sum(i > 2), f2 = dplyr::first(i > 2),
                       l = dplyr::last(dplyr::cumall(b)), n2 = dplyr::nth(!b, 2))
  })
})
