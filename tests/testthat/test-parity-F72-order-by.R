# F72: order_by() / with_order() must never drop their order silently.
#
# dplyr 1.2.1 (R/order-by.R): `order_by(o, f(a1, ...))` is
# `with_order(o, f, a1, ...)`, which evaluates `a1` and `...` in ROW order,
# permutes only `a1` into `o` order (vec_order_radix(): stable, NA last, also
# under desc()), calls `f(a1[perm], ...)` and permutes the result back. So:
#   - a window `f` (cumsum, cummean, lag, ...) runs in `o` order;
#   - an elementwise `f` whose other arguments are scalars gives the plain
#     row-order value (`order_by(o, cumsum(x) * 2)` == `cumsum(x) * 2`);
#   - an elementwise `f` with another row-sized argument gives dplyr's odd
#     `f(a1, b[inverse perm])` value (`cumsum(x) + lag(x)`);
#   - windows inside `a1` itself run in row order (`lag(cumsum(x))`).
# Every test compares against real dplyr, eager and lazy. `f72_native()` also
# pins the shapes the GPU path must compute itself (no CPU fallback).

f72_df <- function() {
  tibble::tibble(
    g = c("a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"),
    # not in row order, with ties and NAs, in and across groups
    o = c(3L, 1L, NA, 2L, 1L, 5L, 2L, NA, 4L, 1L, 3L, 2L),
    x = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    y = c(2.5, NA, 1, 4, -3, 6, 0.5, 8, 2, -1, 7, 3),
    i = c(1L, 2L, 1L, 2L, 3L, 1L, 2L, 3L, 1L, 2L, 2L, 1L),
    b = c(TRUE, TRUE, NA, FALSE, TRUE, TRUE, FALSE, TRUE, NA, TRUE, TRUE, FALSE),
    dt = as.Date("2021-03-01") + c(5L, 3L, 9L, 1L, 2L, NA, 7L, 0L, 4L, 6L, 8L, 10L),
    s = c("b", "A", "a", "B", NA, "c", "C", "a", "b", "A", "c", "B")
  )
}

# A bigger random table: many ties and NAs in `o`, several groups.
f72_big <- function() {
  set.seed(72)
  n <- 300L
  tibble::tibble(
    g = sample(c("p", "q", "r", "s"), n, replace = TRUE),
    o = sample(c(1:20, NA), n, replace = TRUE),
    x = round(stats::rnorm(n) * 10, 1),
    i = sample(1:5, n, replace = TRUE),
    b = sample(c(TRUE, FALSE, NA), n, replace = TRUE, prob = c(0.6, 0.3, 0.1))
  )
}

f72_same <- function(df, pipeline, arrange_by = NULL) {
  expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
}

# Same as f72_same(), and neither run may fall back to the CPU.
f72_native <- function(df, pipeline, arrange_by = NULL) {
  withr::local_options(cuplyr.fallback = "error")
  f72_same(df, pipeline, arrange_by = arrange_by)
}

# dplyr raises an error for this pipeline; the GPU table must raise an error
# of the same class (eager and lazy), not return a value.
f72_same_error <- function(df, pipeline) {
  cpu <- tryCatch(pipeline(tibble::as_tibble(df)), error = identity)
  expect_s3_class(cpu, "error")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tryCatch({
      t <- tbl_gpu(df, lazy = lazy)
      collect(pipeline(t))
    }, error = identity)
    expect_s3_class(gpu, class(cpu)[[1]])
  }
}

# ---------------------------------------------------------------------------
# Desugared window functions: every window the desugar builds runs in `o` order
# ---------------------------------------------------------------------------

test_that("order_by(o, cummean(x)) runs cummean in o order (native)", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(x))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(y))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(o), dplyr::cummean(x))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(-o, dplyr::cummean(x))))
  # an integer argument (cummean accumulates in double)
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(i))))
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(o, dplyr::cummean(x)))
  })
  f72_native(df, function(d) {
    dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(y)), .by = g)
  })
})

test_that("with_order(o, cummean, x) runs cummean in o order (native)", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::cummean, x)))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::with_order(dplyr::desc(o), dplyr::cummean, y)))
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::with_order(o, dplyr::cummean, x))
  })
})

test_that("order_by(o, cumall()/cumany()) run in o order (native)", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cumall(b))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cumany(!b))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(o), dplyr::cumall(b))))
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::mutate(r1 = dplyr::order_by(o, dplyr::cumall(b)), r2 = dplyr::order_by(o, dplyr::cumany(b)))
  })
})

test_that("order_by(o, consecutive_id(x)) runs in o order (native)", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::consecutive_id(i))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::consecutive_id(b))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::consecutive_id(y))))
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(dplyr::desc(o), dplyr::consecutive_id(i)))
  })
})

test_that("desugared windows match dplyr on a larger table with many ties and NAs", {
  skip_if_no_gpu()
  df <- f72_big()
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      m = dplyr::order_by(o, dplyr::cummean(x)),
      a = dplyr::order_by(o, dplyr::cumall(b)),
      y = dplyr::order_by(dplyr::desc(o), dplyr::cumany(b)),
      k = dplyr::order_by(o, dplyr::consecutive_id(i)),
      s = dplyr::order_by(o, cumsum(x)),
      l = dplyr::order_by(dplyr::desc(o), dplyr::lag(x))
    )
  })
})

# ---------------------------------------------------------------------------
# The windows cuplyr maps natively, each directly wrapped (guards)
# ---------------------------------------------------------------------------

test_that("directly wrapped scan and shift windows stay native and ordered", {
  skip_if_no_gpu()
  df <- f72_df()
  for (ord in list(quote(o), quote(dplyr::desc(o)), quote(-o))) {
    p <- function(d) {
      rlang::inject(dplyr::mutate(d,
        cs = dplyr::order_by(!!ord, cumsum(x)),
        cp = dplyr::order_by(!!ord, cumprod(i)),
        cmin = dplyr::order_by(!!ord, cummin(y)),
        cmax = dplyr::order_by(!!ord, cummax(y)),
        lg = dplyr::order_by(!!ord, dplyr::lag(x)),
        ld = dplyr::order_by(!!ord, dplyr::lead(y, 2, default = 0))
      ))
    }
    f72_native(df, p)
    f72_native(df, function(d) dplyr::group_by(d, g) |> p())
  }
})

test_that("integer cumsum under order_by keeps its integer type", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(i))))
})

test_that("rank functions under order_by match dplyr", {
  skip_if_no_gpu()
  df <- f72_df()
  # ties in `i`: row_number()/ntile() break them by position in `o` order
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::row_number(i))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(o), dplyr::row_number(i))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::ntile(i, 3))))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(o, dplyr::ntile(i, 2)))
  })
  # value-only ranks: the order changes nothing, so these stay native
  f72_native(df, function(d) {
    dplyr::mutate(d,
      r1 = dplyr::order_by(o, dplyr::min_rank(i)),
      r2 = dplyr::order_by(o, dplyr::dense_rank(i)),
      r3 = dplyr::order_by(o, dplyr::percent_rank(y)),
      r4 = dplyr::order_by(o, dplyr::cume_dist(y))
    )
  })
})

test_that("roll_*() under order_by match dplyr", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_same(df, function(d) {
    dplyr::mutate(d,
      r1 = dplyr::order_by(o, roll_sum(x, 3)),
      r2 = dplyr::order_by(o, roll_mean(x, 2)),
      r3 = dplyr::order_by(o, roll_min(y, 3, na.rm = TRUE)),
      r4 = dplyr::order_by(dplyr::desc(o), roll_max(x, 2)),
      r5 = dplyr::order_by(o, roll_sd(x, 3))
    )
  })
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, roll_median(x, 3))))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(o, roll_mean(x, 2)))
  })
})

# ---------------------------------------------------------------------------
# Compound calls: dplyr's with_order() permutes the first argument only
# ---------------------------------------------------------------------------

test_that("order_by(o, cumsum(x) + lag(x)) gives dplyr's value", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) + dplyr::lag(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) - dplyr::lag(x, default = 0))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) + y)))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(o), dplyr::cummean(x) * y)))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(o, cumsum(x) + dplyr::lag(x)))
  })
})

test_that("an elementwise call with scalar arguments is the row-order value (native)", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) * 2)))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, abs(cumsum(y)))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, -cumsum(x))))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, x + 1)))
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, round(dplyr::cummean(x), 1))))
  f72_native(df, function(d) {
    dplyr::mutate(d, r = dplyr::order_by(o, dplyr::if_else(cumsum(x) > 10, 1, 0)))
  })
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(o, cumsum(x) / 2))
  })
})

test_that("windows inside the first argument run in row order", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(cumsum(x)))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(dplyr::lag(x)))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(dplyr::lead(y, default = 0)))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::consecutive_id(dplyr::lag(i)))))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::order_by(dplyr::desc(o), dplyr::cumall(dplyr::lag(b))))
  })
  # cumsum(x + 1): an elementwise first argument commutes with the order
  f72_native(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x + 1))))
})

test_that("nested order_by() and with_order() with extra arguments match dplyr", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::order_by(x, cumsum(y)))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::order_by(i, cumsum(x)) * 2)))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::lag, x, 2)))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::lead, x, default = -1)))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::cumall, b)))
})

# ---------------------------------------------------------------------------
# filter() and summarise() arguments
# ---------------------------------------------------------------------------

test_that("order_by() inside filter() keeps its order", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) dplyr::filter(d, dplyr::order_by(o, dplyr::cummean(x)) > 5))
  f72_native(df, function(d) dplyr::filter(d, dplyr::order_by(o, dplyr::cumany(!b))))
  f72_same(df, function(d) dplyr::filter(d, dplyr::order_by(o, cumsum(x) + dplyr::lag(x)) > 10))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::filter(dplyr::order_by(dplyr::desc(o), dplyr::cummean(x)) > 5)
  })
})

test_that("order_by() inside summarise() arguments keeps its order", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_same(df, function(d) dplyr::summarise(d, s = sum(dplyr::order_by(o, dplyr::cummean(x)))))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::summarise(s = max(dplyr::order_by(o, dplyr::cummean(x))),
                       t = sum(dplyr::order_by(o, cumsum(x) + dplyr::lag(x)), na.rm = TRUE))
  }, arrange_by = "g")
  f72_same(df, function(d) {
    # (last(cumall(b)) in summarise() is typed integer even without order_by():
    # a separate, pre-existing gap, so the logical goes through sum() here)
    d |> dplyr::group_by(g) |>
      dplyr::summarise(l = sum(dplyr::order_by(o, dplyr::cumall(b)), na.rm = TRUE),
                       m = dplyr::last(dplyr::order_by(dplyr::desc(o), dplyr::cummean(x))))
  }, arrange_by = "g")
})

# ---------------------------------------------------------------------------
# Shapes dplyr rejects: the GPU table raises the same error
# ---------------------------------------------------------------------------

test_that("order_by() shapes dplyr rejects raise dplyr's error", {
  skip_if_no_gpu()
  df <- f72_df()
  # with_order() has no `x` to bind
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::row_number())))
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::ntile(n = 2))))
  # the first argument is a scalar: `order_by` must have size 1
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, 1 + x)))
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, 2 * cumsum(x))))
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(mean(x)))))
  # an aggregate result can't be permuted back
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, mean(x))))
  # `call` must be a call
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, x)))
  # lag()'s own order_by= is taken by with_order()'s `order_by`
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x, order_by = y))))
  # unary minus is not defined for strings and Dates
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(-s, cumsum(x))))
  f72_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(-dt, cumsum(x))))
  f72_same_error(df, function(d) dplyr::filter(d, dplyr::order_by(o, dplyr::row_number()) > 2))
})

# ---------------------------------------------------------------------------
# Order keys: types, collation, expressions
# ---------------------------------------------------------------------------

test_that("order keys of other types match dplyr", {
  skip_if_no_gpu()
  df <- f72_df()
  df$f <- factor(c("z", "y", "x", "z", "y", "x", "z", NA, "y", "x", "z", "y"), levels = c("z", "x", "y"))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dt, dplyr::cummean(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(dt), cumsum(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(f, dplyr::cummean(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(f), cumsum(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(b, dplyr::cummean(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(-b, cumsum(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(y, dplyr::cummean(x))))
  # a general expression as the order key
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o * -1, dplyr::cummean(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(x %% 4, cumsum(x))))
})

test_that("string order keys follow dplyr under C and non-C collations", {
  skip_if_no_gpu()
  df <- f72_df()
  # ascending: vec_order_radix() orders strings in C order in every locale
  withr::with_collate("C", {
    f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(s, dplyr::cummean(x))))
    f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(s), cumsum(x))))
  })
  withr::local_collate("en_US.UTF-8")
  skip_if_not(identical(Sys.getlocale("LC_COLLATE"), "en_US.UTF-8"), "en_US.UTF-8 collation not available")
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(s, cumsum(x))))
  # desc() is -xtfrm(): the session collation decides
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(s), cumsum(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(s), dplyr::cummean(x))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::lead(x, order_by = dplyr::desc(s))))
  f72_same(df, function(d) dplyr::mutate(d, r = dplyr::lag(x, order_by = dplyr::desc(s))))
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(r = dplyr::first(x, order_by = dplyr::desc(s)))
  })
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(r = dplyr::last(x, order_by = dplyr::desc(s)))
  }, arrange_by = "g")
  f72_same(df, function(d) dplyr::filter(d, dplyr::lag(x, order_by = dplyr::desc(s)) > 3))
})

# ---------------------------------------------------------------------------
# Other order_by= arguments (guards)
# ---------------------------------------------------------------------------

test_that("lag/lead/first/last/nth order_by= arguments keep their order", {
  skip_if_no_gpu()
  df <- f72_df()
  f72_native(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(
      a = dplyr::lag(x, order_by = o),
      b2 = dplyr::lead(x, 2, order_by = dplyr::desc(o)),
      c = dplyr::first(x, order_by = o),
      e = dplyr::last(x, order_by = dplyr::desc(o)),
      f = dplyr::nth(x, 2, order_by = o)
    )
  })
  f72_same(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(
      a = dplyr::first(x, order_by = o),
      e = dplyr::last(x, order_by = dplyr::desc(o)),
      f = dplyr::nth(y, 2, order_by = o)
    )
  }, arrange_by = "g")
  f72_same(df, function(d) dplyr::slice_min(d, o, n = 3))
  f72_same(df, function(d) dplyr::group_by(d, g) |> dplyr::slice_max(o, n = 2))
  f72_same(df, function(d) dplyr::arrange(d, o) |> dplyr::mutate(r = cumsum(x), .by = g))
})

# ---------------------------------------------------------------------------
# Edge inputs
# ---------------------------------------------------------------------------

test_that("order_by() on 0-row and all-NA-order inputs matches dplyr", {
  skip_if_no_gpu()
  df <- f72_df()
  empty <- df[0, ]
  f72_same(empty, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(x))))
  f72_same(empty, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) * 2)))
  f72_same(empty, function(d) dplyr::group_by(d, g) |> dplyr::mutate(r = dplyr::order_by(o, dplyr::cumall(b))))
  all_na <- df
  all_na$o <- NA_integer_
  f72_same(all_na, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::cummean(x))))
  f72_same(all_na, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) + dplyr::lag(x))))
  f72_same(all_na, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(o), dplyr::consecutive_id(i))))
})

test_that("a lazy order_by() plan survives compute()", {
  skip_if_no_gpu()
  df <- f72_df()
  expected <- dplyr::mutate(df, r = dplyr::order_by(o, dplyr::cummean(x)), q = dplyr::order_by(o, cumsum(x) + dplyr::lag(x)))
  got <- tbl_gpu(df) |> as_lazy() |>
    dplyr::mutate(r = dplyr::order_by(o, dplyr::cummean(x)), q = dplyr::order_by(o, cumsum(x) + dplyr::lag(x))) |>
    dplyr::compute() |>
    collect()
  expect_equal(tibble::as_tibble(got), tibble::as_tibble(expected))
})
