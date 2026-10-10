# F06 acceptance tests: join key type coercion (gap A3).
#
# dplyr casts each key pair to its common type (vctrs::vec_ptype2(), via
# dplyr:::join_cast_common()) before matching, and raises
# `dplyr_error_join_incompatible_type` for pairs with no common type.
# Output types depend on how a key reaches the output:
#   - merged equi keys (keep = NULL/FALSE): the common type
#   - keep = TRUE, semi/anti x columns, non-equi columns: original types
# Behaviour verified with tools/r against dplyr 1.2.1
# (scratchpad/parity/features/F06/probe/p1.out).

# Build `df` as the same kind of table as `d`: a tibble for the dplyr oracle,
# a tbl_gpu (lazy if `d` is lazy) for the GPU side.
f06k_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f06k_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f06k_x_int <- data.frame(id = 1:4, a = c("p", "q", "r", "s"))
f06k_y_dbl <- data.frame(id = c(1, 2, 2, 5), b = c(10, 20, 21, 50))

# -----------------------------------------------------------------------------
# INT32 vs FLOAT64
# -----------------------------------------------------------------------------

test_that("mutating joins coerce INT32 x FLOAT64 merged keys to double", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(f06k_x_int, function(d) join(d, f06k_like(d, f06k_y_dbl), by = "id"))
  }
})

test_that("mutating joins coerce FLOAT64 x INT32 merged keys to double", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(f06k_y_dbl, function(d) join(d, f06k_like(d, f06k_x_int), by = "id"))
  }
})

test_that("keep = TRUE keeps each side's original key type", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(f06k_x_int, function(d) {
      join(d, f06k_like(d, f06k_y_dbl), by = "id", keep = TRUE)
    })
  }
})

test_that("non-integer doubles never match an integer key", {
  skip_if_no_gpu()
  y <- data.frame(id = c(1.5, 2, 3.25), b = 1:3)
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(f06k_x_int, function(d) join(d, f06k_like(d, y), by = "id"))
  }
})

test_that("semi_join()/anti_join() match on the common type and keep x's types", {
  skip_if_no_gpu()
  x <- data.frame(id = c(1L, 2L, 3L, NA), a = c("p", "q", "r", "s"))
  y <- data.frame(id = c(1.5, 2, NA))
  for (join in list(semi_join, anti_join)) {
    for (na in c("na", "never")) {
      f06k_both(x, function(d) join(d, f06k_like(d, y), by = "id", na_matches = na))
    }
  }
})

test_that("differently named keys are coerced (by = c(id = 'k'))", {
  skip_if_no_gpu()
  y <- data.frame(k = c(1, 9, 2), z = 1:3)
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(f06k_x_int, function(d) join(d, f06k_like(d, y), by = c(id = "k")))
    f06k_both(f06k_x_int, function(d) {
      join(d, f06k_like(d, y), by = c(id = "k"), keep = TRUE)
    })
  }
  f06k_both(f06k_x_int, function(d) semi_join(d, f06k_like(d, y), by = c(id = "k")))
  f06k_both(f06k_x_int, function(d) anti_join(d, f06k_like(d, y), by = c(id = "k")))
})

test_that("dplyr::join_by() equi keys are coerced", {
  skip_if_no_gpu()
  y <- data.frame(k = c(1, 9, 2), z = 1:3)
  f06k_both(f06k_x_int, function(d) {
    left_join(d, f06k_like(d, y), by = dplyr::join_by(id == k))
  })
  f06k_both(f06k_x_int, function(d) {
    full_join(d, f06k_like(d, y), by = dplyr::join_by(id == k))
  })
})

test_that("NA keys match per na_matches after coercion", {
  skip_if_no_gpu()
  x <- data.frame(id = c(1L, NA, 3L), a = c("p", "q", "r"))
  y <- data.frame(id = c(NA, 1, 4), b = c(1, 2, 3))
  for (join in list(left_join, inner_join, right_join, full_join)) {
    for (na in c("na", "never")) {
      f06k_both(x, function(d) join(d, f06k_like(d, y), by = "id", na_matches = na))
    }
  }
})

test_that("a coerced key next to a same-typed key (multi-column by)", {
  skip_if_no_gpu()
  x <- data.frame(k1 = c(1L, 1L, 2L), k2 = c(1L, 2L, 1L), a = c("p", "q", "r"))
  y <- data.frame(k1 = c(1L, 2L, 2L), k2 = c(2, 1, 3), b = c(10, 20, 30))
  for (join in list(left_join, inner_join, right_join, full_join, semi_join, anti_join)) {
    f06k_both(x, function(d) join(d, f06k_like(d, y), by = c("k1", "k2")))
  }
})

test_that("0-row sides still get the common key type", {
  skip_if_no_gpu()
  x0 <- f06k_x_int[0, , drop = FALSE]
  y0 <- f06k_y_dbl[0, , drop = FALSE]
  for (join in list(left_join, inner_join, right_join, full_join, semi_join, anti_join)) {
    f06k_both(x0, function(d) join(d, f06k_like(d, f06k_y_dbl), by = "id"))
    f06k_both(f06k_x_int, function(d) join(d, f06k_like(d, y0), by = "id"))
  }
})

# -----------------------------------------------------------------------------
# BOOL8 vs INT32 / FLOAT64
# -----------------------------------------------------------------------------

test_that("logical keys are coerced against integer and double keys", {
  skip_if_no_gpu()
  x <- data.frame(k = c(TRUE, FALSE, NA, TRUE), a = 1:4)
  y_int <- data.frame(k = c(1L, 0L, NA, 2L), v = c(10, 20, 30, 40))
  y_dbl <- data.frame(k = c(1, 0.5, 0), v = c(10, 20, 30))
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(x, function(d) join(d, f06k_like(d, y_int), by = "k"))
    f06k_both(x, function(d) join(d, f06k_like(d, y_dbl), by = "k"))
    f06k_both(y_int, function(d) join(d, f06k_like(d, x), by = "k"))
  }
  f06k_both(x, function(d) semi_join(d, f06k_like(d, y_int), by = "k"))
  f06k_both(x, function(d) anti_join(d, f06k_like(d, y_dbl), by = "k"))
})

# -----------------------------------------------------------------------------
# factor vs character
# -----------------------------------------------------------------------------

test_that("factor x character keys join as character", {
  skip_if_no_gpu()
  fx <- data.frame(id = factor(c("a", "b", NA, "c")), v = 1:4)
  cy <- data.frame(id = c("a", "c", "d", NA), w = c(10, 30, 40, 50))
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(fx, function(d) join(d, f06k_like(d, cy), by = "id"))
    f06k_both(cy, function(d) join(d, f06k_like(d, fx), by = "id"))
    f06k_both(fx, function(d) join(d, f06k_like(d, cy), by = "id", na_matches = "never"))
  }
})

test_that("factor x character with keep = TRUE keeps the factor and the character", {
  skip_if_no_gpu()
  fx <- data.frame(id = factor(c("a", "b", "c")), v = 1:3)
  cy <- data.frame(id = c("a", "c", "d"), w = c(10, 30, 40))
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(fx, function(d) join(d, f06k_like(d, cy), by = "id", keep = TRUE))
  }
})

test_that("semi/anti with factor x character keys keep x's factor", {
  skip_if_no_gpu()
  fx <- data.frame(id = factor(c("a", "b", "c", "a")), v = 1:4)
  cy <- data.frame(id = c("a", "c", "z"))
  for (join in list(semi_join, anti_join)) {
    f06k_both(fx, function(d) join(d, f06k_like(d, cy), by = "id"))
    f06k_both(cy, function(d) join(d, f06k_like(d, fx), by = "id"))
  }
})

test_that("factor keys with identical levels stay factor; non-key factors survive", {
  skip_if_no_gpu()
  lv <- c("a", "b", "c")
  x <- data.frame(id = factor(c("a", "b", "c"), levels = lv), k = 1:3,
                  f = factor(c("u", "v", "u")))
  y <- data.frame(id = factor(c("c", "a"), levels = lv), k = c(3, 9), w = 1:2)
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(x, function(d) join(d, f06k_like(d, y), by = c("id", "k")))
  }
})

# -----------------------------------------------------------------------------
# Date vs POSIXct (cuplyr collects every POSIXct as UTC, so use tz = "UTC")
# -----------------------------------------------------------------------------

test_that("Date x POSIXct keys join as POSIXct", {
  skip_if_no_gpu()
  d <- data.frame(t = as.Date(c("2020-01-01", "2020-01-02", NA)), v = 1:3)
  p <- data.frame(
    t = as.POSIXct(c("2020-01-01 00:00:00", "2020-01-02 12:00:00", "2020-01-03 00:00:00"),
                   tz = "UTC"),
    w = c(10, 20, 30)
  )
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06k_both(d, function(x) join(x, f06k_like(x, p), by = "t"))
    f06k_both(p, function(x) join(x, f06k_like(x, d), by = "t"))
  }
  f06k_both(d, function(x) semi_join(x, f06k_like(x, p), by = "t"))
  f06k_both(d, function(x) anti_join(x, f06k_like(x, p), by = "t"))
})

# -----------------------------------------------------------------------------
# Non-equi conditions: matched on the common type, output keeps original types
# -----------------------------------------------------------------------------

test_that("keys of non-equi conditions are not coerced in the output (keep = NULL)", {
  skip_if_no_gpu()
  foo <- data.frame(id = factor(c("a", "b")), col1 = c(1, 2), var1 = "foo")
  bar <- data.frame(id = c("a", "b"), col2 = c(1L, 2L), var2 = "bar")
  f06k_both(foo, function(d) inner_join(d, f06k_like(d, bar), by = dplyr::join_by(id, col1 >= col2)))
  f06k_both(bar, function(d) inner_join(d, f06k_like(d, foo), by = dplyr::join_by(id, col2 <= col1)))
  f06k_both(foo, function(d) left_join(d, f06k_like(d, bar), by = dplyr::join_by(id, col1 > col2)))
})

test_that("pure non-equi INT32 vs FLOAT64 keeps both types", {
  skip_if_no_gpu()
  x <- data.frame(a = 1:3)
  y <- data.frame(b = c(1.5, 2.5))
  f06k_both(x, function(d) {
    inner_join(d, f06k_like(d, y), by = dplyr::join_by(a >= b), relationship = "many-to-many")
  })
  f06k_both(x, function(d) semi_join(d, f06k_like(d, y), by = dplyr::join_by(a > b)))
})

# -----------------------------------------------------------------------------
# Lazy pipelines through the optimizer (compute())
# -----------------------------------------------------------------------------

test_that("coerced keys survive optimizer pushdown/projection in lazy pipelines", {
  skip_if_no_gpu()
  x <- data.frame(id = 1:6, a = c("p", "q", "r", "s", "t", "u"), n = c(1, 2, 3, 4, 5, 6))
  y <- data.frame(id = c(2, 4, 6, 8), b = c(20, 40, 60, 80), junk = letters[1:4])

  f06k_both(x, function(d) {
    d |>
      filter(a != "r") |>
      left_join(f06k_like(d, y), by = "id") |>
      filter(id >= 2) |>
      mutate(c = b * 2) |>
      select(id, c, a)
  })
  f06k_both(x, function(d) {
    d |>
      inner_join(f06k_like(d, y) |> filter(b > 20), by = "id") |>
      select(id, b)
  })
  f06k_both(x, function(d) {
    d |>
      full_join(f06k_like(d, y), by = "id") |>
      filter(is.na(a) | n > 3) |>
      select(id, a)
  })
  f06k_both(x, function(d) {
    d |>
      semi_join(f06k_like(d, y), by = "id") |>
      filter(n > 2) |>
      select(id)
  })
})

test_that("explicit compute() of a coerced-key lazy join matches eager", {
  skip_if_no_gpu()
  fx <- data.frame(id = factor(c("a", "b", "c")), v = 1:3)
  cy <- data.frame(id = c("c", "a"), w = c(30, 10))
  eager <- left_join(tbl_gpu(fx), tbl_gpu(cy), by = "id") |> collect()
  lazy <- left_join(tbl_gpu(fx, lazy = TRUE), tbl_gpu(cy, lazy = TRUE), by = "id") |>
    compute() |>
    collect()
  oracle <- left_join(tibble::as_tibble(fx), tibble::as_tibble(cy), by = "id")
  expect_equal(tibble::as_tibble(eager), oracle)
  expect_equal(tibble::as_tibble(lazy), oracle)
})

# -----------------------------------------------------------------------------
# Incompatible types: dplyr_error_join_incompatible_type, raised at verb time
# -----------------------------------------------------------------------------

# The GPU call is NOT collected: the error must come from the verb itself in
# both eager and lazy mode, with dplyr's classes and exact message.
f06k_expect_incompatible <- function(x, y, join, ...) {
  oracle <- tryCatch(join(tibble::as_tibble(x), tibble::as_tibble(y), ...),
                     error = identity)
  expect_s3_class(oracle, "dplyr_error_join_incompatible_type")
  for (lazy in c(FALSE, TRUE)) {
    gx <- tbl_gpu(x, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    err <- tryCatch(join(gx, gy, ...), error = identity)
    expect_s3_class(err, "dplyr_error_join_incompatible_type")
    expect_s3_class(err, "dplyr_error_join")
    expect_s3_class(err, "dplyr_error")
    expect_identical(conditionMessage(err), conditionMessage(oracle))
  }
}

test_that("numeric/logical vs character keys raise dplyr's incompatible-type error", {
  skip_if_no_gpu()
  chr <- data.frame(id = c("1", "2"), z = 1:2)
  for (join in list(left_join, inner_join, right_join, full_join, semi_join, anti_join)) {
    f06k_expect_incompatible(data.frame(id = 1:3), chr, join, by = "id")
    f06k_expect_incompatible(data.frame(id = c(1, 2)), chr, join, by = "id")
    f06k_expect_incompatible(data.frame(id = c(TRUE, FALSE)), chr, join, by = "id")
    f06k_expect_incompatible(chr, data.frame(id = 1:3), join, by = "id")
  }
})

test_that("factor vs integer keys raise the incompatible-type error", {
  skip_if_no_gpu()
  fx <- data.frame(id = factor(c("a", "b")), v = 1:2)
  iy <- data.frame(id = 1:2, w = 1:2)
  for (join in list(left_join, inner_join, right_join, full_join, semi_join, anti_join)) {
    f06k_expect_incompatible(fx, iy, join, by = "id")
    f06k_expect_incompatible(iy, fx, join, by = "id")
  }
})

test_that("Date/POSIXct vs numeric or character keys raise the incompatible-type error", {
  skip_if_no_gpu()
  d <- data.frame(t = as.Date(c("2020-01-01", "2020-01-02")))
  p <- data.frame(t = as.POSIXct("2020-01-01", tz = "UTC"))
  for (join in list(left_join, inner_join, full_join, semi_join)) {
    f06k_expect_incompatible(d, data.frame(t = c(18262, 1)), join, by = "t")
    f06k_expect_incompatible(d, data.frame(t = "2020-01-01"), join, by = "t")
    f06k_expect_incompatible(p, data.frame(t = 1577836800), join, by = "t")
  }
})

test_that("the error names the right columns for differently named keys", {
  skip_if_no_gpu()
  x <- data.frame(k1 = 1:2, a = c(1, 2))
  y <- data.frame(b = c("x", "y"), k2 = 1:2)
  f06k_expect_incompatible(x, y, left_join, by = c(k1 = "k2", a = "b"))
  f06k_expect_incompatible(x, y, inner_join, by = dplyr::join_by(k1 == k2, a == b))
  f06k_expect_incompatible(x, y, anti_join, by = c(a = "b"))
})

test_that("non-equi conditions with incompatible types raise the same error", {
  skip_if_no_gpu()
  x <- data.frame(a = 1:3)
  y <- data.frame(b = c("1", "2"))
  f06k_expect_incompatible(x, y, inner_join, by = dplyr::join_by(a >= b))
  f06k_expect_incompatible(x, y, left_join, by = dplyr::join_by(a < b))
})

# -----------------------------------------------------------------------------
# Non-equi conditions on compatible but different types (added after round 1,
# review finding B1): matched on the common type, output keeps both types.
# -----------------------------------------------------------------------------

test_that("non-equi Date vs POSIXct conditions match on the common type", {
  skip_if_no_gpu()
  x <- data.frame(d = as.Date(c("2020-01-01", "2020-01-03", NA)), a = 1:3)
  y <- data.frame(t = as.POSIXct(c("2020-01-02 00:00:00", "2020-01-01 00:00:00"), tz = "UTC"),
                  b = c(10, 20))
  f06k_both(x, function(d) {
    inner_join(d, f06k_like(d, y), by = dplyr::join_by(d >= t), relationship = "many-to-many")
  })
  f06k_both(y, function(d) {
    left_join(d, f06k_like(d, x), by = dplyr::join_by(t <= d), relationship = "many-to-many")
  })
  f06k_both(x, function(d) semi_join(d, f06k_like(d, y), by = dplyr::join_by(d > t)))
  f06k_both(x, function(d) anti_join(d, f06k_like(d, y), by = dplyr::join_by(d > t)))
})

test_that("non-equi logical vs integer conditions match on the common type", {
  skip_if_no_gpu()
  x <- data.frame(l = c(TRUE, FALSE, NA))
  y <- data.frame(i = c(0L, 1L, 2L))
  f06k_both(x, function(d) {
    inner_join(d, f06k_like(d, y), by = dplyr::join_by(l <= i), relationship = "many-to-many")
  })
  f06k_both(y, function(d) {
    full_join(d, f06k_like(d, x), by = dplyr::join_by(i < l), relationship = "many-to-many")
  })
  f06k_both(x, function(d) semi_join(d, f06k_like(d, y), by = dplyr::join_by(l >= i)))
})

test_that("mixed equi + non-equi conditions with casts on both kinds of pair", {
  skip_if_no_gpu()
  x <- data.frame(k = c(1L, 1L, 2L, 3L),
                  d = as.Date(c("2020-01-01", "2020-01-05", "2020-01-02", "2020-01-09")),
                  a = 1:4)
  y <- data.frame(k = c(1, 2, 3),
                  t = as.POSIXct(c("2020-01-03", "2020-01-01", "2020-01-10"), tz = "UTC"),
                  b = c(10, 20, 30))
  for (join in list(inner_join, left_join, right_join, full_join, semi_join, anti_join)) {
    f06k_both(x, function(d) join(d, f06k_like(d, y), by = dplyr::join_by(k, d >= t)))
  }
  f06k_both(x, function(d) {
    inner_join(d, f06k_like(d, y), by = dplyr::join_by(k, d >= t), keep = TRUE)
  })
})
