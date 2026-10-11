# F11 acceptance tests: mutate() expression shapes.
#   1. vector literals (A7), recycled like dplyr, native when ungrouped
#   2. `.data$x` auto-naming (rlang::as_label())
#   3. window functions on a non-column scalar (dplyr::row_number("a") etc.)
#   4. `col = NULL` drops a column
# Behavior pinned against dplyr with tools/r; see
# scratchpad/parity/features/F11.md "## Design".

same_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f11_df <- function() {
  tibble::tibble(x = 1:5, g = c(1, 1, 2, 2, 2))
}

# -----------------------------------------------------------------------------
# 1. Vector literals
# -----------------------------------------------------------------------------

test_that("F11: a length-n vector literal becomes a column on the GPU (eager and lazy)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(d, y = 1:5))
  same_both(df, function(d) dplyr::mutate(d, y = c(2.5, NA, 0, -1, 1e10)))
  same_both(df, function(d) dplyr::mutate(d, y = c("a", "b", NA, "d", "e")))
  same_both(df, function(d) dplyr::mutate(d, y = c(TRUE, NA, FALSE, TRUE, TRUE)))
  same_both(df, function(d) dplyr::mutate(d, y = as.Date("2020-01-01") + 0:4))
  # a later dot can use the uploaded column; replacing an existing column
  same_both(df, function(d) dplyr::mutate(d, y = 5:1, z = y + x))
  same_both(df, function(d) dplyr::mutate(d, x = 5:1))
  # inlined constants (`!!`)
  v <- c(10L, 20L, 30L, 40L, 50L)
  same_both(df, function(d) dplyr::mutate(d, out = !!v))
  same_both(df, function(d) dplyr::mutate(d, out = !!quote(1:5)))
  # .keep / .before still work with an uploaded column
  same_both(df, function(d) dplyr::mutate(d, y = 1:5, .before = 1))
  same_both(df, function(d) dplyr::mutate(d, y = 1:5, .keep = "none"))
  # transmute
  same_both(df, function(d) dplyr::transmute(d, y = 1:5, x))
})

test_that("F11: setNames() vector literal keeps its values (names can't live on the GPU)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(df, lazy = lazy)
    res <- collect(dplyr::mutate(gdf, y = setNames(1:5, letters[1:5])))
    expect_equal(unname(res$y), 1:5)
    expect_equal(names(res), c("x", "g", "y"))
  }
})

test_that("F11: 0-row input accepts a length-0 vector literal", {
  skip_if_no_gpu()
  df <- f11_df()[0, ]

  same_both(df, function(d) dplyr::mutate(d, y = integer()))
  same_both(df, function(d) dplyr::mutate(d, y = character()))
})

test_that("F11: a wrong-size vector literal raises dplyr's mutate error", {
  skip_if_no_gpu()
  df <- f11_df()

  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(df, lazy = lazy)
    expect_error(
      collect(dplyr::mutate(gdf, y = 1:3)),
      "`y` must be size 5 or 1, not 3",
      fixed = TRUE, class = "dplyr:::mutate_error"
    )
    expect_error(
      collect(dplyr::mutate(gdf, y = !!(1:3))),
      "Inlined constant `y` must be size 5 or 1, not 3",
      fixed = TRUE, class = "dplyr:::mutate_error"
    )
    expect_error(
      collect(dplyr::transmute(gdf, y = 1:2)),
      class = "dplyr:::mutate_error"
    )
    gdf0 <- tbl_gpu(df[0, ], lazy = lazy)
    expect_error(
      collect(dplyr::mutate(gdf0, y = 1:2)),
      "`y` must be size 0 or 1, not 2",
      fixed = TRUE, class = "dplyr:::mutate_error"
    )
  }
})

test_that("F11: vector literals in grouped and .by mutate follow dplyr's per-group rules", {
  skip_if_no_gpu()
  df6 <- tibble::tibble(g = rep(1:2, each = 3), x = 1:6)

  # a call is evaluated per group: size must be the group size
  same_both(df6, function(d) dplyr::mutate(dplyr::group_by(d, g), y = 1:3))
  same_both(df6, function(d) dplyr::mutate(d, y = 1:3, .by = g))
  # an inlined constant is recycled against the whole table
  v6 <- 6:1
  same_both(df6, function(d) dplyr::mutate(dplyr::group_by(d, g), y = !!v6))

  df <- f11_df()
  for (lazy in c(FALSE, TRUE)) {
    gdf <- dplyr::group_by(tbl_gpu(df, lazy = lazy), g)
    expect_error(collect(dplyr::mutate(gdf, y = 1:5)), class = "dplyr:::mutate_error")
    expect_error(
      collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = 1:5, .by = g)),
      class = "dplyr:::mutate_error"
    )
  }
})

test_that("F11: a vector literal nested inside an expression is elementwise", {
  skip_if_no_gpu()
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(d, y = x + 1:5, z = y * 2))
  same_both(df, function(d) dplyr::mutate(d, y = dplyr::if_else(x > 2, 1:5, 0L)))
})

test_that("F11: lazy plans keep a vector-literal mutate in place (no reordering across it)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  pipes <- list(
    function(d) dplyr::filter(dplyr::mutate(d, y = 1:5), x > 2),
    function(d) dplyr::filter(dplyr::mutate(d, y = 1:5), g == 2, x != 4),
    function(d) dplyr::mutate(dplyr::filter(d, x > 2), y = 1:3),
    function(d) dplyr::filter(dplyr::mutate(dplyr::mutate(d, a = x * 2), y = 5:1, z = a + y), g == 2),
    function(d) dplyr::mutate(dplyr::mutate(d, y = 1:5), y = y * 10L, w = y + x),
    function(d) dplyr::arrange(dplyr::mutate(d, y = c(3L, 1L, 5L, 2L, 4L)), y),
    function(d) dplyr::select(dplyr::mutate(d, y = 1:5), y),
    function(d) dplyr::summarise(dplyr::mutate(d, y = 1:5), s = sum(y)),
    function(d) dplyr::summarise(dplyr::group_by(dplyr::mutate(d, y = 1:5), g), s = sum(y))
  )
  for (p in pipes) {
    same_both(df, p)
  }
})

# -----------------------------------------------------------------------------
# 2. `.data` pronoun auto-naming
# -----------------------------------------------------------------------------

test_that("F11: a bare .data$x / .data[[\"x\"]] dot is named x, without a warning", {
  skip_if_no_gpu()
  df <- f11_df()

  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(df, lazy = lazy)
    expect_no_warning(res <- collect(dplyr::mutate(gdf, .data$x)))
    expect_named(res, c("x", "g"))
    expect_equal(res$x, df$x)
    expect_named(collect(dplyr::mutate(gdf, .data[["x"]])), c("x", "g"))
    expect_named(collect(dplyr::transmute(gdf, .data$x)), "x")
    expect_named(
      collect(dplyr::mutate(dplyr::group_by(gdf, g), .data$x)), c("x", "g")
    )
  }
})

test_that("F11: larger unnamed dots are named like dplyr (as_label)", {
  skip_if_no_gpu()
  df <- f11_df()

  suppressWarnings({
    same_both(df, function(d) dplyr::mutate(d, .data$x + 1))
    same_both(df, function(d) dplyr::mutate(d, .data$x * 2 + .data$g))
    same_both(df, function(d) dplyr::mutate(d, x + 100000000000))
    same_both(df, function(d) dplyr::transmute(d, x + 100000000000))
  })
})

# -----------------------------------------------------------------------------
# 3. Window functions on a non-column scalar
# -----------------------------------------------------------------------------

test_that("F11: filter(dplyr::row_number(<scalar>) == k) broadcasts like dplyr (#781)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  z <- data.frame(a = c(1, 2, 3))
  b <- "a"

  same_both(z, function(d) dplyr::filter(d, dplyr::row_number(b) == 2))
  same_both(z, function(d) dplyr::filter(d, dplyr::row_number(b) == 1))
  df <- f11_df()
  same_both(df, function(d) dplyr::filter(dplyr::group_by(d, g), dplyr::row_number(b) == 1))
  same_both(df, function(d) dplyr::filter(dplyr::group_by(d, g), dplyr::min_rank(b) == 2))
})

test_that("F11: window functions over a scalar argument evaluate on the scalar", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()
  b <- "a"

  same_both(df, function(d) dplyr::mutate(
    d, r = dplyr::row_number(b), m = dplyr::min_rank(b), dr = dplyr::dense_rank(5), p = dplyr::percent_rank(b),
    cd = dplyr::cume_dist(b), nt = dplyr::ntile(b, 2)
  ))
  same_both(df, function(d) dplyr::mutate(
    d, cs = cumsum(2), cm = cummax(3), lg = dplyr::lag(5), ld = dplyr::lead(5), rn = dplyr::row_number(NA)
  ))
  same_both(df, function(d) dplyr::mutate(
    dplyr::group_by(d, g), r = dplyr::row_number(b), cs = cumsum(2), lg = dplyr::lag(5)
  ))
})

test_that("F11: a window call over a constant vector gives the vector result", {
  skip_if_no_gpu()
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(d, cs = cumsum(1:5)))
})

test_that("F11: window calls that read the data are not folded", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(d, r = dplyr::row_number(), n = dplyr::n(), cs = cumsum(x), lg = dplyr::lag(x)))
  same_both(df, function(d) dplyr::mutate(dplyr::group_by(d, g), r = dplyr::row_number(), lg = dplyr::lag(x, default = 0L)))
  same_both(df, function(d) dplyr::filter(d, dplyr::row_number() == 2))
})

# -----------------------------------------------------------------------------
# 4. `col = NULL`
# -----------------------------------------------------------------------------

test_that("F11: mutate(col = NULL) drops the column (eager and lazy)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(d, x = NULL))
  same_both(df, function(d) dplyr::mutate(d, zz = NULL))
  same_both(df, function(d) dplyr::mutate(d, z = 1, z = NULL))
  same_both(df, function(d) dplyr::mutate(d, z = 1, w = z + 1, z = NULL))
  same_both(df, function(d) dplyr::mutate(d, x = NULL, x = 2))
  same_both(df, function(d) dplyr::mutate(d, y = x + 1, z = y, y = NULL, .keep = "used"))
  same_both(df, function(d) dplyr::mutate(d, x = c()))
  same_both(df, function(d) dplyr::mutate(d, r = dplyr::row_number(), x = NULL))

  dff <- tibble::tibble(f = factor(c("b", "a", "b")), x = 1:3)
  same_both(dff, function(d) dplyr::mutate(d, f = NULL))
})

test_that("F11: NULL then re-adding keeps the original location; .before/.after", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df4 <- tibble::tibble(x = 1, y = 2, z = 3, a = 4)

  same_both(df4, function(d) dplyr::mutate(dplyr::group_by(d, z), y = NULL, y = 3, .keep = "all"))
  same_both(df4, function(d) dplyr::mutate(dplyr::group_by(d, z), b = a, y = NULL, y = 3, .keep = "used"))
  same_both(df4, function(d) dplyr::mutate(dplyr::group_by(d, z), b = a, y = NULL, y = 3, .keep = "unused"))
  same_both(df4, function(d) dplyr::mutate(dplyr::group_by(d, z), y = NULL, y = 3, .keep = "all", .before = x))

  same_both(df4, function(d) dplyr::mutate(d, b = NULL, .before = 1))
  same_both(df4, function(d) dplyr::mutate(d, b = 1, b = NULL, .before = 1))
  same_both(df4, function(d) dplyr::mutate(d, b = NULL, b = 1, .before = 1))
  same_both(df4, function(d) dplyr::mutate(d, b = NULL, c = 1, .after = 2))
})

test_that("F11: NULL with groups and .by", {
  skip_if_no_gpu()
  df <- f11_df()

  withr::with_options(list(cuplyr.fallback = "error"), {
    same_both(df, function(d) dplyr::mutate(dplyr::group_by(d, g), x = NULL))
    same_both(df, function(d) dplyr::mutate(d, x = NULL, .by = g))
    same_both(df, function(d) dplyr::mutate(d, g = NULL, .by = g))
    same_both(df, function(d) dplyr::transmute(d, x = NULL, y = g))
    same_both(df, function(d) dplyr::transmute(dplyr::group_by(d, g), x = NULL))
  })

  # dplyr refuses to drop a grouping column of a grouped data frame
  expect_error(dplyr::mutate(dplyr::group_by(df, g), g = NULL))
  for (lazy in c(FALSE, TRUE)) {
    gdf <- dplyr::group_by(tbl_gpu(df, lazy = lazy), g)
    expect_error(collect(dplyr::mutate(gdf, g = NULL)), "`g`", fixed = TRUE)
  }
})

test_that("F11: a column set to NULL is not visible to later dots", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:2, y = 1:2)

  expect_error(dplyr::mutate(df, y = NULL, a = sum(y)), class = "dplyr:::mutate_error")
  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(df, lazy = lazy)
    expect_error(collect(dplyr::mutate(gdf, y = NULL, a = sum(y))), "'y'|`y`")
    expect_error(collect(dplyr::mutate(dplyr::group_by(gdf, x), y = NULL, a = sum(y))), "'y'|`y`")
  }
})

test_that("F11: NULL drops survive lazy fusion and filter pushdown", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(dplyr::mutate(dplyr::mutate(d, a = x + 1), x = NULL), b = a * 2))
  same_both(df, function(d) dplyr::filter(dplyr::mutate(d, x = NULL), g == 2))
  same_both(df, function(d) dplyr::filter(dplyr::mutate(d, y = x * 2, x = NULL), y > 4))
  same_both(df, function(d) dplyr::mutate(dplyr::mutate(d, x = NULL), x = g * 3))
})

# -----------------------------------------------------------------------------
# Added after round 1 review (B1-B3, N1, N2, N4)
# -----------------------------------------------------------------------------

test_that("F11: a column-free dot is evaluated exactly once (RNG and side effects)", {
  skip_if_no_gpu()
  df <- f11_df()

  set.seed(1)
  expected <- dplyr::mutate(df, y = runif(1))$y
  expected_next <- runif(1)
  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = runif(1)))
    expect_equal(res$y, expected)
    expect_equal(runif(1), expected_next)

    set.seed(1)
    res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), y = runif(1)))
    expect_equal(res$y, expected)
    expect_equal(runif(1), expected_next)

    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      1
    }
    collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f()))
    expect_equal(cnt, 1)
  }
})

test_that("F11: window calls over .env$ scalars fold even when the name is a column", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  dfb <- tibble::tibble(x = 1:5, b = c(1, 1, 2, 2, 2), k = 5:1)
  b <- "a"
  k <- 2

  same_both(dfb, function(d) dplyr::mutate(d, r = dplyr::row_number(.env$b), cs = cumsum(.env$k)))
  same_both(dfb, function(d) dplyr::mutate(d, r = dplyr::row_number(.env[["b"]])))
  same_both(dfb, function(d) dplyr::filter(d, dplyr::row_number(.env$b) == 2))
  # the column itself is still a real window input
  same_both(dfb, function(d) dplyr::mutate(d, r = dplyr::row_number(b), cs = cumsum(k)))
})

test_that("F11: aggregate functions over a scalar evaluate on the scalar", {
  skip_if_no_gpu()
  df <- f11_df()

  withr::with_options(list(cuplyr.fallback = "error"), {
    same_both(df, function(d) dplyr::mutate(
      d, s = sum(5), m = mean(5), mx = max(5), mn = min(5), md = median(5),
      nd = dplyr::n_distinct(5), fi = dplyr::first(5)
    ))
    same_both(df, function(d) dplyr::mutate(dplyr::group_by(d, g), s = sum(5), m = mean(5)))
    same_both(df, function(d) dplyr::filter(dplyr::group_by(d, g), sum(1) == 1))
  })
  # var()/sd() of one value is NA
  same_both(df, function(d) dplyr::mutate(d, v = var(5), sdv = sd(5)))
})

# cuplyr stores NaN as a missing value on the GPU (the upload path does the
# same), so a folded NaN collects as NA. Known divergence from dplyr's NaN;
# what matters is that it is missing, not 0 or a number.
test_that("F11: a folded NaN is missing (cuplyr's NaN-as-NA rule)", {
  skip_if_no_gpu()
  df <- f11_df()
  b <- "a"

  for (lazy in c(FALSE, TRUE)) {
    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), p = dplyr::percent_rank(b)))
    expect_true(all(is.na(res$p)))
  }
})

test_that("F11: NULL for a missing column then re-adding it takes dplyr's position", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f11_df()

  same_both(df, function(d) dplyr::mutate(d, a = NULL, b = 1, a = 2))
  same_both(df, function(d) dplyr::transmute(d, a = NULL, b = 1, a = 2))
})

test_that("F11: .before/.after can't name a column dropped in the same call", {
  skip_if_no_gpu()
  df <- f11_df()

  expect_error(dplyr::mutate(df, x = NULL, y = 1, .before = x), "doesn't exist")
  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(df, lazy = lazy)
    expect_error(collect(dplyr::mutate(gdf, x = NULL, y = 1, .before = x)), "doesn't exist")
    expect_error(collect(dplyr::mutate(gdf, x = NULL, y = 1, .after = x)), "doesn't exist")
  }
})

# -----------------------------------------------------------------------------
# Added after round 2 review (B1 residual, B4)
# -----------------------------------------------------------------------------

test_that("F11: a column-free sub-call next to n() is evaluated exactly once", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)

  set.seed(1)
  expected <- dplyr::mutate(df, y = runif(1) + dplyr::n())$y
  expected_next <- runif(1)
  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = runif(1) + dplyr::n()))
    expect_equal(res$y, expected)
    expect_equal(runif(1), expected_next)

    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      1
    }
    collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f() + dplyr::n()))
    expect_equal(cnt, 1)
  }
})

test_that("F11: comparisons with a NaN literal are NA; a NaN value is missing", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, NA))

  same_both(df, function(d) dplyr::filter(d, x != NaN))
  same_both(df, function(d) dplyr::filter(d, x == NaN | x > 1))
  same_both(df, function(d) dplyr::mutate(d, eq = x == NaN, ne = x != NaN, gt = x > NaN))
  for (lazy in c(FALSE, TRUE)) {
    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), ie = dplyr::if_else(x > 1, NaN, x)))
    expect_identical(is.na(res$ie), c(FALSE, TRUE, TRUE))
  }
  # a NaN literal in a value position must behave as missing downstream
  # (is.na(), comparisons, coalesce(), filter row counts), as dplyr's NaN does
  same_both(df, function(d) dplyr::mutate(
    d, r = dplyr::if_else(x > 1, NaN, x), m = is.na(r), m1 = is.na(dplyr::if_else(x > 1, NaN, x)),
    g = r > 0, g1 = dplyr::if_else(x > 1, NaN, x) > 0, c0 = dplyr::coalesce(r, 0)
  ))
  same_both(df, function(d) dplyr::filter(d, !is.na(dplyr::if_else(x > 1, NaN, x))))
  same_both(df, function(d) dplyr::filter(d, !(dplyr::if_else(x > 1, NaN, x) > 0)))
  same_both(df, function(d) dplyr::filter(d, !(dplyr::case_when(x > 1 ~ NaN, .default = x) >= 0)))
})
