# F37 attempt 2 acceptance tests: a mutate()/transmute() dot that reads no
# column is evaluated exactly once, also when its value is a fractional Date
# (B8 from attempt 1's round-4 review) or a factor.
#
# F11 pinned the evaluate-once contract for numeric values
# (test-parity-F11-mutate-shapes.R). A fractional Date (as.Date(19723.5))
# can't be a TIMESTAMP_DAYS literal. dplyr evaluates the dot once; cuplyr
# stores the floored day with `cuplyr_warning_fractional_date` (F37
# decision 2), natively when nothing else in the call reads the value, or
# through the CPU route (which must reuse the value already computed) when a
# later dot does.

f37e_df <- function() tibble::tibble(x = 1:3, g = c(1L, 1L, 2L))

f37e_floor <- function(v) as.Date(floor(unclass(v)))

test_that("F37: a column-free fractional-Date dot draws RNG once (mutate)", {
  skip_if_no_gpu()
  df <- f37e_df()

  set.seed(1)
  expected <- dplyr::mutate(df, y = as.Date(19723 + runif(1) * 10))
  expected_next <- runif(1)
  expect_false(unclass(expected$y[1]) == floor(unclass(expected$y[1])))
  expected$y <- f37e_floor(expected$y)

  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = as.Date(19723 + runif(1) * 10))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a column-free fractional-Date dot draws RNG once (transmute)", {
  skip_if_no_gpu()
  df <- f37e_df()

  set.seed(1)
  expected <- dplyr::transmute(df, y = as.Date(19723 + runif(1) * 10))
  expected_next <- runif(1)
  expected$y <- f37e_floor(expected$y)

  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), y = as.Date(19723 + runif(1) * 10))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a side-effecting fractional-Date dot runs once", {
  skip_if_no_gpu()
  df <- f37e_df()

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      as.Date(19723.5)
    }

    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f())),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(res$y, rep(as.Date("2024-01-01"), 3))

    cnt <- 0
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), y = f())),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(names(res), "y")
  }
})

test_that("F37: later dots see the fractional value of an evaluated-once dot", {
  skip_if_no_gpu()
  df <- f37e_df()

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      as.Date(19723.5)
    }
    expected <- dplyr::mutate(df, y = as.Date(19723.5), z = as.numeric(y), w = x + 1L)
    expected$y <- f37e_floor(expected$y)

    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f(), z = as.numeric(y), w = x + 1L)),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)

    # unnamed sibling dots and .keep/.before still behave like dplyr
    cnt <- 0
    expected <- dplyr::mutate(df, y = as.Date(19723.5), .keep = "none", .before = 1)
    expected$y <- f37e_floor(expected$y)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f(), .keep = "none", .before = 1)),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a column-free factor-valued dot runs once", {
  skip_if_no_gpu()
  df <- f37e_df()

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      factor("b", levels = c("a", "b"))
    }
    expected <- dplyr::mutate(df, h = factor("b", levels = c("a", "b")))

    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), h = f()))
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)

    cnt <- 0
    res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), h = f()))
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), dplyr::transmute(df, h = factor("b", levels = c("a", "b"))))
  }
})

test_that("F37: whole-day Date constants and grouped constants are unchanged", {
  skip_if_no_gpu()
  df <- f37e_df()

  for (lazy in c(FALSE, TRUE)) {
    # a whole-day Date constant stays native and is evaluated once
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      as.Date("2024-01-05")
    }
    expect_no_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f(), z = y + 1L))
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), dplyr::mutate(df, y = as.Date("2024-01-05"), z = y + 1L))

    # grouped: a deterministic fractional constant still floors with the warning
    expected <- dplyr::mutate(dplyr::group_by(df, g), y = as.Date(19723.5))
    expected$y <- f37e_floor(expected$y)
    expect_warning(
      res <- collect(dplyr::mutate(dplyr::group_by(tbl_gpu(df, lazy = lazy), g), y = as.Date(19723.5))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(dplyr::ungroup(res)), tibble::as_tibble(dplyr::ungroup(expected)))
    expect_equal(dplyr::group_vars(res), "g")
  }
})

# --- added after round-1 review (B8 via across(), B9, B10) --------------------

test_that("F37: a column-free fractional-Date across() value draws RNG once", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))

  set.seed(1)
  expected <- dplyr::mutate(df, dplyr::across(x, ~ as.Date(19723 + runif(1) * 10)))
  expected_next <- runif(1)
  expected$x <- f37e_floor(expected$x)

  set.seed(1)
  expected_t <- dplyr::transmute(df, dplyr::across(x, ~ as.Date(19723 + runif(1) * 10)))
  expected_t$x <- f37e_floor(expected_t$x)

  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dplyr::across(x, ~ as.Date(19723 + runif(1) * 10)))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)

    set.seed(1)
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), dplyr::across(x, ~ as.Date(19723 + runif(1) * 10)))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected_t)
  }
})

test_that("F37: across() constants run as often as in dplyr next to a fractional-Date dot", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))

  for (lazy in c(FALSE, TRUE)) {
    # numeric across() constant next to a fractional-Date dot
    set.seed(1)
    expected <- dplyr::mutate(df, dplyr::across(x, ~ runif(1)), dd = as.Date(19723.5))
    expected_next <- runif(1)
    expected$dd <- f37e_floor(expected$dd)
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dplyr::across(x, ~ runif(1)), dd = as.Date(19723.5))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)

    # counter: one call per across() column, plus one for the direct dot
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      cnt
    }
    expected <- dplyr::mutate(df, dplyr::across(c(x, k), ~ f()), y = f(), dd = as.Date(19723.5))
    expected_cnt <- cnt
    expected$dd <- f37e_floor(expected$dd)
    cnt <- 0
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dplyr::across(c(x, k), ~ f()), y = f(), dd = as.Date(19723.5))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, expected_cnt)
    expect_equal(tibble::as_tibble(res), expected)

    cnt <- 0
    expected <- dplyr::transmute(df, dplyr::across(x, ~ f()), y = f(), dd = as.Date(19723.5))
    expected_cnt <- cnt
    expected$dd <- f37e_floor(expected$dd)
    cnt <- 0
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), dplyr::across(x, ~ f()), y = f(), dd = as.Date(19723.5))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, expected_cnt)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

# The B9/B10 tests below add a dot reading `dd` so the call takes the CPU route.
test_that("F37: an evaluated constant is never given to a different, identical dot", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  k <- 100L

  for (lazy in c(FALSE, TRUE)) {
    # `a = k` reads the column; `b = k` (after k = NULL) reads the env value
    expected <- dplyr::mutate(df, a = k, k = NULL, b = k, dd = as.Date(19723.5), zz = as.numeric(dd))
    expected$dd <- f37e_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), a = k, k = NULL, b = k, dd = as.Date(19723.5), zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)

    expected <- dplyr::transmute(df, a = k, k = NULL, b = k, dd = as.Date(19723.5), zz = as.numeric(dd))
    expected$dd <- f37e_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), a = k, k = NULL, b = k, dd = as.Date(19723.5), zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: an unnamed evaluated constant keeps dplyr's column name", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa <- 0.25
  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb <- 0.5

  for (lazy in c(FALSE, TRUE)) {
    expected <- suppressWarnings(dplyr::mutate(
      df, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
      dd = as.Date(19723.5), zz = as.numeric(dd)
    ))
    expected$dd <- f37e_floor(expected$dd)
    res <- suppressWarnings(collect(dplyr::mutate(
      tbl_gpu(df, lazy = lazy), aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
      dd = as.Date(19723.5), zz = as.numeric(dd)
    )))
    expect_equal(names(res), names(expected))
    expect_equal(tibble::as_tibble(res), expected)

    expected <- dplyr::transmute(
      df, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
      dd = as.Date(19723.5), zz = as.numeric(dd)
    )
    expected$dd <- f37e_floor(expected$dd)
    res <- suppressWarnings(collect(dplyr::transmute(
      tbl_gpu(df, lazy = lazy), aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
      dd = as.Date(19723.5), zz = as.numeric(dd)
    )))
    expect_equal(names(res), names(expected))
    expect_equal(tibble::as_tibble(res), expected)
  }
})

# --- added after round-2 review (B8 mixed/staged/vector, B11, B12, B13) -------
# A column-free fractional-Date constant that no later dot of the same call
# reads gives the same column natively (floored, with the warning) as through
# the CPU route, so it must not cost a second evaluation of anything else in
# the call. When a later dot does read it (`zz = as.numeric(dd)`), the CPU
# route is needed so that dot sees the fractional value, and across() must
# still be resolved by dplyr against the columns earlier dots created.

test_that("F37: across() mixing a fractional-Date constant with other shapes draws like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  floor_dates <- function(t) {
    t[] <- lapply(t, function(v) if (inherits(v, "Date")) f37e_floor(v) else v)
    t
  }
  pipelines <- list(
    function(t) dplyr::mutate(t, dplyr::across(x, list(d = ~ as.Date(19723 + runif(1) * 10), v = ~ .x))),
    function(t) dplyr::mutate(t, dplyr::across(x, list(r = ~ runif(1), v = ~ .x)), dd = as.Date(19723.5)),
    function(t) dplyr::mutate(t, dplyr::across(c(k = x, x = k), ~ as.Date(19723 + runif(1) * 10))),
    function(t) dplyr::mutate(t, dplyr::across(x, list(d = ~ as.Date(19723 + runif(1) * 10), r = ~ runif(3)))),
    function(t) dplyr::transmute(t, dplyr::across(x, list(d = ~ as.Date(19723 + runif(1) * 10), v = ~ .x)))
  )
  for (pipeline in pipelines) {
    set.seed(1)
    expected <- floor_dates(pipeline(df))
    expected_next <- runif(1)
    for (lazy in c(FALSE, TRUE)) {
      set.seed(1)
      expect_warning(
        res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(runif(1), expected_next)
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

test_that("F37: a fractional-Date dot doesn't re-run sub-calls folded in other dots", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  pipelines <- list(
    function(t) dplyr::mutate(t, y = runif(1) + dplyr::n(), dd = as.Date(19723.5)),
    function(t) dplyr::mutate(t, b = x + runif(1), dd = as.Date(19723.5))
  )
  for (pipeline in pipelines) {
    set.seed(1)
    expected <- pipeline(df)
    expected_next <- runif(1)
    expected$dd <- f37e_floor(expected$dd)
    for (lazy in c(FALSE, TRUE)) {
      set.seed(1)
      expect_warning(
        res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(runif(1), expected_next)
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

test_that("F37: a grouped call with one group runs a fractional-Date dot once", {
  skip_if_no_gpu()
  d1 <- tibble::tibble(x = 1:3, one = 1L)
  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    g <- function() {
      cnt <<- cnt + 1
      as.Date(19723.5 + cnt)
    }
    expect_warning(
      res <- collect(dplyr::mutate(dplyr::group_by(tbl_gpu(d1, lazy = lazy), one), y = g())),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(res$y, rep(as.Date(19724), 3))

    cnt <- 0
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(d1, lazy = lazy), y = g(), .by = one)),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(res$y, rep(as.Date(19724), 3))
  }
})

test_that("F37: across() on the CPU route sees the columns earlier dots made", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  floor_dates <- function(t) {
    t[] <- lapply(t, function(v) if (inherits(v, "Date")) f37e_floor(v) else v)
    t
  }
  pipelines <- list(
    function(t) dplyr::mutate(t, z = 1L, dplyr::across(everything(), ~ sqrt(4)),
                              dd = as.Date(19723.5), zz = as.numeric(dd)),
    function(t) dplyr::transmute(t, z = 1L, dplyr::across(everything(), ~ sqrt(4)),
                                 dd = as.Date(19723.5), zz = as.numeric(dd)),
    function(t) dplyr::mutate(t, k = NULL, dplyr::across(everything(), ~ sqrt(4)),
                              dd = as.Date(19723.5), zz = as.numeric(dd)),
    function(t) dplyr::mutate(t, x = x > 1L, dplyr::across(where(is.numeric), ~ sqrt(4)),
                              dd = as.Date(19723.5), zz = as.numeric(dd))
  )
  for (pipeline in pipelines) {
    expected <- floor_dates(pipeline(df))
    for (lazy in c(FALSE, TRUE)) {
      expect_warning(
        res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

# --- added after round-3 review (B14, N2, N4) ---------------------------------
# A later across()/if_any() whose columns can include the floored constant
# reads it, as does a later dot reading a fractional-Date vector constant:
# those calls must see the fractional value like dplyr (CPU route), not the
# floored one.

test_that("F37: a later across()/if_any() over the fractional-Date column sees dplyr's value", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  floor_dates <- function(t) {
    t[] <- lapply(t, function(v) if (inherits(v, "Date")) f37e_floor(v) else v)
    t
  }
  pipelines <- list(
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), dplyr::across(everything(), ~ .x + 1L)),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), dplyr::across(everything(), ~ .x + 1L, .names = "{.col}_p")),
    function(t) dplyr::transmute(t, dd = as.Date(19723.5), dplyr::across(everything(), ~ .x + 1L)),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), flag = dplyr::if_any(everything(), ~ .x > 19723.2)),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), flag = dplyr::if_all(everything(), ~ .x != 19723))
  )
  for (pipeline in pipelines) {
    expected <- floor_dates(pipeline(df))
    for (lazy in c(FALSE, TRUE)) {
      expect_warning(
        res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

test_that("F37: a later dot reading a fractional-Date vector constant sees dplyr's value", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  for (lazy in c(FALSE, TRUE)) {
    expected <- dplyr::mutate(df, dd = as.Date(c(19723.5, 19724.5, 19725.5)), z = dd > as.Date(19724))
    expected$dd <- f37e_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   dd = as.Date(c(19723.5, 19724.5, 19725.5)), z = dd > as.Date(19724))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a mask-dependent column-free dot is evaluated by dplyr on the CPU route", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  for (lazy in c(FALSE, TRUE)) {
    expected <- dplyr::mutate(df, y = exists("x"), dd = as.Date(19723.5), zz = as.numeric(dd))
    expected$dd <- f37e_floor(expected$dd)
    expect_true(all(expected$y))
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   y = exists("x"), dd = as.Date(19723.5), zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})
