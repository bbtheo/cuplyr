# Extra evaluate-once cases (F37 attempt 2): which original dot an evaluated
# value belongs to on the CPU route, across() expansions, names, and the
# native floor of a column-free fractional Date constant (round 3). A dot
# reading `dd` (`zz = as.numeric(dd)`) is what sends a call to the CPU route.

f37x_floor <- function(v) as.Date(floor(unclass(v)))

test_that("F37: a name removed by `col = NULL` is not read from the environment", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  k <- 100L

  for (lazy in c(FALSE, TRUE)) {
    expected <- dplyr::mutate(df, k = NULL, b = k, dd = as.Date(19723.5), zz = as.numeric(dd))
    expected$dd <- f37x_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), k = NULL, b = k, dd = as.Date(19723.5),
                                   zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: each across() column and the following dot get their own value", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  cnt <- 0
  f <- function() {
    cnt <<- cnt + 1
    cnt
  }

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    expected <- dplyr::mutate(df, dplyr::across(c(x, k), ~ f()), y = f(), dd = as.Date(19723.5))
    expected$dd <- f37x_floor(expected$dd)
    expect_equal(expected$y, rep(3, 3))
    cnt <- 0
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dplyr::across(c(x, k), ~ f()), y = f(), dd = as.Date(19723.5))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
    expect_equal(cnt, 3)
  }
})

test_that("F37: on the CPU route an across() is left to dplyr and a direct dot runs once", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      cnt * 10
    }
    expected <- dplyr::mutate(df, dplyr::across(c(x, k), ~ .x * 2L), y = f(),
                              dd = as.Date(19723.5), zz = as.numeric(dd))
    expected$dd <- f37x_floor(expected$dd)
    cnt <- 0
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dplyr::across(c(x, k), ~ .x * 2L), y = f(),
                                   dd = as.Date(19723.5), zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: an unnamed evaluated dot gets dplyr's name, also when short", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  a <- 0.25
  b <- 0.5

  for (lazy in c(FALSE, TRUE)) {
    expected <- suppressWarnings(dplyr::transmute(df, a + b, dd = as.Date(19723.5)))
    expected$dd <- f37x_floor(expected$dd)
    res <- suppressWarnings(collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), a + b, dd = as.Date(19723.5))))
    expect_equal(names(res), names(expected))
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a two-column across() whose first value is a fractional Date draws RNG once per column", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))

  for (verb in c("mutate", "transmute")) {
    dverb <- getExportedValue("dplyr", verb)
    for (lazy in c(FALSE, TRUE)) {
      set.seed(1)
      expected <- dverb(df, dplyr::across(c(x, k), ~ as.Date(19723 + runif(1) * 10)))
      expected_next <- runif(1)
      expected$x <- f37x_floor(expected$x)
      expected$k <- f37x_floor(expected$k)

      set.seed(1)
      expect_warning(
        res <- collect(dverb(tbl_gpu(df, lazy = lazy), dplyr::across(c(x, k), ~ as.Date(19723 + runif(1) * 10)))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(runif(1), expected_next)
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

test_that("F37: a two-column across() with a counter returning a fractional Date runs it once per column", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  cnt <- 0
  g <- function() {
    cnt <<- cnt + 1
    as.Date(19723 + cnt + 0.5)
  }

  for (verb in c("mutate", "transmute")) {
    dverb <- getExportedValue("dplyr", verb)
    for (lazy in c(FALSE, TRUE)) {
      cnt <- 0
      expected <- dverb(df, dplyr::across(c(x, k), ~ g()))
      expected_cnt <- cnt
      expected$x <- f37x_floor(expected$x)
      expected$k <- f37x_floor(expected$k)
      expect_equal(expected_cnt, 2)

      cnt <- 0
      expect_warning(
        res <- collect(dverb(tbl_gpu(df, lazy = lazy), dplyr::across(c(x, k), ~ g()))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(cnt, expected_cnt)
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

# --- round 3: native floor of a column-free fractional Date constant ---------

# Every cuplyr_warning_fractional_date raised while running `code`.
f37x_frac_warnings <- function(code) {
  got <- list()
  withCallingHandlers(code, cuplyr_warning_fractional_date = function(w) {
    got[[length(got) + 1L]] <<- w
    invokeRestart("muffleWarning")
  })
  got
}

test_that("F37: a floored constant warns exactly once, native or CPU route, eager and lazy", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, g = c(1L, 1L, 2L))
  pipelines <- list(
    native = function(t) dplyr::mutate(t, dd = as.Date(19723.5)),
    transmute = function(t) dplyr::transmute(t, dd = as.Date(19723.5)),
    grouped = function(t) dplyr::mutate(dplyr::group_by(t, g), dd = as.Date(19723.5)),
    by = function(t) dplyr::mutate(t, dd = as.Date(19723.5), .by = g),
    window = function(t) dplyr::mutate(t, y = dplyr::n(), dd = as.Date(19723.5)),
    reader = function(t) dplyr::mutate(t, dd = as.Date(19723.5), zz = as.numeric(dd)),
    unparseable = function(t) dplyr::mutate(t, dd = as.Date(19723.5), s = toupper(letters[x]))
  )
  for (nm in names(pipelines)) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f37x_frac_warnings(res <- collect(pipelines[[nm]](tbl_gpu(df, lazy = lazy))))
      expect_length(got, 1L)
      expect_match(conditionMessage(got[[1L]]), "`dd`")
      expect_equal(res$dd, rep(as.Date(19723), 3), label = nm)
    }
  }
})

test_that("F37: a floored constant that a later dot drops or overwrites doesn't warn", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    got <- f37x_frac_warnings(res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = as.Date(19723.5), dd = NULL)))
    expect_length(got, 0L)
    expect_equal(tibble::as_tibble(res), dplyr::mutate(df, dd = as.Date(19723.5), dd = NULL))

    got <- f37x_frac_warnings(res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = as.Date(19723.5), dd = 5)))
    expect_length(got, 0L)
    expect_equal(tibble::as_tibble(res), dplyr::mutate(df, dd = as.Date(19723.5), dd = 5))

    # a call that errors after the parse doesn't warn either (dplyr errors)
    expect_error(dplyr::mutate(df, dd = as.Date(19723.5), .before = nope))
    got <- f37x_frac_warnings(
      expect_error(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = as.Date(19723.5), .before = nope))
    )
    expect_length(got, 0L)
  }
})

test_that("F37: a bare copy of a floored constant stays native and is floored too", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expected <- dplyr::mutate(df, dd = as.Date(19723 + runif(1) * 10), ee = dd)
    expected_next <- runif(1)
    expected$dd <- f37x_floor(expected$dd)
    expected$ee <- f37x_floor(expected$ee)
    set.seed(1)
    got <- f37x_frac_warnings(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = as.Date(19723 + runif(1) * 10), ee = dd))
    )
    expect_equal(runif(1), expected_next)
    expect_length(got, 1L)
    expect_match(conditionMessage(got[[1L]]), "`dd`, `ee`")
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a dot reading a floored constant by string sees the fraction", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3)
  pipelines <- list(
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), zz = as.numeric(.data$dd)),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), zz = as.numeric(.data[["dd"]])),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), zz = as.numeric(get("dd"))),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5), ee = dd, zz = as.numeric(ee))
  )
  for (pipeline in pipelines) {
    expected <- pipeline(df)
    expect_equal(expected$zz, rep(19723.5, 3))
    expected[] <- lapply(expected, function(v) if (inherits(v, "Date")) f37x_floor(v) else v)
    for (lazy in c(FALSE, TRUE)) {
      expect_warning(
        res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

test_that("F37: a negative fractional constant floors like the upload", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f <- function() {
      cnt <<- cnt + 1
      as.Date(-0.5)
    }
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), y = f())),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(res$y, rep(as.Date(-1), 2))
  }
})

test_that("F37: a staged across() of a fractional constant is native, temps don't warn", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    g <- function() {
      cnt <<- cnt + 1
      as.Date(19723.5 + cnt)
    }
    expected <- dplyr::mutate(df, dplyr::across(c(k = x, x = k), ~ g()))
    expected[] <- lapply(expected, f37x_floor)
    expect_equal(cnt, 2)
    cnt <- 0
    got <- f37x_frac_warnings(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dplyr::across(c(k = x, x = k), ~ g())))
    )
    expect_equal(cnt, 2)
    expect_length(got, 1L)
    expect_no_match(conditionMessage(got[[1L]]), "across")
    expect_equal(tibble::as_tibble(res), expected)
  }
})

# --- round 4: across()/if_any() readers, fractional Date vector constants ---

test_that("F37: an unread fractional Date vector constant and its bare copy stay native", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  cnt <- 0
  f <- function() {
    cnt <<- cnt + 1
    as.Date(c(19723.5, 19724.5, 19725.5))
  }

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    expected <- dplyr::mutate(df, dd = f(), b = dd)
    expected$dd <- f37x_floor(expected$dd)
    expected$b <- f37x_floor(expected$b)
    cnt <- 0
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = f(), b = dd)),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
    # a vector isn't inlined on the CPU route: one evaluation means native
    expect_equal(cnt, 1)
  }
})

test_that("F37: a later across() counts as a reader even when its .cols skips the floored column", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  for (lazy in c(FALSE, TRUE)) {
    expected <- dplyr::mutate(df, dd = as.Date(19723.5), dplyr::across(x, ~ .x + 1L))
    expected$dd <- f37x_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = as.Date(19723.5),
                                   dplyr::across(x, ~ .x + 1L))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a grouped later across() over the floored column sees dplyr's value", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4, g = c(1L, 1L, 2L, 2L))
  for (lazy in c(FALSE, TRUE)) {
    expected <- df |>
      dplyr::group_by(g) |>
      dplyr::mutate(dd = as.Date(19723.5), dplyr::across(everything(), ~ .x + 1L)) |>
      dplyr::ungroup()
    expected$dd <- f37x_floor(expected$dd)
    expect_equal(as.numeric(expected$dd), rep(19724, 4))
    expect_warning(
      res <- tbl_gpu(df, lazy = lazy) |>
        dplyr::group_by(g) |>
        dplyr::mutate(dd = as.Date(19723.5), dplyr::across(everything(), ~ .x + 1L)) |>
        dplyr::ungroup() |>
        collect(),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(tibble::as_tibble(res), expected)
  }
})
