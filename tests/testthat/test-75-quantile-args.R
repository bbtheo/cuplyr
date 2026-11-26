# #75: the fast quantile() paths (a literal probability matched by position,
# reframe()'s rewrite, the label written out) give what quantile() gives,
# names and errors included, for every argument form they take.

p75a_df <- function() data.frame(x = c(1, 2, 3, 4), y = c("a", "a", "b", "b"))

# The values and the element names of column `col`, eager and lazy, against
# dplyr, from a data.frame and from a tibble (dplyr gives a data.frame's
# result no element names).
p75a_same_names <- function(df, pipeline, col = "q") {
  for (src in list(df, tibble::as_tibble(df))) {
    want <- pipeline(src)[[col]]
    for (lazy in c(FALSE, TRUE)) {
      info <- paste0("lazy = ", lazy, ", ", class(src)[[1L]], ": ", deparse(body(pipeline)))
      got <- collect(pipeline(tbl_gpu(src, lazy = lazy)))[[col]]
      expect_identical(names(got), names(want), info = info)
      expect_equal(unname(got), unname(want), info = info)
    }
  }
}

# dplyr's error message, eager and lazy.
p75a_same_error <- function(df, pipeline) {
  msg <- function(f) tryCatch({
    f()
    NA_character_
  }, error = conditionMessage)
  want <- msg(function() pipeline(tibble::as_tibble(df)))
  expect_false(is.na(want), info = deparse(body(pipeline)))
  for (lazy in c(FALSE, TRUE)) {
    expect_identical(msg(function() collect(pipeline(tbl_gpu(df, lazy = lazy)))), want,
                     info = paste0("lazy = ", lazy, ": ", deparse(body(pipeline))))
  }
}

test_that("reframe() of one probability keeps quantile()'s name", {
  skip_if_no_gpu()
  d <- p75a_df()
  p75a_same_names(d, function(t) reframe(group_by(t, y), q = quantile(x, 0.5)))
  p75a_same_names(d, function(t) reframe(group_by(t, y), q = quantile(x, 0.5, names = TRUE)))
  p75a_same_names(d, function(t) reframe(group_by(t, y), q = quantile(x, 0.25), m = mean(x)))
  p75a_same_names(d, function(t) reframe(t, q = quantile(x, 0.5)))
  p75a_same_names(d, function(t) reframe(t, q = quantile(x, probs = 0.75, na.rm = TRUE)))
  p75a_same_names(d, function(t) reframe(t, q = quantile(x, 0.75, FALSE, TRUE)))
  # names = FALSE has none, as quantile() gives
  p75a_same_names(d, function(t) reframe(group_by(t, y), q = quantile(x, 0.5, names = FALSE)))
  p75a_same_names(d, function(t) reframe(t, q = quantile(x, 0.5, FALSE, FALSE)))
})

test_that("reframe() of one probability stays on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  d <- p75a_df()
  got <- collect(reframe(group_by(tbl_gpu(d, lazy = FALSE), y), q = quantile(x, 0.5)))
  expect_identical(names(got$q), c("50%", "50%"))
  got <- collect(reframe(group_by(tbl_gpu(d, lazy = TRUE), y), q = quantile(x, 0.5, names = FALSE)))
  expect_null(names(got$q))
})

test_that("summarise() quantile() with digits below 1 is dplyr's error", {
  skip_if_no_gpu()
  d <- p75a_df()
  for (digits in list(0, -1, 0.5, NA_real_, TRUE, "3")) {
    p75a_same_error(d, function(t) summarise(group_by(t, y), b = quantile(x, 0.5, digits = !!digits)))
  }
  p75a_same_error(d, function(t) summarise(t, b = quantile(x, 0.5, digits = 0)))
  p75a_same_error(d, function(t) summarise(group_by(t, y), b = quantile(x, probs = 0.5, names = TRUE, digits = 0)))
})

test_that("summarise() quantile() digits are not checked without names, as quantile() doesn't", {
  skip_if_no_gpu()
  d <- p75a_df()
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, 0.5, names = FALSE, digits = 0)), "q")
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, 0.5, names = FALSE, digits = -1)), "q")
})

test_that("summarise() quantile() labels and values follow every argument form", {
  skip_if_no_gpu()
  d <- p75a_df()
  forms <- rlang::exprs(
    quantile(x, 0.5),
    quantile(x, probs = 0.25),
    quantile(x, 1 / 3),
    quantile(x, 0.5, digits = 1),
    quantile(x, 1 / 3, digits = 2L),
    quantile(x, 1 / 3, digits = 2.7),
    quantile(x, 1 / 3, digits = 30),
    quantile(x, 0.5, names = FALSE),
    quantile(x, probs = 0.5, names = FALSE),
    quantile(x, 0.5, type = 7),
    quantile(x, 0.5, type = 7L),
    quantile(x, 0.5, na.rm = TRUE),
    quantile(x, 0.5, TRUE),
    quantile(x, 0.5, FALSE, FALSE),
    quantile(x, pr = 0.5),
    quantile(x, 0.5, nam = FALSE),
    quantile(x, 0.5, dig = 3),
    quantile(probs = 0.5, x),
    quantile(x, 1L),
    quantile(x, 0L),
    quantile(x, 0),
    quantile(x, 1),
    quantile(x, 1e-15),
    quantile(x, 1 - 1e-15),
    quantile(x, -1e-15),
    quantile(x, 1 + 1e-15),
    quantile(x, NA),
    quantile(x, NA_real_),
    quantile(x, 0.5, names = 1),
    quantile(x, 0.5, type = 6),
    quantile(x, 0.5, foo = 1)
  )
  for (e in forms) {
    p75a_same_names(d, function(t) summarise(group_by(t, y), q = !!e))
    p75a_same_names(d, function(t) summarise(t, q = !!e))
  }
})

test_that("summarise() quantile() arguments quantile() rejects are dplyr's errors", {
  skip_if_no_gpu()
  d <- p75a_df()
  d$z <- c(1, NA, 3, 4)
  forms <- rlang::exprs(
    quantile(x, 1.5),
    quantile(x, -0.5),
    quantile(x, 1 + 1e-12),
    quantile(x, 0.5, names = NA),
    quantile(x, 0.5, names = "a"),
    quantile(x, 0.5, type = 10),
    quantile(x, 0.5, na = TRUE),
    quantile(x, 0.5, n = TRUE),
    quantile(x, 0.5, na.rm = NA),
    quantile(z, 0.5, digits = 0),
    quantile(x, 0.5, digits = 0),
    quantile(x, 0.5, digits = NA)
  )
  for (e in forms) {
    p75a_same_error(d, function(t) summarise(group_by(t, y), q = !!e))
  }
})

test_that("summarise() quantile() arguments that read a column are evaluated by dplyr", {
  skip_if_no_gpu()
  d <- p75a_df()
  d$p <- c(0.1, 0.2, 0.3, 0.4)
  d$k <- c(2, 2, 3, 3)
  d$ty <- c(7, 7, 1, 1)
  # a variable of the same name outside the data is not what dplyr reads
  p <- 0.5
  k <- 7
  ty <- 6
  p75a_same_error(d, function(t) summarise(group_by(t, y), q = quantile(x, p)))
  p75a_same_error(d, function(t) summarise(group_by(t, y), q = quantile(x, probs = p, type = 7)))
  p75a_same_error(d, function(t) summarise(group_by(t, y), q = quantile(x, 0.5, names = p > 0)))
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, mean(p))))
  # (1/3's label depends on digits, "33%" and "33.3%" by group, where 0.5's is "50%")
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, 1 / 3, digits = min(k))))
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, 1 / 3, digits = .env$k)))
  # type 7 in one group, 1 in the other (6 outside the data)
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, 1 / 3, type = min(ty))))
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, 1 / 3, type = .env$ty)))
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, .env$p)))
  p75a_same_names(d, function(t) summarise(group_by(t, y), q = quantile(x, .data$p[[1L]])))
})

test_that("summarise() quantile() digits past 50 warn as dplyr does", {
  skip_if_no_gpu()
  d <- p75a_df()
  warns <- function(f) {
    got <- character()
    value <- withCallingHandlers(f(), warning = function(w) {
      got <<- c(got, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
    list(names = names(value$q), warnings = got)
  }
  # (summarise() is dplyr's generic, so the header names it as cuplyr's does)
  want <- warns(function() summarise(dplyr::group_by(d, y), q = quantile(x, 1 / 3, digits = 60)))
  expect_length(want$warnings, 1L)
  for (lazy in c(FALSE, TRUE)) {
    got <- warns(function() collect(summarise(group_by(tbl_gpu(d, lazy = lazy), y), q = quantile(x, 1 / 3, digits = 60))))
    expect_identical(got, want)
  }
})
