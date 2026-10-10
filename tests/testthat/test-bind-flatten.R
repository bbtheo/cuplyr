# bind_rows()/bind_cols() argument flattening matches dplyr 1.2.1 when a
# tbl_gpu is among the inputs: list2(), bind_rows()'s single-bare-list unwrap,
# and vctrs::obj_is_list()-based splicing (unnamed lists only for bind_rows,
# recursive for bind_cols).

# Run `fn` once with tbl_gpu inputs and once with the same data as tibbles;
# both must succeed with the same data or fail with the same message.
bind_vs_dplyr <- function(fn, ...) {
  dfs <- lapply(list(...), tibble::as_tibble)
  expected <- tryCatch(do.call(fn, dfs), error = function(e) e)
  actual <- tryCatch(do.call(fn, lapply(dfs, tbl_gpu)), error = function(e) e)
  if (inherits(expected, "error")) {
    expect_s3_class(actual, "error")
    if (inherits(actual, "error")) {
      expect_identical(conditionMessage(actual), conditionMessage(expected))
    }
    return(invisible(actual))
  }
  if (inherits(actual, "error")) {
    fail(paste("cuplyr errored where dplyr did not:", conditionMessage(actual)))
    return(invisible(actual))
  }
  expect_true(is_tbl_gpu(actual))
  expect_identical(collect(actual), tibble::as_tibble(expected))
  invisible(actual)
}

d1 <- function() data.frame(x = 1:3, s = c("a", "b", "c"))
d2 <- function() data.frame(x = 4:6, s = c("d", "e", "f"))
d3 <- function() data.frame(y = 7:9)

test_that("bind_rows() does not splice an S3 list that doesn't inherit from list (#3924)", {
  skip_if_no_gpu()
  special <- function(...) structure(list(...), class = "special_lst")
  bind_vs_dplyr(function(a) cuplyr::bind_rows(special(a, a)), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_rows(a, special(a, a)), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_rows(NULL, special(a, a)), d1())
  # ... but does splice one whose class ends in "list"
  listy <- function(...) structure(list(...), class = c("my_list", "list"))
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(listy(a, b)), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(a, listy(a, b)), d1(), d2())
})

test_that("bind_rows() ignores empty lists (#2826)", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a) cuplyr::bind_rows(list(a, list())), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_rows(a, list(), NULL), d1())
})

test_that("bind_rows() unwraps a single bare list, then splices one level", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(list(list(a, b), b)), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(list(list(a, b))), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(a, list(list(b))), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(a, !!!list(b, a)), d1(), d2())
})

test_that("bind_rows(.id =) takes names from named lists and dots", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(list(a = a, b = b), .id = "id"), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(a = a, list(b = b, b), .id = "id"), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(list(a = a, b = NULL, c = b), .id = "id"), d1(), d2())
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(a, b, .id = "id"), d1(), d2())
  # names are dropped without .id
  bind_vs_dplyr(function(a, b) cuplyr::bind_rows(p = a, q = b), d1(), d2())
})

test_that("bind_rows(.id =) numbers NA names like empty ones (#7100)", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a) {
    xs <- list(a, a, a)
    names(xs) <- c("p", NA, "r")
    cuplyr::bind_rows(xs, .id = "id")
  }, d1())
})

test_that("bind_rows() treats named lists and named vectors as rows", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a) cuplyr::bind_rows(a, list(x = 7L, s = "q")), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_rows(a, c(x = 7L)), d1())
})

test_that("bind_rows() rejects inputs that aren't data frames like dplyr does", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a) cuplyr::bind_rows(a, 1:3), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_rows(a, .id = 1), d1())
  # a named list of tables is a packed row in dplyr; the GPU can't hold that
  g1 <- tbl_gpu(d1())
  expect_error(
    cuplyr::bind_rows(g1, list(a = g1, b = g1)),
    class = "cuplyr_error_bind_unsupported"
  )
})

test_that("bind_cols() flattens nested lists recursively", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a, c) cuplyr::bind_cols(list(list(a), c)), d1(), d3())
  bind_vs_dplyr(function(a, c) cuplyr::bind_cols(a, list(NULL, list(c))), d1(), d3())
  bind_vs_dplyr(function(a, c) cuplyr::bind_cols(list(p = a, list(q = c))), d1(), d3())
})

test_that("bind_cols() adds vectors as columns next to a tbl_gpu", {
  skip_if_no_gpu()
  bind_vs_dplyr(function(a) cuplyr::bind_cols(a, k = 11:13), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_cols(a, k = 1L), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_cols(k = c(1.5, 2.5, 3.5), a), d1())
  bind_vs_dplyr(function(a) cuplyr::bind_cols(a, list(k = 11:13, m = "z")), d1())
  # A record type stays one column instead of being spliced into its fields
  # (11 columns). dplyr keeps it as a POSIXlt column, which a tbl_gpu can't
  # hold, so the upload fails exactly as tbl_gpu(tibble(y = lt)) does.
  lt <- as.POSIXlt(as.POSIXct(c("2020-01-01", "2020-01-02", "2020-01-03"), tz = "UTC"))
  expect_error(cuplyr::bind_cols(tbl_gpu(d1()), y = lt), "Unsupported column type")
})

test_that("bind_cols() keeps factor levels", {
  skip_if_no_gpu()
  f <- factor(c("a", "b", "a"), levels = c("b", "a"))
  bind_vs_dplyr(function(a, c) cuplyr::bind_cols(a, c), data.frame(f = f), d3())
  bind_vs_dplyr(function(c, a) cuplyr::bind_cols(c, a), data.frame(f = f), d3())
  bind_vs_dplyr(function(c) cuplyr::bind_cols(c, k = f), d3())
  bind_vs_dplyr(function(a) suppressMessages(cuplyr::bind_cols(a, a)), data.frame(f = f))
})
