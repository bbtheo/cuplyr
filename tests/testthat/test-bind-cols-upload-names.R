# bind_cols() uploads each non-GPU piece under its repaired result name, so the
# timestamp upload warnings name result columns (F81).

bcn_warnings <- function(expr) {
  w <- character()
  value <- withCallingHandlers(
    expr,
    warning = function(cnd) {
      w <<- c(w, conditionMessage(cnd))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = w)
}

test_that("bind_cols() warnings name the right column of a multi-column piece", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(a = 1:2))
  piece <- tibble::tibble(a = 1:2, b = as.Date(c(Inf, 2)), a = c(1, 2),
                          .name_repair = "minimal")
  cap <- suppressMessages(bcn_warnings(collect(bind_cols(g, piece))))
  expect_length(cap$warnings, 1L)
  expect_match(cap$warnings, "`b` \\(1 value\\)")
  expect_named(cap$value, c("a...1", "a...2", "b", "a...4"))
})

test_that("bind_cols() takes a grouped data frame piece and warns under its names", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(x = 1:2))
  piece <- dplyr::group_by(tibble::tibble(a = as.Date(c(Inf, 2)), k = 1:2), k)
  cap <- bcn_warnings(collect(bind_cols(g, piece)))
  expect_length(cap$warnings, 1L)
  expect_match(cap$warnings, "`a` \\(1 value\\)")
  expect_named(cap$value, c("x", "a", "k"))
})

test_that("bind_cols(.name_repair = 'minimal') keeps the duplicated name in the warning", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(a = 1:2))
  cap <- bcn_warnings(
    collect(bind_cols(g, a = as.Date(c(Inf, 2)), .name_repair = "minimal"))
  )
  expect_length(cap$warnings, 1L)
  expect_match(cap$warnings, "`a` \\(1 value\\)")
  expect_identical(names(cap$value), c("a", "a"))
})

test_that("mutate() with several constants names each output column in one warning", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(a = 1:2))
  cap <- bcn_warnings(
    collect(mutate(g, d = as.Date(c(Inf, 2)), e = as.Date(c(1, Inf))))
  )
  expect_length(cap$warnings, 1L)
  expect_match(cap$warnings, "`d` \\(1 value\\), `e` \\(1 value\\)")
  expect_identical(names(cap$value), c("a", "d", "e"))
})

test_that("a grouped first input's duplicate-name error comes before any upload warning", {
  skip_if_no_gpu()
  inf_vec <- as.Date(c(Inf, 2))
  for (lazy in c(FALSE, TRUE)) {
    g <- group_by(tbl_gpu(tibble::tibble(a = 1:2), lazy = lazy), a)
    binds <- list(
      function() bind_cols(g, a = inf_vec, .name_repair = "minimal"),
      function() bind_cols(g, tibble::tibble(a = inf_vec), .name_repair = "minimal")
    )
    for (bind in binds) {
      seen <- character()
      expect_error(
        withCallingHandlers(
          bind(),
          warning = function(cnd) {
            seen <<- c(seen, conditionMessage(cnd))
            invokeRestart("muffleWarning")
          }
        ),
        class = "tibble_error_column_names_must_be_unique"
      )
      expect_identical(seen, character())
    }
  }
})

test_that("named bind_cols() pieces of every kind leave schema names plain", {
  skip_if_no_gpu()
  g <- tbl_gpu(tibble::tibble(a = 1:2))
  gp <- tbl_gpu(tibble::tibble(p = 3:4, q = 5:6))
  out <- bind_cols(
    first = g, none = tibble::tibble(.rows = 2), piece = gp,
    df = tibble::tibble(r = 7:8), s = 9:10
  )
  expect_identical(out$schema$names, c("a", "p", "q", "r", "s"))
  expect_null(names(out$schema$names))
  expect_identical(colnames(out), c("a", "p", "q", "r", "s"))
})
