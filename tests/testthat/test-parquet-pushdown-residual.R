# F85 round 3: what a pushed filter leaves above the scan. A conjunct on its own
# is not the same expression as next to its siblings (a bare NA takes the type
# of the operand beside it), so the filter that stays above the scan has to be
# the one the user wrote.

residual_df <- function() {
  tibble::tibble(
    id = 1:8,
    i = c(1L, NA, 3L, 4L, -5L, 6L, 0L, 2L),
    d = c(1.5, NA, NaN, -2.5, 0, 1.5, Inf, 3.25),
    s = c("x", "y", NA, "x", "z", "", "y", "x"),
    l = c(TRUE, NA, FALSE, TRUE, NA, FALSE, TRUE, FALSE)
  )
}

test_that("a pushed filter gives dplyr's rows whatever stays above the scan", {
  skip_if_no_gpu()
  flag_na <- NA
  flag_t <- TRUE
  df <- residual_df()
  path <- pq_write(df)
  pipes <- list(
    bare_na_right = function(x) filter(x, i > 1L & NA),
    bare_na_left = function(x) filter(x, NA & i > 1L),
    variable_na = function(x) filter(x, i > 1L & flag_na),
    bare_true = function(x) filter(x, i > 1L & TRUE),
    bare_false = function(x) filter(x, i > 1L & FALSE),
    variable_true = function(x) filter(x, i > 1L & flag_t),
    na_twice = function(x) filter(x, i > 1L & NA & NA),
    negated_na = function(x) filter(x, i > 1L & !NA),
    na_or_na = function(x) filter(x, d > 0 & (NA | NA)),
    na_or_false = function(x) filter(x, s == "x" & (NA | FALSE)),
    na_and_true = function(x) filter(x, i > 1L & (NA & TRUE)),
    na_compare = function(x) filter(x, i > 1L & (NA > 1)),
    typed_na_compare = function(x) filter(x, i > 1L & (NA_real_ < 1)),
    constant = function(x) filter(x, i > 1L & (1 > 0)),
    constant_false = function(x) filter(x, i > 1L & (1 < 0)),
    all_na_if_else = function(x) filter(x, i > 1L & dplyr::if_else(l, NA, NA)),
    all_na_case_when = function(x) filter(x, i > 1L & dplyr::case_when(l ~ NA)),
    # the same residuals as separate arguments, which the AND split never sees
    dots_na = function(x) filter(x, i > 1L, NA),
    dots_all_na_if_else = function(x) filter(x, i > 1L, dplyr::if_else(l, NA, NA)),
    dots_all_na_case_when = function(x) filter(x, d > 0, dplyr::case_when(l ~ NA)),
    # no surviving row for the residual to run on
    nothing_left = function(x) filter(x, i > 99L & NA),
    nothing_left_dots = function(x) filter(x, i > 99L, dplyr::if_else(l, NA, NA)),
    # a column-free part next to a column that also stays above
    mixed = function(x) filter(x, i > 1L & l & NA),
    mixed_true = function(x) filter(x, d > 0 & l & TRUE)
  )
  for (nm in names(pipes)) {
    want <- pipes[[nm]](df)
    for (push in c(TRUE, FALSE)) {
      got <- withr::with_options(list(cuplyr.parquet_pushdown = push),
                                 collect(pipes[[nm]](read_parquet_gpu(path, lazy = TRUE))))
      expect_equal(tibble::as_tibble(got), want, info = paste(nm, "pushdown", push))
    }
  }
})

test_that("the convertible part of a conjunction is still read as a filter", {
  skip_if_no_gpu()
  df <- residual_df()
  path <- pq_write(df)
  for (pipe in list(function(x) filter(x, i > 1L & NA),
                    function(x) filter(x, i > 1L & l),
                    function(x) filter(x, i > 1L, l))) {
    r <- pq_run(pipe(read_parquet_gpu(path, lazy = TRUE)))
    expect_length(r$read$predicates, 1L)
    expect_equal(r$result$id, pipe(df)$id)
  }
  # nothing to re-check above the scan when every conjunct was pushed
  q <- filter(read_parquet_gpu(path, lazy = TRUE), i > 1L & id > 0L)
  expect_identical(optimize_ast(q$lazy_ops)$type, "source")
  # (a double comparison is one of the things that stays above)
  q <- filter(read_parquet_gpu(path, lazy = TRUE), i > 1L & d > 0)
  expect_identical(optimize_ast(q$lazy_ops)$type, "filter")
})
