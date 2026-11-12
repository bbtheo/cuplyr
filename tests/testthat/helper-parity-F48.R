# F48 acceptance-test helpers: plain-verb type gaps.
#
# The parity oracle (expect_same_as_dplyr()) compares with a tolerance that
# treats integer and double as equal, so the F48 checks also compare the
# column types (typeof) of the collected results, eager and lazy.

f48_types_of <- function(x) {
  vapply(tibble::as_tibble(dplyr::ungroup(x)), typeof, character(1))
}

# Oracle (eager + lazy), plus exact column types on both GPU schedules.
f48_check <- function(df, pipeline, arrange_by = NULL) {
  expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
  want <- f48_types_of(pipeline(tibble::as_tibble(df)))
  for (lazy in c(FALSE, TRUE)) {
    got <- f48_types_of(collect(pipeline(tbl_gpu(df, lazy = lazy))))
    expect_identical(got, want, info = sprintf("column types (lazy = %s)", lazy))
  }
}

# The same check for a pipeline that warns in dplyr: the GPU side must give
# dplyr's warning (eager and lazy) as well as its result.
f48_check_warns <- function(df, pipeline, warning_regexp, arrange_by = NULL) {
  expect_warning(pipeline(tibble::as_tibble(df)), warning_regexp)
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(collect(pipeline(tbl_gpu(df, lazy = lazy))), warning_regexp,
                   info = sprintf("warning (lazy = %s)", lazy))
  }
  suppressWarnings(f48_check(df, pipeline, arrange_by = arrange_by))
}

# dplyr errors; the GPU side must raise dplyr's error (eager and lazy), never
# a raw cuDF failure or a cuplyr-specific type error.
f48_same_error <- function(df, pipeline, regexp = NULL, class = NULL) {
  expect_error(pipeline(tibble::as_tibble(df)), regexp, class = class)
  for (lazy in c(FALSE, TRUE)) {
    err <- tryCatch(collect(pipeline(tbl_gpu(df, lazy = lazy))), error = function(e) e)
    expect_true(inherits(err, "error"), info = sprintf("GPU side errors (lazy = %s)", lazy))
    if (inherits(err, "error")) {
      msg <- conditionMessage(err)
      expect_false(grepl("CUDF failure", msg, fixed = TRUE),
                   info = sprintf("raw cuDF error (lazy = %s): %s", lazy, msg))
      if (!is.null(regexp)) {
        expect_match(msg, regexp, info = sprintf("error message (lazy = %s)", lazy))
      }
      if (!is.null(class)) {
        expect_true(inherits(err, class),
                    info = sprintf("error class %s (lazy = %s): got %s", class, lazy,
                                   paste(class(err), collapse = "/")))
      }
    }
  }
}

# The pipeline must run without any CPU fallback, eager and lazy, and still
# match dplyr (types included).
f48_native <- function(df, pipeline, arrange_by = NULL) {
  withr::local_options(cuplyr.fallback = "error")
  f48_check(df, pipeline, arrange_by = arrange_by)
}

# TRUE iff the collation `coll` can be set on this host.
f48_collate_ok <- function(coll) {
  ok <- tryCatch(
    withr::with_collate(coll, nzchar(Sys.getlocale("LC_COLLATE"))),
    warning = function(w) FALSE,
    error = function(e) FALSE
  )
  isTRUE(ok)
}
