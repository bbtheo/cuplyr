# #30: an across()/if_any() selection error of filter()/summarise() (which are
# not wrapped yet) reaches the user's handlers through rlang's abort(), as
# tidyselect raised it, not through a base stop() re-raise: it prints in rlang's
# format and rlang::last_error() records it.

p30s_signaller <- function(pipeline) {
  calls <- NULL
  tryCatch(
    withCallingHandlers(
      {
        out <- pipeline()
        if (inherits(out, "tbl_gpu")) dplyr::collect(out)
      },
      error = function(e) calls <<- sys.calls()
    ),
    error = function(e) NULL
  )
  n <- length(calls)
  if (n < 2L) NA_character_ else paste(deparse(calls[[n - 1L]][[1L]]), collapse = "")
}

test_that("filter()/summarise() selection errors are signalled through rlang", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 1, 2), a = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    info <- sprintf("lazy = %s", lazy)
    expect_identical(p30s_signaller(function() filter(tbl_gpu(df, lazy = lazy), if_any(no_such_col, ~ .x > 1))),
                     "signalCondition", info = info)
    expect_identical(p30s_signaller(function() filter(tbl_gpu(df, lazy = lazy), if_all(no_such_col, function(v) v > 1))),
                     "signalCondition", info = info)
    expect_identical(p30s_signaller(function() summarise(tbl_gpu(df, lazy = lazy), across(no_such_col, mean))),
                     "signalCondition", info = info)
  }
})
