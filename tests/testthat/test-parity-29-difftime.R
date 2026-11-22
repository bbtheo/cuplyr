# #29 acceptance tests: a difftime made by a verb that ran on the CPU.
#
# `Date - Date` and `POSIXct - POSIXct` give a difftime, which has no GPU
# type, so the CPU route computes the verb and then can't upload its result.
# Before #29 the user got the upload error meant for tbl_gpu(): "Can't upload
# column(s) ...  Convert them first", which names neither the verb nor the
# expression that made the column, and advises converting an input the user
# never had.
#
# cuplyr contract: a verb whose CPU result keeps a column with no GPU type
# raises a classed error (`cuplyr_error_unsupported_result_class`, a subclass
# of `cuplyr_error_unsupported_class`, so F37's contract still holds) that
# names the verb, the column and its class, and says how to keep the result
# on the GPU (`as.numeric()` in the same call). A difftime that does not
# survive into the result gives dplyr's result. Native difftime columns are
# #65's question, not this one's.

d29_df <- function() {
  tibble::tibble(
    g  = c(1L, 1L, 2L, 2L, 2L),
    d  = as.Date(c("2024-01-01", NA, "2024-02-29", "2023-12-31", "1969-12-30")),
    d2 = as.Date(c("2024-01-10", "2024-01-05", NA, "2024-01-01", "1970-01-02")),
    t1 = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-02 12:00:00", NA,
                      "2024-01-01 00:00:00", "1970-01-01 00:00:00"), tz = "UTC"),
    t2 = as.POSIXct(c("2024-01-01 01:00:00", "2024-01-01 00:00:00", "2024-01-01 00:00:00",
                      NA, "1970-01-01 00:00:30"), tz = "Asia/Tokyo")
  )
}

d29_both <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

# The GPU pipeline (eager and lazy) raises the classed result error, naming
# `verb()`, every column in `cols` and the difftime class, with an
# as.numeric() hint. dplyr itself succeeds.
d29_expect_result_error <- function(df, pipeline, verb, cols) {
  expect_no_error(pipeline(df))
  for (lazy in c(FALSE, TRUE)) {
    err <- expect_error(collect(pipeline(tbl_gpu(df, lazy = lazy))),
                        class = "cuplyr_error_unsupported_result_class")
    expect_s3_class(err, "cuplyr_error_unsupported_class")
    msg <- rlang::cnd_message(err)
    expect_match(msg, paste0(verb, "()"), fixed = TRUE, info = paste("lazy =", lazy))
    for (col in cols) {
      expect_match(msg, paste0("`", col, "`"), fixed = TRUE, info = paste("lazy =", lazy))
    }
    expect_match(msg, "difftime", fixed = TRUE)
    expect_match(msg, "as.numeric", fixed = TRUE)
  }
}

test_that("a difftime from Date - Date in mutate() is a classed error naming the verb", {
  skip_if_no_gpu()
  df <- d29_df()
  d29_expect_result_error(df, function(t) mutate(t, y = d2 - d), "mutate", "y")
  d29_expect_result_error(df, function(t) t |> group_by(g) |> mutate(y = d - min(d, na.rm = TRUE)),
                          "mutate", "y")
  d29_expect_result_error(df, function(t) transmute(t, y = d2 - d), "transmute", "y")
})

test_that("a difftime from POSIXct - POSIXct is a classed error naming the verb", {
  skip_if_no_gpu()
  df <- d29_df()
  d29_expect_result_error(df, function(t) mutate(t, y = t2 - t1), "mutate", "y")
})

test_that("`.x - .x` across Date and POSIXct columns names every difftime column (the fuzz case)", {
  skip_if_no_gpu()
  df <- d29_df()[c("d", "t1")]
  d29_expect_result_error(df, function(t) mutate(t, across(everything(), ~ .x - .x)),
                          "mutate", c("d", "t1"))
  d29_expect_result_error(df[0, ], function(t) mutate(t, across(everything(), ~ .x - .x)),
                          "mutate", c("d", "t1"))
})

test_that("a difftime from summarise() is a classed error naming the verb", {
  skip_if_no_gpu()
  df <- d29_df()
  d29_expect_result_error(df, function(t) summarise(t, span = max(d, na.rm = TRUE) - min(d, na.rm = TRUE)),
                          "summarise", "span")
  d29_expect_result_error(df, function(t) t |> group_by(g) |> summarise(span = max(d2) - min(d)),
                          "summarise", "span")
})

test_that("Date and POSIXct differences match dplyr when no difftime reaches the result", {
  skip_if_no_gpu()
  df <- d29_df()
  d29_both(df, function(t) mutate(t, y = as.numeric(d2 - d)))
  d29_both(df, function(t) mutate(t, y = as.numeric(t2 - t1, units = "secs")))
  d29_both(df, function(t) mutate(t, across(c(d, t1), ~ as.numeric(.x - .x))))
  d29_both(df, function(t) filter(t, d2 - d > 3))
  d29_both(df, function(t) summarise(t, span = as.numeric(max(d, na.rm = TRUE) - min(d, na.rm = TRUE))))
})

test_that("uploading a difftime column keeps tbl_gpu()'s own error", {
  skip_if_no_gpu()
  df <- tibble::tibble(y = as.difftime(c(1, 2), units = "days"))
  err <- expect_error(tbl_gpu(df), class = "cuplyr_error_unsupported_class")
  expect_false(inherits(err, "cuplyr_error_unsupported_result_class"))
  expect_match(rlang::cnd_message(err), "`y` <difftime>", fixed = TRUE)
})

test_that("a difftime from a verb built on another verb names the verb the user called", {
  skip_if_no_gpu()
  df <- d29_df()
  # count(wt =) sums the weight in a summarise(); the error is count()'s
  for (lazy in c(FALSE, TRUE)) {
    err <- expect_error(collect(count(tbl_gpu(df, lazy = lazy), g, wt = d2 - d)),
                        class = "cuplyr_error_unsupported_result_class")
    msg <- rlang::cnd_message(err)
    expect_match(msg, "count()", fixed = TRUE)
    expect_false(grepl("summarise()", msg, fixed = TRUE))
  }
  expect_no_error(count(df, g, wt = d2 - d))
})

test_that("a CPU fallback keeps a vctrs unspecified column, as before", {
  skip_if_no_gpu()
  df <- tibble::tibble(u = vctrs::unspecified(2), k = 1:2)
  twice <- function(v) v * 2
  d29_both(df, function(t) mutate(t, y = twice(k)))
})
