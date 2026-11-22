# #29 acceptance tests: `!` on a column type that base R rejects.
#
# Before #29 the `!` registry entry accepted any operand type, so `!s` on a
# string column reached libcudf's AST and failed with its raw
# "Invalid unary operation" (cudf::logic_error).
#
# Base R semantics (checked with dplyr 1.2.1; dplyr adds its `In argument:`
# wrapper, dplyr:::mutate_error in mutate(), a plain rlang_error in filter()):
#   - `!` on a non-empty character vector is an error, "invalid argument
#     type"; `!character(0)` is logical(0), so a 0-row mutate() succeeds.
#   - `!` on a Date or a POSIXct is an error ("unary ! not defined for ...").
#   - `!` on a factor warns "`!` not meaningful for factors" and gives NA.
#   - `!` on logical, integer and double is a logical (`!0` is TRUE).
#
# cuplyr contract: the error cases raise dplyr's own error (the CPU route
# runs the call), never a raw libcudf error; the factor case matches dplyr's
# result and warning; logical/integer/double stay on the GPU.

n29_df <- function() {
  data.frame(
    s = c("abc", NA, "bb", NA, "a"),
    b = c(TRUE, FALSE, NA, TRUE, FALSE),
    i = c(0L, 3L, NA, -1L, 0L),
    x = c(0, 2.5, NA, -0, 1),
    d = as.Date(c("2024-01-01", NA, "2024-02-29", "2023-12-31", "1970-01-01")),
    t = as.POSIXct(c("2024-01-01 00:00:00", NA, "2024-01-02 12:00:00",
                     "2024-01-01 01:00:00", "1970-01-01 00:00:00"), tz = "UTC"),
    f = factor(c("lo", "hi", NA, "lo", "hi"), levels = c("lo", "hi")),
    stringsAsFactors = FALSE
  )
}

# The pipeline errors in dplyr, and the GPU (eager and lazy) raises an error of
# dplyr's class whose message (or a parent's) matches `regexp`. A raw libcudf
# error ("Invalid unary operation") fails both checks.
n29_expect_dplyr_error <- function(df, pipeline, regexp) {
  cpu <- tryCatch(pipeline(tibble::as_tibble(df)), error = identity)
  expect_s3_class(cpu, "error")
  cls <- class(cpu)[1L]
  for (lazy in c(FALSE, TRUE)) {
    gpu <- expect_error(collect(pipeline(tbl_gpu(df, lazy = lazy))),
                        regexp = regexp, class = cls)
    expect_false(grepl("cudf|CUDF|Invalid unary operation", rlang::cnd_message(gpu)),
                 info = paste("lazy =", lazy))
  }
}

test_that("`!` on a string column raises dplyr's error, not a raw libcudf one", {
  skip_if_no_gpu()
  df <- n29_df()
  n29_expect_dplyr_error(df, function(t) mutate(t, y = !s), "invalid argument type")
  n29_expect_dplyr_error(df, function(t) transmute(t, y = !s), "invalid argument type")
  n29_expect_dplyr_error(df, function(t) filter(t, !s), "invalid argument type")
  n29_expect_dplyr_error(df, function(t) t |> group_by(i) |> mutate(y = !s),
                         "invalid argument type")
  # inside a larger expression
  n29_expect_dplyr_error(df, function(t) mutate(t, y = !s | b), "invalid argument type")
})

test_that("`!` through across() on string columns raises dplyr's error (the fuzz cases)", {
  skip_if_no_gpu()
  n29_expect_dplyr_error(n29_df()["s"], function(t) mutate(t, across(everything(), ~ !.x)),
                         "invalid argument type")
  n29_expect_dplyr_error(data.frame(a = "a"), function(t) mutate(t, across("a", ~ !.x)),
                         "invalid argument type")
})

test_that("`!` on a 0-row string column matches dplyr's logical(0) result", {
  skip_if_no_gpu()
  df <- n29_df()[0, ]
  for (pipeline in list(
    function(t) mutate(t, y = !s),
    function(t) mutate(t, across(everything(), ~ !.x))
  )) {
    expect_same_as_dplyr(df["s"], pipeline)
    expect_same_as_dplyr_lazy(df["s"], pipeline)
  }
})

test_that("`!` on a Date or POSIXct column raises dplyr's error", {
  skip_if_no_gpu()
  df <- n29_df()
  n29_expect_dplyr_error(df, function(t) mutate(t, y = !d), "not defined")
  n29_expect_dplyr_error(df, function(t) mutate(t, y = !t), "not defined")
  n29_expect_dplyr_error(df, function(t) filter(t, !d), "not defined")
})

test_that("`!` on a factor column matches dplyr's NA result and warning", {
  skip_if_no_gpu()
  df <- n29_df()
  pipeline <- function(t) mutate(t, y = !f)
  expect_warning(pipeline(tibble::as_tibble(df)), "not meaningful for factors")
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(collect(pipeline(tbl_gpu(df, lazy = lazy))), "not meaningful for factors")
  }
  suppressWarnings({
    expect_same_as_dplyr(df, pipeline)
    expect_same_as_dplyr_lazy(df, pipeline)
  })
})

test_that("`!` on logical, integer and double columns runs on the GPU and matches dplyr", {
  skip_if_no_gpu()
  df <- n29_df()[c("b", "i", "x")]
  withr::local_options(cuplyr.fallback = "error")
  for (pipeline in list(
    function(t) mutate(t, nb = !b, ni = !i, nx = !x),
    function(t) filter(t, !i),
    function(t) filter(t, !x | b),
    function(t) mutate(t, across(everything(), ~ !.x))
  )) {
    expect_same_as_dplyr(df, pipeline)
    expect_same_as_dplyr_lazy(df, pipeline)
  }
})
