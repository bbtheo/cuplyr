# Acceptance tests for #20: count()/tally()/add_count()/add_tally() check
# `name` the way dplyr's check_n_name() does (a single string), and raise
# dplyr's own error (same class and message) at the verb call, eager and lazy.
# Before the fix a non-string `name` reached `:=` and failed with rlang's
# "The LHS of `:=` must be a string" instead.

p20_catch <- function(expr) {
  tryCatch({
    force(expr)
    NULL
  }, error = function(e) e)
}

# `pipeline` must raise the same error on the GPU as dplyr does on `df`.
# The error is expected at the verb call, so the GPU result is not collected.
p20_expect_same_error <- function(df, pipeline) {
  want <- p20_catch(pipeline(tibble::as_tibble(df)))
  if (is.null(want)) {
    stop("dplyr did not error; the test is wrong")
  }
  for (lazy in c(FALSE, TRUE)) {
    got <- p20_catch(pipeline(tbl_gpu(df, lazy = lazy)))
    label <- if (lazy) "lazy" else "eager"
    expect_s3_class(got, "error")
    if (is.null(got)) next
    expect_identical(class(got), class(want), label = paste(label, "error class"))
    expect_identical(conditionMessage(got), conditionMessage(want),
                     label = paste(label, "error message"))
  }
}

p20_df <- data.frame(x = c(1, 2, 2, NA), g = c(1L, 1L, 2L, 2L), w = c(1, 2, 3, 4))

test_that("count() rejects a non-string name like dplyr", {
  skip_if_no_gpu()
  p20_expect_same_error(p20_df, function(t) count(t, x, name = 1))
  p20_expect_same_error(p20_df, function(t) count(t, x, name = letters))
  p20_expect_same_error(p20_df, function(t) count(t, x, name = NA_character_))
  p20_expect_same_error(p20_df, function(t) count(t, x, name = TRUE))
  p20_expect_same_error(p20_df, function(t) count(t, x, name = character()))
  p20_expect_same_error(p20_df, function(t) count(t, name = 1))
  p20_expect_same_error(p20_df, function(t) count(t, x, wt = w, name = 2L))
  p20_expect_same_error(p20_df, function(t) count(t, x, sort = TRUE, name = c("a", "b")))
})

test_that("count() on a grouped table rejects a non-string name like dplyr", {
  skip_if_no_gpu()
  p20_expect_same_error(p20_df, function(t) count(group_by(t, g), x, name = 1))
  p20_expect_same_error(p20_df, function(t) count(group_by(t, g), name = letters))
})

test_that("tally() rejects a non-string name like dplyr", {
  skip_if_no_gpu()
  p20_expect_same_error(p20_df, function(t) tally(t, name = 1))
  p20_expect_same_error(p20_df, function(t) tally(group_by(t, g), name = letters))
  p20_expect_same_error(p20_df, function(t) tally(t, wt = w, name = NA_character_))
})

test_that("add_count() rejects a non-string name like dplyr", {
  skip_if_no_gpu()
  p20_expect_same_error(p20_df, function(t) add_count(t, x, name = 1))
  p20_expect_same_error(p20_df, function(t) add_count(t, x, name = letters))
  p20_expect_same_error(p20_df, function(t) add_count(group_by(t, g), name = TRUE))
})

test_that("add_tally() rejects a non-string name like dplyr", {
  skip_if_no_gpu()
  p20_expect_same_error(p20_df, function(t) add_tally(t, name = 1))
  p20_expect_same_error(p20_df, function(t) add_tally(group_by(t, g), name = c("a", "b")))
})

test_that("a valid name still works, eager and lazy", {
  skip_if_no_gpu()
  for (pipeline in list(
    function(t) count(t, x, name = "cnt"),
    function(t) count(t, x, wt = w, sort = TRUE, name = "total"),
    function(t) count(group_by(t, g), x, name = "cnt"),
    function(t) tally(group_by(t, g), name = "cnt"),
    function(t) add_count(t, g, name = "cnt"),
    function(t) add_tally(group_by(t, g), name = "cnt"),
    function(t) count(t, x)
  )) {
    expect_same_as_dplyr(p20_df, pipeline)
    expect_same_as_dplyr_lazy(p20_df, pipeline)
  }
})

test_that("a non-string name on a zero-row table errors like dplyr", {
  skip_if_no_gpu()
  empty <- p20_df[0, ]
  p20_expect_same_error(empty, function(t) count(t, x, name = 1))
  p20_expect_same_error(empty, function(t) tally(t, name = letters))
  p20_expect_same_error(empty, function(t) add_count(t, x, name = NA_character_))
})
