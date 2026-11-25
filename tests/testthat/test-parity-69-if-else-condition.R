# #69: the if_else() condition rules around the size-1 check. A condition that
# is not logical goes to dplyr (its error); a bare NA condition is a logical NA
# on the GPU; a logical condition built by any op stays native, and so do the
# internal if_else() desugars (cumall(), consecutive_id(), ntile(), slice_*()).
# On a rowwise table with rows each group is one row, so the size-1 check never
# declines there, and that scope ends with the rowwise call; on zero rows, or
# rows only the pending steps can count, it declines and dplyr decides.

for (p69c_fn in c("if_else", "between", "consecutive_id", "ntile", "cumall")) {
  assign(p69c_fn, getExportedValue("dplyr", p69c_fn))
}
rm(p69c_fn)

p69c_df <- function() {
  data.frame(
    g = c(1, 1, 2),
    x = c(3, 0, 2),
    y = 4:6,
    i = c(1L, 0L, NA),
    b = c(TRUE, FALSE, NA),
    s = c("b", "a", "c"),
    f = factor(c("b", "a", "c")),
    stringsAsFactors = FALSE
  )
}

p69c_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("a logical condition built by any op stays native", {
  skip_if_no_gpu()
  df <- p69c_df()
  p69c_native(df, function(d) mutate(d, z = if_else(!b, x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(!x, x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(b & x > 1, x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(xor(b, x > 1), x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(is.na(i), 0L, i)))
  p69c_native(df, function(d) mutate(d, z = if_else(f == "a", x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(f %in% c("a", "b"), x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(grepl("a", s), x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(between(x, 1, 3), x, 0)))
  p69c_native(df, function(d) mutate(d, z = if_else(x > 1, if_else(b, 1, 2), 3)))
  p69c_native(df, function(d) filter(d, if_else(is.na(b), TRUE, b)))
})

test_that("the internal if_else() desugars stay native", {
  skip_if_no_gpu()
  df <- p69c_df()
  p69c_native(df, function(d) mutate(d, z = cumall(x > 1)))
  p69c_native(df, function(d) mutate(d, z = consecutive_id(g)))
  p69c_native(df, function(d) mutate(d, z = ntile(x, 2)))
  # (the grouped slice_*() desugars: a numeric slice_max() takes its own GPU
  # path unless the option says "sort")
  withr::local_options(cuplyr.topk_path = "sort")
  p69c_native(df, function(d) d |> group_by(g) |> slice_max(x, n = 1), arrange_by = "g")
  p69c_native(df, function(d) d |> group_by(g) |> slice_max(x, prop = 0.5), arrange_by = "g")
  p69c_native(df, function(d) d |> group_by(g) |> slice_min(s, n = -1), arrange_by = "g")
  p69c_native(df, function(d) d |> group_by(g) |> slice_head(prop = 0.5), arrange_by = "g")
})

test_that("a typed NA condition and an NA condition in summarise() give dplyr's value", {
  skip_if_no_gpu()
  df <- p69c_df()
  p69c_native(df, function(d) mutate(d, z = if_else(NA, 1, 0) + x))
  p69c_native(df, function(d) mutate(d, z = if_else(NA, mean(x), 0, missing = -1)))
  expect_same_as_dplyr(df, function(d) d |> group_by(g) |> summarise(m = if_else(NA, mean(x), 0)),
                       arrange_by = "g")
})

test_that("a non-logical one-value condition is dplyr's error, whatever its type", {
  skip_if_no_gpu()
  df <- p69c_df()
  for (lazy in c(FALSE, TRUE)) {
    cpu <- tryCatch(mutate(tibble::as_tibble(df), z = if_else(1L, mean(x), 0)), error = function(e) e)
    gpu <- tryCatch(collect(mutate(tbl_gpu(df, lazy = lazy), z = if_else(1L, mean(x), 0))),
                    error = function(e) e)
    expect_s3_class(cpu, "dplyr:::mutate_error")
    expect_identical(class(gpu), class(cpu))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu))
    expect_identical(conditionMessage(gpu$parent), conditionMessage(cpu$parent))
  }
  # the notice gives the type as the reason, not the size
  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(try(collect(mutate(tbl_gpu(df), z = if_else(1, x, y))), silent = TRUE),
                 "condition of type FLOAT64")
})

test_that("rowwise verbs keep a one-value condition native, and the scope ends with them", {
  skip_if_no_gpu()
  df <- p69c_df()
  p69c_native(df, function(d) d |> rowwise() |> filter(if_else(TRUE, x > 1, FALSE)))
  p69c_native(df, function(d) d |> rowwise() |> mutate(z = if_else(TRUE, x, 0)))
  # a non-logical condition is still dplyr's error on a rowwise table
  cpu <- tryCatch(rowwise(tibble::as_tibble(df)) |> mutate(z = if_else(1, x, y)), error = function(e) e)
  gpu <- tryCatch(collect(tbl_gpu(df) |> rowwise() |> mutate(z = if_else(1, x, y))), error = function(e) e)
  expect_s3_class(cpu, "error")
  expect_identical(class(gpu), class(cpu))
  expect_identical(conditionMessage(gpu), conditionMessage(cpu))
  # after a rowwise call (one that ran, one that errored), an ordinary table
  # declines again: dplyr's recycling error
  tbl_gpu(df) |> rowwise() |> mutate(z = if_else(TRUE, x, y)) |> collect()
  expect_identical(rowwise_state$depth, 0L)
  expect_identical(rowwise_state$rewriting, 0L)
  cpu <- tryCatch(mutate(tibble::as_tibble(df), z = if_else(TRUE, x, y)), error = function(e) e)
  gpu <- tryCatch(collect(mutate(tbl_gpu(df), z = if_else(TRUE, x, y))), error = function(e) e)
  expect_s3_class(cpu$parent, "vctrs_error_incompatible_size")
  expect_identical(class(gpu), class(cpu))
  expect_identical(conditionMessage(gpu$parent), conditionMessage(cpu$parent))
})

p69c_same_error <- function(df, pipeline) {
  run <- function(x) {
    tryCatch({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- collect(out)
      NULL
    }, error = function(e) e)
  }
  cpu <- run(tibble::as_tibble(df))
  expect_s3_class(cpu, "error")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- run(tbl_gpu(df, lazy = lazy))
    expect_s3_class(gpu, "error")
    expect_identical(class(gpu), class(cpu))
    expect_identical(rlang::format_error_call(gpu$call), rlang::format_error_call(cpu$call))
    expect_identical(class(gpu$parent), class(cpu$parent))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu))
  }
}

test_that("every rowwise verb on zero rows gives dplyr's recycling error", {
  skip_if_no_gpu()
  df <- p69c_df()
  d0 <- df[0, , drop = FALSE]
  p69c_same_error(d0, function(d) d |> rowwise() |> filter(if_else(TRUE, x > 1, FALSE)))
  p69c_same_error(d0, function(d) d |> rowwise() |> summarise(z = if_else(NA, x, y)))
  p69c_same_error(d0, function(d) d |> rowwise(g) |> mutate(z = if_else(TRUE, x, y)))
  p69c_same_error(df, function(d) d |> filter(x > 100) |> rowwise() |> filter(if_else(TRUE, x > 1, FALSE)))
  p69c_same_error(df, function(d) d |> filter(x > 100) |> rowwise() |> summarise(z = if_else(NA, x, y)))
  # on rows, the same calls stay native
  p69c_native(df, function(d) d |> rowwise() |> summarise(z = if_else(NA, x, y)))
  p69c_native(df, function(d) d |> rowwise() |> transmute(z = if_else(TRUE, x, y)))
  p69c_native(df, function(d) d |> rowwise() |> filter(if_else(NA, x > 1, TRUE)))
})
