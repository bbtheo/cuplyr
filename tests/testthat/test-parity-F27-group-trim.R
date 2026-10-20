# F27 acceptance tests: native group_trim() on a tbl_gpu.
#
# dplyr 1.2.1 (R/group-trim.R): an ungrouped table is returned as is; a
# grouped one has droplevels() applied to every factor GROUP column (other
# factor columns keep their levels), then is regrouped with
# group_by_at(vars, .drop = .drop), `.drop` defaulting to
# group_by_drop_default(). The regroup throws away empty groups kept by
# filter(.preserve = TRUE) or slice(.preserve = TRUE) unless an unused level
# survives a `.drop = FALSE` regroup through another group column.
#
# Every check here runs with options(cuplyr.fallback = "error"): the results
# already matched dplyr through the CPU fallback, the point is that the
# table stays on the GPU. The factor-code remap must happen on the GPU.

f27_df <- function() {
  tibble::tibble(
    f = factor(c("b", "a", "b", NA, "d", "a", "b", "d"), levels = c("e", "a", "b", "c", "d")),
    g = factor(c("x", "y", "x", "y", "x", "x", "y", "y"), levels = c("w", "x", "y", "z")),
    o = factor(c("lo", "hi", "lo", "mid", "hi", "lo", "mid", "hi"),
               levels = c("hi", "mid", "lo", "none"), ordered = TRUE),
    s = c("p", "q", "p", "q", NA, "p", "q", "p"),
    n = c(1L, 2L, 1L, 2L, 3L, NA, 2L, 3L),
    h = factor(c("u", "u", "u", "u", "u", "u", "u", "u"), levels = c("u", "unused")),
    v = c(1, 2, 3, 4, 5, 6, 7, 8)
  )
}

# Same data, group structure and .drop as dplyr, eager and lazy, with no CPU
# fallback anywhere in the pipeline. The exec mode of the input is kept.
f27_native <- function(df, pipe, label) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipe)
  expect_same_as_dplyr_lazy(df, pipe)

  oracle <- pipe(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    lab <- paste(label, if (lazy) "lazy" else "eager")
    res <- pipe(tbl_gpu(df, lazy = lazy))
    expect_true(is_tbl_gpu(res), info = lab)
    expect_identical(res$exec_mode, if (lazy) "lazy" else "eager", info = lab)
    expect_identical(dplyr::group_vars(res), dplyr::group_vars(oracle), info = lab)
    expect_equal(dplyr::group_data(res), dplyr::group_data(oracle), info = lab)
    expect_identical(dplyr::n_groups(res), dplyr::n_groups(oracle), info = lab)
    expect_identical(dplyr::group_by_drop_default(res),
                     dplyr::group_by_drop_default(oracle), info = lab)
    got <- collect(res)
    for (col in names(oracle)) {
      if (is.factor(oracle[[col]])) {
        expect_identical(levels(got[[col]]), levels(oracle[[col]]),
                         info = paste(lab, col))
        expect_identical(class(got[[col]]), class(oracle[[col]]),
                         info = paste(lab, col))
      }
    }
  }
}

test_that("F27: group_trim() after filter(.preserve = TRUE) is native", {
  skip_if_no_gpu()
  pipes <- list(
    one_factor = function(d) d |> group_by(f) |>
      filter(v %in% c(1, 2, 3, 6, 7), .preserve = TRUE) |> group_trim(),
    keeps_na_group = function(d) d |> group_by(f) |>
      filter(v %in% c(4, 5, 8), .preserve = TRUE) |> group_trim(),
    two_factors = function(d) d |> group_by(f, g) |>
      filter(v > 2, .preserve = TRUE) |> group_trim(),
    factor_and_chr = function(d) d |> group_by(s, f) |>
      filter(v < 6, .preserve = TRUE) |> group_trim(),
    integer_group_only = function(d) d |> group_by(n) |>
      filter(v > 4, .preserve = TRUE) |> group_trim(),
    ordered = function(d) d |> group_by(o) |>
      filter(v %in% c(1, 3, 4, 6, 7), .preserve = TRUE) |> group_trim(),
    all_levels_used = function(d) d |> group_by(g) |>
      filter(v > 0, .preserve = TRUE) |> group_trim()
  )
  for (nm in names(pipes)) f27_native(f27_df(), pipes[[nm]], nm)
})

test_that("F27: group_trim() drops unused levels of factor group columns only", {
  skip_if_no_gpu()
  pipes <- list(
    # h is not a group column: its unused level stays
    non_group_factor_untouched = function(d) d |> group_by(f) |> group_trim(),
    # g's unused levels w and z go even without any filter
    plain_grouped = function(d) d |> group_by(g, n) |> group_trim(),
    group_on_h = function(d) d |> group_by(h) |> group_trim(),
    after_mutate = function(d) d |> mutate(w = v * 2) |> group_by(o, f) |>
      filter(w > 6) |> group_trim()
  )
  for (nm in names(pipes)) f27_native(f27_df(), pipes[[nm]], nm)
})

test_that("F27: group_trim() .drop handling matches dplyr", {
  skip_if_no_gpu()
  pipes <- list(
    # .drop = FALSE inherited from group_by(): empty groups only from
    # combinations of the trimmed levels
    inherit_drop_false = function(d) d |> group_by(f, g, .drop = FALSE) |>
      filter(v > 2) |> group_trim(),
    explicit_drop_false = function(d) d |> group_by(f, g) |>
      filter(v > 2) |> group_trim(.drop = FALSE),
    explicit_drop_true = function(d) d |> group_by(f, g, .drop = FALSE) |>
      filter(v > 2) |> group_trim(.drop = TRUE),
    drop_false_with_chr = function(d) d |> group_by(f, s, .drop = FALSE) |>
      group_trim(),
    drop_false_single = function(d) d |> group_by(o, .drop = FALSE) |>
      filter(v %in% c(1, 3, 6)) |> group_trim()
  )
  for (nm in names(pipes)) f27_native(f27_df(), pipes[[nm]], nm)
})

test_that("F27: group_trim() drops slice(.preserve = TRUE) and filter() fixed keys", {
  skip_if_no_gpu()
  pipes <- list(
    slice_preserve = function(d) d |> group_by(n) |>
      slice(0, .preserve = TRUE) |> group_trim(),
    slice_preserve_factor = function(d) d |> group_by(f) |>
      slice(2, .preserve = TRUE) |> group_trim(),
    filter_then_summarise = function(d) d |> group_by(f, g, .drop = FALSE) |>
      filter(v %in% c(1, 3, 7), .preserve = TRUE) |> group_trim() |>
      summarise(k = dplyr::n(), m = sum(v), .groups = "drop")
  )
  for (nm in names(pipes)) f27_native(f27_df(), pipes[[nm]], nm)
})

test_that("F27: group_trim() on empty and all-NA inputs", {
  skip_if_no_gpu()
  pipes <- list(
    zero_rows = function(d) d |> group_by(f) |> filter(v > 100, .preserve = TRUE) |>
      group_trim(),
    zero_rows_drop_false = function(d) d |> group_by(f, g, .drop = FALSE) |>
      filter(v > 100) |> group_trim(),
    only_na_codes = function(d) d |> group_by(f) |> filter(v == 4) |> group_trim()
  )
  for (nm in names(pipes)) f27_native(f27_df(), pipes[[nm]], nm)

  # a zero-row table straight from upload
  f27_native(f27_df()[0, ], function(d) d |> group_by(f, g) |> group_trim(), "upload_0")
})

test_that("F27: group_trim() on an ungrouped tbl_gpu is the identity", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  f27_native(f27_df(), function(d) group_trim(d), "ungrouped")
  f27_native(f27_df(), function(d) d |> filter(v > 3) |> group_trim(.drop = FALSE),
             "ungrouped_lazy_ops")

  # ungrouped input: no GPU work, the same table comes back
  gt <- tbl_gpu(f27_df())
  out <- group_trim(gt)
  expect_identical(out$ptr, gt$ptr)
  expect_identical(out$schema, gt$schema)
  expect_identical(levels(collect(out)$f), c("e", "a", "b", "c", "d"))
})

test_that("F27: group_trim() keeps attributes and class like dplyr", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- as.data.frame(f27_df())
  attr(df, "myattr") <- "hi"

  for (lazy in c(FALSE, TRUE)) {
    # ungrouped: the identity keeps a bare data.frame and its attributes
    got <- collect(group_trim(tbl_gpu(df, lazy = lazy)))
    exp <- group_trim(df)
    expect_identical(class(got), class(exp))
    expect_identical(attr(got, "myattr"), attr(exp, "myattr"))

    # grouped: rebuilt as a grouped tibble, like group_by() after ungroup()
    got <- collect(group_trim(group_by(tbl_gpu(df, lazy = lazy), f)))
    exp <- group_trim(group_by(df, f))
    expect_identical(class(got), class(exp))
    expect_identical(attr(got, "myattr"), attr(exp, "myattr"))
    expect_identical(levels(got$f), levels(exp$f))
  }
})

test_that("F27: group_trim() rejects a .drop that is not TRUE/FALSE", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(f27_df()) |> group_by(f)
  # dplyr fails inside its regroup (`if (.drop)`); any error is fine, a
  # silently accepted value is not
  expect_error(collect(group_trim(gt, .drop = NA)))
  expect_error(collect(group_trim(gt, .drop = "x")))
  expect_error(collect(group_trim(gt, .drop = c(TRUE, FALSE))))
})

test_that("F27: the factor codes are remapped on the GPU (no host round trip)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  # the new levels are the used ones in original order, and every row keeps
  # its label, also where codes shift by more than one position
  df <- tibble::tibble(
    f = factor(c("l9", "l2", "l9", "l5", NA, "l2"), levels = paste0("l", 0:9)),
    v = 1:6
  )
  pipe <- function(d) d |> group_by(f) |> filter(v != 4L, .preserve = TRUE) |>
    group_trim()
  f27_native(df, pipe, "code_remap")

  # later GPU verbs see the remapped codes and the trimmed levels
  pipe2 <- function(d) d |> group_by(f) |> filter(v != 4L, .preserve = TRUE) |>
    group_trim() |> summarise(k = dplyr::n(), m = max(v))
  f27_native(df, pipe2, "codes_seen_by_summarise")
  pipe3 <- function(d) d |> group_by(f) |> filter(v != 4L, .preserve = TRUE) |>
    group_trim() |> ungroup() |> arrange(f, v)
  f27_native(df, pipe3, "codes_seen_by_arrange")
})
