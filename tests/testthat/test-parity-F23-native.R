# F23 acceptance tests, tier B (native paths). Results already match dplyr
# through the CPU fallback; these tests also require that the listed shapes
# run without a CPU fallback (`options(cuplyr.fallback = "error")`).
#
# Native shapes in scope:
# - summarise(): a dot that is exactly `cur_group_id()`. The summarise output
#   rows are in group_data() order (sorted keys for group_by(), empty groups
#   included under .drop = FALSE; first appearance for .by), so the id is the
#   output row number.
# - mutate()/filter(): `cur_group()$key` / `cur_group()[["key"]]` for a group
#   column `key` (no earlier dot of the same call writes a group column).
# - mutate()/filter()/summarise(): `length(cur_group_rows())`, which is n().
# Everything else stays on the CPU and must still match dplyr.

f23n_df <- function() {
  tibble::tibble(
    g = c("b", "a", "b", "c", NA, "a"),
    n = c(2, 1, 2, 3, 1, NA),
    x = c(1L, 2L, 3L, 4L, 5L, 6L),
    f = factor(c("v", "w", "v", "w", "v", "w"), levels = c("u", "v", "w"))
  )
}

f23n_both <- function(df, p, ...) {
  expect_same_as_dplyr(df, p, ...)
  expect_same_as_dplyr_lazy(df, p, ...)
}

# Same result as dplyr AND no CPU fallback, eager and lazy.
f23n_native <- function(df, p, label) {
  f23n_both(df, p, label = label)
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    lab <- paste(label, if (lazy) "lazy" else "eager")
    expect_no_error(collect(p(tbl_gpu(df, lazy = lazy))))
  }
}

test_that("F23 B4: summarise(dplyr::cur_group_id()) is native and matches dplyr", {
  skip_if_no_gpu()
  pipes <- list(
    grouped = function(d) d |> group_by(g) |> summarise(id = dplyr::cur_group_id(), s = sum(x)),
    id_last = function(d) d |> group_by(g) |> summarise(s = sum(x), id = dplyr::cur_group_id()),
    only_id = function(d) d |> group_by(n) |> summarise(id = dplyr::cur_group_id()),
    two_keys = function(d) d |> group_by(g, n) |> summarise(id = dplyr::cur_group_id(), .groups = "drop"),
    two_keys_peel = function(d) d |> group_by(g, n) |> summarise(id = dplyr::cur_group_id(), m = max(x)),
    drop_false = function(d) d |> group_by(f, .drop = FALSE) |> summarise(id = dplyr::cur_group_id(), k = dplyr::n()),
    by = function(d) d |> summarise(id = dplyr::cur_group_id(), m = mean(x), .by = g),
    # min() of a double column: lazy `.by` with an integer min()/max() falls
    # back for an unrelated, pre-existing reason (result type is data-dependent)
    by_two = function(d) d |> summarise(m = min(n), id = dplyr::cur_group_id(), .by = c(n, g)),
    ungrouped = function(d) d |> summarise(id = dplyr::cur_group_id(), s = sum(x)),
    prior_ops = function(d) d |> filter(x > 1L) |> group_by(g) |> summarise(id = dplyr::cur_group_id()),
    zero_rows = function(d) d |> filter(x > 100L) |> group_by(g) |> summarise(id = dplyr::cur_group_id(), s = sum(x))
  )
  for (nm in names(pipes)) {
    f23n_native(f23n_df(), pipes[[nm]], nm)
  }
})

test_that("F23 B5: dplyr::cur_group()$key and length(dplyr::cur_group_rows()) are native", {
  skip_if_no_gpu()
  pipes <- list(
    mutate_dollar = function(d) d |> group_by(g) |> mutate(k = dplyr::cur_group()$g),
    mutate_bracket = function(d) d |> group_by(g, n) |> mutate(k = dplyr::cur_group()[["n"]] * 2),
    mutate_by = function(d) d |> mutate(k = dplyr::cur_group()$g, .by = g),
    filter_dollar = function(d) d |> group_by(g) |> filter(dplyr::cur_group()$g == "b"),
    mutate_rows_len = function(d) d |> group_by(g) |> mutate(k = length(dplyr::cur_group_rows())),
    summarise_rows_len = function(d) d |> group_by(g) |> summarise(k = length(dplyr::cur_group_rows())),
    filter_rows_len = function(d) d |> group_by(g) |> filter(length(dplyr::cur_group_rows()) > 1L)
  )
  for (nm in names(pipes)) {
    f23n_native(f23n_df(), pipes[[nm]], nm)
  }
})

test_that("F23 B: shapes that need a list or data frame still match dplyr via the CPU", {
  skip_if_no_gpu()
  pipes <- list(
    rows = function(d) d |> group_by(g) |> mutate(r = dplyr::cur_group_rows()),
    rows_sum = function(d) d |> group_by(g) |> summarise(r = sum(dplyr::cur_group_rows())),
    cur_group_nrow = function(d) d |> group_by(g) |> mutate(k = nrow(dplyr::cur_group())),
    id_arith_summarise = function(d) d |> group_by(g) |> summarise(id = dplyr::cur_group_id() * 10L + sum(x)),
    # writes the group column first: cur_group() must still give the old key
    overwrite_key = function(d) d |> group_by(g) |> mutate(g = "z", k = dplyr::cur_group()$g),
    later_ref = function(d) d |> group_by(g) |> summarise(id = dplyr::cur_group_id(), id2 = id * 2L),
    # unnamed non-id dots next to a cur_group_id() dot (round-1 review B1)
    unnamed_other = function(d) d |> group_by(g) |> summarise(id = dplyr::cur_group_id(), mean(x)),
    unnamed_other_by = function(d) d |> summarise(id = dplyr::cur_group_id(), mean(x), .by = g),
    unnamed_across = function(d) d |> group_by(g) |> summarise(id = dplyr::cur_group_id(), dplyr::across(x, mean))
  )
  for (nm in names(pipes)) {
    f23n_both(f23n_df(), pipes[[nm]], label = nm)
  }
})

test_that("F23 B: dplyr::cur_data()/dplyr::cur_data_all() keep dplyr's deprecation warning", {
  skip_if_no_gpu()
  df <- f23n_df()
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(
      res <- tbl_gpu(df, lazy = lazy) |> group_by(g) |> mutate(k = nrow(dplyr::cur_data())) |> collect(),
      "cur_data"
    )
    expect_equal(res$k, c(2L, 2L, 2L, 1L, 1L, 2L))
    expect_warning(
      res <- tbl_gpu(df, lazy = lazy) |> group_by(g) |> summarise(k = ncol(dplyr::cur_data_all())) |> collect(),
      "cur_data_all"
    )
    expect_equal(res$k, rep(4L, 4L))
  }
})
