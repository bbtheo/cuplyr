# F23 acceptance tests, tier A (correctness): cur_group_id() in mutate() and
# filter() and the if_any()/if_all() logical-type error.
#
# dplyr 1.2.1 reference (R/context.R, R/by.R, R/grouped-df.R):
# - cur_group_id() is the row of group_data() the current group sits on. For
#   group_by() that is sorted key order and, with `.drop = FALSE`, it counts
#   the empty groups too. For `.by` the groups come from vec_group_loc(), so
#   ids follow FIRST APPEARANCE in the data.
# - Using cur_group_id() never changes the grouping (`.drop` included).
# - if_any()/if_all() over a non-logical result raise dplyr's own error:
#   mutate() -> class `dplyr:::mutate_error`, filter() -> `rlang_error`, both
#   with "In argument: ..." and a parent "`x` must be a logical vector".

f23_df <- function() {
  tibble::tibble(
    g = c("b", "a", "b", "c", NA, "a"),
    n = c(2, 1, 2, 3, 1, NA),
    x = c(1L, 2L, 3L, 4L, 5L, 6L),
    y = c(TRUE, FALSE, NA, TRUE, TRUE, FALSE),
    # level "u" is never used: with .drop = FALSE it is group 1
    f = factor(c("v", "w", "v", "w", "v", "w"), levels = c("u", "v", "w"))
  )
}

f23_both <- function(df, p, ...) {
  expect_same_as_dplyr(df, p, ...)
  expect_same_as_dplyr_lazy(df, p, ...)
}

# ---------------------------------------------------------------------------
# A1: lazy grouped cur_group_id() (the F38 NULL-pointer crash). Regression
# cover: must keep working eager and lazy.
# ---------------------------------------------------------------------------

f23_grouped_pipes <- list(
  mutate = function(d) d |> group_by(g) |> mutate(id = dplyr::cur_group_id()),
  mutate_num_key = function(d) d |> group_by(n) |> mutate(id = dplyr::cur_group_id()),
  mutate_arith = function(d) d |> group_by(g) |> mutate(id = dplyr::cur_group_id() * 10L + x),
  mutate_two_keys = function(d) d |> group_by(g, n) |> mutate(id = dplyr::cur_group_id()),
  prior_ops = function(d) d |> filter(x > 1L) |> mutate(z = x * 2L) |> group_by(g) |> mutate(id = dplyr::cur_group_id()),
  then_filter = function(d) d |> group_by(g) |> mutate(id = dplyr::cur_group_id()) |> filter(id >= 2L),
  filter = function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() == 2L),
  filter_and = function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() >= 2L, x > 2L),
  transmute = function(d) d |> group_by(g) |> transmute(id = dplyr::cur_group_id()),
  ungrouped = function(d) d |> mutate(id = dplyr::cur_group_id()),
  zero_rows = function(d) d |> filter(x > 100L) |> group_by(g) |> mutate(id = dplyr::cur_group_id())
)

test_that("F23 A1: grouped dplyr::cur_group_id() matches dplyr, eager and lazy", {
  skip_if_no_gpu()
  for (nm in names(f23_grouped_pipes)) {
    f23_both(f23_df(), f23_grouped_pipes[[nm]], label = nm)
  }
})

test_that("F23 A1: lazy grouped dplyr::cur_group_id() stays lazy-safe across compute()", {
  skip_if_no_gpu()
  df <- f23_df()
  lz <- tbl_gpu(df, lazy = TRUE) |> mutate(z = x + 1L) |> group_by(g) |>
    mutate(id = dplyr::cur_group_id())
  out <- compute(lz) |> mutate(id2 = id * 2L) |> collect()
  ref <- df |> mutate(z = x + 1L) |> group_by(g) |> mutate(id = dplyr::cur_group_id()) |>
    mutate(id2 = id * 2L)
  expect_equal(tibble::as_tibble(dplyr::ungroup(out)), tibble::as_tibble(dplyr::ungroup(ref)))
  expect_identical(dplyr::group_vars(out), "g")
})

# ---------------------------------------------------------------------------
# A2: `.drop = FALSE`. Ids run over group_data() (empty groups included), and
# the `.drop` flag survives the verb that used cur_group_id().
# ---------------------------------------------------------------------------

f23_drop_pipes <- list(
  mutate = function(d) d |> group_by(f, .drop = FALSE) |> mutate(id = dplyr::cur_group_id()),
  filter = function(d) d |> group_by(f, .drop = FALSE) |> filter(dplyr::cur_group_id() == 2L),
  transmute = function(d) d |> group_by(f, .drop = FALSE) |> transmute(id = dplyr::cur_group_id()),
  two_keys = function(d) d |> group_by(f, g, .drop = FALSE) |> mutate(id = dplyr::cur_group_id()),
  then_summarise = function(d) d |> group_by(f, .drop = FALSE) |> mutate(id = dplyr::cur_group_id()) |>
    summarise(n = dplyr::n(), m = sum(id)),
  filter_then_summarise = function(d) d |> group_by(f, .drop = FALSE) |> filter(dplyr::cur_group_id() >= 1L) |>
    summarise(n = dplyr::n()),
  then_count = function(d) d |> group_by(f, .drop = FALSE) |> mutate(id = dplyr::cur_group_id()) |> count()
)

test_that("F23 A2: dplyr::cur_group_id() under .drop = FALSE counts empty groups (eager and lazy)", {
  skip_if_no_gpu()
  for (nm in names(f23_drop_pipes)) {
    f23_both(f23_df(), f23_drop_pipes[[nm]], label = nm)
  }
})

test_that("F23 A2: dplyr::cur_group_id() keeps .drop = FALSE and the group data", {
  skip_if_no_gpu()
  df <- f23_df()
  keep <- list(
    mutate = function(d) d |> group_by(f, .drop = FALSE) |> mutate(id = dplyr::cur_group_id()),
    filter = function(d) d |> group_by(f, .drop = FALSE) |> filter(dplyr::cur_group_id() >= 2L),
    mutate_chr_key = function(d) d |> group_by(g, .drop = FALSE) |> mutate(id = dplyr::cur_group_id())
  )
  for (nm in names(keep)) {
    ref <- keep[[nm]](df)
    for (lazy in c(FALSE, TRUE)) {
      lab <- paste(nm, if (lazy) "lazy" else "eager")
      res <- keep[[nm]](tbl_gpu(df, lazy = lazy))
      expect_identical(dplyr::group_by_drop_default(res), FALSE, label = lab)
      expect_identical(dplyr::n_groups(res), dplyr::n_groups(ref), label = lab)
      got <- collect(res)
      expect_identical(dplyr::group_by_drop_default(got), FALSE, label = lab)
      expect_equal(
        tibble::as_tibble(dplyr::group_data(got)),
        tibble::as_tibble(dplyr::group_data(ref)),
        label = lab
      )
    }
  }
})

test_that("F23 A2: .drop = TRUE stays TRUE after dplyr::cur_group_id()", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    res <- tbl_gpu(f23_df(), lazy = lazy) |> group_by(f) |> mutate(id = dplyr::cur_group_id())
    expect_true(dplyr::group_by_drop_default(res))
    expect_identical(dplyr::n_groups(res), 2L)
  }
})

test_that("F23 A2: dplyr::cur_group_id() after filter(.preserve = TRUE) numbers the kept groups", {
  skip_if_no_gpu()
  p <- function(d) d |> group_by(g) |> filter(x > 3L, .preserve = TRUE) |> mutate(id = dplyr::cur_group_id())
  f23_both(f23_df(), p)
  p2 <- function(d) d |> group_by(g) |> filter(x > 3L, .preserve = TRUE) |>
    mutate(id = dplyr::cur_group_id()) |> summarise(n = dplyr::n())
  f23_both(f23_df(), p2)
})

# ---------------------------------------------------------------------------
# A4: `.by` numbers groups in order of first appearance (vec_group_loc()),
# not sorted key order.
# ---------------------------------------------------------------------------

f23_by_pipes <- list(
  mutate = function(d) d |> mutate(id = dplyr::cur_group_id(), .by = g),
  mutate_num = function(d) d |> mutate(id = dplyr::cur_group_id(), .by = n),
  mutate_two = function(d) d |> mutate(id = dplyr::cur_group_id(), .by = c(n, g)),
  mutate_factor = function(d) d |> mutate(id = dplyr::cur_group_id(), .by = f),
  filter = function(d) d |> filter(dplyr::cur_group_id() == 1L, .by = g),
  filter_two = function(d) d |> filter(dplyr::cur_group_id() <= 2L, x > 1L, .by = g),
  transmute = function(d) d |> transmute(id = dplyr::cur_group_id(), .by = g),
  prior_ops = function(d) d |> filter(x > 1L) |> mutate(z = x * 2L) |> mutate(id = dplyr::cur_group_id(), .by = g),
  zero_rows = function(d) d |> filter(x > 100L) |> mutate(id = dplyr::cur_group_id(), .by = g)
)

test_that("F23 A4: dplyr::cur_group_id() with .by follows first appearance (eager and lazy)", {
  skip_if_no_gpu()
  for (nm in names(f23_by_pipes)) {
    f23_both(f23_df(), f23_by_pipes[[nm]], label = nm)
  }
})

# ---------------------------------------------------------------------------
# A3: if_any()/if_all() over a non-logical column or function result raise
# dplyr's own error (class and message chain), eager and lazy.
# ---------------------------------------------------------------------------

f23_capture <- function(expr) {
  tryCatch({ force(expr); NULL }, error = function(e) e)
}

f23_expect_same_error <- function(df, p, label) {
  ref <- f23_capture(p(df))
  expect_true(inherits(ref, "error"), label = paste(label, "(dplyr errors)"))
  for (lazy in c(FALSE, TRUE)) {
    lab <- paste(label, if (lazy) "lazy" else "eager")
    got <- f23_capture(collect(p(tbl_gpu(df, lazy = lazy))))
    expect_true(inherits(got, "error"), label = lab)
    if (!inherits(got, "error")) next
    expect_identical(class(got), class(ref), label = lab)
    expect_identical(conditionMessage(got), conditionMessage(ref), label = lab)
    expect_identical(
      conditionMessage(got$parent %||% got),
      conditionMessage(ref$parent %||% ref),
      label = paste(lab, "parent")
    )
  }
}

f23_err_pipes <- list(
  mutate_if_any_bare = function(d) d |> mutate(a = dplyr::if_any(x)),
  mutate_if_any_identity = function(d) d |> mutate(a = dplyr::if_any(x, identity)),
  mutate_if_all_lambda = function(d) d |> mutate(a = dplyr::if_all(c(y, x), ~ .x)),
  mutate_if_any_double = function(d) d |> mutate(a = dplyr::if_any(c(x, n), ~ .x * 2)),
  mutate_grouped = function(d) d |> group_by(g) |> mutate(a = dplyr::if_all(x, ~ .x)),
  mutate_by = function(d) d |> mutate(a = dplyr::if_any(x, ~ .x), .by = g),
  transmute_if_any = function(d) d |> transmute(a = dplyr::if_any(x, identity)),
  mutate_fallback_fn = function(d) d |> mutate(a = dplyr::if_any(c(x, n), ~ scale(.x)[, 1])),
  filter_if_any_bare = function(d) d |> filter(dplyr::if_any(x)),
  filter_if_all_identity = function(d) d |> filter(dplyr::if_all(c(y, x), identity)),
  filter_wrapped = function(d) d |> filter((dplyr::if_any(x, ~ .x))),
  filter_grouped = function(d) d |> group_by(g) |> filter(dplyr::if_any(x, ~ .x))
)

test_that("F23 A3: dplyr::if_any()/dplyr::if_all() non-logical errors match dplyr's", {
  skip_if_no_gpu()
  for (nm in names(f23_err_pipes)) {
    f23_expect_same_error(f23_df(), f23_err_pipes[[nm]], nm)
  }
})

test_that("F23 A3: logical dplyr::if_any()/dplyr::if_all() still work (no false positives)", {
  skip_if_no_gpu()
  ok <- list(
    mutate_logical = function(d) d |> mutate(a = dplyr::if_any(c(y), ~ .x)),
    mutate_pred = function(d) d |> mutate(a = dplyr::if_all(c(x, n), ~ .x > 1)),
    filter_pred = function(d) d |> filter(dplyr::if_any(c(x, n), ~ .x > 2)),
    with_cgid = function(d) d |> group_by(g) |> filter(dplyr::if_all(c(x, n), ~ .x > 0 & dplyr::cur_group_id() >= 1L)),
    mutate_cgid_by = function(d) d |> mutate(a = dplyr::if_any(c(x, n), ~ .x > dplyr::cur_group_id()), .by = g)
  )
  for (nm in names(ok)) {
    f23_both(f23_df(), ok[[nm]], label = nm)
  }
})
