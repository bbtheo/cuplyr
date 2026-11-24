# Issue #24 acceptance tests, part 1: empty mutate(), if_any()/if_all() with a
# NULL body, the vctrs "unspecified" marker, a dead rowwise aggregate's
# warning, and get() of a name that is not a column.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - mutate(.keep = "none") with no expressions keeps only the grouping
#   columns (none when ungrouped): an n x 0 table. transmute() alike.
# - if_any()/if_all() whose body is NULL (a literal or a variable holding
#   NULL) over two or more columns is an error, on zero rows too: mutate()
#   raises `dplyr:::mutate_error`, filter() an rlang error ("must be a logical
#   vector"). A NULL function anywhere in the `.fns` list, the first one
#   included, is an error ("`x_1` must be a logical vector, not `NULL`").
# - A vctrs::unspecified() column stays unspecified through every verb that
#   passes its values through: rename(), rename_with(), select(), relocate(),
#   and mutate() of other columns or a bare copy of it.
# - mutate() evaluates every expression, so a rowwise min()/max() over a row
#   with no non-missing values warns even when a later select() drops it.
# - get("thr") inside the mask reads a column named `thr` if there is one
#   (an earlier expression of the same call counts), else the environment.

p24_err <- function(expr) {
  e <- tryCatch(
    {
      r <- expr
      if (is_tbl_gpu(r)) collect(r)
      NULL
    },
    error = identity
  )
  if (is.null(e)) {
    return("no error")
  }
  list(class = class(e), parent = class(e$parent))
}

# The dplyr side evaluates these by name, as in a session with dplyr attached;
# cuplyr reads the calls by name.
if_any <- dplyr::if_any
if_all <- dplyr::if_all
if_else <- dplyr::if_else

p24_modes <- c(eager = FALSE, lazy = TRUE)

# --- item 1 -------------------------------------------------------------------

test_that("mutate(.keep = 'none') with no expressions matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), g = c(1L, 1L, 2L))
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    check(df, function(t) mutate(t, .keep = "none"))
    check(df, function(t) mutate(t, , .keep = "none"))
    check(df, function(t) mutate(group_by(t, g), .keep = "none"))
    check(df, function(t) mutate(t, .by = g, .keep = "none"))
    check(df, function(t) mutate(rowwise(t, g), .keep = "none"))
    check(df, function(t) transmute(t))
    check(df, function(t) transmute(group_by(t, g)))
  }
  for (lz in p24_modes) {
    res <- collect(mutate(tbl_gpu(df, lazy = lz), .keep = "none"))
    expect_identical(dim(res), c(3L, 0L))
  }
})

# --- item 2 -------------------------------------------------------------------

test_that("if_any()/if_all() with a NULL body over two columns is dplyr's error", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y2 = c(1, 5, 2))
  cfg <- list(trh = NULL)
  pipes <- list(
    mut_any_var = function(t) mutate(t, z = if_any(c(x, y2), ~ cfg$trh)),
    mut_all_var = function(t) mutate(t, z = if_all(c(x, y2), ~ cfg$trh)),
    mut_any_lit = function(t) mutate(t, z = if_any(c(x, y2), ~ NULL)),
    trans_any_var = function(t) transmute(t, z = if_any(c(x, y2), ~ cfg$trh)),
    mut_grouped = function(t) mutate(group_by(t, x), z = if_all(c(x, y2), ~ cfg$trh)),
    filt_any_var = function(t) filter(t, if_any(c(x, y2), ~ cfg$trh)),
    filt_all_var = function(t) filter(t, if_all(c(x, y2), ~ cfg$trh)),
    filt_all_lit = function(t) filter(t, if_all(c(x, y2), ~ NULL))
  )
  for (nm in names(pipes)) {
    for (data in list(rows = df, zero = df[0, ])) {
      want <- p24_err(pipes[[nm]](data))
      expect_false(identical(want, "no error"), info = nm)
      for (lz in p24_modes) {
        got <- p24_err(pipes[[nm]](tbl_gpu(data, lazy = lz)))
        expect_identical(got, want, info = paste(nm, nrow(data), "rows, lazy =", lz))
      }
    }
  }
})

test_that("if_any() over several columns with non-NULL bodies stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y2 = c(1, 5, 2))
  withr::local_options(cuplyr.fallback = "error")
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    check(df, function(t) mutate(t, z = if_any(c(x, y2), ~ .x > 2)))
    check(df, function(t) filter(t, if_all(c(x, y2), list(~ .x > 1, ~ .x < 6))))
  }
})

# --- item 3 -------------------------------------------------------------------

test_that("a NULL function first in an if_any()/if_all() list is dplyr's error", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y2 = c(1, 5, 2))
  pipes <- list(
    mut_lead = function(t) mutate(t, z = if_any(x, list(~ NULL, ~ .x > 1))),
    mut_lead_all = function(t) mutate(t, z = if_all(x, list(~ NULL, ~ .x > 1))),
    mut_two_lead = function(t) mutate(t, z = if_any(x, list(~ NULL, ~ NULL, ~ .x > 1))),
    mut_lead_cols = function(t) mutate(t, z = if_any(c(x, y2), list(~ NULL, ~ .x > 1))),
    mut_grouped = function(t) mutate(group_by(t, x), z = if_any(y2, list(~ NULL, ~ .x > 1))),
    filt_lead = function(t) filter(t, if_any(x, list(~ NULL, ~ .x > 1))),
    filt_lead_all = function(t) filter(t, if_all(c(x, y2), list(~ NULL, ~ .x > 1))),
    mut_trail = function(t) mutate(t, z = if_any(x, list(~ .x > 1, ~ NULL)))
  )
  for (nm in names(pipes)) {
    for (data in list(rows = df, zero = df[0, ])) {
      want <- p24_err(pipes[[nm]](data))
      expect_false(identical(want, "no error"), info = nm)
      for (lz in p24_modes) {
        got <- p24_err(pipes[[nm]](tbl_gpu(data, lazy = lz)))
        expect_identical(got, want, info = paste(nm, nrow(data), "rows, lazy =", lz))
      }
    }
  }
})

# --- item 4 -------------------------------------------------------------------

test_that("the unspecified marker survives verbs that pass the column through", {
  skip_if_no_gpu()
  du <- tibble::tibble(u = vctrs::unspecified(3), y = 1:3)
  abbr <- function(t) vapply(t, vctrs::vec_ptype_abbr, "")
  pipes <- list(
    rename = function(t) rename(t, v = u),
    rename_with = function(t) rename_with(t, toupper),
    select_rename = function(t) select(t, v = u, y),
    select = function(t) select(t, y, u),
    relocate = function(t) relocate(t, y),
    relocate_rename = function(t) relocate(t, v = u, .after = y),
    mutate_other = function(t) mutate(t, z = y + 1L),
    mutate_copy = function(t) mutate(t, w = u),
    filter_then_rename = function(t) rename(filter(t, y > 1L), v = u),
    rename_then_arrange = function(t) arrange(rename(t, v = u), desc(y)),
    rename_twice = function(t) rename(rename(t, v = u), w = v)
  )
  for (nm in names(pipes)) {
    want <- pipes[[nm]](du)
    for (lz in p24_modes) {
      got <- collect(pipes[[nm]](tbl_gpu(du, lazy = lz)))
      expect_identical(abbr(got), abbr(want), info = paste(nm, "lazy =", lz))
      expect_equal(tibble::as_tibble(got), tibble::as_tibble(want), info = paste(nm, "lazy =", lz))
    }
  }
})

# --- item 5 -------------------------------------------------------------------

test_that("a dead rowwise min()/max() under a plain select() keeps its warning", {
  skip_if_no_gpu()
  dr <- tibble::tibble(x = c(NA_real_, 1), y = c(NA_real_, 2), z = c(2, 1), k = 1:2)
  rt <- tibble::tibble(k = 1:2, v = c(3, 4))
  side <- function(t) if (is_tbl_gpu(t)) tbl_gpu(rt, lazy = identical(.subset2(t, "exec_mode"), "lazy")) else rt
  pipes <- list(
    sel_min = list(function(t) t |> rowwise() |> mutate(m = min(c(x, y), na.rm = TRUE)) |>
                     ungroup() |> select(z), "no non-missing arguments to min"),
    sel_minus = list(function(t) t |> rowwise() |> mutate(m = min(c(x, y), na.rm = TRUE)) |>
                       ungroup() |> select(-m), "no non-missing arguments to min"),
    sel_rowwise_max = list(function(t) t |> rowwise() |> mutate(m = max(c(x, y), na.rm = TRUE)) |>
                             select(z), "no non-missing arguments to max"),
    join_sel = list(function(t) t |> rowwise() |> mutate(m = min(c(x, y), na.rm = TRUE)) |>
                      ungroup() |> left_join(side(t), by = "k") |> select(z, v),
                    "no non-missing arguments to min")
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]][[1]]
    msg <- pipes[[nm]][[2]]
    want <- suppressWarnings(p(dr))
    for (lz in p24_modes) {
      expect_warning(got <- collect(p(tbl_gpu(dr, lazy = lz))), msg, info = paste(nm, "lazy =", lz))
      expect_equal(tibble::as_tibble(dplyr::ungroup(got)), tibble::as_tibble(dplyr::ungroup(want)),
                   info = paste(nm, "lazy =", lz))
    }
  }
})

# --- item 10 ------------------------------------------------------------------

test_that("get() of a name that is not a column stays on the GPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y2 = c(1, 5, 2), g = c(1L, 1L, 2L))
  thr <- 1.5
  local_thr <- function(t) {
    thr <- 2.5
    filter(t, x > get("thr"))
  }
  withr::local_options(cuplyr.fallback = "error")
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    check(df, function(t) filter(t, x > get("thr")))
    check(df, function(t) filter(t, x > get0("thr"), y2 > 1))
    check(df, function(t) mutate(t, z = x > get("thr")))
    check(df, local_thr)
  }
})

test_that("get() of a column name still reads the column, as in dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), y2 = c(1, 5, 2), g = c(1L, 1L, 2L))
  thr <- 1.5
  x <- 100
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    # a column of the input
    check(df, function(t) filter(t, y2 > get("x")))
    check(df, function(t) mutate(t, z = y2 > get("x")))
    # a column an earlier expression of the same call creates
    check(df, function(t) mutate(t, thr = x * 10, z = y2 > get("thr")))
    check(df, function(t) mutate(t, thr = x * 10, z = y2 > get0("thr")))
    # a pronoun, and a grouped filter
    check(df, function(t) filter(t, x > get(".data")$y2))
    check(df, function(t) filter(group_by(t, g), x > get("thr")))
  }
})
