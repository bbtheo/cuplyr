# F13 acceptance tests: group_by()/ungroup() semantics.
#
# Behaviour pinned with tools/r against dplyr 1.2.1
# (scratchpad/parity/features/F13/probe*.R):
# - group_by()'s unnamed dots are data-masking, not tidyselect. Only a
#   variable reference (`x`, `.data$x`, `.data[["x"]]`) and a top-level pick()
#   name existing columns; anything else is computed (on the ungrouped data)
#   into a new column named by as_label() and grouped by.
# - ungroup(<grouped>, ...) tidyselects columns and removes those that are
#   groups (others are ignored); unknown columns and renames error; ungroup()
#   of an ungrouped table with dots errors (`...` must be empty).
# - group_data() keys: logical FALSE < TRUE < NA; ordered factors stay
#   ordered; .drop = FALSE expands factor levels within each parent group.
# - count()/tally() on .drop = FALSE groups add zero-count rows for empty
#   factor levels, in group order.
# - group_by(add =) / group_by(.dots =) are defunct errors.

f13_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f13_modes <- function(df) {
  list(eager = tbl_gpu(df), lazy = tbl_gpu(df, lazy = TRUE))
}

f13_df <- data.frame(
  g1 = c(2L, 1L, 2L, 1L),
  a = c(1, 2, 3, 2),
  b = c("x", "y", "z", "y")
)

f13_ignore <- function(x) 1

# --- 1. unnamed computed dots are data-masking ------------------------------

test_that("unnamed computed group_by() dots create a column, not a selection", {
  skip_if_no_gpu()
  f13_both(f13_df, function(d) group_by(d, a + 1))
  f13_both(f13_df, function(d) group_by(d, g1 + 1))
  f13_both(f13_df, function(d) group_by(d, 1 + 1))
  f13_both(f13_df, function(d) group_by(d, 2))
  f13_both(f13_df, function(d) group_by(d, "g1"))
  f13_both(f13_df, function(d) group_by(d, -a))
  f13_both(f13_df, function(d) group_by(d, desc(a)))
  f13_both(f13_df, function(d) group_by(d, is.na(a)))
})

test_that("unnamed computed keys aggregate like dplyr", {
  skip_if_no_gpu()
  f13_both(f13_df, function(d) group_by(d, a * 2) |> summarise(n = n()),
           arrange_by = "a * 2")
  f13_both(f13_df, function(d) group_by(d, g1 + 1) |> summarise(s = sum(a)),
           arrange_by = "g1 + 1")
  f13_both(f13_df, function(d) count(group_by(d, a > 1)), arrange_by = "a > 1")
})

test_that("named and unnamed dots interleave in argument order", {
  skip_if_no_gpu()
  f13_both(f13_df, function(d) group_by(d, h = a * 2, h + 1, b))
  f13_both(f13_df, function(d) group_by(d, b, a + 1, g1))
})

test_that("unnamed computed dots on grouped data compute on the ungrouped data", {
  skip_if_no_gpu()
  f13_both(f13_df, function(d) group_by(d, g1) |> group_by(mean(a)))
  f13_both(f13_df, function(d) group_by(d, g1) |> group_by(a + 1, .add = TRUE))
  f13_both(f13_df, function(d) group_by(d, g1) |> group_by(n(), .add = TRUE))
})

test_that(".data pronoun dots are column references", {
  skip_if_no_gpu()
  f13_both(f13_df, function(d) group_by(d, .data$a))
  f13_both(f13_df, function(d) group_by(d, .data[["b"]], g1))
})

test_that("group_by() errors where dplyr errors", {
  skip_if_no_gpu()
  for (g in f13_modes(f13_df)) {
    expect_error(group_by(g, unknown))
    expect_error(group_by(g, starts_with("g")))
    expect_error(group_by(g, z = nope + 1))
  }
})

# --- 2. ungroup() ------------------------------------------------------------

test_that("ungroup() can remove some of the groups (#6606)", {
  skip_if_no_gpu()
  df <- data.frame(g1 = 1:2, g2 = 3:4, x = 5:6)
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), g1))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), g2))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), g1, g2))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), starts_with("g")))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), c(g2, x)))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), x))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), any_of("zz")))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2)))
  # the remaining groups still work
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), g1) |> summarise(s = sum(x)),
           arrange_by = "g2")
})

test_that("partial ungroup() keeps .drop, full ungroup() resets it", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "a"), levels = c("a", "b")), x = 1:2)
  for (g in f13_modes(df)) {
    gg <- group_by(g, f, x, .drop = FALSE)
    expect_false(group_by_drop_default(ungroup(gg, x)))
    expect_true(group_by_drop_default(ungroup(gg, f, x)))
    expect_equal(
      group_data(ungroup(gg, x)),
      group_data(ungroup(group_by(tibble::as_tibble(df), f, x, .drop = FALSE), x))
    )
  }
})

test_that("ungroup() errors where dplyr errors", {
  skip_if_no_gpu()
  df <- data.frame(x = 1, y = 2)
  for (g in f13_modes(df)) {
    expect_error(ungroup(g, x), class = "rlib_error_dots_nonempty")
    expect_error(ungroup(group_by(g, x, y), z), class = "vctrs_error_subscript_oob")
    expect_error(ungroup(group_by(g, x), x2 = x))
  }
})

# --- 3. logical keys sort FALSE < TRUE < NA ---------------------------------

test_that("logical group keys sort FALSE, TRUE, NA in group_data()", {
  skip_if_no_gpu()
  df <- data.frame(l = c(TRUE, NA, FALSE, TRUE, FALSE), x = 1:5)
  f13_both(df, function(d) group_data(group_by(d, l)))
  f13_both(df, function(d) group_keys(group_by(d, l, x > 2)))
  for (g in f13_modes(df)) {
    expect_equal(group_indices(group_by(g, l)), group_indices(group_by(df, l)))
  }
})

# --- 4. ordered factors stay ordered ----------------------------------------

test_that("ordered factors keep their class through group_data() and collect()", {
  skip_if_no_gpu()
  df <- data.frame(x = ordered(c("b", "a", "b"), levels = c("c", "b", "a")), y = 1:3)
  f13_both(df, function(d) d)
  f13_both(df, function(d) filter(d, y > 1))
  f13_both(df, function(d) group_data(group_by(d, x)))
  f13_both(df, function(d) group_data(group_by(d, x, .drop = FALSE)))
  f13_both(df, function(d) group_keys(group_by(d, x, .drop = FALSE)))
  f13_both(df, function(d) count(d, x, .drop = FALSE))
  for (g in f13_modes(df)) {
    expect_true(is.ordered(group_data(group_by(g, x, .drop = FALSE))$x))
    expect_true(is.ordered(collect(g)$x))
  }
})

# --- 5. quosures with a missing argument ------------------------------------

test_that("group_by() handles a quosure with a missing argument", {
  skip_if_no_gpu()
  named <- function(d, var) group_by(d, g = f13_ignore({{ var }}))
  unnamed <- function(d, var) group_by(d, f13_ignore({{ var }}))
  via_mutate <- function(d, var) mutate(d, g = f13_ignore({{ var }}))
  f13_both(f13_df, function(d) named(d))
  f13_both(f13_df, function(d) unnamed(d))
  f13_both(f13_df, function(d) via_mutate(d))
})

# --- 6. defunct arguments ---------------------------------------------------

test_that("group_by(add =) and group_by(.dots =) are defunct", {
  skip_if_no_gpu()
  df <- data.frame(x = 1, y = 2)
  for (g in f13_modes(df)) {
    expect_error(group_by(g, x, add = TRUE), class = "lifecycle_error_deprecated")
    expect_error(group_by(g, x, add = FALSE), class = "lifecycle_error_deprecated")
    expect_error(group_by(g, .dots = "x"), class = "lifecycle_error_deprecated")
  }
})

# --- 7. .drop = FALSE expansion, count() and tally() ------------------------

f13_fdf <- data.frame(
  f = factor(c("b", "b", "c"), levels = c("a", "b", "c", "d")),
  h = factor(c("p", "q", "p"), levels = c("p", "q", "r")),
  x = c(1, 2, 1),
  w = c(1.5, 2, 3),
  wi = c(1L, 2L, 3L)
)

test_that("count() with .drop = FALSE adds zero-count rows in group order", {
  skip_if_no_gpu()
  f13_both(f13_fdf, function(d) count(d, f, .drop = FALSE))
  f13_both(f13_fdf, function(d) count(d, f, h, .drop = FALSE))
  f13_both(f13_fdf, function(d) count(d, f, wt = w, .drop = FALSE))
  f13_both(f13_fdf, function(d) count(d, f, wt = wi, .drop = FALSE))
  f13_both(f13_fdf, function(d) count(d, f, .drop = FALSE, sort = TRUE))
  f13_both(f13_fdf, function(d) count(d, f, .drop = FALSE, name = "cnt"))
})

test_that("count()/tally() of a .drop = FALSE grouped table add zero-count rows", {
  skip_if_no_gpu()
  f13_both(f13_fdf, function(d) count(group_by(d, f, .drop = FALSE)))
  f13_both(f13_fdf, function(d) tally(group_by(d, f, .drop = FALSE)))
  f13_both(f13_fdf, function(d) tally(group_by(d, f, h, .drop = FALSE)))
  f13_both(f13_fdf, function(d) count(group_by(d, f, .drop = FALSE), h))
  for (g in f13_modes(f13_fdf)) {
    res <- count(group_by(g, f, .drop = FALSE))
    expect_equal(group_vars(res), "f")
    expect_false(group_by_drop_default(res))
  }
})

test_that("count() with .drop = FALSE handles NA levels, 0 rows and non-factor keys", {
  skip_if_no_gpu()
  fna <- data.frame(f = factor(c("b", NA, "b"), levels = c("a", "b")))
  f13_both(fna, function(d) count(d, f, .drop = FALSE))
  f13_both(f13_fdf[0, ], function(d) count(d, f, .drop = FALSE))
  f13_both(f13_fdf, function(d) count(d, f, x, .drop = FALSE))
  f13_both(f13_fdf, function(d) count(d, x, f, .drop = FALSE))
})

test_that(".drop = FALSE group_data() expands factor levels within each parent group", {
  skip_if_no_gpu()
  m <- data.frame(
    x = c(1, 2),
    f = factor(c("a", "a"), levels = c("a", "b")),
    h = factor(c("p", "q"), levels = c("p", "q", "r"))
  )
  f13_both(m, function(d) group_data(group_by(d, x, f, .drop = FALSE)))
  f13_both(m, function(d) group_data(group_by(d, f, x, .drop = FALSE)))
  f13_both(m, function(d) group_data(group_by(d, f, h, .drop = FALSE)))
})

test_that("count() with the default .drop is unchanged", {
  skip_if_no_gpu()
  f13_both(f13_fdf, function(d) count(d, f), arrange_by = "f")
  f13_both(f13_fdf, function(d) count(group_by(d, f)), arrange_by = "f")
})

# --- round 1 review findings (B1, B2, B4) -----------------------------------

test_that("a NULL group_by() dot is ignored; a named NULL is an error", {
  skip_if_no_gpu()
  opt <- function(df, grp = NULL) summarise(group_by(df, {{ grp }}), n = n())
  f13_both(f13_df, function(d) group_by(d, NULL))
  f13_both(f13_df, function(d) opt(d))
  f13_both(f13_df, function(d) summarise(group_by(d, NULL, b), n = n()), arrange_by = "b")
  for (g in f13_modes(f13_df)) {
    expect_error(group_by(g, g = NULL))
  }
})

test_that("count() without dots ignores .drop on a grouped input", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("b", "b", "c"), levels = c("a", "b", "c", "d")), x = 1:3)
  f13_both(df, function(d) count(group_by(d, f), .drop = FALSE), arrange_by = "f")
  f13_both(df, function(d) count(group_by(d, f, .drop = FALSE), .drop = TRUE))
})

test_that("partial ungroup() evaluates type predicates on the columns", {
  skip_if_no_gpu()
  df <- data.frame(g1 = 1:2, b = c("x", "y"), x = 5:6)
  f13_both(df, function(d) ungroup(group_by(d, g1, b), where(is.character)))
  f13_both(df, function(d) ungroup(group_by(d, g1, b), where(is.numeric)))
})

# --- round 2 review finding (B1) --------------------------------------------

test_that("partial ungroup() resolves group_cols() against the grouping", {
  skip_if_no_gpu()
  df <- data.frame(g1 = 1:2, g2 = 3:4, x = 5:6, s = c("p", "q"))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), group_cols()))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), -group_cols()))
  f13_both(df, function(d) ungroup(group_by(d, g1, g2), group_cols() & starts_with("g2")))
})
