# F14 acceptance tests: tidyselect predicates in select()/relocate()/
# rename_with(), grouped select() (missing grouping columns are added with a
# message; groups follow the column), colnames(), and rename by position with
# duplicated names.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - where(fn) tests the column VALUES' type, in every tidyselect slot.
# - select() on a grouped table prepends the grouping columns it doesn't
#   select, with "Adding missing grouping variables: `g`", unless a rename
#   already uses that name; groups follow the column (by position), so
#   select(group_by(d, a), a = b) is ungrouped.
# - colnames(x) is names(x).
# - rename(df, x2 = 2) renames column 2 in place, also when names repeat.

f14s_df <- function() {
  data.frame(
    x = c(1L, 2L, 2L, NA),
    s = c("a", "b", "b", NA),
    v = c(1.5, NA, 2.5, 2.5),
    d = as.Date(c("2024-01-01", NA, "2024-01-03", "2024-01-03")),
    f = factor(c("u", "v", "v", "u")),
    t = c("p", "q", "r", "p"),
    stringsAsFactors = FALSE
  )
}

f14s_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# GPU forced native: any CPU fallback is an error.
f14s_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f14s_both(df, pipeline, ...)
}

# Parity with the "Adding missing grouping variables" message muted on both
# sides (the message itself is checked separately).
f14s_quiet <- function(df, pipeline, ...) {
  f14s_both(df, function(d) suppressMessages(pipeline(d)), ...)
}

f14s_is_date <- function(x) inherits(x, "Date")

# =============================================================================
# where() predicates see column types (A10)
# =============================================================================

test_that("relocate(where()) moves the matching columns (A10)", {
  skip_if_no_gpu()
  df <- f14s_df()
  f14s_native(df, function(d) relocate(d, where(is.character)))
  f14s_native(df, function(d) relocate(d, where(is.numeric), .after = where(is.character)))
  f14s_native(df, function(d) relocate(d, where(is.factor), .before = x))
  f14s_native(df, function(d) relocate(d, where(f14s_is_date)))
  f14s_native(df, function(d) relocate(d, s, .after = where(is.double)))
})

test_that("the dplyr conformance relocate block case works", {
  skip_if_no_gpu()
  df <- data.frame(x = 1, a = "a", y = 2, b = "a", stringsAsFactors = FALSE)
  g <- tbl_gpu(df)
  expect_named(relocate(g, where(is.character)), c("a", "b", "x", "y"))
  expect_named(
    relocate(g, where(is.character), .after = where(is.numeric)),
    c("x", "y", "a", "b")
  )
})

test_that("select(where()) selects by column type", {
  skip_if_no_gpu()
  df <- f14s_df()
  f14s_native(df, function(d) select(d, where(is.numeric)))
  f14s_native(df, function(d) select(d, where(is.character), x))
  f14s_native(df, function(d) select(d, !where(is.numeric)))
  f14s_native(df, function(d) select(d, where(is.factor)))
  f14s_native(df, function(d) select(d, where(is.integer)))
  f14s_native(df, function(d) select(d, where(f14s_is_date), last_col()))
  f14s_native(df, function(d) select(d, where(~ is.double(.x)) & !v))
})

test_that("where() sees the types of columns created by pending lazy ops", {
  skip_if_no_gpu()
  df <- f14s_df()
  f14s_both(df, function(d) d |> mutate(w = v * 2, k = x > 1L) |> select(where(is.logical)))
  f14s_both(df, function(d) d |> mutate(w = x + 1L) |> relocate(where(is.integer)))
})

test_that("where() predicates that read column values select from the real data", {
  # Added after round-1 review B1: a typed proxy can't answer these; they must
  # still match dplyr (a CPU fallback for the selection is fine).
  skip_if_no_gpu()
  df <- data.frame(x = c(1L, NA, 3L), s = c("a", "b", "c"), v = c(NA, 2.5, 1),
                   k = c(1, 1, 1), stringsAsFactors = FALSE)
  f14s_both(df, function(d) select(d, where(~ anyNA(.x))))
  f14s_both(df, function(d) select(d, where(~ !anyNA(.x))))
  f14s_both(df, function(d) select(d, where(~ dplyr::n_distinct(.x) > 1)))
  f14s_both(df, function(d) select(d, where(function(col) is.numeric(col) && !anyNA(col))))
  f14s_both(df, function(d) relocate(d, where(~ !anyNA(.x))))
  f14s_both(df, function(d) relocate(d, k, .after = where(~ anyNA(.x))))
  f14s_both(df, function(d) rename_with(d, toupper, .cols = where(~ anyNA(.x))))
  f14s_both(df, function(d) d |> filter(!is.na(x)) |> select(where(~ !anyNA(.x))))

  # Added after round-2 review B1: thresholds a small proxy can't reach.
  df2 <- data.frame(a = 1:6, b = c(1, 1, 1, 2, 2, 2),
                    s = c("x@y.com", "a", "b", "c", "d", "e"),
                    m = c(NA, NA, 1, 2, 3, 4), stringsAsFactors = FALSE)
  f14s_both(df2, function(d) select(d, where(~ dplyr::n_distinct(.x) > 2)))
  f14s_both(df2, function(d) select(d, where(~ dplyr::n_distinct(.x) < 4)))
  f14s_both(df2, function(d) select(d, where(~ sum(is.na(.x)) > 1)))
  f14s_both(df2, function(d) select(d, where(~ any(grepl("@", .x)))))
  f14s_both(df2, function(d) select(d, where(~ is.numeric(.x) && max(.x, na.rm = TRUE) > 5)))
  f14s_both(df2, function(d) relocate(d, where(~ dplyr::n_distinct(.x) > 2), .after = last_col()))
  f14s_both(df2, function(d) rename_with(d, toupper, .cols = where(~ dplyr::n_distinct(.x) > 2)))
  f14s_quiet(df2, function(d) select(group_by(d, b), where(~ dplyr::n_distinct(.x) > 2)))
})

test_that("where(is.ordered) and level predicates see the real factor type", {
  # Added after round-2 review B4.
  skip_if_no_gpu()
  df <- data.frame(o = factor(c("lo", "hi", "lo"), levels = c("lo", "hi"), ordered = TRUE),
                   f = factor(c("u", "v", "u"), levels = c("u", "v", "w")), x = 1:3)
  f14s_both(df, function(d) select(d, where(is.ordered)))
  f14s_both(df, function(d) relocate(d, where(is.ordered), .after = x))
  f14s_both(df, function(d) rename_with(d, toupper, .cols = where(is.ordered)))
  f14s_both(df, function(d) select(d, where(~ is.factor(.x) && nlevels(.x) > 2)))
})

test_that("rename_with(.cols = where()) renames only the matching columns", {
  skip_if_no_gpu()
  df <- f14s_df()
  f14s_native(df, function(d) rename_with(d, toupper, .cols = where(is.character)))
  f14s_native(df, function(d) rename_with(d, ~ paste0(.x, "_n"), .cols = where(is.numeric)))
})

# =============================================================================
# Grouped select(): missing grouping columns, groups follow the column
# =============================================================================

test_that("select() on a grouped table adds missing grouping columns", {
  skip_if_no_gpu()
  df <- f14s_df()
  f14s_quiet(df, function(d) d |> group_by(s) |> select(x))
  f14s_quiet(df, function(d) d |> group_by(s) |> select(-s))
  f14s_quiet(df, function(d) d |> group_by(x, s) |> select(v))
  f14s_quiet(df, function(d) d |> group_by(s, x) |> select(v, x))
  f14s_quiet(df, function(d) d |> group_by(s) |> select(where(is.numeric)))
  f14s_quiet(df, function(d) d |> group_by(s) |> select())
})

test_that("grouped select() says which grouping columns it adds", {
  skip_if_no_gpu()
  df <- f14s_df()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_message(select(group_by(g, s), x),
                   "Adding missing grouping variables: `s`", fixed = TRUE)
    expect_message(select(group_by(g, x, s), v),
                   "Adding missing grouping variables: `x`, `s`", fixed = TRUE)
    expect_message(select(group_by(g, s)),
                   "Adding missing grouping variables: `s`", fixed = TRUE)
    expect_no_message(select(group_by(g, s), s, x))
    expect_no_message(select(group_by(g, s), s2 = s, x))
    expect_no_message(select(group_by(g, x), x = v))
    expect_no_message(select(g, x))
  }
})

test_that("grouped select() groups follow the selected columns by position", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:2, b = 3:4, c = 5:6)
  # a renamed group column keeps its group under the new name
  f14s_quiet(df, function(d) d |> group_by(a) |> select(a2 = a, b))
  f14s_quiet(df, function(d) d |> group_by(a, b) |> select(b2 = b, a2 = a))
  # another column renamed onto the group's name: the group is gone
  f14s_quiet(df, function(d) d |> group_by(a) |> select(a = b))
  # ... and the other grouping column is added back
  f14s_quiet(df, function(d) d |> group_by(a, b) |> select(a = c))
  f14s_quiet(df, function(d) d |> group_by(a, b) |> select(b = c))
  f14s_quiet(df, function(d) d |> group_by(a, b) |> select(c, a2 = a))
})

test_that("relocate() and rename() keep groups attached to their column", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:2, b = 3:4, c = 5:6)
  f14s_both(df, function(d) d |> group_by(a) |> relocate(a, .after = c))
  f14s_both(df, function(d) d |> group_by(a) |> relocate(a2 = a, .after = c))
  f14s_both(df, function(d) d |> group_by(a, b) |> rename(z = a))
  f14s_both(df, function(d) d |> group_by(a) |> rename(a = b, b = a))
})

test_that("the conformance non-syntactic grouping column case works", {
  skip_if_no_gpu()
  df <- data.frame(`a b` = 1L, check.names = FALSE)
  for (lazy in c(FALSE, TRUE)) {
    g <- group_by(tbl_gpu(df, lazy = lazy), `a b`)
    expect_message(res <- select(g), "Adding missing grouping variables: `a b`",
                   fixed = TRUE)
    expect_named(res, "a b")
    expect_identical(group_vars(res), "a b")
    expect_identical(collect(ungroup(res))$`a b`, 1L)
  }
})

# =============================================================================
# colnames() and rename to UTF-8 names
# =============================================================================

test_that("colnames() of a tbl_gpu are its column names", {
  skip_if_no_gpu()
  df <- f14s_df()
  g <- tbl_gpu(df)
  expect_identical(colnames(g), names(df))
  lz <- tbl_gpu(df, lazy = TRUE) |> mutate(w = v + 1) |> select(w, s)
  expect_identical(colnames(lz), c("w", "s"))
  expect_identical(colnames(group_by(g, s)), names(df))
})

test_that("rename() to a UTF-8 name is visible through names() and colnames()", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(data.frame(a = 1), lazy = lazy) |> rename("幸" := a)
    expect_identical(colnames(g), "幸")
    expect_identical(names(g), "幸")
    expect_identical(names(collect(g)), "幸")
  }
  f14s_both(data.frame(a = 1, b = "z"), function(d) rename(d, "幸" = a, "été" = b))
})

# =============================================================================
# rename by position with duplicated names
# =============================================================================

test_that("rename by position keeps the column in place with duplicated names", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1, x = 2, y = 3, .name_repair = "minimal")
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    r2 <- rename(g, x2 = 2)
    expect_named(r2, c("x", "x2", "y"))
    out <- collect(r2)
    expect_named(out, c("x", "x2", "y"))
    expect_identical(out$x2, 2)
    expect_identical(out$x, 1)

    r1 <- rename(g, x1 = 1)
    expect_named(r1, c("x1", "x", "y"))
    out1 <- collect(r1)
    expect_identical(out1$x1, 1)
    expect_identical(out1$x, 2)
  }
})

test_that("rename by position keeps the position with unique names", {
  skip_if_no_gpu()
  df <- f14s_df()
  f14s_native(df, function(d) rename(d, s2 = 2))
  f14s_native(df, function(d) rename(d, last = 6, first = 1))
  f14s_both(df, function(d) d |> group_by(s) |> rename(s2 = 2))
})
