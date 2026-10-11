# F40 attempt 2 acceptance tests: duplicated column names are supported on an
# allow-list only.
#
# Contract: a tbl_gpu may carry duplicated column names only while it is
# UNGROUPED (no groups, hence no slice(.preserve = TRUE) fixed keys). On such
# a table the positional select/relocate/rename/distinct/`[`/`$`/`[[`/names()/
# collect() paths work (test-parity-F40-dup-names.R); everything else errors.
# A writer that would put duplicated names on a grouped table errors at the
# write, never later:
# - `names<-` on a grouped table: dplyr allows it (names<-.grouped_df), but a
#   name no longer identifies the group column, so cuplyr refuses it with a
#   classed error naming the cause (review B6 of attempt 1: the lazy path
#   regrouped by the wrong column, silently).
# - bind_cols(<grouped>, ..., .name_repair = "minimal"): dplyr raises
#   tibble_error_column_names_must_be_unique at bind_cols() itself.
# Unique-name renames of a grouped table keep working, eager and lazy.

f40a_df <- function() tibble::tibble(x = c(1, 2), y = c(3, 4), z = c(5, 6))

f40a_grouped <- function(lazy) {
  # lazy: a pending op on top, so `names<-` goes through the AST
  g <- tbl_gpu(f40a_df(), lazy = lazy) |> group_by(x)
  if (lazy) g <- filter(g, y > 0)
  g
}

test_that("names<- that duplicates a name on a grouped table errors (review B6)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- f40a_grouped(lazy)
    expect_error({ names(g)[3] <- "x" }, "grouped", class = "cuplyr_error_duplicate_names")
    expect_error({ names(g) <- c("x", "y", "x") }, "grouped", class = "cuplyr_error_duplicate_names")
    expect_error({ names(g) <- c("z", "y", "z") }, "grouped", class = "cuplyr_error_duplicate_names")
    # the failed assignment left g alone
    expect_identical(names(g), c("x", "y", "z"))
    expect_identical(group_vars(g), "x")
  }
  # lazy with no pending ops, and a grouped table whose group column is not first
  g0 <- tbl_gpu(f40a_df(), lazy = TRUE) |> group_by(z)
  expect_error({ names(g0)[1] <- "z" }, "grouped", class = "cuplyr_error_duplicate_names")
})

test_that("the B6 sequence (typo, then fix) keeps the grouping on x", {
  skip_if_no_gpu()
  ref <- f40a_df() |> group_by(x)
  names(ref)[3] <- "b"
  ref_sum <- summarise(ref, n = n())
  for (lazy in c(FALSE, TRUE)) {
    g <- f40a_grouped(lazy)
    try(names(g)[3] <- "x", silent = TRUE)
    names(g)[3] <- "b"
    expect_identical(names(g), c("x", "y", "b"))
    expect_identical(group_vars(g), "x")
    expect_identical(tibble::as_tibble(collect(g)), tibble::as_tibble(ref))
    expect_identical(group_vars(collect(g)), "x")
    expect_identical(
      dplyr::arrange(tibble::as_tibble(collect(summarise(g, n = n()))), x),
      dplyr::arrange(tibble::as_tibble(ref_sum), x)
    )
  }
})

test_that("unique-name names<- and rename() on a grouped table still follow dplyr", {
  skip_if_no_gpu()
  ref <- f40a_df() |> group_by(x)
  names(ref) <- c("a", "y", "b")
  for (lazy in c(FALSE, TRUE)) {
    g <- f40a_grouped(lazy)
    names(g) <- c("a", "y", "b")
    expect_identical(group_vars(g), "a")
    expect_identical(tibble::as_tibble(collect(g)), tibble::as_tibble(ref))
    expect_identical(group_vars(collect(g)), "a")

    g2 <- rename(f40a_grouped(lazy), a = 1)
    expect_identical(group_vars(g2), "a")
    # rename() that would duplicate a name: dplyr's vctrs error
    expect_error(rename(f40a_grouped(lazy), z = y), class = "vctrs_error_names_must_be_unique")
  }
})

test_that("slice(.preserve = TRUE) tables (fixed keys) refuse duplicated names too", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    s <- tbl_gpu(f40a_df(), lazy = lazy) |> group_by(x) |> slice(1, .preserve = TRUE)
    expect_error({ names(s)[2] <- "x" }, "grouped", class = "cuplyr_error_duplicate_names")
  }
})

test_that("bind_cols() of a grouped table with minimal repair errors at bind_cols()", {
  skip_if_no_gpu()
  d <- f40a_df()
  expect_error(dplyr::bind_cols(group_by(d, x), d, .name_repair = "minimal"),
               class = "tibble_error_column_names_must_be_unique")
  for (lazy in c(FALSE, TRUE)) {
    gx <- tbl_gpu(d, lazy = lazy) |> group_by(x)
    expect_error(cuplyr::bind_cols(gx, tbl_gpu(d, lazy = lazy), .name_repair = "minimal"),
                 class = "tibble_error_column_names_must_be_unique")
    expect_error(cuplyr::bind_cols(gx, d, .name_repair = "minimal"),
                 class = "tibble_error_column_names_must_be_unique")
  }
})

test_that("ungrouped duplicated-name tables stay on the allow-list", {
  skip_if_no_gpu()
  d <- f40a_df()
  ref_bind <- dplyr::bind_cols(d, d, .name_repair = "minimal")
  for (lazy in c(FALSE, TRUE)) {
    # bind_cols with minimal repair on ungrouped inputs
    b <- cuplyr::bind_cols(tbl_gpu(d, lazy = lazy), tbl_gpu(d, lazy = lazy), .name_repair = "minimal")
    expect_identical(names(b), names(ref_bind))
    expect_identical(collect(b), ref_bind)
    expect_identical(collect(select(b, 5)), tibble::tibble(y = c(3, 4)))

    # grouped history, then ungroup(), then names<- that duplicates: allowed
    u <- tbl_gpu(d, lazy = lazy) |> group_by(x) |> filter(y > 0) |> ungroup()
    names(u)[3] <- "x"
    expect_identical(group_vars(u), character())
    expect_identical(names(u), c("x", "y", "x"))
    expect_identical(collect(select(u, 3)), tibble::tibble(x = c(5, 6)))
    expect_identical(collect(relocate(u, 3)),
                     tibble::tibble(x = c(5, 6), x = c(1, 2), y = c(3, 4), .name_repair = "minimal"))
    expect_identical(collect(rename(u, b = 3)), tibble::tibble(x = c(1, 2), y = c(3, 4), b = c(5, 6)))
    # and back to unique names, then grouping works by the right column
    names(u)[3] <- "b"
    expect_identical(
      dplyr::arrange(tibble::as_tibble(collect(summarise(group_by(u, b), s = sum(x)))), b),
      tibble::tibble(b = c(5, 6), s = c(1, 2))
    )
    # outside the allow-list: grouping a duplicated-name table errors
    names(u)[3] <- "x"
    expect_error(group_by(u, y), class = "tibble_error_column_names_must_be_unique")
    expect_error(filter(u, y > 0), "duplicate names")
  }
})

# Round-1 review of attempt 2 (B1, B2), dplyr results verified with tools/r.

test_that("bind_rows(.id = <existing column>) replaces that column like dplyr (review B1)", {
  skip_if_no_gpu()
  d <- f40a_df()
  ref_x <- dplyr::bind_rows(d, d, .id = "x")
  ref_y <- dplyr::bind_rows(d, d, .id = "y")
  ref_gx <- dplyr::bind_rows(group_by(d, x), d, .id = "x")
  ref_gy <- dplyr::bind_rows(group_by(d, y), d, .id = "y")
  for (lazy in c(FALSE, TRUE)) {
    expect_identical(collect(cuplyr::bind_rows(tbl_gpu(d, lazy = lazy), tbl_gpu(d, lazy = lazy), .id = "x")), ref_x)
    expect_identical(collect(cuplyr::bind_rows(tbl_gpu(d, lazy = lazy), tbl_gpu(d, lazy = lazy), .id = "y")), ref_y)

    bx <- cuplyr::bind_rows(group_by(tbl_gpu(d, lazy = lazy), x), tbl_gpu(d, lazy = lazy), .id = "x")
    expect_identical(names(bx), c("x", "y", "z"))
    expect_identical(group_vars(bx), "x")
    expect_identical(tibble::as_tibble(collect(bx)), tibble::as_tibble(ref_gx))
    expect_identical(group_vars(collect(bx)), "x")
    expect_identical(group_vars(rename(bx, id = 1)), "id")

    by <- cuplyr::bind_rows(group_by(tbl_gpu(d, lazy = lazy), y), tbl_gpu(d, lazy = lazy), .id = "y")
    expect_identical(names(by), c("x", "y", "z"))
    expect_identical(group_vars(by), "y")
    expect_identical(tibble::as_tibble(collect(by)), tibble::as_tibble(ref_gy))
  }
})

test_that("select() of one column under two names works on a grouped table (review B2)", {
  skip_if_no_gpu()
  d <- f40a_df()
  ref_y <- suppressMessages(select(group_by(d, y), a = x, b = x))
  ref_x <- select(group_by(d, x), a = x, b = x)
  for (lazy in c(FALSE, TRUE)) {
    r_y <- suppressMessages(select(group_by(tbl_gpu(d, lazy = lazy), y), a = x, b = x))
    expect_identical(names(r_y), c("y", "a", "b"))
    expect_identical(group_vars(r_y), "y")
    expect_identical(tibble::as_tibble(collect(r_y)), tibble::as_tibble(ref_y))

    r_x <- select(group_by(tbl_gpu(d, lazy = lazy), x), a = x, b = x)
    expect_identical(names(r_x), c("a", "b"))
    expect_identical(group_vars(r_x), group_vars(ref_x))
    expect_identical(tibble::as_tibble(collect(r_x)), tibble::as_tibble(ref_x))
  }
})

test_that("grouped select twice keeps slice(.preserve = TRUE) empty groups", {
  skip_if_no_gpu()
  d <- tibble::tibble(x = factor(c("a", "b", "c")), y = c(3, 4, 5))
  ref <- d |> group_by(x) |> slice(0, .preserve = TRUE) |> select(a = x, b = x, y)
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(d, lazy = lazy) |> group_by(x) |> slice(0, .preserve = TRUE)
    r <- select(g, a = x, b = x, y)
    expect_identical(group_vars(r), "a")
    expect_identical(tibble::as_tibble(group_data(r)), tibble::as_tibble(group_data(ref)))
  }
})
