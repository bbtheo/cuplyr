# F33 acceptance tests (tier B): row-subsetting verbs keep the empty groups of
# a .drop = FALSE table, as dplyr does.
#
# dplyr 1.2.1 (R/generics.R, dplyr_row_slice.grouped_df()): arrange(),
# distinct() and the filtering joins slice the rows and keep the input's
# group data; empty groups are trimmed only when .drop is TRUE. So after
# group_by(g, .drop = FALSE) |> filter(g == "x") (which keeps the emptied "y"
# and "z" groups), arrange()/distinct()/semi_join()/anti_join() still report
# those empty groups, and a later summarise() gives them a row with n = 0.
# With .drop = TRUE (slice(.preserve = TRUE) keys) the empty groups go.

f33b_rows_of <- function(gd) {
  keys <- tibble::as_tibble(gd[setdiff(names(gd), ".rows")])
  list(keys = keys, rows = lapply(gd$.rows, as.integer))
}

f33b_df <- data.frame(
  g = c("y", "x", "x", "z"),
  x = c(1L, 2L, 3L, 2L),
  stringsAsFactors = FALSE
)
f33b_y <- data.frame(x = 2L)

f33b_check <- function(verb) {
  expected <- f33b_df |> group_by(g, .drop = FALSE) |> filter(g == "x") |>
    verb(y = f33b_y)
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    got <- tbl_gpu(f33b_df, lazy = lazy) |> group_by(g, .drop = FALSE) |>
      filter(g == "x") |> verb(y = tbl_gpu(f33b_y))
    expect_equal(f33b_rows_of(group_data(got)), f33b_rows_of(group_data(expected)),
                 info = paste(label, "group_data"))
    expect_equal(dplyr::n_groups(got), dplyr::n_groups(expected), info = label)
    expect_equal(
      tibble::as_tibble(dplyr::ungroup(collect(got))),
      tibble::as_tibble(dplyr::ungroup(expected)),
      info = paste(label, "rows")
    )
    expect_equal(
      f33b_rows_of(dplyr::group_data(collect(got))),
      f33b_rows_of(dplyr::group_data(expected)),
      info = paste(label, "collected group_data")
    )
    expect_equal(
      tibble::as_tibble(collect(summarise(got, n = n()))),
      tibble::as_tibble(summarise(expected, n = n())),
      info = paste(label, "summarise after")
    )
  }
}

test_that("arrange() keeps the empty groups of a .drop = FALSE table", {
  skip_if_no_gpu()
  f33b_check(function(d, y) arrange(d, desc(x)))
})

test_that("distinct() keeps the empty groups of a .drop = FALSE table", {
  skip_if_no_gpu()
  f33b_check(function(d, y) distinct(d))
  f33b_check(function(d, y) distinct(d, x))
})

test_that("semi_join() and anti_join() keep the empty groups of a .drop = FALSE table", {
  skip_if_no_gpu()
  f33b_check(function(d, y) semi_join(d, y, by = "x"))
  f33b_check(function(d, y) anti_join(d, y, by = "x"))
})

test_that("with .drop = TRUE, arrange() still drops preserved empty groups", {
  skip_if_no_gpu()
  expected <- f33b_df |> group_by(g) |> slice(1, .preserve = TRUE) |>
    filter(g == "x", .preserve = TRUE) |> arrange(x)
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(f33b_df, lazy = lazy) |> group_by(g) |> slice(1, .preserve = TRUE) |>
      filter(g == "x", .preserve = TRUE) |> arrange(x)
    expect_equal(f33b_rows_of(group_data(got)), f33b_rows_of(group_data(expected)))
  }
})
