# F49 coder tests: the safety net around rowwise tables, beyond the acceptance
# files. Every verb on a rowwise tbl_gpu must give dplyr's class, rowwise vars
# (in dplyr's order) and data, eager and lazy, whether it runs natively or on the
# CPU route; the rowwise vars are never real groups.

f49x_df <- function() {
  tibble::tibble(
    id = 1:6,
    g = c("a", "a", "b", "b", "c", "c"),
    i1 = c(1L, NA, 3L, -4L, 5L, 7L),
    i2 = c(2L, 5L, NA, 4L, 0L, 1L),
    d1 = c(1.5, NA, 2.25, -0.5, 1e10, 3),
    d2 = c(0.5, 1, NA, 3, -2, 0.25),
    l1 = c(TRUE, NA, FALSE, TRUE, FALSE, TRUE),
    s = letters[1:6],
    f = factor(c("x", "y", "x", "z", "y", "x"))
  )
}

# `pipeline(x, side)`: x is the tibble or its tbl_gpu; `side(v)` turns a
# second table into the matching kind (a tbl_gpu next to a tbl_gpu).
f49x_check <- function(pipeline, lazy = FALSE) {
  df <- f49x_df()
  withr::local_options(dplyr.summarise.inform = FALSE)
  want <- pipeline(df, function(v) v)
  got <- pipeline(tbl_gpu(df, lazy = lazy), function(v) tbl_gpu(v, lazy = lazy))
  if (is_tbl_gpu(got)) got <- collect(got)
  expect_identical(class(got), class(want))
  if (is.data.frame(want)) {
    expect_identical(dplyr::group_vars(got), dplyr::group_vars(want))
  }
  expect_equal(got, want)
}

f49x_same_error <- function(pipeline, lazy = FALSE) {
  df <- f49x_df()
  want <- tryCatch(pipeline(df, function(v) v), error = identity)
  got <- tryCatch({
    o <- pipeline(tbl_gpu(df, lazy = lazy), function(v) tbl_gpu(v, lazy = lazy))
    if (is_tbl_gpu(o)) collect(o) else o
  }, error = identity)
  expect_s3_class(want, "error")
  expect_s3_class(got, "error")
  expect_identical(class(got), class(want))
}

test_that("every verb on a rowwise table matches dplyr, vars in dplyr's order (eager and lazy)", {
  skip_if_no_gpu()
  y <- tibble::tibble(g = c("a", "b", "d"), w = c(10, 20, 40))
  more <- f49x_df()[1:2, ]
  rw <- function(x, ...) dplyr::rowwise(x, ...)
  pipes <- list(
    function(x, side) dplyr::slice(rw(x, i2, g), 1),
    function(x, side) dplyr::slice_max(rw(x, id), i1),
    function(x, side) dplyr::distinct(rw(x, i2, g), i1),
    function(x, side) dplyr::count(rw(x, g), i1),
    function(x, side) dplyr::tally(rw(x, g)),
    function(x, side) dplyr::add_count(rw(x, g), g),
    function(x, side) dplyr::left_join(rw(x, i2, g), side(y), by = "g"),
    function(x, side) dplyr::full_join(rw(x, g, id), side(y), by = "g"),
    function(x, side) dplyr::semi_join(rw(x, g), side(y), by = "g"),
    function(x, side) dplyr::cross_join(rw(x, id), side(y)),
    function(x, side) cuplyr::bind_rows(rw(x, g), side(more)),
    function(x, side) cuplyr::bind_rows(side(more), rw(x, g)),
    function(x, side) cuplyr::bind_cols(rw(x, g), side(tibble::tibble(zz = 1:6))),
    function(x, side) dplyr::union(rw(x, g), side(more)),
    function(x, side) dplyr::setdiff(rw(x, g), side(more)),
    function(x, side) dplyr::arrange(rw(x, i2, g), dplyr::desc(id), .by_group = TRUE),
    function(x, side) dplyr::mutate(rw(x, i2, g), z = i1 + 1L),
    function(x, side) dplyr::mutate(rw(x, i2, g), z = i1 + 1L, .keep = "none"),
    function(x, side) dplyr::mutate(rw(x, i2, g), z = i1 + 1L, .keep = "used"),
    function(x, side) dplyr::mutate(rw(x, i2, g), z = i1 + 1L, .before = g),
    function(x, side) dplyr::transmute(rw(x, i2, g), z = i1 + 1L),
    function(x, side) dplyr::transmute(rw(x, i2, g), z = i1 + 1L, i2),
    function(x, side) dplyr::select(rw(x, i2, g), g, i2),
    function(x, side) dplyr::select(rw(x, i2, g), i1) |> suppressMessages(),
    function(x, side) dplyr::relocate(rw(x, i2, g), g, .before = id),
    function(x, side) dplyr::rename_with(rw(x, i2, g), toupper, c(i1, d1)),
    function(x, side) rw(x, i2, g)[2:3, c("i2", "g", "id")],
    function(x, side) head(rw(x, i2, g), 3),
    function(x, side) tail(rw(x, i2, g), 2),
    function(x, side) { z <- rw(x, i2, g); z$new <- 1; z },
    function(x, side) dplyr::summarise(rw(x, i2, g), z = i1 + 1L) |> suppressMessages(),
    function(x, side) dplyr::summarise(rw(x, i2, g), z = i1 + 1L, .groups = "rowwise"),
    function(x, side) dplyr::summarise(rw(x, g), z = list(i1)) |> suppressMessages(),
    function(x, side) dplyr::summarise(rw(x, g), z = dplyr::n()) |> suppressMessages(),
    function(x, side) dplyr::filter(rw(x, i2, g), i1 > 0),
    function(x, side) dplyr::filter(rw(x, g), dplyr::row_number() == 1),
    function(x, side) dplyr::filter(rw(x, g), dplyr::if_any(c(i1, i2), ~ .x > 2)),
    function(x, side) dplyr::filter_out(rw(x, g), sum(c(i1, i2), na.rm = TRUE) > 3),
    function(x, side) dplyr::reframe(rw(x, g), z = c(i1, i2)),
    function(x, side) dplyr::mutate(rw(x, g), z = dplyr::row_number()),
    function(x, side) dplyr::mutate(rw(x, g), z = mean(i1)),
    function(x, side) dplyr::mutate(rw(x, g), z = list(i1)),
    function(x, side) dplyr::mutate(rw(x, g), across(c(i1, i2), ~ .x + 1L)),
    function(x, side) dplyr::mutate(rw(x, g), across(c(d1, d2), mean)),
    function(x, side) dplyr::mutate(rw(x, g), z = sum(i1 * 2L)),
    function(x, side) dplyr::mutate(rw(x, g), z = median(c(i1, i2))),
    function(x, side) dplyr::mutate(rw(x, g), z = dplyr::cur_group()$g),
    function(x, side) dplyr::mutate(rw(x, g), z = dplyr::cur_group_id() + dplyr::n()),
    function(x, side) dplyr::mutate(rw(x, g), a = i1 + 1L, b = sum(dplyr::c_across(c(a, i2)))),
    function(x, side) dplyr::mutate(rw(x, g), i1 = i1 * 2L, s2 = sum(dplyr::c_across(c(i1, i2)))),
    function(x, side) dplyr::mutate(rw(x, id), i1 = i1 * 2L, s2 = sum(i1, i2)),
    function(x, side) dplyr::mutate(rw(x, g), z = sum(c(i1, i2)) / dplyr::n()),
    function(x, side) dplyr::group_by(rw(x, g), f, .add = TRUE),
    function(x, side) dplyr::group_by(rw(x, g), k = i1 > 2),
    function(x, side) dplyr::summarise(dplyr::group_by(rw(x, g), f), z = sum(i1, na.rm = TRUE)) |>
      suppressMessages(),
    function(x, side) dplyr::mutate(dplyr::rowwise(dplyr::group_by(x, g)), z = sum(dplyr::c_across(c(i1, i2)), na.rm = TRUE)),
    function(x, side) dplyr::nest_by(rw(x, g)),
    function(x, side) dplyr::group_modify(rw(x, g), ~ .x),
    function(x, side) dplyr::pull(rw(x, g), i1),
    function(x, side) dplyr::nest_join(rw(x, g), side(y), by = "g")
  )
  for (lazy in c(FALSE, TRUE)) {
    for (pipe in pipes) {
      f49x_check(pipe, lazy = lazy)
    }
  }
})

test_that("pipelines whose shape dplyr rejects raise dplyr's error on a rowwise table", {
  skip_if_no_gpu()
  rw <- function(x, ...) dplyr::rowwise(x, ...)
  pipes <- list(
    function(x, side) dplyr::mutate(rw(x, g), g = NULL),
    function(x, side) dplyr::filter(rw(x, g), c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)),
    function(x, side) dplyr::summarise(rw(x, g), z = 1, .by = id),
    function(x, side) dplyr::summarise(rw(x, g), z = 1, .groups = "bogus"),
    function(x, side) dplyr::summarise(rw(x, g), z = c(1, 2)),
    function(x, side) dplyr::mutate(rw(x), z = sum(dplyr::c_across(c(i1, nope)))),
    function(x, side) dplyr::mutate(rw(x, g), z = sum(dplyr::c_across(g)))
  )
  for (lazy in c(FALSE, TRUE)) {
    for (pipe in pipes) {
      f49x_same_error(pipe, lazy = lazy)
    }
  }
})

test_that("a rowwise glimpse() shows the vars, not a group count", {
  skip_if_no_gpu()
  out <- utils::capture.output(dplyr::glimpse(dplyr::rowwise(tbl_gpu(f49x_df()), g, id)))
  expect_true(any(grepl("^Rowwise: g, id", out)))
  expect_false(any(grepl("^Groups:", out)))
})

test_that("a rowwise table's vctrs/cast round trips stay rowwise, never grouped by the vars", {
  skip_if_no_gpu()
  rw <- dplyr::rowwise(tbl_gpu(f49x_df()), g)
  both <- vctrs::vec_ptype2(rw, rw)
  expect_true(is_rowwise_gpu(both))
  sliced <- vctrs::vec_slice(rw, 2:3)
  expect_true(is_rowwise_gpu(sliced))
  expect_identical(class(collect(sliced)), class(dplyr::rowwise(f49x_df(), g)))
})

test_that("mutate_all()/mutate_if() on a rowwise table do not announce ignored groups", {
  skip_if_no_gpu()
  gt <- dplyr::rowwise(tbl_gpu(tibble::tibble(g = c("a", "b"), x = c(1, 2), y = c(3, 4))), g)
  expect_no_message(out <- suppressWarnings(dplyr::mutate_if(gt, is.numeric, ~ .x + 1)))
  expect_equal(collect(out)$x, c(2, 3))
})
