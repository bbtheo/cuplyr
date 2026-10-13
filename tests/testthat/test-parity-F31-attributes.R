# F31 acceptance tests: data-frame attributes and bare data.frame class.
#
# dplyr's rule (R/generics.R dplyr_reconstruct(), dplyr_row_slice(),
# dplyr_col_modify()): a verb on an UNGROUPED data frame copies the input's
# attributes onto the result (except names/row.names/class) and keeps its
# class; summarise()/reframe()/tally() drop the attributes but keep a bare
# data.frame class; group_by() keeps the attributes and gives a grouped tibble;
# any verb on a GROUPED data frame (ungroup() included) drops them. Joins take
# x's, binds the first input's. Row names stay out of scope.
#
# On a tbl_gpu the data-frame attributes are the object's own R attributes
# (other than names/class), so `attr(gt, "foo") <- "bar"` and
# `attr(out, "foo")` work as on a data frame; collect() gives them back.

f31_df <- function() {
  df <- data.frame(g = c(1, 1, 2, 3), x = c(4, 2, 3, 1))
  attr(df, "foo") <- "bar"
  attr(df, "n") <- 1:2
  df
}

f31_tbl <- function() {
  tb <- tibble::tibble(g = c(1, 1, 2, 3), x = c(4, 2, 3, 1))
  attr(tb, "foo") <- "bar"
  tb
}

f31_y <- function() {
  y <- data.frame(g = c(1, 2, 5), z = c(10, 20, 50))
  attr(y, "foo") <- "ybar"
  attr(y, "yonly") <- TRUE
  y
}

# User attributes of a data frame (what dplyr_reconstruct() copies).
f31_frame_attrs <- function(x) {
  a <- attributes(x)
  a <- a[setdiff(names(a), c("names", "row.names", "class", "groups"))]
  a[order(names(a))]
}

# User attributes of a tbl_gpu: its R attributes other than names/class.
f31_gpu_attrs <- function(x) {
  a <- attributes(x)
  a <- a[setdiff(names(a), c("names", "class"))]
  a[order(names(a))]
}

f31_cols <- function(x) lapply(dplyr::ungroup(x), identity)

# Run `pipeline` on `df` with dplyr and on tbl_gpu(df) (eager and lazy), and
# compare attributes on the tbl_gpu, class and attributes of the collected
# result, and the data.
expect_meta_like_dplyr <- function(df, pipeline, lazy = FALSE) {
  want <- pipeline(df)
  got <- pipeline(tbl_gpu(df, lazy = lazy))
  expect_s3_class(got, "tbl_gpu")
  expect_identical(f31_gpu_attrs(got), f31_frame_attrs(want))
  out <- collect(got)
  expect_identical(class(out), class(want))
  expect_identical(f31_frame_attrs(out), f31_frame_attrs(want))
  expect_identical(dplyr::group_vars(out), dplyr::group_vars(want))
  expect_equal(f31_cols(out), f31_cols(want))
  invisible(out)
}

f31_ungrouped_verbs <- list(
  filter = function(d) dplyr::filter(d, x > 1),
  filter_by = function(d) dplyr::filter(d, x > mean(x), .by = g),
  filter_out = function(d) dplyr::filter_out(d, x > 3),
  mutate = function(d) dplyr::mutate(d, y = x * 2),
  mutate_by = function(d) dplyr::mutate(d, m = mean(x), .by = g),
  transmute = function(d) dplyr::transmute(d, y = x + 1),
  select = function(d) dplyr::select(d, x),
  rename = function(d) dplyr::rename(d, w = x),
  relocate = function(d) dplyr::relocate(d, x),
  arrange = function(d) dplyr::arrange(d, x),
  slice = function(d) dplyr::slice(d, 2:3),
  slice_by = function(d) dplyr::slice(d, dplyr::n(), .by = g),
  slice_head = function(d) dplyr::slice_head(d, n = 2),
  slice_max = function(d) dplyr::slice_max(d, x, n = 2),
  distinct = function(d) dplyr::distinct(d, g),
  count = function(d) dplyr::count(d, g),
  add_count = function(d) dplyr::add_count(d, g),
  summarise = function(d) dplyr::summarise(d, s = sum(x)),
  summarise_by = function(d) dplyr::summarise(d, s = sum(x), .by = g),
  tally = function(d) dplyr::tally(d)
)

test_that("ungrouped verbs keep data.frame attributes and class like dplyr (eager)", {
  skip_if_no_gpu()
  for (nm in names(f31_ungrouped_verbs)) {
    expect_meta_like_dplyr(f31_df(), f31_ungrouped_verbs[[nm]])
  }
})

test_that("ungrouped verbs keep tibble attributes like dplyr (eager)", {
  skip_if_no_gpu()
  for (nm in names(f31_ungrouped_verbs)) {
    expect_meta_like_dplyr(f31_tbl(), f31_ungrouped_verbs[[nm]])
  }
})

test_that("ungrouped verbs keep data.frame attributes and class like dplyr (lazy)", {
  skip_if_no_gpu()
  for (nm in names(f31_ungrouped_verbs)) {
    expect_meta_like_dplyr(f31_df(), f31_ungrouped_verbs[[nm]], lazy = TRUE)
  }
})

test_that("summarise() drops attributes but keeps the bare data.frame class", {
  skip_if_no_gpu()
  out <- expect_meta_like_dplyr(f31_df(), function(d) dplyr::summarise(d, n = dplyr::n()))
  expect_identical(class(out), "data.frame")
  expect_null(attr(out, "foo"))
})

test_that("upload and collect round-trip attributes and the bare class", {
  skip_if_no_gpu()
  df <- f31_df()
  gt <- tbl_gpu(df)
  expect_identical(attr(gt, "foo"), "bar")
  expect_identical(attr(gt, "n"), 1:2)
  out <- collect(gt)
  expect_identical(class(out), "data.frame")
  expect_identical(f31_frame_attrs(out), f31_frame_attrs(df))
  expect_equal(f31_cols(out), f31_cols(df))

  tb <- collect(tbl_gpu(f31_tbl()))
  expect_identical(class(tb), c("tbl_df", "tbl", "data.frame"))
  expect_identical(attr(tb, "foo"), "bar")

  # A plain data.frame without attributes stays plain too.
  plain <- collect(tbl_gpu(data.frame(a = 1:3)))
  expect_identical(class(plain), "data.frame")
  expect_identical(names(attributes(plain))[order(names(attributes(plain)))],
                   c("class", "names", "row.names"))
})

test_that("attributes set on a tbl_gpu with attr<- propagate like dplyr's", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(data.frame(g = c(1, 1, 2), x = c(1, 2, 1)), lazy = lazy)
    attr(gt, "foo") <- "bar"
    out <- dplyr::filter(gt, x > mean(x), .by = g)
    expect_identical(attr(out, "foo"), "bar")
    out <- dplyr::mutate(gt, x = mean(x), .by = g)
    expect_identical(attr(out, "foo"), "bar")
    out <- dplyr::slice(gt, dplyr::n(), .by = g)
    expect_identical(attr(out, "foo"), "bar")
    out <- dplyr::select(gt, x)
    expect_identical(attr(out, "foo"), "bar")
    expect_identical(attr(collect(out), "foo"), "bar")
    out <- dplyr::count(gt, g)
    expect_identical(attr(out, "foo"), "bar")
    expect_identical(class(collect(out)), "data.frame")
  }
})

test_that("group_by() keeps attributes and gives a grouped tibble; grouped verbs drop them", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    for (mk in list(f31_df, f31_tbl)) {
      expect_meta_like_dplyr(mk(), function(d) dplyr::group_by(d, g), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::group_by(dplyr::group_by(d, g), x, .add = TRUE), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::group_by(d), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::ungroup(dplyr::group_by(d, g)), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::filter(dplyr::group_by(d, g), x > 1), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::mutate(dplyr::group_by(d, g), y = x), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::arrange(dplyr::group_by(d, g), x), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::summarise(dplyr::group_by(d, g), s = sum(x)), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::count(dplyr::group_by(d, g)), lazy = lazy)
      expect_meta_like_dplyr(mk(), function(d) dplyr::ungroup(d), lazy = lazy)
    }
  }
})

test_that("mutating and filtering joins keep x's attributes and class", {
  skip_if_no_gpu()
  joins <- list(
    left = function(a, b) dplyr::left_join(a, b, by = "g"),
    inner = function(a, b) dplyr::inner_join(a, b, by = "g"),
    right = function(a, b) dplyr::right_join(a, b, by = "g"),
    full = function(a, b) dplyr::full_join(a, b, by = "g"),
    semi = function(a, b) dplyr::semi_join(a, b, by = "g"),
    anti = function(a, b) dplyr::anti_join(a, b, by = "g"),
    cross = function(a, b) dplyr::cross_join(a, b)
  )
  for (lazy in c(FALSE, TRUE)) {
    for (nm in names(joins)) {
      j <- joins[[nm]]
      # x is a data.frame with attributes
      expect_meta_like_dplyr(f31_df(), function(d) {
        j(d, if (is_tbl_gpu(d)) tbl_gpu(f31_y(), lazy = lazy) else f31_y())
      }, lazy = lazy)
      # x is the data.frame y, the right side a tibble: y's attributes and class win
      tb <- f31_tbl()
      expect_meta_like_dplyr(f31_y(), function(d) {
        j(d, if (is_tbl_gpu(d)) tbl_gpu(tb, lazy = lazy) else tb)
      }, lazy = lazy)
    }
    # grouped x: dplyr drops the attributes
    expect_meta_like_dplyr(f31_df(), function(d) {
      dplyr::left_join(dplyr::group_by(d, g), if (is_tbl_gpu(d)) tbl_gpu(f31_y(), lazy = lazy) else f31_y(), by = "g")
    }, lazy = lazy)
  }
})

test_that("nest_join() keeps x's attributes and y's inside the list column", {
  skip_if_no_gpu()
  df1 <- data.frame(x = c(1, 2), y = c(3, 4))
  attr(df1, "foo") <- 1
  df2 <- data.frame(x = c(1, 2), z = c(3, 4))
  attr(df2, "foo") <- 2
  # dplyr names the list column after the `y` expression, so name it here.
  out <- nest_join(tbl_gpu(df1), tbl_gpu(df2), by = "x", name = "df2")
  expect_identical(attr(out, "foo"), 1)
  inner <- if (is_tbl_gpu(out)) collect(out)$df2 else out$df2
  expect_identical(attr(inner[[1]], "foo"), 2)
})

test_that("bind_rows() and bind_cols() take the first input's attributes and class", {
  skip_if_no_gpu()
  df <- f31_df()
  y <- f31_y()
  tb <- f31_tbl()
  check <- function(got, want) {
    expect_identical(f31_gpu_attrs(got), f31_frame_attrs(want))
    out <- collect(got)
    expect_identical(class(out), class(want))
    expect_identical(f31_frame_attrs(out), f31_frame_attrs(want))
    expect_equal(f31_cols(out), f31_cols(want))
  }
  for (lazy in c(FALSE, TRUE)) {
    up <- function(d) tbl_gpu(d, lazy = lazy)
    check(bind_rows(up(df), up(y)), dplyr::bind_rows(df, y))
    check(bind_rows(up(y), up(df)), dplyr::bind_rows(y, df))
    check(bind_rows(up(tb), up(df)), dplyr::bind_rows(tb, df))
    d1 <- data.frame(a = 1:4, b = c(1L, 1L, 2L, 2L))
    check(bind_cols(up(d1), up(df)), dplyr::bind_cols(d1, df))
    check(bind_cols(up(df), up(d1)), dplyr::bind_cols(df, d1))
    check(bind_cols(up(tb), up(d1)), dplyr::bind_cols(tb, d1))
  }
  expect_identical(class(collect(bind_cols(tbl_gpu(df), tbl_gpu(df)))), "data.frame")
})

test_that("compute(), as_lazy() and as_eager() keep attributes and class", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f31_df(), lazy = TRUE) |> dplyr::filter(x > 1)
  for (out in list(compute(gt), as_eager(gt), as_lazy(as_eager(gt)))) {
    expect_identical(f31_gpu_attrs(out), f31_frame_attrs(f31_df()))
    col <- collect(out)
    expect_identical(class(col), "data.frame")
    expect_identical(attr(col, "foo"), "bar")
  }
})

test_that("CPU fallback verbs keep attributes and class as dplyr does", {
  skip_if_no_gpu()
  df <- f31_df()
  y <- data.frame(g = 9, x = 9)
  for (lazy in c(FALSE, TRUE)) {
    expect_meta_like_dplyr(df, function(d) {
      dplyr::rows_append(d, if (is_tbl_gpu(d)) tbl_gpu(y, lazy = lazy) else y)
    }, lazy = lazy)
  }
})

test_that("zero-row inputs keep attributes and class", {
  skip_if_no_gpu()
  df <- f31_df()[0, ]
  attr(df, "foo") <- "bar"
  for (lazy in c(FALSE, TRUE)) {
    expect_meta_like_dplyr(df, function(d) dplyr::filter(d, x > 1), lazy = lazy)
    expect_meta_like_dplyr(df, function(d) dplyr::mutate(d, y = x), lazy = lazy)
    expect_meta_like_dplyr(df, function(d) dplyr::summarise(d, n = dplyr::n()), lazy = lazy)
  }
})

test_that("uploading a grouped_df does not store its groups as an attribute", {
  skip_if_no_gpu()
  gdf <- dplyr::group_by(f31_tbl(), g)
  gt <- tbl_gpu(gdf)
  expect_null(attr(gt, "groups"))
  expect_identical(attr(gt, "foo"), "bar")
  out <- collect(gt)
  expect_identical(class(out), class(gdf))
  expect_identical(f31_frame_attrs(out), f31_frame_attrs(gdf))
})

test_that("as_tibble() and as.data.frame() of a tbl_gpu keep attributes like on the data frame", {
  skip_if_no_gpu()
  df <- f31_df()
  gt <- tbl_gpu(df)
  expect_identical(f31_frame_attrs(tibble::as_tibble(gt)), f31_frame_attrs(tibble::as_tibble(df)))
  expect_identical(class(tibble::as_tibble(gt)), c("tbl_df", "tbl", "data.frame"))
  expect_identical(f31_frame_attrs(as.data.frame(gt)), f31_frame_attrs(as.data.frame(df)))
  expect_identical(class(as.data.frame(gt)), "data.frame")
})

test_that("group_data() of an ungrouped data.frame is a bare data.frame", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    df <- data.frame(x = 1:3)
    gd <- dplyr::group_data(tbl_gpu(df, lazy = lazy))
    expect_identical(class(gd), class(dplyr::group_data(df)))
    expect_equal(gd$.rows, dplyr::group_data(df)$.rows)
    tb <- tibble::tibble(x = 1:3)
    expect_identical(class(dplyr::group_data(tbl_gpu(tb, lazy = lazy))), class(dplyr::group_data(tb)))
    gg <- dplyr::group_by(tbl_gpu(df, lazy = lazy), x)
    expect_identical(class(dplyr::group_data(gg)), class(dplyr::group_data(dplyr::group_by(df, x))))
  }
})

test_that("tbl_sum() reports a tibble with its groups for a grouped table", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 1, 2, 2))
  expect_identical(
    pillar::tbl_sum(dplyr::group_by(tbl_gpu(df), x)),
    pillar::tbl_sum(dplyr::group_by(df, x))
  )
  tb <- tibble::tibble(x = c(1, 1, 2, 2))
  expect_identical(pillar::tbl_sum(tbl_gpu(tb)), pillar::tbl_sum(tb))
  sum_df <- pillar::tbl_sum(tbl_gpu(df))
  expect_identical(names(sum_df), "A data frame")
  expect_identical(unname(sum_df), unname(pillar::tbl_sum(tb)))
})

test_that("with_groups() restores the original class and attributes", {
  skip_if_no_gpu()
  df <- f31_df()
  out <- dplyr::with_groups(tbl_gpu(df), g, dplyr::mutate, m = mean(x))
  want <- dplyr::with_groups(df, g, dplyr::mutate, m = mean(x))
  expect_identical(dplyr::group_vars(out), character())
  col <- collect(out)
  expect_identical(class(col), class(want))
  expect_identical(f31_frame_attrs(col), f31_frame_attrs(want))
  expect_equal(f31_cols(col), f31_cols(want))
  gf <- dplyr::group_by(tbl_gpu(df), g)
  expect_identical(dplyr::group_vars(dplyr::with_groups(gf, x, dplyr::mutate, m = mean(x))), "g")
})

test_that("regrouping a grouped table follows dplyr: partial ungroup keeps, computed or empty group_by drops", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), h = c(1, 2, 2), x = c(1, 2, 3))
  attr(df, "foo") <- "bar"
  gh <- function(d) dplyr::group_by(d, g, h)
  for (lazy in c(FALSE, TRUE)) {
    expect_meta_like_dplyr(df, function(d) dplyr::group_by(gh(d)), lazy = lazy)
    expect_meta_like_dplyr(df, function(d) dplyr::ungroup(gh(d), h), lazy = lazy)
    expect_meta_like_dplyr(df, function(d) dplyr::group_by(gh(d), k = x * 2, .add = TRUE), lazy = lazy)
    expect_meta_like_dplyr(df, function(d) dplyr::group_by(gh(d), x, .add = TRUE), lazy = lazy)
    expect_meta_like_dplyr(df, function(d) dplyr::group_by(gh(d), x), lazy = lazy)
  }
})

test_that("as_tibble() and as.data.frame() of a grouped tbl_gpu drop attributes like dplyr", {
  skip_if_no_gpu()
  df <- f31_df()
  gt <- dplyr::group_by(tbl_gpu(df), g)
  gd <- dplyr::group_by(df, g)
  expect_identical(f31_frame_attrs(tibble::as_tibble(gt)), f31_frame_attrs(tibble::as_tibble(gd)))
  expect_identical(class(tibble::as_tibble(gt)), class(tibble::as_tibble(gd)))
  expect_identical(f31_frame_attrs(as.data.frame(gt)), f31_frame_attrs(as.data.frame(gd)))
  expect_identical(class(as.data.frame(gt)), class(as.data.frame(gd)))
})

test_that("vec_cast() to a tibble or grouped table takes the target's class", {
  skip_if_no_gpu()
  df <- data.frame(g = 1:2, v = 3:4)
  tb <- tibble::tibble(g = 1:2, v = 3:4)
  x <- tbl_gpu(df)
  to_t <- tbl_gpu(tb)
  to_g <- dplyr::group_by(to_t, g)
  expect_identical(class(collect(vctrs::vec_cast(x, to_t))), class(vctrs::vec_cast(df, tb)))
  expect_identical(class(collect(vctrs::vec_cast(x, to_g))),
                   class(vctrs::vec_cast(df, dplyr::group_by(tb, g))))
})
