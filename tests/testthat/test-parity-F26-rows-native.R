# F26 acceptance tests: rows_insert/append/update/patch/upsert/delete run
# natively (no CPU fallback) for the supported shapes.
#
# dplyr 1.2.1 contract (dplyr R/rows.R, verified with tools/r on tibbles):
# - Row order: x's rows keep their order; rows added from y (append, insert,
#   upsert's unmatched rows) follow, in y's order. update/patch keep x's row
#   count; delete returns the unmatched x rows in x order.
# - Columns: x's names, order and types. y must be a subset of x's columns,
#   in any order; x columns missing from y are NA in appended rows and left
#   alone by update/patch. y values are cast to x's type (vec_cast).
# - Keys: matched with vec_match()/vec_in(), so NA matches NA. Multi-column
#   `by` matches on all columns. x keys may be duplicated (every copy is
#   updated/deleted); y keys may be duplicated for insert/append/delete.
# - update overwrites matched rows with y's values, NA included; patch only
#   fills x's NA values (coalesce(x, y)).
# - `by = NULL` uses y's first column with a "Matching, by = ..." message
#   (dplyr_message_matching_by); rows_delete() with extra y columns informs
#   "Ignoring extra `y` columns: ..." (dplyr_message_delete_extra_cols).
# - Groups and frame attributes come from x (dplyr_reconstruct).
#
# Native contract tested here with options(cuplyr.fallback = "error"), in all
# four x/y eager/lazy combinations: logical, integer, double, character, Date
# and POSIXct columns of the same type in x and y, plus the lossless
# widenings integer -> double, logical -> integer and logical -> double;
# NA keys, multi-column `by`, 0-row x or y, grouped x and y, pending lazy ops,
# a bare data.frame x with attributes, `by = NULL`.
# Shapes outside it (see test-parity-F26-rows-errors.R) must still match
# dplyr, through the fallback if needed.

y_new <- function() {
  tibble::tibble(
    id  = c(6L, NA, 7L),
    k2  = c("z", "y", "x"),
    lg  = c(FALSE, TRUE, NA),
    int = c(60L, 70L, NA),
    dbl = c(6.5, NA, 7.5),
    chr = c(NA, "u", "v"),
    dt  = as.Date("2024-02-01") + c(0, NA, 2),
    ts  = as.POSIXct("2024-02-01 08:00:00", tz = "UTC") + c(NA, 1, 2)
  )
}

# --- rows_append() -----------------------------------------------------------

test_that("rows_append() is native: all types, subset and reordered y columns", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, y_new(), function(x, y) rows_append(x, y))
  f26_check(x, tibble::tibble(chr = c("new", NA), id = c(8L, NA), ts = as.POSIXct(NA, tz = "UTC")),
            function(x, y) rows_append(x, y))
})

test_that("rows_append() is native for lossless widenings and 0-row inputs", {
  skip_if_no_gpu()
  x <- f26_x()
  # integer -> double, logical -> integer, logical -> double
  f26_check(x, tibble::tibble(id = 9L, dbl = 2L, int = TRUE, lg = NA),
            function(x, y) rows_append(x, y))
  f26_check(x, tibble::tibble(dbl = c(FALSE, NA, TRUE)), function(x, y) rows_append(x, y))
  f26_check(x, y_new()[0, ], function(x, y) rows_append(x, y))
  f26_check(x[0, ], y_new(), function(x, y) rows_append(x, y))
  f26_check(x[0, ], y_new()[0, ], function(x, y) rows_append(x, y))
})

test_that("rows_append() takes groups from x, ignoring y's", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, y_new(), function(x, y) rows_append(group_by(x, k2), y))
  f26_check(x, y_new(), function(x, y) rows_append(group_by(x, k2, lg, .drop = FALSE), y))
  f26_check(x, y_new(), function(x, y) rows_append(x, group_by(y, k2)))
})

# --- rows_insert() -----------------------------------------------------------

test_that("rows_insert() is native: new keys, duplicated keys, conflict = 'ignore'", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(6L, 7L), chr = c("u", "v")),
            function(x, y) rows_insert(x, y, by = "id"))
  # y keys may repeat
  f26_check(x, tibble::tibble(id = c(8L, 8L), dbl = c(1, 2)),
            function(x, y) rows_insert(x, y, by = "id"))
  # x keys may repeat
  x2 <- f26_x()
  x2$id <- c(1L, 1L, 2L, 2L, NA)
  f26_check(x2, tibble::tibble(id = 3L), function(x, y) rows_insert(x, y, by = "id"))
  # conflict = "ignore": 2 and NA (NA matches NA) already exist in x
  f26_check(x, y_new() |> dplyr::mutate(id = c(2L, NA, 9L)),
            function(x, y) rows_insert(x, y, by = "id", conflict = "ignore"))
  # every y row conflicts
  f26_check(x, tibble::tibble(id = c(1L, 2L)),
            function(x, y) rows_insert(x, y, by = "id", conflict = "ignore"))
})

test_that("rows_insert() is native for multi-column, character and Date keys", {
  skip_if_no_gpu()
  x <- f26_x()
  # (1, "b") is new although id 1 exists; (1, "a") and (NA, "c") conflict
  y <- tibble::tibble(id = c(1L, 1L, NA, NA), k2 = c("b", "a", "c", "c"), dbl = c(1, 2, 3, 4))
  f26_check(x, y, function(x, y) rows_insert(x, y, by = c("id", "k2"), conflict = "ignore"))
  f26_check(x, tibble::tibble(chr = c("q", "w", NA), int = 1:3),
            function(x, y) rows_insert(x, y, by = "chr", conflict = "ignore"))
  f26_check(x, tibble::tibble(dt = as.Date("2024-01-01") + c(1, 20, NA)),
            function(x, y) rows_insert(x, y, by = "dt", conflict = "ignore"))
  f26_check(x, tibble::tibble(id = 11L, k2 = "zz"),
            function(x, y) rows_insert(group_by(x, k2), y, by = "id"))
  f26_check(x[0, ], tibble::tibble(id = 1:2, dbl = c(1, 2)),
            function(x, y) rows_insert(x, y, by = "id"))
})

# --- rows_update() -----------------------------------------------------------

test_that("rows_update() is native: overwrites matched rows, NA included", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(2L, 5L), dbl = c(20, 50), chr = c("Q", NA)),
            function(x, y) rows_update(x, y, by = "id"))
  # every type, reordered columns, NA key matching NA
  y <- y_new() |> dplyr::mutate(id = c(3L, NA, 1L)) |> dplyr::select(ts, dt, chr, id, dbl, int, lg, k2)
  f26_check(x, y, function(x, y) rows_update(x, y, by = "id"))
  # y with only the key column changes nothing
  f26_check(x, tibble::tibble(id = c(1L, 3L)), function(x, y) rows_update(x, y, by = "id"))
  # integer -> double, logical -> integer widening of values
  f26_check(x, tibble::tibble(id = 1L, dbl = 7L, int = FALSE),
            function(x, y) rows_update(x, y, by = "id"))
})

test_that("rows_update() is native: unmatched = 'ignore', multi-column keys, duplicated x keys", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(99L, 2L, 98L), int = c(1L, 2L, 3L)),
            function(x, y) rows_update(x, y, by = "id", unmatched = "ignore"))
  f26_check(x, tibble::tibble(id = c(1L, 3L, NA), k2 = c("a", "a", "c"), chr = c("A", "C", "N")),
            function(x, y) rows_update(x, y, by = c("id", "k2")))
  x2 <- f26_x()
  x2$id <- c(1L, 2L, 1L, 3L, 1L)
  f26_check(x2, tibble::tibble(id = c(1L, 3L), dbl = c(99, 88)),
            function(x, y) rows_update(x, y, by = "id"))
  f26_check(x, tibble::tibble(chr = c("s", "p"), int = c(-1L, -2L)),
            function(x, y) rows_update(x, y, by = "chr"))
  f26_check(x, tibble::tibble(id = integer(), dbl = double()),
            function(x, y) rows_update(x, y, by = "id"))
  f26_check(x[0, ], tibble::tibble(id = integer(), dbl = double()),
            function(x, y) rows_update(x, y, by = "id"))
})

test_that("rows_update() keeps x's groups, also when a group column is updated", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(1L, 2L), k2 = c("b", "new")),
            function(x, y) rows_update(group_by(x, k2), y, by = "id"))
  f26_check(x, tibble::tibble(id = 5L, dbl = 0),
            function(x, y) rows_update(group_by(x, k2, .drop = FALSE), y, by = "id"))
})

# --- rows_patch() ------------------------------------------------------------

test_that("rows_patch() is native: fills only x's NA values", {
  skip_if_no_gpu()
  x <- f26_x()
  # row 2's dbl is NA (patched), row 5's dbl is NA too; row 1's chr is set (kept)
  f26_check(x, tibble::tibble(id = c(2L, 5L, 1L), dbl = c(20, NA, 10), chr = c("Q", "T", "P")),
            function(x, y) rows_patch(x, y, by = "id"))
  y <- y_new() |> dplyr::mutate(id = c(3L, NA, 2L)) |> dplyr::select(ts, dt, chr, id, dbl, int, lg)
  f26_check(x, y, function(x, y) rows_patch(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = 2L, int = TRUE, dbl = 3L),
            function(x, y) rows_patch(x, y, by = "id"))
})

test_that("rows_patch() is native: unmatched = 'ignore', multi-column keys, duplicated x keys, groups", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(99L, 2L), int = c(1L, 2L)),
            function(x, y) rows_patch(x, y, by = "id", unmatched = "ignore"))
  f26_check(x, tibble::tibble(id = c(3L, 5L), k2 = c("a", NA), chr = c("C", "T"), dbl = c(-3, -5)),
            function(x, y) rows_patch(x, y, by = c("id", "k2")))
  x2 <- f26_x()
  x2$id <- c(1L, 2L, 1L, 3L, 1L)
  f26_check(x2, tibble::tibble(id = 1L, dbl = 99, lg = FALSE),
            function(x, y) rows_patch(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = 2L, k2 = "new", lg = TRUE),
            function(x, y) rows_patch(group_by(x, k2), y, by = "id"))
  f26_check(x, tibble::tibble(id = integer(), int = integer()),
            function(x, y) rows_patch(x, y, by = "id"))
})

# --- rows_upsert() -----------------------------------------------------------

test_that("rows_upsert() is native: updates matched rows, appends the rest in y order", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(9L, 2L, 8L, NA), dbl = c(9, 2, 8, 0), chr = c("n", NA, "m", "na")),
            function(x, y) rows_upsert(x, y, by = "id"))
  f26_check(x, y_new() |> dplyr::mutate(id = c(1L, 12L, 3L)),
            function(x, y) rows_upsert(x, y, by = "id"))
  # y with only keys: matched rows unchanged, new keys appended with NA values
  f26_check(x, tibble::tibble(id = c(1L, 13L)), function(x, y) rows_upsert(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = c(1L, 14L), dbl = c(5L, 6L)),
            function(x, y) rows_upsert(x, y, by = "id"))
})

test_that("rows_upsert() is native: multi-column keys, duplicated x keys, 0-row inputs, groups", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(1L, 1L), k2 = c("a", "q"), int = c(-1L, -2L)),
            function(x, y) rows_upsert(x, y, by = c("id", "k2")))
  x2 <- f26_x()
  x2$id <- c(1L, 2L, 1L, 3L, 1L)
  f26_check(x2, tibble::tibble(id = c(1L, 4L), chr = c("one", "four")),
            function(x, y) rows_upsert(x, y, by = "id"))
  f26_check(x[0, ], tibble::tibble(id = c(1L, 2L), dbl = c(1, 2)),
            function(x, y) rows_upsert(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = integer(), dbl = double()),
            function(x, y) rows_upsert(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = c(2L, 20L), k2 = c("c", "new")),
            function(x, y) rows_upsert(group_by(x, k2), y, by = "id"))
})

# --- rows_delete() -----------------------------------------------------------

test_that("rows_delete() is native: NA keys, duplicated keys, unmatched = 'ignore'", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(2L, NA)), function(x, y) rows_delete(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = c(3L, 3L)), function(x, y) rows_delete(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = c(99L, 1L)),
            function(x, y) rows_delete(x, y, by = "id", unmatched = "ignore"))
  x2 <- f26_x()
  x2$id <- c(1L, 2L, 1L, 3L, 1L)
  f26_check(x2, tibble::tibble(id = 1L), function(x, y) rows_delete(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = c(1L, 3L, NA), k2 = c("a", "b", "c")),
            function(x, y) rows_delete(x, y, by = c("id", "k2"), unmatched = "ignore"))
  f26_check(x, tibble::tibble(id = integer()), function(x, y) rows_delete(x, y, by = "id"))
  f26_check(x, tibble::tibble(id = c(1L, 2L, 3L, NA, 5L)), function(x, y) rows_delete(x, y, by = "id"))
  f26_check(x, tibble::tibble(chr = c("q", NA)), function(x, y) rows_delete(x, y, by = "chr"))
  f26_check(x, tibble::tibble(id = 2L), function(x, y) rows_delete(group_by(x, k2), y, by = "id"))
})

test_that("rows_delete() with extra y columns informs once and stays native", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(2L, 3L), dbl = c(0, 0), chr = c("a", "b")),
            function(x, y) rows_delete(x, y, by = "id"))
})

# --- shared behaviour --------------------------------------------------------

test_that("by = NULL uses y's first column, with exactly one message", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(6L, NA), dbl = c(1, 2)),
            function(x, y) rows_insert(x, y, conflict = "ignore"))
  f26_check(x, tibble::tibble(id = c(1L, NA), dbl = c(1, 2)), function(x, y) rows_update(x, y))
  f26_check(x, tibble::tibble(id = c(1L, NA), dbl = c(1, 2)), function(x, y) rows_patch(x, y))
  f26_check(x, tibble::tibble(id = c(1L, 9L), dbl = c(1, 2)), function(x, y) rows_upsert(x, y))
  # two messages, in dplyr's order
  f26_check(x, tibble::tibble(id = c(1L, NA), dbl = c(1, 2)), function(x, y) rows_delete(x, y))
  f26_check(x, tibble::tibble(chr = c("q", "zz"), id = 1:2),
            function(x, y) rows_upsert(x, y))
})

test_that("pending lazy ops on x and y are honoured", {
  skip_if_no_gpu()
  x <- f26_x()
  y <- tibble::tibble(id = c(2L, 3L, 9L), dbl = c(1, 2, 3))
  f26_check(x, y, function(x, y) {
    rows_update(filter(x, !is.na(id)), filter(mutate(y, dbl = dbl * 2), id < 9L), by = "id")
  })
  f26_check(x, y, function(x, y) {
    rows_upsert(mutate(x, dbl = dbl + 1), mutate(y, dbl = -dbl), by = "id")
  })
  f26_check(x, y, function(x, y) {
    rows_delete(arrange(x, desc(id)), select(y, id), by = "id", unmatched = "ignore")
  })
  f26_check(x, y, function(x, y) {
    rows_append(select(x, id, dbl), filter(y, dbl > 1))
  })
})

test_that("a bare data.frame x keeps its class and attributes", {
  skip_if_no_gpu()
  x <- as.data.frame(f26_x())
  attr(x, "foo") <- "bar"
  y <- data.frame(id = c(2L, 9L), dbl = c(-2, -9))
  f26_check(x, y, function(x, y) rows_append(x, y))
  f26_check(x, y, function(x, y) rows_insert(x, y, by = "id", conflict = "ignore"))
  f26_check(x, y, function(x, y) rows_update(x, y, by = "id", unmatched = "ignore"))
  f26_check(x, y, function(x, y) rows_patch(x, y, by = "id", unmatched = "ignore"))
  f26_check(x, y, function(x, y) rows_upsert(x, y, by = "id"))
  f26_check(x, y, function(x, y) rows_delete(x, y, by = "id", unmatched = "ignore"))
})

# --- added after review round 1 ----------------------------------------------

test_that("rows_delete() on a .drop = FALSE grouped x keeps the groups it empties (review B1)", {
  skip_if_no_gpu()
  x <- f26_x()
  # deleting id 2 empties group "b"; dplyr_row_slice() keeps it when .drop = FALSE
  f26_check(x, tibble::tibble(id = 2L),
            function(x, y) rows_delete(group_by(x, k2, .drop = FALSE), y, by = "id"))
  f26_check(x, tibble::tibble(id = c(2L, 4L, NA)),
            function(x, y) rows_delete(group_by(x, k2, .drop = FALSE), y, by = "id",
                                       unmatched = "ignore"))
  # summarise() of a table with preserved empty groups runs on the CPU by
  # design (F33, summarise_impl()), so only the result is checked here
  f26_check(x, tibble::tibble(id = 2L),
            function(x, y) summarise(rows_delete(group_by(x, k2, .drop = FALSE), y, by = "id"),
                                     n = n()),
            native = FALSE)
  # .drop = TRUE drops it
  f26_check(x, tibble::tibble(id = 2L),
            function(x, y) rows_delete(group_by(x, k2), y, by = "id"))
})

test_that("rows_patch() fills a NaN computed on the GPU, like dplyr's coalesce() (review B2)", {
  skip_if_no_gpu()
  x <- tibble::tibble(id = 1:4, v = c(0, 1, NA, 0))
  # x's v becomes NaN, 1, NA, NaN: rows 1 and 3 are matched and filled, row 4
  # is unmatched and keeps its NaN
  f26_check(x, tibble::tibble(id = c(1L, 3L), v = c(5, 6)),
            function(x, y) rows_patch(mutate(x, v = v / v), y, by = "id"))
  # a NaN in y is missing too: dplyr's coalesce() gives NA (not NaN) where
  # both x and y are missing (review round 2), and x's NaN with y's NA is NA
  f26_check(x, tibble::tibble(id = c(3L, 2L), v = c(0, 7)),
            function(x, y) rows_patch(x, mutate(y, v = v / v), by = "id"))
  f26_check(x, tibble::tibble(id = c(1L, 2L, 3L), v = c(NA, 0, 0)),
            function(x, y) rows_patch(mutate(x, v = v / v), mutate(y, v = v / v), by = "id"))
  # rows_update() overwrites with y's NaN and leaves unmatched NaNs alone
  f26_check(x, tibble::tibble(id = c(2L, 3L), v = c(0, 8)),
            function(x, y) rows_update(mutate(x, v = v / v), mutate(y, v = v / v), by = "id"))
})
