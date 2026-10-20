# F26 acceptance tests: the copy = TRUE rule, in_place, dplyr's errors and
# messages, and the shapes outside the native contract (which must still give
# dplyr's result, through the CPU fallback if needed).
#
# dplyr 1.2.1 contract (dplyr R/rows.R, verified with tools/r):
# - Every rows_*() data frame method calls auto_copy(x, y, copy). For a
#   tbl_gpu x and a y that is not a tbl_gpu, copy = FALSE raises dplyr's
#   "`x` and `y` must share the same src." rlang_error (as the GPU joins do
#   since F05); copy = TRUE uploads y.
# - in_place = TRUE is an error for data frames ("Data frames only support
#   `in_place = FALSE`.").
# - Errors compared here by class vector and conditionMessage(), raised by the
#   rows_*() call itself (also when x is lazy), with the messages dplyr emits
#   before them (e.g. "Matching, by = ...") emitted exactly once:
#   conflicts (insert), unmatched keys (update/patch/delete), duplicated y keys
#   (update/patch/upsert, NA duplicates included), lossy casts (double 4.5
#   into integer), incompatible casts (double into character, TRUE into
#   character, character key against integer key), y columns missing from x,
#   `by` columns missing, invalid `conflict`/`unmatched`/`by`.

all_verbs <- list(
  rows_insert = function(x, y, ...) rows_insert(x, y, by = "id", conflict = "ignore", ...),
  rows_append = function(x, y, ...) rows_append(x, y, ...),
  rows_update = function(x, y, ...) rows_update(x, y, by = "id", unmatched = "ignore", ...),
  rows_patch  = function(x, y, ...) rows_patch(x, y, by = "id", unmatched = "ignore", ...),
  rows_upsert = function(x, y, ...) rows_upsert(x, y, by = "id", ...),
  rows_delete = function(x, y, ...) rows_delete(x, y, by = "id", unmatched = "ignore", ...)
)

# A y every verb in all_verbs accepts against f26_x(): id 2 exists in x
# (row 2, whose dbl is NA), id 9 doesn't.
ok_y <- function() tibble::tibble(id = c(2L, 9L), dbl = c(0.25, 0.5))

# --- copy = TRUE -------------------------------------------------------------

test_that("a plain data frame y needs copy = TRUE, like the GPU joins", {
  skip_if_no_gpu()
  for (nm in names(all_verbs)) {
    verb <- all_verbs[[nm]]
    for (lazy in c(FALSE, TRUE)) {
      gx <- tbl_gpu(f26_x(), lazy = lazy)
      expect_error(verb(gx, ok_y()), class = "rlang_error",
                   regexp = "must share the same src", info = nm)
      expect_error(verb(gx, as.data.frame(ok_y())), class = "rlang_error",
                   regexp = "must share the same src", info = nm)
    }
  }
})

test_that("copy = TRUE uploads a plain y and the verb stays native", {
  skip_if_no_gpu()
  y_plain <- ok_y()
  for (nm in names(all_verbs)) {
    verb <- all_verbs[[nm]]
    f26_check(f26_x(), y_plain, function(x, y) verb(x, y_plain, copy = TRUE))
  }
})

test_that("in_place = TRUE is an error", {
  skip_if_no_gpu()
  for (nm in names(all_verbs)) {
    verb <- all_verbs[[nm]]
    for (lazy in c(FALSE, TRUE)) {
      expect_error(verb(tbl_gpu(f26_x(), lazy = lazy), tbl_gpu(ok_y()), in_place = TRUE),
                   regexp = "in_place", info = nm)
    }
  }
})

# --- data-dependent errors ---------------------------------------------------

test_that("rows_insert() conflicts raise dplyr's error", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = c(9L, 2L, NA), dbl = 1:3 / 2),
            function(x, y) rows_insert(x, y, by = "id"), native = FALSE)
  f26_check(x, tibble::tibble(id = c(1L, 1L), k2 = c("q", "a")),
            function(x, y) rows_insert(x, y, by = c("id", "k2")), native = FALSE)
  # message first, then the error
  f26_check(x, tibble::tibble(id = 1L), function(x, y) rows_insert(x, y), native = FALSE)
})

test_that("unmatched y keys raise dplyr's error in update, patch and delete", {
  skip_if_no_gpu()
  x <- f26_x()
  y <- tibble::tibble(id = c(2L, 99L, 98L), dbl = c(1, 2, 3))
  f26_check(x, y, function(x, y) rows_update(x, y, by = "id"), native = FALSE)
  f26_check(x, y, function(x, y) rows_patch(x, y, by = "id"), native = FALSE)
  f26_check(x, y, function(x, y) rows_delete(x, y, by = "id"), native = FALSE)
  # the extra-columns message is not emitted when delete errors first
  f26_check(x, y, function(x, y) rows_delete(x, y), native = FALSE)
  f26_check(x[0, ], tibble::tibble(id = 1L), function(x, y) rows_update(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(id = c(1L, 1L), k2 = c("a", "b")),
            function(x, y) rows_patch(x, y, by = c("id", "k2")), native = FALSE)
})

test_that("duplicated y keys raise dplyr's error in update, patch and upsert", {
  skip_if_no_gpu()
  x <- f26_x()
  y <- tibble::tibble(id = c(2L, 3L, 2L), dbl = c(1, 2, 3))
  f26_check(x, y, function(x, y) rows_update(x, y, by = "id"), native = FALSE)
  f26_check(x, y, function(x, y) rows_patch(x, y, by = "id"), native = FALSE)
  f26_check(x, y, function(x, y) rows_upsert(x, y, by = "id"), native = FALSE)
  # NA keys are duplicates of each other
  f26_check(x, tibble::tibble(id = c(NA, NA)), function(x, y) rows_upsert(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(id = c(1L, 1L), k2 = c("a", "a")),
            function(x, y) rows_update(x, y, by = c("id", "k2"), unmatched = "ignore"),
            native = FALSE)
})

test_that("casts y can't make losslessly raise dplyr's error", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = 4.5), function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(id = 9L, int = 2.5), function(x, y) rows_insert(x, y, by = "id"),
            native = FALSE)
  # an unmatched y key that can't become x's integer key
  f26_check(x, tibble::tibble(id = c(1, 1.5)), function(x, y) rows_upsert(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(id = 2L, int = 0.5), function(x, y) rows_update(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(chr = 1), function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(chr = TRUE), function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(dt = "2024-01-01"), function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(id = "2", dbl = 1), function(x, y) rows_update(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(id = "2"), function(x, y) rows_delete(x, y, by = "id"),
            native = FALSE)
})

# --- argument and column checks ----------------------------------------------

test_that("column and argument checks raise dplyr's errors", {
  skip_if_no_gpu()
  x <- f26_x()
  extra <- tibble::tibble(id = 9L, zzz = 1)
  f26_check(x, extra, function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, extra, function(x, y) rows_insert(x, y, by = "id"), native = FALSE)
  f26_check(x, extra, function(x, y) rows_update(x, y, by = "id"), native = FALSE)
  f26_check(x, extra, function(x, y) rows_upsert(x, y, by = "id"), native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_insert(x, y, by = "nope"), native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_update(x, y, by = c("id", "k2")), native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_delete(x, y, by = "chr"), native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_insert(x, y, by = "id", conflict = "foo"),
            native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_update(x, y, by = "id", unmatched = "foo"),
            native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_delete(x, y, by = "id", unmatched = 1),
            native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_update(x, y, by = 1), native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_patch(x, y, by = character()), native = FALSE)
  f26_check(x, ok_y(), function(x, y) rows_upsert(x, y, by = c(a = "id")), native = FALSE)
})

# --- shapes outside the native contract --------------------------------------

test_that("key and value casts outside the native widenings match dplyr", {
  skip_if_no_gpu()
  x <- f26_x()
  # double y key against integer x key: matched on the common type, x keeps integer
  f26_check(x, tibble::tibble(id = c(2, 3), dbl = c(-1, -2)),
            function(x, y) rows_update(x, y, by = "id"), native = FALSE)
  f26_check(x, tibble::tibble(id = c(2, 3.5)),
            function(x, y) rows_delete(x, y, by = "id", unmatched = "ignore"), native = FALSE)
  f26_check(x, tibble::tibble(id = c(1, 20), int = c(5, 6)),
            function(x, y) rows_upsert(x, y, by = "id"), native = FALSE)
  # integral doubles into an integer column
  f26_check(x, tibble::tibble(id = 9, int = 2), function(x, y) rows_append(x, y), native = FALSE)
  # an all-NA logical column into character / Date / POSIXct columns
  f26_check(x, tibble::tibble(id = 9L, chr = NA, dt = NA, ts = NA),
            function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(id = 2L, chr = NA), function(x, y) rows_update(x, y, by = "id"),
            native = FALSE)
})

test_that("factor columns match dplyr", {
  skip_if_no_gpu()
  x <- tibble::tibble(id = 1:4, f = factor(c("lo", "hi", NA, "lo"), levels = c("lo", "hi")),
                      v = c(1, 2, 3, 4))
  same <- tibble::tibble(id = c(2L, 9L), f = factor(c("lo", "hi"), levels = c("lo", "hi")))
  f26_check(x, same, function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, same, function(x, y) rows_upsert(x, y, by = "id"), native = FALSE)
  f26_check(x, same, function(x, y) rows_patch(x, y, by = "id", unmatched = "ignore"),
            native = FALSE)
  f26_check(x, tibble::tibble(f = factor("hi", levels = c("lo", "hi")), v = 0),
            function(x, y) rows_update(x, y, by = "f", unmatched = "ignore"), native = FALSE)
  f26_check(x, tibble::tibble(id = 3L, f = "hi"), function(x, y) rows_update(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(id = 3L, f = "zz"), function(x, y) rows_update(x, y, by = "id"),
            native = FALSE)
  f26_check(x, tibble::tibble(id = 5L, f = factor("hi", levels = c("hi", "lo"))),
            function(x, y) rows_insert(x, y, by = "id"), native = FALSE)
  f26_check(x, tibble::tibble(id = 5L, f = factor("new")),
            function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(f = factor(c("hi", NA), levels = c("lo", "hi", "mid"))),
            function(x, y) rows_delete(x, y, by = "f", unmatched = "ignore"), native = FALSE)
})

test_that("date-time casts and time zones match dplyr (x's zone wins)", {
  skip_if_no_gpu()
  x <- f26_x()
  ny <- tibble::tibble(id = c(2L, 8L),
                       ts = as.POSIXct("2024-03-01 09:30:00", tz = "America/New_York") + c(0, 1))
  f26_check(x, ny, function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, ny, function(x, y) rows_upsert(x, y, by = "id"), native = FALSE)
  f26_check(x, tibble::tibble(id = 8L, ts = as.Date("2024-05-01")),
            function(x, y) rows_append(x, y), native = FALSE)
  f26_check(x, tibble::tibble(id = 2L, dt = as.POSIXct("2024-05-01", tz = "UTC")),
            function(x, y) rows_update(x, y, by = "id"), native = FALSE)
})

test_that("a duplicated `by` column matches dplyr (review B3)", {
  skip_if_no_gpu()
  x <- f26_x()
  f26_check(x, tibble::tibble(id = 2L, dbl = 20), function(x, y) rows_update(x, y, by = c("id", "id")),
            native = FALSE)
  f26_check(x, tibble::tibble(id = 9L, dbl = 20), function(x, y) rows_insert(x, y, by = c("id", "id")),
            native = FALSE)
  f26_check(x, tibble::tibble(id = c(2L, 9L)), function(x, y) rows_upsert(x, y, by = c("id", "id")),
            native = FALSE)
  f26_check(x, tibble::tibble(id = 2L), function(x, y) rows_delete(x, y, by = c("id", "id")),
            native = FALSE)
})
