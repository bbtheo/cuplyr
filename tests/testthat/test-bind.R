# Tests for bind_rows and bind_cols operations

# Basic S3 wiring --------------------------------------------------------------

test_that("bind_rows and bind_cols register S3 methods for tbl_gpu", {
  expect_true(is.function(getS3method("bind_rows", "tbl_gpu", optional = TRUE)))
  expect_true(is.function(getS3method("bind_cols", "tbl_gpu", optional = TRUE)))
})

test_that("bind_rows delegates to dplyr for non-tbl_gpu inputs", {
  df1 <- data.frame(x = 1:2, y = c("a", "b"))
  df2 <- data.frame(x = 3:4, y = c("c", "d"))

  expect_equal(
    bind_rows(df1, df2),
    dplyr::bind_rows(df1, df2)
  )

  expect_equal(
    bind_rows(list(df1, df2)),
    dplyr::bind_rows(list(df1, df2))
  )
})

test_that("bind_cols delegates to dplyr for non-tbl_gpu inputs", {
  df1 <- data.frame(x = 1:2)
  df2 <- data.frame(y = c("a", "b"))

  expect_equal(
    bind_cols(df1, df2),
    dplyr::bind_cols(df1, df2)
  )

  expect_equal(
    bind_cols(list(df1, df2)),
    dplyr::bind_cols(list(df1, df2))
  )
})

# =============================================================================
# bind_rows: Stacks tables vertically (row concatenation)
# =============================================================================

test_that("bind_rows stacks two tables vertically", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3, b = c("x", "y", "z"))
 df2 <- data.frame(a = 4:6, b = c("p", "q", "r"))

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(nrow(result), 6)
 expect_equal(result$a, 1:6)
 expect_equal(result$b, c("x", "y", "z", "p", "q", "r"))
})

test_that("bind_rows stacks many tables vertically", {
 skip_if_no_gpu()

 dfs <- lapply(1:5, function(i) {
   data.frame(x = (i - 1) * 10 + 1:10, y = letters[(i - 1) * 2 + 1:2])
 })
 gpu_dfs <- lapply(dfs, tbl_gpu)

 result <- do.call(bind_rows, gpu_dfs) |> collect()

 expect_equal(nrow(result), 50)
 expect_equal(result$x, unlist(lapply(dfs, `[[`, "x")))
})

test_that("bind_rows with single table returns equivalent table", {
 skip_if_no_gpu()

 df <- data.frame(a = 1:5, b = c("a", "b", "c", "d", "e"), stringsAsFactors = FALSE)
 gpu_df <- tbl_gpu(df)

 result <- bind_rows(gpu_df) |> collect()

 expect_equal(result$a, df$a)
 expect_equal(result$b, df$b)
})

# =============================================================================
# bind_rows: Matches columns by name, not position
# =============================================================================

test_that("bind_rows matches columns by name when order differs", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3, b = 4:6)
 df2 <- data.frame(b = 7:9, a = 10:12)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(result$a, c(1, 2, 3, 10, 11, 12))
 expect_equal(result$b, c(4, 5, 6, 7, 8, 9))
})

test_that("bind_rows matches columns by name with three tables in different orders", {
 skip_if_no_gpu()

 df1 <- data.frame(x = 1:2, y = 3:4, z = 5:6)
 df2 <- data.frame(z = 7:8, x = 9:10, y = 11:12)
 df3 <- data.frame(y = 13:14, z = 15:16, x = 17:18)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3)) |> collect()

 expect_equal(result$x, c(1, 2, 9, 10, 17, 18))
 expect_equal(result$y, c(3, 4, 11, 12, 13, 14))
 expect_equal(result$z, c(5, 6, 7, 8, 15, 16))
})

test_that("bind_rows result column order follows first table", {
 skip_if_no_gpu()

 df1 <- data.frame(c = 1:2, a = 3:4, b = 5:6)
 df2 <- data.frame(a = 7:8, b = 9:10, c = 11:12)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 # Column order should match first table
 expect_equal(names(result), c("c", "a", "b"))
})

# =============================================================================
# bind_rows: Missing columns filled with NA
# =============================================================================

test_that("bind_rows fills missing columns with NA (second table missing column)", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3, b = 4:6)
 df2 <- data.frame(a = 7:9)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(nrow(result), 6)
 expect_equal(result$a, c(1:3, 7:9))
 expect_equal(result$b, c(4, 5, 6, NA, NA, NA))
})
test_that("bind_rows fills missing columns with NA (first table missing column)", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(a = 4:6, b = 7:9)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(nrow(result), 6)
 expect_equal(result$a, 1:6)
 expect_equal(result$b, c(NA, NA, NA, 7, 8, 9))
})

test_that("bind_rows fills missing columns with NA for multiple disjoint columns", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:2, b = 3:4)
 df2 <- data.frame(c = 5:6, d = 7:8)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(nrow(result), 4)
 expect_equal(result$a, c(1, 2, NA, NA))
 expect_equal(result$b, c(3, 4, NA, NA))
 expect_equal(result$c, c(NA, NA, 5, 6))
 expect_equal(result$d, c(NA, NA, 7, 8))
})

test_that("bind_rows fills NA for different column types", {
 skip_if_no_gpu()

 df1 <- data.frame(int_col = 1L:2L, dbl_col = 1.5:2.5, stringsAsFactors = FALSE)
 df2 <- data.frame(str_col = c("a", "b"), lgl_col = c(TRUE, FALSE),
                   stringsAsFactors = FALSE)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(result$int_col, c(1L, 2L, NA, NA))
 expect_equal(result$dbl_col, c(1.5, 2.5, NA, NA))
 expect_equal(result$str_col, c(NA, NA, "a", "b"))
 expect_equal(result$lgl_col, c(NA, NA, TRUE, FALSE))
})

# =============================================================================
# bind_rows: Type coercion for compatible types
# =============================================================================

test_that("bind_rows promotes integer to double", {
 skip_if_no_gpu()

 df1 <- data.frame(x = 1L:3L)
 df2 <- data.frame(x = c(4.5, 5.5, 6.5))

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_type(result$x, "double")
 expect_equal(result$x, c(1, 2, 3, 4.5, 5.5, 6.5))
})

test_that("bind_rows promotes logical to integer", {
 skip_if_no_gpu()

 df1 <- data.frame(x = c(TRUE, FALSE, TRUE))
 df2 <- data.frame(x = 4L:6L)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 # Logical should be promoted to integer (or double depending on implementation)
 expect_true(is.numeric(result$x))
 expect_equal(result$x[1:3], c(1, 0, 1))
 expect_equal(result$x[4:6], c(4, 5, 6))
})

test_that("bind_rows promotes logical to double", {
 skip_if_no_gpu()

 df1 <- data.frame(x = c(TRUE, FALSE))
 df2 <- data.frame(x = c(2.5, 3.5))

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_type(result$x, "double")
 expect_equal(result$x, c(1, 0, 2.5, 3.5))
})

# =============================================================================
# bind_rows: strict vctrs-style type compatibility (Phase 11 L4)
#
# The old "STRING is widest" rule (promote_types()) is RETIRED: a
# numeric/logical column against a character column of the same name is now
# a hard error, matching real dplyr/vctrs' own `vctrs_error_incompatible_type`
# EXACTLY (verified empirically against dplyr 1.2.1 / vctrs 0.7.3: the error
# text format is `` Can't combine `..1$x` <double> and `..2$x` <character>. ``
# -- see resolve_common_col_type()/abort_bind_rows_incompatible(), R/bind.R).
# `cudf::strings::from_floats()`'s own "4.0" vs. R's "4" formatting
# divergence (previously pinned in this file) is no longer reachable from
# bind_rows() at all: no schema unification can produce a numeric-source/
# STRING-target pair any more (see gpu_cast_to_string()'s updated header
# comment, src/ops_bind.cpp).
# =============================================================================

test_that("bind_rows errors identically to dplyr for integer + character", {
  skip_if_no_gpu()

  df1 <- data.frame(x = 1:3)
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  oracle_err <- tryCatch(dplyr::bind_rows(df1, df2), error = function(e) e)
  cuplyr_err <- tryCatch(bind_rows(tbl_gpu(df1), tbl_gpu(df2)), error = function(e) e)

  expect_s3_class(cuplyr_err, "error")
  expect_equal(conditionMessage(cuplyr_err), conditionMessage(oracle_err))
})

test_that("bind_rows errors identically to dplyr for double + character", {
  skip_if_no_gpu()

  df1 <- data.frame(x = c(1.5, 2.25, 3.75))
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  oracle_err <- tryCatch(dplyr::bind_rows(df1, df2), error = function(e) e)
  cuplyr_err <- tryCatch(bind_rows(tbl_gpu(df1), tbl_gpu(df2)), error = function(e) e)

  expect_equal(conditionMessage(cuplyr_err), conditionMessage(oracle_err))
})

test_that("bind_rows errors identically to dplyr for logical + character", {
  skip_if_no_gpu()

  df1 <- data.frame(x = c(TRUE, FALSE, TRUE))
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  oracle_err <- tryCatch(dplyr::bind_rows(df1, df2), error = function(e) e)
  cuplyr_err <- tryCatch(bind_rows(tbl_gpu(df1), tbl_gpu(df2)), error = function(e) e)

  expect_equal(conditionMessage(cuplyr_err), conditionMessage(oracle_err))
})

test_that("bind_rows errors identically to dplyr regardless of table order", {
  skip_if_no_gpu()

  # Character table FIRST this time -- the `..1$x`/`..2$x` labels always
  # follow the tables' own positional order in `...`, not "whichever side
  # happens to be numeric."
  df1 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)
  df2 <- data.frame(x = 1:3)

  oracle_err <- tryCatch(dplyr::bind_rows(df1, df2), error = function(e) e)
  cuplyr_err <- tryCatch(bind_rows(tbl_gpu(df1), tbl_gpu(df2)), error = function(e) e)

  expect_equal(conditionMessage(cuplyr_err), conditionMessage(oracle_err))
})

test_that("bind_rows reports the FIRST table (..1) against the first genuinely conflicting table across 3+ tables", {
  skip_if_no_gpu()

  # Verified empirically against real dplyr/vctrs (a documented quirk, not a
  # simplification on cuplyr's part): the error's LEFT-hand label is ALWAYS
  # the literal `..1`, even when table 1 doesn't establish the running type
  # itself (e.g. it's missing the conflicting column entirely) -- see
  # abort_bind_rows_incompatible()'s own doc comment, R/bind.R.
  df1 <- data.frame(x = 1)
  df2 <- data.frame(x = 2L)
  df3 <- data.frame(x = "a", stringsAsFactors = FALSE)

  oracle_err <- tryCatch(dplyr::bind_rows(df1, df2, df3), error = function(e) e)
  cuplyr_err <- tryCatch(
    bind_rows(tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3)),
    error = function(e) e
  )

  expect_equal(conditionMessage(cuplyr_err), conditionMessage(oracle_err))
  expect_match(conditionMessage(cuplyr_err), "\\.\\.1\\$x.*\\.\\.3\\$x")
})

test_that("bind_rows factor + character decodes to character labels (Phase 11 L2 Bug 3)", {
  skip_if_no_gpu()

  # dplyr's real contract (verified empirically: vctrs::vec_ptype2(factor,
  # character) -> character, and bind_rows(factor, character) decodes to
  # LABELS): factor's own physical GPU column is INT32 codes (per
  # CLAUDE.md's logical/physical type split), so cast_column() must decode
  # via gpu_decode_factor() (this column's OWN levels), not reuse the
  # numeric integer->string cast path -- that would silently emit codes
  # ("2") instead of labels ("m"), a previously-locked-down (now fixed)
  # silent-corruption bug.
  df1 <- data.frame(x = factor(c("m", "f", "m")))
  df2 <- data.frame(x = c("a", "b"), stringsAsFactors = FALSE)

  expected <- dplyr::bind_rows(df1, df2)

  result_tbl <- bind_rows(tbl_gpu(df1), tbl_gpu(df2))
  expect_equal(result_tbl$schema$types[[1]], "STRING")

  result <- result_tbl |> collect()
  expect_type(result$x, "character")
  expect_equal(result$x, expected$x)
})

test_that("bind_rows factor + factor with different levels unions levels (Phase 11 L2 Bug 2)", {
  skip_if_no_gpu()

  # dplyr's real contract (verified empirically:
  # vctrs::vec_c(factor("a"), factor("b")) -> factor c("a","b") with levels
  # c("a","b"), i.e. union(levels(x), levels(y)), x's order first). Before
  # this fix, bind_rows() concatenated the raw INT32 codes verbatim with no
  # remap AND dropped `schema$factor_levels` entirely -- collect() would
  # present the column as plain integers, not even a factor.
  df1 <- data.frame(x = factor(c("a", "b"), levels = c("a", "b")))
  df2 <- data.frame(x = factor(c("b", "c"), levels = c("b", "c")))

  expected <- dplyr::bind_rows(df1, df2)

  result_tbl <- bind_rows(tbl_gpu(df1), tbl_gpu(df2))
  expect_equal(result_tbl$schema$types[[1]], "DICTIONARY32")

  result <- result_tbl |> collect()
  expect_s3_class(result$x, "factor")
  expect_equal(levels(result$x), levels(expected$x))
  expect_equal(as.character(result$x), as.character(expected$x))
})

test_that("bind_rows factor + factor with identical levels is a no-op union (native, no remap needed)", {
  skip_if_no_gpu()

  lvls <- c("lo", "mid", "hi")
  df1 <- data.frame(x = factor(c("hi", "lo"), levels = lvls))
  df2 <- data.frame(x = factor(c("mid", "hi"), levels = lvls))

  expected <- dplyr::bind_rows(df1, df2)

  result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()
  expect_s3_class(result$x, "factor")
  expect_equal(levels(result$x), lvls)
  expect_equal(as.character(result$x), as.character(expected$x))
})

test_that("bind_rows factor + integer errors in dplyr's own message shape (Phase 11 L4)", {
  skip_if_no_gpu()

  # Real dplyr: `` Can't combine `..1$x` <factor<4d52a>> and `..2$x`
  # <integer>. `` -- the level-derived hash in the factor label can't be
  # byte-matched (bind_rows_type_label()'s own doc comment, R/bind.R;
  # mirrors set_op_type_label()'s identical, pre-existing limitation), so
  # this checks the message SHAPE (prefix/suffix around the un-matchable
  # hash) rather than full equality.
  df1 <- data.frame(x = factor(c("a", "b")))
  df2 <- data.frame(x = 1:2)

  oracle_err <- tryCatch(dplyr::bind_rows(df1, df2), error = function(e) e)
  expect_match(conditionMessage(oracle_err), "^Can't combine `\\.\\.1\\$x` <factor<.*>> and `\\.\\.2\\$x` <integer>\\.$")

  expect_error(
    bind_rows(tbl_gpu(df1), tbl_gpu(df2)),
    "Can't combine `\\.\\.1\\$x` <factor<\\.\\.\\.>> and `\\.\\.2\\$x` <integer>\\."
  )
})

# =============================================================================
# bind_rows: timestamp-mix (Phase 11 L3 -- native TIMESTAMP_DAYS <->
# TIMESTAMP_MICROSECONDS cast, gpu_cast_column())
# =============================================================================

test_that("bind_rows Date + Date is a same-type no-op", {
  skip_if_no_gpu()

  df1 <- data.frame(a = as.Date(c("2020-01-01", "2020-01-02")))
  df2 <- data.frame(a = as.Date("2020-01-03"))

  result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()
  oracle <- dplyr::bind_rows(df1, df2)
  expect_equal(result$a, oracle$a)
})

test_that("bind_rows Date + POSIXct promotes to POSIXct natively (Phase 11 L3, no fallback)", {
  skip_if_no_gpu()

  # Real dplyr: `vec_ptype2(Date, POSIXct)` -> POSIXct (verified
  # empirically). This used to be entirely unreachable for bind_rows() (a
  # raw C++ "Unsupported target type for casting" exception) -- now a
  # genuine `cudf::cast()` between timestamp granularities.
  df1 <- data.frame(a = as.Date("2020-01-01"))
  df2 <- data.frame(a = as.POSIXct("2020-01-02 10:00:00", tz = "UTC"))

  result_tbl <- bind_rows(tbl_gpu(df1), tbl_gpu(df2))
  expect_equal(result_tbl$schema$types[[1]], "TIMESTAMP_MICROSECONDS")

  result <- result_tbl |> collect()
  oracle <- dplyr::bind_rows(df1, df2)
  expect_s3_class(result$a, "POSIXct")
  expect_equal(as.numeric(result$a), as.numeric(oracle$a))
})

test_that("bind_rows POSIXct + Date (order swapped) also promotes to POSIXct natively", {
  skip_if_no_gpu()

  df1 <- data.frame(a = as.POSIXct("2020-01-02 10:00:00", tz = "UTC"))
  df2 <- data.frame(a = as.Date("2020-01-01"))

  result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()
  oracle <- dplyr::bind_rows(df1, df2)
  expect_s3_class(result$a, "POSIXct")
  expect_equal(as.numeric(result$a), as.numeric(oracle$a))
})

# =============================================================================
# bind_rows: .id parameter optionally adds a column identifying source table
# =============================================================================

test_that("bind_rows .id adds source identifier with named inputs", {
 skip_if_no_gpu()

 df1 <- data.frame(x = 1:2)
 df2 <- data.frame(x = 3:4)

 result <- bind_rows(first = tbl_gpu(df1), second = tbl_gpu(df2), .id = "source") |>
   collect()

 expect_true("source" %in% names(result))
 expect_equal(result$source, c("first", "first", "second", "second"))
 expect_equal(result$x, 1:4)
})

test_that("bind_rows .id uses numeric indices for unnamed inputs", {
 skip_if_no_gpu()

 df1 <- data.frame(x = 1:2)
 df2 <- data.frame(x = 3:4)
 df3 <- data.frame(x = 5:6)

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3), .id = "id") |>
   collect()

 expect_true("id" %in% names(result))
 expect_equal(result$id, c("1", "1", "2", "2", "3", "3"))
})

test_that("bind_rows .id column appears first", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:2, b = 3:4)
 df2 <- data.frame(a = 5:6, b = 7:8)

 result <- bind_rows(x = tbl_gpu(df1), y = tbl_gpu(df2), .id = "source") |>
   collect()

 expect_equal(names(result)[1], "source")
})

# =============================================================================
# bind_rows: Groups are discarded after binding
# =============================================================================

test_that("bind_rows discards groups from single grouped table", {
 skip_if_no_gpu()

 df <- data.frame(g = c("a", "a", "b", "b"), x = 1:4)
 gpu_grouped <- tbl_gpu(df) |> group_by(g)

 result <- bind_rows(gpu_grouped)

 expect_equal(result$groups, character(0))
})

test_that("bind_rows discards groups from multiple grouped tables", {
 skip_if_no_gpu()

 df1 <- data.frame(g = c("a", "a"), x = 1:2)
 df2 <- data.frame(g = c("b", "b"), x = 3:4)

 g1 <- tbl_gpu(df1) |> group_by(g)
 g2 <- tbl_gpu(df2) |> group_by(g)

 result <- bind_rows(g1, g2)

 expect_equal(result$groups, character(0))
})

test_that("bind_rows discards groups even with different grouping columns", {
 skip_if_no_gpu()

 df1 <- data.frame(g1 = c("a", "b"), x = 1:2)
 df2 <- data.frame(g2 = c("c", "d"), x = 3:4)

 g1 <- tbl_gpu(df1) |> group_by(g1)
 g2 <- tbl_gpu(df2) |> group_by(g2)

 result <- bind_rows(g1, g2)

 expect_equal(result$groups, character(0))
})

# =============================================================================
# bind_rows: Edge cases
# =============================================================================

test_that("bind_rows handles empty tables", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(a = integer(0))

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(result$a, 1:3)
})

test_that("bind_rows handles all empty tables", {
 skip_if_no_gpu()

 df1 <- data.frame(a = integer(0), b = character(0))
 df2 <- data.frame(a = integer(0), b = character(0))

 result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(nrow(result), 0)
 expect_equal(names(result), c("a", "b"))
})

test_that("bind_rows handles tables with list input", {
 skip_if_no_gpu()

 df1 <- data.frame(x = 1:2)
 df2 <- data.frame(x = 3:4)
 df3 <- data.frame(x = 5:6)

 result <- do.call(bind_rows, list(
   tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3)
 )) |> collect()

 expect_equal(result$x, 1:6)
})

test_that("bind_rows handles mixed tbl_gpu and data.frame", {
 skip_if_no_gpu()

 df1 <- data.frame(x = 1:2)
 df2 <- data.frame(x = 3:4)

 # Second input is plain data.frame - should be converted
 result <- bind_rows(tbl_gpu(df1), df2) |> collect()

 expect_equal(result$x, 1:4)
})

# =============================================================================
# bind_cols: Combines tables horizontally (column concatenation)
# =============================================================================

test_that("bind_cols combines two tables horizontally", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3, b = 4:6)
 df2 <- data.frame(c = 7:9, d = 10:12)

 result <- bind_cols(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(ncol(result), 4)
 expect_equal(names(result), c("a", "b", "c", "d"))
 expect_equal(result$a, 1:3)
 expect_equal(result$c, 7:9)
})

test_that("bind_cols combines many tables horizontally", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(b = 4:6)
 df3 <- data.frame(c = 7:9)
 df4 <- data.frame(d = 10:12)

 result <- bind_cols(tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3), tbl_gpu(df4)) |>
   collect()

 expect_equal(ncol(result), 4)
 expect_equal(names(result), c("a", "b", "c", "d"))
})

test_that("bind_cols with single table returns equivalent table", {
 skip_if_no_gpu()

 df <- data.frame(a = 1:5, b = 6:10)
 gpu_df <- tbl_gpu(df)

 result <- bind_cols(gpu_df) |> collect()

 expect_equal(result$a, df$a)
 expect_equal(result$b, df$b)
})

# =============================================================================
# bind_cols: All tables must have same row count
# =============================================================================

test_that("bind_cols errors when row counts differ", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(b = 1:5)

 expect_error(
   bind_cols(tbl_gpu(df1), tbl_gpu(df2)),
   "same number of rows|row count"
 )
})

test_that("bind_cols errors when first table has more rows", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:10)
 df2 <- data.frame(b = 1:5)

 expect_error(
   bind_cols(tbl_gpu(df1), tbl_gpu(df2)),
   "same number of rows|row count"
 )
})

test_that("bind_cols errors when any table in chain has different row count", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:5)
 df2 <- data.frame(b = 1:5)
 df3 <- data.frame(c = 1:3)  # Different!

 expect_error(
   bind_cols(tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3)),
   "same number of rows|row count"
 )
})

# =============================================================================
# bind_cols: Duplicate column names handled via .name_repair
# =============================================================================

test_that("bind_cols .name_repair='unique' makes duplicate names unique", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3, b = 4:6)
 df2 <- data.frame(a = 7:9, b = 10:12)

 result <- bind_cols(tbl_gpu(df1), tbl_gpu(df2), .name_repair = "unique") |>
   collect()

 expect_equal(ncol(result), 4)
 expect_false(any(duplicated(names(result))))
 # Names should be something like a...1, a...2 or a_1, a_2
 expect_true(all(c("a", "b") %in% substr(names(result), 1, 1)))
})

test_that("bind_cols .name_repair='check_unique' errors on duplicates", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(a = 4:6)

 expect_error(
   bind_cols(tbl_gpu(df1), tbl_gpu(df2), .name_repair = "check_unique"),
   "unique|duplicate"
 )
})

test_that("bind_cols .name_repair='minimal' keeps duplicate names", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(a = 4:6)

 result <- bind_cols(tbl_gpu(df1), tbl_gpu(df2), .name_repair = "minimal")

 # Check tbl_gpu has duplicate names
 expect_equal(names(result), c("a", "a"))

 # Note: collect() would fail because tibble doesn't allow duplicate names
 # This is expected behavior - duplicate names are preserved on GPU
 # but tibble enforces uniqueness
})

# =============================================================================
# bind_cols: Groups from first table are preserved
# =============================================================================

test_that("bind_cols preserves groups from first table", {
 skip_if_no_gpu()

 df1 <- data.frame(g = c("a", "a", "b"), x = 1:3)
 df2 <- data.frame(y = 4:6)

 g1 <- tbl_gpu(df1) |> group_by(g)

 result <- bind_cols(g1, tbl_gpu(df2))

 expect_equal(result$groups, "g")
})

test_that("bind_cols ignores groups from subsequent tables", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(g = c("x", "y", "z"), b = 4:6)

 g2 <- tbl_gpu(df2) |> group_by(g)

 result <- bind_cols(tbl_gpu(df1), g2)

 # Groups from second table should be ignored
 expect_equal(result$groups, character(0))
})

test_that("bind_cols preserves multiple groups from first table", {
 skip_if_no_gpu()

 df1 <- data.frame(g1 = c("a", "a", "b"), g2 = c(1, 2, 1), x = 1:3)
 df2 <- data.frame(y = 4:6)

 g1 <- tbl_gpu(df1) |> group_by(g1, g2)

 result <- bind_cols(g1, tbl_gpu(df2))

 expect_equal(result$groups, c("g1", "g2"))
})

# =============================================================================
# bind_cols: Edge cases
# =============================================================================

test_that("bind_cols handles empty tables (0 rows)", {
 skip_if_no_gpu()

 df1 <- data.frame(a = integer(0))
 df2 <- data.frame(b = character(0))

 result <- bind_cols(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

 expect_equal(nrow(result), 0)
 expect_equal(names(result), c("a", "b"))
})

test_that("bind_cols handles tables with list input", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(b = 4:6)
 df3 <- data.frame(c = 7:9)

 result <- do.call(bind_cols, list(
   tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3)
 )) |> collect()

 expect_equal(names(result), c("a", "b", "c"))
})

test_that("bind_cols handles mixed tbl_gpu and data.frame", {
 skip_if_no_gpu()

 df1 <- data.frame(a = 1:3)
 df2 <- data.frame(b = 4:6)

 # Second input is plain data.frame - should be converted
 result <- bind_cols(tbl_gpu(df1), df2) |> collect()

 expect_equal(names(result), c("a", "b"))
 expect_equal(result$a, 1:3)
 expect_equal(result$b, 4:6)
})

test_that("bind_cols handles different column types", {
 skip_if_no_gpu()

 df1 <- data.frame(int_col = 1L:3L, stringsAsFactors = FALSE)
 df2 <- data.frame(dbl_col = c(1.5, 2.5, 3.5), stringsAsFactors = FALSE)
 df3 <- data.frame(str_col = c("a", "b", "c"), stringsAsFactors = FALSE)
 df4 <- data.frame(lgl_col = c(TRUE, FALSE, TRUE), stringsAsFactors = FALSE)

 result <- bind_cols(
   tbl_gpu(df1), tbl_gpu(df2), tbl_gpu(df3), tbl_gpu(df4)
 ) |> collect()

 expect_equal(ncol(result), 4)
 expect_type(result$int_col, "integer")
 expect_type(result$dbl_col, "double")
 expect_type(result$str_col, "character")
 expect_type(result$lgl_col, "logical")
})

# =============================================================================
# Integration: bind_rows and bind_cols together
# =============================================================================

test_that("bind_cols result can be used with bind_rows", {
 skip_if_no_gpu()

 df1a <- data.frame(a = 1:2)
 df1b <- data.frame(b = 3:4)
 df2a <- data.frame(a = 5:6)
 df2b <- data.frame(b = 7:8)

 combined1 <- bind_cols(tbl_gpu(df1a), tbl_gpu(df1b))
 combined2 <- bind_cols(tbl_gpu(df2a), tbl_gpu(df2b))

 result <- bind_rows(combined1, combined2) |> collect()

 expect_equal(nrow(result), 4)
 expect_equal(result$a, c(1, 2, 5, 6))
 expect_equal(result$b, c(3, 4, 7, 8))
})

test_that("bind_rows result can be used with bind_cols", {
 skip_if_no_gpu()

 df1a <- data.frame(a = 1:2)
 df1b <- data.frame(a = 3:4)
 df2a <- data.frame(b = 5:6)
 df2b <- data.frame(b = 7:8)

 stacked1 <- bind_rows(tbl_gpu(df1a), tbl_gpu(df1b))
 stacked2 <- bind_rows(tbl_gpu(df2a), tbl_gpu(df2b))

 result <- bind_cols(stacked1, stacked2) |> collect()

 expect_equal(ncol(result), 2)
 expect_equal(result$a, 1:4)
 expect_equal(result$b, 5:8)
})
