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
# bind_rows: promotion to STRING (P2 - schema/data corruption bug)
#
# promote_types()/compute_unified_schema() treat STRING as the widest type,
# so binding a numeric/logical column against a character column of the
# same name declares a STRING schema. align_to_schema() used to silently
# `next` past STRING targets ("cudf::cast doesn't support string output"),
# leaving the schema claiming STRING while the GPU column stayed numeric.
# gpu_bind_rows_aligned() then rejects the physical type mismatch between
# the two tables' same-named column -- so the bug currently surfaces as a
# hard error ("Type mismatch at column ...: table 1 has type <N>, table 2
# has type 23 [STRING]"), not silent corruption, but the root cause is the
# same silent-skip.
#
# NOTE on oracle parity: under the installed dplyr/vctrs (dplyr 1.2.1 /
# vctrs 0.7.3), dplyr::bind_rows() no longer coerces mismatched numeric
# vs. character columns to character at all -- it throws a
# `vctrs_error_incompatible_type`. So there is no runnable dplyr oracle for
# these cases any more (this differs from older dplyr, which used to coerce
# with a warning). Expected values below are therefore hand-verified against
# R's own `as.character()`, which is what cudf's string converters are
# expected to match for integers/booleans and for "nice" (non-integer-valued)
# doubles.
# =============================================================================

test_that("dplyr::bind_rows() no longer coerces mismatched types (documents oracle divergence)", {
  df1 <- data.frame(x = 1:2)
  df2 <- data.frame(x = c("a", "b"), stringsAsFactors = FALSE)

  # This is not a cuplyr bug -- it pins down *why* we can't use
  # dplyr::bind_rows() as a live oracle for the STRING-promotion tests below.
  expect_error(dplyr::bind_rows(df1, df2), class = "vctrs_error_incompatible_type")
})

test_that("bind_rows promotes integer + character to STRING with correct values", {
  skip_if_no_gpu()

  df1 <- data.frame(x = 1:3)
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  result_tbl <- bind_rows(tbl_gpu(df1), tbl_gpu(df2))
  expect_equal(result_tbl$schema$types[[1]], "STRING")

  result <- result_tbl |> collect()

  expect_type(result$x, "character")
  expect_equal(result$x, c(as.character(1:3), "p", "q"))
})

test_that("bind_rows promotes double + character to STRING with correct values", {
  skip_if_no_gpu()

  # Non-whole-number values are used here so cudf's from_floats() output
  # matches R's as.character() exactly (see the dedicated divergence test
  # below for whole-number doubles, where cudf appends a trailing ".0").
  df1 <- data.frame(x = c(1.5, 2.25, 3.75))
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  result_tbl <- bind_rows(tbl_gpu(df1), tbl_gpu(df2))
  expect_equal(result_tbl$schema$types[[1]], "STRING")

  result <- result_tbl |> collect()

  expect_type(result$x, "character")
  expect_equal(result$x, c(as.character(c(1.5, 2.25, 3.75)), "p", "q"))
})

test_that("bind_rows STRING-casting a whole-number double diverges from R's as.character() (documented)", {
  skip_if_no_gpu()

  # cudf::strings::from_floats() always emits a decimal point (e.g. "4.0",
  # "100.0"), whereas R's as.character() drops it for whole numbers ("4",
  # "100"). This is a real, verified formatting divergence between cudf and
  # R; normalizing it away would require post-processing every cast float
  # string, which is out of scope for the STRING-promotion fix. We pin cudf's
  # actual behavior here instead of silently disagreeing with it.
  df1 <- data.frame(x = c(4, 100))
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

  expect_equal(result$x, c("4.0", "100.0", "p", "q"))
  # ... while R itself would have produced "4", "100":
  expect_equal(as.character(c(4, 100)), c("4", "100"))
})

test_that("bind_rows promotes logical + character to STRING with correct values", {
  skip_if_no_gpu()

  df1 <- data.frame(x = c(TRUE, FALSE, TRUE))
  df2 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)

  result_tbl <- bind_rows(tbl_gpu(df1), tbl_gpu(df2))
  expect_equal(result_tbl$schema$types[[1]], "STRING")

  result <- result_tbl |> collect()

  expect_type(result$x, "character")
  expect_equal(result$x, c("TRUE", "FALSE", "TRUE", "p", "q"))
})

test_that("bind_rows promotes character + integer to STRING regardless of table order", {
  skip_if_no_gpu()

  # Same as the integer+character case but with the character table first,
  # so the unified schema's declared type ("STRING", taken from table 1's
  # own column) still requires casting *table 2's* integer column.
  df1 <- data.frame(x = c("p", "q"), stringsAsFactors = FALSE)
  df2 <- data.frame(x = 1:3)

  result <- bind_rows(tbl_gpu(df1), tbl_gpu(df2)) |> collect()

  expect_type(result$x, "character")
  expect_equal(result$x, c("p", "q", as.character(1:3)))
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

test_that("bind_rows factor + integer currently errors (pinned; L4 will replace with dplyr-parity text)", {
  skip_if_no_gpu()

  # Real dplyr also errors here (`Can't combine ..1$x <factor<...>> and
  # ..2$x <integer>.`) -- cuplyr's own promote_types()/DICTIONARY32 ladder
  # already refuses to promote a factor against a numeric type, just with
  # cuplyr's own generic message rather than dplyr's exact text/class. This
  # test PINS today's behavior (an error, not a value) so a future L4 pass
  # (replacing this with dplyr's own vctrs-parity error) has a documented
  # starting point instead of silently changing behavior unnoticed.
  df1 <- data.frame(x = factor(c("a", "b")))
  df2 <- data.frame(x = 1:2)

  expect_error(
    bind_rows(tbl_gpu(df1), tbl_gpu(df2)),
    "Cannot promote incompatible types"
  )
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
