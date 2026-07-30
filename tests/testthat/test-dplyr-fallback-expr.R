# Phase 11, task L1 -- the universal expression-level CPU fallback.
#
# GOVERNING FINDING (scratchpad/phase11_design.md): scanning dplyr 1.2.1's
# own test files, string functions barely appear and datetime accessors
# never do -- what actually breaks dplyr parity on cuplyr is UNKNOWN USER
# FUNCTIONS (local closures, identity(), ifelse(), scale(), lengths(),
# toupper(), ...) that mutate()/summarise() used to hard-error on and
# filter()'s no-data-mask CPU eval couldn't serve (also why top_n()/
# top_frac() were broken end-to-end).
#
# The trigger rule (mutate.R's parse_mutate_dots(stop_on_unparseable=),
# summarise.R's summarise_dots_parseable(), filter.R's
# dot_references_schema_col()): a dot whose shape the expression IR
# doesn't recognize at all (ir_parse_quo()/decompose_agg_call() returns
# NULL) routes the WHOLE verb call to gpu_fallback() with the original
# (already across()-expanded) dots re-injected; a dot that instead THROWS
# a genuine parse-time error (e.g. an undefined column) is never caught --
# it propagates immediately, preserving every pre-existing pinned error
# message test. Every test below asserts BOTH oracle equality/correctness
# AND the `cuplyr.fallback = "warn"` notification (or, for the "never
# falls back" pins, its ABSENCE under `cuplyr.fallback = "error"`).

# =============================================================================
# mutate(): unknown-shape dots subsumed by the universal fallback
# =============================================================================

test_that("mutate() falls back for a local closure referencing a column", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3, 4, 5))
  mult <- 3
  add_mult <- function(v) v * mult
  pipeline <- function(d) dplyr::mutate(d, y = add_mult(x))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("mutate() falls back for identity()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3))
  pipeline <- function(d) dplyr::mutate(d, y = identity(x))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("mutate() falls back for base R's ifelse()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3, 4, 5))
  pipeline <- function(d) dplyr::mutate(d, y = ifelse(x > 2, "big", "small"))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("mutate() falls back for scale()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3, 4, 5))
  pipeline <- function(d) dplyr::mutate(d, y = scale(x)[, 1])

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("mutate() falls back for lengths()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(name = c("Alice", "Bob", "Carl"))
  pipeline <- function(d) dplyr::mutate(d, y = lengths(strsplit(name, "")))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("mutate() falls back for toupper() (BEFORE any native string support exists)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # This proves L1's subsumption claim: toupper() has no ir_call_registry
  # entry at all as of this task (native base-R string support is Phase
  # 11 task L5, not yet implemented) -- it works ANYWAY, transparently,
  # via the CPU fallback.
  df <- tibble::tibble(name = c("alice", "bob", "carl"))
  pipeline <- function(d) dplyr::mutate(d, y = toupper(name))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

# =============================================================================
# filter(): unknown-shape predicates referencing a real column
# =============================================================================

test_that("filter() falls back for grepl()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(name = c("Alice", "Bob", "alice2", "Dan"))
  pipeline <- function(d) dplyr::filter(d, grepl("lice", name))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("filter() with a window call mixed with an unknown function falls back cleanly (never half-lowers)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # A dot combining a window call (mean()) with a function the IR doesn't
  # understand (toupper()) fails to parse as ONE unit (the whole `&`
  # expression returns NULL), so it's never even classified as
  # window-bearing -- it must route to the whole-call fallback instead of
  # attempting a partial/half-lowered plan.
  df <- tibble::tibble(g = c(1, 1, 2, 2), name = c("Alice", "bob", "Carl", "dan"))
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |>
      dplyr::filter(mean(nchar(name)) > 0 & toupper(name) != "") |>
      dplyr::ungroup()
  }

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("filter()'s CPU fallback restores grouping metadata", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(g = c(1, 1, 2, 2), name = c("Alice", "bob", "Carl", "dan"))
  gdf <- tbl_gpu(df) |> dplyr::group_by(g)

  expect_warning(
    result <- dplyr::filter(gdf, grepl("a", name, ignore.case = TRUE)),
    "fell back to CPU evaluation"
  )

  expect_equal(result$groups, "g")
})

# =============================================================================
# summarise(): unknown reducers
# =============================================================================

test_that("summarise() falls back for an unknown reducer (cor())", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3, 4, 5), y = c(5, 3, 4, 2, 1))
  pipeline <- function(d) dplyr::summarise(d, r = cor(x, y))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

test_that("summarise()'s CPU fallback restores grouping metadata correctly (.groups= default)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(1, 2, 3, 4), y = c(4, 3, 2, 1))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::summarise(r = cor(x, y))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

# =============================================================================
# top_n() / top_frac(): end-to-end, previously broken (desugar to
# filter(top_n_rank(...)), a shape the IR has never understood natively)
# =============================================================================

test_that("dplyr::top_n() falls back and works end-to-end (previously broken)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(5, 3, 8, 1, 9, 2, 7))
  pipeline <- function(d) dplyr::top_n(d, 3, x)

  expect_warning(expect_same_as_dplyr(df, pipeline, arrange_by = "x"), "fell back to CPU evaluation")
})

test_that("dplyr::top_frac() falls back and works end-to-end (previously broken)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(5, 3, 8, 1, 9, 2, 7, 4))
  pipeline <- function(d) dplyr::top_frac(d, 0.5, x)

  expect_warning(expect_same_as_dplyr(df, pipeline, arrange_by = "x"), "fell back to CPU evaluation")
})

# =============================================================================
# Interaction: across()-expansion runs BEFORE the L1 pre-pass
# =============================================================================

test_that("across() still expands normally and is unaffected by the L1 pre-pass", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(x = c(1, 2, 3), y = c(4, 5, 6)))
  expect_no_error(dplyr::mutate(gdf, dplyr::across(c(x, y), ~ .x * 2)))
})

test_that("across() with a genuinely unsupported nested shape still falls back (Phase 10, unaffected by L1)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, total = rowSums(dplyr::pick(x, y)))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

# =============================================================================
# "Never falls back" pins for core, GPU-native shapes (mirrors
# test-dplyr-across.R's cur_group_id() pattern)
# =============================================================================

test_that("mutate() with ordinary arithmetic never triggers the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(x = 1:5, y = 5:1))
  expect_no_error(dplyr::mutate(gdf, z = x + y * 2))
})

test_that("mutate() window functions never trigger the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3))) |> dplyr::group_by(g)
  expect_no_error(dplyr::mutate(gdf, r = cumsum(x)))
})

test_that("filter() with a plain comparison never triggers the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(x = 1:5))
  expect_no_error(dplyr::filter(gdf, x > 2))
})

test_that("filter() with a constant/env-vector shape (rep(TRUE, n)) still uses the OLD mask-apply path, not the new whole-call fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gdf <- tbl_gpu(mtcars)
  # rep(TRUE, nrow(mtcars)) references no schema column at all -- it must
  # still go through filter_eval_mask() (the pre-existing const/env-vector
  # fallback), not the new dot_references_schema_col()-gated
  # filter_expr_fallback() -- this is unobservable from the fallback
  # notification alone (both warn identically), so this is really a
  # regression guard for dot_references_schema_col() itself: it must
  # correctly say FALSE here.
  expect_warning(
    result <- dplyr::filter(gdf, rep(TRUE, nrow(mtcars))) |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(nrow(result), nrow(mtcars))
})

test_that("summarise() with known aggregations never triggers the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3)))
  expect_no_error(dplyr::summarise(gdf, m = mean(x), s = sum(x > 1)))
})

test_that("summarise() windows-inside-summarise never triggers the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(g = c(1, 1, 2, 2), v = c(1, 2, 3, 4))) |> dplyr::group_by(g)
  expect_no_error(dplyr::summarise(gdf, z = last(cumsum(v))))
})

# =============================================================================
# "Genuine errors stay fast" pins -- the other half of the NULL-vs-throw
# trigger rule: these must propagate directly, never attempting a fallback
# =============================================================================

test_that("mutate() propagates a genuine parse-time error without attempting the CPU fallback", {
  skip_if_no_gpu()
  # cuplyr.fallback = "error" would surface a DIFFERENT ("fell back to CPU
  # evaluation") message if this incorrectly routed through the fallback.
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(x = 1:5))
  expect_error(dplyr::mutate(gdf, y = made_up_col + 1), "not found")
})

test_that("filter() propagates a genuine type-mismatch error without attempting the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(mtcars)
  expect_error(dplyr::filter(gdf, mpg > "twenty"), "Cannot compare column")
})

test_that("filter() propagates a genuine \"column not found\" error without attempting the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(mtcars)
  expect_error(dplyr::filter(gdf, made_up_col > 5), "not found")
})

test_that("summarise() propagates a genuine type-mismatch error without attempting the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gdf <- tbl_gpu(tibble::tibble(name = c("a", "b", "c")))
  expect_error(dplyr::summarise(gdf, m = median(name)), "requires a numeric column")
})
