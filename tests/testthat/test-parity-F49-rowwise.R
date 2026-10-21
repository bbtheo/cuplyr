# F49 acceptance tests, tier A: rowwise() on the GPU -- representation and
# verb semantics. (Tier B, the horizontal-aggregate rewrites, is in
# test-parity-F49-rowwise-rewrites.R.)
#
# Contract (scratchpad/parity/features/F49.md, ## Design):
# - rowwise(<tbl_gpu>) returns a rowwise tbl_gpu, natively (no download, lazy
#   stays lazy); tbl_gpu(<rowwise_df>) uploads one, silently.
# - collect() returns dplyr's rowwise_df; as_tibble()/ungroup() drop it;
#   group_by() replaces it; the group metadata family has one group per row.
# - Every verb on a rowwise tbl_gpu gives dplyr 1.2.1's class, rowwise vars and
#   data. Verbs listed as native below must not fall back; anything else may
#   fall back (CPU, dplyr's own semantics), but never compute with the rowwise
#   vars as real groups.
# - Errors on a rowwise table are dplyr's (same condition class).
# - The conformance harness's cf_collect() reads a rowwise tbl_gpu as a
#   rowwise_df (tools/conformance-lib.R, authorised for F49).

f49_df <- function() {
  tibble::tibble(
    id = 1:6,
    g = c("a", "a", "b", "b", "c", "c"),
    i1 = c(1L, NA, 3L, -4L, 5L, 7L),
    i2 = c(2L, 5L, NA, 4L, 0L, 1L),
    d1 = c(1.5, NA, 2.25, -0.5, 1e10, 3),
    d2 = c(0.5, 1, NA, 3, -2, 0.25),
    l1 = c(TRUE, NA, FALSE, TRUE, FALSE, TRUE),
    s = c("p", "q", "r", "s", "t", "u"),
    f = factor(c("x", "y", "x", "z", "y", "x")),
    dt = as.Date("2020-01-01") + 0:5
  )
}

f49_col_types <- function(x) {
  vapply(x, function(col) paste(c(class(col), typeof(col)), collapse = "/"), character(1))
}

# Run `pipeline` on the plain tibble (dplyr oracle) and on a tbl_gpu. Requires
# the same class, group_vars(), column names/types and data (grouping
# structure included). `native = TRUE` runs the GPU side (verbs AND collect())
# under options(cuplyr.fallback = "error") and requires a tbl_gpu result
# (`tbl = FALSE` for verbs that return a plain object, e.g. as_tibble()).
f49_check <- function(df, pipeline, lazy = FALSE, native = TRUE, tbl = native) {
  withr::local_options(dplyr.summarise.inform = FALSE)
  oracle <- pipeline(tibble::as_tibble(df))
  gin <- tbl_gpu(df, lazy = lazy)
  mode <- if (native) "error" else getOption("cuplyr.fallback", "silent")
  got <- withr::with_options(list(cuplyr.fallback = mode), pipeline(gin))
  if (tbl) {
    expect_true(is_tbl_gpu(got))
  }
  if (is_tbl_gpu(got)) {
    if (native) {
      expect_identical(got$exec_mode, if (lazy) "lazy" else "eager")
    }
    got <- withr::with_options(list(cuplyr.fallback = mode), collect(got))
  }
  expect_identical(class(got), class(oracle))
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(oracle))
  expect_identical(f49_col_types(got), f49_col_types(oracle))
  expect_equal(got, oracle)
  invisible(got)
}

# The condition both sides raise, compared by class (and optionally message).
f49_same_error <- function(df, pipeline, lazy = FALSE, message = NULL) {
  want <- tryCatch(pipeline(tibble::as_tibble(df)), error = identity)
  expect_s3_class(want, "error")
  got <- tryCatch({
    out <- pipeline(tbl_gpu(df, lazy = lazy))
    if (is_tbl_gpu(out)) collect(out) else out
  }, error = identity)
  expect_s3_class(got, "error")
  expect_identical(class(got), class(want))
  if (!is.null(message)) {
    expect_match(conditionMessage(got), message, fixed = TRUE)
  }
}

# ---------------------------------------------------------------------------
# A1. rowwise() of a tbl_gpu
# ---------------------------------------------------------------------------

test_that("rowwise() of a tbl_gpu is a native rowwise tbl_gpu (eager and lazy)", {
  skip_if_no_gpu()
  df <- f49_df()
  for (lazy in c(FALSE, TRUE)) {
    f49_check(df, function(x) dplyr::rowwise(x), lazy = lazy)
    f49_check(df, function(x) dplyr::rowwise(x, g), lazy = lazy)
    f49_check(df, function(x) dplyr::rowwise(x, g, id), lazy = lazy)
    f49_check(df, function(x) dplyr::rowwise(x, "g"), lazy = lazy)
    f49_check(df, function(x) dplyr::rowwise(x, dplyr::starts_with("i")), lazy = lazy)
    f49_check(df, function(x) dplyr::rowwise(x, c(f, dt)), lazy = lazy)
    # a grouped input keeps its group vars
    f49_check(df, function(x) dplyr::rowwise(dplyr::group_by(x, g, f)), lazy = lazy)
    # re-rowwise replaces the vars
    f49_check(df, function(x) dplyr::rowwise(dplyr::rowwise(x, g), id), lazy = lazy)
  }
})

test_that("rowwise() on a lazy table with pending ops stays lazy and computes nothing", {
  skip_if_no_gpu()
  df <- f49_df()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::filter(id > 1) |>
    dplyr::mutate(e = i1 * 2L)
  rw <- dplyr::rowwise(gt, g)
  expect_true(is_tbl_gpu(rw))
  expect_identical(rw$exec_mode, "lazy")
  expect_false(is.null(rw$lazy_ops))
  expect_identical(dplyr::group_vars(rw), "g")
  f49_check(df, function(x) {
    x |> dplyr::filter(id > 1) |> dplyr::mutate(e = i1 * 2L) |> dplyr::rowwise(g)
  }, lazy = TRUE)
})

test_that("rowwise() errors like dplyr", {
  skip_if_no_gpu()
  df <- f49_df()
  for (lazy in c(FALSE, TRUE)) {
    f49_same_error(df, function(x) dplyr::rowwise(dplyr::group_by(x, g), id), lazy = lazy,
                   message = "Can't re-group when creating rowwise data")
    f49_same_error(df, function(x) dplyr::rowwise(x, nope), lazy = lazy)
  }
})

# ---------------------------------------------------------------------------
# A2. Upload, collect and the ways out of rowwise
# ---------------------------------------------------------------------------

test_that("tbl_gpu(<rowwise_df>) uploads a rowwise tbl_gpu, silently", {
  skip_if_no_gpu()
  df <- f49_df()
  for (rdf in list(dplyr::rowwise(df), dplyr::rowwise(df, g), dplyr::rowwise(df, id, g))) {
    for (lazy in c(FALSE, TRUE)) {
      expect_no_warning(gt <- tbl_gpu(rdf, lazy = lazy))
      expect_true(is_tbl_gpu(gt))
      expect_identical(gt$exec_mode, if (lazy) "lazy" else "eager")
      expect_identical(dplyr::group_vars(gt), dplyr::group_vars(rdf))
      out <- collect(gt)
      expect_identical(class(out), class(rdf))
      expect_equal(out, rdf)
    }
    expect_no_warning(ga <- as_tbl_gpu(rdf))
    expect_identical(dplyr::group_vars(ga), dplyr::group_vars(rdf))
    expect_s3_class(collect(ga), "rowwise_df")
  }
})

test_that("an uploaded rowwise_df computes per row, not per rowwise-var group", {
  skip_if_no_gpu()
  df <- f49_df()
  rdf <- dplyr::rowwise(df, g)
  gt <- tbl_gpu(rdf)
  # per row: mean(i1) of one row is i1 itself; n() is 1
  expected <- dplyr::mutate(rdf, m = mean(i1), n = dplyr::n())
  got <- collect(dplyr::mutate(gt, m = mean(i1), n = dplyr::n()))
  expect_identical(class(got), class(expected))
  expect_equal(got, expected)
})

test_that("as_tibble(), ungroup() and group_by() leave rowwise like dplyr", {
  skip_if_no_gpu()
  df <- f49_df()
  for (lazy in c(FALSE, TRUE)) {
    f49_check(df, function(x) tibble::as_tibble(dplyr::rowwise(x, g)), lazy = lazy, tbl = FALSE)
    f49_check(df, function(x) dplyr::ungroup(dplyr::rowwise(x, g)), lazy = lazy)
    f49_check(df, function(x) dplyr::group_by(dplyr::rowwise(x, g), f), lazy = lazy)
    f49_check(df, function(x) dplyr::group_by(dplyr::rowwise(x, g), f, .add = TRUE), lazy = lazy)
    f49_check(df, function(x) dplyr::group_by(dplyr::rowwise(x, g)), lazy = lazy)
    f49_same_error(df, function(x) dplyr::ungroup(dplyr::rowwise(x, g), g), lazy = lazy)
  }
})

test_that("collect() of a rowwise tbl_gpu is identical to dplyr's rowwise_df", {
  skip_if_no_gpu()
  df <- f49_df()
  for (vars in list(character(), "g", c("id", "f"))) {
    got <- collect(dplyr::rowwise(tbl_gpu(df), dplyr::all_of(vars)))
    expect_identical(got, dplyr::rowwise(df, dplyr::all_of(vars)))
  }
  # a data.frame input still gives dplyr's class (rowwise_df is a tibble)
  ddf <- as.data.frame(df)
  got <- collect(dplyr::rowwise(tbl_gpu(ddf), g))
  expect_identical(class(got), class(dplyr::rowwise(ddf, g)))
})

test_that("compute() keeps a lazy rowwise table rowwise", {
  skip_if_no_gpu()
  df <- f49_df()
  withr::local_options(cuplyr.fallback = "error")
  rw <- dplyr::rowwise(tbl_gpu(df, lazy = TRUE), g) |> dplyr::mutate(z = i1 + 1L)
  cm <- dplyr::compute(rw)
  expect_true(is_tbl_gpu(cm))
  expect_identical(dplyr::group_vars(cm), "g")
  expect_identical(class(collect(cm)), class(dplyr::rowwise(df, g)))
})

# ---------------------------------------------------------------------------
# A3. Group metadata and printing
# ---------------------------------------------------------------------------

test_that("group metadata of a rowwise tbl_gpu has one group per row, natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  for (df in list(f49_df(), f49_df()[0, ])) {
    for (vars in list(character(), "g", c("g", "id"))) {
      rdf <- dplyr::rowwise(df, dplyr::all_of(vars))
      for (lazy in c(FALSE, TRUE)) {
        rw <- dplyr::rowwise(tbl_gpu(df, lazy = lazy), dplyr::all_of(vars))
        expect_identical(dplyr::group_vars(rw), dplyr::group_vars(rdf))
        expect_identical(dplyr::n_groups(rw), dplyr::n_groups(rdf))
        expect_equal(dplyr::group_data(rw), dplyr::group_data(rdf))
        expect_equal(dplyr::group_keys(rw), dplyr::group_keys(rdf))
        expect_identical(dplyr::group_indices(rw), dplyr::group_indices(rdf))
        expect_identical(dplyr::group_size(rw), dplyr::group_size(rdf))
        expect_equal(dplyr::group_rows(rw), dplyr::group_rows(rdf))
      }
    }
  }
})

test_that("a rowwise tbl_gpu prints its rowwise vars like dplyr's header", {
  skip_if_no_gpu()
  df <- f49_df()
  rw <- dplyr::rowwise(tbl_gpu(df), g, id)
  expect_identical(pillar::tbl_sum(rw), pillar::tbl_sum(dplyr::rowwise(df, g, id)))
  expect_identical(pillar::tbl_sum(dplyr::rowwise(tbl_gpu(df))), pillar::tbl_sum(dplyr::rowwise(df)))
  out <- utils::capture.output(print(rw))
  expect_true(any(grepl("Rowwise:\\s*g, id", out)))
  expect_false(any(grepl("Groups:", out)))
})

# ---------------------------------------------------------------------------
# A4. Verbs keep rowwise-ness exactly like dplyr
# ---------------------------------------------------------------------------

test_that("row/column-shape verbs keep a rowwise table rowwise, natively", {
  skip_if_no_gpu()
  df <- f49_df()
  native_pipes <- list(
    function(x) dplyr::arrange(x, dplyr::desc(d2)),
    function(x) dplyr::arrange(x, i1, .by_group = TRUE),
    function(x) dplyr::filter(x, i1 > 0),
    function(x) dplyr::filter_out(x, i1 > 0),
    function(x) dplyr::filter(x, s != "q", d1 > 1),
    function(x) dplyr::mutate(x, z = i1 + 1L, w = paste0(s, "!")),
    function(x) dplyr::mutate(x, z = i1 * 2L, .keep = "none"),
    function(x) dplyr::mutate(x, z = d1 - d2, .keep = "unused"),
    function(x) dplyr::mutate(x, z = 1L, .before = 1),
    function(x) dplyr::mutate(x, g = paste0(g, "!")),
    function(x) dplyr::mutate(x, i1 = NULL),
    function(x) dplyr::transmute(x, z = d1 * 2),
    function(x) dplyr::rename(x, G = g, I1 = i1),
    function(x) dplyr::rename_with(x, toupper),
    function(x) suppressMessages(dplyr::select(x, i1, d1)),
    function(x) dplyr::select(x, g, i1),
    function(x) dplyr::relocate(x, g, .after = s),
    function(x) head(x, 2)
  )
  for (vars in list(character(), "g")) {
    for (lazy in c(FALSE, TRUE)) {
      for (pipe in native_pipes) {
        f49_check(df, function(x) pipe(dplyr::rowwise(x, dplyr::all_of(vars))), lazy = lazy)
      }
    }
  }
})

test_that("select() adds the missing rowwise vars with dplyr's message", {
  skip_if_no_gpu()
  rw <- dplyr::rowwise(tbl_gpu(f49_df()), g)
  expect_message(out <- dplyr::select(rw, i1), "Adding missing grouping variables")
  expect_identical(names(collect(out)), c("g", "i1"))
  expect_identical(dplyr::group_vars(out), "g")
})

test_that("every other verb on a rowwise table matches dplyr (class, vars, data)", {
  skip_if_no_gpu()
  df <- f49_df()
  y <- tibble::tibble(g = c("a", "b", "d"), w = c(10, 20, 40))
  more <- f49_df()[1:2, ]
  side <- function(x, v) if (is_tbl_gpu(x)) tbl_gpu(v) else v
  pipes <- list(
    function(x) dplyr::slice(x, c(1, 1)),
    function(x) dplyr::slice(x, 1),
    function(x) dplyr::slice(x, 2),
    function(x) dplyr::slice(x),
    function(x) dplyr::slice_head(x, n = 1),
    function(x) dplyr::slice_tail(x, n = 1),
    function(x) dplyr::slice_max(x, i1),
    function(x) dplyr::slice_min(x, d2, n = 2),
    function(x) dplyr::distinct(x),
    function(x) dplyr::distinct(x, f),
    function(x) dplyr::distinct(x, f, .keep_all = TRUE),
    function(x) dplyr::count(x),
    function(x) dplyr::count(x, f),
    function(x) dplyr::add_count(x, f),
    function(x) dplyr::tally(x),
    function(x) dplyr::reframe(x, z = c(i1, i2)),
    function(x) dplyr::left_join(x, side(x, y), by = "g"),
    function(x) dplyr::inner_join(x, side(x, y), by = "g"),
    function(x) dplyr::semi_join(x, side(x, y), by = "g"),
    function(x) dplyr::anti_join(x, side(x, y), by = "g"),
    function(x) cuplyr::bind_rows(x, side(x, more)),
    function(x) dplyr::union_all(x, side(x, more)),
    function(x) dplyr::union(x, side(x, more))
  )
  for (vars in list(character(), "g")) {
    for (lazy in c(FALSE, TRUE)) {
      for (pipe in pipes) {
        f49_check(df, function(x) pipe(dplyr::rowwise(x, dplyr::all_of(vars))),
                  lazy = lazy, native = FALSE)
      }
    }
  }
})

test_that("pull(), group_split() and group_map() on a rowwise table match dplyr", {
  skip_if_no_gpu()
  df <- f49_df()
  rdf <- dplyr::rowwise(df, g)
  for (lazy in c(FALSE, TRUE)) {
    rw <- dplyr::rowwise(tbl_gpu(df, lazy = lazy), g)
    expect_identical(dplyr::pull(rw, i1), dplyr::pull(rdf, i1))
    got <- dplyr::group_split(rw)
    want <- dplyr::group_split(rdf)
    expect_length(got, length(want))
    got <- lapply(got, function(p) if (is_tbl_gpu(p)) collect(p) else p)
    for (i in seq_along(want)) {
      expect_identical(class(got[[i]]), class(want[[i]]))
      expect_equal(got[[i]], want[[i]])
    }
    expect_equal(
      dplyr::group_map(rw, ~ nrow(.x)),
      dplyr::group_map(rdf, ~ nrow(.x))
    )
  }
})

# ---------------------------------------------------------------------------
# A5. Native rowwise mutate/filter/summarise (elementwise), fallbacks, errors
# ---------------------------------------------------------------------------

test_that("elementwise mutate()/filter() on a rowwise table run natively", {
  skip_if_no_gpu()
  df <- f49_df()
  for (vars in list(character(), "g", c("id", "f"))) {
    for (lazy in c(FALSE, TRUE)) {
      f49_check(df, function(x) {
        dplyr::rowwise(x, dplyr::all_of(vars)) |>
          dplyr::mutate(
            a = i1 + i2 * 2L,
            b = dplyr::if_else(d1 > 1, "hi", "lo"),
            c = dplyr::coalesce(d2, 0),
            d = a / 2,
            e = toupper(s)
          )
      }, lazy = lazy)
      f49_check(df, function(x) {
        dplyr::rowwise(x, dplyr::all_of(vars)) |>
          dplyr::filter(!is.na(i1), d1 > 0 | l1)
      }, lazy = lazy)
      # a chain stays rowwise and native
      f49_check(df, function(x) {
        dplyr::rowwise(x, dplyr::all_of(vars)) |>
          dplyr::mutate(z = i1 + 1L) |>
          dplyr::filter(z > 0) |>
          dplyr::arrange(dplyr::desc(z)) |>
          dplyr::select(dplyr::all_of(vars), z)
      }, lazy = lazy)
    }
  }
})

test_that("rowwise window functions see one row, like dplyr", {
  skip_if_no_gpu()
  df <- f49_df()
  for (lazy in c(FALSE, TRUE)) {
    f49_check(df, function(x) {
      dplyr::rowwise(x) |>
        dplyr::mutate(lg = dplyr::lag(i1), rn = dplyr::row_number(), cs = cumsum(d1))
    }, lazy = lazy, native = FALSE)
    f49_check(df, function(x) {
      dplyr::rowwise(x, g) |> dplyr::filter(dplyr::row_number() == 1)
    }, lazy = lazy, native = FALSE)
  }
})

test_that("elementwise summarise() on a rowwise table is per row, with dplyr's .groups rules", {
  skip_if_no_gpu()
  df <- f49_df()
  for (lazy in c(FALSE, TRUE)) {
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x, g), t = i1 + i2), lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x), t = d1 * 2), lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x, g, id), t = d1 * 2, .groups = "keep"),
              lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x, g, id), t = d1 * 2, .groups = "drop"),
              lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x, g), t = d1 * 2, .groups = "rowwise"),
              lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x), t = d1 * 2, .groups = "rowwise"),
              lazy = lazy)
    # sequential references and a constant
    f49_check(df, function(x) {
      dplyr::summarise(dplyr::rowwise(x, id), a = i1 * 2L, b = a + 1L, k = 1L)
    }, lazy = lazy)
    # zero dots
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x, g, id)), lazy = lazy)
  }
})

test_that("rowwise summarise() messages like dplyr", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f49_df())
  withr::local_options(dplyr.summarise.inform = TRUE)
  expect_message(
    dplyr::summarise(dplyr::rowwise(gt, g), t = i1 + i2),
    "converted the output from a rowwise data frame to a grouped data frame"
  )
  expect_no_message(dplyr::summarise(dplyr::rowwise(gt, g), t = i1 + i2, .groups = "keep"))
  expect_no_message(dplyr::summarise(dplyr::rowwise(gt), t = i1 + i2))
})

test_that("summarise(.groups = ) on a non-rowwise tbl_gpu gives a native rowwise result", {
  skip_if_no_gpu()
  df <- f49_df()
  for (lazy in c(FALSE, TRUE)) {
    f49_check(df, function(x) dplyr::summarise(x, t = sum(i2, na.rm = TRUE), .groups = "rowwise"),
              lazy = lazy)
    f49_check(df, function(x) {
      dplyr::summarise(dplyr::group_by(x, g), t = sum(d2), .groups = "rowwise")
    }, lazy = lazy)
  }
})

test_that("mutate() shapes with no native rowwise form fall back and match dplyr", {
  skip_if_no_gpu()
  df <- f49_df()
  twice <- function(v) v * 2
  inj <- seq_len(nrow(df)) * 10L
  for (lazy in c(FALSE, TRUE)) {
    # an R function per row
    f49_check(df, function(x) dplyr::mutate(dplyr::rowwise(x, g), z = twice(i1)),
              lazy = lazy, native = FALSE)
    # a list result (list-column: the result may stay on the host)
    f49_check(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = list(c(i1, i2))),
              lazy = lazy, native = FALSE)
    # an injected constant of length n is sliced per row
    f49_check(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = !!inj),
              lazy = lazy, native = FALSE)
    # a later native verb after a fallback still sees a rowwise table
    f49_check(df, function(x) {
      dplyr::rowwise(x, g) |> dplyr::mutate(z = twice(i1)) |> dplyr::mutate(w = z + 1)
    }, lazy = lazy, native = FALSE)
  }
})

test_that("errors on a rowwise table are dplyr's", {
  skip_if_no_gpu()
  df <- f49_df()
  v <- seq_len(nrow(df))
  for (lazy in c(FALSE, TRUE)) {
    f49_same_error(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = 1:3), lazy = lazy,
                   message = "must be size 1")
    f49_same_error(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = .env$v), lazy = lazy)
    f49_same_error(df, function(x) dplyr::mutate(dplyr::rowwise(x), z = i1, .by = g), lazy = lazy,
                   message = "rowwise data frame")
    f49_same_error(df, function(x) dplyr::filter(dplyr::rowwise(x, g), c(TRUE, FALSE)), lazy = lazy)
    f49_same_error(df, function(x) dplyr::summarise(dplyr::rowwise(x), z = c(i1, i2)), lazy = lazy)
    f49_same_error(df, function(x) dplyr::summarise(dplyr::rowwise(x, g), z = 1, .groups = "drop_last"),
                   lazy = lazy, message = "drop_last")
    f49_same_error(df, function(x) dplyr::summarise(dplyr::rowwise(x), z = i1, .by = g), lazy = lazy)
  }
})

test_that("rowwise verbs on zero rows match dplyr", {
  skip_if_no_gpu()
  df <- f49_df()[0, ]
  for (lazy in c(FALSE, TRUE)) {
    f49_check(df, function(x) dplyr::mutate(dplyr::rowwise(x, g), z = i1 + 1L), lazy = lazy)
    f49_check(df, function(x) dplyr::filter(dplyr::rowwise(x), d1 > 0), lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x, g), z = d1 * 2), lazy = lazy)
    f49_check(df, function(x) dplyr::summarise(dplyr::rowwise(x), z = d1 * 2), lazy = lazy)
  }
})

# ---------------------------------------------------------------------------
# A6. Conformance harness: cf_collect() reads a rowwise tbl_gpu as rowwise
# ---------------------------------------------------------------------------

f49_harness_lib <- function() {
  tools_dir <- normalizePath(testthat::test_path("..", "..", "tools"), mustWork = FALSE)
  if (!dir.exists(tools_dir)) {
    testthat::skip("tools/ is not present (build-ignored); harness tests need a checkout")
  }
  env <- new.env(parent = globalenv())
  sys.source(file.path(tools_dir, "conformance-lib.R"), envir = env)
  env
}

test_that("the harness's cf_collect() returns a rowwise_df for a rowwise tbl_gpu", {
  skip_if_no_gpu()
  lib <- f49_harness_lib()
  df <- f49_df()
  for (vars in list(character(), "g", c("g", "id"))) {
    got <- lib$cf_collect(dplyr::rowwise(tbl_gpu(df), dplyr::all_of(vars)))
    want <- dplyr::rowwise(df, dplyr::all_of(vars))
    expect_identical(class(got), class(want))
    expect_identical(dplyr::group_vars(got), vars)
    expect_equal(got, want)
  }
  # grouped and plain tables are read exactly as before
  expect_identical(class(lib$cf_collect(dplyr::group_by(tbl_gpu(df), g))),
                   class(dplyr::group_by(df, g)))
  expect_identical(class(lib$cf_collect(tbl_gpu(df))), class(df))
})

test_that("a vendored-style rowwise test runs on a GPU rowwise table under the harness", {
  skip_if_no_gpu()
  lib <- f49_harness_lib()
  dir <- tempfile("f49-fake-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  writeLines(c(
    'test_that("rowwise stays on the gpu", {',
    '  rf <- rowwise(tibble(x = 1:5, y = 5:1), "x")',
    '  out <- mutate(rf, z = x + y)',
    '  expect_true(inherits(rf, "tbl_gpu"))',
    '  expect_true(inherits(out, "tbl_gpu"))',
    '  expect_s3_class(out, "rowwise_df")',
    '  expect_equal(group_vars(out), "x")',
    '  expect_equal(out, rowwise(tibble(x = 1:5, y = 5:1, z = rep(6L, 5)), x))',
    '})'
  ), file.path(dir, "test-f49-fake.R"))
  res <- lib$conformance_run_file(file.path(dir, "test-f49-fake.R"))
  row <- res$ids[res$ids$test == "rowwise stays on the gpu", , drop = FALSE]
  expect_identical(nrow(row), 1L)
  expect_identical(row$status, "pass")
  expect_true(as.logical(row$exercised))
})

# ---------------------------------------------------------------------------
# A7. nest_join() nests pieces of y's own class (needed once rowwise() is a
# tbl_gpu: conformance test-join.R "nest_join respects types of y (#6295)")
# ---------------------------------------------------------------------------

test_that("nest_join() with a rowwise or grouped tbl_gpu y nests pieces of y's class, like dplyr", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1, 2, 3), a = 1:3)
  ydf <- tibble::tibble(k = c(1, 1, 2), z = c(2, 3, 4))
  makers <- list(
    function(t) dplyr::rowwise(t),
    function(t) dplyr::rowwise(t, z),
    function(t) dplyr::group_by(t, z)
  )
  for (lazy in c(FALSE, TRUE)) {
    for (mk in makers) {
      want <- dplyr::nest_join(x, mk(ydf), by = "k", name = "m")
      got <- dplyr::nest_join(tbl_gpu(x, lazy = lazy), mk(tbl_gpu(ydf, lazy = lazy)), by = "k", name = "m")
      if (is_tbl_gpu(got)) got <- collect(got)
      expect_identical(class(got), class(want))
      for (i in seq_along(want$m)) {
        expect_identical(class(got$m[[i]]), class(want$m[[i]]))
      }
      expect_equal(got, want)
    }
  }
})
