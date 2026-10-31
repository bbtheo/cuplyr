# F51: "gibble" is the user-facing name of cuplyr's GPU tables.
#
# Contract (scratchpad/parity/features/F51.md, ## Design):
# - print() of an eager table (or a lazy one with nothing pending) shows
#   tibble's header for the same tbl_sum(), with "A tibble" renamed
#   "A gibble" (padding, the x/× symbol and big marks exactly as pillar
#   prints them). The glimpse-style "$ col <type> values" body stays.
# - print() of a lazy table with pending operations runs the plan once and
#   shows the eager print of the result with "# A gibble [lazy]: ..." (F80);
#   if the plan fails it shows "# A gibble [lazy]: ?? x k", the grouping, the
#   result's schema, the number of pending (non-source) operations and the
#   condition message. The precise preview contract is in
#   test-parity-F80-print.R.
# - pillar::tbl_sum() is unchanged: it still returns dplyr's own values.
# - gibble()/as_gibble()/is_gibble() are exported aliases of
#   tbl_gpu()/as_tbl_gpu()/is_tbl_gpu(); the class stays c("tbl_gpu", "tbl").
# - gibble() is not a generic named "gibble" and cuplyr has no gibble.<class>
#   methods, so the CRAN package gibble (an S3 generic gibble()) can only
#   mask the name, never cross-dispatch.

# Make every plan fail, so print() shows its failure view: the schema,
# grouping and pending-step count of the table, with nothing run.
f51_local_failing_plan <- function(env = parent.frame()) {
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) stop("f51: plan not run"),
    .package = "cuplyr",
    .env = env
  )
}

f51_df <- function() {
  tibble::tibble(g = c(1, 1, 2), x = c("a", "b", NA))
}

# The header lines of a tbl_gpu print: the leading "#" lines.
f51_gpu_header <- function(x) {
  out <- utils::capture.output(print(x))
  n <- match(FALSE, startsWith(out, "#"), nomatch = length(out) + 1L) - 1L
  out[seq_len(n)]
}

# tibble's own header for the same data, renamed. `tib` is a tibble (grouped
# or rowwise as needed). Its header is the leading "#" lines (a wrapped
# Groups/Rowwise element spans several), stopping before the body or before
# a footer line ("# \u2139 ..." / "# i ..." when there is no body).
f51_tibble_header <- function(tib, ...) {
  out <- utils::capture.output(print(tib, ...))
  is_header <- startsWith(out, "#") & !grepl("^# (\u2139|i) ", out)
  k <- match(FALSE, is_header, nomatch = length(out) + 1L) - 1L
  sub("^# A tibble:", "# A gibble:", out[seq_len(k)])
}

f51_expect_header <- function(gpu, tib) {
  expect_identical(f51_gpu_header(gpu), f51_tibble_header(tib))
}

# ---------------------------------------------------------------------------
# 1. Header: eager tables
# ---------------------------------------------------------------------------

test_that("an ungrouped tbl_gpu prints '# A gibble: n x m' like tibble's header", {
  skip_if_no_gpu()
  df <- f51_df()
  gpu <- tbl_gpu(df)
  out <- utils::capture.output(print(gpu))
  expect_identical(out[1], "# A gibble: 3 x 2")
  f51_expect_header(gpu, df)
  # the old glimpse-style header lines are gone, the body stays
  expect_false(any(grepl("^Rows:", out)))
  expect_false(any(grepl("^Columns:", out)))
  expect_true(any(grepl("^\\$ g\\s+<dbl> 1, 1, 2", out)))
  expect_true(any(grepl("^\\$ x\\s+<chr>", out)))
})

test_that("a bare data.frame upload also prints 'A gibble', while tbl_sum() keeps 'A data frame'", {
  skip_if_no_gpu()
  df <- as.data.frame(f51_df())
  gpu <- tbl_gpu(df)
  f51_expect_header(gpu, tibble::as_tibble(df))
  expect_identical(names(pillar::tbl_sum(gpu)), "A data frame")
})

test_that("grouped tables keep the '# Groups:' line, formatted like tibble", {
  skip_if_no_gpu()
  df <- f51_df()
  f51_expect_header(dplyr::group_by(tbl_gpu(df), g), dplyr::group_by(df, g))
  f51_expect_header(dplyr::group_by(tbl_gpu(df), g, x), dplyr::group_by(df, g, x))
  # NA key forms its own group
  f51_expect_header(dplyr::group_by(tbl_gpu(df), x), dplyr::group_by(df, x))
  out <- utils::capture.output(print(dplyr::group_by(tbl_gpu(df), g)))
  expect_identical(out[1:2], c("# A gibble: 3 x 2", "# Groups:   g [2]"))
  expect_false(any(grepl("^Groups:", out)))
})

test_that("rowwise tables show tibble's '# Rowwise:' line, never Groups", {
  skip_if_no_gpu()
  df <- f51_df()
  f51_expect_header(dplyr::rowwise(tbl_gpu(df), g), dplyr::rowwise(df, g))
  f51_expect_header(dplyr::rowwise(tbl_gpu(df), g, x), dplyr::rowwise(df, g, x))
  f51_expect_header(dplyr::rowwise(tbl_gpu(df)), dplyr::rowwise(df))
  out <- utils::capture.output(print(dplyr::rowwise(tbl_gpu(df), g)))
  expect_false(any(grepl("Groups:", out)))
  expect_false(any(grepl("^Rowwise:", out)))
})

test_that("zero rows, zero columns and big row counts format like tibble", {
  skip_if_no_gpu()
  df <- f51_df()
  f51_expect_header(tbl_gpu(df[0, ]), df[0, ])
  f51_expect_header(tbl_gpu(tibble::tibble()), tibble::tibble())
  big <- tibble::tibble(a = seq_len(12345L), k = rep(1:5, length.out = 12345L))
  f51_expect_header(tbl_gpu(big), big)
  expect_identical(f51_gpu_header(tbl_gpu(big))[1], "# A gibble: 12,345 x 2")
  f51_expect_header(dplyr::group_by(tbl_gpu(big), a), dplyr::group_by(big, a))
  f51_expect_header(dplyr::group_by(tbl_gpu(big), k), dplyr::group_by(big, k))
})

# testthat runs with cli.unicode = FALSE (the "x" above); with unicode on,
# pillar writes the multiplication sign and so must the header.
test_that("the header follows pillar's unicode setting", {
  skip_if_no_gpu()
  withr::local_options(cli.unicode = TRUE)
  df <- f51_df()
  expect_identical(f51_gpu_header(tbl_gpu(df))[1], "# A gibble: 3 \u00d7 2")
  f51_expect_header(tbl_gpu(df), df)
  f51_expect_header(dplyr::group_by(tbl_gpu(df), g), dplyr::group_by(df, g))
})

test_that("print() returns its input invisibly", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(f51_df())
  utils::capture.output(res <- withVisible(print(gpu)))
  expect_false(res$visible)
  expect_identical(res$value, gpu)
})

# ---------------------------------------------------------------------------
# 2. Header: lazy tables
# ---------------------------------------------------------------------------

test_that("a lazy table with nothing pending prints the eager header", {
  skip_if_no_gpu()
  df <- f51_df()
  f51_expect_header(tbl_gpu(df, lazy = TRUE), df)
  f51_expect_header(dplyr::group_by(tbl_gpu(df, lazy = TRUE), g), dplyr::group_by(df, g))
  # a computed lazy pipeline has nothing pending either
  done <- dplyr::compute(dplyr::filter(tbl_gpu(df, lazy = TRUE), g > 1))
  f51_expect_header(done, dplyr::filter(df, g > 1))
})

test_that("a lazy table with pending operations prints its computed rows under a [lazy] header", {
  skip_if_no_gpu()
  df <- f51_df()
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::filter(g > 1) |>
    dplyr::mutate(y = g * 2)
  out <- utils::capture.output(print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]: 1 . 3$")
  expect_true(any(grepl("^\\$ y\\s+<dbl> 4", out)))
  expect_false(any(grepl("not materialized|^# Schema|^# Operations", out)))
  expect_true(has_pending_ops(lz))
})

test_that("a lazy table whose plan fails prints the lazy block with the schema and step count", {
  skip_if_no_gpu()
  df <- f51_df()
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::filter(g > 1) |>
    dplyr::mutate(y = g * 2)
  f51_local_failing_plan()
  out <- utils::capture.output(print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]: \\?\\? . 3$")
  expect_true(any(grepl("^# Schema:\\s+g, x, y\\s*$", out)))
  expect_true(any(grepl("^# Operations pending:\\s+2\\s*$", out)))
  expect_true(any(grepl("^# Plan failed: f51: plan not run$", out)))
  # no source-dims glimpse of the un-run table
  expect_false(any(grepl("^Rows:|^Columns:|preview unavailable|A gibble:", out)))
})

test_that("a pending lazy join prints its result, and the lazy block with the result schema if the plan fails", {
  skip_if_no_gpu()
  df <- f51_df()
  keys <- tibble::tibble(g = c(1, 2), lab = c("one", "two"))
  lz <- dplyr::left_join(tbl_gpu(df, lazy = TRUE), tbl_gpu(keys, lazy = TRUE), by = "g")
  out <- utils::capture.output(print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]: 3 . 3$")
  expect_true(any(grepl("^\\$ lab\\s+<chr>", out)))
  f51_local_failing_plan()
  out <- utils::capture.output(print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]: \\?\\? . 3$")
  expect_true(any(grepl("^# Schema:\\s+g, x, lab\\s*$", out)))
  expect_true(any(grepl("^# Operations pending:\\s+1\\s*$", out)))
  expect_false(any(grepl("GPU tibble", out)))
})

test_that("a grouped lazy table with pending operations prints its groups under a [lazy] header", {
  skip_if_no_gpu()
  df <- f51_df()
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::mutate(y = g + 1) |>
    dplyr::group_by(g)
  out <- utils::capture.output(print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]: 3 . 3$")
  expect_identical(out[2], "# Groups:   g [2]")
  # printing does not change what the pipeline returns
  expect_identical(
    tibble::as_tibble(collect(lz)),
    tibble::as_tibble(dplyr::group_by(dplyr::mutate(df, y = g + 1), g))
  )
  f51_local_failing_plan()
  out <- utils::capture.output(print(lz))
  expect_true(any(grepl("^# Operations pending:\\s+1\\s*$", out)))
})

# ---------------------------------------------------------------------------
# 3. tbl_sum() keeps dplyr's values
# ---------------------------------------------------------------------------

test_that("pillar::tbl_sum() still matches dplyr exactly", {
  skip_if_no_gpu()
  df <- f51_df()
  expect_identical(pillar::tbl_sum(tbl_gpu(df)), pillar::tbl_sum(df))
  expect_identical(
    pillar::tbl_sum(dplyr::group_by(tbl_gpu(df), g)),
    pillar::tbl_sum(dplyr::group_by(df, g))
  )
  expect_identical(
    pillar::tbl_sum(dplyr::rowwise(tbl_gpu(df), g)),
    pillar::tbl_sum(dplyr::rowwise(df, g))
  )
})

# ---------------------------------------------------------------------------
# 4. Aliases
# ---------------------------------------------------------------------------

test_that("gibble(), as_gibble() and is_gibble() are exported", {
  exports <- getNamespaceExports("cuplyr")
  expect_true(all(c("gibble", "as_gibble", "is_gibble") %in% exports))
  expect_true(is.function(cuplyr::gibble))
  expect_true(is.function(cuplyr::as_gibble))
  expect_true(is.function(cuplyr::is_gibble))
})

test_that("gibble() has tbl_gpu()'s arguments and gives identical tables", {
  skip_if_no_gpu()
  df <- f51_df()
  expect_identical(formals(gibble), formals(tbl_gpu))

  a <- gibble(df, lazy = FALSE)
  b <- tbl_gpu(df, lazy = FALSE)
  expect_identical(class(a), class(b))
  expect_identical(class(a), c("tbl_gpu", "tbl"))
  expect_identical(collect(a), collect(b))
  expect_identical(.subset2(a, "schema"), .subset2(b, "schema"))
  expect_identical(.subset2(a, "exec_mode"), "eager")

  # bare data.frame class is remembered the same way
  d <- as.data.frame(df)
  expect_identical(collect(gibble(d)), collect(tbl_gpu(d)))

  # lazy argument and the session option
  lz <- gibble(df, lazy = TRUE)
  expect_identical(.subset2(lz, "exec_mode"), "lazy")
  withr::with_options(list(cuplyr.exec_mode = "lazy"), {
    expect_identical(.subset2(gibble(df), "exec_mode"), "lazy")
  })

  # grouped and rowwise inputs, and a tbl_gpu input (tbl_gpu.tbl_gpu)
  g <- dplyr::group_by(df, g)
  expect_identical(dplyr::group_vars(gibble(g)), dplyr::group_vars(tbl_gpu(g)))
  expect_identical(class(gibble(g)), class(tbl_gpu(g)))
  rw <- dplyr::rowwise(df, x)
  expect_identical(collect(gibble(rw)), collect(tbl_gpu(rw)))
  expect_identical(.subset2(gibble(b, lazy = TRUE), "exec_mode"), "lazy")
  expect_identical(collect(gibble(b)), collect(b))

  # verbs work on it as on any tbl_gpu
  for (lazy in c(FALSE, TRUE)) {
    got <- gibble(df, lazy = lazy) |>
      dplyr::group_by(g) |>
      dplyr::summarise(n = dplyr::n()) |>
      collect()
    want <- dplyr::summarise(dplyr::group_by(df, g), n = dplyr::n())
    expect_identical(tibble::as_tibble(got), tibble::as_tibble(want))
  }
})

test_that("gibble() fails exactly like tbl_gpu() on unsupported input", {
  skip_if_no_gpu()
  for (bad in list(1:3, list(a = 1))) {
    e1 <- tryCatch(tbl_gpu(bad), error = identity)
    e2 <- tryCatch(gibble(bad), error = identity)
    expect_s3_class(e2, "error")
    expect_identical(class(e2), class(e1))
    expect_identical(conditionMessage(e2), conditionMessage(e1))
  }
})

test_that("as_gibble() is as_tbl_gpu()", {
  skip_if_no_gpu()
  df <- f51_df()
  expect_identical(formals(as_gibble), formals(as_tbl_gpu))
  a <- as_gibble(df)
  expect_identical(class(a), c("tbl_gpu", "tbl"))
  expect_identical(collect(a), collect(as_tbl_gpu(df)))
  expect_identical(.subset2(as_gibble(df, lazy = TRUE), "exec_mode"), "lazy")
  d <- as.data.frame(df)
  expect_identical(collect(as_gibble(d)), collect(as_tbl_gpu(d)))
  expect_identical(collect(as_gibble(a)), collect(a))
})

test_that("is_gibble() is TRUE for every GPU table and FALSE otherwise", {
  skip_if_no_gpu()
  df <- f51_df()
  gpu_objs <- list(
    tbl_gpu(df),
    tbl_gpu(df, lazy = TRUE),
    dplyr::filter(tbl_gpu(df, lazy = TRUE), g > 1),
    dplyr::group_by(tbl_gpu(df), g),
    dplyr::rowwise(tbl_gpu(df), g),
    dplyr::left_join(tbl_gpu(df, lazy = TRUE), tbl_gpu(df[1, ], lazy = TRUE), by = c("g", "x"))
  )
  for (o in gpu_objs) {
    expect_true(is_gibble(o))
    expect_identical(is_gibble(o), is_tbl_gpu(o))
  }
  other <- list(
    df, as.data.frame(df), dplyr::group_by(df, g), NULL, 1:3, "gibble",
    list(a = 1), structure(list(), class = "gibble"), structure(list(), class = "tbl")
  )
  for (o in other) {
    expect_false(is_gibble(o))
    expect_identical(is_gibble(o), is_tbl_gpu(o))
  }
})

test_that("gibble() cannot cross-dispatch with the CRAN gibble package's generic", {
  ns <- asNamespace("cuplyr")
  # no gibble.<class> S3 method defined or registered by cuplyr
  expect_false(any(grepl("^gibble\\.", ls(ns, all.names = TRUE))))
  s3 <- ls(get(".__S3MethodsTable__.", envir = ns), all.names = TRUE)
  expect_false(any(grepl("^gibble\\.", s3)))
  # and gibble() never dispatches on a generic named "gibble"
  expect_false(any(grepl('UseMethod("gibble"', deparse(body(gibble)), fixed = TRUE)))
  # a class that only the CRAN package knows gets tbl_gpu()'s error, not a method
  skip_if_no_gpu()
  fake_geom <- structure(list(1), class = "POINT")
  e <- tryCatch(gibble(fake_geom), error = identity)
  expect_s3_class(e, "error")
  expect_match(conditionMessage(e), "tbl_gpu", fixed = TRUE)
})

# ---------------------------------------------------------------------------
# 5. Round-1 review additions (B1-B3, N2)
# ---------------------------------------------------------------------------

# B2: long headers wrap exactly like tibble's (pillar's format_comment():
# wrapped at the print width, continuation lines start "#   ").
test_that("long Groups/Rowwise headers wrap like tibble's, and print(width =) is honoured", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    customer_identifier = 1, transaction_category = 2,
    region_of_purchase = 3, payment_method_used = 4
  )
  vars <- c("customer_identifier", "transaction_category", "region_of_purchase", "payment_method_used")
  withr::local_options(width = 80)
  gg <- dplyr::group_by(tbl_gpu(df), dplyr::across(dplyr::all_of(vars)))
  tg <- dplyr::group_by(df, dplyr::across(dplyr::all_of(vars)))
  f51_expect_header(gg, tg)
  expect_true(length(f51_gpu_header(gg)) >= 3L) # it did wrap
  expect_true(all(nchar(f51_gpu_header(gg)) <= 80L))

  rg <- dplyr::rowwise(tbl_gpu(df), dplyr::all_of(vars))
  rt <- dplyr::rowwise(df, dplyr::all_of(vars))
  f51_expect_header(rg, rt)

  # explicit width
  g2 <- dplyr::group_by(tbl_gpu(df), customer_identifier, transaction_category)
  t2 <- dplyr::group_by(df, customer_identifier, transaction_category)
  got <- utils::capture.output(print(g2, width = 40))
  want <- utils::capture.output(print(t2, width = 40))
  k <- match(FALSE, startsWith(want, "#")) - 1L
  want_header <- sub("^# A tibble:", "# A gibble:", want[seq_len(k)])
  expect_identical(got[seq_len(k)], want_header)
  expect_false(startsWith(got[k + 1L], "#"))
})

# B1: the lazy block still names the grouping (metadata only, nothing runs).
test_that("a failed lazy plan still shows its group or rowwise vars, nothing run", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), id = 1:3, x = c(2, 4, 6))
  lab <- tibble::tibble(g = c(1, 2), lab = c("a", "b"))
  grouped <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 0) |> dplyr::group_by(g)
  rw <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 0) |> dplyr::rowwise(g, id)
  joined <- dplyr::group_by(tbl_gpu(df, lazy = TRUE), g) |>
    dplyr::left_join(tbl_gpu(lab, lazy = TRUE), by = "g")
  plain <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 0)
  f51_local_failing_plan()
  out <- utils::capture.output(print(grouped))
  expect_match(out[1], "^# A gibble \\[lazy\\]: \\?\\? . 3$")
  expect_true(any(grepl("^# Groups:\\s+g\\s*$", out)))
  expect_false(any(grepl("Rowwise", out)))

  out <- utils::capture.output(print(rw))
  expect_match(out[1], "^# A gibble \\[lazy\\]: \\?\\? . 3$")
  expect_true(any(grepl("^# Rowwise:\\s+g, id\\s*$", out)))
  expect_false(any(grepl("Groups", out)))

  out <- utils::capture.output(print(joined))
  expect_true(any(grepl("^# Groups:\\s+g\\s*$", out)))
  expect_true(any(grepl("^# Schema:\\s+g, id, x, lab\\s*$", out)))

  out <- utils::capture.output(print(plain))
  expect_false(any(grepl("Groups|Rowwise", out)))
  # every line of the failure view is a "#" comment line
  expect_true(all(startsWith(out, "#")))
})

# N2: pending operations count every non-source node, join children included.
test_that("the pending-operation count includes both sides of a join", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), id = 1:3, x = c(2, 4, 6))
  lab <- tibble::tibble(g = c(1, 2), lab = c("a", "b"))
  y <- tbl_gpu(lab, lazy = TRUE) |> dplyr::mutate(k = g * 2)
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::filter(x > 0) |>
    dplyr::left_join(y, by = "g") |>
    dplyr::mutate(z = k + 1)
  f51_local_failing_plan()
  out <- utils::capture.output(print(lz))
  expect_true(any(grepl("^# Operations pending:\\s+4\\s*$", out)))
  expect_true(any(grepl("^# Schema:\\s+g, id, x, lab, k, z\\s*$", out)))
})

# B3: the header's group count must not build the full group_data() (an
# n-row permutation download and one R vector per group) for an ordinary
# grouped table; it needs only the number of groups.
test_that("printing a grouped table does not build group_data()", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, 1, 2, NA, NaN), h = c("a", "b", "a", NA, "c"), v = 1:5)
  gg <- dplyr::group_by(tbl_gpu(df), g)
  gh <- dplyr::group_by(tbl_gpu(df), g, h)
  testthat::local_mocked_bindings(
    compute_group_data = function(...) stop("print() built group_data()"),
    .package = "cuplyr"
  )
  expect_no_error(utils::capture.output(print(gg)))
  expect_no_error(utils::capture.output(print(gh)))
})

test_that("the header's group count matches dplyr for NA/NaN keys and .drop = FALSE factors", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(2, 1, 2, NA, NaN, NaN),
    f = factor(c("a", "a", NA, "a", "c", "c"), levels = c("a", "b", "c", "d")),
    v = 1:6
  )
  f51_expect_header(dplyr::group_by(tbl_gpu(df), g), dplyr::group_by(df, g))
  f51_expect_header(dplyr::group_by(tbl_gpu(df), f), dplyr::group_by(df, f))
  f51_expect_header(dplyr::group_by(tbl_gpu(df), g, f), dplyr::group_by(df, g, f))
  f51_expect_header(
    dplyr::group_by(tbl_gpu(df), f, .drop = FALSE),
    dplyr::group_by(df, f, .drop = FALSE)
  )
  f51_expect_header(
    dplyr::group_by(tbl_gpu(df), f, g, .drop = FALSE),
    dplyr::group_by(df, f, g, .drop = FALSE)
  )
  # tbl_sum() and n_groups() agree with dplyr too
  for (spec in list(c("g"), c("f"), c("g", "f"))) {
    expect_identical(
      pillar::tbl_sum(dplyr::group_by(tbl_gpu(df), dplyr::across(dplyr::all_of(spec)))),
      pillar::tbl_sum(dplyr::group_by(df, dplyr::across(dplyr::all_of(spec))))
    )
  }
  expect_identical(
    dplyr::n_groups(dplyr::group_by(tbl_gpu(df), f, .drop = FALSE)),
    dplyr::n_groups(dplyr::group_by(df, f, .drop = FALSE))
  )
})

# ---------------------------------------------------------------------------
# 6. Round-2 review additions (B1, B2)
# ---------------------------------------------------------------------------

# B1 (round 2): tables carrying fixed group keys (filter/slice with
# `.preserve = TRUE`, any filter() of a `.drop = FALSE` table) know their
# group count already; printing must not build group_data() for them either.
test_that("printing a table with preserved/fixed group keys does not build group_data()", {
  skip_if_no_gpu()
  df <- tibble::tibble(id = c(1, 2, 3, 3, NA), v = c(0.1, 0.9, 0.2, 0.8, 0.7))
  fp <- dplyr::filter(dplyr::group_by(tbl_gpu(df, lazy = FALSE), id), v > 0.5, .preserve = TRUE)
  sp <- dplyr::slice(dplyr::group_by(tbl_gpu(df, lazy = FALSE), id), 1, .preserve = TRUE)
  gk <- dplyr::filter(dplyr::group_by(tbl_gpu(df, lazy = FALSE), id, .drop = FALSE), v > 0.5)
  # what dplyr prints (computed before the mock)
  want_fp <- dplyr::filter(dplyr::group_by(df, id), v > 0.5, .preserve = TRUE)
  want_sp <- dplyr::slice(dplyr::group_by(df, id), 1, .preserve = TRUE)
  want_gk <- dplyr::filter(dplyr::group_by(df, id, .drop = FALSE), v > 0.5)
  want <- list(f51_tibble_header(want_fp), f51_tibble_header(want_sp), f51_tibble_header(want_gk))
  testthat::local_mocked_bindings(
    compute_group_data = function(...) stop("print() built group_data()"),
    .package = "cuplyr"
  )
  got <- list(f51_gpu_header(fp), f51_gpu_header(sp), f51_gpu_header(gk))
  expect_identical(got, want)
  expect_identical(dplyr::n_groups(fp), dplyr::n_groups(want_fp))
  expect_identical(dplyr::n_groups(gk), dplyr::n_groups(want_gk))
})

# B2 (round 2): the print width follows pillar's order: print(width =), then
# options(pillar.width), then getOption("width").
test_that("the header honours options(pillar.width) like tibble", {
  skip_if_no_gpu()
  df <- tibble::tibble(customer_identifier = 1, transaction_category = 2)
  withr::local_options(width = 80, pillar.width = 40)
  gg <- dplyr::group_by(tbl_gpu(df), customer_identifier, transaction_category)
  tg <- dplyr::group_by(df, customer_identifier, transaction_category)
  f51_expect_header(gg, tg)
  expect_true(length(f51_gpu_header(gg)) >= 3L)
  # an explicit width still wins over the option
  got <- utils::capture.output(print(gg, width = 80))
  k <- match(FALSE, startsWith(got, "#")) - 1L
  expect_identical(got[seq_len(k)], f51_tibble_header(tg, width = 80))
  # the failure view's grouping line wraps at the same width
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::mutate(z = 1) |>
    dplyr::group_by(customer_identifier, transaction_category)
  f51_local_failing_plan()
  out <- utils::capture.output(print(lz))
  grp <- grep("^# Groups:", out)
  expect_length(grp, 1L)
  expect_true(startsWith(out[grp + 1L], "#   "))
})

# ---------------------------------------------------------------------------
# 7. Round-3 review addition (B1)
# ---------------------------------------------------------------------------

# The glimpse-style body follows glimpse()'s width rule: an infinite width
# option (pillar.width / tibble.width = Inf, a common setting) does not let
# body lines run past the console; a finite option narrows them.
test_that("the body stays within the console under an infinite width option", {
  skip_if_no_gpu()
  df <- tibble::as_tibble(stats::setNames(rep(list(strrep("x", 15)), 8), paste0("c", 1:8)))[rep(1, 10), ]
  gpu <- tbl_gpu(df)
  body_of <- function(out) out[startsWith(out, "$ ")]

  for (opt in c("pillar.width", "tibble.width")) {
    withr::with_options(stats::setNames(list(80, Inf), c("width", opt)), {
      out <- utils::capture.output(print(gpu))
      expect_length(body_of(out), 8L)
      expect_true(all(nchar(body_of(out)) <= 80L), info = opt)
      f51_expect_header(gpu, df)
    })
  }

  withr::with_options(list(width = 80, pillar.width = 40), {
    out <- utils::capture.output(print(gpu))
    expect_true(all(nchar(body_of(out)) <= 40L))
  })

  # an explicit print(width =) still sets the body width
  out <- utils::capture.output(print(gpu, width = 50))
  expect_true(all(nchar(body_of(out)) <= 50L))
})
