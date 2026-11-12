# F68a acceptance tests: making lazy verb calls cheap must not change what a
# lazy plan computes, which schema it reports, or when it raises errors.
#
# F68a caches the schema/IR work that recording and lowering used to redo
# (current_schema() re-inferred the whole pending AST on every verb; lowering
# and the optimizer re-inferred every node's input schema). These tests pin the
# contract the caches have to keep:
# - the schema a lazy table reports is exactly what the pending AST infers
#   (names, types, factor levels, time zones), for every verb that records a
#   node, so a cached schema can stand in for the inference;
# - results equal dplyr's and the eager path's, also when plans share
#   prefixes, are collected twice, are extended after a collect, or are built
#   from same-named sources whose types differ;
# - every user-facing error still comes from the verb call, not collect(), and
#   is the eager path's error.

f68_df <- function(n = 40L) {
  set.seed(68L)
  data.frame(
    g = rep_len(c(3L, 1L, 2L, NA), n),
    x = c(NA, round(runif(n - 1L), 3)),
    y = round(rnorm(n), 3),
    w = seq_len(n) / 7,
    s = rep_len(c("a", "b", NA, "c"), n),
    f = factor(rep_len(c("lo", "hi", "mid"), n), levels = c("lo", "mid", "hi")),
    d = as.Date("2024-01-01") + seq_len(n),
    t = as.POSIXct("2024-03-10 12:00:00", tz = "America/New_York") + 3600 * seq_len(n),
    stringsAsFactors = FALSE
  )
}

f68_pipeline <- function(d) {
  d |>
    filter(x > 0.1) |>
    mutate(z = x * y + w) |>
    group_by(g) |>
    summarise(m = mean(z), n = n()) |>
    arrange(desc(m))
}

# The schema a lazy table carries must be what its pending AST infers.
f68_expect_schema_consistent <- function(p, label) {
  expect_true(has_pending_ops(p), info = label)
  expect_identical(current_schema(p), infer_schema_full(p$lazy_ops), info = label)
  expect_identical(.subset2(p, "schema"), infer_schema_full(p$lazy_ops), info = label)
}

# collect() of the lazy plan equals collect() of the same verbs run eagerly.
f68_expect_lazy_is_eager <- function(df, pipeline, label) {
  lz <- pipeline(tbl_gpu(df, lazy = TRUE))
  eg <- pipeline(tbl_gpu(df, lazy = FALSE))
  expect_identical(group_vars(lz), group_vars(eg), info = label)
  expect_identical(tibble::as_tibble(collect(lz)), tibble::as_tibble(collect(eg)), info = label)
  invisible(lz)
}

test_that("F62's pipeline matches dplyr lazily: plain, NA, grouped input, zero rows", {
  skip_if_no_gpu()
  df <- f68_df()
  expect_same_as_dplyr_lazy(df[c("g", "x", "y", "w")], f68_pipeline)
  expect_same_as_dplyr(df[c("g", "x", "y", "w")], f68_pipeline)
  # every row filtered out: an empty summarise
  expect_same_as_dplyr_lazy(df[c("g", "x", "y", "w")], function(d) {
    d |> filter(x > 2) |> mutate(z = x * y + w) |> group_by(g) |>
      summarise(m = mean(z), n = n()) |> arrange(desc(m))
  })
  # zero-row input
  expect_same_as_dplyr_lazy(df[0, c("g", "x", "y", "w")], f68_pipeline)
  # grouped before the pipeline, .groups kept
  expect_same_as_dplyr_lazy(df[c("g", "x", "y", "w", "s")], function(d) {
    d |> group_by(s) |> filter(x > 0.1) |> mutate(z = x * y + w) |>
      group_by(g, .add = TRUE) |> summarise(m = mean(z), n = n(), .groups = "keep") |>
      arrange(desc(m), .by_group = TRUE)
  })
})

test_that("a lazy table's schema is what its pending AST infers, for every recording verb", {
  skip_if_no_gpu()
  df <- f68_df()
  lz <- tbl_gpu(df, lazy = TRUE)
  plans <- list(
    filter = lz |> filter(x > 0.2),
    mutate = lz |> mutate(z = x * 2, s2 = paste0(s, "!"), f2 = f, t2 = t, d2 = d + 1L),
    mutate_replace = lz |> mutate(f = as.character(f), x = NULL),
    transmute = lz |> transmute(g, z = y + 1, f),
    select = lz |> mutate(z = x + 1) |> select(z, f, t, g),
    select_rename = lz |> select(ff = f, tt = t, x),
    rename = lz |> mutate(z = 1) |> rename(ff = f, tt = t),
    relocate = lz |> filter(g > 1) |> relocate(t, f, .before = g),
    arrange = lz |> mutate(z = -x) |> arrange(z, desc(s)),
    group_by = lz |> filter(x > 0) |> group_by(f, g) |> mutate(r = y - mean(y)),
    summarise = lz |> group_by(g, f) |> summarise(m = mean(x, na.rm = TRUE), first_t = first(t),
                                                   .groups = "drop_last"),
    summarise_window = lz |> group_by(g) |> mutate(c = cumsum(w), l = lag(t)) |> filter(c > 1),
    distinct = lz |> mutate(z = g * 2L) |> distinct(z, f, t),
    distinct_all = lz |> filter(g > 1) |> distinct(g, f, .keep_all = TRUE),
    slice = lz |> group_by(g) |> slice_max(x, n = 2) |> mutate(k = 1L),
    head = lz |> mutate(z = 1L) |> head(5),
    count = lz |> filter(x > 0) |> count(g, f),
    collapse = lz |> mutate(z = x * 2) |> collapse() |> filter(z > 0.5),
    join = lz |> filter(g > 1) |> left_join(
      tbl_gpu(data.frame(g = 1:3, lab = factor(c("p", "q", "r"))), lazy = TRUE) |> mutate(k = g * 10L),
      by = "g"),
    semi_join = lz |> mutate(z = 1) |> semi_join(tbl_gpu(data.frame(g = 2:3), lazy = TRUE), by = "g"),
    deep = Reduce(function(d, i) d |> mutate(x = x + 1) |> filter(x > -1e9) |> select(everything()),
                  1:15, lz)
  )
  for (nm in names(plans)) {
    f68_expect_schema_consistent(plans[[nm]], nm)
  }
  # names<- and $<- on a table with pending ops go through the AST too
  p <- lz |> mutate(z = x + 1)
  names(p) <- c(paste0("c", seq_len(ncol(df))), "zz")
  f68_expect_schema_consistent(p, "names<-")
  expect_identical(names(collect(p)), c(paste0("c", seq_len(ncol(df))), "zz"))
  p <- lz |> mutate(z = x + 1)
  p$x <- "const"
  f68_expect_schema_consistent(p, "$<-")
  expect_identical(collect(p |> filter(x == "const") |> select(x, z))$x, rep("const", nrow(df)))
})

test_that("lazy plans equal the eager path for the cached-schema verbs, metadata included", {
  skip_if_no_gpu()
  df <- f68_df()
  f68_expect_lazy_is_eager(df, function(d) d |> mutate(f2 = f, t2 = t) |> select(f2, t2, g) |>
                             rename(gg = g) |> filter(gg > 1), "factor/tz through select+rename")
  f68_expect_lazy_is_eager(df, function(d) d |> group_by(f) |> summarise(t1 = first(t), n = n()),
                           "summarise keeps tz of first()")
  f68_expect_lazy_is_eager(df, function(d) d |> mutate(z = x * 2) |> collapse() |>
                             filter(z > 0.5) |> select(z, f), "collapse barrier")
  f68_expect_lazy_is_eager(df, function(d) d |> mutate(x = x + 1) |> mutate(x = x * 2) |>
                             filter(x > 2.5) |> mutate(x = as.integer(round(x))) |>
                             summarise(s = sum(x)), "column retyped along the chain")
  f68_expect_lazy_is_eager(df, function(d) Reduce(function(d, i) {
    d |> mutate(y = y + i) |> filter(y > -1e9) |> select(-w) |> mutate(w = y)
  }, 1:12, d), "12 rounds of mutate/filter/select")
})

test_that("plans that share a prefix, are collected twice or extended after collect stay correct", {
  skip_if_no_gpu()
  df <- f68_df()
  tb <- tibble::as_tibble(df)
  lz <- tbl_gpu(tb, lazy = TRUE)
  base <- lz |> mutate(z = x * y, k = w + 1) |> filter(g >= 1)
  ref_base <- tb |> mutate(z = x * y, k = w + 1) |> filter(g >= 1)
  a <- base |> select(g, z)                       # prunes k and most columns
  b <- base |> filter(z > 0) |> summarise(s = sum(k))  # filter pushdown candidate
  c2 <- base |> mutate(k = as.character(k)) |> select(k, f)  # retypes a shared column
  expect_identical(collect(a), select(ref_base, g, z))
  expect_equal(collect(b), ref_base |> filter(z > 0) |> summarise(s = sum(k)))
  expect_identical(collect(c2), ref_base |> mutate(k = as.character(k)) |> select(k, f))
  # base itself, twice, then extended after being collected
  expect_identical(collect(base), ref_base)
  expect_identical(collect(base), ref_base)
  ext <- base |> arrange(desc(z)) |> select(z, t)
  expect_identical(collect(ext), ref_base |> arrange(desc(z)) |> select(z, t))
  # compute() in the middle, then keep recording on the computed table
  mid <- compute(base) |> mutate(q = z + k) |> filter(q > 1)
  expect_true(has_pending_ops(mid))
  f68_expect_schema_consistent(mid, "after compute()")
  expect_identical(collect(mid), ref_base |> mutate(q = z + k) |> filter(q > 1))
})

test_that("the same verbs over sources whose same-named columns differ in type stay correct", {
  skip_if_no_gpu()
  plan <- function(d) d |> mutate(y = x) |> filter(!is.na(y)) |> arrange(y) |> select(y)
  dfs <- list(
    dbl = data.frame(x = c(3.5, NA, 1.25)),
    int = data.frame(x = c(3L, NA, 1L)),
    chr = data.frame(x = c("b", NA, "a")),
    fct = data.frame(x = factor(c("b", NA, "a"), levels = c("b", "a"))),
    dt  = data.frame(x = as.Date(c("2024-02-01", NA, "2023-01-01")))
  )
  plans <- lapply(dfs, function(df) plan(tbl_gpu(df, lazy = TRUE)))
  # collected in reverse order of recording, after all plans exist
  for (nm in rev(names(dfs))) {
    expect_identical(tibble::as_tibble(collect(plans[[nm]])),
                     plan(tibble::as_tibble(dfs[[nm]])), info = nm)
    f68_expect_schema_consistent(plans[[nm]], nm)
  }
  # a source re-uploaded under the same variable with a different schema
  src <- tbl_gpu(dfs$dbl, lazy = TRUE)
  p1 <- plan(src)
  src <- tbl_gpu(dfs$chr, lazy = TRUE)
  p2 <- plan(src)
  expect_identical(collect(p1)$y, c(1.25, 3.5))
  expect_identical(collect(p2)$y, c("a", "b"))
})

# Error timing: dplyr raises at the verb. A lazy table with pending ops must
# raise at the verb call as well (no collect() involved), with the error the
# same verb gives on the eager table.
f68_expect_verb_error_like_eager <- function(df, prefix, verb, label) {
  lz <- prefix(tbl_gpu(df, lazy = TRUE))
  eg <- prefix(tbl_gpu(df, lazy = FALSE))
  expect_true(has_pending_ops(lz), info = label)
  e_lazy <- tryCatch({ verb(lz); NULL }, error = function(e) e)
  e_eager <- tryCatch({ verb(eg); NULL }, error = function(e) e)
  expect_false(is.null(e_lazy), info = paste(label, "(lazy verb did not error)"))
  expect_false(is.null(e_eager), info = paste(label, "(eager verb did not error)"))
  if (!is.null(e_lazy) && !is.null(e_eager)) {
    expect_identical(class(e_lazy), class(e_eager), info = label)
    expect_identical(conditionMessage(e_lazy), conditionMessage(e_eager), info = label)
  }
}

test_that("user-facing errors are raised at the verb call on a lazy table, as eagerly", {
  skip_if_no_gpu()
  df <- f68_df()
  pre <- function(d) d |> filter(g > 0) |> mutate(z = x * 2) |> select(-w)
  verbs <- list(
    mutate_missing   = function(d) mutate(d, q = no_such + 1),
    mutate_dropped   = function(d) mutate(d, q = w + 1),
    filter_missing   = function(d) filter(d, no_such > 1),
    filter_dropped   = function(d) filter(d, w > 1),
    summarise_missing = function(d) summarise(group_by(d, g), m = mean(no_such)),
    arrange_missing  = function(d) arrange(d, no_such),
    select_missing   = function(d) select(d, no_such),
    select_dropped   = function(d) select(d, w),
    rename_missing   = function(d) rename(d, a = no_such),
    relocate_missing = function(d) relocate(d, no_such),
    group_by_missing = function(d) group_by(d, no_such),
    distinct_missing = function(d) distinct(d, no_such),
    count_missing    = function(d) count(d, no_such),
    by_and_grouped   = function(d) mutate(group_by(d, g), q = 1, .by = s),
    slice_max_missing = function(d) slice_max(d, no_such, n = 1),
    join_missing_key = function(d) left_join(d, tbl_gpu(data.frame(k = 1L), lazy = TRUE), by = "k")
  )
  for (nm in names(verbs)) {
    f68_expect_verb_error_like_eager(df, pre, verbs[[nm]], nm)
  }
  # a column created earlier in the pending chain is visible to the next verb
  ok <- tbl_gpu(df, lazy = TRUE) |> mutate(z = x * 2) |> filter(z > 0.5) |> mutate(q = z + 1)
  expect_true(has_pending_ops(ok))
  expect_identical(names(ok), c(names(df), "z", "q"))
})

test_that("verb-time warnings stay at the verb on a lazy table", {
  skip_if_no_gpu()
  df <- f68_df()
  lz <- tbl_gpu(df, lazy = TRUE) |> filter(g > 0)
  # dplyr names an unnamed dot after the expression; cuplyr warns at the verb
  expect_warning(p <- mutate(lz, x * 2), "Unnamed mutate expression")
  expect_true(has_pending_ops(p))
  expect_no_warning(res <- collect(p))
  expect_true("x * 2" %in% names(res))
})
