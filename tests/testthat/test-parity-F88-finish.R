# F88 attempt 2 acceptance tests: the round-4 findings of attempt 1.
#
# - B12: vector `[` (and `{`, `:`) is not something cuplyr runs on the GPU, so a
#   reason must not say "`[()` runs on the GPU, but not in this form here".
# - Reverse invariant: every head in `fallback_syntax_heads` (the call heads
#   `fallback_has_gpu_handler()` counts as GPU-handled without an engine op)
#   really runs on the GPU in some verb.
# - B13: when summarise()'s native cur_group_id() path declines, the notice
#   says why it declined, not "only as a whole expression".
# - N27: a plan break or CPU fallback raised inside count()/tally()/
#   add_count()/add_tally() reports that verb and the user's `wt` expression.

# Tests see the package namespace; bind the dplyr functions these pipelines
# call bare, as library(dplyr) would.
for (.f88f in c("first", "cur_group_id", "cur_group", "cur_group_rows", "cur_column",
                "c_across", "pick", "desc", "n", "across", "starts_with")) {
  if (!exists(.f88f, mode = "function")) assign(.f88f, getExportedValue("dplyr", .f88f))
}
rm(.f88f)

f88f_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7),
    g = c("a", "a", "b", "b", "c", "c")
  )
}

# Run `expr`; the value and the classed notices it signalled (warnings muffled)
f88f_capture <- function(expr) {
  cpu <- list()
  pb <- list()
  value <- withCallingHandlers(
    expr,
    cuplyr_cpu_fallback = function(c) cpu[[length(cpu) + 1L]] <<- c,
    cuplyr_plan_break = function(c) pb[[length(pb) + 1L]] <<- c,
    warning = function(w) invokeRestart("muffleWarning")
  )
  list(value = value, cpu = cpu, pb = pb)
}

f88f_flat <- function(x) tibble::as_tibble(dplyr::ungroup(x))

# ---------------------------------------------------------------------------
# B12: vector subsetting has no GPU implementation

test_that("F88 B12: `[` inside mutate/filter/summarise is said to have no GPU implementation", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  cases <- list(
    list(f = function(t) mutate(t, d = x - x[1]), expr = "x - x[1]"),
    list(f = function(t) group_by(t, g) |> mutate(d = x - x[1]), expr = "x - x[1]"),
    list(f = function(t) group_by(t, g) |> summarise(s = sum(x[y > 3])), expr = "sum(x[y > 3])"),
    list(f = function(t) filter(t, x > x[1]), expr = "x > x[1]")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88f_capture(cs$f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_equal(f88f_flat(got$value), f88f_flat(cs$f(df)))
      expect_length(got$cpu, 1L)
      c <- got$cpu[[1L]]
      expect_identical(c$expr, cs$expr)
      expect_match(c$reason, "`[()` has no GPU implementation", fixed = TRUE)
      expect_no_match(c$reason, "runs on the GPU", fixed = TRUE)
    }
  }
  # control: first(x) is the native way to write it
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(tbl_gpu(df, lazy = lazy) |> mutate(d = x - first(x)) |> collect())
    expect_length(got$cpu, 0L)
    expect_equal(f88f_flat(got$value), f88f_flat(mutate(df, d = x - first(x))))
  }
})

test_that("F88 B12: `[`, `{` and `:` do not count as GPU-handled heads", {
  skip_if_no_gpu()
  for (h in c("[", "{", ":")) {
    expect_false(fallback_has_gpu_handler(h), info = h)
  }
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(tbl_gpu(df, lazy = lazy) |> mutate(d = {x + 1}) |> collect())
    expect_equal(f88f_flat(got$value), f88f_flat(mutate(df, d = {x + 1})))
    for (c in got$cpu) {
      expect_no_match(c$reason, "`{()` runs on the GPU", fixed = TRUE)
    }
  }
})

# ---------------------------------------------------------------------------
# Reverse invariant: a head counted as GPU-handled really runs on the GPU

test_that("F88: every head in fallback_syntax_heads runs on the GPU in some verb", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  # One native pipeline per head: a sort, pronoun, selection or context role,
  # or a rewrite some verb runs on the GPU. Each names the head itself.
  uses <- list(
    desc = function(t) arrange(t, desc(x)),
    `$` = function(t) mutate(t, z = .data$x + 1),
    `[[` = function(t) mutate(t, z = .data[["x"]] + 1),
    `(` = function(t) mutate(t, z = (x + 1) * 2),
    `~` = function(t) mutate(t, across(x, ~ .x + 1)),
    `function` = function(t) mutate(t, across(x, function(v) v + 1)),
    c = function(t) filter(t, x %in% c(1, 2)),
    c_across = function(t) rowwise(t) |> mutate(s = sum(c_across(x:y))),
    across = function(t) summarise(t, across(c(x, y), mean)),
    if_any = function(t) filter(t, if_any(c(x, y), ~ .x > 5)),
    if_all = function(t) filter(t, if_all(c(x, y), ~ .x > 2)),
    pick = function(t) arrange(t, pick(y)),
    where = function(t) summarise(t, across(where(is.numeric), mean)),
    everything = function(t) select(t, x, y) |> summarise(across(everything(), mean)),
    all_of = function(t) summarise(t, across(all_of(c("x", "y")), mean)),
    any_of = function(t) summarise(t, across(any_of(c("x", "q")), mean)),
    contains = function(t) summarise(t, across(contains("x"), mean)),
    matches = function(t) summarise(t, across(matches("^x"), mean)),
    num_range = function(t) mutate(t, x1 = x, x2 = y) |> summarise(across(num_range("x", 1:2), mean)),
    last_col = function(t) summarise(t, across(last_col(1), mean)),
    cur_group_id = function(t) group_by(t, g) |> mutate(id = cur_group_id()),
    cur_group = function(t) group_by(t, g) |> mutate(k = cur_group()$g),
    cur_group_rows = function(t) group_by(t, g) |> summarise(k = length(cur_group_rows())),
    cur_column = function(t) mutate(t, across(c(x, y), ~ .x + nchar(cur_column()))),
    n = function(t) group_by(t, g) |> summarise(k = n()),
    range = function(t) reframe(t, r = range(x))
  )
  heads <- fallback_syntax_heads
  expect_true(length(heads) > 0L)
  # every head counted as GPU-handled has a proof
  expect_length(setdiff(heads, names(uses)), 0L)
  for (h in intersect(heads, names(uses))) {
    f <- uses[[h]]
    expect_true(h %in% all.names(body(f)), info = h)
    for (lazy in c(FALSE, TRUE)) {
      got <- f88f_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_length(got$cpu, 0L)
      expect_equal(f88f_flat(got$value), f88f_flat(f(df)), info = h)
    }
  }
})

# ---------------------------------------------------------------------------
# B13: summarise()'s cur_group_id() path says why it declined

test_that("F88 B13: a declined cur_group_id() summarise() says why, not 'whole expression'", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  dfid <- tibble::tibble(id = 1:6, x = c(1, 2, 3, 4, 5, 6), g = c("a", "a", "b", "b", "c", "c"))
  gs <- function(t) group_by(t, g)
  cases <- list(
    # C1: an unnamed dot beside the id takes its names from dplyr
    list(d = df, f = function(t) summarise(gs(t), id = cur_group_id(), across(c(x, y), mean)),
         dots = c("cur_group_id()", "across(c(x, y), mean)"), pattern = "unnamed"),
    list(d = df, f = function(t) summarise(gs(t), id = cur_group_id(), mean(x)),
         dots = c("cur_group_id()", "mean(x)"), pattern = "unnamed"),
    # an unnamed id dot
    list(d = df, f = function(t) summarise(gs(t), cur_group_id(), m = mean(x)),
         dots = c("cur_group_id()", "mean(x)"), pattern = "unnamed"),
    # C2: `id` is already a column of the input
    list(d = dfid, f = function(t) summarise(gs(t), id = cur_group_id(), m = mean(x)),
         dots = c("cur_group_id()", "mean(x)"), pattern = "`id`.*column|column.*`id`"),
    # the id takes a grouping column's name
    list(d = df, f = function(t) summarise(gs(t), g = cur_group_id(), m = mean(x)),
         dots = c("cur_group_id()", "mean(x)"), pattern = "`g`"),
    # C3: a later dot reads `id`
    list(d = df, f = function(t) summarise(gs(t), id = cur_group_id(), m = mean(x) + id),
         dots = c("cur_group_id()", "mean(x) + id"), pattern = "`id`.*read|read.*`id`"),
    # a name used twice
    list(d = df, f = function(t) summarise(gs(t), id = cur_group_id(), m = mean(x), m = max(x)),
         dots = c("cur_group_id()", "mean(x)", "max(x)"), pattern = "`m`"),
    # ungrouped with nothing else to summarise
    list(d = df, f = function(t) summarise(t, id = cur_group_id()),
         dots = "cur_group_id()", pattern = "ungrouped|no group|without group")
  )
  for (k in seq_along(cases)) {
    cs <- cases[[k]]
    for (lazy in c(FALSE, TRUE)) {
      got <- f88f_capture(cs$f(tbl_gpu(cs$d, lazy = lazy)) |> collect())
      expect_equal(f88f_flat(got$value), f88f_flat(cs$f(cs$d)), info = paste("case", k))
      expect_length(got$cpu, 1L)
      c <- got$cpu[[1L]]
      expect_true(is.character(c$expr) && !is.na(c$expr) && c$expr %in% cs$dots, info = paste("case", k))
      expect_true(is.character(c$reason) && nzchar(c$reason), info = paste("case", k))
      expect_no_match(c$reason, "whole expression", fixed = TRUE)
      expect_match(c$reason, cs$pattern, info = paste("case", k))
    }
  }
  # cur_group_id() inside a larger expression keeps the "whole expression" reason
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(
      tbl_gpu(df, lazy = lazy) |> gs() |> summarise(id = cur_group_id() + 1, m = mean(x)) |> collect()
    )
    expect_length(got$cpu, 1L)
    expect_identical(got$cpu[[1L]]$expr, "cur_group_id() + 1")
    expect_match(got$cpu[[1L]]$reason, "only as a whole expression", fixed = TRUE)
  }
  # controls: both native
  controls <- list(
    function(t) summarise(gs(t), id = cur_group_id(), m = mean(x)),
    function(t) summarise(gs(t), across(c(x, y), mean))
  )
  for (f in controls) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88f_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_length(got$cpu, 0L)
      expect_equal(f88f_flat(got$value), f88f_flat(f(df)))
    }
  }
})

# ---------------------------------------------------------------------------
# N27: count()/tally()/add_count()/add_tally() report as themselves

test_that("F88 N27: a plan break inside count()/tally() reports that verb and the wt expression", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  pre <- function(t) filter(t, x > 1) |> mutate(w = x + 1)
  cases <- list(
    list(verb = "count", f = function(t) count(t, g, wt = x * 2)),
    list(verb = "tally", f = function(t) group_by(t, g) |> tally(wt = x * 2))
  )
  for (cs in cases) {
    got <- f88f_capture(cs$f(pre(tbl_gpu(df, lazy = TRUE))) |> collect())
    expect_equal(f88f_flat(got$value), f88f_flat(cs$f(pre(df))))
    expect_length(got$cpu, 0L)
    expect_length(got$pb, 1L)
    p <- got$pb[[1L]]
    expect_identical(p$verb, cs$verb)
    expect_identical(p$expr, "x * 2")
    expect_true(is.numeric(p$steps) && p$steps >= 1)
  }
})

test_that("F88 N27: an add_tally() CPU fallback reports add_tally and the wt expression", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  f <- function(t) group_by(t, g) |> add_tally(wt = qnorm(x / 10))
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
    expect_equal(f88f_flat(got$value), f88f_flat(f(df)))
    expect_length(got$cpu, 1L)
    c <- got$cpu[[1L]]
    expect_identical(c$verb, "add_tally")
    expect_identical(c$expr, "qnorm(x / 10)")
    expect_match(c$reason, "`qnorm()` has no GPU implementation", fixed = TRUE)
  }
})

# ---------------------------------------------------------------------------
# Round 1 review (F88-a2): B14, N32, N33, N37

test_that("F88 B14: arrange() never blames a selection inside across()/pick()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(3, 1, 2, 3, 1, 2), y = c(2, 3, 4, 5, 6, 7),
                       g = c("a", "a", "b", "b", "c", "c"))
  pipelines <- list(
    function(t) arrange(t, across(x:y, desc)),
    function(t) arrange(t, across(x:y)),
    function(t) arrange(t, across(c(x, y), desc)),
    function(t) arrange(t, across(starts_with("x") | y, desc))
  )
  for (f in pipelines) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88f_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_equal(f88f_flat(got$value), f88f_flat(f(df)))
      for (c in got$cpu) {
        expect_no_match(c$reason, "`:()`", fixed = TRUE)
        expect_no_match(c$reason, "`c()`", fixed = TRUE)
        expect_no_match(c$reason, "`starts_with()`", fixed = TRUE)
        expect_no_match(c$reason, "`|()`", fixed = TRUE)
        expect_no_match(c$reason, "`desc()` has no GPU", fixed = TRUE)
      }
    }
  }
  # control: a selection with `:` is native in arrange()
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(tbl_gpu(df, lazy = lazy) |> arrange(pick(x:y)) |> collect())
    expect_length(got$cpu, 0L)
    expect_equal(f88f_flat(got$value), f88f_flat(arrange(df, pick(x:y))))
  }
})

test_that("F88 N33: a cur_group_id() decline names the dot as the user wrote it", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  f <- function(t) group_by(t, g) |> summarise(id = cur_group_id(), k = length(cur_group_rows()) + id)
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
    expect_equal(f88f_flat(got$value), f88f_flat(f(df)))
    expect_length(got$cpu, 1L)
    c <- got$cpu[[1L]]
    expect_true(c$expr %in% c("cur_group_id()", "length(cur_group_rows()) + id"))
    expect_no_match(c$expr, "dplyr::n()", fixed = TRUE)
    expect_no_match(c$reason, "whole expression", fixed = TRUE)
  }
})

test_that("F88 N32: a dot before the cur_group_id() dot is not said to read it later", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  id <- 10
  f <- function(t) group_by(t, g) |> summarise(m = mean(x) + id, id = cur_group_id())
  for (lazy in c(FALSE, TRUE)) {
    got <- f88f_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
    expect_equal(f88f_flat(got$value), f88f_flat(f(df)))
    for (c in got$cpu) {
      expect_no_match(c$reason, "later", fixed = TRUE)
      expect_no_match(c$reason, "whole expression", fixed = TRUE)
    }
  }
})

test_that("F88 N37: a plan break after a count() call is not relabelled as count()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88f_df()
  pre <- function(t) filter(t, x > 1) |> mutate(w = x + 1)
  got1 <- f88f_capture(count(pre(tbl_gpu(df, lazy = TRUE)), g, wt = x * 2) |> collect())
  expect_identical(got1$pb[[1L]]$verb, "count")
  got2 <- f88f_capture(pre(tbl_gpu(df, lazy = TRUE)) |> summarise(m = mean(x * y)) |> collect())
  expect_length(got2$pb, 1L)
  expect_identical(got2$pb[[1L]]$verb, "summarise")
  expect_identical(got2$pb[[1L]]$expr, "mean(x * y)")
})
