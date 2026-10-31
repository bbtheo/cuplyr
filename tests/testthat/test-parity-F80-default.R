# F80: gibbles are lazy by default.
#
# Contract (scratchpad/parity/features/F80.md):
# - resolve_exec_mode() defaults to "lazy"; `lazy =` > options(cuplyr.exec_mode)
#   > CUPLYR_EXEC_MODE > default, as before.
# - Every way a gibble is born follows that rule, and a verb's result keeps its
#   input's mode (a multi-input verb is lazy when any gibble input is lazy, as
#   joins already are). An eager input stays eager everywhere, including a
#   join whose `y` is uploaded by `copy = TRUE`.
# - Errors stay at the verb on the lazy default (F68a's guarantee).
# - No plan-break or fallback notice fires on common pipelines.
# - The dead options cuplyr.lazy / cuplyr.verbose are gone.
# - expect_same_as_dplyr() keeps testing eager, expect_same_as_dplyr_lazy() lazy.

f80_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L),
    x = c(1.5, NA, 3, 4.25, -1),
    s = c("a", "b", NA, "d", "e")
  )
}

f80_keys <- function() {
  tibble::tibble(g = c(1L, 2L, 4L), lab = c("one", "two", "four"))
}

# Clear every switch so the package default decides.
f80_local_default <- function(env = parent.frame()) {
  withr::local_options(cuplyr.exec_mode = NULL, .local_envir = env)
  withr::local_envvar(CUPLYR_EXEC_MODE = NA, .local_envir = env)
}

f80_mode <- function(x) .subset2(x, "exec_mode")

# Rows of a result as a plain tibble sorted by every column, so content can be
# compared whatever the row order.
f80_rows <- function(x) {
  if (is_tbl_gpu(x)) x <- collect(x)
  x <- dplyr::ungroup(tibble::as_tibble(x))
  dplyr::arrange(x, dplyr::across(dplyr::everything()))
}

# ---------------------------------------------------------------------------
# 1. The default and its switches
# ---------------------------------------------------------------------------

test_that("resolve_exec_mode() defaults to lazy, with the old precedence", {
  f80_local_default()
  expect_identical(resolve_exec_mode(NULL), "lazy")

  withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"), {
    expect_identical(resolve_exec_mode(NULL), "eager")
    # option beats the environment variable
    withr::with_options(list(cuplyr.exec_mode = "lazy"),
                        expect_identical(resolve_exec_mode(NULL), "lazy"))
    # the argument beats both
    expect_identical(resolve_exec_mode(TRUE), "lazy")
  })
  withr::with_options(list(cuplyr.exec_mode = "eager"), {
    expect_identical(resolve_exec_mode(NULL), "eager")
    expect_identical(resolve_exec_mode(TRUE), "lazy")
    expect_identical(resolve_exec_mode("lazy"), "lazy")
  })
  withr::with_envvar(c(CUPLYR_EXEC_MODE = "lazy"),
    withr::with_options(list(cuplyr.exec_mode = "eager"),
                        expect_identical(resolve_exec_mode(NULL), "eager")))
  expect_identical(resolve_exec_mode(FALSE), "eager")
  expect_identical(resolve_exec_mode("eager"), "eager")
})

test_that("tbl_gpu(), gibble(), as_gibble() and as_tbl_gpu() make lazy tables by default", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  expect_identical(f80_mode(tbl_gpu(df)), "lazy")
  expect_identical(f80_mode(gibble(df)), "lazy")
  expect_identical(f80_mode(as_gibble(df)), "lazy")
  expect_identical(f80_mode(as_tbl_gpu(df)), "lazy")
  # a grouped_df and a rowwise_df upload lazy too
  expect_identical(f80_mode(tbl_gpu(dplyr::group_by(df, g))), "lazy")
  expect_identical(f80_mode(tbl_gpu(dplyr::rowwise(df))), "lazy")

  expect_identical(f80_mode(tbl_gpu(df, lazy = FALSE)), "eager")
  expect_identical(f80_mode(gibble(df, lazy = FALSE)), "eager")
  withr::with_options(list(cuplyr.exec_mode = "eager"),
                      expect_identical(f80_mode(tbl_gpu(df)), "eager"))
  withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"),
                     expect_identical(f80_mode(tbl_gpu(df)), "eager"))
})

test_that("a verb on a default table records a plan and collects dplyr's result", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  gt <- tbl_gpu(df)
  res <- gt |> dplyr::filter(x > 0) |> dplyr::mutate(z = x * 2)
  expect_true(has_pending_ops(res))
  expect_identical(f80_mode(res), "lazy")
  expect_equal(
    tibble::as_tibble(collect(res)),
    df |> dplyr::filter(x > 0) |> dplyr::mutate(z = x * 2)
  )
  # nrow(), dim(), $ and pull() answer from the plan without changing it
  expect_identical(nrow(res), 3L)
  expect_identical(dim(res), c(3L, 4L))
  expect_equal(res$z, c(3, 6, 8.5))
  expect_equal(dplyr::pull(res, s), c("a", NA, "d"))
  expect_true(has_pending_ops(res))
})

test_that("tbl_gpu() of a gibble, as_lazy() and as_eager() keep their documented behaviour", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  lz <- tbl_gpu(df) |> dplyr::filter(x > 0)
  # no `lazy =`: the table comes back as it is, plan included
  same <- tbl_gpu(lz)
  expect_identical(f80_mode(same), "lazy")
  expect_true(has_pending_ops(same))
  # lazy = FALSE runs the plan and switches to eager
  eg <- tbl_gpu(lz, lazy = FALSE)
  expect_identical(f80_mode(eg), "eager")
  expect_false(has_pending_ops(eg))
  expect_equal(f80_rows(eg), f80_rows(dplyr::filter(df, x > 0)))
  expect_identical(f80_mode(as_eager(lz)), "eager")
  expect_identical(f80_mode(as_lazy(tbl_gpu(df, lazy = FALSE))), "lazy")
  expect_identical(f80_mode(tbl_gpu(tbl_gpu(df, lazy = FALSE))), "eager")
  expect_identical(f80_mode(tbl_gpu(tbl_gpu(df, lazy = FALSE), lazy = TRUE)), "lazy")
})

test_that("compute() and collapse() of a lazy pipeline keep it lazy with nothing pending", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  lz <- tbl_gpu(df) |> dplyr::filter(x > 0)
  done <- dplyr::compute(lz)
  expect_identical(f80_mode(done), "lazy")
  expect_false(has_pending_ops(done))
  expect_equal(f80_rows(done), f80_rows(dplyr::filter(df, x > 0)))
  # a verb after compute() records again
  expect_true(has_pending_ops(dplyr::mutate(done, z = 1)))
  expect_identical(f80_mode(dplyr::collapse(lz)), "lazy")
})

# ---------------------------------------------------------------------------
# 2. Every verb keeps its input's mode
# ---------------------------------------------------------------------------

# Each entry: function(d, yy, ydf) of the main table `d`, a second table `yy`
# of the same kind (gibble or tibble) and the plain tibble `ydf`.
f80_verbs <- list(
  filter        = function(d, yy, ydf) dplyr::filter(d, x > 0),
  mutate        = function(d, yy, ydf) dplyr::mutate(d, z = x * 2),
  mutate_by     = function(d, yy, ydf) dplyr::mutate(d, m = mean(x, na.rm = TRUE), .by = g),
  window_mutate = function(d, yy, ydf) dplyr::mutate(dplyr::group_by(d, g), cs = cumsum(dplyr::coalesce(x, 0))),
  transmute     = function(d, yy, ydf) dplyr::transmute(d, g, z = x + 1),
  select        = function(d, yy, ydf) dplyr::select(d, g, s),
  rename        = function(d, yy, ydf) dplyr::rename(d, gg = g),
  rename_with   = function(d, yy, ydf) dplyr::rename_with(d, toupper),
  relocate      = function(d, yy, ydf) dplyr::relocate(d, s),
  arrange       = function(d, yy, ydf) dplyr::arrange(d, dplyr::desc(x)),
  group_by      = function(d, yy, ydf) dplyr::group_by(d, g),
  ungroup       = function(d, yy, ydf) dplyr::ungroup(dplyr::group_by(d, g)),
  summarise_grp = function(d, yy, ydf) dplyr::summarise(dplyr::group_by(d, g), m = mean(x, na.rm = TRUE), n = dplyr::n()),
  summarise_all = function(d, yy, ydf) dplyr::summarise(d, m = max(x, na.rm = TRUE)),
  summarise_expr = function(d, yy, ydf) dplyr::summarise(dplyr::group_by(d, g), m = mean(x * 2, na.rm = TRUE)),
  reframe       = function(d, yy, ydf) dplyr::reframe(d, r = range(x, na.rm = TRUE)),
  count         = function(d, yy, ydf) dplyr::count(d, g),
  tally         = function(d, yy, ydf) dplyr::tally(dplyr::group_by(d, g)),
  add_count     = function(d, yy, ydf) dplyr::add_count(d, g),
  distinct      = function(d, yy, ydf) dplyr::distinct(d, g),
  slice         = function(d, yy, ydf) dplyr::slice(d, 2:3),
  slice_head    = function(d, yy, ydf) dplyr::slice_head(d, n = 2),
  slice_tail    = function(d, yy, ydf) dplyr::slice_tail(d, n = 2),
  slice_min     = function(d, yy, ydf) dplyr::slice_min(d, x, n = 2),
  slice_max_grp = function(d, yy, ydf) dplyr::slice_max(dplyr::group_by(d, g), x, n = 1),
  slice_sample  = function(d, yy, ydf) dplyr::slice_sample(d, n = 5),
  rowwise       = function(d, yy, ydf) dplyr::mutate(dplyr::rowwise(d), z = x + 1),
  cpu_mutate    = function(d, yy, ydf) dplyr::mutate(d, z = stats::qnorm(stats::pnorm(x))),
  cpu_filter    = function(d, yy, ydf) dplyr::filter(d, stats::qnorm(stats::pnorm(x)) > 0),
  left_join     = function(d, yy, ydf) dplyr::left_join(d, yy, by = "g"),
  inner_join    = function(d, yy, ydf) dplyr::inner_join(d, yy, by = "g"),
  right_join    = function(d, yy, ydf) dplyr::right_join(d, yy, by = "g"),
  full_join     = function(d, yy, ydf) dplyr::full_join(d, yy, by = "g"),
  semi_join     = function(d, yy, ydf) dplyr::semi_join(d, yy, by = "g"),
  anti_join     = function(d, yy, ydf) dplyr::anti_join(d, yy, by = "g"),
  cross_join    = function(d, yy, ydf) dplyr::cross_join(dplyr::select(d, g), dplyr::select(yy, lab)),
  copy_join     = function(d, yy, ydf) dplyr::left_join(d, ydf, by = "g", copy = TRUE),
  copy_semi     = function(d, yy, ydf) dplyr::semi_join(d, ydf, by = "g", copy = TRUE),
  bind_rows     = function(d, yy, ydf) bind_rows(d, d),
  bind_rows_df  = function(d, yy, ydf) bind_rows(d, tibble::tibble(g = 9L, x = 9, s = "z")),
  bind_cols     = function(d, yy, ydf) bind_cols(dplyr::select(d, g), dplyr::transmute(d, x2 = x)),
  union         = function(d, yy, ydf) dplyr::union(d, d),
  union_all     = function(d, yy, ydf) dplyr::union_all(d, d),
  intersect     = function(d, yy, ydf) dplyr::intersect(d, dplyr::filter(d, g > 1)),
  setdiff       = function(d, yy, ydf) dplyr::setdiff(d, dplyr::filter(d, g > 1)),
  symdiff       = function(d, yy, ydf) dplyr::symdiff(d, dplyr::filter(d, g > 1)),
  rows_append   = function(d, yy, ydf) dplyr::rows_append(d, dplyr::filter(d, g == 3L))
)

# `bind_rows()`/`bind_cols()` are cuplyr's (dplyr's refuse a gibble); a tibble
# goes through them to dplyr unchanged.
f80_run_verbs <- function(make, mode) {
  df <- f80_df()
  keys <- f80_keys()
  for (nm in names(f80_verbs)) {
    fn <- f80_verbs[[nm]]
    gpu <- fn(make(df), make(keys), keys)
    cpu <- fn(df, keys, keys)
    expect_true(is_tbl_gpu(gpu), info = nm)
    if (!is_tbl_gpu(gpu)) next
    expect_identical(f80_mode(gpu), mode, info = nm)
    if (!identical(nm, "slice_sample")) {
      expect_equal(f80_rows(gpu), f80_rows(cpu), info = nm)
    }
  }
}

test_that("every verb on a default (lazy) table returns a lazy table with dplyr's rows", {
  skip_if_no_gpu()
  f80_local_default()
  expect_identical(f80_mode(tbl_gpu(f80_df())), "lazy")
  f80_run_verbs(function(d) tbl_gpu(d), "lazy")
})

test_that("every verb on an eager table returns an eager table with dplyr's rows", {
  skip_if_no_gpu()
  f80_local_default()
  f80_run_verbs(function(d) tbl_gpu(d, lazy = FALSE), "eager")
})

test_that("a multi-input verb is lazy when any gibble input is lazy", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  lz <- tbl_gpu(df)
  eg <- tbl_gpu(df, lazy = FALSE)
  for (pair in list(list(lz, eg), list(eg, lz))) {
    a <- pair[[1]]
    b <- pair[[2]]
    expect_identical(f80_mode(bind_rows(a, b)), "lazy")
    expect_identical(f80_mode(dplyr::union_all(a, b)), "lazy")
    expect_identical(f80_mode(dplyr::left_join(a, dplyr::distinct(b, g), by = "g")), "lazy")
  }
  expect_identical(f80_mode(bind_rows(df, lz)), "lazy")
  expect_identical(f80_mode(bind_rows(df, eg)), "eager")
  expect_equal(f80_rows(bind_rows(df, lz)), f80_rows(dplyr::bind_rows(df, df)))
})

test_that("group_split() pieces and column assignment keep the input's mode", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  for (lazy in c(TRUE, FALSE)) {
    mode <- if (lazy) "lazy" else "eager"
    gt <- tbl_gpu(df, lazy = lazy)
    pieces <- dplyr::group_split(dplyr::group_by(gt, g))
    expect_length(pieces, 3L)
    for (p in pieces) expect_identical(f80_mode(p), mode)
    gt$z <- 1
    expect_identical(f80_mode(gt), mode)
    gt$w <- seq_len(5)
    expect_identical(f80_mode(gt), mode)
    expect_equal(f80_rows(gt), f80_rows(dplyr::mutate(df, z = 1, w = seq_len(5))))
  }
})

# ---------------------------------------------------------------------------
# 3. Errors stay at the verb
# ---------------------------------------------------------------------------

f80_capture_error <- function(expr) {
  tryCatch({
    force(expr)
    NULL
  }, error = function(e) e)
}

test_that("user errors are raised at the verb call on the lazy default, as eagerly", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  keys <- f80_keys()
  pre <- function(d) dplyr::mutate(dplyr::filter(d, g > 0), z = g * 2L)
  verbs <- list(
    filter_missing    = function(d) dplyr::filter(d, no_such > 1),
    filter_not_lgl    = function(d) dplyr::filter(d, x),
    mutate_missing    = function(d) dplyr::mutate(d, q = no_such + 1),
    mutate_type       = function(d) dplyr::mutate(d, q = s + 1),
    select_missing    = function(d) dplyr::select(d, no_such),
    rename_missing    = function(d) dplyr::rename(d, a = no_such),
    relocate_missing  = function(d) dplyr::relocate(d, no_such),
    arrange_missing   = function(d) dplyr::arrange(d, no_such),
    group_by_missing  = function(d) dplyr::group_by(d, no_such),
    summarise_missing = function(d) dplyr::summarise(dplyr::group_by(d, g), m = mean(no_such)),
    summarise_groups  = function(d) dplyr::summarise(dplyr::group_by(d, g), n = dplyr::n(), .groups = "bogus"),
    distinct_missing  = function(d) dplyr::distinct(d, no_such),
    count_missing     = function(d) dplyr::count(d, no_such),
    slice_bad_n       = function(d) dplyr::slice_head(d, n = "a"),
    join_missing_key  = function(d) dplyr::left_join(d, tbl_gpu(keys), by = "no_such")
  )
  for (nm in names(verbs)) {
    fn <- verbs[[nm]]
    # dplyr errors too, so this is a user error and not a cuplyr limit
    expect_error(fn(pre(df)), info = paste(nm, "(dplyr)"))
    lz <- pre(tbl_gpu(df))
    expect_true(has_pending_ops(lz), info = nm)
    e_lazy <- f80_capture_error(fn(lz))
    e_eager <- f80_capture_error(fn(pre(tbl_gpu(df, lazy = FALSE))))
    expect_false(is.null(e_lazy), info = paste(nm, "(lazy verb did not error)"))
    expect_false(is.null(e_eager), info = paste(nm, "(eager verb did not error)"))
    if (!is.null(e_lazy) && !is.null(e_eager)) {
      expect_identical(class(e_lazy), class(e_eager), info = nm)
      expect_identical(conditionMessage(e_lazy), conditionMessage(e_eager), info = nm)
    }
  }
})

# ---------------------------------------------------------------------------
# 4. Notices on common pipelines
# ---------------------------------------------------------------------------

f80_common_pipelines <- list(
  summary = function(d, k) d |>
    dplyr::filter(x > 0) |>
    dplyr::mutate(z = x * 2) |>
    dplyr::group_by(g) |>
    dplyr::summarise(m = mean(z), n = dplyr::n()) |>
    dplyr::arrange(dplyr::desc(m)),
  join = function(d, k) d |>
    dplyr::left_join(k, by = "g") |>
    dplyr::filter(!is.na(lab)) |>
    dplyr::select(g, x, lab),
  count = function(d, k) d |> dplyr::count(g) |> dplyr::arrange(dplyr::desc(n)),
  by_window = function(d, k) d |>
    dplyr::mutate(m = mean(x), .by = g) |>
    dplyr::filter(x >= m),
  distinct_slice = function(d, k) d |> dplyr::distinct(g, s) |> dplyr::slice_head(n = 2),
  rename_relocate = function(d, k) d |> dplyr::rename(grp = g) |> dplyr::relocate(s) |> dplyr::arrange(grp)
)

test_that("common lazy-default pipelines raise no notices, even with both notices on", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  keys <- f80_keys()
  for (nm in names(f80_common_pipelines)) {
    fn <- f80_common_pipelines[[nm]]
    # defaults: nothing at all
    expect_silent(res <- collect(fn(tbl_gpu(df), tbl_gpu(keys))))
    expect_equal(f80_rows(res), f80_rows(fn(df, keys)), info = nm)
    # notices on: still nothing
    withr::with_options(list(cuplyr.plan_break = "warn", cuplyr.fallback = "warn"), {
      expect_no_condition(collect(fn(tbl_gpu(df), tbl_gpu(keys))), class = "cuplyr_plan_break")
      expect_no_condition(collect(fn(tbl_gpu(df), tbl_gpu(keys))), class = "cuplyr_cpu_fallback")
      expect_no_warning(collect(fn(tbl_gpu(df), tbl_gpu(keys))))
    })
  }
})

# ---------------------------------------------------------------------------
# 5. Dead options and test helpers
# ---------------------------------------------------------------------------

test_that(".onLoad no longer sets the dead options cuplyr.lazy and cuplyr.verbose", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.lazy = NULL, cuplyr.verbose = NULL)
  cuplyr:::.onLoad(dirname(system.file(package = "cuplyr")), "cuplyr")
  expect_null(getOption("cuplyr.lazy"))
  expect_null(getOption("cuplyr.verbose"))
  expect_false(is.null(getOption("cuplyr.gpu_available")))
})

test_that("expect_same_as_dplyr() tests eager tables and expect_same_as_dplyr_lazy() lazy ones", {
  skip_if_no_gpu()
  f80_local_default()
  df <- f80_df()
  seen <- character()
  pipeline <- function(d) {
    if (is_tbl_gpu(d)) seen <<- c(seen, f80_mode(d))
    dplyr::filter(d, g > 1)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_identical(seen, "eager")
  seen <- character()
  expect_same_as_dplyr_lazy(df, pipeline)
  expect_identical(seen, "lazy")
})

# ---------------------------------------------------------------------------
# 6. Lazy-mode gaps the default exposes (the conformance probe's flips)
# ---------------------------------------------------------------------------

test_that("nrow() and dim() of lazy joins and set operations give the result's size", {
  skip_if_no_gpu()
  f80_local_default()
  x <- tibble::tibble(k = c(1L, 1L, 2L, NA), a = 1:4)
  y <- tibble::tibble(k = c(1L, 3L, NA), b = c("p", "q", "r"))
  cases <- list(
    inner = function(a, b) dplyr::inner_join(a, b, by = "k"),
    left  = function(a, b) dplyr::left_join(a, b, by = "k"),
    full  = function(a, b) dplyr::full_join(a, b, by = "k"),
    semi  = function(a, b) dplyr::semi_join(a, b, by = "k"),
    anti  = function(a, b) dplyr::anti_join(a, b, by = "k"),
    cross = function(a, b) dplyr::cross_join(a, dplyr::select(b, b)),
    pending = function(a, b) dplyr::inner_join(dplyr::filter(a, a > 1), b, by = "k"),
    union = function(a, b) dplyr::union(dplyr::select(a, k), dplyr::select(b, k)),
    intersect = function(a, b) dplyr::intersect(dplyr::select(a, k), dplyr::select(b, k)),
    setdiff = function(a, b) dplyr::setdiff(dplyr::select(a, k), dplyr::select(b, k)),
    bind_rows = function(a, b) bind_rows(dplyr::select(a, k), dplyr::select(b, k))
  )
  for (nm in names(cases)) {
    gpu <- cases[[nm]](tbl_gpu(x), tbl_gpu(y))
    cpu <- cases[[nm]](x, y)
    expect_identical(nrow(gpu), nrow(cpu), info = nm)
    expect_identical(dim(gpu), dim(cpu), info = nm)
    expect_identical(ncol(gpu), ncol(cpu), info = nm)
  }
})

test_that("join checks that read the data (unmatched, relationship) error at the verb on the lazy default", {
  skip_if_no_gpu()
  f80_local_default()
  withr::local_options(lifecycle_verbosity = "quiet")
  x <- tibble::tibble(k = c(1L, 1L, 2L), a = 1:3)
  y <- tibble::tibble(k = c(1L, 3L), b = c("p", "q"))
  one <- tibble::tibble(x = 1:2)
  cases <- list(
    left_unmatched  = function(a, b) dplyr::left_join(a, b, by = "k", unmatched = "error"),
    inner_unmatched = function(a, b) dplyr::inner_join(a, b, by = "k", unmatched = "error"),
    one_to_one      = function(a, b) dplyr::left_join(a, b, by = "k", relationship = "one-to-one"),
    one_to_many     = function(a, b) dplyr::inner_join(b, a, by = "k", relationship = "many-to-one"),
    pending_x       = function(a, b) dplyr::left_join(dplyr::filter(a, a > 0), b, by = "k", relationship = "one-to-one"),
    cross_rel       = function(a, b) dplyr::left_join(dplyr::select(a, a), dplyr::select(a, a), by = character(), relationship = "many-to-one")
  )
  for (nm in names(cases)) {
    fn <- cases[[nm]]
    expect_error(fn(x, y), info = paste(nm, "(dplyr)"))
    e_lazy <- f80_capture_error(fn(tbl_gpu(x), tbl_gpu(y)))
    e_eager <- f80_capture_error(fn(tbl_gpu(x, lazy = FALSE), tbl_gpu(y, lazy = FALSE)))
    expect_false(is.null(e_lazy), info = paste(nm, "(lazy verb did not error)"))
    expect_false(is.null(e_eager), info = paste(nm, "(eager verb did not error)"))
    if (!is.null(e_lazy) && !is.null(e_eager)) {
      expect_identical(class(e_lazy), class(e_eager), info = nm)
      expect_identical(conditionMessage(e_lazy), conditionMessage(e_eager), info = nm)
    }
  }
  # a zero-column x with `unmatched = "error"` (dplyr's own conformance case)
  e0 <- f80_capture_error(dplyr::left_join(tbl_gpu(tibble::tibble()), tbl_gpu(tibble::tibble(x = 1)),
                                           by = character(), unmatched = "error"))
  expect_s3_class(e0, "error")
  # checks that pass keep the join lazy and give dplyr's rows
  ok <- dplyr::left_join(tbl_gpu(x), tbl_gpu(y), by = "k", relationship = "many-to-one")
  expect_identical(f80_mode(ok), "lazy")
  expect_equal(f80_rows(ok), f80_rows(dplyr::left_join(x, y, by = "k", relationship = "many-to-one")))
})

test_that("a vector constant of the wrong size errors at the verb on the lazy default", {
  skip_if_no_gpu()
  f80_local_default()
  df <- tibble::tibble(x = 1:4, g = c(1L, 1L, 2L, 2L))
  z <- 1:5
  cases <- list(
    fresh        = function(d) dplyr::mutate(d, y = 1:2),
    injected     = function(d) dplyr::mutate(d, z = !!z),
    pending      = function(d) dplyr::mutate(dplyr::filter(d, x > 1), y = 1:4),
    twice        = function(d) dplyr::mutate(d, y = 11:14, y = 1:2),
    transmute    = function(d) dplyr::transmute(d, y = 1:2),
    group_by     = function(d) dplyr::group_by(d, z = 1:3)
  )
  for (nm in names(cases)) {
    fn <- cases[[nm]]
    expect_error(fn(df), info = paste(nm, "(dplyr)"))
    e_lazy <- f80_capture_error(fn(tbl_gpu(df)))
    e_eager <- f80_capture_error(fn(tbl_gpu(df, lazy = FALSE)))
    expect_false(is.null(e_lazy), info = paste(nm, "(lazy verb did not error)"))
    expect_false(is.null(e_eager), info = paste(nm, "(eager verb did not error)"))
    if (!is.null(e_lazy) && !is.null(e_eager)) {
      expect_identical(class(e_lazy), class(e_eager), info = nm)
      expect_identical(conditionMessage(e_lazy), conditionMessage(e_eager), info = nm)
    }
  }
  # the right size works, also after pending ops
  good <- dplyr::mutate(dplyr::filter(tbl_gpu(df), x > 1), y = 1:3)
  expect_equal(f80_rows(good), f80_rows(dplyr::mutate(dplyr::filter(df, x > 1), y = 1:3)))
  expect_equal(f80_rows(dplyr::mutate(tbl_gpu(df), y = 4:1)), f80_rows(dplyr::mutate(df, y = 4:1)))
})

test_that("rowwise mutate() that makes list columns gives dplyr's result on the lazy default", {
  skip_if_no_gpu()
  f80_local_default()
  pipelines <- list(
    unlist_new = function(d) dplyr::mutate(dplyr::rowwise(d), y = list(1), z = y),
    chained    = function(d) dplyr::mutate(dplyr::mutate(dplyr::rowwise(d), a = list(1)), b = list(a + 1))
  )
  df <- tibble::tibble(x = 1:3)
  for (nm in names(pipelines)) {
    fn <- pipelines[[nm]]
    lz <- f80_capture_error(r_lazy <- fn(tbl_gpu(df)))
    expect_null(lz, info = nm)
    if (!is.null(lz)) next
    r_eager <- fn(tbl_gpu(df, lazy = FALSE))
    r_cpu <- fn(df)
    if (is_tbl_gpu(r_lazy)) r_lazy <- collect(r_lazy)
    if (is_tbl_gpu(r_eager)) r_eager <- collect(r_eager)
    expect_identical(class(r_lazy), class(r_eager), info = nm)
    expect_equal(r_lazy, r_eager, info = nm)
    expect_equal(tibble::as_tibble(r_lazy), tibble::as_tibble(r_cpu), info = nm)
  }
})

# ---------------------------------------------------------------------------
# 7. An integer sum() past R's integer range is a double, as in dplyr
#    (round-1 review B5): the old eager default matched dplyr here, the lazy
#    default must too.
# ---------------------------------------------------------------------------

f80_sum_pipelines <- list(
  mutate        = function(x) dplyr::mutate(x, s = sum(i)),
  mutate_grp    = function(x) dplyr::mutate(dplyr::group_by(x, g), s = sum(i)),
  mutate_by     = function(x) dplyr::mutate(x, s = sum(i), .by = g),
  summarise_grp = function(x) dplyr::summarise(dplyr::group_by(x, g), s = sum(i)),
  summarise_all = function(x) dplyr::summarise(x, s = sum(i), n = dplyr::n()),
  summarise_by  = function(x) dplyr::summarise(x, s = sum(i), .by = g),
  pending       = function(x) dplyr::summarise(dplyr::group_by(dplyr::filter(x, g > 0L), g), s = sum(i)),
  downstream    = function(x) dplyr::mutate(dplyr::summarise(dplyr::group_by(x, g), s = sum(i)), t = s + 1L),
  rowwise       = function(x) dplyr::mutate(dplyr::rowwise(x), s = sum(dplyr::c_across(c(g, i)))),
  logical_sum   = function(x) dplyr::summarise(dplyr::group_by(x, g), s = sum(b))
)

f80_plain <- function(x) {
  if (is_tbl_gpu(x)) x <- collect(x)
  tibble::as_tibble(dplyr::ungroup(x))
}

test_that("an integer sum() that leaves the integer range is dplyr's double on the lazy default", {
  skip_if_no_gpu()
  f80_local_default()
  big <- tibble::tibble(g = c(1L, 1L, 2L), i = c(.Machine$integer.max, 1L, 5L), b = c(TRUE, TRUE, FALSE))
  small <- tibble::tibble(g = c(1L, 1L, 2L), i = c(7L, 1L, 5L), b = c(TRUE, NA, FALSE))
  for (data in list(big = big, small = small)) {
    for (nm in names(f80_sum_pipelines)) {
      fn <- f80_sum_pipelines[[nm]]
      cpu <- f80_plain(fn(data))
      lz <- f80_capture_error(gpu_lazy <- fn(tbl_gpu(data)))
      expect_null(lz, info = nm)
      if (!is.null(lz)) next
      expect_identical(f80_mode(gpu_lazy), "lazy", info = nm)
      got <- f80_capture_error(res <- f80_plain(gpu_lazy))
      expect_null(got, info = paste(nm, "(collect)"))
      if (!is.null(got)) next
      # values and column types, as dplyr gives them (integer in range, double past it)
      expect_equal(res, cpu, info = nm)
      expect_identical(vapply(res, typeof, ""), vapply(cpu, typeof, ""), info = nm)
      # and as the eager table gives them
      expect_equal(f80_plain(fn(tbl_gpu(data, lazy = FALSE))), res, info = paste(nm, "(eager)"))
    }
  }
})

# ---------------------------------------------------------------------------
# 8. filter() with an integer window sum on the lazy default matches dplyr
#    (round-2 review B1), and an integer sum on a table with nothing pending
#    raises no plan-break notice (round-2 review N2).
# ---------------------------------------------------------------------------

f80_with_warnings <- function(expr) {
  ws <- character()
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e) e),
    warning = function(w) {
      ws <<- c(ws, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = ws)
}

test_that("filter() and filter_out() with an integer window sum match dplyr on the lazy default", {
  skip_if_no_gpu()
  f80_local_default()
  M <- .Machine$integer.max
  d <- tibble::tibble(g = c(1, 1, 2, 2), x = c(M, 5L, 50000L, 50000L))
  e <- tibble::tibble(x = c(1L, 2L), y = c(M, 5L))
  cases <- list(
    by_mixed      = list(d, function(t) dplyr::filter(t, x * sum(x) > 0L, .by = g)),
    grouped_mixed = list(d, function(t) dplyr::filter(dplyr::group_by(t, g), x * sum(x) > 0L)),
    nested        = list(e, function(t) dplyr::filter(t, max(x * sum(y)) > 0L)),
    plain         = list(d, function(t) dplyr::filter(t, sum(x) > 10L, .by = g)),
    pending       = list(d, function(t) dplyr::filter(dplyr::filter(t, g > 0), x * sum(x) > 0L, .by = g)),
    filter_out    = list(d, function(t) dplyr::filter_out(t, x * sum(x) > 0L, .by = g))
  )
  for (nm in names(cases)) {
    data <- cases[[nm]][[1]]
    fn <- cases[[nm]][[2]]
    cpu <- f80_with_warnings(fn(data))
    expect_false(inherits(cpu$value, "error"), info = paste(nm, "(dplyr)"))
    gpu <- f80_with_warnings({
      res <- fn(tbl_gpu(data))
      stopifnot(identical(f80_mode(res), "lazy"))
      collect(res)
    })
    expect_false(inherits(gpu$value, "error"), info = paste(nm, ":", if (inherits(gpu$value, "error")) conditionMessage(gpu$value)))
    if (inherits(gpu$value, "error")) next
    expect_equal(f80_plain(gpu$value), f80_plain(cpu$value), info = nm)
    # dplyr's "NAs produced by integer overflow" warning is raised (or not) alike
    expect_identical(length(gpu$warnings) > 0L, length(cpu$warnings) > 0L, info = nm)
    eg <- f80_with_warnings(collect(fn(tbl_gpu(data, lazy = FALSE))))
    expect_equal(f80_plain(eg$value), f80_plain(gpu$value), info = paste(nm, "(eager)"))
  }
})

test_that("an integer sum on a table with nothing pending raises no plan-break notice", {
  skip_if_no_gpu()
  f80_local_default()
  withr::local_options(cuplyr.plan_break = "warn")
  d <- tibble::tibble(g = c(1L, 1L, 2L), x = c(.Machine$integer.max, 1L, 5L))
  cases <- list(
    summarise_by  = function(t) dplyr::summarise(t, total = sum(x), .by = g),
    summarise_grp = function(t) dplyr::summarise(dplyr::group_by(t, g), total = sum(x)),
    count_wt      = function(t) dplyr::count(t, g, wt = x),
    mutate_by     = function(t) dplyr::mutate(t, total = sum(x), .by = g)
  )
  for (nm in names(cases)) {
    fn <- cases[[nm]]
    expect_no_condition(res <- collect(fn(tbl_gpu(d))), class = "cuplyr_plan_break")
    expect_equal(f80_plain(res), f80_plain(fn(d)), info = nm)
  }
})
