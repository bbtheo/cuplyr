# F111 acceptance, items 1, 2, 4 and 6: a NULL read inside a column-reading dot,
# a column-free dot that reaches the data mask by name, a missing argument read
# through `.env`, and a non-trailing empty dot each match dplyr: the same value,
# or the same error class. Eager and lazy, grouped where it applies.

# Tests see the package namespace, which doesn't import these; bind the dplyr
# functions these pipelines call bare, as library(dplyr) would for a user.
for (.f111s in c("if_else", "if_any", "if_all", "across", "coalesce", "desc", "n", "pick")) {
  if (!exists(.f111s, mode = "function")) assign(.f111s, getExportedValue("dplyr", .f111s))
}
rm(.f111s)

f111s_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3, NA, 5),
    y2 = c(5, 4, 3, 2, 1),
    g = c("a", "a", "b", "b", "b"),
    flag = c(TRUE, FALSE, TRUE, FALSE, TRUE)
  )
}

f111s_cnd <- function(expr) {
  tryCatch(
    {
      force(expr)
      NULL
    },
    error = function(e) e
  )
}

# The same error, in class and in the class of its parent, eager and lazy.
f111s_same_error <- function(df, pipeline, info = NULL) {
  oracle <- f111s_cnd(pipeline(tibble::as_tibble(df)))
  expect_false(is.null(oracle), info = paste("dplyr must error:", info))
  for (lazy in c(FALSE, TRUE)) {
    got <- f111s_cnd(collect(pipeline(tbl_gpu(df, lazy = lazy))))
    label <- paste0(info, if (lazy) " (lazy)" else " (eager)")
    expect_false(is.null(got), info = paste("the GPU side must error:", label))
    if (!is.null(got) && !is.null(oracle)) {
      expect_identical(class(got), class(oracle), info = label)
      expect_identical(class(got$parent), class(oracle$parent), info = label)
    }
  }
}

f111s_same_value <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Runs natively (a CPU fallback is an error here), eager and lazy.
f111s_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f111s_same_value(df, pipeline, ...)
}

# ---------------------------------------------------------------------------
# Item 1: a variable read that returns NULL
# ---------------------------------------------------------------------------

test_that("F111 item 1: filter() on a NULL read inside a column-reading dot errors like dplyr", {
  skip_if_no_gpu()
  df <- f111s_df()
  cfg <- list(thr = 2)
  nul <- NULL
  cases <- list(
    field = function(t) filter(t, x > cfg$trh),
    symbol = function(t) filter(t, x > nul),
    env_dollar = function(t) filter(t, x > .env$nul),
    env_bracket = function(t) filter(t, x > .env[["nul"]]),
    source_null = function(t) filter(t, x > NULL),
    bare_null = function(t) filter(t, NULL),
    bare_field = function(t) filter(t, cfg$trh),
    and = function(t) filter(t, x > 1 & y2 > cfg$trh),
    second_dot = function(t) filter(t, x > 1, y2 > cfg$trh),
    grouped = function(t) filter(group_by(t, g), x > cfg$trh),
    by = function(t) filter(t, x > cfg$trh, .by = g),
    if_any = function(t) filter(t, if_any(c(x, y2), ~ .x > cfg$trh)),
    if_all = function(t) filter(t, if_all(c(x, y2), ~ .x > cfg$trh))
  )
  for (nm in names(cases)) f111s_same_error(df, cases[[nm]], info = nm)
})

test_that("F111 item 1: mutate()/transmute() on a NULL read inside an expression errors like dplyr", {
  skip_if_no_gpu()
  df <- f111s_df()
  cfg <- list(thr = 2)
  nul <- NULL
  cases <- list(
    arith = function(t) mutate(t, y = x + cfg$trh),
    symbol = function(t) mutate(t, y = x + nul),
    env = function(t) mutate(t, y = x + .env$nul),
    source_null = function(t) mutate(t, y = x + NULL),
    if_else_true = function(t) mutate(t, y = if_else(x > 1, cfg$trh, 0)),
    if_else_false = function(t) mutate(t, y = if_else(x > 1, 0, nul)),
    grouped = function(t) mutate(group_by(t, g), y = x + cfg$trh),
    by = function(t) mutate(t, y = x + cfg$trh, .by = g),
    across = function(t) mutate(t, across(x, ~ .x + cfg$trh)),
    across_two = function(t) mutate(t, across(c(x, y2), ~ .x + nul)),
    transmute = function(t) transmute(t, y = x + cfg$trh)
  )
  for (nm in names(cases)) f111s_same_error(df, cases[[nm]], info = nm)
})

test_that("F111 item 1: a NULL through a date fold errors like dplyr (review round 1)", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- f111s_df()
  cfg <- list(thr = 2)
  floor_date <- getExportedValue("lubridate", "floor_date")
  ceiling_date <- getExportedValue("lubridate", "ceiling_date")
  cases <- list(
    filter_floor = function(t) filter(t, is.na(floor_date(cfg$nope, "day")) | y2 > 1),
    filter_ceiling = function(t) filter(t, is.na(ceiling_date(cfg$nope, "day")) | y2 > 1),
    if_else = function(t) mutate(t, z = if_else(y2 > 1, floor_date(cfg$nope, "day"), NA)),
    arith = function(t) mutate(t, z = y2 + is.na(floor_date(cfg$nope, "day")))
  )
  for (nm in names(cases)) f111s_same_error(df, cases[[nm]], info = nm)
})

test_that("F111 item 1: a NULL body of a one-column if_any()/if_all() in mutate() errors like dplyr", {
  skip_if_no_gpu()
  df <- f111s_df()
  cfg <- list(thr = 2)
  nul <- NULL
  cases <- list(
    if_any = function(t) mutate(t, z = if_any(x, ~ cfg$trh)),
    if_all = function(t) mutate(t, z = if_all(x, ~ cfg$trh)),
    fn = function(t) mutate(t, z = if_any(x, function(v) cfg$trh)),
    symbol = function(t) mutate(t, z = if_any(x, ~ nul)),
    grouped = function(t) mutate(group_by(t, g), z = if_any(x, ~ cfg$trh)),
    transmute = function(t) transmute(t, z = if_any(x, ~ cfg$trh))
  )
  for (nm in names(cases)) f111s_same_error(df, cases[[nm]], info = nm)
})

test_that("F111 item 1: an explicit missing = NULL stays native", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_native(df, function(t) mutate(t, z = if_else(y2 > 1, 1, 0, missing = NULL)))
})

test_that("F111 item 1: the shapes where NULL means something still match dplyr", {
  skip_if_no_gpu()
  df <- f111s_df()
  cfg <- list(thr = 2)
  nul <- NULL
  f111s_same_value(df, function(t) mutate(t, y = cfg$trh))
  f111s_same_value(df, function(t) mutate(t, x = cfg$trh))
  f111s_same_value(df, function(t) mutate(t, x = nul))
  f111s_same_value(df, function(t) transmute(t, y = cfg$trh))
  f111s_same_value(df, function(t) summarise(t, y = cfg$trh))
  f111s_same_value(df, function(t) mutate(group_by(t, g), y = cfg$trh))
  f111s_same_value(df, function(t) filter(t, x %in% cfg$trh))
  f111s_same_value(df, function(t) filter(t, x %in% NULL))
  f111s_same_value(df, function(t) mutate(t, y = coalesce(x, cfg$trh)))
  f111s_same_value(df, function(t) mutate(t, y = coalesce(x, nul, 0)))
  f111s_same_value(df, function(t) arrange(t, cfg$trh, desc(y2)))
  f111s_same_value(df, function(t) mutate(t, across(x, ~ cfg$trh)))
})

test_that("F111 item 1: on zero rows a size-0 value is valid, as in dplyr", {
  skip_if_no_gpu()
  df <- f111s_df()
  cfg <- list(thr = 2)
  f111s_same_value(df, function(t) filter(filter(t, y2 > 99), y2 > cfg$trh))
  f111s_same_value(df, function(t) mutate(filter(t, y2 > 99), y = y2 + cfg$trh))
})

test_that("F111 item 1: a variable holding a value stays on the GPU", {
  skip_if_no_gpu()
  df <- f111s_df()
  cfg <- list(thr = 2)
  v <- 3
  f111s_native(df, function(t) filter(t, y2 > cfg$thr))
  f111s_native(df, function(t) filter(t, y2 > v))
  f111s_native(df, function(t) filter(t, y2 > .env$v))
  f111s_native(df, function(t) mutate(t, y = y2 + cfg$thr))
  f111s_native(df, function(t) mutate(t, y = if_else(y2 > 1, cfg$thr, 0)))
  f111s_native(df, function(t) filter(t, if_any(c(x, y2), ~ .x > cfg$thr)))
})

# ---------------------------------------------------------------------------
# Item 2: a column-free dot that reaches the data mask by name
# ---------------------------------------------------------------------------

test_that("F111 item 2: mutate()/transmute() see the mask through exists(), get0() and friends", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_same_value(df, function(t) mutate(t, z = exists("flag")))
  f111s_same_value(df, function(t) mutate(t, z = !exists("flag")))
  f111s_same_value(df, function(t) mutate(t, z = base::exists("flag")))
  f111s_same_value(df, function(t) mutate(t, z = exists("no_such_name_f111")))
  f111s_same_value(df, function(t) mutate(t, x = exists("flag")))
  f111s_same_value(df, function(t) mutate(t, z = get0("y2")))
  f111s_same_value(df, function(t) mutate(t, z = get0("no_such_name_f111", ifnotfound = 5)))
  f111s_same_value(df, function(t) transmute(t, z = exists("flag")))
  f111s_same_value(df, function(t) transmute(t, z = get0("y2")))
  f111s_same_value(df, function(t) mutate(group_by(t, g), z = exists("flag")))
  f111s_same_value(df, function(t) mutate(t, z = exists("flag"), .by = g))
  f111s_same_value(df, function(t) mutate(t, a = y2 + 1, z = exists("a")))
  f111s_same_value(df, function(t) mutate(t, a = y2 + 1, z = get0("a")))
})

test_that("F111 item 2: across() and group_by() bodies see the mask too", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_same_value(df, function(t) mutate(t, across(x, ~ exists("flag"))))
  f111s_same_value(df, function(t) mutate(t, across(c(x, y2), ~ exists("flag"))))
  f111s_same_value(df, function(t) filter(t, if_any(x, ~ exists("flag"))))
  f111s_same_value(df, function(t) mutate(t, z = if_any(x, ~ exists("flag"))))
  f111s_same_value(df, function(t) group_by(t, z = exists("flag")))
  f111s_same_value(df, function(t) ungroup(group_by(t, z = exists("flag"))))
})

test_that("F111 item 2: slice() indices see the mask", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_same_value(df, function(t) slice(t, if (exists("flag")) 2L else 1L))
  f111s_same_value(df, function(t) slice(t, if (base::exists("flag")) 2L else 1L))
  f111s_same_value(df, function(t) slice(t, if (exists("no_such_name_f111")) 2L else 1L))
  f111s_same_value(df, function(t) slice(group_by(t, g), if (exists("flag")) 1L else 2L))
  f111s_same_value(df, function(t) slice(t, if (exists("flag")) 1L else 2L, .by = g), arrange_by = "y2")
  f111s_same_value(df, function(t) slice(filter(t, y2 > 99), if (exists("flag")) 1L else 2L))
})

test_that("F111 item 2: what F79 already fixed stays fixed, and plain constants stay native", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_same_value(df, function(t) filter(t, exists("x")))
  f111s_same_value(df, function(t) filter(t, exists("no_such_name_f111")))
  f111s_same_value(df, function(t) filter(t, y2 > exists("flag")))
  f111s_same_value(df, function(t) summarise(t, z = exists("flag")))
  # no mask reached: no CPU route
  f111s_native(df, function(t) mutate(t, z = 1 + 1))
  f111s_native(df, function(t) mutate(t, z = identity(2)))
  f111s_native(df, function(t) slice(t, 2L))
  f111s_native(df, function(t) slice(t, 1:2))
})

# ---------------------------------------------------------------------------
# Item 4: `.env$name` where `name` is a missing function argument
# ---------------------------------------------------------------------------

test_that("F111 item 4: a missing argument read through .env is dplyr's wrapped error", {
  skip_if_no_gpu()
  df <- f111s_df()
  cases <- list(
    filter_dollar = function(t, thr) filter(t, x > .env$thr),
    filter_bracket = function(t, thr) filter(t, x > .env[["thr"]]),
    filter_bare = function(t, thr) filter(t, .env$thr),
    filter_second = function(t, thr) filter(t, x > 1, y2 > .env$thr),
    filter_grouped = function(t, thr) filter(group_by(t, g), x > .env$thr),
    if_any = function(t, thr) filter(t, if_any(c(x, y2), ~ .x > .env$thr)),
    if_all = function(t, thr) filter(t, if_all(x, ~ .x > .env[["thr"]])),
    mutate_arith = function(t, thr) mutate(t, y = x + .env$thr),
    mutate_bare = function(t, thr) mutate(t, y = .env$thr),
    mutate_bracket = function(t, thr) mutate(t, y = x + .env[["thr"]]),
    mutate_grouped = function(t, thr) mutate(group_by(t, g), y = x + .env$thr),
    mutate_if_any = function(t, thr) mutate(t, y = if_any(x, ~ .x > .env$thr)),
    across = function(t, thr) mutate(t, across(x, ~ .x + .env$thr)),
    transmute = function(t, thr) transmute(t, y = .env$thr),
    group_by = function(t, thr) group_by(t, y = x + .env$thr)
  )
  for (nm in names(cases)) {
    fn <- cases[[nm]]
    f111s_same_error(df, function(t) fn(t), info = nm)
  }
})

test_that("F111 item 4: a missing argument as a .env[[<symbol>]] subscript is dplyr's error", {
  skip_if_no_gpu()
  df <- f111s_df()
  fn <- function(t, thr) filter(t, x > .env[[thr]])
  gn <- function(t, thr) mutate(t, y = x + .env[[thr]])
  f111s_same_error(df, function(t) fn(t), info = "filter")
  f111s_same_error(df, function(t) gn(t), info = "mutate")
})

test_that("F111 item 4: an empty call argument is dplyr's error, not the read's", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_same_error(df, function(t) mutate(t, z = if_else(y2 > 1, , 0)), info = "mutate")
  f111s_same_error(df, function(t) filter(t, if_else(y2 > 1, , TRUE)), info = "filter")
})

test_that("F111 item 4: the same for a plain symbol naming a missing argument", {
  skip_if_no_gpu()
  df <- f111s_df()
  cases <- list(
    filter = function(t, thr) filter(t, x > thr),
    mutate = function(t, thr) mutate(t, y = x + thr),
    if_any = function(t, thr) filter(t, if_any(x, ~ .x > thr))
  )
  for (nm in names(cases)) {
    fn <- cases[[nm]]
    f111s_same_error(df, function(t) fn(t), info = nm)
  }
})

test_that("F111 item 4: a promise that errors through .env is dplyr's wrapped error", {
  skip_if_no_gpu()
  df <- f111s_df()
  fn <- function(t, thr) filter(t, x > .env$thr)
  suppressWarnings(f111s_same_error(df, function(t) fn(t, stop("boom")), info = "promise"))
})

test_that("F111 item 4: a supplied argument read through .env stays on the GPU", {
  skip_if_no_gpu()
  df <- f111s_df()
  fn <- function(t, thr) filter(t, y2 > .env$thr)
  gn <- function(t, thr) mutate(t, y = y2 + .env[["thr"]])
  f111s_native(df, function(t) fn(t, 2))
  f111s_native(df, function(t) gn(t, 2))
})

# ---------------------------------------------------------------------------
# Item 6: a non-trailing empty dot
# ---------------------------------------------------------------------------

test_that("F111 item 6: verbs that drop an empty dot drop it", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_same_value(df, function(t) mutate(t, , y = y2 + 1))
  f111s_same_value(df, function(t) mutate(t, , , y = y2 + 1))
  f111s_same_value(df, function(t) mutate(t, y = y2 + 1, , z = y2 * 2))
  f111s_same_value(df, function(t) mutate(t, , ))
  f111s_same_value(df, function(t) mutate(group_by(t, g), , y = y2 + 1))
  f111s_same_value(df, function(t) filter(t, , y2 > 2))
  f111s_same_value(df, function(t) filter(t, y2 > 2, , x > 1))
  f111s_same_value(df, function(t) filter(t, , ))
  f111s_same_value(df, function(t) filter(group_by(t, g), , y2 > 2))
  f111s_same_value(df, function(t) filter(filter(t, y2 > 99), , y2 > 0))
  f111s_same_value(df, function(t) summarise(t, , s = sum(y2)))
  f111s_same_value(df, function(t) summarise(t, , ))
  f111s_same_value(df, function(t) summarise(group_by(t, g), , s = sum(y2)))
  f111s_same_value(df, function(t) reframe(t, , s = sum(y2)))
  f111s_same_value(df, function(t) arrange(t, , desc(y2)))
  f111s_same_value(df, function(t) arrange(t, , ))
  f111s_same_value(df, function(t) group_by(t, , g))
  f111s_same_value(df, function(t) count(t, , g), arrange_by = "g")
  f111s_same_value(df, function(t) count(t, , ))
  f111s_same_value(df, function(t) add_count(t, , g))
})

test_that("F111 item 6: the verbs that capture with .ignore_empty = \"all\" stay native", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_native(df, function(t) mutate(t, , y = y2 + 1))
  f111s_native(df, function(t) filter(t, , y2 > 2))
  f111s_native(df, function(t) summarise(t, , s = sum(y2)))
  f111s_native(df, function(t) reframe(t, , s = sum(y2)))
})

test_that("F111 item 6: verbs that reject an empty dot reject it the same way", {
  skip_if_no_gpu()
  df <- f111s_df()
  cases <- list(
    transmute = function(t) transmute(t, , y = y2 + 1),
    slice = function(t) slice(t, , 1L),
    distinct = function(t) distinct(t, , g),
    named_empty = function(t) mutate(t, y = , z = y2)
  )
  for (nm in names(cases)) f111s_same_error(df, cases[[nm]], info = nm)
})

test_that("F111 item 6: a trailing empty dot was and stays fine, on the GPU", {
  skip_if_no_gpu()
  df <- f111s_df()
  f111s_native(df, function(t) mutate(t, y = y2 + 1, ))
  f111s_native(df, function(t) filter(t, y2 > 2, ))
  f111s_native(df, function(t) select(t, , y2))
})
