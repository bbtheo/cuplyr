# Acceptance tests for #13: a column that doesn't exist raises dplyr's error.
#
# Contract (dplyr 1.2.1 on a tibble is the oracle, compared live):
# - Every verb that names or evaluates a column raises, for a column the table
#   doesn't have, the condition dplyr raises for the same call: the same class
#   vector, the same call (`Error in `filter()`:`, never an internal
#   `dplyr::summarise(.data = tbl, ...)` or NULL), the same message bullets, the
#   same parent class and the same full message (parent chain included).
#   Eager and lazy tables alike.
#   * filter(): dplyr's "In argument: `<dot>`." wrapper (with the group bullet
#     for grouped and `.by` calls), the original error as parent.
#   * arrange(): dplyr's `dplyr:::mutate_error` "In argument: `..<i> = <expr>`."
#   * summarise()/reframe()/slice_min()/slice_max()/count(wt =)/tally(wt =)/
#     add_count(wt =)/add_tally(wt =)/nest_join(): the user's verb call.
#   * joins: "Join columns in `x`/`y` must be present in the data." with
#     "Problem with `a` and `b`.".
#   * group_by(): no extra bullets beyond dplyr's; every missing column named.
#   * relocate(): tidyselect's "Can't select columns that don't exist.".
#   * `.data$col` of a missing column: rlang's pronoun error as the parent.
# - dplyr gives no "did you mean" hint for any of these, so neither does cuplyr.
# - A missing column is reported as dplyr reports it, not turned into a CPU
#   fallback first: no `cuplyr_cpu_fallback` is signalled, and with
#   `options(cuplyr.fallback = "error")` the error is still dplyr's.
# - A name that is not a column but is bound in the caller's environment is
#   not a missing column: arrange() sorts by it (or ignores a NULL) as dplyr
#   does (dplyr conformance: test-arrange.R "arrange ignores NULLs (#6193)").

p13_df <- function() {
  tibble::tibble(
    x = c(2L, 1L, 3L),
    y = c(1.5, NA, 3),
    g = c("a", "a", "b")
  )
}

p13_df2 <- function() {
  tibble::tibble(x = c(1L, 2L, 4L), z = c(4L, 5L, 6L))
}

# `other` as a table of the same kind as `d` (tibble for dplyr, tbl_gpu of the
# same mode for cuplyr), for the joins.
p13_like <- function(d, other) {
  if (inherits(d, "tbl_gpu")) tbl_gpu(other, lazy = identical(d$exec_mode, "lazy")) else other
}

# The condition `pipeline(x)` raises (collecting a tbl_gpu result), or NULL.
p13_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    }),
    error = function(e) e
  )
}

# The bullets of an rlang error's own message (no parent), as plain strings.
p13_bullets <- function(e) {
  m <- c(e$message, if (is.character(e$body)) e$body)
  stats::setNames(as.character(m), names(m))
}

p13_call <- function(e) {
  call <- e$call
  if (is.null(call)) NA_character_ else rlang::format_error_call(call)
}

# dplyr and cuplyr (eager and lazy) raise the same condition: same class, same
# call, same bullets, same parent class and the same full message. `gpu_opts`
# are options set for the cuplyr side only.
p13_same_error <- function(df, pipeline, gpu_opts = list()) {
  cpu <- p13_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"), info = "dplyr itself errors")
  if (!inherits(cpu, "error")) {
    return(invisible())
  }
  for (lazy in c(FALSE, TRUE)) {
    mode <- if (lazy) "lazy" else "eager"
    gpu <- withr::with_options(gpu_opts, p13_err(tbl_gpu(df, lazy = lazy), pipeline))
    expect_true(inherits(gpu, "error"), info = sprintf("GPU side errors (%s)", mode))
    if (!inherits(gpu, "error")) next
    expect_identical(class(gpu), class(cpu),
                     info = sprintf("error class (%s): %s", mode, conditionMessage(gpu)))
    expect_identical(p13_call(gpu), p13_call(cpu), info = sprintf("error call (%s)", mode))
    expect_identical(p13_bullets(gpu), p13_bullets(cpu), info = sprintf("bullets (%s)", mode))
    expect_identical(class(gpu$parent), class(cpu$parent),
                     info = sprintf("parent class (%s)", mode))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu),
                     info = sprintf("full message (%s)", mode))
  }
  invisible(cpu)
}

test_that("filter() reports a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) filter(d, xx > 1))
  expect_identical(p13_call(cpu), "`filter()`")
  p13_same_error(df, function(d) filter(group_by(d, g), xx > 1))
  p13_same_error(df, function(d) filter(d, xx > 1, .by = g))
  p13_same_error(df, function(d) filter(d, x > 1, xx > 1))
  p13_same_error(df, function(d) filter(d, xx))
  p13_same_error(df, function(d) filter(d, .data$xx > 1))
  p13_same_error(df, function(d) filter(d, dplyr::if_else(xx > 1, TRUE, FALSE)))
  p13_same_error(df, function(d) filter(d, is.na(xx)))
  p13_same_error(df, function(d) filter(d, if_any(c(x, xx), ~ .x > 1)))
  p13_same_error(df, function(d) filter(rowwise(d), xx > 1))
})

test_that("filter() on a grouped table with no rows reports a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()[0, ]
  p13_same_error(df, function(d) filter(d, xx > 1))
  p13_same_error(df, function(d) filter(group_by(d, g), xx > 1))
})

test_that("arrange() reports a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) arrange(d, xx))
  expect_true(inherits(cpu, "dplyr:::mutate_error"))
  p13_same_error(df, function(d) arrange(d, desc(xx)))
  p13_same_error(df, function(d) arrange(d, x, xx))
  p13_same_error(df, function(d) arrange(d, x, desc(xx)))
  p13_same_error(df, function(d) arrange(group_by(d, g), xx))
  p13_same_error(df, function(d) arrange(group_by(d, g), xx, .by_group = TRUE))
  p13_same_error(df, function(d) arrange(d, .data$xx))
})

test_that("arrange() sorts by (or ignores) a name bound outside the table, as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  nul <- NULL
  v <- c(3, 1, 2)
  k <- 5
  expect_same_as_dplyr(df, function(d) arrange(d, nul, desc(x)))
  expect_same_as_dplyr(df, function(d) arrange(d, nul, desc(x), nul))
  expect_same_as_dplyr(df, function(d) arrange(d, v))
  expect_same_as_dplyr(df, function(d) arrange(d, desc(v)))
  expect_same_as_dplyr(df, function(d) arrange(d, k, x))
  expect_same_as_dplyr_lazy(df, function(d) arrange(d, nul, desc(x)))
  expect_same_as_dplyr_lazy(df, function(d) arrange(d, v))
  # the exact shape of dplyr's own test (a table with no column named `y`)
  one <- tibble::tibble(x = 1:2)
  y <- NULL
  expect_same_as_dplyr(one, function(d) arrange(d, y, desc(x)))
  expect_same_as_dplyr(one, function(d) arrange(d, y, desc(x), y))
})

test_that("summarise() and the verbs built on it report a missing column with the user's call", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) summarise(d, m = mean(xx)))
  expect_identical(p13_call(cpu), "`summarise()`")
  p13_same_error(df, function(d) summarise(group_by(d, g), m = mean(xx)))
  p13_same_error(df, function(d) summarise(d, m = mean(xx), .by = g))
  p13_same_error(df, function(d) summarise(d, m = xx))
  p13_same_error(df, function(d) summarise(d, m = n_distinct(xx)))
  p13_same_error(df, function(d) summarise(d, m = sum(.data$xx)))
  p13_same_error(df, function(d) summarise(d, a = mean(x), b = a + xx))
  p13_same_error(df, function(d) summarise(d, across(xx, mean)))
  p13_same_error(df, function(d) summarise(group_by(d, g), across(c(x, xx), mean)))
  p13_same_error(df, function(d) reframe(d, m = range(xx)))
  p13_same_error(df, function(d) count(d, g, wt = xx))
  p13_same_error(df, function(d) tally(d, wt = xx))
})

test_that("summarise() on a grouped table with no rows reports a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()[0, ]
  p13_same_error(df, function(d) summarise(d, m = mean(xx)))
  p13_same_error(df, function(d) summarise(group_by(d, g), m = mean(xx)))
})

test_that("slice_min()/slice_max() report a missing column with the user's call", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) slice_max(d, xx))
  expect_identical(p13_call(cpu), "`slice_max()`")
  p13_same_error(df, function(d) slice_min(d, xx, n = 1))
  p13_same_error(df, function(d) slice_max(group_by(d, g), xx))
})

test_that("mutate() reports a missing column as dplyr does, also from add_count()/add_tally()", {
  skip_if_no_gpu()
  df <- p13_df()
  p13_same_error(df, function(d) mutate(d, z = xx + 1))
  p13_same_error(df, function(d) mutate(group_by(d, g), z = xx + 1))
  p13_same_error(df, function(d) mutate(d, xx))
  p13_same_error(df, function(d) transmute(d, z = xx))
  p13_same_error(df, function(d) mutate(d, z = .data$xx))
  p13_same_error(df, function(d) mutate(group_by(d, g), z = .data$xx))
  p13_same_error(df, function(d) mutate(d, across(c(x, xx), ~ .x + 1)))
  p13_same_error(df, function(d) mutate(rowwise(d), z = xx))
  cpu <- p13_same_error(df, function(d) add_count(d, g, wt = xx))
  expect_identical(p13_call(cpu), "`add_count()`")
  cpu <- p13_same_error(df, function(d) add_tally(d, wt = xx))
  expect_identical(p13_call(cpu), "`add_tally()`")
})

test_that("group_by(), count() and distinct() report missing columns as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  p13_same_error(df, function(d) group_by(d, xx))
  p13_same_error(df, function(d) group_by(d, xx, yy))
  p13_same_error(df, function(d) group_by(d, x, xx))
  p13_same_error(df, function(d) group_by(d, xx, .add = TRUE))
  p13_same_error(df, function(d) group_by(group_by(d, g), xx, .add = TRUE))
  p13_same_error(df, function(d) count(d, xx))
  p13_same_error(df, function(d) add_count(d, xx))
  p13_same_error(df, function(d) distinct(d, xx))
  p13_same_error(df, function(d) distinct(d, .data$xx))
})

test_that("tidyselect verbs report a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  p13_same_error(df, function(d) select(d, xx))
  p13_same_error(df, function(d) select(d, "xx"))
  p13_same_error(df, function(d) select(d, -xx))
  p13_same_error(df, function(d) select(d, x, xx, yy))
  p13_same_error(df, function(d) rename(d, a = xx))
  p13_same_error(df, function(d) rename_with(d, toupper, xx))
  p13_same_error(df, function(d) relocate(d, xx))
  p13_same_error(df, function(d) relocate(d, x, xx))
  p13_same_error(df, function(d) relocate(d, x, .after = xx))
  p13_same_error(df, function(d) pull(d, xx))
  p13_same_error(df, function(d) pull(d, "xx"))
  p13_same_error(df, function(d) d[, "xx"])
  p13_same_error(df, function(d) summarise(d, n = n(), .by = xx))
  p13_same_error(df, function(d) mutate(d, n = 1, .by = xx))
})

test_that("joins report a missing join column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) left_join(d, p13_like(d, p13_df2()), by = "xx"))
  expect_identical(p13_call(cpu), "`left_join()`")
  p13_same_error(df, function(d) left_join(d, p13_like(d, p13_df2()), by = c(x = "zz")))
  p13_same_error(df, function(d) left_join(d, p13_like(d, p13_df2()), by = c("xx", "yy")))
  p13_same_error(df, function(d) inner_join(d, p13_like(d, p13_df2()), by = join_by(xx == x)))
  p13_same_error(df, function(d) right_join(d, p13_like(d, p13_df2()), by = "xx"))
  p13_same_error(df, function(d) full_join(d, p13_like(d, p13_df2()), by = join_by(x == zz)))
  p13_same_error(df, function(d) semi_join(d, p13_like(d, p13_df2()), by = "xx"))
  p13_same_error(df, function(d) anti_join(d, p13_like(d, p13_df2()), by = c(x = "zz")))
  p13_same_error(df, function(d) inner_join(d, p13_like(d, p13_df2()), by = join_by(x >= zz)))
  p13_same_error(df, function(d) inner_join(d, p13_like(d, p13_df2()), by = join_by(xx >= z)))
  p13_same_error(df, function(d) nest_join(d, p13_like(d, p13_df2()), by = "xx"))
})

test_that("a missing column is dplyr's error even when CPU fallbacks are errors", {
  skip_if_no_gpu()
  df <- p13_df()
  strict <- list(cuplyr.fallback = "error")
  p13_same_error(df, function(d) filter(d, xx > 1), gpu_opts = strict)
  p13_same_error(df, function(d) filter(group_by(d, g), xx > 1), gpu_opts = strict)
  p13_same_error(df, function(d) arrange(d, xx), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(d, m = mean(xx)), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(group_by(d, g), m = mean(xx)), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(d, m = sum(.data$xx)), gpu_opts = strict)
  p13_same_error(df, function(d) count(d, g, wt = xx), gpu_opts = strict)
  p13_same_error(df, function(d) mutate(d, z = xx + 1), gpu_opts = strict)
  p13_same_error(df, function(d) group_by(d, xx), gpu_opts = strict)
  p13_same_error(df, function(d) left_join(d, p13_like(d, p13_df2()), by = "xx"), gpu_opts = strict)
})

test_that("a missing column signals no CPU fallback and no warning dplyr doesn't give", {
  skip_if_no_gpu()
  df <- p13_df()
  cases <- list(
    filter = function(d) filter(d, xx > 1),
    arrange = function(d) arrange(d, xx),
    summarise = function(d) summarise(d, m = mean(xx)),
    summarise_unnamed = function(d) summarise(d, mean(xx)),
    summarise_grouped = function(d) summarise(group_by(d, g), m = mean(xx)),
    mutate_bare = function(d) mutate(d, xx),
    mutate_unnamed = function(d) mutate(d, xx + 1),
    transmute_unnamed = function(d) transmute(d, xx + 1)
  )
  for (nm in names(cases)) {
    cpu_warnings <- character()
    withCallingHandlers(
      tryCatch(cases[[nm]](df), error = function(e) NULL),
      warning = function(w) {
        cpu_warnings[[length(cpu_warnings) + 1L]] <<- conditionMessage(w)
        invokeRestart("muffleWarning")
      }
    )
    for (lazy in c(FALSE, TRUE)) {
      d <- tbl_gpu(df, lazy = lazy)
      fallbacks <- 0L
      gpu_warnings <- character()
      err <- withCallingHandlers(
        tryCatch({
          out <- cases[[nm]](d)
          if (inherits(out, "tbl_gpu")) dplyr::collect(out)
          NULL
        }, error = function(e) e),
        cuplyr_cpu_fallback = function(c) fallbacks <<- fallbacks + 1L,
        warning = function(w) {
          gpu_warnings[[length(gpu_warnings) + 1L]] <<- conditionMessage(w)
          invokeRestart("muffleWarning")
        }
      )
      info <- sprintf("%s (%s)", nm, if (lazy) "lazy" else "eager")
      expect_true(inherits(err, "error"), info = info)
      expect_identical(fallbacks, 0L, info = info)
      expect_identical(gpu_warnings, cpu_warnings, info = info)
    }
  }
})

# Added after the first review: names dplyr never evaluates are not missing
# columns, the names earlier dots create (named or not) are visible to later
# ones, and group_by() with a computed dot reports a missing bare name the way
# dplyr's mutate step does.

test_that("summarise() keeps working on dots whose unknown names dplyr never evaluates", {
  skip_if_no_gpu()
  df <- p13_df()
  has_wt <- FALSE
  kind <- "a"
  k <- 10
  v <- 3
  pipelines <- list(
    function(d) summarise(d, m = if (has_wt) weighted.mean(x, wt) else mean(x)),
    function(d) summarise(d, m = tryCatch(mean(xx), error = function(e) NA_real_)),
    function(d) summarise(d, v = switch(kind, a = mean(x), b = mean(zz))),
    function(d) summarise(d, mean(x), b = `mean(x)` * 2),
    function(d) summarise(group_by(d, g), m = mean(x) + k, n = .env$v, s = sum(y, na.rm = TRUE)),
    function(d) summarise(d, across(c(x, y), ~ mean(.x, na.rm = TRUE) + k)),
    function(d) summarise(d, m = base::mean(x), q = stats::median(x)),
    function(d) summarise(d, x = mean(x), m = x * 2)
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("summarise() reports a missing column after dots that create or reuse names", {
  skip_if_no_gpu()
  df <- p13_df()
  p13_same_error(df, function(d) summarise(d, x = mean(x), m = sum(.data$xx)))
  p13_same_error(df, function(d) summarise(d, x = mean(x), m = mean(xx)))
  p13_same_error(df, function(d) summarise(d, mean(x), b = `mean(x)` + xx))
  p13_same_error(df, function(d) summarise(d, across(c(x, y), ~ mean(.x) + xx)))
})

test_that("group_by() with a computed dot reports a missing column as dplyr's mutate step does", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) group_by(d, a = x + 1, xx))
  expect_true(inherits(cpu, "dplyr:::mutate_error"))
  p13_same_error(df, function(d) group_by(d, xx, a = x + 1))
  p13_same_error(df, function(d) group_by(d, x, a = xx + 1))
})

test_that("more filter()/arrange() shapes report a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  p13_same_error(df, function(d) filter(d, if_any(c(x, y), ~ .x > xx)))
  p13_same_error(df, function(d) filter_out(d, xx > 1))
  p13_same_error(df, function(d) arrange(d, -xx))
  strict <- list(cuplyr.fallback = "error")
  p13_same_error(df, function(d) filter(d, if_any(c(x, xx), ~ .x > 1)), gpu_opts = strict)
  p13_same_error(df, function(d) filter(d, .data$xx > 1), gpu_opts = strict)
  p13_same_error(df, function(d) arrange(d, desc(xx)), gpu_opts = strict)
  p13_same_error(df, function(d) arrange(d, .data$xx), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(d, across(xx, mean)), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(d, x = mean(x), m = sum(.data$xx)), gpu_opts = strict)
  p13_same_error(df, function(d) group_by(d, a = x + 1, xx), gpu_opts = strict)
})

# Added after the second review: group_by() with an unnamed computed or pick()
# dot, computed keys of count()/add_count()/distinct(), summarise() dots whose
# arguments R doesn't force (S3 `...`, `..N`) or that return a data frame, and
# reducers wrapped in another call.

test_that("group_by() with an unnamed computed or pick() dot reports a missing column as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  p13_same_error(df, function(d) group_by(d, x + 1, xx))
  p13_same_error(df, function(d) group_by(d, pick(x), xx))
})

test_that("count(), add_count() and distinct() report a missing column in a computed key as dplyr does", {
  skip_if_no_gpu()
  df <- p13_df()
  cpu <- p13_same_error(df, function(d) count(d, a = x + xx))
  expect_true(inherits(cpu, "dplyr:::mutate_error"))
  p13_same_error(df, function(d) count(d, x + xx))
  p13_same_error(df, function(d) count(d, a = x + 1, xx))
  p13_same_error(df, function(d) count(group_by(d, g), a = x + xx))
  p13_same_error(df, function(d) add_count(d, a = x + xx))
  p13_same_error(df, function(d) distinct(d, a = x + xx))
  p13_same_error(df, function(d) distinct(d, a = x + 1, xx))
})

test_that("summarise() keeps working on arguments R doesn't force and on data-frame dots", {
  skip_if_no_gpu()
  df <- p13_df()
  f <- function(d, ...) summarise(d, m = sum(x) + ..1)
  pipelines <- list(
    function(d) f(d, 5),
    function(d) summarise(d, m = mean(x, foo = zz)),
    function(d) summarise(d, dplyr::first(tibble::tibble(a = x)), b = mean(a)),
    function(d) suppressWarnings(summarise(d, m = mean(g, na.rm = zz)))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("summarise() reports a missing column inside other calls and on rowwise or empty input", {
  skip_if_no_gpu()
  df <- p13_df()
  strict <- list(cuplyr.fallback = "error")
  p13_same_error(df, function(d) summarise(d, m = round(mean(xx), 1)))
  p13_same_error(df, function(d) summarise(d, m = quantile(xx, 0.5)))
  p13_same_error(df, function(d) summarise(d, m = n() + xx))
  p13_same_error(df, function(d) summarise(d, m = round(mean(xx), 1)), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(d, m = quantile(xx, 0.5)), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(d, m = n() + xx), gpu_opts = strict)
  p13_same_error(df, function(d) summarise(rowwise(d), m = mean(xx)))
  p13_same_error(df[0, ], function(d) summarise(d, m = mean(xx), .by = g))
})
