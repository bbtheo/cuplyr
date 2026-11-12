# F68b acceptance tests: the cheaper verb front ends behave exactly as before.
#
# F68b removes R work from the verbs that eager and lazy share (selection
# resolution, IR walks, dots screening, summarise screening, schema/metadata
# bookkeeping). These tests pin the shapes those shortcuts touch against
# dplyr on plain tibbles, eagerly and lazily: plain and renamed selections,
# env-shadowed symbols, missing columns, namespaced aggregates, factor/POSIXct/
# Date metadata, frame attributes, groups, and errors raised at the verb.
# They also pin that nothing is cached across calls: a changed environment
# value, a different quosure environment, the session collation and options
# are read on every call.

f68b_tbl <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 3L, 2L, 3L, 1L, 2L),
    h = c(1L, 2L, 1L, 2L, 2L, 1L, 1L, 2L),
    x = c(0.1, 0.9, 0.5, NA, 0.3, 0.7, 0.2, 0.6),
    y = c(1, 2, 3, 4, 5, 6, 7, 8),
    i = c(5L, 3L, NA, 1L, 7L, 2L, 8L, 4L),
    s = c("a", "B", "c", NA, "b", "A", "a", "C"),
    f = factor(c("b", "a", "b", "c", "a", "c", "b", "a"), levels = c("c", "b", "a")),
    t = as.POSIXct(c(0, 3600, 7200, 10800, 14400, 18000, 21600, 25200),
                   origin = "1970-01-01", tz = "America/New_York"),
    d = as.Date("2020-01-01") + c(0L, 3L, 1L, 2L, 5L, 4L, 7L, 6L)
  )
}

f68b_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# The condition `expr` signals (or NULL), for class/message comparisons.
f68b_cnd <- function(expr, type = "error") {
  out <- NULL
  if (type == "error") {
    tryCatch(force(expr), error = function(e) out <<- e)
  } else if (type == "warning") {
    withCallingHandlers(force(expr), warning = function(w) {
      if (is.null(out)) out <<- w
      invokeRestart("muffleWarning")
    })
  } else {
    withCallingHandlers(force(expr), message = function(m) {
      if (is.null(out)) out <<- m
      invokeRestart("muffleMessage")
    })
  }
  out
}

# cuplyr signals the condition dplyr signals (main class and message), eagerly
# and on a lazy table at the verb call (no collect()).
f68b_expect_same_cnd <- function(df, verb, type = "error", label = "") {
  ref <- f68b_cnd(verb(df), type)
  expect_false(is.null(ref), info = paste(label, "(dplyr signalled nothing)"))
  for (mode in c("eager", "lazy")) {
    gt <- tbl_gpu(df, lazy = identical(mode, "lazy"))
    got <- f68b_cnd(verb(gt), type)
    expect_false(is.null(got), info = paste(label, mode, "(cuplyr signalled nothing)"))
    if (!is.null(got)) {
      expect_true(inherits(got, class(ref)[[1]]),
                  info = paste(label, mode, "class:", paste(class(got), collapse = "/")))
      expect_identical(conditionMessage(got), conditionMessage(ref), info = paste(label, mode))
    }
  }
}

test_that("plain and renamed selections match dplyr", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  x <- "y"  # a variable named like a column: the column wins
  vars <- c("g", "x")
  f68b_both(df, function(d) select(d, g, x))
  f68b_both(df, function(d) select(d, x, g))
  f68b_both(df, function(d) select(d, x))
  f68b_both(df, function(d) select(d, x, x))
  f68b_both(df, function(d) select(d, "x", "g"))
  f68b_both(df, function(d) select(d, a = x))
  f68b_both(df, function(d) select(d, x, a = x))
  f68b_both(df, function(d) select(d, a = x, x))
  f68b_both(df, function(d) select(d, f, t, d))
  f68b_both(df, function(d) select(d, -x))
  f68b_both(df, function(d) select(d, 1:2))
  f68b_both(df, function(d) select(d, all_of(vars)))
  f68b_both(df, function(d) select(d, starts_with("g"), everything()))
  f68b_both(df, function(d) select(d, where(is.numeric)))
  f68b_both(df, function(d) select(group_by(d, g), g, x))
  f68b_both(df, function(d) select(group_by(d, g, h), y))
  f68b_both(df, function(d) relocate(d, y))
  f68b_both(df, function(d) relocate(d, y, x, .before = g))
  f68b_both(df, function(d) rename(d, a = x))
  f68b_both(df, function(d) rename(d, a = x, b = g))
  f68b_both(df, function(d) select(filter(d, y > 2), x, y))
})

test_that("selection errors and messages are dplyr's", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  f68b_expect_same_cnd(df, function(d) select(d, nope), label = "select missing symbol")
  f68b_expect_same_cnd(df, function(d) select(d, "nope"), label = "select missing string")
  f68b_expect_same_cnd(df, function(d) select(d, x, nope), label = "select one missing")
  f68b_expect_same_cnd(df, function(d) rename(d, a = nope), label = "rename missing")
  f68b_expect_same_cnd(df, function(d) select(group_by(d, g), x), type = "message",
                       label = "grouped select adds the group")
})

test_that("filter() shapes match dplyr", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  thr <- 0.4
  f68b_both(df, function(d) filter(d, x > 0.5))
  f68b_both(df, function(d) filter(d, x > 0.5, y < 7))
  f68b_both(df, function(d) filter(d, x > thr & !is.na(x)))
  f68b_both(df, function(d) filter(d, .data$x > .env$thr))
  f68b_both(df, function(d) filter(d, is.na(x) | y == 4))
  f68b_both(df, function(d) filter(d, dplyr::between(y, 2, 6)))
  f68b_both(df, function(d) filter(d, g %in% c(1L, 3L)))
  f68b_both(df, function(d) filter(d, i * 2L > 6L))
  f68b_both(df, function(d) filter(d, f == "b"))
  f68b_both(df, function(d) filter(d, d > as.Date("2020-01-03")))
  f68b_both(df, function(d) filter(d, TRUE))
  f68b_both(df, function(d) filter(d, FALSE))
  f68b_both(df, function(d) filter(group_by(d, g), y > mean(y)))
  f68b_both(df, function(d) filter(d, y > mean(y), .by = g))
  f68b_both(df, function(d) filter(d, if_any(c(x, y), ~ .x > 0.8)))
  f68b_both(df, function(d) filter(d, if_all(c(x, y), ~ !is.na(.x))))
  f68b_both(df, function(d) filter(filter(d, y > 1), x < 0.8))
  f68b_both(df[0, ], function(d) filter(d, x > 0.5))
})

test_that("mutate() shapes match dplyr", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  k <- 3
  f68b_both(df, function(d) mutate(d, z = x * 2 + y))
  f68b_both(df, function(d) mutate(d, z = y * k, w = z + 1))
  f68b_both(df, function(d) mutate(d, z = 1L, n = NA))
  f68b_both(df, function(d) mutate(d, x = NULL))
  f68b_both(df, function(d) mutate(d, z = x + y, .keep = "used"))
  f68b_both(df, function(d) mutate(d, z = x + y, .keep = "none"))
  f68b_both(df, function(d) mutate(d, z = y, .before = 1))
  f68b_both(df, function(d) mutate(d, z = dplyr::if_else(y > 4, "hi", "lo")))
  f68b_both(df, function(d) mutate(d, z = i * 2L, q = i %/% 2L))
  f68b_both(df, function(d) mutate(d, across(c(x, y), ~ .x * 2)))
  f68b_both(df, function(d) mutate(d, t2 = t, f2 = f, d2 = d + 1L))
  f68b_both(df, function(d) mutate(d, l = as.character(f)))
  f68b_both(df, function(d) mutate(d, cs = cumsum(y), r = roll_mean(y, 3)))
  f68b_both(df, function(d) mutate(group_by(d, g), m = mean(y), n = n()))
  f68b_both(df, function(d) mutate(d, m = mean(y), .by = g))
  f68b_both(df[0, ], function(d) mutate(d, z = x * 2))
})

test_that("summarise() shapes match dplyr, namespaced aggregates included", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  f68b_both(df, function(d) summarise(group_by(d, g), m = mean(y)))
  f68b_both(df, function(d) summarise(group_by(d, g), n = n(), m = mean(x, na.rm = TRUE)))
  f68b_both(df, function(d) summarise(group_by(d, g), k = dplyr::n(), m = base::mean(y),
                                      s = stats::sd(y)))
  f68b_both(df, function(d) summarise(group_by(d, g), s = sum(i, na.rm = TRUE), mx = max(t)))
  f68b_both(df, function(d) summarise(group_by(d, f), m = mean(y * 2), fi = dplyr::first(y),
                                      nd = dplyr::n_distinct(h)))
  f68b_both(df, function(d) summarise(group_by(d, g, h), m = mean(y), .groups = "drop"))
  f68b_both(df, function(d) summarise(d, m = mean(y), n = n()))
  f68b_both(df, function(d) summarise(d, m = mean(y), .by = g))
  f68b_both(df, function(d) summarise(group_by(d, g), across(c(x, y), mean)))
  f68b_both(df[0, ], function(d) summarise(d, m = mean(y), n = n()))
})

test_that("arrange() shapes match dplyr", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  f68b_both(df, function(d) arrange(d, x))
  f68b_both(df, function(d) arrange(d, desc(x), y))
  f68b_both(df, function(d) arrange(d, g, desc(y)))
  f68b_both(df, function(d) arrange(d, s))
  f68b_both(df, function(d) arrange(d, desc(t)))
  f68b_both(df, function(d) arrange(d, f, d))
  f68b_both(df, function(d) arrange(group_by(d, g), desc(y), .by_group = TRUE))
})

test_that("factor levels, time zones, groups and frame attributes survive every verb", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  attr(df, "meta") <- "m"
  verbs <- list(
    filter = function(d) filter(d, y > 2),
    mutate = function(d) mutate(d, z = y + 1, t2 = t),
    select = function(d) select(d, f, t, d, y),
    arrange = function(d) arrange(d, desc(y)),
    rename = function(d) rename(d, ff = f, tt = t)
  )
  for (nm in names(verbs)) {
    ref <- verbs[[nm]](df)
    for (lazy in c(FALSE, TRUE)) {
      got <- collect(verbs[[nm]](tbl_gpu(df, lazy = lazy)))
      info <- paste(nm, if (lazy) "lazy" else "eager")
      expect_equal(as.data.frame(got), as.data.frame(ref), info = info)
      expect_identical(attr(got, "meta"), "m", info = info)
      for (col in names(ref)) {
        expect_identical(attr(got[[col]], "tzone"), attr(ref[[col]], "tzone"), info = paste(info, col))
        expect_identical(levels(got[[col]]), levels(ref[[col]]), info = paste(info, col))
      }
    }
  }
  # groups carried by the column-only verbs
  gdf <- group_by(f68b_tbl(), g, f)
  for (lazy in c(FALSE, TRUE)) {
    gg <- tbl_gpu(gdf, lazy = lazy)
    expect_identical(group_vars(filter(gg, y > 2)), c("g", "f"))
    expect_identical(group_vars(mutate(gg, z = y)), c("g", "f"))
    expect_identical(group_vars(arrange(gg, y)), c("g", "f"))
    expect_identical(group_vars(rename(gg, gg2 = g)), c("gg2", "f"))
    expect_identical(group_vars(summarise(gg, m = mean(y), .groups = "keep")), c("g", "f"))
  }
  # a rowwise table stays rowwise
  rw <- dplyr::rowwise(f68b_tbl())
  expect_equal(collect(mutate(dplyr::rowwise(tbl_gpu(f68b_tbl())), m = max(x, y))),
               mutate(rw, m = max(x, y)))
})

test_that("verb errors are dplyr's where they match, and come from the verb call", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  f68b_expect_same_cnd(df, function(d) filter(d, x = 1), label = "filter named input")
  f68b_expect_same_cnd(df, function(d) filter(d, x), label = "filter non-logical")
  f68b_expect_same_cnd(df, function(d) summarise(d, m = mean(nope)), label = "summarise missing")
  f68b_expect_same_cnd(df, function(d) mutate(group_by(d, g), z = x, .by = h),
                       label = "mutate .by on grouped")
  # Errors whose text is cuplyr's: a lazy table with pending ops raises the
  # eager error at the verb call.
  verbs <- list(
    filter_missing = function(d) filter(d, nope > 1),
    mutate_missing = function(d) mutate(d, z = nope + 1),
    arrange_missing = function(d) arrange(d, nope),
    summarise_missing = function(d) summarise(group_by(d, g), m = mean(nope)),
    select_missing = function(d) select(d, x, nope),
    relocate_missing = function(d) relocate(d, nope),
    summarise_bad_arg = function(d) summarise(d, m = mean(y, trim = "a")),
    mutate_bad_type = function(d) mutate(d, z = s + 1)
  )
  for (nm in names(verbs)) {
    eg <- tbl_gpu(df, lazy = FALSE)
    lz <- mutate(tbl_gpu(df, lazy = TRUE), y = y + 0)
    e_eager <- f68b_cnd(verbs[[nm]](eg))
    e_lazy <- f68b_cnd(verbs[[nm]](lz))  # no collect(): must fail at the verb
    expect_false(is.null(e_eager), info = nm)
    expect_false(is.null(e_lazy), info = nm)
    if (!is.null(e_eager) && !is.null(e_lazy)) {
      expect_identical(class(e_lazy), class(e_eager), info = nm)
      expect_identical(conditionMessage(e_lazy), conditionMessage(e_eager), info = nm)
    }
  }
})

test_that("integer overflow still warns at the eager verb call", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  eg <- tbl_gpu(df, lazy = FALSE)
  w <- f68b_cnd(mutate(eg, z = i * 1000000000L), type = "warning")
  expect_false(is.null(w))
  expect_match(conditionMessage(w), "integer overflow")
  expect_equal(collect(summarise(group_by(eg, g), s = sum(i * 400000000L, na.rm = TRUE))),
               suppressWarnings(summarise(group_by(df, g), s = sum(i * 400000000L, na.rm = TRUE))))
})

test_that("nothing is cached across calls: environments, collation and options are read each time", {
  skip_if_no_gpu()
  df <- f68b_tbl()
  # the same closure, a different value of its free variable on each call
  run <- function(d) list(
    filter(d, x > thr),
    mutate(d, z = y * thr),
    summarise(group_by(d, g), m = mean(y * thr)),
    arrange(d, desc(y * thr))
  )
  for (thr in c(0.2, 0.5, -1, 0.8)) {
    ref <- run(df)
    for (lazy in c(FALSE, TRUE)) {
      got <- lapply(run(tbl_gpu(df, lazy = lazy)), collect)
      for (k in seq_along(ref)) {
        expect_equal(as.data.frame(got[[k]]), as.data.frame(ref[[k]]),
                     info = paste("thr", thr, "lazy", lazy, "step", k))
      }
    }
  }
  # the same expression text in different environments
  make <- function(k) function(d) filter(d, y > k)
  f1 <- make(2)
  f2 <- make(6)
  eg <- tbl_gpu(df)
  expect_equal(nrow(collect(f1(eg))), nrow(f1(df)))
  expect_equal(nrow(collect(f2(eg))), nrow(f2(df)))
  expect_equal(nrow(collect(f1(eg))), nrow(f1(df)))
  # a column takes precedence over a variable of the same name, every time
  y <- 100
  expect_equal(collect(filter(eg, y > 4)), filter(df, y > 4))
  rm(y)
  # a selection symbol that is a column one call and not the next
  sel <- function(d) select(d, g)
  expect_equal(collect(sel(eg)), sel(df))
  ref <- f68b_cnd(sel(rename(df, g2 = g)))
  got <- f68b_cnd(sel(rename(eg, g2 = g)))
  expect_s3_class(got, class(ref)[[1]])
  expect_identical(conditionMessage(got), conditionMessage(ref))
  expect_equal(collect(sel(eg)), sel(df))

  # the session collation decides native vs CPU string ordering on every call
  have_en <- !identical(suppressWarnings(Sys.setlocale("LC_COLLATE", "en_US.UTF-8")), "")
  old <- Sys.getlocale("LC_COLLATE")
  withr::defer(Sys.setlocale("LC_COLLATE", old))
  locales <- if (have_en) c("C", "en_US.UTF-8", "C", "en_US.UTF-8") else c("C", "C")
  for (loc in locales) {
    Sys.setlocale("LC_COLLATE", loc)
    for (lazy in c(FALSE, TRUE)) {
      gt <- tbl_gpu(df, lazy = lazy)
      expect_equal(collect(summarise(gt, lo = min(s, na.rm = TRUE), hi = max(s, na.rm = TRUE))),
                   summarise(df, lo = min(s, na.rm = TRUE), hi = max(s, na.rm = TRUE)),
                   info = paste(loc, lazy))
      expect_equal(collect(filter(gt, s < "b")), filter(df, s < "b"), info = paste(loc, lazy))
    }
  }

  # options are read per call
  myf <- function(v) v > 0.5
  withr::local_options(cuplyr.fallback = "error")
  expect_error(filter(eg, myf(x)))
  withr::local_options(cuplyr.fallback = NULL)
  expect_equal(collect(filter(eg, myf(x))), filter(df, myf(x)))
})
