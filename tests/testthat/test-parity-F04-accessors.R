# F04 acceptance tests: tibble's `$`, `[[`, `$<-`, `[[<-` and whole-column
# `[<-` (cell assignment errors loudly) on tbl_gpu.
#
# Each case runs the same function on a plain tibble (the oracle) and on a
# tbl_gpu (eager, lazy, lazy with a pending filter, grouped, zero rows). The
# functions are re-homed to the global environment before they run, so they
# call `$`/`[[` exactly as user code does (not as code inside the cuplyr
# namespace, which may read the object's internal fields).

f04_df <- function() {
  data.frame(
    g = c("a", "b", "a", "c"),
    x = c(1.5, 2, NA, 4.5),
    i = c(1L, NA, 3L, 4L),
    s = c("p", "q", NA, "s"),
    f = factor(c("u", "v", "u", NA), levels = c("u", "v", "w")),
    d = as.Date("2024-01-01") + c(0L, 1L, NA, 3L),
    l = c(TRUE, NA, FALSE, TRUE),
    stringsAsFactors = FALSE
  )
}

f04_global <- function(fn) {
  environment(fn) <- globalenv()
  fn
}

# Input shapes: name -> list(prep applied to the tibble / tbl_gpu, lazy?)
f04_shapes <- function() {
  list(
    eager = list(prep = identity, lazy = FALSE),
    lazy = list(prep = identity, lazy = TRUE),
    lazy_pending = list(prep = function(t) dplyr::filter(t, i > 1L | is.na(i)), lazy = TRUE),
    eager_filtered = list(prep = function(t) dplyr::filter(t, i > 1L | is.na(i)), lazy = FALSE),
    grouped = list(prep = function(t) dplyr::group_by(t, g), lazy = FALSE),
    grouped_lazy = list(prep = function(t) dplyr::group_by(t, g), lazy = TRUE),
    zero_rows = list(prep = function(t) dplyr::filter(t, i > 100L), lazy = FALSE),
    zero_rows_lazy = list(prep = function(t) dplyr::filter(t, i > 100L), lazy = TRUE)
  )
}

f04_up <- function(df, lazy = FALSE) {
  tbl_gpu(df, lazy = lazy)
}

# Run `expr`, recording its value, warnings (class + message) and error.
f04_run <- function(expr_fn) {
  warnings <- list()
  value <- tryCatch(
    withCallingHandlers(
      expr_fn(),
      warning = function(w) {
        warnings[[length(warnings) + 1L]] <<- list(class = class(w)[[1]], message = conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  list(value = value, warnings = warnings)
}

# Compare fn(oracle input) with fn(tbl_gpu input) for every shape in `shapes`.
# A data-frame oracle result must be a tbl_gpu with the same group_vars(),
# .drop and collected content (and the same exec_mode as its input); anything
# else must be identical. Warnings must match. Errors must match by class
# (and by message when `same_message`).
f04_same <- function(fn, shapes = names(f04_shapes()), df = f04_df(), same_message = TRUE) {
  fn <- f04_global(fn)
  all_shapes <- f04_shapes()
  for (nm in shapes) {
    sh <- all_shapes[[nm]]
    label <- paste0(nm, ": ", paste(deparse(body(fn)), collapse = " "))
    oracle <- f04_run(function() fn(sh$prep(tibble::as_tibble(df))))
    gpu_in <- sh$prep(f04_up(df, sh$lazy))
    gpu <- f04_run(function() fn(gpu_in))

    expect_identical(gpu$warnings, oracle$warnings, info = paste(label, "- warnings"))

    if (inherits(oracle$value, "error")) {
      expect_s3_class(gpu$value, class(oracle$value)[[1]])
      if (same_message && inherits(gpu$value, "error")) {
        expect_identical(conditionMessage(gpu$value), conditionMessage(oracle$value), info = label)
      }
      next
    }
    if (inherits(gpu$value, "error")) {
      fail(paste(label, "- tbl_gpu errored:", conditionMessage(gpu$value)))
      next
    }
    if (is.data.frame(oracle$value)) {
      out <- gpu$value
      expect_true(is_tbl_gpu(out), info = paste(label, "- result should be a tbl_gpu"))
      if (!is_tbl_gpu(out)) next
      expect_identical(dplyr::group_vars(out), dplyr::group_vars(oracle$value), info = label)
      expect_identical(
        dplyr::group_by_drop_default(out),
        dplyr::group_by_drop_default(oracle$value),
        info = label
      )
      expect_identical(unclass(out)$exec_mode, unclass(gpu_in)$exec_mode, info = paste(label, "- exec_mode"))
      expect_identical(
        tibble::as_tibble(dplyr::ungroup(collect(out))),
        tibble::as_tibble(dplyr::ungroup(oracle$value)),
        info = label
      )
    } else {
      expect_identical(gpu$value, oracle$value, info = label)
    }
  }
}

# ---------------------------------------------------------------------------
# `$` and `[[` read

test_that("x$col returns the column as an R vector for every type and shape", {
  skip_if_no_gpu()
  for (col in names(f04_df())) {
    fn <- function(t) NULL
    body(fn) <- call("$", quote(t), as.name(col))
    f04_same(fn)
  }
})

test_that("x$col on an unknown or partial name warns like tibble and returns NULL", {
  skip_if_no_gpu()
  f04_same(function(t) t$zz)
  f04_same(function(t) t$s2, shapes = c("eager", "lazy_pending", "grouped"))
  f04_same(function(t) t$gg, shapes = c("eager", "lazy"))
})

test_that("x$name reads user columns that share a name with internal fields", {
  skip_if_no_gpu()
  df <- data.frame(
    ptr = 1:3, schema = c("a", "b", "c"), groups = c(1.5, 2.5, NA),
    lazy_ops = c(TRUE, FALSE, NA), exec_mode = 3:1, group_drop = c("x", "y", "z"), i = c(1L, NA, 3L),
    stringsAsFactors = FALSE
  )
  for (col in names(df)) {
    fn <- function(t) NULL
    body(fn) <- call("$", quote(t), as.name(col))
    f04_same(fn, shapes = c("eager", "lazy", "lazy_pending"), df = df)
  }
  # and with no such column, user code gets tibble's warning and NULL
  f04_same(function(t) t$ptr, shapes = c("eager", "lazy"))
  f04_same(function(t) t$schema, shapes = c("eager", "grouped"))
  f04_same(function(t) t[["lazy_ops"]], shapes = c("eager", "lazy"))
})

test_that("x[[name]] and x[[position]] return the column; unknown name gives NULL", {
  skip_if_no_gpu()
  f04_same(function(t) t[["s"]])
  f04_same(function(t) t[["f"]])
  f04_same(function(t) t[[6]])
  f04_same(function(t) t[[2L]])
  f04_same(function(t) t[["zz"]])
  f04_same(function(t) t[["g", exact = FALSE]], shapes = c("eager", "lazy"))
})

test_that("x[[i, j]] returns one cell", {
  skip_if_no_gpu()
  f04_same(function(t) t[[2, "x"]], shapes = c("eager", "lazy", "lazy_pending", "grouped"))
  f04_same(function(t) t[[1, 5]], shapes = c("eager", "lazy", "lazy_pending"))
  f04_same(function(t) t[[3, "d"]], shapes = c("eager", "lazy"))
})

test_that("x[[j]] raises tibble's errors for invalid subscripts", {
  skip_if_no_gpu()
  shapes <- c("eager", "lazy", "grouped")
  f04_same(function(t) t[[99]], shapes = shapes)
  f04_same(function(t) t[[-1]], shapes = shapes)
  f04_same(function(t) t[[0]], shapes = shapes)
  f04_same(function(t) t[[NA]], shapes = shapes)
  f04_same(function(t) t[[NA_character_]], shapes = shapes)
  f04_same(function(t) t[[TRUE]], shapes = shapes)
  f04_same(function(t) t[[1.5]], shapes = shapes)
  f04_same(function(t) t[[c(1, 2)]], shapes = shapes, same_message = FALSE)
})

test_that("the motivating idioms give tibble's answer", {
  skip_if_no_gpu()
  f04_same(function(t) t[which(t$i > 1L), ])
  f04_same(function(t) t[t$l %in% TRUE, c("g", "x")])
  f04_same(function(t) t[nrow(t), ])
  f04_same(function(t) sum(t$x, na.rm = TRUE))
  f04_same(function(t) levels(t$f))
})

test_that("x$col and x[[col]] transfer only the requested column", {
  skip_if_no_gpu()
  orig_collect <- cuplyr:::gpu_collect
  orig_head <- cuplyr:::gpu_head
  local_mocked_bindings(
    gpu_collect = function(ptr, ...) {
      if (gpu_dim(ptr)[2] > 1L) stop("F04: whole table collected")
      orig_collect(ptr, ...)
    },
    gpu_head = function(ptr, ...) {
      if (gpu_dim(ptr)[2] > 1L) stop("F04: whole table collected")
      orig_head(ptr, ...)
    },
    .package = "cuplyr"
  )
  df <- f04_df()
  gt <- tbl_gpu(df)
  rd <- f04_global(function(t) list(t$x, t[["f"]], t[[6]]))
  expected <- list(df$x, df$f, df$d)
  expect_identical(rd(gt), expected)
  expect_identical(rd(dplyr::group_by(gt, g)), expected)
  lzf <- dplyr::filter(as_lazy(gt), i > 1L | is.na(i))
  expect_identical(rd(lzf), lapply(expected, function(v) v[c(2, 3, 4)]))
})

# ---------------------------------------------------------------------------
# `$<-` and `[[<-`

test_that("x$new <- scalar recycles to every row, all types", {
  skip_if_no_gpu()
  f04_same(function(t) { t$z <- 1; t })
  f04_same(function(t) { t$z <- "k"; t })
  f04_same(function(t) { t$z <- NA; t })
  f04_same(function(t) { t$z <- 7L; t }, shapes = c("eager", "lazy_pending", "grouped_lazy"))
  f04_same(function(t) { t$z <- as.Date("2020-02-02"); t }, shapes = c("eager", "lazy", "zero_rows"))
  f04_same(function(t) { t$z <- factor("w", levels = c("v", "w")); t }, shapes = c("eager", "lazy"))
  f04_same(function(t) { t$z <- TRUE; t }, shapes = c("eager", "zero_rows_lazy"))
})

test_that("x$col <- full-length vector adds or replaces in place", {
  skip_if_no_gpu()
  shapes <- c("eager", "lazy", "grouped", "grouped_lazy")
  f04_same(function(t) { t$z <- c(4, 3, 2, 1); t }, shapes = shapes)
  f04_same(function(t) { t$x <- c(10L, NA, 30L, 40L); t }, shapes = shapes)
  f04_same(function(t) { t$s <- c("w", "x", "y", NA); t }, shapes = shapes)
  f04_same(function(t) { t$d <- as.Date("2021-05-05") + 0:3; t }, shapes = shapes)
  f04_same(function(t) { t$f <- factor(c("b", "a", NA, "b")); t }, shapes = shapes)
  f04_same(function(t) { t$l <- c(FALSE, FALSE, NA, TRUE); t }, shapes = shapes)
})

test_that("x$col <- vector on a lazy table with pending ops uses the post-op row count", {
  skip_if_no_gpu()
  # the pending filter keeps 3 of 4 rows
  f04_same(function(t) { t$z <- c(7, 8, 9); t }, shapes = c("lazy_pending", "eager_filtered"))
  f04_same(function(t) { t$z <- c(1, 2, 3, 4); t }, shapes = c("lazy_pending", "eager_filtered"),
           same_message = FALSE)
  f04_same(function(t) { t$z <- c(7, 8, 9); t$z }, shapes = c("lazy_pending", "eager_filtered"))
})

test_that("x[[name]] <- and x[[position]] <- follow tibble", {
  skip_if_no_gpu()
  shapes <- c("eager", "lazy", "grouped")
  f04_same(function(t) { t[["z"]] <- 2.5; t }, shapes = shapes)
  f04_same(function(t) { t[["i"]] <- c(9L, 8L, 7L, 6L); t }, shapes = shapes)
  f04_same(function(t) { t[[2]] <- 99; t }, shapes = shapes)
  f04_same(function(t) { t[[8]] <- c("a", "b", "c", "d"); t }, shapes = shapes)
  f04_same(function(t) { t[[10]] <- 1; t }, shapes = shapes, same_message = FALSE)
})

test_that("assigning NULL drops a column, or does nothing if it is absent", {
  skip_if_no_gpu()
  shapes <- c("eager", "lazy", "lazy_pending", "grouped")
  f04_same(function(t) { t$s <- NULL; t }, shapes = shapes)
  f04_same(function(t) { t[["f"]] <- NULL; t }, shapes = shapes)
  f04_same(function(t) { t[[3]] <- NULL; t }, shapes = shapes)
  f04_same(function(t) { t$zz <- NULL; t }, shapes = shapes)
})

test_that("assignment keeps groups and .drop, regroups on a group column, drops a removed one", {
  skip_if_no_gpu()
  df <- f04_df()
  for (lazy in c(FALSE, TRUE)) {
    prep <- function(t) dplyr::group_by(t, g, .drop = FALSE)
    oracle <- prep(tibble::as_tibble(df))
    gpu <- prep(f04_up(df, lazy))
    set_z <- f04_global(function(t) { t$z <- 1; t })
    set_g <- f04_global(function(t) { t$g <- c("q", "q", "r", "q"); t })
    drop_g <- f04_global(function(t) { t$g <- NULL; t })
    for (fn in list(set_z, set_g, drop_g)) {
      o <- fn(oracle)
      r <- fn(gpu)
      expect_identical(dplyr::group_vars(r), dplyr::group_vars(o))
      expect_identical(dplyr::group_by_drop_default(r), dplyr::group_by_drop_default(o))
      expect_identical(dplyr::n_groups(r), dplyr::n_groups(o))
      expect_identical(tibble::as_tibble(dplyr::ungroup(collect(r))), tibble::as_tibble(dplyr::ungroup(o)))
    }
  }
})

test_that("assignment on zero-row tables follows tibble", {
  skip_if_no_gpu()
  shapes <- c("zero_rows", "zero_rows_lazy")
  f04_same(function(t) { t$z <- 1; t }, shapes = shapes)
  f04_same(function(t) { t$z <- integer(); t }, shapes = shapes)
  f04_same(function(t) { t$z <- 1:2; t }, shapes = shapes, same_message = FALSE)
})

test_that("assignment errors use tibble's classes", {
  skip_if_no_gpu()
  shapes <- c("eager", "lazy", "grouped")
  f04_same(function(t) { t$z <- 1:2; t }, shapes = shapes, same_message = FALSE)
  f04_same(function(t) { t[["z"]] <- c(1, 2, 3, 4, 5); t }, shapes = shapes, same_message = FALSE)
  f04_same(function(t) { t$z <- mean; t }, shapes = shapes, same_message = FALSE)
})

test_that("assignment does not modify other references to the same table", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f04_df())
  fn <- f04_global(function(t) { t$z <- 1; t$x <- NULL; t })
  out <- fn(gt)
  expect_identical(names(gt), names(f04_df()))
  expect_identical(tibble::as_tibble(collect(gt)), tibble::as_tibble(f04_df()))
  expect_identical(names(out), c(setdiff(names(f04_df()), "x"), "z"))
})

test_that("assigning a data frame or matrix column errors loudly", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f04_df())
  fn1 <- f04_global(function(t) { t$z <- data.frame(a = 1:4); t })
  fn2 <- f04_global(function(t) { t$z <- matrix(1:8, ncol = 2); t })
  expect_error(fn1(gt))
  expect_error(fn2(gt))
  expect_identical(tibble::as_tibble(collect(gt)), tibble::as_tibble(f04_df()))
})

test_that("cell assignment errors loudly, pointing at mutate(), and leaves x intact", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04_up(f04_df(), lazy)
    fns <- list(
      function(t) { t[1, "x"] <- 5; t },
      function(t) { t[1:2, ] <- 0; t },
      function(t) { t[[2, "x"]] <- 5; t },
      function(t) { t[] <- 0; t },
      function(t) { t[, ] <- 0; t },
      # f04_df() has as many columns as an uploaded data.frame's tbl_gpu has
      # internal fields: is.na() must not answer about those (F49 review 2, B5)
      function(t) { t[is.na(t)] <- 0; t },
      function(t) { t[matrix(c(TRUE, FALSE), 4L, 7L)] <- 0; t }
    )
    for (fn in fns) {
      fn <- f04_global(fn)
      expect_error(fn(gt), "mutate\\(\\)", info = paste(deparse(body(fn)), collapse = " "))
    }
    expect_error(is.na(gt), class = "cuplyr_error_unsupported_base")
    expect_error(anyNA(gt), class = "cuplyr_error_unsupported_base")
    # A logical column subscript of the wrong length is tibble's error.
    expect_error(f04_global(function(t) { t[c(TRUE, FALSE)] <- 0; t })(gt),
                 class = "vctrs_error_subscript")
    expect_true(is_tbl_gpu(gt))
    expect_identical(tibble::as_tibble(collect(gt)), tibble::as_tibble(f04_df()))
  }
})

# F49 review round 1 (B1): whole-column `[<-` is supported like tibble's
test_that("whole-column `[<-` matches tibble and leaves x intact", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04_up(f04_df(), lazy)
    fns <- list(
      function(t) { t["z"] <- 1; t },
      function(t) { t[, "x"] <- 0; t },
      function(t) { t[c("x", "i")] <- list(1, 2L); t },
      function(t) { t["x"] <- NULL; t }
    )
    for (fn in fns) {
      fn <- f04_global(fn)
      expect_identical(tibble::as_tibble(collect(fn(gt))), tibble::as_tibble(fn(f04_df())),
                       info = paste(deparse(body(fn)), collapse = " "))
    }
    expect_identical(tibble::as_tibble(collect(gt)), tibble::as_tibble(f04_df()))
  }
})

# ---------------------------------------------------------------------------
# Added after round 1 review (B1, B2)

test_that("[[ and [[<- evaluate the subscript once, in the caller's frame", {
  skip_if_no_gpu()
  df <- data.frame(x = 1:3, y = 4:6)
  oracle <- tibble::as_tibble(df)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04_up(df, lazy)
    f <- f04_global(function(tbls) { nm <- "x"; lapply(tbls, `[[`, nm) })
    expect_identical(f(list(gt)), f(list(oracle)))
    g <- f04_global(function(tbls, col) vapply(tbls, `[[`, integer(3), col))
    expect_identical(g(list(gt), "x"), g(list(oracle), "x"))
    h <- f04_global(function(t, ...) t[[...]])
    expect_identical(h(gt, "y"), h(oracle, "y"))
    s <- f04_global(function(tbls, v) lapply(tbls, `[[<-`, "x", value = v))
    expect_identical(
      tibble::as_tibble(collect(s(list(gt), 0L)[[1]])),
      s(list(oracle), 0L)[[1]]
    )

    calls <- 0L
    nxt <- function() { calls <<- calls + 1L; c("x", "y")[[calls]] }
    expect_identical(gt[[nxt()]], df$x)
    expect_identical(calls, 1L)
    calls <- 0L
    t2 <- gt
    t2[[nxt()]] <- 0L
    expect_identical(calls, 1L)
    expect_identical(tibble::as_tibble(collect(t2)), tibble::tibble(x = c(0L, 0L, 0L), y = 4:6))
  }
})

test_that("assignment keeps `...N` column names exactly as tibble does", {
  skip_if_no_gpu()
  f04_same(function(t) { t[[3]] <- 5:6; t$a <- 0L; t }, shapes = c("eager", "lazy"),
           df = data.frame(a = 1:2, b = 3:4))
  df2 <- setNames(data.frame(3:4, 1:2), c("b", "...1"))
  f04_same(function(t) { t$z <- 0; t }, shapes = c("eager", "lazy"), df = df2)
  f04_same(function(t) { t[["...1"]] <- c(9L, 8L); t }, shapes = c("eager", "lazy"), df = df2)
  f04_same(function(t) { t$b <- NULL; t }, shapes = c("eager", "lazy"), df = df2)
})

# ---------------------------------------------------------------------------
# Added after round 2 review (B4)

test_that("x[[i, j]] accepts named i and j", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  for (lazy in c(FALSE, TRUE)) {
    gt <- f04_up(df, lazy)
    expect_identical(gt[[i = 3, "a"]], 3L)
    expect_identical(gt[[3, j = "a"]], 3L)
    expect_identical(gt[[i = 3, j = "a"]], 3L)
    expect_identical(gt[[j = "b", i = 1]], 4L)
    expect_error(gt[[j = "c"]], class = "tibble_error_subset_columns_non_missing_only")
  }
})
