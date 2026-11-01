# F82: the subscript of `.env[[s]]` / `.data[[s]]` is evaluated before the name
# is used. dplyr (rlang 1.2) evaluates `s` in the data mask, so a column named
# like the subscript variable shadows the environment; rlang's capture already
# injects `.data[[s]]` subscripts written in a verb call, so only an injected
# raw expression reaches the parser with a non-literal `.data` subscript.
# Every pipeline below has a decoy: reading `.env$k` instead of `.env[[k]]`
# gives a value that is valid but different.

f82_df <- function() {
  tibble::tibble(
    s = c("k", "hello", "zz", NA, "zz"),
    x = c(1, 5, 10, NA, 3),
    b = c(2L, 3L, 4L, 5L, 6L),
    g = c(1L, 1L, 2L, 2L, 2L)
  )
}

f82_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# The root cause of a chained condition (rlang errors carry `parent`).
f82_root <- function(e) {
  while (!is.null(e$parent)) e <- e$parent
  e
}

# dplyr raises an error; the GPU (eager and lazy, collected) raises one with
# the same class vector and the same root-cause class and message.
f82_expect_same_error <- function(df, pipeline) {
  ref <- tryCatch({
    pipeline(tibble::as_tibble(df))
    NULL
  }, error = identity)
  expect_s3_class(ref, "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch({
      collect(pipeline(tbl_gpu(df, lazy = lazy)))
      NULL
    }, error = identity)
    mode <- if (lazy) "lazy" else "eager"
    expect_s3_class(got, "error")
    if (inherits(got, "error")) {
      expect_identical(class(got), class(ref), info = paste(mode, conditionMessage(got)))
      expect_identical(class(f82_root(got)), class(f82_root(ref)), info = mode)
      expect_identical(conditionMessage(f82_root(got)), conditionMessage(f82_root(ref)), info = mode)
    }
  }
}

# --- .env[[<symbol>]] ---------------------------------------------------------

test_that("F82: filter() reads .env[[k]] as the variable k names", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "hello"
  hello <- "zz"
  f82_both(df, function(d) filter(d, s == .env[[k]]))
  f82_both(df, function(d) filter(d, s != .env[[k]] | is.na(s)))
  f82_both(df, function(d) filter(d, x > 2, s == .env[[k]]))
  f82_both(df, function(d) filter_out(d, s == .env[[k]]))
  f82_both(df, function(d) filter(group_by(d, g), s == .env[[k]]))
  f82_both(df, function(d) filter(d, s == .env[[k]], .by = g))
})

test_that("F82: a column-free filter() predicate reads .env[[k]] correctly", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "hello"
  hello <- "zz"
  f82_both(df, function(d) filter(d, .env[[k]] == "zz"))
  f82_both(df, function(d) filter_out(d, .env[[k]] == "zz"))
})

test_that("F82: mutate() reads .env[[k]] as the variable k names", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "hello"
  hello <- "zz"
  nm <- "lim"
  lim <- 4
  f82_both(df, function(d) mutate(d, y = s == .env[[k]]))
  f82_both(df, function(d) mutate(d, y = paste0(s, .env[[k]])))
  f82_both(df, function(d) mutate(d, y = dplyr::if_else(s == .env[[k]], x, -1)))
  f82_both(df, function(d) mutate(d, y = x + .env[[nm]]))
  f82_both(df, function(d) mutate(d, y = .env[[k]]))
  f82_both(df, function(d) mutate(group_by(d, g), y = s == .env[[k]]))
  f82_both(df, function(d) transmute(d, y = x * .env[[nm]]))
})

test_that("F82: summarise() reads .env[[k]] as the variable k names", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "hello"
  hello <- "zz"
  nm <- "lim"
  lim <- 4
  f82_both(df, function(d) summarise(d, n = sum(s == .env[[k]], na.rm = TRUE)))
  f82_both(df, function(d) summarise(d, m = max(x + .env[[nm]], na.rm = TRUE)))
  f82_both(df, function(d) summarise(group_by(d, g), n = sum(s == .env[[k]], na.rm = TRUE)),
           arrange_by = "g")
})

test_that("F82: arrange() and group_by() read .env[[k]] as the variable k names", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "hello"
  hello <- "zz"
  nm <- "lim"
  lim <- 4
  f82_both(df, function(d) arrange(d, desc(s == .env[[k]]), x))
  f82_both(df, function(d) arrange(d, abs(x - .env[[nm]])))
  f82_both(df, function(d) summarise(group_by(d, z = s == .env[[k]]), n = n()), arrange_by = "z")
  f82_both(df, function(d) count(group_by(d, z = x > .env[[nm]])), arrange_by = "z")
})

test_that("F82: a computed .env[[<call>]] subscript is evaluated", {
  skip_if_no_gpu()
  df <- f82_df()
  hello <- "zz"
  lim <- 4
  f82_both(df, function(d) filter(d, s == .env[[paste0("hel", "lo")]]))
  f82_both(df, function(d) filter(d, s == .env[[tolower("HELLO")]]))
  f82_both(df, function(d) mutate(d, y = s == .env[[paste0("hel", "lo")]]))
  f82_both(df, function(d) mutate(d, y = x + .env[[sprintf("l%sm", "i")]]))
  f82_both(df, function(d) summarise(d, n = sum(s == .env[[tolower("HELLO")]], na.rm = TRUE)))
})

test_that("F82: literal .env[['name']] and .env$name still read the environment", {
  skip_if_no_gpu()
  df <- dplyr::mutate(f82_df(), lim = c(100, 100, 100, 100, 100))
  lim <- 4
  f82_both(df, function(d) filter(d, x > .env[["lim"]]))
  f82_both(df, function(d) filter(d, x > .env$lim))
  f82_both(df, function(d) mutate(d, y = x + .env[["lim"]], z = x + lim))
  # the subscript variable is not a column: it reads the environment
  nm <- "lim"
  f82_both(df, function(d) mutate(d, y = x + .env[[nm]]))
})

# --- .data[[<symbol>]] --------------------------------------------------------

test_that("F82: .data[[nm]] reads the column nm names, not a column called nm", {
  skip_if_no_gpu()
  df <- dplyr::mutate(f82_df(), nm = c(50L, 40L, 30L, 20L, 10L))
  nm <- "b"
  f82_both(df, function(d) filter(d, .data[[nm]] > 3))
  f82_both(df, function(d) mutate(d, y = .data[[nm]] * 2L))
  f82_both(df, function(d) summarise(d, t = sum(.data[[nm]])))
  f82_both(df, function(d) summarise(group_by(d, g), t = sum(.data[[nm]])), arrange_by = "g")
  f82_both(df, function(d) arrange(d, desc(.data[[nm]])))
  f82_both(df, function(d) summarise(group_by(d, .data[[nm]]), n = n()), arrange_by = "b")
  f82_both(df, function(d) mutate(d, across(x, ~ .x + .data[[nm]])))
})

test_that("F82: an injected raw .data[[k]] evaluates k before reading the column", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "x"
  f82_both(df, function(d) filter(d, !!rlang::new_quosure(quote(.data[[k]] > 2), environment())))
  f82_both(df, function(d) mutate(d, z = !!quote(.data[[k]] * 2)))
  f82_both(df, function(d) summarise(d, z = sum(!!quote(.data[[k]]), na.rm = TRUE)))
  f82_both(df, function(d) arrange(d, !!quote(desc(.data[[k]]))))
})

# --- errors -------------------------------------------------------------------

test_that("F82: an invalid .env subscript gives dplyr's error", {
  skip_if_no_gpu()
  df <- f82_df()
  zz <- "nope_f82_var"
  f82_expect_same_error(df, function(d) filter(d, x > .env[[1]]))
  f82_expect_same_error(df, function(d) filter(d, s == .env[[NA_character_]]))
  f82_expect_same_error(df, function(d) mutate(d, y = x + .env[[NA]]))
  f82_expect_same_error(df, function(d) mutate(d, y = .env[[1]]))
  f82_expect_same_error(df, function(d) mutate(d, y = x + .env[[c("a", "b")]]))
  f82_expect_same_error(df, function(d) summarise(d, y = sum(x + .env[[1]])))
  f82_expect_same_error(df, function(d) arrange(d, x + .env[[1]]))
})

test_that("F82: a .env subscript naming a missing variable gives dplyr's error", {
  skip_if_no_gpu()
  df <- f82_df()
  zz <- "nope_f82_var"
  f82_expect_same_error(df, function(d) filter(d, x > .env[[zz]]))
  f82_expect_same_error(df, function(d) mutate(d, y = x + .env[[zz]]))
  f82_expect_same_error(df, function(d) mutate(d, y = x + .env[[nope_f82_subscript]]))
  f82_expect_same_error(df, function(d) summarise(d, y = max(x + .env[[zz]])))
})

test_that("F82: a .env subscript that is a column reads the column (dplyr's error)", {
  skip_if_no_gpu()
  df <- f82_df()
  # the variable `s` would be a valid name; the column `s` shadows it
  s <- "hello"
  hello <- "zz"
  f82_expect_same_error(df, function(d) mutate(d, y = x + .env[[s]]))
  f82_expect_same_error(df, function(d) filter(d, .env[[s]] == "zz"))
  f82_expect_same_error(df, function(d) mutate(d, y = paste0(s, .env[[s]])))
})

test_that("F82: an invalid .data subscript gives dplyr's error", {
  skip_if_no_gpu()
  df <- f82_df()
  f82_expect_same_error(df, function(d) mutate(d, y = .data[[1]] + 1))
  f82_expect_same_error(df, function(d) filter(d, .data[[1]] > 0))
  f82_expect_same_error(df, function(d) mutate(d, y = .data[[NA_character_]]))
  # an injected raw subscript that is a column: dplyr reads the column
  s <- "x"
  f82_expect_same_error(df, function(d) mutate(d, y = !!quote(.data[[s]] + 1)))
})

# --- an injected raw .data[[k]] on the CPU route (round-1 review B2) -----------

test_that("F82: an injected raw .data[[k]] works when the call runs in dplyr", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "x"
  # a local function the GPU can't run: its dot sends the call to dplyr
  twice <- function(v) v * 2
  f82_both(df, function(d) summarise(d, n = n(), z = sum(!!quote(.data[[k]]), na.rm = TRUE)))
  f82_both(df, function(d) summarise(d, z = sum(!!quote(.data[[k]]), na.rm = TRUE),
                                     w = sum(twice(x), na.rm = TRUE)))
  f82_both(df, function(d) summarise(group_by(d, g), z = sum(!!quote(.data[[k]]), na.rm = TRUE),
                                     w = sum(twice(x), na.rm = TRUE)), arrange_by = "g")
  f82_both(df, function(d) transmute(d, z = !!quote(.data[[k]] * 2), w = twice(x)))
  f82_both(df, function(d) mutate(d, z = !!quote(.data[[k]] * 2), w = twice(x)))
  f82_both(df, function(d) arrange(d, !!quote(twice(.data[[k]]))))
  f82_both(df, function(d) summarise(group_by(d, z = !!quote(twice(.data[[k]]))), n = n()),
           arrange_by = "z")
  f82_both(df, function(d) reframe(d, z = !!quote(twice(.data[[k]]))))
})

test_that("F82: an injected raw .data[[s]] whose subscript is a column gives dplyr's error on the CPU route", {
  skip_if_no_gpu()
  df <- f82_df()
  s <- "x"
  twice <- function(v) v * 2
  f82_expect_same_error(df, function(d) summarise(d, n = n(), z = sum(!!quote(.data[[s]]))))
  f82_expect_same_error(df, function(d) transmute(d, z = !!quote(.data[[s]] * 2), w = twice(x)))
  f82_expect_same_error(df, function(d) arrange(d, !!quote(twice(.data[[s]]))))
})

# --- round-2 review: routes that rebuild the call (B2) and labels (B5) ---------

test_that("F82: a raw .data[[k]] in count()/tally()/add_count() wt reads the quosure's k", {
  skip_if_no_gpu()
  df <- f82_df()
  # a same-named variable outside the pipeline names another column
  k <- "b"
  twice <- function(v) v * 2
  f82_both(df, function(d) { k <- "x"; count(d, wt = !!quote(.data[[k]])) })
  f82_both(df, function(d) { k <- "x"; count(d, g, wt = !!quote(.data[[k]])) }, arrange_by = "g")
  f82_both(df, function(d) { k <- "x"; count(d, z = twice(g), wt = !!quote(.data[[k]])) },
           arrange_by = "z")
  f82_both(df, function(d) { k <- "x"; tally(group_by(d, g), wt = !!quote(.data[[k]])) },
           arrange_by = "g")
  f82_both(df, function(d) { k <- "x"; add_count(d, g, wt = !!quote(.data[[k]])) })
  f82_both(df, function(d) { k <- "x"; add_tally(d, wt = !!quote(.data[[k]])) })
})

test_that("F82: a raw .data[[k]] in slice_min()/slice_max()/slice_sample() reads the quosure's k", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "b"
  twice <- function(v) v * 2
  f82_both(df, function(d) { k <- "x"; slice_min(d, !!quote(twice(.data[[k]])), n = 2) })
  f82_both(df, function(d) { k <- "x"; slice_max(d, !!quote(.data[[k]]), n = 2) })
  f82_both(df, function(d) { k <- "x"; slice_min(group_by(d, g), !!quote(.data[[k]]), n = 1) })
  df2 <- dplyr::mutate(df, w = c(0, 0, 1, 0, 0), x = c(1, 0, 0, 0, 0))
  # all the weight on one row: the row the weights name is the only one drawn
  f82_both(df2, function(d) { k <- "w"; slice_sample(d, n = 1, weight_by = !!quote(.data[[k]])) })
})

test_that("F82: an unnamed raw .data[[k]] dot on the CPU route keeps dplyr's name", {
  skip_if_no_gpu()
  df <- f82_df()
  k <- "x"
  twice <- function(v) v * 2
  # (cuplyr's own "Unnamed mutate expression" warning is not under test here)
  suppressWarnings(f82_both(df, function(d) mutate(d, !!quote(.data[[k]] * 2), w = twice(x))))
  suppressWarnings(f82_both(df, function(d) transmute(d, !!quote(.data[[k]] * 2), w = twice(x))))
  f82_both(df, function(d) summarise(d, n = n(), !!quote(sum(.data[[k]], na.rm = TRUE))))
  f82_both(df, function(d) reframe(d, !!quote(twice(.data[[k]]))))
  f82_both(df, function(d) distinct(d, !!quote(twice(.data[[k]]))))
  f82_both(df, function(d) count(d, !!quote(twice(.data[[k]]))), arrange_by = "twice(.data[[k]])")
  f82_both(df, function(d) summarise(group_by(d, !!quote(twice(.data[[k]]))), n = n()),
           arrange_by = "twice(.data[[k]])")
})

# --- round-3 review: named arguments reach dplyr's checks (B1, B2) -------------

test_that("F82: slice() rejects named `...` like dplyr, native and CPU route", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 2), x = c(1, 2))
  f82_expect_same_error(df, function(d) slice(d, 1, foo = g))
  f82_expect_same_error(df, function(d) slice(d, 1, foo = 1))
  f82_expect_same_error(df, function(d) slice(group_by(d, g), 1, foo = g))
})

test_that("F82: transmute() rejects .keep/.before/.after like dplyr, native and CPU route", {
  skip_if_no_gpu()
  df <- f82_df()
  twice <- function(v) v * 2
  f82_expect_same_error(df, function(d) transmute(d, y = twice(x), .keep = "all"))
  f82_expect_same_error(df, function(d) transmute(d, y = twice(x), .before = x))
  f82_expect_same_error(df, function(d) transmute(d, y = twice(x), .after = x))
  f82_expect_same_error(df, function(d) transmute(d, y = x * 2, .keep = "all"))
  f82_expect_same_error(df, function(d) transmute(d, y = x * 2, .before = x))
})

test_that("F82: count()/add_count() pass `add` to dplyr's group_by() check", {
  skip_if_no_gpu()
  df <- f82_df()
  twice <- function(v) v * 2
  f82_expect_same_error(df, function(d) count(d, add = twice(g)))
  f82_expect_same_error(df, function(d) count(d, add = g))
  f82_expect_same_error(df, function(d) add_count(d, add = twice(g)))
})
