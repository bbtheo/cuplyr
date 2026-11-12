# F44 acceptance tests: empty/NA column names, dplyr's informational
# messages (count name bump, colwise grouping vars, join dots), and constant
# mutate()/transmute() on zero-column tables with rows.

# Runs `f()` and returns list(value, msg): `msg` is the first message
# condition signalled (muffled), or NULL.
f44_capture_msg <- function(f) {
  msg <- NULL
  value <- withCallingHandlers(
    f(),
    message = function(m) {
      if (is.null(msg)) msg <<- m
      invokeRestart("muffleMessage")
    }
  )
  list(value = value, msg = msg)
}

f44_names_df <- function(nms) {
  df <- data.frame(x = 1:3, y = c(2.5, NA, 4), z = c("a", "b", NA))
  names(df) <- nms
  df
}

f44_name_cases <- list(
  empty_last  = c("x", "y", ""),
  empty_first = c("", "y", "z"),
  na_last     = c("x", "y", NA),
  na_first    = c(NA, "y", "z")
)

# --- 1. empty / NA names ------------------------------------------------------

test_that("tbl_gpu() keeps empty/NA names and select() picks named columns (eager + lazy)", {
  skip_if_no_gpu()
  for (nm in names(f44_name_cases)) {
    nms <- f44_name_cases[[nm]]
    df <- f44_names_df(nms)
    g <- tbl_gpu(df, lazy = FALSE)
    expect_identical(names(g), names(df), info = nm)

    expected <- tibble::as_tibble(select(df, y))
    expect_identical(tibble::as_tibble(collect(select(g, y))), expected, info = nm)
    expect_identical(tibble::as_tibble(collect(select(as_lazy(g), y))), expected, info = nm)

    named <- nms[!is.na(nms) & nms != ""]
    expected2 <- tibble::as_tibble(select(df, dplyr::all_of(named)))
    expect_identical(
      tibble::as_tibble(collect(select(g, dplyr::all_of(named)))), expected2, info = nm
    )
  }
})

test_that("colnames<- and names<- accept empty and NA names (conformance #3601)", {
  skip_if_no_gpu()
  for (nm in names(f44_name_cases)) {
    nms <- f44_name_cases[[nm]]
    df <- data.frame(x = 1, y = 2, z = 3)
    colnames(df) <- nms

    g <- tbl_gpu(data.frame(x = 1, y = 2, z = 3))
    colnames(g) <- nms
    expect_identical(names(g), names(df), info = nm)
    expect_identical(collect(select(g, y))$y, select(df, y)$y, info = nm)

    g2 <- tbl_gpu(data.frame(x = 1, y = 2, z = 3))
    names(g2) <- nms
    expect_identical(names(g2), names(df), info = nm)
    expect_identical(collect(select(g2, y))$y, 2)
  }
})

test_that("names<- with an empty name works on a lazy table with pending ops", {
  skip_if_no_gpu()
  df <- data.frame(x = 1:3, y = c(2.5, NA, 4), z = c("a", "b", NA))
  l <- mutate(as_lazy(tbl_gpu(df)), w = x + 1L)
  names(l) <- c("x", "", "z", "w")

  ref <- mutate(df, w = x + 1L)
  names(ref) <- c("x", "", "z", "w")

  expect_identical(names(l), names(ref))
  expect_identical(
    tibble::as_tibble(collect(select(l, w, x))),
    tibble::as_tibble(select(ref, w, x))
  )
})

test_that("printing a tbl_gpu with empty or NA names does not error", {
  skip_if_no_gpu()
  for (nm in names(f44_name_cases)) {
    g <- tbl_gpu(f44_names_df(f44_name_cases[[nm]]))
    expect_no_error(utils::capture.output(print(g)))
  }
})

test_that("verbs dplyr refuses on empty/NA names error loudly", {
  skip_if_no_gpu()
  df <- f44_names_df(c("x", "y", ""))
  g <- tbl_gpu(df)

  # dplyr: tidyselect refuses to select an unnamed column
  expect_error(select(df, everything()), class = "vctrs_error_names_cannot_be_empty")
  expect_error(select(g, everything()), class = "vctrs_error_names_cannot_be_empty")

  # dplyr: data-masking verbs refuse the whole table
  msg <- "Can't transform a data frame with `NA` or `\"\"` names."
  expect_error(mutate(df, a = 1), msg, fixed = TRUE)
  expect_error(mutate(g, a = 1), msg, fixed = TRUE)
  expect_error(filter(g, x > 1), msg, fixed = TRUE)
  expect_error(arrange(g, x), msg, fixed = TRUE)

  # dplyr checks missing (NA) names first, with its own message
  df_na <- f44_names_df(c(NA, "y", "z"))
  g_na <- tbl_gpu(df_na)
  msg_na <- "Can't transform a data frame with missing names."
  expect_error(mutate(df_na, a = 1), msg_na, fixed = TRUE)
  expect_error(mutate(g_na, a = 1), msg_na, fixed = TRUE)
  expect_error(filter(g_na, y > 1), msg_na, fixed = TRUE)
  expect_error(mutate(as_lazy(g_na), a = 1) |> collect(), msg_na, fixed = TRUE)

  # repeated NA / "" names: dplyr reports missing / empty names, not duplicates
  df_na2 <- f44_names_df(c("x", NA, NA))
  expect_error(mutate(df_na2, a = 1), msg_na, fixed = TRUE)
  expect_error(mutate(tbl_gpu(df_na2), a = 1), msg_na, fixed = TRUE)
  df_e2 <- f44_names_df(c("x", "", ""))
  expect_error(filter(df_e2, x > 1), msg, fixed = TRUE)
  expect_error(filter(tbl_gpu(df_e2), x > 1), msg, fixed = TRUE)
  expect_error(mutate(as_lazy(tbl_gpu(df_e2)), a = 1) |> collect(), msg, fixed = TRUE)
})

# --- 2a. count()/tally() name bump message -----------------------------------

test_that("count()/tally()/add_count()/add_tally() inform when `n` is already present", {
  skip_if_no_gpu()
  df <- data.frame(n = c(1, 1, 2, 2, 2))
  cases <- list(
    count     = function(d) count(d, n),
    tally     = function(d) tally(group_by(d, n)),
    add_count = function(d) add_count(d, n),
    add_tally = function(d) add_tally(d)
  )
  for (nm in names(cases)) {
    f <- cases[[nm]]
    ref <- f44_capture_msg(function() f(tibble::as_tibble(df)))
    for (lazy in c(FALSE, TRUE)) {
      g <- tbl_gpu(df, lazy = lazy)
      got <- f44_capture_msg(function() collect(f(g)))
      info <- paste(nm, if (lazy) "lazy" else "eager")
      expect_s3_class(got$msg, "rlang_message")
      expect_identical(conditionMessage(got$msg), conditionMessage(ref$msg), info = info)
      expect_identical(names(got$value), names(ref$value), info = info)
    }
  }
  # the exact text dplyr uses
  out <- f44_capture_msg(function() count(tbl_gpu(df), n))
  expect_match(conditionMessage(out$msg),
               "Storing counts in `nn`, as `n` already present in input", fixed = TRUE)
})

test_that("no name-bump message when `name` is given or `n` is free", {
  skip_if_no_gpu()
  df <- data.frame(n = c(1, 1, 2, 2, 2))
  g <- tbl_gpu(df)
  expect_message(out <- count(g, n, name = "n"), NA)
  expect_named(collect(out), "n")
  expect_message(out <- count(g, n, name = "nn"), NA)
  expect_named(collect(out), c("n", "nn"))
  expect_message(tally(g), NA)
  expect_message(count(tbl_gpu(data.frame(x = c(1, 1, 2))), x), NA)

  df2 <- data.frame(n = c(1, 1, 2, 2, 2), nn = 1:5)
  expect_message(out <- count(tbl_gpu(df2), n), "already present")
  expect_named(collect(out), c("n", "nn"))
})

# --- 2b. colwise verbs on grouped tables --------------------------------------

test_that("mutate_all/mutate_if/transmute_all/transmute_if inform about ignored grouping vars", {
  skip_if_no_gpu()
  df <- data.frame(gr1 = rep(1:2, 4), gr2 = rep(1:2, each = 4), x = 1:8)
  cases <- list(
    mutate_all    = function(d) mutate_all(group_by(d, gr1), sqrt),
    transmute_all = function(d) transmute_all(group_by(d, gr1), sqrt),
    mutate_if     = function(d) mutate_if(group_by(d, gr1), is.integer, sqrt),
    transmute_if  = function(d) transmute_if(group_by(d, gr1), is.integer, sqrt)
  )
  for (nm in names(cases)) {
    f <- cases[[nm]]
    ref <- f44_capture_msg(function() f(tibble::as_tibble(df)))
    for (lazy in c(FALSE, TRUE)) {
      g <- tbl_gpu(df, lazy = lazy)
      got <- f44_capture_msg(function() f(g))
      info <- paste(nm, if (lazy) "lazy" else "eager")
      expect_false(is.null(got$msg), info = info)
      expect_identical(conditionMessage(got$msg), conditionMessage(ref$msg), info = info)
      expect_match(conditionMessage(got$msg), "ignored the following grouping variables",
                   info = info)
    }
    expect_same_as_dplyr(df, function(d) suppressMessages(f(d)))
    expect_same_as_dplyr_lazy(df, function(d) suppressMessages(f(d)))
  }
})

test_that("no grouping-vars message when dplyr gives none", {
  skip_if_no_gpu()
  df <- data.frame(gr1 = rep(1:2, 4), x = as.double(1:8))
  g <- tbl_gpu(df)
  gg <- group_by(g, gr1)
  expect_message(mutate_all(g, sqrt), NA)                    # ungrouped
  expect_message(mutate_at(gg, dplyr::vars(x), sqrt), NA)    # explicit selection
  expect_message(mutate(gg, x = sqrt(x)), NA)                # plain verb
  expect_message(transmute(gg, x = sqrt(x)), NA)
  expect_message(summarise_all(gg, mean), NA)                # summarise is silent
})

# --- 2c. joins reject extra arguments in `...` --------------------------------

test_that("joins error on non-empty `...` like dplyr (rlib_error_dots_nonempty)", {
  skip_if_no_gpu()
  df1 <- data.frame(a = 1:3)
  df2 <- data.frame(a = 1L)
  joins <- list(
    inner_join = inner_join, left_join = left_join, right_join = right_join,
    full_join = full_join, semi_join = semi_join, anti_join = anti_join,
    nest_join = nest_join
  )
  for (nm in names(joins)) {
    j <- joins[[nm]]
    expect_error(j(df1, df2, on = "a"), class = "rlib_error_dots_nonempty")
    for (lazy in c(FALSE, TRUE)) {
      x <- tbl_gpu(df1)
      y <- tbl_gpu(df2)
      if (lazy) {
        x <- as_lazy(x)
        y <- as_lazy(y)
      }
      cnd <- expect_error(j(x, y, on = "a"), class = "rlib_error_dots_nonempty",
                          info = paste(nm, lazy))
      expect_match(conditionMessage(cnd), "`...` must be empty", fixed = TRUE)
      expect_match(conditionMessage(cnd), "on = \"a\"", fixed = TRUE)
    }
  }
})

test_that("nest_join() still accepts its own na_matches= and unmatched= arguments", {
  skip_if_no_gpu()
  df1 <- data.frame(a = c(1, NA))
  df2 <- data.frame(a = c(1, NA), b = 1:2)
  ref <- nest_join(df1, df2, by = "a", na_matches = "never")
  got <- collect(nest_join(tbl_gpu(df1), tbl_gpu(df2), by = "a", na_matches = "never",
                             name = "df2"))
  expect_identical(lapply(got$df2, nrow), lapply(ref$df2, nrow))
  expect_length(ref$df2, 2)
  # unmatched = "error": y rows without a match in x error, as in dplyr
  df3 <- data.frame(a = 1, b = 1L)
  expect_error(nest_join(df3, df2, by = "a", unmatched = "error"),
               class = "dplyr_error_join_matches_remaining")
  expect_error(nest_join(tbl_gpu(df3), tbl_gpu(df2), by = "a", unmatched = "error"),
               class = "dplyr_error_join_matches_remaining")
})

# --- 3. constant mutate()/transmute() on zero-column tables -------------------

test_that("constant mutate() on a zero-column table with rows recycles to nrow", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:3)
  pipes <- list(
    function(d) mutate(select(d, -a), b = 1),
    function(d) mutate(select(d, -a), b = 1L, c = "s", d = TRUE, e = NA),
    function(d) mutate(select(d, -a), b = 2.5, c = b * 2),
    function(d) transmute(select(d, -a), b = 2.5),
    function(d) mutate(select(filter(d, a > 1), -a), b = "k"),
    function(d) filter(mutate(select(d, -a), b = 1), b > 0)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("constant mutate() on a zero-column, zero-row table", {
  skip_if_no_gpu()
  df <- data.frame(a = integer())
  p <- function(d) mutate(select(d, -a), b = 1)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  p2 <- function(d) transmute(select(d, -a), b = "x")
  expect_same_as_dplyr(df, p2)
})

test_that("mutate() of a wrong-length vector on a zero-column table errors", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:3)
  expect_error(mutate(select(tibble::as_tibble(df), -a), b = 1:2))
  expect_error(collect(mutate(select(tbl_gpu(df), -a), b = 1:2)))
})

test_that("mutate() on a zero-column table still validates .by", {
  skip_if_no_gpu()
  df <- data.frame(a = 1:3)
  expect_error(mutate(select(tibble::as_tibble(df), -a), b = 1, .by = a))
  expect_error(collect(mutate(select(tbl_gpu(df), -a), b = 1, .by = a)), "exist")
  expect_error(collect(mutate(select(as_lazy(tbl_gpu(df)), -a), b = 1, .by = a)), "exist")
})

test_that("tidyselect inside mutate()/transmute() on a zero-column table never sees helper columns", {
  skip_if_no_gpu()
  # Either dplyr's result or a loud error; never a wrong answer or a leaked
  # internal column.
  df <- data.frame(a = 1:3)
  pipes <- list(
    function(d) mutate(select(d, -a), b = dplyr::if_all(everything(), is.na)),
    function(d) mutate(select(d, -a), b = dplyr::if_any(everything(), ~ .x == 0)),
    function(d) mutate(select(d, -a), across(everything(), ~ .x + 1L, .names = "{.col}_new"), b = 1),
    function(d) transmute(select(d, -a), across(everything(), ~ .x + 1L), b = 2),
    function(d) transmute(select(d, -a), b = dplyr::if_all(everything(), is.na))
  )
  for (i in seq_along(pipes)) {
    p <- pipes[[i]]
    expected <- p(tibble::as_tibble(df))
    for (lazy in c(FALSE, TRUE)) {
      g <- tbl_gpu(df, lazy = lazy)
      res <- tryCatch(collect(p(g)), error = function(e) e)
      if (!inherits(res, "error")) {
        expect_equal(tibble::as_tibble(res), expected, info = paste(i, lazy))
      } else {
        expect_false(grepl("cuplyr_seed", conditionMessage(res)), info = paste(i, lazy))
      }
    }
  }
})
