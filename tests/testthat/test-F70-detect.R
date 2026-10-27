# F70: ir_has_checked_int() mirrors checked_int_kind() kind 1 of
# src/expr_eval.hpp, minus the nodes C++ never reports overflow for.

f70_flag <- function(expr, types = c(a = "INT32", b = "INT32", x = "FLOAT64", l = "BOOL8")) {
  schema <- list(names = names(types), types = unname(types))
  ir <- ir_parse_quo(rlang::new_quosure(expr, globalenv()), schema)
  ir_has_checked_int(ir, schema)
}

test_that("ir_has_checked_int() flags integer +, -, * only", {
  expect_true(f70_flag(quote(a + b)))
  expect_true(f70_flag(quote(a - 1L)))
  expect_true(f70_flag(quote(a * b > 3L)))
  expect_true(f70_flag(quote(l + l)))
  expect_true(f70_flag(quote((a * b) + x)))
  expect_false(f70_flag(quote(x + 1)))
  expect_false(f70_flag(quote(a + x)))
  expect_false(f70_flag(quote(a + 1)))
  expect_false(f70_flag(quote(a %/% b)))
  expect_false(f70_flag(quote(a %% 2L)))
  expect_false(f70_flag(quote(a > b)))
  expect_false(f70_flag(quote(a)))
})

test_that("ir_has_checked_int() skips a node with a null literal operand but walks below it", {
  expect_false(f70_flag(quote(a + NA)))
  expect_false(f70_flag(quote(a * NA_integer_)))
  expect_false(f70_flag(quote(NA_integer_ - b)))
  expect_true(f70_flag(quote((a * b) + NA)))
  expect_true(f70_flag(quote(NA + (a - b))))
})

test_that("a null-literal node does not block pushdown", {
  skip_if_no_gpu()
  d <- data.frame(a = 1:4, b = 4:1, keep = c(TRUE, FALSE, TRUE, TRUE))
  lz <- as_lazy(tbl_gpu(d))
  out <- capture.output(explain(lz |> mutate(y = a + NA) |> filter(keep)))
  p <- out[seq(match("Optimized plan:", out) + 1L, length(out))]
  expect_match(p[1], "ast_mutate")
  expect_match(p[2], "ast_filter")
  expect_equal(
    tibble::as_tibble(collect(lz |> mutate(y = a + NA) |> filter(keep))),
    tibble::as_tibble(d |> dplyr::mutate(y = a + NA) |> dplyr::filter(keep))
  )
})

# ---------------------------------------------------------------------------
# The detector types each subtree once (round-2 review N1). A reference that
# re-infers every call operand with ir_infer_type(), as the first version did,
# must give the same answer on every tree.
# ---------------------------------------------------------------------------

f70_naive <- function(ir, schema) {
  if (is.null(ir) || !identical(ir$kind, "call")) return(FALSE)
  args <- ir$args
  if (length(args) == 2L && ir$op %in% c("+", "-", "*") &&
        !ir_lit_is_null(args[[1]]) && !ir_lit_is_null(args[[2]])) {
    leaf_type <- function(a) {
      if (a$kind == "col") {
        idx <- match(a$name, schema$names)
        if (is.na(idx)) "missing" else schema$types[idx]
      } else a$type
    }
    is_call <- vapply(args, function(a) a$kind == "call", logical(1))
    verdict <- NA
    known <- 0L
    for (a in args[!is_call]) {
      t <- leaf_type(a)
      if (is.null(t)) next
      if (t == "missing") { verdict <- TRUE; break }
      known <- known + 1L
      if (!t %in% c("INT32", "BOOL8")) { verdict <- FALSE; break }
    }
    if (is.na(verdict)) {
      for (a in args[is_call]) {
        t <- tryCatch(ir_infer_type(a, schema), error = function(e) NA_character_)
        if (is.na(t)) { verdict <- TRUE; break }
        known <- known + 1L
        if (!t %in% c("INT32", "BOOL8")) { verdict <- FALSE; break }
      }
      if (is.na(verdict)) verdict <- known > 0L
    }
    if (verdict) return(TRUE)
  }
  any(vapply(args, f70_naive, logical(1), schema = schema))
}

f70_chain <- function(n, bottom, leaf) {
  nm <- c("x0", paste0("v", seq_len(n)))
  schema <- list(names = nm, types = c(bottom, rep(leaf, n)))
  e <- as.name("x0")
  for (v in nm[-1]) e <- call("+", e, as.name(v))
  list(ir = ir_parse_quo(rlang::new_quosure(e, globalenv()), schema), schema = schema)
}

test_that("ir_has_checked_int() on long + chains", {
  # a double at the bottom makes every node of the chain double
  ch <- f70_chain(16, "FLOAT64", "INT32")
  expect_false(ir_has_checked_int(ch$ir, ch$schema))
  ch <- f70_chain(16, "FLOAT64", "FLOAT64")
  expect_false(ir_has_checked_int(ch$ir, ch$schema))
  ch <- f70_chain(16, "INT32", "INT32")
  expect_true(ir_has_checked_int(ch$ir, ch$schema))
  # an integer pair at the bottom, a double leaf above it
  ch <- f70_chain(16, "INT32", "INT32")
  ch$schema$types[17] <- "FLOAT64"
  expect_true(ir_has_checked_int(ch$ir, ch$schema))
  # a double bottom, an integer pair as a call operand higher up
  types <- c(a = "INT32", b = "INT32", x = "FLOAT64")
  expect_true(f70_flag(quote(x + a + (a + b) + a), types))
  expect_false(f70_flag(quote(x + a + (a + 1) + a), types))
})

test_that("ir_has_checked_int() agrees with re-inferring every call operand", {
  withr::local_seed(70)
  sch <- list(names = c("a", "b", "x", "l", "s"),
              types = c("INT32", "INT32", "FLOAT64", "BOOL8", "STRING"))
  # the same IR typed against other schemas, so that some subtrees fail to type
  # (STRING arithmetic) or name a missing column
  schemas <- list(
    sch,
    list(names = sch$names, types = c("FLOAT64", "INT32", "INT32", "BOOL8", "STRING")),
    list(names = sch$names, types = c("STRING", "INT32", "FLOAT64", "INT32", "INT32")),
    list(names = c("b", "x"), types = c("INT32", "FLOAT64"))
  )
  leaves <- list(quote(a), quote(b), quote(x), quote(l), quote(s), 1L, 2.5, NA,
                 NA_integer_, TRUE)
  ops <- c("+", "-", "*", "+", "-", "*", "%/%", ">", "&", "/")
  gen <- function(depth) {
    if (depth == 0 || stats::runif(1) < 0.25) return(leaves[[sample(length(leaves), 1)]])
    r <- stats::runif(1)
    if (r < 0.75) return(call(sample(ops, 1), gen(depth - 1), gen(depth - 1)))
    if (r < 0.85) return(call("abs", gen(depth - 1)))
    call("if_else", call(">", gen(depth - 1), 0L), gen(depth - 1), gen(depth - 1))
  }
  n <- 0L
  mismatches <- list()
  while (n < 400L) {
    e <- gen(sample(1:6, 1))
    ir <- tryCatch(ir_parse_quo(rlang::new_quosure(e, globalenv()), sch),
                   error = function(err) NULL)
    if (is.null(ir)) next
    n <- n + 1L
    for (s in schemas) {
      if (!identical(ir_has_checked_int(ir, s), f70_naive(ir, s))) {
        mismatches[[length(mismatches) + 1L]] <- list(expr = e, types = s$types)
      }
    }
  }
  expect_identical(mismatches, list())
})

# ---------------------------------------------------------------------------
# Pipeline shapes the round-2 review checked by probe: the lazy (and eager)
# warning, values and classes match dplyr on a tibble.
# ---------------------------------------------------------------------------

F70D_M <- .Machine$integer.max

f70d_df <- function() {
  data.frame(
    a = c(F70D_M, 1L, 5L, 2L),
    b = c(1L, 2L, 3L, 4L),
    keep = c(FALSE, TRUE, TRUE, TRUE),
    x = c(1, 2, 3, 4)
  )
}

f70d_run <- function(f) {
  msgs <- character()
  val <- withCallingHandlers(
    {
      r <- f()
      if (is_tbl_gpu(r)) r <- collect(r)
      r
    },
    warning = function(w) {
      msgs <<- c(msgs, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(val = val, overflow = any(grepl("integer overflow", msgs, fixed = TRUE)))
}

# `pipeline(d, src)`: `src(df)` makes a second table of the same kind as `d`.
f70d_check <- function(pipeline, warns, arrange_by = NULL) {
  df <- f70d_df()
  srcs <- list(
    dplyr = function(x) tibble::as_tibble(x),
    lazy = function(x) as_lazy(tbl_gpu(x)),
    eager = function(x) tbl_gpu(x)
  )
  oracle <- f70d_run(function() pipeline(srcs$dplyr(df), srcs$dplyr))
  expect_identical(oracle$overflow, warns, label = "dplyr's warning (test premise)")
  want <- oracle_normalize(oracle$val, arrange_by)
  for (mode in c("lazy", "eager")) {
    src <- srcs[[mode]]
    got <- f70d_run(function() pipeline(src(df), src))
    expect_identical(got$overflow, warns, label = sprintf("%s cuplyr warning", mode))
    got_tbl <- oracle_normalize(got$val, arrange_by)
    expect_equal(got_tbl, want, label = sprintf("%s cuplyr values", mode))
    expect_identical(lapply(got_tbl, class), lapply(want, class),
                     label = sprintf("%s cuplyr classes", mode))
  }
}

test_that("a window mutate with checked arithmetic, then a filter", {
  skip_if_no_gpu()
  f70d_check(function(d, src) d |> mutate(y = a + dplyr::lead(a)) |> filter(keep),
             warns = TRUE)
})

test_that("pruning keeps a checked expression in a join's right input", {
  skip_if_no_gpu()
  yc <- data.frame(b = c(1L, 2L, 3L), c = c(F70D_M, 1L, 2L))
  f70d_check(function(d, src) {
    d |> semi_join(src(yc) |> mutate(w = c * 2L), by = "b")
  }, warns = TRUE)
  f70d_check(function(d, src) {
    d |>
      left_join(src(yc) |> mutate(w = c * 2L), by = "b", relationship = "many-to-many") |>
      select(a, c)
  }, warns = TRUE, arrange_by = "a")
})

test_that("pruning keeps a checked expression below a barrier in a join input", {
  skip_if_no_gpu()
  yy <- data.frame(b = c(2L, 3L, 4L), z = c(10, 20, 30))
  f70d_check(function(d, src) {
    d |>
      mutate(y = a + b) |>
      arrange(x) |>
      left_join(src(yy), by = "b", relationship = "many-to-many") |>
      select(a, z)
  }, warns = TRUE, arrange_by = "a")
})

test_that("filter_out() and if_any() after a filter are not merged with it", {
  skip_if_no_gpu()
  f70d_check(function(d, src) d |> filter(a < 100L) |> filter_out(a + b > 0L),
             warns = FALSE)
  f70d_check(function(d, src) {
    d |> filter(a < 100L) |> filter(dplyr::if_any(c(a, b), ~ .x + b > 0L))
  }, warns = FALSE)
})

test_that("a null literal over a checked node, and checked nodes over call operands", {
  skip_if_no_gpu()
  f70d_check(function(d, src) d |> mutate(z = (a + b) - NA) |> filter(keep),
             warns = TRUE)
  f70d_check(function(d, src) d |> mutate(y = abs(a) * 2L) |> filter(keep),
             warns = TRUE)
  f70d_check(function(d, src) {
    d |> mutate(y = dplyr::if_else(b > 0L, a, NA) * 2L) |> filter(keep)
  }, warns = TRUE)
})
