# F110: the shortcuts added to the per-call screens, each against the general
# path it stands in for (the way test-perf-F68b-fastpaths.R does for F68b's).

# ---------------------------------------------------------------------------
# ir_has_checked_int(): the closure-based walk it replaced, verbatim
# ---------------------------------------------------------------------------

f110_old_checked_int <- function(ir, schema) {
  if (is.null(ir) || !identical(.subset2(ir, "kind"), "call")) {
    return(FALSE)
  }
  found <- FALSE
  # Visit call node `node` and everything below it; when `typed`, also return
  # its type (as ir_infer_type() would, errors included). Returns early once
  # `found` is set.
  walk <- function(node, typed) {
    args <- .subset2(node, "args")
    n <- length(args)
    types <- if (typed) character(n)
    seen <- logical(n)
    isolated <- logical(n)
    # A node with a null literal operand is NA on every row and never reports
    # an overflow (checked_has_null_literal() in src/expr_eval.hpp); its other
    # operand is still evaluated, so the walk goes on below it.
    if (n == 2L && match(.subset2(node, "op"), c("+", "-", "*"), 0L) > 0L &&
          !ir_lit_is_null(.subset2(args, 1L)) && !ir_lit_is_null(.subset2(args, 2L))) {
      # Is every operand INT32/BOOL8? NA: the call operands decide.
      checked <- NA
      known <- 0L
      for (i in 1:2) {
        a <- .subset2(args, i)
        kind <- .subset2(a, "kind")
        if (kind == "call") next
        if (kind == "col") {
          idx <- match(.subset2(a, "name"), schema$names)
          if (is.na(idx)) {
            checked <- TRUE
            break
          }
          type <- schema$types[idx]
        } else {
          type <- .subset2(a, "type")
          # a bare untyped NA literal adopts its sibling's type
          if (is.null(type)) next
        }
        known <- known + 1L
        if (match(type, c("INT32", "BOOL8"), 0L) == 0L) {
          checked <- FALSE
          break
        }
      }
      if (is.na(checked)) {
        for (i in 1:2) {
          a <- .subset2(args, i)
          if (.subset2(a, "kind") != "call") next
          # Inside a typed walk an error fails this node's own type as well, so
          # it is left to the caught walk above, which counts it as checked.
          type <- if (typed) {
            walk(a, TRUE)
          } else {
            tryCatch(walk(a, TRUE), error = function(e) NA_character_)
          }
          if (found) return(NULL)
          seen[[i]] <- TRUE
          if (typed) types[[i]] <- ir_one_type(type)
          if (is.na(type)) {
            checked <- TRUE
            break
          }
          known <- known + 1L
          if (match(type, c("INT32", "BOOL8"), 0L) == 0L) {
            checked <- FALSE
            break
          }
        }
        if (is.na(checked)) checked <- known > 0L
      }
      if (checked) {
        found <<- TRUE
        return(NULL)
      }
    }
    for (i in seq_len(n)) {
      if (seen[[i]]) next
      a <- .subset2(args, i)
      if (.subset2(a, "kind") == "call") {
        type <- walk(a, typed)
        if (found) return(NULL)
      } else if (typed) {
        type <- ir_infer_type(a, schema)
        isolated[[i]] <- identical(.subset2(a, "kind"), "lit") && ir_is_untyped_na(a)
      }
      if (typed) types[[i]] <- ir_one_type(type)
    }
    if (!typed) return(NULL)
    entry <- ir_call_registry[[.subset2(node, "op")]]
    if (is.null(entry)) {
      stop("Unknown IR op: '", .subset2(node, "op"), "'", call. = FALSE)
    }
    ir_call_type(node, entry, types, isolated)
  }
  walk(ir, FALSE)
  found
}

f110_checked_schemas <- function() {
  sch <- list(names = c("a", "b", "x", "l", "s"),
              types = c("INT32", "INT32", "FLOAT64", "BOOL8", "STRING"))
  list(
    sch,
    list(names = sch$names, types = c("FLOAT64", "INT32", "INT32", "BOOL8", "STRING")),
    list(names = sch$names, types = c("STRING", "INT32", "FLOAT64", "INT32", "INT32")),
    list(names = c("b", "x"), types = c("INT32", "FLOAT64"))
  )
}

test_that("ir_has_checked_int() equals the closure-based walk on random trees", {
  withr::local_seed(110)
  schemas <- f110_checked_schemas()
  leaves <- list(quote(a), quote(b), quote(x), quote(l), quote(s), 1L, 2.5, NA,
                 NA_integer_, TRUE, NA_real_)
  ops <- c("+", "-", "*", "+", "-", "*", "%/%", ">", "&", "/")
  gen <- function(depth) {
    if (depth == 0 || stats::runif(1) < 0.25) return(leaves[[sample(length(leaves), 1)]])
    r <- stats::runif(1)
    if (r < 0.75) return(call(sample(ops, 1), gen(depth - 1), gen(depth - 1)))
    if (r < 0.85) return(call("abs", gen(depth - 1)))
    if (r < 0.9) return(call("-", gen(depth - 1)))
    call("if_else", call(">", gen(depth - 1), 0L), gen(depth - 1), gen(depth - 1))
  }
  n <- 0L
  trues <- 0L
  while (n < 600L) {
    e <- gen(sample(1:6, 1))
    ir <- tryCatch(ir_parse_quo(rlang::new_quosure(e, globalenv()), schemas[[1]]),
                   error = function(err) NULL)
    if (is.null(ir)) next
    for (sch in schemas) {
      expect_identical(ir_has_checked_int(ir, sch), f110_old_checked_int(ir, sch),
                       info = paste(deparse(e), collapse = ""))
      trues <- trues + ir_has_checked_int(ir, sch)
    }
    n <- n + 1L
  }
  # (the grid reaches both answers)
  expect_gt(trues, 100L)
  expect_lt(trues, 600L * 4L - 100L)
})

test_that("the checked-integer flag of a record is worked out on demand, once", {
  sch <- list(names = c("a", "b", "x"), types = c("INT32", "INT32", "FLOAT64"))
  q <- function(e) ir_parse_quo(rlang::new_quosure(e, globalenv()), sch)
  p_int <- make_predicate(q(quote(a + b > 3L)), sch)
  p_dbl <- make_predicate(q(quote(a + x > 3)), sch)
  m_int <- make_mutate_expr("z", q(quote(a * b)), sch)
  m_dbl <- make_mutate_expr("z", q(quote(x * 2 + a)), sch)
  expect_true(record_checked_int(p_int))
  expect_false(record_checked_int(p_dbl))
  expect_true(record_checked_int(m_int))
  expect_false(record_checked_int(m_dbl))
  # asked again: the same answer, from the cell
  expect_true(record_checked_int(p_int))
  expect_false(record_checked_int(m_dbl))
  expect_true(is.environment(.subset2(m_int, "checked_int")))
  # a record carrying a plain logical (built by hand) still reads
  expect_true(record_checked_int(list(checked_int = TRUE)))
  expect_false(record_checked_int(list(checked_int = FALSE)))
  expect_false(record_checked_int(list()))
})

test_that("only record_checked_int() reads a record's checked_int flag", {
  root <- file.path(test_path("..", ".."), "R")
  skip_if_not(dir.exists(root))
  for (f in list.files(root, pattern = "[.]R$", full.names = TRUE)) {
    lines <- readLines(f, warn = FALSE)
    lines <- lines[!grepl("^\\s*#", lines)]
    hit <- grep("[$]checked_int|\"checked_int\"", lines, value = TRUE)
    if (basename(f) == "ast.R") {
      # the constructors and the accessor
      hit <- hit[!grepl("checked_int = new_checked_int_cell|\\.subset2\\(record, \"checked_int\"\\)", hit)]
    }
    expect_length(hit, 0L)
  }
})

test_that("a lazy plan keeps its checked-integer behaviour (F70) with the flag on demand", {
  skip_if_no_gpu()
  d <- data.frame(a = 1:4, b = c(2147483647L, 1L, 2L, 3L), keep = c(TRUE, TRUE, FALSE, TRUE))
  lz <- tbl_gpu(d, lazy = TRUE)
  # the filter may not move below the checked mutate, so the overflow warning
  # sees all four rows, as in dplyr
  expect_same_warnings <- function(expr_gpu, expr_cpu) {
    w_gpu <- character()
    w_cpu <- character()
    withCallingHandlers(expr_gpu, warning = function(w) {
      w_gpu <<- c(w_gpu, conditionMessage(w)); invokeRestart("muffleWarning")
    })
    withCallingHandlers(expr_cpu, warning = function(w) {
      w_cpu <<- c(w_cpu, conditionMessage(w)); invokeRestart("muffleWarning")
    })
    expect_equal(sum(grepl("integer overflow", w_gpu)), 1L)
    expect_equal(sum(grepl("integer overflow", w_cpu)), 1L)
  }
  expect_same_warnings(
    collect(lz |> mutate(y = a + b) |> filter(keep)),
    d |> dplyr::mutate(y = a + b) |> dplyr::filter(keep)
  )
})

# ---------------------------------------------------------------------------
# impure_const_call_text(): ir_impure_none() against the walk alone
# ---------------------------------------------------------------------------

test_that("ir_impure_none() never hides a call the walk reports", {
  env <- new.env()
  env$bump <- function() 1
  env$vec <- c(1, 2)
  cols <- c("x", "y", "g")
  schema <- list(names = cols, types = c("FLOAT64", "FLOAT64", "INT32"))
  exprs <- list(
    quote(mean(x)), quote(x > 0.5), quote(x * 2 + y), quote(sum(x, na.rm = TRUE)),
    quote(x + runif(1)), quote(x + bump()), quote(bump()), quote(bump() + 1),
    quote(abs(1) + x), quote(sqrt(x) + abs(-1)), quote(f(x) + g()), quote(mean(x) + bump()),
    quote(if_else(x > 1, bump(), 1)), quote(x[, 1]), quote(x[1]), quote(.data$x + bump()),
    quote(.data$x > 1), quote(.env$vec[1] + x), quote(dplyr::lag(x)), quote(x + stats::runif(1)),
    quote(~ x + bump()), quote(function(v) bump()), quote(ifelse(x > 1, x, bump())),
    quote(across(c(x, y), mean)), quote(across(c(x, y), ~ .x + bump())),
    quote(across(c(x, y), function(v) v + bump())), quote(across(c(x, y), list(a = mean, b = bump))),
    quote(if_any(c(x, y), ~ .x > bump())), quote(pick(x, y)), quote(rowSums(pick(x, y))),
    quote(n()), quote(n() + x), quote(cur_group_id() + x), quote(x), quote(1), quote("a"),
    quote(case_when(x > 1 ~ bump(), TRUE ~ 0)), quote(case_when(x > 1 ~ 1, TRUE ~ 0)),
    quote((x + 1) * (y - bump())), quote(-x), quote(!is.na(x)), quote(x %in% c(1, 2)),
    quote(x %in% vec), quote(paste0(g, "a")), quote(nchar(letters[1])),
    quote(mean(x) / n()), quote(x + (function() 1)()), quote(`my fn`(x))
  )
  strip <- function(r) if (is.null(r)) NULL else list(unclass(r), attr(r, "expr"))
  # a column named like a lambda formal, with a shadowed `list`: the walk scores
  # `.x` inside the lambda as a formal, so the gate must not skip the dot
  env$list <- function(...) base::list(...)
  schema_x <- list(names = c("g", "a", ".x"), types = c("INT32", "FLOAT64", "FLOAT64"))
  dots_x <- list(rlang::new_quosure(quote(across(a, list(m = ~ mean(.x)))), env))
  with_gate <- impure_const_call_text(dots_x, schema_x)
  without <- local({
    testthat::local_mocked_bindings(ir_impure_none = function(expr, cols) FALSE)
    impure_const_call_text(dots_x, schema_x)
  })
  expect_false(is.null(without))
  expect_identical(strip(with_gate), strip(without))
  expect_false(ir_impure_none(quote(across(a, list(m = ~ mean(.x)))), schema_x$names))
  expect_false(ir_impure_none(quote(across(a, function(.x) mean(.x))), schema_x$names))
  rm("list", envir = env)
  for (e in exprs) {
    dots <- list(rlang::new_quosure(e, env))
    for (nm in list(NULL, "out")) {
      names(dots) <- nm
      for (grouped in c(TRUE, FALSE)) {
        with_gate <- impure_const_call_text(dots, schema, grouped = grouped)
        without <- local({
          testthat::local_mocked_bindings(ir_impure_none = function(expr, cols) FALSE)
          impure_const_call_text(dots, schema, grouped = grouped)
        })
        expect_identical(strip(with_gate), strip(without), info = paste(deparse(e), collapse = ""))
      }
    }
  }
  # the gate itself: TRUE only for calls that read a column all the way down
  expect_true(ir_impure_none(quote(mean(x)), cols))
  expect_true(ir_impure_none(quote(x * 2 + y), cols))
  expect_true(ir_impure_none(quote(x > 0.5), cols))
  expect_true(ir_impure_none(quote(x[, 1]), cols))
  expect_false(ir_impure_none(quote(x + bump()), cols))
  expect_false(ir_impure_none(quote(bump()), cols))
  expect_false(ir_impure_none(quote(x), cols))
  expect_false(ir_impure_none(quote(f(1)), cols))
  expect_false(ir_impure_none(quote(.data$x), cols))
  expect_false(ir_impure_none(quote(x$y + x), cols))
  expect_false(ir_impure_none(quote((function() 1)() + x), cols))
})

# ---------------------------------------------------------------------------
# filter(): the top_n screens and the column-free test
# ---------------------------------------------------------------------------

test_that("top_n_rank_any() is FALSE only for dots the top_n screens leave alone", {
  env <- globalenv()
  schema <- list(names = c("x", "y"), types = c("FLOAT64", "FLOAT64"))
  q <- function(e) rlang::new_quosure(e, env)
  dot_sets <- list(
    list(q(quote(x > 1))), list(q(quote(top_n_rank(2, x)))),
    list(q(quote(x > 1)), q(quote(top_n_rank(2, x)))),
    list(q(quote(dplyr:::top_n_rank(2, x)))), list(q(quote(dplyr::top_n_rank(2, x)))),
    list(q(quote(x))), list(q(1)), list(q(TRUE)), list(),
    list(q(quote(top_n_rank))), list(q(quote((top_n_rank)(2, x)))),
    list(q(quote(f(x)(2))))
  )
  .data <- tbl_gpu(data.frame(x = 1:3, y = 3:1), lazy = TRUE)
  for (dots in dot_sets) {
    gated <- list(top_n_rank_decline(dots, .data), rewrite_top_n_rank_dots(dots, .data))
    open <- local({
      testthat::local_mocked_bindings(top_n_rank_any = function(dots) TRUE)
      list(top_n_rank_decline(dots, .data), rewrite_top_n_rank_dots(dots, .data))
    })
    expect_identical(gated, open)
    if (!top_n_rank_any(dots)) {
      expect_null(open[[1]])
      expect_identical(open[[2]], dots)
    }
  }
})

test_that("quo_reads_column_fast() only answers TRUE when expr_has_column_ref() does", {
  env <- globalenv()
  schema <- list(names = c("x", "y", "g"), types = c("FLOAT64", "FLOAT64", "INT32"))
  exprs <- list(
    quote(x > 1), quote(f(x)), quote(1 + z), quote(z), quote(x), quote(.data$x), quote(.env$x + 1),
    quote(function(v) x), quote(~ x), quote(sum(x[1:3])), quote(x$y), quote(g + y),
    quote(mean(c(x, y))), quote(rlang::.data$x), quote(sin(pi)),
    quote(x(1)), quote(`x`(1)), quote(if (a) b else x), quote(NULL), 1, "x"
  )
  for (e in exprs) {
    quo <- rlang::new_quosure(e, env)
    fast <- quo_reads_column_fast(quo, schema)
    full <- expr_has_column_ref(e, schema)
    if (fast) expect_true(full, info = paste(deparse(e), collapse = ""))
    # the verbs' tests that use it give the same answer with it mocked off
    got <- list(filter_dot_is_column_free(quo, schema), mutate_constant_value(quo, schema))
    off <- local({
      testthat::local_mocked_bindings(quo_reads_column_fast = function(quo, schema) FALSE)
      list(filter_dot_is_column_free(quo, schema), mutate_constant_value(quo, schema))
    })
    expect_identical(got, off, info = paste(deparse(e), collapse = ""))
  }
  expect_false(quo_reads_column_fast(quote(x > 1), schema))
})

test_that("quo_is_null_fast() is rlang::quo_is_null() on the verbs' quosures", {
  env <- globalenv()
  cases <- list(rlang::quo(NULL), rlang::quo(x), rlang::quo(), rlang::new_quosure(NULL, env),
                null_quo, rlang::quo(c(a, b)), rlang::quo(g), rlang::quo(!!NULL))
  for (qq in cases) {
    expect_identical(quo_is_null_fast(qq), rlang::quo_is_null(qq))
  }
})

# ---------------------------------------------------------------------------
# ir_parse_quo(): the factor_labels screen needs a DICTIONARY32 column
# ---------------------------------------------------------------------------

test_that("no factor_labels node is built without a DICTIONARY32 column", {
  schema <- list(names = c("a", "x", "s"), types = c("INT32", "FLOAT64", "STRING"))
  for (e in list(quote(as.character(a)), quote(as.character(s) == "b"), quote(as.integer(a)),
                 quote(as.numeric(x)), quote(s == "\xe9"), quote(paste0(s, "a")))) {
    ir <- tryCatch(ir_parse_quo(rlang::new_quosure(e, globalenv()), schema), error = function(err) NULL)
    if (!is.null(ir)) {
      expect_false(ir_has_factor_labels(ir), info = paste(deparse(e), collapse = ""))
    }
  }
  # with one, the screen still runs (a non-UTF-8 literal against decoded labels)
  fschema <- list(names = c("f"), types = c("DICTIONARY32"),
                  factor_levels = list(f = c("a", "b")))
  ir <- ir_parse_quo(rlang::new_quosure(quote(as.character(f)), globalenv()), fschema)
  expect_true(ir_has_factor_labels(ir))
})

# ---------------------------------------------------------------------------
# the dots screens (R/colwise.R, R/across.R)
# ---------------------------------------------------------------------------

test_that("colwise_dot_has_fn_head() is colwise_has_fn_head() of the dot's expression", {
  env <- globalenv()
  fn_call <- as.call(list(function(v) v, quote(x)))
  nested <- as.call(list(as.name("+"), quote(y), fn_call))
  inner_quo <- rlang::new_quosure(fn_call, env)
  exprs <- list(
    quote(x), quote(x > 1), quote(mean(x)), 1, "a", NULL, quote(x[, 1]), quote(f(g(h(x)))),
    fn_call, nested, quote(`(`(x)), as.call(list(as.name("+"), inner_quo, 1)),
    as.call(list(as.name("+"), rlang::new_quosure(quote(x), env), 1)),
    as.call(list(mean, quote(x))), quote(across(c(x, y), mean)), quote(~ x)
  )
  for (e in exprs) {
    quo <- rlang::new_quosure(e, env)
    expect_identical(colwise_dot_has_fn_head(quo), colwise_has_fn_head(rlang::quo_get_expr(quo)),
                     info = paste(deparse(e), collapse = ""))
  }
  expect_false(colwise_dot_has_fn_head(rlang::quo()))
})

test_that("dots_mention() and expand_plain_dots() keep their results", {
  env <- globalenv()
  q <- function(e) rlang::new_quosure(e, env)
  sets <- list(
    list(), list(q(quote(x > 1))), list(a = q(quote(x + y)), b = q(quote(across(c(x, y), mean)))),
    list(q(quote(x)), q(quote(pick(x)))), list(q(1)), list(q(NULL)), list(q(quote(cur_group_id() + x)))
  )
  ref_expand <- function(dots) {
    n <- length(dots)
    in_names <- names(dots)
    out_quos <- vector("list", n)
    out_names <- character(n)
    for (i in seq_len(n)) {
      out_quos[[i]] <- dots[[i]]
      nm <- in_names[i]
      out_names[[i]] <- if (!is.null(nm) && !is.na(nm) && nzchar(nm)) nm else ""
    }
    names(out_quos) <- out_names
    attr(out_quos, "needs_fallback") <- FALSE
    attr(out_quos, "staged_temps") <- character()
    attr(out_quos, "provenance") <- list(origin = seq_len(n), kind = rep("dot", n))
    out_quos
  }
  ref_mention <- function(dots, fns) {
    for (i in seq_along(dots)) {
      if (any(fns %in% all.names(rlang::quo_get_expr(dots[[i]])))) return(TRUE)
    }
    FALSE
  }
  for (dots in sets) {
    expect_identical(expand_plain_dots(dots), ref_expand(dots))
    expect_identical(dots_mention(dots, dots_screen_fns), ref_mention(dots, dots_screen_fns))
    qs <- rlang::as_quosures(unname(dots), env = env)
    expect_identical(dots_are_plain(qs), !ref_mention(qs, dots_screen_fns))
  }
})

test_that("ir_cols() of a call keeps its names and order", {
  sch <- list(names = c("a", "b", "x"), types = c("INT32", "INT32", "FLOAT64"))
  q <- function(e) ir_parse_quo(rlang::new_quosure(e, globalenv()), sch)
  expect_identical(ir_cols(q(quote(a + b))), c("a", "b"))
  expect_identical(ir_cols(q(quote(a + a))), "a")
  expect_identical(ir_cols(q(quote(x > 1))), "x")
  expect_identical(ir_cols(q(quote(a + b * a))), c("a", "b"))
  expect_identical(ir_cols(q(quote(a))), "a")
  expect_identical(ir_cols(q(quote(sqrt(x)))), "x")
})
