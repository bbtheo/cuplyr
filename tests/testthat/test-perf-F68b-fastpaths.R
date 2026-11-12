# F68b: the fast paths of the per-call R work, each held to the general code it
# short-cuts. The acceptance tests (test-parity-F68b-*.R) pin the verbs' results
# against dplyr; these pin the helpers: a fast path may only answer what the
# general path answers.

internal_fn_strings <- function(fn, head_fn) {
  # string constants a function compares `head_fn` against, via
  # `identical(head_fn, "<s>")`, `head_fn == "<s>"` or `head_fn %in% c("<s>", ...)`
  # (or `%in% <variable of the package namespace>`). `other` holds every
  # call that reads `head_fn` some other way that decides a branch (a `switch()`,
  # `startsWith()`, `match()`, `grepl()`, ...): such a test would escape the set
  out <- character()
  other <- character()
  decides <- c("switch", "match", "match.arg", "pmatch", "charmatch", "startsWith", "endsWith",
               "grepl", "regexpr", "substr", "substring", "sub", "gsub", "is.element", "!=",
               "<", ">", "<=", ">=", "nchar", "tolower", "toupper", "which", "vapply", "sapply",
               "lapply", "Map", "mapply", "Filter", "get", "get0", "[[", "[", "$")
  is_head <- function(a) identical(a, as.name(head_fn))
  walk <- function(e) {
    if (!is.call(e)) return(invisible())
    f <- e[[1]]
    fname <- if (is.symbol(f)) as.character(f) else ""
    args <- as.list(e)[-1]
    reads <- any(vapply(args, function(a) !missing(a) && is_head(a), logical(1)))
    if (reads && identical(fname, "identical") && length(e) == 3L &&
        (is_head(e[[2]]) && is.character(e[[3]]) || is_head(e[[3]]) && is.character(e[[2]]))) {
      out <<- c(out, unlist(Filter(is.character, args)))
    } else if (reads && identical(fname, "==") && length(e) == 3L &&
               (is.character(e[[2]]) || is.character(e[[3]]))) {
      out <<- c(out, unlist(Filter(is.character, args)))
    } else if (reads && identical(fname, "%in%") && is_head(e[[2]])) {
      set <- e[[3]]
      if (is.call(set) && identical(set[[1]], as.name("c"))) {
        out <<- c(out, unlist(Filter(is.character, as.list(set)[-1])))
      } else if (is.symbol(set)) {
        out <<- c(out, get(as.character(set), envir = environment(fn)))
      } else {
        other <<- c(other, "%in% <expression>")
      }
    } else if (reads && fname %in% decides) {
      other <<- c(other, fname)
    }
    for (a in args) if (!missing(a)) walk(a)
  }
  walk(body(fn))
  list(strings = unique(out), other = unique(other))
}

test_that("ir_pre_registry_fns holds every head ir_parse_pre_registry() compares", {
  tested <- internal_fn_strings(ir_parse_pre_registry, "fn_name")
  expect_true(length(tested$strings) > 20L)
  # no other kind of test on the head (a switch(), a pattern match, ...) escapes the set
  expect_identical(tested$other, character())
  expect_setequal(setdiff(tested$strings, ir_pre_registry_fns), character())
  # the names it tests through a vector constant
  expect_true(all(ir_foldable_window_fns %in% ir_pre_registry_fns))
  # the checker itself sees the other ways of testing a name
  probe <- function(fn_name) {
    if (switch(fn_name, a = TRUE, FALSE)) 1
    if (startsWith(fn_name, "x")) 2
    if (fn_name == "b") 3
  }
  seen <- internal_fn_strings(probe, "fn_name")
  expect_identical(seen$strings, "b")
  expect_setequal(seen$other, c("switch", "startsWith"))
})

test_that("plain_selection_loc() is eval_select() for plain names and declines otherwise", {
  d <- data.frame(g = 1, x = 2, y = 3, `a b` = 4, check.names = FALSE)
  nms <- names(d)
  ref <- function(...) {
    q <- rlang::quos(...)
    tidyselect::eval_select(rlang::expr(c(!!!q)), d)
  }
  fast <- function(...) plain_selection_loc(rlang::quos(...), nms)
  same <- function(...) expect_identical(fast(...), ref(...))
  same(g)
  same(g, x)
  same(y, g, x)
  same("x")
  same("g", y)
  same(`a b`)
  same(x, "y")
  # anything else is tidyselect's
  for (declined in list(
    quote(fast(a = x)), quote(fast(g, g)), quote(fast(-x)), quote(fast(c(g, x))),
    quote(fast(starts_with("x"))), quote(fast(1)), quote(fast(nope)), quote(fast("nope")),
    quote(fast()), quote(fast(x:y)), quote(fast(.data$x)), quote(fast(everything())),
    quote(fast(c("g", "x"))), quote(fast(NA_character_))
  )) {
    expect_null(eval(declined))
  }
  # an injected string that carries a name (tidyselect: a rename) or any other
  # attribute is not a plain name, whatever the injected value is
  v <- c(a = "x")
  lk <- c(new = "x", other = "y")
  classed <- structure("x", class = "foo")
  attrd <- structure("x", note = "n")
  expect_null(fast(!!v))
  expect_null(fast(!!lk[1]))
  expect_null(fast(!!classed))
  expect_null(fast(!!attrd))
  expect_null(fast(!!factor("x")))
  expect_null(fast(!!as.Date("2020-01-01")))
  expect_null(fast(!!c(1, 2)))
  expect_null(fast(!!NULL))
  expect_null(fast(!!TRUE))
  expect_null(fast(!!as.name("nope")))
  expect_null(fast(!!list("x")))
  plain <- "x"
  expect_identical(fast(!!plain), ref(!!plain))
  expect_identical(fast(!!as.name("x")), ref(!!as.name("x")))
  # a duplicated schema name, and an empty argument
  expect_null(plain_selection_loc(rlang::quos(x), c("x", "x")))
  g <- function(...) rlang::quos(...)
  expect_identical(plain_selection_loc(g(x, ), nms), ref(x))  # a trailing empty argument is dropped
  expect_null(plain_selection_loc(g(, x), nms))
  # a column wins over an environment variable of the same name
  x <- "y"
  expect_identical(fast(x), ref(x))
})

test_that("IR walks keep their definitions", {
  d <- data.frame(g = 1L, x = 2, y = 3, s = "a", f = factor("u"), stringsAsFactors = FALSE)
  schema <- list(names = names(d), types = c("INT32", "FLOAT64", "FLOAT64", "STRING", "DICTIONARY32"),
                 factor_levels = list(f = "u"))
  exprs <- list(quote(x * 2 + y), quote(x > 0.5), quote(1 + 2), quote(TRUE), quote(!is.na(x)),
                quote(mean(x) > 1), quote(cumsum(x) + y), quote(n() > 1), quote(row_number()),
                quote(if_else(x > 1, y, 0)), quote(s == "a"), quote(x %in% c(1, 2)),
                quote(between(x, 1, 2)), quote(abs(x) + sqrt(y)))
  ref_cols <- function(ir) {
    if (is.null(ir)) return(character())
    switch(ir$kind, col = ir$name, lit = ir$reads %||% character(), vec = character(),
           call = unique(unlist(lapply(ir$args, ref_cols), use.names = FALSE)))
  }
  ref_window <- function(ir) {
    if (is.null(ir)) return(FALSE)
    if (inherits(ir, "cuplyr_ir")) {
      if (!identical(ir$kind, "call")) return(FALSE)
      entry <- ir_call_registry[[ir$op]]
      if (!is.null(entry) && !is.null(entry$window)) return(TRUE)
      return(any(vapply(ir$args, ref_window, logical(1))))
    }
    any(vapply(ir, ref_window, logical(1)))
  }
  for (e in exprs) {
    ir <- ir_parse_quo(rlang::new_quosure(e, globalenv()), schema)
    if (is.null(ir)) next
    expect_identical(ir_cols(ir), ref_cols(ir), info = deparse(e))
    expect_identical(ir_is_const(ir), length(ref_cols(ir)) == 0L, info = deparse(e))
    expect_identical(ir_has_window(ir), ref_window(ir), info = deparse(e))
    expect_identical(ir_has_window(ir$args), ref_window(ir$args), info = deparse(e))
    expect_false(ir_decode_meets_non_utf8(ir), info = deparse(e))
    # the factor rule's shortcut leaves the answer alone on a factor-free tree
    expect_identical(ir_factor_rule(ir, schema) == "none", !"f" %in% ref_cols(ir), info = deparse(e))
  }
  expect_identical(ir_cols(NULL), character())
  expect_true(ir_is_const(NULL))
  expect_false(ir_has_window(NULL))
  # a node with no arguments reads nothing, and has the NULL columns of unique(NULL)
  n_call <- ir_call("n", list())
  expect_identical(ir_cols(n_call), NULL)
  expect_true(ir_is_const(n_call))
  # a literal that reads columns (the rowwise rewrite's) is not constant
  lit <- ir_lit(1L, "INT32")
  lit$reads <- c("x", "y")
  expect_identical(ir_cols(lit), c("x", "y"))
  expect_false(ir_is_const(lit))
  expect_false(ir_is_const(ir_call("+", list(ir_col("x"), lit))))
  # ir_bind() puts 0-based indices on column nodes, whatever the nesting
  bound <- ir_bind(ir_parse_quo(rlang::quo(x * 2 + y), schema), schema)
  expect_identical(bound$args[[1]]$args[[1]]$index, 1L)
  expect_identical(bound$args[[2]]$index, 2L)
})

test_that("ir_infer_type() keeps its signals", {
  schema <- list(names = c("x", "d", "p", "s", "b", "i"),
                 types = c("FLOAT64", "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS", "STRING", "BOOL8", "INT32"))
  type <- function(e) ir_infer_type(ir_parse_quo(rlang::new_quosure(e, globalenv()), schema), schema)
  expect_identical(unname(type(quote(x + 1))), "FLOAT64")
  expect_identical(unname(type(quote(i + 1L))), "INT32")
  expect_identical(unname(type(quote(b & b))), "BOOL8")
  expect_identical(unname(type(quote(d > d))), "BOOL8")
  # a timestamp an op can't take, a STRING a reducer can't take: the CPU's
  expect_null(ir_parse_quo(rlang::quo(d * 2), schema))
  expect_null(ir_parse_quo(rlang::quo(mean(s)), schema))
  expect_null(ir_parse_quo(rlang::quo(s == 1), schema))
  expect_null(ir_parse_quo(rlang::quo(sum(d)), schema))
  expect_error(ir_infer_type(ir_col("nope"), schema), "not found")
  expect_error(ir_infer_type(ir_call("no_such_op", list()), schema), "Unknown IR op")
  # a bare NA adopts its sibling's type
  expect_identical(unname(type(quote(i + NA))), "INT32")
  expect_identical(unname(ir_infer_type(ir_lit(NULL, NULL, na = TRUE), schema)), "BOOL8")
})

test_that("expr_has_column_ref() is its walk", {
  schema <- list(names = c("x", "y", "pkg", "obj", ".data"))
  exprs <- list(
    quote(f(.data)), quote(obj$f(1)), quote(obj[["f"]](1)), quote(f(1)(obj)), quote((pkg)(1)),
    quote(x), quote(z), quote(x + 1), quote(1 + 2), quote(f(z)), quote(f(x, y)),
    quote(.data$x), quote(.data[["q"]]), quote(.env$x), quote(.env[["x"]]), quote(x + .env$y),
    quote(function(a) a + x), quote(function(x) 1), quote(~ x), quote(~ z), quote(g(~ x)),
    quote(base::mean(1)), quote(pkg::f(1)), quote(x[, 1]), quote(m[1, ]), quote(f(z)$x),
    quote(x$y), quote(c(a = 1, b = z)), quote(if (z) 1 else 2), quote(if (x) 1 else 2),
    quote(sum(1:3)), quote(list(z)[[1]]), quote(`(`(x)), quote(f()),
    rlang::new_quosure(quote(x), globalenv()), rlang::new_quosure(quote(z + 1), globalenv()),
    quote(g({{ w }})), 1, "x", NULL
  )
  for (e in exprs) {
    expect_identical(expr_has_column_ref(e, schema), expr_has_column_ref_walk(e, schema),
                     info = paste(deparse(e), collapse = " "))
  }
  # an embedded quosure and a function-object head
  q <- rlang::new_quosure(quote(y), globalenv())
  e1 <- call("f", q)
  e2 <- as.call(list(function(a) a, quote(x)))
  for (e in list(e1, e2)) {
    expect_identical(expr_has_column_ref(e, schema), expr_has_column_ref_walk(e, schema))
  }
})

test_that("colwise_has_fn_head() still finds a function-object head, empty slots included", {
  e <- as.call(list(quote(f), as.call(list(mean, quote(x)))))
  expect_true(colwise_has_fn_head(e))
  expect_true(colwise_has_fn_head(as.call(list(mean, quote(x)))))
  expect_false(colwise_has_fn_head(quote(f(g(x)))))
  expect_false(colwise_has_fn_head(quote(x[, 1])))
  expect_false(colwise_has_fn_head(quote(m[1, ])))
  expect_false(colwise_has_fn_head(quote(x)))
  expect_false(colwise_has_fn_head(1))
  expect_true(colwise_has_fn_head(rlang::new_quosure(as.call(list(mean, quote(x))), globalenv())))
  expect_true(colwise_has_fn_head(call("g", rlang::new_quosure(as.call(list(mean, quote(x))), globalenv()))))
})

test_that("expand_plain_dots() is expand_across_dots() of dots with no across()", {
  ref <- function(dots) {
    # the dot branch of expand_across_dots(): what the loop does for a plain dot
    out_quos <- list(); out_names <- character(); origin <- integer(); kind <- character()
    for (i in seq_along(dots)) {
      nm <- names(dots)[i]
      is_named <- !is.null(nm) && !is.na(nm) && nzchar(nm)
      out_quos <- c(out_quos, list(dots[[i]]))
      out_names <- c(out_names, if (is_named) nm else "")
      origin <- c(origin, i)
      kind <- c(kind, "dot")
    }
    names(out_quos) <- out_names
    attr(out_quos, "needs_fallback") <- FALSE
    attr(out_quos, "staged_temps") <- character()
    attr(out_quos, "provenance") <- list(origin = origin, kind = kind)
    out_quos
  }
  schema <- list(names = c("x", "y"), types = c("FLOAT64", "FLOAT64"))
  for (dots in list(rlang::quos(z = x + 1), rlang::quos(x > 1, y < 2), rlang::quos(a = x, y),
                    rlang::quos(m = mean(x), n = n()), rlang::quos(x))) {
    expect_identical(expand_plain_dots(dots), ref(dots))
    for (verb in c("filter", "mutate", "summarise", "transmute")) {
      expect_identical(expand_across_dots(dots, schema, character(), verb), ref(dots), info = verb)
    }
  }
  # a call that mentions across() takes the general path
  d <- rlang::quos(across(c(x, y), ~ .x + 1))
  expect_false(dots_are_plain(d))
  expect_equal(length(expand_across_dots(d, schema, character(), "mutate")), 2L)
  expect_true(dots_are_plain(rlang::quos(x + 1, mean(y))))
  for (fn in c("across", "if_any", "if_all", "pick", "cur_group", "cur_group_rows", "cur_group_id",
               "cur_data", "cur_data_all", "c_across")) {
    expect_false(dots_are_plain(rlang::new_quosures(list(rlang::new_quosure(call(fn), globalenv())))), info = fn)
    expect_false(dots_are_plain(rlang::new_quosures(list(rlang::new_quosure(call("f", call(fn)), globalenv())))), info = fn)
  }
})

test_that("set_diff()/set_intersect() shortcuts give base R's answers", {
  cases <- list(
    list(character(), c("a", "b")), list(NULL, c("a")), list(c("a", "b"), character()),
    list(c("a", "b", "a"), "a"), list(character(), character()), list(integer(), c("a")),
    list(c(x = "a"), "a"), list(structure(character(), names = character()), "a")
  )
  for (cs in cases) {
    expect_identical(set_diff(cs[[1]], cs[[2]]), unname(setdiff(cs[[1]], cs[[2]])) %||% NULL,
                     ignore_attr = FALSE, info = deparse(cs))
    got <- set_intersect(cs[[1]], cs[[2]])
    exp <- if (is.null(cs[[1]]) || is.null(cs[[2]])) NULL else unname(intersect(cs[[1]], cs[[2]]))
    expect_equal(got, exp, info = deparse(cs))
  }
  expect_identical(set_intersect(character(), c("a", "b")), character())
  expect_identical(set_intersect(NULL, c("a", "b")), NULL)
  expect_identical(set_diff(character(), "a"), character())
  # a different type keeps c()'s promotion
  expect_identical(set_intersect(integer(), c("a")), character())
})

test_that("carry_frame_meta() hands over exactly what dplyr_reconstruct() would", {
  skip_if_no_gpu()
  d <- data.frame(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  attrd <- structure(d, note = "keep me")
  cases <- list(
    bare = tbl_gpu(d), tibble = tbl_gpu(tibble::as_tibble(d)), attrs = tbl_gpu(attrd),
    grouped = dplyr::group_by(tbl_gpu(d), g), lazy = tbl_gpu(d, lazy = TRUE)
  )
  for (nm in names(cases)) {
    from <- cases[[nm]]
    out <- dplyr::filter(from, x > 1)
    # the result keeps the input's attributes, bare class and groups as dplyr's
    expect_identical(attr(out, "note"), attr(from, "note"), info = nm)
    expect_identical(isTRUE(unclass(out)[["bare_df"]]), isTRUE(unclass(from)[["bare_df"]]) &&
                       length(unclass(from)$groups) == 0L, info = nm)
    expect_identical(dplyr::group_vars(out), dplyr::group_vars(from), info = nm)
  }
  # carry_frame_meta() against the general route on the same objects
  for (nm in names(cases)) {
    from <- cases[[nm]]
    out <- new_tbl_gpu(ptr = unclass(from)$ptr, schema = unclass(from)$schema,
                       groups = unclass(from)$groups, exec_mode = unclass(from)$exec_mode)
    general <- if (frame_meta(from)$grouped) out else copy_frame_meta(out, from, attrs = TRUE)
    expect_identical(carry_frame_meta(out, from), general, info = nm)
    expect_identical(carry_frame_meta(out, from, attrs = FALSE),
                     if (frame_meta(from)$grouped) out else copy_frame_meta(out, from, attrs = FALSE),
                     info = nm)
  }
})

test_that("propagate_groups() keeps its answers", {
  node <- ast_filter(ast_source(list(names = c("g", "x"), types = c("INT32", "FLOAT64"))), list())
  schema <- list(names = c("g", "x"), types = c("INT32", "FLOAT64"))
  expect_identical(propagate_groups(node, character(), schema), character())
  expect_identical(propagate_groups(node, "g", schema), "g")
  expect_identical(propagate_groups(node, c("g", "gone"), schema), "g")
  ren <- ast_rename(ast_source(schema), "g", "h")
  expect_identical(propagate_groups(ren, "g", list(names = c("h", "x"), types = schema$types)), "h")
  expect_identical(propagate_groups(ren, character(), list(names = c("h", "x"), types = schema$types)), character())
  summ <- ast_summarise(ast_source(schema), list(), "g", result_groups = character())
  expect_identical(propagate_groups(summ, "g", schema), character())
})

test_that("summarise's simple-dot paths decide as the general code does", {
  schema <- list(
    names = c("i", "d", "x", "b", "s", "f", "p", ".data"),
    types = c("INT32", "TIMESTAMP_DAYS", "FLOAT64", "BOOL8", "STRING", "DICTIONARY32", "TIMESTAMP_MICROSECONDS",
              "FLOAT64"),
    factor_levels = list(f = c("u", "v"))
  )
  fns <- c("sum", "mean", "min", "max", "sd", "var", "median")
  # `fn(x = col)` is the same call with a named argument: summarise_simple_dot()
  # declines it, so every step of it runs the general code
  general_dot <- function(fn, col) {
    rlang::new_quosures(list(m = rlang::new_quosure(as.call(list(as.name(fn), x = as.name(col))),
                                                    globalenv())))
  }
  for (fn in fns) for (col in schema$names) {
    dots <- rlang::new_quosures(list(m = rlang::new_quosure(call(fn, as.name(col)), globalenv())))
    gdots <- general_dot(fn, col)
    simple <- summarise_simple_dot(dots[[1]], schema)
    expect_null(summarise_simple_dot(gdots[[1]], schema), info = paste(fn, col))
    expect_false(is.null(simple) && !(fn == "median" && col == "s"))
    # the shortcut and the general rules agree on whether the dot runs natively
    expect_identical(summarise_dots_parseable(dots, schema), summarise_dots_parseable(gdots, schema),
                     info = paste(fn, col))
    # the one-pass plan says what the two separate steps say
    if (!is.null(simple)) {
      plan <- summarise_simple_plan(dots, schema)
      expect_identical(plan$parseable, summarise_dots_parseable(dots, schema), info = paste(fn, col))
      expect_identical(plan$parseable, summarise_dots_parseable(gdots, schema), info = paste(fn, col))
    }
    # and the aggregation it builds is the general one (an error included)
    if (!is.null(simple) && isTRUE(summarise_dots_parseable(dots, schema))) {
      by_hand <- parse_aggregations(schema, gdots, FALSE, NA_real_)
      expect_identical(summarise_simple_aggregations(dots, schema), by_hand, info = paste(fn, col))
      expect_identical(parse_aggregations(schema, dots, FALSE, NA_real_), by_hand, info = paste(fn, col))
    }
  }
  # shapes that are not simple dots
  not_simple <- list(
    quote(sum(i, na.rm = TRUE)), quote(sum(i + 1L)), quote(dplyr::n()), quote(n()),
    quote(base::sum(i)), quote(first(i)), quote(quantile(x, 0.5)), quote(sum(nope)), quote(sum(1)),
    quote(sum()), quote(sum(i, x)), quote(i), quote(sum(.data$i)), quote(nth(x, 2))
  )
  for (e in not_simple) {
    q <- rlang::new_quosure(e, globalenv())
    expect_null(summarise_simple_dot(q, schema), info = deparse(e))
  }
  expect_null(summarise_simple_aggregations(rlang::quos(m = mean(x), k = first(x)), schema))
  expect_null(summarise_simple_plan(rlang::quos(m = mean(x), k = first(x)), schema))
  # several simple dots: parseable only when every one is
  two <- rlang::quos(a = mean(x), b = sum(f))
  expect_false(summarise_simple_plan(two, schema)$parseable)
  expect_identical(summarise_simple_plan(two, schema)$parseable, summarise_dots_parseable(two, schema))
  two <- rlang::quos(a = mean(x), b = sum(i))
  expect_true(summarise_simple_plan(two, schema)$parseable)
  expect_identical(summarise_simple_plan(two, schema)$parseable, summarise_dots_parseable(two, schema))
  # an unnamed dot is named "" by both routes
  expect_identical(summarise_simple_aggregations(rlang::quos(mean(x)), schema),
                   parse_aggregations(schema, rlang::quos(mean(x)), FALSE, NA_real_))
  expect_identical(
    summarise_simple_aggregations(rlang::quos(a = mean(x), b = sum(i)), schema),
    parse_aggregations(schema, rlang::quos(a = mean(x), b = sum(i)), c(FALSE, FALSE), c(NA_real_, NA_real_))
  )
})

test_that("summarise_is_bare_n() is the n() text test", {
  text_test <- function(q) grepl("^n\\(\\)$", strip_ns_prefix(rlang::quo_text(q)))
  qs <- list(
    quote(n()), quote(dplyr::n()), quote(base::n()), quote(dplyr:::n()), quote(n(1)), quote(n),
    quote(m()), quote(sum(n())), quote(1), "n()", quote(`n`()), quote(f()()), quote((n)()),
    quote(.a::n()), quote(a.b::n()), quote(`my-pkg`::n())
  )
  for (e in qs) {
    q <- rlang::new_quosure(e, globalenv())
    expect_identical(summarise_is_bare_n(q), text_test(q), info = paste(deparse(e), collapse = " "))
  }
  inner <- rlang::new_quosure(quote(n()), globalenv())
  q <- rlang::new_quosure(inner, globalenv())
  expect_identical(summarise_is_bare_n(q), text_test(q))
})

test_that("retry_on_sum_overflow() keeps its warning order and its rerun", {
  # warnings of an attempt that completes are re-raised after it, in order
  got <- character()
  res <- withCallingHandlers(
    retry_on_sum_overflow(function() { warning("a"); warning("b"); "done" }, function() "rerun"),
    warning = function(w) { got <<- c(got, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  expect_identical(res, "done")
  expect_identical(got, c("a", "b"))
  for (cls in c("cuplyr_error_sum_overflow", "cuplyr_summarise_rerun", "custom_class")) {
    got <- character()
    res <- withCallingHandlers(
      retry_on_sum_overflow(function() {
        warning("dropped")
        rlang::abort("overflow", class = cls)
      }, function() { warning("from rerun"); "rerun" }, class = cls),
      warning = function(w) { got <<- c(got, conditionMessage(w)); invokeRestart("muffleWarning") }
    )
    expect_identical(res, "rerun", info = cls)
    expect_identical(got, "from rerun", info = cls)
  }
  # another class of error passes through
  expect_error(
    retry_on_sum_overflow(function() stop("boom"), function() "rerun", class = "cuplyr_summarise_rerun"),
    "boom"
  )
})

test_that("quos(...) captures what enquos(...) does, through a generic", {
  A <- function(...) rlang::enquos(...)
  B <- function(...) rlang::quos(...)
  gA <- function(.data, ...) UseMethod("gA")
  gA.foo <- function(.data, ..., .by = NULL) A(...)
  gB <- function(.data, ...) UseMethod("gB")
  gB.foo <- function(.data, ..., .by = NULL) B(...)
  x <- structure(list(), class = "foo")
  norm <- function(res, fr) {
    out <- lapply(unclass(res), function(q) {
      e <- rlang::quo_get_env(q)
      list(expr = rlang::quo_get_expr(q), env = if (identical(e, fr)) "frame" else "other")
    })
    names(out) <- names(res)
    out
  }
  run <- function(body) {
    fa <- function() { fr <- environment(); norm(eval(body, list(G = gA), fr), fr) }
    fb <- function() { fr <- environment(); norm(eval(body, list(G = gB), fr), fr) }
    expect_identical(fb(), fa(), info = paste(deparse(body), collapse = " "))
  }
  run(quote(G(x, z = a + 1, b)))
  run(quote(G(x, a, )))
  run(quote(G(x, a, .by = g)))
  run(quote({ v <- 5; G(x, a + !!v) }))
  run(quote({ q <- rlang::quo(zz); G(x, !!q, k = q) }))
  run(quote({ l <- rlang::quos(a = x1, b = y1); G(x, !!!l) }))
  run(quote({ nm <- "nn"; G(x, !!nm := a) }))
  run(quote({ w <- function(...) w2(...); w2 <- function(...) G(x, ...); w(a, k = b + 1) }))
  run(quote({ f <- function(v) G(x, v); f() }))
  run(quote(G(x)))
  run(quote(G(x, a = 1, a = 2)))
  run(quote(G(x, !!!list(), !!!NULL)))
  expect_identical(gB(x), gA(x))
})

test_that("a verb on a closure-built dot sees the dot's own environment (quos capture)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 5, 9), y = c(2, 4, 6))
  eg <- tbl_gpu(df, lazy = FALSE)
  mk <- function(cut) function(d) dplyr::filter(d, x > cut)
  out <- mk(4)(eg)
  expect_equal(collect(out)$x, c(5, 9))
  out2 <- mk(8)(eg)
  expect_equal(collect(out2)$x, 9)
  # unquoting and splicing reach the verbs' dots
  v <- 6
  expect_equal(collect(dplyr::mutate(eg, z = x + !!v))$z, c(7, 11, 15))
  exprs <- rlang::exprs(a = x * 2, b = y + 1)
  expect_equal(names(collect(dplyr::mutate(eg, !!!exprs))), c("x", "y", "a", "b"))
  nm <- "total"
  expect_equal(collect(dplyr::summarise(eg, !!nm := sum(x)))$total, 15)
  wrapper <- function(d, col) dplyr::summarise(d, m = mean({{ col }}))
  expect_equal(collect(wrapper(eg, y))$m, 4)
  sel <- function(d, ...) dplyr::select(d, ...)
  expect_equal(names(collect(sel(eg, y, x))), c("y", "x"))
})

test_that("an injected named string renames, as in dplyr (select/rename/relocate, eager and lazy)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6, z = 7:9)
  v <- c(a = "x")
  lk <- c(new = "x", other = "y")
  classed <- structure("y", class = "foo")
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    check(df, function(d) dplyr::select(d, !!v))
    check(df, function(d) dplyr::select(d, !!lk[1]))
    check(df, function(d) dplyr::select(d, !!lk[[1]]))
    check(df, function(d) dplyr::select(d, !!lk))
    check(df, function(d) dplyr::select(d, z, !!v))
    check(df, function(d) dplyr::select(d, !!v, -y))
    check(df, function(d) dplyr::relocate(d, !!v))
    check(df, function(d) dplyr::rename(d, !!lk))
    check(df, function(d) dplyr::select(dplyr::group_by(d, x), !!lk[2]))
  }
  # the classed string: whatever dplyr does with it, the verb does too
  dplyr_res <- tryCatch(names(dplyr::select(df, !!classed)), error = function(e) conditionMessage(e))
  gpu_res <- tryCatch(names(collect(dplyr::select(tbl_gpu(df), !!classed))), error = function(e) conditionMessage(e))
  expect_identical(gpu_res, dplyr_res)
})

test_that("the dots pre-check agrees with the general screens on injected values", {
  # `!!` can put anything in a dot: a function object as a call head, a string
  # head, a quosure holding across(), a value with attributes. dots_mention()
  # answers from all.names(); with it forced to TRUE the general walkers run,
  # and must find nothing more than the pre-check let through.
  schema <- list(names = c("x", "y", "across"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  inner <- rlang::new_quosure(quote(across(c(x, y), sqrt)), globalenv())
  shapes <- list(
    as.call(list(dplyr::across, quote(c(x, y)), sqrt)),
    as.call(list(dplyr::pick, quote(x))),
    as.call(list("across", quote(c(x, y)), quote(sqrt))),
    as.call(list(quote(dplyr::across), quote(c(x, y)), quote(sqrt))),
    call("f", inner),
    call("f", "across"),
    call("f", structure(1, class = "foo", note = quote(across(x)))),
    call("f", list(quote(across(x)))),
    call("f", c(a = 1)),
    quote(cur_group_id()), call("+", quote(x), as.call(list("cur_group_id"))),
    as.call(list(dplyr::cur_group_id)),
    quote(x + y), quote(across), quote(f(across)),
    quote(x + across), call("f", quote(.data$across))
  )
  general <- function(expr) {
    dots <- rlang::new_quosures(list(z = rlang::new_quosure(expr, globalenv())))
    list(
      expand = tryCatch(expand_across_dots(dots, schema, character(), "mutate"), error = function(e) conditionMessage(e)),
      ctx = tryCatch(rewrite_context_dots(dots, character()), error = function(e) conditionMessage(e)),
      fallback = tryCatch(dots_need_fallback(dots), error = function(e) conditionMessage(e)),
      fallback_s = tryCatch(dots_need_fallback(dots, TRUE), error = function(e) conditionMessage(e)),
      plain = dots_are_plain(dots),
      dots = dots
    )
  }
  for (e in shapes) {
    fast <- general(e)
    forced <- testthat::with_mocked_bindings(general(e), dots_mention = function(dots, fns) TRUE)
    info <- paste(deparse(e), collapse = " ")
    for (field in c("expand", "ctx", "fallback", "fallback_s")) {
      expect_identical(fast[[field]], forced[[field]], info = paste(field, info))
    }
    # a dot the pre-check calls plain is one every screen leaves alone
    if (fast$plain) {
      expect_identical(forced$expand, expand_plain_dots(fast$dots), info = info)
      expect_identical(forced$ctx, fast$dots, info = info)
      expect_false(forced$fallback, info = info)
    }
  }
})

test_that("dots named like enquos()'s own arguments bind as in dplyr, verb by verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(2, 1, 3), g = c("a", "b", "a"))
  g <- tbl_gpu(df)
  msg <- function(expr) tryCatch({ force(expr); NA_character_ }, error = function(e) conditionMessage(e))
  # dplyr captures arrange()'s and group_by()'s dots with enquos(), and the
  # selection verbs' through tidyselect's enquos(): `.homonyms` is its argument
  for (verb in list(dplyr::arrange, dplyr::group_by, dplyr::select, dplyr::rename, dplyr::relocate)) {
    expect_false(is.na(msg(verb(df, .homonyms = x))))
    expect_identical(msg(verb(g, .homonyms = x)), msg(verb(df, .homonyms = x)))
  }
  # mutate()/summarise()/filter() read them through dplyr_quosures() (quos()): a plain name
  expect_identical(names(collect(dplyr::mutate(g, .homonyms = x))), names(dplyr::mutate(df, .homonyms = x)))
  expect_identical(names(collect(dplyr::summarise(g, .homonyms = sum(x)))),
                   names(dplyr::summarise(df, .homonyms = sum(x))))
  expect_identical(collect(dplyr::mutate(g, .ignore_null = x * 2)),
                   dplyr::mutate(df, .ignore_null = x * 2))
  # transmute() is the exception: dplyr's check_transmute_args() captures its
  # dots with enquos(), so there the name is enquos()'s argument (an error)
  lz <- tbl_gpu(df, lazy = TRUE)
  for (dot in c(".homonyms", ".check_assign", ".ignore_null")) {
    call_df <- rlang::call2(dplyr::transmute, quote(df), !!dot := quote(x + 1))
    expect_false(is.na(msg(eval(call_df))), info = dot)
    for (tb in c("g", "lz")) {
      call_g <- rlang::call2(dplyr::transmute, as.name(tb), !!dot := quote(x + 1))
      expect_identical(msg(eval(call_g)), msg(eval(call_df)), info = paste(dot, tb))
    }
  }
  # filter(): a named input is dplyr's "named input" error, whatever the name
  for (dot in c(".homonyms", ".check_assign", ".ignore_null", "y")) {
    call_df <- rlang::call2(dplyr::filter, quote(df), !!dot := quote(x > 1))
    call_g <- rlang::call2(dplyr::filter, quote(g), !!dot := quote(x > 1))
    err_df <- tryCatch(eval(call_df), error = function(e) e)
    err_g <- tryCatch(eval(call_g), error = function(e) e)
    expect_s3_class(err_df, "error")
    expect_identical(conditionMessage(err_g), conditionMessage(err_df), info = dot)
    expect_identical(class(err_g), class(err_df), info = dot)
  }
  # and lazily, raised at the verb
  expect_identical(msg(dplyr::filter(lz, .homonyms = x > 1)), msg(dplyr::filter(df, .homonyms = x > 1)))
})

test_that("gpu_frame_kind()/gpu_entry_frame() answer as the walk they replace, in every kind of context", {
  ns <- asNamespace("cuplyr")
  # the walk as it was before the global-environment and inline-kind shortcuts
  ref_kind <- function(p, ns) {
    env <- environment(sys.function(p))
    frames <- NULL
    repeat {
      if (is.null(env)) return(0L)
      if (identical(env, ns)) return(1L)
      if (identical(env, .BaseNamespaceEnv)) return(2L)
      if (is.null(frames)) frames <- sys.frames()
      q <- 0L
      for (j in rev(seq_len(p - 1L))) {
        if (identical(frames[[j]], env)) { q <- j; break }
      }
      if (q == 0L) return(0L)
      p <- q
      env <- environment(sys.function(p))
    }
  }
  ref_entry <- function(n) {
    parents <- sys.parents()
    entry <- n
    i <- n
    repeat {
      p <- parents[[i]]
      if (p <= 0L || p >= i) return(entry)
      kind <- ref_kind(p, ns)
      if (kind == 0L) return(entry)
      if (kind == 1L) entry <- p
      i <- p
    }
  }
  # at the top of a stack: the kind of every frame below, and the entry frame
  # of the top frame, by both (everything it needs is passed in, so the probe
  # can live in any environment)
  ctx <- list(ns = ns, ref_kind = ref_kind, ref_entry = ref_entry,
              kind = gpu_frame_kind, entry = gpu_entry_frame)
  probe <- function(ctx) {
    n <- sys.nframe()
    kinds <- vapply(seq_len(n - 1L), function(p) ctx$kind(p, ctx$ns), integer(1))
    ref <- vapply(seq_len(n - 1L), function(p) ctx$ref_kind(p, ctx$ns), integer(1))
    list(same_kinds = identical(kinds, ref), entry = ctx$entry(n), ref_entry = ctx$ref_entry(n))
  }
  check <- function(res, info) {
    expect_true(res$same_kinds, info = info)
    expect_identical(res$entry, res$ref_entry, info = info)
  }
  # user code in the test's environment, through plain calls
  check((function() probe(ctx))(), "user closure")
  # a package function between, and a base closure looked through
  in_ns <- function(f) f()
  environment(in_ns) <- ns
  check(in_ns(function() probe(ctx)), "namespace function calling a user closure")
  check(lapply(1, function(i) in_ns(function() probe(ctx)))[[1]], "lapply over a namespace call")
  check(in_ns(function() lapply(1, function(i) probe(ctx))[[1]]), "namespace call over lapply")
  # a closure made inside a running function (the maker's frame is running)
  maker <- function() { cb <- function() in_ns(function() probe(ctx)); cb() }
  check(maker(), "closure of a running function")
  # top-level code evaluated in the global environment
  global_probe <- probe
  environment(global_probe) <- globalenv()
  global_in_ns <- function(f) in_ns(f)
  environment(global_in_ns) <- globalenv()
  assign("..f68b_probe", global_probe, envir = globalenv())
  assign("..f68b_ctx", ctx, envir = globalenv())
  assign("..f68b_in_ns", in_ns, envir = globalenv())
  withr::defer(rm(list = c("..f68b_probe", "..f68b_ctx", "..f68b_in_ns"), envir = globalenv()))
  check(eval(quote(..f68b_probe(..f68b_ctx)), globalenv()), "global closure, eval() in the global environment")
  check(evalq(..f68b_in_ns(function() ..f68b_probe(..f68b_ctx)), globalenv()), "global closure under a namespace call")
  check(local({ f <- function() probe(ctx); f() }), "closure made in local()")
  check(do.call(probe, list(ctx)), "do.call")
  check(with(list(), probe(ctx)), "with()")
})

test_that("ir_cols() is the unique() of every level's columns", {
  ref_cols <- function(ir) {
    if (is.null(ir)) return(character())
    switch(ir$kind, col = ir$name, lit = ir$reads %||% character(), vec = character(),
           call = unique(unlist(lapply(ir$args, ref_cols), use.names = FALSE)))
  }
  set.seed(7)
  gen <- function(depth) {
    pick <- if (depth <= 0) sample(c("col", "lit", "litreads", "vec", "null"), 1)
            else sample(c("col", "lit", "litreads", "vec", "call", "call", "call0", "null"), 1)
    switch(pick,
      col = ir_col(sample(c("a", "b", "c"), 1)),
      lit = ir_lit(1, "FLOAT64"),
      litreads = { l <- ir_lit(1L, "INT32"); l$reads <- sample(c("a", "d"), sample(1:2, 1)); l },
      vec = ir_vec(1:3, "INT32", NULL, "v"),
      call0 = ir_call("n", list()),
      null = NULL,
      call = ir_call("+", lapply(seq_len(sample(1:3, 1)), function(i) gen(depth - 1)))
    )
  }
  for (i in 1:300) {
    ir <- gen(4)
    expect_identical(ir_cols(ir), ref_cols(ir), info = paste(i, ir_deparse(ir)))
  }
  # a call whose arguments read nothing is character(), not NULL
  expect_identical(ir_cols(ir_call("+", list(ir_lit(1, "FLOAT64"), ir_lit(2, "FLOAT64")))), character())
  expect_identical(ir_cols(ir_call("+", list(ir_call("n", list()), ir_lit(2, "FLOAT64")))), character())
  expect_null(ir_cols(ir_call("n", list())))
})

test_that("propagate_schema_meta() with nothing to carry is the general answer", {
  general <- function(node, input_schema, new_schema) {
    new_schema$factor_levels <- propagate_factor_levels(node, input_schema, new_schema$names)
    new_schema$tz <- propagate_tz(node, input_schema, new_schema)
    new_schema
  }
  base <- list(names = c("g", "x", "t", "f"),
               types = c(g = "INT32", x = "FLOAT64", t = "TIMESTAMP_MICROSECONDS", f = "DICTIONARY32"))
  inputs <- list(
    plain = list(names = c("g", "x"), types = c(g = "INT32", x = "FLOAT64")),
    plain_null_slots = list(names = c("g", "x"), types = c(g = "INT32", x = "FLOAT64"),
                            factor_levels = NULL, tz = NULL),
    empty_levels = list(names = c("g", "x"), types = c(g = "INT32", x = "FLOAT64"),
                        factor_levels = list()),
    levels = base[c("names", "types")] |> c(list(factor_levels = list(f = c("u", "v")))),
    zones = base[c("names", "types")] |> c(list(tz = c(t = "Europe/Paris"))),
    both = base[c("names", "types")] |> c(list(factor_levels = list(f = "u"), tz = c(t = ""))),
    empty_tz = list(names = c("g", "x"), types = c(g = "INT32", x = "FLOAT64"), tz = character())
  )
  mut <- function(out, type) {
    list(output_col = out, ir = ir_col("x"), input_cols = "x", output_type = type)
  }
  for (nm in names(inputs)) {
    sch <- inputs[[nm]]
    src <- ast_source(sch)
    keep <- sch$names[1]
    nodes <- list(
      filter = ast_filter(src, list()), arrange = ast_arrange(src, list()),
      select = ast_select(src, keep), rename = ast_rename(src, keep, "gg"),
      mutate_dbl = ast_mutate(src, list(mut("z", "FLOAT64"))),
      mutate_posixct = ast_mutate(src, list(mut("z", "FLOAT64"), mut("p", "TIMESTAMP_MICROSECONDS"))),
      mutate_copy = ast_mutate(src, list(mut("z", unname(sch$types[[1]])))),
      summarise = ast_summarise(src, list(), keep, character()),
      distinct = ast_distinct(src, keep), barrier = ast_node("barrier", input = src)
    )
    for (k in names(nodes)) {
      node <- nodes[[k]]
      new <- tryCatch(infer_schema_from(node, sch), error = function(e) NULL)
      if (is.null(new)) next
      expect_identical(propagate_schema_meta(node, sch, new), general(node, sch, new),
                       info = paste(nm, k))
    }
  }
  # a new schema that already holds metadata takes the general path
  sch <- inputs$plain
  new <- list(names = "g", types = c(g = "INT32"), factor_levels = list(g = "a"))
  node <- ast_select(ast_source(sch), "g")
  expect_identical(propagate_schema_meta(node, sch, new), general(node, sch, new))
})

test_that("push_op() hands the frame metadata over as new_tbl_gpu() + carry_frame_meta() would", {
  skip_if_no_gpu()
  d <- data.frame(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  cases <- list(
    bare = tbl_gpu(d), tibble = tbl_gpu(tibble::as_tibble(d)), attrs = tbl_gpu(structure(d, note = "keep me")),
    grouped = dplyr::group_by(tbl_gpu(d), g), lazy = tbl_gpu(d, lazy = TRUE),
    lazy_pending = dplyr::filter(tbl_gpu(d, lazy = TRUE), x > 0),
    rowwise = dplyr::rowwise(tbl_gpu(d))
  )
  for (nm in names(cases)) {
    from <- cases[[nm]]
    for (verb in c("mutate", "arrange", "select")) {
      out <- switch(verb, mutate = dplyr::mutate(from, z = x + 1), arrange = dplyr::arrange(from, x),
                    select = dplyr::select(from, x, g))
      u <- unclass(out)
      # the table as the general route builds it: no bare_df from new_tbl_gpu(), the
      # class and attributes handed over by carry_frame_meta()
      ref <- do.call(new_tbl_gpu, u[intersect(names(u), c("ptr", "schema", "lazy_ops", "groups", "exec_mode",
                                                            "group_drop", "nrow", "fixed_keys", "unspecified",
                                                            "rowwise"))])
      ref <- carry_frame_meta(ref, from, attrs = TRUE)
      expect_identical(names(u), names(unclass(ref)), info = paste(nm, verb))
      expect_identical(isTRUE(u$bare_df), isTRUE(unclass(ref)$bare_df), info = paste(nm, verb))
      expect_identical(attributes(out)[setdiff(names(attributes(out)), c("names", "class"))],
                       attributes(ref)[setdiff(names(attributes(ref)), c("names", "class"))],
                       info = paste(nm, verb))
    }
  }
})

test_that("the IR walks that take a leaf inline agree with their recursive definitions", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "INT32", "BOOL8"))
  # the walks as they were: one recursive call per node, leaves included
  ref_bind <- function(ir) {
    if (is.null(ir)) return(ir)
    kind <- ir$kind
    if (kind == "col") { ir$index <- match(ir$name, schema$names) - 1L; return(ir) }
    if (kind == "lit" || kind == "vec") return(ir)
    if (identical(ir$op, "factor_codes")) return(ref_bind(ir$args[[1]]))
    args <- ir$args
    for (i in seq_along(args)) args[i] <- list(ref_bind(.subset2(args, i)))
    ir$args <- args
    ir
  }
  meta_nodes <- function(node) Filter(function(m) is.list(m) && !is.null(m$kind), node$meta)
  ref_window <- function(ir) {
    if (is.null(ir)) return(FALSE)
    if (inherits(ir, "cuplyr_ir")) {
      if (!identical(ir$kind, "call")) return(FALSE)
      entry <- ir_call_registry[[ir$op]]
      if (!is.null(entry) && !is.null(entry$window)) return(TRUE)
      return(any(vapply(ir$args, ref_window, logical(1))))
    }
    any(vapply(ir, ref_window, logical(1)))
  }
  ref_labels <- function(node) {
    if (inherits(node, "cuplyr_ir")) {
      if (!identical(node$kind, "call")) return(FALSE)
      if (identical(node$op, "factor_labels")) return(TRUE)
      return(any(vapply(c(node$args, meta_nodes(node)), ref_labels, logical(1))))
    }
    if (is.list(node)) return(any(vapply(node, ref_labels, logical(1))))
    FALSE
  }
  ref_dict <- function(node) {
    if (is.null(node)) return(FALSE)
    if (identical(node$kind, "call")) return(any(vapply(c(node$args, meta_nodes(node)), ref_dict, logical(1))))
    (identical(node$kind, "lit") || identical(node$kind, "vec")) && identical(unname(node$type), "DICTIONARY32")
  }
  set.seed(11)
  gen <- function(depth) {
    pick <- if (depth <= 0) sample(c("col", "lit", "dict", "vec", "null"), 1)
            else sample(c("col", "lit", "dict", "call", "call", "call", "win", "labels", "codes", "meta", "null"), 1)
    kids <- function() lapply(seq_len(sample(1:3, 1)), function(i) gen(depth - 1))
    switch(pick,
      col = ir_col(sample(schema$names, 1)),
      lit = ir_lit(1, "FLOAT64"),
      dict = ir_lit(1L, "DICTIONARY32"),
      vec = ir_vec(1:3, "INT32", NULL, "v"),
      null = NULL,
      call = ir_call(sample(c("+", "*", "-", "=="), 1), kids()),
      win = ir_call("cumsum", kids()[1]),
      labels = ir_call("factor_labels", kids()),
      codes = ir_call("factor_codes", list(ir_col("a"))),
      meta = ir_call("lag", kids()[1], meta = list(n = 1L, default = gen(depth - 1)))
    )
  }
  for (i in 1:400) {
    ir <- gen(4)
    info <- paste(i, if (is.null(ir)) "NULL" else ir_deparse(ir))
    expect_identical(ir_has_window(ir), ref_window(ir), info = info)
    expect_identical(ir_has_factor_labels(ir), ref_labels(ir), info = info)
    expect_identical(ir_has_dictionary_value(ir), ref_dict(ir), info = info)
    if (!is.null(ir) && !any(vapply(list(ir), ref_window, logical(1)))) {
      expect_identical(ir_bind(ir, schema), ref_bind(ir), info = info)
    }
  }
  # a plain list of nodes, as ir_has_window() takes
  lst <- list(ir_col("a"), ir_call("cumsum", list(ir_col("a"))))
  expect_identical(ir_has_window(lst), ref_window(lst))
  expect_identical(ir_has_factor_labels(list(ir_call("factor_labels", list()))), TRUE)
  expect_error(ir_bind(ir_call("+", list(ir_col("nope"))), schema), "Column 'nope' not found")
})

test_that("values injected with !! reach the literal and symbol fast paths as dplyr sees them", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3), g = c("a", "b", "a"), i = 1:3)
  named <- c(a = 1)
  classed <- structure(1, class = "foo")
  noted <- structure(1, note = "n")
  pipelines <- list(
    named_num = function(d) dplyr::mutate(d, z = x + !!named),
    classed_num = function(d) dplyr::mutate(d, z = x + !!classed),
    attr_num = function(d) dplyr::mutate(d, z = x + !!noted),
    factor1 = function(d) dplyr::mutate(d, z = !!factor("u")),
    date1 = function(d) dplyr::mutate(d, z = !!as.Date("2020-01-02")),
    posixct1 = function(d) dplyr::mutate(d, z = !!as.POSIXct("2020-01-02 03:04:05", tz = "UTC")),
    named_chr = function(d) dplyr::mutate(d, z = paste0(g, !!c(a = "k"))),
    named_filter = function(d) dplyr::filter(d, x > !!named),
    classed_filter = function(d) dplyr::filter(d, x > !!classed),
    named_lgl_filter = function(d) dplyr::filter(d, !!c(a = TRUE)),
    sym_arith = function(d) dplyr::mutate(d, z = !!as.name("x") * 2),
    sym_sum = function(d) dplyr::summarise(d, s = sum(!!as.name("x"))),
    sym_mean_grouped = function(d) dplyr::summarise(dplyr::group_by(d, g), m = mean(!!as.name("x")))
  )
  for (nm in names(pipelines)) {
    expect_same_as_dplyr(df, pipelines[[nm]], arrange_by = NULL) |> suppressMessages()
    expect_same_as_dplyr_lazy(df, pipelines[[nm]], arrange_by = NULL) |> suppressMessages()
  }
})
