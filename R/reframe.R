# reframe() (F22)
#
# dplyr's reframe() returns any number of rows per group. The common shapes
# are native without list-columns, in two steps that mirror dplyr's own
# summarise_cols() / summarise_build():
#
#   1. WIDE: every size-k expression is rewritten into k scalar aggregations
#      (`quantile(x, c(.1, .9))` -> `quantile(x, .1)`, `quantile(x, .9)`;
#      `range(x)` -> `min(x)`, `max(x)`), and the ordinary summarise_impl()
#      computes one row per group (keys, then the scalars).
#   2. LONG: one `ast_reframe` node reshapes that table on the device
#      (gpu_reframe_expand(), src/ops_reframe.cpp): keys and size-1 results
#      are repeated k times, the k columns of a size-k result are interleaved,
#      literal companions (`prob = c(.25, .5)`) are tiled.
#
# This is only right where the size-k call is an element-wise bundle of
# scalar aggregations (`quantile(x, p)[j] == quantile(x, p[j])`, `range(x) ==
# c(min(x), max(x))`, NA handling, na.rm, empty-slice values and the NA error
# included), and where summarise() computes every scalar exactly as R does, so
# the plan below is a positive allowlist of call and argument shapes (see "The
# native allowlist"). Everything else, and anything the plan is unsure about,
# is dplyr's on the CPU (the whole-call fallback), decided before any GPU work.

# TRUE while the wide step of a native reframe() runs (see
# notify_materialization()).
reframe_state <- new.env(parent = emptyenv())

# Internal: evaluate `expr` with the wide-step flag set.
# @keywords internal
with_reframe_wide <- function(expr) {
  old <- reframe_state$wide
  reframe_state$wide <- TRUE
  on.exit(reframe_state$wide <- old, add = TRUE)
  expr
}

#' Summarise each group to any number of rows
#'
#' [dplyr::reframe()] for a `tbl_gpu`. The result is always ungrouped:
#' `group_by()` keys come out sorted, `.by` keys in first-appearance order, and
#' size-1 results are recycled to each group's size.
#'
#' @section Native shapes:
#' The call runs on the GPU when every expression is named and is one of:
#' * `quantile(x, probs)` (`stats::`) with `probs` a column-free constant in
#'   \[0, 1\], a literal `na.rm`, `type` absent or 7;
#' * `range(x)` (`base::`) with a literal `na.rm`;
#' * `n()`, or `sum()`, `mean()`, `min()`, `max()`, `sd()`, `var()`, `median()`
#'   of one argument with a literal `na.rm`;
#' * a column-free constant vector (`c(0.25, 0.75)`, `"lo"`), recycled.
#'
#' The argument `x` must be an input column, or arithmetic (`+ - * / %% %/%`,
#' `abs()`, `exp()`; `%%` and `%/%` take an integer divisor or a whole-number
#' double literal), comparisons, `& | !` and `is.na()` over integer, double
#' and logical columns and literals, with no window or aggregate call inside.
#' Comparisons, `& | !` and `is.na()` take only integer and logical columns,
#' integer arithmetic on them, and literals.
#' `quantile()` takes integers and doubles, `median()` only doubles; `min()`,
#' `max()` and `range()` also take a bare Date or POSIXct column, and a bare
#' string column under a C collation. Sizes above 1 must agree, and no
#' expression may read an earlier one's result.
#'
#' Anything else runs dplyr's `reframe()` on the CPU (see [cuplyr_fallback_notify()]).
#' One divergence remains on both paths: `quantile()`'s element names (`"10%"`)
#' are dropped.
#'
#' @param .data A `tbl_gpu`.
#' @param ... Name-value pairs, as in [dplyr::reframe()].
#' @param .by Optional per-operation grouping, as in [dplyr::reframe()].
#' @return An ungrouped `tbl_gpu` (lazy if `.data` is lazy).
#' @export
#' @importFrom dplyr reframe
reframe.tbl_gpu <- function(.data, ..., .by = NULL) {
  if (is_rowwise_gpu(.data)) {
    return(rowwise_cpu(.data, "reframe"))
  }
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(.by)
  caller_env <- rlang::caller_env()
  cpu <- function() {
    gpu_fallback("reframe", .data, function(tbl) {
      rlang::inject(dplyr::reframe(tbl, !!!dots, .by = !!by_quo))
    })
  }

  schema <- current_schema(.data)
  by_given <- !rlang::quo_is_null(by_quo)
  if (!reframe_input_is_native(.data, schema) || (by_given && length(.data$groups) > 0L)) {
    return(cpu())
  }
  plan <- reframe_native_plan(dots, schema)
  if (is.null(plan)) {
    return(cpu())
  }

  by_cols <- if (by_given) {
    tryCatch(resolve_by(.data, by_quo, ".by"), error = function(e) NULL)
  } else {
    .data$groups
  }
  if (is.null(by_cols) || (by_given && length(by_cols) == 0L) ||
      any(names(dots) %in% by_cols)) {
    return(cpu())
  }

  # Step 1: the wide table, one row per group. `.groups = "drop"` (or `.by`)
  # gives the ungrouped, key-ordered result reframe() always returns.
  # An INT32 sum() can leave R's integer range, where dplyr gives a double.
  # A lazy plan would abort at collect(), so such a call runs now, eagerly
  # (as summarise_run_now() does for integer min()/max()), and goes back to
  # lazy at the end.
  run_now <- identical(.data$exec_mode, "lazy") && length(plan$int32_sum) > 0L
  input <- .data
  if (run_now) {
    if (has_pending_ops(.data)) {
      cuplyr_fallback_notify("reframe", paste0(
        "integer sum() (result type depends on the data): ",
        paste(plan$int32_sum, collapse = ", ")
      ))
    }
    input <- compute(.data)
    input$exec_mode <- "eager"
  }
  wide <- with_reframe_wide(summarise_impl(
    input, plan$wide_dots, if (by_given) NULL else "drop", by_quo, by_cols, by_given, caller_env
  ))

  # The wide table is computed; the reshape below stays pending again.
  if (run_now) {
    wide$exec_mode <- "lazy"
  }

  # Step 2: reshape. With one row per group and nothing to tile, the wide
  # table already is the answer (the names are the user's).
  if (plan$k == 1L && !any(vapply(plan$outputs, function(o) o$kind == "literal", logical(1)))) {
    result <- wide
  } else {
    keys <- lapply(by_cols, function(key) list(name = key, kind = "rep", src = key))
    result <- push_op(wide, ast_reframe(input_node(wide), c(keys, plan$outputs), plan$k,
                                        plan$first_k_name))
  }
  result
}

# Internal: can reframe() of this input run natively at all? Fixed group keys
# (slice(.preserve = TRUE)) and zero-column or oddly named tables are the
# CPU's, as for summarise().
# @keywords internal
reframe_input_is_native <- function(.data, schema) {
  nms <- schema$names
  is.null(fixed_keys_of(.data)) && length(nms) > 0L && !anyNA(nms) &&
    all(nzchar(nms)) && !anyDuplicated(nms)
}

# Functions a column-free constant (`c(.25, .5)`, `1:3`, `rep("a", 2)`) may
# call: evaluating one once must equal dplyr's evaluation per group.
reframe_constant_fns <- c("c", ":", "+", "-", "*", "/", "(", "rep", "rep_len", "seq",
                          "seq_len", "rev", "as.integer", "as.numeric", "as.double",
                          "as.character", "paste", "paste0")

# Internal: does the call head of `expr` mean `pkg::fn` when evaluated in
# `env`? A bare `range` counts only if the lookup (functions only, like R's)
# finds base's own, so a user's `range <- function(x, ...) ...` is not
# rewritten; `pkg::fn` is that function by construction.
# @keywords internal
reframe_head_is <- function(expr, env, pkg, fn) {
  head <- expr[[1]]
  if (is.call(head)) {
    return(length(head) == 3L && identical(head[[1]], as.name("::")) &&
             identical(head[[2]], as.name(pkg)) && identical(head[[3]], as.name(fn)))
  }
  is.symbol(head) && identical(as.character(head), fn) &&
    identical(get0(fn, envir = env, mode = "function", inherits = TRUE),
              getExportedValue(pkg, fn))
}

# Internal: is `expr` a deterministic constant that reads no column? Symbols
# are fine unless they name a column (they resolve in the quosure's environment
# then); calls only to base's `reframe_constant_fns` (not a masked
# redefinition); an injected atomic vector is a constant.
# @keywords internal
reframe_is_constant_expr <- function(expr, env, col_names) {
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    return(nzchar(nm) && !nm %in% c(col_names, ".data", ".env"))
  }
  if (is.call(expr)) {
    head <- expr[[1]]
    fn <- if (is.call(head) && length(head) == 3L && identical(head[[1]], as.name("::"))) {
      head[[3]]
    } else {
      head
    }
    if (!is.symbol(fn) || !as.character(fn) %in% reframe_constant_fns ||
        !reframe_head_is(expr, env, "base", as.character(fn))) {
      return(FALSE)
    }
    return(all(vapply(as.list(expr)[-1], reframe_is_constant_expr, logical(1), env, col_names)))
  }
  is.atomic(expr) && !is.null(expr)
}

# Internal: the value of a constant expression, or NULL when it can't be
# evaluated or isn't a plain atomic logical/integer/double/character vector
# (no class, names or dim, no NaN, at least one element).
# @keywords internal
reframe_constant_value <- function(expr, env, col_names) {
  if (!reframe_is_constant_expr(expr, env, col_names)) {
    return(NULL)
  }
  value <- tryCatch(rlang::eval_tidy(rlang::new_quosure(expr, env)), error = function(e) NULL)
  if (!is.atomic(value) || is.null(value) || is.object(value) || !is.null(names(value)) ||
      !is.null(dim(value)) ||
      !typeof(value) %in% c("logical", "integer", "double", "character") ||
      length(value) < 1L || (is.double(value) && any(is.nan(value)))) {
    return(NULL)
  }
  value
}

# Internal: `pkg::` stripped call head as a string, or NULL. Only `allowed`
# namespaces count (`stats::quantile`, `base::range`).
# @keywords internal
reframe_call_name <- function(expr, ns) {
  if (!is.call(expr)) {
    return(NULL)
  }
  head <- expr[[1]]
  if (is.call(head) && length(head) == 3L && identical(head[[1]], as.name("::"))) {
    if (!is.symbol(head[[2]]) || !as.character(head[[2]]) %in% ns) {
      return(NULL)
    }
    head <- head[[3]]
  }
  if (is.symbol(head)) as.character(head) else NULL
}

# Call head as a string whatever its namespace, or NULL.
# @keywords internal
reframe_head_name <- function(expr) {
  if (!is.call(expr)) {
    return(NULL)
  }
  head <- expr[[1]]
  if (is.call(head) && length(head) == 3L && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]
  }
  if (is.symbol(head)) as.character(head) else NULL
}

# A literal TRUE/FALSE written in the call, else NULL.
# @keywords internal
reframe_lgl_literal <- function(expr) {
  if (is.logical(expr) && length(expr) == 1L && !is.na(expr)) expr else NULL
}

# Internal: the arguments of `quantile(x, probs, na.rm, names, type)` when the
# call is one the rewrite is exact for, else NULL: `x`, the probabilities (a
# column-free constant in [0, 1], no NA), a literal `na.rm`, an ignorable
# `names`, `type` absent or 7 (cudf's LINEAR is R's type 7).
# @return `list(x, probs, na_rm)` or NULL
# @keywords internal
reframe_parse_quantile <- function(expr, env, col_names) {
  proto <- function(x, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = FALSE,
                    names = TRUE, type = 7) NULL
  given <- names(expr)
  given <- given[!is.na(given) & nzchar(given)]
  if (!all(given %in% names(formals(proto)))) {
    return(NULL)  # digits=, a partial name, ...
  }
  matched <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(matched)) {
    return(NULL)
  }
  args <- as.list(matched)[-1]
  if (is.null(args[["x"]]) || !all(names(args) %in% names(formals(proto)))) {
    return(NULL)
  }
  na_rm <- if (is.null(args[["na.rm"]])) FALSE else reframe_lgl_literal(args[["na.rm"]])
  if (is.null(na_rm)) {
    return(NULL)
  }
  if (!is.null(args[["names"]]) && is.null(reframe_lgl_literal(args[["names"]]))) {
    return(NULL)
  }
  if (!is.null(args[["type"]]) && !(is.numeric(args[["type"]]) && length(args[["type"]]) == 1L &&
                               identical(as.double(args[["type"]]), 7))) {
    return(NULL)
  }
  probs <- if (is.null(args[["probs"]])) {
    c(0, 0.25, 0.5, 0.75, 1)
  } else {
    reframe_constant_value(args[["probs"]], env, col_names)
  }
  if (is.null(probs) || !is.numeric(probs) || anyNA(probs) || any(probs < 0 | probs > 1)) {
    return(NULL)
  }
  list(x = args[["x"]], probs = as.double(probs), na_rm = na_rm)
}

# Internal: the argument of `range(x, na.rm =)` (one data argument, `finite=`
# and the like are not ours), else NULL.
# @return `list(x, na_rm)` or NULL
# @keywords internal
reframe_parse_range <- function(expr) {
  args <- as.list(expr)[-1]
  nms <- names(args) %||% rep("", length(args))
  na_pos <- which(nms == "na.rm")
  rest <- if (length(na_pos) > 0L) nms[-na_pos] else nms
  if (length(na_pos) > 1L || any(rest != "") || length(rest) != 1L) {
    return(NULL)
  }
  na_rm <- if (length(na_pos) == 1L) reframe_lgl_literal(args[[na_pos]]) else FALSE
  if (is.null(na_rm)) {
    return(NULL)
  }
  x <- if (length(na_pos) == 1L) args[-na_pos][[1]] else args[[1]]
  list(x = x, na_rm = na_rm)
}

# --- The native allowlist ------------------------------------------------------
#
# A wide dot is native only in a shape listed here; everything else is dplyr's.
# The list is positive on purpose: "whatever summarise() can parse" made
# reframe() inherit every divergence of the IR, window and summarise kernels
# (string comparisons in byte order, a NaN that compares as a number, an
# integer median() as double, ...) that the CPU path doesn't have.
#
# A wide dot is `n()` or one call `fn(arg, na.rm = <TRUE/FALSE literal>)` with
# `fn` in `reframe_agg_ns` (its head resolving to that namespace's function).
# `arg` is a bare input column, or a computed expression over INT32 / FLOAT64 /
# BOOL8 columns and numeric/logical literals calling only `reframe_arg_fns`
# (no window or aggregate call, so no window kernel runs). Its type must be one
# the aggregation takes natively with dplyr's result type
# (reframe_agg_arg_types()).

# Aggregations a wide dot may be, and the namespace each head must resolve to.
reframe_agg_ns <- c(n = "dplyr", sum = "base", mean = "base", min = "base", max = "base",
                    sd = "stats", var = "stats", median = "stats", quantile = "stats")

# Base functions a computed aggregation argument may call. Left out because
# the GPU disagrees with R: `^` (R's `NA ^ 0` is 1), sqrt()/log() (R warns
# "NaNs produced"), `%%`/`%/%` with a double divisor column (zero and infinite
# divisors), unary `+` (the IR has none).
reframe_arg_fns <- c("(", "+", "-", "*", "/", "%%", "%/%", "abs", "exp",
                     "==", "!=", "<", "<=", ">", ">=", "&", "|", "!", "is.na")

# Internal: the argument types `fn` takes natively. median() of an INT32/BOOL8
# is R's middle element (integer/logical) for an odd-sized group, so only a
# double is native; quantile() of a logical is dplyr's; a timestamp, or a
# string (bytes order, so only under a C collation), only as a bare min()/max()
# argument.
# @keywords internal
reframe_agg_arg_types <- function(fn, bare) {
  num <- c("INT32", "FLOAT64", "BOOL8")
  switch(fn,
    quantile = c("INT32", "FLOAT64"),
    median = "FLOAT64",
    sum = , mean = , sd = , var = num,
    min = , max = c(num, if (bare) {
      c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS", if (session_collation_is_c()) "STRING")
    }),
    character()
  )
}

# Internal: `list(fn, arg)` of a wide dot in an allowlisted call shape (`arg`
# NULL for n()), else NULL. `verified` dots come from the quantile()/range()
# rewrite, whose head was resolved already (the rewrite writes `quantile`,
# `min`, `max` bare); the user's own dots must resolve here.
# @keywords internal
reframe_agg_parts <- function(expr, env, verified) {
  fn <- reframe_head_name(expr)
  if (is.null(fn) || !fn %in% names(reframe_agg_ns) ||
      (!verified && !reframe_head_is(expr, env, reframe_agg_ns[[fn]], fn))) {
    return(NULL)
  }
  args <- as.list(expr)[-1]
  if (identical(fn, "n")) {
    return(if (length(args) == 0L) list(fn = fn, arg = NULL) else NULL)
  }
  nms <- names(args) %||% rep("", length(args))
  named <- nms[nzchar(nms)]
  allowed <- c("na.rm", if (identical(fn, "quantile")) "probs")
  if (sum(!nzchar(nms)) != 1L || anyDuplicated(named) > 0L || !all(named %in% allowed)) {
    return(NULL)
  }
  if ("na.rm" %in% named && is.null(reframe_lgl_literal(args[["na.rm"]]))) {
    return(NULL)
  }
  if (identical(fn, "quantile")) {
    p <- args[["probs"]]
    if (!is.numeric(p) || length(p) != 1L || is.na(p) || p < 0 || p > 1) {
      return(NULL)
    }
  }
  list(fn = fn, arg = args[[which(!nzchar(nms))]])
}

# Internal: the R type of an allowlisted computed argument, walked on the R
# expression, or NULL when the expression isn't on the list.
#
# `nan_free` tracks whether the value can be NaN. A NaN is a value on the GPU,
# and since F55 is.na() and the comparisons read it as missing there, like R,
# so this gate is no longer needed for correctness. It still keeps FLOAT64
# comparisons and logic on the CPU route: comparisons, `!`, `&`, `|` and
# is.na() only take operands that can't be NaN (INT32/BOOL8 columns, integer
# arithmetic on them, literals). (A NaN reaching the aggregation itself is
# fine: gpu_summarise() treats it as missing.)
# @return `list(type, nan_free, cols)` (`cols`: reads a column) or NULL
# @keywords internal
reframe_arg_info <- function(expr, schema, env) {
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    type <- unname(schema$types[match(nm, schema$names)])
    if (nm %in% c(".data", ".env") || length(type) != 1L || is.na(type) ||
        !type %in% c("INT32", "FLOAT64", "BOOL8")) {
      return(NULL)  # an environment variable, a pronoun, a string/timestamp/factor
    }
    return(list(type = type, nan_free = type != "FLOAT64", cols = TRUE))
  }
  if (is.atomic(expr)) {
    type <- switch(typeof(expr), logical = "BOOL8", integer = "INT32", double = "FLOAT64", NULL)
    if (is.null(type) || length(expr) != 1L || !is.null(attributes(expr)) || is.nan(expr)) {
      return(NULL)
    }
    return(list(type = type, nan_free = TRUE, cols = FALSE))
  }
  fn <- reframe_head_name(expr)
  if (is.null(fn) || !fn %in% reframe_arg_fns || !reframe_head_is(expr, env, "base", fn)) {
    return(NULL)
  }
  args <- as.list(expr)[-1]
  if (length(args) == 0L || (!is.null(names(args)) && any(nzchar(names(args))))) {
    return(NULL)
  }
  info <- lapply(args, reframe_arg_info, schema, env)
  if (any(vapply(info, is.null, logical(1)))) {
    return(NULL)
  }
  types <- vapply(info, function(a) a$type, character(1))
  nan_free <- all(vapply(info, function(a) a$nan_free, logical(1)))
  cols <- any(vapply(info, function(a) a$cols, logical(1)))
  ints <- all(types %in% c("INT32", "BOOL8"))
  out <- function(type, nan_free) list(type = type, nan_free = nan_free, cols = cols)
  unary <- length(args) == 1L
  binary <- length(args) == 2L

  if (fn == "(" && unary) {
    return(info[[1]])
  }
  if (fn == "-" && unary) {
    # R's -TRUE is an integer; the GPU keeps a logical
    return(if (types == "BOOL8") NULL else info[[1]])
  }
  if (fn %in% c("+", "-", "*") && binary) {
    return(if (ints) out("INT32", nan_free) else out("FLOAT64", FALSE))
  }
  if (fn == "/" && binary) {
    return(out("FLOAT64", FALSE))
  }
  if (fn %in% c("%%", "%/%") && binary) {
    if (ints) {
      return(out("INT32", TRUE))
    }
    # An integer by a whole, non-zero double literal within the integer range
    # (`i %% 2`) is exact. A fractional divisor is not: R corrects the floored
    # quotient for the divisor's binary value (`1L %/% 0.1` is 9, while the
    # GPU's floor(x / y) gives 10), so those calls run on the CPU.
    rhs <- args[[2]]
    if (types[[1]] != "FLOAT64" && is.double(rhs) && is.finite(rhs) && rhs != 0 &&
        rhs == trunc(rhs) && abs(rhs) <= .Machine$integer.max) {
      return(out("FLOAT64", TRUE))
    }
    return(NULL)
  }
  if (fn == "abs" && unary) {
    return(out(if (types == "FLOAT64") "FLOAT64" else "INT32", nan_free))
  }
  if (fn == "exp" && unary) {
    return(out("FLOAT64", FALSE))
  }
  if (fn %in% c("==", "!=", "<", "<=", ">", ">=", "&", "|") && binary && nan_free) {
    return(out("BOOL8", TRUE))
  }
  if (fn %in% c("!", "is.na") && unary && nan_free) {
    return(out("BOOL8", TRUE))
  }
  NULL
}

# Internal: the type of an aggregation argument and whether it is a bare
# column, or NULL when the argument isn't allowlisted. A computed argument
# must read a column (one that reads none is dplyr's: `quantile(v, p)`), parse
# in the IR, and get from the IR the type R gives it (so the GPU result type
# is dplyr's).
# @return `list(type, bare)` or NULL
# @keywords internal
reframe_agg_arg_type <- function(arg, schema, env) {
  if (is.symbol(arg)) {
    nm <- as.character(arg)
    if (!nm %in% schema$names || nm %in% c(".data", ".env")) {
      return(NULL)
    }
    return(list(type = unname(schema$types[match(nm, schema$names)]), bare = TRUE))
  }
  info <- reframe_arg_info(arg, schema, env)
  if (is.null(info) || !info$cols) {
    return(NULL)
  }
  ir <- ir_parse_quo(rlang::new_quosure(arg, env), schema)
  if (is.null(ir) || ir_has_window(ir) ||
      !identical(unname(ir_infer_type(ir, schema)), info$type)) {
    return(NULL)
  }
  list(type = info$type, bare = FALSE)
}

# Internal: is every wide dot an allowlisted aggregation (see above)? Also
# requires summarise_dots_parseable(), so summarise() computes each dot
# natively. Pure schema work, no GPU.
#
# @param quos the wide dots
# @param verified logical, per dot: produced by the quantile()/range() rewrite
# @return NULL when a dot is off the list, else `list(int32_sum)`: the names of
#   the dots that `sum()` an INT32 argument (bare or computed). Such a sum can
#   leave R's integer range, which a lazy plan can't turn into a double.
# @keywords internal
reframe_wide_dots_ok <- function(quos, schema, verified) {
  tryCatch({
    parts <- vector("list", length(quos))
    for (i in seq_along(quos)) {
      parts[[i]] <- reframe_agg_parts(rlang::quo_get_expr(quos[[i]]),
                                      rlang::quo_get_env(quos[[i]]), verified[[i]])
      if (is.null(parts[[i]])) {
        return(NULL)
      }
    }
    # The k dots of one quantile() share fn and argument: check each distinct
    # (fn, argument, environment) once.
    keys <- lapply(seq_along(quos), function(i) {
      list(parts[[i]]$fn, parts[[i]]$arg, rlang::quo_get_env(quos[[i]]))
    })
    first <- !duplicated(keys)
    # (match() would compare the environments as text.)
    same_as <- vapply(keys, function(k) {
      which(vapply(keys[first], identical, logical(1), k))[[1]]
    }, integer(1))
    same_as <- vapply(same_as, function(j) which(first)[[j]], integer(1))
    int32_sum <- character()
    for (i in which(first)) {
      if (is.null(parts[[i]]$arg)) {
        next  # n()
      }
      arg <- reframe_agg_arg_type(parts[[i]]$arg, schema, rlang::quo_get_env(quos[[i]]))
      if (is.null(arg) || !arg$type %in% reframe_agg_arg_types(parts[[i]]$fn, arg$bare)) {
        return(NULL)
      }
      if (identical(parts[[i]]$fn, "sum") && identical(arg$type, "INT32")) {
        int32_sum <- c(int32_sum, names(quos)[same_as == same_as[[i]]])
      }
    }
    if (!summarise_dots_parseable(quos[first], schema)) {
      return(NULL)
    }
    list(int32_sum = int32_sum)
  }, error = function(e) NULL)
}

# Internal: the native plan of a reframe() call, or NULL when dplyr must run
# it. Touches no GPU data (schema only), so the caller can decide before any
# work. See the file header for the approach.
#
# @param dots enquos(...) of the call
# @param schema the input's current schema
# @return NULL, or `list(wide_dots, outputs, k, int32_sum, first_k_name)`: the named
#   quosures for summarise_impl(), the output columns after the keys in dot
#   order (`list(name, kind = "rep"|"interleave"|"literal", src | value)`),
#   the common size, the (user) names of the wide dots that sum an INT32
#   argument (character(0) if none), and the
#   first dot of that size (for dplyr's recycling
#   error on an empty table)
# @keywords internal
reframe_native_plan <- function(dots, schema) {
  nms <- names(dots)
  if (length(dots) == 0L || is.null(nms) || anyNA(nms) || any(!nzchar(nms)) ||
      anyDuplicated(nms) > 0L) {
    return(NULL)
  }
  col_names <- schema$names
  # Dots that read earlier dots' results are sequential in dplyr (the second
  # `quantile(x, probs)` below sees the `probs` column dot, ...).
  if (summarise_dots_are_sequential(dots) || dots_need_fallback(dots)) {
    return(NULL)
  }

  reserved <- c(col_names, nms)
  wide_dots <- list()
  verified <- logical()  # per wide dot: from the rewrite (head already resolved)
  outputs <- list()
  sizes <- integer(length(dots))
  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)
    name <- nms[[i]]

    lit <- reframe_constant_value(expr, env, col_names)
    if (!is.null(lit)) {
      outputs[[length(outputs) + 1L]] <- list(name = name, kind = "literal", value = lit)
      sizes[[i]] <- length(lit)
      next
    }

    # The call must be the real stats::quantile / base::range, not a function
    # of that name in the caller's environment.
    q_name <- reframe_call_name(expr, "stats")
    r_name <- reframe_call_name(expr, "base")
    is_quantile <- identical(q_name, "quantile") && reframe_head_is(expr, env, "stats", "quantile")
    is_range <- identical(r_name, "range") && reframe_head_is(expr, env, "base", "range")
    if (is_quantile) {
      parsed <- reframe_parse_quantile(expr, env, col_names)
      if (is.null(parsed)) return(NULL)
      calls <- lapply(parsed$probs, function(p) {
        rlang::call2("quantile", parsed$x, probs = p, na.rm = parsed$na_rm)
      })
    } else if (is_range) {
      parsed <- reframe_parse_range(expr)
      if (is.null(parsed)) return(NULL)
      calls <- list(rlang::call2("min", parsed$x, na.rm = parsed$na_rm),
                    rlang::call2("max", parsed$x, na.rm = parsed$na_rm))
    } else {
      # A multi-row call we don't rewrite (another namespace, ...) is dplyr's.
      if (isTRUE(reframe_head_name(expr) %in% c("quantile", "range"))) return(NULL)
      # Anything else must be an allowlisted scalar aggregation (checked
      # below); it keeps the user's name.
      wide_dots[[name]] <- quo
      verified[[name]] <- FALSE
      outputs[[length(outputs) + 1L]] <- list(name = name, kind = "rep", src = name)
      sizes[[i]] <- 1L
      next
    }

    sizes[[i]] <- length(calls)
    if (length(calls) == 1L) {
      wide_dots[[name]] <- rlang::new_quosure(calls[[1]], env)
      verified[[name]] <- TRUE
      outputs[[length(outputs) + 1L]] <- list(name = name, kind = "rep", src = name)
    } else {
      temps <- character(length(calls))
      for (j in seq_along(calls)) {
        temps[[j]] <- unique_temp_name(sprintf(".reframe_%d_%d", i, j), reserved)
        reserved <- c(reserved, temps[[j]])
        wide_dots[[temps[[j]]]] <- rlang::new_quosure(calls[[j]], env)
        verified[[temps[[j]]]] <- TRUE
      }
      outputs[[length(outputs) + 1L]] <- list(name = name, kind = "interleave", src = temps)
    }
  }

  # At least one aggregation to compute; the multi-row sizes must agree (the
  # rest is size 1, which recycles); a size mismatch is dplyr's error.
  big <- unique(sizes[sizes > 1L])
  if (length(wide_dots) == 0L || length(big) > 1L) {
    return(NULL)
  }
  # Every wide dot must be an allowlisted scalar aggregation.
  wide_quos <- rlang::new_quosures(wide_dots)
  if (summarise_dots_are_sequential(wide_quos)) {
    return(NULL)
  }
  wide_info <- reframe_wide_dots_ok(wide_quos, schema, unname(verified[names(wide_quos)]))
  if (is.null(wide_info)) {
    return(NULL)
  }

  k <- if (length(big) == 1L) big else 1L
  list(
    wide_dots = wide_quos,
    outputs = outputs,
    k = as.integer(k),
    int32_sum = wide_info$int32_sum,
    first_k_name = if (k > 1L) nms[[which(sizes > 1L)[[1]]]] else NA_character_
  )
}
