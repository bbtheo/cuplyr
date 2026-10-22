# =============================================================================
# F19: superseded colwise verbs (`mutate_at()`, `summarise_all()`,
# `filter_if()`, `arrange_at()`, ...) on the GPU
# =============================================================================
#
# dplyr's colwise verbs are plain functions: they resolve the columns on the
# CPU, then call the plain verb with one dot per (column, function) whose
# expression is a call with a FUNCTION OBJECT as its head:
#
#   `<mean>(x, na.rm = TRUE)`          a function, with extra `...` arguments
#   `<inline_colwise_function>(x)`     a formula (`~ . + y`)
#   `<rlang_lambda_function>(x) & ...` filter_all()/_at()/_if() predicates
#
# The IR parses calls by name, so colwise_rewrite_dots() turns each such call
# into the plain call dplyr would evaluate, per dot (dots stay separate, hence
# sequential). Two scoping rules, both from dplyr:
#
#   - an inline_colwise_function body is evaluated IN THE DATA MASK, with the
#     formula's environment behind it: a free variable may be a column.
#   - any other function is called as a value: its free variables must NOT be
#     columns (across()'s VALUE rule, across_value_spec()).
#
# The rewrite only turns function-object heads into plain calls; it checks no
# types. The plain verbs' type gate (ir_type_gate()) decides native vs CPU for
# the rewritten dots, falling back with the user's original dots. A dot with
# any call the rewrite can't reproduce exactly is returned untouched, and the
# verb's CPU fallback then runs dplyr on the original dots.

# Does `expr` contain a call whose head is a function object? (cheap scan)
# @keywords internal
colwise_has_fn_head <- function(expr) {
  if (rlang::is_missing(expr)) return(FALSE)
  if (rlang::is_quosure(expr)) return(colwise_has_fn_head(rlang::quo_get_expr(expr)))
  if (!is.call(expr)) return(FALSE)
  if (is.function(expr[[1L]])) return(TRUE)
  for (part in as.list(expr)) {
    if (colwise_has_fn_head(part)) return(TRUE)
  }
  FALSE
}

# `dots` with every function-object-headed call rewritten to a plain call;
# a dot that can't be rewritten exactly is kept as is. A no-op (returns
# `dots` itself) when no dot has such a call.
# @param dots Named list of quosures
# @param schema The table's current schema
# @keywords internal
colwise_rewrite_dots <- function(dots, schema) {
  if (!any(vapply(dots, function(q) colwise_has_fn_head(rlang::quo_get_expr(q)), logical(1)))) {
    return(dots)
  }
  nms <- names(dots) %||% character(length(dots))
  scope <- c(schema$names, nms[nzchar(nms)])
  for (i in seq_along(dots)) {
    q <- dots[[i]]
    if (!colwise_has_fn_head(rlang::quo_get_expr(q))) next
    out <- tryCatch(
      colwise_rewrite(rlang::quo_get_expr(q), rlang::quo_get_env(q), schema, scope),
      colwise_giveup = function(e) NULL
    )
    if (is.null(out)) next
    out <- if (rlang::is_quosure(out)) out else rlang::new_quosure(out, rlang::quo_get_env(q))
    dots[[i]] <- out
  }
  dots
}

colwise_giveup <- function() {
  rlang::abort("", class = "colwise_giveup")
}

# Rewrite `expr` (evaluated in `env`). A call that needs an environment of its
# own comes back as a quosure, nested in the tree; the caller unwraps a
# top-level one.
# @keywords internal
colwise_rewrite <- function(expr, env, schema, scope) {
  if (rlang::is_missing(expr)) return(expr)
  if (rlang::is_quosure(expr)) {
    if (!colwise_has_fn_head(expr)) return(expr)
    out <- colwise_rewrite(rlang::quo_get_expr(expr), rlang::quo_get_env(expr), schema, scope)
    return(if (rlang::is_quosure(out)) out else rlang::new_quosure(out, rlang::quo_get_env(expr)))
  }
  if (!is.call(expr) || !colwise_has_fn_head(expr)) return(expr)
  head <- expr[[1L]]
  if (is.function(head)) return(colwise_rewrite_call(expr, head, env, schema, scope))
  parts <- lapply(as.list(expr), colwise_rewrite, env = env, schema = schema, scope = scope)
  as.call(parts)
}

# One `<fn>(col, ...)` call
# @keywords internal
colwise_rewrite_call <- function(expr, fn, env, schema, scope) {
  args <- as.list(expr)[-1L]
  if (length(args) < 1L || !is.symbol(args[[1L]]) || nzchar(names(args)[1L] %||% "")) {
    colwise_giveup()
  }
  col <- args[[1L]]
  extra <- args[-1L]
  placeholders <- c(".", ".x")

  if (inherits(fn, "inline_colwise_function")) {
    fml <- attr(fn, "formula")
    if (length(extra) > 0L || !rlang::is_formula(fml, lhs = FALSE) || !across_formula_inlinable(fml) ||
        colwise_calls_cur(fml) || "..." %in% all.names(fml) ||
        any(placeholders %in% scope)) {
      colwise_giveup()
    }
    body <- substitute_across_placeholders(fml[[2L]], col, "")
    return(colwise_native_quo(body, environment(fml)))
  }

  if (inherits(fn, "rlang_lambda_function")) {
    body <- body(fn)
    if (length(extra) > 0L || !across_formula_inlinable(body) || colwise_calls_cur(body) ||
        "..." %in% all.names(body) || any(placeholders %in% scope) ||
        across_free_vars_hit_data(set_diff(all.vars(body), placeholders), scope)) {
      colwise_giveup()
    }
    return(colwise_native_quo(substitute_across_placeholders(body, col, ""), environment(fn)))
  }

  # a bare function the arrange verbs take by name. dplyr keeps `<desc>(s)`
  # as a value (-xtfrm(s)), which sorts strings in the session collation; the
  # GPU's desc(s) is the C locale.
  if (identical(fn, dplyr::desc) && length(extra) == 0L &&
      identical(get0("desc", env, mode = "function"), fn)) {
    if (identical(unname(schema$types[match(as.character(col), schema$names)]), "STRING")) {
      colwise_giveup()
    }
    return(call("desc", col))
  }

  # a closure is called as a value: cur_column() is an error there in dplyr
  if (colwise_calls_cur(body(fn))) colwise_giveup()

  if (!all(vapply(extra, function(a) is.atomic(a) && length(a) == 1L, logical(1)))) colwise_giveup()
  spec <- across_value_spec(fn, NULL, env, scope, extra)
  if (is.null(spec)) colwise_giveup()
  body <- spec$build(col, as.character(col))
  if (is.null(spec$env)) body else colwise_native_quo(body, spec$env)
}

# A rewritten body, or give up when it uses a call the IR doesn't lower by name
# @keywords internal
colwise_native_quo <- function(body, env) {
  if (!across_expr_native(body)) colwise_giveup()
  rlang::new_quosure(body, env)
}

# Does `expr` call a `cur_*()` context function? (call heads only)
# @keywords internal
colwise_calls_cur <- function(expr) {
  if (rlang::is_missing(expr) || !is.call(expr)) return(FALSE)
  head <- expr[[1L]]
  if (is.call(head) && length(head) == 3L && identical(head[[1L]], as.name("::"))) head <- head[[3L]]
  if (is.symbol(head) && startsWith(as.character(head), "cur_")) return(TRUE)
  any(vapply(as.list(expr), colwise_calls_cur, logical(1)))
}

# =============================================================================
# tbl_ptype(): dplyr's `_if` verbs pull every column to test `.p` on it
# =============================================================================

#' Prototype of a GPU table for dplyr's `_if` verbs
#'
#' `dplyr::tbl_if_vars()` (behind `mutate_if()`, `select_if()`, ...) calls
#' `tbl_ptype()` and then `pull()`s every column to test the predicate on. A
#' type-only predicate is answered by a one-row proxy of the column classes,
#' with no GPU read; any other predicate gets the data, downloaded once. Any
#' other caller of the generic gets `.data` back unchanged.
#'
#' @param .data A `tbl_gpu`.
#' @return A data frame, or `.data`.
#' @importFrom dplyr tbl_ptype
#' @export
#' @keywords internal
tbl_ptype.tbl_gpu <- function(.data) {
  caller <- tryCatch(sys.function(sys.parent()), error = function(e) NULL)
  if (!identical(caller, utils::getFromNamespace("tbl_if_vars", "dplyr"))) {
    return(.data)
  }
  frame <- parent.frame()
  pred <- get0(".p", frame, inherits = FALSE)
  # `.env` can be an unsupplied argument (filter_if()), only used to look up a
  # bare predicate name
  env <- tryCatch(get(".env", frame, inherits = FALSE), error = function(e) globalenv())
  if (is.list(pred) && length(pred) == 1L) pred <- pred[[1L]]
  if (inherits(pred, "inline_colwise_function")) pred <- attr(pred, "formula")
  if (!is.null(pred) && !is.null(env) &&
      isTRUE(tryCatch(where_predicate_is_type_only(pred, env), error = function(e) FALSE))) {
    return(tidyselect_proxy(current_schema(.data)))
  }
  collect_frame(.data)
}
