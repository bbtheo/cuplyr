# Rowwise tables (F49)
#
# A rowwise tbl_gpu is a table whose verbs run once per row. Its rowwise vars
# (the `rowwise(x, ...)` selection, kept by summarise()) live in `$groups`,
# which is what group_vars(), select()'s "add missing grouping variables", the
# rename/relocate name tracking and dplyr's own model of a rowwise_df already
# do right, and the object carries `$rowwise = TRUE`, a field that exists ONLY
# on rowwise tables (see new_tbl_gpu(): the conformance harness recognises an
# uploaded fixture by its fields).
#
# THE DANGER of that choice is any code that treats `$groups` as real groups: it
# would compute per rowwise-var group, silently wrong. Hence the rule (the
# classification table is in .claude/reference/architecture.md, "Rowwise
# tables"): a verb either has an explicit rowwise-aware branch, or its whole call
# goes to dplyr on a CPU rowwise_df (rowwise_cpu(), gpu_fallback()). Never both
# half-way.
#
# Inside rowwise() every column reference is a length-1 value, so an
# ELEMENTWISE expression gives the same result vectorised: those run on the
# existing native path unchanged. Only calls that aggregate within a row need a
# rewrite into horizontal (across-columns) GPU ops: sum/mean/min/max/prod of
# c_across()/c()/several columns, n(), cur_group_id(), length(c(...)) (tier B,
# rowwise_rewrite_dots() below). The native attempt (rowwise_attempt()) is
# all-or-nothing: anything outside the allowlist bails and the WHOLE call runs
# in dplyr with the user's original dots, so dplyr produces its own result or
# error.

# TRUE for a rowwise tbl_gpu.
# @keywords internal
is_rowwise_gpu <- function(x) {
  is_tbl_gpu(x) && isTRUE(.subset2(x, "rowwise"))
}

# `x` as a rowwise table with rowwise vars `vars` (metadata only: no GPU work,
# a lazy table keeps its pending ops).
# @keywords internal
rowwise_mark <- function(x, vars) {
  x <- set_tbl_gpu_field(x, "groups", as.character(vars))
  x <- set_tbl_gpu_field(x, "group_drop", TRUE)
  x <- set_tbl_gpu_field(x, "fixed_keys", NULL)
  x <- set_tbl_gpu_field(x, "bare_df", NULL)
  set_tbl_gpu_field(x, "rowwise", TRUE)
}

# A rowwise table whose vars are listed in the order of its columns. dplyr's
# verbs that rebuild the table's columns (mutate(), transmute(), select(),
# relocate(), `[`, head()/tail()) leave the rowwise vars in column order; the
# row-keeping ones (filter(), arrange(), slice(), joins, ...) in the order
# rowwise() was given them. Metadata only.
# @keywords internal
rowwise_column_order <- function(x) {
  if (!is_rowwise_gpu(x) || length(.subset2(x, "groups")) < 2L) {
    return(x)
  }
  cols <- current_schema(x)$names
  set_tbl_gpu_field(x, "groups", cols[cols %in% .subset2(x, "groups")])
}

# `x` grouped by `vars` (summarise()'s .groups = "keep"/"drop" on a rowwise
# table): the rowwise marker goes, the vars become real groups.
# @keywords internal
rowwise_to_grouped <- function(x, vars) {
  x <- set_tbl_gpu_field(x, "rowwise", NULL)
  x <- set_tbl_gpu_field(x, "groups", as.character(vars))
  x <- set_tbl_gpu_field(x, "group_drop", TRUE)
  set_tbl_gpu_field(x, "fixed_keys", NULL)
}

#' Group a GPU table by rows
#'
#' Native: `rowwise()` of a `tbl_gpu` only records which columns identify a row
#' (metadata, no GPU work and no download; a lazy table keeps its pending
#' operations). The result is a *rowwise* `tbl_gpu`: [collect()] returns
#' dplyr's `rowwise_df`, `as_tibble()` and `ungroup()` drop the rowwise-ness,
#' `group_by()` replaces it, and [tbl_gpu()] of a `rowwise_df` uploads one.
#'
#' Elementwise expressions in `mutate()`/`filter()`/`summarise()` on a rowwise
#' table run on the GPU as they would on a plain one (every column is one value
#' per row). The horizontal aggregates `sum()`/`mean()`/`min()`/`max()`/`prod()`
#' of `c_across(<tidyselect>)`, of `c(a, b, ...)` or of several column
#' arguments, `n()`, `cur_group_id()` and `length(c_across(...))` are rewritten
#' into GPU operations over the columns of each row, with dplyr's NA rules and
#' result types (integer inputs give an integer `sum()`/`min()`/`max()`, a
#' double sum if the integer range is left; an all-`NA` row of
#' `min()`/`max(na.rm = TRUE)` is `Inf`/`-Inf` with dplyr's warning). Only
#' logical, integer and double columns take part. Everything else (a function
#' that needs the row as a vector, list-columns, window functions, `.by`) runs
#' the whole call in dplyr on the CPU and comes back as a rowwise `tbl_gpu`.
#'
#' @param .data A `tbl_gpu`.
#' @param ... <[`tidy-select`][dplyr::dplyr_tidy_select]> Columns to keep as
#'   identifying variables in `summarise()`. A grouped table takes its group
#'   variables (naming others is an error, as in dplyr).
#' @return A rowwise `tbl_gpu`.
#' @export
#' @importFrom dplyr rowwise
rowwise.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  cpu <- function() {
    gpu_fallback("rowwise", .data, function(tbl) rlang::inject(dplyr::rowwise(tbl, !!!dots)))
  }
  schema <- current_schema(.data)
  groups <- .subset2(.data, "groups")

  if (length(groups) > 0L && !is_rowwise_gpu(.data)) {
    # rowwise.grouped_df(): the group vars, and nothing else may be named
    if (length(dots) > 0L) {
      rlang::abort(c(
        "Can't re-group when creating rowwise data.",
        i = "Either first `ungroup()` or call `rowwise()` without arguments."
      ))
    }
    vars <- groups
  } else if (length(dots) == 0L) {
    vars <- character()
  } else {
    # A selection that may read values, or a duplicated name, is dplyr's.
    if (anyDuplicated(schema$names) > 0L || !selection_is_type_only(dots, NULL, schema$names)) {
      return(cpu())
    }
    loc <- tidyselect::eval_select(rlang::expr(c(!!!dots)), tidyselect_proxy(schema),
                                   error_call = rlang::current_env())
    # dplyr keeps the SOURCE names of a renaming selection as the vars
    vars <- schema$names[loc]
  }
  check_dup_names_ungrouped(schema$names, vars)
  rowwise_mark(.data, vars)
}

# =============================================================================
# The CPU route for a rowwise input
# =============================================================================

# Depth of rowwise_attempt() calls in progress. While a native rowwise attempt
# runs, gpu_fallback() bails at once (before computing anything) so the attempt
# is abandoned and the whole call reruns in dplyr; see rowwise_attempt().
# `failed`/`why`: the message of the error the last attempt swallowed, and the
# one the CPU rerun in progress reports (rowwise_rerun()).
rowwise_state <- new.env(parent = emptyenv())
rowwise_state$depth <- 0L

# Abandon the native attempt in progress (it is always inside rowwise_attempt()).
# @keywords internal
rowwise_bail <- function(why = "no native rowwise form") {
  rlang::abort(why, class = "cuplyr_rowwise_bail")
}

# Run `native()`, a rowwise verb's native attempt, and return its result, or
# NULL if it cannot be used: it bailed (rowwise_bail()), a CPU fallback was
# requested inside it, or it raised ANY error but an OOM that survived the
# retry (`cuplyr_error_gpu_oom`, F58), which propagates. The caller then runs the whole
# call in dplyr with the user's original arguments, so dplyr produces its own
# result or its own error (cuplyr's wording of an error would differ). Nothing
# the attempt built leaks: it works on copies of the immutable input.
# @keywords internal
rowwise_attempt <- function(native) {
  rowwise_state$depth <- rowwise_state$depth + 1L
  on.exit(rowwise_state$depth <- rowwise_state$depth - 1L, add = TRUE)
  rowwise_state$failed <- NULL
  tryCatch(
    native(),
    cuplyr_cpu_fallback = function(cnd) NULL,
    error = function(cnd) {
      # (re-raised from this outermost handler: from a sibling handler it would
      # be caught by this one)
      if (inherits(cnd, "cuplyr_error_gpu_oom")) stop(cnd)
      rowwise_state$failed <- conditionMessage(cnd)
      NULL
    }
  )
}

# Run `cpu()`, the CPU rerun of a rowwise verb whose native attempt or rewrite
# gave NULL, with the message of the error that attempt swallowed (if any)
# passed on to gpu_fallback()'s notification, so a native bug is not hidden
# behind a plain "fell back" (options(cuplyr.fallback = "warn")).
# @keywords internal
rowwise_rerun <- function(cpu) {
  rowwise_state$why <- rowwise_state$failed
  rowwise_state$failed <- NULL
  on.exit(rowwise_state$why <- NULL, add = TRUE)
  cpu()
}

# Run the verb that called this on the CPU: dplyr's own verb on a rowwise_df of
# the input, with the arguments exactly as the user wrote them (the call is
# rebuilt from the method's call and evaluated in the caller's environment, so
# tidy evaluation, `!!`, `{{ }}` and defaults behave as for a direct call). The
# whole-call route of every verb that has no rowwise-aware native form.
#
# Called first thing in a method, before any argument but the first is
# evaluated. `tables` names arguments holding another table (a join/set-op `y`):
# they are collected to data frames here.
# @param .data The rowwise tbl_gpu (the method's first argument)
# @param verb The dplyr function name
# @keywords internal
rowwise_cpu <- function(.data, verb, tables = character()) {
  idx <- sys.parent()
  fn <- sys.function(idx)
  call <- sys.call(idx)
  caller <- parent.frame(2)
  mc <- match.call(fn, call, expand.dots = TRUE, envir = caller)
  first <- names(formals(fn))[[1L]]
  for (nm in set_intersect(tables, names(mc))) {
    val <- rlang::eval_bare(mc[[nm]], caller)
    mc[[nm]] <- if (is_tbl_gpu(val)) collect_other_side(val) else val
  }
  verb_fn <- rlang::call2("::", quote(dplyr), rlang::sym(verb))
  gpu_fallback(verb, .data, function(tbl) {
    mc[[first]] <- tbl
    mc[[1L]] <- verb_fn
    rlang::eval_bare(mc, caller)
  })
}

# =============================================================================
# Group metadata of a rowwise table: one group per row
# =============================================================================

# group_data() of a rowwise table, as dplyr builds it: the rowwise vars, as the
# rows hold them, and `.rows = list_of(1L, 2L, ...)`.
# @keywords internal
rowwise_group_data <- function(.data) {
  m <- if (has_pending_ops(.data)) compute(.data) else .data
  vars <- .subset2(m, "groups")
  n <- dim(m)[[1L]]
  rows <- vctrs::new_list_of(as.list(seq_len(n)), ptype = integer())
  if (length(vars) == 0L) {
    return(tibble::new_tibble(list(.rows = rows), nrow = n))
  }
  keys <- dplyr::select(ungrouped(m), dplyr::all_of(vars))
  keys <- tibble::as_tibble(collect_frame(keys))
  keys$.rows <- rows
  keys
}

# =============================================================================
# Tier B: the horizontal-aggregate rewrite
# =============================================================================

# Calls that mean something only inside the group/row machinery a rewrite does
# not reproduce: any that survive the rewrite send the call to dplyr.
rowwise_forbidden_calls <- c("c_across", "cur_group", "cur_group_rows", "cur_data",
                             "cur_data_all", "pick", "across", "if_any", "if_all",
                             "cur_column")

# Column-free calls dplyr evaluates once PER ROW but the native path folds once:
# the same value either way only for these. Anything else (runif(), Sys.time(),
# a user function) sends the call to dplyr.
rowwise_pure_fns <- c(
  "+", "-", "*", "/", "^", "%%", "%/%", "(", "==", "!=", "<", ">", "<=", ">=", "&", "|",
  "!", "&&", "||", "c", "abs", "sqrt", "exp", "log", "log2", "log10", "floor", "ceiling",
  "round", "signif", "as.integer", "as.numeric", "as.double", "as.character", "as.logical",
  "paste", "paste0", "min", "max", "sum", "mean", "length", "rep", "seq_len", "seq", "rev",
  "nchar", "toupper", "tolower", "is.na", "ifelse", "identity", "list", "numeric", "integer",
  "character", "logical", ":", "$", "[[", "[", ".env", ".data", "::", "cur_group_id"
)

# The dots of a rowwise verb call, rewritten for the native path, or NULL when
# the call has no native form (the caller runs it in dplyr).
#
# - `n()` is 1L and `length(c_across(<sel>))` / `length(c(a, b))` the number of
#   columns;
# - `sum`/`mean`/`min`/`max`/`prod` of `c_across(<sel>)`, of `c(<columns>)`, of
#   several column arguments or of one column become one `.cuplyr_rw(<IR node>,
#   <columns>)` call that ir_parse_expr() returns the node of; `na.rm` must be a
#   literal TRUE/FALSE. `mean(a, b)` is `mean(a, trim = b)`: not rewritten.
#   Operands are plain (logical/integer/double) columns, including ones an
#   earlier dot wrote (the running schema types them);
# - `cur_group_id()` is left for the verb (a row-number column);
# - anything left that only the group machinery can evaluate (c_across(),
#   cur_*(), pick(), across()), or a column-free impure call, gives NULL.
# A rewritten dot must be named when `named` (an unnamed mutate()/summarise()
# dot takes its name from the user's expression, which dplyr derives; filter()
# dots have none).
# @param .data The rowwise tbl_gpu
# @param dots Quosures (the user's dots)
# @param named Whether a rewritten dot needs a name
# @keywords internal
rowwise_rewrite_dots <- function(.data, dots, named = TRUE) {
  rowwise_state$failed <- NULL
  tryCatch(
    rowwise_rewrite_dots_(.data, dots, named),
    error = function(cnd) {
      # an OOM that survived the retry (F58) is the user's to see, not a reason
      # to rerun the call on the CPU
      if (inherits(cnd, "cuplyr_error_gpu_oom")) stop(cnd)
      rowwise_state$failed <- conditionMessage(cnd)
      NULL
    }
  )
}

rowwise_rewrite_dots_ <- function(.data, dots, named) {
  schema <- current_schema(.data)
  if (anyDuplicated(schema$names) > 0L) {
    return(NULL)
  }
  # An eager table can say whether an integer sum overflows / an integer
  # min()/max() has an all-NA row (the type then depends on the data); a lazy
  # one cannot.
  eager <- identical(.subset2(.data, "exec_mode"), "eager") && !has_pending_ops(.data)
  # `schema` is the running schema: later dots see the columns earlier dots
  # wrote (c_across() selects them, aggregates read them), with their types.
  ctx <- list(schema = schema, orig_schema = schema, orig_names = schema$names,
              vars = .subset2(.data, "groups"),
              eager = eager, ptr = if (eager) .subset2(.data, "ptr"), produced = character())
  nms <- names(dots) %||% rep("", length(dots))
  out <- dots
  for (i in seq_along(dots)) {
    expr <- rlang::quo_get_expr(dots[[i]])
    env <- rlang::quo_get_env(dots[[i]])
    new <- rowwise_rewrite_expr(expr, env, ctx)
    if (!identical(new, expr)) {
      if (named && !nzchar(nms[i])) {
        return(NULL)
      }
      out[[i]] <- rlang::new_quosure(new, env)
    }
    if (any(all.names(new) %in% rowwise_forbidden_calls) ||
        !rowwise_consts_ok(new, ctx$schema)) {
      return(NULL)
    }
    if (named) {
      # the dot's output joins the running schema (parse_mutate_dots()'s rule)
      name <- if (nzchar(nms[i])) nms[i] else {
        rlang::with_options("rlang:::use_as_label_infix" = FALSE, rlang::as_label(dots[[i]]))
      }
      ctx$produced <- set_union(ctx$produced, name)
      if (is.null(new)) {
        keep <- ctx$schema$names != name
        ctx$schema$names <- ctx$schema$names[keep]
        ctx$schema$types <- ctx$schema$types[keep]
        ctx$schema <- set_schema_factor_levels(ctx$schema, name, NULL)
      } else {
        # (cur_group_id() is an INT32 row number the verb adds later)
        typed <- substitute_zero_arg_call(rlang::quo_get_expr(out[[i]]), "cur_group_id", 1L)
        ir <- ir_parse_quo(rlang::new_quosure(typed, env), ctx$schema)
        if (is.null(ir)) {
          return(NULL)
        }
        ctx$schema <- update_schema_for_expr(ctx$schema, make_mutate_expr(name, ir, ctx$schema))
      }
    }
  }
  out
}

# Every column-free call in `expr` is one rowwise_pure_fns / a registry op.
# @keywords internal
rowwise_consts_ok <- function(expr, schema) {
  if (rlang::is_missing(expr) || !is.call(expr)) {
    return(TRUE)
  }
  if (rlang::is_quosure(expr)) {
    return(rowwise_consts_ok(rlang::quo_get_expr(expr), schema))
  }
  if (!expr_has_column_ref(expr, schema)) {
    known <- c(rowwise_pure_fns, set_diff(names(ir_call_registry), ir_internal_ops))
    return(all(rowwise_call_heads(expr) %in% known))
  }
  parts <- as.list(expr)[-1]
  all(vapply(parts, rowwise_consts_ok, logical(1), schema = schema))
}

# The function names called anywhere in `expr` (a head `pkg::f` is `f`; a head
# that is not a name, such as a function literal, is NA).
# @keywords internal
rowwise_call_heads <- function(expr) {
  if (rlang::is_missing(expr)) {
    return(character())
  }
  if (rlang::is_quosure(expr)) {
    return(rowwise_call_heads(rlang::quo_get_expr(expr)))
  }
  if (!is.call(expr)) {
    return(character())
  }
  head <- call_head_name(expr)
  c(head %||% NA_character_, unlist(lapply(as.list(expr)[-1L], rowwise_call_heads), use.names = FALSE))
}

rowwise_rewrite_expr <- function(expr, env, ctx) {
  if (rlang::is_missing(expr)) {
    return(expr)
  }
  if (rlang::is_quosure(expr)) {
    inner <- rowwise_rewrite_expr(rlang::quo_get_expr(expr), rlang::quo_get_env(expr), ctx)
    return(if (identical(inner, rlang::quo_get_expr(expr))) expr else
      rlang::new_quosure(inner, rlang::quo_get_env(expr)))
  }
  if (!is.call(expr)) {
    return(expr)
  }
  head <- call_head_name(expr)
  # a lambda or formula is not evaluated per row here: leave it (any c_across()
  # inside is caught by the forbidden-call scan)
  if (!is.null(head) && head %in% c("function", "~")) {
    return(expr)
  }
  if (!is.null(head)) {
    if (identical(head, "n") && length(expr) == 1L) {
      return(1L)
    }
    if (identical(head, "length") && length(expr) == 2L && rowwise_is_fn(expr[[1]], env, "length")) {
      cols <- rowwise_operands(list(expr[[2]]), env, ctx, "length")
      if (!is.null(cols)) {
        # a literal that still reads its operands, so .keep = "used"/"unused"
        # count them (dplyr's mask marks them used)
        lit <- ir_lit_from_r(length(cols))
        lit$reads <- cols
        return(rlang::call2(".cuplyr_rw", lit, !!!rlang::syms(unique(cols))))
      }
    }
    if (head %in% c("sum", "mean", "min", "max", "prod")) {
      agg <- rowwise_rewrite_agg(expr, head, env, ctx)
      if (!is.null(agg)) {
        return(agg)
      }
    }
  }
  parts <- as.list(expr)
  for (j in seq_along(parts)[-1L]) {
    if (!rlang::is_missing(parts[[j]])) {
      parts[j] <- list(rowwise_rewrite_expr(parts[[j]], env, ctx))
    }
  }
  as.call(parts)
}

# Does the call head `head` (a symbol or pkg::symbol) name base R's `name` as
# seen from `env`? (A user's own `sum` or `c` must not be rewritten.)
# @keywords internal
rowwise_is_fn <- function(head, env, name) {
  if (rlang::is_call(head, "::", n = 2L)) {
    return(identical(as.character(head[[3]]), name) &&
             as.character(head[[2]]) %in% c("base", "dplyr"))
  }
  is.symbol(head) && identical(as.character(head), name) &&
    identical(get0(name, envir = env, mode = "function"), get(name, envir = baseenv(), mode = "function"))
}

# The columns a horizontal aggregate / length() reads, from its positional
# arguments `args`: `c_across(<sel>)`, `c(<bare columns>)`, or bare columns.
# NULL if the arguments have no such shape.
# @keywords internal
rowwise_operands <- function(args, env, ctx, fn) {
  as_column <- function(a) {
    if (is.symbol(a) && as.character(a) %in% ctx$schema$names) as.character(a) else NULL
  }
  cols <- NULL
  if (length(args) == 1L) {
    a <- args[[1L]]
    if (rlang::is_call(a) && identical(call_head_name(a), "c_across")) {
      cols <- rowwise_c_across_cols(a, env, ctx)
    } else if (rlang::is_call(a) && rowwise_is_fn(a[[1L]], env, "c")) {
      parts <- as.list(a)[-1L]
      if (length(parts) > 0L && is.null(names(parts))) {
        cols <- vapply(parts, function(p) as_column(p) %||% NA_character_, character(1))
        if (anyNA(cols)) cols <- NULL
      }
    } else {
      cols <- as_column(a)
    }
  } else if (length(args) > 1L && !identical(fn, "mean")) {
    cols <- vapply(args, function(p) as_column(p) %||% NA_character_, character(1))
    if (anyNA(cols)) cols <- NULL
  }
  if (is.null(cols) || length(cols) == 0L) {
    return(NULL)
  }
  types <- ctx$schema$types[match(cols, ctx$schema$names)]
  if (!all(types %in% c("BOOL8", "INT32", "FLOAT64"))) {
    return(NULL)
  }
  unname(cols)
}

# c_across(<selection>) resolved against the table's columns minus the rowwise
# vars (dplyr's rule), or NULL.
# @keywords internal
rowwise_c_across_cols <- function(call, env, ctx) {
  args <- as.list(call)[-1L]
  if (length(args) != 1L || !(is.null(names(args)) || names(args) %in% c("", "cols"))) {
    return(NULL)
  }
  sel <- resolve_across_cols(args[[1L]], env, ctx$schema, ctx$vars)
  if (is.null(sel)) {
    return(NULL)
  }
  cols <- as.character(sel)
  cols
}

# sum/mean/min/max/prod over the columns of a row -> `.cuplyr_rw(<node>, cols)`.
# @keywords internal
rowwise_rewrite_agg <- function(expr, fn, env, ctx) {
  if (!rowwise_is_fn(expr[[1L]], env, fn)) {
    return(NULL)
  }
  args <- as.list(expr)[-1L]
  nms <- names(args) %||% rep("", length(args))
  na_rm <- FALSE
  is_narm <- nms == "na.rm"
  if (any(is_narm)) {
    v <- args[[which(is_narm)[[1L]]]]
    if (sum(is_narm) > 1L || !(isTRUE(v) || isFALSE(v))) {
      return(NULL)
    }
    na_rm <- v
    args <- args[!is_narm]
    nms <- nms[!is_narm]
  }
  if (any(nzchar(nms)) || length(args) == 0L) {
    return(NULL)
  }
  cols <- rowwise_operands(args, env, ctx, fn)
  if (is.null(cols)) {
    return(NULL)
  }
  rowwise_agg_call(fn, na_rm, cols, ctx)
}

# The IR node of one horizontal aggregate and the call embedding it.
#
# Result types are dplyr's: mean()/prod() and anything with a double operand
# are doubles; sum()/min()/max() of logical/integer operands are integers. Two
# of those depend on the data: an integer sum() that leaves the integer range
# is a double, and an integer min()/max(na.rm = TRUE) of an all-NA row is
# +-Inf (a double, with a warning). An eager table asks the GPU first and
# builds the double form when it has to; a lazy plan has a fixed schema, so
# its integer sum() is the INT32 form that fails loudly (cuplyr_error_sum_overflow,
# like F42's window sums) and its integer min()/max(na.rm = TRUE) has no native form.
# On zero rows dplyr still evaluates min()/max() once (+-Inf, a double, with
# base R's warning): an eager table sends that to dplyr; in a lazy plan the
# kernel flags it (check_rowwise_flags()).
# @keywords internal
rowwise_agg_call <- function(fn, na_rm, cols, ctx) {
  if (ctx$eager && fn %in% c("min", "max") && gpu_dim(ctx$ptr)[[1L]] == 0L) {
    rowwise_bail("a min()/max() of a zero-row table")
  }
  types <- ctx$schema$types[match(cols, ctx$schema$names)]
  any_dbl <- "FLOAT64" %in% types
  mk <- function(op) {
    ir_call(op, lapply(cols, ir_col), meta = list(fn = fn, na_rm = na_rm))
  }
  op <- if (fn %in% c("mean", "prod") || any_dbl) "rw_agg_f64" else "rw_agg_i32"
  node <- mk(op)
  data_dependent <- identical(op, "rw_agg_i32") &&
    ((fn == "sum" && length(cols) > 1L) || (fn %in% c("min", "max") && na_rm))
  if (data_dependent) {
    if (ctx$eager) {
      # the probe runs on the input table: every operand must be one of its
      # own columns, not one an earlier dot wrote
      if (!all(cols %in% ctx$orig_names) || any(cols %in% ctx$produced)) {
        rowwise_bail("a data-dependent aggregate of a column written by an earlier dot")
      }
      bound <- ir_bind(node, ctx$orig_schema)
      probe <- wrap_gpu_call("rowwise", gpu_compute_column(ctx$ptr, bound))
      flags <- attr(probe, "rw_flags") %||% 0L
      if (bitwAnd(flags, 3L) != 0L) {
        node <- mk("rw_agg_f64")
      }
    } else if (fn != "sum") {
      rowwise_bail("an integer min()/max(na.rm = TRUE) has a data-dependent type")
    }
  }
  rlang::call2(".cuplyr_rw", node, !!!rlang::syms(unique(cols)))
}

# The post-parse allowlist (mutate/transmute/summarise): the parsed dots of a
# rowwise call may not hold a window call (it would see one row, dplyr's result
# differs from a whole-column window) or a vector literal.
# @param expressions parse_mutate_dots()'s result
# @keywords internal
rowwise_expressions_ok <- function(expressions) {
  !is.null(expressions) && !isTRUE(attr(expressions, "has_window")) &&
    !isTRUE(attr(expressions, "has_vec"))
}

# =============================================================================
# cur_group_id(): the row number
# =============================================================================

# `.data` (materialized) with an INT32 column `helper` holding each row's
# number, which is what cur_group_id() is on a rowwise table. Same shape as
# substitute_cur_group_id()'s other branch: every field of `.data` kept, only
# the data and schema change.
# @keywords internal
rowwise_with_row_ids <- function(.data, helper) {
  n <- dim(.data)[[1L]]
  ids <- tbl_gpu(tibble::tibble(!!helper := seq_len(n)), lazy = FALSE)
  joined <- bind_cols(set_tbl_gpu_field(ungrouped(.data), "exec_mode", "eager"), ids)
  if (has_pending_ops(joined)) {
    joined <- compute(joined)
  }
  new_data <- .data
  new_data <- set_tbl_gpu_field(new_data, "ptr", joined$ptr)
  new_data <- set_tbl_gpu_field(new_data, "schema", joined$schema)
  new_data <- set_tbl_gpu_field(new_data, "lazy_ops", NULL)
  new_data
}

# =============================================================================
# summarise()
# =============================================================================

# summarise() of a rowwise table: one output row per input row, the rowwise
# vars first, then one column per dot (each must have one value per row). That
# is transmute() of the rewritten dots, followed by dplyr's `.groups` rules for a
# rowwise input: NULL/"keep" group by the vars (with a message when summarise()
# was not told `.groups`), "drop" ungroups, "rowwise" stays rowwise; anything
# else is dplyr's error.
# @keywords internal
summarise_rowwise <- function(.data, dots, .groups, by_quo, caller_env) {
  cpu <- function(detail = NULL) {
    gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!dots, .groups = .groups, .by = !!by_quo))
    }, detail = detail)
  }
  # F77: a function of the user's with a native op's name runs in dplyr
  shadowed <- shadow_screen_text(dots)
  if (!is.null(shadowed)) {
    return(cpu(shadowed))
  }
  vars <- .subset2(.data, "groups")
  nms <- names(dots) %||% rep("", length(dots))
  if (!rlang::quo_is_null(by_quo) ||
      !(is.null(.groups) || (is.character(.groups) && length(.groups) == 1L &&
                              .groups %in% c("keep", "drop", "rowwise"))) ||
      (length(dots) == 0L && length(vars) == 0L) ||
      !all(nzchar(nms)) || anyDuplicated(nms) > 0L || any(nms %in% vars)) {
    return(cpu())
  }
  rw <- rowwise_rewrite_dots(.data, dots)
  if (is.null(rw)) {
    return(rowwise_rerun(cpu))
  }
  res <- rowwise_attempt(function() transmute_run(.data, rw, rowwise = TRUE))
  if (is.null(res)) {
    return(rowwise_rerun(cpu))
  }
  # summarise() rebuilds the frame: nothing of the input's attributes stays
  res <- strip_frame_attrs(res)
  how <- .groups %||% "keep"
  if (identical(how, "rowwise")) {
    return(res)
  }
  if (identical(how, "keep")) {
    if (is.null(.groups) && length(vars) > 0L && summarise_should_inform(caller_env)) {
      inform_rowwise_regrouped(vars)
    }
    return(rowwise_to_grouped(res, vars))
  }
  rowwise_to_grouped(res, character())
}

# "a", "a and b", "a, b, and c" (cli's list style, as dplyr's messages use)
and_list <- function(x) {
  n <- length(x)
  if (n <= 2L) paste(x, collapse = " and ") else
    paste0(paste(x[-n], collapse = ", "), ", and ", x[[n]])
}

# dplyr's message for the implicit "keep" of a rowwise summarise().
# @keywords internal
inform_rowwise_regrouped <- function(groups) {
  rlang::inform(c(
    "`summarise()` has converted the output from a rowwise data frame to a grouped data frame.",
    i = "Summaries were computed rowwise.",
    i = paste0("Output is grouped by ", and_list(groups), "."),
    i = "Use `summarise(.groups = \"keep\")` to silence this message."
  ))
}
