# Expression IR - rlang quosure -> IR -> (eventually) cudf::compute_column
#
# See scratchpad/phase1_expression_engine.md, section 1, for the full design
# authority this file implements verbatim (node shapes 1.1, parse rule 1.2,
# type inference 1.3) and section 6 for the ir_call_registry extension
# contract that T2/T3 will consume via each entry's `lower` field.
#
# This file is dormant in Phase 1 task T1: nothing else in the package calls
# ir_parse_quo()/ir_infer_type()/etc. yet. filter.R/mutate.R keep using their
# existing parsers until T3/T5 cut them over.

# -----------------------------------------------------------------------------
# Node Constructors
# -----------------------------------------------------------------------------

#' Create an IR column-reference node
#'
#' @param name Character scalar, the column name
#' @return A `cuplyr_ir_col` node: `list(kind = "col", name = <chr1>)`. Gains
#'   an `index` field (0-based) only after [ir_bind()] runs.
#' @keywords internal
ir_col <- function(name) {
  node <- list(kind = "col", name = name)
  class(node) <- c("cuplyr_ir_col", "cuplyr_ir")
  node
}

#' Create an IR literal node
#'
#' @param value Atomic value (length 1, or length > 1 only for a
#'   set-argument literal such as `%in%`'s RHS), or `NULL`
#' @param type GPU type string, or `NULL` for a bare `NA` (type is adopted
#'   from the sibling operand at build/lowering time; defaults to `FLOAT64`
#'   when inferred in isolation, see [ir_infer_type()])
#' @param na `TRUE` iff this literal is the bare, untyped `NA` constant
#' @return A `cuplyr_ir_lit` node
#' @keywords internal
ir_lit <- function(value, type, na = FALSE) {
  node <- list(kind = "lit", value = value, type = type, na = na)
  class(node) <- c("cuplyr_ir_lit", "cuplyr_ir")
  node
}

#' Create an IR function-call node
#'
#' @param op Canonical cuplyr op name (not necessarily the R function name
#'   that produced it -- see [ir_call_registry] and `ir_op_alias()`, e.g.
#'   `ceiling` normalizes to `ceil`)
#' @param args List of IR nodes (the call's arguments, already parsed)
#' @param meta Forward-compatibility slot for future lowering metadata
#'   (e.g. window-function args); unused in Phase 1
#' @return A `cuplyr_ir_call` node
#' @keywords internal
ir_call <- function(op, args, meta = list()) {
  node <- list(kind = "call", op = op, args = args, meta = meta)
  class(node) <- c("cuplyr_ir_call", "cuplyr_ir")
  node
}

#' Create an IR vector-literal node (mutate()/transmute() only)
#'
#' A whole-dot constant vector (`mutate(y = 1:5)`): its length is the table's
#' row count (or the call is an error), so it is uploaded as a column at
#' lowering time (`lower_mutate()`), never evaluated by the GPU expression
#' engine. Only ever the top node of a mutate expression record.
#'
#' @param value Atomic vector (names stripped), length != 1
#' @param type GPU type string
#' @param quo The named quosure of the dot, used to raise dplyr's own error
#'   when the length doesn't fit
#' @param name Output column name
#' @return A `cuplyr_ir_vec` node
#' @keywords internal
ir_vec <- function(value, type, quo, name) {
  structure(
    list(kind = "vec", value = value, type = type, quo = quo, name = name),
    class = c("cuplyr_ir_vec", "cuplyr_ir")
  )
}

#' Build an IR literal node from an already-evaluated R value
#'
#' Implements the length/NA rules from section 1.2: a bare, untyped `NA`
#' (the logical `NA` constant) becomes `na = TRUE` with `type = NULL`
#' (adopts the sibling operand's type later; [ir_infer_type()] defaults it
#' to `FLOAT64` when asked in isolation). Any other length-1 atomic value is
#' typed via [gpu_type_from_r()]. Length != 1 is only legal when
#' `allow_vector = TRUE` (the `%in%` RHS set-argument position); otherwise
#' it's a diagnosable user error.
#'
#' A length-0 vector is legal in the `allow_vector = TRUE` position too:
#' `x %in% numeric(0)` (or `character(0)`, etc.) is well-defined, ordinary R
#' -- always `FALSE`, for every element including `NA` -- not a user error,
#' so it must not be rejected here (T4 correctness finding: this used to
#' `stop()` for *any* length-0 vector, `allow_vector` or not, which broke
#' `filter(x %in% <empty vector>)` parity with dplyr; the `%in%` handler in
#' `src/expr_eval.hpp` special-cases an empty/`NULL` RHS set to "always
#' FALSE" to match).
#'
#' @param value An R value (atomic length 1, an atomic vector, or `NULL`)
#' @param allow_vector `TRUE` iff a length != 1 vector is acceptable here
#'   (used for `%in%`'s RHS)
#' @return A `cuplyr_ir_lit` node
#' @keywords internal
ir_lit_from_r <- function(value, allow_vector = FALSE) {
  if (is.null(value)) {
    return(ir_lit(value = NULL, type = NULL, na = FALSE))
  }

  # A plain scalar (no class or other attribute) is neither a tbl_gpu nor a
  # timestamp and has the type of its storage; a NA logical is the bare NA
  # below. (Checked first: it is what a constant in an expression almost always is.)
  if (is.atomic(value) && length(value) == 1L && is.null(attributes(value)) &&
        !(is.logical(value) && is.na(value))) {
    type <- switch(typeof(value), double = "FLOAT64", integer = "INT32", character = "STRING",
                   logical = "BOOL8", NULL)
    if (!is.null(type)) {
      return(ir_lit(value = value, type = type, na = FALSE))
    }
  }

  if (is_tbl_gpu(value)) {
    # length() of a tbl_gpu is its column count, so it must not be mistaken
    # for a (scalar) literal.
    stop("A tbl_gpu can't be used as a value inside an expression.", call. = FALSE)
  }

  # An integer-backed Date/POSIXct is still a timestamp; give it double
  # storage like the uploaded columns have.
  value <- timestamp_as_double(value)

  n <- length(value)

  if (n != 1) {
    if (!allow_vector) {
      # Classed so mutate()/transmute() can route a nested vector literal
      # (`x + 1:5`) to the CPU fallback instead of failing; every other
      # caller still sees an ordinary error with this message.
      stop(structure(
        class = c("cuplyr_vector_literal", "error", "condition"),
        list(message = paste0(
          "Expected a numeric scalar, but got a value of length ", n,
          ". (Vector literals are only supported on the right-hand side ",
          "of %in%.)"
        ), call = NULL)
      ))
    }
    return(ir_lit(value = value, type = gpu_type_from_r(value), na = FALSE))
  }

  if (is.logical(value) && is.na(value)) {
    return(ir_lit(value = NULL, type = NULL, na = TRUE))
  }

  ir_lit(value = value, type = gpu_type_from_r(value), na = FALSE)
}

# -----------------------------------------------------------------------------
# Parsing: rlang quosure -> IR (or NULL for "unsupported, caller should
# fall back")
# -----------------------------------------------------------------------------

# Canonical op-name aliasing, applied to the R function name before registry
# lookup (section 1.1: "op is a canonical cuplyr op name, not an R function
# name: the parser normalizes (ceiling -> ceil, ...)").
ir_op_alias <- function(fn_name) {
  switch(fn_name, ceiling = "ceil", fn_name)
}

# The formal names of the R function a registry op is named after (base,
# dplyr, stats or lubridate, first found), read from the function itself
# (`args()` gives a primitive's), cached per name. NULL when no such function
# exists (an op name only cuplyr builds) or it has no formals list.
# @keywords internal
ir_op_formals_cache <- new.env(parent = emptyenv())
ir_op_formals <- function(fn_name) {
  cached <- ir_op_formals_cache[[fn_name]]
  if (!is.null(cached)) {
    return(if (identical(cached, NA)) NULL else cached)
  }
  fn <- get0(fn_name, envir = baseenv(), mode = "function", inherits = FALSE)
  for (ns in c("dplyr", "stats", "lubridate")) {
    if (!is.null(fn)) break
    if (requireNamespace(ns, quietly = TRUE)) {
      fn <- tryCatch(getExportedValue(ns, fn_name), error = function(e) NULL)
      if (!is.function(fn)) fn <- NULL
    }
  }
  fmls <- if (is.null(fn)) NULL else names(formals(args(fn)))
  ir_op_formals_cache[[fn_name]] <- if (is.null(fmls)) NA else fmls
  fmls
}

# TRUE iff R binds every argument of the registry call `fn_name(<args_raw>)`
# to the formal at the argument's own position, which is how the registry
# loop and its parse hooks read them. Each named argument must carry exactly
# the name of the formal at its position, before any `...`: a reordered name
# (`between(x, right = 3, left = 1)`), a partial one (`round(x, d = 1)`), an
# unknown one, or one that only exists after `...` (`coalesce(x, .ptype = 0)`,
# `if_else(c, 1, 0, ptype = )`) is FALSE, as is a positional argument that R
# would put in `...` or reject (`round(x, 1, 2)`, `log(x, 2, 3)`). FALSE means
# the call runs in dplyr, which matches by name. Base R's arithmetic
# primitives ignore names, so a reordered `-`(e2 =, e1 =) going to dplyr is
# only slower. A call with unnamed arguments only and a fixed arity or fewer
# than three arguments is TRUE without looking anything up.
# @keywords internal
ir_args_bind_by_position <- function(fn_name, args_raw, arity) {
  arg_names <- names(args_raw)
  n_args <- length(args_raw)
  if (is.null(arg_names) && (n_args < 3L || !is.na(arity))) {
    return(TRUE)
  }
  fmls <- ir_op_formals(fn_name)
  if (is.null(fmls)) {
    return(is.null(arg_names) || !any(nzchar(arg_names)))
  }
  dots <- match("...", fmls)
  positional <- if (is.na(dots)) fmls else fmls[seq_len(dots - 1L)]
  if (n_args > length(positional) && !identical(dots, 1L)) {
    return(FALSE)
  }
  if (is.null(arg_names)) {
    return(TRUE)
  }
  named <- which(nzchar(arg_names))
  if (identical(dots, 1L) && is.primitive(get0(fn_name, envir = baseenv(), inherits = FALSE))) {
    # base R's max()/min() take the data in `...` and ignore a name there
    # (`max(x = i)` is max(i)); only a formal's own name (`na.rm`) binds
    return(!any(arg_names[named] %in% fmls))
  }
  all(named <= length(positional)) && identical(unname(arg_names[named]), positional[named])
}

# TRUE iff `expr` is `.data$name` or `.data[["name"]]`.
is_data_pronoun_call <- function(expr) {
  is.call(expr) &&
    length(expr) == 3 &&
    (identical(expr[[1]], as.name("$")) || identical(expr[[1]], as.name("[["))) &&
    identical(expr[[2]], as.name(".data"))
}

# TRUE iff `expr` is `.env$name` or `.env[["name"]]`.
is_env_pronoun_call <- function(expr) {
  is.call(expr) &&
    length(expr) == 3 &&
    (identical(expr[[1]], as.name("$")) || identical(expr[[1]], as.name("[["))) &&
    identical(expr[[2]], as.name(".env"))
}

# Extract the bare name out of a `.data$name`/`.data[["name"]]` (or `.env`
# equivalent) call's third element, which is a symbol for `$` and a string
# for `[[`.
pronoun_call_name <- function(expr) {
  as.character(expr[[3]])
}

# TRUE iff `expr` contains a symbol matching a schema column name anywhere
# in its subtree (used to decide whether an unrecognized call's subtree can
# be constant-folded per section 1.2's governing rule). A forced `.data`
# reference anywhere in the subtree is conservatively always treated as a
# column reference, even if the referenced column is missing (an
# unsupported call is not the place to raise the "not found" diagnostic;
# the caller falls back instead).
expr_has_column_ref <- function(expr, schema) {
  # Two exact shortcuts for an expression of calls, symbols and constants,
  # both from C walks of its names (all.names(): every symbol, call heads
  # included; all.vars(): those that are not a call head):
  # - the walk can only take a symbol it finds in the expression, so one with
  #   no column name and no `.data` in it is FALSE;
  # - with no `.data`/`.env` pronoun, function literal or formula/quosure (the
  #   cases the walk treats specially), a non-head symbol that is a column is
  #   what the walk finds.
  if (is.call(expr)) {
    nm <- all.names(expr)
    if (!any(nm %in% schema$names) && !".data" %in% nm) {
      return(FALSE)
    }
    if (!any(c(".data", ".env", "function", "~") %in% nm) && any(all.vars(expr) %in% schema$names)) {
      return(TRUE)
    }
  }
  expr_has_column_ref_walk(expr, schema)
}

# The recursive walk behind expr_has_column_ref().
# @keywords internal
expr_has_column_ref_walk <- function(expr, schema) {
  # An empty/missing argument slot (e.g. the elided row-index in `x[, 1]`)
  # is a special R sentinel: forcing it (is.symbol()/is.call(), even just
  # reading it as a plain value) always raises "argument ... is missing,
  # with no default", regardless of context -- rlang::is_missing() is the
  # one check that inspects it safely. Found via Phase 11 task L1 test
  # coverage (mutate(y = scale(x)[, 1]), which reaches this walker once
  # `[` isn't found in ir_call_registry) -- a pre-existing landmine, not
  # something L1 itself introduces (see the identical guard in
  # R/across.R's raw-expression walkers).
  if (rlang::is_missing(expr)) {
    return(FALSE)
  }
  if (is.symbol(expr)) {
    return(as.character(expr) %in% schema$names)
  }
  if (is_data_pronoun_call(expr)) {
    return(TRUE)
  }
  # `.env$b` / `.env[["b"]]` read the environment even when a column is named b.
  if (is_env_pronoun_call(expr)) {
    return(FALSE)
  }
  # A quosure embedded in the tree (`{{ arg }}`, `!!quo`) is a `~` call to R;
  # look through it to its expression. An unsupplied `{{ arg }}` wraps the
  # missing-argument sentinel, which the guard above answers.
  if (rlang::is_quosure(expr)) {
    return(expr_has_column_ref_walk(rlang::quo_get_expr(expr), schema))
  }
  if (is.call(expr)) {
    # unclass(): the subsetting below must not dispatch on a formula-like
    # class, whose `[` method would force a missing-argument element.
    parts <- unclass(as.list(expr))
    # A function symbol is looked up as a function: a column that happens to
    # share its name (`mutate(c = 1, y = x %in% c("a", "b"))`) is skipped by
    # R's own call evaluation, so it is not a column reference.
    if (is.symbol(parts[[1]])) {
      parts <- parts[-1]
    }
    return(any(vapply(parts, expr_has_column_ref_walk, logical(1), schema = schema)))
  }
  FALSE
}

# -----------------------------------------------------------------------------
# Impure column-free calls (F75)
#
# dplyr evaluates a dot once per group (summarise(), grouped mutate()/filter()),
# so a column-free call in it (`bump()`, `runif(1)`) runs once per group and
# each group sees its own value. The native path evaluates such a call once, in
# R, while parsing (ir_parse_expr() folds it) and broadcasts the value. That is
# only the same thing when the call is deterministic and has no side effects,
# which cuplyr cannot prove for an arbitrary function. So it decides from the
# syntax alone, evaluating nothing: a column-free call is PURE iff its head
# resolves, in the dot's environment, to the identical function object as the
# canonical one (base, or an already-loaded dplyr/stats/lubridate/utils/methods/
# data.table/cuplyr/tidyselect export) of an allowlisted name. A call that reads
# a column is never folded, so its head does not matter. Purity is by name: an S3 method
# or an active binding behind an allowlisted name can hide a side effect.
# -----------------------------------------------------------------------------

# Heads that are syntax: their contents are checked, the node itself is not.
# (An inlined `.fns` lambda never reaches this: ir_impure_walk_fns() walks its
# body with its formals bound. Any other `~` or `function` is only syntax.)
ir_transparent_heads <- c("(", "{", "~", "function", "if")

# Allowlisted names of deterministic, side-effect free functions, besides the
# registry ops, the foldable window reducers and what the native `.cols`
# resolver accepts (ir_pure_names()). A higher-order function (one that calls a
# function it is given: with_order(), sapply(), Reduce(), do.call(), ...) is
# never pure: the function may be a user closure passed by name, which the walk
# cannot see. across()/if_any()/if_all() call their `.fns`, but only as a whole
# dot, where the expansion inlines the call over a column (ir_across_fns_root());
# where() calls its predicate once per column, as tidyselect does in dplyr.
ir_pure_extra_names <- c(
  # constructors, subsetting, shape
  "c", ":", "seq", "seq_len", "seq_along", "rep", "rep_len", "rev", "sort", "order", "unique",
  "length", "lengths", "nrow", "ncol", "names", "levels", "nlevels", "$", "[[", "[", "list", "numeric",
  "integer", "character", "logical", "which", "any", "all", "diff", "setNames", "is.null",
  "head", "tail", "match", "unlist", "as.vector", "append", "union", "intersect", "setdiff",
  "is.element", "format", "getOption", "suppressWarnings", "suppressMessages",
  # coercion
  "as.numeric", "as.integer", "as.double", "as.character", "as.logical", "as.factor", "factor",
  "as.Date", "as.POSIXct", "ISOdate", "ISOdatetime", "Sys.Date",
  # math and summaries
  "max", "min", "sum", "mean", "median", "quantile", "range", "prod", "sd", "var", "cor", "cov",
  "pmin", "pmax", "abs", "sqrt", "exp", "log", "log1p", "expm1", "floor", "ceiling", "trunc",
  "round", "signif", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
  "cumsum", "cumprod", "cummax", "cummin", "xor", "isTRUE", "isFALSE", "is.finite",
  "is.infinite", "is.numeric", "is.character", "is.logical", "is.na", "is.nan",
  "%in%", "%%", "%/%", "&&", "||", "!", "&", "|",
  # strings
  "paste", "paste0", "sprintf", "nchar", "toupper", "tolower", "substr", "substring", "grep",
  # control
  "ifelse", "identity", "if_else", "switch", "case_when", "case_match", "recode", "coalesce", "between",
  "na_if", "desc", "order_by", "recode_values", "replace_values", "when_all", "when_any",
  # dplyr context and column-wise helpers (their column selections are skipped)
  "n", "cur_group_id", "cur_group", "cur_group_rows", "cur_data", "cur_data_all", "cur_column",
  "row_number", "min_rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead",
  "across", "if_any", "if_all", "c_across", "pick",
  # tidyselect helpers (in `.cols`)
  "everything", "last_col", "starts_with", "ends_with", "contains", "matches", "num_range",
  "all_of", "any_of", "one_of", "where", "group_cols",
  # lubridate parsers and constructors
  "ymd", "ydm", "mdy", "myd", "dmy", "dym", "ymd_hms", "ymd_hm", "ymd_h", "mdy_hms", "dmy_hms",
  "make_date", "make_datetime", "days", "weeks", "months", "years", "hours", "minutes",
  "seconds", "period", "duration", "ddays", "dweeks", "dhours", "dminutes", "dseconds",
  "today", "as_date", "as_datetime"
)

# The canonical function objects of a pure name: the ones R finds in base, then
# in the namespaces below (the lookup order of ir_op_formals(), plus cuplyr's own
# exports, e.g. `roll_mean()`, data.table's `froll*()` and the tidyselect
# helpers dplyr re-exports). A function resolved
# from the user's environment must be identical to one. Only namespaces that are
# already loaded are consulted: a function of an unloaded namespace is not what
# the environment resolved to (except through `pkg::fn`, where resolving it
# loads the namespace, as R itself does), and looking one up must not load
# lubridate or data.table as a side effect. The answer is memoised per name
# together with the namespaces that were not loaded then; it holds until one of
# them loads, so the common lookup costs one isNamespaceLoaded() per namespace
# that is still unloaded.
# @keywords internal
ir_canonical_memo <- new.env(parent = emptyenv())
ir_canonical_namespaces <- c("dplyr", "stats", "lubridate", "utils", "methods", "data.table", "cuplyr",
                             "tidyselect")
ir_canonical_fns <- function(name) {
  hit <- ir_canonical_memo[[name]]
  if (!is.null(hit)) {
    fresh <- TRUE
    for (ns in hit$unloaded) {
      if (isNamespaceLoaded(ns)) {
        fresh <- FALSE
        break
      }
    }
    if (fresh) {
      return(hit$fns)
    }
  }
  fns <- list()
  base_fn <- get0(name, envir = baseenv(), mode = "function", inherits = FALSE)
  if (!is.null(base_fn)) fns[[1L]] <- base_fn
  unloaded <- character()
  for (ns in ir_canonical_namespaces) {
    if (!isNamespaceLoaded(ns)) {
      unloaded <- c(unloaded, ns)
      next
    }
    fn <- tryCatch(getExportedValue(ns, name), error = function(e) NULL)
    if (is.function(fn)) fns[[length(fns) + 1L]] <- fn
  }
  ir_canonical_memo[[name]] <- list(fns = fns, unloaded = unloaded)
  fns
}

# dplyr's context and column-wise functions. across()/if_any()/if_all()/pick()
# are recognised by name in dplyr 1.2.1 (expanded before evaluation), so they
# work with dplyr unattached; n(), cur_*() and c_across() must resolve from the
# environment (dplyr attached or `dplyr::`), or dplyr itself fails. Taking them
# as pure when they do not resolve changes nothing: the call fails either way.
# tidyselect supplies its own helpers to a selection when the environment has
# none (`starts_with()` with neither dplyr nor tidyselect attached).
ir_mask_provided_names <- c("n", "cur_group_id", "cur_group", "cur_group_rows", "cur_data",
                            "cur_data_all", "cur_column", "across", "if_any", "if_all",
                            "c_across", "pick", "everything", "last_col", "starts_with",
                            "ends_with", "contains", "matches", "num_range", "all_of",
                            "any_of", "one_of", "where", "group_cols")

# TRUE iff the function `name` (from namespace `pkg`, or looked up from `env`
# when `pkg` is NULL) is an allowlisted pure function.
# @keywords internal
ir_head_is_pure <- function(name, env, pkg = NULL) {
  if (!name %in% ir_pure_names()) {
    return(FALSE)
  }
  fn <- if (is.null(pkg)) {
    get0(name, envir = env, mode = "function")
  } else {
    tryCatch(getExportedValue(pkg, name), error = function(e) NULL)
  }
  if (is.null(fn)) {
    # (see ir_mask_provided_names)
    return(is.null(pkg) && name %in% ir_mask_provided_names)
  }
  for (canon in ir_canonical_fns(name)) {
    if (identical(fn, canon)) {
      return(TRUE)
    }
  }
  FALSE
}

# (built once: the registry is fixed at load time). Everything the native
# `.cols` resolver (selection_is_type_only(), R/select.R) accepts is pure: its
# type tests and its name builders, so a `.cols` of `where(~ is.double(.x))` or
# `any_of(colnames(lk))` stays native. with_order() is in ir_pre_registry_fns
# (the parser dispatches on it) but calls its `fun`: not pure.
ir_pure_names_cache <- new.env(parent = emptyenv())
ir_pure_names <- function() {
  cached <- ir_pure_names_cache$names
  if (is.null(cached)) {
    cached <- unique(c(set_diff(names(ir_call_registry), ir_internal_ops), "ceiling",
                       ir_foldable_window_fns, ir_pure_extra_names,
                       set_diff(ir_pre_registry_fns, c(".cuplyr_rw", "with_order")),
                       base_type_test_names, "inherits", name_builder_fns))
    ir_pure_names_cache$names <- cached
  }
  cached
}

# Lambda formals. The names a lambda binds stand for a column only where
# cuplyr's across() expansion replaces them with one: in the `.fns` of an
# across()/if_any()/if_all() that is a whole dot (expand_across_dots() expands
# only those), when parse_single_fn() inlines the lambda. `f(.x)` there becomes
# `f(<column>)`, which the parser never folds. Anywhere else the expansion
# substitutes nothing, so the names are ordinary variables: the `.x` of a
# two-sided `cond ~ value` (case_when()), the formal of a function literal
# passed to with_order(), a lambda in a nested across() (which goes to dplyr
# whole). A call over them is column-free there, and checked.

# The formals a `.fns` lambda binds when the expansion inlines it: `.x` and `.`
# of a one-sided formula without `.y`/`..N`/return() (across_formula_inlinable();
# as_function() also binds `.y`/`..N`, but such a formula is not inlined), the
# one formal of a function literal without return(). NULL for anything else (not
# a lambda, or one the expansion hands to dplyr with the whole call).
# @keywords internal
ir_inlined_lambda_formals <- function(fn) {
  if (is_formula_expr(fn)) {
    return(if (across_formula_inlinable(fn)) c(".x", ".") else NULL)
  }
  if (is_lambda_expr(fn) && length(fn[[2L]]) == 1L && !"return" %in% all.names(fn[[3L]])) {
    return(names(fn[[2L]]))
  }
  NULL
}

# For a dot that is an across()/if_any()/if_all() call the expansion inlines
# lambdas of: the call matched to its prototype as expand_one_across()/
# expand_if_any_all() match it, and the position of `.fns` in it. NULL when the
# dot is anything else: another call, a named across() (a data-frame column,
# which goes to dplyr), extra `...` arguments or a `.unpack` (no lambda is
# inlined then), or a call the prototype does not match.
# @keywords internal
ir_across_fns_root <- function(expr, named) {
  fn <- call_head_name(expr)
  if (is.null(fn) || !fn %in% c("across", "if_any", "if_all") || (named && fn == "across")) {
    return(NULL)
  }
  across <- fn == "across"
  matched <- tryCatch(rlang::call_match(expr, if (across) across_proto else if_any_all_proto,
                                        defaults = FALSE),
                      error = function(e) NULL)
  if (is.null(matched)) {
    return(NULL)
  }
  arg_names <- names(matched) %||% character(length(matched))
  if (!all(arg_names[-1L] %in% c(".cols", ".fns", ".names", if (across) ".unpack")) ||
      (across && ".unpack" %in% arg_names && !identical(matched$.unpack, FALSE))) {
    return(NULL)
  }
  fns_at <- match(".fns", arg_names)
  if (is.na(fns_at)) {
    return(NULL)
  }
  list(call = matched, fns_at = fns_at)
}

# One bottom-up walk of `expr` (in `env`): returns 2 when the subtree reads a
# column (`cols`: the schema names), 1 when it reads only `formals` (the bound
# formals of an enclosing inlined `.fns` lambda), else 0, and sets `st$impure`
# when a call that reads neither and whose head is not pure turns up. `fns_at`
# are the positions of `expr`'s arguments that are `.fns` (ir_impure_walk_fns()).
# A formal stands for a column (see "Lambda formals" above): `f(.x)` in an
# inlined `.fns` lambda is a call over a column, while `.x + bump()` is still
# caught at `bump()`. A `~` or `function` anywhere else is syntax, like `(`.
# The `.cols` of across()/if_any()/if_all() is walked like any argument: dplyr
# evaluates it once, but so does cuplyr's expansion, and a CPU exit after the
# expansion hands dplyr the original call, which evaluates it again; the
# tidyselect helpers are pure. pick() and c_across() send the call to dplyr
# before cuplyr evaluates their selection: they count as reading a column and
# their arguments are not walked, so neither they nor a call over them
# (`rowSums(pick(x, y))`) is reported. An embedded quosure keeps its own env.
# Known limits: a default argument of a `function` literal is not walked.
# @keywords internal
ir_impure_walk <- function(expr, env, cols, st, formals = character(), fns_at = integer()) {
  if (st$impure) {
    return(2L)
  }
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    return(if (nm %in% formals) 1L else if (nm %in% cols) 2L else 0L)
  }
  if (inherits(expr, "quosure")) {
    if (rlang::quo_is_missing(expr)) {
      return(0L)
    }
    return(ir_impure_walk(rlang::quo_get_expr(expr), rlang::quo_get_env(expr), cols, st, formals))
  }
  if (!is.call(expr)) {
    return(0L)
  }
  if (is_data_pronoun_call(expr)) {
    return(2L)
  }
  if (is_env_pronoun_call(expr)) {
    return(0L)
  }
  # `pkg::fn` names a function (as the argument of with_order(), across()); it
  # is not a call of one
  if (length(expr) == 3L && (identical(expr[[1L]], as.name("::")) || identical(expr[[1L]], as.name(":::")))) {
    return(0L)
  }
  parts <- unclass(as.list(expr))
  head <- parts[[1L]]
  head_name <- NULL
  head_pkg <- NULL
  if (is.symbol(head)) {
    head_name <- as.character(head)
  } else if (is.call(head) && length(head) == 3L && is.symbol(head[[3L]]) &&
             (identical(head[[1L]], as.name("::")) || identical(head[[1L]], as.name(":::"))) &&
             (is.symbol(head[[2L]]) || is.character(head[[2L]]))) {
    head_name <- as.character(head[[3L]])
    head_pkg <- as.character(head[[2L]])
  }
  reads <- 0L
  if (is.null(head_name)) {
    # `obj$fn()`, a function literal or an injected function object
    if (is.call(head)) {
      reads <- ir_impure_walk(head, env, cols, st, formals)
    }
  }
  if (!is.null(head_name) && head_name %in% c("pick", "c_across")) {
    # reads columns, and sends the call to dplyr whole (deferred_fallback_fns)
    return(2L)
  }
  for (i in seq_along(parts)[-1L]) {
    if (identical(parts[[i]], quote(expr = ))) {
      next
    }
    reads <- max(reads, if (length(fns_at) > 0L && i %in% fns_at) {
      ir_impure_walk_fns(parts[[i]], env, cols, st, formals)
    } else {
      ir_impure_walk(parts[[i]], env, cols, st, formals)
    })
    if (st$impure) {
      return(2L)
    }
  }
  if (!is.null(head_name) && head_name %in% ir_transparent_heads && is.null(head_pkg)) {
    # syntax: the node is not a call to check
    return(reads)
  }
  if (reads == 0L && (is.null(head_name) || !ir_head_is_pure(head_name, env, head_pkg))) {
    st$impure <- TRUE
    st$call <- expr
  }
  reads
}

# Walk one `.fns` argument (or one element of a `list()` of them) as
# parse_across_fns()/parse_single_fn() read it: a quosure (`{{ fn }}`) is looked
# through to its own expression and env, an inlined lambda's body is walked with
# its formals bound, a `list()`'s elements are `.fns` themselves, and anything
# else (a value: a name, `pkg::fn`, `make_fn()`) is an ordinary expression.
# @keywords internal
ir_impure_walk_fns <- function(fns, env, cols, st, formals) {
  if (inherits(fns, "quosure")) {
    if (rlang::quo_is_missing(fns)) {
      return(0L)
    }
    return(ir_impure_walk_fns(rlang::quo_get_expr(fns), rlang::quo_get_env(fns), cols, st, formals))
  }
  bound <- ir_inlined_lambda_formals(fns)
  if (!is.null(bound)) {
    body <- if (is_formula_expr(fns)) fns[[2L]] else fns[[3L]]
    reads <- ir_impure_walk(body, env, cols, st, c(formals, bound))
    # the lambda reads a column only through a name that is not its own formal
    return(if (reads < 2L) 0L else reads)
  }
  if (identical(call_head_name(fns), "list")) {
    return(ir_impure_walk(fns, env, cols, st, formals, fns_at = seq_along(fns)[-1L]))
  }
  ir_impure_walk(fns, env, cols, st, formals)
}

# The text of the first dot (a quosure) that contains a column-free call whose
# head is not known to be pure, with the call itself; NULL when every call is
# pure. See the comment above. Evaluates nothing.
# @param dots A list of quosures
# @param schema The schema the dots are evaluated against
# @param grouped Whether the verb has groups (or `.by`); only words the text
# @keywords internal
impure_const_call_text <- function(dots, schema, grouped = TRUE) {
  cols <- schema$names
  st <- new.env(parent = emptyenv())
  st$impure <- FALSE
  dot_names <- names(dots) %||% character(length(dots))
  for (i in seq_along(dots)) {
    dot <- dots[[i]]
    root <- NULL
    if (rlang::is_quosure(dot) && !rlang::quo_is_missing(dot)) {
      root <- ir_across_fns_root(rlang::quo_get_expr(dot), named = !is.na(dot_names[i]) && nzchar(dot_names[i]))
    }
    if (is.null(root)) {
      ir_impure_walk(dot, NULL, cols, st)
    } else {
      ir_impure_walk(root$call, rlang::quo_get_env(dot), cols, st, fns_at = root$fns_at)
    }
    if (st$impure) {
      one_line <- function(x) trimws(gsub("\\s+", " ", paste(x, collapse = " ")))
      # (the dot is the notice's `expr`; the reason names the call in it)
      return(structure(
        sprintf("`%s` is not known to be pure, and dplyr evaluates it %s.",
                substr(one_line(deparse(st$call)), 1L, 40L),
                if (grouped) "once per group" else "exactly once"),
        expr = notice_expr_text(dot)
      ))
    }
  }
  NULL
}

# -----------------------------------------------------------------------------
# F77: a function of the user's that shadows a native op's name.
#
# dplyr evaluates every dot in a data mask whose parent is the quosure's env, so
# a call head is found by R's ordinary function lookup from there: a local, an
# enclosing function's or a global `abs()` wins over `base::abs()`. The IR maps a
# call by its NAME, so every expression-taking verb first screens the user's
# dots (shadow_screen_text()): when the head of any call resolves to a function
# that is none of the name's native canonical ones, the whole call runs in
# dplyr, which calls the user's function. Nothing in the parser changes. `pkg::fn`
# is resolved as written (`stats::lag` is not `dplyr::lag`). The canonical
# functions are F75's (ir_canonical_fns()), minus the homonyms below.
# -----------------------------------------------------------------------------

# Functions of another package with the name of a native op but other
# semantics: not the native op, though ir_canonical_fns() (purity) lists them.
# lubridate's accessors that data.table exports too (`second()` is integer
# seconds there, fractional in lubridate); `between()`/`first()`/`last()`
# differ in data.table; `stats::lag()` is a time-series shift.
ir_shadow_homonyms <- list(
  stats = "lag",
  data.table = c("between", "first", "last", "hour", "isoweek", "isoyear", "mday", "minute",
                 "month", "quarter", "second", "wday", "week", "yday", "year")
)
ir_shadow_homonym_names <- unlist(ir_shadow_homonyms, use.names = FALSE)
ir_shadow_homonym_set <- list2env(stats::setNames(as.list(rep(TRUE, length(ir_shadow_homonym_names))),
                                                  ir_shadow_homonym_names))

# The native canonical functions of `name`: ir_canonical_fns() minus homonyms.
# Memoised with the list ir_canonical_fns() answered from (which that memo
# keeps while no further namespace loads), so it is as fresh as that one.
# @keywords internal
ir_native_memo <- new.env(parent = emptyenv())
ir_native_canonical_fns <- function(name) {
  fns <- ir_canonical_fns(name)
  if (is.null(ir_shadow_homonym_set[[name]])) {
    return(fns)
  }
  hit <- ir_native_memo[[name]]
  if (!is.null(hit) && identical(hit$from, fns)) {
    return(hit$native)
  }
  native <- fns
  for (pkg in names(ir_shadow_homonyms)) {
    if (name %in% ir_shadow_homonyms[[pkg]] && isNamespaceLoaded(pkg)) {
      other <- tryCatch(getExportedValue(pkg, name), error = function(e) NULL)
      native <- native[!vapply(native, identical, logical(1), other)]
    }
  }
  ir_native_memo[[name]] <- list(from = fns, native = native)
  native
}

# TRUE iff the head `name` (of namespace `pkg`, or looked up from `env` when
# `pkg` is NULL) is a function that is not one of the native canonical ones of
# `name`. A name that is neither checked (in ir_pure_names()) nor has a
# canonical function is not a shadow (a user's `myfun`: the existing paths send
# a call reading a column to the CPU and fold a column-free one, F75 deciding
# purity). A bare head that does not resolve is not a shadow (dplyr fails, or
# the mask provides the name). A `pkg::fn` is resolved as R's `::` does (the
# export only, `pkg:::fn` the namespace without its imports), and one that does
# not resolve is a shadow: dplyr raises its own error for it.
# @keywords internal
ir_head_is_shadowed <- function(name, env, pkg = NULL, triple = FALSE) {
  fn <- if (is.null(pkg)) {
    get0(name, envir = env, mode = "function")
  } else if (triple) {
    tryCatch(get(name, envir = asNamespace(pkg), inherits = FALSE), error = function(e) NULL)
  } else {
    tryCatch(getExportedValue(pkg, name), error = function(e) NULL)
  }
  if (is.null(fn)) {
    return(!is.null(pkg))
  }
  homonym <- !is.null(ir_shadow_homonym_set[[name]])
  if (!homonym) {
    # the common case: a function already known as a canonical one (no
    # freshness check needed, the cached ones are real)
    for (canon in ir_canonical_memo[[name]]$fns) {
      if (identical(fn, canon)) {
        return(FALSE)
      }
    }
  }
  canonical <- ir_canonical_fns(name)
  if (length(canonical) == 0L) {
    # no canonical function loaded: checked only if natively dispatched
    return(name %in% ir_pure_names())
  }
  for (canon in if (homonym) ir_native_canonical_fns(name) else canonical) {
    if (identical(fn, canon)) {
      return(FALSE)
    }
  }
  TRUE
}

# Is `name` checked at all: natively dispatched (ir_pure_names()) or the name
# of a canonical function? Any other name is never a shadow.
# @keywords internal
ir_shadow_name_checked <- function(name) {
  length(ir_canonical_fns(name)) > 0L || name %in% ir_pure_names()
}

# The state of one screen (one verb call): the first shadowed head (`name`) and
# the head memo, keyed by name (and package) and dropped when the env changes.
# @keywords internal
ir_shadow_state <- function() {
  st <- new.env(parent = emptyenv())
  st$name <- NULL
  st$env <- NULL
  st$cache <- NULL
  st
}

# Head lookup with the per-screen memo (`st`).
# @keywords internal
ir_shadow_check_head <- function(name, env, pkg, st, triple = FALSE) {
  if (!identical(env, st$env)) {
    st$env <- env
    st$cache <- new.env(parent = emptyenv())
  }
  key <- if (is.null(pkg)) name else paste0(pkg, if (triple) ":::" else "::", name)
  hit <- st$cache[[key]]
  if (is.null(hit)) {
    hit <- ir_head_is_shadowed(name, env, pkg, triple)
    st$cache[[key]] <- hit
  }
  hit
}

ir_sym_colon2 <- as.name("::")
ir_sym_colon3 <- as.name(":::")

# `pkg` and `name` of a `pkg::name` / `pkg:::name` call (the head of a call, or
# a function argument), or NULL.
# @keywords internal
ir_qualified_parts <- function(f) {
  if (is.call(f) && length(f) == 3L && is.symbol(f[[3L]]) &&
      (identical(f[[1L]], ir_sym_colon2) || identical(f[[1L]], ir_sym_colon3)) &&
      (is.symbol(f[[2L]]) || is.character(f[[2L]]))) {
    list(pkg = as.character(f[[2L]]), name = as.character(f[[3L]]),
         triple = identical(f[[1L]], ir_sym_colon3))
  }
}

# A function given as a value (the `.fns` of across(), the `fun` of
# with_order()): a name or `pkg::name` is looked up like a call head. TRUE, with
# `st$name` set, when it is shadowed.
# @keywords internal
ir_shadow_check_fn_value <- function(f, env, st) {
  if (is.symbol(f)) {
    if (ir_shadow_check_head(as.character(f), env, NULL, st)) {
      st$name <- as.character(f)
      return(TRUE)
    }
    return(FALSE)
  }
  q <- ir_qualified_parts(f)
  if (!is.null(q) && ir_shadow_check_head(q$name, env, q$pkg, st, q$triple)) {
    st$name <- paste0(q$pkg, if (q$triple) ":::" else "::", q$name)
    return(TRUE)
  }
  FALSE
}

# Calls whose arguments are functions given by name (not calls): the plain walk
# of the arguments never looks at them.
ir_shadow_value_heads <- c("across", "if_any", "if_all", "with_order")

# Walk `expr` (in `env`), setting `st$name` (the shadowed head) at the first
# call whose head is shadowed. Skips syntax heads, pronoun calls and non-symbol
# heads; looks through embedded quosures (own env), formula and function
# literal bodies, and every argument. The `.fns` of across()/if_any()/if_all()
# and the `fun` of with_order() written as a name are heads too.
# @keywords internal
ir_shadow_walk <- function(expr, env, st) {
  if (!is.call(expr)) {
    return(invisible())
  }
  # (.subset2(): rlang gives quosures slow `[[` and length() methods)
  head <- .subset2(expr, 1L)
  start <- 2L
  if (is.symbol(head)) {
    head_name <- as.character(head)
    if (head_name == "~" && inherits(expr, "quosure")) {
      # an embedded quosure: its own expression and env
      if (!rlang::quo_is_missing(expr)) {
        ir_shadow_walk(.subset2(expr, 2L), attr(expr, ".Environment"), st)
      }
      return(invisible())
    }
    if ((head_name == "$" || head_name == "[[") && length(expr) == 3L &&
        (identical(expr[[2L]], quote(.data)) || identical(expr[[2L]], quote(.env)))) {
      # a pronoun call: only a `[[` subscript can hold a call
      start <- 3L
    } else if (!(head_name %in% ir_transparent_heads) && ir_shadow_check_head(head_name, env, NULL, st)) {
      st$name <- head_name
      return(invisible())
    } else if (head_name %in% ir_shadow_value_heads) {
      ir_shadow_walk_value_args(expr, head_name, env, st)
    }
  } else if (is.call(head)) {
    q <- ir_qualified_parts(head)
    if (!is.null(q)) {
      if (ir_shadow_check_head(q$name, env, q$pkg, st, q$triple)) {
        st$name <- paste0(q$pkg, if (q$triple) ":::" else "::", q$name)
        return(invisible())
      }
      if (q$name %in% ir_shadow_value_heads) {
        ir_shadow_walk_value_args(expr, q$name, env, st)
      }
    } else {
      # `obj$fn()`, a function literal call
      ir_shadow_walk(head, env, st)
    }
  }
  if (!is.null(st$name)) {
    return(invisible())
  }
  # a `pkg::fn` that is not a head names a function: nothing to look up
  if (length(expr) == 3L && (identical(head, ir_sym_colon2) || identical(head, ir_sym_colon3))) {
    return(invisible())
  }
  for (i in seq_along(expr)[-seq_len(start - 1L)]) {
    # (an empty argument, as in `x[, 1]`, is the missing arg: never assigned)
    if (is.call(expr[[i]])) {
      ir_shadow_walk(expr[[i]], env, st)
      if (!is.null(st$name)) {
        return(invisible())
      }
    }
  }
  invisible()
}

ir_with_order_proto <- function(order_by, fun, x, ...) NULL

# The function-valued arguments of across()/if_any()/if_all() (`.fns`, alone or
# in a list(); an embedded quosure keeps its own env) and with_order() (`fun`),
# checked like call heads.
# @keywords internal
ir_shadow_walk_value_args <- function(expr, head_name, env, st) {
  proto <- if (head_name == "across") across_proto else if (head_name == "with_order") ir_with_order_proto else if_any_all_proto
  matched <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  fns <- if (is.null(matched)) NULL else if (head_name == "with_order") matched$fun else matched$.fns
  if (is.null(fns)) {
    return(invisible())
  }
  if (inherits(fns, "quosure")) {
    env <- rlang::quo_get_env(fns)
    fns <- rlang::quo_get_expr(fns)
  }
  cands <- if (head_name != "with_order" && identical(call_head_name(fns), "list")) as.list(fns)[-1L] else list(fns)
  for (f in cands) {
    fenv <- env
    if (inherits(f, "quosure")) {
      fenv <- rlang::quo_get_env(f)
      f <- rlang::quo_get_expr(f)
    }
    if (ir_shadow_check_fn_value(f, fenv, st)) {
      return(invisible())
    }
  }
  invisible()
}

# TRUE iff a call in `expr` (a formula right-hand side or a closure body,
# evaluated in `env`) has a shadowed head.
# @keywords internal
ir_shadow_hit <- function(expr, env) {
  st <- ir_shadow_state()
  ir_shadow_walk(expr, env, st)
  !is.null(st$name)
}

# Names the screen walks a dot for whatever they resolve to: a formula or a
# qualified call can embed another env, and the others take functions as values.
ir_shadow_walk_triggers <- c("~", "::", ":::", ir_shadow_value_heads)
ir_shadow_trigger_set <- list2env(stats::setNames(as.list(rep(TRUE, length(ir_shadow_walk_triggers))),
                                                  ir_shadow_walk_triggers))
# the syntax heads that are never looked up (`~` walks)
ir_shadow_skip_set <- list2env(list(`(` = TRUE, `{` = TRUE, `function` = TRUE, `if` = TRUE))

# The detail for gpu_fallback() when a head of the dots `dots` (a list of
# quosures, as the user wrote them) is shadowed by a function of the user's, or
# NULL. Call it first in a verb, before anything is rewritten or parsed. Evaluates
# nothing (but forces promises bound to head names, as dplyr's lookup does).
#
# Cost: this runs on every verb call. A dot is first screened by the names it
# CALLS (a symbol head, found with two all.names() passes; a column or variable
# is never looked up, as dplyr never evaluates a promise bound to one). Each
# that is checked at all (ir_shadow_name_checked()) is looked up once from the
# dot's env (memoised for the whole call) and only a dot
# with a head that resolves to a shadowing function is walked, to see whether it
# really is a call head (a head's name can also be a column). A dot with a
# formula, a `pkg::` qualification, across(), if_any(), if_all() or with_order()
# is always walked: it can embed another env or name a function as a value.
# @param dots A list of quosures
# @keywords internal
shadow_screen_text <- function(dots) {
  st <- NULL
  ok <- NULL
  ok_env <- NULL
  for (dot in dots) {
    # (a quosure is `~expr` with its env as an attribute; .subset2() because
    # rlang gives quosures slow `[[` and length() methods. A missing one has
    # the empty symbol for `expr`, which must not be assigned)
    expr <- .subset2(dot, 2L)
    if (!is.call(expr)) {
      next
    }
    all <- all.names(expr)
    values <- all.names(expr, functions = FALSE)
    n_heads <- length(all) - length(values)
    if (n_heads == 0L) {
      next
    }
    # the names called: the root's head if it is the only one (all.names()
    # lists a call's head first), else those that never occur as a value,
    # unless a name occurs in both roles (then the counts tell)
    heads <- if (n_heads == 1L && is.symbol(.subset2(expr, 1L))) {
      all[[1L]]
    } else {
      only_called <- all[match(all, values, 0L) == 0L]
      if (length(only_called) == n_heads) {
        unique(only_called)
      } else {
        uniq <- unique(all)
        uniq[tabulate(match(all, uniq), length(uniq)) > tabulate(match(values, uniq, 0L), length(uniq))]
      }
    }
    env <- attr(dot, ".Environment")
    if (!is.null(ok) && !identical(env, ok_env)) {
      ok <- NULL
    }
    walk <- FALSE
    for (nm in heads) {
      if (!is.null(ir_shadow_trigger_set[[nm]])) {
        walk <- TRUE
        break
      }
      if (!is.null(ir_shadow_skip_set[[nm]]) || (!is.null(ok) && match(nm, ok, 0L) > 0L)) {
        next
      }
      memo_fns <- ir_canonical_memo[[nm]]$fns
      if (length(memo_fns) == 0L && !ir_shadow_name_checked(nm)) {
        # never a shadow (ir_head_is_shadowed()), so never looked up. That
        # also spares most names all.names() counts as heads only because they
        # sit inside a call-valued head (`fn` in `obj$fn(x)`, `v` in
        # `(function(v) v)(x)`); a checked name there still is (`c` in
        # `fs[[c]](x)`), which can force a promise dplyr leaves alone
        ok_env <- env
        ok <- c(ok, nm)
        next
      }
      fn <- get0(nm, envir = env, mode = "function")
      if (!is.null(fn)) {
        # the common case first: already known as a canonical function (no
        # freshness check needed, the cached ones are real)
        known <- FALSE
        if (is.null(ir_shadow_homonym_set[[nm]])) {
          for (canon in memo_fns) {
            if (identical(fn, canon)) {
              known <- TRUE
              break
            }
          }
        }
        if (!known && ir_head_is_shadowed(nm, env)) {
          walk <- TRUE
          break
        }
      }
      ok_env <- env
      ok <- c(ok, nm)
    }
    if (!walk) {
      next
    }
    if (is.null(st)) {
      st <- ir_shadow_state()
    }
    ir_shadow_walk(expr, env, st)
    if (!is.null(st$name)) {
      one_line <- function(x) trimws(gsub("\\s+", " ", paste(x, collapse = " ")))
      return(structure(
        sprintf("`%s`: `%s` is a function of yours (or another package's) that dplyr calls, not the native one",
                substr(one_line(rlang::quo_text(dot)), 1L, 80L), st$name),
        expr = notice_expr_text(dot)
      ))
    }
  }
  NULL
}

# Window-function names whose all-constant calls are folded in R (see
# ir_parse_expr()).
ir_foldable_window_fns <- c(
  "row_number", "min_rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
  "cumsum", "cumprod", "cummax", "cummin", "cummean", "cumall", "cumany",
  "lag", "lead", "consecutive_id",
  "sum", "mean", "min", "max", "sd", "var", "median", "n_distinct",
  "first", "last", "nth"
)

# Registry ops that only cuplyr's own rewrites build (ntile()'s final cast,
# F50's factor decode, F61's recode lookup). A user call by one of these names is an unknown
# function: constant-folded or sent to the CPU, where the user's own function
# of that name runs.
ir_internal_ops <- c("cast_int32", "factor_codes", "factor_labels", "cast_float64",
                     "rw_agg_i32", "rw_agg_f64", "recode_lookup")

# The call heads ir_parse_expr() treats specially before registry dispatch
# (everything ir_parse_pre_registry() tests `fn_name` against). A head outside
# this set goes straight to the registry; test-perf-F68b-fastpaths.R checks
# that the set holds every name the helper tests.
ir_pre_registry_fns <- c(
  ".cuplyr_rw", "&&", "||", "as.character", "as.integer", "as.numeric", "as.double",
  ir_foldable_window_fns,
  "case_when", "case_match", "recode_values", "recode", "replace_values",
  "replace_when", "when_all", "when_any", "lag", "lead", "first", "last", "nth",
  "ntile", "order_by", "with_order", "sum", "mean", "median",
  "roll_mean", "roll_sum", "roll_min", "roll_max", "roll_sd", "roll_median",
  "frollmean", "frollsum", "frollmin", "frollmax", "frollsd", "frollmedian",
  "nchar", "substr", "trimws", "paste0", "paste", "grepl", "sub", "gsub",
  "startsWith", "endsWith",
  "year", "month", "day", "hour", "minute", "second", "quarter", "yday", "wday",
  "as.Date", "as.POSIXct", "floor_date", "ceiling_date", "round_date"
)

# Returned by ir_parse_pre_registry() when no special case took the call.
ir_no_special <- new.env(parent = emptyenv())

# The special cases of a call head that precede registry dispatch in
# ir_parse_expr() (moved out of it unchanged so the common heads skip the long
# run of name tests). Returns the parse, `NULL` for unsupported, or
# `ir_no_special` when the call falls through to the registry.
ir_parse_pre_registry <- function(expr, env, schema, allow_vector, fn_name, head) {
  # F49: an IR node the rowwise rewrite (R/rowwise.R) built for a horizontal
  # aggregate, embedded as `.cuplyr_rw(<node>, <columns it reads>)`. The
  # trailing symbols only let walkers of the raw expression (all.vars(), the
  # constant-folding checks) see the columns; the node is the value.
  if (identical(fn_name, ".cuplyr_rw")) {
    return(expr[[2]])
  }

  if (fn_name %in% c("&&", "||")) {
    replacement <- if (fn_name == "&&") "&" else "|"
    stop("'", fn_name, "' is not supported in filter(); use '", replacement, "'",
         call. = FALSE)
  }

  # F50: as.character()/as.integer()/as.numeric()/as.double() of a bare
  # factor column. Registry dispatch (below) runs before constant folding, so
  # these names are intercepted only for that one shape; every other
  # argument flows on exactly as before (`as.integer(lim)` still folds).
  # Base-only assumption: the match is on the name alone (namespace already
  # stripped), so a user-masked `as.character` would get the same GPU decode.
  if (fn_name %in% c("as.character", "as.integer", "as.numeric", "as.double")) {
    converted <- ir_parse_factor_conversion(expr, schema, fn_name)
    if (!is.null(converted)) {
      return(converted)
    }
  }

  # A window call whose arguments read no column (`row_number("a")`,
  # `cumsum(2)`, `lag(5)`) runs on a length-1 constant in dplyr and the
  # result is recycled; evaluate it in R and use the value. Zero-arg
  # `row_number()`/`n()` and anything reading a column are left alone (an
  # evaluation error outside the data mask also means "not constant").
  if (fn_name %in% ir_foldable_window_fns && length(expr) > 1 &&
      !any(vapply(as.list(expr)[-1], expr_has_column_ref, logical(1), schema = schema))) {
    folded <- tryCatch(
      list(value = rlang::eval_tidy(rlang::new_quosure(expr, env), data = list())),
      error = function(e) NULL
    )
    if (!is.null(folded) && is.atomic(folded$value) && !is.null(folded$value)) {
      if (length(folded$value) != 1) {
        return(NULL)
      }
      return(ir_lit_from_r(folded$value))
    }
  }

  # case_when()'s dots are `~` formulas, not ordinary expressions the
  # generic per-arg parse loop below understands (a formula's LHS/RHS need
  # pulling apart before recursing, and `.default=` is a named value
  # argument, not a formula at all) -- so it's special-cased here, before
  # canonical/registry dispatch, the same way `&&`/`||` are. See
  # ir_parse_case_when()'s own docs for exactly what it accepts.
  if (identical(fn_name, "case_when")) {
    return(ir_parse_case_when(expr, env, schema))
  }

  # Phase 4 wave 2: case_match()/recode_values()/replace_values()/
  # replace_when()'s dots are ALSO `~` formulas (case_match()/
  # recode_values() additionally take a leading `.x`/`x` vector argument
  # before the formulas), and when_all()/when_any()'s dots need their
  # `na_rm=`/`size=` named controls pulled out before the remaining dots
  # are parsed as ordinary logical expressions -- none of that fits the
  # generic per-arg parse loop below, so all are special-cased here
  # exactly like case_when() above. See each parse function's own docs.
  if (identical(fn_name, "case_match")) {
    return(ir_parse_case_match(expr, env, schema))
  }
  if (identical(fn_name, "recode_values")) {
    return(ir_parse_recode_values(expr, env, schema))
  }
  if (identical(fn_name, "recode")) {
    return(ir_parse_recode(expr, env, schema))
  }
  if (identical(fn_name, "replace_values")) {
    return(ir_parse_replace_values(expr, env, schema))
  }
  if (identical(fn_name, "replace_when")) {
    return(ir_parse_replace_when(expr, env, schema))
  }
  if (fn_name %in% c("when_all", "when_any")) {
    return(ir_parse_when_reduce(expr, env, schema, fn_name))
  }

  # Phase 5 (window functions): lag()/lead() (n=/default=/order_by= named
  # controls), first()/last()/nth() (order_by=/default=/na_rm=), and
  # ntile() (x= is optional with a non-trivial default, and its
  # "supplied or not" distinction changes the desugar -- see
  # ir_parse_ntile()'s own docs) all need real argument-name-aware
  # matching the generic per-arg registry loop below can't do (it only
  # ever sees positional args, see ir_parse_shift_call()'s own docs) --
  # so, like case_when()/case_match()/etc. above, these are special-cased
  # here, before registry dispatch.
  if (fn_name %in% c("lag", "lead")) {
    return(ir_parse_shift_call(expr, env, schema, fn_name))
  }
  if (fn_name %in% c("first", "last", "nth")) {
    return(ir_parse_first_last_nth(expr, env, schema, fn_name))
  }
  if (identical(fn_name, "ntile")) {
    return(ir_parse_ntile(expr, env, schema))
  }
  if (identical(fn_name, "order_by")) {
    return(ir_parse_order_by_call(expr, env, schema))
  }
  if (identical(fn_name, "with_order")) {
    return(ir_parse_with_order_call(expr, env, schema))
  }

  # Phase 5, task W9: `sum()`'s ordinary registry entry only accepts the
  # bare 1-arg form (see ir_parse_sum_call()'s own docs for why the 2-arg
  # `sum(x, na.rm = TRUE)` shape needs real argument-NAME inspection the
  # generic per-arg registry loop below can't do) -- special-cased here
  # alongside lag()/lead()/first()/last()/nth()/ntile() above, for the
  # same reason.
  if (identical(fn_name, "sum")) {
    return(ir_parse_sum_call(expr, env, schema))
  }
  if (fn_name %in% c("mean", "median") && length(expr) == 3L) {
    return(ir_parse_mean_median_call(expr, env, schema, fn_name))
  }

  # Rolling windows (scratchpad/todo.md "Rolling-window benchmark
  # target"): cuplyr's own roll_*() names, and an alias for
  # data.table::froll*() (verified against benchmark_rolling.R's actual
  # cuplyr probe call shape) -- both need real argument-name matching
  # (na.rm=/fill=/align=/... can be omitted or out of position), so both
  # get dedicated dispatch like lag()/lead()/ntile() above. See the
  # file-level comment above ir_parse_roll_call()'s definition.
  if (fn_name %in% c("roll_mean", "roll_sum", "roll_min", "roll_max", "roll_sd", "roll_median")) {
    return(ir_parse_roll_call(expr, env, schema, fn_name))
  }
  if (fn_name %in% c("frollmean", "frollsum", "frollmin", "frollmax", "frollsd", "frollmedian")) {
    return(ir_parse_froll_call(expr, env, schema, fn_name))
  }

  # Phase 11 L5 (base-R strings): nchar()/substr()/trimws()/paste0()/
  # paste()/grepl()/sub()/gsub()/startsWith()/endsWith() all have optional
  # named controls (type=/which=/sep=/collapse=/fixed=/ignore.case=/...)
  # the generic per-arg registry loop can't validate (it only ever sees
  # positional args -- same reason lag()/lead()/ntile()/roll_*() above
  # need dedicated dispatch). toupper()/tolower() need no such dispatch
  # (fixed 1-arg signature, ordinary registry entries below).
  if (identical(fn_name, "nchar")) {
    return(ir_parse_nchar(expr, env, schema))
  }
  if (identical(fn_name, "substr")) {
    return(ir_parse_substr(expr, env, schema))
  }
  if (identical(fn_name, "trimws")) {
    return(ir_parse_trimws(expr, env, schema))
  }
  if (fn_name %in% c("paste0", "paste")) {
    return(ir_parse_paste_call(expr, env, schema, fn_name))
  }
  if (identical(fn_name, "grepl")) {
    return(ir_parse_grepl(expr, env, schema))
  }
  if (fn_name %in% c("sub", "gsub")) {
    return(ir_parse_sub_gsub(expr, env, schema, fn_name))
  }
  if (fn_name %in% c("startsWith", "endsWith")) {
    return(ir_parse_starts_ends_with(expr, env, schema, fn_name))
  }

  # Phase 11 L6 (lubridate datetime accessors): MANDATORY namespace guard
  # -- year()/month()/day()/hour()/minute()/second()/quarter()/yday()/
  # wday() are lubridate names, not base R (base R has no year()) -- go
  # native ONLY if the call's function symbol resolves (in the quosure's
  # own environment, or via an explicit `lubridate::` qualification) to
  # exactly lubridate's own function; otherwise NULL (fall back), so a
  # user's own same-named function (or lubridate not being
  # installed/attached at all) reproduces dplyr's ordinary "could not
  # find function"/user-function behavior on the CPU fallback path
  # instead of cuplyr silently misinterpreting an unrelated call. See
  # ir_require_lubridate_fn()'s own docs.
  if (fn_name %in% c("year", "month", "day", "hour", "minute", "second",
                      "quarter", "yday", "wday")) {
    if (!ir_require_lubridate_fn(head, env, fn_name)) {
      return(NULL)
    }
    return(ir_parse_lubridate_accessor(expr, env, schema, fn_name))
  }

  # F41/F46: a column-free as.Date()/as.POSIXct()/floor_date()/
  # ceiling_date()/round_date() is a constant, whatever its arguments (`tz =`,
  # `format =`, a vector set for `%in%`, a unit the native rounding doesn't
  # cover): the registry entries below only take one parsed argument. An
  # error means "not constant" (NULL, CPU), as in the generic fold further
  # down. Folded before the namespace guard, so a user's own floor_date() is
  # evaluated as it would be on dev.
  if (fn_name %in% c("as.Date", "as.POSIXct", "floor_date", "ceiling_date",
                      "round_date") &&
      length(expr) > 1 && !expr_has_column_ref(expr, schema)) {
    folded <- tryCatch(
      list(value = rlang::eval_tidy(rlang::new_quosure(expr, env))),
      error = function(e) NULL
    )
    if (is.null(folded)) {
      return(NULL)
    }
    return(ir_lit_from_r(folded$value, allow_vector = allow_vector))
  }

  # F46: lubridate floor_date()/ceiling_date()/round_date(), same guard.
  if (fn_name %in% c("floor_date", "ceiling_date", "round_date")) {
    if (!ir_require_lubridate_fn(head, env, fn_name)) {
      return(NULL)
    }
    return(ir_parse_lubridate_rounding(expr, env, schema, fn_name))
  }
  ir_no_special
}

# The `(` symbol, compared against call heads.
ir_paren_sym <- as.name("(")
ir_dollar_sym <- as.name("$")
ir_bracket2_sym <- as.name("[[")

# The recursive walker behind ir_parse_quo(). Carries the quosure's
# environment explicitly (rather than re-deriving a quosure at every
# recursive step) and an `allow_vector` flag threaded down only to the
# `%in%` RHS argument position.
ir_parse_expr <- function(expr, env, schema, allow_vector = FALSE) {
  # Case 2 first (a symbol is neither a quosure nor a call, so no case before
  # it applies to one): a bare symbol. Columns shadow the environment.
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    if (match(nm, schema$names, 0L) > 0L) {
      return(ir_col(nm))
    }
    val <- tryCatch(
      rlang::eval_tidy(rlang::new_quosure(expr, env)),
      error = function(e) stop(conditionMessage(e), call. = FALSE)
    )
    return(ir_lit_from_r(val, allow_vector = allow_vector))
  }

  # Case 0: a quosure embedded INSIDE the expression tree (as opposed to
  # the top-level quosure ir_parse_quo() itself unwraps) -- unwrap to its
  # own expr/env and recurse. This arises whenever a language tree is built
  # by unquoting an already-captured quosure into another call, e.g.
  # `rlang::expr(base::sum(!!wt, na.rm = TRUE))` where `wt <-
  # rlang::enquo(wt_arg)` (exactly what real, unmodified
  # `dplyr:::tally_n()` does for `add_tally(wt = )`/`add_count(wt = )`'s
  # internal `mutate()` call, Phase 5 task W9) -- `!!wt` splices the
  # quosure OBJECT itself into that argument position (a one-sided
  # formula, `is.call()`-true but with head `` ` ~` ``, not the wrapped
  # function's own name), not just its bare expression. Ordinary tidy-eval
  # (`rlang::eval_tidy()`) resolves nested quosures like this
  # transparently; this walker doesn't use tidy-eval for structural
  # dispatch (it inspects the raw language tree directly), so it needs this
  # explicit unwrap instead.
  if (inherits(expr, "quosure")) {
    return(ir_parse_expr(rlang::quo_get_expr(expr), rlang::quo_get_env(expr), schema, allow_vector))
  }

  # Case 1: "(" is transparent. (A symbol head is compared directly; any other
  # head, e.g. a namespaced one, takes rlang's matching.)
  is_call <- is.call(expr)
  if (is_call) {
    head <- expr[[1L]]
    is_paren <- if (is.symbol(head)) identical(head, ir_paren_sym) else rlang::is_call(expr, "(")
    if (is_paren && length(expr) == 2) {
      return(ir_parse_expr(expr[[2]], env, schema, allow_vector))
    }
  }

  # Cases 3 and 4 are the `$` / `[[` calls of the `.data` / `.env` pronouns
  # (three elements, that head): any other call skips both tests.
  if (is_call && length(expr) == 3L &&
        (identical(head, ir_dollar_sym) || identical(head, ir_bracket2_sym))) {
    # Case 3: .data[["x"]] / .data$x - forced column reference.
    if (is_data_pronoun_call(expr)) {
      nm <- pronoun_call_name(expr)
      if (!nm %in% schema$names) {
        stop("Column '", nm, "' not found.\n",
             "Available columns: ", paste(schema$names, collapse = ", "),
             call. = FALSE)
      }
      return(ir_col(nm))
    }

    # Case 4: .env$x / .env[["x"]] - forced environment lookup.
    if (is_env_pronoun_call(expr)) {
      nm <- pronoun_call_name(expr)
      val <- tryCatch(
        get(nm, envir = env, inherits = TRUE),
        error = function(e) stop("object '", nm, "' not found", call. = FALSE)
      )
      return(ir_lit_from_r(val, allow_vector = allow_vector))
    }
  }

  # Case 5: literal constant (numeric/character/logical/NA/NULL).
  if (is.null(expr)) {
    return(ir_lit_from_r(NULL))
  }
  if (is.atomic(expr) && length(expr) == 1) {
    return(ir_lit_from_r(expr, allow_vector = allow_vector))
  }

  # Case 6: call f(...).
  if (is_call) {
    head <- expr[[1]]

    if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
      fn_name <- as.character(head[[3]])  # strip namespace: pkg::fn -> fn
    } else if (is.symbol(head)) {
      fn_name <- as.character(head)
    } else {
      return(NULL)  # exotic call head (e.g. a function literal): unsupported
    }

    if (match(fn_name, ir_pre_registry_fns, 0L) > 0L) {
      special <- ir_parse_pre_registry(expr, env, schema, allow_vector, fn_name, head)
      if (!identical(special, ir_no_special)) {
        if (is.null(special)) {
          # (F88: a shape of this head the GPU has no form for, when nothing
          # inside it recorded a more precise cause first)
          fallback_cause(fallback_no_gpu_text(fn_name))
        }
        return(special)
      }
    }

    canonical <- ir_op_alias(fn_name)
    args_raw <- as.list(expr)[-1]

    # The registry reads arguments by position, R binds names first: a call
    # where the two differ is dplyr's (ir_args_bind_by_position()).
    entry <- ir_call_registry[[canonical]]
    in_registry <- !is.null(entry) && match(canonical, ir_internal_ops, 0L) == 0L
    if (in_registry && !ir_args_bind_by_position(fn_name, args_raw, entry$arity)) {
      fallback_cause(sprintf("`%s()` is not on the GPU with these named arguments.", fn_name))
      return(NULL)
    }

    # F29 A0: a rank-family call whose single argument is `desc(e)` ranks
    # `e` descending. Parse `e` as the value argument and flag the node;
    # R/window.R turns the flag into the spec's `order_desc`. Any other
    # `desc()` shape (nested, extra args) parses `desc` as an unknown call
    # and falls back as before.
    if (length(args_raw) == 1L && match(canonical, ir_rank_desc_fns, 0L) > 0L) {
      # F53: an unsupplied `{{ v }}` (`min_rank({{ v }})`, `min_rank(desc({{ v }}))`)
      # is dplyr's to evaluate: it raises its own "`x` must be a vector, not
      # absent." error through the CPU route.
      if (ir_has_missing_quosure(args_raw[[1]])) {
        return(NULL)
      }
      inner <- ir_unwrap_desc(args_raw[[1]])
      if (!is.null(inner)) {
        parsed <- ir_parse_expr(inner, env, schema)
        if (is.null(parsed)) {
          return(NULL)
        }
        # dplyr's desc() is `-xtfrm(x)`: a character `x` is ranked in the
        # session collation, which the GPU (bytes) matches only under C.
        # (A bare min_rank(s) is vec_rank(), C-ordered, so stays native.)
        if (!session_collation_is_c()) {
          type <- tryCatch(ir_infer_type(parsed, schema), error = function(e) "STRING")
          if (identical(unname(type), "STRING")) {
            return(NULL)
          }
        }
        node <- ir_call(canonical, list(parsed), meta = list(rank_desc = TRUE))
        return(node)
      }
    }

    if (in_registry) {
      if (!is.na(entry$arity) && length(args_raw) != entry$arity) {
        # wrong arg count for this op: not this shape, fall back
        fallback_cause(sprintf("`%s()` is not on the GPU with %d arguments.", fn_name, length(args_raw)))
        return(NULL)
      }

      parsed_args <- vector("list", length(args_raw))
      is_in_op <- identical(canonical, "%in%")
      all_lit <- TRUE
      for (i in seq_along(args_raw)) {
        # %in%'s RHS (2nd argument) is the one set-argument position where
        # a vector literal is legal (section 1.2/1.1).
        arg_allow_vector <- is_in_op && i == 2L
        parsed <- ir_parse_expr(args_raw[[i]], env, schema, allow_vector = arg_allow_vector)
        if (is.null(parsed)) {
          return(NULL)
        }
        parsed_args[[i]] <- parsed
        if (!identical(.subset2(parsed, "kind"), "lit")) {
          all_lit <- FALSE
        }
      }

      # F53: a pure op over literals (`k == ""` with `k` an R variable) is a
      # scalar in R; fold it here, bottom-up, so no literal-only subtree
      # reaches the evaluator. (ir_fold_literal_call() declines a call with a
      # non-literal argument, so only an all-literal call asks.)
      if (all_lit) {
        folded <- ir_fold_literal_call(canonical, parsed_args)
        if (!is.null(folded)) {
          return(folded)
        }
      }

      if (!is.null(entry$parse)) {
        return(entry$parse(parsed_args, schema))
      }
      return(ir_call(canonical, parsed_args))
    }

    # f not in the registry: constant-fold if the subtree has no column
    # reference (how c(4, 6), nrow(df), Sys.Date() work), else unsupported.
    if (!expr_has_column_ref(expr, schema)) {
      # An error here is NOT proof of a user error: dplyr evaluates the call
      # inside the data mask, where pick()/n()/cur_*() (possibly inside user
      # functions) are valid but fail out here. NULL = unsupported, so the
      # caller's CPU fallback reruns it through dplyr (which raises its own
      # error if the call really is broken).
      folded <- tryCatch(
        list(value = rlang::eval_tidy(rlang::new_quosure(expr, env))),
        error = function(e) NULL
      )
      if (is.null(folded)) {
        return(NULL)
      }
      return(ir_lit_from_r(folded$value, allow_vector = allow_vector))
    }

    # (F88: the reason a CPU-fallback notice gives, recorded where it is true)
    fallback_cause(fallback_no_gpu_text(fn_name))
    return(NULL)
  }

  NULL
}

# F53: pure registry ops that are evaluated in R when every argument is a
# scalar literal (see ir_fold_literal_call()).
ir_foldable_pure_ops <- c(
  "==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "^", "%%", "%/%",
  "&", "|", "!", "xor", "is.na", "is.nan", "abs", "sqrt", "exp", "log", "log2",
  "log10", "floor", "ceil", "round", "sin", "cos", "tan", "tolower", "toupper"
)

# The R value of a scalar literal node, or `ok = FALSE` when it has none the
# fold can use (NULL, a timestamp, a vector set, an object).
# @keywords internal
ir_lit_r_value <- function(node) {
  no <- list(ok = FALSE)
  if (!inherits(node, "cuplyr_ir") || !identical(node$kind, "lit")) {
    return(no)
  }
  if (is.null(node$type)) {
    return(if (isTRUE(node$na)) list(ok = TRUE, value = NA) else no)
  }
  v <- node$value
  if (!is.atomic(v) || length(v) != 1L || is.object(v) ||
      isTRUE(startsWith(unname(node$type), "TIMESTAMP"))) {
    return(no)
  }
  list(ok = TRUE, value = v)
}

# Evaluate the pure op `op` (ir_foldable_pure_ops) on R values with base R:
# `ok = FALSE` when R errors or warns, or the result isn't a plain scalar.
# @keywords internal
ir_eval_pure_op <- function(op, vals) {
  no <- list(ok = FALSE)
  fn <- get0(if (identical(op, "ceil")) "ceiling" else op, envir = baseenv(),
             mode = "function", inherits = FALSE)
  if (is.null(fn)) {
    return(no)
  }
  value <- tryCatch(do.call(fn, vals), error = function(e) NULL,
                    warning = function(w) NULL)
  if (!is.atomic(value) || length(value) != 1L || is.object(value)) {
    return(no)
  }
  list(ok = TRUE, value = value)
}

# F53: constant-fold a pure op whose arguments are all scalar literals, with
# R's own semantics and types (`5L %/% 2L` is an integer, `NA_character_ == "a"`
# a logical NA). Returns the literal node, or NULL when the call isn't foldable
# (an op outside the pure list, a non-literal or vector argument, a warning or
# error in R, a result that isn't a plain scalar): the call then parses as before.
# @keywords internal
ir_fold_literal_call <- function(op, args) {
  if (!op %in% ir_foldable_pure_ops || length(args) == 0L) {
    return(NULL)
  }
  # (only literal arguments have a value to fold)
  for (i in seq_along(args)) {
    if (!identical(.subset2(.subset2(args, i), "kind"), "lit")) {
      return(NULL)
    }
  }
  # `-3` stays a unary minus of the literal (its IR shape is relied on)
  if (identical(op, "-") && length(args) == 1L) {
    return(NULL)
  }
  vals <- vector("list", length(args))
  for (i in seq_along(args)) {
    got <- ir_lit_r_value(args[[i]])
    if (!got$ok) {
      return(NULL)
    }
    vals[i] <- list(got$value)
  }
  got <- ir_eval_pure_op(op, vals)
  if (!got$ok) {
    return(NULL)
  }
  value <- got$value
  # A computed NaN (`0/0`) is left to the evaluator as the calculation.
  if (is.double(value) && is.nan(value)) {
    return(NULL)
  }
  ir_lit_from_r(value)
}

# The type ir_parse_quo() just inferred for `ir` against `schema`, which the verb
# asks for again right after (make_mutate_expr()): inference is a pure function
# of the two, so the answer is reused when both are the very same objects.
# Only a successful inference is kept; any other outcome is computed again.
.ir_type_memo <- new.env(parent = emptyenv())

ir_type_memo_set <- function(ir, schema, type) {
  # `type` is a promise whose evaluation may fail: nothing is stored then.
  force(type)
  .ir_type_memo$ir <- ir
  .ir_type_memo$schema <- schema
  .ir_type_memo$type <- type
  invisible(type)
}

ir_infer_type_memo <- function(ir, schema) {
  memo <- .ir_type_memo
  if (!is.null(memo$ir) && identical(memo$ir, ir) && identical(memo$schema, schema)) {
    return(memo$type)
  }
  ir_infer_type(ir, schema)
}

#' Parse an rlang quosure into cuplyr IR
#'
#' Implements the parse rule of section 1.2: columns shadow the
#' environment; any subtree with no column reference is constant-folded by
#' evaluating it in the quosure's environment. Returns `NULL` for anything
#' the IR doesn't (yet) understand -- callers (filter/mutate parsers, from
#' T3/T5 onward) fall back to the pre-existing CPU-eval or error path.
#'
#' A parse-time type check that fails only because a factor (DICTIONARY32)
#' operand is involved signals `cuplyr_factor_needs_cpu` (see
#' [stop_factor_needs_cpu()]); that class becomes `NULL` here, so every verb
#' sends the dot to its CPU fallback, where dplyr combines the factor itself.
#'
#' @param quo An `rlang` quosure
#' @param schema List with `names`/`types` (as from [current_schema()])
#' @return An IR node, or `NULL` if unsupported
#' @keywords internal
ir_parse_quo <- function(quo, schema) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)
  # F88: a new dot has no cause yet; the parse records one where it gives up
  # (fallback_cause()), and a parse that succeeds drops one a sub-attempt left.
  col <- fallback_reasons$cur
  if (!is.null(col)) {
    col$cause <- NULL
  }
  ir <- tryCatch(
    {
      ir <- ir_parse_expr(expr, env, schema, allow_vector = FALSE)
      if (!is.null(ir) && ir_decode_meets_non_utf8(ir)) {
        ir <- NULL
      }
      # Timestamp arithmetic (`t + 3600`) parses fine and only its type check
      # says "CPU", as does every timestamp argument the GPU can't take
      # (ir_check_ts_operands(), F37); verbs infer the type after the parse
      # (outside this handler), so run it here too, plus the zone rule of
      # every POSIXct op (F32: an op without one goes to the CPU). Only the
      # timestamp signal and the typed-allowlist signal (F48,
      # ir_check_type_operands()) are acted on: a factor signal must still
      # come back as IR for ir_factor_rule(), and any other type error is
      # left for the verb to raise.
      if (!is.null(ir)) {
        tryCatch({
          ir_type_memo_set(ir, schema, ir_infer_type(ir, schema))
          ir_check_tz_rules(ir, schema)
        }, error = function(e) {
          if (inherits(e, c("cuplyr_timestamp_needs_cpu", "cuplyr_type_needs_cpu"))) stop(e)
        })
      }
      ir
    },
    cuplyr_factor_needs_cpu = function(e) {
      fallback_cause(fallback_type_text(e))
      NULL
    }
  )
  if (!is.null(ir) && !is.null(col)) {
    col$cause <- NULL
  }
  ir
}

#' Parse a `case_when()` call into IR (Phase 4)
#'
#' `case_when(cond1 ~ val1, cond2 ~ val2, ..., .default = val)` dots are
#' two-sided formulas (`~` calls), not ordinary expressions -- so unlike
#' every other `ir_call_registry` entry, this is invoked directly from
#' `ir_parse_expr()`'s call-handling branch (see the `identical(fn_name,
#' "case_when")` check there) rather than going through the registry's
#' generic per-arg parse loop. Recognizes exactly: any number of two-sided
#' `cond ~ value` clauses, plus an optional `.default = value` named
#' argument. Returns `NULL` (unsupported, caller falls back per the normal
#' contract) for anything else this Phase-4 wave doesn't implement yet:
#' `.ptype=`, `.size=`, `.unmatched=`, one-sided formulas, non-formula dots,
#' or zero clauses.
#'
#' Normalizes into a single flat `ir_call("case_when", args)` where `args`
#' is `list(cond1, val1, cond2, val2, ..., valN, default)` -- always
#' terminated by a default (a bare untyped-NA literal is synthesized when
#' the user didn't supply `.default=`, matching dplyr's "unmatched rows are
#' a typed NA" rule). `meta$n_when` records the clause count for
#' diagnostics/`show_query()`; the C++ side derives the same count from
#' `length(args)` being odd (`2 * n_when + 1`), so `meta` isn't load-bearing
#' there.
#'
#' Bare untyped `NA` literals among the value positions (any `val_i` or the
#' `.default`/synthesized default) are resolved against a sibling value's
#' type via [ir_resolve_value_nas()] -- the same value-position type
#' adoption if_else()/coalesce() use, generalized to `case_when`'s N+1
#' value slots.
#'
#' @param expr The raw `case_when(...)` call
#' @param env The quosure's environment (conditions/values are parsed
#'   against this, exactly like any other IR subtree)
#' @param schema List with `names`/`types`
#' @return An `ir_call("case_when", ...)` node, or `NULL` if unsupported
#' @keywords internal
ir_parse_case_when <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }

  # .ptype=/.size=/.unmatched=/etc.: not implemented this wave -- fall back.
  unsupported_named <- set_diff(arg_names[nzchar(arg_names)], ".default")
  if (length(unsupported_named) > 0) {
    return(NULL)
  }

  default_pos <- which(arg_names == ".default")
  default_raw <- if (length(default_pos) == 1) args_raw[[default_pos]] else NULL
  clause_raw <- if (length(default_pos) == 1) args_raw[-default_pos] else args_raw

  if (length(clause_raw) == 0) {
    return(NULL)  # nothing to match against
  }

  conds <- vector("list", length(clause_raw))
  vals <- vector("list", length(clause_raw))

  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)  # not a two-sided formula: unsupported shape
    }
    cond_ir <- ir_parse_expr(f[[2]], env, schema)
    if (is.null(cond_ir)) {
      return(NULL)
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    conds[[i]] <- cond_ir
    vals[[i]] <- val_ir
  }

  default_ir <- if (!is.null(default_raw)) {
    parsed_default <- ir_parse_expr(default_raw, env, schema)
    if (is.null(parsed_default)) {
      return(NULL)
    }
    parsed_default
  } else {
    ir_lit(value = NULL, type = NULL, na = TRUE)  # synthesized untyped NA
  }

  values <- ir_resolve_value_nas(c(vals, list(default_ir)), schema)
  n_when <- length(conds)

  interleaved <- vector("list", n_when * 2L)
  interleaved[seq(1L, n_when * 2L, by = 2L)] <- conds
  interleaved[seq(2L, n_when * 2L, by = 2L)] <- values[seq_len(n_when)]

  ir_call("case_when", c(interleaved, values[n_when + 1L]), meta = list(n_when = n_when))
}

# -----------------------------------------------------------------------------
# Phase 4 wave 2: case_match()/recode_values()/replace_values()/
# replace_when()/when_all()/when_any() -- verified empirically against
# dplyr 1.2.1 first (see the Phase 4 wave 2 task notes/scratchpad/todo.md).
#
# case_match()/recode_values(): "match .x/x by VALUE against a set of old
# values" desugars EXACTLY into `%in%` conditions feeding a `case_when()`
# chain -- `case_match(x, c(1,3) ~ "a", .default = "z")` is, at the IR
# level, indistinguishable from `case_when(x %in% c(1,3) ~ "a", .default =
# "z")`. This reuses `%in%`'s own NA-matching semantics verbatim (`NA %in%
# NA` is TRUE in base R, so `case_match(x, NA ~ "unknown", ...)` correctly
# replaces x's own NA elements) -- which required a real pre-existing `%in%`
# handler bug fix (a bare scalar `NA` literal RHS, e.g. `x %in% NA`, was
# wrongly routed into the "empty RHS set" always-FALSE shortcut instead of
# matching x's NA rows; see `src/expr_eval.hpp`'s `%in%` handler). Because
# both fully reduce to an existing "case_when" node, neither needs its own
# registry entry, `type` function, or C++ handler.
#
# replace_values()/replace_when(): same idea (value-match / boolean-cond
# clauses respectively) but TYPE STABLE to `x` (dplyr's own docs: "type
# stable, unlike case_match()") and with `x` itself as the implicit,
# non-overridable "default" for unmatched rows -- case_when()'s own type
# rule (vctrs common-type unify across every value + default) is WRONG for
# this, so both lower through a dedicated "replace_when" IR op instead (see
# `type_replace_when()` below and the `replace_when` handler in
# `src/expr_eval.hpp`).
#
# when_all()/when_any(): `when_any(x, y, z)` is `x | y | z` generalized to N
# inputs (`when_all()` the `&` analogue) -- fully desugars into a folded
# chain of the existing `|`/`&` ops, needing no IR op of its own at all.
# `na_rm = TRUE` does NOT simply drop NAs from the *result*; verified
# empirically it treats NA as each operator's identity element instead
# (`when_any(NA, NA, na_rm = TRUE)` is `FALSE`, not `NA`; `when_all(NA, NA,
# na_rm = TRUE)` is `TRUE`) -- implemented by `coalesce()`-ing every input
# against that identity value before folding, reusing coalesce()'s existing
# handler too.
#
# Every one of these follows wave 1's own scope-narrowing precedent: any
# named argument this wave doesn't implement (recode_values()'s
# `from=`/`to=`/`unmatched=`/`ptype=`, case_match()'s `.ptype=`,
# replace_values()'s `from=`/`to=`, when_all()/when_any()'s `size=`) being
# present at all makes the parse function return `NULL` (caller falls
# back/errors), exactly like case_when()'s own `.ptype=`/`.size=`/
# `.unmatched=` handling above.
# -----------------------------------------------------------------------------

# na_if()'s output keeps `x`'s own type; replace_when()/replace_values() are
# the same idea generalized to N clauses -- the result is always `x`'s own
# type (arg 1), never a vctrs-style unify across the replacement values.
type_replace_when <- function(arg_types) {
  n <- length(arg_types)
  value_idx <- c(1L, if (n >= 3L) seq(3L, n, by = 2L))  # x, val_1, ..., val_N
  ir_check_cast_to_x(arg_types[value_idx], "replace_when")
  arg_types[1]
}

# For the ops whose result is `x` with other values cast into it (na_if(),
# replace_when()): vctrs casts a Date to local midnight in a POSIXct `x`'s
# zone and rejects a POSIXct into a Date `x`, neither of which the GPU cast
# does (it reinterprets the raw ticks), so a Date + POSIXct mix goes to the
# CPU (F32). So does any other mix with a timestamp (F37): vctrs rejects a
# number into a Date `x` (`na_if(d, 19725)`) and a Date into a number `x`,
# where the GPU would compare the raw values.
# @keywords internal
ir_check_cast_to_x <- function(types, op_name) {
  if (all(c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS") %in% types)) {
    stop_timestamp_needs_cpu(op_name, "(): can't combine a Date and a POSIXct on the GPU.")
  }
  if (any(types %in% cuplyr_timestamp_types) && length(unique(unname(types))) > 1L) {
    stop_timestamp_needs_cpu(op_name, "(): can't cast between a timestamp and ",
                             "another type on the GPU.")
  }
  invisible()
}

#' Shared clause-parsing helper for case_match()/recode_values() (Phase 4 wave 2)
#'
#' Both functions share an identical clause shape once "match by value" is
#' expressed as `%in%`: `f(x, old_values_1 ~ new_value_1, ..., <default_key>
#' = value)`. Builds `x %in% old_values_i` as each clause's condition
#' (reusing the existing `%in%` op verbatim, including its NA-matching
#' semantics) and hands the interleaved cond/value list to `ir_call(
#' "case_when", ...)` directly -- case_match()/recode_values() are, at the
#' IR level, indistinguishable from case_when() once desugared this way.
#'
#' Only the two-sided-formula clause interface is implemented this wave:
#' recode_values()'s `from=`/`to=`/`unmatched=`/`ptype=` vector-lookup-table
#' interface, and case_match()'s `.ptype=`, are NOT implemented -- ANY named
#' argument other than `default_key` present in the call makes this return
#' `NULL` (caller falls back).
#'
#' A clause's LHS ("old value(s)") must resolve to a literal (a constant, a
#' `c(...)` vector, or an environment lookup) -- an LHS referencing a column
#' is unsupported this wave (mirrors `%in%`'s own RHS restriction, since
#' that's exactly the op this desugars into). A STRING-vs-non-STRING type
#' mismatch between `x` and a clause's LHS is diagnosed here (mirroring
#' `na_if()`'s own STRING compatibility check) rather than surfacing as a
#' raw cudf cast error at lowering time.
#'
#' @param expr The raw `case_match(...)`/`recode_values(...)` call
#' @param env The quosure's environment
#' @param schema List with `names`/`types`
#' @param default_key `".default"` (case_match) or `"default"`
#'   (recode_values) -- the one named control argument this wave supports
#' @return An `ir_call("case_when", ...)` node, or `NULL` if unsupported
#' @keywords internal
ir_parse_value_match_clauses <- function(expr, env, schema, default_key) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) < 1) {
    return(NULL)  # need at least `.x`/`x`
  }
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }

  # First positional (unnamed) argument is .x/x -- case_match()/
  # recode_values() both always take it first and unnamed in every
  # realistic call site (the same simplifying assumption if_else()/
  # case_when() already make for their own positional args).
  if (nzchar(arg_names[1])) {
    return(NULL)
  }
  x_raw <- args_raw[[1]]
  rest_raw <- args_raw[-1]
  rest_names <- arg_names[-1]

  unsupported_named <- set_diff(rest_names[nzchar(rest_names)], default_key)
  if (length(unsupported_named) > 0) {
    return(NULL)
  }

  default_pos <- which(rest_names == default_key)
  default_raw <- if (length(default_pos) == 1) rest_raw[[default_pos]] else NULL
  clause_raw <- if (length(default_pos) == 1) rest_raw[-default_pos] else rest_raw

  if (length(clause_raw) == 0) {
    return(NULL)  # nothing to match against
  }

  x_ir <- ir_parse_expr(x_raw, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))

  conds <- vector("list", length(clause_raw))
  vals <- vector("list", length(clause_raw))

  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)  # not a two-sided formula: unsupported shape
    }
    lhs_ir <- ir_parse_expr(f[[2]], env, schema, allow_vector = TRUE)
    if (is.null(lhs_ir) || !identical(lhs_ir$kind, "lit")) {
      return(NULL)  # non-literal (e.g. column-referencing) LHS: unsupported
    }
    if (!isTRUE(lhs_ir$na)) {
      lhs_type <- unname(ir_infer_type(lhs_ir, schema))
      if (identical(x_type, "DICTIONARY32") || identical(lhs_type, "DICTIONARY32")) {
        return(NULL)  # a factor needs dplyr's labels: CPU fallback (F18)
      }
      if (identical(x_type, "STRING") != identical(lhs_type, "STRING")) {
        stop("case_match()/recode_values(): can't convert the matched value (",
             lhs_type, ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    conds[[i]] <- ir_call("%in%", list(x_ir, lhs_ir))
    vals[[i]] <- val_ir
  }

  default_ir <- if (!is.null(default_raw)) {
    parsed_default <- ir_parse_expr(default_raw, env, schema)
    if (is.null(parsed_default)) {
      return(NULL)
    }
    parsed_default
  } else {
    ir_lit(value = NULL, type = NULL, na = TRUE)  # synthesized untyped NA
  }

  values <- ir_resolve_value_nas(c(vals, list(default_ir)), schema)
  n_when <- length(conds)

  interleaved <- vector("list", n_when * 2L)
  interleaved[seq(1L, n_when * 2L, by = 2L)] <- conds
  interleaved[seq(2L, n_when * 2L, by = 2L)] <- values[seq_len(n_when)]

  ir_call("case_when", c(interleaved, values[n_when + 1L]), meta = list(n_when = n_when))
}

#' One scalar `recode()` value, by shape
#'
#' Accepts only what the F30 allowlist means to: a length-1, attribute-free
#' atomic of typeof character/integer/double/logical (doubles finite or
#' `NA_real_`; R parses `NA`, `NA_character_`, ... as such constants, so a
#' symbol is never accepted), or unary `-` on a numeric literal. Nothing is
#' evaluated.
#'
#' @return `list(value =, type =)` (`type` is the R typeof), or `NULL`
#' @keywords internal
ir_recode_scalar <- function(a, env, schema) {
  if (is.call(a) && length(a) == 2L && identical(a[[1]], as.name("-"))) {
    inner <- ir_recode_scalar(a[[2]], env, schema)
    if (is.null(inner) || !inner$type %in% c("integer", "double") || is.na(inner$value)) {
      return(NULL)
    }
    return(list(value = -inner$value, type = inner$type))
  }
  if (!is.atomic(a) || length(a) != 1L || !is.null(attributes(a)) ||
      !typeof(a) %in% c("character", "integer", "double", "logical")) {
    return(NULL)
  }
  if (is.double(a) && !is.na(a) && !is.finite(a)) {
    return(NULL)  # Inf
  }
  if (is.double(a) && is.nan(a)) {
    return(NULL)
  }
  list(value = a, type = typeof(a))
}

#' Parse `dplyr::recode(.x, old = new, ..., .default =, .missing =)`
#'
#' A POSITIVE allowlist (F30): anything that doesn't match exactly returns
#' `NULL`, i.e. the CPU fallback, which already reproduces dplyr's results,
#' warnings and errors. Accepted: head resolving to `dplyr::recode`; `.x` a
#' bare STRING/INT32/FLOAT64 (non-factor) column; scalar-literal values of one
#' shared typeof; string keys = names, numeric keys = `as.double(names)` (whole
#' int32 for an INT32 `.x`) or positions; no duplicate keys; and, without
#' `.default`, a value type equal to `.x`'s (otherwise dplyr's data-dependent
#' "Unreplaced values treated as NA" warning must stay on the CPU). Lowers to
#' one `recode_lookup(x, keys, values, default, missing)` node (F61: the keys
#' and values are two vector literals, one hash lookup on the GPU instead of
#' a clause per key), every literal explicitly typed. See `features/F30.md`.
#' @keywords internal
ir_parse_recode <- function(expr, env, schema) {
  head <- expr[[1]]
  target <- tryCatch(getExportedValue("dplyr", "recode"), error = function(e) NULL)
  resolved <- resolve_call_fn(head, env)
  if (is.null(target) || is.null(resolved) || !identical(resolved, target)) {
    return(NULL)
  }

  args <- as.list(expr)[-1]
  if (length(args) < 2L) {
    return(NULL)
  }
  arg_names <- names(args) %||% rep("", length(args))

  if (!arg_names[1] %in% c("", ".x")) {
    return(NULL)
  }
  if (rlang::is_missing(args[[1]]) || !is.symbol(args[[1]])) {
    return(NULL)
  }
  x_name <- as.character(args[[1]])
  idx <- match(x_name, schema$names)
  if (is.na(idx) || anyDuplicated(schema$names) > 0L ||
      !is.null(schema$factor_levels[[x_name]])) {
    return(NULL)
  }
  x_gpu <- unname(schema$types[idx])
  if (!x_gpu %in% c("STRING", "INT32", "FLOAT64")) {
    return(NULL)
  }
  x_r <- c(STRING = "character", INT32 = "integer", FLOAT64 = "double")[[x_gpu]]

  rest <- args[-1]
  rnames <- arg_names[-1]
  if (any(is.na(rnames)) || any(rnames == ".x")) {
    return(NULL)
  }
  is_ctl <- rnames %in% c(".default", ".missing")
  if (anyDuplicated(rnames[is_ctl]) > 0L) {
    return(NULL)
  }
  rep_names <- rnames[!is_ctl]
  if (any(startsWith(rep_names, "."))) {
    return(NULL)  # partial match of .x, typo of .default/.missing
  }
  reps <- rest[!is_ctl]
  if (length(reps) == 0L) {
    return(NULL)
  }

  # Scalar-by-shape values; an empty argument (`recode(x, a = )`) is NULL.
  parse_val <- function(i) {
    if (rlang::is_missing(rest[[i]])) NULL else ir_recode_scalar(rest[[i]], env, schema)
  }
  rep_pos <- which(!is_ctl)
  rep_vals <- lapply(rep_pos, parse_val)
  has_default <- ".default" %in% rnames
  has_missing <- ".missing" %in% rnames
  def_val <- if (has_default) parse_val(match(".default", rnames))
  mis_val <- if (has_missing) parse_val(match(".missing", rnames))
  all_vals <- c(rep_vals, if (has_default) list(def_val), if (has_missing) list(mis_val))
  if (any(vapply(all_vals, is.null, logical(1)))) {
    return(NULL)
  }
  val_type <- all_vals[[1]]$type
  if (!all(vapply(all_vals, function(v) identical(v$type, val_type), logical(1)))) {
    return(NULL)
  }
  if (!has_default && !identical(val_type, x_r)) {
    return(NULL)
  }

  # Keys.
  named <- nzchar(rep_names)
  if (x_gpu == "STRING") {
    if (!all(named) || anyDuplicated(rep_names) > 0L) {
      return(NULL)
    }
    keys <- rep_names
  } else {
    if (any(named) && !all(named)) {
      return(NULL)
    }
    if (all(named)) {
      keys <- tryCatch(as.double(rep_names), warning = function(w) NULL)
      if (is.null(keys) || any(!is.finite(keys)) || anyDuplicated(keys) > 0L) {
        return(NULL)
      }
    } else {
      keys <- as.double(seq_along(reps))
    }
    if (x_gpu == "INT32") {
      if (any(keys != trunc(keys)) || any(abs(keys) > .Machine$integer.max)) {
        return(NULL)
      }
      keys <- as.integer(keys)
    }
  }

  # Every literal carries its own type (a bare NA would adopt a sibling's).
  lit_of <- function(v) {
    if (is.na(v)) {
      ir_lit(value = NULL, type = gpu_type_from_r(v), na = TRUE)
    } else {
      ir_lit_from_r(v)
    }
  }
  x_ir <- ir_col(x_name)
  na_of_t <- switch(val_type,
    character = NA_character_, integer = NA_integer_, double = NA_real_, logical = NA)

  # One lookup node (F61): the K keys and values are two vector literals, built
  # in one go, instead of K `==` conditions. A missing `.x` row never matches a
  # key and, with `.default`/`.missing`, is only replaced by `.missing`.
  # dplyr's is.na() also covers NaN, and so does the handler (F55). The missing
  # step is needed for a FLOAT64 `.x` even with neither argument: dplyr then
  # returns NA (not NaN) for such a row.
  na_step <- has_default || has_missing || x_gpu == "FLOAT64"
  vals <- vapply(rep_vals, function(v) v$value, FUN.VALUE = vector(val_type, 1L))
  ir_call("recode_lookup", list(
    x_ir,
    ir_lit(value = keys, type = x_gpu),
    ir_lit(value = vals, type = gpu_type_from_r(vals)),
    if (has_default) lit_of(def_val$value) else x_ir,
    lit_of(if (has_missing) mis_val$value else na_of_t)
  ), meta = list(na_step = na_step))
}

#' Parse `case_match(.x, old_values ~ new_value, ..., .default=, .ptype=)`
#'
#' See [ir_parse_value_match_clauses()] -- `case_match()` is deprecated in
#' dplyr 1.2.1 in favor of `recode_values()`/`replace_values()` (verified:
#' calling it emits a lifecycle warning), but is still exported and
#' functional, and is explicitly in this wave's scope, so it's implemented
#' identically to `recode_values()`'s formula interface, just with `.default`
#' as the named key instead of `default`.
#'
#' @keywords internal
ir_parse_case_match <- function(expr, env, schema) {
  ir_parse_value_match_clauses(expr, env, schema, default_key = ".default")
}

#' Parse `recode_values(x, old_values ~ new_value, ..., default=, unmatched=, ptype=)`
#'
#' See [ir_parse_value_match_clauses()]. `from=`/`to=`/`unmatched=`/`ptype=`
#' are not implemented this wave.
#'
#' @keywords internal
ir_parse_recode_values <- function(expr, env, schema) {
  ir_parse_value_match_clauses(expr, env, schema, default_key = "default")
}

#' Parse `replace_values(x, old_values ~ new_value, ..., from=, to=)`
#'
#' Value-match clauses like [ir_parse_value_match_clauses()] (`x %in%
#' old_values_i`), but TYPE STABLE: the result always keeps `x`'s own type,
#' and unmatched rows keep `x` itself (there is no `.default=`/`default=`
#' argument at all) -- expressed as a dedicated `ir_call("replace_when",
#' ...)` node (see `type_replace_when()` and this file's module comment
#' above) rather than reusing `case_when`'s node, since case_when's own
#' common-type-unify rule is wrong here.
#'
#' Only the two-sided-formula clause interface is implemented this wave:
#' `from=`/`to=` (the vector-lookup-table interface) being present at all
#' makes this return `NULL` (fall back).
#'
#' @keywords internal
ir_parse_replace_values <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) < 2) {
    return(NULL)  # need x + at least one clause
  }
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }
  if (nzchar(arg_names[1])) {
    return(NULL)
  }
  if (any(nzchar(arg_names[-1]))) {
    return(NULL)  # from=/to=: not implemented this wave
  }

  x_raw <- args_raw[[1]]
  clause_raw <- args_raw[-1]

  x_ir <- ir_parse_expr(x_raw, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))

  interleaved <- vector("list", length(clause_raw) * 2L)
  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)
    }
    lhs_ir <- ir_parse_expr(f[[2]], env, schema, allow_vector = TRUE)
    if (is.null(lhs_ir) || !identical(lhs_ir$kind, "lit")) {
      return(NULL)
    }
    if (!isTRUE(lhs_ir$na)) {
      lhs_type <- unname(ir_infer_type(lhs_ir, schema))
      if (identical(x_type, "DICTIONARY32") || identical(lhs_type, "DICTIONARY32")) {
        return(NULL)  # a factor needs dplyr's labels: CPU fallback (F18)
      }
      if (identical(x_type, "STRING") != identical(lhs_type, "STRING")) {
        stop("replace_values(): can't convert the matched value (", lhs_type,
             ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    if (!(identical(val_ir$kind, "lit") && isTRUE(val_ir$na) && is.null(val_ir$type))) {
      val_type <- unname(ir_infer_type(val_ir, schema))
      if (identical(x_type, "DICTIONARY32") || identical(val_type, "DICTIONARY32")) {
        return(NULL)  # a factor needs dplyr's labels: CPU fallback (F18)
      }
      if (identical(x_type, "STRING") != identical(val_type, "STRING")) {
        stop("replace_values(): can't convert a replacement value (", val_type,
             ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    interleaved[[2 * i - 1]] <- ir_call("%in%", list(x_ir, lhs_ir))
    interleaved[[2 * i]] <- val_ir
  }

  ir_call("replace_when", c(list(x_ir), interleaved), meta = list(n_when = length(clause_raw)))
}

#' Parse `replace_when(x, cond1 ~ val1, ..., condN ~ valN)`
#'
#' Boolean-condition clauses -- exactly `case_when()`'s own clause shape,
#' see [ir_parse_case_when()] -- but TYPE STABLE to `x`, with `x` itself as
#' the implicit, non-overridable final "default" (unmatched rows keep their
#' original `x` value; there is no `.default=`/`default=` argument at all).
#' Lowered via the same dedicated "replace_when" op as
#' [ir_parse_replace_values()].
#'
#' No named arguments are supported this wave (the real `replace_when()`
#' signature is just `replace_when(x, ...)` anyway) -- any named dot present
#' makes this return `NULL` (fall back).
#'
#' @keywords internal
ir_parse_replace_when <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) < 2) {
    return(NULL)  # need x + at least one clause
  }
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }
  if (nzchar(arg_names[1])) {
    return(NULL)
  }
  if (any(nzchar(arg_names[-1]))) {
    return(NULL)  # no named controls supported this wave
  }

  x_raw <- args_raw[[1]]
  clause_raw <- args_raw[-1]

  x_ir <- ir_parse_expr(x_raw, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))

  interleaved <- vector("list", length(clause_raw) * 2L)
  for (i in seq_along(clause_raw)) {
    f <- clause_raw[[i]]
    if (!rlang::is_call(f, "~") || length(f) != 3) {
      return(NULL)
    }
    cond_ir <- ir_parse_expr(f[[2]], env, schema)
    if (is.null(cond_ir)) {
      return(NULL)
    }
    val_ir <- ir_parse_expr(f[[3]], env, schema)
    if (is.null(val_ir)) {
      return(NULL)
    }
    if (!(identical(val_ir$kind, "lit") && isTRUE(val_ir$na) && is.null(val_ir$type))) {
      val_type <- unname(ir_infer_type(val_ir, schema))
      if (identical(x_type, "DICTIONARY32") || identical(val_type, "DICTIONARY32")) {
        return(NULL)  # a factor needs dplyr's labels: CPU fallback (F18)
      }
      if (identical(x_type, "STRING") != identical(val_type, "STRING")) {
        stop("replace_when(): can't convert a replacement value (", val_type,
             ") to match the type of `x` (", x_type, ").", call. = FALSE)
      }
    }
    interleaved[[2 * i - 1]] <- cond_ir
    interleaved[[2 * i]] <- val_ir
  }

  ir_call("replace_when", c(list(x_ir), interleaved), meta = list(n_when = length(clause_raw)))
}

#' Parse `when_all(..., na_rm=FALSE, size=NULL)` / `when_any(...)`
#'
#' See this file's module comment above for the full empirical rationale.
#' Fully desugars at parse time into a right-folded chain of the existing
#' `|`/`&` ops -- no new IR op, registry entry, or C++ handler needed.
#'
#' `size=` (only meaningful for the zero-input edge case) is NOT implemented
#' this wave -- its presence, or zero dots, makes this return `NULL`.
#'
#' @param fn_name `"when_all"` or `"when_any"`
#' @keywords internal
ir_parse_when_reduce <- function(expr, env, schema, fn_name) {
  args_raw <- as.list(expr)[-1]
  arg_names <- names(args_raw)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args_raw))
  }

  unsupported_named <- set_diff(arg_names[nzchar(arg_names)], "na_rm")
  if (length(unsupported_named) > 0) {
    return(NULL)  # size=: not implemented this wave
  }

  na_rm_pos <- which(arg_names == "na_rm")
  if (length(na_rm_pos) > 1) {
    return(NULL)  # dplyr's own "matched by multiple actual arguments" error
  }
  na_rm <- FALSE
  if (length(na_rm_pos) == 1) {
    na_rm <- ir_na_flag_value(args_raw[[na_rm_pos]], env, schema)
    if (is.null(na_rm)) {
      return(NULL)
    }
  }

  dot_raw <- if (length(na_rm_pos) == 1) args_raw[-na_rm_pos] else args_raw
  dot_names <- if (length(na_rm_pos) == 1) arg_names[-na_rm_pos] else arg_names
  if (any(nzchar(dot_names))) {
    return(NULL)  # dots should be positional
  }
  if (length(dot_raw) == 0) {
    return(NULL)  # zero-input edge case: not implemented this wave
  }

  fold_op <- if (identical(fn_name, "when_all")) "&" else "|"
  identity_value <- identical(fn_name, "when_all")  # TRUE for &, FALSE for |

  parsed <- vector("list", length(dot_raw))
  for (i in seq_along(dot_raw)) {
    node <- ir_parse_expr(dot_raw[[i]], env, schema)
    if (is.null(node)) {
      return(NULL)
    }
    if (na_rm) {
      node <- ir_call("coalesce", list(node, ir_lit_from_r(identity_value)))
    }
    parsed[[i]] <- node
  }

  Reduce(function(a, b) ir_call(fold_op, list(a, b)), parsed)
}

# -----------------------------------------------------------------------------
# Phase 5 (window functions, scratchpad/phase5_window_design.md section 1.1):
# lag()/lead(), first()/last()/nth(), ntile(), order_by()/with_order() --
# parsed via dedicated top-level dispatch (see the `fn_name %in% c(...)`
# checks in ir_parse_expr()'s call-handling branch above), not the
# registry's generic per-arg loop, because they have named controls that
# can be supplied out of position or omitted entirely (`lag(x, default =
# 0)` skipping `n=`, `nth(x, k, order_by = y)`, ...) -- something the
# generic loop can't express, since it only ever sees `as.list(expr)[-1]`
# positionally (see ir_parse_case_when()'s sibling functions above for the
# file's established "named args need dedicated dispatch" precedent).
# Argument matching itself reuses rlang::call_match() (rlang >= 1.1.0,
# already a hard dependency) against a small prototype function whose
# formals mirror the real dplyr signature -- this gets R's own
# argument-matching semantics (named, positional, or mixed, in any order)
# for free, rather than hand-rolling it.
#
# Phase 5 W4: `order_by=` (on `lag()`/`lead()`/`first()`/`last()`/`nth()`)
# and the standalone `order_by()`/`with_order()` wrappers are now
# implemented, but ONLY when the order argument resolves to a bare column
# reference (or `desc(column)`/`-column`) via `ir_parse_order_ref()` --
# a general expression (e.g. `order_by(a + b, ...)`) is still unsupported
# and falls back, same as before.
# -----------------------------------------------------------------------------

#' Evaluate a raw (unparsed) expression as a constant in `env`
#'
#' No data mask is supplied, so a genuine column reference (or anything
#' else that isn't resolvable from `env` alone) raises an ordinary R
#' error, caught here and reported as "not a constant" (`ok = FALSE`)
#' rather than propagating -- the same "no data mask" trick
#' `resolve_slice_dots()` (`R/slice.R`) already relies on to distinguish
#' plain R values from column references.
#'
#' @param raw_expr A language object (not a quosure)
#' @param env The environment to evaluate against
#' @return `list(ok = <logical>, value = <the evaluated R value, or NULL>)`
#' @keywords internal
ir_eval_constant <- function(raw_expr, env) {
  tryCatch(
    list(ok = TRUE, value = rlang::eval_tidy(rlang::new_quosure(raw_expr, env))),
    error = function(e) list(ok = FALSE, value = NULL)
  )
}

# The value of an NA flag argument (`na.rm =` / `na_rm =`) when the native
# path can take it exactly: a single TRUE or FALSE, evaluated in `env`. NULL
# (run it in dplyr) for anything else: a value base R or dplyr would coerce
# (`1`, `"yes"`) or reject (`NA`, a vector), an expression that fails outside
# the data mask (`n() > 0`), or one that reads a column, which dplyr's data
# mask resolves before an environment variable of the same name. Shared by
# summarise()'s extract_na_rm() and the IR parsers that take a flag.
# @keywords internal
ir_na_flag_value <- function(raw_expr, env, schema) {
  if (expr_has_column_ref(raw_expr, schema)) {
    return(NULL)
  }
  evaluated <- ir_eval_constant(raw_expr, env)
  value <- evaluated$value
  if (!evaluated$ok || !is.logical(value) || length(value) != 1L || is.na(value)) {
    return(NULL)
  }
  as.vector(value)
}

#' Parse `lag(x, n = 1L, default = NULL, order_by = NULL)` / `lead(...)`
#'
#' Builds a window-class `ir_call(fn_name, list(x_ir), meta = list(n = ,
#' default = , order_override = ))` node (never lowered directly -- see the
#' [ir_call_registry] entry / [ir_bind()]'s guard). `n` must be a
#' non-negative whole-number constant (dplyr itself requires this);
#' `default`, if supplied, must be a CONSTANT expression -- evaluable via
#' [ir_eval_constant()] with no data mask, so `default = -1`, `default = 2L
#' + 3L`, or any other column-free expression all work (not just a bare
#' literal token), but `default = first(x)` (a real, column-referencing
#' dplyr default) is not implemented this wave.
#'
#' `order_by=`, when supplied, must resolve via [ir_parse_order_ref()] (a
#' bare column, `desc(column)`, or `-column`); anything else (a general
#' expression) is not implemented this wave and returns `NULL` (Phase 5
#' W4: `order_by=` re-frames this spec's own frame exactly like
#' `order_by()`/`with_order()` would if wrapped around the same call --
#' see [ir_parse_with_order()]'s docs -- so `lag(x, order_by = y)` and
#' `order_by(y, lag(x))` produce the same `meta$order_override`).
#'
#' @keywords internal
ir_parse_shift_call <- function(expr, env, schema, fn_name) {
  proto <- function(x, n = 1L, default = NULL, order_by = NULL) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  order_override <- NULL
  if (!is.null(present_args$order_by)) {
    order_override <- ir_parse_order_ref(present_args$order_by, schema)
    if (is.null(order_override)) {
      return(NULL)  # unsupported order_by= shape: not implemented this wave
    }
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  n_value <- 1L
  if (!is.null(present_args$n)) {
    n_eval <- ir_eval_constant(present_args$n, env)
    n_val <- n_eval$value
    # A Period/Duration `n` is dplyr's error to raise (F48): CPU
    if (!n_eval$ok || is_timespan_value(n_val) || !is.numeric(n_val) || length(n_val) != 1 ||
        is.na(n_val) || n_val != round(n_val) || n_val < 0) {
      return(NULL)
    }
    n_value <- as.integer(n_val)
  }

  default_ir <- NULL
  if (!is.null(present_args$default)) {
    default_eval <- ir_eval_constant(present_args$default, env)
    if (!default_eval$ok) {
      return(NULL)  # non-constant default: not implemented this wave
    }
    default_val <- default_eval$value
    if (!is.null(default_val)) {
      if (is_timespan_value(default_val)) {
        return(NULL)  # F48: a Period/Duration default is not a plain number
      }
      if (!is.atomic(default_val) || length(default_val) != 1) {
        return(NULL)  # non-scalar default: not implemented this wave
      }
      default_ir <- ir_lit_from_r(default_val)
      # `meta` is outside ir_infer_type()'s walk: a fractional Date default
      # goes to the CPU here (F37, ir_check_ts_lit())
      ir_check_ts_lit(default_ir)
      # A Date default for a POSIXct `x` (or the reverse): vctrs casts it
      # (local midnight / an error), the GPU would reinterpret its ticks (F32)
      x_type <- unname(ir_infer_type(x_ir, schema))
      def_type <- unname(default_ir$type)
      if (!isTRUE(default_ir$na) && !identical(x_type, def_type) &&
          any(c(x_type, def_type) %in% cuplyr_timestamp_types)) {
        return(NULL)
      }
    }
  }

  ir_call(fn_name, list(x_ir), meta = list(n = n_value, default = default_ir,
                                            order_override = order_override))
}

#' Parse `first(x, ...)` / `last(x, ...)` / `nth(x, n, ...)`
#'
#' The bare `x` (and, for `nth()`, a constant-integer `n`) shape, plus
#' `order_by=` (Phase 5 W4: see [ir_parse_order_ref()]/
#' [ir_parse_with_order()] -- `first(x, order_by = y)` re-frames this
#' spec's frame to be ordered by `y`, matching dplyr's own
#' "first/last/nth row of the group IN `order_by`'s ORDER, not the
#' group's current order" semantics, verified empirically against dplyr
#' 1.2.1). `default=`/`na_rm=` being present at all -- even at their own
#' default value -- still makes this return `NULL` (not implemented this
#' wave). Builds a window-class `ir_call(fn_name, list(x_ir), meta =
#' list(order_override = ))` node (`nth()` additionally carries `n = <int>`,
#' the (1-based, possibly negative) element position).
#'
#' @keywords internal
ir_parse_first_last_nth <- function(expr, env, schema, fn_name) {
  proto <- if (identical(fn_name, "nth")) {
    function(x, n, order_by = NULL, default = NULL, na_rm = FALSE) NULL
  } else {
    function(x, order_by = NULL, default = NULL, na_rm = FALSE) NULL
  }

  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args$default) || !is.null(present_args$na_rm)) {
    return(NULL)  # not implemented this wave
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  order_override <- NULL
  if (!is.null(present_args$order_by)) {
    order_override <- ir_parse_order_ref(present_args$order_by, schema)
    if (is.null(order_override)) {
      return(NULL)  # unsupported order_by= shape: not implemented this wave
    }
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  if (identical(fn_name, "nth")) {
    if (is.null(present_args$n)) {
      return(NULL)
    }
    n_eval <- ir_eval_constant(present_args$n, env)
    n_val <- n_eval$value
    if (!n_eval$ok || is_timespan_value(n_val) || !is.numeric(n_val) || length(n_val) != 1 ||
        is.na(n_val) || n_val != round(n_val) || n_val == 0) {
      return(NULL)  # a Period/Duration `n` is dplyr's error to raise (F48): CPU
    }
    return(ir_call("nth", list(x_ir),
                    meta = list(n = as.integer(n_val), order_override = order_override)))
  }

  ir_call(fn_name, list(x_ir), meta = list(order_override = order_override))
}

# Rank-family functions that accept `desc(e)` as their value argument (F29).
ir_rank_desc_fns <- c("row_number", "min_rank", "dense_rank", "percent_rank",
                      "cume_dist")

# TRUE iff `expr` is, or holds anywhere in its call tree, a quosure of the
# missing argument (what `{{ v }}` becomes when `v` was not supplied).
# @keywords internal
ir_has_missing_quosure <- function(expr) {
  if (rlang::is_quosure(expr)) {
    return(rlang::quo_is_missing(expr) || ir_has_missing_quosure(rlang::quo_get_expr(expr)))
  }
  if (rlang::is_missing(expr) || !is.call(expr)) {
    return(FALSE)
  }
  any(vapply(as.list(expr), ir_has_missing_quosure, logical(1)))
}

# `e` for a call `desc(e)` / `dplyr::desc(e)` (one argument), else `NULL`.
# A quosure spliced in by a wrapper (`{{ }}`) is looked through, and `e` then
# comes back as a quosure of that environment, not the caller's.
ir_unwrap_desc <- function(expr) {
  quo_env <- NULL
  if (rlang::is_quosure(expr)) {
    quo_env <- rlang::quo_get_env(expr)
    expr <- rlang::quo_get_expr(expr)
  }
  if (!is.call(expr) || length(expr) != 2L || (!is.null(names(expr)) && nzchar(names(expr)[2]))) {
    return(NULL)
  }
  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.symbol("::")) &&
      identical(head[[2]], as.symbol("dplyr"))) {
    head <- head[[3]]
  }
  if (!is.symbol(head) || !identical(as.character(head), "desc")) {
    return(NULL)
  }
  if (is.null(quo_env)) expr[[2]] else rlang::new_quosure(expr[[2]], quo_env)
}

#' Parse a window "order by" reference (`order_by=` argument, or
#' `order_by()`/`with_order()`'s own first argument) into a schema column
#' reference
#'
#' Recognizes a bare column name, `desc(column)`/`dplyr::desc(column)`, or
#' `-column` (dplyr's own `desc()` shorthand, also usable directly as an
#' `arrange()`/`order_by()` sort key) and resolves it against `schema`.
#' Anything else -- a general expression such as `order_by(a + b, ...)` --
#' is not implemented this wave (mirroring `R/window.R`'s rank-family
#' design note: only a bare, already-existing column doubles as an order
#' key so far, no `..winarg..`-style hoisting for order columns yet) and
#' returns `NULL`, same as any other unsupported IR shape (callers fall
#' back).
#'
#' @param expr A language object (not a quosure)
#' @param schema List with `names`/`types`
#' @return `list(col = <chr1>, desc = <lgl1>)`, or `NULL`
#' @keywords internal
ir_parse_order_ref <- function(expr, schema) {
  if (is.symbol(expr)) {
    col <- as.character(expr)
    if (!(col %in% schema$names)) {
      return(NULL)
    }
    return(list(col = col, desc = FALSE))
  }

  if (is.call(expr) && length(expr) == 2) {
    head <- expr[[1]]
    if (is.call(head) && length(head) == 3 && identical(head[[1]], as.symbol("::"))) {
      head <- head[[3]]  # strip namespace: dplyr::desc -> desc
    }
    # dplyr evaluates an order_by= value as an ordinary call, where
    # `desc(foo = y)` is an "unused argument" error: only desc()'s own `x`
    # name (or none) is that column
    arg_name <- if (is.null(names(expr))) "" else names(expr)[[2]]
    if (is.symbol(head) && as.character(head) %in% c("desc", "-") &&
        arg_name %in% c("", if (identical(as.character(head), "desc")) "x")) {
      inner <- expr[[2]]
      if (!is.symbol(inner)) {
        return(NULL)
      }
      col <- as.character(inner)
      if (!(col %in% schema$names)) {
        return(NULL)
      }
      type <- unname(schema$types[match(col, schema$names)])
      if (length(type) != 1L || is.na(type)) {
        return(NULL)
      }
      if (identical(as.character(head), "-")) {
        # `-f` is arithmetic on a factor (F18's rule): dplyr warns and ignores
        # the order, so it can't be a native sort key
        if (type == "DICTIONARY32") {
          if (ir_factor_rule(ir_call("-", list(ir_col(col))), schema) == "cpu") {
            return(NULL)
          }
        } else if (!(type %in% ir_negatable_types)) {
          return(NULL)  # `-s` / `-date` is an error in R
        }
      } else if (type == "STRING" && !session_collation_is_c()) {
        # desc() is `-xtfrm(s)`: the session collation orders it, the GPU
        # sorts bytes
        return(NULL)
      }
      return(list(col = col, desc = TRUE))
    }
  }

  NULL
}

# The GPU types a unary minus is defined for (dplyr's `-col` order key is the
# R expression `-col`: an error for a string or a Date/POSIXct).
ir_negatable_types <- c("INT32", "INT64", "FLOAT64", "BOOL8")

# TRUE iff `ir` provably has one value per row. Used to tell the first
# argument of an `order_by()` call (which dplyr permutes, and which must be
# row-sized) from a scalar (dplyr: "`order_by` must have size 1").
# @keywords internal
ir_is_row_sized <- function(ir) {
  if (identical(ir$kind, "col")) {
    return(TRUE)
  }
  if (!identical(ir$kind, "call")) {
    return(FALSE)  # literal, vector literal
  }
  entry <- ir_call_registry[[ir$op]]
  if (!is.null(entry) && !is.null(entry$window)) {
    if (identical(entry$window$kind, "agg")) {
      return(FALSE)
    }
    if (length(ir$args) == 0L) {
      return(identical(ir$op, "row_number"))
    }
    return(ir_is_row_sized(ir$args[[1]]))
  }
  any(vapply(ir$args, ir_is_row_sized, logical(1)))
}

# Window ops whose value does not depend on the row order: dplyr's
# `order_by()` has no observable effect on them.
ir_order_invariant_aggs <- c("sum", "n", "mean", "min", "max", "sd", "var",
                             "median", "n_distinct")
ir_order_invariant_ranks <- c("min_rank", "dense_rank", "percent_rank", "cume_dist")

# Apply `order_ref` to every window node of `ir` -- the windows the head
# function's own parse/desugar built -- and nothing else. `NULL` when some
# window cannot take it exactly (see ir_parse_with_order()).
# @keywords internal
ir_order_windows <- function(ir, order_ref) {
  ok <- TRUE
  walk <- function(node) {
    if (!ok || !identical(node$kind, "call")) {
      return(node)
    }
    entry <- ir_call_registry[[node$op]]
    if (is.null(entry) || is.null(entry$window)) {
      node$args <- lapply(node$args, walk)
      return(node)
    }
    kind <- entry$window$kind
    if (kind %in% c("scan", "shift", "rolling")) {
      # defensive: only the head's own windows are walked, and `with_order()`
      # captures an `order_by =` argument, so none can carry an override yet
      if (!is.null(node$meta$order_override)) {
        ok <<- FALSE
        return(node)
      }
      node$args <- lapply(node$args, walk)
      node$meta$order_override <- order_ref
      return(node)
    }
    if (identical(kind, "rank")) {
      if (identical(node$op, "row_number") && length(node$args) == 0L &&
          is.null(node$meta$order_override)) {
        # the position in the frame sorted by group + order
        node$meta$order_override <- order_ref
        return(node)
      }
      if (node$op %in% ir_order_invariant_ranks) {
        node$args <- lapply(node$args, walk)
        return(node)
      }
      ok <<- FALSE  # row_number(x): ties are broken by the order
      return(node)
    }
    if (identical(kind, "agg") && node$op %in% ir_order_invariant_aggs) {
      node$args <- lapply(node$args, walk)
      return(node)
    }
    ok <<- FALSE  # first/last/nth, an unknown window
    node
  }
  out <- walk(ir)
  if (ok) out else NULL
}

# `ir` with every column `name` replaced by `repl`, in the arguments and in
# the IR nodes a hook keeps in `meta`.
# @keywords internal
ir_subst_col <- function(ir, name, repl) {
  if (identical(ir$kind, "col")) {
    return(if (identical(ir$name, name)) repl else ir)
  }
  if (!identical(ir$kind, "call")) {
    return(ir)
  }
  ir$args <- lapply(ir$args, ir_subst_col, name = name, repl = repl)
  for (m in names(ir$meta)) {
    node <- ir$meta[[m]]
    if (is.list(node) && !is.null(node$kind)) {
      ir$meta[m] <- list(ir_subst_col(node, name, repl))
    }
  }
  ir
}

#' Parse dplyr's `with_order(order_by, fun, x, ...)` (what `order_by()` is)
#'
#' dplyr 1.2.1: `with_order(o, f, a1, ...)` evaluates `a1` and `...` in ROW
#' order, permutes only `a1` into `o` order (stable, NA last), calls
#' `f(a1[perm], ...)` and permutes the result back. Hence:
#' \itemize{
#'   \item the windows `f` itself builds (a plain `cumsum`, or the
#'     `cumsum()`/`row_number()` of `cummean()`, the `lag()` of
#'     `consecutive_id()`, ...) run in `o` order: each gets
#'     `meta$order_override`;
#'   \item windows inside `a1` run in row order and get no override;
#'   \item an elementwise `f` whose other arguments are all scalar literals
#'     is the plain row-order value (permutation commutes with it);
#'   \item an elementwise `f` with another row-sized argument gives
#'     `f(a1, b[inverse perm])`, which has no native form: `NULL`.
#' }
#' Everything dplyr rejects (no `a1`, a scalar `a1`, an aggregate `f`, a
#' name `with_order()` would capture) and everything the frame cannot do
#' exactly (`row_number(x)`/`ntile(x)` tie-breaking by the order,
#' `first`/`last`/`nth`, a nested `order_by()`) returns `NULL`, so the verb's
#' whole-call CPU fallback gives dplyr's value or error: never an unordered
#' native result.
#'
#' Every user sub-expression is evaluated at most once, in dplyr's order:
#' the first argument, then the others. The first argument is parsed once;
#' when it is not a plain column, `f` is parsed over a placeholder column of
#' the argument's type (and zone), the order is applied to that tree, and the
#' argument's IR then replaces the placeholder. A re-parse of the raw argument
#' would run a column-free call inside it (`cumsum(x + runif(1))`) again and
#' build the IR from the second value.
#'
#' @param order_ref `list(col =, desc =)` from [ir_parse_order_ref()]
#' @param head The raw expression of `f` (a symbol or `pkg::sym`)
#' @param a1 The raw expression of the permuted first argument
#' @param rest Named list of the raw remaining arguments (none may read a
#'   column, and each must evaluate to a non-NULL atomic vector of length 1)
#' @param env The environment the expressions are evaluated in
#' @param schema The input schema (`names`, `types`)
#' @return An IR node, or `NULL`
#' @keywords internal
ir_parse_with_order <- function(order_ref, head, a1, rest, env, schema) {
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.symbol("::"))) {
    head_name <- head[[3]]
  } else {
    head_name <- head
  }
  if (!is.symbol(head_name) ||
      as.character(head_name) %in% c("order_by", "with_order")) {
    return(NULL)
  }

  # Every other argument must be a non-NULL scalar (length exactly 1): a
  # row-sized one is not permuted by dplyr, which no native op expresses, and
  # NULL/empty ones are not scalars (`lag(x, n = NULL)` is dplyr's error;
  # the parse hooks read NULL as an absent argument). All are checked for a
  # column reference before anything is evaluated, so a call that goes to the
  # CPU for that reason has run nothing here.
  for (arg in rest) {
    if (expr_has_column_ref(arg, schema)) {
      return(NULL)
    }
  }

  # The value-only rank functions don't see the order: the plain call is the
  # value, whatever its argument is (`min_rank(desc(x))` is not an
  # expression of its own, only the rank call parses it). This parse is the
  # first argument's one evaluation.
  if (as.character(head_name) %in% ir_order_invariant_ranks && length(rest) == 0L) {
    plain <- ir_parse_expr(as.call(list(head, a1)), env, schema)
    if (!is.null(plain) && identical(plain$kind, "call") &&
        identical(ir_call_registry[[plain$op]]$window$kind, "rank") &&
        ir_is_row_sized(plain$args[[1]])) {
      return(plain)
    }
    return(NULL)
  }

  # The first argument's one evaluation (dplyr evaluates it before `...`).
  a1_ir <- ir_parse_expr(a1, env, schema)
  if (is.null(a1_ir) || !ir_is_row_sized(a1_ir)) {
    return(NULL)
  }

  # The others, each evaluated once; the call is parsed with the VALUE in its
  # place. `.env$k` resolves: the mask is empty, as no column is read.
  for (i in seq_along(rest)) {
    value <- tryCatch(
      list(ok = TRUE, value = rlang::eval_tidy(rest[[i]], data = list(), env = env)),
      error = function(e) list(ok = FALSE, value = NULL)
    )
    if (!value$ok || !is.atomic(value$value) || length(value$value) != 1L) {
      return(NULL)
    }
    rest[i] <- list(value$value)
  }

  placeholder <- NULL
  call_schema <- schema
  if (identical(a1_ir$kind, "col")) {
    # A column (`x`, `.data$x`): its name evaluates nothing, and the hooks
    # see the real column (factor levels, zone)
    call_a1 <- as.symbol(a1_ir$name)
  } else {
    a1_type <- tryCatch(unname(ir_infer_type(a1_ir, schema)), error = function(e) NA_character_)
    # A placeholder carries a type and a zone, not factor levels
    if (length(a1_type) != 1L || is.na(a1_type) || a1_type == "DICTIONARY32" ||
        (a1_type %in% cuplyr_timestamp_types &&
         !a1_type %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS"))) {
      return(NULL)
    }
    placeholder <- unique_temp_name("..order_by_a1..", schema$names)
    call_schema$names <- c(schema$names, placeholder)
    call_schema$types <- c(schema$types, a1_type)
    if (a1_type == "TIMESTAMP_MICROSECONDS") {
      zone <- tryCatch(ir_infer_tz(a1_ir, schema), error = function(e) NULL)
      if (is.null(zone)) {
        return(NULL)
      }
      call_schema$tz <- tz_from_positions(call_schema$names, call_schema$types,
                                          c(tz_by_position(schema), zone))
    }
    call_a1 <- as.symbol(placeholder)
  }

  call_ir <- ir_parse_expr(as.call(c(list(head, call_a1), rest)), env, call_schema)
  if (is.null(call_ir)) {
    return(NULL)
  }
  if (identical(call_ir$kind, "call")) {
    entry <- ir_call_registry[[call_ir$op]]
    if (!is.null(entry) && identical(entry$window$kind, "agg")) {
      return(NULL)  # an aggregate has no row to put back
    }
  }
  call_ir <- ir_order_windows(call_ir, order_ref)
  if (is.null(call_ir) || is.null(placeholder)) {
    return(call_ir)
  }
  ir_subst_col(call_ir, placeholder, a1_ir)
}

# Bind the arguments of `with_order(order_by, fun, x, ...)` the way R does
# (`present`: the call_match()ed call) and parse. `NULL` when `x` is absent
# or `fun` is not a (namespaced) function name.
# @keywords internal
ir_parse_with_order_args <- function(present, env, schema) {
  args <- as.list(present)[-1]
  if (is.null(args[["order_by"]]) || is.null(args[["fun"]]) || is.null(args[["x"]])) {
    return(NULL)
  }
  order_ref <- ir_parse_order_ref(args[["order_by"]], schema)
  if (is.null(order_ref)) {
    return(NULL)
  }
  rest <- args[!(names(args) %in% c("order_by", "fun", "x"))]
  if (any(vapply(rest, function(a) rlang::is_missing(a) || identical(a, quote(...)), logical(1)))) {
    return(NULL)
  }
  ir_parse_with_order(order_ref, args[["fun"]], args[["x"]], rest, env, schema)
}

#' Parse `order_by(order_by, call)`
#'
#' dplyr rewrites it to `with_order(order_by, <head of call>, <args of
#' call>)`, so this builds that call, lets R bind its arguments (a name
#' `with_order()` would capture, such as `lag(x, order_by = y)`, makes the
#' call an error, i.e. `NULL` here) and parses it like [ir_parse_with_order()].
#'
#' @keywords internal
ir_parse_order_by_call <- function(expr, env, schema) {
  proto <- function(order_by, call) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args[["order_by"]]) || !is.call(present_args[["call"]])) {
    return(NULL)
  }

  wrapped <- present_args[["call"]]
  rewritten <- as.call(c(list(as.symbol("with_order"), present_args[["order_by"]], wrapped[[1]]),
                         as.list(wrapped)[-1]))
  present <- tryCatch(
    rlang::call_match(rewritten, function(order_by, fun, x, ...) NULL, defaults = FALSE),
    error = function(e) NULL
  )
  if (is.null(present)) {
    return(NULL)
  }
  ir_parse_with_order_args(present, env, schema)
}

#' Parse `with_order(order_by, fun, x, ...)`
#'
#' `fun` is a function reference (`cummean`, `dplyr::lag`); `x` and the
#' further arguments are `fun`'s. Evaluated exactly as `order_by(order_by,
#' fun(x, ...))`: see [ir_parse_with_order()].
#'
#' @keywords internal
ir_parse_with_order_call <- function(expr, env, schema) {
  present <- tryCatch(
    rlang::call_match(expr, function(order_by, fun, x, ...) NULL, defaults = FALSE),
    error = function(e) NULL
  )
  if (is.null(present)) {
    return(NULL)
  }
  ir_parse_with_order_args(present, env, schema)
}

# -----------------------------------------------------------------------------
# Rolling windows (Theo-requested benchmark task, scratchpad/todo.md
# "Rolling-window benchmark target"): a fifth window kind, `"rolling"`,
# alongside shift/rank/scan/agg. Surfaced under two call shapes, both
# needing real argument-NAME matching (same reason as lag()/lead()/ntile()
# above -- na.rm=/fill=/align= are named controls that can be omitted or
# supplied out of position), so both get dedicated top-level dispatch
# rather than the generic per-arg registry loop:
#   * cuplyr's own names -- `roll_mean(x, n, na.rm = FALSE)`,
#     `roll_sum()`/`roll_min()`/`roll_max()`/`roll_sd()`/`roll_median()` --
#     RcppRoll/slider-adjacent naming picked as the dplyr-ecosystem-idiomatic
#     surface (see ir_parse_roll_call()).
#   * an ALIAS for `data.table::frollmean(x, n, ...)` and its five
#     siblings (`frollsum`/`frollmin`/`frollmax`/`frollsd`/`frollmedian`) --
#     this is the exact call shape `benchmark/benchmark_rolling.R`'s cuplyr
#     probe constructs (`rlang::call2(paste0("froll", workload), sym("x"),
#     window, .ns = "data.table")`), verified by reading that script before
#     picking this alias surface. `pkg::fn(...)` is already stripped to a
#     bare `fn_name` by `ir_parse_expr()`'s call-head handling before any
#     dispatch runs (`data.table::frollmean` arrives here as plain
#     `"frollmean"`), so `ir_parse_froll_call()` only has to translate
#     data.table's richer argument set (`fill=`/`algo=`/`align=`/`hasNA=`/
#     `adaptive=`) down to the same `list(n=, na_rm=)` meta shape, rejecting
#     (NULL, i.e. "not implemented this wave, fall back") any non-default
#     `align=`/`fill=`/`adaptive=` -- this task is right-aligned-only, per
#     scratchpad/todo.md's explicit scope cut. `algo=`/`hasNA=` are accepted
#     but ignored: both are pure performance hints in data.table (exact vs.
#     approximate *computation strategy*, and a "does x contain NA at all"
#     hint respectively) that never change the correct *result*, and cuplyr
#     always computes the exact result regardless.
#
# Both shapes desugar to the SAME five (well, six, see roll_median below)
# canonical registry ops (`"roll_mean"`/`"roll_sum"`/`"roll_min"`/
# `"roll_max"`/`"roll_sd"`/`"roll_median"`), reusing `window_spec()`'s
# EXISTING `n`/`na_rm` fields verbatim (no new field needed: `n` already
# carries lag()/lead()'s shift amount and nth()'s element position, `na_rm`
# already threads through generically in R/window.R's
# extract_one_window_call() even though no pre-existing window op had set
# it to anything but FALSE before this task) -- see
# scratchpad/phase5_window_design.md section 2's `window_spec()` field
# docs, all reused unchanged.
#
# F59: cudf's rolling kernel rescans every window, so its cost grows with the
# width (and its STD loses accuracy on offset data). `src/ops_rolling.cpp`
# computes roll_sum/roll_mean/roll_sd/roll_min/roll_max from segmented scans
# (segments of w rows; sums of squares about a member of each part, so
# nothing cancels), and the sum of an integer exactly in INT64; the NA rules are the same ones the empirical notes below derive,
# applied from counts of the finite values and infinities (a window never
# crosses a group once its row-in-group is >= w - 1). roll_sd always takes the
# scans. For the other four, `roll_native()` (src/ops_rolling.cpp) keeps
# cudf's grouped_rolling_window() below a measured crossover width (about
# 100 rows on one 1e7-row group, 56 to 160 depending on groups, NA and
# na.rm), where its
# direct kernel is cheaper than a constant number of passes. What the native
# code does not cover at all (min/max of a timestamp or boolean) stays on
# cudf too, and the notes below describe the min_periods scheme of that path.
#
# Empirically verified (a standalone probe against this environment's
# installed cudf 25.12, NOT re-derived from the header comments alone,
# since "min_periods" is ambiguous between "row count in window" and
# "valid (non-null) count in window" until tested):
#   * `cudf::rolling_window(x, preceding = w, following = 0, min_periods =
#     w, agg)` reproduces R's na.rm=FALSE poisoning FOR FREE, with no
#     separate is-null-indicator masking pass at all (unlike the W4
#     cum*/rank sticky-NA fix this file's window entries otherwise mirror):
#     the window is a FIXED w-row range, so "w valid values required"
#     among exactly w physical rows already means "zero nulls tolerated" --
#     confirmed on `x = c(1,2,NA,4,5)`, w=3: every output is NA the moment
#     the row-range includes the NA, matching data.table's own
#     `frollsum(x, 3, na.rm = FALSE)`.
#   * na.rm=TRUE is NOT simply "cudf's own default behavior" (cudf rolling
#     aggregations always skip nulls when computing, regardless of
#     min_periods): using `min_periods = 1` alone would incorrectly
#     compute a value for the leading (w-1) rows too, using FEWER than w
#     physical rows -- confirmed empirically that real
#     `data.table::frollmean(x, n, na.rm = TRUE)` still keeps exactly the
#     leading (n-1) positions NA (a ROW-COUNT requirement, independent of
#     na.rm), and only skips individual NA *values* once a window has
#     become physically complete. `src/ops_window.cpp`'s rolling branch
#     therefore computes the aggregate with `min_periods = 1` (skip nulls,
#     as low a valid-count bar as possible) and separately masks to NA any
#     row whose PHYSICAL row-count-in-window (a second rolling COUNT_ALL
#     pass, `null_policy::INCLUDE`, `min_periods = 1`, which counts rows
#     regardless of their own nullity) is `< w` -- verified bit-for-bit
#     against `data.table::frollmean(c(1,2,NA,4,5), 3, na.rm = TRUE)` ==
#     `c(NA, NA, 1.5, 3, 4.5)`.
#   * Grouped rolling comes "almost free": `cudf::grouped_rolling_window()`
#     clamps window boundaries at GROUP edges (not just table edges) when
#     given a `group_keys` table, and this file's existing frame
#     architecture already produces a stable-sorted-by-group-only `work`
#     view (no order_cols for a rolling frame, exactly like the scan
#     family) whose ROW ORDER is the group's original relative order --
#     exactly what `grouped_rolling_window()` requires ("presorted by the
#     group_key values"). `src/ops_window.cpp` therefore uses
#     `grouped_rolling_window()` UNIFORMLY for both the truly-grouped and
#     the ungrouped case (the ungrouped case's substituted constant
#     zero-key column is trivially "one sorted group"), the same "one code
#     path via a substituted key column" idiom this file already uses for
#     the scan/rank/agg families -- no separate `cudf::rolling_window()`
#     call needed at all.
#   * `cudf::make_median_aggregation<T>()` has NO `rolling_aggregation`
#     explicit template instantiation in this environment's compiled
#     `libcudf.so` (confirmed via `nm -DC libcudf.so | grep
#     make_median_aggregation`: only `aggregation`/`reduce_aggregation`/
#     `groupby_aggregation` instantiations exist) -- MEDIAN is genuinely
#     unsupported by `cudf::rolling_window`/`grouped_rolling_window` in
#     cudf 25.12, not merely undocumented. `roll_median()` is therefore
#     registered (so it parses, types, and participates in window
#     decomposition identically to its five siblings) but
#     `window_spec_lowerable()` (R/lower.R) explicitly excludes it, routing
#     any `roll_median()`/`data.table::frollmedian()` mutate() call through
#     the existing `mutate_window_fallback()` CPU path -- which needs
#     `roll_median()` (cuplyr's own name) to be a REAL, correct, exported R
#     function for that CPU re-run to work at all; see `R/roll.R`.
#   * MEAN/SUM/MIN/MAX/STD all have confirmed `rolling_aggregation`
#     instantiations (`nm -DC libcudf.so`), matching scratchpad/todo.md's
#     expectation exactly. `roll_sd()` uses `make_std_aggregation<
#     rolling_aggregation>(1)` (ddof = 1, sample standard deviation,
#     matching R's own `sd()` and this file's existing groupby `sd()`
#     entry's ddof choice).
# -----------------------------------------------------------------------------

#' Parse `roll_mean(x, n, na.rm = FALSE)` and its five siblings
#'
#' cuplyr's own rolling-window surface (RcppRoll/slider-adjacent naming --
#' see the file-level comment above this section for the full design
#' rationale). Right-aligned only: there is no `align=` parameter on this
#' surface at all (unlike the `frollmean()`-alias path below, which has to
#' reject a non-default `align=` explicitly since data.table's own
#' signature offers one).
#'
#' @param fn_name One of `"roll_mean"`/`"roll_sum"`/`"roll_min"`/
#'   `"roll_max"`/`"roll_sd"`/`"roll_median"`
#' @keywords internal
ir_parse_roll_call <- function(expr, env, schema, fn_name) {
  proto <- function(x, n, na.rm = FALSE) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$x) || is.null(present_args$n)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  n_value <- ir_parse_roll_window_width(present_args$n, env)
  if (is.null(n_value)) {
    return(NULL)
  }

  na_rm <- ir_parse_roll_na_rm(present_args$na.rm, env, schema)
  if (is.null(na_rm)) {
    return(NULL)
  }

  ir_call(fn_name, list(x_ir), meta = list(n = n_value, na_rm = na_rm))
}

#' Parse `data.table::frollmean(x, n, fill=, algo=, align=, na.rm=, hasNA=,
#' adaptive=)` and its five siblings as an alias for `roll_mean()`/etc.
#'
#' See the file-level comment above this section: only the default
#' `align = "right"`, `fill = NA`, `adaptive = FALSE`, `partial = FALSE`,
#' `give.names = FALSE` are supported this wave (a non-default value
#' returns `NULL`, i.e. "not implemented, fall back" -- mutate()'s usual
#' contract for an unrecognized shape: `partial = TRUE` changes windowing
#' semantics entirely -- allows a genuinely incomplete window to produce a
#' value instead of `NA` -- and `give.names = TRUE` changes the RETURN
#' SHAPE to a named `data.table`/list instead of a plain vector, so both
#' must be rejected, not silently ignored); `algo=`/`has.nf=`/`hasNA=` are
#' accepted and ignored (pure performance hints, verified against
#' data.table's own docs to never change the *result*). Proto signature
#' verified against this environment's actual installed
#' `data.table::frollmean` (`args(data.table::frollmean)`), not assumed
#' from memory/docs alone -- note the real signature has `has.nf=` (NOT
#' `hasNA=`, which is a separate, no-default legacy/back-compat argument
#' that's ALSO present, verified empirically).
#'
#' @param dt_fn_name One of `"frollmean"`/`"frollsum"`/`"frollmin"`/
#'   `"frollmax"`/`"frollsd"`/`"frollmedian"`
#' @keywords internal
ir_parse_froll_call <- function(expr, env, schema, dt_fn_name) {
  canonical <- switch(dt_fn_name,
    frollmean = "roll_mean", frollsum = "roll_sum", frollmin = "roll_min",
    frollmax = "roll_max", frollsd = "roll_sd", frollmedian = "roll_median"
  )

  proto <- function(x, n, fill = NA, algo = c("fast", "exact"),
                     align = c("right", "left", "center"), na.rm = FALSE,
                     has.nf = NA, adaptive = FALSE, partial = FALSE,
                     give.names = FALSE, hasNA) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$x) || is.null(present_args$n)) {
    return(NULL)
  }

  if (!is.null(present_args$align)) {
    align_eval <- ir_eval_constant(present_args$align, env)
    if (!align_eval$ok || !identical(align_eval$value, "right")) {
      return(NULL)  # align="left"/"center": not implemented this wave
    }
  }
  if (!is.null(present_args$fill)) {
    fill_eval <- ir_eval_constant(present_args$fill, env)
    if (!fill_eval$ok || length(fill_eval$value) != 1 || !is.na(fill_eval$value)) {
      return(NULL)  # a non-NA fill: not implemented this wave
    }
  }
  if (!is.null(present_args$adaptive)) {
    adaptive_eval <- ir_eval_constant(present_args$adaptive, env)
    if (!adaptive_eval$ok || !isFALSE(adaptive_eval$value)) {
      return(NULL)  # adaptive=TRUE (per-row window vector): not implemented this wave
    }
  }
  if (!is.null(present_args$partial)) {
    partial_eval <- ir_eval_constant(present_args$partial, env)
    if (!partial_eval$ok || !isFALSE(partial_eval$value)) {
      return(NULL)  # partial=TRUE: different windowing semantics, not implemented this wave
    }
  }
  if (!is.null(present_args$give.names)) {
    give_names_eval <- ir_eval_constant(present_args$give.names, env)
    if (!give_names_eval$ok || !isFALSE(give_names_eval$value)) {
      return(NULL)  # give.names=TRUE: changes return shape entirely, not implemented this wave
    }
  }
  # algo=/has.nf=/hasNA=: accepted, deliberately unread (see docs above).

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  n_value <- ir_parse_roll_window_width(present_args$n, env)
  if (is.null(n_value)) {
    return(NULL)
  }

  na_rm <- ir_parse_roll_na_rm(present_args$na.rm, env, schema)
  if (is.null(na_rm)) {
    return(NULL)
  }

  ir_call(canonical, list(x_ir), meta = list(n = n_value, na_rm = na_rm))
}

# A rolling window width must be a positive whole-number CONSTANT (never a
# column reference/vector of multiple sizes -- data.table's own "adaptive"
# multi-window-size shape, rejected above via `adaptive=`, is the only
# vector-`n` case that would otherwise reach here, and `length(...) != 1`
# rejects it defensively too). Returns an integer scalar, or `NULL` (not
# implemented this wave / not a constant).
ir_parse_roll_window_width <- function(n_expr, env) {
  n_eval <- ir_eval_constant(n_expr, env)
  n_val <- n_eval$value
  if (!n_eval$ok || is_timespan_value(n_val) || !is.numeric(n_val) || length(n_val) != 1 ||
      is.na(n_val) || n_val != round(n_val) || n_val < 1) {
    return(NULL)
  }
  as.integer(n_val)
}

# `na.rm=` must be a constant TRUE/FALSE (absent -> FALSE, dplyr/data.table's
# shared default). Returns a logical scalar, or `NULL` (not a constant; see
# ir_na_flag_value()).
ir_parse_roll_na_rm <- function(na_rm_expr, env, schema) {
  if (is.null(na_rm_expr)) {
    return(FALSE)
  }
  ir_na_flag_value(na_rm_expr, env, schema)
}

#' Parse `ntile(x = row_number(), n)`
#'
#' Fully desugars at parse time into dplyr's own `ntile()` algorithm
#' (verified against `dplyr:::ntile`'s actual body, not just its docs;
#' cross-checked numerically against a battery of cases including ties, a
#' non-multiple group size, all-`NA`, and a zero-row input) expressed in
#' plain IR plus two nested window calls -- `row_number(x)` (or bare
#' `row_number()` when `x` is omitted) and, only when `x` is supplied,
#' `sum(if_else(is.na(x), 0L, 1L))` (the non-missing count; when `x` is
#' omitted, dplyr's own `len` is just the group size, i.e. plain `n()`).
#' Both are ordinary window-class `ir_call` nodes at this point -- their
#' actual extraction into `..win*..` temp columns happens later, in
#' `R/window.R`'s `ir_extract_windows()`/`plan_window_stages()`, not here.
#'
#' Whether `x` was actually supplied changes the desugar (dplyr's own
#' source only re-ranks `x` -- `x <- row_number(x)` -- when it's NOT the
#' default), so this dispatches on `rlang::call_match(..., defaults =
#' FALSE)` (which omits an unsupplied formal entirely, unlike `defaults =
#' TRUE`) rather than merely on whether `x`'s resolved value looks like
#' `row_number()`.
#'
#' `n` must be a constant, positive, whole-number scalar (dplyr's own
#' `check_number_whole()` requirement); `n` missing entirely (dplyr's `n`
#' has no default) falls back, since there's nothing sensible to build.
#'
#' The division-then-floor steps in dplyr's own formula (`(x + (larger_size
#' - 1L)) / larger_size`, then a final `as.integer(floor(bins))`) are
#' expressed via `%/%` (floor division) directly: every numerator/denominator
#' pair in both branches is non-negative by construction (ranks start at 1,
#' sizes are always >= 1), and floor division equals
#' true-division-then-floor for non-negative operands. The sizes follow
#' dplyr's FLOAT64 `len / n` / `ceiling()` / `floor()`, and the bins are cast
#' to INT32 once at the end (`cast_int32`), matching ntile()'s own
#' always-integer output.
#'
#' @keywords internal
ir_parse_ntile <- function(expr, env, schema) {
  proto <- function(x = row_number(), n) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (is.null(present_args$n)) {
    return(NULL)  # n is mandatory (no default); dplyr itself errors here
  }
  n_eval <- ir_eval_constant(present_args$n, env)
  n_val <- n_eval$value
  if (!n_eval$ok || is_timespan_value(n_val) || !is.numeric(n_val) || length(n_val) != 1 ||
      is.na(n_val) || n_val != round(n_val) || n_val <= 0) {
    return(NULL)  # a Period/Duration `n` is dplyr's error to raise (F48): CPU
  }
  n_lit <- ir_lit_from_r(as.integer(n_val))

  x_supplied <- !is.null(present_args$x)

  if (x_supplied) {
    x_ir <- ir_parse_expr(present_args$x, env, schema)
    if (is.null(x_ir)) {
      return(NULL)
    }
    rn <- ir_call("row_number", list(x_ir))
    # dplyr counts a NaN as missing, and so does is.na() (F55).
    missing_ir <- ir_call("is.na", list(x_ir))
    indicator <- ir_call("if_else", list(missing_ir, ir_lit_from_r(0L), ir_lit_from_r(1L)))
    len <- ir_call("sum", list(indicator))
  } else {
    rn <- ir_call("row_number", list())
    len <- ir_call("n", list())
  }

  # dplyr's own formula: size = len / n, larger_size = ceiling(size),
  # smaller_size = floor(size). It runs in FLOAT64 and is cast to INT32 once
  # at the end (ntile() returns an integer); the same chain in INT32 would pay
  # F42's range check per node.
  n_larger         <- ir_call("%%", list(len, n_lit))
  size             <- ir_call("/", list(len, n_lit))
  larger_size      <- ir_call("ceil", list(size))
  smaller_size     <- ir_call("floor", list(size))
  larger_threshold <- ir_call("*", list(larger_size, n_larger))

  branch_a <- ir_call("%/%", list(
    ir_call("+", list(rn, ir_call("-", list(larger_size, ir_lit_from_r(1L))))),
    larger_size
  ))
  branch_b <- ir_call("+", list(
    ir_call("%/%", list(
      ir_call("+", list(
        ir_call("-", list(rn, larger_threshold)),
        ir_call("-", list(smaller_size, ir_lit_from_r(1L)))
      )),
      smaller_size
    )),
    n_larger
  ))

  bins <- ir_call("if_else", list(
    ir_call("<=", list(rn, larger_threshold)),
    branch_a,
    branch_b
  ))

  ir_call("cast_int32", list(ir_call("if_else", list(
    ir_call("==", list(len, ir_lit_from_r(0L))),
    ir_lit(value = NULL, type = "INT32", na = TRUE),
    bins
  ))))
}

#' Parse `sum(x)` / `sum(x, na.rm = <literal>)` (Phase 5, task W9)
#'
#' `sum()`'s ordinary registry entry (`arity = 1L`, agg-family window kind)
#' only accepts the bare single-argument form -- the overwhelmingly common
#' shape, and the one this window-agg family's `na.rm = FALSE` default (see
#' `src/ops_window.cpp`'s "sticky-NA"-style whole-group-NA-propagation
#' rule, matching R's own `sum()` default) is built for. But real,
#' unmodified `dplyr::add_tally(wt = )` (a plain, non-generic dplyr
#' function we don't control -- `dplyr:::tally_n()`) UNCONDITIONALLY emits
#' `base::sum(<wt>, na.rm = TRUE)` as its internal `mutate()` dot,
#' regardless of whether the weight column actually has any NAs -- so this
#' 2-arg shape must parse too, or `add_tally(wt = )` hard-errors. (This is
#' the Phase 5 task W9 fix-forward found while deleting
#' `grouped_aggregate_mutate()`, which used to accept this exact shape via
#' a real NA-excluding `summarise()`-then-join-back call.)
#'
#' `na.rm = TRUE` desugars `x` to `coalesce(x, 0)` before the window `sum`
#' call: replacing every `NA` with the additive identity is exactly
#' equivalent to excluding it from the sum (matching what R's own
#' `sum(x, na.rm = TRUE)` computes), and since the desugared column has no
#' nulls left at all, the window kernel's whole-group-NA rule never fires
#' -- so no C++ change is needed. `na.rm = FALSE` (if named explicitly) is
#' a pure no-op, since propagating NA is already this family's own default.
#'
#' A second argument with any other name (or a non-literal-logical value)
#' doesn't match this shape -- returns `NULL`, same as any other
#' unsupported call shape (`mutate()` hard-errors on it; `filter()` falls
#' back to CPU evaluation).
#'
#' @keywords internal
ir_parse_sum_call <- function(expr, env, schema) {
  args_raw <- as.list(expr)[-1]
  if (length(args_raw) == 0 || length(args_raw) > 2) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(args_raw[[1]], env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }

  if (length(args_raw) == 1) {
    return(ir_call("sum", list(x_ir)))
  }

  arg_names <- names(args_raw)
  if (is.null(arg_names) || !identical(arg_names, c("", "na.rm"))) {
    return(NULL)  # only the exact `sum(x, na.rm = <literal>)` shape matches
  }

  na_rm_val <- ir_na_flag_value(args_raw[[2]], env, schema)
  if (is.null(na_rm_val)) {
    return(NULL)
  }

  if (isTRUE(na_rm_val)) {
    x_ir <- ir_call("coalesce", list(x_ir, ir_lit_from_r(0L)))
  }

  ir_call("sum", list(x_ir))
}

# F41: `mean(x, na.rm = <literal>)` / `median(...)` in mutate()/filter(). The
# window agg path has no na.rm argument, so the two-argument form is only
# native for a Date/POSIXct x (where "all NA" is NA either way); it is carried
# as `meta$na_rm` into the window spec. `na.rm = FALSE` is the one-argument
# call. Any other shape is NULL (CPU), as before.
ir_parse_mean_median_call <- function(expr, env, schema, fn_name) {
  args_raw <- as.list(expr)[-1]
  arg_names <- names(args_raw)
  if (is.null(arg_names) || !identical(arg_names, c("", "na.rm"))) {
    return(NULL)
  }
  x_ir <- ir_parse_expr(args_raw[[1]], env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  na_rm_val <- ir_na_flag_value(args_raw[[2]], env, schema)
  if (is.null(na_rm_val)) {
    return(NULL)
  }
  if (!na_rm_val) {
    return(ir_call(fn_name, list(x_ir)))
  }
  x_type <- tryCatch(unname(ir_infer_type(x_ir, schema)), error = function(e) NA_character_)
  if (!x_type %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")) {
    return(NULL)
  }
  ir_call(fn_name, list(x_ir), meta = list(na_rm = TRUE))
}

# -----------------------------------------------------------------------------
# Tree Helpers
# -----------------------------------------------------------------------------

#' Collect the unique columns referenced by an IR subtree
#'
#' @param ir An IR node (or `NULL`)
#' @return Unique character vector of column names, in first-encountered
#'   order
#' @keywords internal
ir_cols <- function(ir) {
  if (is.null(ir)) {
    return(character())
  }
  switch(.subset2(ir, "kind"),
    col = .subset2(ir, "name"),
    # (F49: a rewritten length(c_across(..)) is a literal that still reads its
    # operand columns, which .keep = "used"/"unused" must see)
    lit = .subset2(ir, "reads") %||% character(),
    vec = character(),
    call = {
      args <- .subset2(ir, "args")
      # no arguments: unique(unlist(list())), which is NULL
      if (length(args) == 0L) NULL else unique(ir_arg_cols(args))
    }
  )
}

# The columns every argument of a call reads, in order and with repeats (the
# caller takes unique() once, which is what unique() at each level of nesting
# gives): the walk of ir_cols() without a vector and an unlist() per call node.
# @keywords internal
ir_arg_cols <- function(args) {
  out <- NULL
  for (i in seq_along(args)) {
    arg <- .subset2(args, i)
    kind <- .subset2(arg, "kind")
    out <- c(out, if (identical(kind, "col")) {
      .subset2(arg, "name")
    } else if (identical(kind, "call")) {
      ir_arg_cols(.subset2(arg, "args"))
    } else if (identical(kind, "lit")) {
      .subset2(arg, "reads") %||% character()
    } else if (identical(kind, "vec") || is.null(arg)) {
      character()
    })
  }
  out
}

#' Estimate the relative evaluation cost of an IR subtree
#'
#' Used by `reorder_filters()` to run cheaper predicates first. Literal = 1,
#' column reference = 2, and every call node adds 1 on top of the summed
#' cost of its arguments -- so a scalar comparison (`x > 5`, cost 4) is
#' cheaper than a column-column comparison (`x > y`, cost 5), which in turn
#' is cheaper than any deeper nesting.
#'
#' @param ir An IR node (or `NULL`)
#' @return Integer cost
#' @keywords internal
ir_cost <- function(ir) {
  if (is.null(ir)) {
    return(0L)
  }
  switch(ir$kind,
    lit = 1L,
    vec = 1L,
    col = 2L,
    call = sum(vapply(ir$args, ir_cost, integer(1))) + 1L
  )
}

#' Does an IR subtree hold an F42 checked integer node?
#'
#' A checked node is a binary `+`, `-` or `*` whose operands are all
#' INT32/BOOL8 (a bare untyped NA literal adopts its sibling's type): the
#' rule of `checked_int_kind()` kind 1 in `src/expr_eval.hpp`. Such a node can
#' warn "NAs produced by integer overflow", and dplyr's warning depends on the
#' rows its verb receives, so the optimizer must not change them (F70).
#' `%/%` and `%%` never warn and are not checked, nor is a node with a null
#' literal operand (`a + NA`; its other operand is still walked). Cached by
#' `make_predicate()` and `make_mutate_expr()`.
#'
#' One walk visits each node once and stops at the first checked node. Column,
#' literal and vector operands are typed off the schema/IR (both first, so a
#' double leaf settles a node without typing a call operand). A call operand is
#' typed bottom-up as the walk descends into it (`ir_call_type()`, the step
#' [ir_infer_type()] takes per node), so a chain of `+` nodes is typed once,
#' not once per level. A type that cannot be inferred counts as checked.
#'
#' @param ir An IR node (or `NULL`)
#' @param schema List with `names`/`types` the IR is bound against
#' @return A single logical
#' @keywords internal
ir_has_checked_int <- function(ir, schema) {
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

# A single type string, else an error (ir_infer_type() raises one for it too).
ir_one_type <- function(type) {
  if (!is.character(type) || length(type) != 1L) {
    stop("An IR node type must be a single string.", call. = FALSE)
  }
  type
}

# Mirrors checked_lit_is_null() in src/expr_eval.hpp.
ir_lit_is_null <- function(a) {
  if (!identical(.subset2(a, "kind"), "lit")) {
    return(FALSE)
  }
  value <- .subset2(a, "value")
  isTRUE(.subset2(a, "na")) || is.null(value) ||
    ((is.numeric(value) || is.logical(value)) && length(value) == 1L && is.na(value))
}

#' Is an IR subtree constant (no column references)?
#'
#' @param ir An IR node
#' @return `TRUE` iff [ir_cols()] is empty
#' @keywords internal
ir_is_const <- function(ir) {
  # (walks like ir_cols() but stops at the first column it would collect)
  if (is.null(ir)) {
    return(TRUE)
  }
  switch(.subset2(ir, "kind"),
    col = FALSE,
    lit = length(ir$reads) == 0L,
    call = {
      args <- .subset2(ir, "args")
      for (i in seq_along(args)) {
        if (!ir_is_const(.subset2(args, i))) {
          return(FALSE)
        }
      }
      TRUE
    },
    TRUE
  )
}

#' Does an IR (sub)tree contain a window-function call anywhere?
#'
#' Phase 5 (window functions, `scratchpad/phase5_window_design.md` section
#' 1.1): `TRUE` iff any `call` node in the subtree has a non-`NULL` `window`
#' field in its [ir_call_registry] entry. Used by [ir_bind()]'s window
#' guard, by the decomposition machinery in `R/window.R`
#' (`ir_extract_windows()`/`plan_window_stages()`), and will be used by the
#' as-yet-unwired grouped `mutate()`/`filter()` (W3/W5) to detect when a
#' parsed expression needs the window decomposition path at all.
#'
#' Accepts either a single IR node (identified by the `cuplyr_ir` class
#' every [ir_col()]/[ir_lit()]/[ir_call()] node carries) or a plain
#' (possibly unnamed) list of IR nodes -- e.g. a call node's own `$args` --
#' so callers can check a call's argument list directly without unpacking
#' it first.
#'
#' @param ir An IR node, a list of IR nodes, or `NULL`
#' @return Logical scalar
#' @keywords internal
ir_has_window <- function(ir) {
  if (is.null(ir)) {
    return(FALSE)
  }

  if (inherits(ir, "cuplyr_ir")) {
    if (!identical(.subset2(ir, "kind"), "call")) {
      return(FALSE)
    }
    entry <- ir_call_registry[[.subset2(ir, "op")]]
    if (!is.null(entry) && !is.null(entry$window)) {
      return(TRUE)
    }
    args <- .subset2(ir, "args")
    for (i in seq_along(args)) {
      arg <- .subset2(args, i)
      # (an IR leaf is no window call: answered without the call)
      if (inherits(arg, "cuplyr_ir") && !identical(.subset2(arg, "kind"), "call")) {
        next
      }
      if (ir_has_window(arg)) {
        return(TRUE)
      }
    }
    return(FALSE)
  }

  # A plain list of IR nodes (e.g. a call node's `$args`).
  for (i in seq_along(ir)) {
    if (ir_has_window(.subset2(ir, i))) {
      return(TRUE)
    }
  }
  FALSE
}

#' Bind 0-based column indices onto every `col` node in an IR tree
#'
#' Per section 2.1's decision, indices are resolved against a schema only
#' at lowering time (never at parse time), since projection pushdown can
#' shift column positions in between. Returns a new tree (does not mutate
#' in place, matching R's copy-on-modify semantics) with an `index` field
#' (0-based, for C++) added to every `col` node.
#'
#' @param ir An IR node (or `NULL`)
#' @param schema List with `names`/`types` to resolve column positions
#'   against
#' @return The IR tree with `index` filled in on every `col` node
#' @keywords internal
ir_bind <- function(ir, schema) {
  if (is.null(ir)) {
    return(ir)
  }

  kind <- .subset2(ir, "kind")
  if (kind == "col") {
    idx <- match(ir$name, schema$names)
    if (is.na(idx)) {
      stop("Column '", ir$name, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    ir$index <- idx - 1L
    return(ir)
  }

  if (kind == "lit" || kind == "vec") {
    return(ir)
  }

  # Phase 5 (window functions) invariant: a window-class op (registry entry
  # has a non-NULL `window` field, see ir_call_registry's window entries and
  # ir_has_window()) must NEVER reach lowering directly -- it can only ever
  # appear as an intermediate node that R/window.R's ir_extract_windows()/
  # plan_window_stages() rewrites away (into an ir_col() reference to a
  # materialized window-spec's output column) before a verb's own
  # ast_mutate()/ast_filter()/etc. node is built. Nothing in the package
  # wires window ops into a verb yet (W1 is dormant), so this guard should
  # be unreachable in practice; it exists so that a future verb (or a
  # currently-live mutate()/filter() call that now happens to *parse*
  # successfully because "mean"/"lag"/etc. are registry entries) fails
  # loudly here instead of silently mis-executing as an ordinary AST op.
  op <- .subset2(ir, "op")
  entry <- ir_call_registry[[op]]
  if (!is.null(entry) && !is.null(entry$window)) {
    stop("internal: window op '", op, "' reached lowering", call. = FALSE)
  }

  # F50: a factor column already IS its INT32 codes, so factor_codes(f) binds
  # to the column itself (a comparison on it stays one cudf AST kernel).
  if (identical(op, "factor_codes")) {
    return(ir_bind(.subset2(ir, "args")[[1]], schema))
  }

  args <- .subset2(ir, "args")
  for (i in seq_along(args)) {
    arg <- .subset2(args, i)
    arg_kind <- .subset2(arg, "kind")
    # (a literal binds to itself, and a column is bound here, without a call
    # per leaf; a call node, or anything else, goes through ir_bind())
    if (identical(arg_kind, "lit") || identical(arg_kind, "vec")) {
      next
    }
    if (identical(arg_kind, "col")) {
      idx <- match(.subset2(arg, "name"), schema$names)
      if (is.na(idx)) {
        stop("Column '", .subset2(arg, "name"), "' not found.\n",
             "Available columns: ", paste(schema$names, collapse = ", "),
             call. = FALSE)
      }
      arg$index <- idx - 1L
      args[i] <- list(arg)
    } else {
      args[i] <- list(ir_bind(arg, schema))
    }
  }
  ir$args <- args
  ir
}

#' Create a side-tagged IR column reference for a non-equi `join_by()`
#' condition (Phase 7 J7)
#'
#' A plain [ir_col()] node is resolved against a *single* schema by name
#' ([ir_bind()]) -- fine for `filter()`/`mutate()`, which only ever see one
#' table. A join condition (`join_by(a >= b)`) references columns from TWO
#' tables at once, and -- crucially -- a self-join (`inner_join(x, x,
#' join_by(a >= b))`) has IDENTICAL left/right schemas, so a name alone
#' can't say which side `a`/`b` belong to. `join_by()`'s own pre-normalized
#' spec already tells us explicitly, though: `x`/`y` are parallel vectors
#' where `x[i]` is ALWAYS the left-table column and `y[i]` is ALWAYS the
#' right-table column for condition `i` (see `join_by_is_non_equi()`'s own
#' roxygen, R/join.R). `join_cond_ir_from_spec()` (R/join.R) uses that
#' knowledge directly when building each condition's IR, tagging every
#' `col` node with an explicit `side` (`0L` = left/x, `1L` = right/y) at
#' construction time -- deliberately NOT the design doc's original
#' "resolve left-first-then-right by lookup order" sketch, which silently
#' breaks on a self-join. [ir_bind_join()] then resolves `$index` against
#' the correct side's schema using this tag, never by cross-schema lookup.
#' @param name Character scalar, the column name
#' @param side `0L` (left/x) or `1L` (right/y)
#' @return A `cuplyr_ir_col` node with an extra `side` field
#' @keywords internal
ir_col_join <- function(name, side) {
  node <- ir_col(name)
  node$side <- as.integer(side)
  node
}

#' Bind 0-based, per-side column indices onto a non-equi join condition IR
#' tree (Phase 7 J7)
#'
#' The join-condition analogue of [ir_bind()]. Every `col` node in a
#' condition IR built by `join_cond_ir_from_spec()` (R/join.R) already
#' carries an explicit `side` tag (see [ir_col_join()]'s docs for why this
#' package deviates from the design doc's lookup-order sketch: a self-join
#' has identical left/right schemas, so name-based resolution against
#' `left_schema` first would silently mis-bind a right-only column, or
#' bind a same-named column to the wrong side entirely). This function
#' simply resolves each `col` node's `$index` against `left_schema` (side
#' `0L`) or `right_schema` (side `1L`) directly, using that tag -- never a
#' generic single-schema lookup.
#' @param ir An IR node (or `NULL`), every `col` descendant carrying a
#'   `side` field ([ir_col_join()])
#' @param left_schema,right_schema List with `names`/`types` for the join's
#'   left/right inputs
#' @return The IR tree with `index` filled in on every `col` node
#' @keywords internal
ir_bind_join <- function(ir, left_schema, right_schema) {
  if (is.null(ir)) {
    return(ir)
  }

  if (ir$kind == "col") {
    schema <- if (identical(ir$side, 1L)) right_schema else left_schema
    idx <- match(ir$name, schema$names)
    if (is.na(idx)) {
      side_name <- if (identical(ir$side, 1L)) "Right" else "Left"
      stop(side_name, " join condition column '", ir$name, "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    ir$index <- idx - 1L
    return(ir)
  }

  # kind == "call": every non-equi join condition IR is built exclusively
  # from comparison ops folded with "&" (join_cond_ir_from_spec()) -- never
  # a literal (join_by() itself only ever accepts column-to-column
  # comparisons, e.g. `join_by(a >= b)`; `join_by(a >= 5)` is a dplyr parse
  # error before this package ever sees it) -- so no "lit" branch is needed
  # here, unlike ir_bind()'s generic single-schema version.
  ir$args <- lapply(ir$args, ir_bind_join, left_schema = left_schema, right_schema = right_schema)
  ir
}

#' Deparse an IR node back to readable text
#'
#' For diagnostics (`show_query()`, error messages) -- not intended to be a
#' faithful/round-trippable R deparser, just a readable rendering.
#'
#' @param ir An IR node (or `NULL`)
#' @return Character scalar
#' @keywords internal
ir_deparse <- function(ir) {
  if (is.null(ir)) {
    return("NULL")
  }

  switch(ir$kind,
    col = ir$name,
    vec = paste0("<", ir$type, "[", length(ir$value), "]>"),
    lit = {
      if (isTRUE(ir$na)) {
        "NA"
      } else if (is.null(ir$value)) {
        "NULL"
      } else if (length(ir$value) > 1) {
        paste0("c(", paste(format(ir$value), collapse = ", "), ")")
      } else {
        format(ir$value)
      }
    },
    call = {
      arg_strs <- vapply(ir$args, ir_deparse, character(1))
      paste0(ir$op, "(", paste(arg_strs, collapse = ", "), ")")
    }
  )
}

#' Where may a factor stay on the GPU? (F18's rule, one bottom-up pass)
#'
#' The GPU holds only a factor's integer codes, while dplyr/base R resolve a
#' factor through its labels. A *factor operand* is a DICTIONARY32 column
#' reference, or a DICTIONARY32-typed `lit`/`vec` node (a factor value from
#' the environment). It may stay on the GPU only
#'
#' 1. as a bare column reference (plain copy, key, or sort key on its own);
#' 2. directly inside `is.na()` / `n_distinct()`;
#' 3. as the column of F50's `factor_codes()`/`factor_labels()` (the parse
#'    hooks turn a label comparison or `as.character(f)` into these);
#' 4. as both operands of `==`/`!=` when both are factor *columns* with
#'    identical level vectors (labels, order and `ordered` flag); `<`/`<=`/
#'    `>`/`>=` additionally need both ordered. (`%in%` is not here: the GPU's
#'    `%in%` only takes a literal set.)
#'
#' Anything else that touches a factor operand (comparisons with non-factors
#' or other levels, arithmetic, helpers, windows, aggregates, factor values
#' from the environment) needs dplyr's labels, warnings and errors, so callers
#' send it to their CPU fallback.
#'
#' Each node is visited once: a node's status is derived from its children's
#' statuses, never by re-walking a subtree, so the cost is linear in IR size.
#' IR nodes stored in a call's `meta` (e.g. `lag()`'s `default`) count as
#' children.
#'
#' @param ir An IR node (or `NULL`)
#' @param schema List with `names`/`types` (and `factor_levels`)
#' @return `"none"` (no factor operand outside the allowed places), `"bare"`
#'   (the root itself is a bare factor column), or `"cpu"`
#' @keywords internal
ir_factor_rule <- function(ir, schema) {
  # no DICTIONARY32 column and no such literal: nothing in the tree is "bare" or
  # "cpu", so every node is "none"
  if (!"DICTIONARY32" %in% schema$types && !ir_has_dictionary_value(ir)) {
    return("none")
  }
  dup <- anyDuplicated(schema$names) > 0L
  levels_of <- function(name) {
    if (dup) {
      return(NULL)
    }
    schema$factor_levels[[name]]
  }
  same_levels <- function(a, b) {
    la <- levels_of(a$name)
    lb <- levels_of(b$name)
    !is.null(la) && !is.null(lb) && identical(la, lb)
  }
  visit <- function(node) {
    if (is.null(node)) {
      return("none")
    }
    switch(node$kind,
      col = if (identical(unname(schema$types[match(node$name, schema$names)]), "DICTIONARY32")) "bare" else "none",
      lit = ,
      vec = if (identical(unname(node$type), "DICTIONARY32")) "cpu" else "none",
      call = {
        meta_nodes <- Filter(function(m) is.list(m) && !is.null(m$kind), node$meta)
        st <- vapply(c(node$args, meta_nodes), visit, character(1))
        if (any(st == "cpu")) {
          return("cpu")
        }
        if (!any(st == "bare")) {
          return("none")
        }
        op <- node$op
        if (op %in% c("is.na", "n_distinct")) {
          return("none")
        }
        # F50: the rewrites of ir_parse_factor_cmp()/ir_parse_factor_conversion()
        # (codes of / labels of a bare factor column) leave no factor operand.
        if (op %in% c("factor_codes", "factor_labels") && identical(node$args[[1]]$kind, "col") &&
            st[1] == "bare") {
          return("none")
        }
        if (length(node$args) == 2L && length(meta_nodes) == 0L && all(st == "bare") &&
            identical(node$args[[1]]$kind, "col") && identical(node$args[[2]]$kind, "col") &&
            same_levels(node$args[[1]], node$args[[2]])) {
          if (op %in% c("==", "!=")) {
            return("none")
          }
          if (op %in% c("<", "<=", ">", ">=") &&
              isTRUE(attr(levels_of(node$args[[1]]$name), "ordered"))) {
            return("none")
          }
        }
        "cpu"
      },
      "none"
    )
  }
  visit(ir)
}

# Is there a DICTIONARY32 literal or vector node in `node` (call arguments and
# IR nodes in `meta`, as ir_factor_rule() visits them)?
# @keywords internal
ir_has_dictionary_value <- function(node) {
  if (is.null(node)) {
    return(FALSE)
  }
  kind <- .subset2(node, "kind")
  if (identical(kind, "call")) {
    args <- .subset2(node, "args")
    for (i in seq_along(args)) {
      arg <- .subset2(args, i)
      arg_kind <- .subset2(arg, "kind")
      # (a call argument recurses; a leaf is answered here, without a call)
      if (identical(arg_kind, "call")) {
        if (ir_has_dictionary_value(arg)) {
          return(TRUE)
        }
      } else if ((identical(arg_kind, "lit") || identical(arg_kind, "vec")) &&
                   identical(unname(.subset2(arg, "type")), "DICTIONARY32")) {
        return(TRUE)
      }
    }
    for (m in .subset2(node, "meta")) {
      if (is.list(m) && !is.null(m$kind) && ir_has_dictionary_value(m)) {
        return(TRUE)
      }
    }
    return(FALSE)
  }
  (identical(kind, "lit") || identical(kind, "vec")) &&
    identical(unname(.subset2(node, "type")), "DICTIONARY32")
}

#' Infer the GPU output type of an IR node
#'
#' Recurses; for arithmetic call nodes this delegates to the existing
#' [infer_mutate_output_type()] unchanged, so D5's rules (and its pinned
#' STRING/DICTIONARY32/TIMESTAMP_* error messages) survive verbatim. Every
#' other op class is resolved via its [ir_call_registry] entry's `type`
#' function (section 1.3).
#'
#' @param ir An IR node
#' @param schema List with `names`/`types` to resolve column types against
#' @return GPU type string
#' @keywords internal
ir_infer_type <- function(ir, schema) {
  # .subset2(): the nodes are classed lists, `$` would dispatch on them
  kind <- .subset2(ir, "kind")
  if (kind == "col") {
    idx <- match(.subset2(ir, "name"), schema$names)
    if (is.na(idx)) {
      stop("Column '", .subset2(ir, "name"), "' not found.\n",
           "Available columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }
    return(schema$types[idx])
  }

  if (kind == "vec") {
    return(.subset2(ir, "type"))
  }

  if (kind == "lit") {
    # A bare NA (type = NULL) defaults to BOOL8 when asked in isolation --
    # matching `typeof(NA)` ("logical") in R, so `mutate(y = NA)` types `y`
    # as a logical column exactly like dplyr (T6 finding: this used to
    # default to FLOAT64 on the theory that it's always asked as part of a
    # call's arg_types where its contribution is a promotion-union no-op;
    # empirically wrong for a bare NA as an entire top-level mutate
    # expression, which never reaches a call node at all). See the "call"
    # branch below for the sibling-adoption logic that keeps this default
    # from over-eagerly promoting a *nested* NA in e.g. `x + NA`.
    type <- .subset2(ir, "type")
    if (is.null(type)) {
      return("BOOL8")
    }
    # (both literal checks return at once unless the literal is a Date value
    # or a Period/Duration/difftime object)
    if (identical(unname(type), "TIMESTAMP_DAYS")) {
      ir_check_ts_lit(ir)
    }
    value <- .subset2(ir, "value")
    if (isS4(value) || inherits(value, "difftime")) {
      ir_check_timespan_lit(value)
    }
    return(type)
  }

  op <- .subset2(ir, "op")
  entry <- ir_call_registry[[op]]
  if (is.null(entry)) {
    stop("Unknown IR op: '", op, "'", call. = FALSE)
  }

  args <- .subset2(ir, "args")
  n <- length(args)
  arg_types <- character(n)
  is_isolated_na <- logical(n)
  for (i in seq_len(n)) {
    a <- .subset2(args, i)
    a_kind <- .subset2(a, "kind")
    # (a column's type is read here, without a call per leaf)
    t <- if (identical(a_kind, "col")) {
      idx <- match(.subset2(a, "name"), schema$names)
      if (is.na(idx)) {
        stop("Column '", .subset2(a, "name"), "' not found.\n",
             "Available columns: ", paste(schema$names, collapse = ", "),
             call. = FALSE)
      }
      schema$types[idx]
    } else {
      ir_infer_type(a, schema)
    }
    if (!is.character(t) || length(t) != 1L) {
      # not a single type string: vapply() raises its error for it
      vapply(args, ir_infer_type, character(1), schema = schema)
    }
    arg_types[[i]] <- t
    is_isolated_na[[i]] <- identical(a_kind, "lit") && ir_is_untyped_na(a)
  }
  ir_call_type(ir, entry, arg_types, is_isolated_na)
}

# A bare, untyped NA literal (its kind already known to be "lit").
ir_is_untyped_na <- function(a) {
  isTRUE(.subset2(a, "na")) && is.null(.subset2(a, "type"))
}

# The type of call node `ir` (registry entry `entry`) given the inferred types
# of its arguments, in order, and which arguments are bare untyped NA literals:
# the step ir_infer_type() takes once its arguments are typed, shared with
# ir_has_checked_int(), which types each subtree once.
ir_call_type <- function(ir, entry, arg_types, is_isolated_na) {
  args <- .subset2(ir, "args")
  op <- .subset2(ir, "op")
  # A bare, untyped NA literal argument (kind="lit", na=TRUE, type=NULL) is
  # a type no-op in R's arithmetic: `1L + NA` is integer, not double (NA is
  # coerced to the OTHER operand's type, it never forces a promotion). Since
  # each arg's type is inferred independently, an isolated NA arg
  # would otherwise surface as the "in isolation" BOOL8 default from the
  # branch above and incorrectly feed into infer_mutate_output_type()'s
  # promotion union. Substitute it with a sibling arg's type first (mirrors
  # src/expr_eval.hpp's resolve_static_type(), which does the same
  # adoption when building the actual AST) -- T6 finding: without this,
  # `mutate(y = int_col + NA)` came back FLOAT64 instead of matching
  # dplyr's INT32.
  if (!is.null(nms <- names(args))) {
    names(arg_types) <- nms
  }
  if (any(is_isolated_na) && !all(is_isolated_na)) {
    arg_types[is_isolated_na] <- arg_types[!is_isolated_na][1]
  }

  # F37: a timestamp argument the GPU can't take goes to the CPU, whatever
  # built the node (see ir_check_ts_operands(), which returns at once unless
  # an argument is a timestamp).
  if (any(match(arg_types, cuplyr_timestamp_types, 0L) > 0L)) {
    ir_check_ts_operands(ir, arg_types)
  }
  # F48: the same for the STRING/BOOL8/INT32 arguments of reducers, windows
  # and comparisons (see ir_native_arg_types). ir_check_type_operands()
  # judges the ops of ir_native_arg_types, and a comparison or `%in%` that has
  # a STRING argument; it returns at once for anything else.
  if (match(op, ir_native_arg_op_names, 0L) > 0L ||
        (match(op, ir_string_checked_ops, 0L) > 0L && "STRING" %in% arg_types)) {
    ir_check_type_operands(ir, arg_types)
  }

  entry$type(arg_types)
}

#' Infer the time zone of a POSIXct-valued IR node (F32)
#'
#' The sibling of [ir_infer_type()] for the `tzone` attribute the result
#' would carry in R. Only meaningful for a TIMESTAMP_MICROSECONDS result (it
#' returns `"UTC"` for anything else): a column has its schema zone, a POSIXct
#' literal its own `tzone`, and every op that can return a POSIXct has an
#' explicit rule in [ir_tz_rules]. An op without one raises
#' `cuplyr_timestamp_needs_cpu` (the expression goes to the CPU when the
#' parse sees it, [ir_parse_quo()]) rather than guess a zone.
#'
#' @param ir An IR node
#' @param schema List with `names`/`types`/`tz`
#' @return A zone string as stored in `schema$tz`: a name, `""` (local), `NA`
#'   (no tzone attribute) or `"UTC"`
#' @keywords internal
ir_infer_tz <- function(ir, schema) {
  zone <- ir_tz_of(ir, schema)
  if (is.null(zone)) "UTC" else zone
}

# F46: the native lubridate rounding ops, "<fn>_date_<unit>" (the unit is
# part of the op name; ir_parse_lubridate_rounding() picks the combination).
# floor_date()/ceiling_date()/round_date() return x's class and zone.
# @keywords internal
ir_round_ops <- as.vector(outer(
  c("floor_date", "ceiling_date", "round_date"),
  c("second", "minute", "hour", "day", "week"),
  paste, sep = "_"
))

#' How each POSIXct-valued op gets its zone (F32)
#'
#' Verified against dplyr 1.2 / vctrs 0.7 on plain tibbles, for local,
#' absent, UTC and New York operands:
#'
#' - `x`: the result is `x` itself, a slice of it, or other values cast to
#'   `x`'s type (`vec_cast(y, x)`), so it has `x`'s zone (absent becomes
#'   `""`): `na_if()`, `replace_when()` (also `replace_values()`), `lag()`/
#'   `lead()` (with `default = `), `first()`/`last()`/`nth()`, `min()`/
#'   `max()`, `mean()`/`median()` (F41), and cuplyr's `roll_min()`/`roll_max()`.
#' - `common`: vctrs' common type of the value operands in order
#'   ([tz_fold()]), then absent becomes `""`: `if_else()` (true, false,
#'   missing), `case_when()` (values in order, then `.default`),
#'   `coalesce()`.
#' - `as_posixct`: a POSIXct keeps its zone as is, a Date becomes UTC
#'   midnight (R's `as.POSIXct.Date`).
#' - `shift` (F46): `ts_add_us`/`ts_sub_us` keep `x`'s zone attribute as is
#'   (an absent one stays absent).
#' - `round_date` (F46): `round_date()` of a local or absent zone returns the
#'   session zone's name; floor_date()/ceiling_date() (in `x`) keep `""`.
#'
#' Not listed, so CPU: unary `-`, `abs()`, `cummax()`/`cummin()` (R's
#' `Math.POSIXt` rejects them, so dplyr errors). A bare `t +/- n` never
#' reaches here: its type check sends it to the CPU
#' ([infer_mutate_output_type()]); only the rounding calls rewrite it
#' ([ir_posixct_shift()]).
#'
#' @format A named list of op names per rule
#' @keywords internal
ir_tz_rules <- list(
  x = c("na_if", "replace_when", "lag", "lead", "first", "last", "nth",
        "min", "max", "mean", "median", "roll_min", "roll_max",
        grep("^(floor|ceiling)_date_", ir_round_ops, value = TRUE)),
  common = c("if_else", "case_when", "coalesce"),
  as_posixct = "as_posixct",
  # F46: `t +/- n` keeps x's zone attribute as is (an absent one stays absent)
  shift = c("ts_add_us", "ts_sub_us"),
  # F46: timechange::time_round() takes the zone from lubridate's tz(), so a
  # local ("" / absent) zone comes back as the session zone's name, unlike
  # floor_date()/ceiling_date() which keep "".
  round_date = grep("^round_date_", ir_round_ops, value = TRUE)
)

# The value operands of a vctrs-common op, in fold order.
# @keywords internal
ir_tz_value_args <- function(ir) {
  args <- ir$args
  n <- length(args)
  switch(ir$op,
    if_else = args[-1],                                 # true, false, missing
    case_when = args[c(seq(2L, n - 1L, by = 2L), n)],   # val_1, ..., default
    args                                                # coalesce
  )
}

# `NULL` when `ir` is not POSIXct-valued, else its zone.
# @keywords internal
ir_tz_of <- function(ir, schema) {
  if (identical(ir$kind, "col")) {
    idx <- match(ir$name, schema$names)
    if (is.na(idx) || !identical(unname(schema$types[idx]), "TIMESTAMP_MICROSECONDS")) {
      return(NULL)
    }
    return(tz_by_position(schema)[[idx]])
  }
  if (identical(ir$kind, "lit") || identical(ir$kind, "vec")) {
    if (!identical(unname(ir$type), "TIMESTAMP_MICROSECONDS") || isTRUE(ir$na)) {
      return(NULL)
    }
    return(tz_of_attr(ir$value))
  }
  if (!identical(unname(ir_infer_type(ir, schema)), "TIMESTAMP_MICROSECONDS")) {
    return(NULL)
  }
  op <- ir$op
  if (op %in% ir_tz_rules$x) {
    # the result has x's type, so x is the POSIXct that sets the zone
    return(tz_localize(ir_tz_of(ir$args[[1]], schema) %||% "UTC"))
  }
  if (op %in% ir_tz_rules$common) {
    zone <- NULL
    for (arg in ir_tz_value_args(ir)) {
      zone <- tz_fold(zone, ir_tz_of(arg, schema))
    }
    return(tz_localize(zone %||% "UTC"))
  }
  if (op %in% ir_tz_rules$shift) {
    return(ir_tz_of(ir$args[[1]], schema) %||% "UTC")
  }
  if (op %in% ir_tz_rules$round_date) {
    zone <- ir_tz_of(ir$args[[1]], schema) %||% "UTC"
    return(if (is.na(zone) || !nzchar(zone)) session_tz() else zone)
  }
  if (op %in% ir_tz_rules$as_posixct) {
    # a Date becomes UTC midnight; a POSIXct keeps its zone
    return(ir_tz_of(ir$args[[1]], schema) %||% "UTC")
  }
  stop_timestamp_needs_cpu(
    op, "() on a POSIXct value: no time zone rule on the GPU path."
  )
}

# Run ir_tz_of() over every POSIXct-valued call of a parsed expression, so an
# op without a zone rule (ir_tz_rules) sends the whole expression to the CPU
# at parse time, wherever it sits in the tree. A no-op without a POSIXct
# column in the schema (constant subtrees are already folded in R).
# @keywords internal
ir_check_tz_rules <- function(ir, schema) {
  if (!any(unname(schema$types) == "TIMESTAMP_MICROSECONDS")) {
    return(invisible())
  }
  visit <- function(node) {
    if (!identical(node$kind, "call")) {
      return(invisible())
    }
    meta_nodes <- Filter(function(m) is.list(m) && !is.null(m$kind), node$meta)
    for (child in c(node$args, meta_nodes)) visit(child)
    ir_tz_of(node, schema)
  }
  visit(ir)
  invisible()
}

# -----------------------------------------------------------------------------
# ir_call_registry - the extension contract (section 6)
# -----------------------------------------------------------------------------
#
# Adding a function is two edits, never a parser change:
#   R side: one entry here -- arity (int or NA for variable), an optional
#     `parse` desugaring hook (NULL if the op needs none), a `type`
#     function computing the GPU output type from argument types, and a
#     `lower` spec: list(ast_op = "<ENUM>") for an AST-native op (T2 looks
#     the enum name up directly), or list(handler = "<name>") for a
#     non-AST cuDF call (T2 adds one case in apply_handler()). `lower` is
#     NULL for ops that are always fully desugared at parse time (between,
#     log2, log10) and therefore never survive to reach lowering under
#     their own op name.
#   C++ side: nothing for ast_op entries; one apply_handler() case for
#     handler entries.

# F50: the labels of a bare factor column node, or NULL when the node is not
# one, the schema's names are duplicated (the levels can't be told apart, as in
# ir_factor_rule()), the levels are unknown, or a level is NA (an NA level is
# lost on upload, so the rewrites below would be wrong): the callers then keep
# the plain node and F18's rule sends it to the CPU as before.
# @keywords internal
ir_factor_col_levels <- function(node, schema) {
  if (!identical(node$kind, "col") || !"DICTIONARY32" %in% schema$types ||
      anyDuplicated(schema$names) > 0L) {
    return(NULL)
  }
  i <- match(node$name, schema$names)
  if (is.na(i) || !identical(unname(schema$types[i]), "DICTIONARY32")) {
    return(NULL)
  }
  lv <- schema$factor_levels[[node$name]]
  if (is.null(lv) || anyNA(lv)) {
    return(NULL)
  }
  lv
}

# F50: `factor_codes(f)` for a bare factor column node.
# @keywords internal
ir_factor_codes <- function(col) {
  ir_call("factor_codes", list(col))
}

# F50: the label a comparison operand stands for: a length-1 STRING literal
# (NA_character_ included), or, when `allow_na` is set, the bare logical NA.
# Returns list(label = <chr1>) or NULL.
# @keywords internal
ir_factor_label_operand <- function(node, allow_na = TRUE) {
  if (!identical(node$kind, "lit")) {
    return(NULL)
  }
  if (isTRUE(node$na) && is.null(node$type) && allow_na) {
    return(list(label = NA_character_))
  }
  if (identical(unname(node$type), "STRING") && length(node$value) == 1L && is.character(node$value)) {
    return(list(label = unname(node$value)))
  }
  NULL
}

# F50: the parse hook of ==, !=, <, <=, >, >=. A bare factor column against a
# STRING label becomes a comparison of INT32 codes against the label's level
# position, keeping the operand order. A label that is not a level matches no
# code: 0L for ==/!= (FALSE / TRUE, NA rows stay NA), NA for the ordering ops
# (NA everywhere, no warning, as in base R's Ops.ordered); an NA label gives an
# NA code either way. The ordering ops are rewritten for an ordered factor
# only (an unordered one keeps the node so R's "not meaningful for factors"
# warning comes from the CPU). Every other shape keeps the plain node.
# @keywords internal
ir_parse_factor_cmp <- function(op) {
  force(op)
  is_order <- op %in% c("<", "<=", ">", ">=")
  function(args, schema) {
    plain <- ir_call(op, args)
    if (length(args) != 2L || !"DICTIONARY32" %in% schema$types) {
      return(plain)
    }
    levels_ <- lapply(args, ir_factor_col_levels, schema = schema)
    is_fct <- !vapply(levels_, is.null, logical(1))
    if (sum(is_fct) != 1L) {
      return(plain)
    }
    f_pos <- which(is_fct)
    lv <- levels_[[f_pos]]
    if (is_order && !isTRUE(attr(lv, "ordered"))) {
      return(plain)
    }
    operand <- ir_factor_label_operand(args[[3L - f_pos]], allow_na = !is_order)
    if (is.null(operand)) {
      return(plain)
    }
    k <- match(operand$label, lv)
    if (is.na(operand$label) || (is_order && is.na(k))) {
      code <- ir_lit(NULL, "INT32", na = TRUE)
    } else {
      code <- ir_lit(if (is.na(k)) 0L else k, "INT32")
    }
    new_args <- args
    new_args[[f_pos]] <- ir_factor_codes(args[[f_pos]])
    new_args[[3L - f_pos]] <- code
    ir_call(op, new_args)
  }
}

# F50: the parse hook of %in%. A bare factor column against a set of labels (a
# STRING literal/vector, the bare NA, an empty set, or a factor value, whose
# labels are matched, as base R's match() does) becomes `codes %in% positions`.
# Non-levels drop out of the set; an NA in it stays NA and matches the NA
# codes. Everything else (a number set, a factor on the right) keeps the
# plain node.
# @keywords internal
ir_parse_factor_in <- function(args, schema) {
  plain <- ir_call("%in%", args)
  if (length(args) != 2L || !"DICTIONARY32" %in% schema$types) {
    return(plain)
  }
  lv <- ir_factor_col_levels(args[[1]], schema)
  set <- args[[2]]
  if (is.null(lv) || !identical(set$kind, "lit")) {
    return(plain)
  }
  labels <- if (isTRUE(set$na) && is.null(set$type)) {
    NA_character_
  } else if (is.null(set$type) && is.null(set$value)) {
    character(0)
  } else if (identical(unname(set$type), "STRING")) {
    set$value
  } else if (identical(unname(set$type), "DICTIONARY32") && is.factor(set$value)) {
    as.character(set$value)
  } else {
    return(plain)
  }
  pos <- match(labels, lv)
  pos <- unique(pos[!is.na(pos) | is.na(labels)])
  codes <- if (length(pos) == 1L && is.na(pos)) {
    ir_lit(NULL, "INT32", na = TRUE)
  } else {
    ir_lit(as.integer(pos), "INT32")
  }
  ir_call("%in%", list(ir_factor_codes(args[[1]]), codes))
}

# F50: TRUE iff every string in `x` reaches the GPU as UTF-8. Strings are
# uploaded as their raw CHAR() bytes and compared bytewise there, so a
# latin1/bytes string (read.csv(encoding = "latin1")) would never equal a
# UTF-8 one, while dplyr translates both. A string counts when it is marked
# UTF-8, or unmarked ("unknown", the native encoding) and either ASCII or
# valid UTF-8 in a UTF-8 session. Literals in a UTF-8 session's source code
# are marked UTF-8, so the usual non-ASCII labels ("\u00e9t\u00e9") pass.
# @keywords internal
ir_strings_are_utf8 <- function(x) {
  enc <- Encoding(x)
  if (any(enc %in% c("latin1", "bytes"))) {
    return(FALSE)
  }
  native <- x[enc == "unknown" & !is.na(x)]
  if (length(native) == 0L) {
    return(TRUE)
  }
  if (isTRUE(l10n_info()[["UTF-8"]])) {
    return(all(validUTF8(native)))
  }
  # a non-UTF-8 session: native non-ASCII bytes are in that locale's encoding
  !anyNA(iconv(native, "ASCII", "ASCII"))
}

# Does a factor_labels call occur anywhere in `node`, walked as
# ir_decode_meets_non_utf8()'s collecting walk does (call arguments, IR nodes
# in `meta`, plain lists)?
# @keywords internal
ir_has_factor_labels <- function(node) {
  if (inherits(node, "cuplyr_ir")) {
    if (!identical(.subset2(node, "kind"), "call")) {
      return(FALSE)
    }
    if (identical(.subset2(node, "op"), "factor_labels")) {
      return(TRUE)
    }
    args <- .subset2(node, "args")
    for (i in seq_along(args)) {
      arg <- .subset2(args, i)
      # (an IR leaf has no factor_labels call: answered without the call)
      if (inherits(arg, "cuplyr_ir") && !identical(.subset2(arg, "kind"), "call")) {
        next
      }
      if (ir_has_factor_labels(arg)) {
        return(TRUE)
      }
    }
    for (m in .subset2(node, "meta")) {
      if (is.list(m) && !is.null(m$kind) && ir_has_factor_labels(m)) {
        return(TRUE)
      }
    }
    return(FALSE)
  }
  if (is.list(node)) {
    for (i in seq_along(node)) {
      if (ir_has_factor_labels(.subset2(node, i))) {
        return(TRUE)
      }
    }
  }
  FALSE
}

# F50: TRUE iff `ir` decodes factor labels (factor_labels, whose levels
# ir_parse_factor_conversion() only builds from UTF-8 strings) and also holds
# a string literal that isn't UTF-8 (`as.character(f) == lit` with a latin1
# `lit`): the GPU would compare the two byte encodings and never match.
# @keywords internal
ir_decode_meets_non_utf8 <- function(ir) {
  # the result needs a factor_labels call; look for one before collecting strings
  if (!ir_has_factor_labels(ir)) {
    return(FALSE)
  }
  decodes <- FALSE
  strings <- list()
  walk <- function(node) {
    if (inherits(node, "cuplyr_ir")) {
      if (identical(node$kind, "call")) {
        decodes <<- decodes || identical(node$op, "factor_labels")
        lapply(node$args, walk)
        # IR operands held in meta (lag()'s `default`), as ir_factor_rule() visits
        lapply(Filter(function(m) is.list(m) && !is.null(m$kind), node$meta), walk)
      } else if (identical(node$kind, "lit") && is.character(node$value)) {
        strings[[length(strings) + 1L]] <<- node$value
      }
    } else if (is.list(node)) {
      lapply(node, walk)
    }
    invisible()
  }
  walk(ir)
  decodes && !ir_strings_are_utf8(unlist(strings))
}

# F50: as.character()/as.integer()/as.numeric()/as.double() of a bare factor
# column (a symbol or `.data$f`): labels (NA for NA codes), codes, or codes as
# a double. Returns NULL for any other call, which then parses exactly as it
# did before (these names have no registry entry, so a constant argument
# folds, a column argument falls back).
# @keywords internal
ir_parse_factor_conversion <- function(expr, schema, fn_name) {
  if (length(expr) != 2L || (!is.null(names(expr)) && nzchar(names(expr)[2])) ||
      !"DICTIONARY32" %in% schema$types) {
    return(NULL)
  }
  # `(f)` and an embedded quosure (`as.character({{ v }})`) are transparent;
  # a symbol naming a column is that column whatever the quosure's env. An
  # unsupplied `{{ v }}` wraps the missing-argument sentinel, which can't be
  # read once bound to `arg`, so each slot is checked before it is extracted;
  # declining lets as.character(<missing>) fold to "" as in dplyr.
  if (rlang::is_missing(expr[[2]])) {
    return(NULL)
  }
  arg <- expr[[2]]
  repeat {
    if (rlang::is_quosure(arg)) {
      if (rlang::quo_is_missing(arg)) {
        return(NULL)
      }
      arg <- rlang::quo_get_expr(arg)
    } else if (rlang::is_call(arg, "(") && length(arg) == 2L) {
      if (rlang::is_missing(arg[[2]])) {
        return(NULL)
      }
      arg <- arg[[2]]
    } else {
      break
    }
  }
  name <- if (is.symbol(arg)) {
    as.character(arg)
  } else if (is_data_pronoun_call(arg)) {
    pronoun_call_name(arg)
  } else {
    return(NULL)
  }
  if (!name %in% schema$names) {
    return(NULL)
  }
  col <- ir_col(name)
  lv <- ir_factor_col_levels(col, schema)
  if (is.null(lv)) {
    return(NULL)
  }
  if (fn_name == "as.character" && !ir_strings_are_utf8(lv)) {
    return(NULL)
  }
  switch(fn_name,
    as.character = ir_call("factor_labels", list(col, ir_lit(as.character(lv), "STRING"))),
    as.integer = ir_factor_codes(col),
    ir_call("cast_float64", list(ir_factor_codes(col)))
  )
}

# R gives a logical operand a numeric type the GPU evaluator doesn't: window
# max()/min() and unary `-` of a logical are integers, round() of a logical is
# a double (and cuDF's round rejects a BOOL8 column). The parse hooks of those
# ops wrap a BOOL8 first argument in `cast_to` (`cast_int32`/`cast_float64`,
# nulls kept), so the node's inferred type and the evaluated column are R's
# type, for the plain verbs and everything that rewrites into them
# (across(), the colwise verbs). summarise()'s own max()/min() aggregates are
# parsed elsewhere (make_aggregation() casts in F34) and never get here.
# @keywords internal
ir_cast_bool8_first <- function(args, schema, cast_to) {
  if (length(args) < 1L) {
    return(args)
  }
  type <- tryCatch(ir_infer_type(args[[1L]], schema), error = function(e) NA_character_)
  if (identical(unname(type), "BOOL8")) {
    args[[1L]] <- ir_call(cast_to, list(args[[1L]]))
  }
  args
}

ir_parse_bool8_cast <- function(op, cast_to) {
  force(op)
  force(cast_to)
  function(args, schema) {
    ir_call(op, ir_cast_bool8_first(args, schema, cast_to))
  }
}

# The R value of a column-free IR subtree made of literals and pure ops
# (`-1`, `-(2)`, `1 + 1`), or `ok = FALSE`. Unlike ir_fold_literal_call() it also
# evaluates a unary minus, which that fold leaves as a call.
# @keywords internal
ir_const_value <- function(node) {
  got <- ir_lit_r_value(node)
  if (got$ok) {
    return(got)
  }
  if (!inherits(node, "cuplyr_ir") || !identical(node$kind, "call") ||
      !node$op %in% ir_foldable_pure_ops) {
    return(list(ok = FALSE))
  }
  vals <- vector("list", length(node$args))
  for (i in seq_along(node$args)) {
    v <- ir_const_value(node$args[[i]])
    if (!v$ok) {
      return(list(ok = FALSE))
    }
    vals[i] <- list(v$value)
  }
  ir_eval_pure_op(node$op, vals)
}

# F67: the parse hook of round(). The GPU rule is R's fround() for a constant
# `digits`, so the second argument must evaluate in R (column-free) to one
# non-NA numeric/logical value, which is passed on as a FLOAT64 literal with
# the raw double (the handler applies `floor(digits + 0.5)` itself). A column,
# NA/NaN, a vector or anything else returns NULL: the verb's CPU fallback gives
# dplyr's value or error. A logical x is cast to a double first (F19-a2).
ir_parse_round <- function(args, schema) {
  if (length(args) == 2L) {
    d <- ir_const_value(args[[2L]])
    if (!d$ok || !(is.numeric(d$value) || is.logical(d$value)) || is.na(d$value)) {
      return(NULL)
    }
    args[[2L]] <- ir_lit(as.double(d$value), "FLOAT64")
  } else if (length(args) != 1L) {
    return(NULL)
  }
  ir_call("round", ir_cast_bool8_first(args, schema, "cast_float64"))
}

# F37: the parse hook of "+" and "-" (it only sees parsed args). A Date
# shifted by a whole number of days becomes a native date_add_days/
# date_sub_days node; every other timestamp shape that dplyr can compute goes
# to the CPU (`stop_timestamp_needs_cpu()`): a fractional/NA/double shift,
# `ts - ts` (a difftime, which surfaces F18's upload error unless the result
# drops it), and a unary sign (base R errors on a unary minus). Shapes base R
# rejects (`n - d`, `d + d2`) and non-timestamp calls keep the plain node, so
# the type check raises its usual error.
ir_parse_ts_arith <- function(op) {
  force(op)
  function(args, schema) {
    if (op == "-" && length(args) == 1L) {
      args <- ir_cast_bool8_first(args, schema, "cast_int32")
    }
    plain <- ir_call(op, args)
    if (!ir_args_may_be_timestamp(args, schema)) {
      return(plain)
    }
    types <- tryCatch(
      vapply(args, ir_infer_type, character(1), schema = schema),
      error = function(e) NULL
    )
    if (is.null(types) || !any(types %in% cuplyr_timestamp_types)) {
      return(plain)
    }
    is_ts <- types %in% cuplyr_timestamp_types
    if (length(args) == 1L) {
      stop_timestamp_needs_cpu("unary ", op, " on a timestamp")
    }
    if (all(is_ts)) {
      if (op == "-") {
        stop_timestamp_needs_cpu("`-` of two timestamps is a difftime")
      }
      return(plain)
    }
    # a bare NA (`d + NA`) adopts the date's type, so the type check would
    # say "Date + Date"; dplyr gives an NA Date
    if (any(vapply(args, function(a) identical(a$kind, "lit") && isTRUE(a$na), logical(1)))) {
      stop_timestamp_needs_cpu("a timestamp shifted by NA")
    }
    ts_pos <- which(is_ts)
    if (op == "-" && ts_pos == 2L) {
      return(plain)   # number - date: R errors
    }
    if (!identical(unname(types[ts_pos]), "TIMESTAMP_DAYS")) {
      return(plain)   # POSIXct: F32's CPU route (ir_posixct_shift() for F46)
    }
    n <- args[[3L - ts_pos]]
    if (identical(n$kind, "lit")) {
      v <- n$value
      # Unclassed only, as in ir_posixct_shift(): a lubridate Period/Duration
      # (`days(1)`) is a double-storage S4 object that passes is.numeric(),
      # and as.integer() on it gives seconds, not days. The plain node's type
      # check sends it to the CPU (a difftime fails is.numeric() already).
      if (is.object(v) || !n$type %in% c("INT32", "FLOAT64")) {
        return(plain)
      }
      if (!(is.numeric(v) && length(v) == 1L && is.finite(v) && v == trunc(v) &&
            abs(v) <= .Machine$integer.max)) {
        return(plain)
      }
      n <- ir_lit(as.integer(v), "INT32")
    } else if (!identical(unname(types[3L - ts_pos]), "INT32")) {
      return(plain)   # a double/logical shift: dplyr may be fractional
    }
    ir_call(if (op == "+") "date_add_days" else "date_sub_days",
            list(args[[ts_pos]], n))
  }
}

# F46: the operand of a native floor_date()/ceiling_date()/round_date() may be a
# POSIXct shifted by a whole-number literal of seconds (`floor_date(t + 3600,
# "hour")`). Elsewhere `t +/- n` stays on F32's CPU route (the result's absent
# tzone attribute can't be told from "" in the rounded result, but can in a
# bare shift), so the rewrite happens here, only at the rounding call.
# @keywords internal
ir_posixct_shift <- function(ir, schema) {
  if (!identical(ir$kind, "call") || !ir$op %in% c("+", "-") || length(ir$args) != 2L) {
    return(ir)
  }
  types <- tryCatch(
    vapply(ir$args, ir_infer_type, character(1), schema = schema),
    error = function(e) NULL
  )
  ts_pos <- which(unname(types) == "TIMESTAMP_MICROSECONDS")
  if (length(ts_pos) != 1L || (ir$op == "-" && ts_pos == 2L)) {
    return(ir)
  }
  n <- ir$args[[3L - ts_pos]]
  v <- n$value
  # Unclassed only: a Period (`hours(1)`) is a double-storage S4 object whose
  # data part is 0, and a difftime/Duration has its own unit.
  if (!identical(n$kind, "lit") || isTRUE(n$na) || is.object(v) ||
      !n$type %in% c("INT32", "FLOAT64") || !is.numeric(v) || length(v) != 1L ||
      !is.finite(v) || v != trunc(v) || abs(v) > 1e9) {
    return(ir)
  }
  ir_call(if (ir$op == "+") "ts_add_us" else "ts_sub_us",
          list(ir$args[[ts_pos]], ir_lit(v * 1e6, "INT64")))
}

# Cheap pre-check for ir_parse_ts_arith(): the Date shift can only be built
# when the schema has a timestamp column or an argument is a timestamp
# literal. Skips re-inferring whole subtrees for every `x1 + x2 + ...` step.
# A timestamp buried deeper (`(as.Date(lit) + n) + 1L`) keeps the plain
# node, which is correct, just not native: every CPU decision of the hook is
# also made by the type inference (ir_check_ts_operands() for a unary sign
# and a bare NA, infer_mutate_output_type() for a shift and `ts - ts`).
# @keywords internal
ir_args_may_be_timestamp <- function(args, schema) {
  if (any(match(schema$types, cuplyr_timestamp_types, 0L) > 0L)) {
    return(TRUE)
  }
  for (i in seq_along(args)) {
    a <- .subset2(args, i)
    kind <- .subset2(a, "kind")
    if ((identical(kind, "lit") || identical(kind, "vec")) &&
          match(unname(.subset2(a, "type") %||% ""), cuplyr_timestamp_types, 0L) > 0L) {
      return(TRUE)
    }
  }
  FALSE
}

# TRUE iff the node's inferred type is a timestamp (Date/POSIXct).
# @keywords internal
ir_type_is_timestamp <- function(ir, schema) {
  type <- tryCatch(ir_infer_type(ir, schema), error = function(e) NA_character_)
  unname(type) %in% cuplyr_timestamp_types
}

# F37: the registry ops the GPU computes correctly when an argument is a
# timestamp (Date/POSIXct). ir_check_ts_operands() sends every other op with a
# timestamp argument to the CPU, where dplyr gives its result or its error.
# Before Date shifts were native, a shifted Date never reached these ops
# (`d + 1` itself went to the CPU); now `d + 1L` is a plain TIMESTAMP_DAYS
# value like a Date column, so the rule goes by type, never by node kind.
# Listed ops still check their own argument mixes: `+`/`-` (ir_parse_ts_arith()
# and infer_mutate_output_type()), the comparisons (same timestamp type only,
# below), if_else()/case_when()/coalesce() (ir_unify_types()), na_if()/
# replace_when() (ir_check_cast_to_x()), as.Date()/as.POSIXct() (their own
# type fns), and POSIXct-valued ops (ir_check_tz_rules()).
# `%in%` is listed but only native for a timestamp x against a set literal of
# the same timestamp type (ir_check_ts_operands()); `match()` compares the raw
# numbers, so any other mix stays on the CPU.
# Not listed, so CPU: `*`, `/`, `^`, `%%`, `%/%`, `&`, `|`, `!`, xor(), the math functions
# (round() keeps a Date in R, floor()/abs()/sqrt()/... error), cumulative
# functions (R errors except for cummean()), sum/sd/var (no cudf
# reduction over a timestamp), near(), the
# string functions (R formats the Date first), roll_mean/sum/sd/median.
ir_ts_native_ops <- c(
  "+", "-", "date_add_days", "date_sub_days", "ts_add_us", "ts_sub_us", "%in%",
  "==", "!=", "<", "<=", ">", ">=",
  "is.na", "is.nan", "if_else", "case_when", "coalesce", "na_if", "replace_when",
  "lag", "lead", "first", "last", "nth", "min", "max", "mean", "median", "n_distinct",
  "row_number", "min_rank", "dense_rank", "percent_rank", "cume_dist",
  "consecutive_id", "roll_min", "roll_max",
  "as.Date", "as.POSIXct", "as_date", "as_posixct",
  "year", "month", "day", "hour", "minute", "second", "quarter", "yday", "wday",
  ir_round_ops
)

# F37: the timestamp rule of ir_infer_type(), run on every call node `ir`
# with its argument types (bare NAs already resolved to a sibling's type).
# These signal `cuplyr_timestamp_needs_cpu`: a timestamp argument of an op
# outside ir_ts_native_ops; a unary `+`/`-` or a `+`/`-` with a bare NA; a
# comparison of a timestamp with anything but the same timestamp type (with a
# number or string the GPU truncates the literal or has no such comparison;
# with a Date vs a POSIXct R compares the raw numbers and warns), except a
# numeric NA/NaN literal (`d > NaN` is NA either way, F11); a timestamp in a
# condition position. Literal values are checked on their own
# (ir_check_ts_lit()). ir_parse_quo() runs the type inference over the whole
# tree, so this covers every way a node gets built: the parser, the desugars
# (between() expands to `>=` and `<=`), and nesting at any depth.
# @keywords internal
ir_check_ts_operands <- function(ir, arg_types) {
  # F41: a timestamp mean()/median() is only native as a whole value (the
  # aggregate is rounded/floored to the column type, which dplyr's double
  # isn't). Used inside any other expression (`t >= mean(t)`, `mean(d) + 1L`)
  # the result must stay unrounded: CPU.
  for (i in seq_along(ir$args)) {
    a <- ir$args[[i]]
    if (identical(a$kind, "call") && a$op %in% c("mean", "median") &&
        unname(arg_types[i]) %in% cuplyr_timestamp_types) {
      stop_timestamp_needs_cpu(a$op, "() of a timestamp inside `", ir$op, "()`: not on the GPU.")
    }
  }
  is_ts <- arg_types %in% cuplyr_timestamp_types
  if (!any(is_ts)) {
    return(invisible())
  }
  op <- ir$op
  if (op %in% c("mean", "median") && !unname(arg_types[1]) %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")) {
    stop_timestamp_needs_cpu(op, "() of a ", unname(arg_types[1]), ": not on the GPU.")
  }
  if (!op %in% ir_ts_native_ops) {
    stop_timestamp_needs_cpu(op, "() of a timestamp: not on the GPU.")
  }
  # ir_parse_ts_arith() sends these to the CPU too, but it skips a node whose
  # timestamp is nested (`-(as.Date(lit) + n)` without a Date column), and
  # the type fns would keep a unary sign's Date type or call `d + NA` a
  # "Date + Date". Base R errors on a unary minus, returns the timestamp for
  # a unary plus, and gives an NA timestamp for a bare-NA shift.
  if (op %in% c("+", "-")) {
    if (length(arg_types) == 1L) {
      stop_timestamp_needs_cpu("unary ", op, " on a timestamp: not on the GPU.")
    }
    if (any(vapply(ir$args, function(a) identical(a$kind, "lit") && isTRUE(a$na), logical(1)))) {
      stop_timestamp_needs_cpu("a timestamp shifted by NA: not on the GPU.")
    }
  }
  if (identical(op, "%in%")) {
    # F41: a timestamp x against a set literal of the same timestamp type (a
    # POSIXct set compares instants whatever its tzone). Anything else (mixed
    # Date/POSIXct, a numeric or string set, a timestamp set against another
    # x) is match() on raw numbers or strings: CPU. A fractional Date set is
    # caught by ir_check_ts_lit().
    if (length(arg_types) != 2L || length(unique(unname(arg_types))) != 1L ||
        !identical(ir$args[[2]]$kind, "lit")) {
      stop_timestamp_needs_cpu("%in% of a timestamp and a different type: not on the GPU.")
    }
  }
  if (op %in% c("==", "!=", "<", "<=", ">", ">=")) {
    missing_num <- vapply(ir$args, function(a) {
      identical(a$kind, "lit") && is.numeric(a$value) && length(a$value) == 1L && is.na(a$value)
    }, logical(1))
    if (length(unique(unname(arg_types[!missing_num]))) > 1L) {
      stop_timestamp_needs_cpu(
        "`", op, "` of ", paste(unname(arg_types), collapse = " and "), ": not on the GPU."
      )
    }
  }
  n <- length(arg_types)
  cond_idx <- switch(op,
    if_else = 1L,
    case_when = if (n >= 3L) seq(1L, n - 1L, by = 2L),
    replace_when = if (n >= 3L) seq(2L, n - 1L, by = 2L),
    NULL
  )
  if (any(is_ts[cond_idx])) {
    stop_timestamp_needs_cpu(op, "(): a timestamp condition.")
  }
  invisible()
}

# F37: the literal half of the timestamp rule, run by ir_infer_type() on every
# literal it reaches (and by ir_parse_shift_call() on a lag()/lead()
# `default`, which sits in `meta`). A Date literal becomes a TIMESTAMP_DAYS
# scalar, an int32 count of days (build_scalar() truncates), but an R Date
# can hold a fraction: `seq(a, b, length.out = k)` breakpoints, midpoints,
# mean() of Dates. `d >= 19724.5` would become `d >= 19724`, and a fractional
# value in if_else()/coalesce()/... would be truncated rather than floored
# like an upload (tbl_gpu()) is. So a Date literal with any non-whole (or
# out of int32 range) value sends its expression to the CPU, where dplyr
# compares the real value. NA values are fine (a null scalar). POSIXct
# literals are left alone: build_scalar() rounds them to microseconds exactly
# as posixct_to_gpu() rounds the columns they meet.
# @keywords internal
ir_check_ts_lit <- function(ir) {
  if (!identical(unname(ir$type), "TIMESTAMP_DAYS") || is.null(ir$value)) {
    return(invisible())
  }
  v <- unclass(ir$value)
  v <- v[!is.na(v)]
  if (!all(is.finite(v) & v == trunc(v) & abs(v) <= .Machine$integer.max)) {
    stop_timestamp_needs_cpu("a Date value that isn't a whole day: not on the GPU.")
  }
  invisible()
}

# F48: is `x` a lubridate Period/Duration/Interval or a base difftime? Their
# numeric payload is not what R computes with (`x + days(1)` errors, `x <
# days(1)` compares against the Period's lubridate value), so a literal of
# one of these classes goes to the CPU wherever it appears.
# @keywords internal
is_timespan_value <- function(v) {
  (isS4(v) || inherits(v, "difftime")) &&
    any(vapply(c("Period", "Duration", "Interval", "difftime"),
               function(cl) methods::is(v, cl), logical(1)))
}

# The literal half of F48's rule, run by ir_infer_type() on every literal.
# @keywords internal
ir_check_timespan_lit <- function(value) {
  if (is_timespan_value(value)) {
    stop_type_needs_cpu("a ", class(value)[1], " value outside Date/POSIXct arithmetic: not on the GPU.")
  }
  invisible()
}

# F48: the typed allowlist. For each reducer/window op, the GPU types of its
# first (value) argument that give dplyr's result natively; any other
# STRING/BOOL8/INT32/INT64/FLOAT64 argument goes to the CPU, where dplyr
# gives its result, warning or error (a raw cuDF error never reaches the
# user). A positive list: an op not named here is not judged. DICTIONARY32
# is ir_factor_rule()'s and the timestamps are ir_check_ts_operands()'s, so
# neither is judged here either. Comparisons and `%in%` are
# ir_check_type_operands()'s own rules, except that one with a DICTIONARY32
# operand is ir_factor_rule()'s too (F50).
# - the numeric reducers/windows take no STRING (dplyr coerces, warns or
#   errors);
# - median() of an INT32/BOOL8 is an integer or a double depending on each
#   group's count, which a lazy plan can't type: CPU;
# - cummax()/cummin() of a BOOL8 are integers in R (the GPU keeps BOOL8): CPU;
# - cumall()/cumany()/any()/all() natively take a logical only;
# - min()/max() of a STRING follow the session collation (ir_type_gate()).
# @keywords internal
ir_native_arg_types <- local({
  num <- c("BOOL8", "INT32", "INT64", "FLOAT64")
  c(
    stats::setNames(rep(list(num), 14L),
             c("mean", "sd", "var", "sum", "quantile", "cumsum", "cumprod",
               "cummean", "roll_mean", "roll_sum", "roll_sd", "roll_median",
               "roll_min", "roll_max")),
    list(
      median = c("INT64", "FLOAT64"),
      cummax = c("INT32", "INT64", "FLOAT64"),
      cummin = c("INT32", "INT64", "FLOAT64"),
      any = "BOOL8", all = "BOOL8", cumall = "BOOL8", cumany = "BOOL8",
      min = num, max = num,
      # a STRING gets dplyr's "non-numeric argument" error from the CPU
      round = num
    )
  )
})

# The ops ir_check_type_operands() judges: those of ir_native_arg_types, and the
# comparisons and `%in%` when an argument is a STRING.
ir_native_arg_op_names <- names(ir_native_arg_types)
ir_string_checked_ops <- c("==", "!=", "<", "<=", ">", ">=", "%in%")

# Ops whose STRING argument is native only under a byte-order collation: the
# one place the collation rule lives (ir_type_gate()). The comparisons are
# not in ir_native_arg_types (ir_check_type_operands() judges them), but
# share this rule.
ir_collated_ops <- c("min", "max", "<", "<=", ">", ">=")

ir_gated_types <- c("STRING", "BOOL8", "INT32", "INT64", "FLOAT64")

# TRUE iff `op` may run on the GPU with a value argument of GPU type `type`.
# `c_order` marks a comparison dplyr makes with vctrs (between()), which is
# byte order whatever the session collation.
# @keywords internal
ir_type_gate <- function(op, type, c_order = FALSE) {
  type <- unname(type)
  if (identical(type, "STRING") && op %in% ir_collated_ops) {
    return(c_order || session_collation_is_c())
  }
  allowed <- ir_native_arg_types[[op]]
  if (is.null(allowed) || !type %in% ir_gated_types) {
    return(TRUE)
  }
  type %in% allowed
}

# F48: the typed-allowlist rule of ir_infer_type(), run on every call node
# `ir` with its argument types (bare NAs already resolved to a sibling's
# type). Signals `cuplyr_type_needs_cpu` for: an op in ir_native_arg_types
# whose first argument type isn't listed; a comparison or `%in%` mixing a
# STRING with a number/logical (dplyr compares as character); an ordering
# comparison with a STRING operand outside a byte-order collation (except
# between()'s, flagged `meta$c_order`).
# @keywords internal
ir_check_type_operands <- function(ir, arg_types) {
  op <- ir$op
  types <- unname(arg_types)
  if (op %in% names(ir_native_arg_types)) {
    if (length(types) >= 1L && !ir_type_gate(op, types[1])) {
      stop_type_needs_cpu(op, "() of a ", types[1], ": not on the GPU.")
    }
    return(invisible())
  }
  is_cmp <- op %in% c("==", "!=", "<", "<=", ">", ">=")
  if (!is_cmp && !identical(op, "%in%")) {
    return(invisible())
  }
  # F50: a factor operand is ir_factor_rule()'s, not judged by the STRING
  # rules below. A label comparison was rewritten to INT32 codes by
  # ir_parse_factor_cmp() and never gets here; whatever is left is routed by
  # F18's rule whatever the session collation.
  if ("DICTIONARY32" %in% types) {
    return(invisible())
  }
  if (!"STRING" %in% types) {
    return(invisible())
  }
  if (any(types %in% c("BOOL8", "INT32", "INT64", "FLOAT64"))) {
    stop_type_needs_cpu("`", op, "` of a STRING and a number: not on the GPU.")
  }
  if (is_cmp && !ir_type_gate(op, "STRING", isTRUE(ir$meta$c_order))) {
    stop_type_needs_cpu("`", op, "` of strings under a non-C collation: not on the GPU.")
  }
  invisible()
}

# Small type-fn builders shared by several registry entries below.
type_arith_op <- function(op) {
  force(op)
  function(arg_types) {
    type <- ir_numeric_arith_type(arg_types)
    if (is.null(type)) infer_mutate_output_type(op, arg_types, NULL) else type
  }
}
# infer_mutate_output_type()'s rule 5 (`+ - * %% %/%`, no scalar) for operands
# that are all BOOL8/INT32/INT64/FLOAT64, the types no earlier rule rejects:
# FLOAT64 if any is, else INT64 if any is, else INT32. NULL for any other mix.
ir_plain_numeric_types <- c("BOOL8", "INT32", "INT64", "FLOAT64")
ir_numeric_arith_type <- function(arg_types) {
  if (anyNA(match(arg_types, ir_plain_numeric_types))) {
    return(NULL)
  }
  if (match("FLOAT64", arg_types, 0L) > 0L) "FLOAT64" else if (match("INT64", arg_types, 0L) > 0L) "INT64" else "INT32"
}
type_bool8 <- function(arg_types) "BOOL8"
type_float64 <- function(arg_types) "FLOAT64"
type_int32 <- function(arg_types) "INT32"
type_arg1 <- function(arg_types) arg_types[1]
# F41: mean()/median() of a Date/POSIXct is that timestamp type (computed on
# the integer rep, converted back); every other input gives a double.
type_mean_median <- function(arg_types) ts_mean_type(arg_types[1])
ts_mean_type <- function(input_type) {
  input_type <- unname(input_type)
  if (length(input_type) == 1L && input_type %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")) {
    return(input_type)
  }
  "FLOAT64"
}

# The 5 cuDF timestamp granularities (Phase 11 L3) -- shared by
# `ir_check_type_operands()`, `resolve_common_col_type()`
# (R/bind.R), and the `as.Date`/`as.POSIXct` registry entries below.
cuplyr_timestamp_types <- c("TIMESTAMP_DAYS", "TIMESTAMP_SECONDS",
                            "TIMESTAMP_MILLISECONDS", "TIMESTAMP_MICROSECONDS",
                            "TIMESTAMP_NANOSECONDS")

# Window-class `sum()`'s output type -- reuses make_aggregation()'s own
# switch verbatim (R/ast.R): INT32 promotes to INT64, everything else
# (including INT64 itself, matching that function's existing behavior)
# defaults to FLOAT64. Phase 5 (window functions) note: this intentionally
# mirrors that pre-existing table rather than "fixing" it, per
# scratchpad/phase5_window_design.md section 4's "output types reuse
# make_aggregation()'s table verbatim" instruction.
type_window_sum <- function(arg_types) {
  if (identical(arg_types[1], "INT32")) "INT64" else "FLOAT64"
}

# mutate()'s window sum(): like dplyr, the sum of an integer or logical is an
# integer (F42). The GPU total is INT64; an overflow is flagged by gpu_window()
# and handled by the caller (eager mutate() reruns the call on the CPU, a lazy
# plan raises `cuplyr_error_sum_overflow`). Everything else stays double.
type_window_int_sum <- function(arg_types) {
  if (arg_types[1] %in% c("INT32", "BOOL8")) "INT32" else "FLOAT64"
}

# abs() preserves its argument's type -- EXCEPT a logical (BOOL8) argument,
# which R promotes to integer (`typeof(abs(TRUE))` is "integer", not
# "logical"; base R's abs() coerces via as.integer()/as.double() before
# taking the absolute value, it never returns a logical). Verified against
# the dplyr oracle (T6 finding; phase1_expression_engine.md section 1.3's
# "abs, round | argument type" table entry was correct for abs() only on
# the non-logical case).
type_abs <- function(arg_types) {
  if (identical(arg_types[1], "BOOL8")) "INT32" else arg_types[1]
}

# Unary `-x` preserves the argument's type (section 1.3); binary `x - y`
# defers to infer_mutate_output_type() like the other D5-governed ops
# (+, *, %%, %/%). Both share the single registry key "-" (arity = NA)
# since R's parser also uses the same call head for both arities.
type_minus <- function(arg_types) {
  if (length(arg_types) == 1) {
    return(arg_types[1])
  }
  type <- ir_numeric_arith_type(arg_types)
  if (is.null(type)) infer_mutate_output_type("-", arg_types, NULL) else type
}

# -----------------------------------------------------------------------------
# Phase 4: value-position type unification (if_else/case_when/coalesce) and
# the shared bare-NA-adoption helper they (and na_if) all need.
#
# Verified empirically against dplyr 1.2.1 (see the Phase 4 task's trap
# list): `if_else()`/`case_when()`/`coalesce()` unify their "value position"
# arguments (if_else's yes/no/missing; case_when's val_i's + default;
# coalesce's whole arg list) via a vctrs-style common type, NOT the strict
# "must be identical type" rule -- e.g. `if_else(cond, 1L, 2.5)` succeeds as
# a double (dplyr's vctrs::vec_ptype2() freely promotes across
# integer/double/logical), it only errors on a genuinely incompatible mix
# like character vs double. This mirrors -- but is not literally the same
# function as -- infer_mutate_output_type()'s D5 arithmetic promotion
# (BOOL8 < INT32 < INT64 < FLOAT64); STRING here can only unify with STRING
# (no numeric<->STRING promotion path exists either way).
# -----------------------------------------------------------------------------

ir_value_type_rank <- c(BOOL8 = 1L, INT32 = 2L, INT64 = 3L, FLOAT64 = 4L)

#' Signal a type error caused by a factor operand (F18)
#'
#' The GPU types a factor as DICTIONARY32 and can't combine it with anything
#' else, but dplyr can (vctrs makes `factor` + `character` a character), so a
#' type check that fails because of a DICTIONARY32 operand is not a user
#' error. It signals this class instead of a plain error; [ir_parse_quo()]
#' turns it into `NULL` (CPU fallback). Outside a parse the message is the
#' same one the plain error had.
#'
#' @param ... Message pieces, pasted together
#' @keywords internal
stop_factor_needs_cpu <- function(...) {
  stop(structure(
    class = c("cuplyr_factor_needs_cpu", "error", "condition"),
    list(message = paste0(...), call = NULL)
  ))
}

#' Signal a type error dplyr resolves for a timestamp operand (F32)
#'
#' The timestamp sibling of [stop_factor_needs_cpu()]: `t + 3600` (a POSIXct in
#' `t`'s zone) and a Date mixed with a POSIXct in `if_else()`/`case_when()`/
#' `coalesce()` are valid R that the GPU engine can't compute. It is a
#' subclass, so [ir_parse_quo()]'s handler turns it into `NULL` as well, but
#' only this class is raised from the type check [ir_parse_quo()] runs: a
#' factor expression has to come back as IR for [ir_factor_rule()].
#'
#' @param ... Message pieces, pasted together
#' @keywords internal
stop_timestamp_needs_cpu <- function(...) {
  stop(structure(
    class = c("cuplyr_timestamp_needs_cpu", "cuplyr_factor_needs_cpu", "error", "condition"),
    list(message = paste0(...), call = NULL)
  ))
}

#' Signal a type combination the GPU does not compute like dplyr (F48)
#'
#' Same shape as [stop_timestamp_needs_cpu()] (a subclass of
#' `cuplyr_factor_needs_cpu`, so [ir_parse_quo()] turns it into `NULL`): a
#' STRING/BOOL8/INT32 argument of a reducer, window or comparison that
#' `ir_native_arg_types` does not list, or a Period/Duration value, goes to the
#' CPU, where dplyr gives its result, warning or error.
#'
#' @param ... Message pieces, pasted together
#' @keywords internal
stop_type_needs_cpu <- function(...) {
  stop(structure(
    class = c("cuplyr_type_needs_cpu", "cuplyr_factor_needs_cpu", "error", "condition"),
    list(message = paste0(...), call = NULL)
  ))
}

# @param types Character vector of GPU types for every "value position" arg
#   of an if_else()/case_when()/coalesce() call (already resolved -- no
#   remaining bare/isolated NA placeholders, see ir_resolve_value_nas())
# @param op_name Verb name, for the error message
# @return The unified GPU type
# @keywords internal
ir_unify_types <- function(types, op_name) {
  uniq <- unique(types)
  if (length(uniq) == 0) {
    return("BOOL8")  # typeof(NA) in R, when every value position was an isolated NA
  }
  if (length(uniq) == 1) {
    return(uniq)
  }

  # A factor among the values: dplyr may well combine it, the GPU can't. The
  # same for a Date next to a POSIXct (F32): vctrs unifies them to a POSIXct
  # by casting the Date to local midnight. Any other timestamp mix goes to the
  # CPU too (F37), so the user gets vctrs' own error ("Can't combine <date>
  # and <double>") rather than ours.
  fail <- if ("DICTIONARY32" %in% uniq) {
    stop_factor_needs_cpu
  } else if (any(uniq %in% cuplyr_timestamp_types)) {
    stop_timestamp_needs_cpu
  } else {
    function(...) stop(..., call. = FALSE)
  }

  if ("STRING" %in% uniq) {
    fail(op_name, "(): can't combine ",
         paste(paste0("<", uniq, ">"), collapse = " and "),
         " -- STRING values can't be combined with a non-STRING value.")
  }

  ranks <- ir_value_type_rank[uniq]
  if (anyNA(ranks)) {
    fail(op_name, "(): can't unify types ", paste(uniq, collapse = ", "))
  }

  names(ranks)[which.max(ranks)]
}

type_if_else <- function(arg_types) {
  ir_unify_types(arg_types[-1], "if_else")  # drop cond's own (BOOL8) type
}

type_case_when <- function(arg_types) {
  n <- length(arg_types)
  value_idx <- c(seq(2L, n - 1L, by = 2L), n)  # val_1, val_2, ..., default
  ir_unify_types(arg_types[value_idx], "case_when")
}

type_coalesce <- function(arg_types) {
  ir_unify_types(arg_types, "coalesce")
}

# na_if()'s output keeps `x`'s own type exactly (verified empirically: e.g.
# `na_if(1:3, 2)` -- an integer `x` against a double literal `y` -- stays
# integer; `y` is just compared against, never promoted into the result).
type_na_if <- function(arg_types) {
  ir_check_cast_to_x(arg_types, "na_if")
  arg_types[1]
}

#' Resolve a bare untyped `NA` literal's type from a sibling value-position arg
#'
#' `if_else()`/`case_when()`/`coalesce()`/`na_if()` each have several "value
#' position" argument slots that must share a common output type (if_else's
#' yes/no/missing; case_when's val_i's + default; coalesce's whole arg list;
#' na_if's x/y). A bare, untyped `NA` in one of those slots (e.g. `if_else(
#' cond, x, NA)`) must adopt a *sibling* value-position arg's type -- exactly
#' like arithmetic's `1L + NA` being integer, generalized from 2 args to N
#' (section 1.2/1.3).
#'
#' This is done here, at PARSE time (mutating each isolated node's own
#' `type` field directly, in place), rather than relying on
#' [ir_infer_type()]'s own generic isolated-NA substitution (which already
#' exists for arithmetic's 2-arg case): that generic mechanism picks the
#' FIRST non-isolated arg among a call node's ENTIRE `args` list, which for
#' `if_else` would incorrectly be `cond` (always the first arg, always
#' non-isolated, and NOT a value-position arg) rather than a true sibling
#' value. Calling this here on ONLY the value-position args (never `cond`)
#' avoids that pitfall entirely, and once it runs, the isolated node's
#' `type` field is no longer `NULL` -- so `ir_infer_type()`'s later generic
#' walk of the finished node is a no-op for these args, as intended.
#'
#' A typed `NA` (e.g. `NA_character_`, `NA_integer_`) is untouched: it
#' already carries its own `type` and is not "isolated" by this function's
#' definition (matching dplyr: `na_if(numeric_x, NA_character_)` still
#' errors, only a bare untyped `NA` adopts a sibling's type).
#'
#' @param nodes A list of IR nodes -- the "value position" args only (e.g.
#'   if_else()'s yes/no/missing, never its cond)
#' @param schema List with `names`/`types`
#' @return `nodes`, with every isolated bare-NA node's `type` field filled
#'   in from the first non-isolated sibling's inferred type (returned
#'   unchanged if none, or all, of `nodes` are isolated)
#' @keywords internal
ir_resolve_value_nas <- function(nodes, schema) {
  is_isolated_na <- vapply(nodes, function(n) {
    identical(n$kind, "lit") && isTRUE(n$na) && is.null(n$type)
  }, logical(1))

  if (!any(is_isolated_na) || all(is_isolated_na)) {
    return(nodes)
  }

  fallback_type <- ir_infer_type(nodes[[which(!is_isolated_na)[1]]], schema)

  for (i in which(is_isolated_na)) {
    nodes[[i]]$type <- fallback_type
  }
  nodes
}

# -----------------------------------------------------------------------------
# Phase 5 (window functions): cummean()/cumall()/cumany()/consecutive_id()
# registry `parse` hooks -- these are ordinary registry entries (single
# positional `x`, no ambiguous named controls), so unlike lag()/lead()/
# first()/last()/nth()/ntile() above they need no dedicated top-level
# dispatch: the registry's own generic per-arg loop in ir_parse_expr()
# already parses their one argument correctly before calling `entry$parse`.
#
# Each fully desugars into plain IR containing nested window-class calls
# (cumsum/cummin/cummax/row_number/sum/n) -- see R/window.R's
# ir_extract_windows()/plan_window_stages() for how those get decomposed
# into actual `..win*..` temp columns later; nothing here needs to know
# about that. Every formula below was verified empirically against dplyr
# 1.2.1 (not re-derived from the design doc's table alone) via a battery of
# hand-checked cases including NA-adjacency runs and ties; see the W1 task
# notes for the exact cases run.
# -----------------------------------------------------------------------------

#' Desugar `cummean(x)` into `cumsum(x) / row_number()`
#'
#' Verified against dplyr 1.2.1: `cummean(c(1,2,NA,4))` is `1, 1.5, NA, NA`
#' -- i.e. cummean inherits cumsum()'s own (sticky, once cudf's scan gets
#' its Phase-5 sticky-NA fix in W2 -- see phase5_window_design.md section
#' 4) NA propagation for free, simply by dividing by row_number() (which is
#' never NA), with no special-casing needed here.
#'
#' @keywords internal
ir_parse_cummean <- function(args, schema) {
  x <- args[[1]]
  # F48: the `x + 0` below would turn a STRING into an arithmetic error; the
  # allowlist sends it to the CPU (dplyr coerces to numeric) while the op
  # still has its own name.
  x_type <- unname(ir_infer_type(x, schema))
  if (!ir_type_gate("cummean", x_type)) {
    stop_type_needs_cpu("cummean() of a ", x_type, ": not on the GPU.")
  }
  # dplyr accumulates in double: an integer/logical argument must not go
  # through the (overflow-checked) integer cumsum (F42).
  if (!identical(x_type, "FLOAT64")) {
    x <- ir_call("+", list(x, ir_lit_from_r(0)))
  }
  ir_call("/", list(ir_call("cumsum", list(x)), ir_call("row_number", list())))
}

#' Desugar `cumall(x)`/`cumany(x)` into a tri-state `cummin`/`cummax` encoding
#'
#' Verified against dplyr 1.2.1: `cumall(c(T,T,NA,T,F,T))` is `T,T,NA,NA,
#' F,F`; `cumany(c(F,NA,T,F))` is `F,NA,T,T` -- both are exactly `cummin`/
#' `cummax` (respectively) over the tri-state encoding FALSE=0, NA=1,
#' TRUE=2, then decoded back to BOOL8 (`scratchpad/phase5_window_design.md`
#' section 0's ground truth table, cross-checked directly here rather than
#' taken on faith).
#'
#' `if_else()`'s bare-NA value-position adoption (`ir_resolve_value_nas()`)
#' is applied explicitly to the decode step's `yes`/`no` pair here, exactly
#' as if_else()'s own registry `parse` hook would -- this desugar builds
#' `ir_call("if_else", ...)` nodes directly rather than routing back through
#' `ir_parse_expr()`, so that adoption has to be done by hand.
#'
#' @param scan_op `"cummin"` (cumall) or `"cummax"` (cumany)
#' @keywords internal
ir_parse_cumall_cumany <- function(args, schema, scan_op) {
  x <- args[[1]]
  # The tri-state encoding below only takes a logical x; dplyr accepts any
  # type (F48), so the typed allowlist is checked here, where the op still
  # has its own name.
  x_type <- tryCatch(unname(ir_infer_type(x, schema)), error = function(e) NA_character_)
  if (!is.na(x_type) && !ir_type_gate(if (scan_op == "cummin") "cumall" else "cumany", x_type)) {
    stop_type_needs_cpu(if (scan_op == "cummin") "cumall" else "cumany",
                        "() of a ", x_type, ": not on the GPU.")
  }
  encoded <- ir_call("if_else", list(
    ir_call("is.na", list(x)),
    ir_lit_from_r(1L),
    ir_call("if_else", list(x, ir_lit_from_r(2L), ir_lit_from_r(0L)))
  ))
  s <- ir_call(scan_op, list(encoded))
  decode_values <- ir_resolve_value_nas(
    list(ir_lit(value = NULL, type = NULL, na = TRUE), ir_call("==", list(s, ir_lit_from_r(2L)))),
    schema
  )
  ir_call("if_else", c(list(ir_call("==", list(s, ir_lit_from_r(1L)))), decode_values))
}

#' Desugar single-column `consecutive_id(x)` into a two-stage `lag`/
#' `row_number`/`cumsum` formula
#'
#' Multi-column `consecutive_id(a, b, ...)` is not implemented this wave
#' (returns `NULL`, i.e. falls back) -- this is enforced for free by the
#' registry entry's `arity = 1L` (a 2+-arg call fails the generic arity
#' check in `ir_parse_expr()` before `parse` is ever invoked), so this
#' function itself only ever sees exactly one argument.
#'
#' Verified against dplyr 1.2.1, INCLUDING NA-adjacency (the tricky part:
#' two consecutive `NA`s are treated as "the same value", i.e. NOT a new
#' group, which the `is.na(x) & is.na(lx)` disjunct captures):
#' `consecutive_id(c(1,NA,NA,2))` is `1,2,2,3`; `consecutive_id(c(NA,NA,1,
#' 1,NA))` is `1,1,2,2,3`; `consecutive_id(c(1,2,2,NA,NA,2))` is
#' `1,2,2,3,3,4`.
#'
#' Builds `cumsum(if_else(rn==1L, 1L, if_else(same_as_prev, 0L, 1L)))`
#' where `rn = row_number()` and `same_as_prev = coalesce(x == lag(x),
#' FALSE) | (is.na(x) & is.na(lag(x)))`; for a double `x` the "both missing"
#' clause is split into "both NaN" and "both NA but not NaN" (vctrs keeps NaN
#' and NA apart). `lag(x)`/`row_number()` are
#' themselves window calls nested inside `cumsum()`'s argument -- exactly
#' the "nested window calls needing 2 decomposition passes" shape `R/
#' window.R`'s `ir_extract_windows()`/`plan_window_stages()` handles
#' generically (see test-window-plan.R), so this desugar doesn't need to
#' know anything about staging itself.
#'
#' @keywords internal
ir_parse_consecutive_id <- function(args, schema) {
  x <- args[[1]]
  lx <- ir_call("lag", list(x), meta = list(n = 1L, default = NULL))
  rn <- ir_call("row_number", list())

  equal_prev <- ir_call("coalesce", list(ir_call("==", list(x, lx)), ir_lit_from_r(FALSE)))
  x_type <- tryCatch(unname(ir_infer_type(x, schema)), error = function(e) NA_character_)
  same_as_prev <- if (identical(x_type, "FLOAT64")) {
    # F55: is.na() is TRUE for NaN, but vctrs' run identification keeps NaN and
    # NA apart: two NaN are one run, two NA are one run, a NaN next to an NA
    # starts a new one. So "both missing" is split into "both NaN" and "both
    # NA (not NaN)".
    is_nan_x <- ir_call("is.nan", list(x))
    is_nan_lx <- ir_call("is.nan", list(lx))
    both_na <- ir_call("&", list(
      ir_call("&", list(ir_call("is.na", list(x)), ir_call("!", list(is_nan_x)))),
      ir_call("&", list(ir_call("is.na", list(lx)), ir_call("!", list(is_nan_lx))))
    ))
    ir_call("|", list(
      ir_call("|", list(equal_prev, ir_call("&", list(is_nan_x, is_nan_lx)))),
      both_na
    ))
  } else {
    ir_call("|", list(
      equal_prev,
      ir_call("&", list(ir_call("is.na", list(x)), ir_call("is.na", list(lx))))
    ))
  }

  increment <- ir_call("if_else", list(
    ir_call("==", list(rn, ir_lit_from_r(1L))),
    ir_lit_from_r(1L),
    ir_call("if_else", list(same_as_prev, ir_lit_from_r(0L), ir_lit_from_r(1L)))
  ))

  ir_call("cumsum", list(increment))
}

# -----------------------------------------------------------------------------
# Phase 11 L5 (scratchpad/phase11_design.md): base-R string functions.
# STRING is never `ast_expressible()` (src/expr_eval.hpp), so every one of
# these always materializes as its own `apply_handler()` column -- there is
# no AST-native string op, unlike arithmetic/comparison/logic.
#
# Every dedicated dispatch function below follows the lag()/lead()/ntile()
# precedent (module comment above ir_parse_shift_call()): `rlang::call_match()`
# against a prototype mirroring the REAL base-R signature, so named/
# positional/reordered calls are matched exactly like real R would, then any
# OPTIONAL argument this wave doesn't implement being present AT ALL (even at
# its own default value) makes the whole dot return `NULL` (fall back to L1's
# CPU path) -- the same "narrow scope, cut liberally" convention every prior
# phase in this file established (see ir_parse_first_last_nth()'s docs for
# the canonical statement of this rule).
#
# `pattern`/`start`/`stop`/`which`/`sep`/etc. arguments are all evaluated as
# R-side CONSTANTS via [ir_eval_constant()] (no data mask) -- exactly the
# `lag()`'s `n=`/`default=` precedent -- never as column expressions; a
# column-valued position/pattern falls back.
# -----------------------------------------------------------------------------

# TRUE iff `ir_node`'s inferred GPU type is STRING. Shared guard for every
# L5 entry/dispatch function below (toupper()/tolower()'s own `parse` hooks,
# and every dedicated dispatch function's own `x` check) -- mirrors
# na_if()'s/case_match()'s own `unname()` landmine (a schema column's
# inferred type comes back NAMED; identical() treats a `names` attribute as
# significant).
# @keywords internal
ir_is_string_type <- function(ir_node, schema) {
  identical(unname(ir_infer_type(ir_node, schema)), "STRING")
}

#' Parse `nchar(x, type = "chars", allowNA = FALSE, keepNA = NA)` (Phase 11 L5)
#'
#' Only the default `type = "chars"` shape is implemented -- `allowNA=`/
#' `keepNA=` present at all, or an explicit `type=` other than `"chars"`,
#' falls back: cuDF's `count_characters()` (`strings/attributes.hpp`) counts
#' Unicode code points, matching R's own `nchar(type = "chars")` (the
#' default) exactly, but has no byte-count (`type = "bytes"`) or
#' display-width (`type = "width"`) equivalent. `x` must be a genuine
#' STRING-typed expression.
#' @keywords internal
ir_parse_nchar <- function(expr, env, schema) {
  proto <- function(x, type = "chars", allowNA = FALSE, keepNA = NA) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args$allowNA) || !is.null(present_args$keepNA)) {
    return(NULL)
  }
  if (!is.null(present_args$type)) {
    type_eval <- ir_eval_constant(present_args$type, env)
    if (!type_eval$ok || !identical(type_eval$value, "chars")) {
      return(NULL)  # type = "bytes"/"width": not implemented this wave
    }
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir) || !ir_is_string_type(x_ir, schema)) {
    return(NULL)
  }

  ir_call("nchar", list(x_ir))
}

#' Parse `substr(x, start, stop)` (Phase 11 L5)
#'
#' `start`/`stop` must be constant (non-column) whole numbers -- resolved
#' via [ir_eval_constant()] with no data mask, exactly like `lag()`'s `n=`.
#' Converts R's inclusive 1-based `[start, stop]` into cuDF
#' `slice_strings()`'s (`strings/slice.hpp`) exclusive 0-based `[start,
#' stop)`, replicating R's own clamping (verified empirically against real
#' `substr()`): a `start` below 1 clamps UP to 1 *before* the 0-based
#' conversion (`substr("hello", -2, 3)` is `"hel"`, identical to
#' `substr("hello", 1, 3)`), and a `stop` below the already-clamped 0-based
#' start clamps up to it, producing an empty string (`substr("hello", 5,
#' 2)` is `""`) rather than ever handing `slice_strings()` a negative
#' scalar. A non-numeric/`NA` `start`/`stop`, a non-constant `start`/`stop`,
#' or a non-STRING `x`, falls back.
#' @keywords internal
ir_parse_substr <- function(expr, env, schema) {
  proto <- function(x, start, stop) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (is.null(present_args$x) || is.null(present_args$start) || is.null(present_args$stop)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir) || !ir_is_string_type(x_ir, schema)) {
    return(NULL)
  }

  start_eval <- ir_eval_constant(present_args$start, env)
  stop_eval <- ir_eval_constant(present_args$stop, env)
  if (!start_eval$ok || !stop_eval$ok) {
    return(NULL)
  }
  start_val <- start_eval$value
  stop_val <- stop_eval$value
  if (!is.numeric(start_val) || length(start_val) != 1 || is.na(start_val) ||
      !is.numeric(stop_val) || length(stop_val) != 1 || is.na(stop_val)) {
    return(NULL)
  }

  start_r <- max(as.integer(start_val), 1L)
  start0 <- start_r - 1L
  stop0 <- max(as.integer(stop_val), start0)

  ir_call("substr", list(x_ir, ir_lit_from_r(start0), ir_lit_from_r(stop0)))
}

#' Parse `trimws(x, which = "both", whitespace = "[ \\t\\r\\n]")` (Phase 11 L5)
#'
#' Only the default `whitespace=` (space/tab/CR/LF -- cuDF `strip()`'s
#' (`strings/strip.hpp`) own "empty `to_strip` means strip whitespace"
#' behavior) is implemented -- an explicit `whitespace=` argument at all
#' falls back. `which=` maps to `cudf::strings::side_type`
#' (`"both"`/`"left"`/`"right"`), matched via `pmatch()` against the 3 full
#' names (real `trimws()` uses `match.arg()` internally, which also accepts
#' an unambiguous partial match like `"l"`).
#' @keywords internal
ir_parse_trimws <- function(expr, env, schema) {
  proto <- function(x, which = "both", whitespace = "[ \t\r\n]") NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args$whitespace)) {
    return(NULL)
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir) || !ir_is_string_type(x_ir, schema)) {
    return(NULL)
  }

  which_val <- "both"
  if (!is.null(present_args$which)) {
    which_eval <- ir_eval_constant(present_args$which, env)
    if (!which_eval$ok || !is.character(which_eval$value) || length(which_eval$value) < 1) {
      return(NULL)
    }
    matched <- pmatch(which_eval$value[1], c("both", "left", "right"))
    if (is.na(matched)) {
      return(NULL)
    }
    which_val <- c("both", "left", "right")[matched]
  }

  ir_call("trimws", list(x_ir, ir_lit_from_r(which_val)))
}

#' Parse `paste0(..., collapse = NULL)` / `paste(..., sep = " ", collapse =
#' NULL)` (Phase 11 L5)
#'
#' `collapse=` (reduce-to-one-string) is not implemented this wave -- its
#' presence at all falls back. `sep=` (`paste()` only; `paste0()` is always
#' `sep = ""`) must be a constant string. Every dot must resolve to a
#' STRING-typed IR node -- ANY non-STRING dot (numeric, logical, ...) falls
#' back rather than replicating R's `as.character()` formatting rules
#' (verified empirically: `paste0("x", 4)` is `"x4"`, NOT `"x4.0"` --
#' faithfully reproducing `as.character()`'s numeric-formatting edge cases
#' for every type is explicitly out of scope, per the design doc's own
#' "x4 vs 4.0 trap" note). The lowered `ir_call("paste_cols", ...)` node's
#' LAST argument is always a STRING literal carrying the separator (never a
#' per-row value) -- `apply_handler()`'s `narep = "NA"` reproduces R's own
#' "a genuine NA argument becomes the literal two-character string `\"NA\"`"
#' behavior (verified: `paste0("a", NA, "b")` is `"aNAb"`, not `NA`).
#' @keywords internal
ir_parse_paste_call <- function(expr, env, schema, fn_name) {
  proto <- if (identical(fn_name, "paste0")) {
    function(..., collapse = NULL) NULL
  } else {
    function(..., sep = " ", collapse = NULL) NULL
  }
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args$collapse)) {
    return(NULL)
  }

  sep_val <- if (identical(fn_name, "paste0")) "" else " "
  if (!is.null(present_args$sep)) {
    sep_eval <- ir_eval_constant(present_args$sep, env)
    if (!sep_eval$ok || !is.character(sep_eval$value) || length(sep_eval$value) != 1 ||
        is.na(sep_eval$value)) {
      return(NULL)
    }
    sep_val <- sep_eval$value
  }

  arg_names <- names(present_args)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(present_args))
  }
  dots_raw <- present_args[!(arg_names %in% c("sep", "collapse"))]
  if (length(dots_raw) == 0) {
    return(NULL)  # zero-dot edge case: not implemented this wave
  }

  dot_irs <- vector("list", length(dots_raw))
  for (i in seq_along(dots_raw)) {
    node <- ir_parse_expr(dots_raw[[i]], env, schema)
    if (is.null(node) || !ir_is_string_type(node, schema)) {
      return(NULL)  # non-STRING/unparseable arg: fall back (no as.character())
    }
    dot_irs[[i]] <- node
  }

  ir_call("paste_cols", c(dot_irs, list(ir_lit_from_r(sep_val))))
}

#' Parse `grepl(pattern, x, ignore.case = FALSE, perl = FALSE, fixed =
#' FALSE, useBytes = FALSE)` (Phase 11 L5)
#'
#' `pattern` must be a constant (non-column) string -- resolved via
#' [ir_eval_constant()]. `ignore.case=`/`perl=`/`useBytes=` present at all
#' falls back (cuDF's `regex_flags` -- `strings/regex/flags.hpp` -- has no
#' case-insensitive flag to map `ignore.case=TRUE` onto, and PCRE
#' (`perl=TRUE`) is a materially different regex dialect from cuDF's own --
#' cuDF-regex-vs-R divergence is scoped to simple ASCII patterns only, per
#' the design doc). `fixed=TRUE` routes to a literal substring search
#' (`contains()`, `strings/find.hpp`) instead of regex (`contains_re()`,
#' `strings/contains.hpp`). The lowered node's 3rd argument is always a
#' BOOL8 literal carrying `fixed`. Result is always non-NA: R's `grepl()`
#' treats a `NA` element of `x` as `FALSE` (verified empirically), enforced
#' by the C++ handler via `replace_nulls()` after the (null-propagating)
#' cuDF call.
#' @keywords internal
ir_parse_grepl <- function(expr, env, schema) {
  proto <- function(pattern, x, ignore.case = FALSE, perl = FALSE, fixed = FALSE, useBytes = FALSE) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args[["ignore.case"]]) || !is.null(present_args$perl) ||
      !is.null(present_args$useBytes)) {
    return(NULL)
  }
  if (is.null(present_args$pattern) || is.null(present_args$x)) {
    return(NULL)
  }

  pattern_eval <- ir_eval_constant(present_args$pattern, env)
  if (!pattern_eval$ok || !is.character(pattern_eval$value) ||
      length(pattern_eval$value) != 1 || is.na(pattern_eval$value)) {
    return(NULL)
  }

  fixed_val <- FALSE
  if (!is.null(present_args$fixed)) {
    fixed_eval <- ir_eval_constant(present_args$fixed, env)
    if (!fixed_eval$ok || !is.logical(fixed_eval$value) || length(fixed_eval$value) != 1 ||
        is.na(fixed_eval$value)) {
      return(NULL)
    }
    fixed_val <- fixed_eval$value
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir) || !ir_is_string_type(x_ir, schema)) {
    return(NULL)
  }

  ir_call("grepl", list(x_ir, ir_lit_from_r(pattern_eval$value), ir_lit_from_r(fixed_val)))
}

#' Parse `sub(pattern, replacement, x, ignore.case=, perl=, fixed=,
#' useBytes=)` / `gsub(...)` (Phase 11 L5)
#'
#' Same restrictions as [ir_parse_grepl()]: `pattern`/`replacement` must be
#' constant strings, `ignore.case=`/`perl=`/`useBytes=` present at all
#' falls back, `fixed=TRUE` routes to literal `replace()`
#' (`strings/replace.hpp`) instead of `replace_re()`
#' (`strings/replace_re.hpp`). `sub()` replaces only the first match per
#' string (`max_replace_count = 1`); `gsub()` replaces every match
#' (`std::nullopt`) -- both share one C++ handler keyed off the op name
#' (`"sub"`/`"gsub"`, both registered separately below purely so
#' `ir_infer_type()` can look them up by their own name).
#' @keywords internal
ir_parse_sub_gsub <- function(expr, env, schema, fn_name) {
  proto <- function(pattern, replacement, x, ignore.case = FALSE, perl = FALSE,
                     fixed = FALSE, useBytes = FALSE) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args[["ignore.case"]]) || !is.null(present_args$perl) ||
      !is.null(present_args$useBytes)) {
    return(NULL)
  }
  if (is.null(present_args$pattern) || is.null(present_args$replacement) ||
      is.null(present_args$x)) {
    return(NULL)
  }

  pattern_eval <- ir_eval_constant(present_args$pattern, env)
  if (!pattern_eval$ok || !is.character(pattern_eval$value) ||
      length(pattern_eval$value) != 1 || is.na(pattern_eval$value)) {
    return(NULL)
  }
  repl_eval <- ir_eval_constant(present_args$replacement, env)
  if (!repl_eval$ok || !is.character(repl_eval$value) ||
      length(repl_eval$value) != 1 || is.na(repl_eval$value)) {
    return(NULL)
  }

  fixed_val <- FALSE
  if (!is.null(present_args$fixed)) {
    fixed_eval <- ir_eval_constant(present_args$fixed, env)
    if (!fixed_eval$ok || !is.logical(fixed_eval$value) || length(fixed_eval$value) != 1 ||
        is.na(fixed_eval$value)) {
      return(NULL)
    }
    fixed_val <- fixed_eval$value
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir) || !ir_is_string_type(x_ir, schema)) {
    return(NULL)
  }

  ir_call(fn_name, list(x_ir, ir_lit_from_r(pattern_eval$value),
                        ir_lit_from_r(repl_eval$value), ir_lit_from_r(fixed_val)))
}

#' Parse `startsWith(x, prefix)` / `endsWith(x, suffix)` (Phase 11 L5)
#'
#' `prefix`/`suffix` must be a constant (non-column, non-`NA`) string --
#' matching a per-row column-valued target (`starts_with()`/`ends_with()`'s
#' `targets` column overload, `strings/find.hpp`) is not implemented this
#' wave. NA propagation needs no special handling in the C++ handler: cuDF's
#' `starts_with()`/`ends_with()` already null-propagate exactly like R's own
#' `startsWith()`/`endsWith()` (verified empirically: `startsWith(c("abc",
#' NA), "a")` is `TRUE, NA`, not `TRUE, FALSE`) -- unlike `grepl()` above,
#' which needs an explicit NA-to-FALSE fixup.
#' @keywords internal
ir_parse_starts_ends_with <- function(expr, env, schema, fn_name) {
  proto <- if (identical(fn_name, "startsWith")) {
    function(x, prefix) NULL
  } else {
    function(x, suffix) NULL
  }
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  target_name <- if (identical(fn_name, "startsWith")) "prefix" else "suffix"

  if (is.null(present_args$x) || is.null(present_args[[target_name]])) {
    return(NULL)
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir) || !ir_is_string_type(x_ir, schema)) {
    return(NULL)
  }

  target_eval <- ir_eval_constant(present_args[[target_name]], env)
  if (!target_eval$ok || !is.character(target_eval$value) ||
      length(target_eval$value) != 1 || is.na(target_eval$value)) {
    return(NULL)
  }

  op <- if (identical(fn_name, "startsWith")) "starts_with" else "ends_with"
  ir_call(op, list(x_ir, ir_lit_from_r(target_eval$value)))
}

# -----------------------------------------------------------------------------
# Phase 11 L6 (scratchpad/phase11_design.md): lubridate datetime accessors.
#
# MANDATORY NAMESPACE GUARD: year()/month()/day()/hour()/minute()/second()/
# quarter()/yday()/wday() are lubridate's own names -- base R has no year()
# (unlike, say, `nchar()`/`substr()` above, which really are base R). Going
# native for a call that merely happens to be spelled `year(x)` would be
# wrong if the user has their own unrelated `year <- function(x) ...` in
# scope, or if lubridate isn't installed/attached at all (in which case real
# dplyr's own CPU evaluation would raise "could not find function \"year\""
# -- cuplyr must reproduce that, not silently misinterpret the call).
# `ir_require_lubridate_fn()` resolves the call's OWN function symbol --
# `ir_parse_expr()`'s call-handling branch already derived it from `head`,
# either a bare symbol (looked up in the quosure's environment, exactly like
# any other symbol resolution in this file) or an explicit `pkg::fn`
# qualification (resolved directly via that package's namespace, bypassing
# environment lookup entirely, so `lubridate::year(x)` is honored even when
# lubridate isn't attached) -- and accepts the call ONLY if that resolves to
# the REAL, identical `lubridate::<name>` function object.
# -----------------------------------------------------------------------------

#' Resolve a call's function symbol to the actual function object it invokes
#'
#' @param head The call's own head (`expr[[1]]`, as derived by
#'   `ir_parse_expr()`'s call-handling branch) -- a bare symbol, or a
#'   `pkg::fn`/`pkg:::fn` call
#' @param env The quosure's environment (used only for a bare symbol)
#' @return The resolved function, or `NULL` if it can't be resolved (an
#'   undefined symbol, or a non-function binding)
#' @keywords internal
resolve_call_fn <- function(head, env) {
  if (is.call(head) && length(head) == 3 &&
      (identical(head[[1]], as.name("::")) || identical(head[[1]], as.name(":::")))) {
    pkg <- as.character(head[[2]])
    fn <- as.character(head[[3]])
    return(tryCatch(getExportedValue(pkg, fn), error = function(e) NULL))
  }
  if (is.symbol(head)) {
    return(tryCatch(get(as.character(head), envir = env, mode = "function", inherits = TRUE),
                     error = function(e) NULL))
  }
  NULL
}

#' The Phase 11 L6 namespace guard: TRUE iff a call's function symbol
#' resolves to exactly `lubridate::<name>`
#'
#' `FALSE` (never an error) whenever lubridate isn't installed, `name` isn't
#' one of its exports, or the call's own function symbol resolves to
#' anything else (including simply not resolving at all) -- every one of
#' those cases must fall back to L1's CPU path, never hard-error here.
#' @keywords internal
ir_require_lubridate_fn <- function(head, env, name) {
  if (!requireNamespace("lubridate", quietly = TRUE)) {
    return(FALSE)
  }
  target <- tryCatch(getExportedValue("lubridate", name), error = function(e) NULL)
  if (is.null(target)) {
    return(FALSE)
  }
  resolved <- resolve_call_fn(head, env)
  if (is.null(resolved)) {
    return(FALSE)
  }
  identical(resolved, target)
}

# TRUE iff `ir` is POSIXct-valued in a zone that is not UTC-equivalent, i.e.
# a calendar component of it depends on the zone (DST included) and must be
# computed on the CPU (F32).
# @keywords internal
ir_needs_cpu_tz <- function(ir, schema) {
  zone <- ir_tz_of(ir, schema)
  !is.null(zone) && !tz_is_utc(zone)
}

#' Parse a lubridate accessor call (Phase 11 L6), once
#' [ir_require_lubridate_fn()] has already confirmed the call really
#' invokes lubridate's own function
#'
#' Every one of these 9 functions takes `x` as its sole required argument;
#' each also has 1-4 OPTIONAL controls (`label=`/`abbr=`/`locale=`
#' (month()/wday()), `week_start=` (wday() only), `with_year=`/
#' `fiscal_start=` (quarter() only)) that are NOT implemented this wave --
#' any of them present at all (even at its own default value) falls back,
#' matched via `rlang::call_match()` against a prototype mirroring each
#' function's REAL lubridate signature (so `year(x)`/`year(x = d)` both
#' match, but e.g. `month(x, label = TRUE)` does not).
#'
#' `x` must already be one of the 5 TIMESTAMP_* granularities (a STRING or
#' numeric `x` is not attempted natively, mirroring `as.Date()`/
#' `as.POSIXct()`'s own scope cut in L3).
#'
#' `wday()`'s own default depends on the GLOBAL `lubridate.week.start`
#' option (`getOption("lubridate.week.start", 7)`) -- verified empirically
#' (plain Rscript, no GPU needed): with the option at its own default (7,
#' i.e. Sunday), `lubridate::wday(d)` returns `Sun=1, Mon=2, ..., Sat=7`.
#' cuDF's `WEEKDAY` component (`datetime.hpp`) is ISO `Mon=1..Sun=7`;
#' `(w %% 7) + 1` maps cuDF's convention onto lubridate's exactly (verified
#' algebraically against the empirical ground truth: Mon `(1 %% 7) + 1 =
#' 2`; Sun `(7 %% 7) + 1 = 1`) -- implemented in the C++ handler
#' (`src/expr_eval.hpp`). If the user has changed the global option away
#' from its own default, this remap would be wrong, so `wday()` additionally
#' checks the CURRENT option value at parse time and falls back if it isn't
#' the default 7.
#'
#' KNOWN, DOCUMENTED DIVERGENCE (not fixed this wave, see
#' `test-dplyr-datetime.R`): `lubridate::second()` returns a FRACTIONAL
#' value for a POSIXct with sub-second precision (e.g. `second(x)` where
#' `x` is `13:45:30.75` is `30.75`, not `30`) -- cuDF's `SECOND` component
#' is always a whole integer, so `second()` on a genuinely sub-second-
#' precision column silently truncates natively instead of falling back.
#' Whole-second POSIXct values (the overwhelming majority of real usage,
#' and every `Date`/whole-second `POSIXct`) are unaffected.
#'
#' @keywords internal
ir_parse_lubridate_accessor <- function(expr, env, schema, fn_name) {
  proto <- switch(fn_name,
    year = ,
    day = ,
    hour = ,
    minute = ,
    second = ,
    yday = function(x) NULL,
    month = function(x, label = FALSE, abbr = TRUE, locale = Sys.getlocale("LC_TIME")) NULL,
    quarter = function(x, with_year = FALSE, fiscal_start = 1) NULL,
    wday = function(x, label = FALSE, abbr = TRUE,
                     week_start = getOption("lubridate.week.start", 7),
                     locale = Sys.getlocale("LC_TIME")) NULL
  )
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]

  if (length(set_diff(names(present_args), "x")) > 0) {
    return(NULL)  # any optional control present at all: not implemented this wave
  }
  if (is.null(present_args$x)) {
    return(NULL)
  }

  if (identical(fn_name, "wday") && !identical(getOption("lubridate.week.start", 7), 7)) {
    return(NULL)  # non-default global week_start: the (w %% 7) + 1 remap wouldn't hold
  }

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_type <- unname(ir_infer_type(x_ir, schema))
  if (!x_type %in% cuplyr_timestamp_types) {
    return(NULL)  # STRING/numeric source: not attempted natively (mirrors L3)
  }
  if (ir_needs_cpu_tz(x_ir, schema)) {
    return(NULL)  # F32: wall clock in a non-UTC zone is computed by R itself
  }

  ir_call(fn_name, list(x_ir))
}

#' Parse a lubridate floor_date()/ceiling_date()/round_date() call (F46)
#'
#' Native only for a UTC-equivalent POSIXct with unit second/minute/hour/day
#' and a Date with day/week (week starting Sunday, the default
#' `lubridate.week.start`), `unit` a single character literal in one of
#' timechange's spellings with no multiplier. Anything else (other zones,
#' `week_start`/`change_on_boundary` arguments, calendar units, a Date with a
#' sub-day unit, which lubridate turns into a POSIXct, unknown spellings)
#' returns `NULL` and runs on the CPU.
#'
#' @keywords internal
ir_parse_lubridate_rounding <- function(expr, env, schema, fn_name) {
  proto <- switch(fn_name,
    floor_date = function(x, unit = "seconds", week_start = getOption("lubridate.week.start", 7)) NULL,
    ceiling_date = function(x, unit = "seconds", change_on_boundary = NULL,
                            week_start = getOption("lubridate.week.start", 7)) NULL,
    round_date = function(x, unit = "second", week_start = getOption("lubridate.week.start", 7)) NULL
  )
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    return(NULL)
  }
  present_args <- as.list(present)[-1]
  if (length(set_diff(names(present_args), c("x", "unit"))) > 0 || is.null(present_args$x)) {
    return(NULL)
  }

  unit <- if ("unit" %in% names(present_args)) present_args$unit else
    if (identical(fn_name, "round_date")) "second" else "seconds"
  if (!is.character(unit) || length(unit) != 1L || is.na(unit)) {
    return(NULL)
  }
  unit <- switch(unit,
    second = , seconds = , sec = , secs = , S = , s = "second",
    minute = , minutes = , min = , mins = , M = "minute",
    hour = , hours = , h = , H = "hour",
    day = , days = , d = , D = "day",
    week = , weeks = , w = , W = "week",
    return(NULL)
  )

  x_ir <- ir_parse_expr(present_args$x, env, schema)
  if (is.null(x_ir)) {
    return(NULL)
  }
  x_ir <- ir_posixct_shift(x_ir, schema)
  x_type <- unname(ir_infer_type(x_ir, schema))
  if (identical(x_type, "TIMESTAMP_DAYS")) {
    if (!unit %in% c("day", "week") ||
        (identical(unit, "week") && !isTRUE(getOption("lubridate.week.start", 7) == 7))) {
      return(NULL)
    }
  } else if (identical(x_type, "TIMESTAMP_MICROSECONDS")) {
    if (identical(unit, "week") || ir_needs_cpu_tz(x_ir, schema)) {
      return(NULL)
    }
  } else {
    return(NULL)
  }

  ir_call(paste(fn_name, unit, sep = "_"), list(x_ir))
}

#' Registry of supported expression-IR operations
#'
#' See the module-level comment above and section 6 of
#' `scratchpad/phase1_expression_engine.md` for the extension contract this
#' implements. Each entry: `list(arity, parse, type, lower)`.
#'
#' Phase 5 (`scratchpad/phase5_window_design.md` section 1.1) adds a fifth
#' field, `window`: `NULL` for every pre-Phase-5 entry (unchanged), or
#' `list(kind = "shift"|"rank"|"scan"|"agg")` for a window-class op. `window`
#' non-`NULL` and `lower` non-`NULL` are mutually exclusive -- a window-class
#' entry always has `lower = NULL` (window ops are never `ir_call` handlers
#' in `src/expr_eval.hpp`, see this file's own module comment for that
#' boundary, and are structurally prevented from reaching lowering at all
#' by [ir_bind()]'s guard). `ir_infer_type()` needs no change: a window
#' entry's `type` function is looked up and called exactly like any other
#' entry's.
#'
#' @keywords internal
ir_call_registry <- list(
  "+" = list(arity = 2L, parse = ir_parse_ts_arith("+"), type = type_arith_op("+"), lower = list(ast_op = "ADD")),
  "-" = list(arity = NA_integer_, parse = ir_parse_ts_arith("-"), type = type_minus, lower = list(ast_op = "SUB")),
  # F37: Date +/- a whole number of days; only ir_parse_ts_arith() builds them.
  "date_add_days" = list(arity = 2L, parse = NULL, type = function(arg_types) "TIMESTAMP_DAYS",
                         lower = list(handler = "date_add_days")),
  "date_sub_days" = list(arity = 2L, parse = NULL, type = function(arg_types) "TIMESTAMP_DAYS",
                         lower = list(handler = "date_sub_days")),
  # F46: a POSIXct shifted by a whole number of seconds, the literal in INT64
  # microseconds; only ir_posixct_shift() builds them.
  "ts_add_us" = list(arity = 2L, parse = NULL, type = function(arg_types) arg_types[[1]],
                     lower = list(handler = "ts_add_us")),
  "ts_sub_us" = list(arity = 2L, parse = NULL, type = function(arg_types) arg_types[[1]],
                     lower = list(handler = "ts_sub_us")),
  "*" = list(arity = 2L, parse = NULL, type = type_arith_op("*"), lower = list(ast_op = "MUL")),
  "/" = list(arity = 2L, parse = NULL, type = type_float64, lower = list(ast_op = "TRUE_DIV")),
  "^" = list(arity = 2L, parse = NULL, type = type_float64, lower = list(ast_op = "POW")),
  "%%" = list(arity = 2L, parse = NULL, type = type_arith_op("%%"), lower = list(ast_op = "PYMOD")),
  "%/%" = list(arity = 2L, parse = NULL, type = type_arith_op("%/%"), lower = list(ast_op = "FLOOR_DIV")),

  "==" = list(arity = 2L, parse = ir_parse_factor_cmp("=="), type = type_bool8, lower = list(ast_op = "EQUAL")),
  "!=" = list(arity = 2L, parse = ir_parse_factor_cmp("!="), type = type_bool8, lower = list(ast_op = "NOT_EQUAL")),
  "<"  = list(arity = 2L, parse = ir_parse_factor_cmp("<"), type = type_bool8, lower = list(ast_op = "LESS")),
  "<=" = list(arity = 2L, parse = ir_parse_factor_cmp("<="), type = type_bool8, lower = list(ast_op = "LESS_EQUAL")),
  ">"  = list(arity = 2L, parse = ir_parse_factor_cmp(">"), type = type_bool8, lower = list(ast_op = "GREATER")),
  ">=" = list(arity = 2L, parse = ir_parse_factor_cmp(">="), type = type_bool8, lower = list(ast_op = "GREATER_EQUAL")),

  "&" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "NULL_LOGICAL_AND")),
  "|" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "NULL_LOGICAL_OR")),
  "!" = list(arity = 1L, parse = NULL, type = type_bool8, lower = list(ast_op = "NOT")),
  "xor" = list(arity = 2L, parse = NULL, type = type_bool8, lower = list(ast_op = "BITWISE_XOR")),

  "is.na" = list(arity = 1L, parse = NULL, type = type_bool8, lower = list(ast_op = "IS_NULL")),

  # F55: TRUE only for a valid NaN (NA is FALSE; a non-double column has none).
  # A STRING or factor argument is R's own error/answer: CPU.
  "is.nan" = list(arity = 1L, parse = function(args, schema) {
    x_type <- tryCatch(unname(ir_infer_type(args[[1]], schema)), error = function(e) NA_character_)
    if (is.na(x_type) || x_type %in% c("STRING", "DICTIONARY32")) {
      return(NULL)
    }
    ir_call("is.nan", args)
  }, type = type_bool8, lower = list(handler = "is.nan")),

  "between" = list(
    arity = 3L,
    parse = function(args, schema) {
      x <- args[[1]]
      lo <- args[[2]]
      hi <- args[[3]]
      # dplyr's between() compares with vctrs (byte order for strings), so the
      # comparisons skip F48's session-collation rule.
      ir_call("&", list(ir_call(">=", list(x, lo), meta = list(c_order = TRUE)),
                        ir_call("<=", list(x, hi), meta = list(c_order = TRUE))))
    },
    type = type_bool8,
    lower = NULL  # always desugared at parse time; never lowered as "between"
  ),

  "%in%" = list(arity = 2L, parse = ir_parse_factor_in, type = type_bool8, lower = list(handler = "in")),

  "sqrt" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "SQRT")),

  "log" = list(
    arity = NA_integer_,  # log(x) or log(x, base)
    parse = function(args, schema) {
      if (length(args) == 2) {
        base_node <- args[[2]]
        if (!identical(base_node$kind, "lit") || isTRUE(base_node$na) ||
            is.null(base_node$value) || length(base_node$value) != 1) {
          return(NULL)  # non-constant base: unsupported, fall back
        }
        return(ir_call("/", list(
          ir_call("log", list(args[[1]])),
          ir_lit_from_r(log(base_node$value))
        )))
      }
      ir_call("log", args)
    },
    type = type_float64,
    lower = list(ast_op = "LOG")
  ),
  "log2" = list(
    arity = 1L,
    parse = function(args, schema) {
      ir_call("/", list(ir_call("log", args), ir_lit_from_r(log(2))))
    },
    type = type_float64,
    lower = NULL  # always desugared at parse time
  ),
  "log10" = list(
    arity = 1L,
    parse = function(args, schema) {
      ir_call("/", list(ir_call("log", args), ir_lit_from_r(log(10))))
    },
    type = type_float64,
    lower = NULL  # always desugared at parse time
  ),

  "exp" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "EXP")),
  "abs" = list(arity = 1L, parse = NULL, type = type_abs, lower = list(ast_op = "ABS")),
  "floor" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "FLOOR")),
  "ceil" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "CEIL")),
  "sin" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "SIN")),
  "cos" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "COS")),
  "tan" = list(arity = 1L, parse = NULL, type = type_float64, lower = list(ast_op = "TAN")),

  # round() ALWAYS returns a double in R, regardless of the input's type
  # (`typeof(round(1L))`, `typeof(round(TRUE))`, `typeof(round(1L, 2))` are
  # all "double") -- unlike abs(), which does preserve non-BOOL8 input
  # types. Verified against the dplyr oracle (T6 finding; corrects
  # phase1_expression_engine.md section 1.3's table, which grouped round()
  # with abs() as "argument type" -- empirically wrong for round()).
  "round" = list(arity = NA_integer_, parse = ir_parse_round,
                 type = type_float64, lower = list(handler = "round")),

  # --- Phase 4: conditional & vector functions ---
  # `if_else(cond, yes, no, missing = NULL)`: 3 or 4 positional args (the
  # `arity = NA` + validating `parse` hook is the same pattern `log(x[,
  # base])` already uses above). `ir_resolve_value_nas()` fixes up a bare
  # untyped NA among yes/no/missing before the call node is built (never
  # touching `cond`, see that function's own docs for why that matters).
  "if_else" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) < 3 || length(args) > 4) {
        return(NULL)  # wrong arg count: not this shape, fall back
      }
      cond <- args[[1]]
      values <- ir_resolve_value_nas(args[-1], schema)
      ir_call("if_else", c(list(cond), values))
    },
    type = type_if_else,
    lower = list(handler = "if_else")
  ),

  # `case_when(...)` itself never reaches this entry's `parse`/generic
  # per-arg loop -- see `ir_parse_case_when()`, invoked directly from
  # `ir_parse_expr()`'s call-handling branch because its dots are `~`
  # formulas. This entry exists only so `ir_infer_type()` can look up
  # `type`/`lower` for the `ir_call("case_when", ...)` node
  # `ir_parse_case_when()` builds.
  "case_when" = list(
    arity = NA_integer_,
    parse = NULL,
    type = type_case_when,
    lower = list(handler = "case_when")
  ),

  # F61: `recode(x, !!!lookup)` as one hash lookup. Only ir_parse_recode()
  # builds it: args are x, the key vector literal, the value vector literal,
  # the default (a literal, or x itself) and the `.missing` literal; the result
  # has the values' type.
  "recode_lookup" = list(
    arity = 5L,
    parse = NULL,
    type = function(arg_types) arg_types[[3]],
    lower = list(handler = "recode_lookup")
  ),

  # `coalesce(...)`: variadic, at least one argument.
  "coalesce" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) < 1) {
        return(NULL)
      }
      values <- ir_resolve_value_nas(args, schema)
      ir_call("coalesce", values)
    },
    type = type_coalesce,
    lower = list(handler = "coalesce")
  ),

  # `na_if(x, y)`: output keeps x's own type (type_na_if); a STRING/
  # non-STRING mismatch between x and y is diagnosed here, at parse time,
  # with a message naming both sides and their types (mirroring dplyr's own
  # "Can't convert `y` <type> to match type of `x` <type>." wording) rather
  # than surfacing as a raw cudf type error later.
  "na_if" = list(
    arity = 2L,
    parse = function(args, schema) {
      resolved <- ir_resolve_value_nas(args, schema)
      x <- resolved[[1]]
      y <- resolved[[2]]
      # unname(): a `col` node's type comes back from ir_infer_type() as
      # schema$types[idx] -- a NAMED single-element subset when
      # current_schema()'s own $types vector is named (as it is for any
      # real tbl_gpu), and identical() treats a "names" attribute as
      # significant (identical(c(s = "STRING"), "STRING") is FALSE even
      # though the values are equal). Same landmine, same fix, as
      # ir_check_type_operands()'s unname() call
      # -- without it, every na_if() STRING/non-STRING
      # mismatch silently skipped this check and only surfaced as a raw
      # cudf cast error at GPU-execution time instead.
      x_type <- unname(ir_infer_type(x, schema))
      y_type <- unname(ir_infer_type(y, schema))
      if (identical(x_type, "DICTIONARY32") || identical(y_type, "DICTIONARY32")) {
        return(NULL)  # a factor needs dplyr's labels: CPU fallback (F18)
      }
      if (identical(x_type, "STRING") != identical(y_type, "STRING")) {
        stop("na_if(): can't convert `y` (", y_type, ") to match the type of ",
             "`x` (", x_type, ").", call. = FALSE)
      }
      ir_call("na_if", list(x, y))
    },
    type = type_na_if,
    lower = list(handler = "na_if")
  ),

  # --- Phase 4 wave 2: near(), replace_when() (case_match()/recode_values()/
  # replace_values()/when_all()/when_any() all fully desugar at parse time
  # into "case_when"/"&"/"|" nodes -- see the module comment above
  # ir_parse_value_match_clauses() and never reach the registry under their
  # own op name at all) ---

  # `near(x, y, tol = .Machine$double.eps^0.5)`: verified against dplyr's
  # own source (`dplyr:::near`) -- `abs(x - y) < tol`, nothing more. Always
  # desugared at parse time (like `between`/`log2`/`log10` above), so "near"
  # never reaches lowering under its own op name.
  "near" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) < 2 || length(args) > 3) {
        return(NULL)  # wrong arg count: not this shape, fall back
      }
      x <- args[[1]]
      y <- args[[2]]
      tol <- if (length(args) == 3) args[[3]] else ir_lit_from_r(sqrt(.Machine$double.eps))
      ir_call("<", list(ir_call("abs", list(ir_call("-", list(x, y)))), tol))
    },
    type = type_bool8,
    lower = NULL  # always desugared at parse time; never lowered as "near"
  ),

  # `replace_when(x, ...)`/`replace_values(x, ...)`: never reach this
  # entry's `parse` -- both are built directly by their own dedicated parse
  # functions (`ir_parse_replace_when()`/`ir_parse_replace_values()`), since
  # `x` (arg 1) is a required leading positional argument the generic
  # per-arg registry loop has no special handling for. This entry exists
  # only so `ir_infer_type()`/lowering can look up `type`/`lower` for the
  # `ir_call("replace_when", ...)` node those parse functions build.
  "replace_when" = list(
    arity = NA_integer_,
    parse = NULL,
    type = type_replace_when,
    lower = list(handler = "replace_when")
  ),

  # --- Phase 5 (window functions): window-class entries ---
  # See scratchpad/phase5_window_design.md sections 1.1/4. Every entry here
  # has `lower = NULL` (window ops never reach lowering under their own op
  # name -- ir_bind()'s guard enforces this defensively) and a `window`
  # field carrying the broad C++ lowering family ("shift"/"rank"/"scan"/
  # "agg") that R/window.R's ir_extract_windows() reads (only for "rank",
  # to decide whether the value column doubles as the frame's order
  # column) -- the family itself has no other R-side effect this wave;
  # W2/W4 are where each family actually gets lowered.
  #
  # lag()/lead(): parsed by ir_parse_shift_call() (dedicated dispatch,
  # never via this entry's own `parse`/generic loop -- see that function's
  # docs for why named args force this). `type = type_arg1`: the shifted
  # value keeps x's own type (the fill default, when absent, is an invalid
  # scalar of that same type -- W2's C++ concern, not this entry's).
  "lag" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
               lower = NULL, window = list(kind = "shift")),
  "lead" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                lower = NULL, window = list(kind = "shift")),

  # row_number(): 0- or 1-arg (dplyr: bare `row_number()` inside a mutate/
  # filter is sequential position; `row_number(x)` ranks by x, ties broken
  # by original position -- rank_method::FIRST). Needs its own `parse` hook
  # (arity = NA) purely to re-validate the 0-or-1 arg count, since the
  # registry's generic arity check only fires for a single fixed arity;
  # the args themselves are ordinary positional parses, so no dedicated
  # top-level dispatch is needed (unlike lag()/lead()'s named controls).
  "row_number" = list(
    arity = NA_integer_,
    parse = function(args, schema) {
      if (length(args) > 1) {
        return(NULL)
      }
      ir_call("row_number", args)
    },
    type = type_int32,
    lower = NULL,
    window = list(kind = "rank")
  ),
  "min_rank" = list(arity = 1L, parse = NULL, type = type_int32,
                     lower = NULL, window = list(kind = "rank")),
  "dense_rank" = list(arity = 1L, parse = NULL, type = type_int32,
                       lower = NULL, window = list(kind = "rank")),
  "percent_rank" = list(arity = 1L, parse = NULL, type = type_float64,
                         lower = NULL, window = list(kind = "rank")),
  "cume_dist" = list(arity = 1L, parse = NULL, type = type_float64,
                      lower = NULL, window = list(kind = "rank")),

  # cumsum/cummax/cummin/cumprod: single-arg scan family. cumsum's type
  # reuses "+"'s own D5 promotion function verbatim (calling
  # infer_mutate_output_type("+", arg_types, NULL) with a single input type
  # gives exactly BOOL8/INT32 -> INT32, INT64 -> INT64, else FLOAT64 --
  # dplyr's own cumsum() promotion, confirmed by inspection of that
  # function's Rule 5 branch); cummax/cummin preserve the argument's type
  # (matching base R); cumprod always returns FLOAT64 (matching base R:
  # `typeof(cumprod(1:3))` is "double").
  "cumsum" = list(arity = 1L, parse = NULL, type = type_arith_op("+"),
                   lower = NULL, window = list(kind = "scan")),
  "cummax" = list(arity = 1L, parse = NULL, type = type_arg1,
                   lower = NULL, window = list(kind = "scan")),
  "cummin" = list(arity = 1L, parse = NULL, type = type_arg1,
                   lower = NULL, window = list(kind = "scan")),
  "cumprod" = list(arity = 1L, parse = NULL, type = type_float64,
                    lower = NULL, window = list(kind = "scan")),

  # mean/sum/min/max/n/sd/var: agg-family window entries (aggregate +
  # broadcast, see phase5_window_design.md section 4). These double as
  # summarise()'s own aggregation function names, but summarise()'s parser
  # (parse_aggregations(), R/summarise.R) never reaches ir_call_registry at
  # all -- it has its own independent dispatch table -- so registering
  # these here only affects expressions parsed via ir_parse_quo() (mutate()/
  # filter(), and summarise()'s temp-column sub-expression preprocessing,
  # which parses the *argument* of an aggregation call, never the
  # aggregation function name itself). Output types reuse
  # make_aggregation()'s own table verbatim (R/ast.R) except where that
  # table doesn't cover a case at all (sum(): type_window_sum reimplements
  # its exact switch branch, since arg_types here come from a window call's
  # own arg rather than an aggregation struct's `input_type` field).
  "mean" = list(arity = 1L, parse = NULL, type = type_mean_median,
                 lower = NULL, window = list(kind = "agg")),
  "sum" = list(arity = 1L, parse = NULL, type = type_window_int_sum,
                lower = NULL, window = list(kind = "agg")),
  "min" = list(arity = 1L, parse = ir_parse_bool8_cast("min", "cast_int32"),
                type = type_arg1, lower = NULL, window = list(kind = "agg")),
  "max" = list(arity = 1L, parse = ir_parse_bool8_cast("max", "cast_int32"),
                type = type_arg1, lower = NULL, window = list(kind = "agg")),
  "n" = list(arity = 0L, parse = NULL, type = type_int32,
              lower = NULL, window = list(kind = "agg")),
  "sd" = list(arity = 1L, parse = NULL, type = type_float64,
               lower = NULL, window = list(kind = "agg")),
  "var" = list(arity = 1L, parse = NULL, type = type_float64,
                lower = NULL, window = list(kind = "agg")),

  # median()/n_distinct() (Phase 6, task 6.2): join the agg family "for
  # free" once summarise() learned them, reusing get_groupby_agg()
  # (src/ops_groupby.cpp, genuinely shared source between the summarise()
  # and window-mutate() paths) -- verified empirically against dplyr 1.2.1
  # (test-dplyr-summarise.R's window-parity oracle tests). Like every
  # other "agg"-family window function, neither supports a na.rm=/na_rm=
  # ARGUMENT here (arity is fixed at 1L, same as mean()/sum()/...; a
  # `median(x, na.rm = TRUE)` mutate() call falls through to mutate()'s own
  # "unsupported named argument" error, exactly like `mean(x, na.rm =
  # TRUE)` already does) -- median() always gets na.rm=FALSE's whole-
  # group-null propagation (src/ops_window.cpp's na_propagates list),
  # n_distinct() always counts NA as its own distinct value
  # (null_policy::INCLUDE, get_groupby_agg()'s "n_distinct" branch).
  # quantile()/any()/all() are NOT added here -- see is_agg_fn()'s own
  # docs (src/ops_window.cpp) for why.
  "median" = list(arity = 1L, parse = NULL, type = type_mean_median,
                   lower = NULL, window = list(kind = "agg")),
  "n_distinct" = list(arity = 1L, parse = NULL, type = type_int32,
                       lower = NULL, window = list(kind = "agg")),

  # first()/last()/nth(): agg-family, but order-sensitive (dplyr's own
  # "first"/"last"/"nth row *of the group, in its current order*" -- W1
  # only supports the no-order_by/no-default/no-na_rm shape, hence the
  # dedicated dispatch in ir_parse_first_last_nth(); type = type_arg1
  # (preserves the argument's own type, per phase5_window_design.md
  # section 4's "-> arg type", NOT make_aggregation()'s generic table,
  # which doesn't cover these three functions at all).
  "first" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                  lower = NULL, window = list(kind = "agg")),
  "last" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                 lower = NULL, window = list(kind = "agg")),
  "nth" = list(arity = NA_integer_, parse = NULL, type = type_arg1,
                lower = NULL, window = list(kind = "agg")),

  # --- Phase 5: pure parse-time desugars that happen to produce window IR
  # (window = NULL / lower = NULL, exactly like "between"/"log2"/"near"
  # above -- these op names never survive to appear in a finished tree
  # under their own name, only the window-class calls their `parse` hooks
  # build do) ---
  "cummean" = list(arity = 1L, parse = ir_parse_cummean, type = type_float64,
                    lower = NULL, window = NULL),
  "cumall" = list(
    arity = 1L,
    parse = function(args, schema) ir_parse_cumall_cumany(args, schema, "cummin"),
    type = type_bool8,
    lower = NULL,
    window = NULL
  ),
  "cumany" = list(
    arity = 1L,
    parse = function(args, schema) ir_parse_cumall_cumany(args, schema, "cummax"),
    type = type_bool8,
    lower = NULL,
    window = NULL
  ),
  "consecutive_id" = list(arity = 1L, parse = ir_parse_consecutive_id, type = type_int32,
                           lower = NULL, window = NULL),

  # --- Rolling windows (Theo-requested benchmark task, a sixth window
  # kind alongside shift/rank/scan/agg): roll_mean()/roll_sum()/roll_min()/
  # roll_max()/roll_sd()/roll_median(), parsed via the dedicated
  # ir_parse_roll_call()/ir_parse_froll_call() dispatch above (never via
  # this entry's own `parse`/generic loop -- na.rm= is a named control the
  # generic per-arg loop can't express, same reason as lag()/lead()).
  # Output types: roll_mean/roll_sd/roll_median always FLOAT64 (matching
  # data.table::frollmean()/frollsd()/frollmedian(), always double);
  # roll_sum reuses type_window_sum verbatim (identical promotion rule to
  # window sum()/summarise's sum(): INT32 -> INT64, else FLOAT64);
  # roll_min/roll_max preserve the argument's own type (type_arg1, matching
  # cudf::rolling_window()'s own documented "MIN/MAX return a column of the
  # same type as the input" rule). `window_spec_lowerable()` (R/lower.R)
  # excludes "roll_median" specifically -- cudf's rolling_window()/
  # grouped_rolling_window() have no MEDIAN aggregation instantiation in
  # this environment (verified via `nm -DC libcudf.so`) -- routing it
  # through the existing CPU-fallback path instead (see R/roll.R).
  "roll_mean" = list(arity = 1L, parse = NULL, type = type_float64,
                      lower = NULL, window = list(kind = "rolling")),
  "roll_sum" = list(arity = 1L, parse = NULL, type = type_window_sum,
                     lower = NULL, window = list(kind = "rolling")),
  "roll_min" = list(arity = 1L, parse = NULL, type = type_arg1,
                     lower = NULL, window = list(kind = "rolling")),
  "roll_max" = list(arity = 1L, parse = NULL, type = type_arg1,
                     lower = NULL, window = list(kind = "rolling")),
  "roll_sd" = list(arity = 1L, parse = NULL, type = type_float64,
                    lower = NULL, window = list(kind = "rolling")),
  "roll_median" = list(arity = 1L, parse = NULL, type = type_float64,
                        lower = NULL, window = list(kind = "rolling")),

  # --- Phase 11 L3: as.Date()/as.POSIXct() on an existing TIMESTAMP_*
  # column, or a column-independent (constant) argument ---
  #
  # `arity = 1L` deliberately: any additional argument (`format=`, `tz=`,
  # `origin=`, ...) makes `length(args_raw) != 1`, so the registry's own
  # dispatch loop returns NULL before `parse` is even called -- falling
  # back to L1's whole-call CPU path, exactly like a STRING-source column
  # (see `parse` below) rather than attempting to replicate those
  # semantics natively.
  #
  # Two shapes reach `parse`:
  #   - `args[[1]]$kind == "col"`: a real schema column. Native only when
  #     its OWN inferred type is already one of the 5 TIMESTAMP_*
  #     granularities (`ir_call("as_date"/"as_posixct", ...)`, lowered by
  #     `src/expr_eval.hpp`'s `apply_handler()` via a single
  #     `cudf::cast()`); a STRING or numeric source column returns NULL
  #     (design's explicit scope cut -- STRING parsing/numeric-origin
  #     interpretation isn't attempted), falling back to L1.
  #   - `args[[1]]$kind == "lit"`: a column-independent constant (already
  #     evaluated by the generic per-arg parser -- this is what makes
  #     `as.Date("2010-01-05")`/`as.Date(Sys.Date())` reach here instead of
  #     the old pre-registration constant-fold path, since registering
  #     "as.Date"/"as.POSIXct" here intercepts EVERY call to it, including
  #     a purely literal one). Reproduces the exact value the old
  #     constant-fold would have produced by applying the real R function
  #     to the literal's own value (or `NA` for a bare untyped NA node) and
  #     re-wrapping via `ir_lit_from_r()` -- which is what
  #     `gpu_type_from_r()` already types as TIMESTAMP_DAYS/
  #     TIMESTAMP_MICROSECONDS for a `Date`/`POSIXct` result. This keeps
  #     e.g. `filter(d < as.Date("2010-01-05"))` on the fully native path
  #     (a `<` comparing a real "col" node against a TIMESTAMP_DAYS "lit"
  #     node) instead of silently falling back to the CPU for a
  #     column-referencing dot (see `dot_references_schema_col()`,
  #     R/filter.R).
  #   - Any other node kind (a derived, non-literal, non-column
  #     expression, e.g. `as.Date(x + 1)`): NULL, conservative fallback --
  #     not attempted natively.
  #
  # `type`/`lower` on THIS entry are never actually consulted:
  # `ir_infer_type()`/documentation lookups key off the node's OWN `op`
  # field, and `parse` always rewrites the node to either a `lit` (handled
  # entirely in R, no "as.Date" op ever reaches lowering) or an
  # `ir_call("as_date"/"as_posixct", ...)` (looked up under ITS OWN
  # registry key, below) -- kept here only for the same
  # documentation-consistency reason `case_when`/`between` do the same.
  "as.Date" = list(
    arity = 1L,
    parse = function(args, schema) {
      x <- args[[1]]
      if (identical(x$kind, "col")) {
        x_type <- unname(ir_infer_type(x, schema))
        if (x_type %in% cuplyr_timestamp_types) {
          return(ir_call("as_date", list(x)))
        }
        return(NULL)
      }
      if (identical(x$kind, "lit")) {
        val <- if (isTRUE(x$na)) NA else x$value
        return(ir_lit_from_r(as.Date(val)))
      }
      NULL
    },
    type = type_float64,
    lower = NULL
  ),
  "as.POSIXct" = list(
    arity = 1L,
    parse = function(args, schema) {
      x <- args[[1]]
      if (identical(x$kind, "col")) {
        x_type <- unname(ir_infer_type(x, schema))
        if (x_type %in% cuplyr_timestamp_types) {
          return(ir_call("as_posixct", list(x)))
        }
        return(NULL)
      }
      if (identical(x$kind, "lit")) {
        val <- if (isTRUE(x$na)) NA else x$value
        return(ir_lit_from_r(as.POSIXct(val)))
      }
      NULL
    },
    type = type_float64,
    lower = NULL
  ),

  # The actual lowered ops (see the "as.Date"/"as.POSIXct" `parse` hooks
  # above): a single-arg `cudf::cast()` to a fixed target granularity,
  # `apply_handler()`'s "as_date"/"as_posixct" case (src/expr_eval.hpp).
  # `type` is fixed regardless of the (already-verified-timestamp) input
  # granularity, matching `as.Date()`/`as.POSIXct()`'s own R semantics
  # (always returns a `Date`/`POSIXct`, never a mix).
  #
  # F32: this key is also what a user's `lubridate::as_date(x)` resolves to,
  # and lubridate takes the calendar day in the column's own zone while the
  # cast below is the UTC day. `as.Date()` (UTC by default in R 4.6) builds
  # the node directly and is not re-parsed, so only the user-callable form
  # goes through this hook: a non-UTC POSIXct goes to the CPU.
  "as_date" = list(arity = 1L,
                    parse = function(args, schema) {
                      if (ir_needs_cpu_tz(args[[1]], schema)) {
                        return(NULL)
                      }
                      ir_call("as_date", args)
                    },
                    type = function(arg_types) "TIMESTAMP_DAYS",
                    lower = list(handler = "as_date")),
  "as_posixct" = list(arity = 1L, parse = NULL,
                       type = function(arg_types) "TIMESTAMP_MICROSECONDS",
                       lower = list(handler = "as_posixct")),

  # --- Phase 11 L5: base-R string functions ---
  # toupper()/tolower() reach here via the ORDINARY registry per-arg loop
  # (fixed arity = 1, no optional named controls) -- unlike every other L5
  # entry below, which is parsed entirely by its own dedicated dispatch
  # function (ir_parse_nchar()/ir_parse_substr()/etc., invoked directly from
  # ir_parse_expr()'s call-handling branch, mirroring lag()/lead()/ntile())
  # because real R lets their optional arguments be named/omitted/reordered
  # in ways the registry's positional-only per-arg loop can't validate. Every
  # dedicated-dispatch entry below exists only so ir_infer_type()/lowering
  # can look up `type`/`lower` for the finished ir_call(...) node each
  # dispatch function builds (same "documentation, not dispatch" role
  # "case_when"/"replace_when" play above).
  "toupper" = list(
    arity = 1L,
    parse = function(args, schema) {
      if (!ir_is_string_type(args[[1]], schema)) return(NULL)
      ir_call("toupper", args)
    },
    type = function(arg_types) "STRING",
    lower = list(handler = "toupper")
  ),
  "tolower" = list(
    arity = 1L,
    parse = function(args, schema) {
      if (!ir_is_string_type(args[[1]], schema)) return(NULL)
      ir_call("tolower", args)
    },
    type = function(arg_types) "STRING",
    lower = list(handler = "tolower")
  ),
  "nchar" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "nchar")),
  # F48: ntile()'s final step (cudf::cast of its FLOAT64 bins to INT32);
  # internal only (ir_internal_ops)
  "cast_int32" = list(arity = 1L, parse = NULL, type = type_int32,
                      lower = list(handler = "cast_int32")),
  # F50: a factor column's codes / labels / codes as a double. Built only by
  # ir_parse_factor_cmp()/ir_parse_factor_in()/ir_parse_factor_conversion();
  # listed in ir_internal_ops, so a user call by these names never lands here.
  # factor_codes is stripped to its column by ir_bind(); factor_labels' second
  # argument is a STRING lit holding the level vector.
  "factor_codes" = list(arity = 1L, parse = NULL, type = type_int32,
                        lower = list(handler = "factor_codes")),
  "factor_labels" = list(arity = 2L, parse = NULL, type = function(arg_types) "STRING",
                         lower = list(handler = "factor_labels")),
  "cast_float64" = list(arity = 1L, parse = NULL, type = type_float64,
                        lower = list(handler = "cast_float64")),
  # F49: a rowwise horizontal aggregate (sum/mean/min/max/prod over the
  # columns of one row); meta$fn and meta$na_rm say which. Built only by
  # R/rowwise.R's rewrite and embedded in the dot as `.cuplyr_rw(<node>, cols)`
  # (ir_parse_expr()); `_i32` is the INT32-typed form (integer sum/min/max),
  # `_f64` the FLOAT64 one. Arguments are columns.
  "rw_agg_i32" = list(arity = NA_integer_, parse = NULL, type = type_int32,
                      lower = list(handler = "rw_agg_i32")),
  "rw_agg_f64" = list(arity = NA_integer_, parse = NULL, type = type_float64,
                      lower = list(handler = "rw_agg_f64")),
  "substr" = list(arity = 3L, parse = NULL, type = function(arg_types) "STRING",
                   lower = list(handler = "substr")),
  "trimws" = list(arity = 2L, parse = NULL, type = function(arg_types) "STRING",
                   lower = list(handler = "trimws")),
  "paste_cols" = list(arity = NA_integer_, parse = NULL, type = function(arg_types) "STRING",
                       lower = list(handler = "paste_cols")),
  "grepl" = list(arity = 3L, parse = NULL, type = type_bool8, lower = list(handler = "grepl")),
  "sub" = list(arity = 4L, parse = NULL, type = function(arg_types) "STRING",
               lower = list(handler = "sub")),
  "gsub" = list(arity = 4L, parse = NULL, type = function(arg_types) "STRING",
                lower = list(handler = "gsub")),
  "starts_with" = list(arity = 2L, parse = NULL, type = type_bool8,
                        lower = list(handler = "starts_with")),
  "ends_with" = list(arity = 2L, parse = NULL, type = type_bool8,
                      lower = list(handler = "ends_with")),

  # --- Phase 11 L6: lubridate datetime accessors ---
  # Dispatched only via ir_parse_lubridate_accessor(), reached only after
  # the mandatory ir_require_lubridate_fn() namespace guard passes (both
  # above) -- these entries exist purely for ir_infer_type()/lowering
  # lookup. Each cuDF call (extract_datetime_component()/extract_quarter()/
  # day_of_year(), datetime.hpp) returns INT16, cast immediately to INT32
  # inside the C++ handler itself (not left to a later "declared output
  # type" cast step) so a nested use (e.g. `filter(year(d) == 2020)`, inside
  # build_ast()'s own static type resolution) sees the real INT32 type --
  # see resolve_static_type()'s matching new cases, src/expr_eval.hpp.
  "year" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "year")),
  "month" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "month")),
  "day" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "day")),
  "hour" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "hour")),
  "minute" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "minute")),
  "second" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "second")),
  "quarter" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "quarter")),
  "yday" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "yday")),
  # wday: cuDF WEEKDAY is ISO Mon=1..Sun=7; lubridate::wday()'s default
  # (week_start = 7, i.e. Sunday) is Sun=1..Sat=7 -- verified empirically
  # (plain Rscript, no GPU) -- remapped via `(w %% 7) + 1` inside the C++
  # handler itself (src/expr_eval.hpp).
  "wday" = list(arity = 1L, parse = NULL, type = type_int32, lower = list(handler = "wday"))

  # ntile() is dispatched directly from ir_parse_expr() (ir_parse_ntile()),
  # never through this registry at all (its "x supplied or not" distinction
  # needs raw-call inspection the generic per-arg loop can't provide) -- so,
  # unlike cummean()/cumall()/cumany()/consecutive_id() above, it has no
  # entry here; ir_infer_type()/ir_bind() never need to look up "ntile" as
  # an op name, since it never appears in a finished tree either.
)

# F46: the lubridate rounding ops all return their operand's type.
for (.op in ir_round_ops) {
  ir_call_registry[[.op]] <- list(
    arity = 1L, parse = NULL, type = function(arg_types) arg_types[[1]],
    lower = list(handler = .op)
  )
}
rm(.op)
