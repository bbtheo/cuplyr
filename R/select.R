#' Select columns from a GPU table
#'
#' Keeps only the specified columns from a GPU table, similar to
#' `dplyr::select()`. Supports tidyselect syntax for flexible column selection.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names or tidyselect expressions specifying which columns
#'   to keep. Supports:
#'   \itemize{
#'     \item Column names: `select(x, y, z)`
#'     \item Negative selection: `select(-x)`
#'     \item Range: `select(x:z)`
#'     \item Predicates: `where(is.numeric)`. Base type tests (`is.numeric`,
#'       `is.double`, `is.integer`, `is.character`, `is.logical`, `is.factor`,
#'       `is.ordered`, `inherits(x, "Date")`), bare or combined in a lambda
#'       such as `~ is.numeric(.x) || is.factor(.x)`, are answered from the
#'       column types. Any other predicate (`where(~ anyNA(.x))`,
#'       `lubridate::is.Date`), and any call that is not a tidyselect helper
#'       or a base name builder (`sort(names(x))`, `paste0()`, `rev()`,
#'       `setdiff()`, `.data$x`, ... with safe arguments), is resolved on the
#'       data: a notified CPU fallback that collects the table
#'     \item Helpers: `starts_with()`, `ends_with()`, `contains()`, etc.
#'     \item Renaming while selecting: `select(new_name = old_name)`
#'   }
#'
#' @return A `tbl_gpu` object containing only the selected columns.
#'   Column order matches the order specified in the selection; any
#'   `new = old` selections are renamed in the result.
#'
#' @details
#' Column selection creates a new GPU table with only the selected columns.
#' The original data remains in GPU memory until garbage collected.
#'
#' ## Renaming
#' `select(new_name = old_name)` both selects and renames `old_name`. This
#' is schema-only (see [rename.tbl_gpu()]): the underlying selection still
#' runs as one `gpu_select()` gather, with the rename applied on top of it
#' as a separate, schema-only [ast_rename()] step whenever the tidyselect
#' resolution renamed a column.
#'
#' ## Grouped tables
#' Grouping columns that the selection leaves out are added in front, with
#' the message "Adding missing grouping variables", as in dplyr. Groups
#' follow their column by position, so `select(group_by(df, a), a = b)` is
#' ungrouped.
#'
#' ## Zero columns
#' A selection that keeps no column (`select(df, -everything())`) gives a
#' zero-column table that still knows its row count. A lazy table with
#' pending operations is computed first to get that count.
#'
#' ## Performance
#' Select operations involve copying column data to a new table structure.
#' For very wide tables, selecting fewer columns can significantly reduce
#' memory usage and improve performance of subsequent operations.
#'
#' @seealso
#' \code{\link{filter.tbl_gpu}} for filtering rows,
#' \code{\link{mutate.tbl_gpu}} for creating columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr select
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Select specific columns
#'   result <- gpu_mtcars |>
#'     select(mpg, cyl, hp) |>
#'     collect()
#'
#'   # Select with tidyselect helpers
#'   result <- gpu_mtcars |>
#'     select(starts_with("d")) |>
#'     collect()
#'
#'   # Reorder columns
#'   result <- gpu_mtcars |>
#'     select(hp, mpg, wt) |>
#'     collect()
#' }
select.tbl_gpu <- function(.data, ...) {
  schema <- current_schema(.data)

  # tidyselect resolves against a frame of the schema's column types, so
  # type-test where() predicates see types (and a duplicated name stays
  # addressable by position); any other predicate is resolved on the real
  # data (see resolve_selection()).
  quos <- rlang::enquos(...)
  call <- rlang::current_env()
  sel <- resolve_selection(.data, schema, quos, "select", function(data) {
    tidyselect::eval_select(rlang::expr(c(!!!quos)), data, error_call = call)
  }, plain = TRUE)
  vars <- sel$loc
  .data <- sel$data

  groups <- .subset2(.data, "groups")
  if (length(groups) > 0L) {
    vars <- select_ensure_group_vars(vars, schema$names, groups)
  }

  if (length(vars) == 0L) {
    return(zero_column_result(.data, "select"))
  }

  # `names(vars)` are the OUTPUT names tidyselect resolved -- for a plain
  # selection these are identical to the selected columns' own names, but
  # for `select(new = old)` they carry the rename. Uniqueness of the
  # renamed set is already validated by eval_select() itself (unlike
  # rename()'s partial-touch case, select()'s output IS the full result,
  # so there's no separate untouched-column collision to check -- see
  # R/rename.R's check_rename_unique() doc comment for the contrast).
  selected_cols <- schema$names[vars]
  output_names <- names(vars)

  out <- push_select_rename(.data, schema$names, vars)

  if (length(groups) > 0L) {
    out <- set_groups_by_position(out, .data, vars, schema$names)
  }
  if (is_rowwise_gpu(.data) && !is_rowwise_gpu(out)) {
    out <- rowwise_mark(out, character())  # (a rowwise table with duplicated names, no vars)
  }
  rowwise_column_order(out)
}

# =============================================================================
# Shared tidyselect resolution for select(), relocate(), rename(),
# rename_with()
#
# tidyselect runs where() predicates on the data it is given. The GPU table
# has no R-side data, so a one-row frame of PROXY columns stands in: per
# schema column, a value of the class collect() gives it (a factor with the
# column's levels and `ordered`). That answers a predicate exactly only when
# the answer depends on the class alone, so the proxy is used only when a
# static walk (selection_is_type_only()) proves every predicate is a base
# type test: `is.numeric`, `is.double`, `is.integer`, `is.character`,
# `is.logical`, `is.factor`, `is.ordered`, or `inherits(x, "<constant>")`,
# bare or in a lambda/closure combining them with `!`, `&&`, `||`. Names,
# positions, ranges and the name-pattern helpers need no predicate at all.
# Anything else (a predicate that reads values, a type test the walk can't
# see through such as `lubridate::is.Date`, a user function, an unknown call)
# resolves the selection on the real data: a notified CPU fallback that
# computes a lazy plan once and builds the verb's result on the computed
# table.
# =============================================================================

# One proxy value per GPU type, built once (the POSIXct constructor is not
# free): what collect() returns for that type. Types collect() has no
# conversion for come back as doubles.
proxy_cache <- new.env(parent = emptyenv())

tidyselect_proxy_value <- function(type, levels = NULL) {
  if (identical(type, "DICTIONARY32") && !is.null(levels)) {
    code <- if (length(levels) > 0L) 1L else NA_integer_
    return(structure(
      code, levels = as.character(levels),
      class = c(if (isTRUE(attr(levels, "ordered"))) "ordered", "factor")
    ))
  }
  if (!is.null(hit <- proxy_cache[[type]])) {
    return(hit)
  }
  v <- switch(type,
    BOOL8 = TRUE,
    INT32 = 1L,
    STRING = "a",
    TIMESTAMP_DAYS = as.Date("1970-01-01"),
    TIMESTAMP_MICROSECONDS = as.POSIXct("1970-01-01", tz = "UTC"),
    DICTIONARY32 = factor("a"),
    1
  )
  proxy_cache[[type]] <- v
  v
}

# One-row frame of `schema`'s column classes for tidyselect, grouped like the
# table so group_cols() works. Built with new_data_frame() so duplicated
# column names are allowed.
# @keywords internal
tidyselect_proxy <- function(schema, groups = character()) {
  cols <- Map(tidyselect_proxy_value, unname(schema$types), levels_by_position(schema))
  names(cols) <- schema$names
  frame <- vctrs::new_data_frame(cols, n = 1L)
  if (length(groups) == 0L) {
    return(frame)
  }
  # one row, one group
  keys <- cols[match(groups, schema$names)]
  dplyr::new_grouped_df(
    tibble::new_tibble(frame),
    groups = tibble::new_tibble(c(keys, list(.rows = vctrs::list_of(1L))), nrow = 1L)
  )
}

# Is every predicate in this selection (a quosure, a list of them, or an
# expression walked inside one) answered by the column classes alone?
# Symbols that name a column, constants and the name/pattern helpers carry no
# predicate. A where() needs where_predicate_is_type_only(); a function
# standing for a predicate (a bare function symbol, a spliced function) is
# checked the same way. Any other call may expand to a value predicate.
# @param x Quosure, list of quosures, or expression
# @param env Environment for the symbols of a bare expression
# @param cols Column names of the schema the selection resolves against
# @keywords internal
selection_is_type_only <- function(x, env, cols) {
  if (rlang::is_missing(x)) {
    return(TRUE)
  }
  if (rlang::is_quosure(x)) {
    return(selection_is_type_only(rlang::quo_get_expr(x), rlang::quo_get_env(x), cols))
  }
  if (is.list(x)) {
    return(all(vapply(x, selection_is_type_only, logical(1), env = env, cols = cols)))
  }
  if (is.function(x)) {
    return(where_predicate_is_type_only(x, env))
  }
  if (is.symbol(x)) {
    # a symbol that is not a column is looked up in the environment, where it
    # could be a predicate function (a deprecated but accepted selection)
    name <- as.character(x)
    if (name %in% cols) {
      return(TRUE)
    }
    fn <- rlang::env_get(env, name, default = NULL, inherit = TRUE)
    return(!is.function(fn) || where_predicate_is_type_only(fn, env))
  }
  if (!is.call(x)) {
    return(TRUE)
  }
  name <- pick_call_name(x[[1L]])
  if (is.null(name)) {
    return(FALSE)
  }
  if (identical(name, "where")) {
    return(length(x) == 2L && where_predicate_is_type_only(x[[2L]], env))
  }
  if (name %in% c("c", "-", "!", "&", "|", "(", ":")) {
    return(selection_is_type_only(as.list(x)[-1L], env, cols))
  }
  if (name %in% c(
    "everything", "last_col", "starts_with", "ends_with", "contains", "matches",
    "num_range", "all_of", "any_of", "one_of", "group_cols"
  )) {
    return(TRUE)
  }
  name_builder_is_safe(x, env)
}

# Base functions that build names or positions out of their arguments and
# can't return a predicate; `selection_is_type_only()` accepts a call to one
# only when every argument is itself safe (name_builder_is_safe()).
name_builder_fns <- c(
  "names", "colnames", "sort", "rev", "unique", "setdiff", "union", "intersect",
  "paste", "paste0", "sprintf", "toupper", "tolower", "seq_len", "seq", "head",
  "tail", "[", "c", ":"
)

# The functions a name builder `name` may legitimately be: the base one, plus
# the generic dplyr attaches over setdiff/union/intersect (it falls back to
# base on vectors), and utils' head/tail.
# @keywords internal
name_builder_originals <- function(name) {
  home <- if (name %in% c("head", "tail")) asNamespace("utils") else baseenv()
  out <- list(base::get0(name, home))
  if (name %in% c("setdiff", "union", "intersect")) {
    out <- c(out, list(getExportedValue("dplyr", name)))
  }
  out
}

# Is `x` a name-building expression: a constant, a symbol that is not a
# function, `.data$x` / `.data[["x"]]`, or a call to a base name builder
# (looked up in `env` and compared by identity, so a user function named
# `sort` doesn't pass) whose arguments are all safe in turn? Anything else
# (a user helper, `c(helper(), ...)` inside a builder) may hide a predicate.
# @keywords internal
name_builder_is_safe <- function(x, env) {
  if (rlang::is_quosure(x)) {
    return(name_builder_is_safe(rlang::quo_get_expr(x), rlang::quo_get_env(x)))
  }
  if (rlang::is_missing(x)) {
    return(TRUE)
  }
  if (is.symbol(x)) {
    fn <- rlang::env_get(env %||% globalenv(), as.character(x), default = NULL, inherit = TRUE)
    return(!is.function(fn))
  }
  if (!is.call(x)) {
    return(!is.function(x))
  }
  if (is_data_pronoun_call(x)) {
    return(TRUE)
  }
  name <- pick_call_name(x[[1L]])
  if (is.null(name) || !name %in% name_builder_fns) {
    return(FALSE)
  }
  fn <- if (is.symbol(x[[1L]])) {
    rlang::env_get(env %||% globalenv(), name, default = NULL, inherit = TRUE)
  } else {
    base::get0(name, baseenv())
  }
  if (!any(vapply(name_builder_originals(name), identical, logical(1), fn))) {
    return(FALSE)
  }
  all(vapply(as.list(x)[-1L], name_builder_is_safe, logical(1), env = env))
}

# A where() argument whose answer depends only on the column's class: a base
# type test (bare, `base::` or as a value), or a lambda (`~ body`, a formula
# value, `function(x) body`, a closure value) whose body only applies type
# tests to its own argument (type_only_body()).
# @param pred The where() argument: expression, quosure or evaluated value
# @param env Environment `pred`'s symbols are looked up in
# @keywords internal
where_predicate_is_type_only <- function(pred, env) {
  if (rlang::is_quosure(pred)) {
    return(where_predicate_is_type_only(rlang::quo_get_expr(pred), rlang::quo_get_env(pred)))
  }
  if (rlang::is_call(pred, "~", n = 1L)) {
    # a lambda written in place, or a formula value with its own environment
    return(type_only_body(pred[[2L]], c(".x", ".", "..1"),
                          attr(pred, ".Environment") %||% env))
  }
  if (rlang::is_call(pred, "function")) {
    arg <- names(pred[[2L]])[1L]
    return(!is.na(arg) && !identical(arg, "...") && type_only_body(pred[[3L]], arg, env))
  }
  if (is.symbol(pred) || rlang::is_call(pred, "::", n = 2L)) {
    pred <- tryCatch(eval(pred, env), error = function(e) NULL)
    if (rlang::is_formula(pred)) {
      return(where_predicate_is_type_only(pred, env))
    }
  }
  if (!is.function(pred)) {
    return(FALSE)
  }
  if (is_base_type_test(pred)) {
    return(TRUE)
  }
  if (is.primitive(pred)) {
    return(FALSE)
  }
  # a closure value: its body, over its first argument, in its own environment
  arg <- if (inherits(pred, "rlang_lambda_function")) c(".x", ".", "..1") else names(formals(pred))[1L]
  !anyNA(arg) && !identical(arg, "...") && type_only_body(body(pred), arg, environment(pred))
}

# Base R's class tests a proxy of the right class answers exactly.
# @keywords internal
is_base_type_test <- function(fn) {
  for (test in list(is.numeric, is.double, is.integer, is.character, is.logical,
                    is.factor, is.ordered)) {
    if (identical(fn, test)) return(TRUE)
  }
  FALSE
}

# Is `expr` built only from base type tests and `inherits(<arg>, "<constant>")`
# applied to the predicate's own argument (one of `args`), joined by `!`,
# `&&`, `||`, `&`, `|`, `(` or a one-expression `{`? Call heads are looked up
# in `env`, so a user function named like a type test doesn't pass.
# @keywords internal
type_only_body <- function(expr, args, env) {
  if (rlang::is_call(expr, "(", n = 1L) || rlang::is_call(expr, "{", n = 1L) ||
        rlang::is_call(expr, "!", n = 1L)) {
    return(type_only_body(expr[[2L]], args, env))
  }
  if (rlang::is_call(expr, c("&&", "||", "&", "|"), n = 2L)) {
    return(type_only_body(expr[[2L]], args, env) && type_only_body(expr[[3L]], args, env))
  }
  if (!is.call(expr) || !(is.symbol(expr[[1L]]) || rlang::is_call(expr[[1L]], "::", n = 2L))) {
    return(FALSE)
  }
  fn <- tryCatch(eval(expr[[1L]], env), error = function(e) NULL)
  call_args <- as.list(expr)[-1L]
  arg_names <- names(call_args) %||% rep("", length(call_args))
  is_own_arg <- function(a) is.symbol(a) && as.character(a) %in% args
  if (is.function(fn) && is_base_type_test(fn)) {
    return(length(call_args) == 1L && arg_names %in% c("", "x") && is_own_arg(call_args[[1L]]))
  }
  if (identical(fn, base::inherits)) {
    return(length(call_args) == 2L && arg_names[1L] %in% c("", "x") &&
             arg_names[2L] %in% c("", "what") && is_own_arg(call_args[[1L]]) &&
             is.character(call_args[[2L]]) && !anyNA(call_args[[2L]]))
  }
  FALSE
}

# `eval_select(c(!!!quos), data)` for the commonest selection, without
# tidyselect: every dot unnamed and a bare symbol (or an unnamed length-1 string
# with no attributes) that names a column, no column repeated, on a table with
# unique column names. A column wins over an environment variable of the same
# name, so the result is just the positions named by the columns. Returns NULL for anything else
# (named dots, calls, numbers, `-x`, helpers, an empty selection, a name that is
# no column, `.data`/`.env`, duplicated schema names, a repeated column), which
# keeps tidyselect's errors, warnings and messages. Reads nothing but the
# quosures' expressions and the schema, so nothing is cached across calls.
# @keywords internal
plain_selection_loc <- function(quos, names) {
  n <- length(quos)
  if (n == 0L) {
    return(NULL)
  }
  if (!is.null(qn <- names(quos)) && any(nzchar(qn))) {
    return(NULL)
  }
  cols <- character(n)
  for (i in seq_len(n)) {
    q <- .subset2(quos, i)
    # .subset2(): `[[.quosure` is deprecated and signals. No variable may hold
    # the expression: an empty argument is the missing arg
    if (is.symbol(.subset2(q, 2L))) {
      cols[[i]] <- as.character(.subset2(q, 2L))
    } else if (is.character(.subset2(q, 2L)) && length(.subset2(q, 2L)) == 1L &&
                 !is.na(.subset2(q, 2L)) && is.null(attributes(.subset2(q, 2L)))) {
      # an injected string with a name (`!!c(a = "x")`: tidyselect renames) or
      # any other attribute is not a plain column name
      cols[[i]] <- .subset2(q, 2L)
    } else {
      return(NULL)
    }
  }
  if (anyDuplicated(cols) > 0L || anyDuplicated(names) > 0L ||
        any(cols == "" | cols == ".data" | cols == ".env")) {
    return(NULL)
  }
  loc <- match(cols, names)
  if (anyNA(loc)) {
    return(NULL)
  }
  names(loc) <- cols
  loc
}

# Resolve a selection to positions, for select()/relocate()/rename()/
# rename_with(): on the proxy when selection_is_type_only(), otherwise on the
# real data with a notified CPU fallback.
#
# @param .data The input tbl_gpu
# @param schema `current_schema(.data)`
# @param quos The quosures of the selection (all slots that take one)
# @param verb Verb name, for the fallback notification
# @param resolve `function(data)` running tidyselect on a data frame
# @param plain Whether `plain_selection_loc()` may answer the selection without
#   tidyselect (only for `eval_select()`'s own contract: select())
# @return list(loc = positions as `resolve()` returns them, data = the input
#   to build the result on: `.data`, or its computed table when the real data
#   had to be read, so the plan does not run twice)
# @keywords internal
resolve_selection <- function(.data, schema, quos, verb, resolve, plain = FALSE) {
  if (plain && !is.null(loc <- plain_selection_loc(quos, schema$names))) {
    return(list(loc = loc, data = .data))
  }
  groups <- .subset2(.data, "groups")
  if (selection_is_type_only(quos, NULL, schema$names)) {
    return(list(loc = resolve(tidyselect_proxy(schema, groups)), data = .data))
  }

  if (anyDuplicated(schema$names) > 0L) {
    stop("`", verb, "()` cannot evaluate this selection on a table with ",
         "duplicated column names: it needs the column values (a `where()` ",
         "predicate that is not a base type test, or a function call).", call. = FALSE)
  }
  cuplyr_fallback_notify(verb, "a selection that may read column values (resolved on the data)")
  data <- if (has_pending_ops(.data)) compute(.data) else .data
  frame <- collect_frame(data)
  if (length(groups) > 0L) {
    frame <- dplyr::grouped_df(frame, groups)
  }
  list(loc = resolve(frame), data = data)
}

# A select node keeping the columns at positions `loc` of a table named
# `names`: by name, or by position when the names are duplicated (where a name
# is ambiguous, and push_op() refuses a by-name select).
# @keywords internal
ast_select_at <- function(input, names, loc) {
  ast_select(input, names[loc], positions = if (anyDuplicated(names) > 0L) unname(loc))
}

# The select node for `loc` (named positions into `names`: the kept columns
# in order, named by their output names), plus a rename on top when
# tidyselect renamed a column. A table with duplicated names, or a selection
# repeating a source column (`select(d, a = x, b = x)`), is addressed by
# position end to end: the select by `loc`, the rename by the output slots.
# The intermediate select of a repeated source column carries duplicated names,
# which a grouped table must not; it is built on the ungrouped table, and the
# caller puts the groups (and the fixed keys of `slice(.preserve = TRUE)`) back
# by position (set_groups_by_position()).
# @keywords internal
push_select_rename <- function(.data, names, loc) {
  selected_cols <- names[loc]
  output_names <- names(loc)
  dup <- anyDuplicated(names) > 0L || anyDuplicated(selected_cols) > 0L
  if (dup) {
    .data <- ungrouped(.data)
  }
  select_node <- ast_select(input_node(.data), selected_cols,
                            positions = if (dup) unname(loc))
  if (identical(selected_cols, output_names)) {
    return(push_op(.data, select_node))
  }
  changed <- selected_cols != output_names
  if (dup || length(current_schema(.data)$factor_levels) > 0L ||
      length(current_schema(.data)$tz) > 0L) {
    # two steps, so the rename's factor levels (and POSIXct zones) are read off
    # the select's result (by position) rather than the original table's: a
    # combined node would give `select(x = y)` the levels of the dropped
    # factor `x`
    selected <- push_op(.data, select_node)
    return(push_op(selected, ast_rename(input_node(selected), selected_cols[changed],
                                        output_names[changed],
                                        positions = if (dup) which(changed))))
  }
  node <- ast_rename(select_node, selected_cols[changed], output_names[changed])
  push_op(.data, node)
}

# dplyr's ensure_group_vars(): grouping columns the selection leaves out are
# put in front, with a message, unless an output name already uses theirs.
# @param vars Named integer positions from eval_select()
# @return `vars` with the added grouping columns prepended
# @keywords internal
select_ensure_group_vars <- function(vars, names, groups) {
  group_loc <- match(groups, names)
  missing <- set_diff(group_loc, vars)
  if (length(missing) == 0L) {
    return(vars)
  }
  added <- stats::setNames(missing, names[missing])
  added <- added[!names(added) %in% names(vars)]
  if (length(added) > 0L) {
    rlang::inform(paste0(
      "Adding missing grouping variables: ",
      paste0("`", names(added), "`", collapse = ", ")
    ))
  }
  c(added, vars)
}

# Groups follow their column by position (dplyr regroups by position): each
# original group column keeps its group under the name it now has, and a
# column renamed onto a group's name does not inherit the group. The fixed
# keys of `slice(.preserve = TRUE)` (see fixed_keys_of()) follow the same
# mapping, and are kept only when every group column survives.
# @param out The pushed select/relocate result
# @param before The input table
# @param loc Named positions (into `names`) of the output columns, in order
# @keywords internal
set_groups_by_position <- function(out, before, loc, names) {
  pos <- match(match(before$groups, names), loc)
  groups <- names(loc)[pos[!is.na(pos)]]
  keys <- fixed_keys_of(before)
  if (!is.null(keys) && !anyNA(pos)) {
    names(keys) <- groups
    out$fixed_keys <- keys
  } else if (!is.null(out[["fixed_keys"]])) {
    out <- drop_fixed_keys(out)
  }
  out$groups <- groups
  out$group_drop <- if (length(groups) > 0L) before$group_drop %||% TRUE else TRUE
  # a rowwise table stays rowwise (F49)
  if (is_rowwise_gpu(before)) out$rowwise <- TRUE
  out
}
