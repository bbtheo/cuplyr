# dplyr's error context for mutate()/transmute() (#30).
#
# dplyr evaluates every dot inside `withCallingHandlers(error = )`
# (mutate_cols(), R/mutate.R) and re-raises what a dot throws as
#
#   abort(c(i = "In argument: `<label>`.", i = "In group 1: `<keys>`."),
#         class = "dplyr:::mutate_error", parent = <the error>, call = <verb>)
#
# This file is that handler for the native route. A verb call opens a context
# (mutate_errors_open()), runs inside ONE `withCallingHandlers(error = )`, and
# parse_mutate_dots()/expand_across_dots() say which dot they are on
# (`ctx$dot`: the index among the user's dots, NULL between dots). An error
# raised while a dot is current is re-raised the way dplyr does; any other error
# (lowering, a CPU fallback, the rowwise rerun, `.by`/`.keep` checks) passes
# through untouched, because no dot is current then and dplyr doesn't wrap those
# either. The helpers are parametrised by the error class so summarise() could
# reuse them.
#
# Nothing here runs on the success path but the open/close and a few field
# assignments per dot: labels and group keys are worked out in the handler.

# The innermost open context (an environment, see mutate_errors_open()), or NULL.
mutate_error_state <- new.env(parent = emptyenv())

# Open the context of one verb call. `dots` are the user's quosures (before any
# rewrite), `call` the verb method's frame (what dplyr reports as the error's
# call). Returns the context; close it with mutate_errors_close() (a stack: a
# mutate() inside a dot restores the outer one).
# @keywords internal
mutate_errors_open <- function(dots, call, error_class = "dplyr:::mutate_error") {
  ctx <- new.env(parent = emptyenv())
  ctx$dots <- dots
  ctx$call <- call
  ctx$error_class <- error_class
  ctx$prev <- mutate_error_state$cur
  mutate_error_state$cur <- ctx
  ctx
}

# @keywords internal
mutate_errors_close <- function(ctx) {
  mutate_error_state$cur <- ctx$prev
  invisible(NULL)
}

# Record the groups of the call (`.data` is the table before cur_group_id()'s
# substitution; `by` is TRUE for `.by =`). Called once the verb has resolved them,
# so an error before that (resolve_by()'s) is not wrapped with a group.
# @keywords internal
mutate_errors_set_groups <- function(.data, groups, by) {
  ctx <- mutate_error_state$cur
  if (!is.null(ctx)) {
    ctx$data <- .data
    ctx$groups <- groups
    ctx$by <- by
  }
  invisible(NULL)
}

# Conditions that are never wrapped: dplyr's own (a CPU route raised it) and
# cuplyr's classed errors (GPU, out of memory, `cuplyr.fallback = "error"`, ...),
# which are not what the user's dot threw.
mutate_error_passes <- function(cnd, error_class) {
  inherits(cnd, error_class) || any(startsWith(class(cnd), "cuplyr_"))
}

# The `error =` handler of a verb call (`ctx` from mutate_errors_open()).
# @keywords internal
mutate_errors_handle <- function(ctx, cnd) {
  i <- ctx$dot
  if (is.null(i) || mutate_error_passes(cnd, ctx$error_class)) {
    return(invisible(NULL))
  }
  mutate_errors_rethrow(ctx, cnd, i)
}

# Raise `cnd` as dplyr would. Anything that goes wrong while building the
# context (a label, the group keys) leaves the user's own error as it is.
# @keywords internal
mutate_errors_rethrow <- function(ctx, cnd, i) {
  phase <- ctx$phase
  col <- ctx$col
  # (no dot is current from here on: nothing the handler runs is wrapped)
  ctx$dot <- NULL

  label <- tryCatch(mutate_error_label(ctx$dots, i), error = function(e) NULL)
  if (is.null(label)) {
    return(invisible(NULL))
  }
  bullets <- c(i = paste0("In argument: `", label, "`."))
  # A group is current only while a dot is evaluated, not while across() is
  # expanded (tidyselect, `.fns`).
  if (identical(phase, "eval") && length(ctx$groups) > 0L) {
    group <- tryCatch(mutate_error_group_bullet(ctx), error = function(e) NULL)
    if (!is.null(group)) {
      bullets <- c(bullets, i = group)
    }
  }
  parent <- cnd
  if (identical(phase, "eval") && !is.null(col)) {
    # an across() expansion that failed while it was computed
    parent <- rlang::error_cnd(
      message = paste0("Can't compute column `", col, "`."),
      call = call("across"),
      parent = cnd
    )
  }
  rlang::abort(bullets, class = ctx$error_class, parent = parent, call = ctx$call)
}

# dplyr's error_label_named(name, expr) of the user's dot `i`.
# @keywords internal
mutate_error_label <- function(dots, i) {
  name <- names(dots)[i]
  label <- mutate_error_expr_label(rlang::quo_get_expr(dots[[i]]))
  if (length(name) == 0L || is.na(name) || !nzchar(name)) label else paste0(name, " = ", label)
}

# dplyr's expr_as_label(): as_label() strips `.data$x` / `.data[["x"]]`, so
# those are deparsed; rlang's infix labelling is off.
# @keywords internal
mutate_error_expr_label <- function(expr) {
  if (rlang::is_call(expr, c("[[", "$")) && identical(expr[[2L]], quote(.data))) {
    deparse(expr)[[1L]]
  } else {
    rlang::with_options("rlang:::use_as_label_infix" = FALSE, rlang::as_label(expr))
  }
}

# "In group 1: `g = 1`." for the first group the call evaluates, or NULL when it
# has no rows (dplyr: a mask of size 0 is not grouped). A grouped table's first
# group is the first of its sorted keys; `.by` groups are in order of first
# appearance, so the first is the key of row 1.
# @keywords internal
mutate_error_group_bullet <- function(ctx) {
  data <- ctx$data
  groups <- ctx$groups
  if (isTRUE(ctx$by)) {
    first <- tibble::as_tibble(dplyr::collect(dplyr::slice_head(data, n = 1L)))
    if (nrow(first) == 0L) {
      return(NULL)
    }
    keys <- first[1L, groups, drop = FALSE]
  } else {
    keys <- dplyr::group_keys(data)
    if (nrow(keys) == 0L) {
      return(NULL)
    }
    keys <- keys[1L, , drop = FALSE]
  }
  paste0("In group 1: ", mutate_error_group_details(keys), ".")
}

# dplyr's group_labels_details(): "`a = 1, b = 2`" (cli's code style).
# @keywords internal
mutate_error_group_details <- function(keys) {
  keys <- vapply(keys, pillar::format_glimpse, character(1))
  labels <- paste0(names(keys), " = ", keys)
  labels <- cli::cli_vec(labels, style = list("vec-last" = ", ", "vec-sep2" = ", "))
  cli::format_inline("{.code {labels}}")
}
