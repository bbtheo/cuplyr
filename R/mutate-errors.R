# dplyr's error context for mutate()/transmute() (#30).
#
# dplyr evaluates every dot inside `withCallingHandlers(error = )`
# (mutate_cols(), R/mutate.R) and re-raises what a dot throws as
#
#   abort(c(i = "In argument: `<label>`.", i = "In group 1: `<keys>`."),
#         class = "dplyr:::mutate_error", parent = <the error>, call = <verb>)
#
# This file is that handler for the native route, built the way dplyr builds
# its own (dplyr_error_handler(i_sym = "i")): the two loops over the dots,
# parse_mutate_dots() ("eval": a dot is parsed and evaluated) and
# expand_across_dots() ("expand": across()/if_any()/if_all() are resolved), run
# inside ONE `withCallingHandlers(error = )` each, and its handler reads the
# loop's index `i` from the loop's own frame when an error comes. Nothing is
# recorded per call or per dot: the user's dots, the table before cur_group_id()'s
# substitution and the groups are read from the frame of mutate_run() /
# transmute_run() (the caller of the loop's function), the verb's call from the
# frame that called it. An error raised anywhere else (lowering, a CPU fallback,
# `.by`/`.keep`/relocate) is not wrapped, as dplyr doesn't wrap those either.
#
# What dplyr does per kind of dot, which `mutate_dot_error()` reproduces:
# - a plain dot is evaluated once per group: a group bullet, the error as parent;
# - an unnamed `across()` with no `...` is expanded before the groups run
#   (expand_across()): an error in the expansion (`.cols`, `.fns`) has no group
#   bullet; one that computes a column has the group bullet and an extra layer,
#   "Can't compute column `<col>`." (call `across()`), between it and the error;
# - any other across() (extra arguments), and if_any()/if_all() in mutate(), run
#   per group as functions: the group bullet always; the "Can't compute column"
#   layer for an error while a column is computed (its call is the one as the
#   user wrote it), none for a selection error.
#
# Nothing here runs on the success path but the withCallingHandlers() call.
#
# A CPU route runs dplyr's own mutate()/transmute() from inside the verb, so dplyr
# reports ITS call (`dplyr::mutate()`), not the user's: cpu_verb_call() gives the
# errors that name it the verb's name (`mutate()`) and leaves everything else of
# the condition (class, parent, message) as dplyr made it.

# Conditions the loops never wrap: cuplyr's own runtime errors (GPU, memory, a
# fallback the option turned into an error, an internal error, a column class
# with no GPU form, an integer sum overflow a caller reruns on the CPU), which
# are not what the user's dot threw.
mutate_error_passthrough <- c(
  "cuplyr_error_gpu", "cuplyr_error_gpu_oom", "cuplyr_error_fallback",
  "cuplyr_error_internal", "cuplyr_error_unsupported_class", "cuplyr_error_sum_overflow"
)

# The `error =` handler of a dot loop. Called from a closure defined in the
# loop's function, whose frame is its enclosure: `i` is the index of the dot the
# loop is on (of `provenance`'s expanded dots in the "eval" phase, of the user's
# in the "expand" phase), `names` the expanded dots' names, `verb` "mutate" or
# "transmute". Returns (the error goes on) unless it can raise dplyr's.
# @keywords internal
mutate_dot_error <- function(cnd, i, phase, provenance, names, verb) {
  if (inherits(cnd, mutate_error_passthrough)) {
    return(invisible(NULL))
  }
  # the loop's function, then mutate_run()/transmute_run(), then the verb's method
  loop <- parent.env(parent.frame())
  run <- rlang::eval_bare(quote(base::parent.frame()), loop)
  # (a rowwise call's native attempt abandons on any error: dplyr gives its own
  # from the rerun; a loop called from elsewhere has no user dots to name)
  user_dots <- get0("user_dots", run, inherits = FALSE)
  if (is.null(user_dots) || isTRUE(get0("rowwise", run, inherits = FALSE))) {
    return(invisible(NULL))
  }
  args <- tryCatch(
    mutate_dot_error_args(cnd, i, phase, provenance, names, verb, run, user_dots),
    error = function(e) NULL
  )
  if (is.null(args)) {
    return(invisible(NULL))
  }
  rlang::abort(
    args$bullets, class = "dplyr:::mutate_error", parent = args$parent,
    call = rlang::eval_bare(quote(base::parent.frame()), run)
  )
}

# What mutate_dot_error() raises: the bullets and the parent. Anything that
# goes wrong while it is worked out leaves the user's own error as it is.
# @keywords internal
mutate_dot_error_args <- function(cnd, i, phase, provenance, names, verb, run, user_dots) {
  eval_phase <- identical(phase, "eval")
  origin <- if (eval_phase && !is.null(provenance)) provenance$origin[i] else i
  kind <- if (eval_phase && !is.null(provenance)) provenance$kind[i] else "dot"
  quo <- user_dots[[origin]]
  expr <- rlang::quo_get_expr(quo)
  user_name <- names(user_dots)[origin]
  named <- length(user_name) > 0L && !is.na(user_name) && nzchar(user_name)

  # dplyr expands an unnamed across() without `...` before the groups run
  head <- call_head_name(expr)
  across_dot <- !named && identical(head, "across")
  expands <- across_dot && mutate_error_across_expands(expr)
  # (the call of the function the user wrote, for an error it reports itself)
  written <- if (is.call(expr)) as.call(list(expr[[1L]]))

  bullets <- c(i = paste0("In argument: `", mutate_error_label(user_name, expr), "`."))
  # A group is current only while a dot is evaluated, not while the across()
  # that dplyr expands first is resolved.
  if (!(identical(phase, "expand") && expands)) {
    group <- tryCatch(mutate_error_group_bullet(run, phase, verb), error = function(e) NULL)
    if (!is.null(group)) {
      bullets <- c(bullets, i = group)
    }
  }

  parent <- cnd
  # An error while a column is computed has the layer of across() / if_any()
  layer_call <- NULL
  col <- NULL
  if (eval_phase && identical(kind, "across")) {
    col <- provenance$col[i] %||% names[i]
    layer_call <- if (expands) call("across") else written
  } else if (eval_phase && identical(kind, "combined")) {
    col <- mutate_error_first_col(run, expr, quo, verb)
    layer_call <- written
  }
  if (!is.null(col) && !is.na(col) && nzchar(col)) {
    parent <- rlang::error_cnd(
      message = paste0("Can't compute column `", col, "`."),
      call = layer_call,
      parent = cnd
    )
  }
  list(bullets = bullets, parent = parent)
}

# TRUE iff dplyr's expand_across() expands this unnamed across() call: it has no
# `...` (an argument other than `.cols`, `.fns`, `.names`, `.unpack`) and no
# `.unpack`.
# @keywords internal
mutate_error_across_expands <- function(expr) {
  present <- names(as.list(rlang::call_match(expr, across_proto, defaults = FALSE))[-1L])
  all(present %in% c(".cols", ".fns", ".names"))
}

# dplyr's error_label_named(name, expr) of a user dot.
# @keywords internal
mutate_error_label <- function(name, expr) {
  label <- mutate_error_expr_label(expr)
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

# The column an if_any()/if_all() reports in "Can't compute column": dplyr
# computes the columns in order, so its first (the failing one unless the
# predicate depends on the data). NULL when it can't be told.
# @keywords internal
mutate_error_first_col <- function(run, expr, quo, verb) {
  proto <- if_any_all_proto
  present <- as.list(rlang::call_match(expr, proto, defaults = FALSE))[-1L]
  cols_expr <- if (".cols" %in% names(present)) present$.cols else default_across_cols(call_head_name(expr))
  data <- get0(".data", run, inherits = FALSE)
  cols <- resolve_across_cols(cols_expr, rlang::quo_get_env(quo), current_schema(data),
                              mutate_error_groups(run, verb)$groups)
  if (length(cols) > 0L) unname(cols)[1L]
}

# The groups of the call (`.by` ones included) and whether they came from `.by`,
# from the frame of mutate_run()/transmute_run().
# @keywords internal
mutate_error_groups <- function(run, verb) {
  if (identical(verb, "mutate")) {
    list(groups = get0("verb_groups", run, inherits = FALSE),
         by = isTRUE(get0("by_given", run, inherits = FALSE)))
  } else {
    list(groups = get0("eff_groups", run, inherits = FALSE), by = FALSE)
  }
}

# "In group 1: `g = 1`." for the first group the call evaluates, or NULL when it
# is not grouped or has no rows (dplyr: a mask of size 0 is not grouped). A
# grouped table's first group is the first of its sorted keys; `.by` groups are
# in order of first appearance, so the first is the key of row 1. The table is
# the one before cur_group_id()'s substitution (`fallback_data`, or `.data` while
# across() is still being expanded).
# @keywords internal
mutate_error_group_bullet <- function(run, phase, verb) {
  g <- mutate_error_groups(run, verb)
  if (length(g$groups) == 0L) {
    return(NULL)
  }
  data <- get0(if (identical(phase, "eval")) "fallback_data" else ".data", run, inherits = FALSE)
  if (isTRUE(g$by)) {
    first <- tibble::as_tibble(dplyr::collect(dplyr::slice_head(data, n = 1L)))
    if (nrow(first) == 0L) {
      return(NULL)
    }
    keys <- first[1L, g$groups, drop = FALSE]
  } else {
    gd <- dplyr::group_data(data)
    # (a table with no rows has no current group, also when it has empty groups)
    if (sum(lengths(gd$.rows)) == 0L) {
      return(NULL)
    }
    keys <- gd[1L, setdiff(names(gd), ".rows"), drop = FALSE]
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

# Run `expr`, a CPU route's `dplyr::mutate()`/`dplyr::transmute()` call, so that an
# error dplyr raises for the route's own frame (its `dplyr:::mutate_error`, or an
# argument error of the verb) names the user's verb, `verb`, as dplyr's does when
# the user calls it. Every error of a dot is wrapped by dplyr's own handler (below
# this one) into one whose call is the route's frame, so only that kind reaches
# here with a call to `dplyr::mutate()`/`dplyr::transmute()`. Nothing is added to
# the success path of a native call: only a CPU route pays for the handler.
# @keywords internal
cpu_verb_call <- function(expr, verb) {
  withCallingHandlers(expr, error = function(cnd) {
    call <- cnd$call
    if (is.call(call) &&
        (identical(call[[1L]], quote(dplyr::mutate)) || identical(call[[1L]], quote(dplyr::transmute)))) {
      cnd$call <- call(verb)
      stop(cnd)
    }
  })
}
