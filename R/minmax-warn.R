# F124: min()/max() over no non-missing value, as dplyr reports it.
#
# base R warns "no non-missing arguments to min; returning Inf" for each empty
# vector, and dplyr folds every warning of a verb call into ONE condition
# ("There was 1 warning in `summarise()`." / "There were N warnings in
# `mutate()`.", the first warning in full, a pointer to
# `last_dplyr_warnings()` for the rest). The native paths know the hits by
# group, not the R calls, so they count them and signal that one condition
# here, built with the calls dplyr makes (rlang::warning_cnd(),
# cli::format_warning(), rlang::warn()), so the message is identical.
#
# A hit is (dot, group, call). The first warning is the first in dplyr's
# evaluation order: summarise()/reframe()/mutate() evaluate dot by dot, each
# over all groups in group order; filter() group by group, all dots per group.

# Package state of the call being built (never held past the verb): the dots a
# summarise()/reframe() aggregation came from (`verb`, `dots`, `ids`, set by
# minmax_ctx_open()), and the sink that gathers hits instead of signalling
# them (`collector`, minmax_collect()).
minmax_state <- new.env(parent = emptyenv())

# Name the dots of the verb running, so the aggregations built for them can
# say which dot a hit belongs to (minmax_attach_dots()). `dots` is the quosure
# list the aggregations are parsed from, one per aggregation in order; `ids`
# the user's dot each one belongs to (reframe() rewrites a dot into several).
# @return the previous state, for minmax_ctx_close()
# @keywords internal
minmax_ctx_open <- function(verb, dots, ids = seq_along(dots)) {
  prev <- list(verb = minmax_state$verb, dots = minmax_state$dots, ids = minmax_state$ids)
  minmax_state$verb <- verb
  minmax_state$dots <- dots
  minmax_state$ids <- ids
  prev
}

# @keywords internal
minmax_ctx_close <- function(prev) {
  minmax_state$verb <- prev$verb
  minmax_state$dots <- prev$dots
  minmax_state$ids <- prev$ids
  invisible()
}

# Put the dot (id, name, expression, verb) on every min()/max() aggregation:
# the expression, not the quosure, so a plan does not hold the environment.
# Only the aggregations that can hit an empty group (`identity`) pay for it.
# @keywords internal
minmax_attach_dots <- function(aggregations) {
  dots <- minmax_state$dots
  n <- length(aggregations)
  if (is.null(dots) || !(n == length(dots) || n == length(dots) + 1L)) {
    return(aggregations)
  }
  for (i in seq_len(min(n, length(dots)))) {
    if (isTRUE(.subset2(.subset2(aggregations, i), "identity"))) {
      aggregations[[i]]$dot <- list(
        id = minmax_state$ids[[i]], name = rlang::names2(dots)[[i]],
        expr = rlang::quo_get_expr(dots[[i]]), verb = minmax_state$verb
      )
    }
  }
  aggregations
}

# Run `expr`, collecting the hits lower_summarise() finds instead of signalling
# them (summarise_with_empty_groups() merges them with the empty groups').
# @return `list(value, items)`
# @keywords internal
minmax_collect <- function(expr) {
  prev <- minmax_state$collector
  sink <- new.env(parent = emptyenv())
  sink$items <- list()
  minmax_state$collector <- sink
  on.exit(minmax_state$collector <- prev)
  value <- expr
  list(value = value, items = sink$items)
}

# The label dplyr gives a dot's expression (`expr_as_label()`): the deparsed
# call for `.data$x` / `.data[["x"]]`, rlang's label for anything else.
# @keywords internal
minmax_expr_label <- function(expr) {
  if (is.call(expr) && (identical(expr[[1L]], quote(`[[`)) || identical(expr[[1L]], quote(`$`))) &&
      identical(expr[[2L]], quote(.data))) {
    return(deparse(expr)[[1L]])
  }
  opts <- options("rlang:::use_as_label_infix" = FALSE)
  on.exit(options(opts))
  rlang::as_label(expr)
}

# `error_label_named()`: `name = expr` for a named dot, `expr` for an unnamed
# one. A name equal to the expression's own label is the auto name cuplyr gave
# an unnamed dot.
# @keywords internal
minmax_dot_label <- function(name, expr) {
  label <- minmax_expr_label(expr)
  if (is.null(name) || is.na(name) || !nzchar(name) || identical(name, label)) label else paste0(name, " = ", label)
}

# `cur_group_label()` of a grouped mask: "group <id>: <key> = <value>, ...",
# values through pillar::format_glimpse() like dplyr.
# @param keys One-row data frame of the group's key values
# @keywords internal
minmax_group_label <- function(id, keys) {
  vals <- vapply(keys, pillar::format_glimpse, character(1))
  labels <- cli::cli_vec(paste0(names(vals), " = ", vals),
                         style = list("vec-last" = ", ", "vec-sep2" = ", "))
  paste0("group ", id, ": ", cli::format_inline("{.code {labels}}"))
}

# The warning min()/max() raised in R for an empty input of `type`: the call it
# came from (the Summary methods of Date and POSIXct reach min.default()) and
# its text.
# @keywords internal
minmax_cause <- function(op, type) {
  type <- unname(type %||% "")
  ts <- type %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")
  list(
    call = rlang::call2(if (ts) paste0(op, ".default") else op, quote(x)),
    msg = if (identical(type, "STRING")) {
      "no non-missing arguments, returning NA"
    } else {
      paste0("no non-missing arguments to ", op, "; returning ", if (identical(op, "min")) "Inf" else "-Inf")
    }
  )
}

# Signal dplyr's one condition for `n` hits in `verb`, the first being a hit
# of dot (`name`, `expr`) by call `op` over an input of `type`, in group `id`
# with key values `keys` (`id` NA: no group line, i.e. ungrouped or no rows).
# @keywords internal
minmax_signal <- function(verb, n, name, expr, op, type, id = NA_integer_, keys = NULL) {
  cause <- minmax_cause(op, type)
  bullets <- c(
    i = paste0("In argument: `", minmax_dot_label(name, expr), "`."),
    if (!is.na(id)) c(i = paste0("In ", minmax_group_label(id, keys), "."))
  )
  first <- rlang::warning_cnd(
    message = bullets,
    parent = simpleWarning(cause$msg, call = cause$call),
    call = NULL
  )
  call <- rlang::format_error_call(rlang::call2(verb))
  msg <- paste(c(
    cli::format_warning(c(
      "There {cli::qty(n)} {?was/were} {n} warning{?s} in {call}.",
      if (n > 1) "The first warning was:"
    )),
    rlang::cnd_message(first),
    if (n > 1) {
      cli::format_warning(c(
        i = "Run {.run dplyr::last_dplyr_warnings()} to see the {n - 1} remaining warning{?s}."
      ))
    }
  ), collapse = "\n")
  rlang::warn(msg, use_cli_format = FALSE)
}

# The upload rule's warning for the +-Inf dplyr has in a Date/POSIXct and the
# GPU cannot hold: `cols` and `counts` name the output columns and their hits.
# @keywords internal
minmax_warn_infinite_timestamp <- function(cols, counts) {
  listed <- paste0("`", cols, "` (", format(counts, scientific = FALSE, trim = TRUE),
                   ifelse(counts == 1, " value)", " values)"), collapse = ", ")
  rlang::warn(
    paste0("A GPU Date or POSIXct cannot hold +-Inf: replaced with NA in ", listed, "."),
    class = c("cuplyr_warning_infinite_timestamp", "cuplyr_warning_timestamp_unrepresentable")
  )
}

# Signal the folded warning of a summarise()/reframe() whose aggregations hit
# empty groups. `records` has one entry per aggregation with hits:
# `list(dot, op, type, col, n, id, keys)`: the aggregation's dot, call, input
# type and output column, its hit count, and its first hit group (`id` NA for
# none; `keys` a function giving its key values, called only when needed).
# @keywords internal
minmax_emit <- function(records) {
  if (length(records) == 0L) {
    return(invisible())
  }
  n <- sum(vapply(records, `[[`, numeric(1), "n"))
  # first warning: the first dot, and within a dot (reframe's several
  # aggregations) the first group, then the first aggregation
  dot_id <- vapply(records, function(r) r$dot$id, numeric(1))
  grp <- vapply(records, function(r) if (is.na(r$id)) 0 else r$id, numeric(1))
  first <- records[[order(dot_id, grp)[[1L]]]]
  keys <- if (!is.na(first$id)) first$keys()
  minmax_signal(first$dot$verb %||% "summarise", n, first$dot$name, first$dot$expr, first$op,
                first$type, first$id, keys)
  ts <- vapply(records, function(r) unname(r$type %||% "") %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS"),
               logical(1))
  if (any(ts)) {
    minmax_warn_infinite_timestamp(
      vapply(records[ts], `[[`, character(1), "col"),
      vapply(records[ts], `[[`, numeric(1), "n")
    )
  }
  invisible()
}

# The records of one gpu_summarise() result (lower_summarise()), from its
# `identity_rows` report: per aggregation the 1-based result rows that hit (an
# NA row is the evaluation dplyr makes on a zero-row grouped input). `rank` is
# the dplyr group order of the result rows when it is not their order (the
# `.by` desugar leaves them unsorted), `key_at(row)` the key values of a result
# row, `grouped` whether the call has group lines at all.
# @keywords internal
minmax_summarise_records <- function(aggs, hits, rank, key_at, grouped) {
  records <- list()
  for (i in which(lengths(hits) > 0L)) {
    agg <- .subset2(aggs, i)
    rows <- hits[[i]]
    rows <- rows[!is.na(rows)]
    first_row <- NA_integer_
    id <- NA_integer_
    if (grouped && length(rows) > 0L) {
      ids <- if (is.null(rank)) rows else rank[rows]
      j <- which.min(ids)
      first_row <- rows[[j]]
      id <- as.integer(ids[[j]])
    }
    records[[length(records) + 1L]] <- local({
      row <- first_row
      list(
        dot = agg$dot %||% list(id = i, name = agg$output_col, expr = rlang::call2(agg$fn, as.name(agg$input_col)),
                                verb = "summarise"),
        op = agg$fn, type = agg$input_type, col = agg$output_col,
        n = length(hits[[i]]), id = id, keys = function() key_at(row)
      )
    })
  }
  records
}

# -----------------------------------------------------------------------------
# Window min()/max() in mutate() and filter()
# -----------------------------------------------------------------------------
#
# An empty group is a hit there too: a zero-row table (dplyr evaluates each dot
# once on the empty input), or a group of a `.drop = FALSE` / fixed-key grouping
# that no row belongs to. dplyr's value for an integer or logical min()/max()
# over one is a double (`Inf`), and vctrs makes the WHOLE column double, so the
# type is the data's to decide: the verb reads the row count (and the empty
# groups) before the dots are parsed (minmax_window_facts()), and the parse
# hook of min()/max() (ir_parse_minmax()) casts an INT32/BOOL8 argument to
# FLOAT64 when there is a hit, so the node, later operators, later dots and a
# lazy schema all say double.

# TRUE iff a dot mentions min() or max() (any call, any depth).
# @keywords internal
minmax_dots_mention <- function(dots) {
  for (d in dots) {
    if (any(c("min", "max") %in% all.names(rlang::quo_get_expr(d)))) {
      return(TRUE)
    }
  }
  FALSE
}

# The row count of a lazy table the pending steps cannot change (mutate,
# select, rename, window, arrange: a walk of the plan), else NA.
# @keywords internal
minmax_lazy_rows <- function(.data) {
  node <- .subset2(.data, "lazy_ops")
  while (inherits(node, "ast_node")) {
    if (!node$type %in% c("source", "mutate", "select", "rename", "window", "arrange")) {
      return(NA_integer_)
    }
    node <- node$input
  }
  base_dim(.data)[[1L]]
}

# Does the grouping keep empty groups (summarise_wants_empty_groups())?
# `groups` are the verb's: a `.by` call has none that do.
# @keywords internal
minmax_wants_empty <- function(.data, by_given) {
  !by_given && !is_rowwise_gpu(.data) && summarise_wants_empty_groups(.data)
}

# What the verb knows about empty groups, from the data: `E`, the number of
# empty groups each min()/max() call meets (a zero-row input is one evaluation
# unless the grouping lists groups), `n_in` the rows, and the first empty
# group's position `id` and `keys` (NA/NULL: no group line, as dplyr's mask is
# ungrouped over zero rows). `rows` is the known row count.
# @keywords internal
minmax_window_facts <- function(.data, rows, wants_empty) {
  if (wants_empty) {
    # (a lazy table with steps pending is computed for it: the empty groups
    # are the data's)
    if (has_pending_ops(.data)) .data <- compute(.data)
    gd <- compute_group_data(.data)
    n_in <- base_dim(.data)[[1L]]
    empty <- lengths(gd$.rows) == 0L
    E <- sum(empty)
    if (E == 0L && n_in == 0L) E <- 1L
    id <- NA_integer_
    keys <- NULL
    if (E > 0L && n_in > 0L) {
      id <- which(empty)[[1L]]
      keys <- gd[id, .subset2(.data, "groups"), drop = FALSE]
    }
    return(list(E = E, n_in = n_in, id = id, keys = keys))
  }
  list(E = as.integer(identical(as.integer(rows), 0L)), n_in = rows, id = NA_integer_, keys = NULL)
}

# The window min()/max() state of the verb call being run: set before its dots
# are parsed, read by ir_parse_minmax() (`double`), parse_mutate_dots() /
# filter_run() (`records`), and the emit. `carry`: the plan, not the verb,
# signals the warning (a lazy table: it warns when it runs).
# @keywords internal
minmax_win_open <- function(verb, facts, carry = FALSE, typed = TRUE) {
  prev <- minmax_state$win
  minmax_state$win <- list2env(list(
    verb = verb, facts = facts, carry = carry,
    double = typed && !is.null(facts) && facts$E > 0L, records = list()
  ), parent = emptyenv())
  prev
}

# @keywords internal
minmax_win_close <- function(prev) {
  minmax_state$win <- prev
  invisible()
}

# The min()/max() calls of a parsed expression in R's evaluation order
# (arguments left to right, then the call): `list(op, type)` each, `type` the
# GPU type of the argument.
# @keywords internal
minmax_ir_calls <- function(ir, schema) {
  calls <- list()
  walk <- function(node) {
    if (!inherits(node, "cuplyr_ir") || !identical(node$kind, "call")) {
      return(invisible())
    }
    for (a in node$args) walk(a)
    if (node$op %in% c("min", "max") && length(node$args) == 1L) {
      type <- tryCatch(unname(ir_infer_type(node$args[[1L]], schema)), error = function(e) NA_character_)
      calls[[length(calls) + 1L]] <<- list(op = node$op, type = type)
    }
  }
  walk(ir)
  calls
}

# Note a parsed dot's min()/max() calls (a no-op outside a verb that counts them).
# @keywords internal
minmax_record_dot <- function(name, expr, ir, schema) {
  w <- minmax_state$win
  if (is.null(w) || is.null(ir)) {
    return(invisible())
  }
  calls <- minmax_ir_calls(ir, schema)
  if (length(calls) > 0L) {
    w$records[[length(w$records) + 1L]] <- list(name = name, expr = expr, calls = calls)
  }
  invisible()
}

# The warning every window min()/max() call of the verb meets for the empty
# groups: `n` = calls x empty groups, the first being the first call of the
# first dot that has one, in the first empty group. NULL when there is nothing
# to say.
# @return `list(verb, n, name, expr, op, type, id, keys)` or NULL
# @keywords internal
minmax_win_warning <- function(w, per_group = w$facts$E) {
  if (is.null(w) || length(w$records) == 0L || per_group == 0L) {
    return(NULL)
  }
  first <- w$records[[1L]]
  n <- per_group * sum(vapply(w$records, function(r) length(r$calls), integer(1)))
  list(verb = w$verb, n = n, name = first$name, expr = first$expr,
       op = first$calls[[1L]]$op, type = first$calls[[1L]]$type,
       id = w$facts$id, keys = w$facts$keys)
}

# Signal the verb's window min()/max() warning now (the rows are known).
# @keywords internal
minmax_win_emit <- function() {
  w <- minmax_state$win
  if (is.null(w) || isTRUE(w$carry)) {
    return(invisible())
  }
  p <- minmax_win_warning(w)
  if (!is.null(p)) {
    minmax_signal(p$verb, p$n, p$name, p$expr, p$op, p$type, p$id, p$keys)
  }
  invisible()
}

# The warning a lazy plan carries when it cannot know its row count: it fires
# when the node that holds it meets a zero-row input (one evaluation, no group
# line). NULL when the verb has none or is not in that case.
# @keywords internal
minmax_win_lazy_prep <- function() {
  w <- minmax_state$win
  if (is.null(w) || !isTRUE(w$carry)) {
    return(NULL)
  }
  p <- minmax_win_warning(w, per_group = 1L)
  if (!is.null(p)) {
    p$id <- NA_integer_
    p$keys <- NULL
  }
  p
}

# Fire a prepared warning (minmax_win_lazy_prep()).
# @keywords internal
minmax_signal_prepared <- function(p) {
  minmax_signal(p$verb, p$n, p$name, p$expr, p$op, p$type)
}

# How the verb knows its row count and empty groups: `facts` (minmax_window_facts())
# when it can read them now (an eager table, a lazy one with nothing pending or
# only row-preserving steps, a grouping that keeps empty groups), else NULL. A
# lazy table without empty-group grouping only needs the row count for the
# TYPE; its warning is the plan's (`carry`), signalled whenever it runs on an
# empty input (F80: a plan warns when it runs).
# @keywords internal
minmax_window_setup <- function(.data, by_given) {
  lazy <- identical(.subset2(.data, "exec_mode"), "lazy")
  pending <- lazy && has_pending_ops(.data)
  rows <- if (pending) minmax_lazy_rows(.data) else base_dim(.data)[[1L]]
  wants_empty <- minmax_wants_empty(.data, by_given)
  facts <- if (!is.na(rows) || wants_empty) minmax_window_facts(.data, rows, wants_empty)
  list(facts = facts, carry = lazy && !wants_empty)
}

# Does a lazy mutate() have to run its pending steps now to type its window
# min()/max() (the type of an integer one is the data's to decide)? The reason
# for the plan break, or NULL. Only a lazy table with steps pending can need it:
# a grouping that keeps empty groups (they are the data's), or an integer
# min()/max() over a row count the steps can change. A double, string or
# timestamp one only needs a warning, which the plan carries.
# @keywords internal
minmax_window_run_now <- function(.data, dots, by_given) {
  if (!identical(.subset2(.data, "exec_mode"), "lazy") || !has_pending_ops(.data)) {
    return(NULL)
  }
  if (minmax_wants_empty(.data, by_given)) {
    return("min()/max() over groups that may be empty (result type depends on the data)")
  }
  if (is.na(minmax_lazy_rows(.data)) && minmax_dots_int_call(dots, current_schema(.data))) {
    return("an integer min()/max() (result type depends on the data)")
  }
  NULL
}

# Does a dot call min()/max() on an argument that is, or may be, an INT32 or
# logical? (a bare column by its type; any other argument by its inferred type,
# NA when it cannot be told)
# @keywords internal
minmax_dots_int_call <- function(dots, schema) {
  found <- FALSE
  walk <- function(e, env) {
    if (found || !is.call(e)) return(invisible())
    head <- e[[1L]]
    if (is.call(head) && length(head) == 3L && identical(head[[1L]], quote(`::`))) head <- head[[3L]]
    if (is.symbol(head) && as.character(head) %in% c("min", "max") && length(e) == 2L) {
      arg <- e[[2L]]
      type <- if (is.symbol(arg) && as.character(arg) %in% schema$names) {
        unname(schema$types[match(as.character(arg), schema$names)])
      } else {
        tryCatch(unname(ir_infer_type(ir_parse_quo(rlang::new_quosure(arg, env), schema), schema)),
                 error = function(e) NA_character_)
      }
      if (is.na(type) || type %in% c("INT32", "BOOL8")) found <<- TRUE
    }
    for (a in as.list(e)[-1L]) walk(a, env)
  }
  for (d in dots) walk(rlang::quo_get_expr(d), rlang::quo_get_env(d))
  found
}
