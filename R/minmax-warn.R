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

# Where the dot of a summarise()/reframe() aggregation is kept: `ir_cache` (one
# per summarise_impl() call, handed to every stage of it) holds `.minmax`, the
# user's dots the aggregations were built from (`user_dots`, one per
# aggregation in order, through `origin`; `ids` give the dot order: reframe()
# rewrites a dot into several), and the S3 method whose call the warning names
# (`method`, `default`), built from `.minmax_src` only when an aggregation can
# hit (minmax_dot_info()). Those aggregations carry a
# small `$dot` built from it (minmax_attach_dots()): the dot's name and
# expression (not the quosure, so a plan does not hold the environment), the
# verb's call head and `ctx`, the environment that routes the hits of
# summarise_with_empty_groups() to its collector (minmax_collect()). Nothing is
# package-global: a verb run while another one builds its dots (a gibble verb
# called inside a dot) cannot see or emit the outer verb's records.

# The user's dots (`user_dots`, quosures) behind the aggregations or dots of
# `dots` (their `provenance` says which: `origin`), for the label of a
# warning.
# @return a list with `user_dots`, `origin` and `ids`
# @keywords internal
minmax_dot_source <- function(user_dots, dots, method, default, ids = NULL) {
  origin <- attr(dots, "provenance")$origin
  if (length(origin) != length(dots) || length(user_dots) < max(origin, 0L)) {
    # (no provenance: the dots are the user's own)
    origin <- seq_along(dots)
    user_dots <- dots
  }
  list(user_dots = user_dots, origin = origin, ids = ids %||% seq_along(origin),
       method = method, default = default)
}

# The environment shared by the aggregations of one summarise_impl() call.
# @keywords internal
minmax_ctx <- function(ir_cache) {
  ctx <- .subset2(ir_cache, ".minmax_ctx")
  if (is.null(ctx)) {
    ctx <- new.env(parent = emptyenv())
    ir_cache$.minmax_ctx <- ctx
  }
  ctx
}

# The call head the verb was called by (`summarise`, `summarize`,
# `dplyr::mutate`): the head of the call of its dplyr generic, the frame under
# the S3 `method`'s (R rewrites the head of the method's own call to the method
# name): dplyr names the call as written. `frame` is the method's frame number
# as its helper saw it (sys.parent()), so the usual call reads two frames; a
# caller that is not the method (an internal rerun) is looked for on the stack.
# Pointer comparisons only: identical() of two different closures compares
# their bodies.
# @keywords internal
minmax_verb_head <- function(method, default, frame = NA_integer_) {
  if (is.na(frame) || frame < 2L || !rlang::is_reference(sys.function(frame), method)) {
    frame <- NA_integer_
    for (i in rev(seq_len(sys.nframe() - 1L))) {
      if (rlang::is_reference(sys.function(i), method)) {
        frame <- i
        break
      }
    }
    if (is.na(frame) || frame < 2L) {
      return(as.name(default))
    }
  }
  if (rlang::is_reference(sys.function(frame - 1L), getExportedValue("dplyr", default))) {
    head <- sys.call(frame - 1L)[[1L]]
    if (is.symbol(head) ||
        (is.call(head) && length(head) == 3L && identical(head[[1L]], quote(`::`)))) {
      return(head)
    }
  }
  as.name(default)
}

# The dot source of one summarise_impl() call (minmax_dot_source()), built the
# first time an aggregation needs it from what summarise_impl() left in
# `ir_cache$.minmax_src`: `list(source, user_dots, expanded, frame)`, `source`
# being reframe()'s own (or NULL) and `frame` the method's frame number.
# @keywords internal
minmax_dot_info <- function(ir_cache) {
  info <- .subset2(ir_cache, ".minmax")
  if (is.null(info)) {
    src <- .subset2(ir_cache, ".minmax_src")
    if (is.null(src)) {
      return(NULL)
    }
    info <- src[[1L]] %||% minmax_dot_source(src[[2L]], src[[3L]], summarise.tbl_gpu, "summarise")
    info$frame <- src[[4L]]
    ir_cache$.minmax <- info
  }
  info
}

# Put the dot (id, name, expression, call head) on every min()/max()
# aggregation that can hit an empty group (`identity`), when one can: an
# aggregation hits only an empty group (or an all-NA one under `na.rm = TRUE`),
# so with no `na.rm = TRUE` one a table whose row count is known and not zero
# (`data`: eager, or a lazy plan that keeps its rows, minmax_lazy_rows()) has
# nothing to label. Its groups are dplyr's, each with a row: a grouping that
# keeps empty ones is summarised by summarise_with_empty_groups(), whose empty
# slice has no rows. A count the source does not explain is a bug of the caller.
# @keywords internal
minmax_attach_dots <- function(aggregations, ir_cache, data = NULL) {
  n <- length(aggregations)
  hit <- FALSE
  na_rm <- FALSE
  for (i in seq_len(n)) {
    agg <- .subset2(aggregations, i)
    if (isTRUE(.subset2(agg, "identity"))) {
      hit <- TRUE
      if (isTRUE(.subset2(agg, "na_rm"))) {
        na_rm <- TRUE
        break
      }
    }
  }
  if (!hit || is.null(ir_cache)) {
    return(aggregations)
  }
  if (!na_rm && !is.null(data)) {
    rows <- minmax_lazy_rows(data)
    if (!is.na(rows) && rows > 0L) {
      return(aggregations)
    }
  }
  info <- minmax_dot_info(ir_cache)
  if (is.null(info)) {
    return(aggregations)
  }
  k <- length(info$origin)
  # (one more: the `.by` call's first-row position, last, summarise_by_desugar())
  if (!(n == k || (n == k + 1L && isTRUE(.subset2(.subset2(aggregations, n), "first_pos"))))) {
    stop("internal error: ", n, " aggregations for ", k, " dots (min()/max() warning labels)",
         call. = FALSE)
  }
  ctx <- minmax_ctx(ir_cache)
  head <- ctx$head
  if (is.null(head)) {
    head <- ctx$head <- minmax_verb_head(info$method, info$default, info$frame %||% NA_integer_)
  }
  names <- rlang::names2(info$user_dots)
  for (i in seq_len(k)) {
    if (isTRUE(.subset2(.subset2(aggregations, i), "identity"))) {
      o <- info$origin[[i]]
      aggregations[[i]]$dot <- list(
        id = info$ids[[i]], name = names[[o]],
        expr = rlang::quo_get_expr(info$user_dots[[o]]), head = head, ctx = ctx
      )
    }
  }
  aggregations
}

# Run `expr`, collecting the hits lower_summarise() finds instead of signalling
# them (summarise_with_empty_groups() merges them with the empty groups').
# The sink hangs on the call's own `ctx`, which only its aggregations carry.
# @return `list(value, items)`
# @keywords internal
minmax_collect <- function(ir_cache, expr) {
  ctx <- minmax_ctx(ir_cache)
  sink <- new.env(parent = emptyenv())
  sink$items <- list()
  ctx$sink <- sink
  on.exit(ctx$sink <- NULL)
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
# one; `name` is the user's own (`names2()` of their dots), "" if they gave none.
# @keywords internal
minmax_dot_label <- function(name, expr) {
  label <- minmax_expr_label(expr)
  if (is.null(name) || is.na(name) || !nzchar(name)) label else paste0(name, " = ", label)
}

# Is `op` (min/max) given to the across()/if_any()/if_all() call `expr` as a
# function VALUE (`across(c(a, b), max)`, `list(m = max)`)? dplyr calls it
# without a call of its own, so the warning it raises has none ("Caused by
# warning:"); a lambda (`~ max(.x)`) is a call of `max()`.
# @keywords internal
minmax_fn_value <- function(expr, op) {
  if (!is.call(expr) || !isTRUE(call_head_name(expr) %in% c("across", "if_any", "if_all"))) {
    return(FALSE)
  }
  matched <- tryCatch(rlang::call_match(expr, dplyr::across), error = function(e) NULL)
  fns <- matched[[".fns"]]
  is_op <- function(f) {
    if (is.symbol(f)) {
      return(identical(as.character(f), op))
    }
    if (is.call(f) && length(f) == 3L && identical(f[[1L]], quote(`::`))) {
      return(identical(as.character(f[[3L]]), op))
    }
    if (is.call(f) && isTRUE(call_head_name(f) %in% c("list", "c"))) {
      return(any(vapply(as.list(f)[-1L], is_op, logical(1))))
    }
    FALSE
  }
  is_op(fns)
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
# came from (the Summary methods of Date and POSIXct reach min.default(); none
# for a function value given to across()) and its text.
# @keywords internal
minmax_cause <- function(op, type, expr = NULL) {
  type <- unname(type %||% "")
  ts <- type %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")
  list(
    call = if (!minmax_fn_value(expr, op)) rlang::call2(if (ts) paste0(op, ".default") else op, quote(x)),
    msg = if (identical(type, "STRING")) {
      "no non-missing arguments, returning NA"
    } else {
      paste0("no non-missing arguments to ", op, "; returning ", if (identical(op, "min")) "Inf" else "-Inf")
    }
  )
}

# Signal dplyr's one condition for `n` hits in the verb called as `head`
# (`summarise`, `dplyr::mutate`, ...), the first being a hit of dot (`name`,
# `expr`) by call `op` over an input of `type`, in group `id` with key values
# `keys` (`id` NA: no group line, i.e. ungrouped or no rows).
# @keywords internal
minmax_signal <- function(head, n, name, expr, op, type, id = NA_integer_, keys = NULL) {
  cause <- minmax_cause(op, type, expr)
  bullets <- c(
    i = paste0("In argument: `", minmax_dot_label(name, expr), "`."),
    if (!is.na(id)) c(i = paste0("In ", minmax_group_label(id, keys), "."))
  )
  first <- rlang::warning_cnd(
    message = bullets,
    parent = simpleWarning(cause$msg, call = cause$call),
    call = NULL
  )
  call <- rlang::format_error_call(rlang::call2(head))
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
  minmax_signal(first$dot$head, n, first$dot$name, first$dot$expr, first$op,
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
                                head = quote(summarise)),
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
# type is the data's to decide. The verb reads the row count and the empty
# groups before its dots are parsed when it can (minmax_window_setup(): an eager
# table, a lazy one nothing or only row-preserving steps are pending on), and
# the parse hook of min()/max() (ir_parse_minmax()) then casts an INT32/BOOL8
# argument to FLOAT64 when there is a hit, so the node, later operators, later
# dots and a lazy schema all say double. When it cannot (a lazy table whose
# pending steps change the rows), mutate() parses once as if nothing were empty
# and looks at the PARSED calls: only an INT32/BOOL8 one needs the data's
# answer, and runs the plan now (minmax_win_int_call()). A double, string or
# timestamp call only has a warning to raise, which the plan carries.
#
# A call whose argument has one value per group (`max(n())`, `max(sum(x))`,
# ir_is_group_scalar()) never meets an empty vector: it neither warns nor turns
# double. A Date shifted by an integer min()/max() (`d + max(a)`) is a Date
# whatever that call's type, so the shift keeps the integer call (native) and
# the call's type decides nothing (`typed = FALSE` in its record).
#
# The state of one verb call is the environment minmax_win_open() makes, only
# when a hit is possible (the facts say an empty group, or cannot say yet); the
# labels and the call head are built when a warning is prepared. The verb hands
# it to parse_mutate_dots() and to its own emit; nothing is global.

# TRUE iff a dot mentions min() or max() (any call, any depth).
# @keywords internal
minmax_dots_mention <- function(dots) {
  dots_mention(dots, c("min", "max"))
}

# TRUE iff a dot CALLS min() or max() (a column named `max` is not a call).
# @keywords internal
minmax_dots_call <- function(dots) {
  calls <- function(x) {
    if (!is.call(x)) {
      return(FALSE)
    }
    head <- x[[1L]]
    if (is.call(head) && length(head) == 3L && identical(head[[1L]], quote(`::`))) {
      head <- head[[3L]]
    }
    if (is.symbol(head) && as.character(head) %in% c("min", "max")) {
      return(TRUE)
    }
    for (a in as.list(x)) {
      if (rlang::is_quosure(a)) a <- rlang::quo_get_expr(a)
      if (calls(a)) {
        return(TRUE)
      }
    }
    FALSE
  }
  for (dot in dots) {
    if (calls(rlang::quo_get_expr(dot))) {
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

# The empty groups of an eager (or lazy, nothing pending) table whose grouping
# keeps them (`.drop = FALSE` factor keys, fixed keys), counted before they
# are looked for: the number of groups is the keys' (their count, or the product
# of the levels of all-factor keys with no NA), the number with rows a hash-set
# count, so a table with no empty group pays one distinct count and no group
# index (no sort, no permutation download). Only an empty group makes it match
# the observed keys (k rows) against all of them.
# @return `list(E, id, keys)`: the number of empty groups, the first one's
#   1-based position in group order and its key values (NA/NULL when the table
#   has no rows: dplyr's mask is ungrouped there, no group line)
# @keywords internal
minmax_empty_groups <- function(.data, n_in) {
  none <- list(E = 0L, id = NA_integer_, keys = NULL)
  groups <- .subset2(.data, "groups")
  ptr <- .subset2(.data, "ptr")
  idx0 <- group_col_index0(.data)
  fixed <- fixed_keys_of(.data)
  levels <- current_schema(.data)$factor_levels
  n_keys <- if (!is.null(fixed)) {
    nrow(fixed)
  } else if (all(groups %in% names(levels)) && !any(gpu_null_counts(ptr, idx0) > 0L)) {
    prod(lengths(levels[groups]))
  } else {
    NA_real_
  }
  if (!is.na(n_keys)) {
    n_obs <- if (n_in == 0L) 0L else distinct_group_count(.data)
    if (n_obs >= n_keys) {
      return(none)
    }
    if (n_in == 0L) {
      return(list(E = as.integer(n_keys), id = NA_integer_, keys = NULL))
    }
  }
  keys_schema <- group_keys_schema(.data)
  observed <- group_keys_frame(wrap_gpu_call("group_data", gpu_group_keys(ptr, idx0)), keys_schema)
  full <- fixed
  if (is.null(full)) {
    gd <- observed
    gd$.rows <- vctrs::new_list_of(rep(list(integer()), nrow(gd)), ptype = integer())
    full <- slice_group_keys_of(expand_dropped_groups(gd, groups, keys_schema$factor_levels))
  }
  empty <- which(is.na(vctrs::vec_match(full, observed[names(full)])))
  if (length(empty) == 0L) {
    return(none)
  }
  first <- empty[[1L]]
  list(E = length(empty), id = if (n_in > 0L) first else NA_integer_,
       keys = if (n_in > 0L) full[first, , drop = FALSE])
}

# What the verb knows about empty groups, from the data: `E`, the number of
# empty groups each min()/max() call meets (a zero-row input is one evaluation
# unless the grouping lists groups), `n_in` the rows, and the first empty
# group's position `id` and `keys` (NA/NULL: no group line). `rows` is the known
# row count. The table has no steps pending when its grouping keeps empty groups.
# @keywords internal
minmax_window_facts <- function(.data, rows, wants_empty) {
  if (!wants_empty) {
    return(list(E = as.integer(identical(as.integer(rows), 0L)), n_in = rows,
                id = NA_integer_, keys = NULL))
  }
  n_in <- base_dim(.data)[[1L]]
  g <- minmax_empty_groups(.data, n_in)
  E <- g$E
  if (E == 0L && n_in == 0L) E <- 1L
  list(E = E, n_in = n_in, id = g$id, keys = g$keys)
}

# The label dplyr gives parsed dot `i` of a verb call: `list(name, expr)` of the
# user's dot it comes from (across() expanded), the name as the user wrote it
# ("" for none). `origin` is the user dot of each parsed dot (the `provenance`
# of expand_across_dots()). Built only for the dot a warning names.
# @keywords internal
minmax_label_of <- function(w, i) {
  user_dots <- w$user_dots
  origin <- w$origin
  if (length(origin) != length(w$dots) || (length(origin) > 0L && max(origin) > length(user_dots))) {
    origin <- seq_along(w$dots)
    user_dots <- w$dots
  }
  o <- origin[[i]]
  list(name = rlang::names2(user_dots)[[o]], expr = rlang::quo_get_expr(user_dots[[o]]))
}

# The window min()/max() state of one verb call, or NULL when no call of it can
# meet an empty group (the facts are known and none is empty: nothing is built,
# the dots are parsed as they would be without F124). It keeps what the labels
# and the call head are built from when a warning is prepared: the verb's S3
# `method`, its dplyr name `default`, the method's `frame` number (the verb's
# frames are on the stack until it returns), the user's dots, the parsed `dots`
# and their `origin`. Then `facts` (NULL: not known yet), `carry` (the plan, not
# the verb, signals the warning: a lazy table warns when it runs), `double`
# (parse an INT32/BOOL8 argument as a double, read by ir_parse_minmax() through
# the schema), and the `records` of the parsed dots. `typed = FALSE` (filter():
# the predicate does not change type) never sets `double`.
# @keywords internal
minmax_win_open <- function(.data, by_given, method, default, frame, user_dots, dots, origin,
                            typed = TRUE) {
  setup <- minmax_window_setup(.data, by_given)
  if (!is.null(setup$facts) && setup$facts$E == 0L) {
    return(NULL)
  }
  w <- new.env(parent = emptyenv())
  w$method <- method
  w$default <- default
  w$frame <- frame
  w$user_dots <- user_dots
  w$dots <- dots
  w$origin <- origin
  w$typed <- typed
  w$records <- list()
  minmax_win_set_setup(w, setup)
  w
}

# @keywords internal
minmax_win_set_setup <- function(w, setup) {
  w$facts <- setup$facts
  w$carry <- setup$carry
  w$unknown_empty <- setup$unknown_empty
  w$double <- w$typed && !is.null(setup$facts) && setup$facts$E > 0L
  invisible(w)
}

# Does the verb need the data's answer (its steps run now) before it can say
# what its parsed min()/max() calls do? Only a lazy table whose facts are not
# known: when the grouping keeps empty groups that pending steps may leave
# (any call: the warning names them), or when the rows may be none and a call's
# INT32/BOOL8 argument would become a double (not for filter(): its type never
# changes). A column called `max`, or a double call over unknown rows, needs
# nothing: the plan carries that warning.
# @keywords internal
minmax_win_needs_run <- function(w) {
  if (is.null(w) || !is.null(w$facts) || length(w$records) == 0L) {
    return(FALSE)
  }
  isTRUE(w$unknown_empty) || (w$typed && minmax_win_int_call(w))
}

# The min()/max() calls of a parsed expression that can meet an empty group, in
# R's evaluation order (arguments left to right, then the call): `list(op,
# type, typed)` each, `type` the GPU type of the argument, `typed = FALSE` for
# the shift of a Date (`date_add_days(d, max(a))`: the result is a Date whatever
# the call's type). A call whose argument has one value per group
# (ir_is_group_scalar(): `max(n())`, the outer max() of `max(max(x))`) is not
# one: dplyr hands it one value.
# @keywords internal
minmax_ir_calls <- function(ir, schema) {
  calls <- list()
  walk <- function(node, shift = FALSE) {
    if (!inherits(node, "cuplyr_ir") || !identical(node$kind, "call")) {
      return(invisible())
    }
    shifts <- node$op %in% c("date_add_days", "date_sub_days")
    for (k in seq_along(node$args)) walk(node$args[[k]], shifts && k == 2L)
    if (node$op %in% c("min", "max") && length(node$args) == 1L && !ir_is_group_scalar(node$args[[1L]])) {
      type <- tryCatch(unname(ir_infer_type(node$args[[1L]], schema)), error = function(e) NA_character_)
      calls[[length(calls) + 1L]] <<- list(op = node$op, type = type, typed = !shift)
    }
  }
  walk(ir)
  calls
}

# Note the min()/max() calls of parsed dot `i` (a no-op without a state).
# @keywords internal
minmax_record_dot <- function(w, i, ir, schema) {
  if (is.null(w) || is.null(ir)) {
    return(invisible())
  }
  calls <- minmax_ir_calls(ir, schema)
  if (length(calls) > 0L) {
    w$records[[length(w$records) + 1L]] <- list(i = i, calls = calls)
  }
  invisible()
}

# Does a parsed call of the verb take an INT32 or logical argument (or one whose
# type is not known) and decide a type? Then the type of its result is the
# data's to decide.
# @keywords internal
minmax_win_int_call <- function(w) {
  for (r in w$records) {
    for (cl in r$calls) {
      if (!isFALSE(cl$typed) && (is.na(cl$type) || cl$type %in% c("INT32", "BOOL8"))) {
        return(TRUE)
      }
    }
  }
  FALSE
}

# The warning every window min()/max() call of the verb meets for the empty
# groups: `n` = calls x empty groups, the first being the first call of the
# first dot that has one, in the first empty group. NULL when there is nothing
# to say. Called while the verb runs (its frames give the call head).
# @return `list(head, n, name, expr, op, type, id, keys)` or NULL
# @keywords internal
minmax_win_warning <- function(w, per_group = w$facts$E) {
  if (is.null(w) || length(w$records) == 0L || per_group == 0L) {
    return(NULL)
  }
  first <- w$records[[1L]]
  lab <- minmax_label_of(w, first$i)
  n <- per_group * sum(vapply(w$records, function(r) length(r$calls), integer(1)))
  list(head = minmax_verb_head(w$method, w$default, w$frame), n = n, name = lab$name,
       expr = lab$expr, op = first$calls[[1L]]$op, type = first$calls[[1L]]$type,
       id = w$facts$id, keys = w$facts$keys)
}

# Signal the verb's window min()/max() warning now (the rows are known).
# @keywords internal
minmax_win_emit <- function(w) {
  if (is.null(w) || isTRUE(w$carry)) {
    return(invisible())
  }
  p <- minmax_win_warning(w)
  if (!is.null(p)) {
    minmax_signal(p$head, p$n, p$name, p$expr, p$op, p$type, p$id, p$keys)
  }
  invisible()
}

# The warning a lazy plan carries when it cannot know its row count: it fires
# when the node that holds it meets a zero-row input (one evaluation, no group
# line). NULL when the verb has none or is not in that case.
# @keywords internal
minmax_win_lazy_prep <- function(w) {
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
  minmax_signal(p$head, p$n, p$name, p$expr, p$op, p$type)
}

# How the verb knows its row count and empty groups: `facts` (minmax_window_facts())
# when it can read them now (an eager table, a lazy one with nothing pending or
# only row-preserving steps; with a grouping that keeps empty groups, nothing
# pending), else NULL; no step is ever run for it. A lazy table's warning is the
# plan's (`carry`) unless its grouping keeps empty groups and the facts are
# known: the plan cannot name them. It signals whenever the plan runs on an
# empty input (F80: a plan warns when it runs).
# @keywords internal
minmax_window_setup <- function(.data, by_given) {
  lazy <- identical(.subset2(.data, "exec_mode"), "lazy")
  pending <- lazy && has_pending_ops(.data)
  wants_empty <- minmax_wants_empty(.data, by_given)
  rows <- if (pending) minmax_lazy_rows(.data) else base_dim(.data)[[1L]]
  facts <- if (!is.na(rows) && !(wants_empty && pending)) minmax_window_facts(.data, rows, wants_empty)
  list(facts = facts, carry = lazy && (is.null(facts) || !wants_empty),
       unknown_empty = lazy && pending && wants_empty)
}
