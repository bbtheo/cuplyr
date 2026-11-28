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
# rewrites a dot into several), the dplyr verb whose call the warning names
# (`default`) and the frame of its S3 method (`frame`), built from
# `.minmax_src` only when an aggregation can hit (minmax_dot_info()). Those
# aggregations carry a small `$dot` built from it (minmax_attach_dots()): the
# dot's name and
# expression (not the quosure, so a plan does not hold the environment), the
# verb's call head and `ctx`, the environment that routes the hits of
# summarise_with_empty_groups() to its collector (minmax_collect()). Nothing is
# package-global: a verb run while another one builds its dots (a tbl_gpu verb
# called inside a dot) cannot see or emit the outer verb's records.

# The user's dots (`user_dots`, quosures) behind the aggregations or dots of
# `dots` (their `provenance` says which: `origin`), for the label of a
# warning.
# @return a list with `user_dots`, `origin` and `ids`
# @keywords internal
minmax_dot_source <- function(user_dots, dots, default, ids = NULL) {
  origin <- attr(dots, "provenance")$origin
  if (length(origin) != length(dots) || length(user_dots) < max(origin, 0L)) {
    # (no provenance: the dots are the user's own)
    origin <- seq_along(dots)
    user_dots <- dots
  }
  list(user_dots = user_dots, origin = origin, ids = ids %||% seq_along(origin),
       default = default)
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
# the S3 method's (R rewrites the head of the method's own call to the method
# name): dplyr names the call as written, or the one a caller marked for it
# (minmax_marked_call(): `add_tally()`). `frame` is the method's frame number
# as its helper saw it (sys.parent()), so the usual call reads two frames; a
# caller that is not the method (an internal rerun) is looked for down the
# stack. The method's frame is known by the `.Generic` UseMethod() defines in
# it, an environment lookup (sys.function() gives a copy of the closure, so no
# pointer comparison can match it, and identical() would compare bodies).
# @keywords internal
minmax_verb_head <- function(default, frame = NA_integer_) {
  n <- sys.nframe()
  if (is.na(frame) || frame < 2L || frame >= n || !minmax_dispatched_at(frame, default)) {
    frame <- NA_integer_
    for (i in rev(seq_len(n - 1L))) {
      if (minmax_dispatched_at(i, default)) {
        frame <- i
        break
      }
    }
    if (is.na(frame) || frame < 2L) {
      return(as.name(default))
    }
  }
  head <- minmax_marked_call(frame)
  head <- if (is.null(head)) sys.call(frame - 1L)[[1L]] else head[[1L]]
  if (is.symbol(head) ||
      (is.call(head) && length(head) == 3L && identical(head[[1L]], quote(`::`)))) {
    return(head)
  }
  as.name(default)
}

# The call dplyr names instead of the verb's own, as dplyr_error_call() finds it
# from the method's frame `frame` (#74): a `.__dplyr_error_call__.` binding in
# the frame that called the verb (dplyr_local_error_call(): add_tally() calling
# mutate(), add_count() calling add_tally()), followed while the frame it points
# at has a caller with one. The call of the frame it ends at (of its generic,
# for an S3 method), or the bound call itself; NULL when nothing is bound.
# @keywords internal
minmax_marked_call <- function(frame) {
  parents <- sys.parents()
  call <- NULL
  k <- frame
  repeat {
    p <- parents[[k]]
    marker <- if (p > 0L) get0(".__dplyr_error_call__.", envir = sys.frame(p), inherits = FALSE)
    if (is.null(marker)) {
      break
    }
    if (!is.environment(marker)) {
      # (a call, or anything else dplyr would format as one)
      return(if (is.call(marker)) marker)
    }
    k <- 0L
    for (j in rev(seq_len(p))) {
      if (identical(sys.frame(j), marker)) {
        k <- j
        break
      }
    }
    if (k == 0L) {
      return(NULL)
    }
    call <- if (k > 1L && !is.null(get0(".Generic", envir = marker, inherits = FALSE))) {
      sys.call(k - 1L)
    } else {
      sys.call(k)
    }
  }
  call
}

# Is frame `i` a method UseMethod() dispatched for the dplyr generic `generic`?
# @keywords internal
minmax_dispatched_at <- function(i, generic) {
  identical(get0(".Generic", envir = sys.frame(i), inherits = FALSE), generic)
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
    info <- src[[1L]] %||% minmax_dot_source(src[[2L]], src[[3L]], "summarise")
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
  if (is.null(ir_cache)) {
    return(aggregations)
  }
  # (this check runs on every summarise() with a min()/max(): a plain loop,
  # which costs less than a vapply() call for the usual few aggregations)
  hit <- FALSE
  na_rm <- FALSE
  for (agg in aggregations) {
    if (isTRUE(.subset2(agg, "identity"))) {
      hit <- TRUE
      if (isTRUE(.subset2(agg, "na_rm"))) {
        na_rm <- TRUE
        break
      }
    }
  }
  if (!hit) {
    return(aggregations)
  }
  if (!na_rm && !is.null(data)) {
    rows <- if (inherits(.subset2(data, "lazy_ops"), "ast_node")) minmax_lazy_rows(data) else base_dim(data)[[1L]]
    if (!is.na(rows) && rows > 0L) {
      return(aggregations)
    }
  }
  n <- length(aggregations)
  info <- minmax_dot_info(ir_cache)
  if (is.null(info)) {
    return(aggregations)
  }
  k <- length(info$origin)
  if (n != k) {
    stop("internal error: ", n, " aggregations for ", k, " dots (min()/max() warning labels)",
         call. = FALSE)
  }
  ctx <- minmax_ctx(ir_cache)
  head <- ctx$head
  if (is.null(head)) {
    head <- ctx$head <- minmax_verb_head(info$default, info$frame %||% NA_integer_)
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
# NA row is the evaluation dplyr makes on a zero-row grouped input). The result
# rows are in dplyr's group order (sorted keys, or first appearance for `.by`:
# gpu_summarise() reorders the hit masks with their rows), `key_at(row)` the key
# values of a result row, `grouped` whether the call has group lines at all.
# @keywords internal
minmax_summarise_records <- function(aggs, hits, key_at, grouped) {
  records <- list()
  for (i in which(lengths(hits) > 0L)) {
    agg <- .subset2(aggs, i)
    rows <- hits[[i]]
    rows <- rows[!is.na(rows)]
    first_row <- NA_integer_
    id <- NA_integer_
    if (grouped && length(rows) > 0L) {
      first_row <- min(rows)
      id <- as.integer(first_row)
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
# type is the data's to decide, but only the TYPE: the groups that have rows
# compute in integers (F42's overflow and `%/%` by 0L rules), as dplyr's do. So
# a dot is parsed once, integer, and when a group is empty the dot's ROOT is cast
# to the type it has in dplyr (minmax_dot_root(), ir_infer_type_minmax_double()).
# A dot is type-dependent iff that type differs from the parsed one (`m = max(a)`,
# `max(a) + 1L`; not `a == max(a)`). Later dots read the cast column.
#
# The verb reads the row count and the empty groups before its dots are parsed
# when it can (minmax_window_setup(): an eager table, a lazy one nothing or only
# row-preserving steps are pending on, a bare parquet scan's count coming from
# its footer). When it cannot (a lazy table whose pending steps change the rows),
# mutate() parses once as if nothing were empty and the data is asked only if a
# parsed dot is type-dependent (minmax_win_needs_run()): an in-memory plan runs
# its steps now (F80's rule for an integer sum()); a plan over a parquet scan
# NEVER does (it would read every column of the file): a pruned facts query over
# the pending plan answers instead (minmax_scan_resolve()), and the plan stays
# lazy. A double, string or timestamp call, or a dot whose type does not depend
# on the data, only has a warning to raise, which the plan carries.
#
# A call whose argument has one value per group (`max(n())`, `max(sum(x))`,
# ir_is_group_scalar()) never meets an empty vector: it neither warns nor turns
# double. A Date shifted by an integer amount made of min()/max() calls
# (`d + max(a)`) is a Date whatever those calls' type.
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
  for (dot in dots) {
    if (expr_calls_any(rlang::quo_get_expr(dot), c("min", "max"))) {
      return(TRUE)
    }
  }
  FALSE
}

# TRUE iff every min()/max() call of the dots is `min(col)` / `max(col)` of a
# bare column of the table whose GPU type is FLOAT64, STRING or a timestamp, and
# no dot of the call makes a column of that name: such a call is the same type
# whatever the data (an empty group gives Inf, or NA for a string, not another
# type), so nothing needs to be known of the rows before the dots are parsed.
# Any other shape (`max(a + 1L)`, `max(x, na.rm = TRUE)`, `.data$x`, a name an
# earlier dot makes) is not known to be: FALSE. A dot with no call gives TRUE.
# @param dots the (expanded) quosures of the verb call
# @param schema the schema they are evaluated against
# @keywords internal
minmax_dots_type_fixed <- function(dots, schema) {
  names_ok <- .subset2(schema, "names")
  types <- .subset2(schema, "types")
  made <- names(dots)
  fixed_arg <- function(a) {
    if (!is.symbol(a)) {
      return(FALSE)
    }
    nm <- as.character(a)
    pos <- which(names_ok == nm)
    length(pos) == 1L && !(nm %in% made) &&
      types[[pos]] %in% c("FLOAT64", "STRING", "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")
  }
  fixed <- function(x) {
    if (!is.call(x)) {
      return(TRUE)
    }
    head <- x[[1L]]
    if (is.call(head) && length(head) == 3L && identical(head[[1L]], quote(`::`))) {
      head <- head[[3L]]
    }
    if (is.symbol(head) && as.character(head) %in% c("min", "max")) {
      return(length(x) == 2L && is.null(names(x)) && fixed_arg(x[[2L]]))
    }
    parts <- as.list(x)
    for (i in seq_along(parts)) {
      if (identical(parts[[i]], quote(expr = ))) next
      a <- parts[[i]]
      if (rlang::is_quosure(a)) a <- rlang::quo_get_expr(a)
      if (!fixed(a)) {
        return(FALSE)
      }
    }
    TRUE
  }
  for (dot in dots) {
    if (!fixed(rlang::quo_get_expr(dot))) {
      return(FALSE)
    }
  }
  TRUE
}

# The row count of a lazy table the pending steps cannot change (mutate,
# select, rename, window, arrange: a walk of the plan), else NA. A plan that
# ends in a parquet scan with nothing pushed into it has its footers' count (as
# dim() does, bare_scan_of()), so no read happens for it; any other scan is NA.
# @param footer when TRUE, the result carries `attr(, "footer")` = TRUE if the
#   count is a parquet footer's (a type decided from it must be checked when the
#   file is read, lower_scan())
# @keywords internal
minmax_lazy_rows <- function(.data, footer = FALSE) {
  node <- .subset2(.data, "lazy_ops")
  while (inherits(node, "ast_node")) {
    type <- .subset2(node, "type")
    if (identical(type, "source")) {
      scan <- .subset2(node, "scan")
      if (!is.null(scan)) {
        if (length(.subset2(scan, "predicates")) > 0L || !is.null(.subset2(scan, "limit"))) {
          return(NA_integer_)
        }
        rows <- as.integer(parquet_nrow(.subset2(scan, "paths")))
        if (footer) attr(rows, "footer") <- TRUE
        return(rows)
      }
    } else if (!type %in% c("mutate", "select", "rename", "window", "arrange")) {
      return(NA_integer_)
    }
    node <- .subset2(node, "input")
  }
  base_dim(.data)[[1L]]
}

# `.data`'s plan with its scan marked: a min()/max() type was decided from the
# scan's non-zero footer count, which lower_scan() holds the file to. The plan
# is a row-preserving chain over the scan (minmax_lazy_rows()).
# @keywords internal
minmax_mark_scan <- function(.data) {
  mark <- function(node) {
    if (identical(.subset2(node, "type"), "source")) {
      node$scan$typed_nonzero <- TRUE
      return(node)
    }
    node$input <- mark(.subset2(node, "input"))
    node
  }
  set_tbl_gpu_field(.data, "lazy_ops", mark(.subset2(.data, "lazy_ops")))
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
# and the call head are built from when a warning is prepared: the verb's dplyr
# name `default`, its S3 method's `frame` number (the verb's frames are on the
# stack until it returns), the user's dots, the parsed `dots`
# and their `origin`. Then `facts` (NULL: not known yet), `carry` (the plan, not
# the verb, signals the warning: a lazy table warns when it runs), `double`
# (a type-dependent dot's root is cast to a double, minmax_dot_root()), and the
# `records` of the parsed dots (`type_dep`: the dot's type is the data's to decide). `typed = FALSE` (filter():
# the predicate does not change type) never sets `double`.
# @keywords internal
minmax_win_open <- function(.data, by_given, default, frame, user_dots, dots, origin,
                            typed = TRUE) {
  # (the usual call, inline: nothing pending, a grouping that cannot keep an
  # empty group, rows: no call can hit)
  if (!inherits(.subset2(.data, "lazy_ops"), "ast_node") &&
      (by_given || length(.subset2(.data, "groups")) == 0L ||
         (is.null(.subset2(.data, "fixed_keys")) && !isFALSE(.subset2(.data, "group_drop"))))) {
    rows <- base_dim(.data)[[1L]]
    if (!is.na(rows) && rows > 0L) {
      return(NULL)
    }
  }
  setup <- minmax_window_setup(.data, by_given)
  # (a type decided from a parquet footer's count keeps its state: the scan is
  # marked when a dot's type depends on it)
  if (!is.null(setup$facts) && setup$facts$E == 0L && !(typed && isTRUE(setup$footer))) {
    return(NULL)
  }
  w <- new.env(parent = emptyenv())
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
  # (nothing to warn about when no group is empty)
  w$carry <- isTRUE(setup$carry) && !(!is.null(setup$facts) && setup$facts$E == 0L)
  w$fixed <- w$carry && isTRUE(setup$fixed)
  w$unknown_empty <- setup$unknown_empty
  w$scan <- isTRUE(setup$scan)
  w$footer <- isTRUE(setup$footer)
  w$double <- w$typed && !is.null(setup$facts) && setup$facts$E > 0L
  invisible(w)
}

# Does the verb need the data's answer before it can say what its parsed
# min()/max() calls do? Only a lazy table whose facts are not known: when the
# grouping keeps empty groups that pending steps may leave (any call: the warning
# names them), or when the rows may be none and a dot's type depends on it (an
# INT32/BOOL8 call that changes the dot's type, minmax_dot_root(); not for
# filter(): its type never changes). A column called `max`, a double call or a
# dot like `a == max(a)` over unknown rows needs nothing: the plan carries that
# warning.
# @keywords internal
minmax_win_needs_run <- function(w) {
  if (is.null(w) || !is.null(w$facts) || length(w$records) == 0L) {
    return(FALSE)
  }
  isTRUE(w$unknown_empty) ||
    (w$typed && any(vapply(w$records, function(r) isTRUE(r$type_dep), logical(1))))
}

# Did a dot's type depend on a min()/max() call over a table whose rows were
# known? (what a parquet footer's count decided)
# @keywords internal
minmax_win_type_decided <- function(w) {
  !is.null(w) && w$typed && any(vapply(w$records, function(r) isTRUE(r$type_dep), logical(1)))
}

# The min()/max() calls of a parsed expression that can meet an empty group, in
# R's evaluation order (arguments left to right, then the call): `list(op,
# type)` each, `type` the GPU type of the argument. A call whose argument has one
# value per group (ir_is_group_scalar(): `max(n())`, the outer max() of
# `max(max(x))`) is not one: dplyr hands it one value.
# @keywords internal
minmax_ir_calls <- function(ir, schema) {
  calls <- list()
  walk <- function(node) {
    if (!inherits(node, "cuplyr_ir") || !identical(node$kind, "call")) {
      return(invisible())
    }
    for (k in seq_along(node$args)) walk(node$args[[k]])
    if (node$op %in% c("min", "max") && length(node$args) == 1L && !ir_is_group_scalar(node$args[[1L]])) {
      type <- tryCatch(unname(ir_infer_type(node$args[[1L]], schema)), error = function(e) NA_character_)
      calls[[length(calls) + 1L]] <<- list(op = node$op, type = type)
    }
  }
  walk(ir)
  calls
}

# Note the min()/max() calls of parsed dot `i` (a no-op without a state) and,
# for the root cast: is the dot's type the data's to decide? It is iff the type
# it has with every INT32 min()/max() call that can meet an empty group a double
# (ir_infer_type_minmax_double()) differs from the parsed one. When the facts
# say a group is empty (`w$double`) the dot's root is cast to that type, and
# later dots read the cast column; the inner calls stay integer, so the groups
# with rows keep F42's integer rules. A type inference that fails, or one that
# is not a double, can't be cast: NULL, the verb takes the CPU route.
# @return `ir`, cast if needed; NULL when the dot can't be cast
# @keywords internal
minmax_dot_root <- function(w, i, ir, schema) {
  if (is.null(w)) {
    return(ir)
  }
  calls <- minmax_ir_calls(ir, schema)
  if (length(calls) == 0L) {
    return(ir)
  }
  type_dep <- FALSE
  cast <- FALSE
  if (w$typed && any(vapply(calls, function(cl) is.na(cl$type) || cl$type %in% c("INT32", "BOOL8"), logical(1)))) {
    t_i <- tryCatch(unname(ir_infer_type(ir, schema)), error = function(e) NA_character_)
    t_d <- tryCatch(unname(ir_infer_type_minmax_double(ir, schema)), error = function(e) NA_character_)
    type_dep <- is.na(t_i) || is.na(t_d) || !identical(t_i, t_d)
    if (type_dep && isTRUE(w$double)) {
      if (!identical(t_d, "FLOAT64")) {
        return(NULL)
      }
      cast <- TRUE
    }
  }
  w$records[[length(w$records) + 1L]] <- list(i = i, calls = calls, type_dep = type_dep)
  if (cast) ir_call("cast_float64", list(ir)) else ir
}

# @keywords internal
minmax_record_dot <- function(w, i, ir, schema) {
  if (!is.null(w) && !is.null(ir)) {
    minmax_dot_root(w, i, ir, schema)
  }
  invisible()
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
  list(head = minmax_verb_head(w$default, w$frame), n = n, name = lab$name,
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
# line). A plan over a parquet scan whose empty groups a facts query found at
# the verb (`fixed`, minmax_scan_resolve()) carries the verb's whole warning
# instead (count, first empty group and its keys): `always`, it fires whenever
# the node runs, whatever its input, so it comes at collect() after the
# warnings of the steps below it, in dplyr's order. NULL when the verb has none
# or is not in either case.
# @keywords internal
minmax_win_lazy_prep <- function(w) {
  if (is.null(w) || !isTRUE(w$carry)) {
    return(NULL)
  }
  if (isTRUE(w$fixed)) {
    p <- minmax_win_warning(w)
    if (!is.null(p)) p$always <- TRUE
    return(p)
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
  minmax_signal(p$head, p$n, p$name, p$expr, p$op, p$type, p$id %||% NA_integer_, p$keys)
}

# How the verb knows its row count and empty groups: `facts` (minmax_window_facts())
# when it can read them now (an eager table, a lazy one with nothing pending or
# only row-preserving steps; with a grouping that keeps empty groups, nothing
# pending), else NULL; no step is ever run for it. A lazy table's warning is the
# plan's (`carry`) unless its grouping keeps empty groups and the facts are
# known: the plan cannot name them. It signals whenever the plan runs on an
# empty input (F80: a plan warns when it runs). `scan`: the plan reads a parquet
# scan somewhere, so when the facts are not known they are asked by a pruned
# query, never by running the plan (minmax_scan_resolve()); `footer`: the row
# count came from a parquet footer.
# @keywords internal
minmax_window_setup <- function(.data, by_given) {
  lazy <- identical(.subset2(.data, "exec_mode"), "lazy")
  pending <- lazy && has_pending_ops(.data)
  wants_empty <- minmax_wants_empty(.data, by_given)
  rows <- if (pending) minmax_lazy_rows(.data, footer = TRUE) else base_dim(.data)[[1L]]
  footer <- isTRUE(attr(rows, "footer", exact = TRUE))
  rows <- as.integer(rows)
  facts <- if (!is.na(rows) && !(wants_empty && pending)) minmax_window_facts(.data, rows, wants_empty)
  list(facts = facts, carry = lazy && (is.null(facts) || !wants_empty),
       unknown_empty = lazy && pending && wants_empty,
       scan = pending && plan_has_scan(.subset2(.data, "lazy_ops")),
       footer = footer && !is.null(facts))
}

# Run a facts query (minmax_scan_facts()) over a pending plan: quietly, and
# reading only what the query reads. Every warning and message it could raise
# comes from a step of the plan, which collect() runs again and reports then
# (F80: a plan warns when it runs), so none is shown here, except one said once
# per session (an invalid option value, suppress_probe_warnings()'s rule; the
# roll_median() engine note): muffled, it would be recorded as given and never
# shown. The mutates nothing of the query reads are not computed
# (`prune_unread`, keep_unread_mutates()), so the columns they read are not
# read. The flag is restored however the query ends (its restore is registered
# before it is set).
# @param x a lazy table, the query on top of the pending plan
# @keywords internal
minmax_facts_compute <- function(x) {
  prev <- .optimizer_opts$prune_unread
  on.exit(.optimizer_opts$prune_unread <- prev, add = TRUE)
  .optimizer_opts$prune_unread <- TRUE
  withCallingHandlers(
    suppress_probe_warnings(compute(x)),
    message = function(m) {
      if (!inherits(m, "cuplyr_message_roll_median_exact")) {
        tryInvokeRestart("muffleMessage")
      }
    }
  )
}

# The observed group keys of a grouped lazy plan, as a computed table with the
# plan's grouping (groups, `.drop`, fixed keys): one row per key, from a quiet
# pruned query (minmax_facts_compute()), so the scan reads only the columns the
# keys and the pending steps' rows need, and the plan is not run. For a plan
# over a parquet scan, which is never run whole for its keys (minmax_scan_facts(),
# row_slice_prepare()).
# @keywords internal
plan_group_keys <- function(.data) {
  keys <- minmax_facts_compute(push_op(.data, ast_distinct(input_node(.data), .subset2(.data, "groups"))))
  if (!is.null(fixed <- .subset2(.data, "fixed_keys"))) {
    keys <- set_tbl_gpu_field(keys, "fixed_keys", fixed)
  }
  keys
}

# The facts of a lazy plan over a parquet scan whose rows (or empty groups) the
# verb could not read, by a query over the pending plan through the normal
# compute path: the optimizer prunes it, so the scan reads only the columns the
# query needs (the predicate and key columns). The plan itself is not run, not
# broken and not read whole. `.data` is the verb's input, lazy.
# - a grouping that keeps empty groups: the distinct group keys (k rows), then
#   the same empty-group logic as an eager table's (minmax_empty_groups())
# - otherwise the row count, `n()` of the whole plan
# @return what minmax_window_facts() returns
# @keywords internal
minmax_scan_facts <- function(.data, wants_empty) {
  if (wants_empty) {
    keys <- plan_group_keys(.data)
    n_keys <- gpu_dim(.subset2(keys, "ptr"))[[1L]]
    # (the rows are only asked for being none: the keys of an empty input are none)
    g <- minmax_empty_groups(keys, n_keys)
    E <- g$E
    if (E == 0L && n_keys == 0L) E <- 1L
    return(list(E = E, n_in = n_keys, id = g$id, keys = g$keys))
  }
  flat <- dplyr::ungroup(.data)
  counted <- minmax_facts_compute(push_op(flat, ast_summarise(
    input_node(flat), list(make_aggregation(".n", NA_character_, "n")), character()
  )))
  minmax_window_facts(.data, as.integer(collect(counted)[[1L]]), FALSE)
}

# Answer the facts of the verb state `w` (a plan over a parquet scan) with a
# pruned query, instead of running the plan (minmax_scan_facts()). The plan stays
# lazy and carries the warning, so it comes when the plan runs, after the
# warnings of the steps below (which the query does not show), every run: for
# an empty input, as an in-memory plan's; for a grouping that keeps empty
# groups, the verb's own warning, as the query found them (`fixed`: the type
# was decided from them too).
# @keywords internal
minmax_scan_resolve <- function(w, .data, by_given) {
  wants_empty <- minmax_wants_empty(.data, by_given)
  facts <- minmax_scan_facts(.data, wants_empty)
  minmax_win_set_setup(w, list(facts = facts, carry = TRUE, fixed = wants_empty,
                               unknown_empty = FALSE, scan = TRUE, footer = FALSE))
  invisible(w)
}

# -----------------------------------------------------------------------------
# min()/max() with na.rm = TRUE over a group with no value left (#74)
# -----------------------------------------------------------------------------
#
# dplyr evaluates `max(x, na.rm = TRUE)` once per group, and base R warns for
# each group whose values are all NA/NaN (and for each empty one, which the
# F124 machinery above names). The window computes -Inf/Inf for those groups
# natively, so the verb plans the warning: `narm_warn_plan()` lists the calls
# of the dots in dplyr's evaluation order, the window node that holds them
# carries the list (`narm_warn`), and lower_window_narm_warn() finds the groups
# when the node runs, with the aggregate summarise() uses for this (the same
# group order and the same hit rows), and signals dplyr's one condition.

# Is `spec` a window min()/max() with na.rm = TRUE?
# @keywords internal
narm_minmax_spec <- function(spec) {
  isTRUE(.subset2(spec, "na_rm")) && !is.null(.subset2(spec, "cse_key"))
}

# The calls of a verb's dots that warn for an all-NA group.
# @param expressions the parsed dots (`ir`, `dot`: the dot's position)
# @param plan plan_mutate_window_batches()'s result
# @param user_dots,origin the user's dots and the user dot each parsed dot comes from
# @param by_given whether the call has `.by =` (groups in first-appearance order)
# @param minmax the verb's F124 state (NULL when no group can be empty)
# @return NULL when no call warns; the `narm_warn` of the window node; FALSE when
#   the warning can't be planned (the calls are in several stages, or the verb's
#   F124 warning for empty groups would be a second condition)
# @keywords internal
narm_warn_plan <- function(expressions, plan, user_dots, origin, by_given, minmax) {
  specs <- plan$all_specs
  is_narm <- vapply(specs, narm_minmax_spec, logical(1))
  if (!any(is_narm)) {
    return(NULL)
  }
  by_key <- stats::setNames(specs[is_narm], vapply(specs[is_narm], `[[`, character(1), "cse_key"))
  in_stage <- 0L
  for (bp in plan$batch_plans) {
    for (stage in bp$stages) {
      in_stage <- in_stage + any(vapply(stage$specs, narm_minmax_spec, logical(1)))
    }
  }
  if (in_stage > 1L) {
    return(FALSE)
  }
  if (!is.null(minmax) && (isTRUE(minmax$unknown_empty) ||
        (!is.null(minmax$facts) && minmax$facts$E > 0L && minmax$facts$n_in > 0L))) {
    return(FALSE)
  }
  head <- minmax_verb_head("mutate", NA_integer_)
  names_user <- rlang::names2(user_dots)
  calls <- list()
  for (e in expressions) {
    walk <- function(node) {
      if (!inherits(node, "cuplyr_ir") || !identical(node$kind, "call")) {
        return(invisible())
      }
      for (a in node$args) walk(a)
      if (node$op %in% c("min", "max") && isTRUE(node$meta$na_rm)) {
        spec <- by_key[[rlang::hash(list(op = node$op, args = node$args, meta = node$meta))]]
        if (is.null(spec)) {
          stop("internal error: a min()/max() with na.rm = TRUE has no window spec", call. = FALSE)
        }
        d <- e$dot
        o <- if (length(origin) >= d && !is.na(origin[[d]])) origin[[d]] else d
        calls[[length(calls) + 1L]] <<- list(
          dot = list(id = d, name = names_user[[o]], expr = rlang::quo_get_expr(user_dots[[o]]), head = head),
          op = node$op, value_col = spec$value_col, spec_col = spec$output_col
        )
      }
    }
    walk(e$ir)
  }
  list(by_first = by_given, calls = calls)
}

# Signal the warning of a window node's min()/max(na.rm = TRUE) calls: the
# groups with no value left, found with summarise()'s own min()/max() (na.rm,
# identity report) over the node's input. An input with no row is the F124
# warning's (one evaluation, no group line).
# @keywords internal
lower_window_narm_warn <- function(narm, input_ptr, input_schema, groups) {
  if (gpu_dim(input_ptr)[[1L]] == 0L) {
    return(invisible())
  }
  calls <- narm$calls
  pair <- vapply(calls, function(cl) paste(cl$op, cl$value_col, sep = "\r"), character(1))
  # A group has no value left only when its column holds a NA or NaN, and the
  # sum of the whole column is NA or NaN then: one reduction settles the common
  # case (an Inf - Inf sum takes the grouped pass below, which is exact)
  value_idx <- match(unique(vapply(calls, `[[`, character(1), "value_col")), input_schema$names) - 1L
  total <- gpu_summarise(
    input_ptr, integer(), value_idx, rep("sum", length(value_idx)),
    rep(FALSE, length(value_idx)), rep(NA_real_, length(value_idx)), rep(FALSE, length(value_idx)),
    "keys"
  )
  if (!anyNA(unlist(gpu_collect(total, paste0("v", seq_along(value_idx))), use.names = FALSE))) {
    return(invisible())
  }
  uniq <- unique(pair)
  first <- match(uniq, pair)
  k <- length(uniq)
  group_idx <- if (length(groups) > 0L) match(groups, input_schema$names) - 1L else integer()
  res <- gpu_summarise(
    input_ptr, group_idx,
    match(vapply(calls[first], `[[`, character(1), "value_col"), input_schema$names) - 1L,
    vapply(calls[first], `[[`, character(1), "op"),
    rep(TRUE, k), rep(NA_real_, k), rep(TRUE, k),
    if (isTRUE(narm$by_first)) "first" else "keys"
  )
  hits <- attr(res, "identity_rows")
  grouped <- length(groups) > 0L
  key_at <- function(row) {
    group_keys_frame(
      gpu_gather_rows(list(gpu_select(res, seq_along(groups) - 1L)), as.numeric(row - 1L)),
      group_keys_schema(list(schema = input_schema, groups = groups))
    )
  }
  records <- list()
  for (i in seq_along(calls)) {
    rows <- hits[[match(pair[[i]], uniq)]]
    rows <- rows[!is.na(rows)]
    if (length(rows) == 0L) next
    records[[length(records) + 1L]] <- local({
      row <- min(rows)
      list(dot = calls[[i]]$dot, op = calls[[i]]$op, type = "FLOAT64", col = calls[[i]]$spec_col,
           n = length(rows), id = if (grouped) as.integer(row) else NA_integer_,
           keys = function() key_at(row))
    })
  }
  minmax_emit(records)
}
