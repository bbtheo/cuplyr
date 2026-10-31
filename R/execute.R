# Execution Helpers - Unified eager/lazy op dispatch
#
# This module implements the "one node, two schedules" execution model
# (see scratchpad/unification_design.md, decisions D1/D3/D4). Every verb
# builds a single AST node over `input_node(.data)` and hands it to
# `push_op()`, which is the *only* place that branches on `exec_mode`:
# lazy mode stores the node in `$lazy_ops` with no GPU work; eager mode
# lowers and executes the node immediately via `lower_and_execute()`.
#
# `R/lower.R` (lowering) and `R/ast.R` (schema inference) are shared
# verbatim by both schedules.

#' Infer a node's output schema, INCLUDING `factor_levels`
#'
#' `infer_schema()` (R/ast.R) computes `names`/`types` recursively via each
#' node type's own S3 method, but only `infer_schema.ast_join` (Phase 11 L2,
#' Bug 1) threads `factor_levels` through on its own -- every other node
#' type (filter/mutate/select/arrange/summarise/distinct/slice/rename/
#' window/barrier) relies entirely on `push_op()`'s separate
#' `propagate_factor_levels()` step, which only ever ran at op-construction
#' time (each verb's own call site, one node at a time against that node's
#' OWN immediate input schema). That's fine for a freshly-built single-node
#' AST, but `current_schema()`/`compute.tbl_gpu()` both need to recompute
#' the schema for an ARBITRARY-depth pending AST from scratch (a multi-op
#' lazy chain, or a chain built on top of another lazy op) by calling plain
#' `infer_schema()` directly -- which, for any of those non-join node
#' types, silently drops `factor_levels` for the whole subtree above the
#' leaf `ast_source` (the one place besides `ast_join` that genuinely
#' carries `factor_levels`, via `input_node()`'s `ast_source(.data$schema)`).
#'
#' This wraps `infer_schema()` to recurse down to each node's input FIRST
#' (recovering the leaf's real `factor_levels` from its `ast_source`
#' schema, or an `ast_join`'s own self-contained computation), then
#' reapplies `propagate_factor_levels()` on the way back up -- exactly what
#' `push_op()` does for one node, generalized to a whole AST.
#'
#' @param node An AST node (or `NULL`)
#' @return List with `names`, `types`, and (if any survive) `factor_levels`
#' @keywords internal
infer_schema_full <- function(node) {
  if (is.null(node)) {
    return(list(names = character(0), types = character(0)))
  }
  if (identical(node$type, "source")) {
    return(node$schema)
  }
  if (identical(node$type, "join")) {
    return(infer_schema(node))  # already factor_levels-aware, see build_join_schema()
  }

  input_schema <- infer_schema_full(node$input)
  propagate_schema_meta(node, input_schema, infer_schema_from(node, input_schema))
}

#' Get the schema a new op should be built against
#'
#' The table's stored `$schema`. For a lazy table with pending operations it is
#' the output schema of `$lazy_ops`, factor levels and POSIXct zones included:
#' every writer of `lazy_ops` (`push_op()`, `push_join()`, `compute()`,
#' `collapse()`'s barrier, the callers that copy both fields) stores
#' the schema it implies, so the invariant `schema == infer_schema_full(lazy_ops)`
#' holds and reading it is O(1), whatever the plan's depth. Set
#' `options(cuplyr.check_schema_cache = TRUE)` or the environment variable
#' `CUPLYR_CHECK_SCHEMA_CACHE=1` to verify the invariant on every call (internal
#' debug switch for test runs; aborts with `cuplyr_error_internal`).
#'
#' @param .data A `tbl_gpu` object
#' @return List with `names` and `types` (and possibly `factor_levels`, `tz`)
#' @keywords internal
current_schema <- function(.data) {
  schema <- .subset2(.data, "schema")
  # (the cheap tests first: the switch reads an option on every call. A lazy
  # verb calls this a few times per node, so has_pending_ops() and
  # check_schema_cache_on() are inline: the same tests, in the same order.)
  if (identical(.subset2(.data, "exec_mode"), "lazy") &&
      inherits(.subset2(.data, "lazy_ops"), "ast_node") && inherits(.data, "tbl_gpu")) {
    on <- getOption("cuplyr.check_schema_cache")
    if (is.null(on)) {
      on <- .schema_memo$check_env
      if (is.null(on)) on <- check_schema_cache_on()
    } else {
      on <- isTRUE(on)
    }
  } else {
    on <- FALSE
  }
  if (on) {
    expected <- infer_schema_full(.subset2(.data, "lazy_ops"))
    if (!identical(schema, expected)) {
      rlang::abort(c(
        "Internal error: a lazy table's stored schema differs from its pending plan's.",
        i = paste0("stored: ", paste0(schema$names, collapse = ", "), " (",
                   paste0(unname(schema$types), collapse = ", "), ")"),
        i = paste0("plan:   ", paste0(expected$names, collapse = ", "), " (",
                   paste0(unname(expected$types), collapse = ", "), ")"),
        i = paste0("difference: ", paste(
          if (isTRUE(d <- all.equal(schema, expected))) {
            "none up to representation (storage type, NULL slots; see waldo::compare())"
          } else {
            d
          }, collapse = "; "))
      ), class = "cuplyr_error_internal")
    }
  }
  schema
}

# The debug switch of current_schema(): option `cuplyr.check_schema_cache`, else
# env var `CUPLYR_CHECK_SCHEMA_CACHE` ("1"/"true"). Off unless asked for.
# @keywords internal
check_schema_cache_on <- function() {
  opt <- getOption("cuplyr.check_schema_cache")
  if (!is.null(opt)) {
    return(isTRUE(opt))
  }
  on <- .schema_memo$check_env
  if (is.null(on)) {
    on <- Sys.getenv("CUPLYR_CHECK_SCHEMA_CACHE") %in% c("1", "true", "TRUE", "yes")
    .schema_memo$check_env <- on  # read once per session
  }
  on
}

# dplyr's verbs that read columns (filter, mutate, summarise, arrange, ...)
# refuse a data frame with duplicated names; select()/relocate()/rename()
# are what repair them. Verbs call this at their entry so dplyr's message
# comes before any other check; push_op() enforces the same rule on every
# node (node_is_position_safe()), so a verb that forgets still can't run.
# @keywords internal
check_unique_input_names <- function(.data, call = rlang::caller_env()) {
  nms <- current_schema(.data)$names
  # dplyr's DataMask (R/data-mask.R) checks missing names, then "" names, then
  # duplicates, so a repeated NA or "" gets the missing/empty message.
  if (anyNA(nms)) {
    rlang::abort("Can't transform a data frame with missing names.", call = call)
  }
  if (any(nms == "")) {
    rlang::abort("Can't transform a data frame with `NA` or `\"\"` names.", call = call)
  }
  if (anyDuplicated(nms) > 0L) {
    abort_duplicate_names(call)
  }
  invisible(.data)
}

abort_duplicate_names <- function(call = rlang::caller_env()) {
  rlang::abort("Can't transform a data frame with duplicate names.", call = call)
}

# A tbl_gpu may carry duplicated column names only while it is ungrouped: a
# group column is identified by name, and so are the fixed keys of
# `slice(.preserve = TRUE)`. Every writer that could put duplicated names on a
# grouped table calls this first (names<-, push_op()'s output, new_tbl_gpu()).
# @param names,groups The column names and group columns the table would get
# @keywords internal
check_dup_names_ungrouped <- function(names, groups, call = rlang::caller_env()) {
  if (length(groups) > 0L && anyDuplicated(names) > 0L) {
    rlang::abort(c(
      "Can't give a grouped tbl_gpu duplicated column names.",
      i = "`ungroup()` first, or pick unique names."
    ), class = "cuplyr_error_duplicate_names", call = call)
  }
  invisible()
}

# dplyr's join_cols() check: every join refuses duplicated names on either
# side. Called at each join verb's entry and again by push_join().
# @keywords internal
check_unique_join_inputs <- function(x, y, call = rlang::caller_env()) {
  for (side in c("x", "y")) {
    nms <- current_schema(if (side == "x") x else y)$names
    dup <- anyDuplicated(nms)
    if (dup > 0L) {
      rlang::abort(c(
        sprintf("Input columns in `%s` must be unique.", side),
        x = sprintf("Problem with `%s`.", nms[[dup]])
      ), call = call)
    }
  }
  invisible()
}

#' Can a node run on a table with duplicated column names?
#'
#' On such a table a column name is ambiguous, so any node that finds a
#' column by name (a predicate, an expression, a sort or join key, a by-name
#' select) would silently take the first of the same-named columns.
#' `push_op()` refuses every node on a duplicated-name input except the ones
#' listed here, which address columns by position or touch only rows:
#' - `select`/`rename` carrying `positions` (select(), relocate(), rename(),
#'   `[` and `$`/`[[` build them whenever names are duplicated);
#' - `distinct` carrying `key_positions` (distinct()/unique());
#' - `slice` without an ordering column: head()/tail(), `x[i, ]` and the
#'   row-index slices (the slice verbs themselves refuse duplicated names
#'   at their entry, as dplyr does);
#' - `barrier`.
#' A new node type, or a by-name variant of these, fails closed.
#'
#' @param node An AST node about to be pushed
#' @return `TRUE` when `node` is position-correct
#' @keywords internal
node_is_position_safe <- function(node) {
  switch(node$type,
    select = ,
    rename = !is.null(node$positions),
    distinct = !is.null(node$key_positions),
    slice = is.null(node$order_col),
    barrier = TRUE,
    FALSE
  )
}

#' Build the AST node that a new op should be layered on top of
#'
#' - lazy + pending ops: returns `.data$lazy_ops` as-is.
#' - lazy + no pending ops: returns a fresh `ast_source(.data$schema)`,
#'   matching the historical lazy-verb convention (e.g. the now-deleted
#'   `filter_lazy()`, `select_lazy()`) of leaving the source pointer unset; `compute()`/
#'   `collect()` supply `.data$ptr` as the lowering fallback `source_ptr`.
#' - eager: returns a fresh `ast_source(.data$schema, source_ptr = .data$ptr)`
#'   so `push_op()` can call `lower_and_execute()` immediately without a
#'   separate source table.
#'
#' @param .data A `tbl_gpu` object
#' @return An `ast_node` to use as the `input` of a new node
#' @keywords internal
input_node <- function(.data) {
  if (identical(.subset2(.data, "exec_mode"), "lazy")) {
    if (has_pending_ops(.data)) {
      return(.subset2(.data, "lazy_ops"))
    }
    return(ast_source(.subset2(.data, "schema")))
  }

  ast_source(.subset2(.data, "schema"), source_ptr = .subset2(.data, "ptr"))
}

#' Factor levels by column position
#'
#' `schema$factor_levels` is a list named by column. When a table has
#' duplicated column names (positional select()/relocate()/rename()) a name no
#' longer identifies a column, so an entry belongs to the n-th `DICTIONARY32`
#' column carrying its name: entries are kept in column order, and a duplicated
#' name may repeat in the list. Unique names keep the plain by-name lookup.
#'
#' Only `levels_from_positions()` lays a list out in column order, and it tags
#' the list (`attr(, "by_position")`) when the names repeat. A list for
#' duplicated names without that tag was written by name somewhere, so its
#' order says nothing about which column an entry belongs to: that is an error
#' rather than a guess. Every writer that can produce duplicated names
#' (`names<-`, positional nodes in `propagate_factor_levels()`, `tbl_gpu()`,
#' `bind_cols()`) goes through `levels_from_positions()`.
#'
#' @param schema A schema with `names`, `types` and `factor_levels`
#' @return A list the length of `schema$names`: the level vector of a factor
#'   column, `NULL` for any other
#' @keywords internal
levels_by_position <- function(schema) {
  nms <- schema$names
  fl <- schema$factor_levels
  if (is.null(fl)) {
    return(vector("list", length(nms)))
  }
  if (!anyDuplicated(nms)) {
    return(unname(fl[nms]))
  }
  dict <- which(unname(schema$types) == "DICTIONARY32")
  if (!isTRUE(attr(fl, "by_position")) || length(fl) != length(dict)) {
    rlang::abort(c(
      "Internal error: factor levels of duplicated column names are not tracked by position.",
      i = "Please report this to the cuplyr maintainers."
    ), class = "cuplyr_error_internal")
  }
  out <- vector("list", length(nms))
  fl_names <- names(fl)
  seen <- list()
  for (i in dict) {
    nm <- nms[[i]]
    k <- (seen[[nm]] %||% 0L) + 1L
    seen[[nm]] <- k
    hit <- which(fl_names == nm)[k]
    if (is.na(hit)) {
      rlang::abort(c(
        "Internal error: factor levels of duplicated column names are not tracked by position.",
        i = "Please report this to the cuplyr maintainers."
      ), class = "cuplyr_error_internal")
    }
    out[i] <- list(fl[[hit]])
  }
  out
}

#' Rebuild `schema$factor_levels` from per-position level vectors
#'
#' The inverse of `levels_by_position()`: entries in column order, named by
#' the output column names (which may repeat). When they repeat, the list is
#' tagged `by_position` so `levels_by_position()` can trust its order.
#'
#' @param names Output column names
#' @param by_pos List the length of `names` (`NULL` for a non-factor column)
#' @return A named list, or `NULL` when no column is a factor
#' @keywords internal
levels_from_positions <- function(names, by_pos) {
  keep <- !vapply(by_pos, is.null, logical(1))
  if (!any(keep)) {
    return(NULL)
  }
  out <- stats::setNames(by_pos[keep], names[keep])
  if (anyDuplicated(names) > 0L) {
    attr(out, "by_position") <- TRUE
  }
  out
}

#' Restrict factor level metadata to columns that survive an op
#'
#' Implements the factor_levels half of D3: `schema$factor_levels` is
#' always recomputed from the *input* schema's factor_levels, filtered to
#' the names present in the node's output schema. This fixes the bug where
#' eager `select()` of a factor column dropped `factor_levels`, causing
#' `collect()` to return integer codes instead of a factor.
#'
#' @param node The AST node being applied (used to translate factor_levels
#'   keys through a `rename` node's name map, see `ast_rename()`,
#'   `R/ast.R` -- without this, a renamed factor column's levels would be
#'   silently dropped here since its key in `input_schema$factor_levels`
#'   is the OLD name, which no longer appears in `surviving_names`)
#' @param input_schema The schema of the op's input (pre-op)
#' @param surviving_names Character vector of column names in the new schema
#' @return A named list of factor levels, or `NULL` if none survive
#' @keywords internal
propagate_factor_levels <- function(node, input_schema, surviving_names) {
  factor_levels <- input_schema$factor_levels
  if (is.null(factor_levels) || length(factor_levels) == 0) {
    return(NULL)
  }

  # Duplicated names: the levels follow their column by position, the way
  # the select/rename itself addresses it.
  # slice and barrier keep every column in place.
  # A select always goes by position: `select(x = y)` drops the old `x`, and a
  # by-name filter would hand the new `x` the dropped factor's levels.
  if (identical(node$type, "select") ||
      (node$type %in% c("rename", "distinct", "slice", "barrier") &&
       (anyDuplicated(input_schema$names) > 0L || anyDuplicated(surviving_names) > 0L))) {
    by_pos <- levels_by_position(input_schema)
    idx <- switch(node$type,
      select = node$positions %||% match(node$columns, input_schema$names),
      distinct = if (!isTRUE(node$keep_all)) {
        node$key_positions %||% match(node$key_cols, input_schema$names)
      },
      NULL
    )
    if (!is.null(idx)) by_pos <- by_pos[idx]
    return(levels_from_positions(surviving_names, by_pos))
  }

  if (identical(node$type, "rename")) {
    nm <- names(factor_levels)
    idx <- match(nm, node$old_names)
    renamed <- !is.na(idx)
    nm[renamed] <- node$new_names[idx[renamed]]
    names(factor_levels) <- nm
  }

  if (identical(node$type, "summarise")) {
    # F07: an aggregate output only keeps a factor column's levels when it IS
    # that column carried through first()/last()/nth(); an aggregate named
    # like a factor (typically a grouping key it replaced) is a plain
    # integer/double column.
    for (agg in node$aggregations) {
      carries <- agg$fn %in% c("first", "last", "nth") &&
        identical(agg$input_col, agg$output_col)
      if (!carries) {
        factor_levels[[agg$output_col]] <- NULL
      }
    }
  }

  # A bare copy of a factor column (`mutate(h = f)`) carries its levels to
  # the output name; any other expression stops a replaced column being a
  # factor (mutate_output_levels(), R/mutate.R).
  if (identical(node$type, "mutate")) {
    for (expr in node$expressions) {
      factor_levels[[expr$output_col]] <- mutate_output_levels(factor_levels, expr)
    }
  }

  factor_levels <- factor_levels[names(factor_levels) %in% surviving_names]

  if (length(factor_levels) == 0) {
    return(NULL)
  }

  factor_levels
}

#' Propagate group columns across an AST node
#'
#' A `join` keeps `x`'s groups that survive in its output (`push_join()`
#' computes that itself; this branch is only the defensive default). `summarise`
#' yields whatever `node$result_groups` says (Phase 6, task 3 -- computed by
#' `resolve_summarise_groups()`, R/summarise.R, from the node's `.groups=`
#' semantics; `character()` for `.groups = "drop"`/an ungrouped input, all
#' pre-summarise groups for `"keep"`, all-but-the-last for the default/
#' `"drop_last"`). Every other node type keeps whichever of the existing
#' `groups` still appear in the node's (post-op) schema. This single rule
#' reproduces today's per-verb behavior (select drops de-selected group
#' columns; filter/mutate/arrange preserve all of them).
#'
#' `rename` needs one extra step before that final `intersect()`: a
#' group column that got renamed (e.g. `rename(gdf, grp = g)` where `g` is
#' grouped) must be translated to its NEW name first, or the plain
#' `intersect()` would miss it entirely (`schema$names` has the new name,
#' `groups` still has the old one) and silently drop the grouping.
#'
#' @param node The AST node being applied
#' @param groups Character vector of the input's group columns
#' @param schema The node's output schema (post-op), as from `infer_schema()`
#' @return Character vector of group columns to carry forward
#' @keywords internal
propagate_groups <- function(node, groups, schema) {
  abort_positional_rename_with_groups(node, groups)
  type <- .subset2(node, "type")
  if (identical(type, "summarise")) {
    return(set_intersect(node$result_groups, schema$names))
  }
  # reframe() always returns an ungrouped table
  if (identical(type, "reframe")) {
    return(character())
  }

  # (an ungrouped table: what set_intersect() gives for no groups)
  if (length(groups) == 0L && is.character(groups) && is.character(schema$names)) {
    return(character())
  }

  if (identical(type, "rename")) {
    idx <- match(groups, node$old_names)
    renamed <- !is.na(idx)
    groups[renamed] <- node$new_names[idx[renamed]]
  }

  set_intersect(groups, schema$names)
}

# A positional rename (duplicated names) can't map group columns by name. The
# invariant above keeps it off grouped tables, so this is unreachable.
# @keywords internal
abort_positional_rename_with_groups <- function(node, groups) {
  if (length(groups) > 0L && identical(.subset2(node, "type"), "rename") &&
      !is.null(node$positions)) {
    rlang::abort("Internal error: positional rename on a grouped tbl_gpu.",
                 class = "cuplyr_error_internal")
  }
  invisible()
}

#' Refuse verbs on a zero-column table that has rows
#'
#' A cudf table with no columns always has 0 rows, so a zero-column `tbl_gpu`
#' carries its row count on the R side (`x$nrow`, see `new_tbl_gpu()`). Verbs
#' lower to cudf calls that cannot see it, so running one on such a table
#' would silently return a table with the wrong row count. Until zero-column
#' tables are supported by every verb, fail loudly instead. Called by
#' `push_op()`, `push_join()` (both sides), the binds and `vec_proxy()`;
#' `summarise()` and the `slice()` family handle this input before it reaches
#' `push_op()`.
#'
#' @param .data A `tbl_gpu`
#' @param input_schema `current_schema(.data)`
#' @param verb Character, the node type, for the message
#' @keywords internal
check_zero_column_input <- function(.data, input_schema, verb) {
  if (length(input_schema$names) == 0) {
    n <- .subset2(.data, "nrow")
    if (is.null(n)) {
      stop("`", verb, "()` is not supported on a tbl_gpu with no columns ",
           "and an unknown number of rows.", call. = FALSE)
    }
    if (n > 0) {
      stop("`", verb, "()` is not supported on a tbl_gpu with no columns but ",
           n, " rows.", call. = FALSE)
    }
  }
  invisible(NULL)
}

#' A zero-column result with a known number of rows
#'
#' The one place verbs that end with no columns (`select()`, `transmute()`,
#' `[`, `distinct()`) build their result: an ungrouped zero-column `tbl_gpu`
#' of `n` rows in the input's exec_mode. `n` defaults to the input's row
#' count; pending lazy operations are computed first to get it (a
#' materialization point, as for `summarise()` with no expressions).
#'
#' @param .data The input `tbl_gpu`
#' @param verb Character, for the plan-break notification
#' @param n Row count of the result, or `NULL` for `.data`'s own
#' @keywords internal
zero_column_result <- function(.data, verb, n = NULL) {
  if (is.null(n)) {
    n <- zero_column_nrow(.data, verb)
  }
  carry_frame_meta(
    tbl_gpu(tibble::tibble(.rows = n), lazy = identical(.subset2(.data, "exec_mode"), "lazy")),
    .data
  )
}

# Row count of `.data`, computing pending lazy operations to get it.
# @keywords internal
zero_column_nrow <- function(.data, verb) {
  if (has_pending_ops(.data)) {
    cuplyr_plan_break_notify(
      verb, "a result with no columns keeps only its row count, so the pending steps run first",
      data = .data
    )
    .data <- compute(.data)
  }
  n <- base_dim(.data)[1]
  if (is.na(n)) {
    stop("`", verb, "()` is not supported on a tbl_gpu with no columns ",
         "and an unknown number of rows.", call. = FALSE)
  }
  n
}

# Names and types of `node`, a verb's node over `input_node(.data)`, from the
# schema of that input alone. A verb may push a short chain of fresh nodes
# (select_node under a rename, ...): the walk goes down the fresh nodes only,
# stopping at `base` (the pending plan, by reference) or at a source, so the
# cost is the chain's length, never the plan's depth.
# @param base `.data$lazy_ops`, or NULL when `.data` has no pending ops
# @keywords internal
infer_push_schema <- function(node, input_schema, base) {
  input <- .subset2(node, "input")
  if (!is.null(input) && !(!is.null(base) && is_reference(input, base))) {
    if (identical(.subset2(input, "type"), "source")) {
      # `input_node(.data)`'s source; one that is not (a node built over another
      # table's plan) is inferred from its own plan, as before
      if (!identical(.subset2(.subset2(input, "schema"), "names"), input_schema$names)) {
        return(infer_schema(node))
      }
    } else {
      input_schema <- infer_push_schema(input, input_schema, base)
    }
  }
  infer_schema_from(node, input_schema)
}

#' Apply an AST node to a tbl_gpu, dispatching on exec_mode
#'
#' The only exec_mode branch point (D1). Lazy mode stores `node` as the new
#' `$lazy_ops` and does no GPU work; eager mode lowers and executes `node`
#' immediately. Both modes uniformly compute `schema` from `input_schema` alone
#' (via `infer_schema_from()`, with factor_levels reattached per D3) and `groups` (via
#' `propagate_groups()`).
#'
#' @param .data A `tbl_gpu` object
#' @param node The new AST node (already wrapping `input_node(.data)`)
#' @param input_schema `current_schema(.data)`, when the caller already has it
#'   (a node chain, `push_chain()`: the previous push's `$schema`)
#' @param carry `FALSE` leaves the frame metadata (attributes, bare class) off
#'   the result: a step of a `push_chain()`, which carries it once at the end
#' @return A new `tbl_gpu` object
#' @keywords internal
push_op <- function(.data, node, input_schema = current_schema(.data), carry = TRUE) {
  type <- .subset2(node, "type")
  check_zero_column_input(.data, input_schema, type)
  if (anyDuplicated(input_schema$names) > 0L && !node_is_position_safe(node)) {
    abort_duplicate_names(rlang::caller_env())
  }

  base <- if (has_pending_ops(.data)) .subset2(.data, "lazy_ops")
  new_schema <- propagate_schema_meta(node, input_schema, infer_push_schema(node, input_schema, base))

  new_groups <- propagate_groups(node, .subset2(.data, "groups"), new_schema)
  check_dup_names_ungrouped(new_schema$names, new_groups)
  # `.drop` belongs to the grouping: kept while any group column survives, back
  # to the default once the table is ungrouped (a later group_by() starts over).
  new_group_drop <- if (length(new_groups) > 0L) .subset2(.data, "group_drop") %||% TRUE else TRUE
  new_keys <- carry_fixed_keys(node, .data, new_groups)
  new_unspec <- if (type %in% c("filter", "arrange", "slice", "barrier")) {
    .subset2(.data, "unspecified")
  }
  # F49: a rowwise table stays rowwise through every node but the ones that
  # rebuild the rows or their grouping (their verbs never reach push_op() with a
  # rowwise input: they run through the whole-call fallback or build their own).
  new_rowwise <- is_rowwise_gpu(.data) && !type %in% c("summarise", "reframe", "distinct")

  # F31: summarise() rebuilds the frame, so only the bare data.frame class
  # survives it; every other node is dplyr_reconstruct(out, .data).
  keep_attrs <- !identical(type, "summarise")

  # A table with only its names and class as attributes (the usual one) hands
  # nothing over but the bare data.frame class, which an ungrouped input gives
  # to an ungrouped result: new_tbl_gpu() sets that field itself, which is what
  # carry_frame_meta() would do with one more copy of the table.
  plain_frame <- carry && identical(names(attributes(.data)), c("names", "class"))
  bare_df <- plain_frame && isTRUE(.subset2(.data, "bare_df")) &&
    length(.subset2(.data, "groups")) == 0L && !isTRUE(.subset2(.data, "rowwise"))

  if (identical(.subset2(.data, "exec_mode"), "lazy")) {
    out <- new_tbl_gpu(
      ptr = .subset2(.data, "ptr"),
      schema = new_schema,
      lazy_ops = node,
      groups = new_groups,
      exec_mode = "lazy",
      group_drop = new_group_drop,
      nrow = .subset2(.data, "nrow"),
      fixed_keys = new_keys,
      unspecified = new_unspec,
      bare_df = bare_df,
      rowwise = new_rowwise
    )
    return(if (carry && !plain_frame) carry_frame_meta(out, .data, attrs = keep_attrs) else out)
  }

  new_ptr <- wrap_gpu_call(type, lower_and_execute(node, .subset2(.data, "ptr")))

  out <- new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = new_groups,
    exec_mode = .subset2(.data, "exec_mode"),
    group_drop = new_group_drop,
    nrow = .subset2(.data, "nrow"),
    fixed_keys = new_keys,
    unspecified = new_unspec,
    bare_df = bare_df,
    rowwise = new_rowwise
  )
  if (plain_frame) out else carry_frame_meta(out, .data, attrs = keep_attrs)
}

#' Run a verb's node chain as ONE eager execution
#'
#' A few eager verbs push several nodes (a window mutate is window -> mutate ->
#' select). Pushed one by one, each node builds a full table; here `build()` gets
#' a lazy view of `.data` (so its `push_op()`s only do schema work), and the
#' chain it returns is lowered once, with no optimizer (as for any eager node),
#' into one result table. Fields and groups come from the lazy pushes, which
#' derive them exactly as the eager ones do; the frame metadata (attributes, bare
#' class) is carried once from `.data` at the end, which is what carrying it
#' through each step comes to. A lazy `.data` is just handed to `build()`.
#'
#' @param .data A `tbl_gpu`
#' @param build `function(x)` pushing nodes on `x` with `push_op(carry = FALSE)`;
#'   returns the resulting `tbl_gpu`
#' @return A `tbl_gpu` in `.data`'s exec_mode
#' @keywords internal
push_chain <- function(.data, build) {
  if (!identical(.subset2(.data, "exec_mode"), "eager")) {
    return(carry_frame_meta(build(.data), .data))
  }
  view <- set_tbl_gpu_field(.data, "exec_mode", "lazy")
  res <- build(view)
  node <- .subset2(res, "lazy_ops")
  new_ptr <- .subset2(.data, "ptr")
  if (inherits(node, "ast_node")) {
    memo_prev <- .schema_memo$plain
    on.exit(.schema_memo$plain <- memo_prev, add = TRUE)
    schema_memo_open()
    ops <- character()
    n <- node
    while (!is.null(n) && !identical(n$type, "source")) {
      ops <- c(n$type, ops)
      n <- n$input
    }
    new_ptr <- wrap_gpu_call(paste(ops, collapse = "+"), lower_and_execute(node, new_ptr))
  }
  out <- unclass(res)
  out$ptr <- new_ptr
  out["lazy_ops"] <- list(NULL)
  out$exec_mode <- "eager"
  class(out) <- oldClass(res)
  carry_frame_meta(out, .data)
}

# The fixed group keys (`slice(.preserve = TRUE)`, see fixed_keys_of()) that
# survive `node`: a column-only node (select/rename/mutate and its window
# stages, relocate is a select) that leaves every group column's values and
# the grouping alone, or (F33) an arrange/distinct of a `.drop = FALSE` table,
# which dplyr's dplyr_row_slice() leaves with the whole group structure (with
# `.drop = TRUE` it trims the empty groups, so they are dropped here too).
# Anything else returns NULL, so the keys can never end up on a table with
# different groups.
# @keywords internal
carry_fixed_keys <- function(node, .data, new_groups) {
  abort_positional_rename_with_groups(node, .subset2(.data, "groups"))
  keys <- fixed_keys_of(.data)
  if (is.null(keys)) {
    return(NULL)
  }
  if (node$type %in% c("arrange", "distinct")) {
    if (!isFALSE(.subset2(.data, "group_drop"))) return(NULL)
  } else if (!node$type %in% c("select", "rename", "mutate", "window")) {
    return(NULL)
  }
  old <- .subset2(.data, "groups")
  outputs <- switch(node$type,
    mutate = vapply(node$expressions, function(e) e$output_col, character(1)),
    window = vapply(node$specs, function(s) s$output_col, character(1)),
    character()
  )
  if (any(outputs %in% old)) {
    return(NULL)
  }
  expected <- old
  if (identical(node$type, "rename")) {
    idx <- match(old, node$old_names)
    expected[!is.na(idx)] <- node$new_names[idx[!is.na(idx)]]
  }
  if (!identical(new_groups, expected)) {
    return(NULL)
  }
  names(keys) <- new_groups
  keys
}

#' Build one side of a join's AST, always anchoring the source pointer
#'
#' Like `input_node()`, but *unconditionally* attaches `tbl$ptr` to the leaf
#' `ast_source` node, even when `tbl` is lazy with no pending ops (the case
#' where `input_node()` deliberately leaves the pointer unset and relies on
#' `lower_and_execute()`'s single `source_ptr` argument as a fallback, filled
#' in later by `compute()`/`collect()`). A join has *two* independent input
#' trees, so that single-fallback mechanism can't serve both leaves --
#' instead, each side's own concrete GPU pointer must be anchored explicitly
#' at construction time. This is safe because `tbl$ptr` is always a valid
#' pointer to that side's own (possibly stale, pre-pending-ops) base table:
#' `push_op()` carries it forward unchanged across lazy operations, only
#' `compute()`/`collect()` ever replace it.
#'
#' @param tbl A `tbl_gpu` object (one side of a join)
#' @return An `ast_node` with its leaf `source_ptr` set to `tbl$ptr`
#' @keywords internal
join_input_node <- function(tbl) {
  node <- if (has_pending_ops(tbl)) tbl$lazy_ops else ast_source(tbl$schema)
  set_ast_source_ptr(node, tbl$ptr)
}

# Does a join's `unmatched` / `relationship` check need the joined rows?
# ("many-to-many" and NULL, the advisory default, check nothing that stops the
# join.)
# @keywords internal
join_check_reads_data <- function(unmatched, relationship) {
  any(unmatched != "drop") ||
    (!is.null(relationship) &&
       relationship %in% c("one-to-one", "one-to-many", "many-to-one"))
}

#' Apply a two-input join AST node, dispatching on exec_mode
#'
#' The join analogue of `push_op()` (D1). Joins can't reuse `push_op()`
#' as-is because it is built around a single `.data` input: it reads exactly
#' one `exec_mode`/`ptr`/`schema` to decide the schedule. A join has two
#' independent inputs (`x`/`y`) whose *either* side being lazy forces the
#' whole join lazy (mirroring the pre-unification behavior), and whose eager
#' execution needs both sides' pointers anchored -- see `join_input_node()`.
#'
#' Groups (F06): every dplyr join ends with `dplyr_reconstruct(out, x)`, so
#' for EVERY join type the result is grouped by `intersect(x$groups,
#' names(out))` and keeps `x`'s `.drop` setting (`x$group_drop`); `y`'s
#' grouping is irrelevant. A group column that gets suffixed away (a name
#' collision, or the key under `keep = TRUE`) simply drops out of the
#' intersection; a merged key that is also a group column stays grouped (with
#' its coerced type). Semi/anti outputs are `x`'s schema verbatim, so all of
#' `x`'s groups survive.
#'
#' @param join_type One of `"left"`, `"inner"`, `"full"`, `"right"`,
#'   `"semi"`, `"anti"`, `"cross"`
#' @param x,y The two `tbl_gpu` join inputs
#' @param join_spec List with `left`/`right`/`op`/`filter` (and, when key
#'   types differ, `key_casts`), as returned by `parse_join_by()` /
#'   `resolve_join_key_types()`
#' @param suffix,keep,na_matches,multiple,unmatched,relationship As
#'   documented on the join verbs (all are unused for `"semi"`/`"anti"`,
#'   which have no `suffix`/`keep`/`multiple`/`unmatched`/`relationship` in
#'   dplyr's own signature; `"cross"` has none of `keep`/`na_matches`/
#'   `multiple`/`unmatched`/`relationship` either, and always passes `keep =
#'   TRUE` internally purely to get `build_join_output_info()`'s
#'   all-common-names-suffixed behavior -- see `cross_join.tbl_gpu()`,
#'   `R/join.R`)
#' @param cross_by `TRUE` for dplyr's deprecated `by = character()` spelling:
#'   an ordinary `join_type` join over a constant key (see
#'   `lower_cross_by_join()`, `R/lower.R`)
#' @return A new `tbl_gpu`
#' @keywords internal
push_join <- function(join_type, x, y, join_spec, suffix, keep, na_matches,
                      multiple = "all", unmatched = "drop", relationship = NULL,
                      cross_by = FALSE) {
  # Joins lower to cudf, which sees no rows in a zero-column side, so a
  # `by = character()` join against one would silently get the row count
  # wrong (F07).
  verb <- paste0(join_type, "_join")
  check_unique_join_inputs(x, y, rlang::caller_env())
  check_zero_column_input(x, current_schema(x), verb)
  check_zero_column_input(y, current_schema(y), verb)

  # F71: a filtering join slices x's rows, so (dplyr_row_slice()) a grouped
  # `.drop = FALSE` x keeps its whole group structure, emptied groups included.
  # row_slice_finish() fixes x's keys on the result when a group can be lost;
  # that reads x's data, so pending ops are computed first (as filter_impl()
  # does) and the join result stays lazy if either side is.
  prep <- row_slice_prepare(x, eligible = join_type %in% c("semi", "anti"))
  x <- prep$data

  join_ast <- ast_join(
    join_type,
    join_input_node(x),
    join_input_node(y),
    join_spec,
    keep = keep,
    suffix = suffix,
    na_matches = na_matches,
    multiple = multiple,
    unmatched = unmatched,
    relationship = relationship,
    cross_by = cross_by
  )

  # both sides' schemas are stored on x and y: no walk of either plan
  new_schema <- infer_join_schema(join_ast, current_schema(x), current_schema(y))

  new_groups <- set_intersect(x$groups, new_schema$names)
  # As in push_op(): `.drop` goes with the last surviving group column.
  new_group_drop <- if (length(new_groups) > 0L) x$group_drop %||% TRUE else TRUE

  # Filtering joins return x verbatim, so its unspecified() columns survive.
  join_unspec <- if (join_type %in% c("semi", "anti")) x[["unspecified"]]

  # F31: x's attributes and class win (dplyr_reconstruct(out, x)).
  if (identical(x$exec_mode, "lazy") || identical(y$exec_mode, "lazy")) {
    # The size estimate reads the row counts of the two uploaded tables, so it
    # is made only while neither side has pending steps (their row counts are
    # not known before the plan runs).
    if (!has_pending_ops(x) && !has_pending_ops(y)) {
      warn_if_join_too_large(if (cross_by) "cross" else join_type, x, y, join_spec,
                             suffix, keep)
    }
    out <- new_tbl_gpu(
      ptr = NULL,
      schema = new_schema,
      lazy_ops = join_ast,
      groups = new_groups,
      exec_mode = "lazy",
      group_drop = new_group_drop,
      unspecified = join_unspec,
      rowwise = is_rowwise_gpu(x)
    )
    out <- row_slice_finish(carry_frame_meta(out, x), prep)
    # `unmatched = "error"` and a one-to-one / one-to-many / many-to-one
    # `relationship` are checks on the join's rows, and dplyr raises them at the
    # verb: the join runs now (the result is the computed table, so it is not
    # run twice).
    if (join_check_reads_data(unmatched, relationship)) {
      cuplyr_plan_break_notify(
        verb, "its unmatched = \"error\" / relationship check reads the joined rows, which dplyr raises at the verb",
        data = out, own_step = TRUE
      )
      out <- compute(out)
    }
    return(out)
  }

  warn_if_join_too_large(if (cross_by) "cross" else join_type, x, y, join_spec,
                         suffix, keep)

  new_ptr <- wrap_gpu_call(
    paste0(join_type, "_join"),
    lower_and_execute(join_ast, x$ptr)
  )

  out <- new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = new_groups,
    exec_mode = "eager",
    group_drop = new_group_drop,
    unspecified = join_unspec,
    rowwise = is_rowwise_gpu(x)
  )
  row_slice_finish(carry_frame_meta(out, x), prep)
}

#' Auto-name unnamed dots, warning as dplyr's verbs do
#'
#' `mutate()` and `summarise()` both accept unnamed expressions (e.g.
#' `mutate(mpg * 2)`) and, matching dplyr, fall back to the expression's
#' deparsed text as the output column name -- while warning that an explicit
#' name would be clearer. Historically this logic was duplicated verbatim
#' inside both `mutate.tbl_gpu()` (twice -- once for its since-deleted eager
#' path, once for its since-deleted lazy path) and `summarise_lazy()` (now
#' also deleted); this is the single shared implementation, parameterized by
#' `verb` so each caller's warning names itself correctly.
#'
#' @param dots A list of quosures, as returned by `rlang::enquos(...)`
#' @param verb Character, the verb name to use in the warning text (e.g.
#'   `"mutate"`, `"summarise"`)
#' @return `dots`, with `names(dots)` filled in for any unnamed entries
#' @keywords internal
auto_name_dots <- function(dots, verb) {
  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }

  for (i in seq_along(dots)) {
    new_name <- dot_names[i]

    if (is.na(new_name) || new_name == "") {
      # as_label() is dplyr's own naming rule (a bare `.data$x` is `x`).
      new_name <- rlang::as_label(dots[[i]])
      # A `.data$x` / `.data[["x"]]` dot is a column reference: no warning.
      if (!is_data_pronoun_call(rlang::quo_get_expr(dots[[i]]))) {
        warning("Unnamed ", verb, " expression '", new_name, "' will use expression as column name.\n",
                "Consider using explicit names: ", verb, "(name = ", new_name, ")",
                call. = FALSE)
      }
      dot_names[i] <- new_name
    }
  }

  names(dots) <- dot_names
  dots
}

#' Resolve a verb's `.by=`/`by=` quosure into group column names
#'
#' Implements `scratchpad/phase5_window_design.md` section 6: the shared
#' resolution logic for on-the-fly grouping across `mutate()`/`filter()`
#' (and, from task W7 onward, the `slice()` family). `.by=`/`by=` is
#' mutually exclusive with an already-grouped `.data` -- verified
#' empirically against dplyr 1.2.1, both `mutate()` and `filter()` raise
#' the identical message below verbatim (including the backticks) when
#' `.data` is grouped and `.by=` is also supplied.
#'
#' @param .data A `tbl_gpu` object
#' @param by_quo A quosure for the verb's `.by=`/`by=` argument, as
#'   returned by `rlang::enquo(.by)` -- NOT yet evaluated
#' @param arg_name Character scalar, the argument's name as written in the
#'   verb's own signature (`".by"` for `mutate()`/`filter()`/`slice()`,
#'   `"by"` for the `slice_*()` family) -- interpolated into the
#'   grouped-`.data` error message so it names the argument the caller
#'   actually used
#' @param call Environment of the user-facing verb, the call the error reports
#' @param verb The verb's name for the CPU-fallback notice, when it is not the
#'   name of `call`'s frame
#' @return A character vector of column names: `character()` when `by_quo`
#'   is NULL (absent or explicit `.by = NULL`, matching dplyr: both behave
#'   identically, see the roxygen on the calling verb), otherwise the
#'   tidyselect-resolved column names in selection order (via
#'   `tidyselect::eval_select()`'s own `names()`, so `c(g, h)`,
#'   `starts_with("g")`, etc. all work exactly as they do for real
#'   dplyr's own `.by=`)
#' @keywords internal
resolve_by <- function(.data, by_quo, arg_name = ".by", call = rlang::caller_env(), verb = NULL) {
  if (rlang::quo_is_null(by_quo)) {
    return(character())
  }

  if (length(.subset2(.data, "groups")) > 0) {
    rlang::abort(paste0("Can't supply `", arg_name,
                        "` when `.data` is a grouped data frame."), call = call)
  }

  # Same rule as select(): a selection made of names and base type tests is
  # answered by the typed proxy; anything that may read values runs on the
  # real data (only the names come back, so the verb itself stays native).
  schema <- current_schema(.data)
  if (selection_is_type_only(by_quo, NULL, schema$names)) {
    loc <- tidyselect::eval_select(by_quo, tidyselect_proxy(schema))
  } else {
    # (`verb`: the user's verb, when the caller's frame is an internal helper
    # such as mutate_run() whose name is not it)
    verb <- verb %||% tryCatch(rlang::call_name(rlang::frame_call(call)), error = function(e) NULL)
    data <- if (has_pending_ops(.data)) compute(.data) else .data
    size <- fallback_size(data)
    cuplyr_fallback_notify(
      verb %||% arg_name, reason = selection_fallback_reason,
      rows = size$rows, cols = size$cols, bytes = size$bytes
    )
    loc <- tidyselect::eval_select(by_quo, collect_frame(data))
  }
  names(loc)
}

# F42: run `attempt()`, verb pushes that may raise `cuplyr_error_sum_overflow`
# (an eager table executes them on the spot), and run `on_overflow()` instead
# if one does. The attempt's warnings are held back: they're re-raised when the
# attempt succeeds and dropped when it overflows, since the rerun raises its
# own (`mutate(s = sum(x + 1L))` would otherwise warn twice), except a notice
# about an invalid option, which is only said once. `class` names the
# condition to catch (summarise() uses its own, see resolve_agg_window_arg()).
# @keywords internal
retry_on_sum_overflow <- function(attempt, on_overflow, class = "cuplyr_error_sum_overflow") {
  held <- list()
  on.exit(for (w in held) warning(w), add = TRUE)
  overflowed <- FALSE
  on_class <- function(e) {
    overflowed <<- TRUE
    NULL
  }
  # (the two classes callers pass are spelled out: a computed handler name
  # needs rlang::inject(), which costs more than the verb's own wrapping)
  result <- if (identical(class, "cuplyr_error_sum_overflow")) {
    tryCatch(
      withCallingHandlers(attempt(), warning = function(w) {
        held[[length(held) + 1L]] <<- w
        invokeRestart("muffleWarning")
      }),
      cuplyr_error_sum_overflow = on_class
    )
  } else if (identical(class, "cuplyr_summarise_rerun")) {
    tryCatch(
      withCallingHandlers(attempt(), warning = function(w) {
        held[[length(held) + 1L]] <<- w
        invokeRestart("muffleWarning")
      }),
      cuplyr_summarise_rerun = on_class
    )
  } else {
    handlers <- list(on_class)
    names(handlers) <- class
    rlang::inject(tryCatch(
      withCallingHandlers(attempt(), warning = function(w) {
        held[[length(held) + 1L]] <<- w
        invokeRestart("muffleWarning")
      }),
      !!!handlers
    ))
  }
  if (overflowed) {
    # (an invalid-option warning is said once per session: not one to lose)
    held <- Filter(function(w) inherits(w, "cuplyr_warning_invalid_option"), held)
    return(on_overflow())
  }
  result
}

#' CPU fallbacks and plan breaks: what cuplyr tells you, and how to hear it
#'
#' cuplyr runs a dplyr verb on the GPU whenever it can. Two things can happen
#' that you may want to know about. Both are silent by default, and both are
#' always *signalled* as conditions, so a calling handler sees them whatever the
#' options say.
#'
#' @section CPU fallbacks:
#' A verb (or part of one) that has no GPU implementation runs in dplyr on the
#' host: cuplyr collects the table, runs the real dplyr verb and uploads the
#' result. This is correct but slow for a big table. The condition has class
#' `cuplyr_cpu_fallback` and the fields
#' * `verb`: the verb, as you wrote it (`"mutate"`);
#' * `expr`: the expression that forced the fallback (`"qnorm(x / 10)"`), or
#'   `NA` when the call as a whole goes;
#' * `reason`: why (`` "`qnorm()` has no GPU implementation." ``, a rowwise
#'   table, a function of yours that shadows a native one, an unsupported
#'   type, ...);
#' * `rows`, `cols`, `bytes`: what is collected to the host (the table's
#'   dimensions and an estimate of its size; a two-table verb adds both
#'   tables).
#'
#' `options(cuplyr.fallback = )` decides what happens next: `"silent"` (the
#' default), `"warn"` (a warning of class `cuplyr_warning_cpu_fallback`) or
#' `"error"` (an error of class `cuplyr_error_fallback`, for a script that must
#' stay on the GPU). `"warning"` is accepted as an alias of `"warn"`. The
#' message reads
#'
#' ```
#' cuplyr: mutate() fell back to CPU evaluation for: qnorm(x / 10)
#' i Reason: `qnorm()` has no GPU implementation.
#' i Collected 1,000 rows x 3 cols (about 24.4 kB) to run on the CPU.
#' ```
#'
#' @section Plan breaks:
#' A lazy table (the default, `tbl_gpu(df)`) collects its verbs into a plan and
#' runs it when you `compute()` or `collect()`. A few steps need the result of
#' the pending plan before it is asked for (a `summarise()` of an expression
#' such as `mean(x * y)`, an integer `max()`, a verb that leaves no column).
#' cuplyr then computes the plan early. This stays on the GPU and nothing is
#' collected, but the plan is not fused across that step. The condition has
#' class `cuplyr_plan_break` and the fields `verb`, `expr`, `reason` and
#' `steps` (the number of pending steps that ran early).
#'
#' `options(cuplyr.plan_break = )` is `"silent"` (the default) or `"warn"` (a
#' warning of class `cuplyr_warning_plan_break`); `"warning"` is an alias of
#' `"warn"`. A plan break is never an error: nothing left the GPU.
#'
#' @section Invalid option values:
#' Any other value of either option (including `"error"` for
#' `cuplyr.plan_break`) warns once per session and value, with class
#' `cuplyr_warning_invalid_option` (fields `option` and `value`), and the
#' default (`"silent"`) is used.
#'
#' @examples
#' \dontrun{
#' # Count the fallbacks of a pipeline without changing its behaviour
#' n <- 0L
#' res <- withCallingHandlers(
#'   my_pipeline(),
#'   cuplyr_cpu_fallback = function(cnd) {
#'     message("fallback in ", cnd$verb, ": ", cnd$reason)
#'     n <<- n + 1L
#'   }
#' )
#'
#' options(cuplyr.fallback = "warn", cuplyr.plan_break = "warn")
#' }
#' @name cuplyr-fallback
#' @aliases cuplyr-fallback cuplyr.fallback cuplyr.plan_break
#' @seealso [cuplyr_fallback_notify()], [fallback-verbs]
NULL

# The valid values of the notice options. "warning" is accepted as "warn".
cuplyr_notice_values <- list(
  cuplyr.fallback = c("silent", "warn", "error"),
  cuplyr.plan_break = c("silent", "warn")
)

# Invalid option values already warned about this session ("option value")
cuplyr_notice_state <- new.env(parent = emptyenv())

# The mode `option` ("cuplyr.fallback" or "cuplyr.plan_break") asks for:
# "silent", "warn" or "error". An unknown value warns once per session and
# value and means "silent". Runs on the notify path only.
# @keywords internal
cuplyr_notice_mode <- function(option) {
  value <- getOption(option)
  if (is.null(value)) {
    return("silent")
  }
  valid <- cuplyr_notice_values[[option]]
  if (is.character(value) && length(value) == 1L && !is.na(value)) {
    if (identical(value, "warning")) {
      value <- "warn"
    }
    if (value %in% valid) {
      return(value)
    }
  }
  shown <- paste(deparse(value, nlines = 1L), collapse = "")
  key <- paste(option, shown)
  if (is.null(cuplyr_notice_state[[key]])) {
    cuplyr_notice_state[[key]] <- TRUE
    rlang::warn(
      sprintf("`options(%s = )` is %s, which is not one of %s (or \"warning\" for \"warn\"). Using \"silent\".",
              option, shown, paste0("\"", valid, "\"", collapse = ", ")),
      class = "cuplyr_warning_invalid_option",
      option = option, value = getOption(option)
    )
  }
  "silent"
}

# Estimated bytes per element of each GPU column type (strings estimated),
# plus the validity bitmask. The one table gpu_memory_usage() uses.
# @keywords internal
gpu_type_bytes <- function(types) {
  vapply(types, function(type) {
    switch(type,
      "FLOAT64" = 8,
      "FLOAT32" = 4,
      "INT64" = 8,
      "INT32" = 4,
      "INT16" = 2,
      "INT8" = 1,
      "BOOL8" = 1,
      "STRING" = 32,  # Average estimate
      "TIMESTAMP_DAYS" = 4,
      "TIMESTAMP_MICROSECONDS" = 8,
      "TIMESTAMP_NANOSECONDS" = 8,
      "DICTIONARY32" = 4,
      8  # Default
    )
  }, numeric(1), USE.NAMES = FALSE)
}

# What a CPU fallback collects: rows, cols and an estimated byte count of the
# tbl_gpu objects in `...` (summed), from their schemas and the row counts the
# tables already carry. Reads no data and runs no plan: a table with pending
# ops reports its output columns and 0 rows (callers compute first).
# @keywords internal
fallback_size <- function(...) {
  rows <- 0
  cols <- 0
  bytes <- 0
  for (tbl in list(...)) {
    if (!is_tbl_gpu(tbl)) {
      next
    }
    types <- current_schema(tbl)$types
    n <- if (has_pending_ops(tbl)) NA_real_ else tryCatch(base_dim(tbl)[1L], error = function(e) NA_real_)
    if (is.na(n)) {
      n <- 0
    }
    rows <- rows + n
    cols <- cols + length(types)
    bytes <- bytes + sum(n * gpu_type_bytes(types)) + ceiling(n / 8) * length(types)
  }
  list(rows = rows, cols = cols, bytes = bytes)
}

# "1,000 rows x 3 cols (about 24.4 kB)"
# @keywords internal
fallback_size_text <- function(rows, cols, bytes) {
  sprintf("%s rows x %s cols (about %s)",
          format(rows, big.mark = ",", scientific = FALSE, trim = TRUE),
          format(cols, big.mark = ",", scientific = FALSE, trim = TRUE),
          format(structure(bytes, class = "object_size"), units = "auto", standard = "SI"))
}

# One-line form of a quosure's text, the `expr` of a notice.
# @keywords internal
notice_expr_text <- function(x) {
  if (rlang::is_quosure(x)) {
    x <- rlang::quo_squash(x)
  }
  if (!is.character(x)) {
    x <- rlang::expr_deparse(x, width = Inf)
  }
  trimws(gsub("\\s+", " ", paste(x, collapse = " ")))
}

#' Notify about a CPU-eval fallback, gated by `options(cuplyr.fallback = )`
#'
#' The one channel for "this ran on the CPU". It always signals a
#' `cuplyr_cpu_fallback` condition (fields `verb`, `expr`, `reason`, `rows`,
#' `cols`, `bytes`) before any CPU work; `options(cuplyr.fallback = )` then
#' decides whether that also warns or errors (default silent). See
#' [cuplyr-fallback] for the options, classes and message. A lazy plan that is
#' only computed early on the GPU goes through `cuplyr_plan_break_notify()`
#' instead.
#'
#' @param verb Character, the verb name (e.g. `"filter"`, `"summarise"`)
#' @param expr Character, the offending expression as text, or `NA` when the
#'   call as a whole goes
#' @param reason Character, why this runs on the CPU
#' @param rows,cols,bytes What is collected to the host (see `fallback_size()`)
#' @param nothing_collected `TRUE` when the CPU only evaluates an expression
#'   that reads no column
#' @return `invisible(NULL)`
#' @keywords internal
cuplyr_fallback_notify <- function(verb, expr = NA_character_, reason = NULL,
                                   rows = 0, cols = 0, bytes = 0,
                                   nothing_collected = FALSE) {
  if (is.null(reason) || length(reason) != 1L || is.na(reason) || !nzchar(reason)) {
    reason <- "this operation has no GPU implementation here."
  }
  expr <- if (is.null(expr) || length(expr) != 1L) NA_character_ else as.character(expr)
  # A verb built on another one reports as itself (fallback_notice_as())
  as <- fallback_reasons$as
  if (!is.null(as)) {
    verb <- as$verb
    if (!is.na(expr)) expr <- as$expr
  }
  head <- sprintf("cuplyr: %s() fell back to CPU evaluation for: %s",
                  verb, if (is.na(expr)) sprintf("%s()", verb) else expr)
  msg <- paste0(
    head,
    "\ni Reason: ", reason,
    "\ni ", if (nothing_collected) {
      "Nothing was collected: the expression reads no column."
    } else {
      sprintf("Collected %s to run on the CPU.", fallback_size_text(rows, cols, bytes))
    }
  )
  fields <- list(verb = verb, expr = expr, reason = reason,
                 rows = rows, cols = cols, bytes = bytes)

  # Lets a caller that can do without the CPU (summarise()'s cur_group_id()
  # path) catch the fallback. gpu_fallback() has already computed any pending
  # ops, but the CPU evaluation itself has not started.
  rlang::cnd_signal(rlang::cnd("cuplyr_cpu_fallback", message = msg, !!!fields,
                               use_cli_format = FALSE))
  mode <- cuplyr_notice_mode("cuplyr.fallback")
  switch(mode,
    "warn" = rlang::exec(rlang::warn, msg, class = "cuplyr_warning_cpu_fallback", !!!fields),
    # classed (but a plain rlang error otherwise) so wrap_gpu_call() lets it
    # through unworded: nothing failed on the GPU
    "error" = rlang::exec(rlang::abort, msg, class = "cuplyr_error_fallback", call = NULL, !!!fields),
    NULL
  )
  invisible(NULL)
}

# F80: a verb on a lazy table whose result TYPE depends on the data (an integer
# sum() that can leave R's integer range) can't be planned, since a plan has a
# fixed schema. It runs at the verb instead, the way an eager table runs it:
# the pending steps run now (a plan break, notified with `reason`), the verb
# then sees an eager table, and its result goes back to lazy with nothing
# pending (`plan_back_to_lazy()`). `.data` is a lazy `tbl_gpu`. The notice is
# about the user's steps: `notice_data` is the verb's input as the user gave it
# when `.data` holds steps of the verb's own (a cur_group_id() helper column).
# @keywords internal
plan_run_now <- function(.data, verb, reason, notice_data = .data) {
  if (has_pending_ops(notice_data)) {
    cuplyr_plan_break_notify(verb, reason, data = notice_data)
  }
  set_tbl_gpu_field(compute(.data), "exec_mode", "eager")
}

# `plan_run_now()`'s result `now` with only the columns `names`: the verb's
# input without the helper columns the verb added before the break, for a CPU
# rerun, which then starts from the computed table instead of running the
# user's plan again. It is a lazy select, run only if the rerun happens.
# @keywords internal
plan_now_columns <- function(now, names) {
  now <- set_tbl_gpu_field(now, "exec_mode", "lazy")
  push_op(now, ast_select(input_node(now), names))
}

# The lazy form of `plan_run_now()`'s eager result (anything that is not a
# gibble, such as a CPU route's data frame, is returned as it is).
# @keywords internal
plan_back_to_lazy <- function(result) {
  if (is_tbl_gpu(result) && identical(.subset2(result, "exec_mode"), "eager") &&
      !has_pending_ops(result)) {
    result <- set_tbl_gpu_field(result, "exec_mode", "lazy")
  }
  result
}

#' Notify about a plan break, gated by `options(cuplyr.plan_break = )`
#'
#' A lazy plan that had to run early, on the GPU. Always signals a
#' `cuplyr_plan_break` condition (fields `verb`, `expr`, `reason`, `steps`);
#' `options(cuplyr.plan_break = "warn")` also warns. Never an error. Call it
#' before the plan is computed: `steps` counts the pending non-source nodes of
#' `data`. See [cuplyr-fallback].
#'
#' @param verb Character, the verb name
#' @param reason Character, why the plan had to run
#' @param expr Character, the expression that forced it, or `NA`
#' @param data The lazy `tbl_gpu` whose plan is about to run
#' @param own_step `TRUE` when `data` already holds the verb's own step (a join
#'   that has to check its rows at the verb), so the notice says the verb ran
#'   with its pending steps rather than that it ran them early.
#' @return `invisible(NULL)`
#' @keywords internal
cuplyr_plan_break_notify <- function(verb, reason, expr = NA_character_, data, own_step = FALSE) {
  steps <- max(1L, ast_count_ops(.subset2(data, "lazy_ops")))
  expr <- if (is.null(expr) || length(expr) != 1L) NA_character_ else as.character(expr)
  # (as in cuplyr_fallback_notify(): count()/tally() report their own wt)
  as <- fallback_reasons$as
  if (!is.null(as)) {
    verb <- as$verb
    if (!is.na(expr)) expr <- as$expr
  }
  earlier <- if (own_step) steps - 1L else steps
  msg <- paste0(
    if (own_step && earlier == 0L) {
      sprintf("cuplyr: %s() ran on the GPU at the verb (a plan break)", verb)
    } else if (own_step) {
      sprintf("cuplyr: %s() ran on the GPU at the verb, with %d pending step%s (a plan break)",
              verb, earlier, if (earlier == 1L) "" else "s")
    } else {
      sprintf("cuplyr: %s() ran %d pending step%s on the GPU early (a plan break)",
              verb, steps, if (steps == 1L) "" else "s")
    },
    "\ni Reason: ", reason, if (!is.na(expr)) paste0(" (", expr, ")"), "."
  )
  fields <- list(verb = verb, expr = expr, reason = reason, steps = steps)
  rlang::cnd_signal(rlang::cnd("cuplyr_plan_break", message = msg, !!!fields,
                               use_cli_format = FALSE))
  if (identical(cuplyr_notice_mode("cuplyr.plan_break"), "warn")) {
    rlang::exec(rlang::warn, msg, class = "cuplyr_warning_plan_break", !!!fields)
  }
  invisible(NULL)
}
