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
  new_schema <- infer_schema(node)
  new_schema$factor_levels <- propagate_factor_levels(node, input_schema, new_schema$names)
  new_schema
}

#' Get the schema a new op should be built against
#'
#' Pending-lazy-AST-aware: if `.data` is lazy and has pending operations,
#' infers the schema from the AST (`$lazy_ops`) via `infer_schema_full()`
#' (factor_levels-aware, see its own docs); otherwise returns the table's
#' base `$schema`. This is the canonical replacement for the
#' `current_schema`/`current schema` blocks duplicated inline across verbs.
#'
#' @param .data A `tbl_gpu` object
#' @return List with `names` and `types` (and possibly `factor_levels`)
#' @keywords internal
current_schema <- function(.data) {
  if (identical(.subset2(.data, "exec_mode"), "lazy") && has_pending_ops(.data)) {
    infer_schema_full(.subset2(.data, "lazy_ops"))
  } else {
    .subset2(.data, "schema")
  }
}

# dplyr's verbs that read columns (filter, mutate, summarise, arrange, ...)
# refuse a data frame with duplicated names; select()/relocate()/rename()
# are what repair them. Verbs call this at their entry so dplyr's message
# comes before any other check; push_op() enforces the same rule on every
# node (node_is_position_safe()), so a verb that forgets still can't run.
# @keywords internal
check_unique_input_names <- function(.data, call = rlang::caller_env()) {
  if (anyDuplicated(.subset2(.data, "schema")$names) > 0L) {
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
  if (node$type %in% c("select", "rename", "distinct", "slice", "barrier") &&
      (anyDuplicated(input_schema$names) > 0L || anyDuplicated(surviving_names) > 0L)) {
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
  # factor.
  if (identical(node$type, "mutate")) {
    for (expr in node$expressions) {
      is_factor <- identical(unname(expr$output_type), "DICTIONARY32")
      src <- if (identical(expr$ir$kind, "col")) factor_levels[[expr$ir$name]] else NULL
      if (is_factor && !is.null(src)) {
        factor_levels[[expr$output_col]] <- src
      } else if (!is_factor) {
        factor_levels[[expr$output_col]] <- NULL
      }
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
  if (identical(node$type, "summarise")) {
    return(intersect(node$result_groups, schema$names))
  }

  if (identical(node$type, "rename")) {
    idx <- match(groups, node$old_names)
    renamed <- !is.na(idx)
    groups[renamed] <- node$new_names[idx[renamed]]
  }

  intersect(groups, schema$names)
}

# A positional rename (duplicated names) can't map group columns by name. The
# invariant above keeps it off grouped tables, so this is unreachable.
# @keywords internal
abort_positional_rename_with_groups <- function(node, groups) {
  if (identical(node$type, "rename") && !is.null(node$positions) && length(groups) > 0L) {
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
#' @param verb Character, for the fallback notification
#' @param n Row count of the result, or `NULL` for `.data`'s own
#' @keywords internal
zero_column_result <- function(.data, verb, n = NULL) {
  if (is.null(n)) {
    n <- zero_column_nrow(.data, verb)
  }
  tbl_gpu(tibble::tibble(.rows = n), lazy = identical(.data$exec_mode, "lazy"))
}

# Row count of `.data`, computing pending lazy operations to get it.
# @keywords internal
zero_column_nrow <- function(.data, verb) {
  if (has_pending_ops(.data)) {
    cuplyr_fallback_notify(verb, "zero columns (pending operations run first)")
    .data <- compute(.data)
  }
  n <- base_dim(.data)[1]
  if (is.na(n)) {
    stop("`", verb, "()` is not supported on a tbl_gpu with no columns ",
         "and an unknown number of rows.", call. = FALSE)
  }
  n
}

#' Apply an AST node to a tbl_gpu, dispatching on exec_mode
#'
#' The only exec_mode branch point (D1). Lazy mode stores `node` as the new
#' `$lazy_ops` and does no GPU work; eager mode lowers and executes `node`
#' immediately. Both modes uniformly recompute `schema` (via `infer_schema()`,
#' with factor_levels reattached per D3) and `groups` (via
#' `propagate_groups()`).
#'
#' @param .data A `tbl_gpu` object
#' @param node The new AST node (already wrapping `input_node(.data)`)
#' @return A new `tbl_gpu` object
#' @keywords internal
push_op <- function(.data, node) {
  input_schema <- current_schema(.data)
  check_zero_column_input(.data, input_schema, node$type)
  if (anyDuplicated(input_schema$names) > 0L && !node_is_position_safe(node)) {
    abort_duplicate_names(rlang::caller_env())
  }

  new_schema <- infer_schema(node)
  new_schema$factor_levels <- propagate_factor_levels(node, input_schema, new_schema$names)

  new_groups <- propagate_groups(node, .subset2(.data, "groups"), new_schema)
  check_dup_names_ungrouped(new_schema$names, new_groups)
  # `.drop` belongs to the grouping: kept while any group column survives, back
  # to the default once the table is ungrouped (a later group_by() starts over).
  new_group_drop <- if (length(new_groups) > 0L) .subset2(.data, "group_drop") %||% TRUE else TRUE
  new_keys <- carry_fixed_keys(node, .data, new_groups)

  if (identical(.subset2(.data, "exec_mode"), "lazy")) {
    return(new_tbl_gpu(
      ptr = .subset2(.data, "ptr"),
      schema = new_schema,
      lazy_ops = node,
      groups = new_groups,
      exec_mode = "lazy",
      group_drop = new_group_drop,
      nrow = .subset2(.data, "nrow"),
      fixed_keys = new_keys
    ))
  }

  new_ptr <- wrap_gpu_call(node$type, lower_and_execute(node, .subset2(.data, "ptr")))

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = new_groups,
    exec_mode = .subset2(.data, "exec_mode"),
    group_drop = new_group_drop,
    nrow = .subset2(.data, "nrow"),
    fixed_keys = new_keys
  )
}

# The fixed group keys (`slice(.preserve = TRUE)`, see fixed_keys_of()) that
# survive `node`: only a column-only node (select/rename/mutate and its window
# stages, relocate is a select) that leaves every group column's values and
# the grouping alone. Anything else returns NULL, so the keys can never end up
# on a table with different groups.
# @keywords internal
carry_fixed_keys <- function(node, .data, new_groups) {
  abort_positional_rename_with_groups(node, .subset2(.data, "groups"))
  keys <- fixed_keys_of(.data)
  if (is.null(keys) || !node$type %in% c("select", "rename", "mutate", "window")) {
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

  new_schema <- infer_schema(join_ast)

  new_groups <- intersect(x$groups, new_schema$names)
  # As in push_op(): `.drop` goes with the last surviving group column.
  new_group_drop <- if (length(new_groups) > 0L) x$group_drop %||% TRUE else TRUE

  if (identical(x$exec_mode, "lazy") || identical(y$exec_mode, "lazy")) {
    return(new_tbl_gpu(
      ptr = NULL,
      schema = new_schema,
      lazy_ops = join_ast,
      groups = new_groups,
      exec_mode = "lazy",
      group_drop = new_group_drop
    ))
  }

  warn_if_join_too_large(if (cross_by) "cross" else join_type, x, y, join_spec,
                         suffix, keep)

  new_ptr <- wrap_gpu_call(
    paste0(join_type, "_join"),
    lower_and_execute(join_ast, x$ptr)
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = NULL,
    groups = new_groups,
    exec_mode = "eager",
    group_drop = new_group_drop
  )
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
#' @return A character vector of column names: `character()` when `by_quo`
#'   is NULL (absent or explicit `.by = NULL`, matching dplyr: both behave
#'   identically, see the roxygen on the calling verb), otherwise the
#'   tidyselect-resolved column names in selection order (via
#'   `tidyselect::eval_select()`'s own `names()`, so `c(g, h)`,
#'   `starts_with("g")`, etc. all work exactly as they do for real
#'   dplyr's own `.by=`)
#' @keywords internal
resolve_by <- function(.data, by_quo, arg_name = ".by", call = rlang::caller_env()) {
  if (rlang::quo_is_null(by_quo)) {
    return(character())
  }

  if (length(.data$groups) > 0) {
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
    verb <- tryCatch(rlang::call_name(rlang::frame_call(call)), error = function(e) NULL)
    cuplyr_fallback_notify(verb %||% arg_name,
                           "a selection that may read column values (resolved on the data)")
    data <- if (has_pending_ops(.data)) compute(.data) else .data
    loc <- tidyselect::eval_select(by_quo, collect_frame(data))
  }
  names(loc)
}

#' Notify about a CPU-eval fallback, gated by `options(cuplyr.fallback = )`
#'
#' Per D4: default is silent (keeps existing eager behavior unchanged,
#' e.g. `filter(rep(TRUE, n))` is legitimate dplyr, not an error). Set
#' `options(cuplyr.fallback = "warn")` or `"error"` to surface fallbacks.
#'
#' @param verb Character, the verb name (e.g. `"filter"`, `"summarise"`)
#' @param expr_text Character, the offending expression as text
#' @return `invisible(NULL)`
#' @keywords internal
cuplyr_fallback_notify <- function(verb, expr_text) {
  mode <- getOption("cuplyr.fallback", "silent")
  msg <- sprintf(
    "cuplyr: %s() fell back to CPU evaluation for: %s",
    verb, expr_text
  )

  switch(mode,
    "warn" = warning(msg, call. = FALSE),
    "error" = stop(msg, call. = FALSE),
    invisible(NULL)
  )

  invisible(NULL)
}
