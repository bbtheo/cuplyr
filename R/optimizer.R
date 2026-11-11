# AST Optimizer for GPU Operations
#
# This module implements optimization passes for the lazy evaluation AST:
# 1. Projection pruning (push down selects)
# 2. Mutate fusion
# 3. Dead column pruning (drop unused mutate outputs)
# 4. Filter pushdown (across mutate when safe)
# 5. Filter reordering
#
# Filter fusion is no longer a separate pass (Phase 1 expression-engine
# cutover, task T3): `lower_filter()` always folds every predicate in a
# filter node into one `&`-combined expression and evaluates it as a single
# `compute_column()` kernel, so a filter node with N predicates is already
# exactly one kernel with no `ast$fused` flag needed.
#
# Pass order matters! Run in sequence:
# projection -> mutate fusion -> dead column pruning -> filter pushdown
# -> filter reorder

# Set helpers of the passes below: base `union()`/`setdiff()`/`intersect()` for
# the character vectors (or NULL) the passes carry, without their
# `as.vector()`/class dispatch, which the passes pay at every node. Same values
# and order as the base functions (first occurrence, names dropped).
opt_union <- function(x, y) unique.default(c(x, y))
opt_diff <- function(x, y) {
  x <- unique.default(x)
  x[match(x, y, 0L) == 0L]
}
opt_intersect <- function(x, y) {
  if (is.null(x) || is.null(y)) return(NULL)
  x <- unique.default(x)
  x[match(x, y, 0L) > 0L]
}
# The output column names of a mutate node's expressions.
mutate_output_cols <- function(expressions) {
  if (length(expressions) == 1L) {
    .subset2(.subset2(expressions, 1L), "output_col")
  } else {
    vapply(expressions, `[[`, character(1), "output_col")
  }
}
# unique(unlist(lapply(items, `[[`, field))): the columns a node's predicates or
# expressions read. One item is the common case.
node_field_union <- function(items, field) {
  if (length(items) == 1L) {
    unique.default(.subset2(.subset2(items, 1L), field))
  } else {
    unique.default(unlist(lapply(items, `[[`, field)))
  }
}

#' Optimize an AST for GPU execution
#'
#' Applies all optimization passes in the correct order.
#'
#' @param ast Root AST node
#' @return Optimized AST
#' @keywords internal
optimize_ast <- function(ast) {
  if (is.null(ast)) return(NULL)
  if (!inherits(ast, "ast_node")) {
    stop("Invalid lazy_ops: expected an AST node or NULL.", call. = FALSE)
  }

  # Schema memo (R/ast.R): the passes ask for the same nodes' schemas again
  # and again; compute() has already opened one that lowering shares (then
  # there is nothing to open, or to restore on exit).
  if (is.null(.schema_memo$plain)) {
    on.exit(.schema_memo$plain <- NULL, add = TRUE)
    schema_memo_open()
  }

  # Handle barriers: optimize segments separately
  optimize_with_barriers(ast)
}

#' Optimize AST respecting barrier nodes
#'
#' A plan is optimized one segment at a time: the nodes above a barrier (down
#' to it), then the barrier, then what is below it as a plan of its own. Each
#' segment is told what its consumer reads of its output, so the columns a
#' barrier does not need are pruned below it as well (F99).
#'
#' @param ast Root AST node
#' @param required Columns the consumer of `ast` reads from its output
#'   (`NULL`: all of them)
#' @return Optimized AST
#' @keywords internal
optimize_with_barriers <- function(ast, required = NULL) {
  if (is.null(ast)) return(NULL)

  # `required` as names of ast's own output, in its order (a barrier's input
  # requirement may name outputs the input does not have); everything, or a
  # duplicated name that cannot be told apart, is NULL.
  if (!is.null(required)) {
    out <- infer_schema_memo(ast)$names
    required <- if (anyDuplicated(out) > 0L) NULL else opt_intersect(out, required)
    if (length(required) == length(out)) required <- NULL
  }

  if (is_barrier(ast)) {
    if (identical(.subset2(ast, "type"), "slice")) ast <- rewrite_slice_over_arrange(ast)
    # Optimize below the barrier, not across (an input that comes back as it
    # was is not assigned again: the node stays the same object)
    input <- .subset2(ast, "input")
    if (!is.null(input)) {
      optimized <- optimize_with_barriers(input, barrier_input_required(ast, required))
      if (!is_reference(optimized, input)) ast$input <- optimized
    }
    return(ast)
  }

  # A settled chain that leaves no later pass anything to do (settled_chain()
  # gives 2L) reaches its source through filters and mutates only, so it
  # holds no barrier, and optimize_segment() would return settled_prune() of
  # it. (Only when the consumer reads every column: a narrower requirement
  # makes projection pushdown insert a select.)
  if (is.null(required) && settled_chain(ast, NULL) == 2L) {
    return(settled_prune(ast))
  }

  # Find the next barrier down the tree
  barrier_node <- find_next_barrier(ast)

  if (is.null(barrier_node)) {
    # No barrier: optimize entire subtree
    optimize_segment(ast, required, keep_mutates = !is.null(required))
  } else {
    # The segment above the barrier ends in a stub source holding the
    # barrier's output schema, so it is a complete chain: every schema lookup
    # inside it (and the passes' use of it) answers for real. What the segment
    # reads of the barrier's output is what the pruning select
    # on the stub keeps (projection pushdown puts one there when it can drop a
    # column; none means all of them). That, mapped through the barrier, is what the
    # barrier's input has to deliver, and the barrier plus everything below it
    # is optimized on that basis.
    stub <- ast_source(infer_schema_memo(barrier_node)[c("names", "types")])
    segment_root <- extract_segment_above_barrier(ast, barrier_node, stub)
    optimized_segment <- optimize_segment(segment_root, required, keep_mutates = !is.null(required))

    # (after the extraction, which finds the barrier by identity; the stub's
    # schema is the same either way)
    if (identical(.subset2(barrier_node, "type"), "slice")) {
      barrier_node <- rewrite_slice_over_arrange(barrier_node)
    }
    input <- .subset2(barrier_node, "input")
    optimized <- optimize_with_barriers(
      input,
      barrier_input_required(barrier_node, segment_stub_requirement(optimized_segment))
    )
    if (!is_reference(optimized, input)) barrier_node$input <- optimized

    replace_segment_stub(optimized_segment, barrier_node)
  }
}

#' Turn `arrange(x) |> head(n)` into a top-k slice
#'
#' `head(n)` over a one-key `arrange()` is `slice_min(x, n, with_ties = FALSE)`
#' (or `slice_max()` for `desc()`): both stable-sort with NA last and take the
#' first `compute_slice_size()` rows. The rank-mode slice node runs
#' `gpu_slice_rank()`, which selects the cut value instead of sorting the
#' table when the cut is selective. Only a head slice sitting directly on an
#' ungrouped one-key arrange of a numeric, Date or POSIXct column is rewritten;
#' strings (collation), factors and logicals, several keys, `.by_group` and
#' anything between the two nodes keep the sort. The new node is a slice, so it
#' stays a barrier: no filter crosses it, and `barrier_input_required()` reports
#' its key. Called on a barrier node only, and only slices are tested at the
#' call sites.
#'
#' @param node A slice node
#' @return The rewritten node, or `node` itself
#' @keywords internal
rewrite_slice_over_arrange <- function(node) {
  if (!identical(.subset2(node, "mode"), "head")) return(node)
  arr <- .subset2(node, "input")
  if (!identical(.subset2(arr, "type"), "arrange")) return(node)
  specs <- .subset2(arr, "sort_specs")
  if (length(specs) != 1L || length(.subset2(arr, "groups")) > 0L) return(node)

  spec <- .subset2(specs, 1L)
  below <- .subset2(arr, "input")
  schema <- infer_schema_memo(below)
  col <- .subset2(spec, "col_name")
  pos <- match(col, schema$names)
  if (is.na(pos) || anyDuplicated(schema$names) > 0L) return(node)
  if (!(schema$types[[pos]] %in% topk_key_types)) return(node)

  ast_slice(below, mode = "rank", amount = .subset2(node, "amount"),
            is_prop = .subset2(node, "is_prop"), order_col = col,
            order_is_temp = FALSE, descending = isTRUE(.subset2(spec, "descending")),
            with_ties = FALSE, na_rm = FALSE, top_k = TRUE)
}

# The key types gpu_slice_rank() has a top-k route for
topk_key_types <- c("INT32", "INT64", "FLOAT32", "FLOAT64",
                    "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")

#' What a barrier's input has to deliver
#'
#' The single rule for "the columns below this node that its consumer and the
#' node itself read", for the barrier nodes (`optimize_with_barriers()`,
#' `push_down_projections()` and `prune_dead_columns()` all ask it). The result
#' may name columns the input does not have (a window's outputs, a mutate's
#' intermediates above): they are harmless where a source intersects with its
#' names, but a select must be built from real names only.
#'
#' @param node A barrier node
#' @param required Columns read from `node`'s output (`NULL`: all)
#' @return Character vector, or `NULL` for "all the input's columns" (what a
#'   node that reads positions, the whole table, or has duplicated input names
#'   needs)
#' @keywords internal
barrier_input_required <- function(node, required) {
  type <- .subset2(node, "type")
  if (!(type %in% c("summarise", "arrange", "distinct", "window", "rename", "slice"))) {
    return(NULL)  # collapse(), reframe(), a positional select
  }
  # (summarise and distinct read their own columns whatever is above; every
  # other node needs its consumer's requirement to narrow anything)
  if (is.null(required) && type != "summarise" && type != "distinct") {
    return(NULL)
  }
  if (anyDuplicated(infer_schema_memo(.subset2(node, "input"))$names) > 0L) {
    return(NULL)
  }
  if (identical(type, "summarise")) {
    agg_inputs <- unique.default(unlist(lapply(.subset2(node, "aggregations"), `[[`, "input_col")))
    return(opt_union(.subset2(node, "groups"), agg_inputs[!is.na(agg_inputs)]))
  }
  if (identical(type, "distinct")) {
    if (isTRUE(.subset2(node, "keep_all")) || !is.null(.subset2(node, "key_positions"))) {
      return(NULL)
    }
    return(opt_union(character(), .subset2(node, "key_cols")))
  }
  switch(type,
    # `.by_group = TRUE` sorts by the groups too (lower_arrange())
    arrange = {
      sort_cols <- vapply(.subset2(node, "sort_specs"), `[[`, character(1), "col_name")
      opt_union(opt_union(required, sort_cols), .subset2(node, "groups"))
    },
    window = {
      spec_cols <- unique.default(unlist(
        lapply(.subset2(node, "specs"), function(s) c(s$value_col, s$order_cols)),
        use.names = FALSE
      ))
      opt_union(opt_union(required, .subset2(node, "group_cols")), spec_cols[!is.na(spec_cols)])
    },
    # `required` is in post-rename names, the input's schema has the old ones.
    # Every old name stays: infer_schema_from() looks them up in the input, and
    # a missing one is an NA subscript. A positional rename (duplicated names)
    # can't be translated by name.
    rename = {
      if (!is.null(.subset2(node, "positions"))) return(NULL)
      idx <- match(required, .subset2(node, "new_names"))
      hit <- !is.na(idx)
      required[hit] <- .subset2(node, "old_names")[idx[hit]]
      opt_union(required, .subset2(node, "old_names"))
    },
    # `mode = "rank"`'s order_col is read for the ranking even when nothing
    # above reads it (it may also be a temp column absent from the output);
    # `"rank_grouped"` reads its group columns too
    slice = opt_union(opt_union(required, .subset2(node, "order_col")),
                      .subset2(node, "group_cols"))
  )
}

#' What a segment reads of the stub at its bottom
#'
#' The columns of the pruning select that projection pushdown put directly on
#' the stub (`pruned = TRUE`: it exists because some stub columns are not read),
#' or `NULL` when the node on the stub is anything else: every stub column is
#' then read or passed on. A select higher up is not looked at: one above a
#' filter keeps fewer columns than the filter reads (the filter case wraps
#' itself in a select to drop its predicate columns), and a user select on the
#' stub keeps only what the stub delivers anyway.
#'
#' @param segment An optimized segment ending in a stub source
#' @return Character vector or `NULL`
#' @keywords internal
segment_stub_requirement <- function(segment) {
  node <- segment
  while (!identical(.subset2(.subset2(node, "input"), "type"), "source")) {
    node <- .subset2(node, "input")
    if (is.null(node)) return(NULL)  # (the segment is the stub)
  }
  if (isTRUE(.subset2(node, "pruned")) && is.null(.subset2(node, "positions"))) {
    .subset2(node, "columns")
  }
}

#' Optimize a single segment (no barriers)
#'
#' @param ast Root of segment
#' @param required_cols Columns required by whatever consumes this
#'   segment's output (`NULL` means all of `ast`'s own output columns)
#' @param keep_mutates `TRUE` when `required_cols` is a barrier's narrower
#'   requirement (F99): that pruning removes columns only, so a mutate
#'   expression nothing reads stays and is computed (it can warn, as in dplyr),
#'   and `prune_dead_columns()` does not run. Its inputs stay required
#'   (`push_down_projections()` keeps them).
#' @return Optimized segment
#' @keywords internal
optimize_segment <- function(ast, required_cols = NULL, keep_mutates = FALSE) {
  if (is.null(ast)) return(NULL)

  # The segment's output columns (`required_cols = NULL`), inferred once from
  # the tree as given, and only when a pass needs them: projection pushdown
  # keeps them as they are, so the second pass would infer the same names from
  # its (rebuilt) tree.
  root <- ast
  own <- is.null(required_cols)

  # A chain of filters and mutates on a source whose every output column is
  # required (settled_chain()): projection pushdown leaves it as it is, and
  # dead-column pruning only re-subsets each mutate's expression list.
  settled <- settled_chain(ast, required_cols)
  if (settled == 2L) {
    # ... and no later pass applies either (segment_features() would find no
    # mutate or filter over a mutate or filter): the pruning is all there is.
    return(if (keep_mutates) ast else settled_prune(ast))
  }
  settled <- settled == 1L
  if (!settled) {
    if (own) required_cols <- infer_schema_memo(root)$names
    ast <- push_down_projections(ast, required_cols)
  }

  # A cheap scan of the node types says which later passes can change the
  # plan; a pass that can't is skipped (it would rebuild the same tree). Each
  # pass that runs can create what the next one looks for, so the scan is
  # repeated after it.
  feats <- segment_features(ast)
  if (feats[["mutate_pair"]]) {
    ast <- fuse_mutates(ast)
    if (settled) {
      # (fusion keeps a settled chain settled; checked again all the same)
      if (own) required_cols <- infer_schema_memo(root)$names
      settled <- settled_chain(ast, required_cols) > 0L
    }
  }
  if (feats[["column_nodes"]] && !keep_mutates) {
    if (settled) {
      # (same node types and links: the scan would read the same features)
      ast <- settled_prune(ast)
    } else {
      if (is.null(required_cols)) required_cols <- infer_schema_memo(root)$names
      ast <- prune_dead_columns(ast, required_cols)
      feats <- segment_features(ast)
    }
  }
  if (feats[["filter_over"]]) {
    ast <- push_down_filters(ast)
    feats <- segment_features(ast)
  }
  if (feats[["filter_pair"]]) {
    ast <- reorder_filters(ast)
  }
  # F85: what a parquet reader can decide of a filter on a scan moves into it
  # (segment_filters_on_scan() before the option: a plan with no scan pays one walk)
  if (segment_filters_on_scan(ast) && scan_pushdown_enabled()) {
    ast <- push_into_scan(ast)
  }
  ast
}

#' What a segment holds, for skipping optimizer passes that cannot apply
#'
#' One walk over the plan (both sides of a join):
#' - `mutate_pair`: a mutate directly over a mutate (`fuse_mutates()`)
#' - `column_nodes`: any mutate or select (`prune_dead_columns()` only ever
#'   changes those)
#' - `filter_over`: a filter directly over a mutate, select or join
#'   (`push_down_filters()`)
#' - `filter_pair`: a filter directly over a filter (`reorder_filters()`)
#'
#' @param ast Root of a segment
#' @return Named logical vector
#' @keywords internal
segment_features <- function(ast) {
  mutate_pair <- column_nodes <- filter_over <- filter_pair <- FALSE
  pending <- list(ast)
  while (length(pending) > 0L) {
    node <- pending[[length(pending)]]
    pending[[length(pending)]] <- NULL
    while (!is.null(node)) {
      type <- .subset2(node, "type")
      if (identical(type, "join")) {
        pending[[length(pending) + 1L]] <- .subset2(node, "right")
        node <- .subset2(node, "left")
        next
      }
      input <- .subset2(node, "input")
      input_type <- .subset2(input, "type")
      if (identical(type, "mutate")) {
        column_nodes <- TRUE
        if (identical(input_type, "mutate")) mutate_pair <- TRUE
      } else if (identical(type, "select")) {
        column_nodes <- TRUE
      } else if (identical(type, "filter") && !is.null(input_type)) {
        if (input_type == "filter") {
          filter_pair <- TRUE
        } else if (input_type %in% c("mutate", "select", "join")) {
          filter_over <- TRUE
        }
      }
      node <- input
    }
  }
  c(mutate_pair = mutate_pair, column_nodes = column_nodes,
    filter_over = filter_over, filter_pair = filter_pair)
}

#' Is a segment a chain no projection or pruning pass can change?
#'
#' Settled when `ast` is a chain of filter and mutate nodes (each mutate with
#' at least one expression) ending in a source, every predicate reads only
#' columns of its filter's input, and `required_cols` holds every output
#' column of `ast` (`NULL`: they are `ast`'s own output columns, as
#' `optimize_segment()` takes it). Then, by induction from the top (a node's
#' required set always covers its output columns: a filter's output is its
#' input, and a mutate's is its input plus its outputs, so taking its new
#' outputs away and adding what its expressions read still covers the input):
#' - `push_down_projections()` returns the same tree: the source keeps all
#'   its columns, and no filter has a predicate column that isn't required.
#' - `prune_dead_columns()` keeps every mutate expression (each output column
#'   is required), so its only change is `exprs[keep]` with `keep` all TRUE,
#'   which `settled_prune()` repeats.
#' Anything else (a select, a join, a barrier, an empty mutate) is not settled,
#' and the passes run as usual. A segment above a barrier ends in a stub source
#' and is settled like any other. Only output column NAMES are needed, as sets,
#' so they are built bottom-up without inferring types.
#'
#' A narrower `required_cols` (F99: what a barrier above reads) is settled when
#' the requirement walked down the chain needs no select above a filter, finds
#' no dead mutate expression and keeps every source column. One set does for
#' both passes: a mutate's new outputs, the only names projection pushdown
#' takes back out of its requirement, are neither source columns nor a lower
#' node's input, so they change no answer here. When every output is required
#' the walk passes by the induction above.
#'
#' @param ast Root of a segment
#' @param required_cols Columns required by whatever consumes the segment, or
#'   `NULL` for all of its output columns
#' @return `0L` not settled; `1L` settled; `2L` settled, and no mutate or
#'   filter sits directly on a mutate or filter, so `segment_features()` would
#'   find no pair for `fuse_mutates()`, `push_down_filters()` or
#'   `reorder_filters()` either
#' @keywords internal
settled_chain <- function(ast, required_cols) {
  # First walk: the node types, the pairs, and the predicates of a filter on
  # the source (whose input columns are the source's).
  pair <- FALSE
  node <- ast
  repeat {
    type <- .subset2(node, "type")
    if (is.null(type)) {
      return(0L)
    }
    input <- .subset2(node, "input")
    input_type <- .subset2(input, "type")
    switch(type,
      source = break,
      mutate = {
        if (length(.subset2(node, "expressions")) == 0L) return(0L)
        if (identical(input_type, "mutate")) pair <- TRUE
      },
      filter = {
        if (identical(input_type, "source")) {
          cols <- node_field_union(.subset2(node, "predicates"), "cols")
          if (!all(match(cols, .subset2(.subset2(input, "schema"), "names"), 0L) > 0L)) return(0L)
        } else {
          # (a filter on a mutate or a filter; anything else ends the walk below)
          pair <- TRUE
        }
      },
      return(0L)
    )
    node <- input
  }
  # A scan has the passes of F85's pushdown to take (projection into the reader,
  # predicates into the reader): the chain over it is never settled
  if (!is.null(.subset2(node, "scan")) && scan_pushdown_enabled()) {
    return(0L)
  }
  if (!pair && is.null(required_cols)) {
    return(2L)
  }

  # A narrower requirement, walked down from the root (see above): a filter
  # whose predicates are not all required needs a select above it, a dead
  # expression is one prune_dead_columns() would drop, and a source column
  # left over needs a select below.
  if (!is.null(required_cols)) {
    need <- required_cols
    node <- ast
    repeat {
      type <- .subset2(node, "type")
      if (type == "source") break
      if (type == "filter") {
        if (!all(node_field_union(.subset2(node, "predicates"), "cols") %in% need)) return(0L)
      } else {
        exprs <- .subset2(node, "expressions")
        for (j in rev(seq_along(exprs))) {
          expr <- .subset2(exprs, j)
          if (!(.subset2(expr, "output_col") %in% need)) return(0L)
          need <- opt_union(need, .subset2(expr, "input_cols"))
        }
      }
      node <- .subset2(node, "input")
    }
    if (!all(.subset2(.subset2(node, "schema"), "names") %in% need)) return(0L)
  }

  # With a filter on a mutate or a filter, every predicate must read columns
  # of its filter's input: built bottom-up. (A filter on the source was
  # checked in the first walk.)
  if (!pair) {
    return(2L)
  }
  chain <- list()
  node <- ast
  while (!identical(.subset2(node, "type"), "source")) {
    chain[[length(chain) + 1L]] <- node
    node <- .subset2(node, "input")
  }
  cols <- .subset2(.subset2(node, "schema"), "names")
  for (i in rev(seq_along(chain))) {
    node <- chain[[i]]
    if (identical(.subset2(node, "type"), "filter")) {
      if (!all(node_field_union(.subset2(node, "predicates"), "cols") %in% cols)) return(0L)
    } else {
      cols <- opt_union(cols, mutate_output_cols(.subset2(node, "expressions")))
    }
  }
  1L
}

#' `prune_dead_columns()` on a settled chain
#'
#' What the pass returns for a chain `settled_chain()` accepts: the same nodes,
#' each mutate's expressions replaced by `exprs[keep]` with every `keep`
#' TRUE. That subset drops the list's attributes but its names (the
#' parse-time flags `parse_mutate_dots()` sets), so a mutate whose list has
#' other attributes is rebuilt with it; a node nothing changes, with nothing
#' changed below it, is returned as it is.
#'
#' @param ast A chain `settled_chain()` accepted
#' @return The pruned chain
#' @keywords internal
settled_prune <- function(ast) {
  if (identical(.subset2(ast, "type"), "source")) {
    return(ast)
  }
  input <- .subset2(ast, "input")
  pruned <- settled_prune(input)
  if (!is_reference(pruned, input)) {
    ast$input <- pruned
  }
  if (identical(.subset2(ast, "type"), "mutate")) {
    exprs <- .subset2(ast, "expressions")
    attrs <- names(attributes(exprs))
    if (!(is.null(attrs) || identical(attrs, "names"))) {
      ast$expressions <- exprs[rep(TRUE, length(exprs))]
    }
  }
  ast
}

#' Find the next barrier node walking down from root
#'
#' @param ast AST node
#' @return Barrier node or NULL
#' @keywords internal
find_next_barrier <- function(ast) {
  while (!is.null(ast)) {
    if (is_barrier(ast)) return(ast)
    ast <- .subset2(ast, "input")
  }
  NULL
}

#' Extract segment above a barrier (returns copy)
#'
#' @param ast Root node
#' @param barrier Target barrier node
#' @param stub Source node that takes the barrier's place
#' @return Copy of segment with `stub` at the barrier point
#' @keywords internal
extract_segment_above_barrier <- function(ast, barrier, stub) {
  if (identical(ast, barrier)) {
    return(stub)
  }

  # Shallow copy the node
  new_node <- ast
  new_node$input <- extract_segment_above_barrier(ast$input, barrier, stub)
  new_node
}

#' Put the barrier back where the segment's stub source is
#'
#' @param segment Optimized segment, a chain of nodes ending in the stub
#' @param barrier Node to put in its place
#' @return The segment on top of `barrier`
#' @keywords internal
replace_segment_stub <- function(segment, barrier) {
  if (identical(.subset2(segment, "type"), "source")) {
    return(barrier)
  }
  segment$input <- replace_segment_stub(.subset2(segment, "input"), barrier)
  segment
}

# -----------------------------------------------------------------------------
# Pass 1: Projection Pruning
# -----------------------------------------------------------------------------

#' Push down column projections to reduce data width early
#'
#' @param ast Root AST node
#' @param required_cols Columns required by parent (NULL = all output cols)
#' @param group_cols Group columns that must be preserved
#' @return Optimized AST with select nodes inserted
#' @keywords internal
push_down_projections <- function(ast, required_cols = NULL, group_cols = character()) {
  if (is.null(ast)) return(NULL)

  if (is.null(required_cols)) {
    required_cols <- infer_schema_memo(ast)$names
  }

  # Always preserve group columns
  # With no group columns `required_cols` is used as passed and may hold
  # duplicates (the base `union()` this replaced always deduped it). Most
  # consumers dedupe again with opt_union()/opt_intersect(); the filter case's
  # `else required_cols` branch (a hand-cut chain) does not, but duplicates can
  # only come from a segment root with duplicated output names, which only
  # positional (barrier) nodes produce.
  if (length(group_cols) > 0L) required_cols <- opt_union(required_cols, group_cols)

  switch(ast$type,
    "source" = {
      # a source with duplicated names can't be pruned by name
      if (anyDuplicated(ast$schema$names) > 0L) {
        return(ast)
      }
      keep <- opt_intersect(ast$schema$names, required_cols)
      # A downstream node that reads no column at all (a constant
      # `mutate(a = TRUE)` under a select) would prune the source to zero
      # columns, and a zero-column table has no row count to broadcast the
      # constant over: keep one column.
      # (a parquet scan keeps a column it can read, F85)
      into_scan <- !is.null(ast$scan) && scan_pushdown_enabled()
      if (length(keep) == 0 && length(ast$schema$names) > 0) {
        keep <- if (into_scan) scan_row_count_column(ast) else ast$schema$names[1]
      }
      if (length(keep) < length(ast$schema$names)) {
        # F85: a parquet scan reads just those columns (still in the schema's
        # order); any other source gets a select
        if (into_scan) scan_project(ast, keep) else ast_select(ast, keep, pruned = TRUE)
      } else {
        ast
      }
    },
    "filter" = {
      pred_cols <- node_field_union(.subset2(ast, "predicates"), "cols")
      needed <- opt_union(required_cols, pred_cols)
      ast$input <- push_down_projections(ast$input, needed, group_cols)

      extra_cols <- opt_diff(pred_cols, required_cols)
      if (length(extra_cols) > 0) {
        # With a complete input chain, keep the input's column order and
        # only names it actually has: `required_cols` is in the parent's
        # order (it may name a mutate's intermediates, or a window's outputs
        # above this filter in a join side), and a reordered join side changes
        # which column gets which stacked suffix (`v.x.x` vs `v.x`). (A
        # segment's chain is complete: it ends in the barrier's stub source.)
        keep <- if (input_chain_complete(ast$input)) {
          opt_intersect(infer_schema_memo(ast$input)$names, required_cols)
        } else {
          required_cols
        }
        # Nothing above reads a column (a constant `mutate(b = 1)` under a
        # select): a zero-column table has no row count to broadcast over, so
        # keep one predicate column (same guard as the "source" case).
        if (length(keep) == 0L) {
          keep <- if (input_chain_complete(ast$input)) {
            opt_intersect(infer_schema_memo(ast$input)$names, pred_cols)[1]
          } else {
            pred_cols[1]
          }
          if (is.na(keep)) keep <- pred_cols[1]
        }
        ast_select(ast, keep)
      } else {
        ast
      }
    },
    "mutate" = {
      expr_inputs <- node_field_union(.subset2(ast, "expressions"), "input_cols")
      outputs <- mutate_output_cols(.subset2(ast, "expressions"))
      # Only genuinely NEW output columns can be safely dropped from the
      # requirement passed upstream -- an output whose name already exists
      # in the input schema is a REPLACE-in-place (e.g. `mutate(s = sin(x))`
      # when `s` is an existing column), and lower_mutate()/gpu_mutate_expr()
      # rely on that name still being present (at its original position) in
      # the immediate input schema to recognize it as a replacement rather
      # than an append. Dropping it here (as this code used to do
      # unconditionally) let this "source" case insert an ast_select that
      # silently removed the replaced column from the projected source
      # table -- gpu_mutate_expr() then appended the new value at the END
      # instead of replacing in place, while the (unoptimized) inferred
      # schema used to label the collected result still expected the
      # original column order, producing a silent column-value rotation
      # bug (found via the T6 oracle tests -- see test-dplyr-mutate.R).
      input_schema_names <- infer_schema_memo(ast$input)$names
      new_outputs <- opt_diff(outputs, input_schema_names)
      needed <- opt_union(opt_diff(required_cols, new_outputs), expr_inputs)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "select" = {
      # a positional select (duplicated names) can't be translated by name
      needed <- NULL
      if (is.null(ast$positions)) {
        # narrowed to what is read, as prune_dead_columns() does (a select
        # nothing reads from keeps its columns)
        needed <- opt_intersect(ast$columns, required_cols)
        if (length(needed) == 0L) needed <- ast$columns
        ast$columns <- needed
      }
      inner <- push_down_projections(ast$input, needed, group_cols)
      # (a pruning select just inserted below that keeps the same columns
      # makes this one redundant, and stays: it is the one that holds a filter)
      if (isTRUE(inner$pruned) && identical(inner$columns, needed)) return(inner)
      ast$input <- inner
      ast
    },
    "join" = {
      if (ast$join_type %in% c("semi", "anti")) {
        # semi/anti (Phase 7 J2, design section 1.2): the output is x's
        # (left's) schema verbatim -- every `required_cols` entry is
        # already a left column, no origin split needed. The right side
        # contributes nothing to the output at all; it only needs its own
        # join-key columns (`by$right`) to probe against.
        left_needed <- opt_union(required_cols, ast$by$left)
        right_needed <- unique(ast$by$right)
      } else {
        needed <- join_side_requirements(ast, required_cols)
        left_needed <- needed$left
        right_needed <- needed$right
      }

      left_needed <- join_keep_a_column(ast, ast$left, left_needed)
      right_needed <- join_keep_a_column(ast, ast$right, right_needed)
      # `group_cols` are OUTPUT-namespace names, already folded into
      # `required_cols` above and mapped to source names by the side
      # requirements: they must not be forwarded into the sides.
      ast$left <- push_down_projections(ast$left, left_needed)
      ast$right <- push_down_projections(ast$right, right_needed)
      ast
    },
    # Phase 12 audit note: "arrange"/"summarise"/"window" (and, added here,
    # "rename"/"distinct"/"slice") are all barriers (`is_barrier()`,
    # R/ast.R), so none of these cases are reached via the *normal*
    # barrier-splitting entry point (`optimize_with_barriers()` ends a
    # segment in a stub right before the first barrier it finds walking down
    # `$input`, so a barrier node is never the `ast` a segment's own
    # `push_down_projections()` call is invoked on). They ARE reached,
    # however, via the "join" case below: `ast$left`/`ast$right` are
    # recursed into with a direct `push_down_projections()` call, NOT
    # `optimize_with_barriers()`, so a join's own input side can be (and
    # commonly is -- `x |> rename(...) |> left_join(y, ...)`,
    # `x |> distinct(...) |> inner_join(y, ...)`) a barrier-type node
    # directly. Confirmed empirically (Phase 12 audit): before this case
    # existed, a join's `rename`/`distinct`/`slice` input side had its
    # required columns silently mistranslated (`rename`, whose input
    # schema still uses the OLD names) or under-required (`distinct`'s own
    # `key_cols`, `slice(mode="rank")`'s `order_col`, not automatically
    # part of what's needed downstream), so a subsequent `ast_select`
    # inserted further down could drop a column the barrier node itself
    # still needed -- a real, reproducible miscorrectness bug, not just a
    # missed optimization (see `tests/testthat/test-optimizer.R`).
    "arrange" = {
      ast$input <- push_down_projections(ast$input, barrier_input_required(ast, required_cols), group_cols)
      ast
    },
    "summarise" = {
      ast$input <- push_down_projections(ast$input, barrier_input_required(ast, required_cols), ast$groups)
      ast
    },
    "reframe" = {
      # reads every column of its (wide) input; nothing is pruned below it
      ast$input <- push_down_projections(ast$input, infer_schema_memo(ast$input)$names)
      ast
    },
    "window" = {
      ast$input <- push_down_projections(ast$input, barrier_input_required(ast, required_cols), ast$group_cols)
      ast
    },
    # (the rules for what each of these needs below it, rename's mapping
    # from post- to pre-rename names included, are in barrier_input_required())
    "rename" = , "distinct" = , "slice" = {
      ast$input <- push_down_projections(ast$input, barrier_input_required(ast, required_cols), group_cols)
      ast
    },
    {
      if (!is.null(ast$input)) {
        ast$input <- push_down_projections(ast$input, required_cols, group_cols)
      }
      ast
    }
  )
}

# -----------------------------------------------------------------------------
# Pass 2: Mutate Fusion
# -----------------------------------------------------------------------------

#' Fuse consecutive mutate nodes when safe
#'
#' @param ast Root AST node
#' @return AST with fused mutates
#' @keywords internal
fuse_mutates <- function(ast) {
  if (is.null(ast)) return(NULL)

  # Phase 12 audit: a "join" node has no `$input` field at all (its two
  # inputs are `$left`/`$right`), so without this case, any mutate chain
  # sitting directly on either side of a join (`x |> mutate(...) |>
  # mutate(...) |> inner_join(y, ...)`) was silently skipped by this whole
  # pass -- not a correctness bug (unfused mutates still execute correctly,
  # just as separate kernels), but a real missed-optimization gap now
  # closed the same way push_down_projections()/prune_dead_columns()
  # already recurse into both sides.
  if (ast$type == "join") {
    ast$left <- fuse_mutates(ast$left)
    ast$right <- fuse_mutates(ast$right)
    return(ast)
  }

  if (ast$type != "mutate") {
    if (!is.null(ast$input)) {
      ast$input <- fuse_mutates(ast$input)
    }
    return(ast)
  }

  # Recursively process input first
  ast$input <- fuse_mutates(ast$input)

  # Check if input is also a mutate
  if (!is.null(ast$input) && ast$input$type == "mutate") {
    combined <- try_fuse_mutate_pair(ast$input, ast)
    if (!is.null(combined)) {
      return(combined)
    }
  }

  ast
}

# TRUE iff a mutate node carries a vector-literal expression.
mutate_has_vector_literal <- function(node) {
  any(vapply(node$expressions, function(e) identical(e$ir$kind, "vec"), logical(1)))
}

# F49: TRUE iff `ir` (an IR node or a list of them) holds a rowwise aggregate
# whose flags depend on which rows it sees (check_rowwise_flags()): an integer
# sum()/min()/max() (`rw_agg_i32`: overflow, all-NA and zero-row checks that
# change dplyr's type) or a double min()/max() (base R's warnings). dplyr
# computes it on every row its verb gets, so no filter may be moved below a
# mutate holding one, or merged with the filters under one that holds one.
ir_has_rw_flag_agg <- function(ir) {
  if (!is.list(ir)) {
    return(FALSE)
  }
  if (inherits(ir, "cuplyr_ir")) {
    if (!identical(ir$kind, "call")) {
      return(FALSE)
    }
    if (identical(ir$op, "rw_agg_i32") ||
        (identical(ir$op, "rw_agg_f64") && isTRUE(ir$meta$fn %in% c("min", "max")))) {
      return(TRUE)
    }
    return(ir_has_rw_flag_agg(ir$args))
  }
  any(vapply(ir, ir_has_rw_flag_agg, logical(1)))
}

# F70: TRUE iff a mutate node holds an F42 checked integer expression
# (`ir_has_checked_int()`, asked through `record_checked_int()`).
# dplyr evaluates it on every row its verb receives and warns on overflow, so
# no filter may be moved below it and it may not be pruned as dead.
mutate_has_checked_int <- function(node) {
  any(vapply(.subset2(node, "expressions"), record_checked_int, logical(1)))
}

# F70: TRUE iff a predicate holds an F42 checked integer expression.
pred_checked_int <- function(pred) record_checked_int(pred)

#' Try to fuse two consecutive mutate nodes
#'
#' @param lower Lower (earlier) mutate node
#' @param upper Upper (later) mutate node
#' @return Fused mutate node or NULL if fusion not possible
#' @keywords internal
try_fuse_mutate_pair <- function(lower, upper) {
  combined_exprs <- c(lower$expressions, upper$expressions)

  # Guard 1: Expression count limit
  if (length(combined_exprs) > 8) {
    return(NULL)
  }

  # No reordering: `lower` then `upper` is already dplyr's sequential order,
  # and lower_mutate() binds each expression against the columns as the
  # earlier ones left them. So write-after-read and write-after-write
  # overwrites fuse correctly with no temps.
  lower_outputs <- vapply(lower$expressions, `[[`, character(1), "output_col")
  upper_inputs <- unique(unlist(lapply(upper$expressions, `[[`, "input_cols")))

  # Guard 2: Count intermediate columns (lower outputs the upper reads)
  if (length(opt_intersect(upper_inputs, lower_outputs)) > 4) {
    return(NULL)
  }

  ast_mutate(lower$input, combined_exprs)
}

# -----------------------------------------------------------------------------
# Pass 3: Dead Column Pruning
# -----------------------------------------------------------------------------

#' Prune unused mutate outputs to reduce intermediate width
#'
#' Walks the AST from root to leaves, tracking columns required downstream.
#' Any mutate expression whose output is not required is dropped; empty mutate
#' nodes are removed entirely.
#'
#' @param ast Root AST node
#' @param required_cols Columns required by parent (NULL = all output cols)
#' @param group_cols Group columns that must be preserved
#' @return AST with unused mutate outputs pruned
#' @keywords internal
prune_dead_columns <- function(ast, required_cols = NULL, group_cols = character()) {
  if (is.null(ast)) return(NULL)

  if (is.null(required_cols)) {
    required_cols <- infer_schema_memo(ast)$names
  }

  # As in push_down_projections(), with no group columns `required_cols` is
  # used as passed and may hold duplicates. Most cases dedupe again with
  # opt_union()/opt_intersect(); the mutate, rename and default cases can pass
  # it down as is (mutate only tests membership with it), and duplicates can
  # only come from a segment root with duplicated output names, which only
  # positional (barrier) nodes produce.
  if (length(group_cols) > 0L) required_cols <- opt_union(required_cols, group_cols)

  switch(ast$type,
    "source" = {
      ast
    },
    "filter" = {
      pred_cols <- node_field_union(.subset2(ast, "predicates"), "cols")
      needed <- opt_union(required_cols, pred_cols)
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "mutate" = {
      exprs <- ast$expressions
      if (length(exprs) == 0) {
        return(prune_dead_columns(ast$input, required_cols, group_cols))
      }

      required <- required_cols
      keep <- logical(length(exprs))

      for (i in rev(seq_along(exprs))) {
        expr <- exprs[[i]]
        # A vector literal is kept even when dead: its length check is an error
        # dplyr raises regardless of use.
        if (identical(expr$ir$kind, "vec")) {
          keep[i] <- TRUE
        }
        # Likewise a checked integer expression (F70): its overflow warning
        # does not depend on the column being used. Its inputs stay required.
        if (record_checked_int(expr)) {
          keep[i] <- TRUE
          required <- opt_union(required, expr$input_cols)
        }
        if (expr$output_col %in% required) {
          keep[i] <- TRUE
          # The name stays required: an earlier writer of it (in this node or
          # a lower one) may be the expression that CREATES the column, and
          # gpu_mutate_expr() appends new columns in order of first writing.
          # Dropping that writer (a dead overwrite) would move the column
          # behind later-created ones while the result is still named by the
          # unoptimized schema -- values under the wrong names. Cost: every
          # earlier writer of a still-required name is kept, including pure
          # dead overwrites (`mutate(a = x) |> mutate(a = y)` computes both).
          required <- opt_union(required, expr$input_cols)
        }
      }

      if (!any(keep)) {
        return(prune_dead_columns(ast$input, required, group_cols))
      }

      ast$expressions <- exprs[keep]
      ast$input <- prune_dead_columns(ast$input, required, group_cols)
      ast
    },
    "summarise" = {
      ast$input <- prune_dead_columns(ast$input, barrier_input_required(ast, required_cols), ast$groups)
      ast
    },
    "select" = {
      if (!is.null(ast$positions)) {
        ast$input <- prune_dead_columns(ast$input, NULL, group_cols)
        return(ast)
      }
      keep <- opt_intersect(ast$columns, required_cols)
      if (length(keep) == 0) {
        keep <- ast$columns
      }
      ast$columns <- keep
      ast$input <- prune_dead_columns(ast$input, keep, group_cols)
      ast
    },
    "join" = {
      if (ast$join_type %in% c("semi", "anti")) {
        # Same rationale as push_down_projections()'s "join" case above:
        # semi/anti's output is x's schema verbatim, right only needs its
        # own join keys.
        left_needed <- opt_union(required_cols, ast$by$left)
        right_needed <- unique(ast$by$right)
      } else {
        needed <- join_side_requirements(ast, required_cols)
        left_needed <- needed$left
        right_needed <- needed$right
      }

      left_needed <- join_keep_a_column(ast, ast$left, left_needed)
      right_needed <- join_keep_a_column(ast, ast$right, right_needed)
      # group_cols are output-namespace names: see push_down_projections()
      ast$left <- prune_dead_columns(ast$left, left_needed)
      ast$right <- prune_dead_columns(ast$right, right_needed)
      ast
    },
    # Same reachability rationale as push_down_projections()'s equivalent
    # cases (Phase 12 audit note above it): "arrange"/"summarise"/"window"/
    # "rename"/"distinct"/"slice" are all barriers, so these cases are only
    # ever reached via a join's `left`/`right` recursion below, never via
    # the normal barrier-splitting entry point.
    "arrange" = {
      ast$input <- prune_dead_columns(ast$input, barrier_input_required(ast, required_cols), group_cols)
      ast
    },
    "reframe" = {
      ast$input <- prune_dead_columns(ast$input, infer_schema_memo(ast$input)$names)
      ast
    },
    "window" = {
      ast$input <- prune_dead_columns(ast$input, barrier_input_required(ast, required_cols), ast$group_cols)
      ast
    },
    # (same rules as push_down_projections(): see barrier_input_required())
    "rename" = , "distinct" = , "slice" = {
      ast$input <- prune_dead_columns(ast$input, barrier_input_required(ast, required_cols), group_cols)
      ast
    },
    {
      if (!is.null(ast$input)) {
        ast$input <- prune_dead_columns(ast$input, required_cols, group_cols)
      }
      ast
    }
  )
}

#' Columns each side of a mutating join must keep for `required_cols`
#'
#' Shared by push_down_projections() and prune_dead_columns() so the two
#' passes cannot drift. Besides the join keys and the source column of each
#' required output, pruning must not change any required column's output
#' NAME: a column suffixed because of a collision (`v.x`) would come out as
#' plain `v` if its collision partner were pruned from the other side. So
#' the partner is kept too, and the result is verified by recomputing the
#' output names on the pruned schemas (falling back to keeping both sides
#' whole if the simple rule is ever not enough, e.g. double suffixes).
#'
#' Invariant the verification relies on: it recomputes names on each side's
#' columns in NATURAL (schema) order, so every pass that prunes a join side
#' must keep that order. The returned sets are unordered on purpose; when the
#' input chain is complete, an inserted `ast_select` must use
#' `intersect(<input names>, needed)`, never `needed` itself, because stacked
#' suffixes depend on column order. (A join side is always a complete
#' subtree.)
#' @keywords internal
join_side_requirements <- function(ast, required_cols) {
  left_schema <- infer_schema_memo(ast$left)
  right_schema <- infer_schema_memo(ast$right)
  info <- build_join_output_info(left_schema, right_schema, ast$by,
                                 suffix = ast$suffix, keep = ast$keep)
  idx <- match(opt_intersect(required_cols, info$names), info$names)
  idx <- idx[!is.na(idx)]
  is_left <- info$origin[idx] == "left"
  src <- info$source_names[idx]
  left_needed <- unique(c(ast$by$left, src[is_left]))
  right_needed <- unique(c(ast$by$right, src[!is_left]))

  # Collision partners of renamed (suffixed) required columns.
  renamed <- info$names[idx] != src
  left_needed <- unique(c(left_needed, opt_intersect(src[renamed & !is_left], left_schema$names)))
  right_needed <- unique(c(right_needed, opt_intersect(src[renamed & is_left], right_schema$names)))

  prune <- function(schema, keep) {
    k <- schema$names %in% keep
    list(names = schema$names[k], types = schema$types[k])
  }
  pinfo <- build_join_output_info(prune(left_schema, left_needed),
                                  prune(right_schema, right_needed), ast$by,
                                  suffix = ast$suffix, keep = ast$keep)
  key <- function(i) paste(i$origin, i$source_names)
  m <- match(key(info)[idx], key(pinfo))
  if (anyNA(m) || !identical(pinfo$names[m], info$names[idx])) {
    return(list(left = left_schema$names, right = right_schema$names))
  }
  list(left = left_needed, right = right_needed)
}

#' Does an input chain reach a real leaf?
#'
#' A segment of `optimize_with_barriers()` ends in a stub source, so it is
#' complete. "source" and "join" are the only node types without an `$input`;
#' a chain ending anywhere else was cut by hand (a test).
#' @keywords internal
input_chain_complete <- function(node) {
  if (is.null(node)) return(FALSE)
  while (!is.null(node$input)) node <- node$input
  node$type %in% c("source", "join")
}

#' Keep at least one column on each side of a keyless join
#'
#' A `cross` join or a `by = character()` join has no key columns, so
#' projection pruning can leave a side with no needed column at all -- and a
#' zero-column table has lost its row count, which the join still needs
#' (cross product size; "does y have rows" for semi/anti).
#' @keywords internal
join_keep_a_column <- function(ast, side, needed) {
  if (length(needed) > 0L || !(isTRUE(ast$cross_by) || identical(ast$join_type, "cross"))) {
    return(needed)
  }
  infer_schema_memo(side)$names[1L]
}

# -----------------------------------------------------------------------------
# Pass 4: Filter Pushdown
# -----------------------------------------------------------------------------

#' Rewrite a predicate pushed below a join to the side's source names
#'
#' A predicate is written against the join's OUTPUT names (`v.x`, `k.y`), but
#' once pushed to one side it reads that side's own table, where the column
#' is still called by its source name. Only IR `col` nodes are renamed.
#' @keywords internal
join_source_predicate <- function(pred, info, side) {
  map <- info$source_names[info$origin == side]
  names(map) <- info$names[info$origin == side]
  rename_ir <- function(ir) {
    if (ir$kind == "col") {
      if (ir$name %in% names(map)) ir$name <- unname(map[[ir$name]])
    } else if (ir$kind == "call") {
      ir$args <- lapply(ir$args, rename_ir)
    }
    ir
  }
  pred$ir <- rename_ir(pred$ir)
  pred$cols <- ir_cols(pred$ir)
  pred
}

#' Push filters below mutates when predicates do not depend on mutate outputs
#'
#' Phase 12 audit -- filter-pushdown-across-`distinct()` (flagged as a known,
#' unimplemented opportunity back in section 3.1 of the design notes):
#' `filter(distinct(df, k), k > 1)` COULD safely push the filter below
#' `distinct()` when the predicate only references surviving key columns (or,
#' when `.keep_all = TRUE`, any column at all) -- a predicate that's a pure
#' function of the dedup key is constant within each dedup group, so
#' filtering before or after the group-representative pick selects the same
#' groups either way. A predicate touching a NON-key column (only reachable
#' with `.keep_all = TRUE`) is NOT safe to push, though: it could remove the
#' very row `distinct()` would have picked as a group's representative,
#' changing which row (not just which groups) survives.
#'
#' Deliberately NOT implemented here. `distinct` is a barrier
#' (`is_barrier()`, R/ast.R), so under the current `optimize_with_barriers()`
#' architecture a `filter` immediately above a `distinct` is split into two
#' INDEPENDENT segments before any pass runs (see `optimize_with_barriers()`'s
#' own comments) -- the filter's segment ends in a stub source in place
#' of the `distinct`, so `distinct` is never actually present as `ast$input` in
#' the same `push_down_filters()` call for the ordinary linear-chain case.
#' Reaching across deliberately would require a barrier-aware special case in
#' `optimize_with_barriers()` itself (matching on "is the barrier specifically
#' a distinct node, and is the segment's bottom node specifically a single
#' filter with only-key-column predicates"), not a `push_down_filters()`
#' switch case -- meaningfully more invasive than the analogous mutate/
#' select pushdown cases below, and with a real correctness footgun if the
#' key-vs-non-key-column safety check is ever gotten wrong. Left for a future
#' phase; the two-line summary above is the reason it isn't attempted now.
#'
#' @param ast Root AST node
#' @return AST with filters pushed down across mutates where safe
#' @keywords internal
push_down_filters <- function(ast) {
  if (is.null(ast)) return(NULL)

  # Phase 12 audit: a "join" node has no `$input` field (its inputs are
  # `$left`/`$right`), so without this case any filter chain sitting
  # entirely *within* one side of a join (not directly wrapping the join
  # itself -- that case is handled below, by the `input$type == "join"`
  # branch, but only one level deep) was never visited by this pass at
  # all. Recurse into both sides independently; the existing
  # `input$type == "join"` logic further down still handles the one-level
  # swap of a filter that directly wraps this join from above.
  if (ast$type == "join") {
    ast$left <- push_down_filters(ast$left)
    ast$right <- push_down_filters(ast$right)
    return(ast)
  }

  # Recurse first
  if (!is.null(ast$input)) {
    ast$input <- push_down_filters(ast$input)
  }

  if (ast$type != "filter") {
    return(ast)
  }

  input <- ast$input
  if (is.null(input)) {
    return(ast)
  }

  filter_cols <- node_field_union(.subset2(ast, "predicates"), "cols")

  if (input$type == "mutate") {
    # A vector literal is tied to row positions (and its length to the row
    # count): a filter must stay above it.
    if (mutate_has_vector_literal(input)) {
      return(ast)
    }
    # A rowwise aggregate's flags must see dplyr's rows (F49).
    if (any(vapply(input$expressions, function(e) ir_has_rw_flag_agg(e$ir), logical(1)))) {
      return(ast)
    }
    # So must a checked integer expression's overflow warning (F70).
    if (mutate_has_checked_int(input)) {
      return(ast)
    }
    mutate_outputs <- mutate_output_cols(.subset2(input, "expressions"))

    # Do not push if filter references any mutate outputs (including replacements)
    if (length(opt_intersect(filter_cols, mutate_outputs)) > 0) {
      return(ast)
    }

    # Swap: filter below mutate
    new_filter <- ast_filter(input$input, ast$predicates, grouped = isTRUE(.subset2(ast, "grouped")))
    input$input <- new_filter
    return(input)
  }

  if (input$type == "select") {
    # Only push if select keeps all predicate columns (a positional select
    # renames nothing by name, but its columns can't be matched by name)
    # (and not below a select that pruning inserted: the filter then sees only
    # the columns that are read, F99)
    if (!is.null(input$positions) || isTRUE(input$pruned) ||
        !all(filter_cols %in% input$columns)) {
      return(ast)
    }

    new_filter <- ast_filter(input$input, ast$predicates, grouped = isTRUE(.subset2(ast, "grouped")))
    input$input <- new_filter
    return(input)
  }

  if (input$type == "join") {
    if (input$join_type %in% c("semi", "anti")) {
      # semi/anti (Phase 7 J2, design section 1.2): behave like "left" for
      # filter pushdown, but even more permissively -- the join's OUTPUT
      # schema is x's (left's) schema verbatim, so every predicate column
      # is necessarily a left column (there's no right-side name in the
      # output namespace for a predicate to reference at all). Pushing the
      # whole filter below the join is always safe: semi/anti only ever
      # subsets x's ROWS without touching any column's values, so
      # filter(semi_join(x, y), pred) and semi_join(filter(x, pred), y)
      # select the identical row set. Right is NEVER a pushdown target --
      # it isn't part of the output at all.
      #
      # A checked integer predicate (F70) stays above: below the join it would
      # also be evaluated on the x rows the join drops. dplyr evaluates every
      # predicate of one filter() on the same rows, so its siblings stay with it
      # (pushing them alone would hide rows from the checked predicate).
      if (any(vapply(ast$predicates, pred_checked_int, logical(1)))) {
        return(ast)
      }
      input$left <- ast_filter(input$left, ast$predicates, grouped = isTRUE(.subset2(ast, "grouped")))
      return(input)
    }

    left_schema <- infer_schema_memo(input$left)
    right_schema <- infer_schema_memo(input$right)
    info <- build_join_output_info(left_schema, right_schema, input$by,
                                   suffix = input$suffix, keep = input$keep)

    classify_predicate <- function(pred) {
      # A merged key cast to a common type (A3) has a different type than the
      # source column a pushed-down filter would read (e.g. a character
      # predicate against a factor key), so keep such predicates above the join.
      if (any(pred$cols %in% info$cast_keys)) return("both")
      sides <- unique(info$origin[match(pred$cols, info$names)])
      sides <- sides[!is.na(sides)]
      if (length(sides) != 1) return("both")
      sides
    }

    left_preds <- list()
    right_preds <- list()
    stay_preds <- list()

    # dplyr's join_rows() runs its cardinality checks (unmatched = "error",
    # relationship = one-to-one/one-to-many/many-to-one) over the FULL
    # inputs, so a filter below the join could hide or create an error:
    # push nothing. multiple = first/last/any picks among ALL of an x row's
    # y matches, so y must not be filtered first (x-side pushdown is fine).
    # The same goes for a closest() condition (below).
    # A join that can raise dplyr's advisory many-to-many warning (default
    # relationship = NULL, multiple = "all") checks the FULL inputs, so a
    # filter pushed below it would suppress the warning or change its row
    # numbers: push nothing, the join then warns exactly like dplyr (join_by()
    # conditions dplyr never checks already carry relationship = "many-to-many",
    # see join_may_warn_many_to_many()). Trade-off: a default-relationship
    # join loses pushdown; say relationship = "many-to-many" (or another
    # relationship) to keep it.
    if (input$join_type %in% c("inner", "left", "right") &&
        is.null(input$relationship) && identical(input$multiple, "all")) {
      return(ast)
    }
    if (input$join_type %in% c("inner", "left", "right") &&
        (any(input$unmatched == "error") ||
         any(input$relationship %in% c("one-to-one", "one-to-many", "many-to-one")))) {
      return(ast)
    }
    # A closest() join keeps only the extremal y rows of each x row's matches
    # (F24), so a predicate on y columns pushed below it would change which
    # y row is closest, exactly like multiple = first/last/any.
    right_push_ok <- !(input$join_type %in% c("inner", "left", "right") &&
                         !identical(input$multiple, "all")) &&
      !any(input$by$filter != "none")

    for (pred in ast$predicates) {
      side <- classify_predicate(pred)
      # cross (Phase 7 J3, design section 1.2/2): behaves like "inner" for
      # filter pushdown on BOTH sides -- a cross join has no unmatched-row
      # concept at all (every left row pairs with every right row), so a
      # left-only predicate can always push to the left side and a
      # right-only predicate can always push to the right side without
      # changing the result, exactly like an inner join's own left/right
      # pushdown rule.
      #
      # F70: a checked integer predicate is evaluated on every row of the input
      # it moves into, so it may only go where every row reaches the join
      # output: the left input of a left join, the right input of a right join.
      checked <- pred_checked_int(pred)
      if (side == "left" && input$join_type %in% c("inner", "left", "cross") &&
          (!checked || input$join_type == "left")) {
        left_preds <- c(left_preds, list(join_source_predicate(pred, info, "left")))
      } else if (side == "right" && input$join_type %in% c("inner", "right", "cross") &&
                 right_push_ok && (!checked || input$join_type == "right")) {
        right_preds <- c(right_preds, list(join_source_predicate(pred, info, "right")))
      } else {
        stay_preds <- c(stay_preds, list(pred))
      }
    }

    # F70: all predicates of one filter node see the same rows in dplyr. If a
    # checked predicate stays above the join, pushing a sibling would hide rows
    # from it, so the whole node stays (a checked predicate that moves is
    # evaluated on every row that reaches the output, siblings may go as usual).
    if (any(vapply(stay_preds, pred_checked_int, logical(1)))) {
      return(ast)
    }

    if (length(left_preds) > 0) {
      input$left <- ast_filter(input$left, left_preds, grouped = isTRUE(.subset2(ast, "grouped")))
    }
    if (length(right_preds) > 0) {
      input$right <- ast_filter(input$right, right_preds, grouped = isTRUE(.subset2(ast, "grouped")))
    }

    if (length(stay_preds) == 0) {
      return(input)
    }

    ast$predicates <- stay_preds
    ast$input <- input
    return(ast)
  }

  ast
}

# -----------------------------------------------------------------------------
# Pass 5: Filter Reordering
# -----------------------------------------------------------------------------

#' Reorder filters by estimated cost (cheapest first)
#'
#' @param ast Root AST node
#' @return AST with reordered filters
#' @keywords internal
reorder_filters <- function(ast) {
  if (is.null(ast)) return(NULL)

  # Same "join" has-no-$input gap as fuse_mutates()/push_down_filters()
  # above (Phase 12 audit): without this, a consecutive filter chain
  # sitting entirely within one side of a join was never reordered.
  if (ast$type == "join") {
    ast$left <- reorder_filters(ast$left)
    ast$right <- reorder_filters(ast$right)
    return(ast)
  }

  if (ast$type != "filter") {
    if (!is.null(ast$input)) {
      ast$input <- reorder_filters(ast$input)
    }
    return(ast)
  }

  # Collect consecutive filter chain. A node holding a checked integer
  # predicate ends the group (F70): the later filter() call sees only the rows
  # the filters below it keep, so it may absorb the filters above it (it is
  # then the bottom of the merged group and sees the same rows) but not those
  # below.
  filters <- list()
  current <- ast
  grouped <- isTRUE(.subset2(ast, "grouped"))

  # (filters on a grouped table and on an ungrouped one are not merged: a
  # parquet scan can take the predicates of the ungrouped one, F85)
  while (!is.null(current) && current$type == "filter" &&
         identical(isTRUE(.subset2(current, "grouped")), grouped)) {
    filters <- c(filters, list(current$predicates))
    checked_here <- any(vapply(current$predicates, pred_checked_int, logical(1)))
    current <- current$input
    if (checked_here) break
  }

  if (length(filters) <= 1) {
    ast$input <- reorder_filters(ast$input)
    return(ast)
  }

  # Flatten predicates
  all_preds <- unlist(filters, recursive = FALSE)

  # A rowwise aggregate's flags must see only the rows the filters below it
  # keep (F49): leave this node, merge what is under it.
  if (any(vapply(all_preds, function(p) ir_has_rw_flag_agg(p$ir), logical(1)))) {
    ast$input <- reorder_filters(ast$input)
    return(ast)
  }

  # Safety check: don't reorder non-deterministic predicates
  safe_to_reorder <- all(vapply(all_preds, function(p) {
    isTRUE(p$is_deterministic)
  }, logical(1)))

  if (safe_to_reorder) {
    costs <- vapply(all_preds, `[[`, integer(1), "estimated_cost")
    all_preds <- all_preds[order(costs)]
  }

  # Rebuild single filter node
  result <- ast_filter(current, all_preds, grouped = grouped)
  result$input <- reorder_filters(result$input)
  result
}
