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
#' @param ast Root AST node
#' @return Optimized AST
#' @keywords internal
optimize_with_barriers <- function(ast) {
  if (is.null(ast)) return(NULL)

  if (is_barrier(ast)) {
    # Optimize below the barrier, not across (an input that comes back as it
    # was is not assigned again: the node stays the same object)
    input <- .subset2(ast, "input")
    if (!is.null(input)) {
      optimized <- optimize_with_barriers(input)
      if (!is_reference(optimized, input)) ast$input <- optimized
    }
    return(ast)
  }

  # A settled chain that leaves no later pass anything to do (settled_chain()
  # gives 2L) reaches its source through filters and mutates only, so it
  # holds no barrier, and optimize_segment() would return settled_prune() of
  # it.
  if (settled_chain(ast, NULL) == 2L) {
    return(settled_prune(ast))
  }

  # Find the next barrier down the tree
  barrier_node <- find_next_barrier(ast)

  if (is.null(barrier_node)) {
    # No barrier: optimize entire subtree
    optimize_segment(ast)
  } else {
    # `extract_segment_above_barrier()` truncates the segment's bottom
    # node's `$input` to NULL (the barrier itself, and everything below
    # it, is optimized separately -- see the recursive call a few lines
    # down). That truncation means `infer_schema()` can no longer be
    # called on any node *within* the segment without hitting the severed
    # NULL and returning an empty schema (`infer_schema.NULL()`) instead
    # of the barrier's real output schema -- which is exactly what
    # `push_down_projections()`/`prune_dead_columns()` do internally
    # whenever they're handed a NULL `required_cols` (their signal to
    # infer it from the ast they were just given). So `required_cols`
    # MUST be precomputed here, from the real (untruncated) `ast`, and
    # threaded through explicitly -- letting `optimize_segment()` compute
    # its own default from `segment_root` would silently produce an empty
    # required-columns set and `push_down_projections()` would insert a
    # zero-column `ast_select` on top of the segment (found via the
    # slice()-composes-with-filter oracle test, but equally reproducible
    # with any pre-existing barrier, e.g. `arrange() |> filter()` in lazy
    # mode -- this was a latent, pre-existing bug in the barrier-splitting
    # logic itself, not specific to any one barrier type).
    required_cols <- infer_schema_memo(ast)$names

    segment_root <- extract_segment_above_barrier(ast, barrier_node)
    optimized_segment <- optimize_segment(segment_root, required_cols)

    # Reconnect: find the bottom of optimized segment and attach barrier
    barrier_node$input <- optimize_with_barriers(barrier_node$input)

    attach_ast_bottom(optimized_segment, barrier_node)
  }
}

#' Optimize a single segment (no barriers)
#'
#' @param ast Root of segment
#' @param required_cols Columns required by whatever consumes this
#'   segment's output (`NULL` means "infer from `ast` itself", correct
#'   only when `ast` is a genuine, untruncated subtree -- see the caller
#'   comment in `optimize_with_barriers()` for why a barrier-truncated
#'   segment must instead pass this explicitly)
#' @return Optimized segment
#' @keywords internal
optimize_segment <- function(ast, required_cols = NULL) {
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
    return(settled_prune(ast))
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
  if (feats[["column_nodes"]]) {
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
#' Anything else (a select, a join, a truncated segment cut at a barrier, an
#' empty mutate) is not settled, and the passes run as usual. Only output
#' column NAMES are needed, as sets, so they are built bottom-up without
#' inferring types.
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
  if (!pair && is.null(required_cols)) {
    return(2L)
  }

  # Otherwise the output columns, bottom-up: the input columns of every filter
  # (above a mutate or a filter) and the segment's own.
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
  if (!is.null(required_cols) && !all(cols %in% required_cols)) {
    return(0L)
  }
  if (pair) 1L else 2L
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
#' @return Copy of segment with NULL input at barrier point
#' @keywords internal
extract_segment_above_barrier <- function(ast, barrier) {
  if (is.null(ast) || identical(ast, barrier)) {
    return(NULL)
  }

  # Shallow copy the node
  new_node <- ast
  new_node$input <- extract_segment_above_barrier(ast$input, barrier)
  new_node
}

#' Find the bottom node of an AST (deepest non-NULL input)
#'
#' @param ast AST node
#' @return Bottom node
#' @keywords internal
find_ast_bottom <- function(ast) {
  if (is.null(ast$input)) return(ast)
  find_ast_bottom(ast$input)
}

#' Attach a node at the bottom of an AST
#'
#' Returns a new AST with the bottom input replaced by `new_input`.
#'
#' @param ast AST node
#' @param new_input Node to attach at bottom
#' @return AST with bottom input replaced
#' @keywords internal
attach_ast_bottom <- function(ast, new_input) {
  if (is.null(ast)) return(NULL)
  if (is.null(ast$input)) {
    ast$input <- new_input
    return(ast)
  }
  ast$input <- attach_ast_bottom(ast$input, new_input)
  ast
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
  # truncated-chain `else required_cols` branch and the rename pass-through do
  # not, but duplicates can only come from a segment root with duplicated
  # output names, which only positional (barrier) nodes produce.
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
      if (length(keep) == 0 && length(ast$schema$names) > 0) {
        keep <- ast$schema$names[1]
      }
      if (length(keep) < length(ast$schema$names)) {
        ast_select(ast, keep)
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
        # order (and a window above this filter in a join side passes its
        # own outputs down), and a reordered join side changes which column
        # gets which stacked suffix (`v.x.x` vs `v.x`). In a segment cut at
        # a barrier the input schema is partial or empty, so fall back to
        # `required_cols` (computed from the untruncated tree by
        # optimize_with_barriers()).
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
      needed <- if (is.null(ast$positions)) ast$columns
      ast$input <- push_down_projections(ast$input, needed, group_cols)
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
    # barrier-splitting entry point (`optimize_with_barriers()` truncates a
    # segment right before the first barrier it finds walking down `$input`,
    # so a barrier node is never the `ast` a segment's own
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
      # `.by_group = TRUE` sorts by `ast$groups` too (lower_arrange())
      sort_cols <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
      needed <- opt_union(opt_union(required_cols, sort_cols), ast$groups)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "summarise" = {
      agg_inputs <- unique(unlist(lapply(ast$aggregations, `[[`, "input_col")))
      agg_inputs <- agg_inputs[!is.na(agg_inputs)]
      needed <- opt_union(ast$groups, agg_inputs)
      ast$input <- push_down_projections(ast$input, needed, ast$groups)
      ast
    },
    "reframe" = {
      # reads every column of its (wide) input; nothing is pruned below it
      ast$input <- push_down_projections(ast$input, infer_schema_memo(ast$input)$names)
      ast
    },
    "window" = {
      spec_cols <- unique(unlist(
        lapply(ast$specs, function(s) c(s$value_col, s$order_cols)),
        use.names = FALSE
      ))
      spec_cols <- spec_cols[!is.na(spec_cols)]
      needed <- opt_union(opt_union(required_cols, ast$group_cols), spec_cols)
      ast$input <- push_down_projections(ast$input, needed, ast$group_cols)
      ast
    },
    "rename" = {
      # `required_cols` is expressed in POST-rename names (the schema seen
      # by everything above this node); `ast$input`'s own schema still uses
      # the PRE-rename names, so translate any renamed entry back through
      # `new_names -> old_names` before recursing. Columns untouched by the
      # rename pass through unchanged (their name is the same on both sides
      # of the map).
      rev_idx <- match(required_cols, ast$new_names)
      translated <- required_cols
      hit <- !is.na(rev_idx)
      translated[hit] <- ast$old_names[rev_idx[hit]]
      # A positional rename (duplicated names) can't be translated by name:
      # everything below it stays.
      if (!is.null(ast$positions)) translated <- NULL
      ast$input <- push_down_projections(ast$input, translated, group_cols)
      ast
    },
    "distinct" = {
      # `key_cols` are required for the dedup itself regardless of whether
      # `required_cols` (computed from this node's own OUTPUT schema)
      # references them -- true even for `.keep_all = TRUE`, where
      # `required_cols` can be a strict subset of the full input schema.
      needed <- opt_union(required_cols, ast$key_cols)
      # keyed by position (duplicated names): nothing below is pruned by name
      if (!is.null(ast$key_positions)) needed <- NULL
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "slice" = {
      # `mode = "rank"`'s `order_col` is required for the ranking itself
      # even when it isn't otherwise referenced downstream (it may be a
      # real user column not selected further up, or a temp column already
      # excluded from this node's own output schema by
      # `infer_schema.ast_slice`).
      extra <- if (!is.null(ast$order_col)) ast$order_col else character(0)
      needed <- opt_union(required_cols, extra)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
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
# (`ir_has_checked_int()`, cached as `checked_int` by make_mutate_expr()).
# dplyr evaluates it on every row its verb receives and warns on overflow, so
# no filter may be moved below it and it may not be pruned as dead.
mutate_has_checked_int <- function(node) {
  any(vapply(.subset2(node, "expressions"), function(e) isTRUE(e$checked_int), logical(1)))
}

# F70: TRUE iff a predicate holds an F42 checked integer expression.
pred_checked_int <- function(pred) isTRUE(pred$checked_int)

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
        if (isTRUE(expr$checked_int)) {
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
      agg_inputs <- unique(unlist(lapply(ast$aggregations, `[[`, "input_col")))
      agg_inputs <- agg_inputs[!is.na(agg_inputs)]
      needed <- opt_union(ast$groups, agg_inputs)
      ast$input <- prune_dead_columns(ast$input, needed, ast$groups)
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
      # `.by_group = TRUE` sorts by `ast$groups` too (lower_arrange())
      sort_cols <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
      needed <- opt_union(opt_union(required_cols, sort_cols), ast$groups)
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "reframe" = {
      ast$input <- prune_dead_columns(ast$input, infer_schema_memo(ast$input)$names)
      ast
    },
    "window" = {
      spec_cols <- unique(unlist(
        lapply(ast$specs, function(s) c(s$value_col, s$order_cols)),
        use.names = FALSE
      ))
      spec_cols <- spec_cols[!is.na(spec_cols)]
      needed <- opt_union(opt_union(required_cols, ast$group_cols), spec_cols)
      ast$input <- prune_dead_columns(ast$input, needed, ast$group_cols)
      ast
    },
    "rename" = {
      # Same old_names/new_names translation as push_down_projections()'s
      # "rename" case: without it, a mutate producing the pre-rename column
      # further down could be wrongly judged "not required" (since
      # `required_cols` would only ever contain the post-rename name) and
      # pruned away entirely -- confirmed empirically (Phase 12 audit),
      # worse than the projection-only bug since it drops a whole
      # expression, not just narrows a source projection.
      rev_idx <- match(required_cols, ast$new_names)
      translated <- required_cols
      hit <- !is.na(rev_idx)
      translated[hit] <- ast$old_names[rev_idx[hit]]
      # A positional rename (duplicated names) can't be translated by name:
      # everything below it stays.
      if (!is.null(ast$positions)) translated <- NULL
      ast$input <- prune_dead_columns(ast$input, translated, group_cols)
      ast
    },
    "distinct" = {
      needed <- opt_union(required_cols, ast$key_cols)
      if (!is.null(ast$key_positions)) needed <- NULL
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "slice" = {
      extra <- if (!is.null(ast$order_col)) ast$order_col else character(0)
      needed <- opt_union(required_cols, extra)
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
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
#' `extract_segment_above_barrier()` truncates a segment's bottom `$input`
#' to NULL, so `infer_schema()` inside such a segment sees a partial (or
#' empty) schema. "source" and "join" are the only node types without an
#' `$input`; a chain ending anywhere else was cut at a barrier.
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
#' own comments) -- the filter's segment has its `$input` truncated to NULL
#' at extraction, so `distinct` is never actually present as `ast$input` in
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
    new_filter <- ast_filter(input$input, ast$predicates)
    input$input <- new_filter
    return(input)
  }

  if (input$type == "select") {
    # Only push if select keeps all predicate columns (a positional select
    # renames nothing by name, but its columns can't be matched by name)
    if (!is.null(input$positions) || !all(filter_cols %in% input$columns)) {
      return(ast)
    }

    new_filter <- ast_filter(input$input, ast$predicates)
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
      input$left <- ast_filter(input$left, ast$predicates)
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
      input$left <- ast_filter(input$left, left_preds)
    }
    if (length(right_preds) > 0) {
      input$right <- ast_filter(input$right, right_preds)
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

  while (!is.null(current) && current$type == "filter") {
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
  result <- ast_filter(current, all_preds)
  result$input <- reorder_filters(result$input)
  result
}
