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

  # Handle barriers: optimize segments separately
  ast <- optimize_with_barriers(ast)

  ast
}

#' Optimize AST respecting barrier nodes
#'
#' @param ast Root AST node
#' @return Optimized AST
#' @keywords internal
optimize_with_barriers <- function(ast) {
  if (is.null(ast)) return(NULL)

  if (is_barrier(ast)) {
    # Optimize below the barrier, not across
    if (!is.null(ast$input)) {
      ast$input <- optimize_with_barriers(ast$input)
    }
    return(ast)
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
    required_cols <- infer_schema(ast)$names

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

  ast |>
    push_down_projections(required_cols) |>
    fuse_mutates() |>
    prune_dead_columns(required_cols) |>
    push_down_filters() |>
    reorder_filters()
}

#' Find the next barrier node walking down from root
#'
#' @param ast AST node
#' @return Barrier node or NULL
#' @keywords internal
find_next_barrier <- function(ast) {
  if (is.null(ast)) return(NULL)
  if (is_barrier(ast)) return(ast)
  find_next_barrier(ast$input)
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
    required_cols <- infer_schema(ast)$names
  }

  # Always preserve group columns
  required_cols <- union(required_cols, group_cols)

  switch(ast$type,
    "source" = {
      keep <- intersect(ast$schema$names, required_cols)
      if (length(keep) < length(ast$schema$names)) {
        ast_select(ast, keep)
      } else {
        ast
      }
    },
    "filter" = {
      pred_cols <- unique(unlist(lapply(ast$predicates, `[[`, "cols")))
      needed <- union(required_cols, pred_cols)
      ast$input <- push_down_projections(ast$input, needed, group_cols)

      extra_cols <- setdiff(pred_cols, required_cols)
      if (length(extra_cols) > 0) {
        ast_select(ast, required_cols)
      } else {
        ast
      }
    },
    "mutate" = {
      expr_inputs <- unique(unlist(lapply(ast$expressions, `[[`, "input_cols")))
      outputs <- vapply(ast$expressions, `[[`, character(1), "output_col")
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
      input_schema_names <- infer_schema(ast$input)$names
      new_outputs <- setdiff(outputs, input_schema_names)
      needed <- union(setdiff(required_cols, new_outputs), expr_inputs)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "select" = {
      ast$input <- push_down_projections(ast$input, ast$columns, group_cols)
      ast
    },
    "join" = {
      if (ast$join_type %in% c("semi", "anti")) {
        # semi/anti (Phase 7 J2, design section 1.2): the output is x's
        # (left's) schema verbatim -- every `required_cols` entry is
        # already a left column, no origin split needed. The right side
        # contributes nothing to the output at all; it only needs its own
        # join-key columns (`by$right`) to probe against.
        left_needed <- union(required_cols, ast$by$left)
        right_needed <- unique(ast$by$right)
      } else {
        left_schema <- infer_schema(ast$left)
        right_schema <- infer_schema(ast$right)
        info <- build_join_output_info(left_schema, right_schema, ast$by,
                                       suffix = ast$suffix, keep = ast$keep)
        req <- intersect(required_cols, info$names)
        idx <- match(req, info$names)
        idx <- idx[!is.na(idx)]
        left_needed <- unique(c(ast$by$left, info$source_names[idx][info$origin[idx] == "left"]))
        right_needed <- unique(c(ast$by$right, info$source_names[idx][info$origin[idx] == "right"]))
      }

      ast$left <- push_down_projections(ast$left, left_needed, group_cols)
      ast$right <- push_down_projections(ast$right, right_needed, group_cols)
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
      sort_cols <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
      needed <- union(required_cols, sort_cols)
      ast$input <- push_down_projections(ast$input, needed, group_cols)
      ast
    },
    "summarise" = {
      agg_inputs <- unique(unlist(lapply(ast$aggregations, `[[`, "input_col")))
      agg_inputs <- agg_inputs[!is.na(agg_inputs)]
      needed <- union(ast$groups, agg_inputs)
      ast$input <- push_down_projections(ast$input, needed, ast$groups)
      ast
    },
    "window" = {
      spec_cols <- unique(unlist(
        lapply(ast$specs, function(s) c(s$value_col, s$order_cols)),
        use.names = FALSE
      ))
      spec_cols <- spec_cols[!is.na(spec_cols)]
      needed <- union(union(required_cols, ast$group_cols), spec_cols)
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
      ast$input <- push_down_projections(ast$input, translated, group_cols)
      ast
    },
    "distinct" = {
      # `key_cols` are required for the dedup itself regardless of whether
      # `required_cols` (computed from this node's own OUTPUT schema)
      # references them -- true even for `.keep_all = TRUE`, where
      # `required_cols` can be a strict subset of the full input schema.
      needed <- union(required_cols, ast$key_cols)
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
      needed <- union(required_cols, extra)
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

  # Guard 2: Check for dependencies
  lower_outputs <- vapply(lower$expressions, `[[`, character(1), "output_col")
  upper_inputs <- unique(unlist(lapply(upper$expressions, `[[`, "input_cols")))

  has_dependency <- any(upper_inputs %in% lower_outputs)

  if (has_dependency) {
    combined_exprs <- toposort_expressions(combined_exprs)
    if (is.null(combined_exprs)) {
      return(NULL)  # Cycle detected or too complex
    }
  }

  # Guard 3: Count intermediate columns
  intermediate_uses <- table(upper_inputs[upper_inputs %in% lower_outputs])
  n_intermediates <- length(intermediate_uses)
  max_reuse <- if (length(intermediate_uses) > 0) max(intermediate_uses) else 0

  if (n_intermediates > 4 || max_reuse > 3) {
    return(NULL)
  }

  ast_mutate(lower$input, combined_exprs)
}

#' Topologically sort expressions to respect dependencies
#'
#' @param exprs List of expression structures
#' @return Sorted list or NULL if cycle detected
#' @keywords internal
toposort_expressions <- function(exprs) {
  if (length(exprs) == 0) return(list())

  outputs <- vapply(exprs, `[[`, character(1), "output_col")
  names(exprs) <- outputs

  # Kahn's algorithm
  in_degree <- integer(length(exprs))
  names(in_degree) <- outputs

  for (i in seq_along(exprs)) {
    deps <- intersect(exprs[[i]]$input_cols, outputs)
    in_degree[i] <- length(deps)
  }

  result <- list()
  queue <- names(in_degree)[in_degree == 0]

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]
    result <- c(result, list(exprs[[current]]))

    for (i in seq_along(exprs)) {
      if (current %in% exprs[[i]]$input_cols) {
        in_degree[i] <- in_degree[i] - 1
        if (in_degree[i] == 0) {
          queue <- c(queue, outputs[i])
        }
      }
    }
  }

  if (length(result) != length(exprs)) {
    return(NULL)  # Cycle detected
  }

  result
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
    required_cols <- infer_schema(ast)$names
  }

  required_cols <- union(required_cols, group_cols)

  switch(ast$type,
    "source" = {
      ast
    },
    "filter" = {
      pred_cols <- unique(unlist(lapply(ast$predicates, `[[`, "cols")))
      needed <- union(required_cols, pred_cols)
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
        if (expr$output_col %in% required) {
          keep[i] <- TRUE
          required <- union(setdiff(required, expr$output_col), expr$input_cols)
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
      needed <- union(ast$groups, agg_inputs)
      ast$input <- prune_dead_columns(ast$input, needed, ast$groups)
      ast
    },
    "select" = {
      keep <- intersect(ast$columns, required_cols)
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
        left_needed <- union(required_cols, ast$by$left)
        right_needed <- unique(ast$by$right)
      } else {
        left_schema <- infer_schema(ast$left)
        right_schema <- infer_schema(ast$right)
        info <- build_join_output_info(left_schema, right_schema, ast$by,
                                       suffix = ast$suffix, keep = ast$keep)
        req <- intersect(required_cols, info$names)
        idx <- match(req, info$names)
        idx <- idx[!is.na(idx)]
        left_needed <- unique(c(ast$by$left, info$source_names[idx][info$origin[idx] == "left"]))
        right_needed <- unique(c(ast$by$right, info$source_names[idx][info$origin[idx] == "right"]))
      }

      ast$left <- prune_dead_columns(ast$left, left_needed, group_cols)
      ast$right <- prune_dead_columns(ast$right, right_needed, group_cols)
      ast
    },
    # Same reachability rationale as push_down_projections()'s equivalent
    # cases (Phase 12 audit note above it): "arrange"/"summarise"/"window"/
    # "rename"/"distinct"/"slice" are all barriers, so these cases are only
    # ever reached via a join's `left`/`right` recursion below, never via
    # the normal barrier-splitting entry point.
    "arrange" = {
      sort_cols <- vapply(ast$sort_specs, `[[`, character(1), "col_name")
      needed <- union(required_cols, sort_cols)
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "window" = {
      spec_cols <- unique(unlist(
        lapply(ast$specs, function(s) c(s$value_col, s$order_cols)),
        use.names = FALSE
      ))
      spec_cols <- spec_cols[!is.na(spec_cols)]
      needed <- union(union(required_cols, ast$group_cols), spec_cols)
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
      ast$input <- prune_dead_columns(ast$input, translated, group_cols)
      ast
    },
    "distinct" = {
      needed <- union(required_cols, ast$key_cols)
      ast$input <- prune_dead_columns(ast$input, needed, group_cols)
      ast
    },
    "slice" = {
      extra <- if (!is.null(ast$order_col)) ast$order_col else character(0)
      needed <- union(required_cols, extra)
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

# -----------------------------------------------------------------------------
# Pass 4: Filter Pushdown
# -----------------------------------------------------------------------------

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

  filter_cols <- unique(unlist(lapply(ast$predicates, `[[`, "cols")))

  if (input$type == "mutate") {
    mutate_outputs <- vapply(input$expressions, `[[`, character(1), "output_col")

    # Do not push if filter references any mutate outputs (including replacements)
    if (length(intersect(filter_cols, mutate_outputs)) > 0) {
      return(ast)
    }

    # Swap: filter below mutate
    new_filter <- ast_filter(input$input, ast$predicates)
    input$input <- new_filter
    return(input)
  }

  if (input$type == "select") {
    # Only push if select keeps all predicate columns
    if (!all(filter_cols %in% input$columns)) {
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
      input$left <- ast_filter(input$left, ast$predicates)
      return(input)
    }

    left_schema <- infer_schema(input$left)
    right_schema <- infer_schema(input$right)
    info <- build_join_output_info(left_schema, right_schema, input$by,
                                   suffix = input$suffix, keep = input$keep)

    classify_predicate <- function(pred) {
      sides <- unique(info$origin[match(pred$cols, info$names)])
      sides <- sides[!is.na(sides)]
      if (length(sides) != 1) return("both")
      sides
    }

    left_preds <- list()
    right_preds <- list()
    stay_preds <- list()

    for (pred in ast$predicates) {
      side <- classify_predicate(pred)
      # cross (Phase 7 J3, design section 1.2/2): behaves like "inner" for
      # filter pushdown on BOTH sides -- a cross join has no unmatched-row
      # concept at all (every left row pairs with every right row), so a
      # left-only predicate can always push to the left side and a
      # right-only predicate can always push to the right side without
      # changing the result, exactly like an inner join's own left/right
      # pushdown rule.
      if (side == "left" && input$join_type %in% c("inner", "left", "cross")) {
        left_preds <- c(left_preds, list(pred))
      } else if (side == "right" && input$join_type %in% c("inner", "right", "cross")) {
        right_preds <- c(right_preds, list(pred))
      } else {
        stay_preds <- c(stay_preds, list(pred))
      }
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

  # Collect consecutive filter chain
  filters <- list()
  current <- ast

  while (!is.null(current) && current$type == "filter") {
    filters <- c(filters, list(current$predicates))
    current <- current$input
  }

  if (length(filters) <= 1) {
    ast$input <- reorder_filters(ast$input)
    return(ast)
  }

  # Flatten predicates
  all_preds <- unlist(filters, recursive = FALSE)

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
