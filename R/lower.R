# AST Lowering - Convert AST to GPU Operations
#
# This module translates the optimized AST into actual GPU operations
# by calling the C++ functions in the cuplyr package.

#' Lower and execute an AST against a GPU table
#'
#' @param ast Optimized AST root node
#' @param source_ptr External pointer to source GPU table
#' @return External pointer to result GPU table
#' @keywords internal
lower_and_execute <- function(ast, source_ptr) {
  if (is.null(ast)) {
    return(source_ptr)
  }

  switch(ast$type,
    "source" = if (!is.null(ast$source_ptr)) ast$source_ptr else source_ptr,
    "select" = lower_select(ast, source_ptr),
    "filter" = lower_filter(ast, source_ptr),
    "mutate" = lower_mutate(ast, source_ptr),
    "arrange" = lower_arrange(ast, source_ptr),
    "summarise" = lower_summarise(ast, source_ptr),
    "join" = lower_join(ast, source_ptr),
    "distinct" = lower_distinct(ast, source_ptr),
    "slice" = lower_slice(ast, source_ptr),
    "rename" = lower_rename(ast, source_ptr),
    "window" = lower_window(ast, source_ptr),
    "barrier" = lower_and_execute(ast$input, source_ptr),
    stop("Unknown AST node type: ", ast$type, call. = FALSE)
  )
}

#' Lower select node
#' @keywords internal
lower_select <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)
  indices <- match(ast$columns, input_schema$names) - 1L
  gpu_select(input_ptr, indices)
}

#' Lower filter node
#'
#' Task T3 (Phase 1 expression-engine cutover, `scratchpad/phase1_expression_engine.md`
#' section 5): every predicate in the node is a `cuplyr_ir` node (see
#' `R/ir.R`, `R/ast.R::make_predicate()`). All of them are folded together
#' with `&` into a single expression -- one filter node is always exactly
#' one fused GPU kernel now, regardless of how many predicates it holds, so
#' there is no separate "fused vs sequential" branch (and no `ast$fused`
#' flag) to check.
#' @keywords internal
lower_filter <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  if (length(ast$predicates) == 0) {
    return(input_ptr)
  }

  combined_ir <- ast$predicates[[1]]$ir
  for (pred in ast$predicates[-1]) {
    combined_ir <- ir_call("&", list(combined_ir, pred$ir))
  }

  bound_ir <- ir_bind(combined_ir, input_schema)
  gpu_filter_expr(input_ptr, bound_ir)
}

#' Lower mutate node
#'
#' Task T5 (Phase 1 expression-engine cutover, `scratchpad/phase1_expression_engine.md`
#' section 5): each expression's IR is bound (`ir_bind()`) against a running
#' index map that mirrors `gpu_mutate_expr()`'s contract (`src/ops_expr.cpp`):
#' expression `i`'s top-level result always lands at ctx index `n_input + i`,
#' so a `col` node referencing an *earlier* expression's output (however many
#' `mutate()` calls it originally came from -- `fuse_mutates()` may have
#' merged several `ast_mutate` nodes into this one) must bind to that
#' *appended* index, never the original input column's position.
#'
#' `index_names`/`index_types` track this: they start as the input schema,
#' and after each expression, any stale (superseded) occurrence of its
#' output name is blanked to `NA` -- so a subsequent `match()` can't find the
#' old position -- before the name is appended at the new slot. This keeps
#' `match(name, index_names)` always returning the single live occurrence,
#' whose 0-based position is exactly the ctx index `gpu_mutate_expr()` will
#' place that expression's result at.
#' @keywords internal
lower_mutate <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  index_names <- input_schema$names
  index_types <- input_schema$types
  bound_exprs <- vector("list", length(ast$expressions))

  for (i in seq_along(ast$expressions)) {
    expr <- ast$expressions[[i]]

    lookup_schema <- list(names = index_names, types = index_types)
    bound_exprs[[i]] <- ir_bind(expr$ir, lookup_schema)

    stale <- which(index_names == expr$output_col)
    if (length(stale) > 0) {
      index_names[stale] <- NA_character_
    }
    index_names <- c(index_names, expr$output_col)
    index_types <- c(index_types, expr$output_type)
  }

  out_names <- vapply(ast$expressions, `[[`, character(1), "output_col")
  out_types <- vapply(ast$expressions, `[[`, character(1), "output_type")

  gpu_mutate_expr(input_ptr, bound_exprs, out_names, out_types, input_schema$names)
}

#' Lower arrange node
#' @keywords internal
lower_arrange <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  col_indices <- integer(length(ast$sort_specs))
  descending <- logical(length(ast$sort_specs))

  for (i in seq_along(ast$sort_specs)) {
    spec <- ast$sort_specs[[i]]
    col_indices[i] <- match(spec$col_name, input_schema$names) - 1L
    descending[i] <- isTRUE(spec$descending)
  }

  # Handle grouped arrange (.by_group = TRUE): dplyr's arrange.grouped_df
  # always prepends the group columns ASCENDING ahead of the user's own sort
  # keys (D8), even if the user's dots already sort by one of those columns
  # (possibly with desc()) -- that user key simply becomes a later, no-op
  # tiebreaker since the ascending group prepend already fully orders it.
  if (length(ast$groups) > 0) {
    group_indices <- match(ast$groups, input_schema$names) - 1L
    col_indices <- c(group_indices, col_indices)
    descending <- c(rep(FALSE, length(group_indices)), descending)
  }

  gpu_arrange(input_ptr, col_indices, descending)
}

#' Lower summarise node
#' @keywords internal
lower_summarise <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  # Build aggregation specs
  group_indices <- if (length(ast$groups) > 0) {
    match(ast$groups, input_schema$names) - 1L
  } else {
    integer(0)
  }

  agg_col_indices <- integer(length(ast$aggregations))
  agg_fns <- character(length(ast$aggregations))

  for (i in seq_along(ast$aggregations)) {
    agg <- ast$aggregations[[i]]
    agg_fns[i] <- agg$fn
    if (agg$fn == "n") {
      agg_col_indices[i] <- 0L  # n() doesn't need a column
    } else {
      agg_col_indices[i] <- match(agg$input_col, input_schema$names) - 1L
    }
  }

  gpu_summarise(input_ptr, group_indices, agg_col_indices, agg_fns)
}

#' Lower distinct node
#'
#' `gpu_distinct()` (`src/ops_distinct.cpp`) always returns every column of
#' its input, deduped on `ast$key_cols` (`cudf::stable_distinct()`, which
#' preserves input row order and keeps the first occurrence of each key --
#' matching dplyr's contract exactly). `ast$key_cols`'s positions in the
#' (already-mutated, if the caller had computed key expressions) input
#' schema are unchanged by the dedup itself, so the same `key_idx` used to
#' build the dedup key can be reused to project down to just those columns,
#' in the caller's requested order, for the `.keep_all = FALSE` case.
#' @keywords internal
lower_distinct <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  key_idx <- match(ast$key_cols, input_schema$names) - 1L

  result_ptr <- gpu_distinct(input_ptr, key_idx)

  if (!isTRUE(ast$keep_all)) {
    result_ptr <- gpu_select(result_ptr, key_idx)
  }

  result_ptr
}

#' Lower slice node
#'
#' Dispatches on `ast$mode` to the matching C++ entry point
#' (`src/ops_slice.cpp`). `nrow`-dependent resolution (the `n=`/`prop=`
#' clamp, and `slice()`'s index-vector validation/negative-index handling)
#' happens entirely in C++, since the input's actual row count is only
#' available once `ast$input` has actually been lowered/executed -- it
#' can't be computed from `infer_schema()` (which only tracks
#' names/types, not row counts) at R-side parse time, and a lazy AST may
#' not even have a concrete row count until this point in the pipeline
#' runs. See `src/ops_common.hpp::compute_slice_size()`.
#' @keywords internal
lower_slice <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)

  switch(ast$mode,
    "head" = gpu_slice_head(input_ptr, ast$amount, ast$is_prop),
    "tail" = gpu_slice_tail(input_ptr, ast$amount, ast$is_prop),
    "index" = gpu_slice_indices(input_ptr, ast$raw_indices),
    "rank" = {
      input_schema <- infer_schema(ast$input)
      col_idx <- match(ast$order_col, input_schema$names) - 1L

      result_ptr <- gpu_slice_rank(input_ptr, col_idx, ast$descending, ast$amount,
                                   ast$is_prop, ast$with_ties, ast$na_rm)

      if (isTRUE(ast$order_is_temp)) {
        keep_idx <- setdiff(seq_along(input_schema$names) - 1L, col_idx)
        result_ptr <- gpu_select(result_ptr, keep_idx)
      }

      result_ptr
    },
    stop("Unknown slice mode: ", ast$mode, call. = FALSE)
  )
}

#' Lower rename node
#'
#' Pure schema-only passthrough: column names live entirely in the R-side
#' schema (see `ast_rename()`, `R/ast.R`), never on the underlying GPU
#' table, so there is no GPU call to make here at all -- this simply lowers
#' `ast$input` and returns its pointer unchanged. `push_op()`
#' (`R/execute.R`) is what actually applies the new (renamed) schema to the
#' resulting `tbl_gpu`, via `infer_schema()`.
#' @keywords internal
lower_rename <- function(ast, source_ptr) {
  lower_and_execute(ast$input, source_ptr)
}

#' Lower join node
#' @keywords internal
lower_join <- function(ast, source_ptr) {
  left_ptr <- lower_and_execute(ast$left, source_ptr)
  right_ptr <- lower_and_execute(ast$right, source_ptr)

  left_schema <- infer_schema(ast$left)
  right_schema <- infer_schema(ast$right)

  left_key_idx <- match(ast$by$left, left_schema$names) - 1L
  right_key_idx <- match(ast$by$right, right_schema$names) - 1L

  right_drop <- if (!isTRUE(ast$keep)) ast$by$right else character(0)
  right_drop_idx <- if (length(right_drop) > 0) {
    match(right_drop, right_schema$names) - 1L
  } else {
    integer(0)
  }

  switch(ast$join_type,
    "left" = gpu_left_join(left_ptr, right_ptr, left_key_idx, right_key_idx, right_drop_idx),
    "inner" = gpu_inner_join(left_ptr, right_ptr, left_key_idx, right_key_idx, right_drop_idx),
    "full" = gpu_full_join(left_ptr, right_ptr, left_key_idx, right_key_idx, right_drop_idx),
    "right" = {
      # Implement right join via swapped left join, then reorder columns.
      # Shared with the eager path in right_join.tbl_gpu() (R/join.R) via
      # build_right_join_via_left() / resolve_right_join_select_idx() --
      # right_join.tbl_gpu() builds this exact ast_join("right", ...) node
      # whenever either input is lazy, so this branch is the live lazy path
      # and must stay in lockstep with the eager one (including the
      # keep = FALSE join-key coalescing fix in build_right_join_via_left()).
      plan <- build_right_join_via_left(left_schema, right_schema, ast$by,
                                        suffix = ast$suffix, keep = ast$keep)
      out <- gpu_left_join(right_ptr, left_ptr, right_key_idx, left_key_idx,
                           integer(0))
      idx <- resolve_right_join_select_idx(plan, left_schema, right_schema, ast$suffix)
      gpu_select(out, idx - 1L)
    },
    stop("Unknown join type: ", ast$join_type, call. = FALSE)
  )
}

#' Lower window node
#'
#' Phase 5, task W2 (`scratchpad/phase5_window_design.md` section 2.1):
#' binds every spec's `value_col`/`order_cols` (and the node's own
#' `group_cols`) to 0-based indices against `ast$input`'s schema, groups the
#' specs into "frames" keyed by `(order_cols, order_desc)`, and hands ONE
#' `gpu_window()` call (`src/ops_window.cpp`) to the C++ side: the
#' six-step stable-sort / gather / groupby-scan-or-shift / scatter-back
#' algorithm (section 2.2).
#'
#' `out_pos` (each spec's 0-based position among the node's own appended
#' columns) is simply the spec's position in `ast$specs` -- the same order
#' `infer_schema.ast_window()` (`R/ast.R`) appends spec output columns in, so
#' the two always agree on where a given spec's result lands in the final
#' schema.
#'
#' W2 scope only: the "shift" family (`lag()`/`lead()`) and the "scan"
#' family (`cumsum()`/`cummax()`/`cummin()`/`cumprod()`, plus bare
#' `row_number()`'s no-order COUNT_ALL form) -- see
#' `validate_window_spec_w2()`. Every other window kind (any `"rank"` spec
#' other than bare `row_number()`, and the entire `"agg"` family) is W4
#' work and hits a hard internal error here (never a user-facing message:
#' no verb decomposes an expression into one of those specs yet, so
#' reaching this function with one is a cuplyr bug, not a user mistake).
#' @keywords internal
lower_window <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  specs <- ast$specs
  if (length(specs) == 0) {
    return(input_ptr)
  }

  for (spec in specs) {
    validate_window_spec_w2(spec)
  }

  group_indices <- if (length(ast$group_cols) > 0) {
    match(ast$group_cols, input_schema$names) - 1L
  } else {
    integer(0)
  }

  # Group specs into frames keyed by (order_cols, order_desc), preserving
  # each frame's first-appearance order (irrelevant for correctness --
  # gpu_window() assembles the output purely from each spec's own
  # out_pos -- but keeps `show_query()`/debugging output deterministic).
  frame_keys <- vapply(specs, function(s) {
    paste(c(s$order_cols, ifelse(s$order_desc, "D", "A")), collapse = "")
  }, character(1))
  uniq_keys <- unique(frame_keys)

  frames <- lapply(uniq_keys, function(k) {
    member_idx <- which(frame_keys == k)
    s0 <- specs[[member_idx[1]]]

    order_idx <- if (length(s0$order_cols) > 0) {
      match(s0$order_cols, input_schema$names) - 1L
    } else {
      integer(0)
    }

    cpp_specs <- lapply(member_idx, function(i) {
      build_window_cpp_spec(specs[[i]], i - 1L, input_schema)
    })

    list(order_idx = order_idx, order_desc = s0$order_desc, specs = cpp_specs)
  })

  gpu_window(input_ptr, frames, group_indices)
}

#' Is a window spec in W2/W3's lowerable set?
#'
#' The non-throwing predicate half of `validate_window_spec_w2()` (below),
#' factored out so Phase 5 task W3's `mutate()` wiring (`R/mutate.R`,
#' `mutate_window()`) can check every spec a decomposition plan produces
#' *before* pushing any AST node -- a window call whose spec isn't lowerable
#' yet (any `"rank"` spec other than bare `row_number()`, or the entire
#' `"agg"` family, both W4 work) must fall back to `gpu_fallback()` cleanly,
#' never surface `validate_window_spec_w2()`'s internal-error text to a
#' user, and never partially push GPU ops before discovering the need to
#' fall back (see `scratchpad/phase5_window_design.md` section 1.3's "Never
#' half-lower").
#'
#' See `lower_window()`'s own docs and section 2.2's RANK-spec constraint
#' ("a rank spec's value col must equal the frame's sole order col"): bare
#' `row_number()` (no value column at all, i.e. `R/window.R`'s
#' `extract_one_window_call()` only sets `order_cols` for a `"rank"`-kind
#' spec that HAS a value column) is the one `"rank"`-kind spec that
#' trivially satisfies this (it has no order col to compare against, and
#' none is needed for a COUNT_ALL scan) -- every other `"rank"` spec
#' (`row_number(x)`, `min_rank()`, `dense_rank()`, `percent_rank()`,
#' `cume_dist()`) and the entire `"agg"` family are W4 work.
#'
#' @param spec A `window_spec()` structure (`R/window.R`)
#' @return `TRUE`/`FALSE`
#' @keywords internal
window_spec_w3_lowerable <- function(spec) {
  if (identical(spec$fn, "row_number") && length(spec$order_cols) == 0) {
    return(TRUE)
  }

  entry <- ir_call_registry[[spec$fn]]
  kind <- if (!is.null(entry)) entry$window$kind else NULL
  identical(kind, "shift") || identical(kind, "scan")
}

#' Reject any window spec whose kind W2's C++ kernel doesn't implement yet
#'
#' See `window_spec_w3_lowerable()`'s docs for the shared predicate this
#' wraps with user-facing (well, internal-error-facing: see that function's
#' docs) `stop()` calls.
#' @keywords internal
validate_window_spec_w2 <- function(spec) {
  if (window_spec_w3_lowerable(spec)) {
    return(invisible(TRUE))
  }

  entry <- ir_call_registry[[spec$fn]]
  kind <- if (!is.null(entry)) entry$window$kind else NULL

  if (identical(kind, "rank")) {
    stop("internal: window kind 'rank' (fn = '", spec$fn, "') is not implemented ",
         "until Phase 5 W4 (ranked row_number()/min_rank()/dense_rank()/",
         "percent_rank()/cume_dist()); W2 only supports bare row_number() ",
         "(the no-order COUNT_ALL form).", call. = FALSE)
  }
  if (identical(kind, "agg")) {
    stop("internal: window kind 'agg' (fn = '", spec$fn, "') is not implemented ",
         "until Phase 5 W4 (mean()/sum()/min()/max()/n()/sd()/var()/first()/",
         "last()/nth()); W2 only supports the scan and shift families.",
         call. = FALSE)
  }
  stop("internal: lower_window() received an unrecognized window fn '",
       spec$fn, "'.", call. = FALSE)
}

#' Translate one R-side `window_spec()` into the plain list `gpu_window()`
#' (`src/ops_window.cpp`) expects
#'
#' `default_value`/`default_valid` collapse `window_spec()`'s `default`
#' field (an IR literal node, or `NULL`) into the shape `build_scalar()`
#' (`src/expr_eval.hpp`) already accepts elsewhere: no default supplied,
#' and an explicit `default = NA` (`spec$default$na`), both mean the same
#' thing for `lag()`/`lead()` (an invalid/NA fill scalar of the value
#' column's own type) -- so both collapse to `default_valid = FALSE`,
#' `default_value = NULL`.
#'
#' @param spec A `window_spec()` structure
#' @param out_pos 0-based position of this spec among the whole
#'   `ast_window` node's appended columns
#' @param schema The node input's schema (`list(names=, types=)`)
#' @return A plain list: `list(fn, value_idx, out_type, out_pos, n,
#'   default_value, default_valid, na_rm)`
#' @keywords internal
build_window_cpp_spec <- function(spec, out_pos, schema) {
  value_idx <- if (is.na(spec$value_col)) {
    NA_integer_
  } else {
    match(spec$value_col, schema$names) - 1L
  }

  default_value <- NULL
  default_valid <- FALSE
  if (!is.null(spec$default) && !isTRUE(spec$default$na)) {
    default_value <- spec$default$value
    default_valid <- TRUE
  }

  list(
    fn = spec$fn,
    value_idx = value_idx,
    out_type = spec$output_type,
    out_pos = as.integer(out_pos),
    n = if (is.na(spec$n)) NA_integer_ else as.integer(spec$n),
    default_value = default_value,
    default_valid = default_valid,
    na_rm = isTRUE(spec$na_rm)
  )
}
