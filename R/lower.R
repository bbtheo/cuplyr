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
  expressions <- ast$expressions

  # Vector literals (`y = 1:5`): upload each as a temp column appended to the
  # input, make the expression a copy of it, and drop the temps afterwards.
  vec_idx <- which(vapply(expressions, function(e) identical(e$ir$kind, "vec"), logical(1)))
  n_input_cols <- length(input_schema$names)
  if (length(vec_idx) > 0) {
    n_rows <- gpu_dim(input_ptr)[[1L]]
    tmp_names <- paste0("..vec", seq_along(vec_idx), "..")
    tmp_cols <- list()
    for (k in seq_along(vec_idx)) {
      vec <- expressions[[vec_idx[k]]]$ir
      if (length(vec$value) != n_rows) {
        mutate_vector_size_error(n_rows, vec)
      }
      tmp_cols[[tmp_names[k]]] <- vec$value
      expressions[[vec_idx[k]]]$ir <- ir_col(tmp_names[k])
    }
    upload <- tbl_gpu(tibble::as_tibble(tmp_cols, .name_repair = "minimal"), lazy = FALSE)
    input_ptr <- gpu_bind_cols_impl(list(input_ptr, upload$ptr))
    input_schema$names <- c(input_schema$names, tmp_names)
    input_schema$types <- c(input_schema$types, upload$schema$types)
  }

  index_names <- input_schema$names
  index_types <- input_schema$types
  bound_exprs <- vector("list", length(expressions))

  for (i in seq_along(expressions)) {
    expr <- expressions[[i]]

    lookup_schema <- list(names = index_names, types = index_types)
    bound_exprs[[i]] <- ir_bind(expr$ir, lookup_schema)

    stale <- which(index_names == expr$output_col)
    if (length(stale) > 0) {
      index_names[stale] <- NA_character_
    }
    index_names <- c(index_names, expr$output_col)
    index_types <- c(index_types, expr$output_type)
  }

  out_names <- vapply(expressions, `[[`, character(1), "output_col")
  out_types <- vapply(expressions, `[[`, character(1), "output_type")
  # A bare factor-column copy is typed DICTIONARY32 (logical); the column
  # that comes out is the INT32 codes. Any other factor-valued expression is
  # sent to the CPU by parse_mutate_dots(); if one gets here anyway, fail
  # rather than return codes with no levels.
  is_copy <- vapply(expressions, function(e) identical(e$ir$kind, "col"), logical(1))
  bad <- out_types == "DICTIONARY32" & !is_copy
  if (any(bad)) {
    stop("Can't compute factor-valued expression `", out_names[bad][1],
         "` on the GPU.", call. = FALSE)
  }
  out_types <- gpu_physical_type(out_types)

  result <- gpu_mutate_expr(input_ptr, bound_exprs, out_names, out_types, input_schema$names)
  if (length(vec_idx) > 0) {
    # The temps are input columns, so they kept their slots after the originals.
    temp_slots <- n_input_cols + seq_along(vec_idx) - 1L
    result <- gpu_select(result, setdiff(seq_len(gpu_dim(result)[[2L]]) - 1L, temp_slots))
  }
  result
}

# A vector literal whose length isn't the table's row count: run dplyr itself
# on an empty-column tibble of that height with the original dot, so the error
# (class, "must be size N or 1, not K", the "Inlined constant" variant) is
# dplyr's own.
mutate_vector_size_error <- function(n_rows, vec) {
  tb <- tibble::new_tibble(list(), nrow = n_rows)
  rlang::inject(dplyr::mutate(tb, !!vec$name := !!vec$quo))
  stop("Can't recycle `", vec$name, "` (size ", length(vec$value), ") to size ", n_rows, ".",
       call. = FALSE)
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
  # Phase 6, task 6.1 (Fix A): per-aggregation na.rm flag, threaded through
  # to gpu_summarise() (src/ops_groupby.cpp) so it knows which aggregations
  # need the extra whole-group-null propagation step (na_rm == FALSE, R's
  # own default) vs. cudf's native null-excluding behavior (na_rm == TRUE).
  na_rm_flags <- logical(length(ast$aggregations))
  # Phase 6, task 6.2: per-aggregation extra scalar shape parameter --
  # quantile()'s probability, nth()'s (possibly negative) element
  # position, NA_real_ for every other function -- see make_aggregation()'s
  # own docs.
  agg_extra <- numeric(length(ast$aggregations))

  for (i in seq_along(ast$aggregations)) {
    agg <- ast$aggregations[[i]]
    agg_fns[i] <- agg$fn
    na_rm_flags[i] <- isTRUE(agg$na_rm)
    agg_extra[i] <- if (is.null(agg$extra)) NA_real_ else agg$extra
    if (agg$fn == "n") {
      agg_col_indices[i] <- 0L  # n() doesn't need a column
    } else {
      agg_col_indices[i] <- match(agg$input_col, input_schema$names) - 1L
    }
  }

  # F07: min()/max() over INT32/FLOAT64 apply base R's empty-group identity
  # (see make_aggregation()'s `identity`).
  identity_flags <- vapply(ast$aggregations, function(agg) isTRUE(agg$identity), logical(1))

  result <- gpu_summarise(input_ptr, group_indices, agg_col_indices, agg_fns,
                          na_rm_flags, agg_extra, identity_flags)

  # dplyr warns whenever min()/max() hit the identity, with base R's text.
  used <- attr(result, "identity_used")
  for (i in which(used)) {
    fn <- ast$aggregations[[i]]$fn
    warning("no non-missing arguments to ", fn, "; returning ",
            if (identical(fn, "min")) "Inf" else "-Inf", call. = FALSE)
  }

  # An aggregate named like a grouping column replaces that key in place.
  agg_names <- vapply(ast$aggregations, `[[`, character(1), "output_col")
  out_idx <- summarise_output_index(ast$groups, agg_names)
  if (!identical(out_idx, seq_along(out_idx))) {
    result <- gpu_select(result, out_idx - 1L)
  }

  result
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
    # `x[i, ]`: tibble semantics (NA / out-of-range -> all-NA row), resolved
    # against the real row count (unless `[.tbl_gpu` already did), then one
    # NULLIFY gather.
    "bracket" = {
      pos <- if (isTRUE(ast$rows_resolved)) {
        ast$raw_indices
      } else {
        bracket_row_positions(ast$raw_indices, gpu_dim(input_ptr)[1],
                              ast$bracket_arg, ast$bracket_call)
      }
      gpu_gather_rows(list(input_ptr), as.numeric(pos) - 1)
    },
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
#'
#' The four "mutating" join types (`"left"`/`"inner"`/`"full"`/`"right"`) all
#' share the same C++ calling convention -- `left_key_idx`/`right_key_idx`/
#' `right_drop_idx`/`nulls_equal` -- and their GPU entry points
#' (`gpu_left_join()`/`gpu_inner_join()`/`gpu_full_join()`/`gpu_right_join()`,
#' `src/ops_join.cpp`) all return `list(ptr, diag)` (Phase 7 J4's map-pipeline
#' restructure): `ptr` is the result table's external pointer, `diag` is a
#' first-offender diagnostics list consumed by `check_join_cardinality()`
#' below (multiple=/unmatched=/relationship=, Phase 7 J5). `right_join()`
#' used to be lowered via a swapped `left_join(y, x)` plus a column
#' reorder/select (`build_right_join_via_left()`/
#' `resolve_right_join_select_idx()`, since removed) -- `gpu_right_join()` is
#' now native and takes exactly the same argument shape as the other three,
#' so no special-casing is needed here beyond the switch itself.
#'
#' Non-equi `join_by()` (Phase 7 J7, `scratchpad/phase7_joins_design.md`
#' section 1.4/2): `ast$by$op` carries each condition's real operator (not
#' just `"=="`) whenever `join_by_natively_supported()` (R/join.R) let the
#' join reach this AST shape at all -- [join_equi_spec()]/[join_cond_spec()]
#' split it back into an equality-key subset (routed through the same
#' hash-join machinery every equi join uses, via `mixed_join`'s own
#' equality tables) and a residual non-equi subset, folded into one AST
#' condition tree by `join_cond_ir_from_spec()` + [ir_bind_join()] and
#' handed to `gpu_cond_join()` (`src/ops_join.cpp`), which dispatches to
#' `mixed_*`/`conditional_*` cuDF entry points depending on whether any
#' equi keys survived the split. `gpu_cond_join()` reuses the exact same
#' `list(ptr, diag)` / bare-`ptr` return shapes as the equi paths below (see
#' its own C++ comment for the `join_kind == "right"` swapped-role decision),
#' so `check_join_cardinality()` applies identically either way.
#' @keywords internal
lower_join <- function(ast, source_ptr) {
  left_ptr <- lower_and_execute(ast$left, source_ptr)
  right_ptr <- lower_and_execute(ast$right, source_ptr)

  left_schema <- infer_schema(ast$left)
  right_schema <- infer_schema(ast$right)

  if (isTRUE(ast$cross_by)) {
    return(lower_cross_by_join(ast, left_ptr, right_ptr, left_schema, right_schema))
  }

  # na_matches = "na" (the default) matches NA keys against each other
  # (cudf::null_equality::EQUAL); "never" treats every NA key as distinct,
  # so no NA ever matches anything, on either side (::UNEQUAL). See
  # scratchpad/phase7_joins_design.md task J1.
  nulls_equal <- !identical(ast$na_matches, "never")

  mutating_join_types <- c("left", "inner", "full", "right")

  # Key position per spec pair (0-based), then the A3 key-cast plan: when an
  # equi pair's two sides have different (compatible) types,
  # apply_join_key_casts() casts them to the common type -- in place for a
  # merged key, on trailing temporary copies when the original column must
  # survive (keep = TRUE, semi/anti). `temps` is NULL (and nothing below
  # changes) when all key types already match.
  left_pair_idx <- match(ast$by$left, left_schema$names) - 1L
  right_pair_idx <- match(ast$by$right, right_schema$names) - 1L
  temps <- NULL
  cond_by <- ast$by
  cond_left_schema <- left_schema
  cond_right_schema <- right_schema
  if (!is.null(ast$by$key_casts)) {
    plan <- apply_join_key_casts(ast, left_ptr, right_ptr, left_schema, right_schema,
                                 left_pair_idx, right_pair_idx)
    left_ptr <- plan$left_ptr
    right_ptr <- plan$right_ptr
    left_pair_idx <- plan$left_pair_idx
    right_pair_idx <- plan$right_pair_idx
    temps <- plan$temps
    # Non-equi conditions are bound against the schemas extended with the
    # temporary cast copies (named in cond_by), so both operands already
    # have the common type when the evaluator sees them.
    cond_by <- plan$cond_by
    cond_left_schema <- plan$left_schema
    cond_right_schema <- plan$right_schema
  }
  finish <- function(ptr) {
    if (is.null(temps)) ptr else drop_join_key_temps(ptr, temps)
  }

  # Phase 7 J7: at least one condition's operator isn't "==" -- a native
  # non-equi join_by() (join_by_natively_supported() already screened out
  # closest()/rolling filters, unrecognized operators, and STRING columns
  # before this AST node was ever built; see R/join.R). `keep` is NOT
  # unconditionally TRUE here (Phase 7 J7 finding, see
  # resolve_join_keep()/build_join_output_info()'s own docs, R/join.R):
  # dplyr's real `keep = NULL` default drops the right key column (and
  # coalesces) PER CONDITION, only for the genuinely equi (`op == "=="`)
  # entries -- a mixed `join_by(c == d, a >= b)` drops `d` but keeps both
  # `a`/`b`. `right_drop`/coalescing below therefore only ever applies to
  # `equi_spec`'s columns, mirroring gpu_full_join()'s/gpu_right_join()'s
  # own `right_drop`-gated coalescing exactly.
  is_non_equi <- length(ast$by$op) > 0 && any(ast$by$op != "==")

  if (is_non_equi) {
    equi_spec <- join_equi_spec(ast$by)
    cond_spec <- join_cond_spec(cond_by)

    is_eq <- ast$by$op == "=="
    equi_left_idx <- left_pair_idx[is_eq]
    equi_right_idx <- right_pair_idx[is_eq]

    right_drop <- if (!isTRUE(ast$keep)) equi_spec$right else character(0)
    right_drop_idx <- if (length(right_drop) > 0) {
      match(right_drop, right_schema$names) - 1L
    } else {
      integer(0)
    }

    cond_ir <- ir_bind_join(join_cond_ir_from_spec(cond_spec), cond_left_schema,
                            cond_right_schema)

    if (ast$join_type %in% mutating_join_types) {
      result <- gpu_cond_join(left_ptr, right_ptr, equi_left_idx, equi_right_idx,
                              right_drop_idx, cond_ir, ast$join_type, nulls_equal,
                              ast$multiple)

      check_join_cardinality(result$diag, ast$join_type, ast$multiple,
                             ast$relationship, ast$unmatched)

      return(finish(result$ptr))
    }

    # semi/anti (no multiple=/unmatched=/relationship= in dplyr's own
    # signature for either -- see the equi semi/anti branch below; `keep`
    # is likewise not part of dplyr's semi_join()/anti_join() signature at
    # all, and the output is x's own schema verbatim regardless, so
    # right_drop_idx is computed uniformly above but never consulted by
    # gpu_cond_join()'s semi/anti branch).
    return(finish(gpu_cond_join(left_ptr, right_ptr, equi_left_idx, equi_right_idx,
                                right_drop_idx, cond_ir, ast$join_type, nulls_equal, "all")))
  }

  left_key_idx <- left_pair_idx
  right_key_idx <- right_pair_idx

  right_drop <- if (!isTRUE(ast$keep)) ast$by$right else character(0)
  right_drop_idx <- if (length(right_drop) > 0) {
    match(right_drop, right_schema$names) - 1L
  } else {
    integer(0)
  }

  if (ast$join_type %in% mutating_join_types) {
    # `multiple=` filtering happens device-side (src/ops_join.cpp
    # build_join_result(), Phase 7 J5) -- passed straight through as a
    # string; "all" (the default) is a zero-cost no-op there. `diag`
    # (join_map_stats(), computed BEFORE filtering) is consumed by
    # check_join_cardinality() below to raise dplyr's exact
    # relationship=/unmatched= conditions -- see its own roxygen
    # (R/join.R) for the full precedence rules.
    result <- switch(ast$join_type,
      "left" = gpu_left_join(left_ptr, right_ptr, left_key_idx, right_key_idx,
                             right_drop_idx, nulls_equal, ast$multiple),
      "inner" = gpu_inner_join(left_ptr, right_ptr, left_key_idx, right_key_idx,
                               right_drop_idx, nulls_equal, ast$multiple),
      "full" = gpu_full_join(left_ptr, right_ptr, left_key_idx, right_key_idx,
                             right_drop_idx, nulls_equal, ast$multiple),
      # right (Phase 7 J4): native via cudf::inner_join() for matched pairs
      # plus a filtered_join anti-join for unmatched-y rows, assembled and
      # fed through the same stable-sort/gather/coalesce pipeline every
      # other mutating join uses -- see gpu_right_join()'s own comment
      # (src/ops_join.cpp) for the sentinel/sort-order argument that
      # reproduces dplyr's x-matched-order-then-unmatched-y-tail contract.
      "right" = gpu_right_join(left_ptr, right_ptr, left_key_idx, right_key_idx,
                               right_drop_idx, nulls_equal, ast$multiple)
    )

    check_join_cardinality(result$diag, ast$join_type, ast$multiple,
                           ast$relationship, ast$unmatched)

    return(finish(result$ptr))
  }

  switch(ast$join_type,
    # semi/anti (Phase 7 J2): native via cudf::filtered_join
    # (gpu_semi_anti_join(), src/ops_join.cpp) -- builds the hash table from
    # the right/key table, probes with the left table, sorts the returned
    # (unordered, non-duplicated) left-row index vector ascending before
    # gathering the FULL left table (no right columns at all -- see
    # infer_schema.ast_join's semi/anti branch, x's schema verbatim). No
    # diag list here -- semi_join()/anti_join() have no multiple=/unmatched=/
    # relationship= in dplyr's own signatures.
    "semi" = finish(gpu_semi_anti_join(left_ptr, right_ptr, left_key_idx, right_key_idx,
                                       FALSE, nulls_equal)),
    "anti" = finish(gpu_semi_anti_join(left_ptr, right_ptr, left_key_idx, right_key_idx,
                                       TRUE, nulls_equal)),
    # cross (Phase 7 J3): native via cudf::cross_join (gpu_cross_join(),
    # src/ops_join.cpp) -- no keys, no drops; the cartesian product's column
    # order (left's columns, then right's) already matches
    # infer_schema.ast_join's regular (non-semi/anti) branch, since
    # cross_join.tbl_gpu() (R/join.R) builds an empty spec + keep = TRUE, and
    # build_join_output_info() with keep = TRUE never drops any right column.
    "cross" = gpu_cross_join(left_ptr, right_ptr),
    stop("Unknown join type: ", ast$join_type, call. = FALSE)
  )
}

#' Cast one join key column to its common type (A3)
#'
#' A factor key meeting a character key is decoded to its labels with
#' `gpu_decode_factor()` -- never `gpu_cast_column()`, which would only see
#' the physical INT32 codes. Everything else is a numeric/timestamp ladder
#' step handled by `gpu_cast_column()`. The column is replaced in place.
#' @keywords internal
cast_join_key_column <- function(ptr, idx, from_type, to_type, levels) {
  if (identical(from_type, "DICTIONARY32")) {
    if (is.null(levels)) {
      stop("Can't join on a factor column whose levels are unknown: ",
           "its factor metadata was lost earlier in the pipeline, so the ",
           "codes cannot be decoded to labels.", call. = FALSE)
    }
    gpu_decode_factor(ptr, idx, levels)
  } else {
    gpu_cast_column(ptr, idx, to_type)
  }
}

#' Apply a join's key-cast plan (`by$key_casts`, see `resolve_join_key_types()`)
#'
#' Merged equi keys (`keep != TRUE`, mutating joins): cast IN PLACE on both
#' sides -- the single output key column has the common type anyway, and
#' right/full coalescing then fills unmatched-y rows from the cast y key.
#'
#' Every other cast pair -- equi keys whose original column must survive
#' (`keep = TRUE`, and semi/anti, whose output is x verbatim) and non-equi
#' conditions (whose columns always keep their own types): match on cast
#' COPIES of just the side(s) that need a cast, appended as trailing temporary
#' columns. Equi pairs repoint the key indices at the copies; a non-equi pair's
#' condition is rewritten to name the copies (`cond_by`, with the schemas
#' extended by them) so the evaluator never sees mismatched operand types.
#' `drop_join_key_temps()` removes the temporaries from the result. (They are
#' dropped after the join rather than via `right_drop` so the C++ side's
#' right/full key coalescing, which is gated on `right_drop`, never touches the
#' surviving originals.)
#'
#' @return `list(left_ptr, right_ptr, left_pair_idx, right_pair_idx, temps,
#'   cond_by, left_schema, right_schema)`; `temps` is `NULL` when every cast
#'   was applied in place.
#' @keywords internal
apply_join_key_casts <- function(ast, left_ptr, right_ptr, left_schema, right_schema,
                                 left_pair_idx, right_pair_idx) {
  in_place <- !isTRUE(ast$keep) && !(ast$join_type %in% c("semi", "anti"))
  n_left <- length(left_schema$names)
  n_right <- length(right_schema$names)

  left_temps <- list()   # each: list(src = 0-based source column, cast = key_casts entry)
  right_temps <- list()
  cond_by <- ast$by
  for (cast in ast$by$key_casts) {
    i <- cast$idx
    if (in_place && ast$by$op[i] == "==") {
      if (cast$cast_left) {
        left_ptr <- cast_join_key_column(left_ptr, left_pair_idx[i], cast$left_type,
                                         cast$type, cast$left_levels)
      }
      if (cast$cast_right) {
        right_ptr <- cast_join_key_column(right_ptr, right_pair_idx[i], cast$right_type,
                                          cast$type, cast$right_levels)
      }
      next
    }

    if (cast$cast_left) {
      left_temps[[length(left_temps) + 1L]] <- list(src = left_pair_idx[i], cast = cast)
      left_pair_idx[i] <- n_left + length(left_temps) - 1L
      cond_by$left[i] <- paste0(".cuplyr_key_x", length(left_temps))
    }
    if (cast$cast_right) {
      right_temps[[length(right_temps) + 1L]] <- list(src = right_pair_idx[i], cast = cast)
      right_pair_idx[i] <- n_right + length(right_temps) - 1L
      cond_by$right[i] <- paste0(".cuplyr_key_y", length(right_temps))
    }
  }

  if (length(left_temps) == 0L && length(right_temps) == 0L) {
    return(list(left_ptr = left_ptr, right_ptr = right_ptr,
                left_pair_idx = left_pair_idx, right_pair_idx = right_pair_idx,
                temps = NULL, cond_by = cond_by,
                left_schema = left_schema, right_schema = right_schema))
  }

  # Appends the copies, casts them to the common type, and extends the schema
  # (names/types only: all the condition binder reads).
  append_temps <- function(ptr, schema, temps, n, side, levels_of) {
    if (length(temps) == 0L) return(list(ptr = ptr, schema = schema))
    srcs <- vapply(temps, function(t) as.integer(t$src), integer(1))
    ptr <- gpu_select(ptr, c(seq_len(n) - 1L, srcs))
    for (k in seq_along(temps)) {
      cast <- temps[[k]]$cast
      ptr <- cast_join_key_column(ptr, n + k - 1L, cast[[paste0(side, "_type")]],
                                  cast$type, cast[[paste0(side, "_levels")]])
    }
    schema$names <- c(schema$names, paste0(".cuplyr_key_", if (side == "left") "x" else "y",
                                           seq_along(temps)))
    schema$types <- c(schema$types, vapply(temps, function(t) t$cast$type, character(1)))
    list(ptr = ptr, schema = schema)
  }
  left <- append_temps(left_ptr, left_schema, left_temps, n_left, "left")
  right <- append_temps(right_ptr, right_schema, right_temps, n_right, "right")

  list(left_ptr = left$ptr, right_ptr = right$ptr,
       left_pair_idx = left_pair_idx, right_pair_idx = right_pair_idx,
       temps = list(n_left = n_left, n_right = n_right,
                    n_left_temp = length(left_temps),
                    n_right_temp = length(right_temps),
                    semi_anti = ast$join_type %in% c("semi", "anti")),
       cond_by = cond_by, left_schema = left$schema, right_schema = right$schema)
}

#' Drop the temporary key copies `apply_join_key_casts()` appended
#'
#' A mutating join's columns are `[left cols, left temps, right cols kept,
#' right temps]`; semi/anti return `[left cols, left temps]`.
#' @keywords internal
drop_join_key_temps <- function(ptr, temps) {
  n_cols <- gpu_dim(ptr)[2]
  left_temp <- temps$n_left + seq_len(temps$n_left_temp) - 1L
  right_temp <- if (temps$semi_anti) integer(0) else {
    n_cols - seq_len(temps$n_right_temp)
  }
  gpu_select(ptr, setdiff(seq_len(n_cols) - 1L, c(left_temp, right_temp)))
}

#' Lower a `by = character()` join (gap A4)
#'
#' dplyr runs its ordinary join machinery with a constant key
#' (`join_rows(cross = TRUE)`), so every x row matches every y row. That is a
#' cartesian product when both sides have rows (`gpu_cross_join()`, x-major),
#' but an empty side behaves like an outer join: left/full keep x with NA y
#' columns, right/full keep y with NA x columns, inner gives 0 rows; semi
#' returns x iff y has rows, anti the opposite. Row counts are only known
#' here, at execution time, so the empty-side cases are decided in R.
#' Deliberately NOT a hash join on a constant key column (a hash build over
#' n identical keys is quadratic).
#'
#' `multiple = "first"/"any"/"last"` keep y's first/last row for each x row;
#' for right/full the other y rows are then unmatched and appended with NA x
#' columns, like dplyr. `relationship`/`unmatched` run through
#' `check_join_cardinality()` on a diag synthesized from the two row counts;
#' the default relationship is `"many-to-many"` (no warning).
#' @keywords internal
lower_cross_by_join <- function(ast, left_ptr, right_ptr, left_schema, right_schema) {
  nx <- gpu_dim(left_ptr)[1]
  ny <- gpu_dim(right_ptr)[1]
  type <- ast$join_type

  if (type %in% c("semi", "anti")) {
    out <- gpu_select(left_ptr, seq_along(left_schema$names) - 1L)
    keep_x <- (ny > 0) == identical(type, "semi")
    return(if (keep_x) out else gpu_slice_head(out, 0, FALSE))
  }

  diag <- list(
    left_multi_first = if (nx > 0 && ny > 1) 0L else -1L,
    right_multi_first = if (ny > 0 && nx > 1) 0L else -1L,
    left_unmatched_first = if (nx > 0 && ny == 0) 0L else -1L,
    right_unmatched_first = if (ny > 0 && nx == 0) 0L else -1L
  )
  check_join_cardinality(diag, type, ast$multiple,
                         ast$relationship %||% "many-to-many", ast$unmatched)

  null_cols <- function(schema, n) {
    lapply(seq_along(schema$names), function(i) {
      gpu_make_null_column(n, gpu_physical_type(unname(schema$types[i])))
    })
  }

  if (nx == 0 || ny == 0) {
    if (identical(type, "left") || identical(type, "full")) {
      if (ny == 0 && nx > 0) {
        return(gpu_bind_cols_impl(c(list(left_ptr), null_cols(right_schema, nx))))
      }
    }
    if (identical(type, "right") || identical(type, "full")) {
      if (nx == 0 && ny > 0) {
        return(gpu_bind_cols_impl(c(null_cols(left_schema, ny), list(right_ptr))))
      }
    }
    return(gpu_cross_join(left_ptr, right_ptr))
  }

  pick <- ast$multiple
  if (ny > 1 && pick %in% c("first", "any", "last")) {
    first <- !identical(pick, "last")
    matched_y <- if (first) gpu_slice_head(right_ptr, 1, FALSE) else gpu_slice_tail(right_ptr, 1, FALSE)
    out <- gpu_cross_join(left_ptr, matched_y)
    if (type %in% c("right", "full")) {
      rest_y <- if (first) gpu_slice_tail(right_ptr, ny - 1, FALSE) else gpu_slice_head(right_ptr, ny - 1, FALSE)
      rest <- gpu_bind_cols_impl(c(null_cols(left_schema, ny - 1L), list(rest_y)))
      out <- gpu_bind_rows_aligned(list(out, rest))
    }
    return(out)
  }

  gpu_cross_join(left_ptr, right_ptr)
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
#' As of Phase 5 task W4, every window kind is lowerable: the "shift"
#' family (`lag()`/`lead()`), the "scan" family (`cumsum()`/`cummax()`/
#' `cummin()`/`cumprod()`, plus bare `row_number()`'s no-order COUNT_ALL
#' form), the "rank" family (`row_number(x)`/`min_rank()`/`dense_rank()`/
#' `percent_rank()`/`cume_dist()`), and the "agg" family
#' (`mean()`/`sum()`/`min()`/`max()`/`n()`/`sd()`/`var()`/`first()`/
#' `last()`/`nth()`) -- see `validate_window_spec()`. A later task (the
#' rolling-window benchmark, scratchpad/todo.md) adds a sixth, "rolling"
#' (`roll_mean()`/`roll_sum()`/`roll_min()`/`roll_max()`/`roll_sd()`, but
#' NOT `roll_median()` -- see `window_spec_lowerable()`'s own docs). A spec
#' whose `fn` isn't recognized AT ALL (no registry entry, or a registry
#' entry with a `NULL` `window` field) hits a hard internal error here
#' (never a user-facing message: no verb decomposes an expression into an
#' unrecognized spec, so reaching this function with one is a cuplyr bug,
#' not a user mistake).
#' @keywords internal
lower_window <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)

  specs <- ast$specs
  if (length(specs) == 0) {
    return(input_ptr)
  }

  for (spec in specs) {
    validate_window_spec(spec)
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

#' Is a window spec lowerable by `src/ops_window.cpp`'s `gpu_window()`?
#'
#' The non-throwing predicate half of `validate_window_spec()` (below),
#' factored out so `mutate()`'s window wiring (`R/mutate.R`,
#' `mutate_window()`) can check every spec a decomposition plan produces
#' *before* pushing any AST node -- a window call whose spec isn't lowerable
#' (an unrecognized `fn`, i.e. a genuine cuplyr bug rather than a
#' user-facing shape) must fall back to `gpu_fallback()` cleanly, never
#' surface `validate_window_spec()`'s internal-error text to a user, and
#' never partially push GPU ops before discovering the need to fall back
#' (see `scratchpad/phase5_window_design.md` section 1.3's "Never
#' half-lower").
#'
#' As of Phase 5 task W4, every window `fn` with a registry `window` entry
#' is lowerable (`"shift"`, `"rank"`, `"scan"`, and `"agg"` kinds all have
#' C++ support in `gpu_window()`) -- this predicate now only guards against
#' an `fn` with NO registry entry at all, or a registry entry whose
#' `window` field is `NULL` (not a window-class op), which should be
#' unreachable in practice (nothing in `R/window.R`'s decomposition
#' machinery can produce such a spec) but is checked defensively so a
#' bug surfaces as a clear internal error instead of silently mis-executing.
#'
#' The rolling-window task (scratchpad/todo.md) adds a sixth kind,
#' `"rolling"`, lowerable for every `fn` EXCEPT `"roll_median"`: cudf
#' 25.12's `rolling_window()`/`grouped_rolling_window()` have no MEDIAN
#' aggregation instantiation at all in this environment (verified via `nm
#' -DC libcudf.so`, see the file-level comment above
#' `ir_parse_roll_call()`, R/ir.R) -- a genuine cudf capability gap, not a
#' cuplyr bug, so `roll_median()`/`data.table::frollmedian()` mutate()
#' calls route through `mutate_window_fallback()` exactly like the
#' not-yet-implemented-kind case below, needing `R/roll.R`'s real
#' `roll_median()` R function to make that CPU re-run work.
#'
#' @param spec A `window_spec()` structure (`R/window.R`)
#' @return `TRUE`/`FALSE`
#' @keywords internal
window_spec_lowerable <- function(spec) {
  entry <- ir_call_registry[[spec$fn]]
  kind <- if (!is.null(entry)) entry$window$kind else NULL
  if (identical(kind, "rolling")) {
    return(!identical(spec$fn, "roll_median"))
  }
  identical(kind, "shift") || identical(kind, "rank") ||
    identical(kind, "scan") || identical(kind, "agg")
}

#' Reject any window spec `gpu_window()` doesn't implement
#'
#' See `window_spec_lowerable()`'s docs for the shared predicate this
#' wraps with an internal-error `stop()` call.
#' @keywords internal
validate_window_spec <- function(spec) {
  if (window_spec_lowerable(spec)) {
    return(invisible(TRUE))
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
