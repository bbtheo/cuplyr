# AST Node Infrastructure for Lazy Evaluation
#
# This module defines the Abstract Syntax Tree (AST) node types used for
# lazy evaluation of GPU operations. Each dplyr verb creates an AST node
# that is only executed when collect() or compute() is called.

# -----------------------------------------------------------------------------
# Base Node Constructor
# -----------------------------------------------------------------------------

#' Create an AST node
#'
#' @param type Character string identifying the node type
#' @param ... Additional named arguments stored on the node
#' @param input The input AST node (NULL for source nodes)
#' @return An ast_node object
#' @keywords internal
ast_node <- function(type, ..., input = NULL) {
  node <- list(
    type = type,
    input = input,
    ...
  )
  # (the class vectors of the node types the verbs build on every call are
  # kept ready; any other type builds its own)
  class(node) <- ast_node_classes[[type]] %||% c(paste0("ast_", type), "ast_node")
  node
}

# `c(paste0("ast_", type), "ast_node")` for the node types built per verb call.
# @keywords internal
ast_node_classes <- lapply(
  stats::setNames(nm = c("source", "filter", "mutate", "select", "arrange", "summarise", "rename",
                         "window", "slice", "distinct", "join", "barrier", "reframe")),
  function(type) c(paste0("ast_", type), "ast_node")
)

# -----------------------------------------------------------------------------
# Node Type Constructors
# -----------------------------------------------------------------------------

#' Create a source AST node (leaf node)
#'
#' @param schema List with names and types vectors
#' @return An ast_source node
#' @keywords internal
ast_source <- function(schema, source_ptr = NULL) {
  ast_node("source", schema = schema, source_ptr = source_ptr, input = NULL)
}

#' Create a filter AST node
#'
#' @param input Input AST node
#' @param predicates List of predicate structures
#' @return An ast_filter node
#' @keywords internal
ast_filter <- function(input, predicates) {
  ast_node("filter", input = input, predicates = predicates)
}

#' Create a mutate AST node
#'
#' @param input Input AST node
#' @param expressions List of expression structures
#' @return An ast_mutate node
#' @keywords internal
ast_mutate <- function(input, expressions) {
  ast_node("mutate", input = input, expressions = expressions)
}

#' Create a select AST node
#'
#' @param input Input AST node
#' @param columns Character vector of column names to keep
#' @param positions `NULL` (select by `columns`), or the integer positions of
#'   the kept columns in `input`'s schema: used instead of the names when the
#'   input has duplicated column names, where a name is ambiguous. Such a node
#'   is an optimizer barrier and never pruned through.
#' @param pruned `TRUE` for a select the optimizer inserted to drop columns
#'   nothing reads (`push_down_projections()`); `push_down_filters()` leaves a
#'   filter above it, as moving the filter below would undo the pruning. The
#'   field is only present when `TRUE`.
#' @return An ast_select node
#' @keywords internal
ast_select <- function(input, columns, positions = NULL, pruned = FALSE) {
  node <- ast_node("select", input = input, columns = columns, positions = positions)
  if (isTRUE(pruned)) node$pruned <- TRUE
  node
}

#' Create an arrange AST node
#'
#' @param input Input AST node
#' @param sort_specs List of sort specifications (col_name, descending)
#' @param groups Character vector of group columns (for .by_group)
#' @return An ast_arrange node
#' @keywords internal
ast_arrange <- function(input, sort_specs, groups = character()) {
  ast_node("arrange", input = input, sort_specs = sort_specs, groups = groups)
}

#' Create a summarise AST node
#'
#' @param input Input AST node
#' @param aggregations List of aggregation structures
#' @param groups Character vector of group columns this node aggregates BY
#'   (drives the actual groupby op -- always the full pre-summarise group
#'   set, regardless of `.groups=`)
#' @param result_groups Character vector of group columns RETAINED on the
#'   summarise's *output* (Phase 6, task 3): a subset of `groups`, computed
#'   from `.groups=` semantics by `resolve_summarise_groups()`
#'   (R/summarise.R) -- `character()` for `.groups = "drop"`/ungrouped
#'   input, all of `groups` for `"keep"`, all-but-the-last for the default/
#'   `"drop_last"`. Read by `propagate_groups()` (R/execute.R) instead of
#'   the old hardcoded "summarise always drops all groups" rule.
#' @param on_sum_overflow (F34) What to do when an integer/logical `sum()`
#'   leaves R's integer range at execution. `"double"`: the column comes back
#'   FLOAT64, as in dplyr (eager mode reconciles the schema from the result).
#'   `"error"` (lazy plans, whose schema was declared INT32 statically): raise
#'   `cuplyr_error_sum_overflow`. A lazy `summarise()` plans only a sum of a
#'   logical this way (F80): an INT32 sum runs at the verb with `"double"`.
#' @param sort_groups (F33) Sort the grouped result by the group keys
#'   (ascending, `NA` last), dplyr's `group_by()` order. `FALSE` only for
#'   `summarise_by_desugar()`, which re-sorts by first appearance anyway.
#' @return An ast_summarise node
#' @keywords internal
ast_summarise <- function(input, aggregations, groups, result_groups = character(),
                          on_sum_overflow = "error", sort_groups = TRUE) {
  ast_node("summarise", input = input, aggregations = aggregations, groups = groups,
            result_groups = result_groups, on_sum_overflow = on_sum_overflow,
            sort_groups = sort_groups)
}

#' Create a reframe AST node
#'
#' The LONG step of a native `reframe()` (F22, `R/reframe.R`): reshapes the
#' wide one-row-per-group table `input` into `k` rows per group.
#'
#' @param input Input AST node (the wide table)
#' @param outputs The output columns in order, each `list(name, kind, ...)`:
#'   `kind = "rep"` (`src` = one wide column, every row repeated `k` times; the
#'   group keys and size-1 results), `"interleave"` (`src` = `k` wide columns
#'   interleaved row by row; a size-`k` result) or `"literal"` (`value`, an R
#'   vector of length 1 or `k`, tiled)
#' @param k Rows per group
#' @param first_k_name Name of the first output of size `k` (NA when `k == 1`):
#'   the column in dplyr's "Can't recycle" error for an empty table
#' @return An ast_reframe node
#' @keywords internal
ast_reframe <- function(input, outputs, k, first_k_name = NA_character_) {
  ast_node("reframe", input = input, outputs = outputs, k = k, first_k_name = first_k_name)
}

#' Create a barrier AST node (optimization fence)
#'
#' @param input Input AST node
#' @return An ast_barrier node
#' @keywords internal
ast_barrier <- function(input) {
  ast_node("barrier", input = input)
}

#' Create a distinct AST node
#'
#' @param input Input AST node (any dot expressions that compute new key
#'   columns, e.g. `distinct(df, z = x + 1)`, are already lowered into an
#'   `ast_mutate` wrapping this node by `distinct.tbl_gpu()` -- this node
#'   only ever does the dedup step)
#' @param key_cols Character vector of column names to dedup on, in the
#'   exact order the `.keep_all = FALSE` output should have (group columns,
#'   if any, already prepended by the caller)
#' @param keep_all Logical, `TRUE` to keep every column of `input` (dplyr's
#'   `.keep_all = TRUE`), `FALSE` to project down to just `key_cols`
#' @param key_positions `NULL` (key on `key_cols` by name), or the integer
#'   positions of the key columns in `input`'s schema: used instead of the
#'   names when the input has duplicated column names. Such a node is never
#'   pruned through by name.
#' @return An ast_distinct node
#' @keywords internal
ast_distinct <- function(input, key_cols, keep_all = FALSE, key_positions = NULL) {
  ast_node("distinct", input = input, key_cols = key_cols, keep_all = keep_all,
           key_positions = key_positions)
}

#' Create a slice AST node
#'
#' One node type covers the whole slice family (`slice()`/`slice_head()`/
#' `slice_tail()`/`slice_min()`/`slice_max()`), distinguished by `mode` --
#' unlike most other verbs (which each get their own `ast_*` constructor),
#' these five share the same "select a subset of rows, in some order,
#' from a single input" shape and (mostly) the same `n=`/`prop=` sizing
#' rule, so one node with a mode-specific field set avoids several
#' near-identical constructors/lowering functions/optimizer barrier
#' entries. `slice_sample()` and every *grouped* variant are out of scope
#' for this node (Phase 3 scope decision, see `R/slice.R`) and keep using
#' the CPU fallback (`R/fallback.R`) instead.
#'
#' @param input Input AST node
#' @param mode One of `"head"`, `"tail"`, `"index"`, `"rank"`, `"bracket"` (tibble
#'   `x[i, ]` row subscript in `raw_indices`, NA / out-of-range give NA rows)
#' @param amount For `mode` `"head"`/`"tail"`/`"rank"`: the raw (unresolved)
#'   `n`/`prop` value (sign preserved; resolved against the actual row
#'   count only at lowering time, see `src/ops_common.hpp::compute_slice_size()`)
#' @param is_prop For `mode` `"head"`/`"tail"`/`"rank"`: `TRUE` if `amount`
#'   is a `prop`, `FALSE` if it's an `n`
#' @param raw_indices For `mode` `"index"`: the unresolved numeric index
#'   vector from `slice()`'s dots (may contain 0/NA/duplicates/out-of-range/
#'   negative values -- fully resolved only at lowering time, since negative
#'   indices and out-of-range dropping both need the actual row count)
#' @param bracket_arg,bracket_call,rows_resolved For `mode` `"bracket"`:
#'   `raw_indices` is either already resolved to 1-based positions against the
#'   real row count (`rows_resolved = TRUE`, `NA` = all-`NA` row), or still
#'   the user's raw subscript, in which case `bracket_arg` (the user's
#'   subscript expression) and `bracket_call` (the `[` call) label any error
#'   raised when it is resolved at lowering
#' @param order_col For `mode` `"rank"`: the column name (in `input`'s
#'   output schema) to rank by -- always a real column name, even when
#'   `order_by` was a computed expression (see `order_is_temp`)
#' @param order_is_temp For `mode` `"rank"`: `TRUE` if `order_col` is a
#'   synthetic column added by a wrapping `ast_mutate` (computed
#'   `order_by`, e.g. `slice_min(df, x + y)`) that must NOT appear in the
#'   final output -- mirrors `distinct()`'s same computed-key pattern
#'   (`R/distinct.R`)
#' @param descending For `mode` `"rank"`: `FALSE` for `slice_min()`, `TRUE`
#'   for `slice_max()`
#' @param with_ties,na_rm For `mode` `"rank"`: as documented on
#'   `slice_min()`/`slice_max()`
#' @param top_k For `mode` `"rank"`: `TRUE` on a node the optimizer made from
#'   `arrange(x) |> head(n)` (`rewrite_slice_over_arrange()`); the node is then
#'   `slice_min(x, n, with_ties = FALSE)` and nothing else
#' @return An ast_slice node
#' @keywords internal
ast_slice <- function(input, mode, amount = NULL, is_prop = NULL,
                      raw_indices = NULL, order_col = NULL,
                      order_is_temp = FALSE, descending = NULL,
                      with_ties = NULL, na_rm = NULL,
                      bracket_arg = NULL, bracket_call = NULL,
                      rows_resolved = FALSE, top_k = FALSE) {
  ast_node("slice", input = input, mode = mode, amount = amount, is_prop = is_prop,
           raw_indices = raw_indices, order_col = order_col,
           order_is_temp = order_is_temp, descending = descending,
           with_ties = with_ties, na_rm = na_rm,
           bracket_arg = bracket_arg, bracket_call = bracket_call,
           rows_resolved = rows_resolved, top_k = top_k)
}

#' Create a window AST node
#'
#' One node per window-decomposition "stage" (`scratchpad/phase5_window_design.md`
#' section 2): evaluates every `window_spec()` in `specs` (`R/window.R`)
#' against `input`'s current row set/order and APPENDS one new column per
#' spec (never replaces an existing column -- temp names guarantee this).
#' Built by a verb's own window-wiring (`R/mutate.R`/`R/filter.R`/`R/slice.R`,
#' from W3/W5/W6/W7 onward) or, in W2, by hand for `test-window-native.R`.
#'
#' @param input Input AST node
#' @param specs List of `window_spec()` structures (`R/window.R`)
#' @param group_cols Character vector of group-by column names --
#'   `character()` for an ungrouped window (section 3: lowering substitutes
#'   a constant key column and skips the sort when there's also no
#'   `order_cols` on every spec)
#' @param on_sum_overflow (F42) What to do when an integer/logical window
#'   `sum()` (declared INT32) leaves R's integer range at execution. `"error"`:
#'   raise `cuplyr_error_sum_overflow` (mutate() and eager filter(): eager
#'   reruns the call on the CPU; a lazy `mutate()` only plans a sum of a
#'   logical, an INT32 sum runs at the verb, F80). `"double"`: the column
#'   comes back FLOAT64, as in dplyr, for a consumer that reads the real column
#'   type (lazy `filter()`'s predicate). A spec's own `on_sum_overflow` field,
#'   when set, overrides this for that spec (`filter()` sets `"error"` on a sum
#'   a later stage consumes).
#' @return An ast_window node
#' @keywords internal
ast_window <- function(input, specs, group_cols = character(),
                       on_sum_overflow = "error") {
  ast_node("window", input = input, specs = specs, group_cols = group_cols,
           on_sum_overflow = on_sum_overflow)
}

#' Create a rename AST node
#'
#' Schema-only: `old_names[i]` in the *input* schema becomes `new_names[i]`
#' in the output schema; every other column (position, type, and GPU
#' column data) is untouched. This is the one AST node type where lowering
#' does no GPU work at all (see `lower_rename()`, `R/lower.R`) -- column
#' names live entirely in the R-side schema (`$schema$names`), never on the
#' underlying `cudf::table`/`GpuTablePtr` itself (confirmed by
#' `gpu_collect()`, `src/transfer_io.cpp`, which takes the R-side names as
#' a parameter and applies them positionally to the physical columns), so a
#' pure rename never needs to touch the GPU pointer.
#'
#' Used directly by `rename()`/`rename_with()` (`R/rename.R`), and stacked
#' on top of an `ast_select()` node by `select()` (the `new = old` rename
#' fix, `R/select.R`) and `relocate()` (`R/relocate.R`) whenever their
#' tidyselect resolution renames a column.
#'
#' Treated as an optimizer barrier (see `is_barrier()` below): the simplest
#' safe choice, since letting a filter/projection above a rename push below
#' it would need to translate column references through `old_names`/
#' `new_names` -- correct but not worth the complexity for a schema-only
#' verb. Optimization opportunity for later: translate `required_cols`/
#' `pred$cols` through the name map instead of blocking pushdown entirely.
#'
#' @param input Input AST node
#' @param old_names Character vector of column names (in `input`'s output
#'   schema) being renamed
#' @param new_names Character vector of the same length: `new_names[i]` is
#'   what `old_names[i]` becomes
#' @param positions `NULL` (rename by `old_names`), or the integer positions
#'   of the renamed columns: used instead of the names when the input has
#'   duplicated column names, where a name is ambiguous. Such a node is
#'   never pruned through by the optimizer.
#' @return An ast_rename node
#' @keywords internal
ast_rename <- function(input, old_names, new_names, positions = NULL) {
  ast_node("rename", input = input, old_names = old_names, new_names = new_names,
           positions = positions)
}

#' Create a join AST node
#'
#' @param type Join type: "inner", "left", "right", "full", "semi", "anti",
#'   "cross"
#' @param left Left input AST node
#' @param right Right input AST node
#' @param by Join specification `list(left, right, op, filter)`, as returned
#'   by [parse_join_by()]
#' @param keep Logical, keep both key columns when names match
#' @param suffix Character vector of length 2
#' @param na_matches Character, "na" or "never"
#' @param multiple Character, one of "all"/"any"/"first"/"last" (dormant --
#'   only "all" is honored until Phase 7 J5)
#' @param unmatched Character, one or two of "drop"/"error" (dormant --
#'   only "drop" is honored until Phase 7 J5)
#' @param relationship `NULL` or one of the four cardinality strings
#'   (dormant -- not enforced until Phase 7 J5)
#' @param cross_by `TRUE` for dplyr's deprecated `by = character()` cross
#'   spelling (`by` is then the empty spec; see `lower_cross_by_join()`)
#' @return An ast_join node
#' @keywords internal
ast_join <- function(type, left, right, by, keep = FALSE,
                     suffix = c(".x", ".y"), na_matches = "na",
                     multiple = "all", unmatched = "drop", relationship = NULL,
                     cross_by = FALSE) {
  ast_node("join", left = left, right = right, join_type = type, by = by,
           keep = keep, suffix = suffix, na_matches = na_matches,
           multiple = multiple, unmatched = unmatched, relationship = relationship,
           cross_by = cross_by)
}

# -----------------------------------------------------------------------------
# Predicate and Expression Structures
# -----------------------------------------------------------------------------

#' Create a filter predicate structure
#'
#' Section 3 of `scratchpad/phase1_expression_engine.md`: the IR cutover's
#' predicate record. `ir` is the parsed expression IR (see `R/ir.R`); `cols`
#' is cached from `ir_cols()` since every optimizer filter pass needs it and
#' none of them need to walk the IR tree themselves.
#'
#' @param ir An IR node (the parsed predicate expression)
#' @param schema List with `names`/`types` the IR is bound against; read only
#'   by [ir_has_checked_int()] for `checked_int`
#' @return A predicate list structure: `list(ir, cols, estimated_cost,
#'   is_deterministic, na_sensitive, checked_int)`; `checked_int` (F70) is
#'   the cell of new_checked_int_cell(): read whether the predicate holds an
#'   F42 checked integer node with record_checked_int()
#' @keywords internal
make_predicate <- function(ir, schema) {
  list(
    ir = ir,
    cols = ir_cols(ir),
    estimated_cost = ir_cost(ir),
    is_deterministic = TRUE,
    na_sensitive = TRUE,
    checked_int = new_checked_int_cell(ir, schema)
  )
}

# The F70 checked-integer flag of a predicate or mutate expression, on demand
#
# Only the optimizer reads the flag (a filter may not move below, and a
# checked expression may not be pruned as dead: `mutate_has_checked_int()`,
# `pred_checked_int()`, `prune_dead_columns()`), and an eager node is lowered
# and dropped without ever reaching it. So a record holds a cell with the
# inputs of [ir_has_checked_int()] and the answer is worked out the first time
# it is asked for (F110). Always read the flag through record_checked_int():
# the cell itself is not a logical.
#
# @param ir The record's IR
# @param schema The schema the IR is bound against
# @return An environment holding `ir`, `schema` and, once asked, `flag`
# @keywords internal
new_checked_int_cell <- function(ir, schema) {
  cell <- new.env(hash = FALSE, parent = emptyenv())
  cell$ir <- ir
  cell$schema <- schema
  cell
}

# @param record A predicate or mutate expression from [make_predicate()] or
#   [make_mutate_expr()]
# @return `record_checked_int()`: `TRUE` iff the record holds an F42 checked
#   integer node ([ir_has_checked_int()])
# @keywords internal
record_checked_int <- function(record) {
  cell <- .subset2(record, "checked_int")
  if (!is.environment(cell)) {
    return(isTRUE(cell))
  }
  flag <- cell[["flag"]]
  if (is.null(flag)) {
    flag <- ir_has_checked_int(cell[["ir"]], cell[["schema"]])
    cell[["flag"]] <- flag
  }
  flag
}

#' Create a mutate expression structure
#'
#' Section 3 of `scratchpad/phase1_expression_engine.md`: the IR cutover's
#' mutate expression record. `ir` is the parsed expression IR (see
#' `R/ir.R`); `input_cols`/`output_type` are cached from `ir_cols()`/
#' `ir_infer_type()` since every optimizer pass
#' (`fuse_mutates`/`try_fuse_mutate_pair`/
#' `prune_dead_columns`/`push_down_projections`) reads only those two fields
#' plus `output_col` -- none of them walk the IR tree themselves. `op`/
#' `scalar`/`input_types` are gone: only `lower_mutate()` (via
#' `gpu_mutate_expr()`) ever needed them, and it now reads `ir` directly.
#'
#' @param output_col Output column name
#' @param ir An IR node (the parsed mutate expression)
#' @param schema List with `names`/`types`, used to resolve `output_type`
#'   via [ir_infer_type()] (which delegates to [infer_mutate_output_type()]
#'   unchanged for arithmetic ops, per section 1.3)
#' @return An expression list structure: `list(output_col, ir, input_cols,
#'   output_type, checked_int)`; `checked_int` (F70) is the cell of
#'   new_checked_int_cell(): read whether the expression holds an F42 checked
#'   integer node with record_checked_int()
#' @keywords internal
make_mutate_expr <- function(output_col, ir, schema) {
  list(
    output_col = output_col,
    ir = ir,
    input_cols = ir_cols(ir),
    output_type = ir_infer_type_memo(ir, schema),
    checked_int = new_checked_int_cell(ir, schema)
  )
}

#' Infer output type for a mutate expression
#'
#' Implements D5's ordered promotion rules (see
#' `scratchpad/unification_design.md`). Note this deliberately mirrors R's own
#' arithmetic promotion, not SQL/cudf defaults: `2L^2L` is a double in R, and
#' `TRUE+TRUE` is an integer, and dplyr agrees with both.
#'
#' @param op The operation
#' @param input_types Types of input columns
#' @param scalar Scalar value if any
#' @return GPU type string
#' @keywords internal
infer_mutate_output_type <- function(op, input_types, scalar) {
  # Rule 6 (checked first as a guard): unknown/no input type information.
  # In practice every production call site in R/mutate.R supplies
  # input_types; this only fires for defensive/test-only callers that omit
  # it, so checking it up front (rather than only as a final fallback) can't
  # diverge from any real call.
  if (is.null(input_types)) {
    return("FLOAT64")
  }

  # Rule 1: copy preserves the source column's type exactly.
  if (op == "copy") {
    return(input_types[1])
  }

  # Rule 2: no arithmetic on STRING/DICTIONARY32/TIMESTAMP_* columns.
  # (startsWith() with NA taken as FALSE is grepl("^TIMESTAMP_", .), faster)
  is_ts <- startsWith(input_types, "TIMESTAMP_") & !is.na(input_types)
  is_unsupported <- input_types == "STRING" |
    input_types == "DICTIONARY32" |
    is_ts
  if (any(is_unsupported)) {
    bad_type <- input_types[is_unsupported][1]
    msg <- paste0("mutate() does not support arithmetic on ", bad_type, " columns")
    # A factor operand: dplyr warns and returns NA, so the parse falls back
    # (stop_factor_needs_cpu()) instead of erroring. So does a timestamp
    # `+`/`-` a number (F32): `t + 3600` is valid R (a POSIXct in t's zone)
    # that the GPU expression engine has no duration arithmetic for; dplyr
    # computes it on the CPU. So does `t - t2` (F37: a difftime; the CPU
    # route then raises F18's upload error unless the result drops it). The
    # other operators keep the plain error here, though an IR expression
    # with `*`, `%%`, ... on a timestamp never gets this far:
    # ir_check_ts_operands() (R/ir.R) sends it to the CPU first.
    if ("DICTIONARY32" %in% input_types) {
      stop_factor_needs_cpu(msg)
    }
    if (op %in% c("+", "-") && !any(is_unsupported & !is_ts) &&
        (sum(is_ts) == 1L || (op == "-" && sum(is_ts) == 2L))) {
      stop_timestamp_needs_cpu(msg)
    }
    stop(msg, call. = FALSE)
  }

  # Rule 3: division always promotes to FLOAT64 (dplyr: `1L/2L` is double).
  if (op == "/") {
    return("FLOAT64")
  }

  # Rule 4: power always promotes to FLOAT64 (R: `2L^2L` is double).
  if (op == "^") {
    return("FLOAT64")
  }

  # Rule 5: +, -, *, %%, %/% promote over input types plus the scalar's own
  # R type (double -> FLOAT64, integer -> INT32; NULL if no scalar/not
  # numeric-typed, which drops out of the union below).
  if (op %in% c("+", "-", "*", "%%", "%/%")) {
    scalar_type <- if (is.double(scalar)) {
      "FLOAT64"
    } else if (is.integer(scalar)) {
      "INT32"
    } else {
      NULL
    }
    all_types <- c(input_types, scalar_type)

    if (any(all_types == "FLOAT64" | all_types == "FLOAT32")) {
      return("FLOAT64")
    }
    if (any(all_types == "INT64")) {
      return("INT64")
    }
    # Anything else (INT32, BOOL8, INT16, INT8, ...) promotes to INT32.
    # This covers BOOL8, matching R's `TRUE + TRUE == 2L`.
    return("INT32")
  }

  # Rule 6: unrecognized operation -> FLOAT64 for safety.
  "FLOAT64"
}

#' Create an aggregation structure for summarise
#'
#' @param output_col Output column name
#' @param input_col Input column name
#' @param fn Aggregation function name (sum, mean, min, max, n, sd, var,
#'   median, quantile, n_distinct, first, last, nth, any, all)
#' @param input_type Input column type
#' @param na_rm Logical scalar (Phase 6, task 6.1; extended Phase 6, task
#'   6.2). `FALSE` (the default, matching R's own `na.rm = FALSE` default
#'   for `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()`/`median()`) means a
#'   group with ANY null input value must produce a NULL aggregation result
#'   for the WHOLE group -- cudf's own groupby MEAN/SUM/MIN/MAX/STD/
#'   VARIANCE/MEDIAN aggregations always exclude nulls, so
#'   `gpu_summarise()` (`src/ops_groupby.cpp`) applies an extra
#'   whole-group-null propagation step when this is `FALSE` and the input
#'   column actually has nulls (same technique as the window-aggregate
#'   path, `src/ops_window.cpp`, W4). `TRUE` means "use cudf's native
#'   null-excluding behavior as-is" -- no extra step. Meaningless for `n()`
#'   (counts rows regardless of nulls, matching dplyr already). For
#'   `quantile()`, `na_rm = FALSE` with an actual null present is a hard
#'   ERROR (matching base R's own `quantile.default()`), not a NA
#'   propagation -- see `gpu_summarise()`. For `n_distinct()`, `na_rm`
#'   selects cudf's `null_policy` directly (`INCLUDE`/`EXCLUDE`) rather
#'   than a post-hoc propagation step -- NA is its own distinct value when
#'   `na_rm = FALSE`, matching dplyr's `n_distinct()` exactly. For
#'   `first()`/`last()`/`nth()`, `na_rm` also selects `null_policy`
#'   directly: `FALSE` (default) never skips a NA when picking the
#'   position (may return NA), `TRUE` skips NA values when picking. For
#'   `any()`/`all()`, `na_rm` implements R's own three-valued logic
#'   (`any(c(TRUE, NA))` is `TRUE`, `any(c(FALSE, NA))` is `NA` when
#'   `na_rm = FALSE`, `FALSE` when `na_rm = TRUE`).
#' @param extra Numeric scalar, `NA_real_` unless `fn` needs an extra
#'   scalar parameter beyond the input column: `quantile()`'s probability
#'   (`[0, 1]`) or `nth()`'s (possibly negative) element position.
#' @param identity Logical scalar (F07). `TRUE` (the default) lets a
#'   `min()`/`max()` over an INT32/FLOAT64 column apply base R's empty-vector
#'   identity (`Inf`/`-Inf`, with dplyr's warning) for an empty group, which
#'   promotes an INT32 result column to FLOAT64 when it happens; see
#'   `gpu_summarise()` (`src/ops_groupby.cpp`). `FALSE` is for internal helper
#'   aggregations (`summarise_by_desugar()`'s row-position `min()`) that must
#'   stay INT32 and never warn. Only meaningful for `min`/`max` over
#'   INT32/FLOAT64; stored as `FALSE` for everything else.
#' @return An aggregation list structure
#' @keywords internal
make_aggregation <- function(output_col, input_col, fn, input_type = NULL, na_rm = FALSE,
                              extra = NA_real_, identity = TRUE) {
  # F34: min()/max() of a logical is an integer in dplyr; gpu_summarise()
  # casts the BOOL8 input to INT32, so the aggregation is an INT32 one.
  if (fn %in% c("min", "max") && identical(unname(input_type), "BOOL8")) {
    input_type <- "INT32"
  }
  output_type <- switch(fn,
    "n" = "INT32",
    # F34: dplyr's sum() of an integer/logical is an R integer. gpu_summarise()
    # reduces in INT64 and narrows to INT32; the type is FLOAT64 instead when
    # a sum leaves R's integer range (see ast_summarise(on_sum_overflow=)).
    "sum" = if (!is.null(input_type) && unname(input_type) %in% c("INT32", "BOOL8")) "INT32" else "FLOAT64",
    # F41: mean()/median() of a Date/POSIXct is that timestamp type
    # (gpu_summarise() converts back from the FLOAT64 aggregate).
    "mean" = ts_mean_type(input_type),
    "min" = input_type %||% "FLOAT64",
    "max" = input_type %||% "FLOAT64",
    "median" = ts_mean_type(input_type),
    "quantile" = "FLOAT64",
    "n_distinct" = "INT32",
    "first" = input_type %||% "FLOAT64",
    "last" = input_type %||% "FLOAT64",
    "nth" = input_type %||% "FLOAT64",
    "any" = "BOOL8",
    "all" = "BOOL8",
    "FLOAT64"
  )

  list(
    output_col = output_col,
    input_col = input_col,
    fn = fn,
    input_type = input_type,
    output_type = output_type,
    na_rm = isTRUE(na_rm),
    extra = as.double(extra),
    identity = isTRUE(identity) && fn %in% c("min", "max") &&
      !is.null(input_type) && unname(input_type) %in% c("INT32", "FLOAT64")
  )
}

# -----------------------------------------------------------------------------
# Schema Inference
# -----------------------------------------------------------------------------

#' Infer output schema from an AST node
#'
#' @param node An AST node
#' @return List with names and types vectors
#' @export
infer_schema <- function(node) {
  UseMethod("infer_schema")
}

#' `infer_schema()` through the schema memo
#'
#' What the methods, the optimizer and the lowering call to get a node's
#' schema. Outside a memo scope (see `schema_memo_open()`) it is plain
#' `infer_schema()`. Inside one (an outermost `compute()`, an eager chain's
#' lowering, `show_query()`) the result is remembered by node identity, so
#' inferring every node's input schema along a plan is linear in its depth
#' instead of quadratic.
#'
#' @param node An AST node (or `NULL`)
#' @return List with `names`, `types` (plus whatever `infer_schema()` gives)
#' @keywords internal
infer_schema_memo <- function(node) {
  memo <- .schema_memo$plain
  if (is.null(memo) || !is.list(node)) {
    # (a source node's schema is infer_schema.ast_source()'s answer)
    if (is.list(node) && identical(.subset2(node, "type"), "source")) {
      return(.subset2(node, "schema"))
    }
    return(infer_schema(node))
  }
  key <- obj_address(node)
  hit <- memo[[key]]
  if (!is.null(hit)) {
    return(.subset2(hit, "schema"))
  }
  # What the infer_schema() methods say, without their S3 dispatch.
  schema <- switch(.subset2(node, "type"),
    source = .subset2(node, "schema"),
    join = infer_schema(node),
    infer_schema_from(node, infer_schema_memo(.subset2(node, "input")))
  )
  # The entry holds `node`, so its address cannot be reused while the memo
  # lives, and a rewritten node (`node$input <- ...` copies a shared list) has
  # a new address: an entry is never stale.
  memo[[key]] <- list(node = node, schema = schema)
  schema
}

# Memo of `infer_schema()` by node identity, alive for one outermost
# `compute()` / eager chain lowering. Nodes are never annotated: the
# optimizer rewrites them with `node$input <- ...`.
.schema_memo <- new.env(parent = emptyenv())

# Start a memo scope unless an enclosing one is active (nested
# compute()/lowering), which is shared: entries are keyed by identity of nodes
# the memo holds, so sharing is always correct. Callers save the old state and
# register the restoring handler BEFORE opening, so an interrupt can't leave a
# scope open for the session (the pattern of wrap_gpu_call()):
#   memo_prev <- .schema_memo$plain
#   on.exit(.schema_memo$plain <- memo_prev, add = TRUE)
#   schema_memo_open()
# @return TRUE when this call opened the scope.
# @keywords internal
schema_memo_open <- function() {
  if (!is.null(.schema_memo$plain)) {
    return(FALSE)
  }
  .schema_memo$plain <- new.env(hash = TRUE, parent = emptyenv())
  TRUE
}

#' Output `names`/`types` of a node from its input's schema
#'
#' The body of every `infer_schema.ast_*()` method, with the input schema
#' passed in: `push_op()` calls it with the schema it already has, so recording
#' a verb never walks the pending plan. Joins have two inputs and go through
#' `infer_join_schema()`.
#'
#' @param node A single-input AST node
#' @param input_schema Schema of `node$input` (extra fields such as
#'   `factor_levels` are tolerated; pass-through nodes return it unchanged)
#' @return List with `names` and `types`
#' @keywords internal
infer_schema_from <- function(node, input_schema) {
  switch(.subset2(node, "type"),
    filter = , arrange = , barrier = input_schema,
    select = {
      idx <- node$positions %||% match(node$columns, input_schema$names)
      list(
        names = input_schema$names[idx],
        types = input_schema$types[idx]
      )
    },
    mutate = {
      result_names <- input_schema$names
      result_types <- input_schema$types

      for (expr in node$expressions) {
        existing_idx <- match(expr$output_col, result_names)
        if (!is.na(existing_idx)) {
          result_types[existing_idx] <- expr$output_type
        } else {
          result_names <- c(result_names, expr$output_col)
          result_types <- c(result_types, expr$output_type)
        }
      }

      list(names = result_names, types = result_types)
    },
    rename = {
      result_names <- input_schema$names
      idx <- node$positions %||% match(node$old_names, result_names)
      result_names[idx] <- node$new_names
      list(names = result_names, types = input_schema$types)
    },
    distinct = {
      if (isTRUE(node$keep_all)) {
        return(input_schema)
      }

      idx <- node$key_positions %||% match(node$key_cols, input_schema$names)
      list(names = input_schema$names[idx], types = input_schema$types[idx])
    },
    slice = {
      if (identical(node$mode, "rank") && isTRUE(node$order_is_temp)) {
        idx <- input_schema$names != node$order_col
        return(list(names = input_schema$names[idx], types = input_schema$types[idx]))
      }

      input_schema
    },
    summarise = {
      # Group columns come first
      if (length(node$groups) > 0) {
        group_idx <- match(node$groups, input_schema$names)
        group_names <- input_schema$names[group_idx]
        group_types <- input_schema$types[group_idx]
      } else {
        group_names <- character(0)
        group_types <- character(0)
      }

      # Then aggregation columns
      aggs <- .subset2(node, "aggregations")
      agg_names <- character(length(aggs))
      agg_types <- character(length(aggs))
      for (i in seq_along(aggs)) {
        agg_names[[i]] <- .subset2(.subset2(aggs, i), "output_col")
        agg_types[[i]] <- .subset2(.subset2(aggs, i), "output_type")
      }
      if (!is.null(agg_labels <- names(aggs))) {
        names(agg_names) <- agg_labels
        names(agg_types) <- agg_labels
      }

      # F07: an aggregate named like a grouping column replaces that key IN
      # PLACE (see summarise_output_index()).
      idx <- summarise_output_index(node$groups, agg_names)

      list(
        names = c(group_names, agg_names)[idx],
        types = c(group_types, agg_types)[idx]
      )
    },
    reframe = {
      types <- vapply(node$outputs, function(o) {
        switch(o$kind,
          literal = gpu_type_from_r(o$value),
          {
            src <- unname(input_schema$types[match(o$src, input_schema$names)])
            if (length(unique(src)) == 1L) src[[1]] else "FLOAT64"  # INT32 vs FLOAT64 drift
          }
        )
      }, character(1))
      list(names = vapply(node$outputs, `[[`, character(1), "name"), types = types)
    },
    window = {
      out_names <- vapply(node$specs, `[[`, character(1), "output_col")
      out_types <- vapply(node$specs, `[[`, character(1), "output_type")

      list(
        names = c(input_schema$names, out_names),
        types = c(input_schema$types, out_types)
      )
    },
    source = node$schema,
    infer_schema(node)  # join: two inputs
  )
}

#' @export
infer_schema.ast_source <- function(node) {
  node$schema
}

#' @export
infer_schema.ast_filter <- function(node) {
  infer_schema_memo(node$input)
}

#' @export
infer_schema.ast_select <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.ast_mutate <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.ast_arrange <- function(node) {
  infer_schema_memo(node$input)
}

#' @export
infer_schema.ast_barrier <- function(node) {
  infer_schema_memo(node$input)
}

#' @export
infer_schema.ast_rename <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.ast_distinct <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.ast_slice <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.ast_summarise <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' Output column order of a summarise node
#'
#' `gpu_summarise()` returns `[group keys, aggregates]`. dplyr instead lets an
#' aggregate named like a grouping column replace that key in place (same
#' position, the aggregate's type); every other aggregate follows the keys in
#' call order. This returns the 1-based permutation of `c(keys, aggregates)`
#' that yields dplyr's layout, shared by `infer_schema.ast_summarise()` and
#' `lower_summarise()` so the schema and the table cannot disagree. It is the
#' identity when no aggregate collides with a key. Duplicate aggregate names
#' never reach a node (`summarise()` routes them to the CPU fallback).
#'
#' @param groups Character vector of group columns the node aggregates by
#' @param agg_names Character vector of aggregate output names
#' @return Integer vector indexing `c(keys, aggregates)`
#' @keywords internal
summarise_output_index <- function(groups, agg_names) {
  n_groups <- length(groups)
  over <- match(groups, agg_names)
  if (all(is.na(over))) {
    return(seq_len(n_groups + length(agg_names)))  # nothing replaced: [keys, aggregates]
  }
  key_idx <- ifelse(is.na(over), seq_len(n_groups), n_groups + over)
  rest <- set_diff(seq_along(agg_names), over[!is.na(over)])
  as.integer(c(key_idx, n_groups + rest))
}

#' @export
infer_schema.ast_reframe <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.ast_join <- function(node) {
  # *_full(): the children's factor levels and POSIXct zones (F32) are
  # metadata that plain infer_schema() drops above a source.
  left_schema <- infer_schema_full(node$left)
  right_schema <- if (!node$join_type %in% c("semi", "anti")) infer_schema_full(node$right)
  infer_join_schema(node, left_schema, right_schema)
}

# A join's output schema from the (full) schemas of its two inputs;
# `right_schema` is unused for semi/anti. push_join() passes the schemas it
# already has, so recording a join doesn't walk either side's plan.
# @keywords internal
infer_join_schema <- function(node, left_schema, right_schema) {
  # semi/anti joins are row filters, not column merges (Phase 7 J2): the
  # output is x's (the left input's) own schema verbatim, factor_levels
  # included -- there is no right-side column to merge in and no
  # suffix/keep decision to make.
  if (node$join_type %in% c("semi", "anti")) {
    # a row filter: vec_slice() turns an absent tzone into "" (F32)
    left_schema$tz <- tz_from_positions(left_schema$names, left_schema$types,
                                        tz_localize(tz_by_position(left_schema)))
    return(left_schema)
  }

  # cross (Phase 7 J3): reuses this same general (non-semi/anti) branch
  # unchanged -- cross_join.tbl_gpu() (R/join.R) already builds an empty
  # `by` spec + `keep = TRUE`, which makes build_join_schema()/
  # build_join_output_info() suffix EVERY common name (not just would-be
  # join keys, since there are none) and drop nothing from either side --
  # exactly dplyr's own cross_join() column-naming contract (verified
  # empirically against dplyr 1.2.1).
  build_join_schema(left_schema, right_schema, node$by,
                    suffix = node$suffix, keep = node$keep)
}

#' @export
infer_schema.ast_window <- function(node) {
  infer_schema_from(node, infer_schema_memo(node$input))
}

#' @export
infer_schema.NULL <- function(node) {
  list(names = character(0), types = character(0))
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Recursively find all function calls in an R expression
#'
#' @param expr An R expression (from parse())
#' @return Character vector of function names
#' @keywords internal
find_calls <- function(expr) {
  if (is.call(expr)) {
    fn_name <- as.character(expr[[1]])
    # Recurse into arguments
    arg_calls <- unlist(lapply(as.list(expr)[-1], find_calls))
    unique(c(fn_name, arg_calls))
  } else if (is.recursive(expr)) {
    unique(unlist(lapply(expr, find_calls)))
  } else {
    character(0)
  }
}

#' Check if an expression is opaque (contains unknown functions)
#'
#' @param expr_text Expression as text string
#' @return TRUE if expression contains unknown/unsafe functions
#' @keywords internal
is_opaque_expression <- function(expr_text) {
  safe_ops <- c("+", "-", "*", "/", "^", "(", "c")
  safe_fns <- c("desc", "n", "sum", "mean", "min", "max", "sqrt", "abs", "log",
                "log10", "log2", "exp", "ceiling", "floor", "round", "trunc",
                "is.na", "!is.na", "!", "%%", "%/%")

  parsed <- tryCatch(parse(text = expr_text)[[1]], error = function(e) NULL)
  if (is.null(parsed)) return(TRUE)

  calls <- find_calls(parsed)
  unknown <- set_diff(calls, c(safe_ops, safe_fns))

  length(unknown) > 0
}

#' Check if an AST node is an optimization barrier
#'
#' `distinct` is treated as a barrier like `summarise`/`arrange`: the
#' simplest safe choice, since a filter above a `distinct` node may reference
#' columns the `distinct` projected away (`.keep_all = FALSE`), and pushing
#' it below would change which columns are visible to it. Optimization
#' opportunity for later: a filter referencing only surviving key columns
#' (or any column when `.keep_all = TRUE`) could safely push below distinct,
#' the same way filter pushdown already works across `select`.
#'
#' `slice` (the whole family: `slice()`/`slice_head()`/`slice_tail()`/
#' `slice_min()`/`slice_max()`) is a barrier for the same reason `arrange`
#' is: it's a row-selection op whose result depends on the current row
#' order/content, so reordering or pushing other ops across it (e.g. a
#' filter that would change which rows `slice_head()` sees) would change
#' the result.
#'
#' `rename` is a barrier for the reasons documented on `ast_rename()`
#' above: the simplest safe choice for a schema-only rename, avoiding the
#' need to translate column references through its name map for anything
#' pushed across it.
#'
#' `window` (`scratchpad/phase5_window_design.md` section 2.1) is a barrier:
#' every window spec's result depends on the FULL row set (which rows exist)
#' and the current row order (frames are built from a stable sort of
#' group+order columns) -- nothing may be pushed across it, the same
#' reasoning as `arrange`/`slice`.
#'
#' Phase 12 audit note: barrier classification only matters for node TYPES
#' that actually appear in an AST at all. `do()`/`rows_insert()`/
#' `rows_append()`/`rows_update()`/`rows_patch()`/`rows_upsert()`/
#' `rows_delete()`, and every colwise/pick-bearing call (`mutate_at()`,
#' `pick()`-inside-an-expression, ...) are whole-call CPU
#' fallbacks (`gpu_fallback()`, R/fallback.R): they materialize any pending
#' lazy ops via `compute()` FIRST, then run the real dplyr verb on the
#' collected data and re-upload -- the result always has `lazy_ops = NULL`
#' (see `tests/testthat/test-fallback.R`'s "fallback verbs materialize
#' pending lazy ops" test). None of these verbs ever construct an AST node
#' of their own, so "is this node type a barrier" is simply inapplicable to
#' them -- there is nothing downstream of a fallback call for a barrier to
#' block, since nothing downstream can ever see a pending op that predates
#' the fallback.
#'
#' @param node An AST node
#' @return TRUE if node is a barrier
#' @keywords internal
is_barrier <- function(node) {
  if (is.null(node)) return(FALSE)
  switch(.subset2(node, "type"),
    arrange = , barrier = , summarise = , distinct = , slice = , rename = , window = ,
    reframe = TRUE,
    select = !is.null(.subset2(node, "positions")),
    FALSE
  )
}

#' Get the depth of an AST tree
#'
#' @param node Root AST node
#' @return Integer depth
#' @keywords internal
ast_depth <- function(node) {
  if (is.null(node)) {
    return(0L)
  }
  if (node$type == "join") {
    return(1L + max(ast_depth(node$left), ast_depth(node$right)))
  }
  if (is.null(node$input)) {
    return(1L)
  }
  1L + ast_depth(node$input)
}

#' Count nodes in an AST tree
#'
#' @param node Root AST node
#' @return Integer count
#' @keywords internal
ast_count <- function(node) {
  if (is.null(node)) {
    return(0L)
  }
  if (node$type == "join") {
    return(1L + ast_count(node$left) + ast_count(node$right))
  }
  1L + ast_count(node$input)
}

#' Count the operations pending in an AST tree
#'
#' Like [ast_count()], but source nodes (the data a plan starts from) do not
#' count.
#'
#' @param node Root AST node (or NULL)
#' @return Integer count
#' @keywords internal
ast_count_ops <- function(node) {
  if (is.null(node) || node$type == "source") {
    return(0L)
  }
  if (node$type == "join") {
    return(1L + ast_count_ops(node$left) + ast_count_ops(node$right))
  }
  1L + ast_count_ops(node$input)
}

#' Attach a source pointer to the source node in an AST
#'
#' @param node Root AST node
#' @param ptr External pointer to GPU table
#' @return AST node with source pointer attached
#' @keywords internal
set_ast_source_ptr <- function(node, ptr) {
  if (is.null(node)) return(NULL)
  # A lazy join result has no pointer of its own: its sources already carry
  # theirs (attached by the inner push_join()), so there is nothing to do.
  if (is.null(ptr)) return(node)
  if (.subset2(node, "type") == "source") {
    # Never overwrite an attached pointer (a chained join's inner sources).
    if (is.null(.subset2(node, "source_ptr"))) node$source_ptr <- ptr
    return(node)
  }
  # (fields read by exact name, without `$`'s dispatch on the node class: this
  # runs once per node of a join side's pending plan; only a join node has
  # `left`/`right`)
  child <- .subset2(node, "input")
  if (!is.null(child)) {
    node$input <- set_ast_source_ptr(child, ptr)
  }
  child <- .subset2(node, "left")
  if (!is.null(child)) {
    node$left <- set_ast_source_ptr(child, ptr)
  }
  child <- .subset2(node, "right")
  if (!is.null(child)) {
    node$right <- set_ast_source_ptr(child, ptr)
  }
  node
}

# -----------------------------------------------------------------------------
# Print Methods
# -----------------------------------------------------------------------------

#' Describe a join's `by` spec as "left op right, ..." condition pairs
#'
#' Shared by `print.ast_node()`/`ast_to_string()` (Phase 12, `show_query()`
#' coverage audit): renders every condition, not just the equi ones, so a
#' non-equi/mixed `join_by()` (`op` other than `"=="`) is visible in the
#' rendered query -- e.g. `join_by(a == b, x >= y)` shows as
#' `"a == b, x >= y"`. A `closest()` condition shows as `closest(x >= y)`.
#'
#' @param by A join spec `list(left, right, op, filter)` (see [parse_join_by()])
#' @return Character scalar, `""` if `by` has no conditions
#' @keywords internal
describe_join_conditions <- function(by) {
  if (is.null(by) || length(by$left) == 0) return("")
  cond <- paste(by$left, by$op, by$right, sep = " ")
  # A closest() condition (F24) carries `filter != "none"`.
  rolling <- length(by$filter) == length(cond) & by$filter != "none"
  cond[rolling] <- paste0("closest(", cond[rolling], ")")
  paste(cond, collapse = ", ")
}

#' Describe a slice node's mode-specific sizing/ordering detail
#'
#' @param node An `ast_slice` node
#' @return Character scalar, `""` if there's nothing mode-specific to add
#'   (shouldn't happen for any of the supported modes)
#' @keywords internal
describe_slice_detail <- function(node) {
  size <- function() paste0(if (isTRUE(node$is_prop)) "prop=" else "n=", node$amount)
  switch(node$mode,
    "head" = size(),
    "tail" = size(),
    "index" = paste0("indices=[", paste(node$raw_indices, collapse = ","), "]"),
    "bracket" = paste0("`[` rows=[", paste(utils::head(node$raw_indices, 10), collapse = ","),
                       if (length(node$raw_indices) > 10) ",..." else "", "]"),
    "rank" = paste0(
      "order_by=", node$order_col,
      ", desc=", node$descending,
      ", ", size(),
      ", with_ties=", node$with_ties,
      if (isTRUE(node$top_k)) ", top_k" else ""
    ),
    ""
  )
}

#' Describe a window node's specs as "fn->output_col" pairs
#'
#' @param specs List of `window_spec()` structures (`R/window.R`)
#' @return Character scalar
#' @keywords internal
describe_window_specs <- function(specs) {
  paste(vapply(specs, function(s) paste0(s$fn, "->", s$output_col), character(1)),
        collapse = ", ")
}

#' @export
print.ast_node <- function(x, ..., indent = 0) {
  prefix <- paste0(rep("| ", indent), collapse = "")
  cat(prefix, "ast_", x$type, sep = "")

  # Print type-specific info
  switch(x$type,
    "source" = {
      cat(" [", length(x$schema$names), " cols]", sep = "")
    },
    "filter" = {
      cat(" [", length(x$predicates), " predicates]", sep = "")
    },
    "mutate" = {
      cat(" [", length(x$expressions), " expressions]", sep = "")
    },
    "select" = {
      cat(" [", length(x$columns), " cols]", sep = "")
    },
    "arrange" = {
      cat(" [", length(x$sort_specs), " keys]", sep = "")
    },
    "summarise" = {
      cat(" [", length(x$groups), " groups, ",
          length(x$aggregations), " aggs]", sep = "")
    },
    "join" = {
      conds <- describe_join_conditions(x$by)
      cat(" [", x$join_type, " join",
          if (nzchar(conds)) paste0("; on ", conds) else "",
          "]", sep = "")
    },
    "distinct" = {
      cat(" [keys=", paste(x$key_cols, collapse = ","),
          ", keep_all=", x$keep_all, "]", sep = "")
    },
    "slice" = {
      cat(" [mode=", x$mode, ", ", describe_slice_detail(x), "]", sep = "")
    },
    "rename" = {
      pairs <- paste(x$old_names, "->", x$new_names, collapse = ", ")
      cat(" [", pairs, "]", sep = "")
    },
    "reframe" = {
      cat(" [k=", x$k, ", ", length(x$outputs), " cols]", sep = "")
    },
    "window" = {
      cat(" [", length(x$specs), " specs: ", describe_window_specs(x$specs),
          "; group_cols=", paste(x$group_cols, collapse = ","), "]", sep = "")
    }
  )

  cat("\n")

  if (x$type == "join") {
    print(x$left, indent = indent + 1)
    print(x$right, indent = indent + 1)
  } else if (!is.null(x$input)) {
    print(x$input, indent = indent + 1)
  }

  invisible(x)
}

#' Format AST as a string for debugging
#'
#' @param node Root AST node
#' @return Character string representation
#' @keywords internal
ast_to_string <- function(node) {
  if (is.null(node)) return("NULL")

  info <- switch(node$type,
    "source" = paste0("source[", paste(node$schema$names, collapse = ","), "]"),
    "filter" = paste0("filter[", length(node$predicates), "]"),
    "mutate" = paste0("mutate[", paste(vapply(node$expressions, `[[`, character(1), "output_col"), collapse = ","), "]"),
    "select" = paste0("select[", paste(node$columns, collapse = ","), "]"),
    "arrange" = paste0("arrange[", length(node$sort_specs), "]"),
    "summarise" = paste0("summarise[", length(node$aggregations), "]"),
    "barrier" = "barrier",
    "join" = {
      conds <- describe_join_conditions(node$by)
      paste0("join[", node$join_type, if (nzchar(conds)) paste0("; on ", conds) else "", "]")
    },
    "distinct" = paste0("distinct[", paste(node$key_cols, collapse = ","),
                        "; keep_all=", node$keep_all, "]"),
    "slice" = paste0("slice[", node$mode, "; ", describe_slice_detail(node), "]"),
    "rename" = paste0("rename[", paste(node$old_names, "->", node$new_names, collapse = ","), "]"),
    "window" = paste0("window[", describe_window_specs(node$specs), "]"),
    "reframe" = paste0("reframe[k=", node$k, "]"),
    node$type
  )

  if (node$type == "join") {
    paste0(info, " -> left(", ast_to_string(node$left),
           "), right(", ast_to_string(node$right), ")")
  } else if (!is.null(node$input)) {
    paste0(info, " -> ", ast_to_string(node$input))
  } else {
    info
  }
}
