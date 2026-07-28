#' Filter rows of a GPU table
#'
#' Selects rows from a GPU table where conditions are TRUE, similar to
#' `dplyr::filter()`. Filtering is performed entirely on the GPU for
#' maximum performance on large datasets, and matches dplyr's row-keeping
#' and three-valued-logic (`NA`) semantics exactly (see "NA semantics"
#' below).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Logical expressions to filter by (see "Supported expressions"
#'   below for the full surface). Multiple expressions, whether passed as
#'   separate dots or combined with `&` within one dot, are always combined
#'   with AND (a row is kept only if every expression is TRUE for it).
#' @param .preserve Ignored. Included for compatibility with dplyr generic.
#'
#' @return A `tbl_gpu` object containing only rows where all conditions are TRUE.
#'   The GPU memory for the filtered result is newly allocated.
#'
#' @details
#' ## Supported expressions
#' `filter()` parses its arguments through an internal expression IR (see
#' `R/ir.R`) that lowers every predicate in the call to a **single fused GPU
#' kernel** (`cudf::compute_column()` under the hood) -- one kernel per
#' `filter()` call, no matter how many dots or how deeply the predicate is
#' nested. The IR understands:
#' \itemize{
#'   \item Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=` -- against a
#'     literal, another column, an environment variable, or an arithmetic
#'     expression (`(x - y) > 0`)
#'   \item Logic: `&`, `|`, `!`, `xor()`, arbitrarily nested
#'     (`x > 2 & (y < 5 | z == 1)`); `&&`/`||` are rejected with a message
#'     pointing at `&`/`|` (matching dplyr 1.1+)
#'   \item `is.na()`
#'   \item `%in%`, including a set containing `NA` (R's never-NA rule: a
#'     value that itself is `NA` is TRUE for `%in%` iff the set contains
#'     `NA`, never `NA` itself) and an empty set (`character(0)`, `c()`,
#'     etc. -- always FALSE, for every row, including `NA` rows)
#'   \item `between(x, lo, hi)` (dplyr's definition: `x >= lo & x <= hi`,
#'     including its `NA` propagation), `near(x, y, tol = )`
#'     (`abs(x - y) < tol`, matching `dplyr::near()` exactly)
#'   \item String-column comparisons (`==`/`!=`/`<`/`<=`/`>`/`>=`), against
#'     a literal or another string column
#'   \item Column-vs-column comparisons for every operator above
#'   \item Math on columns inside a predicate (`+ - * / %% %/% ^`, `sqrt`,
#'     `log`, `abs`, etc. -- the same set `mutate()` supports)
#'   \item `if_else()`, `case_when()`, `coalesce()`, `na_if()`,
#'     `case_match()`, `recode_values()`, `replace_values()`,
#'     `replace_when()` -- the same conditional/vector functions `mutate()`
#'     supports (see its own docs for the exact type-unification rules),
#'     usable anywhere inside a predicate, including as the whole predicate
#'     itself (e.g. `filter(case_when(x > 1 ~ TRUE, .default = FALSE))`)
#'   \item `when_any(...)`/`when_all(...)` -- particularly useful inside
#'     `filter()` to combine several comma-separated conditions with `|`
#'     instead of `&` (e.g. `filter(when_any(x > 10, y < 0))`)
#'   \item The `.data`/`.env` pronouns, and ordinary R symbols (a bare name
#'     that matches a column resolves to that column; otherwise it's
#'     evaluated as an environment variable -- "columns shadow the
#'     environment")
#'   \item Window/aggregate functions (Phase 5, task W5), usable anywhere
#'     inside a predicate, including combined with plain column
#'     comparisons in the same dot (`filter(x > mean(x) & y < 5)`):
#'     `row_number()`/`row_number(x)`, `min_rank()`/`dense_rank()`/
#'     `percent_rank()`/`cume_dist()`, `lag()`/`lead()`,
#'     `cumsum()`/`cummax()`/`cummin()`/`cumprod()`/`cummean()`/
#'     `cumall()`/`cumany()`/`consecutive_id()`, `ntile()`,
#'     `mean()`/`sum()`/`min()`/`max()`/`n()`/`sd()`/`var()`/`first()`/
#'     `last()`/`nth()` -- the same set `mutate()` supports (see its own
#'     docs for exact semantics/argument support), evaluated per `group_by()`
#'     group when `.data` is grouped (`.by=` is not yet supported for
#'     `filter()`, a later phase). A grouped filter -- like dplyr's own --
#'     preserves the ORIGINAL row order in the result, it does not reorder
#'     into group-key order.
#' }
#' Comparing a column against a literal of an incompatible type (e.g. a
#' numeric column against a character literal) errors immediately, naming
#' the column and its type, rather than silently falling back or producing
#' a raw GPU-side type error.
#'
#' ## Multi-dot semantics with window/aggregate predicates
#' Multiple dots (or multiple `&`-joined predicates within one dot) are
#' always combined with AND, and -- confirmed empirically against real
#' dplyr, since this is easy to get wrong -- every dot's predicate is
#' evaluated against the SAME, ORIGINAL (pre-`filter()`) data, never
#' against a progressively-narrowed intermediate result. Concretely,
#' `filter(x > 0, cumsum(x) < 10)` computes `cumsum(x)` over every row of
#' the original table, then ANDs that mask with `x > 0` -- it does NOT
#' filter to `x > 0` first and then compute `cumsum()` only over the
#' surviving rows (which would generally give a different, smaller
#' cumulative sum and therefore a different result). `cuplyr` reproduces
#' this exactly: every window-bearing dot's temp columns are computed
#' against the table as it stood when `filter()` was called, before any
#' dot's predicate (window-derived or plain) has removed a single row.
#'
#' ## NA semantics
#' `filter()` keeps a row iff its combined predicate is non-missing *and*
#' TRUE -- identical to dplyr (`cudf::apply_boolean_mask()`'s "non-null AND
#' true" rule matches `dplyr::filter()`'s own row-keeping rule exactly).
#' Concretely:
#' \itemize{
#'   \item `filter(x > 5)` drops rows where `x` is `NA` (the comparison is
#'     `NA`, and `NA` isn't kept), same as dplyr
#'   \item `&`/`|` use R's three-valued logic, not C's: `NA & FALSE` is
#'     `FALSE`, `NA | TRUE` is `TRUE`, `NA & TRUE`/`NA | FALSE`/`NA & NA`/
#'     `NA | NA` are all `NA` (`cudf::ast_operator::NULL_LOGICAL_AND`/
#'     `NULL_LOGICAL_OR`, chosen specifically because they match)
#'   \item `filter(x > 2 | is.na(x))` is the idiom for "keep NA rows too"
#'     -- works exactly as it does in dplyr
#'   \item `%in%` never produces `NA` (see "Supported expressions" above) --
#'     this is R's `%in%`/`match()` semantics, not null-propagating
#'   \item `filter(TRUE)` is a no-op (returns `.data` unchanged, no GPU
#'     work); `filter(FALSE)` and `filter(NA)` both return a zero-row
#'     result (a bare `NA` predicate is `FALSE`-equivalent for row-keeping,
#'     same as dplyr)
#' }
#' One documented, deliberate divergence from dplyr: cuDF's floating-point
#' `NaN` is not the same value as SQL/cuDF null, so `is.na()` on a *computed*
#' `NaN` (e.g. `0/0`) returns `FALSE` on the GPU where R's `is.na(NaN)` is
#' `TRUE`. `NA` values ingested from R (including in a `FLOAT64` column)
#' are unaffected -- they round-trip as genuine GPU nulls and behave
#' identically to dplyr in every case above.
#'
#' ## Not yet supported
#' \itemize{
#'   \item `.by=` on-the-fly grouping combined with a window/aggregate
#'     predicate (a later phase; use `group_by()` first instead)
#'   \item Arbitrary R functions with no cuDF equivalent -- these fall back
#'     to CPU evaluation, see below
#' }
#'
#' ## Expressions the IR doesn't understand (CPU fallback)
#' An expression shape the IR doesn't recognize (see `R/ir.R`'s
#' `ir_parse_quo()`) is handed to a CPU-side fallback: it's evaluated with
#' `rlang::eval_tidy()` and, if the result is a logical scalar or vector,
#' applied back to the GPU table as a boolean mask (materializing first,
#' via `compute()`, if `.data` is lazy with pending operations). Because
#' this evaluation happens with *no data mask*, it can only succeed for
#' expressions that evaluate to a logical value on their own without
#' referencing table columns -- e.g. a pre-computed logical vector or
#' `filter(rep(TRUE, n))`. An expression that references a table column
#' but doesn't parse (a rare shape given how much of dplyr's filter surface
#' the IR now covers) will still error with the "only supports comparisons"
#' message, since the column symbol can't resolve outside the table. This
#' fallback path is legitimate, supported behavior for the cases it does
#' cover, not an error condition -- see `getOption("cuplyr.fallback")`
#' below to be notified when it's used.
#'
#' ## `options(cuplyr.fallback = ...)`
#' Controls whether the CPU fallback described above (used by `filter()`
#' and by `summarise()`'s temp-column preprocessing) notifies the caller.
#' One of:
#' \itemize{
#'   \item `"silent"` (default) - no notification
#'   \item `"warn"` - emits a `warning()` naming the verb and expression
#'   \item `"error"` - `stop()`s instead of falling back
#' }
#'
#' ## Performance
#' Filtering on GPU is highly parallel and can process billions of rows
#' per second. For best performance, prefer expressions the IR parses
#' directly into a GPU predicate (the entire "Supported expressions" list
#' above) over expressions that require the CPU fallback.
#'
#' @seealso
#' \code{\link{mutate.tbl_gpu}} for creating new columns,
#' \code{\link{select.tbl_gpu}} for selecting columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr filter
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Filter with single condition
#'   efficient_cars <- gpu_mtcars |>
#'     filter(mpg > 25)
#'
#'   # Multiple conditions (combined with AND)
#'   result <- gpu_mtcars |>
#'     filter(mpg > 20) |>
#'     filter(cyl == 4) |>
#'     collect()
#'
#'   # Compare two columns
#'   gpu_cars <- tbl_gpu(cars)
#'   fast_stops <- gpu_cars |>
#'     filter(dist < speed) |>
#'     collect()
#' }
filter.tbl_gpu <- function(.data, ..., .preserve = FALSE) {
  dots <- rlang::enquos(...)

  if (length(dots) == 0) return(.data)

  # Schema is stable across this whole call for every PLAIN dot: filter()
  # never adds/removes real columns. The one exception is Phase 5's
  # window-predicate handling directly below: when at least one dot's IR
  # contains a window-function call, the schema is temporarily extended
  # with `..win*..`/`..winarg*..` temp columns for the remainder of this
  # call; a trailing ast_select drops them again before returning, so the
  # table's real, user-visible schema is unaffected either way.
  orig_data <- .data
  schema <- current_schema(.data)
  orig_names <- schema$names
  verb_groups <- .data$groups

  # First pass: parse every dot up front (no push_op()/GPU work yet) so we
  # know, before touching `.data` at all, whether any dot needs the window
  # path below. Doing this classification BEFORE the per-dot loop's
  # ir_is_const() check (further down) matters, not just for staging: a
  # bare window call like `n()` has zero column ARGUMENTS of its own
  # (ir_cols() only walks `$args`), so `ir_is_const(ir_parse_quo(quo(n() >
  # 2)))` would itself read TRUE -- without routing window calls through
  # this dedicated path first, `filter(n() > 2)` fell into the
  # constant-expression branch instead (eval_tidy()-ing `n() > 2` with no
  # table in scope, which either errors or silently misbehaves) rather than
  # being planned as a real window predicate. This was the crash this task
  # fixes (see the regression test): checking ir_has_window() here, before
  # any const-folding logic ever runs, routes it correctly instead.
  parsed <- vector("list", length(dots))
  window_idx <- integer()

  for (i in seq_along(dots)) {
    ir <- parse_filter_ir(dots[[i]], schema)
    # `parsed[i] <- list(ir)`, NOT `parsed[[i]] <- ir`: the latter, when
    # `ir` is NULL (an opaque expression the IR doesn't understand -- a
    # perfectly ordinary, common case, see the opaque-dot branch below),
    # *deletes* that list element instead of storing NULL in it (R's
    # `x[[i]] <- NULL` list-element-removal special case), silently
    # shrinking `parsed` below `length(dots)` and desyncing every
    # subsequent index -- caught by the crash-regression/opaque-dot tests.
    parsed[i] <- list(ir)
    if (!is.null(ir) && ir_has_window(ir)) {
      window_idx <- c(window_idx, i)
    }
  }

  # Phase 5, task W5: at least one dot's predicate contains a window call
  # (mean(x), n(), row_number(), cumsum(x), lag(x), ...). dplyr's multi-dot
  # semantics (verified empirically -- see this file's roxygen "Multi-dot
  # semantics" section): every dot's predicate is evaluated against the
  # SAME, ORIGINAL (pre-filter()) data, then ANDed together -- never
  # sequential/progressive narrowing, where a later dot's window aggregate
  # would "see" an earlier dot's already-filtered rows. We reproduce this by
  # planning and materializing EVERY window dot's temp columns up front,
  # against `.data` exactly as it stood at function entry -- before any
  # predicate (plain or window-derived) has removed a single row -- and
  # only afterward building/pushing ONE combined ast_filter node folding in
  # every dot's (possibly rewritten) predicate, in original dot order.
  if (length(window_idx) > 0) {
    window_exprs <- stats::setNames(
      parsed[window_idx],
      paste0("..filterwin", seq_along(window_idx), "..")
    )

    plan <- filter_plan_window(window_exprs, schema, verb_groups)

    if (is.null(plan)) {
      # The plan failed outright, or produced a window spec gpu_window()
      # can't lower (window_spec_lowerable(), R/lower.R -- unreachable for
      # any window kind currently in the registry, kept here defensively
      # for a future kind). "Never half-lower" (design doc section 1.3):
      # nothing has been pushed onto `.data` yet at this point, so falling
      # back to CPU evaluation of the WHOLE call, from the pristine
      # original table and dots, is safe.
      return(filter_window_fallback(orig_data, dots))
    }

    for (stage in plan$stages) {
      if (length(stage$pre) > 0) {
        .data <- push_op(.data, ast_mutate(input_node(.data), stage$pre))
      }
      .data <- push_op(.data, ast_window(input_node(.data), stage$specs, verb_groups))
    }

    for (k in seq_along(window_idx)) {
      parsed[window_idx[k]] <- plan$post[k]
    }
    schema <- plan$schema
  }

  predicates <- list()

  flush_predicates <- function() {
    if (length(predicates) > 0) {
      .data <<- push_op(.data, ast_filter(input_node(.data), predicates))
      predicates <<- list()
    }
  }

  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    ir <- parsed[[i]]

    if (is.null(ir)) {
      # Opaque expression (the IR doesn't understand this shape): flush any
      # predicates accumulated so far into a real ast_filter node, then
      # fall back to CPU-side evaluation of this expression (D4).
      flush_predicates()
      .data <- filter_eval_mask(.data, quo)
      next
    }

    if (ir_is_const(ir)) {
      # A constant expression (ir_cols() empty -- no column reference
      # anywhere in the subtree, e.g. TRUE, FALSE, `1 > 2`, `rep(TRUE, n)`):
      # section 3's design decision is that these never reach the IR/GPU
      # expression path. `compute_column()`/`gpu_filter_expr()` need a real
      # per-row column to broadcast an n-row mask against; a schema-less
      # constant subtree gives them nothing to anchor that broadcast to.
      # Evaluate in R instead (on the original quosure, not the already-
      # constant-folded IR) and route to the same no-op / impossible-
      # predicate / mask-apply paths the pre-IR parser used.
      value <- tryCatch(rlang::eval_tidy(quo), error = function(e) NULL)

      if (is.logical(value) && length(value) == 1 && isTRUE(value)) {
        next  # TRUE: genuine no-op, contributes nothing (D4/S4 semantics)
      }

      if (is.logical(value) && length(value) == 1) {
        # FALSE or NA: an impossible predicate. This is accumulated as a
        # normal predicate (preserving laziness exactly like any other
        # predicate, replacing the deleted "col != col" synthesis) rather
        # than executed eagerly: a bare FALSE literal never reaches
        # compute_column()'s AST-build path -- gpu_filter_expr()'s
        # materialize() has a dedicated `lit` branch that broadcasts a
        # literal scalar directly across the table's real row count, no
        # AST tree involved.
        predicates <- c(predicates, list(make_predicate(ir_lit_from_r(FALSE), schema)))
        next
      }

      # Anything else constant (a non-logical value, or a logical vector
      # such as `rep(TRUE, n)`): same CPU-eval fallback as an opaque
      # expression -- filter_eval_mask() has the exact error/mask-apply
      # logic for these shapes already.
      flush_predicates()
      .data <- filter_eval_mask(.data, quo)
      next
    }

    check_filter_comparison_types(ir, schema)
    predicates <- c(predicates, list(make_predicate(ir, schema)))
  }

  flush_predicates()

  if (length(window_idx) > 0) {
    # Drop every `..win*../..winarg*..` temp column introduced by the
    # window-planning block above, restoring the table's real,
    # user-visible schema (section 1.3's "trailing select" step).
    .data <- push_op(.data, ast_select(input_node(.data), orig_names))
  }

  .data
}

# Internal: plan + validate the window stages for filter()'s window-bearing
# dots (Phase 5, task W5).
#
# A thin wrapper around plan_window_stages() (R/window.R) -- the exact same
# planning machinery mutate_window()/plan_mutate_window_batches() (R/
# mutate.R) drives -- that additionally validates every produced spec is
# lowerable (window_spec_lowerable(), R/lower.R), mirroring mutate_window()'s
# own two-pass "plan then validate, only emit if everything checks out"
# shape: `filter.tbl_gpu()` must never push a single AST node before it
# knows the WHOLE plan can be lowered (never half-lower).
#
# Unlike mutate()'s multi-dot chunking (window_batches()'s "does a later
# dot's window-call argument reference an earlier dot's OWN output column"
# rule), filter() dots never define a named output column for another dot
# to reference -- each dot is an independent boolean predicate -- so every
# window-bearing dot here is always planned jointly, in one batch (sharing
# CSE across dots, e.g. two dots both calling `mean(x)` still produce a
# single temp column). `exprs`' names are purely internal bookkeeping
# plan_window_stages() requires (never surfaced as a real column, since the
# trailing ast_select in filter.tbl_gpu() always restores the pre-window
# column set) -- they exist only so plan_window_stages()'s own chunking
# logic has something to compare against, which never actually splits
# anything here for exactly the reason above.
#
# @param exprs A *named* list of per-dot window-bearing IR nodes (only the
#   dots identified by `window_idx` in filter.tbl_gpu(), not every dot)
# @param schema `.data`'s schema before this filter() call
# @param group_cols Character vector, `.data$groups` (the `.by=` variant is
#   Phase 5 task W6, not handled here)
# @return `plan_window_stages()`'s result list, or `NULL` if planning
#   errored or produced an unlowerable spec
# @keywords internal
filter_plan_window <- function(exprs, schema, group_cols) {
  plan <- tryCatch(plan_window_stages(exprs, schema, group_cols), error = function(e) NULL)
  if (is.null(plan)) {
    return(NULL)
  }

  all_specs <- unlist(lapply(plan$stages, `[[`, "specs"), recursive = FALSE)
  if (length(all_specs) > 0 && !all(vapply(all_specs, window_spec_lowerable, logical(1)))) {
    return(NULL)
  }

  plan
}

# Internal: whole-call CPU fallback for a window-bearing filter() call whose
# plan isn't (yet) fully lowerable (Phase 5, task W5).
#
# Mirrors mutate_window_fallback() (R/mutate.R): as of Phase 5 task W4,
# every window kind has C++ support (window_spec_lowerable()), so this is
# only reachable for a genuinely new, not-yet-implemented window kind added
# by some FUTURE phase -- there is currently no dplyr syntax that reaches
# it. Unlike filter_eval_mask() (which evaluates a single quosure with NO
# data mask, and so can never work for a window call referencing real table
# columns), this re-runs the real dplyr::filter() call, with every original
# dot, on the CPU via gpu_fallback() -- the only correct way to evaluate a
# window/aggregate predicate that cuplyr can't lower natively.
#
# @param .data The ORIGINAL tbl_gpu (BEFORE this filter() call -- nothing
#   has been pushed onto it yet, see the "never half-lower" comment at the
#   call site)
# @param dots The original (already rlang::enquos()'d) dots, every one of
#   them (not just the window-bearing ones), so the re-run dplyr::filter()
#   call reproduces the exact original semantics
# @return A new tbl_gpu (via gpu_fallback(), which also emits the
#   cuplyr_fallback_notify() notification)
# @keywords internal
filter_window_fallback <- function(.data, dots) {
  gpu_fallback("filter", .data, function(tbl) {
    rlang::inject(dplyr::filter(tbl, !!!dots))
  })
}

# ir_parse_quo() throws (rather than returning NULL) when a constant-folded
# subtree evaluates to a vector outside %in%'s RHS position
# (ir_lit_from_r()'s scalar-only rule, section 1.2 of
# scratchpad/phase1_expression_engine.md) -- exactly right for mutate(),
# where a bare vector literal is always a user error, but filter() has
# always accepted a directly-evaluable logical vector as a mask
# (`filter(rep(TRUE, n))`), which must still fall back to CPU evaluation
# rather than error at parse time. Catch specifically that "expected a
# scalar" shape and treat it like any other unsupported/opaque expression
# (NULL, triggering the fallback in filter.tbl_gpu()); anything else (e.g.
# "not found", or the `&&`/`||` guard) propagates as a real error.
#
# @param quo A quosure containing a filter expression
# @param schema The current schema (names/types) to parse against
# @return An IR node, or NULL if unsupported
# @keywords internal
parse_filter_ir <- function(quo, schema) {
  tryCatch(
    ir_parse_quo(quo, schema),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("scalar", msg, fixed = TRUE) && grepl("length", msg, fixed = TRUE)) {
        return(NULL)
      }
      stop(e)
    }
  )
}

# Diagnosable filter()-specific type check: comparing a non-STRING column
# against a STRING literal (or vice versa) used to be caught by the old
# parser's "numeric scalar" RHS-type guard; the IR parses this shape
# successfully (both `col` and `lit` are valid nodes), so without this
# check it would reach the GPU and fail with a raw cudf type-mismatch
# error instead of a clear message naming the column and its type
# (scratchpad/phase1_expression_engine.md section 4's error-contract
# ledger). Recurses through the whole IR subtree since a single filter()
# dot may already parse into a compound `&`/`|` expression under the IR
# cutover.
#
# @param ir An IR node
# @param schema The current schema (names/types)
# @return `invisible(NULL)`; `stop()`s on a detected mismatch
# @keywords internal
check_filter_comparison_types <- function(ir, schema) {
  if (is.null(ir) || !identical(ir$kind, "call")) {
    return(invisible(NULL))
  }

  cmp_ops <- c("==", "!=", "<", "<=", ">", ">=")
  if (ir$op %in% cmp_ops && length(ir$args) == 2) {
    check_pair <- function(col_node, lit_node) {
      if (!identical(col_node$kind, "col") || !identical(lit_node$kind, "lit")) {
        return(invisible(NULL))
      }
      if (is.null(lit_node$type) || !identical(lit_node$type, "STRING")) {
        return(invisible(NULL))
      }
      # unname(): `schema$types[match(...)]` is a *named* single-element
      # subset (name = the matched column), and identical() considers the
      # "names" attribute significant -- identical(c(s = "STRING"),
      # "STRING") is FALSE even though the values are equal. Without
      # unname() here, every STRING-column-vs-STRING-literal comparison
      # (a completely valid, supported shape) fell through to the stop()
      # below instead of returning early (T4 oracle-testing finding: this
      # broke every string filter comparison against a literal).
      col_type <- unname(schema$types[match(col_node$name, schema$names)])
      if (is.na(col_type) || identical(col_type, "STRING")) {
        return(invisible(NULL))
      }
      stop("Cannot compare column '", col_node$name, "' (", col_type,
           ") with a character literal.", call. = FALSE)
    }
    check_pair(ir$args[[1]], ir$args[[2]])
    check_pair(ir$args[[2]], ir$args[[1]])
  }

  for (arg in ir$args) {
    check_filter_comparison_types(arg, schema)
  }

  invisible(NULL)
}

# Internal: CPU-eval fallback for filter expressions that aren't a
# recognized comparison shape.
#
# Materializes any pending lazy operations first (so the fallback always
# operates on a real GPU table), eval_tidy()s the quosure (no explicit data
# mask - matches the historical behavior of only supporting expressions
# that evaluate directly, e.g. `rep(TRUE, n)`, not column references),
# validates the result is logical, and applies it via gpu_filter_bool()
# (scalar/uniform case) or gpu_filter_mask() (mixed mask), same as the
# renamed-from filter_logical() did. Notifies via cuplyr_fallback_notify()
# (D4: silent by default).
#
# @param .data A tbl_gpu object
# @param quo A quosure whose evaluated result must be logical
# @return A filtered tbl_gpu object
# @keywords internal
filter_eval_mask <- function(.data, quo) {
  if (identical(.data$exec_mode, "lazy") && has_pending_ops(.data)) {
    .data <- compute(.data)
  }

  expr_chr <- rlang::quo_text(quo)
  eval_result <- tryCatch(rlang::eval_tidy(quo), error = function(e) NULL)

  if (is.null(eval_result) || !is.logical(eval_result)) {
    stop("filter() only supports comparisons: ==, !=, >, >=, <, <=\n",
         "Or logical values: TRUE, FALSE, logical vectors\n",
         "Expression: ", expr_chr, call. = FALSE)
  }

  cuplyr_fallback_notify("filter", expr_chr)

  n_rows <- dim(.data)[1]

  if (length(eval_result) == 1) {
    # Single boolean: TRUE keeps all rows, FALSE keeps none
    if (isTRUE(eval_result)) {
      new_ptr <- wrap_gpu_call("filter_bool_true", gpu_filter_bool(.data$ptr, TRUE))
    } else {
      new_ptr <- wrap_gpu_call("filter_bool_false", gpu_filter_bool(.data$ptr, FALSE))
    }
  } else {
    # Logical vector: use as mask
    if (length(eval_result) != n_rows) {
      stop("Logical vector length (", length(eval_result),
           ") must match number of rows (", n_rows, ")", call. = FALSE)
    }

    # Check for all TRUE or all FALSE (optimize common cases)
    if (all(eval_result, na.rm = TRUE) && !any(is.na(eval_result))) {
      new_ptr <- wrap_gpu_call("filter_bool_all_true", gpu_filter_bool(.data$ptr, TRUE))
    } else if (!any(eval_result, na.rm = TRUE)) {
      new_ptr <- wrap_gpu_call("filter_bool_all_false", gpu_filter_bool(.data$ptr, FALSE))
    } else {
      # Mixed: apply mask
      new_ptr <- wrap_gpu_call("filter_mask", gpu_filter_mask(.data$ptr, eval_result))
    }
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}
