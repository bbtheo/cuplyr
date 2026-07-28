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
#' }
#' Comparing a column against a literal of an incompatible type (e.g. a
#' numeric column against a character literal) errors immediately, naming
#' the column and its type, rather than silently falling back or producing
#' a raw GPU-side type error.
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
#'   \item Aggregates or window functions inside a filter predicate (e.g.
#'     `filter(x > mean(x))`, `filter(row_number() == 1)`) -- planned for a
#'     later phase (window functions need group-aware evaluation that
#'     `compute_column()` cannot do; see `scratchpad/phase1_expression_engine.md`
#'     section 6)
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

  # Schema is stable across this whole call: filter() never adds/removes
  # columns, so a single current_schema() snapshot is valid for every dot,
  # even across an intervening eval-mask fallback materialization.
  schema <- current_schema(.data)
  predicates <- list()

  flush_predicates <- function() {
    if (length(predicates) > 0) {
      .data <<- push_op(.data, ast_filter(input_node(.data), predicates))
      predicates <<- list()
    }
  }

  for (quo in dots) {
    ir <- parse_filter_ir(quo, schema)

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

  .data
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
