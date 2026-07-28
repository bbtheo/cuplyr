#' Summarise groups in a GPU table
#'
#' Computes aggregations on groups defined by [group_by()]. Operations are
#' performed entirely on the GPU for maximum performance.
#'
#' @param .data A grouped `tbl_gpu` object created by [group_by()].
#' @param ... Name-value pairs of summary functions. The name will be the
#'   name of the variable in the result. The value must be a single aggregation
#'   expression in the form `fun(column)`, where `column` can also be any
#'   expression the internal expression IR understands (see "Aggregation
#'   sub-expressions" below).
#' @param .groups Controls grouping structure of the result. Currently only
#'   "drop" is supported (default).
#' @param .by Optional on-the-fly grouping columns (tidyselect), GPU-native
#'   (Phase 5, task W9). Unlike `mutate()`/`filter()`'s `.by=` (which behave
#'   exactly like `group_by()`), `summarise(.by=)` has one genuinely
#'   different rule verified empirically against dplyr 1.2.1: groups are
#'   emitted in FIRST-APPEARANCE order (the order each distinct `.by`
#'   combination first appears in `.data`), not sorted group-key order --
#'   see "`.by` group order" below. The result is always ungrouped (same as
#'   a `group_by()`-driven `summarise()`). Supplying `.by` when `.data` is
#'   already grouped (via `group_by()`) is an error, matching dplyr exactly.
#'
#' @return A `tbl_gpu` object with one row per group containing the grouping
#'   columns and computed aggregations.
#'
#' @details
#' ## Supported aggregation functions
#' \itemize{
#'   \item `sum(x)` - Sum of values
#'   \item `mean(x)` - Arithmetic mean
#'   \item `min(x)` - Minimum value
#'   \item `max(x)` - Maximum value
#'   \item `n()` - Count of rows in each group
#'   \item `sd(x)` - Standard deviation
#'   \item `var(x)` - Variance
#' }
#'
#' ## Aggregation sub-expressions
#' `column` is not limited to a bare column name: any expression the same
#' expression IR that powers [filter.tbl_gpu()]/[mutate.tbl_gpu()] understands
#' is computed as a hidden temporary column first, then aggregated. This
#' covers comparisons and logic (`sum(carb > 3 & wt < 4)`), arithmetic
#' (`mean(hp / wt)`), and math functions (`mean(sqrt(hp))`, `sum(carb %% 2 ==
#' 0)`), arbitrarily nested -- the same surface documented in
#' \code{\link{mutate.tbl_gpu}}. A comparison/logical sub-expression is
#' summed as `TRUE`/`FALSE` -> `1`/`0`, matching R's own coercion
#' (`sum(c(TRUE, TRUE))` is `2L`). An expression shape the IR doesn't
#' recognize raises the same "Invalid aggregation expression" error as any
#' other unsupported aggregation shape.
#'
#' ## NA handling
#' `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()` all match R's own
#' `na.rm = FALSE` default (Phase 6, task 6.1): if a group contains ANY `NA`
#' in the aggregated column, the WHOLE group's result is `NA` for that
#' aggregation -- e.g. `summarise(m = mean(x))` is `NA` for a group with even
#' one `NA` value of `x`, matching real dplyr exactly. Pass `na.rm = TRUE`
#' explicitly (e.g. `sum(x, na.rm = TRUE)`) to exclude `NA` values instead
#' (cudf's own native aggregation behavior). `n()` is unaffected by
#' `na.rm=` (it counts every row regardless of nulls, like dplyr's own
#' `n()`).
#'
#' ## Ungrouped summarise
#' If `.data` is not grouped, summarise will compute aggregations over all
#' rows, returning a single-row table.
#'
#' ## `.by` group order
#' Verified empirically against dplyr 1.2.1: `summarise(.by=)` emits one row
#' per distinct `.by` combination in the order that combination FIRST
#' appears in `.data` -- e.g. `summarise(tibble(g = c("b","a","b")), n =
#' n(), .by = g)` returns `g` in the order `c("b", "a")`, not `c("a", "b")`.
#' This differs from `group_by(g) |> summarise(...)`, whose group order is
#' cuDF's own (unspecified, hash-groupby-derived) order. Implemented via a
#' desugar (`summarise_by_desugar()`): an ungrouped `row_number()` marks
#' each row's original position, a grouped summarise computes the user's
#' aggregations plus `min()` of that position per group, and the result is
#' sorted by that minimum (each group's first-appearance position) before
#' the helper column is dropped.
#'
#' @seealso
#' \code{\link{group_by.tbl_gpu}} for grouping data,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr summarise summarize
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Simple aggregation over all rows
#'   total <- gpu_mtcars |>
#'     summarise(avg_mpg = mean(mpg)) |>
#'     collect()
#'
#'   # Grouped aggregation
#'   by_cyl <- gpu_mtcars |>
#'     group_by(cyl) |>
#'     summarise(
#'       avg_mpg = mean(mpg),
#'       max_hp = max(hp),
#'       count = n()
#'     ) |>
#'     collect()
#'
#'   # Multiple grouping columns
#'   by_cyl_gear <- gpu_mtcars |>
#'     group_by(cyl, gear) |>
#'     summarise(
#'       mean_mpg = mean(mpg),
#'       min_wt = min(wt)
#'     ) |>
#'     collect()
#' }
summarise.tbl_gpu <- function(.data, ..., .groups = "drop", .by = NULL) {
  by_quo <- rlang::enquo(.by)

  # Phase 5, task W9: `.by=` resolution -- checked up front (before the
  # zero-dots error just below), matching mutate()/filter()'s own `.by=`
  # resolution order: resolve_by() (R/execute.R) errors immediately if
  # `.data` is already grouped, and this must fire even when `...` is empty
  # (verified empirically: `summarise(group_by(df, g), .by = g)` raises the
  # grouped-`.data` error, it does NOT reach summarise()'s own "requires at
  # least one aggregation expression" check first).
  by_cols <- resolve_by(.data, by_quo, ".by")
  by_given <- !rlang::quo_is_null(by_quo)

  dots <- rlang::enquos(...)

  if (length(dots) == 0) {
    stop("summarise() requires at least one aggregation expression.",
         call. = FALSE)
  }

  dots <- auto_name_dots(dots, "summarise")

  if (by_given) {
    return(summarise_by_desugar(.data, dots, by_cols))
  }

  summarise_core(.data, dots)
}

# Internal: the shared aggregation pipeline -- temp-column preprocessing,
# aggregation parsing, one ast_summarise push -- used both by the ordinary
# group_by()-or-ungrouped path and by summarise_by_desugar()'s own internal
# grouped summarise call (with its extra `..summarise_first..` dot already
# appended).
#
# @param .data A tbl_gpu (its own `$groups` drive the aggregation)
# @param dots Already auto_name_dots()'d aggregation quosures
# @return A new (ungrouped -- ast_summarise always drops groups, see
#   propagate_groups(), R/execute.R) tbl_gpu
# @keywords internal
summarise_core <- function(.data, dots) {
  # Phase 6, task 6.1 (Fix A): extract and strip any `na.rm = <literal>`
  # named argument BEFORE temp-column preprocessing -- this must happen
  # first so a sub-expression call like `sum(sqrt(x), na.rm = TRUE)` is
  # reduced to the `fn(single_arg)` shape preprocess_agg_expressions() (and
  # parse_aggregations()'s own regex) already understand, with na.rm
  # tracked out-of-band per dot instead of embedded in the call shape.
  extracted <- lapply(dots, extract_na_rm)
  na_rm_flags <- vapply(extracted, `[[`, logical(1), "na_rm")
  dots <- lapply(extracted, `[[`, "quo")

  # Pre-process: create temporary columns for expressions inside agg functions
  # (e.g. sum(carb == 4)). Per D4, if a temp column is needed and .data is
  # lazy with pending ops, this materializes (compute()) before creating the
  # temp column and notifies via cuplyr_fallback_notify().
  preprocess_result <- preprocess_agg_expressions(.data, dots)
  working_data <- preprocess_result$data
  processed_dots <- preprocess_result$dots

  # Parse the aggregation expressions (now with simple column refs), emitting
  # aggregation structs that carry the cudf-accepted function name (fixes
  # lazy sd()/var(), which used to reach C++ as "sd"/"var" and error out).
  aggregations <- parse_aggregations(current_schema(working_data), processed_dots, na_rm_flags)

  push_op(working_data, ast_summarise(input_node(working_data), aggregations,
                                      working_data$groups))
}

# Internal: Extract and strip an `na.rm = <literal>` named argument from a
# single summarise() aggregation dot (Phase 6, task 6.1, Fix A)
#
# Real dplyr's `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()` all default to
# `na.rm = FALSE` (verified empirically against dplyr 1.2.1: any NA in a
# group makes the WHOLE group's aggregation result NA). cudf's own groupby
# aggregations always exclude nulls, so this default requires an extra
# whole-group-null propagation step in `gpu_summarise()`
# (`src/ops_groupby.cpp`) -- `na_rm = FALSE` (the R-side default emitted
# when no `na.rm=` is present) requests that step; `na.rm = TRUE` opts back
# into cudf's native (NA-excluding) behavior directly, no extra step.
#
# Mirrors `ir_parse_sum_call()`'s (R/ir.R) validation style: `na.rm`'s value
# is resolved via `eval_tidy()` in the dot's own environment (so a bound
# variable like `na.rm = drop_na` works, not just a literal `TRUE`/`FALSE`
# token), but unlike `ir_parse_sum_call()` (which returns `NULL` to signal
# "try a different shape" to its caller, since mutate()/filter() have
# fallback paths), an unevaluable or non-logical `na.rm` is a hard error
# here -- summarise()'s aggregation parsing has no CPU fallback (see this
# file's own module docs).
#
# @param quo A quosure, one summarise() dot (post `auto_name_dots()`)
# @return `list(quo = <possibly rewritten quosure, na.rm stripped>, na_rm =
#   logical(1))`
# @keywords internal
extract_na_rm <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)

  if (!is.call(expr)) {
    return(list(quo = quo, na_rm = FALSE))
  }

  arg_names <- names(expr)
  if (is.null(arg_names)) {
    return(list(quo = quo, na_rm = FALSE))
  }

  na_rm_pos <- which(arg_names == "na.rm")
  if (length(na_rm_pos) == 0) {
    return(list(quo = quo, na_rm = FALSE))
  }
  if (length(na_rm_pos) > 1) {
    stop("Multiple na.rm= arguments in aggregation expression: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  na_rm_val <- tryCatch(
    rlang::eval_tidy(rlang::new_quosure(expr[[na_rm_pos]], env)),
    error = function(e) NULL
  )
  if (!is.logical(na_rm_val) || length(na_rm_val) != 1 || is.na(na_rm_val)) {
    stop("na.rm= must evaluate to a single TRUE or FALSE in aggregation expression: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  stripped_expr <- expr[-na_rm_pos]
  list(quo = rlang::new_quosure(stripped_expr, env), na_rm = isTRUE(na_rm_val))
}

# Internal: `.by=` desugar for summarise() (Phase 5, task W9)
#
# scratchpad/phase5_window_design.md section 6:
#   1. mutate(..rowid.. = row_number()) -- an UNGROUPED window call (safe:
#      resolve_by() already guarantees `.data` itself carries no groups
#      when `.by=` is supplied, so this is never a grouped row_number()).
#   2. Group by `by_cols` and run the user's own aggregation dots PLUS one
#      extra `..first.. = min(..rowid..)` aggregation -- the smallest
#      original row index surviving in each group is exactly that group's
#      first-appearance position (every row of a group shares the same
#      grouped-summarise result, so `min()` over the whole group is safe
#      even though only one row's rowid "matters").
#   3. arrange(..first..) orders the one-row-per-group result by that
#      first-appearance position -- reproducing dplyr's empirically-verified
#      `.by=` group order (see summarise.tbl_gpu()'s own roxygen).
#   4. Drop `..first..`. The result is already ungrouped (ast_summarise
#      always drops groups), matching dplyr's own `.by=` contract.
#
# @param .data A tbl_gpu, confirmed ungrouped by resolve_by() before this is
#   ever called
# @param dots The already-auto_name_dots()'d aggregation quosures (`...`)
# @param by_cols Character vector, the resolved `.by=` columns
# @return A new (ungrouped) tbl_gpu, one row per distinct `by_cols`
#   combination, in first-appearance order
# @keywords internal
summarise_by_desugar <- function(.data, dots, by_cols) {
  reserved_names <- c(current_schema(.data)$names, names(dots))

  rowid_name <- unique_temp_name("..summarise_rowid..", reserved_names)
  with_rowid <- dplyr::mutate(.data, !!rowid_name := dplyr::row_number())

  # Set groups directly (equivalent to group_by(by_cols), but avoids a
  # round trip through the group_by() generic for a purely internal step).
  grouped <- new_tbl_gpu(
    ptr = with_rowid$ptr,
    schema = with_rowid$schema,
    lazy_ops = with_rowid$lazy_ops,
    groups = by_cols,
    exec_mode = with_rowid$exec_mode
  )

  first_name <- unique_temp_name("..summarise_first..", c(reserved_names, rowid_name))
  dots[[first_name]] <- rlang::new_quosure(
    rlang::call2("min", as.name(rowid_name)), rlang::base_env()
  )

  result <- summarise_core(grouped, dots)
  result <- dplyr::arrange(result, !!rlang::sym(first_name))
  dplyr::select(result, -dplyr::all_of(first_name))
}

#' @rdname summarise.tbl_gpu
#' @export
summarize.tbl_gpu <- summarise.tbl_gpu

# Internal: Strip a leading `pkg::` qualifier from an aggregation call's
# text (e.g. `dplyr::n()` -> `n()`, `stats::sd(mpg)` -> `sd(mpg)`), so the
# shape-matching regexes below (and in parse_aggregations()) see the bare
# call regardless of how the user namespaced it.
#
# @param expr_text Character, expression text from rlang::quo_text()
# @return Character, with any leading `identifier::` removed
# @keywords internal
strip_ns_prefix <- function(expr_text) {
  sub("^[a-zA-Z_][a-zA-Z0-9_.]*::", "", trimws(expr_text))
}

# Internal: decompose an aggregation dot's raw expression into a bare
# `fn(arg)` (or no-arg `fn()`) shape, stripping one leading `pkg::`
# qualifier from the call head, e.g. `dplyr::n()` -> list(fn_name = "n",
# arg = NULL), `stats::sd(mpg)` -> list(fn_name = "sd", arg = quote(mpg)).
# Returns `NULL` if `expr` isn't a call, its head isn't a (possibly
# namespaced) bare symbol, or it has more than one argument (multi-arg
# aggregations like `quantile(x, 0.5)` aren't supported -- left alone here,
# parse_aggregations() raises the actual diagnosable error for these).
#
# @param expr A language object (rlang::quo_get_expr() of a summarise() dot)
# @return `list(fn_name, arg)` (`arg` is `NULL` for a no-arg call), or `NULL`
# @keywords internal
decompose_agg_call <- function(expr) {
  if (!is.call(expr)) {
    return(NULL)
  }

  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]  # strip namespace: pkg::fn -> fn
  }
  if (!is.symbol(head)) {
    return(NULL)
  }

  n_args <- length(expr) - 1L
  if (n_args == 0) {
    return(list(fn_name = as.character(head), arg = NULL))
  }
  if (n_args == 1) {
    return(list(fn_name = as.character(head), arg = expr[[2]]))
  }
  NULL
}

# Internal: Pre-process aggregation expressions
#
# Detects expressions inside aggregation functions (e.g., sum(carb == 4),
# mean(sqrt(x)), sum(x %% 2 == 0)) and creates temporary columns for them
# via the expression IR (R/ir.R) -- the same evaluator filter()/mutate()
# use. Since the aggregation's argument is parsed with `ir_parse_quo()`,
# any expression shape the IR understands is now a valid aggregation
# sub-expression (T7 widens this from the previous single-operator-only
# regex whitelist to "ir_parse_quo() succeeds").
#
# Per D4: if a dot needs a temp column and `.data` is lazy with pending
# operations, this materializes (`compute()`) once, up front, before
# building any temp column, and notifies via `cuplyr_fallback_notify()` --
# `create_temp_column()` always operates on a materialized GPU pointer
# rather than reaching into an in-progress AST.
#
# @param .data A tbl_gpu object
# @param dots Quosures from summarise()
# @return List with modified data and simplified dots
# @keywords internal
preprocess_agg_expressions <- function(.data, dots) {
  working_data <- .data
  new_dots <- vector("list", length(dots))
  names(new_dots) <- names(dots)

  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    raw_expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)

    decomposed <- decompose_agg_call(raw_expr)

    if (is.null(decomposed) || is.null(decomposed$arg)) {
      # Not a `fn(single_arg)` shape (includes bare `n()`), or not a call
      # at all: nothing to preprocess. parse_aggregations() validates the
      # shape (and raises its own diagnosable error) from here.
      new_dots[[i]] <- quo
      next
    }

    arg_expr <- decomposed$arg

    # A bare column reference (the common case, e.g. sum(mpg)) needs no
    # temp column -- and is safe to check even while lazy ops are still
    # pending, since it's a pure schema lookup with no GPU work involved.
    if (is.symbol(arg_expr) && as.character(arg_expr) %in% current_schema(working_data)$names) {
      new_dots[[i]] <- quo
      next
    }

    arg_quo <- rlang::new_quosure(arg_expr, env)
    ir <- ir_parse_quo(arg_quo, current_schema(working_data))

    if (is.null(ir) || ir_is_const(ir)) {
      # Either genuinely unparseable by the expression IR, or a constant
      # subtree with no column reference at all (e.g. `sum(5)`): neither is
      # a temp-column shape. Leave the dot untouched -- parse_aggregations()
      # raises the actual diagnosable error for these rare/degenerate
      # shapes (a literal that isn't a valid column reference).
      new_dots[[i]] <- quo
      next
    }

    # Materialize before building the temp column if lazy ops are pending.
    if (has_pending_ops(working_data)) {
      working_data <- compute(working_data)
      cuplyr_fallback_notify("summarise", rlang::quo_text(quo))
    }

    # Create a temporary column name (index-based, so it stays unique even
    # though not every dot ends up needing one).
    temp_col_name <- paste0(".temp_agg_", i)

    working_data <- create_temp_column(working_data, temp_col_name, ir)

    # Create a new quosure with the temp column name, e.g. `sum(.temp_agg_3)`.
    new_dots[[i]] <- rlang::new_quosure(
      as.call(list(as.name(decomposed$fn_name), as.name(temp_col_name))),
      rlang::base_env()
    )
  }

  list(data = working_data, dots = new_dots)
}

# Internal: Create a temporary column from a parsed IR expression
#
# One IR-based implementation for every aggregation sub-expression shape
# (comparison, arithmetic, logical, math function, arbitrarily nested) --
# built via a single `ir_call`-equivalent step through `gpu_mutate_expr()`,
# the same evaluator entry point `mutate()` uses.
#
# @param .data A tbl_gpu object (already materialized -- no pending lazy
#   ops; `preprocess_agg_expressions()` guarantees this)
# @param col_name Name for the new column
# @param ir A parsed (not yet bound) IR node, non-const (references at
#   least one column)
# @return Modified tbl_gpu with the new column appended
# @keywords internal
create_temp_column <- function(.data, col_name, ir) {
  schema <- .data$schema
  inferred_type <- ir_infer_type(ir, schema)

  # A boolean-producing expression (comparisons, &/|/!, is.na(), between(),
  # %in%, ...) is declared as INT32 here, matching R's own coercion of
  # logical values in arithmetic contexts (`sum(c(TRUE, TRUE))` is `2L`, an
  # integer, not logical) -- exactly what sum() (the overwhelmingly common
  # aggregation over a comparison, e.g. `sum(carb > 3)`) expects downstream
  # via make_aggregation()'s INT32->INT64 promotion rule. gpu_mutate_expr()
  # performs the actual GPU-side cast (BOOL8 -> INT32) since the declared
  # output type differs from the computed one (src/ops_expr.cpp).
  new_type <- if (identical(inferred_type, "BOOL8")) "INT32" else inferred_type

  bound_ir <- ir_bind(ir, schema)

  new_ptr <- wrap_gpu_call(
    "summarise_mutate_expr",
    gpu_mutate_expr(.data$ptr, list(bound_ir), col_name, new_type, schema$names)
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(
      names = c(schema$names, col_name),
      types = c(schema$types, new_type)
    ),
    lazy_ops = .data$lazy_ops,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
}

# Internal: Parse aggregation expressions into make_aggregation() structs
#
# The one parser used by both eager and lazy scheduling (D1): given a
# schema and quosures already simplified by preprocess_agg_expressions()
# (so every dot's argument is either a bare column name or the no-arg
# n()), validates function/column names -- raising the same three
# diagnosable errors the pre-unification eager parser did, verbatim -- and
# emits aggregation structs (via make_aggregation(), which also folds in
# former get_agg_result_type()'s result-type inference) carrying the
# *cudf*-accepted function name. This is the fix for lazy sd()/var(),
# which used to reach C++ as "sd"/"var" (gpu_summarise()/get_groupby_agg()
# in src/ops_groupby.cpp only accept "std"/"variance") and error out with
# "Unknown aggregation type"; the eager path already translated these
# names and continues to do so here.
#
# A leading `pkg::` qualifier (`dplyr::n()`, `stats::sd(mpg)`, ...) is
# stripped before shape-matching so namespaced aggregation calls resolve
# the same way as their bare equivalents.
#
# @param schema Current schema (list(names=, types=))
# @param dots Quosures from summarise(), already temp-column-preprocessed
# @param na_rm_flags Logical vector, same length/order as `dots` (Phase 6,
#   task 6.1): each dot's already-extracted `na.rm` value (see
#   `extract_na_rm()`), `FALSE` when not supplied by the user, matching R's
#   own default
# @return List of aggregation structs (see make_aggregation())
# @keywords internal
parse_aggregations <- function(schema, dots, na_rm_flags = rep(FALSE, length(dots))) {
  # dplyr-facing aggregation function names (order matches the historical
  # error message from parse_agg_expressions()) and their cudf-accepted
  # equivalents (src/ops_groupby.cpp::get_groupby_agg()).
  agg_functions <- c("sum", "mean", "min", "max", "n", "sd", "var", "count")
  agg_fn_map <- c(
    sum = "sum", mean = "mean", min = "min", max = "max",
    n = "n", sd = "std", var = "variance", count = "n"
  )

  aggregations <- vector("list", length(dots))

  for (i in seq_along(dots)) {
    expr <- dots[[i]]
    expr_text <- rlang::quo_text(expr)
    match_text <- strip_ns_prefix(expr_text)

    # Check for n() which has no column argument
    if (grepl("^n\\(\\)$", match_text)) {
      aggregations[[i]] <- make_aggregation(names(dots)[i], NA_character_, "n")
      next
    }

    # Parse function(column) pattern
    match_result <- regmatches(
      match_text,
      regexec("^([a-zA-Z_][a-zA-Z0-9_]*)\\(([^)]+)\\)$", match_text)
    )[[1]]

    if (length(match_result) != 3) {
      stop("Invalid aggregation expression: ", expr_text,
           "\nExpected format: function(column), e.g., mean(mpg)",
           call. = FALSE)
    }

    func_name <- match_result[2]
    col_name <- trimws(match_result[3])

    # Validate function
    if (!func_name %in% agg_functions) {
      stop("Unsupported aggregation function: ", func_name,
           "\nSupported functions: ", paste(agg_functions, collapse = ", "),
           call. = FALSE)
    }

    # Validate column
    col_idx <- match(col_name, schema$names)
    if (is.na(col_idx)) {
      stop("Column '", col_name, "' not found.",
           "\nAvailable columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }

    input_type <- schema$types[col_idx]
    cudf_fn <- unname(agg_fn_map[func_name])

    aggregations[[i]] <- make_aggregation(names(dots)[i], col_name, cudf_fn, input_type,
                                          na_rm = isTRUE(na_rm_flags[i]))
  }

  aggregations
}
