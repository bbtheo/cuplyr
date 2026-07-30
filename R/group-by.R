#' Group a GPU table by one or more columns
#'
#' Marks columns to group by for subsequent operations like [summarise()].
#' The grouping is stored as metadata and does not perform any computation
#' until an aggregation is requested.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names to group by. Can be unquoted column names,
#'   tidyselect expressions, or `name = expression` computed columns (see
#'   "Computed groups" below).
#' @param .add If `FALSE` (default), will override existing groups. If `TRUE`,
#'   will add to existing groups.
#' @param .drop Whether to drop unobserved factor-level combinations from
#'   [group_data()]/[group_keys()]/[n_groups()]/etc. Defaults to `TRUE`
#'   (dplyr's own default). See "`.drop` semantics" below for the exact
#'   scope of what's honored.
#'
#' @return A grouped `tbl_gpu` object. The object has the same data but with
#'   grouping columns recorded for use by [summarise()].
#'
#' @details
#' Unlike operations like [dplyr::filter()] or [mutate()], `group_by()` does not
#' perform any expensive GPU aggregation itself. It simply records which
#' columns should be used for grouping in subsequent aggregation operations
#' (computed group columns, see below, DO run a real `mutate()`).
#'
#' The actual groupby computation happens when you call [summarise()] (or
#' one of the [group_data()] family) on the grouped table. This lazy
#' approach allows you to chain multiple operations before executing the
#' expensive groupby operation.
#'
#' ## Computed groups
#' Verified empirically against dplyr 1.2.1: a **named** argument (e.g.
#' `group_by(df, g = x %/% 2)`, or even a bare-symbol rename like
#' `group_by(df, g = x)`) is ALWAYS treated as a `mutate()` expression --
#' the original column (if any) is kept, a new (or overwritten) column is
#' added under the given name, and the table is then grouped by that name.
#' Later dots may reference earlier dots' newly-computed columns, exactly
#' like `mutate()`'s own chain semantics (e.g. `group_by(df, g = x %/% 2, h
#' = g * 10)`). Unnamed arguments are resolved via ordinary tidyselect
#' against the *current* schema (after any computed columns from this same
#' call have been added), and every dot's resulting group column name(s)
#' are combined in the exact order the arguments were written (even when
#' named and unnamed dots are interleaved).
#'
#' Calling `group_by()` with **no** arguments and `.add = FALSE` (the
#' default) clears any existing grouping entirely (verified empirically --
#' this differs from simply returning `.data` unchanged, which was this
#' function's historical, incorrect behavior for that case).
#'
#' ## `.drop` semantics
#' `.drop = FALSE` (verified empirically against dplyr 1.2.1) adds one
#' placeholder row (`.rows = integer(0)`) to [group_data()]'s output for
#' every declared *factor level combination* of the FACTOR grouping columns
#' that has no observed row anywhere in the data -- non-factor grouping
#' columns in that placeholder row are `NA` (dplyr does not attempt to
#' cross the placeholder against every observed non-factor value; see
#' `compute_group_data()`, R/group-by.R, for the exact algorithm).
#'
#' This is honored for the immediate `group_by(..., .drop = FALSE) |>
#' group_data()`-style call chain, but -- like `$groups` normally does not
#' need special handling to survive intervening verbs -- `.drop` is stored
#' as a single boolean field on the returned `tbl_gpu` (`$group_drop`) that
#' is **not** re-threaded through every other verb's own `tbl_gpu`
#' construction (`push_op()`, `push_join()`, `gpu_fallback()`, etc. all
#' default it back to `TRUE`). Full propagation would require touching
#' every construction site in the codebase for a rarely-used flag --
#' deliberately deferred (documented, not silently wrong: the common
#' `group_by(.drop = FALSE) |> group_data()` pattern works; `group_by(.drop
#' = FALSE) |> filter(...) |> group_data()` silently reverts to `.drop =
#' TRUE` semantics, matching neither an error nor a crash, just dplyr's
#' default).
#'
#' @seealso
#' \code{\link{summarise.tbl_gpu}} for aggregating grouped data,
#' \code{\link{ungroup.tbl_gpu}} for removing grouping,
#' \code{\link{group_data.tbl_gpu}} for the grouping metadata query family
#'
#' @export
#' @importFrom dplyr group_by
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Group by a single column
#'   by_cyl <- gpu_mtcars |>
#'     group_by(cyl)
#'
#'   # Group by multiple columns
#'   by_cyl_gear <- gpu_mtcars |>
#'     group_by(cyl, gear)
#'
#'   # Computed group column
#'   by_half_cyl <- gpu_mtcars |>
#'     group_by(half_cyl = cyl %/% 2)
#'
#'   # Use with summarise for aggregation
#'   result <- gpu_mtcars |>
#'     group_by(cyl) |>
#'     summarise(mean_mpg = mean(mpg)) |>
#'     collect()
#' }
group_by.tbl_gpu <- function(.data, ..., .add = FALSE, .drop = TRUE) {
  dots <- rlang::enquos(...)

  if (length(dots) == 0) {
    new_groups <- if (.add) .data$groups else character()
    new_drop <- if (.add) (.data$group_drop %||% TRUE) else TRUE
    return(new_tbl_gpu(
      ptr = .data$ptr,
      schema = .data$schema,
      lazy_ops = .data$lazy_ops,
      groups = new_groups,
      group_drop = new_drop,
      exec_mode = .data$exec_mode
    ))
  }

  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }
  is_named <- !is.na(dot_names) & dot_names != ""

  # Named args are ALWAYS a mutate() expression (see "Computed groups"
  # above) -- delegated straight to mutate.tbl_gpu() (in the dots' own
  # relative order, so later named dots can reference earlier ones' new
  # columns) rather than reimplementing expression evaluation here.
  working <- .data
  if (any(is_named)) {
    named_dots <- dots[is_named]
    working <- rlang::inject(dplyr::mutate(working, !!!named_dots))
  }

  # Resolve every dot -- named -> its own name; unnamed -> tidyselect
  # against the schema AFTER the mutate() above -- IN THE ORIGINAL ARGUMENT
  # ORDER, since dplyr preserves the exact order args were written even
  # when named/unnamed dots are interleaved (verified empirically).
  new_groups <- character()
  for (i in seq_along(dots)) {
    if (is_named[i]) {
      new_groups <- c(new_groups, dot_names[i])
    } else {
      new_groups <- c(new_groups, parse_group_cols(working, dots[i]))
    }
  }
  new_groups <- unique(new_groups)

  missing <- setdiff(new_groups, current_schema(working)$names)
  if (length(missing) > 0) {
    stop("Column(s) not found: ", paste(missing, collapse = ", "),
         "\nAvailable columns: ", paste(current_schema(working)$names, collapse = ", "),
         call. = FALSE)
  }

  if (.add && length(.data$groups) > 0) {
    new_groups <- unique(c(.data$groups, new_groups))
  }

  new_tbl_gpu(
    ptr = working$ptr,
    schema = working$schema,
    lazy_ops = working$lazy_ops,
    groups = new_groups,
    group_drop = .drop,
    exec_mode = working$exec_mode
  )
}

#' Remove grouping from a GPU table
#'
#' Removes all grouping information from a grouped `tbl_gpu` object.
#'
#' @param x A `tbl_gpu` object.
#' @param ... Ignored. Included for compatibility with dplyr generic.
#'
#' @return An ungrouped `tbl_gpu` object.
#'
#' @export
#' @importFrom dplyr ungroup
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   grouped <- gpu_mtcars |>
#'     group_by(cyl)
#'
#'   # Remove grouping
#'   ungrouped <- grouped |>
#'     ungroup()
#'
#'   # Verify groups are removed
#'   length(group_vars(ungrouped))  # 0
#' }
ungroup.tbl_gpu <- function(x, ...) {
  new_tbl_gpu(
    ptr = x$ptr,
    schema = x$schema,
    lazy_ops = x$lazy_ops,
    groups = character(),
    group_drop = TRUE,
    exec_mode = x$exec_mode
  )
}

#' Get grouping variables from a GPU table
#'
#' Returns the names of columns used for grouping.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A character vector of grouping column names.
#'
#' @export
#' @importFrom dplyr group_vars
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   grouped <- gpu_mtcars |>
#'     group_by(cyl, gear)
#'
#'   group_vars(grouped)  # c("cyl", "gear")
#' }
group_vars.tbl_gpu <- function(x) {
  x$groups
}

#' Get grouping information from a GPU table
#'
#' Returns a list of symbols representing the grouping columns.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A list of symbols for the grouping columns.
#'
#' @export
#' @importFrom dplyr groups
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   grouped <- gpu_mtcars |>
#'     group_by(cyl)
#'
#'   groups(grouped)  # list(as.symbol("cyl"))
#' }
groups.tbl_gpu <- function(x) {
  lapply(x$groups, as.symbol)
}

#' Get all column names from a GPU table (dplyr's `tbl_vars()` generic)
#'
#' `dplyr::tbl_vars()` is a thin generic: its top-level function body always
#' returns `new_sel_vars(tbl_vars_dispatch(x), group_vars(x))`, where
#' `tbl_vars_dispatch()` re-dispatches to `tbl_vars.<class>()` if one exists
#' (falling back to `tbl_vars.default <- function(x) names(x)` otherwise).
#' So a `tbl_vars.tbl_gpu` method plugs straight into that machinery -- it
#' only needs to return every column name (grouped or not); `group_vars()`
#' (already implemented, see above) supplies the grouping half.
#'
#' Added to support `dplyr::add_tally()`, which is a plain (non-generic)
#' function in dplyr 1.2 that calls `tbl_vars(x)` directly for its name-
#' collision check (`check_n_name(name, tbl_vars(x))`, i.e. it avoids
#' colliding with ANY existing column, not just the grouping columns -- see
#' `tally()`'s own narrower `check_n_name(name, group_vars(x))` check for
#' the contrasting case, R/count.R). Before this method existed,
#' `dplyr::add_tally(gpu_table)` errored inside `tbl_vars_dispatch()`
#' (`UseMethod("tbl_vars")` found no applicable method for a plain-list
#' `tbl_gpu` object, and the generic's own `tbl_vars.default` doesn't apply
#' since `tbl_gpu` isn't a data.frame subclass).
#'
#' `dplyr::tbl_nongroup_vars()` (Phase 9, task 1) is a plain, non-generic
#' function defined purely in terms of `setdiff(tbl_vars(x),
#' group_vars(x))` -- it works transparently for `tbl_gpu` via this method
#' plus `group_vars.tbl_gpu()`, with no `tbl_gpu`-specific code of its own.
#'
#' @param x A `tbl_gpu` object.
#' @return A character vector of every column name (schema-aware: reflects
#'   pending lazy ops via `current_schema()`, not just the base `$schema`).
#'
#' @export
#' @importFrom dplyr tbl_vars
#' @keywords internal
tbl_vars.tbl_gpu <- function(x) {
  current_schema(x)$names
}

#' Report whether `group_by()` drops unobserved factor levels by default
#'
#' `dplyr::group_by_drop_default()` is a generic dplyr's own internals call
#' to decide the default `.drop=` value for verbs built on top of
#' `group_by()` (e.g. `group_trim()`'s own default, see R/fallback.R). The
#' `.default` method always returns `TRUE`; this method instead reports
#' whichever value the table's own most recent `group_by(..., .drop=)` call
#' recorded (`$group_drop`, see `group_by.tbl_gpu()`'s own "`.drop`
#' semantics" docs for the scope of what's tracked).
#'
#' @param .tbl A `tbl_gpu` object.
#' @return `TRUE` or `FALSE`.
#'
#' @export
#' @importFrom dplyr group_by_drop_default
#' @keywords internal
group_by_drop_default.tbl_gpu <- function(.tbl) {
  isTRUE(.tbl$group_drop %||% TRUE)
}

# Internal: Parse column names from group_by expressions
#
# Handles both bare column names and tidyselect expressions
#
# @param .data A tbl_gpu object
# @param dots Quosures from group_by()
# @return Character vector of column names
# @keywords internal
parse_group_cols <- function(.data, dots) {
  col_names <- character()
  schema_names <- current_schema(.data)$names

  for (expr in dots) {
    expr_text <- rlang::quo_text(expr)

    # Check if it's a simple column name
    if (expr_text %in% schema_names) {
      col_names <- c(col_names, expr_text)
    } else {
      # Try tidyselect evaluation
      tryCatch({
        # Create a named vector for tidyselect
        col_positions <- stats::setNames(
          seq_along(schema_names),
          schema_names
        )
        selected <- tidyselect::eval_select(expr, data = col_positions)
        col_names <- c(col_names, names(selected))
      }, error = function(e) {
        # If tidyselect fails, assume it's a column name
        col_names <<- c(col_names, expr_text)
      })
    }
  }

  unique(col_names)
}

# =============================================================================
# Grouping metadata query family (Phase 9, task 1): group_data()/
# group_keys()/group_indices()/group_size()/n_groups() are all GPU-native,
# sharing one C++ primitive (`gpu_group_info()`, src/ops_groupby.cpp) via
# `compute_group_data()` below. `group_rows()` needs no method of its own:
# `dplyr::group_rows()` is a plain (non-generic) function defined as
# `group_data(.data)[[".rows"]]`, so it already works transparently once
# `group_data.tbl_gpu()` exists. Previously (Phase 2) all six were CPU
# fallbacks (see R/fallback.R's now-removed registrations); the iteration
# family (group_split()/group_map()/group_modify()/group_nest()/nest_by())
# inherently produces R-side objects
# and stays on the CPU fallback (R/fallback.R).
# =============================================================================

#' Compute the grouping-metadata tibble shared by the whole query family
#'
#' Materializes `.data` if needed, then either builds the trivial
#' single-row ungrouped `.rows` tibble (verified empirically against dplyr
#' 1.2.1: `group_data()` on an ungrouped table is one row, `.rows` holding
#' every row 1:n in original order -- no GPU work needed at all), or calls
#' `gpu_group_info()` (src/ops_groupby.cpp) once and assembles the result:
#' the small distinct-keys table is collected via the ordinary
#' `collect.tbl_gpu()` path (reusing its factor-level/INT64 handling
#' rather than duplicating it), and `.rows` is sliced out of the returned
#' `perm` vector using `offsets` as 0-based boundary positions (see
#' `gpu_group_info()`'s own docs for the exact contract).
#'
#' `.drop = FALSE` (see `group_by.tbl_gpu()`'s own docs) is honored as a
#' post-hoc R-side expansion over the (small, group-count-sized) result --
#' never over the underlying GPU data -- via `expand_dropped_groups()`.
#'
#' Not cached across calls (each call re-runs the GPU query): correctness
#' first, per this task's own scope -- see scratchpad/workflow_state.md's
#' Phase 9 kickoff note. The query itself is already a single GPU groupby,
#' so repeated calls are cheap relative to any actual aggregation.
#'
#' @param .data A `tbl_gpu` object
#' @return A tibble: grouping columns (if grouped) plus a `.rows` list-of
#'   column of 1-based row indices, one row per group (or one row for the
#'   whole table if ungrouped)
#' @keywords internal
compute_group_data <- function(.data) {
  materialized <- if (has_pending_ops(.data)) compute(.data) else .data

  if (length(materialized$groups) == 0) {
    nr <- dim(materialized)[1]
    # `tibble::tibble(.rows = ...)` treats a literal `.rows` argument as its
    # OWN row-count-override special case (like `tibble(.rows = 3)`), not
    # an ordinary column -- `new_tibble()` bypasses that dots-parsing
    # entirely, building the single `.rows` list-of column directly.
    rows <- vctrs::new_list_of(list(seq_len(nr)), ptype = integer())
    return(tibble::new_tibble(list(.rows = rows), nrow = length(rows)))
  }

  group_names <- materialized$groups
  schema_idx <- match(group_names, materialized$schema$names)
  group_col_idx0 <- as.integer(schema_idx - 1L)

  info <- wrap_gpu_call("group_data", gpu_group_info(materialized$ptr, group_col_idx0))

  keys_schema <- list(
    names = group_names,
    types = unname(materialized$schema$types[schema_idx])
  )
  if (!is.null(materialized$schema$factor_levels)) {
    fl <- materialized$schema$factor_levels[
      names(materialized$schema$factor_levels) %in% group_names
    ]
    keys_schema$factor_levels <- if (length(fl) > 0) fl else NULL
  }

  keys_tbl <- new_tbl_gpu(ptr = info$keys, schema = keys_schema, exec_mode = "eager")
  keys_df <- tibble::as_tibble(collect(keys_tbl))

  offsets <- info$offsets
  perm <- info$perm
  num_groups <- length(offsets) - 1L

  rows_list <- vector("list", num_groups)
  for (i in seq_len(num_groups)) {
    lo <- offsets[i] + 1L
    hi <- offsets[i + 1L]
    rows_list[[i]] <- if (hi >= lo) perm[lo:hi] else integer()
  }

  gd <- keys_df
  gd$.rows <- vctrs::new_list_of(rows_list, ptype = integer())

  if (!isTRUE(materialized$group_drop %||% TRUE)) {
    gd <- expand_dropped_groups(gd, group_names, keys_schema$factor_levels)
  }

  gd
}

# Honor group_by(.drop = FALSE) (see group_by.tbl_gpu()'s own docs): adds
# one placeholder row (.rows = integer(0)) per declared factor-level
# combination of the FACTOR grouping columns that has no row anywhere in
# `gd` -- verified empirically against dplyr 1.2.1 that non-factor grouping
# columns in a placeholder row are NA (dplyr does NOT cross the placeholder
# against every distinct value of the non-factor columns; it only expands
# the cartesian product of the factor columns' own declared levels). A
# no-op when there are no factor grouping columns at all (nothing declared
# to expand against).
# @keywords internal
expand_dropped_groups <- function(gd, group_names, factor_levels) {
  if (is.null(factor_levels) || length(factor_levels) == 0) {
    return(gd)
  }
  factor_group_names <- intersect(group_names, names(factor_levels))
  if (length(factor_group_names) == 0) {
    return(gd)
  }

  level_lists <- lapply(factor_group_names, function(nm) factor_levels[[nm]])
  names(level_lists) <- factor_group_names
  full_grid <- do.call(
    expand.grid,
    c(level_lists, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE))
  )
  for (nm in factor_group_names) {
    full_grid[[nm]] <- factor(full_grid[[nm]], levels = factor_levels[[nm]])
  }

  key_of <- function(df) do.call(paste, c(unname(as.list(df)), list(sep = "\r")))
  observed_key <- key_of(gd[factor_group_names])
  full_key <- key_of(full_grid)
  missing <- full_grid[!(full_key %in% observed_key), , drop = FALSE]

  if (nrow(missing) == 0) {
    return(gd)
  }

  other_names <- setdiff(group_names, factor_group_names)
  for (nm in other_names) {
    missing[[nm]] <- rep(gd[[nm]][NA_integer_], nrow(missing))
  }
  missing$.rows <- vctrs::new_list_of(
    replicate(nrow(missing), integer(), simplify = FALSE),
    ptype = integer()
  )
  missing <- missing[names(gd)]

  out <- vctrs::vec_rbind(gd, missing)
  ord_cols <- lapply(group_names, function(nm) out[[nm]])
  out[do.call(order, c(ord_cols, list(na.last = TRUE))), , drop = FALSE]
}

#' Grouping metadata: keys, row indices, per-row group id, sizes, and count
#'
#' `group_data()` (and the queries derived from it below) report the
#' *grouping structure* dplyr's own `grouped_df` carries as an attribute --
#' `cuplyr` computes it on demand via one native GPU groupby instead
#' (`compute_group_data()`, above), since `tbl_gpu` has no such attribute to
#' read. All are GPU-native (Phase 9, task 1) -- previously CPU fallbacks.
#'
#' `group_data()` returns a tibble of the grouping columns (in ascending key
#' order, `NA` sorted last -- cudf's own default sort-groupby order, which
#' is exactly dplyr's own convention, verified empirically) plus a `.rows`
#' list-of column of 1-based row indices; called on an ungrouped table it
#' returns a single row whose `.rows` holds every row, in original order.
#' `group_keys()` is `group_data()` minus `.rows`; `group_rows()` is just
#' `.rows`; `group_indices()` is the per-ROW group id (1-based, matching
#' `group_data()`'s row order); `group_size()` is the per-group row count
#' (`group_data()` order); `n_groups()` is the group count.
#'
#' @param .data,x,.tbl A `tbl_gpu` object.
#' @param ... For `group_keys()`/`group_indices()`: no longer accepted
#'   (matching dplyr 1.0.0+, where the `...` argument of both was
#'   deprecated then made defunct -- passing anything here is an error).
#' @name group-data
#' @keywords internal
NULL

#' @rdname group-data
#' @export
#' @importFrom dplyr group_data
group_data.tbl_gpu <- function(.data) {
  compute_group_data(.data)
}

#' @rdname group-data
#' @export
#' @importFrom dplyr group_keys
group_keys.tbl_gpu <- function(.tbl, ...) {
  dots <- rlang::enquos(...)
  if (length(dots) > 0) {
    stop(
      "The `...` argument of `group_keys()` was deprecated in dplyr 1.0.0 ",
      "and is now defunct.\nPlease `group_by()` first.",
      call. = FALSE
    )
  }
  gd <- compute_group_data(.tbl)
  gd[setdiff(names(gd), ".rows")]
}

#' @rdname group-data
#' @export
#' @importFrom dplyr group_indices
group_indices.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)
  if (length(dots) > 0) {
    stop(
      "The `...` argument of `group_indices()` was deprecated in dplyr 1.0.0 ",
      "and is now defunct.\nPlease `group_by()` first.",
      call. = FALSE
    )
  }
  rows <- compute_group_data(.data)$.rows
  n <- sum(lengths(rows))
  idx <- integer(n)
  for (i in seq_along(rows)) {
    idx[rows[[i]]] <- i
  }
  idx
}

#' @rdname group-data
#' @export
#' @importFrom dplyr group_size
group_size.tbl_gpu <- function(x) {
  lengths(compute_group_data(x)$.rows)
}

#' @rdname group-data
#' @export
#' @importFrom dplyr n_groups
n_groups.tbl_gpu <- function(x) {
  nrow(compute_group_data(x))
}
