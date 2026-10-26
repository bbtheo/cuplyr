#' Group a GPU table by one or more columns
#'
#' Marks columns to group by for subsequent operations like [summarise()].
#' The grouping is stored as metadata and does not perform any computation
#' until an aggregation is requested.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Variables to group by: column names (`x`, `.data$x`), a
#'   top-level [dplyr::pick()], or computed columns (`name = expression`, or an
#'   unnamed expression such as `a + 1`; see "Computed groups" below). Like
#'   dplyr, tidyselect helpers are only allowed inside `pick()`.
#' @param .add If `FALSE` (default), will override existing groups. If `TRUE`,
#'   will add to existing groups.
#' @param .drop Whether to drop unobserved factor-level combinations from
#'   [group_data()]/[group_keys()]/[n_groups()]/etc. Defaults to
#'   `group_by_drop_default(.data)`: `TRUE` for an ungrouped table, the
#'   table's current setting for a grouped one (as in dplyr). See "`.drop`
#'   semantics" below for the exact scope of what's honored.
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
#' = g * 10)`). An **unnamed** argument is a column reference (`x`,
#' `.data$x`, `.data[["x"]]`), a top-level `pick()`, or -- anything else --
#' a data-masking expression computed into a new column named
#' `rlang::as_label()` of the expression (`group_by(df, a + 1)` makes a column
#' called `a + 1`). All computed dots (named and unnamed) run as one
#' `mutate()` on the ungrouped data, in argument order. Every dot's resulting
#' group column name(s) are combined in the order the arguments were written.
#' The old `add =` and `.dots =` arguments are defunct and error.
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
#' that has no observed row, expanded recursively like dplyr: under each
#' parent group a factor column takes all its levels (plus `NA` if observed
#' there) and a non-factor column the values observed under that parent (a
#' single `NA` under an empty parent); see `expand_dropped_groups()`.
#'
#' `.drop` is stored as a single boolean on the returned `tbl_gpu`
#' (`$group_drop`) and carried by `push_op()`/`push_join()` through every verb
#' that keeps a group column (and by `count()`, `add_count()` and
#' `slice_sample()`, which build their result directly); it is reset to `TRUE`
#' once no group column is left. Like dplyr, `.drop` defaults to the table's
#' current setting, so regrouping a `.drop = FALSE` table keeps it.
#'
#' Groups are recomputed from the data each time, and empty factor-level
#' groups are re-added from the declared levels. Row-subsetting verbs also
#' keep emptied groups of any key type, as dplyr does. On a `.drop = FALSE`
#' table, `filter()`, `filter_out()`, `slice()`, `slice_head()`,
#' `slice_tail()`, `slice_min()`, `slice_max()`, `slice_sample()`,
#' `semi_join()`, `anti_join()` and `rows_delete()` keep every group,
#' emptied ones included. `.preserve = TRUE` does the same for `slice()` and
#' `filter()` on a `.drop = TRUE` table. The result fixes the input's
#' group-keys table (`$fixed_keys`), only when a group can have been lost,
#' which `group_data()` and friends, `collect()`, `summarise()`/`count()`/
#' `tally()` (on the GPU, with a row for each emptied group) and the CPU
#' fallbacks honour.
#' `mutate()`, `select()`, `rename()` and `relocate()` carry it as long as
#' they leave the group columns alone, and a CPU fallback result carries
#' whatever empty groups dplyr's own result kept (see `gpu_fallback()`: a
#' fallback `mutate()` keeps them, a fallback `arrange()` on a `.drop = TRUE`
#' table drops them); every
#' other verb, and any change of grouping, drops it.
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
group_by.tbl_gpu <- function(.data, ..., .add = FALSE,
                             .drop = dplyr::group_by_drop_default(.data)) {
  dots <- rlang::enquos(...)

  # dplyr's group_by() (with or without dots) rebuilds the table with tibble,
  # which refuses duplicated names; grouping metadata would name an ambiguous
  # column.
  dup <- anyDuplicated(.data$schema$names)
  if (dup > 0L) {
    rlang::abort(paste0("Column name `", .data$schema$names[[dup]], "` must not be duplicated.\n",
                        "Use `.name_repair` to specify repair."),
                 class = c("tibble_error_column_names_must_be_unique", "tibble_error"))
  }

  if (length(dots) == 0) {
    new_groups <- if (.add) .data$groups else character()
    new_drop <- if (.add && length(.data$groups) > 0L) .drop else TRUE
    # group_by() keeps the attributes but always gives a tibble (F31). On an
    # already grouped table dplyr rebuilds through as_tibble() and loses them
    # unless `.add` leaves the grouping as it was.
    keep_attrs <- (length(.data$groups) == 0L && !is_rowwise_gpu(.data)) || length(new_groups) > 0L
    return(copy_frame_meta(new_tbl_gpu(
      ptr = .data$ptr,
      schema = .data$schema,
      lazy_ops = .data$lazy_ops,
      groups = new_groups,
      group_drop = new_drop,
      exec_mode = .data$exec_mode,
      nrow = .data[["nrow"]],
      unspecified = .data[["unspecified"]]
    ), .data, attrs = keep_attrs, bare = FALSE))
  }

  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }
  if (any(c("add", ".dots") %in% dot_names)) {
    if ("add" %in% dot_names) {
      lifecycle::deprecate_stop("1.0.0", "group_by(add = )", "group_by(.add = )")
    }
    lifecycle::deprecate_stop("1.0.0", "group_by(.dots = )")
  }
  is_named <- !is.na(dot_names) & dot_names != ""

  # Whole-call CPU fallback for shapes with no tbl_gpu representation or that
  # only dplyr can resolve (pick() inside wrappers, across(), a named pick()
  # that makes a df-column, ...). Always reruns the ORIGINAL call, from the
  # original `.data`, so nothing half-applied leaks.
  group_by_fallback <- function() {
    gpu_fallback("group_by", .data, function(tbl) {
      dplyr::group_by(tbl, !!!dots, .add = .add, .drop = .drop)
    })
  }

  # Classify each dot like dplyr's group_by_prepare(): a variable reference
  # (`x`, `.data$x`, `.data[["x"]]`) or a top-level pick() names existing
  # columns; everything else -- every named dot, and any unnamed expression
  # (`a + 1`, `2`, `"g"`, `-a`, ...) -- is data-masking and computed into a
  # column called as_label(expr).
  kind <- character(length(dots))     # "named", "var", "pick", "computed"
  var_name <- character(length(dots))
  comp_names <- dot_names
  for (i in seq_along(dots)) {
    if (is_named[i]) {
      kind[i] <- "computed"
      next
    }
    expr <- rlang::quo_get_expr(dots[[i]])
    if (is.null(expr)) {
      # an unnamed NULL (e.g. an unused `{{ arg }}`) is ignored by dplyr
      kind[i] <- "null"
    } else if (is.symbol(expr)) {
      kind[i] <- "var"
      var_name[i] <- as.character(expr)
    } else if (is_data_pronoun_call(expr)) {
      # `.data[[var]]` arrives as `.data[["name"]]`: rlang evaluates the
      # subscript when capturing the quosure
      kind[i] <- "var"
      var_name[i] <- pronoun_call_name(expr)
    } else if (rlang::is_call(expr) &&
               (call_head_name(expr) %in% c("pick", "across"))) {
      kind[i] <- "pick"
    } else {
      kind[i] <- "computed"
      comp_names[i] <- rlang::as_label(dots[[i]])
    }
  }
  is_comp <- kind == "computed"
  # A column named twice has no single "type at this position"
  if (anyDuplicated(comp_names[is_comp])) {
    return(group_by_fallback())
  }

  # Computed dots run as ONE mutate() on the UNGROUPED data (#5598), so n(),
  # pick() and friends see the whole table even when `.data` is already
  # grouped, and later dots see earlier ones' columns.
  working <- .data
  if (any(is_comp)) {
    comp_dots <- stats::setNames(dots[is_comp], comp_names[is_comp])
    working <- rlang::inject(dplyr::mutate(dplyr::ungroup(working), !!!comp_dots))
    if (!is_tbl_gpu(working)) {
      return(group_by_fallback())
    }
  }

  working_schema <- current_schema(working)
  orig_schema <- current_schema(.data)
  orig_names <- orig_schema$names
  new_groups <- character()
  for (i in seq_along(dots)) {
    if (kind[i] == "null") {
      next
    } else if (kind[i] == "computed") {
      # a computed dot that evaluated to NULL made no column: dplyr ignores it
      # when unnamed and rejects it when named
      if (!comp_names[i] %in% working_schema$names) {
        if (is_named[i]) {
          rlang::abort(c(
            "Must group by variables found in `.data`.",
            x = paste0("Column `", comp_names[i], "` is not found.")
          ))
        }
        next
      }
      new_groups <- c(new_groups, comp_names[i])
    } else if (kind[i] == "var") {
      if (!var_name[i] %in% working_schema$names) {
        rlang::abort(c(
          "Must group by variables found in `.data`.",
          x = paste0("Column `", var_name[i], "` is not found."),
          i = paste0("Available columns: ",
                     paste(working_schema$names, collapse = ", "))
        ))
      }
      new_groups <- c(new_groups, var_name[i])
    } else {
      # A pick() only sees the columns as they are at its position: the
      # original ones (with their original type until a computed dot
      # redefines them) plus those made by earlier computed dots.
      earlier <- comp_names[seq_len(i - 1L)][is_comp[seq_len(i - 1L)]]
      visible <- working_schema$names %in% c(orig_names, earlier)
      pick_names <- working_schema$names[visible]
      use_working <- pick_names %in% earlier
      pick_types <- ifelse(
        use_working,
        working_schema$types[match(pick_names, working_schema$names)],
        orig_schema$types[match(pick_names, orig_names)]
      )
      pick_schema <- list(names = pick_names, types = unname(pick_types))
      picked <- resolve_pick_call(dots[[i]], pick_schema)
      if (is.null(picked)) {
        return(group_by_fallback())
      }
      new_groups <- c(new_groups, picked)
    }
  }
  new_groups <- unique(new_groups)

  if (.add && length(.data$groups) > 0) {
    new_groups <- unique(c(.data$groups, new_groups))
  }

  # A grouped input keeps its attributes only through a plain regroup (every
  # dot a column reference, new_grouped_df()); computed dots go through
  # dplyr_col_modify.grouped_df() and an empty grouping through as_tibble(),
  # which drop them. An ungrouped input always keeps them.
  keep_attrs <- (length(.data$groups) == 0L && !is_rowwise_gpu(.data)) ||
    (all(kind == "var") && length(new_groups) > 0L)

  copy_frame_meta(new_tbl_gpu(
    ptr = working$ptr,
    schema = working$schema,
    lazy_ops = working$lazy_ops,
    groups = new_groups,
    # dots that select no column leave the table ungrouped, `.drop` and all
    group_drop = if (length(new_groups) > 0L) .drop else TRUE,
    exec_mode = working$exec_mode,
    nrow = working[["nrow"]],
    unspecified = working[["unspecified"]]
  ), .data, attrs = keep_attrs, bare = FALSE)
}

#' Remove grouping from a GPU table
#'
#' Removes grouping from a grouped `tbl_gpu` object: all of it, or only the
#' group columns selected in `...`.
#'
#' @param x A `tbl_gpu` object.
#' @param ... Columns to remove from the grouping (tidyselect; renaming is an
#'   error). Empty removes all groups. Must be empty for an ungrouped table.
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
  groups <- x$groups
  drop <- TRUE
  if (is_rowwise_gpu(x)) {
    # ungroup.rowwise_df(): nothing may be named, and it is as_tibble()
    rlang::check_dots_empty()
    return(new_tbl_gpu(
      ptr = x$ptr, schema = x$schema, lazy_ops = x$lazy_ops, exec_mode = x$exec_mode,
      nrow = x[["nrow"]], unspecified = x[["unspecified"]]
    ))
  }
  if (...length() > 0L) {
    if (length(groups) == 0L) {
      rlang::check_dots_empty()
    }
    # tidyselect over every column; the selected group columns stop being
    # groups (other selections are ignored), `.drop` is kept while any is left
    # (predicates see the schema's type proxies, like pick()/across())
    schema <- current_schema(x)
    dots <- rlang::enquos(...)
    if (!all(vapply(dots, function(q) {
      pick_selection_is_type_only(rlang::quo_get_expr(q), rlang::quo_get_env(q), schema$names)
    }, logical(1)))) {
      return(gpu_fallback("ungroup", x, function(tbl) dplyr::ungroup(tbl, !!!dots)))
    }
    # the proxy carries the grouping: group_cols() reads it from peek_data()
    proxy <- tidyselect_proxy(schema, groups)
    sel <- tidyselect::eval_select(rlang::expr(c(!!!dots)), proxy, allow_rename = FALSE)
    groups <- set_diff(groups, names(sel))
    if (length(groups) > 0L) {
      drop <- group_by_drop_default(x)
    }
  } else {
    groups <- character()
  }
  # An ungrouped x comes back as is. A grouped one is rebuilt: a full ungroup
  # goes through as_tibble() and loses its attributes, a partial one through
  # group_by(x, <remaining>) which keeps them (as a tibble, F31).
  out <- new_tbl_gpu(
    ptr = x$ptr,
    schema = x$schema,
    lazy_ops = x$lazy_ops,
    groups = groups,
    group_drop = drop,
    exec_mode = x$exec_mode,
    nrow = x[["nrow"]],
    unspecified = x[["unspecified"]]
  )
  if (length(x$groups) > 0L && length(groups) > 0L) {
    copy_frame_meta(out, x, bare = FALSE)
  } else {
    carry_frame_meta(out, x)
  }
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
#' `group_by()` (e.g. `group_trim()`'s own default, see below). The
#' `.default` method always returns `TRUE`; this method returns `TRUE` for an
#' ungrouped table and otherwise whichever value the table's own most recent
#' `group_by(..., .drop=)` call recorded (`$group_drop`, see
#' `group_by.tbl_gpu()`'s own "`.drop` semantics" docs for the scope of
#' what's tracked).
#'
#' @param .tbl A `tbl_gpu` object.
#' @return `TRUE` or `FALSE`.
#'
#' @export
#' @importFrom dplyr group_by_drop_default
#' @keywords internal
group_by_drop_default.tbl_gpu <- function(.tbl) {
  # dplyr reports TRUE for any ungrouped table; a stale `$group_drop` left on
  # one must not leak into the next group_by().
  if (length(.subset2(.tbl, "groups")) == 0L) {
    return(TRUE)
  }
  isTRUE(.subset2(.tbl, "group_drop") %||% TRUE)
}

#' Drop unused levels of the factor group columns (dplyr backend method)
#'
#' Native: [dplyr::group_trim()] on a grouped `tbl_gpu` finds the codes still
#' used by each factor group column on the GPU (a distinct over that one
#' column; only the at most `n_levels + 1` distinct codes reach R), keeps the
#' used levels in their original order (an ordered factor stays ordered),
#' remaps the codes with `gpu_remap_codes()` and regroups with
#' `ungroup()` + `group_by(.drop = )`, which also drops any `fixed_keys` left
#' by `.preserve = TRUE`. Pending ops are computed first (the new level set
#' depends on the data). An ungrouped table is returned unchanged, without
#' looking at `.drop`, like dplyr. Shapes outside the allowlist (duplicated
#' names, a factor group column without known levels) use the CPU fallback.
#'
#' @param .tbl A `tbl_gpu` object.
#' @param .drop Whether the regroup drops empty groups.
#' @return A `tbl_gpu`.
#' @export
#' @importFrom dplyr group_trim
group_trim.tbl_gpu <- function(.tbl, .drop = dplyr::group_by_drop_default(.tbl)) {
  if (is_rowwise_gpu(.tbl)) {
    # (dplyr: a rowwise_df has no groups to trim)
    return(.tbl)
  }
  if (length(.tbl$groups) == 0L) {
    return(.tbl)
  }
  if (!rlang::is_bool(.drop)) {
    rlang::abort("`.drop` must be `TRUE` or `FALSE`.")
  }

  fallback <- function() {
    gpu_fallback("group_trim", .tbl, function(tbl) {
      dplyr::group_trim(tbl, .drop = .drop)
    })
  }

  vars <- .tbl$groups
  tbl <- if (has_pending_ops(.tbl)) compute(.tbl) else .tbl
  names_ <- tbl$schema$names
  if (anyDuplicated(names_) || !all(vars %in% names_)) {
    return(fallback())
  }
  idx0 <- match(vars, names_) - 1L
  is_fct <- tbl$schema$types[idx0 + 1L] == "DICTIONARY32"
  levels_ <- tbl$schema$factor_levels
  if (any(is_fct) && !all(vars[is_fct] %in% names(levels_))) {
    return(fallback())
  }

  for (k in which(is_fct)) {
    col <- vars[[k]]
    old <- levels_[[col]]
    one <- new_tbl_gpu(
      ptr = wrap_gpu_call("group_trim", gpu_distinct(gpu_select(tbl$ptr, idx0[[k]]), 0L)),
      schema = list(names = "k", types = "INT32"), exec_mode = "eager"
    )
    codes <- collect_frame(one)[[1L]]
    used <- sort(unique(codes[!is.na(codes)]))
    if (length(used) == length(old)) {
      next
    }
    new <- old[used]
    if (isTRUE(attr(old, "ordered"))) attr(new, "ordered") <- TRUE
    map <- as.integer(match(seq_along(old), used))
    tbl$ptr <- wrap_gpu_call("group_trim", gpu_remap_codes(tbl$ptr, idx0[[k]], map))
    tbl$schema$factor_levels[[col]] <- new
  }

  # ungroup() sheds custom attributes / bare data.frame class like dplyr's
  # as_tibble(); group_by() then rebuilds the groups (no fixed_keys)
  group_by(ungroup(tbl), !!!rlang::syms(vars), .drop = .drop)
}

# =============================================================================
# Grouping metadata query family (Phase 9, task 1): group_data()/
# group_keys()/group_indices()/group_size()/n_groups() are all GPU-native.
# All but n_groups() share one C++ primitive (`gpu_group_info()`,
# src/ops_groupby.cpp) via `compute_group_data()` below; n_groups() (and the
# printed group count) goes through `count_groups()`, which needs no row
# indices (it falls back to `compute_group_data()` only for a `.drop = FALSE`
# factor key). `group_rows()` needs no method of its own:
# `dplyr::group_rows()` is a plain (non-generic) function defined as
# `group_data(.data)[[".rows"]]`, so it already works transparently once
# `group_data.tbl_gpu()` exists. Previously (Phase 2) all six were CPU
# fallbacks (see R/fallback.R's now-removed registrations); the iteration
# family (group_map()/group_modify()/group_nest()/nest_by(); group_split()
# is native, R/fallback.R; group_trim() is native, above)
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

  if (is_rowwise_gpu(materialized)) {
    return(rowwise_group_data(materialized))
  }

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
  group_col_idx0 <- group_col_index0(materialized)

  info <- wrap_gpu_call("group_data", gpu_group_info(materialized$ptr, group_col_idx0))

  keys_schema <- group_keys_schema(materialized)
  keys_df <- group_keys_frame(info$keys, keys_schema)

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

  fixed <- fixed_keys_of(materialized)
  if (!is.null(fixed)) {
    # slice(.preserve = TRUE): the fixed keys are the groups, observed or not
    gd <- fixed_group_rows(gd, fixed)
  } else if (!isTRUE(materialized$group_drop %||% TRUE)) {
    gd <- expand_dropped_groups(gd, group_names, keys_schema$factor_levels)
  }

  # dplyr's grouped_df group_data() carries `.drop` as an attribute
  attr(gd, ".drop") <- isTRUE(materialized$group_drop %||% TRUE)

  gd
}

# 0-based positions of `x`'s group columns in its schema.
# @keywords internal
group_col_index0 <- function(x) {
  as.integer(match(.subset2(x, "groups"), .subset2(x, "schema")$names) - 1L)
}

# The schema of `x`'s group-key table: its group columns, with their factor
# levels and time zones.
# @keywords internal
group_keys_schema <- function(x) {
  schema <- .subset2(x, "schema")
  group_names <- .subset2(x, "groups")
  schema_idx <- match(group_names, schema$names)
  keys_schema <- list(
    names = group_names,
    types = unname(schema$types[schema_idx])
  )
  if (!is.null(schema$factor_levels)) {
    fl <- schema$factor_levels[names(schema$factor_levels) %in% group_names]
    keys_schema$factor_levels <- if (length(fl) > 0) fl else NULL
  }
  keys_schema$tz <- tz_from_positions(
    group_names, keys_schema$types, tz_by_position(schema)[schema_idx]
  )
  keys_schema
}

# A distinct-keys GPU table (`gpu_group_info()$keys`, `gpu_group_keys()`) as a
# tibble.
# @keywords internal
group_keys_frame <- function(keys_ptr, keys_schema) {
  keys_tbl <- new_tbl_gpu(ptr = keys_ptr, schema = keys_schema, exec_mode = "eager")
  tibble::as_tibble(collect_frame(keys_tbl))
}

# The number of observed key combinations of `x` (eager, grouped, not
# rowwise): one GPU groupby, with no row-index upload, permutation download or
# `.drop = FALSE` level expansion (what count_groups() adds on top).
# @keywords internal
observed_group_count <- function(x) {
  wrap_gpu_call(
    "group_data",
    gpu_group_count(.subset2(x, "ptr"), group_col_index0(x))
  )
}

# The same number as observed_group_count(), counted with a hash set rather
# than a sorted groupby (about 50x faster on 5e6 string keys): row_slice_keys()
# only compares two counts.
# @keywords internal
distinct_group_count <- function(x) {
  wrap_gpu_call(
    "group_data",
    gpu_distinct_key_count(.subset2(x, "ptr"), group_col_index0(x))
  )
}

# `nrow(compute_group_data(.data))` without the group data, checked in the
# same order as compute_group_data():
# - fixed keys (filter()/slice() with `.preserve = TRUE`, filter() of a
#   `.drop = FALSE` table): compute_group_data() returns exactly the fixed keys
#   (fixed_group_rows()), so the count is their row count, with no GPU work;
# - `.drop = FALSE` with a factor key: unused level combinations are groups,
#   so the count needs the expanded group data;
# - otherwise the number of observed key combinations from one GPU groupby
#   (`gpu_group_count()`), with no row-index upload, n-row permutation
#   download or per-group R vectors.
# @keywords internal
count_groups <- function(.data) {
  # fixed keys are the groups (compute() copies them), so a lazy table's plan
  # is not run to count them
  fixed <- fixed_keys_of(.data)
  if (!is.null(fixed) && length(.subset2(.data, "groups")) > 0L && !is_rowwise_gpu(.data)) {
    return(nrow(fixed))
  }
  materialized <- if (has_pending_ops(.data)) compute(.data) else .data

  if (is_rowwise_gpu(materialized)) {
    return(dim(materialized)[[1L]])
  }
  group_names <- .subset2(materialized, "groups")
  if (length(group_names) == 0L) {
    return(1L)
  }
  fixed <- fixed_keys_of(materialized)
  if (!is.null(fixed)) {
    return(nrow(fixed))
  }
  schema <- .subset2(materialized, "schema")
  keeps_unused_levels <-
    !isTRUE(.subset2(materialized, "group_drop") %||% TRUE) &&
    any(group_names %in% names(schema$factor_levels))
  if (keeps_unused_levels) {
    return(nrow(compute_group_data(materialized)))
  }

  observed_group_count(materialized)
}

# The group keys `slice(.preserve = TRUE)` fixed on `x`, or NULL. Only trusted
# while they still describe the table's grouping, so a key table that outlived
# a change of groups can never be read.
# @keywords internal
fixed_keys_of <- function(x) {
  keys <- .subset2(x, "fixed_keys")
  if (is.null(keys) || !identical(names(keys), .subset2(x, "groups"))) {
    return(NULL)
  }
  keys
}

# Group data (keys + `.rows`) of `fixed` keys, with the rows of the observed
# groups `gd` (keys + `.rows`) put on the matching key; a key with no observed
# row gets `integer()`. NA keys match NA.
# @keywords internal
fixed_group_rows <- function(gd, fixed) {
  m <- vctrs::vec_match(fixed, gd[names(fixed)])
  rows <- unclass(gd$.rows)[m]
  rows[is.na(m)] <- list(integer())
  out <- fixed
  out$.rows <- vctrs::new_list_of(rows, ptype = integer())
  out
}

# `frame` (a data frame) as a grouped_df whose groups are exactly the `fixed`
# keys, empty ones included.
# @keywords internal
with_fixed_groups <- function(frame, fixed, drop = TRUE) {
  frame <- tibble::as_tibble(frame)
  observed <- dplyr::group_data(dplyr::grouped_df(frame, names(fixed)))
  gd <- tibble::as_tibble(fixed_group_rows(observed, fixed))
  attr(gd, ".drop") <- drop
  dplyr::new_grouped_df(frame, gd)
}

# Honor group_by(.drop = FALSE) (see group_by.tbl_gpu()'s own docs) with
# dplyr's recursive rule, column by column in grouping order: under each
# parent group a FACTOR column expands to all its declared levels (plus NA if
# NA is observed there) and any other column takes the values observed under
# that parent; under an EMPTY parent a non-factor column is a single NA while
# a factor column still expands all levels. `gd` is the observed
# keys + `.rows` (one row per observed group, ascending key order), so the
# result is in group order. A no-op without factor grouping columns.
# @keywords internal
expand_dropped_groups <- function(gd, group_names, factor_levels) {
  if (is.null(factor_levels) || !any(group_names %in% names(factor_levels))) {
    return(gd)
  }

  # Every partial key ("parent") is processed at once per column: `row` holds
  # the observed gd rows ordered by `pid`, the id of the parent they sit under.
  row <- seq_len(nrow(gd))
  pid <- rep.int(1L, length(row))
  np <- 1L
  keys <- list()
  for (nm in group_names) {
    col <- gd[[nm]]
    lv <- factor_levels[[nm]]
    if (!is.null(lv)) {
      # every level under each parent, plus NA where NA is observed
      nlev <- length(lv)
      code <- as.integer(col[row])
      na_code <- is.na(code)
      rk <- (pid - 1) * (nlev + 1) + ifelse(na_code, nlev + 1L, code)
      has_na <- logical(np)
      has_na[pid[na_code]] <- TRUE
      per <- nlev + has_na
      par <- rep.int(seq_len(np), per)
      j <- sequence(per)
      gid <- match(rk, (par - 1) * (nlev + 1) + j)
      j[j > nlev] <- NA_integer_
      piece <- factor_from_codes(seq_along(lv), lv)[j]
    } else {
      # the values observed under each parent (first-appearance = ascending
      # key order); a parent with no rows gets a single NA
      vals <- vctrs::vec_slice(col, row)
      g <- vctrs::vec_group_loc(vctrs::data_frame(p = pid, v = vals))
      gid_obs <- integer(length(row))
      gid_obs[unlist(g$loc)] <- rep.int(seq_len(nrow(g)), lengths(g$loc))
      empty <- which(!seq_len(np) %in% pid)
      all_pid <- c(g$key$p, empty)
      all_val <- vctrs::vec_c(
        g$key$v, vctrs::vec_slice(col, rep.int(NA_integer_, length(empty)))
      )
      ord <- order(all_pid)
      rank <- integer(length(ord))
      rank[ord] <- seq_along(ord)
      gid <- rank[gid_obs]
      par <- all_pid[ord]
      piece <- vctrs::vec_slice(all_val, ord)
    }
    keys <- lapply(keys, function(v) vctrs::vec_slice(v, par))
    keys[[nm]] <- piece
    o <- order(gid)
    row <- row[o]
    pid <- gid[o]
    np <- length(par)
  }

  # each leaf group holds at most one observed gd row
  first <- row[match(seq_len(np), pid)]
  rows_out <- unclass(gd$.rows)[first]
  rows_out[is.na(first)] <- list(integer())
  out <- tibble::as_tibble(keys)
  out$.rows <- vctrs::new_list_of(rows_out, ptype = integer())
  out
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
  gd <- compute_group_data(.data)
  # dplyr: an ungrouped plain data.frame answers with a bare data.frame (F31)
  if (isTRUE(.subset2(.data, "bare_df"))) {
    class(gd) <- "data.frame"
  }
  gd
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
  attr(gd, ".drop") <- NULL
  gd[set_diff(names(gd), ".rows")]
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
  count_groups(x)
}
