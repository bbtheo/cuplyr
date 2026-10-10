# Bind operations for tbl_gpu
# bind_rows: vertical concatenation (stacking rows)
# bind_cols: horizontal concatenation (adding columns)

# =============================================================================
# bind_cols - S3 generic that dispatches based on first argument
# =============================================================================

#' Bind multiple data frames/tables by column
#'
#' Combines objects horizontally by adding columns. For tbl_gpu objects,
#' operations are performed on the GPU. For other objects, delegates to dplyr.
#'
#' @param ... Objects to bind (tbl_gpu, data.frame, vectors, or lists of
#'   these, flattened recursively as in `dplyr::bind_cols()`). A vector
#'   becomes one column named by its argument name. `NULL`s are dropped.
#' @param .name_repair How to handle duplicate column names
#'
#' @details
#' If any input (including one inside a list) is a `tbl_gpu`, the result is a
#' `tbl_gpu`, wherever the `tbl_gpu` sits in the arguments. Call this function
#' (not `dplyr::bind_cols()` or `vctrs::vec_cbind()`) on a `tbl_gpu`: those
#' cannot be intercepted by S3 dispatch and raise an error for a `tbl_gpu`.
#'
#' @return Combined data frame or tbl_gpu
#' @export
bind_cols <- function(..., .name_repair = c("unique", "universal",
                                             "check_unique", "minimal")) {
  UseMethod("bind_cols")
}

#' @export
bind_cols.tbl_gpu <- function(..., .name_repair = c("unique", "universal",
                                                     "check_unique", "minimal")) {
  .name_repair <- match.arg(.name_repair)
  bind_cols_route(..., .name_repair = .name_repair)
}

#' @export
bind_cols.default <- function(..., .name_repair = c("unique", "universal",
                                                     "check_unique", "minimal")) {
  .name_repair <- match.arg(.name_repair)
  bind_cols_route(..., .name_repair = .name_repair)
}

# A tbl_gpu that is not the first argument (`NULL`, a tibble or a list comes
# first) dispatches to the default method, so both methods route here: any
# tbl_gpu among dplyr's flattened inputs sends the call to the GPU, anything
# else goes to dplyr unchanged (and so raises dplyr's own errors).
# The captured arguments are passed on to dplyr as a list, so a spliced
# `!!!f()` is evaluated once.
bind_cols_route <- function(..., .name_repair) {
  args <- rlang::list2(...)
  dots <- bind_cols_inputs(args)
  if (any(vapply(dots, is_tbl_gpu, logical(1)))) {
    return(bind_cols_gpu(dots, .name_repair = .name_repair))
  }
  dplyr::bind_cols(!!!args, .name_repair = .name_repair)
}


#' Bind multiple data frames/tables by row
#'
#' Combines objects vertically by stacking rows. For tbl_gpu objects,
#' operations are performed on the GPU. For other objects, delegates to dplyr.
#'
#' @param ... Objects to bind (tbl_gpu, data.frame, or lists of these,
#'   flattened as in `dplyr::bind_rows()`). `NULL`s are dropped.
#' @param .id Optional column name to identify source tables
#'
#' @details
#' If any input (including one inside a list) is a `tbl_gpu`, the result is a
#' `tbl_gpu`, wherever the `tbl_gpu` sits in the arguments. Call this function
#' (not `dplyr::bind_rows()`, `purrr::map_dfr()` or `vctrs::vec_rbind()`) on a
#' `tbl_gpu`: those cannot be intercepted by S3 dispatch and raise an error for
#' a `tbl_gpu`.
#'
#' @return Combined data frame or tbl_gpu
#' @export
bind_rows <- function(..., .id = NULL) {
  UseMethod("bind_rows")
}

#' @export
bind_rows.tbl_gpu <- function(..., .id = NULL) {
  bind_rows_route(..., .id = .id)
}

#' @export
bind_rows.default <- function(..., .id = NULL) {
  bind_rows_route(..., .id = .id)
}

#' Base `rbind()` / `cbind()` for GPU tables
#'
#' `rbind(gt, ...)` is [bind_rows()] (keeping the first table's groups) and
#' `cbind(gt, ...)` is [bind_cols()], so
#' the result is a `tbl_gpu` built on the GPU (base R would otherwise bind the
#' internal fields of the list). Base R picks the method from the arguments, so
#' this applies when a `tbl_gpu` comes first (or no argument is a plain
#' `data.frame`).
#'
#' @param ... `tbl_gpu`s, data frames and vectors, as for [bind_rows()] /
#'   [bind_cols()].
#' @param deparse.level Ignored.
#' @return A `tbl_gpu`.
#' @name tbl_gpu-rbind
#' @export
rbind.tbl_gpu <- function(..., deparse.level = 1) {
  out <- bind_rows(...)
  # Like `rbind()` of data frames (and dplyr's `bind_rows()`), the result keeps
  # the first input's grouping. (cuplyr's `bind_rows()` clears it.)
  first <- ..1
  if (is_tbl_gpu(first) && is_tbl_gpu(out)) {
    out$groups <- intersect(first$groups, out$schema$names)
    out$group_drop <- if (length(out$groups) > 0L) first$group_drop %||% TRUE else TRUE
  }
  out
}

#' @rdname tbl_gpu-rbind
#' @export
cbind.tbl_gpu <- function(..., deparse.level = 1) {
  bind_cols(...)
}

# See bind_cols_route(). After flattening, the inputs are checked the way
# dplyr::bind_rows() checks them, so a call dplyr rejects fails with dplyr's
# message on the GPU path too.
bind_rows_route <- function(..., .id) {
  args <- rlang::list2(...)
  dots <- bind_rows_inputs(args)
  is_gpu <- vapply(dots, is_tbl_gpu, logical(1))
  if (!any(is_gpu)) {
    return(dplyr::bind_rows(!!!args, .id = .id))
  }
  # Only the other inputs are checked: vctrs::vec_is() on a lazy tbl_gpu
  # would compute it, and bind_rows_gpu() computes it again.
  bindable <- is_gpu
  bindable[!is_gpu] <- vapply(dots[!is_gpu], bind_dataframe_ish, logical(1))
  # dplyr turns named non-table inputs into the columns of one tibble
  # (bind_rows(a = 1, b = 2)); with a tbl_gpu among them that tibble would
  # need a data frame column, which a tbl_gpu can't hold.
  if (rlang::is_named(dots) && !all(bindable)) {
    rlang::abort(
      c(
        "Can't bind named non-table inputs together with a tbl_gpu.",
        i = "dplyr packs them into data frame columns, which a tbl_gpu can't hold."
      ),
      class = "cuplyr_error_bind_unsupported"
    )
  }

  for (i in seq_along(dots)) {
    if (!bindable[[i]]) {
      rlang::abort(sprintf(
        "Argument %d must be a data frame or a named atomic vector.", i
      ))
    }
    if (!is_gpu[[i]] && !is.data.frame(dots[[i]])) {
      dots[[i]] <- bind_row_as_df(dots[[i]])
    }
  }

  if (!is.null(.id)) {
    # dplyr's own .id check (and message), with no inputs to bind
    dplyr::bind_rows(.id = .id)
    if (!rlang::is_named(dots)) {
      nms <- names(dots) %||% rep("", length(dots))
      unnamed <- which(is.na(nms) | nms == "")
      nms[unnamed] <- as.character(unnamed)
      names(dots) <- nms
    }
  } else {
    names(dots) <- NULL
  }

  bind_rows_gpu(dots, .id = .id)
}

# dplyr 1.2.1's argument flattening (`dplyr::bind_rows()` / `bind_cols()`
# and their `list_flatten()` helper), so the GPU path binds exactly the inputs
# dplyr would. Only vctrs lists are spliced (bare lists, or classes ending in
# "list"): a data frame, a tbl_gpu, a POSIXlt or an S3 container like
# `structure(list(...), class = "x")` stays one input. bind_rows() first
# unwraps a single bare-list argument, then splices unnamed lists one level;
# bind_cols() splices every list recursively. NULLs are dropped afterwards.
bind_rows_inputs <- function(dots) {
  if (length(dots) == 1L && rlang::is_bare_list(dots[[1L]])) {
    dots <- dots[[1L]]
  }
  dots <- bind_list_flatten(dots, fn = function(x) !rlang::is_named(x))
  dots[!vapply(dots, is.null, logical(1))]
}

bind_cols_inputs <- function(dots) {
  dots <- bind_list_flatten(dots, recursive = TRUE)
  dots[!vapply(dots, is.null, logical(1))]
}

# Splice the elements of `x` that are vctrs lists (and pass `fn`, if given)
# into `x`. A spliced list's own name is dropped; its elements keep theirs.
bind_list_flatten <- function(x, fn = NULL, recursive = FALSE) {
  x <- unclass(x)
  loc <- vapply(x, vctrs::obj_is_list, logical(1))
  if (!is.null(fn)) {
    loc[loc] <- vapply(x[loc], fn, logical(1))
  }
  nms <- names(x)
  if (!is.null(nms)) {
    nms[loc] <- ""
    names(x) <- nms
  }
  x[loc] <- lapply(x[loc], unclass)
  x[!loc] <- lapply(x[!loc], list)
  out <- vctrs::list_unchop(x, ptype = list())
  if (recursive && any(loc)) {
    out <- bind_list_flatten(out, fn = fn, recursive = TRUE)
  }
  out
}

# dplyr's `dataframe_ish()`: a data frame, or a named vector (one row).
bind_dataframe_ish <- function(x) {
  is.data.frame(x) || (vctrs::vec_is(x) && rlang::is_named(x))
}

# A named list or named atomic vector as the one-row data frame dplyr binds.
bind_row_as_df <- function(x) {
  if (vctrs::obj_is_list(x)) {
    if (any(vapply(x, is_tbl_gpu, logical(1)))) {
      rlang::abort(
        c(
          "Can't bind a named list of tbl_gpu as one row.",
          i = "dplyr packs such a list into data frame columns, which a tbl_gpu can't hold."
        ),
        class = "cuplyr_error_bind_unsupported"
      )
    }
    return(vctrs::data_frame(!!!x, .name_repair = "minimal"))
  }
  vctrs::vec_rbind(x)
}


# =============================================================================
# GPU implementations
# =============================================================================

#' GPU bind_cols implementation
#' @param dots Inputs as flattened by `bind_cols_inputs()`: tbl_gpu, data
#'   frames and vectors (a vector becomes one column, named by its element
#'   name, recycled to the table height like `vctrs::vec_cbind()` does)
#' @param .name_repair Name repair method
#' @return A tbl_gpu
#' @keywords internal
bind_cols_gpu <- function(dots, .name_repair = "unique") {
  if (length(dots) == 0) {
    stop("No tables provided to bind_cols", call. = FALSE)
  }

  if (length(dots) == 1 && is_tbl_gpu(dots[[1]])) {
    return(dots[[1]])
  }

  # dplyr::bind_cols() ignores the names of data frame inputs; a vector's
  # name is its column name.
  nms <- names(dots) %||% rep("", length(dots))
  nms[is.na(nms)] <- ""
  is_gpu <- vapply(dots, is_tbl_gpu, logical(1))

  # Materialize any lazy tables
  dots[is_gpu] <- lapply(dots[is_gpu], function(x) {
    if (identical(x$exec_mode, "lazy") && has_pending_ops(x)) {
      compute(x)
    } else {
      x
    }
  })
  n <- nrow(dots[[which(is_gpu)[[1L]]]])

  # Convert data frames and vectors to tbl_gpu. A vector is uploaded under a
  # placeholder name: its real name may be "" until repair_names() below.
  col_names <- vector("list", length(dots))
  dots <- lapply(seq_along(dots), function(i) {
    x <- dots[[i]]
    if (is_gpu[[i]]) {
      col_names[[i]] <<- x$schema$names
      x
    } else if (is.data.frame(x)) {
      col_names[[i]] <<- names(x)
      tbl_gpu(x, lazy = FALSE)
    } else if (vctrs::vec_is(x)) {
      col_names[[i]] <<- nms[[i]]
      col <- vctrs::vec_recycle(x, n, x_arg = nms[[i]])
      tbl_gpu(vctrs::new_data_frame(list(value = col), n = n), lazy = FALSE)
    } else {
      stop("All inputs must be tbl_gpu, data.frame or vectors", call. = FALSE)
    }
  })

  # Zero-column tables carry their row count on the R side (new_tbl_gpu()), so
  # cudf can't see it: binding only zero-column tables of one size is just
  # that size; mixing one that has rows with real columns is unsupported.
  no_cols <- vapply(dots, function(x) length(x$schema$names) == 0L, logical(1))
  if (all(no_cols)) {
    sizes <- unique(vapply(dots, function(x) x[["nrow"]] %||% NA_integer_, integer(1)))
    if (length(sizes) != 1L || is.na(sizes)) {
      stop("`bind_cols()` of tables with no columns needs one known, common ",
           "number of rows.", call. = FALSE)
    }
    return(new_tbl_gpu(
      ptr = dots[[1]]$ptr,
      schema = dots[[1]]$schema,
      groups = dots[[1]]$groups,
      exec_mode = "eager",
      nrow = sizes
    ))
  }
  for (x in dots[no_cols]) {
    check_zero_column_input(x, x$schema, "bind_cols")
  }

  # Collect column names and handle duplicates
  all_names <- unlist(col_names)
  repaired_names <- repair_names(all_names, .name_repair)

  # Collect types (unname to avoid issues)
  all_types <- unname(unlist(lapply(dots, function(x) x$schema$types)))

  # Factor levels are keyed by column name, so re-key them by position onto
  # the repaired names.
  all_levels <- unlist(lapply(dots, function(x) {
    lapply(x$schema$names, function(nm) x$schema$factor_levels[[nm]])
  }), recursive = FALSE)
  has_levels <- !vapply(all_levels, is.null, logical(1))
  factor_levels <- if (any(has_levels)) {
    rlang::set_names(all_levels[has_levels], repaired_names[has_levels])
  }

  # Collect pointers
  ptrs <- lapply(dots, function(x) x$ptr)

  # Call C++ function
  new_ptr <- gpu_bind_cols_impl(ptrs)

  # Preserve groups (and their `.drop`) from first table. dplyr regroups the
  # output by name, so a group column renamed by repair (`bind_cols(g, g)`)
  # no longer groups it.
  groups <- intersect(dots[[1]]$groups, repaired_names)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = repaired_names, types = all_types,
                  factor_levels = factor_levels),
    groups = groups,
    group_drop = if (length(groups) > 0L) dots[[1]]$group_drop %||% TRUE else TRUE,
    exec_mode = "eager"
  )
}

#' GPU bind_rows implementation
#' @param dots Inputs as flattened and checked by `bind_rows_route()`: tbl_gpu
#'   and data frames, named for every input when `.id` is given
#' @param .id Optional column name to identify source
#' @return A tbl_gpu
#' @keywords internal
bind_rows_gpu <- function(dots, .id = NULL) {
  if (length(dots) == 0) {
    stop("No tables provided to bind_rows", call. = FALSE)
  }

  source_names <- names(dots) %||% as.character(seq_along(dots))

  # Convert any data.frames to tbl_gpu
  dots <- lapply(dots, function(x) {
    if (is.data.frame(x) && !is_tbl_gpu(x)) {
      tbl_gpu(x, lazy = FALSE)
    } else if (!is_tbl_gpu(x)) {
      stop("All inputs must be tbl_gpu or data.frame", call. = FALSE)
    } else {
      x
    }
  })

  # Materialize any lazy tables
  dots <- lapply(dots, function(x) {
    if (identical(x$exec_mode, "lazy") && has_pending_ops(x)) {
      compute(x)
    } else {
      x
    }
  })

  if (length(dots) == 1 && is.null(.id)) {
    # Single table, no .id - just return with groups cleared
    result <- dots[[1]]
    result$groups <- character()
    return(result)
  }

  # Zero-column tables carry their row count on the R side (new_tbl_gpu()),
  # which the concatenation can't see. All zero-column: the counts add up.
  # Mixed with real columns, one that has rows is unsupported (it would need
  # all-NA padding rows), but an empty one adds nothing.
  no_cols <- vapply(dots, function(x) length(x$schema$names) == 0L, logical(1))
  if (all(no_cols) && is.null(.id)) {
    sizes <- vapply(dots, function(x) x[["nrow"]] %||% NA_integer_, integer(1))
    if (anyNA(sizes)) {
      stop("`bind_rows()` of tables with no columns needs known row counts.",
           call. = FALSE)
    }
    return(new_tbl_gpu(
      ptr = dots[[1]]$ptr,
      schema = dots[[1]]$schema,
      exec_mode = "eager",
      nrow = sum(sizes)
    ))
  }
  for (x in dots[no_cols]) {
    check_zero_column_input(x, x$schema, "bind_rows")
  }

  # Compute unified schema
  unified <- compute_unified_schema(dots)

  # Align each table to unified schema
  aligned <- lapply(dots, function(tbl) {
    align_to_schema(tbl, unified)
  })

  # Collect pointers
  ptrs <- lapply(aligned, function(x) x$ptr)

  # Concatenate
  new_ptr <- gpu_bind_rows_aligned(ptrs)

  # Build result
  result <- new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = unified$names, types = unified$types,
                  factor_levels = unified$factor_levels),
    groups = character(),
    exec_mode = "eager"
  )

  # Add .id column if requested
  if (!is.null(.id)) {
    result <- add_id_column(result, .id, dots, source_names)
  }

  result
}

# =============================================================================
# Helper functions
# =============================================================================

#' Repair duplicate column names
#' @param names Character vector of column names
#' @param method Repair method
#' @return Character vector with repaired names
#' @keywords internal
repair_names <- function(names, method = "unique") {
  if (method == "minimal") {
    return(names)
  }

  if (method == "check_unique") {
    if (anyDuplicated(names)) {
      dups <- names[duplicated(names)]
      stop("Column names must be unique. Duplicates: ",
           paste(unique(dups), collapse = ", "), call. = FALSE)
    }
    return(names)
  }

  if (method == "unique" || method == "universal") {
    # Use vctrs if available for better name repair
    if (requireNamespace("vctrs", quietly = TRUE)) {
      return(vctrs::vec_as_names(names, repair = method))
    }
    # Fallback: add numeric suffixes to duplicates
    make.unique(names, sep = "_")
  } else {
    names
  }
}

#' Resolve the common GPU type for one column across a bind/set operation
#' (Phase 11 L4)
#'
#' The type-compatibility ladder extracted out of `check_set_op_compatible()`
#' (R/sets.R, Phase 8) so `bind_rows()`'s `compute_unified_schema()` and the
#' set-op verbs share EXACTLY one promotion rule, instead of `bind_rows()`
#' having its own looser one (the old `promote_types()`'s unconditional
#' "STRING is widest" rule, which silently coerced e.g. a numeric column
#' against a character column of the same name -- a real bug: real dplyr's
#' own `vctrs::vec_ptype2()`-backed `bind_rows()` has never allowed this,
#' it's a hard `vctrs_error_incompatible_type`, see
#' `abort_bind_rows_incompatible()` below).
#'
#' Four cases, all pinned via oracle-verified `dplyr`/`vctrs` behavior:
#' \itemize{
#'   \item Both factor (`DICTIONARY32`): union of both level sets (or a
#'     no-op if identical) -- `vctrs::vec_c(factor, factor)`'s contract.
#'   \item Exactly one factor, the other `STRING`: decodes to `STRING`
#'     (dplyr's factor + character -> character contract). Exactly one
#'     factor, the other anything else (numeric, timestamp, ...): a hard
#'     incompatibility (`$ok = FALSE`) -- real `vctrs::vec_ptype2()` refuses
#'     this too.
#'   \item Identical types: a no-op.
#'   \item Both in the numeric ladder (`BOOL8 < INT32 < INT64 < FLOAT64`) or
#'     both in the timestamp ladder (`TIMESTAMP_DAYS < ... <
#'     TIMESTAMP_NANOSECONDS`): promotes to the wider member. The timestamp
#'     cast (Date <-> POSIXct, etc.) is genuinely native as of Phase 11 L3
#'     (`gpu_cast_column()`, `src/ops_bind.cpp`, gained the 5 TIMESTAMP_*
#'     targets) -- this ladder used to require `check_set_op_compatible()`'s
#'     own caller to set a `$needs_fallback` flag for exactly this case
#'     (no native `TIMESTAMP_DAYS` -> `TIMESTAMP_MICROSECONDS` cast existed
#'     yet); that flag is retired, this ladder is now unconditionally native
#'     for both callers.
#'   \item Anything else (e.g. numeric vs. character, factor vs. numeric,
#'     numeric vs. timestamp): `$ok = FALSE` -- the caller reports its own
#'     verb-specific incompatibility error (sets.R's `"`x` and `y` are not
#'     compatible."` bulleted form vs. bind.R's `vctrs`-style `"Can't
#'     combine ...`"` form, see `abort_bind_rows_incompatible()`).
#' }
#'
#' @param tx,ty GPU type strings for the two occurrences of this column
#'   (`unname()`d already by the caller)
#' @param fx,fy Logical: is this occurrence `DICTIONARY32` (a factor)?
#' @param lvls_x,lvls_y Level vectors for `tx`/`ty` when `fx`/`fy` is `TRUE`
#'   (`NULL` otherwise)
#' @return A list with `$ok` (logical) and, when `TRUE`, `$type` (promoted
#'   type string) and `$factor_levels` (level vector, or `NULL` when the
#'   result isn't `DICTIONARY32`)
#' @keywords internal
resolve_common_col_type <- function(tx, ty, fx = FALSE, fy = FALSE,
                                     lvls_x = NULL, lvls_y = NULL) {
  numeric_ladder <- c("BOOL8", "INT32", "INT64", "FLOAT64")
  timestamp_ladder <- c("TIMESTAMP_DAYS", "TIMESTAMP_SECONDS",
                         "TIMESTAMP_MILLISECONDS", "TIMESTAMP_MICROSECONDS",
                         "TIMESTAMP_NANOSECONDS")

  if (fx && fy) {
    lvls <- if (identical(lvls_x, lvls_y)) lvls_x else union(lvls_x, lvls_y)
    return(list(ok = TRUE, type = "DICTIONARY32", factor_levels = lvls))
  }

  if (xor(fx, fy)) {
    other_type <- if (fx) ty else tx
    if (identical(other_type, "STRING")) {
      return(list(ok = TRUE, type = "STRING", factor_levels = NULL))
    }
    return(list(ok = FALSE))
  }

  if (identical(tx, ty)) {
    return(list(ok = TRUE, type = tx, factor_levels = NULL))
  }

  if (tx %in% numeric_ladder && ty %in% numeric_ladder) {
    idx <- max(match(tx, numeric_ladder), match(ty, numeric_ladder))
    return(list(ok = TRUE, type = numeric_ladder[idx], factor_levels = NULL))
  }

  if (tx %in% timestamp_ladder && ty %in% timestamp_ladder) {
    idx <- max(match(tx, timestamp_ladder), match(ty, timestamp_ladder))
    return(list(ok = TRUE, type = timestamp_ladder[idx], factor_levels = NULL))
  }

  list(ok = FALSE)
}

#' Friendly, `vctrs`-bracket-style type label for a `bind_rows()`
#' incompatible-type error (Phase 11 L4)
#'
#' Mirrors real dplyr/vctrs' own `<double>`/`<character>`/`<date>`/
#' `<datetime<UTC>>` labels (verified empirically: `bind_rows(data.frame(x =
#' 1), data.frame(x = as.Date("2020-01-01")))` -> `` Can't combine `..1$x`
#' <double> and `..2$x` <date>. ``). The one label that can't be
#' byte-matched is a factor's: real `vctrs::vec_ptype_full()` embeds a
#' levels-derived hash (`<factor<4d52a>>`) this package has no equivalent
#' for -- `set_op_type_label()` (R/sets.R) documents the identical
#' limitation for its own, differently-formatted set-op error text.
#' @keywords internal
bind_rows_type_label <- function(type, is_factor = FALSE) {
  if (isTRUE(is_factor)) {
    return("factor<...>")
  }
  switch(type,
    BOOL8 = "logical",
    INT32 = "integer",
    INT64 = "integer64",
    FLOAT64 = "double",
    STRING = "character",
    TIMESTAMP_DAYS = "date",
    TIMESTAMP_SECONDS = ,
    TIMESTAMP_MILLISECONDS = ,
    TIMESTAMP_MICROSECONDS = ,
    TIMESTAMP_NANOSECONDS = "datetime<UTC>",
    type
  )
}

#' Raise dplyr's own `bind_rows()` incompatible-type error (Phase 11 L4)
#'
#' Reproduces `vctrs`' exact message shape, verified empirically against
#' dplyr 1.2.1/vctrs 0.7.3 across 2-, 3-, and 4-table `bind_rows()` calls,
#' named and unnamed dots, and a table where the conflicting column is
#' entirely absent from table 1: the LEFT-hand label is ALWAYS the literal
#' `..1` (dplyr's own dots are always relabeled `..1`/`..2`/... positionally
#' for this error regardless of the caller's actual names, and -- a genuine
#' vctrs quirk -- the accumulator side of a >2-way reduce keeps citing `..1`
#' even when the running common type was actually last updated by some
#' later table), and the RIGHT-hand label is the actual 1-based position of
#' the table whose column first failed to fit the type accumulated so far.
#' `compute_unified_schema()`'s own left-to-right table loop naturally
#' produces this: it only ever calls this function the first time some
#' table's column doesn't fit the running `name_types[[nm]]`, and never
#' updates a "which table established this" tracker -- so the constant `1L`
#' below isn't a simplification, it's the literal, verified rule.
#'
#' @param other_idx 1-based index (in the original bind_rows() dots) of the
#'   table whose column conflicted with the running accumulated type
#' @param col_name The conflicting column's name
#' @param type1,type2 GPU type strings: the accumulated type so far, and
#'   `other_idx`'s own type for this column
#' @param is_factor1,is_factor2 Logical: is each side `DICTIONARY32`?
#' @keywords internal
abort_bind_rows_incompatible <- function(other_idx, col_name, type1, type2,
                                          is_factor1 = FALSE, is_factor2 = FALSE) {
  label1 <- bind_rows_type_label(type1, is_factor1)
  label2 <- bind_rows_type_label(type2, is_factor2)
  stop(sprintf("Can't combine `..1$%s` <%s> and `..%d$%s` <%s>.",
               col_name, label1, other_idx, col_name, label2),
       call. = FALSE)
}

#' Compute unified schema from multiple tables
#'
#' Phase 11 L4: shares `resolve_common_col_type()`'s type ladder with the
#' set-op verbs (R/sets.R) -- the old `promote_types()`'s unconditional
#' "STRING is widest" rule is RETIRED; a numeric-vs-character (or any other
#' non-ladder) mismatch is now a hard error matching real dplyr/vctrs'
#' `vctrs_error_incompatible_type` text (`abort_bind_rows_incompatible()`),
#' not a silent STRING coercion.
#'
#' Factor columns (Phase 11 L2, Bug 2/3, unchanged by L4): a column present
#' as `DICTIONARY32` in every table that has it unifies to `DICTIONARY32`
#' with `factor_levels` set to the LEVEL UNION across every occurrence
#' (`base::union()`, first-occurrence order -- verified empirically against
#' `vctrs::vec_c(factor("a"), factor("b"))` -> levels `c("a", "b")`). A
#' column that's `DICTIONARY32` in some tables and `STRING` in at least one
#' other unifies to plain `STRING` with no `factor_levels` entry at all
#' (dplyr's own factor + character -> character contract) --
#' `align_to_schema()` decodes each factor occurrence to its labels via
#' `gpu_decode_factor()` (`cast_column()`, below). A factor column against
#' any OTHER type (numeric, timestamp, ...) is now a hard error too (real
#' dplyr's own behavior -- verified empirically), where it used to silently
#' fall through `promote_types()`'s STRING-is-widest rule.
#'
#' @param tables List of tbl_gpu objects
#' @return List with `names`, `types`, and `factor_levels` (`NULL` if no
#'   column in the union ends up `DICTIONARY32`) for the unified schema
#' @keywords internal
compute_unified_schema <- function(tables) {
  # Union of all column names (preserving order from first occurrence)
  all_names <- character()
  name_types <- list()
  name_factor_levels <- list()   # nm -> current level vector (only set while DICTIONARY32)

  for (t in seq_along(tables)) {
    tbl <- tables[[t]]
    tbl_factor_levels <- tbl$schema$factor_levels

    for (i in seq_along(tbl$schema$names)) {
      nm <- tbl$schema$names[i]
      # unname(): `tbl$schema$types` is a NAMED vector (tbl_gpu() builds it
      # via vapply()) -- identical(ty, "DICTIONARY32") would otherwise
      # compare a named single-element vector against an unnamed literal
      # and always be FALSE (identical() treats "names" as significant).
      ty <- unname(tbl$schema$types[i])
      is_factor <- identical(ty, "DICTIONARY32")
      lvls <- if (is_factor) tbl_factor_levels[[nm]] else NULL

      if (!(nm %in% all_names)) {
        all_names <- c(all_names, nm)
        name_types[[nm]] <- ty
        name_factor_levels[[nm]] <- lvls
        next
      }

      existing_type <- name_types[[nm]]
      existing_is_factor <- identical(existing_type, "DICTIONARY32")
      existing_lvls <- name_factor_levels[[nm]]

      res <- resolve_common_col_type(existing_type, ty, existing_is_factor, is_factor,
                                      existing_lvls, lvls)
      if (!isTRUE(res$ok)) {
        abort_bind_rows_incompatible(t, nm, existing_type, ty, existing_is_factor, is_factor)
      }
      name_types[[nm]] <- res$type
      name_factor_levels[[nm]] <- res$factor_levels
    }
  }

  unified_types <- vapply(all_names, function(nm) name_types[[nm]], character(1))

  factor_levels <- name_factor_levels[!vapply(name_factor_levels, is.null, logical(1))]

  list(
    names = all_names,
    types = unname(unified_types),
    factor_levels = if (length(factor_levels) > 0) factor_levels else NULL
  )
}

#' Do a table's factor columns already carry the target's exact level sets?
#'
#' Helper for `align_to_schema()`'s already-aligned fast path: `identical()`
#' on `names`/`types` alone isn't enough for a `DICTIONARY32` column -- two
#' tables can agree on the type string while their actual level VECTORS
#' differ (Bug 2, Phase 11 L2), which still needs a code remap even though
#' no type-string cast is triggered.
#' @keywords internal
factor_levels_match_target <- function(source_factor_levels, target_factor_levels,
                                       names, types) {
  factor_idx <- which(types == "DICTIONARY32")
  for (i in factor_idx) {
    nm <- names[i]
    if (!identical(source_factor_levels[[nm]], target_factor_levels[[nm]])) {
      return(FALSE)
    }
  }
  TRUE
}

#' Remap a factor column's physical codes onto a different level ordering
#'
#' Used by `align_to_schema()` when unifying two `DICTIONARY32` columns
#' under the same name whose level VECTORS differ (Bug 2, Phase 11 L2):
#' dplyr's own `bind_rows()`/`vctrs::vec_c()` contract for two factors is a
#' LEVEL UNION, not a raw code concatenation -- naively concatenating this
#' table's codes as-is (the pre-fix behavior) would silently misinterpret
#' them once the union's code assignment differs from this table's own.
#' `match(current_levels, target_levels)` gives, for each of this table's
#' OLD (1-based) levels in order, its NEW 1-based code in `target_levels`
#' (always found -- `target_levels` is always a superset, see
#' `compute_unified_schema()`); `gpu_remap_codes()` (`src/ops_bind.cpp`)
#' gathers using the column's own (null-safe) codes as the index.
#' @param tbl A tbl_gpu object
#' @param col_name Name of the factor column to remap
#' @param current_levels This table's own level vector for `col_name`
#' @param target_levels The unified level vector to remap onto
#' @return A tbl_gpu with `col_name`'s codes remapped (schema/types/groups
#'   otherwise unchanged)
#' @keywords internal
remap_factor_column <- function(tbl, col_name, current_levels, target_levels) {
  col_idx <- match(col_name, tbl$schema$names) - 1L
  map <- as.integer(match(current_levels, target_levels))

  new_ptr <- gpu_remap_codes(tbl$ptr, col_idx, map)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = tbl$schema$names, types = tbl$schema$types,
                  factor_levels = tbl$schema$factor_levels),
    groups = tbl$groups,
    group_drop = tbl$group_drop %||% TRUE,
    exec_mode = "eager"
  )
}

#' Align a table to a target schema
#' @param tbl A tbl_gpu object
#' @param target_schema List with `names`, `types`, and optionally
#'   `factor_levels` (the unified level set for any `DICTIONARY32` column,
#'   see `compute_unified_schema()`)
#' @return A tbl_gpu aligned to the target schema
#' @keywords internal
align_to_schema <- function(tbl, target_schema) {
  current_names <- tbl$schema$names
  current_types <- unname(tbl$schema$types)  # Remove names for comparison
  target_names <- target_schema$names
  target_types <- unname(target_schema$types)
  target_factor_levels <- target_schema$factor_levels

  # Captured ONCE, up front: `add_null_columns()`/`remap_factor_column()`/
  # `cast_column()` below each return a *new* tbl_gpu, and
  # `add_null_columns()` in particular rebuilds its own schema without
  # necessarily preserving every existing column's `factor_levels` entry --
  # reading from this snapshot (rather than the evolving `tbl$schema
  # $factor_levels`) avoids losing a column's levels partway through this
  # function's own multi-step rewrite.
  source_factor_levels <- tbl$schema$factor_levels

  # Check if already aligned (common case for same-schema tables) --
  # names/types identical AND (for any DICTIONARY32 column) this table's
  # own levels already match the target's exactly.
  if (identical(current_names, target_names) &&
      identical(current_types, target_types) &&
      factor_levels_match_target(source_factor_levels, target_factor_levels,
                                 target_names, target_types)) {
    return(tbl)
  }

  nrows <- nrow(tbl)

  # Identify missing columns
  missing_cols <- setdiff(target_names, current_names)

  # If there are missing columns, add them
  if (length(missing_cols) > 0) {
    missing_types <- target_types[match(missing_cols, target_names)]
    tbl <- add_null_columns(tbl, missing_cols, missing_types)
    current_names <- tbl$schema$names
    current_types <- unname(tbl$schema$types)
  }

  # Now handle type coercion (and factor level remapping) for each column
  for (i in seq_along(target_names)) {
    target_name <- target_names[i]
    target_type <- target_types[i]

    if (identical(target_type, "DICTIONARY32")) {
      # Same LOGICAL type doesn't mean the physical codes already agree:
      # remap unless this table's own levels already match the union (or
      # this is an all-null column just added above, which has no levels
      # of its own to remap -- every row is null regardless).
      current_levels <- source_factor_levels[[target_name]]
      col_target_levels <- target_factor_levels[[target_name]]
      if (!is.null(current_levels) && !identical(current_levels, col_target_levels)) {
        tbl <- remap_factor_column(tbl, target_name, current_levels, col_target_levels)
      }
      next
    }

    current_idx <- match(target_name, current_names)
    current_type <- current_types[current_idx]

    if (!identical(current_type, target_type)) {
      # Need to cast this column
      source_levels <- source_factor_levels[[target_name]]
      tbl <- cast_column(tbl, target_name, target_type, source_levels = source_levels)
      current_types <- unname(tbl$schema$types)
    }
  }

  # Reorder columns to match target
  col_indices <- match(target_names, tbl$schema$names) - 1L
  new_ptr <- gpu_select(tbl$ptr, col_indices)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = target_names, types = target_types,
                  factor_levels = target_factor_levels),
    groups = character(),
    exec_mode = "eager"
  )
}

#' Add null columns to a table
#' @param tbl A tbl_gpu object
#' @param col_names Names of columns to add
#' @param col_types Types of columns to add
#' @return A tbl_gpu with additional null columns
#' @keywords internal
add_null_columns <- function(tbl, col_names, col_types) {
  nrows <- nrow(tbl)

  # Create null columns and bind them. `gpu_physical_type()` (R/utils.R):
  # `gpu_make_null_column()` is a PHYSICAL-column C++ primitive with no
  # DICTIONARY32 case (a factor's actual GPU column is INT32) -- passing
  # the logical type string straight through would error.
  null_tbls <- lapply(seq_along(col_names), function(i) {
    null_ptr <- gpu_make_null_column(nrows, gpu_physical_type(col_types[i]))
    new_tbl_gpu(
      ptr = null_ptr,
      schema = list(names = col_names[i], types = col_types[i]),
      groups = character(),
      exec_mode = "eager"
    )
  })

  # Bind null columns to original table
  all_tbls <- c(list(tbl), null_tbls)
  ptrs <- lapply(all_tbls, function(x) x$ptr)

  new_ptr <- gpu_bind_cols_impl(ptrs)

  new_names <- c(tbl$schema$names, col_names)
  new_types <- c(tbl$schema$types, col_types)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = new_names, types = new_types,
                  factor_levels = tbl$schema$factor_levels),
    groups = character(),
    exec_mode = "eager"
  )
}

#' Cast a column to a different type
#' @param tbl A tbl_gpu object
#' @param col_name Name of column to cast
#' @param target_type Target type string
#' @param source_levels For a `DICTIONARY32` source column being cast to
#'   `STRING` (dplyr's factor + character -> character contract, Bug 3 of
#'   Phase 11 L2): this column's OWN level vector, decoded via
#'   `gpu_decode_factor()` instead of `gpu_cast_to_string()`'s numeric-only
#'   path (which only ever sees the PHYSICAL INT32 column and would
#'   silently emit its raw codes as strings, e.g. `"2"` instead of `"m"`).
#'   `NULL`/unused for every other source type.
#' @return A tbl_gpu with the column cast to the target type
#' @keywords internal
cast_column <- function(tbl, col_name, target_type, source_levels = NULL) {
  col_idx <- match(col_name, tbl$schema$names) - 1L

  if (is.na(col_idx) || col_idx < 0) {
    stop("Column not found: ", col_name, call. = FALSE)
  }

  new_ptr <- if (identical(target_type, "STRING")) {
    current_type <- tbl$schema$types[[col_idx + 1L]]
    if (identical(current_type, "DICTIONARY32")) {
      if (is.null(source_levels)) {
        stop("bind_rows(): cannot cast factor column '", col_name,
             "' to STRING -- its levels are unknown to cast_column().",
             call. = FALSE)
      }
      gpu_decode_factor(tbl$ptr, col_idx, source_levels)
    } else {
      gpu_cast_to_string(tbl$ptr, col_idx)
    }
  } else {
    gpu_cast_column(tbl$ptr, col_idx, target_type)
  }

  new_types <- tbl$schema$types
  new_types[col_idx + 1L] <- target_type

  new_factor_levels <- tbl$schema$factor_levels
  if (!is.null(new_factor_levels) && !is.null(new_factor_levels[[col_name]])) {
    new_factor_levels[[col_name]] <- NULL  # no longer a factor after this cast
    if (length(new_factor_levels) == 0) new_factor_levels <- NULL
  }

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = tbl$schema$names, types = new_types,
                  factor_levels = new_factor_levels),
    groups = tbl$groups,
    group_drop = tbl$group_drop %||% TRUE,
    exec_mode = "eager"
  )
}

#' Add .id column to identify source tables
#' @param result The combined tbl_gpu
#' @param id_col_name Name for the .id column
#' @param original_tables List of original tables (for row counts)
#' @param source_names Names/identifiers for each source
#' @return A tbl_gpu with .id column prepended
#' @keywords internal
add_id_column <- function(result, id_col_name, original_tables, source_names) {
  # Create character vector with source identifiers
  row_counts <- vapply(original_tables, nrow, integer(1))
  id_values <- rep(source_names, times = row_counts)

  # Create data frame with .id column and transfer to GPU
  id_df <- data.frame(x = id_values, stringsAsFactors = FALSE)
  names(id_df) <- id_col_name
  id_tbl <- tbl_gpu(id_df, lazy = FALSE)

  # Bind .id column at front
  bind_cols(id_tbl, result, .name_repair = "minimal")
}
