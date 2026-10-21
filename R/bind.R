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
  .name_repair <- match.arg(.name_repair)
  bind_cols_route(..., .name_repair = .name_repair, call = rlang::current_env())
}

# Like dplyr's own, `bind_cols()`/`bind_rows()` are plain functions, not S3
# generics: UseMethod() would force the first `...` element for dispatch and
# so evaluate a spliced `!!!list` as a plain `!`. Any tbl_gpu among dplyr's
# flattened inputs sends the call to the GPU, anything else goes to dplyr
# unchanged (and so raises dplyr's own errors). The captured arguments are
# passed on to dplyr as a list, so a spliced `!!!f()` is evaluated once.
# `call` is the user's call, for errors raised on the GPU path.
bind_cols_route <- function(..., .name_repair, call) {
  args <- rlang::list2(...)
  dots <- bind_cols_inputs(args)
  if (any(vapply(dots, is_tbl_gpu, logical(1)))) {
    if (bind_first_is_rowwise(dots)) {
      return(bind_rowwise_first("bind_cols", dots, .name_repair = .name_repair))
    }
    return(bind_cols_gpu(dots, .name_repair = .name_repair, call = call))
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
  bind_rows_route(..., .id = .id, call = rlang::current_env())
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
  bind_rows(...)
}

#' @rdname tbl_gpu-rbind
#' @export
cbind.tbl_gpu <- function(..., deparse.level = 1) {
  bind_cols(...)
}

# See bind_cols_route() (`call` is the user's call). After flattening, the
# inputs are checked the way dplyr::bind_rows() checks them, so a call dplyr
# rejects fails with dplyr's message on the GPU path too.
bind_rows_route <- function(..., .id, call) {
  args <- rlang::list2(...)
  dots <- bind_rows_inputs(args)
  is_gpu <- vapply(dots, is_tbl_gpu, logical(1))
  if (!any(is_gpu)) {
    return(dplyr::bind_rows(!!!args, .id = .id))
  }
  if (bind_first_is_rowwise(dots)) {
    return(bind_rowwise_first("bind_rows", dots, .id = .id))
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
      class = "cuplyr_error_bind_unsupported",
      call = call
    )
  }

  for (i in seq_along(dots)) {
    if (!bindable[[i]]) {
      rlang::abort(sprintf(
        "Argument %d must be a data frame or a named atomic vector.", i
      ), call = call)
    }
    if (!is_gpu[[i]] && !is.data.frame(dots[[i]])) {
      dots[[i]] <- bind_row_as_df(dots[[i]])
    }
    dots[[i]] <- bind_rows_repair_names(dots[[i]], call)
  }

  if (!is.null(.id)) {
    # dplyr's own .id check and message, re-raised from the user's call
    tryCatch(
      dplyr::bind_rows(.id = .id),
      error = function(e) rlang::abort(conditionMessage(e), call = call)
    )
    if (!rlang::is_named(dots)) {
      nms <- names(dots) %||% rep("", length(dots))
      unnamed <- which(is.na(nms) | nms == "")
      nms[unnamed] <- as.character(unnamed)
      names(dots) <- nms
    }
  } else {
    names(dots) <- NULL
  }

  bind_rows_gpu(dots, .id = .id, call = call)
}

# vctrs (under dplyr's bind_rows()) repairs an input's duplicated names
# ("unique", with its "New names:" message) before matching columns by name;
# a tbl_gpu gets the same names through a positional rename.
# @keywords internal
bind_rows_repair_names <- function(d, call) {
  nms <- if (is_tbl_gpu(d)) current_schema(d)$names else names(d)
  if (!anyDuplicated(nms)) {
    return(d)
  }
  new <- vctrs::vec_as_names(nms, repair = "unique", call = call)
  if (!is_tbl_gpu(d)) {
    names(d) <- new
    return(d)
  }
  changed <- which(nms != new)
  push_op(d, ast_rename(input_node(d), nms[changed], new[changed], positions = changed))
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
#'
#' Sizes follow vctrs' rules (`vctrs::vec_size_common()`): inputs of size 1
#' recycle, every other size must match, and an input with no columns
#' contributes its size only. A lazy input is computed once, so its size is the
#' post-op row count.
#' @param dots Inputs as flattened by `bind_cols_inputs()`: tbl_gpu, data
#'   frames and vectors (a vector becomes one column, named by its element
#'   name)
#' @param .name_repair Name repair method
#' @param call The user's call, for errors
#' @return A tbl_gpu
#' @keywords internal
bind_cols_gpu <- function(dots, .name_repair = "unique", call = rlang::caller_env()) {
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

  # Only the non-tbl_gpu inputs are checked (vec_is() on a tbl_gpu could
  # compute it).
  ok <- is_gpu
  ok[!is_gpu] <- vapply(dots[!is_gpu], function(x) {
    is.data.frame(x) || vctrs::vec_is(x)
  }, logical(1))
  if (!all(ok)) {
    rlang::abort(sprintf(
      "Argument %d must be a data frame or a vector.", which(!ok)[[1L]]
    ), call = call)
  }

  no_cols <- vapply(dots, function(x) {
    if (is_tbl_gpu(x)) length(x$schema$names) == 0L
    else is.data.frame(x) && ncol(x) == 0L
  }, logical(1))
  sizes <- vapply(dots, function(x) {
    if (is_tbl_gpu(x) && length(x$schema$names) == 0L) {
      as.integer(x[["nrow"]] %||% NA_integer_)
    } else if (is_tbl_gpu(x) || is.data.frame(x)) {
      as.integer(nrow(x))
    } else {
      as.integer(vctrs::vec_size(x))
    }
  }, integer(1))
  if (anyNA(sizes)) {
    stop("`bind_cols()` of tables with no columns needs a known number of rows.",
         call. = FALSE)
  }

  # vctrs' own size rule and message, with its argument labels: `..N`, or the
  # name for a vector (dplyr clears the names of data frame inputs first).
  nms[is_gpu | vapply(dots, is.data.frame, logical(1))] <- ""
  labels <- ifelse(nms != "", nms, paste0("..", seq_along(dots)))
  proxies <- lapply(sizes, function(s) vctrs::new_data_frame(n = s))
  names(proxies) <- labels
  n <- vctrs::vec_size_common(!!!proxies, .call = call)

  first <- dots[[1L]]
  keep <- which(!no_cols)
  if (length(keep) == 0L) {
    # Zero-column inputs only: the result is just the common size.
    out <- tbl_gpu(tibble::tibble(.rows = n), lazy = FALSE)
    return(if (bind_has_meta(first)) carry_frame_meta(out, first) else out)
  }

  # Convert data frames and vectors to tbl_gpu and recycle size-1 inputs.
  col_names <- vector("list", length(keep))
  pieces <- vector("list", length(keep))
  for (j in seq_along(keep)) {
    i <- keep[[j]]
    x <- dots[[i]]
    if (is_gpu[[i]]) {
      if (sizes[[i]] != n) {
        x <- dplyr_row_slice.tbl_gpu(x, rep(1L, n))
      }
      col_names[[j]] <- x$schema$names
    } else if (is.data.frame(x)) {
      if (sizes[[i]] != n) {
        x <- vctrs::vec_recycle(x, n)
      }
      col_names[[j]] <- names(x)
      x <- upload_other_side(x, lazy = FALSE)
    } else {
      col_names[[j]] <- nms[[i]]
      col <- vctrs::vec_recycle(x, n, x_arg = nms[[i]])
      x <- tbl_gpu(vctrs::new_data_frame(list(value = col), n = n), lazy = FALSE)
    }
    pieces[[j]] <- x
  }

  # Collect column names and handle duplicates
  all_names <- unlist(col_names)
  repaired_names <- repair_names(all_names, .name_repair)

  # dplyr regroups the output with tibble, which refuses duplicated names.
  if (is_tbl_gpu(first) && length(first$groups) > 0L && anyDuplicated(repaired_names) > 0L) {
    dups <- unique(repaired_names[duplicated(repaired_names)])
    rlang::abort(
      c(paste0("Column names ", paste0("`", dups, "`", collapse = ", "),
               " must not be duplicated."),
        "Use `.name_repair` to specify repair."),
      class = c("tibble_error_column_names_must_be_unique", "tibble_error"),
      call = call
    )
  }

  # Collect types (unname to avoid issues)
  all_types <- unname(unlist(lapply(pieces, function(x) x$schema$types)))

  # Factor levels by position (an input may carry duplicated names), re-keyed
  # onto the repaired names.
  all_levels <- unlist(lapply(pieces, function(x) levels_by_position(x$schema)),
                       recursive = FALSE)
  factor_levels <- levels_from_positions(repaired_names, all_levels)
  tz <- tz_from_positions(repaired_names, all_types,
                          unlist(lapply(pieces, function(x) tz_by_position(x$schema))))

  # Collect pointers
  ptrs <- lapply(pieces, function(x) x$ptr)

  # Call C++ function
  new_ptr <- gpu_bind_cols_impl(ptrs)

  # Preserve groups (and their `.drop`) from the first input. dplyr regroups
  # the output by name, so a group column renamed by repair (`bind_cols(g, g)`)
  # no longer groups it.
  groups <- if (is_tbl_gpu(first)) intersect(first$groups, repaired_names) else character()

  out <- new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = repaired_names, types = all_types,
                  factor_levels = factor_levels, tz = tz),
    groups = groups,
    group_drop = if (length(groups) > 0L) first$group_drop %||% TRUE else TRUE,
    exec_mode = "eager"
  )
  # the first input's attributes and class (F31); a bare vector has none
  if (bind_has_meta(first)) carry_frame_meta(out, first) else out
}

# A rowwise first input (a host rowwise_df, or a rowwise tbl_gpu, F49) decides
# the result's class (a rowwise_df): collect the tbl_gpu inputs and let dplyr
# bind on the CPU. With a rowwise tbl_gpu first the result goes back up as a
# rowwise tbl_gpu, like any other fallback.
bind_first_is_rowwise <- function(dots) {
  length(dots) > 0L && (inherits(dots[[1L]], "rowwise_df") || is_rowwise_gpu(dots[[1L]]))
}

bind_rowwise_first <- function(verb, dots, ...) {
  cuplyr_fallback_notify(verb, "a rowwise data frame as the first input")
  first_gpu <- is_rowwise_gpu(dots[[1L]])
  was_lazy <- first_gpu && identical(dots[[1L]]$exec_mode, "lazy")
  dots[-1L] <- lapply(dots[-1L], collect_other_side)
  dots[[1L]] <- if (first_gpu) collect(dots[[1L]]) else dots[[1L]]
  # Called by name so dplyr's errors read `bind_rows()` / `bind_cols()`.
  out <- switch(verb,
    bind_rows = rlang::inject(dplyr::bind_rows(!!!dots, ...)),
    bind_cols = rlang::inject(dplyr::bind_cols(!!!dots, ...))
  )
  if (first_gpu && is.data.frame(out) && !has_list_column(out)) {
    out <- tbl_gpu(out, lazy = was_lazy)
  }
  out
}

# A bind input that has data-frame attributes and a class to hand over: a table,
# not a bare vector.
bind_has_meta <- function(x) is.data.frame(x) || is_tbl_gpu(x)

#' GPU bind_rows implementation
#'
#' Keeps the first input's groups and `.drop` (a tbl_gpu's, or a grouped data
#' frame's), as `dplyr::bind_rows()` does. An input with no columns adds its
#' rows as all-NA rows of the unified schema.
#' @param dots Inputs as flattened and checked by `bind_rows_route()`: tbl_gpu
#'   and data frames, named for every input when `.id` is given
#' @param .id Optional column name to identify source
#' @param call The user's call, for errors
#' @return A tbl_gpu
#' @keywords internal
bind_rows_gpu <- function(dots, .id = NULL, call = rlang::caller_env()) {
  if (length(dots) == 0) {
    stop("No tables provided to bind_rows", call. = FALSE)
  }

  source_names <- names(dots) %||% as.character(seq_along(dots))

  first <- dots[[1L]]
  if (is_tbl_gpu(first)) {
    first_groups <- first$groups
    first_drop <- first$group_drop %||% TRUE
  } else if (dplyr::is_grouped_df(first)) {
    first_groups <- dplyr::group_vars(first)
    first_drop <- dplyr::group_by_drop_default(first)
  } else {
    first_groups <- character()
    first_drop <- TRUE
  }

  # Convert any data.frames to tbl_gpu
  dots <- lapply(dots, function(x) {
    if (is.data.frame(x) && !is_tbl_gpu(x)) {
      upload_other_side(x, lazy = FALSE)
    } else if (!is_tbl_gpu(x)) {
      stop("All inputs must be tbl_gpu or data.frame", call. = FALSE)
    } else {
      x
    }
  })

  # dplyr_reconstruct(out, first): the first input's attributes and class (F31)
  meta_from <- first
  done <- function(res) reset_frame_meta(res, meta_from)

  # Materialize any lazy tables
  dots <- lapply(dots, function(x) {
    if (identical(x$exec_mode, "lazy") && has_pending_ops(x)) {
      compute(x)
    } else {
      x
    }
  })

  if (length(dots) == 1 && is.null(.id)) {
    # Single table, no .id: nothing to bind, groups stay
    return(dots[[1]])
  }

  # Zero-column tables carry their row count on the R side (new_tbl_gpu()),
  # which the concatenation can't see.
  no_cols <- vapply(dots, function(x) length(x$schema$names) == 0L, logical(1))
  row_counts <- vapply(dots, function(x) {
    as.integer(if (length(x$schema$names) == 0L) x[["nrow"]] %||% NA_integer_ else nrow(x))
  }, integer(1))
  if (anyNA(row_counts)) {
    stop("`bind_rows()` of tables with no columns needs known row counts.",
         call. = FALSE)
  }

  if (all(no_cols)) {
    # Only the row count (and the .id column) is left.
    if (is.null(.id)) {
      return(done(new_tbl_gpu(
        ptr = dots[[1]]$ptr,
        schema = dots[[1]]$schema,
        exec_mode = "eager",
        nrow = sum(row_counts)
      )))
    }
    return(done(make_id_table(.id, source_names, row_counts)))
  }

  # Compute unified schema over the inputs that have columns
  real <- which(!no_cols)
  unified <- compute_unified_schema(dots[real], positions = real, call = call)

  if (isTRUE(unified$cpu_tz)) {
    # A Date next to a non-UTC POSIXct: vctrs casts the Date to local midnight
    # of that zone, which R computes exactly (DST included).
    cuplyr_fallback_notify("bind_rows", "a Date combined with a non-UTC POSIXct")
    frames <- lapply(dots, collect_frame)
    names(frames) <- if (!is.null(.id)) source_names
    result <- tbl_gpu(dplyr::bind_rows(!!!frames, .id = .id), lazy = FALSE)
    groups <- intersect(first_groups, result$schema$names)
    check_dup_names_ungrouped(result$schema$names, groups, call = call)
    result$groups <- groups
    result$group_drop <- if (length(groups) > 0L) first_drop else TRUE
    return(done(result))
  }

  # Align each table to unified schema; a zero-column one becomes all-NA rows
  aligned <- lapply(seq_along(dots), function(i) {
    if (!no_cols[[i]]) {
      align_to_schema(dots[[i]], unified)
    } else if (row_counts[[i]] > 0L) {
      null_table_for_schema(unified, row_counts[[i]])
    }
  })
  aligned <- aligned[!vapply(aligned, is.null, logical(1))]

  # Collect pointers
  ptrs <- lapply(aligned, function(x) x$ptr)

  # Concatenate
  new_ptr <- gpu_bind_rows_aligned(ptrs)

  # Build result
  result <- new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = unified$names, types = unified$types,
                  factor_levels = unified$factor_levels, tz = unified$tz),
    groups = character(),
    exec_mode = "eager"
  )

  # Add .id column if requested
  if (!is.null(.id)) {
    result <- add_id_column(result, .id, row_counts, source_names)
  }

  groups <- intersect(first_groups, result$schema$names)
  check_dup_names_ungrouped(result$schema$names, groups, call = call)
  result$groups <- groups
  result$group_drop <- if (length(groups) > 0L) first_drop else TRUE
  done(result)
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
#' @param call The user's call; the error has vctrs' classes
#' @keywords internal
abort_bind_rows_incompatible <- function(other_idx, col_name, type1, type2,
                                          is_factor1 = FALSE, is_factor2 = FALSE,
                                          call = rlang::caller_env()) {
  label1 <- bind_rows_type_label(type1, is_factor1)
  label2 <- bind_rows_type_label(type2, is_factor2)
  rlang::abort(
    sprintf("Can't combine `..1$%s` <%s> and `..%d$%s` <%s>.",
            col_name, label1, other_idx, col_name, label2),
    class = c("vctrs_error_ptype2", "vctrs_error_incompatible_type",
              "vctrs_error_incompatible", "vctrs_error"),
    call = call
  )
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
#' @param positions Each table's position among the user's inputs (for the
#'   error's `..N` labels)
#' @param call The user's call, for errors
#' @return List with `names`, `types`, and `factor_levels` (`NULL` if no
#'   column in the union ends up `DICTIONARY32`) for the unified schema
#' @keywords internal
compute_unified_schema <- function(tables, positions = seq_along(tables),
                                   call = rlang::caller_env()) {
  # Union of all column names (preserving order from first occurrence)
  all_names <- character()
  name_types <- list()
  name_factor_levels <- list()   # nm -> current level vector (only set while DICTIONARY32)
  name_tz <- list()              # nm -> vctrs' common tzone of its POSIXct occurrences (F32)
  name_has_date <- character()   # names with a Date occurrence

  for (t in seq_along(tables)) {
    tbl <- tables[[t]]
    tbl_factor_levels <- tbl$schema$factor_levels
    tbl_tz <- tz_by_position(tbl$schema)

    for (i in seq_along(tbl$schema$names)) {
      nm <- tbl$schema$names[i]
      # unname(): `tbl$schema$types` is a NAMED vector (tbl_gpu() builds it
      # via vapply()) -- identical(ty, "DICTIONARY32") would otherwise
      # compare a named single-element vector against an unnamed literal
      # and always be FALSE (identical() treats "names" as significant).
      ty <- unname(tbl$schema$types[i])
      is_factor <- identical(ty, "DICTIONARY32")
      lvls <- if (is_factor) tbl_factor_levels[[nm]] else NULL
      ty_tz <- if (identical(ty, "TIMESTAMP_MICROSECONDS")) tbl_tz[[i]]
      if (identical(ty, "TIMESTAMP_DAYS")) name_has_date <- c(name_has_date, nm)

      if (!(nm %in% all_names)) {
        all_names <- c(all_names, nm)
        name_types[[nm]] <- ty
        name_factor_levels[[nm]] <- lvls
        name_tz[nm] <- list(ty_tz)
        next
      }
      name_tz[nm] <- list(tz_fold(name_tz[[nm]], ty_tz))

      existing_type <- name_types[[nm]]
      existing_is_factor <- identical(existing_type, "DICTIONARY32")
      existing_lvls <- name_factor_levels[[nm]]

      res <- resolve_common_col_type(existing_type, ty, existing_is_factor, is_factor,
                                      existing_lvls, lvls)
      if (!isTRUE(res$ok)) {
        abort_bind_rows_incompatible(positions[[t]], nm, existing_type, ty,
                                     existing_is_factor, is_factor, call = call)
      }
      name_types[[nm]] <- res$type
      name_factor_levels[[nm]] <- res$factor_levels
    }
  }

  unified_types <- vapply(all_names, function(nm) name_types[[nm]], character(1))

  factor_levels <- name_factor_levels[!vapply(name_factor_levels, is.null, logical(1))]

  # F32: a POSIXct column takes vctrs' common tzone. A Date unified with a
  # POSIXct is cast to midnight in THAT zone (DST-dependent), which the GPU's
  # UTC cast only matches for a UTC-equivalent zone: otherwise the caller runs
  # the bind on the CPU (`cpu_tz`).
  zones <- vapply(all_names, function(nm) name_tz[[nm]] %||% "UTC", character(1))
  zones_out <- tz_localize(zones)  # vec_rbind(): an absent tzone becomes ""
  cpu_tz <- any(vapply(seq_along(all_names), function(k) {
    nm <- all_names[[k]]
    identical(unname(unified_types[[k]]), "TIMESTAMP_MICROSECONDS") &&
      nm %in% name_has_date && !tz_is_utc(zones[[k]])
  }, logical(1)))

  list(
    names = all_names,
    types = unname(unified_types),
    factor_levels = if (length(factor_levels) > 0) factor_levels else NULL,
    tz = tz_from_positions(all_names, unname(unified_types), unname(zones_out)),
    cpu_tz = cpu_tz
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
    # A zone-only difference is metadata: the GPU holds UTC instants either
    # way, so no column is touched (F32).
    if (!tz_matches_target(tbl$schema, target_schema)) {
      tbl$schema$tz <- target_schema$tz
    }
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
                  factor_levels = target_factor_levels,
                  tz = target_schema$tz),
    groups = character(),
    exec_mode = "eager"
  )
}

#' An all-NA table of `n` rows with a unified schema (a zero-column input's
#' rows in `bind_rows()`)
#' @param schema List with `names`, `types` and `factor_levels`
#' @param n Row count
#' @keywords internal
null_table_for_schema <- function(schema, n) {
  # Physical types: a factor's GPU column is INT32 (see add_null_columns()).
  cols <- lapply(unname(schema$types), function(ty) {
    gpu_make_null_column(n, gpu_physical_type(ty))
  })
  new_ptr <- if (length(cols) == 1L) cols[[1L]] else gpu_bind_cols_impl(cols)
  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(names = schema$names, types = unname(schema$types),
                  factor_levels = schema$factor_levels),
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

#' The .id column alone: `source_names` repeated by each input's row count
#' @keywords internal
make_id_table <- function(id_col_name, source_names, row_counts) {
  id_df <- data.frame(x = rep(source_names, times = row_counts),
                      stringsAsFactors = FALSE)
  names(id_df) <- id_col_name
  tbl_gpu(id_df, lazy = FALSE)
}

#' Add .id column to identify source tables
#' @param result The combined tbl_gpu
#' @param id_col_name Name for the .id column
#' @param row_counts Row count of each original input (zero-column ones too)
#' @param source_names Names/identifiers for each source
#' @return A tbl_gpu with .id column prepended
#' @keywords internal
add_id_column <- function(result, id_col_name, row_counts, source_names) {
  id_tbl <- make_id_table(id_col_name, source_names, row_counts)

  # Bind .id column at front
  out <- bind_cols(id_tbl, result, .name_repair = "minimal")

  # vctrs' `.names_to` rule: an `.id` naming an existing column replaces that
  # column in place, so the result never carries a duplicate. `out` is
  # ungrouped here, where the transient duplicated name is allowed; the
  # positional select drops the old column and puts the id in its slot.
  existing <- match(id_col_name, result$schema$names)
  if (!is.na(existing)) {
    n <- length(result$schema$names)
    positions <- seq_len(n + 1L)[-1L]
    positions[existing] <- 1L
    out <- push_op(out, ast_select(input_node(out), out$schema$names[positions],
                                   positions = positions))
  }
  out
}
