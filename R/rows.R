# rows_insert()/rows_append()/rows_update()/rows_patch()/rows_upsert()/
# rows_delete() for tbl_gpu (F26).
#
# dplyr's recipe is key matching (vec_match/vec_in), row slicing, column
# patching and vec_rbind, so each verb is composed from native primitives:
# bind_rows, semi/anti/left joins, distinct, mutate(if_else/coalesce). One
# shared driver, rows_run():
#   1. dplyr's own check order (dots, in_place, auto_copy);
#   2. the real dplyr verb on 0-row prototypes of x and y: every schema-level
#      error and message comes from dplyr itself. Its messages are held back
#      and replayed only when the native path completes (a CPU fallback makes
#      dplyr emit them itself, so each appears exactly once, in dplyr's order);
#   3. a positive allowlist of shapes (rows_native_ok()); anything else, and
#      every data-dependent error (conflicts, unmatched or duplicated y keys),
#      runs the verb through gpu_fallback() so dplyr produces the result or
#      the exact error, row locations included.

#' Row-wise modification of a `tbl_gpu`
#'
#' `rows_insert()`, `rows_append()`, `rows_update()`, `rows_patch()`,
#' `rows_upsert()` and `rows_delete()` for `tbl_gpu`. Tables whose columns are
#' logical, integer, double, character, Date or POSIXct (a `y` column may also
#' be a lossless widening of `x`'s: integer to double, logical to integer or
#' double) run on the GPU. Anything else (factors, other casts, a zone that
#' differs from `x`'s) and every data-dependent error runs dplyr's own method
#' on the CPU, which gives dplyr's result or its exact error. Like the GPU
#' joins, a `y` that is not a `tbl_gpu` needs `copy = TRUE`. A lazy `x` is
#' computed at call time (the verbs are not nodes of the lazy plan), and the
#' result stays lazy.
#'
#' @param x A `tbl_gpu`.
#' @param y A `tbl_gpu`, or a data frame with `copy = TRUE`.
#' @param by,conflict,unmatched,copy,in_place As in [dplyr::rows_insert()].
#' @param ... Must be empty.
#' @name rows-tbl_gpu
#' @keywords internal
NULL

#' @rdname rows-tbl_gpu
#' @export
#' @importFrom dplyr rows_insert
rows_insert.tbl_gpu <- function(x, y, by = NULL, ..., conflict = c("error", "ignore"),
                                copy = FALSE, in_place = FALSE) {
  if (is_rowwise_gpu(x)) {
    return(rowwise_cpu(x, "rows_insert", tables = "y"))
  }
  rlang::check_dots_empty()
  rows_run("rows_insert", x, y, by, copy, in_place, list(conflict = conflict))
}

#' @rdname rows-tbl_gpu
#' @export
#' @importFrom dplyr rows_append
rows_append.tbl_gpu <- function(x, y, ..., copy = FALSE, in_place = FALSE) {
  if (is_rowwise_gpu(x)) {
    return(rowwise_cpu(x, "rows_append", tables = "y"))
  }
  rlang::check_dots_empty()
  rows_run("rows_append", x, y, NULL, copy, in_place, list())
}

#' @rdname rows-tbl_gpu
#' @export
#' @importFrom dplyr rows_update
rows_update.tbl_gpu <- function(x, y, by = NULL, ..., unmatched = c("error", "ignore"),
                                copy = FALSE, in_place = FALSE) {
  if (is_rowwise_gpu(x)) {
    return(rowwise_cpu(x, "rows_update", tables = "y"))
  }
  rlang::check_dots_empty()
  rows_run("rows_update", x, y, by, copy, in_place, list(unmatched = unmatched))
}

#' @rdname rows-tbl_gpu
#' @export
#' @importFrom dplyr rows_patch
rows_patch.tbl_gpu <- function(x, y, by = NULL, ..., unmatched = c("error", "ignore"),
                               copy = FALSE, in_place = FALSE) {
  if (is_rowwise_gpu(x)) {
    return(rowwise_cpu(x, "rows_patch", tables = "y"))
  }
  rlang::check_dots_empty()
  rows_run("rows_patch", x, y, by, copy, in_place, list(unmatched = unmatched))
}

#' @rdname rows-tbl_gpu
#' @export
#' @importFrom dplyr rows_upsert
rows_upsert.tbl_gpu <- function(x, y, by = NULL, ..., copy = FALSE, in_place = FALSE) {
  if (is_rowwise_gpu(x)) {
    return(rowwise_cpu(x, "rows_upsert", tables = "y"))
  }
  rlang::check_dots_empty()
  rows_run("rows_upsert", x, y, by, copy, in_place, list())
}

#' @rdname rows-tbl_gpu
#' @export
#' @importFrom dplyr rows_delete
rows_delete.tbl_gpu <- function(x, y, by = NULL, ..., unmatched = c("error", "ignore"),
                                copy = FALSE, in_place = FALSE) {
  if (is_rowwise_gpu(x)) {
    return(rowwise_cpu(x, "rows_delete", tables = "y"))
  }
  rlang::check_dots_empty()
  rows_run("rows_delete", x, y, by, copy, in_place, list(unmatched = unmatched))
}

# The shared driver. `opts` holds the verb's own `conflict`/`unmatched`
# argument exactly as the user gave it (dplyr validates it on the prototypes).
# @keywords internal
rows_run <- function(verb, x, y, by, copy, in_place, opts) {
  has_by <- !identical(verb, "rows_append")
  verb_args <- function(extra) {
    c(if (has_by) list(by = by), opts, extra)
  }

  # dplyr: check_dots_empty(), rows_df_in_place(), auto_copy(). Only a TRUE
  # in_place is an error (rows_df_in_place() uses is_true()); the error is
  # dplyr's own, from a call that fails on it before anything else.
  if (rlang::is_true(in_place)) {
    px <- gpu_proto(x)
    rows_call(verb, px, px, verb_args(list(in_place = in_place)))
  }
  y <- dplyr::auto_copy(x, y, copy = copy)

  # Schema-level checks and messages from dplyr itself, on 0-row prototypes.
  msgs <- list()
  proto <- withCallingHandlers(
    tryCatch(
      {
        rows_call(verb, gpu_proto(x), gpu_proto(y), verb_args(list()))
        NULL
      },
      error = function(e) e
    ),
    message = function(m) {
      msgs[[length(msgs) + 1L]] <<- m
      invokeRestart("muffleMessage")
    }
  )
  replay <- function() {
    for (m in msgs) rlang::cnd_signal(m)
  }
  fallback <- function(reason) {
    y <- fallback_materialize(y)
    y_tbl <- collect_other_side(y)
    gpu_fallback(verb, x, function(tbl) {
      rows_call(verb, tbl, y_tbl, verb_args(list(copy = copy)))
    }, reason = reason, also = y)
  }
  if (!is.null(proto)) {
    # An all-NA logical column casts to anything in dplyr, but its 0-row
    # prototype is not all-NA: only the real data can tell.
    if (rows_has_logical_y(x, y)) {
      return(fallback("an all-NA logical column in `y` casts to any type in dplyr; only the real data can tell."))
    }
    replay()
    rlang::cnd_signal(proto)
  }

  by_cols <- if (has_by) (by %||% current_schema(y)$names[[1L]]) else character()
  if (!rows_native_ok(verb, x, y, by_cols)) {
    return(fallback(paste0(verb, "() with these key or column types has no native GPU form.")))
  }

  mode <- if (!has_by || identical(verb, "rows_upsert")) {
    NULL
  } else {
    # the proto call validated the argument, so the default vector means "error"
    (opts$conflict %||% opts$unmatched)[[1L]]
  }
  xe <- as_eager(x)
  res <- rows_native(verb, dplyr::ungroup(xe), rows_eager(y), by_cols, mode)
  if (is.null(res)) {
    return(fallback("the keys conflict, repeat or are missing in a way dplyr reports itself (an error or its own result)."))
  }

  replay()
  rows_finish(res, x, rows_delete_keys(verb, xe, res))
}

# dplyr_row_slice() keeps a `.drop = FALSE` table's whole group structure
# when rows_delete() slices it, so a group whose rows all went stays in
# group_data(). Those keys are fixed on the result (as filter() does), but
# only when a group was actually lost: otherwise a plain regroup gives the
# same groups and fixed keys would only push later verbs onto the CPU (the
# rule of result_fixed_keys(), R/fallback.R). The allowlist keeps factor
# group columns off this path, so x's keys are its observed groups and
# counting distinct keys on both sides tells whether one was lost. `xe` is
# eager and grouped, `res` ungrouped.
# @keywords internal
rows_delete_keys <- function(verb, xe, res) {
  groups <- xe$groups
  if (!identical(verb, "rows_delete") || length(groups) == 0L || !isFALSE(xe$group_drop)) {
    return(NULL)
  }
  n_keys <- function(t) {
    base_dim(dplyr::distinct(dplyr::select(dplyr::ungroup(t), !!!rlang::syms(groups))))[[1L]]
  }
  if (n_keys(res) == n_keys(xe)) {
    return(NULL)
  }
  slice_group_keys(xe)
}

# `<verb>(x, y, <args>)` run as dplyr's own generic, from an environment that
# binds the verb, `x`, `y` and each argument by name, so an error reads "Error
# in `rows_insert()`:" and its call neither embeds the data nor the verb
# function.
# @keywords internal
rows_call <- function(verb, x, y, args) {
  env <- rlang::new_environment(c(
    stats::setNames(list(getExportedValue("dplyr", verb)), verb),
    list(x = x, y = y), args
  ))
  call <- rlang::call2(rlang::sym(verb), quote(x), quote(y),
                       !!!rlang::syms(names(args)) |> stats::setNames(names(args)))
  eval(call, env)
}

# Does y have a logical column that x holds as something a logical does not
# widen into (character, Date, POSIXct)?
# @keywords internal
rows_has_logical_y <- function(x, y) {
  xs <- current_schema(x)
  ys <- current_schema(y)
  for (j in which(unname(ys$types) == "BOOL8")) {
    i <- match(ys$names[[j]], xs$names)
    if (!is.na(i) && !(unname(xs$types)[[i]] %in% c("BOOL8", "INT32", "FLOAT64"))) {
      return(TRUE)
    }
  }
  FALSE
}

# --- allowlist ---------------------------------------------------------------

rows_native_types <- c("BOOL8", "INT32", "FLOAT64", "STRING", "TIMESTAMP_DAYS",
                       "TIMESTAMP_MICROSECONDS")

# y type -> x types it widens into without loss
rows_widenings <- list(INT32 = "FLOAT64", BOOL8 = c("INT32", "FLOAT64"))

# The positive allowlist (see the file header and the architecture notes): x
# and y columns of plain GPU types, each y column x's type or a lossless
# widening of it, POSIXct zones equal and named (or UTC), no duplicated names,
# x without fixed (empty) groups, no repeated `by` column. `by` is resolved.
# rows_delete() looks at the key columns only (dplyr ignores the others).
# @keywords internal
rows_native_ok <- function(verb, x, y, by) {
  if (!is.null(fixed_keys_of(x))) {
    return(FALSE)
  }
  # a repeated `by` column is something dplyr tolerates and the joins do not
  if (anyDuplicated(by)) {
    return(FALSE)
  }
  xs <- current_schema(x)
  ys <- current_schema(y)
  xn <- xs$names
  yn <- if (identical(verb, "rows_delete")) by else ys$names
  if (length(xn) == 0L || length(yn) == 0L || anyDuplicated(xn) || anyDuplicated(ys$names)) {
    return(FALSE)
  }
  xt <- unname(xs$types)
  if (!all(xt %in% rows_native_types) || !all(yn %in% xn)) {
    return(FALSE)
  }
  # a zone of "" or an absent tzone attribute: left to dplyr (vctrs' rules)
  x_tz <- tz_by_position(xs)
  if (any(xt == "TIMESTAMP_MICROSECONDS" & (is.na(x_tz) | !nzchar(x_tz)))) {
    return(FALSE)
  }
  y_tz <- tz_by_position(ys)
  for (nm in yn) {
    i <- match(nm, xn)
    j <- match(nm, ys$names)
    xty <- xt[[i]]
    yty <- unname(ys$types)[[j]]
    if (!(yty %in% rows_native_types)) {
      return(FALSE)
    }
    if (!identical(xty, yty) && !(xty %in% rows_widenings[[yty]])) {
      return(FALSE)
    }
    if (identical(xty, "TIMESTAMP_MICROSECONDS") && !identical(x_tz[[i]], y_tz[[j]])) {
      return(FALSE)
    }
  }
  TRUE
}

# --- native composition ------------------------------------------------------

# An eager, ungrouped table (groups are metadata; x's are put back at the end).
# @keywords internal
rows_eager <- function(t) {
  dplyr::ungroup(as_eager(t))
}

# `y`'s columns cast to the types of the same-named columns of `x` (x's type
# always wins; the allowlist only lets lossless widenings through).
# @keywords internal
rows_cast_y <- function(y, x) {
  tz <- y$schema$tz
  for (nm in y$schema$names) {
    want <- x$schema$types[[match(nm, x$schema$names)]]
    if (!identical(unname(y$schema$types[[match(nm, y$schema$names)]]), unname(want))) {
      y <- cast_column(y, nm, unname(want))
    }
  }
  y$schema$tz <- tz
  y
}

# A column name not in `taken`.
# @keywords internal
rows_temp_name <- function(taken, stem) {
  nm <- stem
  i <- 0L
  while (nm %in% taken) {
    i <- i + 1L
    nm <- paste0(stem, i)
  }
  nm
}

# Rows of `a` (already cast to x's types) appended below `x`, x's columns.
# @keywords internal
rows_bind <- function(x, a) {
  bind_rows_gpu(list(x, a))
}

# x's rows with the matched ones patched from `y` (keys unique in y, same
# types as x): left join of x with y's value columns under temp names, then
# update takes y's value on a matched row (marker), patch only fills x's
# missing value (NA or NaN) on a matched row.
# @keywords internal
rows_patch_from <- function(x, y, by, update) {
  vals <- setdiff(y$schema$names, by)
  if (length(vals) == 0L) {
    return(x)
  }
  taken <- c(x$schema$names, y$schema$names)
  tmp <- character(length(vals))
  for (i in seq_along(vals)) {
    tmp[[i]] <- rows_temp_name(c(taken, tmp), paste0(".cuplyr_rows_v", i))
  }
  marker <- rows_temp_name(c(taken, tmp), ".cuplyr_rows_m")
  yt <- dplyr::select(y, !!!rlang::syms(by), !!!stats::setNames(rlang::syms(vals), tmp))
  yt <- dplyr::mutate(yt, !!marker := TRUE)
  j <- dplyr::left_join(x, yt, by = by)
  exprs <- lapply(seq_along(vals), function(i) {
    v <- rlang::sym(vals[[i]])
    t <- rlang::sym(tmp[[i]])
    m <- rlang::sym(marker)
    if (update) {
      rlang::expr(if_else(is.na(!!m), !!v, !!t))
    } else {
      # "missing" is NA or NaN, as for vctrs, and is.na() sees both (F55).
      missing <- rlang::expr(is.na(!!v))
      # dplyr's coalesce() gives NA, never NaN, where x and y are both
      # missing, so a FLOAT64 y value that is itself missing is taken as NA.
      fill <- if (identical(unname(y$schema$types[[match(vals[[i]], y$schema$names)]]), "FLOAT64")) {
        rlang::expr(if_else(is.na(!!t), NA_real_, !!t))
      } else {
        t
      }
      rlang::expr(if_else(!is.na(!!m) & !!missing, !!fill, !!v))
    }
  })
  names(exprs) <- vals
  j <- dplyr::mutate(j, !!!exprs)
  dplyr::select(j, !!!rlang::syms(x$schema$names))
}

# Returns the result table, or NULL to run the CPU fallback (a data-dependent
# error dplyr must raise itself). `x`, `y` eager and ungrouped.
# @keywords internal
rows_native <- function(verb, x, y, by, mode) {
  n_rows <- function(t) base_dim(t)[[1L]]
  switch(verb,
    rows_append = rows_bind(x, rows_cast_y(y, x)),
    rows_insert = {
      yc <- rows_cast_y(y, x)
      new <- dplyr::anti_join(yc, x, by = by)
      if (identical(mode, "error") && n_rows(new) != n_rows(yc)) {
        return(NULL)
      }
      rows_bind(x, new)
    },
    rows_delete = {
      yk <- rows_cast_y(dplyr::select(y, !!!rlang::syms(by)), x)
      if (identical(mode, "error") &&
          n_rows(dplyr::anti_join(yk, x, by = by)) > 0L) {
        return(NULL)
      }
      dplyr::anti_join(x, yk, by = by)
    },
    {
      # update / patch / upsert: y's keys must be unique (NA counts)
      yc <- rows_cast_y(y, x)
      keys <- dplyr::select(yc, !!!rlang::syms(by))
      if (n_rows(dplyr::distinct(keys)) != n_rows(yc)) {
        return(NULL)
      }
      if (identical(verb, "rows_upsert")) {
        out <- rows_patch_from(x, yc, by, update = TRUE)
        rows_bind(out, dplyr::anti_join(yc, x, by = by))
      } else {
        if (identical(mode, "error") && n_rows(dplyr::anti_join(keys, x, by = by)) > 0L) {
          return(NULL)
        }
        rows_patch_from(x, yc, by, update = identical(verb, "rows_update"))
      }
    }
  )
}

# x's groups (and, for a `.drop = FALSE` rows_delete(), its full group keys),
# `.drop`, frame class and attributes, and exec mode on the result.
# @keywords internal
rows_finish <- function(res, x, keys = NULL) {
  res <- reset_frame_meta(res, x)
  groups <- .subset2(x, "groups")
  res$groups <- groups
  res$group_drop <- if (length(groups) > 0L) isTRUE(.subset2(x, "group_drop") %||% TRUE) else TRUE
  if (!is.null(keys)) {
    res$fixed_keys <- keys
  }
  if (identical(.subset2(x, "exec_mode"), "lazy")) as_lazy(res) else res
}
