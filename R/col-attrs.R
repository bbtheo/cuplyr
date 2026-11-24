# Per-column R metadata that has no GPU form: element names, extra attributes
# (`label`, `scaled:center`, ...) and the shape of a one-column matrix (#21).
#
# `$col_attrs` on a tbl_gpu is a named list keyed by column name, present only
# when some column has any of it (like `$unspecified`), so a table without names
# or extra attributes pays one NULL test per verb. One entry per column:
#   attrs        the constant attributes, put back verbatim at collect (all but
#                names/dim/dimnames and what the column's class owns: class,
#                levels, tzone)
#   names        row-aligned element names (or a matrix's rownames), one per row
#   names_const  ONE name repeated on every row (quantile()'s "50%", or
#                upload of a column whose names are all the same); it needs no
#                row tracking, so it carries like an attribute
#   matrix       TRUE for a one-column matrix; `dim` is c(nrow, 1) at collect
#   colnames     its column name, when it has one
#
# vctrs slices names with the rows and keeps every other attribute
# (vec_restore_default()), so `names` is the only part tied to rows. The safety
# rule: row-aligned `names` are carried only through nodes that cannot move rows
# (select, rename, mutate, window, barrier); a verb that moves rows and should
# keep them takes its CPU route first (`col_names_move()`/`col_cpu_route()`). A
# node that forgets this loses the names, it never puts them on the wrong rows.

# The entries for the columns of `data` that have metadata, as `$col_attrs`
# (NULL when none). O(ncol); a column with no attributes is one NULL test.
# @keywords internal
col_attrs_extract <- function(data) {
  cols <- unclass(data)
  has <- !vapply(cols, has_no_attributes, logical(1), USE.NAMES = FALSE)
  if (!any(has)) {
    return(NULL)
  }
  nms <- names(data)
  # a name that can't be looked up (blank, NA, duplicated) can't key an entry
  usable <- !is.na(nms) & nzchar(nms) & !(duplicated(nms) | duplicated(nms, fromLast = TRUE))
  out <- list()
  for (j in which(has & usable)) {
    e <- col_attrs_entry(cols[[j]])
    if (!is.null(e)) out[[nms[[j]]]] <- e
  }
  if (length(out) > 0L) out
}

has_no_attributes <- function(col) is.null(attributes(col))

# The entry of one column (NULL when everything it has is owned by its type).
# @keywords internal
col_attrs_entry <- function(col) {
  a <- attributes(col)
  drop <- c("names", "dim", "dimnames", "class")
  if (is.factor(col)) drop <- c(drop, "levels")
  if (inherits(col, "POSIXct")) drop <- c(drop, "tzone")
  attrs <- a[!names(a) %in% drop]
  entry <- list()
  if (length(attrs) > 0L) entry$attrs <- attrs
  dims <- a[["dim"]]
  dn <- a[["dimnames"]]
  nm <- a[["names"]]
  if (length(dims) == 2L && dims[[2L]] == 1L && !is.data.frame(col)) {
    entry$matrix <- TRUE
    if (length(dn) == 2L) {
      nm <- dn[[1L]]
      if (!is.null(dn[[2L]])) entry$colnames <- dn[[2L]]
    }
  }
  if (!is.null(nm)) {
    if (length(nm) > 0L && !anyNA(nm) && all(nm == nm[[1L]])) {
      entry$names_const <- nm[[1L]]
    } else {
      entry$names <- nm
    }
  }
  if (length(entry) > 0L) entry
}

# Put the metadata back on a collected frame (`result`, a tibble).
# @keywords internal
col_attrs_restore <- function(result, col_attrs) {
  n <- nrow(result)
  cl <- oldClass(result)
  cols <- unclass(result)
  nms <- names(cols)
  if (anyDuplicated(nms) > 0L) {
    return(result)
  }
  for (nm in names(col_attrs)) {
    j <- match(nm, nms)
    if (is.na(j)) next
    e <- col_attrs[[nm]]
    col <- cols[[j]]
    for (a in names(e$attrs)) attr(col, a) <- e$attrs[[a]]
    row_names <- e[["names"]]
    if (is.null(row_names) && !is.null(e$names_const)) row_names <- rep(e$names_const, n)
    if (!is.null(row_names) && length(row_names) != n) row_names <- NULL # (never wrong rows)
    if (isTRUE(e$matrix)) {
      dim(col) <- c(n, 1L)
      if (!is.null(row_names) || !is.null(e$colnames)) {
        dimnames(col) <- list(row_names, e$colnames)
      }
    } else if (!is.null(row_names)) {
      names(col) <- row_names
    }
    cols[[j]] <- col
  }
  class(cols) <- cl
  cols
}

# Does any column carry row-aligned names (`names`: one per row, which a verb
# that moves rows has to move with them)?
# @keywords internal
col_names_move <- function(x, cols = NULL) {
  for (nm in (cols %||% names(.subset2(x, "col_attrs")))) {
    if (!is.null(.subset2(.subset2(.subset2(x, "col_attrs"), nm), "names"))) return(TRUE)
  }
  FALSE
}

# The CPU-route reason for a table whose columns have row-aligned names.
# @keywords internal
col_names_reason <- function(x, cols = NULL) {
  nms <- cols %||% names(Filter(function(e) !is.null(.subset2(e, "names")), .subset2(x, "col_attrs")))
  paste0("column ", paste0("`", nms, "`", collapse = ", "),
         " has element names, which dplyr moves with the rows.")
}

# Does a join of `x` and `y` (a tbl_gpu) meet element names of any kind in
# either? dplyr gathers them with the rows, null-extended rows included.
# @keywords internal
join_names_need_cpu <- function(x, y) {
  (!is.null(.subset2(x, "col_attrs")) && col_names_any(x)) ||
    (is_tbl_gpu(y) && !is.null(.subset2(y, "col_attrs")) && col_names_any(y))
}

# The columns of `x` and `y` that have names of any kind.
# @keywords internal
join_names_cols <- function(x, y) {
  named <- function(t) {
    if (!is_tbl_gpu(t)) return(character())
    names(Filter(function(e) !is.null(e[["names"]]) || !is.null(e$names_const), .subset2(t, "col_attrs")))
  }
  unique(c(named(x), named(y)))
}

# Does a column hold what a row bind must combine by dplyr's own rules (names
# of any kind, which vctrs concatenates with `""` for an input without; a
# one-column matrix, which it stacks)? Constant attributes are dropped by a
# bind, which is what the GPU bind does.
# @keywords internal
col_binds_by_rule <- function(col) {
  e <- col_attrs_entry(col)
  !is.null(e) && (!is.null(e[["names"]]) || !is.null(e$names_const) || isTRUE(e$matrix))
}

# The reason a bind/set operation/rows_*() of these inputs (tbl_gpus and data
# frames) runs on the CPU for a column's names or matrix shape; NULL if none.
# @keywords internal
col_bind_reason <- function(dots) {
  for (d in dots) {
    cols <- if (is_tbl_gpu(d)) {
      if (is.null(.subset2(d, "col_attrs"))) next
      names(Filter(function(e) !is.null(e[["names"]]) || !is.null(e$names_const) || isTRUE(e$matrix),
                   .subset2(d, "col_attrs")))
    } else if (is.data.frame(d)) {
      nms <- names(d)
      nms[vapply(d, function(col) !is.null(attributes(col)) && col_binds_by_rule(col), logical(1))]
    } else {
      next
    }
    if (length(cols) > 0L) {
      return(paste0("column ", paste0("`", cols, "`", collapse = ", "),
                    " has element names or a matrix shape, which dplyr combines by its own rules."))
    }
  }
  NULL
}

# bind_cols(): each piece's entries under its columns' repaired names (the
# rows are every piece's own, in order).
# @keywords internal
bind_cols_col_attrs <- function(pieces, repaired_names, at) {
  out <- list()
  for (j in seq_along(pieces)) {
    ca <- .subset2(pieces[[j]], "col_attrs")
    if (is.null(ca)) next
    piece_names <- repaired_names[at[[j]] + seq_along(pieces[[j]]$schema$names)]
    names(ca) <- piece_names[match(names(ca), pieces[[j]]$schema$names)]
    out <- c(out, ca)
  }
  out <- out[!is.na(names(out)) & !(names(out) %in% repaired_names[duplicated(repaired_names)])]
  if (length(out) > 0L) out
}

# A join's `$col_attrs`: x's columns keep x's constant entries, y's kept
# columns y's, a merged key takes x's (never one that was cast to a common
# type). A join with names of any kind is routed to the CPU before it gets
# here, so the names are dropped, never carried.
# @keywords internal
join_col_attrs <- function(join_type, x, y, join_spec, suffix, keep, new_schema) {
  xa <- col_attrs_without_names(.subset2(x, "col_attrs"))
  if (join_type %in% c("semi", "anti")) {
    return(xa)
  }
  ya <- col_attrs_without_names(.subset2(y, "col_attrs"))
  info <- build_join_output_info(current_schema(x), current_schema(y), join_spec,
                                 suffix = suffix, keep = keep)
  if (anyDuplicated(info$names) > 0L) {
    return(NULL)
  }
  out <- lapply(seq_along(info$names), function(i) {
    if (info$names[[i]] %in% info$cast_keys) return(NULL)
    (if (identical(info$origin[[i]], "left")) xa else ya)[[info$source_names[[i]]]]
  })
  names(out) <- info$names
  out <- out[!vapply(out, is.null, logical(1))]
  if (length(out) > 0L) out
}

# Has a constant value (a mutate() dot that read no column) attributes besides
# names that its column would keep in dplyr? (Its names go with the column,
# see the `const_names` of the mutate expression.)
# @keywords internal
col_value_has_attrs <- function(value) {
  a <- attributes(value)
  !is.null(a) && any(!names(a) %in% c("names", "class", "levels", "tzone"))
}

# Why a mutate()/transmute() whose `expressions` (the parsed dots) read a
# column of `attributed` (the names of columns with `$col_attrs` entries) in
# anything but a bare copy runs on dplyr: R's own rules decide what the result
# keeps. NULL when none does. A bare copy keeps the entry, so its output is
# attributed too; any other output isn't.
# @keywords internal
col_attrs_read_reason <- function(attributed, expressions) {
  for (e in expressions) {
    ir <- .subset2(e, "ir")
    out <- .subset2(e, "output_col")
    if (identical(.subset2(ir, "kind"), "col")) {
      attributed <- if (.subset2(ir, "name") %in% attributed) union(attributed, out) else setdiff(attributed, out)
      next
    }
    hit <- intersect(ir_cols(ir), attributed)
    if (length(hit) > 0L) {
      return(paste0("column ", paste0("`", hit, "`", collapse = ", "),
                    " has names or attributes, which R carries through an expression by its own rules."))
    }
    attributed <- setdiff(attributed, out)
  }
  NULL
}

# Does any column have element names of any kind (row-aligned or constant)?
# @keywords internal
col_names_any <- function(x) {
  for (e in .subset2(x, "col_attrs")) {
    if (!is.null(.subset2(e, "names")) || !is.null(.subset2(e, "names_const"))) return(TRUE)
  }
  FALSE
}

# `col_attrs` without the row-aligned names (what a row-moving node may carry).
# @keywords internal
col_attrs_without_names <- function(col_attrs) {
  for (nm in names(col_attrs)) {
    e <- col_attrs[[nm]]
    if (!is.null(e[["names"]])) {
      e$names <- NULL
      col_attrs[nm] <- list(if (length(e) > 0L) e)
    }
  }
  col_attrs <- col_attrs[!vapply(col_attrs, is.null, logical(1))]
  if (length(col_attrs) > 0L) col_attrs
}

# For each output column of `node`, the position of the input column it is an
# unchanged copy of (NA: a new or recomputed column, or a node that gives no
# column its input's values). The one column-origin map the per-column metadata
# rides on.
# @keywords internal
col_origin <- function(node, old_names, new_names) {
  none <- rep(NA_integer_, length(new_names))
  switch(.subset2(node, "type"),
    select = {
      pos <- .subset2(node, "positions") %||% match(.subset2(node, "columns"), old_names)
      if (length(pos) == length(new_names)) as.integer(pos) else none
    },
    rename = {
      # select(x = y) is one rename node over a fresh select node: the select's
      # columns are the table's own names, where a select already in the plan
      # has the names the table has now
      input <- .subset2(node, "input")
      sel <- if (identical(.subset2(input, "type"), "select") &&
                 !identical(.subset2(input, "columns"), old_names)) input
      if (!is.null(sel)) {
        pos <- .subset2(sel, "positions") %||% match(.subset2(sel, "columns"), old_names)
        if (length(pos) == length(new_names)) as.integer(pos) else none
      } else if (length(old_names) == length(new_names)) {
        seq_along(new_names)
      } else {
        none
      }
    },
    barrier = , filter = , arrange = , slice = {
      if (length(old_names) == length(new_names)) seq_along(new_names) else none
    },
    window = {
      out <- vapply(.subset2(node, "specs"), .subset2, "", "output_col")
      origin <- match(new_names, old_names)
      origin[new_names %in% out] <- NA_integer_
      origin
    },
    mutate = {
      cur <- as.list(stats::setNames(seq_along(old_names), old_names))
      for (e in .subset2(node, "expressions")) {
        ir <- .subset2(e, "ir")
        src <- if (identical(.subset2(ir, "kind"), "col")) cur[[.subset2(ir, "name")]]
        cur[.subset2(e, "output_col")] <- list(src)
      }
      vapply(new_names, function(nm) cur[[nm]] %||% NA_integer_, integer(1), USE.NAMES = FALSE)
    },
    distinct = match(new_names, old_names),
    summarise = {
      keys <- setdiff(.subset2(node, "groups"), summarise_output_names(node))
      origin <- match(new_names, keys)
      ifelse(is.na(origin), NA_integer_, match(new_names, old_names))
    },
    none
  )
}

# The output names of a summarise node's aggregations.
# @keywords internal
summarise_output_names <- function(node) {
  vapply(.subset2(node, "aggregations"), function(a) .subset2(a, "output_col") %||% NA_character_, "")
}

# `$col_attrs` of the output of `node`: each output column that is an unchanged
# copy of an input column takes that column's entry; the row-aligned names go
# only through the nodes that cannot move rows.
# @keywords internal
carry_col_attrs <- function(node, col_attrs, old_names, new_names) {
  type <- .subset2(node, "type")
  # summarise: quantile() names its output (R/summarise.R)
  new_entries <- list()
  if (identical(type, "mutate")) {
    # a constant with names (`b = c(p = 1)`, `b = setNames(1:3, letters[1:3])`):
    # one name for every row, or one per row (a vector constant is the table's
    # length); a later dot of the same name replaces it
    for (e in .subset2(node, "expressions")) {
      out_col <- .subset2(e, "output_col")
      cn <- .subset2(e, "const_names")
      new_entries[out_col] <- list(if (is.null(cn)) NULL else if (length(cn) == 1L && identical(.subset2(.subset2(e, "ir"), "kind"), "lit")) {
        list(names_const = cn[[1L]])
      } else {
        list(names = cn)
      })
    }
    new_entries <- new_entries[!vapply(new_entries, is.null, logical(1))]
  }
  if (identical(type, "summarise")) {
    for (agg in .subset2(node, "aggregations")) {
      qn <- .subset2(agg, "qnames")
      if (!is.null(qn)) new_entries[[.subset2(agg, "output_col")]] <- list(names_const = qn[[1L]])
    }
  }
  if (is.null(col_attrs) && length(new_entries) == 0L) {
    return(NULL)
  }
  if (anyDuplicated(old_names) > 0L || anyDuplicated(new_names) > 0L) {
    return(NULL)
  }
  out <- list()
  if (!is.null(col_attrs)) {
    origin <- col_origin(node, old_names, new_names)
    out <- col_attrs[old_names[origin]]
    names(out) <- new_names
    out <- out[!vapply(out, is.null, logical(1))]
    if (!type %in% c("select", "rename", "mutate", "window", "barrier")) {
      out <- col_attrs_without_names(out)
    }
  }
  out[names(new_entries)] <- new_entries
  if (length(out) > 0L) out
}

# Run the verb that called this on the CPU: dplyr's own verb on the collected
# table, with the arguments exactly as the user wrote them (rowwise_cpu()'s
# mechanism: the call is rebuilt from the method's call and evaluated in the
# caller's environment). The route of a row-moving verb whose table has names
# that dplyr moves with the rows. Called first thing in a method, as
# `return(col_cpu_route(.data, "filter"))`; `tables` names the arguments that
# hold another table (a join's `y`), collected here.
# @keywords internal
col_cpu_route <- function(.data, verb, tables = character(),
                          reason = col_names_reason(.data)) {
  idx <- sys.parent()
  fn <- sys.function(idx)
  call <- sys.call(idx)
  caller <- parent.frame(2)
  mc <- match.call(fn, call, expand.dots = TRUE, envir = caller)
  first <- names(formals(fn))[[1L]]
  also <- list()
  for (nm in set_intersect(tables, names(mc))) {
    val <- rlang::eval_bare(mc[[nm]], caller)
    if (is_tbl_gpu(val)) {
      val <- fallback_materialize(val)
      also[[length(also) + 1L]] <- val
      val <- collect_other_side(val)
    }
    mc[[nm]] <- val
  }
  verb_fn <- rlang::call2("::", quote(dplyr), rlang::sym(verb))
  res <- gpu_fallback(verb, .data, function(tbl) {
    mc[[first]] <- tbl
    mc[[1L]] <- verb_fn
    rlang::eval_bare(mc, caller)
  }, reason = reason, also = also)
  result_mode_of_inputs(res, c(list(.data), also))
}
