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
# keep them takes its CPU route first (`col_names_move()` and `rowwise_cpu()` with a `reason`). A
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
  if (length(attrs) > 0L) {
    entry$attrs <- attrs
    entry$ptype <- col_attrs_ptype(col)
  }
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

# A zero-length copy of `col` that holds its constant attributes (no names,
# shape, class, levels or zone: the type owns those). vctrs copies attributes
# onto a result without checking them, which `attr<-` can't do for one R
# validates against the length (`tsp`); this is what col_attrs_restore() copies
# from. NULL when vctrs can't slice the column.
# @keywords internal
col_attrs_ptype <- function(col) {
  p <- tryCatch(vctrs::vec_slice(col, 0L), error = function(e) NULL)
  if (is.null(p)) {
    return(NULL)
  }
  for (a in c("names", "dim", "dimnames", "class", "levels", "tzone")) attr(p, a) <- NULL
  p
}

# `col` with the constant attributes of entry `e` on it, as vec_slice() would
# leave them; what the type owns (class, levels, zone) stays the column's own.
# @keywords internal
col_attrs_put <- function(col, e) {
  ptype <- .subset2(e, "ptype")
  if (is.null(ptype)) {
    for (a in names(e$attrs)) attr(col, a) <- e$attrs[[a]]
    return(col)
  }
  own <- attributes(col)
  out <- vctrs::vec_restore(col, ptype)
  for (a in c("class", "levels", "tzone")) attr(out, a) <- own[[a]]
  out
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
    if (!is.null(e$attrs)) col <- col_attrs_put(col, e)
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

# Does any of `cols` (default: every column with an entry) carry row-aligned
# names (`names`: one per row, which a verb that moves rows has to move with
# them)? `cols = NULL` is no column.
# @keywords internal
col_names_move <- function(x, cols = names(.subset2(x, "col_attrs"))) {
  ca <- .subset2(x, "col_attrs")
  for (nm in cols) {
    if (!is.null(.subset2(.subset2(ca, nm), "names"))) return(TRUE)
  }
  FALSE
}

# "column `b` has" / "columns `b`, `d` have", for the reasons below.
# @keywords internal
col_list_phrase <- function(cols) {
  paste0(if (length(cols) == 1L) "column " else "columns ",
         paste0("`", cols, "`", collapse = ", "),
         if (length(cols) == 1L) " has" else " have")
}

# Can the row subscript `i` of `[` add rows (an NA or out-of-range position,
# which gets NA for a name), on a table of `n` rows (NA: not known yet)?
# @keywords internal
bracket_may_pad <- function(i, n) {
  is.na(n) || is.character(i) || anyNA(i) || (is.numeric(i) && any(i > n))
}

# The CPU-route reason for a `[` that may add rows to a table whose columns
# have element names: the added rows' names are NA, which no entry says.
# @keywords internal
col_names_pad_reason <- function(x) {
  nms <- names(Filter(function(e) !is.null(.subset2(e, "names_const")), .subset2(x, "col_attrs")))
  paste0(col_list_phrase(nms), " element names, which are NA on a row the subscript adds.")
}

# The CPU-route reason for a table whose columns have row-aligned names: the
# columns of `cols` (default: all) that do.
# @keywords internal
col_names_reason <- function(x, cols = names(.subset2(x, "col_attrs"))) {
  ca <- .subset2(x, "col_attrs")
  nms <- cols[vapply(cols, function(nm) !is.null(.subset2(.subset2(ca, nm), "names")), logical(1))]
  paste0(col_list_phrase(nms), " element names, which dplyr moves with the rows.")
}

# The CPU-route reason of a join of type `join_type` whose inputs have element
# names it can't give (`join_names_need_cpu()`).
# @keywords internal
join_names_reason <- function(x, y, join_type) {
  paste0(col_list_phrase(join_names_cols(x, y, join_type)), " element names, which dplyr gathers with the rows.")
}

# Does dplyr rebuild the group keys of `x` (compute_groups(), for group_data(),
# group_keys(), summarise() and count())? Only for a `.drop = FALSE` grouping
# with a factor key: a factor key is then made from its codes and levels alone
# (no other attribute, no names), any other key from its unique values
# (attributes kept, the names of the first group in key order with that value;
# `""` on a row under a parent key with no rows, which slices it with NA).
# Without a factor key, `.drop = FALSE` keys are those of `.drop = TRUE`.
# @keywords internal
group_keys_rebuilt <- function(x, schema = current_schema(x)) {
  isFALSE(.subset2(x, "group_drop")) &&
    any(match(.subset2(x, "groups"), names(.subset2(schema, "factor_levels")), 0L) > 0L)
}

# The group data `gd` of `x` (a table with `$col_attrs`, no pending steps) with
# the attributes of its key columns on them. A key's names are those of its
# group's first row (`group_keys_rebuilt()`: dplyr's rebuilt keys). Fixed keys
# (`fixed`: a `.drop = FALSE` table that lost a group, slice(.preserve = TRUE))
# get no names; a rebuilt factor key no attributes, fixed or not (dplyr's
# row slice keeps the group data group_by() rebuilt).
# @keywords internal
group_keys_col_attrs <- function(gd, x, fixed) {
  keys <- .subset2(x, "groups")
  ca <- .subset2(x, "col_attrs")[keys]
  ca <- ca[!vapply(ca, is.null, logical(1))]
  rebuilt <- group_keys_rebuilt(x)
  if (rebuilt) {
    ca <- ca[!names(ca) %in% names(.subset2(x, "schema")$factor_levels)]
  }
  if (length(ca) == 0L) {
    return(gd)
  }
  named <- !fixed &
    vapply(ca, function(e) !is.null(e[["names"]]) || (rebuilt && !is.null(e$names_const)), logical(1))
  first <- if (any(named)) {
    vapply(gd$.rows, function(r) if (length(r) > 0L) r[[1L]] else NA_integer_, integer(1))
  }
  for (nm in names(ca)) {
    e <- ca[[nm]]
    if (fixed) {
      e$names <- NULL
      e$names_const <- NULL
    } else if (rebuilt && named[[nm]]) {
      e <- group_key_rebuilt_names(e, gd, match(nm, keys), keys, first)
    } else if (!is.null(e[["names"]])) {
      e$names <- e[["names"]][first]
    }
    ca[nm] <- list(e)
  }
  col_attrs_restore(gd, ca)
}

# The entry `e` of the key at position `p` of `keys` (not a factor) as dplyr's
# rebuilt keys name it (`group_keys_rebuilt()`): each row takes the names of the
# first row of the first observed group (in key order) with its value, and a row
# under a parent key with no rows `""`. `first` is each group's first row (NA
# for an empty group).
# @keywords internal
group_key_rebuilt_names <- function(e, gd, p, keys, first) {
  obs <- !is.na(first)
  n <- length(first)
  if (!is.null(e[["names"]])) {
    id <- vctrs::vec_group_id(gd[[keys[[p]]]])
    src <- first[obs][match(seq_len(attr(id, "n")), id[obs])]
    out <- e[["names"]][src[id]]
  } else {
    out <- rep(e$names_const, n)
  }
  under_rows <- if (p == 1L) {
    rep(any(obs), n)
  } else {
    pid <- vctrs::vec_group_id(gd[keys[seq_len(p - 1L)]])
    (tabulate(pid[obs], nbins = attr(pid, "n")) > 0L)[pid]
  }
  out[!under_rows] <- ""
  e$names_const <- NULL
  e$names <- out
  e
}

# `col_attrs` (a summarise's result) as dplyr's rebuilt keys have them
# (`group_keys_rebuilt()` of `x`, the summarise's input), for the keys among
# `keys`: a rebuilt factor has no attributes beyond its levels, and any other
# key loses a constant name, which dplyr leaves `""` on a row under an empty
# parent key. (summarise() and reframe() send such a key to the CPU first
# (`summarise_key_names_cols()`); a path that misses that loses the name, it
# never puts it on the wrong rows.)
# @keywords internal
col_attrs_rebuilt_keys <- function(col_attrs, x, keys, schema) {
  if (!group_keys_rebuilt(x, schema)) {
    return(col_attrs)
  }
  fl <- names(schema$factor_levels)
  out <- col_attrs[!names(col_attrs) %in% intersect(keys, fl)]
  for (nm in intersect(setdiff(keys, fl), names(out))) {
    e <- out[[nm]]
    e$names_const <- NULL
    out[nm] <- list(if (length(e) > 0L) e)
  }
  out <- out[!vapply(out, is.null, logical(1))]
  if (length(out) > 0L) out
}

# The keys among `keys` whose names a summarise() of `x` can't give natively:
# row-aligned names (a group's first row is dplyr's to pick), and, with keys
# dplyr rebuilds (`group_keys_rebuilt()`), a constant name on a key that isn't a
# factor (the rows under an empty parent get `""`).
# @keywords internal
summarise_key_names_cols <- function(x, keys) {
  ca <- .subset2(x, "col_attrs")
  rebuilt <- group_keys_rebuilt(x)
  fl <- names(current_schema(x)$factor_levels)
  keys[vapply(keys, function(nm) {
    e <- .subset2(ca, nm)
    !is.null(.subset2(e, "names")) ||
      (rebuilt && !is.null(.subset2(e, "names_const")) && !nm %in% fl)
  }, logical(1))]
}

# The CPU-route reason for the keys `summarise_key_names_cols()` found.
# @keywords internal
key_names_reason <- function(keys) {
  paste0(col_list_phrase(keys), " element names, which dplyr gives the group keys by its own rules.")
}

# `out` (a summarise() or reframe() result of a plain data.frame) without any
# element names: dplyr's methods for a data.frame end with as.data.frame(),
# which drops them (a matrix's rownames too) and keeps the other attributes.
# @keywords internal
col_attrs_drop_names <- function(out) {
  # (a CPU result that stays on the host, a list column's, is dplyr's own)
  ca <- if (is_tbl_gpu(out)) .subset2(out, "col_attrs")
  if (is.null(ca)) {
    return(out)
  }
  for (nm in names(ca)) {
    e <- ca[[nm]]
    e$names <- NULL
    e$names_const <- NULL
    ca[nm] <- list(if (length(e) > 0L) e)
  }
  ca <- ca[!vapply(ca, is.null, logical(1))]
  set_tbl_gpu_field(out, "col_attrs", if (length(ca) > 0L) ca)
}

# Does a join of type `join_type` of `x` and `y` (a tbl_gpu) meet element names
# the GPU join can't give? Row-aligned names on either side (dplyr gathers them
# with the rows); a constant name (`names_const`) only on a side whose rows the
# join can null-extend (y's in left and full joins, x's in right and full
# joins), where dplyr gives the added rows `""`. Elsewhere every row of the
# column comes from a row of its side, which has the one name.
# @keywords internal
join_names_need_cpu <- function(x, y, join_type) {
  if ((!is_tbl_gpu(x) || is.null(.subset2(x, "col_attrs"))) &&
      (!is_tbl_gpu(y) || is.null(.subset2(y, "col_attrs")))) {
    return(FALSE)
  }
  length(join_names_cols(x, y, join_type)) > 0L
}

# The columns of `x` and `y` whose names send a join of type `join_type` to
# dplyr (`join_names_need_cpu()`).
# @keywords internal
join_names_cols <- function(x, y, join_type) {
  named <- function(t, null_extended) {
    ca <- if (is_tbl_gpu(t)) .subset2(t, "col_attrs")
    if (is.null(ca)) return(character())
    names(Filter(function(e) !is.null(e[["names"]]) || (null_extended && !is.null(e$names_const)), ca))
  }
  unique(c(named(x, join_type %in% c("right", "full")), named(y, join_type %in% c("left", "full"))))
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
      return(paste0(col_list_phrase(cols),
                    " element names or a matrix shape, which dplyr combines by its own rules."))
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
# columns y's. A merged key is dplyr's vec_cast() of x's key to the common
# type: it keeps x's entry when x's key already has that type (only y's side
# was cast), and has none when x's key was cast or is a factor (vctrs makes
# a new factor, even with the same levels). A join with row-aligned names,
# or a constant name on a side it null-extends, is routed to the CPU before
# it gets here (`join_names_need_cpu()`), so those are dropped, never carried.
# @keywords internal
join_col_attrs <- function(join_type, x, y, join_spec, suffix, keep, new_schema) {
  xa <- col_attrs_without_names(.subset2(x, "col_attrs"))
  if (join_type %in% c("semi", "anti")) {
    return(xa)
  }
  ya <- col_attrs_without_names(.subset2(y, "col_attrs"))
  x_schema <- current_schema(x)
  info <- build_join_output_info(x_schema, current_schema(y), join_spec,
                                 suffix = suffix, keep = keep)
  if (anyDuplicated(info$names) > 0L) {
    return(NULL)
  }
  # x's merged keys that come out without x's entry
  bare_keys <- character()
  if (!isTRUE(keep) && !is.null(xa)) {
    merged <- join_spec$left[join_spec$op == "=="]
    bare_keys <- intersect(merged, names(x_schema$factor_levels))
    for (cast in join_spec$key_casts) {
      if (isTRUE(cast$cast_left)) bare_keys <- c(bare_keys, join_spec$left[[cast$idx]])
    }
    bare_keys <- intersect(bare_keys, merged)
  }
  out <- lapply(seq_along(info$names), function(i) {
    if (identical(info$origin[[i]], "left")) {
      if (info$source_names[[i]] %in% bare_keys) return(NULL)
      return(xa[[info$source_names[[i]]]])
    }
    ya[[info$source_names[[i]]]]
  })
  names(out) <- info$names
  out <- out[!vapply(out, is.null, logical(1))]
  if (length(out) > 0L) out
}

# Has a constant value (a mutate() dot that read no column) attributes besides
# names that its column would keep in dplyr? (Its names go with the column,
# see the `const_names` of the mutate expression.) Only a class a column can
# hold counts: the `units` of a difftime belong to its class, and a value of
# a class the GPU has no type for is the type check's to report.
# @keywords internal
col_value_has_attrs <- function(value) {
  a <- attributes(value)
  if (is.null(a)) {
    return(FALSE)
  }
  holdable <- c("factor", "ordered", "Date", "POSIXct", "POSIXt", "integer64")
  all(oldClass(value) %in% holdable) && any(!names(a) %in% c("names", "class", "levels", "tzone"))
}

# Why a mutate()/transmute() whose `expressions` (the parsed dots) read a
# column of `attributed` (the names of columns with `$col_attrs` entries) in
# anything but a bare copy runs on dplyr: R's own rules decide what the result
# keeps. NULL when none does. A bare copy keeps the entry, so its output is
# attributed too; so is a constant with names (`const_names`), for the dots
# after it; any other output isn't.
# @keywords internal
col_attrs_read_reason <- function(attributed, expressions) {
  for (e in expressions) {
    ir <- .subset2(e, "ir")
    out <- .subset2(e, "output_col")
    if (identical(.subset2(ir, "kind"), "col")) {
      attributed <- if (.subset2(ir, "name") %in% attributed) union(attributed, out) else setdiff(attributed, out)
      next
    }
    if (!is.null(.subset2(e, "const_names"))) {
      attributed <- union(attributed, out)
      next
    }
    hit <- intersect(ir_cols(ir), attributed)
    if (length(hit) > 0L) {
      return(paste0(col_list_phrase(hit),
                    " names or attributes, which R carries through an expression by its own rules."))
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

# For each output column of `node`, which entry it takes: the position of the
# input column it is an unchanged copy of (a bare `b = a` copy included), or,
# for a mutate dot that makes a constant with names, `length(old_names)` plus
# that dot's index (the entry the node builds itself, `col_attrs_made()`); NA
# for a new or recomputed column, or a node that gives no column its input's
# values. The one column-origin map the per-column metadata and
# `$unspecified` ride on.
# @keywords internal
col_origin <- function(node, old_names, new_names) {
  none <- rep(NA_integer_, length(new_names))
  switch(.subset2(node, "type"),
    select = {
      pos <- .subset2(node, "positions") %||% match(.subset2(node, "columns"), old_names)
      if (length(pos) == length(new_names)) as.integer(pos) else none
    },
    rename = {
      # the columns keep their places; the node's new names map back to its old
      # ones (select(x = y) is one rename node over a fresh select, where `y`
      # is the column the output `x` is). A positional rename (duplicated
      # names) renames in place.
      if (!is.null(.subset2(node, "positions"))) {
        if (length(old_names) == length(new_names)) seq_along(new_names) else none
      } else {
        before <- new_names
        before[match(.subset2(node, "new_names"), new_names)] <- .subset2(node, "old_names")
        match(before, old_names)
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
      k <- 0L
      for (e in .subset2(node, "expressions")) {
        k <- k + 1L
        ir <- .subset2(e, "ir")
        src <- if (identical(.subset2(ir, "kind"), "col")) {
          cur[[.subset2(ir, "name")]]
        } else if (!is.null(.subset2(e, "const_names"))) {
          length(old_names) + k
        }
        cur[.subset2(e, "output_col")] <- list(src)
      }
      vapply(new_names, function(nm) as.integer(cur[[nm]] %||% NA_integer_), integer(1), USE.NAMES = FALSE)
    },
    distinct = match(new_names, old_names),
    summarise = {
      keys <- setdiff(.subset2(node, "groups"), summarise_output_names(node))
      origin <- match(new_names, keys)
      ifelse(is.na(origin), NA_integer_, match(new_names, old_names))
    },
    reframe = {
      # the group keys, repeated to the new size
      outs <- .subset2(node, "outputs")
      src <- vapply(outs, function(o) if (identical(.subset2(o, "kind"), "rep")) .subset2(o, "src") else NA_character_, "")
      match(src[match(new_names, vapply(outs, .subset2, "", "name"))], old_names)
    },
    none
  )
}

# The output names of a summarise node's aggregations.
# @keywords internal
summarise_output_names <- function(node) {
  vapply(.subset2(node, "aggregations"), function(a) .subset2(a, "output_col") %||% NA_character_, "")
}

# The entry a mutate dot that makes a constant with names gives its column:
# one name for every row when the constant is a scalar literal, one per row
# otherwise (a vector constant is the table's length). NULL for any other dot.
# @keywords internal
const_names_entry <- function(e) {
  cn <- .subset2(e, "const_names")
  if (is.null(cn)) {
    return(NULL)
  }
  if (length(cn) == 1L && identical(.subset2(.subset2(e, "ir"), "kind"), "lit")) {
    list(names_const = cn[[1L]])
  } else {
    list(names = cn)
  }
}

# `$col_attrs` of the output of `node`: each output column takes the entry of
# the column it is an unchanged copy of (`col_origin()`); the row-aligned names
# go only through the nodes that cannot move rows. A mutate adds the entries of
# its constants with names, a summarise those of quantile()'s names.
# @keywords internal
carry_col_attrs <- function(node, col_attrs, old_names, new_names) {
  type <- .subset2(node, "type")
  made <- switch(type,
    mutate = lapply(.subset2(node, "expressions"), const_names_entry),
    summarise = {
      q <- list()
      for (agg in .subset2(node, "aggregations")) {
        qn <- .subset2(agg, "qnames")
        if (!is.null(qn)) q[[.subset2(agg, "output_col")]] <- list(names_const = qn[[1L]])
      }
      q
    }
  )
  if (is.null(col_attrs) && !any(!vapply(made, is.null, logical(1)))) {
    return(NULL)
  }
  if (anyDuplicated(old_names) > 0L || anyDuplicated(new_names) > 0L) {
    return(NULL)
  }
  out <- list()
  if (!is.null(col_attrs) || identical(type, "mutate")) {
    origin <- col_origin(node, old_names, new_names)
    have <- if (is.null(col_attrs)) vector("list", length(old_names)) else unname(col_attrs[old_names])
    pool <- c(have, if (identical(type, "mutate")) unname(made))
    out <- pool[origin]
    names(out) <- new_names
    out <- out[!vapply(out, is.null, logical(1))]
    if (!type %in% c("select", "rename", "mutate", "window", "barrier")) {
      out <- col_attrs_without_names(out)
    }
  }
  if (identical(type, "summarise")) {
    out[names(made)] <- made
  }
  if (length(out) > 0L) out
}
