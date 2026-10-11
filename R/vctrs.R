# vctrs contract for tbl_gpu (F02)
#
# vctrs treats an S3 list with no `vec_proxy()` method as a scalar, so every
# vctrs-based helper (vec_size(), vec_slice(), vec_c(), dplyr::coalesce(),
# n_distinct(), consecutive_id(), if_else(), ...) used to fail on a whole
# tbl_gpu. This file gives tbl_gpu a proper contract.
#
# The proxy is a ROW-HANDLE frame, not the data: a two-column data frame with
# one row per table row, `.src` (integer id of the source table) and `.row`
# (1-based row in that table). `.row` is R's compact `seq_len(n)` and `.src` is
# a constant ALTREP vector (`gpu_handle_src()`, src/ops_gather.cpp), so
# vec_size() is O(1) host work and touches the GPU only for `gpu_dim()`
# metadata. vctrs slices / combines / assigns handle frames with its ordinary
# data-frame machinery; `vec_restore.tbl_gpu()` then decodes the handles and
# materializes the result with ONE GPU gather (`gpu_gather_rows()`),
# concatenating the referenced source tables first if the handles name several
# of them (this is how `vec_c(a, b)`, `vec_assign(a, i, b)`, `coalesce(a, b)`
# ... work). NA handles become all-null rows.
#
# Lifetime of sources. The registry maps id -> source table through WEAK
# references, so by itself it never keeps GPU memory alive, and ids are never
# reused within a session: a handle can only ever resolve to the table it was
# made from, or fail loudly (`cuplyr_error_vctrs_dead_source`). That alone is
# not enough, because vctrs copies the handle values out of the proxy and
# often drops the table they came from before it restores (casts and NA fills
# inside `vec_c()`, a lazy table's computed result). So `vec_proxy()` also PINS
# the source strongly until the vctrs entry point that is running it returns
# (`vctrs_entry_frame()`: the innermost non-generic vctrs closure on the call
# stack, e.g. `vec_c`, `vec_slice`, `vec_assign`, `list_unchop`). After that
# only the weak reference is left. What is NOT guaranteed: handles that are
# kept past the vctrs call that made them (a bare `vec_proxy(x)` result kept
# around) stay valid only while the caller keeps the source table (or, for a
# lazy table, the proxy vector itself, which holds its computed table) alive.
#
# Value-based generics (equality / ordering) collect: they are per-element
# over the whole table, so a CPU result is the right answer there.
#
# A lazy table with pending ops is executed (`compute()`) by vec_proxy(); the
# caller's object is not modified. Within one vctrs entry call the computed
# table is reused (`vctrs_computed()`); a later vctrs call recomputes.
#
# Unsupported: calling `vctrs::vec_rbind()` / `vec_cbind()` (and so
# `dplyr::bind_rows()` / `dplyr::bind_cols()`) on a tbl_gpu directly. They treat
# a non-data-frame vector as a row/column and cannot be intercepted by S3
# dispatch; `vec_proxy.tbl_gpu()` raises a clear error there. Use
# `cuplyr::bind_rows()` / `cuplyr::bind_cols()`.

vctrs_max_source_id <- .Machine$integer.max

vctrs_state <- new.env(parent = emptyenv())

vctrs_state_init <- function() {
  if (is.null(vctrs_state$refs)) {
    vctrs_state$refs <- new.env(parent = emptyenv())      # id -> entry
    vctrs_state$by_addr <- new.env(parent = emptyenv())   # pointer address -> id
    vctrs_state$pins <- new.env(parent = emptyenv())      # id -> list(tbl, n)
    vctrs_state$next_id <- 1
    vctrs_state$n_entries <- 0L
    vctrs_state$sweep_at <- 256L
  }
  invisible(vctrs_state)
}

# Structural identity of a schema for gather purposes: column types and
# factor levels by position (names are irrelevant to the physical gather).
vctrs_schema_sig <- function(schema) {
  list(
    types = unname(as.character(schema$types)),
    levels = levels_by_position(schema)
  )
}

# Drop registry entries whose source table has been garbage collected. Ids are
# NOT recycled: a stale handle must never resolve to a different table.
vctrs_sweep <- function() {
  st <- vctrs_state_init()
  for (key in ls(st$refs, all.names = TRUE)) {
    entry <- st$refs[[key]]
    if (is.null(rlang::wref_key(entry$wref)) && is.null(st$pins[[key]])) {
      rm(list = key, envir = st$refs)
      if (identical(st$by_addr[[entry$addr]], entry$id)) {
        rm(list = entry$addr, envir = st$by_addr)
      }
      st$n_entries <- st$n_entries - 1L
    }
  }
  st$sweep_at <- max(256L, 2L * st$n_entries)
  invisible(NULL)
}

vctrs_new_id <- function() {
  st <- vctrs_state_init()
  if (st$n_entries >= st$sweep_at) {
    vctrs_sweep()
  }
  if (st$next_id > vctrs_max_source_id) {
    rlang::abort(
      "Too many tbl_gpu sources registered with vctrs in this session (2^31 - 1).",
      class = "cuplyr_error_vctrs_registry_full"
    )
  }
  id <- st$next_id
  st$next_id <- id + 1
  id
}

# Register a materialized table (non-NULL `ptr`) and return its source id.
vctrs_register_source <- function(tbl) {
  st <- vctrs_state_init()
  ptr <- tbl$ptr
  addr <- rlang::obj_address(ptr)
  sig <- vctrs_schema_sig(tbl$schema)

  id <- st$by_addr[[addr]]
  if (!is.null(id)) {
    entry <- st$refs[[as.character(id)]]
    live <- if (is.null(entry)) NULL else rlang::wref_key(entry$wref)
    if (!is.null(live) && identical(rlang::obj_address(live), addr)) {
      entry$sig <- sig
      st$refs[[as.character(id)]] <- entry
      return(id)
    }
  }

  id <- vctrs_new_id()
  st$refs[[as.character(id)]] <- list(
    id = id, addr = addr, sig = sig, wref = rlang::new_weakref(ptr)
  )
  st$by_addr[[addr]] <- id
  st$n_entries <- st$n_entries + 1L
  id
}

# id -> list(ptr, sig, n) with `ptr` a STRONG reference that is verified to
# still hold a GPU table; aborts if the source is gone. A weak reference can
# still return a key whose finalizer has already released the GPU table (the
# key is cleared one collection later), so liveness is checked on the pointer
# itself, and callers use the returned reference, never the weak one again.
vctrs_lookup_source <- function(id) {
  st <- vctrs_state_init()
  entry <- st$refs[[as.character(id)]]
  ptr <- if (is.null(entry)) NULL else rlang::wref_key(entry$wref)
  dims <- if (is.null(ptr)) NULL else tryCatch(gpu_dim(ptr), error = function(e) NULL)
  if (is.null(dims)) {
    rlang::abort(
      c(
        "A tbl_gpu row handle refers to a table that no longer exists.",
        i = "Keep the source tbl_gpu alive while its vctrs proxy is in use."
      ),
      class = "cuplyr_error_vctrs_dead_source"
    )
  }
  list(ptr = ptr, sig = entry$sig, n = as.double(dims[[1L]]))
}

# --- strong pins scoped to a vctrs call ------------------------------------

# vctrs generics (UseMethod dispatchers). Their frames end as soon as the
# method returns, so they are never the scope a pin should live in.
vctrs_generic_names <- c(
  "vec_proxy", "vec_restore", "vec_cast", "vec_ptype2", "vec_proxy_equal",
  "vec_proxy_compare", "vec_proxy_order", "vec_ptype_abbr", "vec_ptype_full",
  "vec_proxy_assign", "obj_print_data"
)

# Index (into sys.frames()) of the innermost non-generic vctrs closure on the
# stack below frame `from`, or 0L if there is none. vctrs' C code calls S3
# methods through a throwaway zero-argument closure (`function() vec_proxy(x =
# x)`) and `vec_*_dispatch()` helpers; those frames end with the method call,
# so they are skipped like the generics.
vctrs_entry_frame <- function(from) {
  if (from <= 1L) {
    return(0L)
  }
  ns <- asNamespace("vctrs")
  generics <- NULL
  for (i in seq.int(from - 1L, 1L)) {
    fn <- sys.function(i)
    env <- environment(fn)
    if (is.null(env) || !identical(topenv(env), ns) || length(formals(fn)) == 0L) {
      next
    }
    head <- sys.call(i)[[1L]]
    if (is.symbol(head) && grepl("dispatch", as.character(head), fixed = TRUE)) {
      next
    }
    if (is.call(head) && length(head) == 3L && is.symbol(head[[3L]]) &&
        grepl("dispatch", as.character(head[[3L]]), fixed = TRUE)) {
      next
    }
    if (is.null(generics)) {
      generics <- lapply(vctrs_generic_names, function(nm) ns[[nm]])
    }
    if (!any(vapply(generics, identical, logical(1), fn))) {
      return(i)
    }
  }
  0L
}

vctrs_unpin <- function(id) {
  st <- vctrs_state_init()
  key <- as.character(id)
  pin <- st$pins[[key]]
  if (is.null(pin)) {
    return(invisible(NULL))
  }
  if (pin$n <= 1L) {
    rm(list = key, envir = st$pins)
  } else {
    pin$n <- pin$n - 1L
    st$pins[[key]] <- pin
  }
  invisible(NULL)
}

# Hold `tbl` strongly until frame `frame` (an environment from sys.frame())
# returns, on any exit path.
vctrs_pin_until_exit <- function(id, tbl, frame) {
  st <- vctrs_state_init()
  key <- as.character(id)
  pin <- st$pins[[key]]
  st$pins[[key]] <- list(tbl = tbl, n = if (is.null(pin)) 1L else pin$n + 1L)
  release <- as.call(list(vctrs_unpin, id))
  do.call(base::on.exit, list(release, add = TRUE), envir = frame)
  invisible(NULL)
}

# A column-less all-null table of `n` rows with `schema`'s physical layout.
vctrs_null_table_ptr <- function(schema, n) {
  cols <- lapply(seq_along(schema$names), function(i) {
    gpu_make_null_column(n, gpu_physical_type(unname(schema$types[i])))
  })
  if (length(cols) == 1L) cols[[1L]] else gpu_bind_cols_impl(cols)
}

# 0-row tibble with the R types a collect() of `x` would produce.
gpu_proto <- function(x) {
  schema <- current_schema(x)
  if (length(schema$names) == 0L) {
    return(tibble::tibble())
  }
  ptr <- vctrs_null_table_ptr(schema, 0L)
  collect_frame(new_tbl_gpu(ptr = ptr, schema = schema))
}

# ---------------------------------------------------------------------------
# proxy / restore

#' @exportS3Method vctrs::vec_proxy
vec_proxy.tbl_gpu <- function(x, ...) {
  entry <- vctrs_entry_frame(sys.nframe())
  if (entry > 0L && is_vctrs_bind_entry(sys.function(entry))) {
    rlang::abort(
      c(
        "Can't combine a tbl_gpu with `vec_rbind()`/`vec_cbind()` (so not with `dplyr::bind_rows()`/`dplyr::bind_cols()` either).",
        i = "Use `cuplyr::bind_rows()` / `cuplyr::bind_cols()`, which stay on the GPU."
      ),
      class = "cuplyr_error_vctrs_bind_unsupported"
    )
  }

  m <- if (has_pending_ops(x)) vctrs_computed(x, entry) else x
  if (is.null(m$ptr)) {
    rlang::abort("Cannot take the vctrs proxy of a tbl_gpu with no GPU table.")
  }
  # The proxy's size is cudf's row count, which is 0 for a zero-column table
  # whatever its R-side `$nrow` says (F07; full support is F14).
  check_zero_column_input(m, m$schema, "vctrs::vec_proxy")
  n <- gpu_dim(m$ptr)[[1L]]
  id <- vctrs_register_source(m)
  if (entry > 0L) {
    vctrs_pin_until_exit(id, m, sys.frame(entry))
  }
  vctrs::new_data_frame(
    list(.src = gpu_handle_src(id, n, m), .row = seq_len(n)),
    n = n
  )
}

# compute() a lazy table for vec_proxy(), once per vctrs entry call: vctrs
# proxies the same input many times within one call (list_sizes(rep(list(lz),
# 40)), vec_c(lz, lz)), and every pinned recompute would hold its own copy of
# the result until the call returns. The memo lives exactly as long as the
# pins (it is dropped when frame `entry` exits) and holds `x$ptr` and
# `x$lazy_ops`, so their addresses can't be reused by other objects while it
# exists: the same key means the same source and the same pending ops.
vctrs_computed <- function(x, entry) {
  if (entry <= 0L) {
    return(compute(x))
  }
  st <- vctrs_state_init()
  if (is.null(st$memo)) {
    st$memo <- new.env(parent = emptyenv())
  }
  frame <- sys.frame(entry)
  key <- paste(
    rlang::obj_address(frame), rlang::obj_address(x$ptr),
    rlang::obj_address(x$lazy_ops), x$exec_mode
  )
  hit <- st$memo[[key]]
  if (!is.null(hit)) {
    return(hit$m)
  }
  m <- compute(x)
  st$memo[[key]] <- list(m = m, ptr = x$ptr, ops = x$lazy_ops, frame = frame)
  release <- as.call(list(vctrs_memo_drop, key))
  do.call(base::on.exit, list(release, add = TRUE), envir = frame)
  m
}

vctrs_memo_drop <- function(key) {
  memo <- vctrs_state$memo
  if (!is.null(memo) && exists(key, envir = memo, inherits = FALSE)) {
    rm(list = key, envir = memo)
  }
  invisible(NULL)
}

is_vctrs_bind_entry <- function(fn) {
  ns <- asNamespace("vctrs")
  identical(fn, ns$vec_rbind) || identical(fn, ns$vec_cbind)
}

is_handle_frame <- function(x) {
  is.data.frame(x) && identical(names(x), c(".src", ".row")) &&
    is.integer(x[[1L]]) && is.integer(x[[2L]])
}

restore_from_data <- function(x, to) {
  out <- tbl_gpu(tibble::as_tibble(x), lazy = identical(to$exec_mode, "lazy"))
  out$groups <- intersect(to$groups, out$schema$names)
  out$group_drop <- if (length(out$groups) > 0L) to$group_drop %||% TRUE else TRUE
  out
}

#' @exportS3Method vctrs::vec_restore
vec_restore.tbl_gpu <- function(x, to, ...) {
  if (!is_handle_frame(x)) {
    return(restore_from_data(x, to))
  }

  schema <- current_schema(to)
  schema <- list(
    names = schema$names,
    types = schema$types,
    factor_levels = schema$factor_levels
  )
  src <- x[[1L]]
  row <- x[[2L]]
  n <- length(src)

  valid <- !is.na(src) & !is.na(row)
  if (!any(valid)) {
    ptr <- vctrs_null_table_ptr(schema, n)
  } else {
    ids <- unique(src[valid])
    # Strong references for the rest of this function: they, not the weak
    # registry entries, are what the gather below uses.
    sources <- lapply(ids, vctrs_lookup_source)

    want <- vctrs_schema_sig(schema)
    for (s in sources) {
      if (!identical(s$sig, want)) {
        rlang::abort(
          "Internal error: tbl_gpu row handles come from a table whose layout differs from the restore target.",
          class = "cuplyr_error_vctrs_schema_mismatch"
        )
      }
    }

    ptrs <- lapply(sources, function(s) s$ptr)
    sizes <- vapply(sources, function(s) s$n, numeric(1))
    offsets <- c(0, cumsum(sizes))[seq_along(ids)]
    pos <- match(src, ids)
    if (any(row[valid] < 1L | row[valid] > sizes[pos[valid]])) {
      rlang::abort("Internal error: tbl_gpu row handle out of range.")
    }
    gidx <- offsets[pos] + row - 1
    gidx[!valid] <- NA_real_
    ptr <- wrap_gpu_call("vec_restore", gpu_gather_rows(ptrs, gidx))
  }

  new_tbl_gpu(
    ptr = ptr,
    schema = schema,
    groups = to$groups,
    exec_mode = to$exec_mode,
    group_drop = to$group_drop %||% TRUE
  )
}

# ---------------------------------------------------------------------------
# value-based proxies: per-element over the whole table, so they collect

collected_frame <- function(x, what) {
  cuplyr_fallback_notify("vctrs", what)
  as.data.frame(collect_frame(x))
}

#' @exportS3Method vctrs::vec_proxy_equal
vec_proxy_equal.tbl_gpu <- function(x, ...) collected_frame(x, "vec_proxy_equal() of a tbl_gpu")

#' @exportS3Method vctrs::vec_proxy_compare
vec_proxy_compare.tbl_gpu <- function(x, ...) collected_frame(x, "vec_proxy_compare() of a tbl_gpu")

#' @exportS3Method vctrs::vec_proxy_order
vec_proxy_order.tbl_gpu <- function(x, ...) collected_frame(x, "vec_proxy_order() of a tbl_gpu")

# ---------------------------------------------------------------------------
# common type: tbl_gpu plays the role grouped_df plays for tibbles. The
# result is a 0-row tbl_gpu so combining with a GPU table stays on the GPU.

proto_of <- function(x) {
  if (is_tbl_gpu(x)) {
    gpu_proto(x)
  } else if (inherits(x, "grouped_df")) {
    vctrs::vec_ptype(dplyr::ungroup(x))
  } else {
    vctrs::vec_ptype(x)
  }
}

groups_of <- function(x) {
  if (is_tbl_gpu(x)) {
    x$groups
  } else if (inherits(x, "grouped_df")) {
    dplyr::group_vars(x)
  } else {
    character()
  }
}

mode_of <- function(x) {
  if (is_tbl_gpu(x) && identical(x$exec_mode, "lazy")) "lazy" else "eager"
}

ptype2_tbl_gpu <- function(x, y, ..., x_arg = "", y_arg = "", call = rlang::caller_env()) {
  proto <- vctrs::vec_ptype2(
    proto_of(x), proto_of(y), x_arg = x_arg, y_arg = y_arg, call = call
  )
  lazy <- identical(mode_of(x), "lazy") || identical(mode_of(y), "lazy")
  out <- tbl_gpu(tibble::as_tibble(proto), lazy = lazy)
  out$groups <- intersect(union(groups_of(x), groups_of(y)), out$schema$names)
  out
}

#' @exportS3Method vctrs::vec_ptype2 tbl_gpu.tbl_gpu
vec_ptype2.tbl_gpu.tbl_gpu <- ptype2_tbl_gpu
#' @exportS3Method vctrs::vec_ptype2 tbl_gpu.tbl_df
vec_ptype2.tbl_gpu.tbl_df <- ptype2_tbl_gpu
#' @exportS3Method vctrs::vec_ptype2 tbl_gpu.data.frame
vec_ptype2.tbl_gpu.data.frame <- ptype2_tbl_gpu
#' @exportS3Method vctrs::vec_ptype2 tbl_gpu.grouped_df
vec_ptype2.tbl_gpu.grouped_df <- ptype2_tbl_gpu
#' @exportS3Method vctrs::vec_ptype2 tbl_df.tbl_gpu
vec_ptype2.tbl_df.tbl_gpu <- ptype2_tbl_gpu
#' @exportS3Method vctrs::vec_ptype2 data.frame.tbl_gpu
vec_ptype2.data.frame.tbl_gpu <- ptype2_tbl_gpu
#' @exportS3Method vctrs::vec_ptype2 grouped_df.tbl_gpu
vec_ptype2.grouped_df.tbl_gpu <- ptype2_tbl_gpu

# ---------------------------------------------------------------------------
# casting

cast_gpu_to_gpu <- function(x, to, ..., x_arg = "", to_arg = "", call = rlang::caller_env()) {
  sx <- current_schema(x)
  st <- current_schema(to)
  if (identical(unname(sx$names), unname(st$names)) &&
      identical(vctrs_schema_sig(sx), vctrs_schema_sig(st))) {
    # Same layout: no GPU work, only `to`'s grouping metadata.
    x$groups <- to$groups
    x$group_drop <- to$group_drop %||% TRUE
    return(x)
  }
  cuplyr_fallback_notify("vctrs", "vec_cast() between tbl_gpu layouts")
  casted <- vctrs::vec_cast(
    collect_frame(x), gpu_proto(to), x_arg = x_arg, to_arg = to_arg, call = call
  )
  restore_from_data(casted, to)
}

cast_df_to_gpu <- function(x, to, ..., x_arg = "", to_arg = "", call = rlang::caller_env()) {
  if (inherits(x, "grouped_df")) {
    x <- dplyr::ungroup(x)
  }
  casted <- vctrs::vec_cast(
    tibble::as_tibble(x), gpu_proto(to), x_arg = x_arg, to_arg = to_arg, call = call
  )
  restore_from_data(casted, to)
}

cast_gpu_to_df <- function(x, to, ..., x_arg = "", to_arg = "", call = rlang::caller_env()) {
  if (inherits(to, "grouped_df")) {
    to <- dplyr::ungroup(to)
  }
  vctrs::vec_cast(collect_frame(x), to, x_arg = x_arg, to_arg = to_arg, call = call)
}

#' @exportS3Method vctrs::vec_cast tbl_gpu.tbl_gpu
vec_cast.tbl_gpu.tbl_gpu <- cast_gpu_to_gpu
#' @exportS3Method vctrs::vec_cast tbl_gpu.tbl_df
vec_cast.tbl_gpu.tbl_df <- cast_df_to_gpu
#' @exportS3Method vctrs::vec_cast tbl_gpu.data.frame
vec_cast.tbl_gpu.data.frame <- cast_df_to_gpu
#' @exportS3Method vctrs::vec_cast tbl_gpu.grouped_df
vec_cast.tbl_gpu.grouped_df <- cast_df_to_gpu
#' @exportS3Method vctrs::vec_cast tbl_df.tbl_gpu
vec_cast.tbl_df.tbl_gpu <- cast_gpu_to_df
#' @exportS3Method vctrs::vec_cast data.frame.tbl_gpu
vec_cast.data.frame.tbl_gpu <- cast_gpu_to_df
#' @exportS3Method vctrs::vec_cast grouped_df.tbl_gpu
vec_cast.grouped_df.tbl_gpu <- cast_gpu_to_df

# ---------------------------------------------------------------------------
# length

#' @export
length.tbl_gpu <- function(x) {
  length(current_schema(x)$names)
}
