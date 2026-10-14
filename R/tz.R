# POSIXct time zone metadata (F32)
#
# The GPU holds every POSIXct as UTC microseconds (cudf has no tz-aware
# type); the column's `tzone` attribute is R-side metadata, carried in
# `schema$tz` the way `schema$factor_levels` carries factor labels:
#
#   schema$tz  a named character vector, column name -> zone, only for
#              TIMESTAMP_MICROSECONDS columns whose zone is not "UTC".
#              "" is a local-time column (tzone ""), NA_character_ a column
#              whose tzone attribute is absent. No entry means "UTC", what
#              every code path that does not know the zone has always
#              produced. NULL when every POSIXct column is UTC.
#
# With duplicated column names an entry belongs to the n-th
# TIMESTAMP_MICROSECONDS column (entries for ALL of them, in column order,
# tagged `by_position`), like `levels_by_position()`.
#
# Accessors (hour(), wday(), ...) on a column whose zone is not UTC-equivalent
# are computed by dplyr itself on the CPU (parse returns NULL, the existing
# whole-call fallback): R's tz database is the single source of truth for DST
# rules, and a wrong wall-clock hour on a DST day would be a silent error.

# Names that R's tz database treats as plain UTC (no offset changes ever).
tz_utc_names <- c("UTC", "GMT", "Etc/UTC", "Etc/GMT", "UCT", "Etc/UCT",
                  "Etc/Universal", "Universal", "Zulu", "Etc/Zulu", "GMT0",
                  "Etc/GMT0", "Etc/GMT+0", "Etc/GMT-0", "Greenwich",
                  "Etc/Greenwich")

# The zone R uses for a tz-less POSIXct: the TZ environment variable, else the
# system zone. `NA` when it can't be determined.
# @keywords internal
session_tz <- function() {
  tz <- Sys.getenv("TZ")
  if (nzchar(tz)) tz else Sys.timezone()
}

#' Is a tzone attribute value UTC-equivalent?
#'
#' `""` and `NA` (attribute absent) both mean local time, so they resolve to
#' the session zone first. An undeterminable session zone is not UTC.
#'
#' @param tz Character scalar: a zone, `""` or `NA`
#' @return `TRUE` when instants and wall-clock times coincide in `tz`
#' @keywords internal
tz_is_utc <- function(tz) {
  if (is.null(tz) || length(tz) != 1L) {
    return(FALSE)
  }
  if (is.na(tz) || !nzchar(tz)) {
    tz <- session_tz()
  }
  !is.na(tz) && tz %in% tz_utc_names
}

#' vctrs' common tzone of two POSIXct operands
#'
#' `vec_ptype2()` keeps `x`'s zone unless it is local (`""`/absent), then
#' takes `y`'s. `NULL` stands for "no POSIXct operand" (a Date or a
#' non-timestamp) and is skipped.
#'
#' @param x,y Zones as stored in `schema$tz` (`""`, `NA`, a name), or `NULL`
#' @return The common zone, or `NULL` when both are `NULL`
#' @keywords internal
tz_fold <- function(x, y) {
  if (is.null(x)) return(y)
  if (is.null(y)) return(x)
  if (is.na(x) || !nzchar(x)) y else x
}

# A column's tzone attribute as a schema value: absent -> NA.
# @keywords internal
tz_of_attr <- function(col) {
  tz <- attr(col, "tzone", exact = TRUE)
  if (is.null(tz)) NA_character_ else tz[[1L]]
}

# An absent tzone as dplyr/vctrs leave it after combining or slicing values:
# "" (local time). Only select/rename/relocate, a bare copy and `t +/- n`
# keep the attribute absent. Vectorised over a character vector of zones.
# @keywords internal
tz_localize <- function(zone) {
  zone[is.na(zone)] <- ""
  zone
}

#' Do a table's POSIXct zones already match a target schema's?
#'
#' The metadata counterpart of `factor_levels_match_target()`: the GPU values
#' are UTC instants either way, but a set operation has to hand back the
#' target's zones. An absent tzone and `""` count as equal (slicing turns the
#' former into the latter).
#'
#' @param schema,target_schema Schemas with the same column names and types
#' @return `TRUE` when every column has the same zone
#' @keywords internal
tz_matches_target <- function(schema, target_schema) {
  identical(tz_localize(tz_by_position(schema)), tz_localize(tz_by_position(target_schema)))
}

#' Zone of every column of a schema, by position
#'
#' @param schema A schema with `names`, `types` and (maybe) `tz`
#' @return A character vector the length of `schema$names`: the zone of a
#'   TIMESTAMP_MICROSECONDS column (`"UTC"` when it has no entry), `"UTC"`
#'   for every other column
#' @keywords internal
tz_by_position <- function(schema) {
  nms <- schema$names
  out <- rep("UTC", length(nms))
  tz <- schema$tz
  if (is.null(tz) || length(nms) == 0L) {
    return(out)
  }
  ts <- which(unname(schema$types) == "TIMESTAMP_MICROSECONDS")
  if (!anyDuplicated(nms)) {
    hit <- match(names(tz), nms)
    ok <- !is.na(hit) & unname(schema$types)[hit] == "TIMESTAMP_MICROSECONDS"
    ok[is.na(ok)] <- FALSE
    out[hit[ok]] <- unname(tz)[ok]
    return(out)
  }
  if (!isTRUE(attr(tz, "by_position")) || length(tz) != length(ts)) {
    rlang::abort(c(
      "Internal error: time zones of duplicated column names are not tracked by position.",
      i = "Please report this to the cuplyr maintainers."
    ), class = "cuplyr_error_internal")
  }
  out[ts] <- unname(tz)
  out
}

#' Rebuild `schema$tz` from per-position zones
#'
#' The inverse of `tz_by_position()`.
#'
#' @param names,types Output column names and GPU types
#' @param by_pos Character vector the length of `names` (`"UTC"` for UTC and
#'   non-timestamp columns)
#' @return A named character vector, or `NULL` when every column is UTC
#' @keywords internal
tz_from_positions <- function(names, types, by_pos) {
  ts <- which(unname(types) == "TIMESTAMP_MICROSECONDS")
  if (length(ts) == 0L) {
    return(NULL)
  }
  tz <- by_pos[ts]
  non_utc <- is.na(tz) | tz != "UTC"
  if (!any(non_utc)) {
    return(NULL)
  }
  if (anyDuplicated(names) > 0L) {
    out <- stats::setNames(tz, names[ts])
    attr(out, "by_position") <- TRUE
    return(out)
  }
  stats::setNames(tz[non_utc], names[ts][non_utc])
}

#' Zone of one named column of a (uniquely named) schema
#'
#' @param schema A schema
#' @param name Column name
#' @return The zone string, `"UTC"` for a column without an entry or a
#'   non-timestamp column
#' @keywords internal
schema_tz_of <- function(schema, name) {
  tz <- schema$tz
  if (is.null(tz)) {
    return("UTC")
  }
  idx <- match(name, schema$names)
  if (is.na(idx)) {
    return("UTC")
  }
  tz_by_position(schema)[[idx]]
}

# Set (or, for "UTC", drop) one column's entry in a running, uniquely named
# schema's `tz`, leaving no empty vector behind.
# @keywords internal
set_schema_tz <- function(schema, name, zone) {
  tz <- schema$tz
  if (!is.null(tz)) {
    tz <- tz[names(tz) != name]
  }
  if (is.na(zone) || zone != "UTC") {
    tz <- c(tz, stats::setNames(zone, name))
  }
  schema$tz <- if (length(tz) == 0L) NULL else tz
  schema
}

# Append columns to a schema, zones included (`tz` per new column, "UTC" for
# non-timestamp ones).
# @keywords internal
append_schema_cols <- function(schema, names, types, tz = rep("UTC", length(names))) {
  by_pos <- c(tz_by_position(schema), tz)
  schema$names <- c(schema$names, names)
  schema$types <- c(schema$types, types)
  schema$tz <- tz_from_positions(schema$names, schema$types, by_pos)
  schema
}

#' Output zones of a node, by position
#'
#' The `schema$tz` of `node`'s output, from its input schema's. The node-wise
#' sibling of `propagate_factor_levels()`: row and column nodes carry zones
#' with their columns; mutate/window/summarise compute the zone of each new
#' POSIXct column (`ir_infer_tz()`; an aggregate or window op over a POSIXct
#' column keeps that column's zone). Joins and binds build their own schema
#' (`build_join_schema()`, `compute_unified_schema()`).
#'
#' @param node The AST node being applied
#' @param input_schema The schema of the op's input (pre-op)
#' @param new_schema The node's output `names`/`types` (from `infer_schema()`)
#' @return A `schema$tz` value
#' @keywords internal
propagate_tz <- function(node, input_schema, new_schema) {
  if (is.null(input_schema$tz) && !identical(node$type, "mutate")) {
    return(NULL)
  }
  nms <- input_schema$names

  if (identical(node$type, "mutate")) {
    s <- input_schema
    for (expr in node$expressions) {
      s <- update_schema_for_expr(s, expr)
    }
    return(s$tz)
  }

  by_pos <- tz_by_position(input_schema)
  if (identical(node$type, "window")) {
    spec_tz <- vapply(node$specs, function(spec) {
      if (!identical(unname(spec$output_type), "TIMESTAMP_MICROSECONDS") ||
          is.na(spec$value_col)) {
        return("UTC")
      }
      tz_localize(schema_tz_of(input_schema, spec$value_col))
    }, character(1))
    return(tz_from_positions(new_schema$names, new_schema$types,
                             c(by_pos, unname(spec_tz))))
  }
  if (identical(node$type, "reframe")) {
    # each output takes the zone of its (first) wide source column; a literal
    # is never a POSIXct
    out <- vapply(node$outputs, function(o) {
      if (identical(o$kind, "literal")) "UTC" else by_pos[[match(o$src[[1]], nms)]]
    }, character(1))
    return(tz_from_positions(new_schema$names, new_schema$types, out))
  }
  if (identical(node$type, "summarise")) {
    key_tz <- tz_localize(by_pos[match(node$groups, nms)])
    agg_tz <- vapply(node$aggregations, function(agg) {
      if (!identical(unname(agg$output_type), "TIMESTAMP_MICROSECONDS") ||
          is.na(agg$input_col)) {
        return("UTC")
      }
      tz_localize(schema_tz_of(input_schema, agg$input_col))
    }, character(1))
    agg_names <- vapply(node$aggregations, `[[`, character(1), "output_col")
    out <- c(key_tz, unname(agg_tz))[summarise_output_index(node$groups, agg_names)]
    return(tz_from_positions(new_schema$names, new_schema$types, out))
  }

  # Row-slicing nodes rebuild the column through vctrs: absent becomes "".
  slices_rows <- node$type %in% c("filter", "arrange", "slice", "distinct")

  if (anyDuplicated(nms) > 0L || anyDuplicated(new_schema$names) > 0L) {
    # Duplicated names: the zones follow their column by position, the way the
    # select/rename itself addresses it (as propagate_factor_levels()).
    out <- switch(node$type,
      select = by_pos[node$positions %||% match(node$columns, nms)],
      distinct = if (isTRUE(node$keep_all)) {
        by_pos
      } else {
        by_pos[node$key_positions %||% match(node$key_cols, nms)]
      },
      by_pos  # rename, slice, filter, arrange, barrier: columns stay in place
    )
    if (length(out) != length(new_schema$names)) {
      rlang::abort("Internal error: could not carry POSIXct zones through a positional node.",
                   class = "cuplyr_error_internal")
    }
    if (slices_rows) out <- tz_localize(out)
    return(tz_from_positions(new_schema$names, new_schema$types, out))
  }

  # Unique names: by name, through a rename's old -> new mapping.
  tz <- input_schema$tz
  if (identical(node$type, "rename")) {
    nm <- names(tz)
    idx <- match(nm, node$old_names)
    renamed <- !is.na(idx)
    nm[renamed] <- node$new_names[idx[renamed]]
    names(tz) <- nm
  }
  hit <- match(new_schema$names, names(tz))
  out <- rep("UTC", length(new_schema$names))
  out[!is.na(hit)] <- unname(tz)[hit[!is.na(hit)]]
  if (slices_rows) out <- tz_localize(out)
  tz_from_positions(new_schema$names, new_schema$types, out)
}

# `infer_schema()` plus the metadata that rides on a schema (factor levels and
# zones), for a node whose input schema is known.
# @keywords internal
propagate_schema_meta <- function(node, input_schema, new_schema) {
  new_schema$factor_levels <- propagate_factor_levels(node, input_schema, new_schema$names)
  new_schema$tz <- propagate_tz(node, input_schema, new_schema)
  new_schema
}

# Give the POSIXct columns of a collected tibble their schema zone (the C++
# collect tags every one "UTC").
# @keywords internal
restore_tz <- function(result, schema) {
  if (!any(unname(schema$types) == "TIMESTAMP_MICROSECONDS")) {
    return(result)
  }
  by_pos <- tz_by_position(schema)
  for (i in which(unname(schema$types) == "TIMESTAMP_MICROSECONDS")) {
    zone <- by_pos[[i]]
    if (!is.na(zone) && zone == "UTC") next  # what the C++ collect tags already
    col <- result[[i]]
    if (is.na(zone)) {
      attr(col, "tzone") <- NULL
    } else {
      attr(col, "tzone") <- zone
    }
    result[[i]] <- col
  }
  result
}

# One preview column (gpu_head()) as display strings' source: a POSIXct gets
# its schema zone, and dates/timestamps are formatted so that the NA
# replacement the printers do is a character assignment.
# @keywords internal
preview_values <- function(vals, zone) {
  if (inherits(vals, "POSIXct")) {
    attr(vals, "tzone") <- if (is.na(zone)) NULL else zone
  }
  if (inherits(vals, c("POSIXct", "Date"))) {
    vals <- format(vals)
  }
  vals
}
