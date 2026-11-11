# Pushdown into a parquet scan (F85, phase B)
#
# `push_into_scan()` is the last pass of optimize_segment(): a filter() sitting
# directly on a scan gives the reader the conjuncts that libcudf's AST filter can
# decide exactly, and keeps the rest above the scan. Column projection is
# push_down_projections()'s source case (scan_project()).
#
# A wrong row set is the worst outcome of this feature, so a conjunct is pushed
# only as one of a few atoms whose meaning in the reader is the meaning in R
# (see scan_atom()), and anything doubtful stays where it was.
#
# What the reader evaluates is the AST on the FILE's values: no cast, no
# normalisation, no NA sentinel. So an atom is pushed only on a column whose
# schema type is its file type (no cast column, no factor), against a literal
# that converts to that type exactly, and the NA sentinel of INT32/INT64
# (R/io.R reads it as missing, but the reader sees the raw value) is excluded by
# an extra `col > MIN` atom wherever the comparison could admit it.
#
# The reader decides `==` in up to three stages: min/max statistics, a Bloom
# filter if the file has one, and the row-by-row AST. A Bloom filter holds the
# BITS of each value, so an `==` atom is exact only if every value R calls equal
# to the literal has the literal's bits.
#
# Only integer, integer64, Date and POSIXct columns are pushed; doubles and
# strings run above the scan. libcudf 25.12 takes a NaN in a row group's footer
# min or max for a bound and skips the group, losing its rows (the Parquet spec
# says to ignore such a statistic), so no FLOAT64 comparison is pushed. For a
# string column it takes the footer's deprecated min/max fields, which hold
# signed byte order, as byte-order bounds, so a row group can be skipped wrongly
# for a string comparison too. Both filters give the same rows above the scan;
# only the row-group skipping is given up.

# The pushdown option: `cuplyr.parquet_pushdown = FALSE` turns off both the
# projection and the predicate pushdown.
scan_pushdown_enabled <- function() {
  !identical(getOption("cuplyr.parquet_pushdown", TRUE), FALSE)
}

# File types the reader can be trusted to compare without any of our
# normalisation (not FLOAT64 or STRING: see the top of this file).
scan_pushable_types <- c("INT32", "INT64", "TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")

# The scan record of a node, or NULL.
node_scan <- function(node) {
  if (identical(.subset2(node, "type"), "source")) .subset2(node, "scan")
}

# -----------------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------------

# The scan node reading only `keep` (file order of the node's schema).
scan_project <- function(node, keep) {
  scan <- node$scan
  node$schema <- parquet_schema_for(keep, list(names = scan$file_names, types = scan$file_types),
                                    scan$meta)
  scan$columns <- keep
  node$scan <- scan
  node
}

# The column a plan that reads none still has to read, for its row count: the
# first one R can hold (a LIST column would be an error), else the first.
scan_row_count_column <- function(node) {
  scan <- node$scan
  nms <- node$schema$names
  types <- scan$file_types[match(nms, scan$file_names)]
  ok <- which(vapply(types, parquet_type_supported, logical(1)))
  nms[[if (length(ok) > 0L) ok[[1L]] else 1L]]
}

# -----------------------------------------------------------------------------
# Predicates
# -----------------------------------------------------------------------------

# The conjuncts of `ir` split at its top-level `&`s, or NULL when `ir` isn't a
# `&` of logical operands (`i & d > 0` on an integer `i` is not an AND of two
# conditions, and a conjunct left over on its own would be a numeric filter).
scan_split_conjuncts <- function(ir, schema) {
  if (!identical(.subset2(ir, "kind"), "call") || !identical(.subset2(ir, "op"), "&") ||
      length(.subset2(ir, "args")) != 2L) {
    return(NULL)
  }
  parts <- list()
  add <- function(node) {
    if (identical(.subset2(node, "kind"), "call") && identical(.subset2(node, "op"), "&") &&
        length(.subset2(node, "args")) == 2L) {
      for (a in .subset2(node, "args")) add(a)
    } else {
      type <- tryCatch(ir_infer_type(node, schema), error = function(e) NA_character_)
      if (!identical(type, "BOOL8")) {
        parts <<- NULL
      } else if (!is.null(parts)) {
        parts[[length(parts) + 1L]] <<- node
      }
    }
  }
  add(ir)
  parts
}

scan_compare_ops <- c(">" = "GREATER", ">=" = "GREATER_EQUAL", "<" = "LESS", "<=" = "LESS_EQUAL",
                      "==" = "EQUAL", "!=" = "NOT_EQUAL")
scan_flipped_ops <- c(">" = "<", ">=" = "<=", "<" = ">", "<=" = ">=", "==" = "==", "!=" = "!=")

# The literal of a comparison against a column of file type `type`, as the value
# the reader gets, or NULL when it does not convert EXACTLY (then the conjunct
# stays above the scan).
scan_literal <- function(lit, type) {
  v <- .subset2(lit, "value")
  lt <- .subset2(lit, "type")
  if (isTRUE(.subset2(lit, "na")) || is.null(v) || is.null(lt) || length(v) != 1L || is.na(v)) {
    return(NULL)
  }
  whole <- function(x, bound) is.finite(x) && x == floor(x) && abs(x) <= bound
  plain <- is.null(attributes(v))
  switch(type,
    # INT32 file values the reader sees as they are: R's NA_integer_ is read as
    # missing, so the literal can't be that value either (a double -2^31 is not
    # an R integer)
    INT32 = {
      if (!plain || !lt %in% c("INT32", "FLOAT64")) return(NULL)
      x <- as.numeric(v)
      if (whole(x, 2147483647)) x
    },
    # below 2^53: the GPU compares an INT64 with a double literal as doubles,
    # which rounds values past 2^53 (2^53 + 1 == 2^53), and an atom must say
    # what the residual predicate would. A whole double literal arrives here
    # typed INT64 by ir_int64_retype_cmp(), not integer64, and stays above
    INT64 = {
      if (lt %in% c("INT32", "FLOAT64") && plain) {
        x <- as.numeric(v)
        if (whole(x, 2^53 - 1)) x
      } else if (identical(lt, "INT64") && inherits(v, "integer64") && requireNamespace("bit64", quietly = TRUE)) {
        # (compared as integer64 first: converting one past 2^53 warns)
        lim <- bit64::as.integer64("9007199254740991")
        if (isTRUE(v <= lim) && isTRUE(v >= -lim)) as.numeric(v)
      }
    },
    TIMESTAMP_DAYS = {
      if (!identical(lt, "TIMESTAMP_DAYS") || !inherits(v, "Date")) return(NULL)
      x <- as.numeric(v)
      if (whole(x, 2147483647)) x
    },
    # whole seconds only: the literal is a double count of seconds
    TIMESTAMP_MICROSECONDS = {
      if (!identical(lt, "TIMESTAMP_MICROSECONDS") || !inherits(v, "POSIXct")) return(NULL)
      x <- as.numeric(v)
      if (whole(x, 9007199254) && abs(x * 1e6) <= 2^53) x * 1e6
    },
    NULL
  )
}

# The smallest value of an integer file type: R's NA, which the read turns into
# a missing value
scan_int_min <- function(type) if (identical(type, "INT32")) -2147483648 else -9223372036854775808

# The atoms (and their text) a conjunct stands for, or NULL when it can't be
# pushed. An atom is list(col, op, type, value); the record is
# list(text, atoms).
#
# Atoms, with the reasons they are exact:
# - `col <cmp> lit` (or the literal first, flipped): a null column value gives a
#   null comparison, which drops the row like R's NA. The literal is in the
#   file type of the column.
#   * INT32/INT64: the reader sees R's NA value (INT_MIN) as a number, so `<`,
#     `<=` and `!=` get a `col > MIN` atom that drops those rows as the read will
#     make them missing (`>`, `>=` and `==` already drop them: the literal is
#     above MIN).
# - `!is.na(col)`: INT32/INT64 only, as `col > MIN` (null and the sentinel both
#   drop). `is.na()` is not pushed, and IS_NULL itself is not used: it skips row
#   groups from each column's null count, which a writer may leave out next to
#   min/max. `col > MIN` on such a file (a patched footer with no null count,
#   tests/testthat/fixtures/parquet/no_null_count_stats.parquet) is tested and
#   passes. Nor `is.na(col)` on INT: the sentinel is not null.
scan_atom <- function(ir, scan, schema) {
  if (!identical(.subset2(ir, "kind"), "call")) {
    return(NULL)
  }
  op <- .subset2(ir, "op")
  args <- .subset2(ir, "args")
  col_ok <- function(name) {
    i <- match(name, schema$names)
    if (is.na(i) || anyDuplicated(schema$names) > 0L) return(NULL)
    f <- match(name, scan$file_names)
    # the schema type is the file's: no cast column, and a factor is not one
    if (is.na(f) || !identical(schema$types[[i]], scan$file_types[[f]]) ||
        !scan$file_types[[f]] %in% scan_pushable_types) {
      return(NULL)
    }
    scan$file_types[[f]]
  }
  is_col <- function(a) identical(.subset2(a, "kind"), "col")
  is_lit <- function(a) identical(.subset2(a, "kind"), "lit")
  int_min <- function(col, type) list(col = col, op = "GREATER", type = type, value = scan_int_min(type))

  if (op %in% names(scan_compare_ops) && length(args) == 2L) {
    a <- args[[1L]]
    b <- args[[2L]]
    if (is_lit(a) && is_col(b)) {
      tmp <- a
      a <- b
      b <- tmp
      op <- scan_flipped_ops[[op]]
    }
    if (!is_col(a) || !is_lit(b)) return(NULL)
    col <- .subset2(a, "name")
    type <- col_ok(col)
    if (is.null(type)) return(NULL)
    value <- scan_literal(b, type)
    if (is.null(value)) return(NULL)
    atoms <- list(list(col = col, op = scan_compare_ops[[op]], type = type, value = value))
    if (type %in% c("INT32", "INT64") && op %in% c("<", "<=", "!=")) {
      atoms <- c(atoms, list(int_min(col, type)))
    }
    return(list(text = paste(col, op, scan_value_text(b)), atoms = atoms))
  }

  # !is.na(col) on an integer column
  if (identical(op, "!") && length(args) == 1L) {
    inner <- args[[1L]]
    if (identical(.subset2(inner, "kind"), "call") && identical(.subset2(inner, "op"), "is.na") &&
        length(.subset2(inner, "args")) == 1L && is_col(.subset2(inner, "args")[[1L]])) {
      col <- .subset2(.subset2(inner, "args")[[1L]], "name")
      type <- col_ok(col)
      if (isTRUE(type %in% c("INT32", "INT64"))) {
        return(list(text = paste0("!is.na(", col, ")"), atoms = list(int_min(col, type))))
      }
    }
  }
  NULL
}

# How a literal reads in the plan text (explain(), parquet_last_read()).
scan_value_text <- function(lit) {
  v <- .subset2(lit, "value")
  format(v)
}

# A filter() directly on a parquet scan, with what the reader can decide moved
# into the scan. Returns the scan node when every conjunct was pushed, the filter
# over the narrower scan when some were, or `node` itself when none was.
#
# Whatever has to run above the scan runs as the user wrote it, pushed conjuncts
# included. A part alone is not the expression it was next to its siblings: a
# bare NA, or a call of NAs like if_else(l, NA, NA), takes the type of the
# operand beside it and is FLOAT64 on its own, which gpu_filter_expr() refuses.
# The pushed conjuncts only cost one more comparison per row that survived them.
scan_absorb_filter <- function(node) {
  scan_node <- .subset2(node, "input")
  scan <- .subset2(scan_node, "scan")
  schema <- .subset2(scan_node, "schema")
  # grouped, `.by` and rowwise filters stay whole; so does a filter holding an
  # integer-overflow check (F70): every predicate of one filter() sees the same
  # rows in dplyr, and the check warns for the rows its siblings would drop
  if (isTRUE(.subset2(node, "grouped"))) {
    return(node)
  }
  preds <- .subset2(node, "predicates")
  if (any(vapply(preds, record_checked_int, logical(1)))) {
    return(node)
  }

  pushed <- list()
  all_pushed <- TRUE
  for (p in preds) {
    ir <- .subset2(p, "ir")
    for (part in scan_split_conjuncts(ir, schema) %||% list(ir)) {
      hit <- scan_atom(part, scan, schema)
      if (is.null(hit)) {
        all_pushed <- FALSE
      } else {
        pushed[[length(pushed) + 1L]] <- hit
      }
    }
  }
  if (length(pushed) == 0L) {
    return(node)
  }
  scan$predicates <- c(scan$predicates, pushed)
  scan_node$scan <- scan
  if (all_pushed) {
    return(scan_node)
  }
  node$input <- scan_node
  node
}

# Does a segment hold a filter directly on a scan?
segment_filters_on_scan <- function(node) {
  while (!is.null(node)) {
    type <- .subset2(node, "type")
    if (identical(type, "join")) {
      return(segment_filters_on_scan(.subset2(node, "left")) ||
               segment_filters_on_scan(.subset2(node, "right")))
    }
    input <- .subset2(node, "input")
    if (identical(type, "filter") && !is.null(node_scan(input))) {
      return(TRUE)
    }
    node <- input
  }
  FALSE
}

# The pass: every filter directly on a scan (both sides of a join included).
push_into_scan <- function(ast) {
  if (is.null(ast) || identical(.subset2(ast, "type"), "source")) {
    return(ast)
  }
  if (identical(.subset2(ast, "type"), "join")) {
    ast$left <- push_into_scan(.subset2(ast, "left"))
    ast$right <- push_into_scan(.subset2(ast, "right"))
    return(ast)
  }
  input <- .subset2(ast, "input")
  if (identical(.subset2(ast, "type"), "filter") && !is.null(node_scan(input))) {
    return(scan_absorb_filter(ast))
  }
  pushed <- push_into_scan(input)
  if (!is_reference(pushed, input)) {
    ast$input <- pushed
  }
  ast
}

# Does a plan read a parquet scan anywhere?
plan_has_scan <- function(node) {
  while (!is.null(node)) {
    if (identical(.subset2(node, "type"), "join")) {
      return(plan_has_scan(.subset2(node, "left")) || plan_has_scan(.subset2(node, "right")))
    }
    if (!is.null(node_scan(node))) {
      return(TRUE)
    }
    node <- .subset2(node, "input")
  }
  FALSE
}

# A verb that has to run the pending plan at the verb (summarise_run_now()) knows
# which columns it reads. A select of those, appended to a plan over a parquet
# scan, lets projection pushdown keep the other columns out of the read; any
# other table is returned as it is.
plan_keep_columns <- function(.data, cols) {
  ops <- .subset2(.data, "lazy_ops")
  if (!scan_pushdown_enabled() || !inherits(ops, "ast_node") || !plan_has_scan(ops)) {
    return(.data)
  }
  names <- current_schema(.data)$names
  keep <- names[names %in% cols]
  if (length(keep) == 0L || length(keep) == length(names) || anyDuplicated(names) > 0L) {
    return(.data)
  }
  push_op(.data, ast_select(input_node(.data), keep))
}
