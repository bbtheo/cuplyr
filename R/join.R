# Join operations for tbl_gpu

#' Is a `join_by()` object non-equi?
#'
#' `TRUE` only for a real `dplyr_join_by` object (class from `dplyr::join_by()`)
#' with at least one condition other than `"=="` (an inequality/rolling
#' condition, e.g. `join_by(a >= b)`, `join_by(between(...))`) or any
#' `filter` other than `"none"` (a `closest()` join). Empirically verified
#' against dplyr 1.2.1 (`str(join_by(...))` for `join_by(a)`, `join_by(a ==
#' b, c >= d)`, `join_by(x$a == y$b)`, `join_by(between(a, lo, hi))`,
#' `join_by(closest(a >= b))`, `join_by(within(...))`, `join_by(overlaps(...))`):
#' the object is a PRE-NORMALIZED `list(exprs, condition, filter, x, y)`
#' where `condition`/`filter`/`x`/`y` are already parallel, per-condition
#' character vectors (NOT a single scalar `filter` -- `between()`/`within()`/
#' `overlaps()` each pre-expand to two `condition` entries with `filter =
#' c("none", "none")`, and mixing `closest()` with a plain equi condition in
#' the same call, e.g. `join_by(closest(a >= b), c == d)`, produces `filter =
#' c("max", "none")`). This function's "any" check is deliberate: a `join_by()`
#' call mixing an equi condition with even ONE non-equi/closest condition
#' must route the WHOLE join to the CPU fallback (see the callers in this
#' file), since there is no such thing as a partially-equi join in this
#' package's hash-join path.
#' @keywords internal
join_by_is_non_equi <- function(by) {
  inherits(by, "dplyr_join_by") &&
    (any(by$condition != "==") || any(by$filter != "none"))
}

#' The non-equi operators J7's native `mixed_join`/`conditional_join`
#' lowering understands
#'
#' `join_by()`'s own infix parser only ever produces one of these five for
#' a condition (verified empirically: `==`, `>=`, `>`, `<=`, `<`; `!=` isn't
#' reachable via any of `join_by()`'s own helpers -- `between()`/`within()`/
#' `overlaps()` all pre-expand to `>=`/`<=` pairs, `closest()` wraps one of
#' the five -- but is included here defensively since `ast_op_from_name()`
#' (`src/expr_eval.hpp`) already has an entry for it and nothing about this
#' set is dplyr-specific).
#' @keywords internal
join_native_non_equi_ops <- c("==", "!=", ">=", ">", "<=", "<")

#' Is this `by=` spec natively lowerable end to end (Phase 7 J7)?
#'
#' `TRUE` for everything [join_by_is_non_equi()] already called equi (no
#' change there -- J6 native hash-join path). For a genuinely non-equi
#' `join_by()` object, native lowering additionally requires:
#'   1. Every condition's `filter` is `"none"` -- a `closest()`/rolling
#'      condition (`filter = "max"/"min"`) is a nearest-match query, not a
#'      plain inequality; `gpu_cond_join()`/`build_join_ast()`
#'      (src/ops_join.cpp / src/expr_eval.hpp) have no such "keep only the
#'      extremal match per group" step, and dplyr's `closest()` GPU route is
#'      out of scope until Phase 12+ (J8 pins this fallback).
#'   2. Every condition's operator is one this package's AST lowering
#'      understands ([join_native_non_equi_ops]).
#'   3. Neither side of any genuinely non-equi (`condition != "=="`)
#'      condition is a STRING column -- `build_join_ast()`'s comparison
#'      lowering reuses `get_compare_op()`/AST comparison ops, which cudf's
#'      AST evaluator can't apply to STRING columns at all (unlike the
#'      dedicated `binary_operation()` handler `ir_op_is_comparison()`
#'      routes string filter/mutate comparisons through); STRING non-equi
#'      joins stay on the CPU fallback (J8).
#' A missing/misspelled column name is NOT checked here -- left to
#' `validate_join_cols()`'s own proper error downstream on the native path
#' (safer than silently routing an outright user typo to a fallback that
#' would just re-raise the same error one layer later).
#' @keywords internal
join_by_natively_supported <- function(by, x, y) {
  if (!join_by_is_non_equi(by)) {
    return(TRUE)
  }

  if (any(by$filter != "none")) {
    return(FALSE)
  }
  if (!all(by$condition %in% join_native_non_equi_ops)) {
    return(FALSE)
  }

  is_non_equi <- by$condition != "=="
  if (!any(is_non_equi)) {
    return(TRUE)
  }

  left_idx <- match(by$x[is_non_equi], x$schema$names)
  right_idx <- match(by$y[is_non_equi], y$schema$names)
  if (anyNA(left_idx) || anyNA(right_idx)) {
    return(TRUE)  # let validate_join_cols() raise its own error natively
  }

  left_types <- x$schema$types[left_idx]
  right_types <- y$schema$types[right_idx]
  !any(left_types == "STRING") && !any(right_types == "STRING")
}

#' Parse a join specification into a 4-vector spec
#'
#' Returns `list(left, right, op, filter)`: `left`/`right`/`op` are parallel
#' character vectors (one entry per key/condition pair), `filter` is a single
#' scalar describing the whole join's filter semantics (`"none"` for a plain
#' equi join; `"max"`/`"min"` for `closest()` joins -- Phase 7 J7, which never
#' reach this function, see below). Every form this function currently
#' understands (`NULL`, an unnamed character vector, a named character
#' vector, or an EQUI-ONLY `join_by()` object) produces an all-equi spec:
#' `op` is `"=="` for every key and `filter` is `"none"`. A non-equi
#' `join_by()` object (any condition other than `"=="`, or any `closest()`
#' filter -- see `join_by_is_non_equi()`) is never passed here: every join
#' verb checks `join_by_is_non_equi()` FIRST and routes straight to the CPU
#' fallback (`gpu_fallback()`, mirroring `nest_join()`'s two-table fallback
#' shape in `R/fallback.R`) before ever calling `parse_join_by()` -- Phase 7
#' J7 replaces that fallback with a native `mixed_join`/`conditional_join`
#' lowering, at which point this function gains a real non-equi branch. The
#' `stop()` below is an internal-only safety net: every join verb calls
#' [join_by_natively_supported()] FIRST and routes the unsupported subset
#' (`closest()`/rolling `filter=`, an unrecognized operator, or a STRING
#' non-equi column -- see that function's own docs) to the CPU fallback
#' before ever reaching this function, so a non-equi `join_by()` that does
#' reach here is always one J7's native `mixed_join`/`conditional_join`
#' lowering (`gpu_cond_join()`, `src/ops_join.cpp`) can handle: `op`
#' carries each condition's real operator (not just `"=="`) straight
#' through, and [join_equi_spec()]/[join_cond_spec()] (below) split it back
#' into its equi/non-equi halves at lowering time (`R/lower.R::lower_join()`).
#' @keywords internal
parse_join_by <- function(by, x, y) {
  if (is.null(by)) {
    common <- intersect(x$schema$names, y$schema$names)
    if (length(common) == 0) {
      stop("No common columns for natural join. Specify `by` argument.",
           call. = FALSE)
    }
    return(list(left = common, right = common,
                op = rep("==", length(common)), filter = "none"))
  }

  if (inherits(by, "dplyr_join_by")) {
    if (join_by_is_non_equi(by) && !all(by$filter == "none")) {
      stop("Internal error: a closest()/rolling join_by() must be routed ",
           "to the CPU fallback before reaching parse_join_by().",
           call. = FALSE)
    }
    return(list(left = by$x, right = by$y,
                op = by$condition, filter = "none"))
  }

  if (is.character(by) && is.null(names(by))) {
    return(list(left = by, right = by,
                op = rep("==", length(by)), filter = "none"))
  }

  if (is.character(by) && !is.null(names(by))) {
    left_cols <- names(by)
    right_cols <- unname(by)
    empty_names <- left_cols == ""
    left_cols[empty_names] <- right_cols[empty_names]
    return(list(left = left_cols, right = right_cols,
                op = rep("==", length(left_cols)), filter = "none"))
  }

  stop("Invalid `by` specification. Use NULL, character vector, or named vector.",
       call. = FALSE)
}

#' Is a join spec entirely equi (`==`)?
#'
#' A spec with no condition pairs at all (`length(spec$op) == 0`) counts as
#' equi -- vacuously true, and matches `join_is_equi()`'s only current caller
#' ([resolve_join_keep()]) needing a sensible answer for the natural-join
#' (`by = NULL`, zero common columns already rejected earlier) edge case.
#' @keywords internal
join_is_equi <- function(spec) {
  length(spec$op) == 0 || all(spec$op == "==")
}

#' The equi (`==`) subset of a join spec
#'
#' Every current spec is 100% equi (Phase 7 J1); this returns `spec`
#' unchanged in that case. Kept as a named subsetting helper (rather than
#' inlining `spec$op == "=="` at call sites) so the future non-equi callers
#' (Phase 7 J7's hash-join key selection) have one place to read the equi
#' subset from.
#' @keywords internal
join_equi_spec <- function(spec) {
  is_eq <- spec$op == "=="
  list(left = spec$left[is_eq], right = spec$right[is_eq],
       op = spec$op[is_eq], filter = spec$filter)
}

#' The non-equi subset of a join spec
#'
#' Feeds `join_cond_ir_from_spec()` below: the residual AST condition
#' `gpu_cond_join()` (`src/ops_join.cpp`) evaluates via `mixed_join`'s
#' `binary_predicate` argument (equi entries are handled entirely by
#' `mixed_join`'s own equality-key hashing, via [join_equi_spec()] instead --
#' re-including them in the AST condition too would be redundant, not
#' wrong, but this package's `gpu_cond_join()` never does).
#' @keywords internal
join_cond_spec <- function(spec) {
  is_eq <- spec$op == "=="
  list(left = spec$left[!is_eq], right = spec$right[!is_eq],
       op = spec$op[!is_eq], filter = spec$filter)
}

#' Build a single, `&`-folded IR condition tree from a join spec's non-equi
#' subset (Phase 7 J7)
#'
#' One `ir_call(op, list(x_col, y_col))` comparison node per condition
#' entry (`cond_spec$left[i] <op> cond_spec$right[i]`), each column tagged
#' with its side via [ir_col_join()] (`0L` = left/x, `1L` = right/y --
#' straight from `cond_spec`'s own parallel `left`/`right` vectors, which
#' `parse_join_by()` populated from `join_by()`'s own `x`/`y` fields; see
#' [ir_bind_join()]'s docs for why this explicit tag -- not a lookup-order
#' convention -- is required for a self-join to bind correctly), folded
#' together with `"&"` exactly like `lower_filter()` folds a filter node's
#' predicate list into one expression. `cond_spec` must be non-empty
#' (callers only invoke this when `length(cond_spec$op) > 0`, i.e. the
#' join_by() has at least one genuinely non-equi condition -- see
#' `lower_join()`, R/lower.R).
#' @param cond_spec The non-equi subset of a join spec, as returned by
#'   [join_cond_spec()]
#' @return An unbound IR node (run [ir_bind_join()] before lowering)
#' @keywords internal
join_cond_ir_from_spec <- function(cond_spec) {
  n <- length(cond_spec$op)
  combined <- NULL
  for (i in seq_len(n)) {
    node <- ir_call(cond_spec$op[i], list(
      ir_col_join(cond_spec$left[i], 0L),
      ir_col_join(cond_spec$right[i], 1L)
    ))
    combined <- if (is.null(combined)) node else ir_call("&", list(combined, node))
  }
  combined
}

#' Resolve `keep`'s `NULL` default against a join spec
#'
#' Ported directly from `dplyr:::join_cols()` (verified via `body()` against
#' dplyr 1.2.1, Phase 7 J7 finding): `keep = NULL`'s real resolution is NOT
#' a single whole-join TRUE/FALSE decision -- it is PER CONDITION.
#' `join_cols()`'s own `is_null(keep)` branch drops the right key column
#' (and coalesces, no suffix distinction) only for conditions whose
#' `by$condition == "=="`; a genuinely non-equi condition (`join_by(a >=
#' b)`) always keeps BOTH sides' key columns, suffixed like any other
#' colliding column -- verified empirically: `join_by(c == d, a >= b)`
#' drops `d` (the equi pair) but keeps both `a` and `b` (the non-equi
#' pair). An all-equi spec's per-condition resolution collapses to
#' "drop every right key" (this package's historical `FALSE` behavior,
#' unchanged); an all-non-equi spec's per-condition resolution collapses to
#' "keep every column, suffixed uniformly" (this package's historical
#' `TRUE`-forced behavior for a pure non-equi join, also unchanged) -- only
#' a MIXED equi/non-equi spec actually needs the per-condition machinery to
#' differ from either extreme, which is why this was never caught before
#' J7 (no spec was ever both equi and non-equi at once).
#'
#' Both `NULL` and a *legal* explicit `FALSE` resolve to the same
#' `FALSE` return value here: `build_join_output_info()`'s own `keep !=
#' TRUE` branch always applies the per-condition (`join_spec$op == "=="`)
#' logic, which -- per `dplyr:::join_cols()`'s own `is_null`/`is_false`
#' branches -- are byte-identical whenever `is_false(keep)` is even legal
#' (i.e. an all-equi spec, where `by$x[equi] == by$x` and `by$y[equi] ==
#' by$y` make the two branches' cross-sets identical); explicit `keep =
#' FALSE` on any spec with a genuinely non-equi condition is a hard error
#' with dplyr's own text, exactly as before.
#' @param keep `NULL`, `TRUE`, or `FALSE`, as passed by the user
#' @param spec A join spec as returned by [parse_join_by()]
#' @return `TRUE`/`FALSE`
#' @keywords internal
resolve_join_keep <- function(keep, spec) {
  if (is.null(keep)) {
    return(FALSE)
  }

  if (isFALSE(keep) && !join_is_equi(spec)) {
    stop("Can't set `keep = FALSE` when using an inequality, rolling, or overlap join.",
         call. = FALSE)
  }

  keep
}

#' Validate a join spec's per-side column list exists in that side's schema
#'
#' Deliberately does NOT require `cols` to be unique (Phase 7 J7 finding): a
#' repeated column IS a legitimate non-equi shape --
#' `join_by(between(a, lo, hi))` compares the SAME left column `a` against
#' TWO different right columns (`lo` via `>=`, `hi` via `<=`), so
#' `join_spec$left` is `c("a", "a")` by construction (`within()`/
#' `overlaps()` similarly duplicate a column across their pre-expanded
#' condition pairs) -- this is exactly what `between()`/`within()`/
#' `overlaps()` need to keep working once J7 routes them off the CPU
#' fallback, not a user error to reject.
#' @keywords internal
validate_join_cols <- function(cols, tbl, side) {
  missing <- setdiff(cols, tbl$schema$names)
  if (length(missing) > 0) {
    stop(side, " join columns not found: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
}

validate_key_types <- function(x, y, join_spec) {
  left_types <- x$schema$types[match(join_spec$left, x$schema$names)]
  right_types <- y$schema$types[match(join_spec$right, y$schema$names)]

  numeric_types <- c("INT8", "INT16", "INT32", "INT64", "FLOAT32", "FLOAT64", "BOOL8")

  for (i in seq_along(left_types)) {
    lt <- left_types[i]
    rt <- right_types[i]

    if (identical(lt, rt)) {
      next
    }

    if (lt %in% numeric_types && rt %in% numeric_types) {
      next
    }

    stop(sprintf(
      "Key column type mismatch: %s (%s) vs %s (%s). Cannot join incompatible types.",
      join_spec$left[i], lt, join_spec$right[i], rt
    ), call. = FALSE)
  }
}

#' Port of `dplyr:::add_suffixes()`
#'
#' The real column-naming primitive behind every dplyr join's suffix
#' handling (verified via `body(dplyr:::add_suffixes)` against dplyr 1.2.1)
#' -- NOT a one-shot "does this name collide with the other side" check like
#' `build_join_output_info()` used before Phase 7 J6. `x` is the set of
#' names to (maybe) suffix; `y` is the "cross" name set to disambiguate
#' against (the other side's names, plus, for the ignore-aware caller below,
#' this side's own ignored/key names); `suffix` is appended to any entry of
#' `x` that collides with `c(y, x)`, and the check is REPEATED (each
#' iteration re-checks the whole combined vector for duplicates) until every
#' entry is unique -- so an "already-taken suffixed name" (e.g. left already
#' has a literal `val.y` column when `val` needs a `.y` suffix from a
#' collision with the right table) gets a SECOND suffix (`val.y.y`),
#' matching dplyr exactly rather than silently producing a duplicate output
#' column name. `suffix` equal to the literal empty string `""` is dplyr's
#' own escape hatch: `identical(suffix, "")` short-circuits to `x`
#' unchanged, even when that leaves a genuine name collision -- dplyr
#' resolves that downstream via a name-keyed column *overwrite*
#' (`join_mutate()`'s `out[names(y_out)] <- vec_slice(y_out, ...)`, which
#' replaces same-named columns in place rather than erroring or renaming);
#' see `join_output_would_collide()` below for how cuplyr detects
#' this specific degenerate case and routes around it (CPU fallback) instead
#' of trying to reproduce a column-count-changing overwrite in the native
#' gather path.
#' @keywords internal
add_suffixes <- function(x, y, suffix) {
  if (length(x) == 0) {
    return(x)
  }
  if (identical(suffix, "")) {
    return(x)
  }

  combined <- c(y, x)
  dup <- duplicated(combined)
  while (any(dup)) {
    combined[dup] <- paste0(combined[dup], suffix)
    dup <- duplicated(combined)
  }

  n_y <- length(y)
  combined[seq.int(n_y + 1L, n_y + length(x))]
}

#' Compute a join's output column names/types/origin, mirroring
#' `dplyr:::join_cols()` exactly (Phase 7 J6/J7)
#'
#' Ported directly from dplyr 1.2.1's own `join_cols()` (verified via
#' `body(dplyr:::join_cols)`), specialized to this package's always-resolved
#' `keep` (never `NULL` here -- every join verb calls `resolve_join_keep()`
#' first, which -- Phase 7 J7 finding -- returns `FALSE` for BOTH the real
#' `NULL` default AND a legal explicit `FALSE`; see its own roxygen for why
#' those two collapse to one value here).
#'
#' `keep = TRUE`: every column on both sides is suffixed uniformly via
#' `add_suffixes()` (no column is "ignored"), and nothing is dropped from
#' the right table -- reproduces e.g. `join_by(a)` + `keep = TRUE` giving
#' `a.x`/`a.y` for BOTH copies of the shared key.
#'
#' `keep != TRUE` (the `NULL`-default/legal-`FALSE` case): PER CONDITION,
#' mirroring `dplyr:::join_cols()`'s own `is_null(keep)` branch exactly
#' (Phase 7 J7 finding -- verified empirically against dplyr 1.2.1 that this
#' branch, not a whole-join TRUE/FALSE collapse, governs the real `keep =
#' NULL` default: `join_by(c == d, a >= b)` drops `d` but keeps BOTH `a` and
#' `b`). `is_eq <- join_spec$op == "=="` selects which condition PAIRS
#' collapse to one coalesced column (the join KEY columns for those pairs
#' specifically are excluded from suffixing -- they keep their bare
#' original name -- and the right table's key column for that pair is
#' dropped from the output) versus which pairs are kept verbatim on BOTH
#' sides, suffixed like any other colliding column (a genuinely non-equi
#' condition, e.g. `join_by(a >= b)`, has no single shared key value to
#' collapse to -- there IS no "coalesced" column for it). An all-equi spec
#' (`is_eq` all `TRUE`) collapses to this package's historical `keep =
#' FALSE` behavior exactly; an all-non-equi spec (`is_eq` all `FALSE`)
#' collapses to byte-identical output to the `keep = TRUE` branch above
#' (every column suffixed against the full cross set, nothing dropped) --
#' verified algebraically: `x_ignore`/`y_ignore` are both `character(0)`,
#' so `y_aux` becomes all of `right_names` and every column check is
#' unconditional, exactly reproducing `add_suffixes(left_names,
#' right_names, ...)`/`add_suffixes(right_names, left_names, ...)`. Only a
#' MIXED equi/non-equi spec actually needs this per-condition machinery to
#' diverge from either extreme.
#' @keywords internal
build_join_output_info <- function(left_schema, right_schema, join_spec,
                                   suffix = c(".x", ".y"), keep = FALSE) {
  left_names <- left_schema$names
  right_names <- right_schema$names

  if (isTRUE(keep)) {
    left_out_names <- add_suffixes(left_names, right_names, suffix[1])
    right_out_names <- add_suffixes(right_names, left_names, suffix[2])
    right_keep <- right_names
  } else {
    is_eq <- join_spec$op == "=="
    x_ignore <- join_spec$left[is_eq]
    y_ignore <- join_spec$right[is_eq]

    y_aux <- setdiff(right_names, c(x_ignore, y_ignore))
    x_check <- !left_names %in% x_ignore

    left_out_names <- left_names
    left_out_names[x_check] <- add_suffixes(left_names[x_check],
                                            c(x_ignore, y_aux), suffix[1])

    right_out_all <- add_suffixes(right_names, left_names, suffix[2])
    keep_idx <- !right_names %in% y_ignore
    right_keep <- right_names[keep_idx]
    right_out_names <- right_out_all[keep_idx]
  }

  out_names <- c(left_out_names, right_out_names)
  origin <- c(rep("left", length(left_out_names)), rep("right", length(right_out_names)))
  source_names <- c(left_names, right_keep)

  list(
    names = out_names,
    types = c(left_schema$types, right_schema$types[match(right_keep, right_names)]),
    origin = origin,
    source_names = source_names,
    factor_levels = join_output_factor_levels(left_schema, right_schema,
                                              out_names, origin, source_names)
  )
}

#' Propagate `factor_levels` through a join's output-name mapping
#'
#' `build_join_schema()`/`build_join_output_info()` compute each output
#' column's post-suffix NAME and which side's SOURCE column it came from
#' (`origin`/`source_names`, parallel to `names`) -- this reattaches
#' `factor_levels`, keyed by that post-suffix output name, by looking up
#' each output column's own origin schema's `factor_levels` under its
#' PRE-suffix source name. This is the Bug 1 fix (Phase 11 L2): before this,
#' `build_join_schema()` returned only `names`/`types`, so every mutating
#' join (`left_join()`/`inner_join()`/`full_join()`/`right_join()`, and
#' `cross_join()`) silently dropped factor metadata entirely -- `collect()`
#' would then present a factor column as raw integer codes.
#'
#' A coalesced equi-join KEY column (kept unsuffixed under `keep = FALSE`,
#' see `build_join_output_info()`'s `is_eq` branch) always has `origin ==
#' "left"` here (the right-side copy of that key is dropped, never part of
#' `out_names`) -- correct, since the join's native gather always keeps the
#' LEFT key column's own physical data for a coalesced key (see
#' `src/ops_join.cpp`), so its levels are always the left side's.
#' `join_factor_key_levels_conflict()` (below) is the correctness guard for
#' the case this alone can't resolve: an equi key that's factor-typed on
#' BOTH sides with DIFFERENT level sets would still coalesce to "the left
#' side's levels" here, but the native GPU join itself compares the two
#' sides' raw INT32 codes directly -- which would silently mismatch two
#' logically-equal labels encoded under different codes. That case is
#' caught and routed to the CPU fallback before ever reaching this
#' function (every join verb calls it right after `validate_key_types()`).
#' @keywords internal
join_output_factor_levels <- function(left_schema, right_schema, out_names, origin, source_names) {
  factor_levels <- list()
  for (i in seq_along(out_names)) {
    src_schema <- if (identical(origin[i], "left")) left_schema else right_schema
    lvls <- src_schema$factor_levels[[source_names[i]]]
    if (!is.null(lvls)) {
      factor_levels[[out_names[i]]] <- lvls
    }
  }
  if (length(factor_levels) == 0) NULL else factor_levels
}

#' Would this join's equi keys compare two factor columns with different
#' level sets?
#'
#' Native GPU joins match equi keys by comparing their PHYSICAL INT32 codes
#' directly (`validate_key_types()` only checks `identical()`/numeric-
#' compatible type strings, never level compatibility) -- if both sides of
#' an equi condition are factor columns whose level VECTORS differ, two
#' logically-equal labels can be encoded under different codes (e.g. `"a"`
#' is code 1 on the left but code 2 on the right), so comparing raw codes
#' would silently produce wrong matches. This is detectable from schemas
#' alone, before any GPU op runs -- every join verb (`left_join()`/
#' `inner_join()`/`full_join()`/`right_join()`/`semi_join()`/`anti_join()`)
#' calls this right after `validate_key_types()` and routes the WHOLE join
#' to the CPU fallback when it's `TRUE`, mirroring
#' `join_output_would_collide()`'s existing pattern (Phase 8 sets.R
#' precedent: `check_set_op_compatible()` uses the identical
#' identical-levels-vs-not split to decide native vs. fallback).
#' Both-factor-with-IDENTICAL-levels keys compare correctly natively (same
#' code domain) and are not flagged here.
#' @keywords internal
join_factor_key_levels_conflict <- function(x_schema, y_schema, join_spec) {
  if (length(join_spec$op) == 0) {
    return(FALSE)
  }
  is_eq <- join_spec$op == "=="
  if (!any(is_eq)) {
    return(FALSE)
  }

  left_keys <- join_spec$left[is_eq]
  right_keys <- join_spec$right[is_eq]

  for (i in seq_along(left_keys)) {
    lvls_x <- x_schema$factor_levels[[left_keys[i]]]
    lvls_y <- y_schema$factor_levels[[right_keys[i]]]
    if (!is.null(lvls_x) && !is.null(lvls_y) && !identical(lvls_x, lvls_y)) {
      return(TRUE)
    }
  }

  FALSE
}

#' Would this join's output have a duplicate column name?
#'
#' With the `add_suffixes()` port above, this can only happen when at least
#' one `suffix` element is the literal empty string `""` AND that side's
#' would-be-suffixed name genuinely collides with the other side (see
#' `add_suffixes()`'s own roxygen for why `""` is special: dplyr itself
#' resolves the resulting name clash via a name-keyed column *overwrite*
#' downstream in `join_mutate()`, not a rename or an error) -- with any
#' other suffix value, `add_suffixes()`'s iterate-until-unique loop
#' guarantees `build_join_output_info()`'s `names` has no duplicates.
#' Callers use this to detect the degenerate case up front (computable from
#' schemas alone, no GPU work needed) and route the whole join to the CPU
#' fallback instead, since cuplyr's fixed-column-count gather has no native
#' way to reproduce "the later column's data silently replaces the earlier
#' same-named column, and the total column count drops by one".
#' @keywords internal
join_output_would_collide <- function(left_schema, right_schema, join_spec,
                                      suffix, keep) {
  info <- build_join_output_info(left_schema, right_schema, join_spec,
                                 suffix = suffix, keep = keep)
  anyDuplicated(info$names) > 0
}

build_join_schema <- function(left_schema, right_schema, join_spec,
                              suffix = c(".x", ".y"), keep = FALSE) {
  info <- build_join_output_info(left_schema, right_schema, join_spec,
                                 suffix = suffix, keep = keep)
  list(names = info$names, types = info$types, factor_levels = info$factor_levels)
}

estimate_gpu_bytes <- function(nrow, types) {
  bytes_per_type <- vapply(types, function(type) {
    switch(type,
      "FLOAT64" = 8,
      "FLOAT32" = 4,
      "INT64" = 8,
      "INT32" = 4,
      "INT16" = 2,
      "INT8" = 1,
      "BOOL8" = 1,
      "STRING" = 32,
      "TIMESTAMP_DAYS" = 4,
      "TIMESTAMP_MICROSECONDS" = 8,
      "TIMESTAMP_NANOSECONDS" = 8,
      "DICTIONARY32" = 4,
      8
    )
  }, numeric(1))

  data_bytes <- sum(nrow * bytes_per_type)
  mask_bytes <- ceiling(nrow / 8) * length(types)

  data_bytes + mask_bytes
}

warn_if_join_too_large <- function(join_type, x, y, join_spec, suffix, keep) {
  dims_x <- tryCatch(dim(x), error = function(e) NULL)
  dims_y <- tryCatch(dim(y), error = function(e) NULL)
  if (is.null(dims_x) || is.null(dims_y)) {
    return(invisible(NULL))
  }

  n_left <- dims_x[1]
  n_right <- dims_y[1]

  est_rows <- switch(join_type,
    "left" = n_left,
    "right" = n_right,
    "inner" = min(n_left, n_right),
    "full" = n_left + n_right,
    "semi" = n_left,
    "anti" = n_left,
    "cross" = n_left * n_right,
    n_left
  )

  # semi/anti never merge columns (output is x's own schema verbatim, see
  # infer_schema.ast_join's semi/anti branch) -- build_join_output_info()'s
  # merged-schema estimate would be wrong here (and keep/suffix are unused
  # for these join types), so estimate straight from x's own schema.
  schema <- if (join_type %in% c("semi", "anti")) {
    x$schema
  } else {
    build_join_schema(x$schema, y$schema, join_spec,
                      suffix = suffix, keep = keep)
  }
  est_bytes <- estimate_gpu_bytes(est_rows, schema$types)

  mem <- gpu_memory_state()
  if (!isTRUE(mem$available) || is.na(mem$free_bytes)) {
    return(invisible(NULL))
  }

  if (est_bytes > mem$free_bytes * 0.8) {
    warning(
      sprintf(
        "Join output is estimated at ~%.2f GB with only %.2f GB free on GPU. ",
        est_bytes / 1e9,
        mem$free_bytes / 1e9
      ),
      "Many-to-many joins can exceed this estimate; consider filtering or ",
      "calling gpu_gc() before joining.",
      call. = FALSE
    )
  }
}

#' Format a scalar value for a "not one of ..." style error message
#'
#' Strings render quoted (`"bogus"`); anything else falls back to `deparse()`
#' so the message stays readable without crashing on non-character input.
#' @keywords internal
format_join_arg_value <- function(value) {
  if (is.character(value) && length(value) == 1) {
    return(paste0('"', value, '"'))
  }
  paste(deparse(value), collapse = " ")
}

#' Validate `multiple=`, replicating dplyr's exact bad-value text
#'
#' Note dplyr's own message (from `vctrs::vec_locate_matches()`) omits the
#' offending value for this argument specifically (unlike `unmatched=`/
#' `relationship=`/`na_matches=`, which all name it) -- this is dplyr's
#' actual behavior, verified empirically against dplyr 1.2.1, not an
#' oversight here.
#' @keywords internal
validate_join_multiple <- function(multiple) {
  if (!is.character(multiple) || length(multiple) != 1) {
    stop("`multiple` must be a string.", call. = FALSE)
  }
  if (!multiple %in% c("all", "any", "first", "last")) {
    stop('`multiple` must be one of "all", "any", "first", or "last".',
         call. = FALSE)
  }
  multiple
}

#' Validate `unmatched=`, replicating dplyr's exact bad-value text
#'
#' `max_length` is 2 only for `inner_join()` (independent x-side/y-side
#' checks); every other join type allows length 1 only. `full_join()` has no
#' `unmatched=` parameter at all in dplyr (it never drops rows), so it never
#' calls this.
#' @keywords internal
validate_join_unmatched <- function(unmatched, max_length = 1L) {
  if (!is.character(unmatched)) {
    stop(sprintf("`unmatched` must be a character vector, not %s.",
                 format_join_arg_value(unmatched)), call. = FALSE)
  }
  if (length(unmatched) < 1 || length(unmatched) > max_length) {
    expected <- if (max_length == 1L) "length 1" else paste0("length 1 or ", max_length)
    stop(sprintf("`unmatched` must be %s, not %d.", expected, length(unmatched)),
         call. = FALSE)
  }
  bad <- setdiff(unmatched, c("drop", "error"))
  if (length(bad) > 0) {
    stop(sprintf('`unmatched` must be one of "drop" or "error", not "%s".', bad[1]),
         call. = FALSE)
  }
  unmatched
}

#' Validate `relationship=`, replicating dplyr's exact bad-value text
#' @keywords internal
validate_join_relationship <- function(relationship) {
  if (is.null(relationship)) {
    return(relationship)
  }
  if (!is.character(relationship) || length(relationship) != 1) {
    stop("`relationship` must be a string or character vector.", call. = FALSE)
  }
  valid <- c("one-to-one", "one-to-many", "many-to-one", "many-to-many")
  if (!relationship %in% valid) {
    stop(sprintf(
      '`relationship` must be one of "one-to-one", "one-to-many", "many-to-one", or "many-to-many", not "%s".',
      relationship
    ), call. = FALSE)
  }
  relationship
}

#' Validate `na_matches=`, replicating dplyr's exact bad-value text
#' @keywords internal
validate_join_na_matches <- function(na_matches) {
  if (!is.character(na_matches) || length(na_matches) != 1) {
    stop("`na_matches` must be a string or character vector.", call. = FALSE)
  }
  if (!na_matches %in% c("na", "never")) {
    stop(sprintf('`na_matches` must be one of "na" or "never", not "%s".', na_matches),
         call. = FALSE)
  }
  na_matches
}

#' Describe a `suffix=` value the way dplyr's own bad-value message would
#'
#' A minimal, purpose-built port of just enough of dplyr's
#' `obj_type_friendly()`-driven text (walked via
#' `body(dplyr:::standardise_join_suffix)`, empirically verified against
#' dplyr 1.2.1's actual message for `character(0)`/a length-1 string/a
#' length-3 vector/a bare `NA`) to reproduce `validate_join_suffix()`'s
#' messages exactly -- not a general-purpose friendly-type formatter.
#' @keywords internal
describe_suffix_value <- function(x) {
  if (!is.character(x)) {
    if (length(x) == 1 && is.na(x)) {
      return("`NA`")
    }
    kind <- if (is.logical(x)) "logical" else if (is.integer(x)) "integer" else
      if (is.double(x)) "double" else typeof(x)
    return(sprintf("a %s vector", kind))
  }
  if (length(x) == 0) {
    return("an empty character vector")
  }
  if (length(x) == 1) {
    return(sprintf('the string "%s"', x))
  }
  "a character vector"
}

#' Validate `suffix=`, replicating dplyr's exact bad-value text
#'
#' Mirrors `dplyr:::standardise_join_suffix()` exactly (verified empirically
#' against dplyr 1.2.1): must be a length-2 character vector with no `NA`
#' element. A single literal `""` element is legal (it means "don't suffix
#' this side at all" -- see `add_suffixes()`'s own roxygen) and is NOT
#' rejected here.
#' @keywords internal
validate_join_suffix <- function(suffix) {
  if (!is.character(suffix) || length(suffix) != 2) {
    rlang::abort(sprintf(
      "`suffix` must be a character vector of length 2, not %s of length %d.",
      describe_suffix_value(suffix), length(suffix)
    ))
  }
  if (anyNA(suffix)) {
    rlang::abort("`suffix` can't be `NA`.")
  }
  suffix
}

# =============================================================================
# check_join_cardinality(): multiple=/relationship=/unmatched= activation
# (Phase 7 J5, scratchpad/phase7_joins_design.md).
#
# Every rlang class + message/bullet text below was empirically transcribed
# against dplyr 1.2.1 (`Rscript`, system R, dplyr 1.2.1/rlang 1.2.0 --
# matches this package's declared dependency) by walking dplyr's own
# internal condition-building functions (`dplyr:::stop_join_matches_multiple`,
# `dplyr:::rethrow_error_join_relationship_one_to_one`/`_one_to_many`/
# `_many_to_one`, `dplyr:::rethrow_warning_join_relationship_many_to_many`,
# `dplyr:::rethrow_error_join_matches_nothing`/`_remaining`, `dplyr:::warn_join`/
# `stop_join`/`warn_dplyr`/`stop_dplyr` for the class-vector assembly) and
# cross-checking every text/class/first-offender-row claim against live
# `left_join()`/`inner_join()`/`full_join()`/`right_join()` calls (see
# tests/testthat/test-dplyr-join-rows.R for the pinned expect_snapshot()
# comparisons). dplyr's own machinery ultimately delegates to
# `vctrs::vec_locate_matches()`, which this package does not reimplement --
# only the four first-offender diagnostics `join_map_stats()`
# (src/ops_join.cpp) already computes are needed to reconstruct dplyr's
# exact observable behavior (message text, class, offending row number).
#
# `diag` (`list(left_multi_first, right_multi_first, left_unmatched_first,
# right_unmatched_first)`, all 0-based C++ row indices with -1 meaning "no
# such row") is computed by `join_map_stats()` on the RAW (pre-`multiple=`
# -filter) sanitized join maps -- deliberately, since dplyr's own
# `relationship=`/`unmatched=` checks fire on the raw match cardinality
# regardless of `multiple=` (empirically verified: `relationship =
# "one-to-one"` still errors on a genuine duplicate even when `multiple =
# "first"` would have resolved it down to one match). "x" always means
# `ast$left`/left_map and "y" always means `ast$right`/right_map here,
# regardless of `join_type` -- dplyr's own `join_rows()` never swaps the
# x/y ("needles"/"haystack") roles based on join type either, including for
# `right_join()` (verified: `gpu_right_join()`'s own diag is computed with
# left_view = x, right_view = y, same as every other join type).
#
# Precedence (empirically verified, all four combinations of "does a
# multiplicity violation exist" x "does an unmatched violation exist"):
# multiplicity (relationship=/auto many-to-many) always checked BEFORE
# unmatched=, and within unmatched=, x-side (no_match) always checked before
# y-side (remaining).
# =============================================================================

#' Raise a `dplyr_error_join*` condition with dplyr's own class-vector shape
#'
#' Mirrors `dplyr:::stop_join()`/`dplyr:::stop_dplyr()`: the final class
#' vector is `c(class, "dplyr_error_join", "dplyr_error")`, with
#' `rlang::abort()` appending `"rlang_error"`/`"error"`/`"condition"`.
#' No `call` is threaded through (unlike dplyr's own `error_call` -- the
#' user's `left_join()`/etc. call) -- this package's `lower_join()` call
#' chain doesn't preserve that frame the way dplyr's own `error_call =
#' caller_env()` threading does, and `rlang::abort()`'s own default (`call =
#' NULL`, no call shown) is fine here: `conditionMessage()`/`cnd_message()`
#' (what every test in test-dplyr-join-rows.R compares) are unaffected
#' either way -- the call only ever shows in the *printed* "Error in ...:"
#' header, never the message text itself.
#' @keywords internal
abort_join <- function(message, class) {
  rlang::abort(message, class = c(class, "dplyr_error_join", "dplyr_error"))
}

#' Raise a `dplyr_warning_join*` condition with dplyr's own class-vector shape
#'
#' Mirrors `dplyr:::warn_join()`/`dplyr:::warn_dplyr()`: the final class
#' vector is `c(class, "dplyr_warning_join", "dplyr_warning")`, with
#' `rlang::warn()` appending `"rlang_warning"`/`"warning"`/`"condition"`.
#' `rlang::warn()` has no `call` parameter at all (unlike `rlang::abort()`),
#' so there's nothing to suppress here.
#' @keywords internal
warn_join <- function(message, class) {
  rlang::warn(message, class = c(class, "dplyr_warning_join", "dplyr_warning"))
}

#' The shared "Each row in X must match at most 1 row in Y" error
#'
#' Mirrors `dplyr:::stop_join_matches_multiple()` exactly -- the single
#' message-template function backing all three relationship-cardinality
#' violations below (one-to-one's two directions, one-to-many, many-to-one)
#' plus the (unreachable from any join verb this package implements --
#' dplyr's own `multiple=` never resolves to `vctrs::vec_locate_matches()`'s
#' "error"/"warning" values from any join verb; confirmed empirically by
#' grepping dplyr's whole namespace source for a literal `multiple =
#' "error"`/`"warning"`, found only in `rows_*()`, out of scope here)
#' `dplyr_error_join_matches_multiple` class dplyr itself defines for the
#' same template.
#' @param i 1-based row number to report
#' @param x_name,y_name Which table name each half of the message names
#' @param class The specific `dplyr_error_join_relationship_*` class
#' @keywords internal
stop_join_matches_multiple <- function(i, x_name, y_name, class) {
  abort_join(
    c(
      sprintf("Each row in `%s` must match at most 1 row in `%s`.", x_name, y_name),
      i = sprintf("Row %d of `%s` matches multiple rows in `%s`.", i, x_name, y_name)
    ),
    class = class
  )
}

#' `relationship = "one-to-one"` violation
#'
#' Checks BOTH directions (unlike one-to-many/many-to-one, which each check
#' only one). When both `left_multi`/`right_multi` indicate a violation, the
#' SMALLER 0-based index wins (ties go to `left`/x) -- empirically verified
#' against dplyr 1.2.1 via `vctrs::vec_locate_matches()` directly (the
#' underlying engine dplyr delegates to): the reported offender is always
#' whichever side's first duplicate appears earliest in the (left_map,
#' right_map)-sorted match sequence, not a fixed x-then-y or y-then-x
#' priority.
#' @keywords internal
check_relationship_one_to_one <- function(left_multi, right_multi) {
  if (left_multi < 0 && right_multi < 0) {
    return(invisible(NULL))
  }
  left_wins <- right_multi < 0 || (left_multi >= 0 && left_multi <= right_multi)
  if (left_wins) {
    stop_join_matches_multiple(left_multi + 1L, "x", "y",
                               "dplyr_error_join_relationship_one_to_one")
  } else {
    stop_join_matches_multiple(right_multi + 1L, "y", "x",
                               "dplyr_error_join_relationship_one_to_one")
  }
}

#' `relationship = "one-to-many"` violation: forbids a `y` row matching
#' multiple `x` rows (i.e. `right_multi_first`) -- ALWAYS reported as
#' "y matches multiple in x", regardless of any `x`-side duplication that
#' may also exist (dplyr's own `rethrow_error_join_relationship_one_to_many()`
#' hardcodes this direction, no `which`-based branch, unlike one-to-one).
#' @keywords internal
check_relationship_one_to_many <- function(right_multi) {
  if (right_multi < 0) return(invisible(NULL))
  stop_join_matches_multiple(right_multi + 1L, "y", "x",
                             "dplyr_error_join_relationship_one_to_many")
}

#' `relationship = "many-to-one"` violation: forbids an `x` row matching
#' multiple `y` rows (i.e. `left_multi_first`) -- ALWAYS reported as
#' "x matches multiple in y" (mirror image of one-to-many above).
#' @keywords internal
check_relationship_many_to_one <- function(left_multi) {
  if (left_multi < 0) return(invisible(NULL))
  stop_join_matches_multiple(left_multi + 1L, "x", "y",
                             "dplyr_error_join_relationship_many_to_one")
}

#' The many-to-many auto-warning (`relationship = NULL`, the default)
#'
#' Fires only when BOTH directions have a genuine duplicate (an x row
#' matching multiple y rows, AND a y row matching multiple x rows) --
#' verified empirically: a one-directional duplication pattern (e.g. many
#' x-rows sharing a key that matches a single y-row) is a many-to-one
#' pattern, not many-to-many, and does NOT warn. Mirrors
#' `dplyr:::rethrow_warning_join_relationship_many_to_many()`'s exact
#' message (3 `i` bullets) and class.
#' @keywords internal
warn_relationship_many_to_many <- function(left_multi, right_multi) {
  if (left_multi < 0 || right_multi < 0) return(invisible(NULL))
  warn_join(
    c(
      "Detected an unexpected many-to-many relationship between `x` and `y`.",
      i = sprintf("Row %d of `x` matches multiple rows in `y`.", left_multi + 1L),
      i = sprintf("Row %d of `y` matches multiple rows in `x`.", right_multi + 1L),
      i = paste0("If a many-to-many relationship is expected, set `relationship = ",
                '"many-to-many"` to silence this warning.')
    ),
    class = "dplyr_warning_join_relationship_many_to_many"
  )
}

#' `unmatched = "error"`, x-side: "Each row of x must have a match in y"
#'
#' Mirrors `dplyr:::rethrow_error_join_matches_nothing()` exactly (also the
#' handler for `na_matches`-driven "incomplete" x-side violations --
#' `rethrow_error_join_matches_incomplete()` just delegates to this same
#' function, so no separate na_matches-specific class/text is needed here).
#' @keywords internal
abort_join_matches_nothing <- function(left_unmatched) {
  abort_join(
    c(
      "Each row of `x` must have a match in `y`.",
      i = sprintf("Row %d of `x` does not have a match.", left_unmatched + 1L)
    ),
    class = "dplyr_error_join_matches_nothing"
  )
}

#' `unmatched = "error"`, y-side: "Each row of y must be matched by x"
#'
#' Mirrors `dplyr:::rethrow_error_join_matches_remaining()` exactly.
#' @keywords internal
abort_join_matches_remaining <- function(right_unmatched) {
  abort_join(
    c(
      "Each row of `y` must be matched by `x`.",
      i = sprintf("Row %d of `y` was not matched.", right_unmatched + 1L)
    ),
    class = "dplyr_error_join_matches_remaining"
  )
}

#' Split a validated `unmatched=` value into its x-side/y-side components
#'
#' Mirrors `dplyr:::check_unmatched()`'s own splitting (validation itself
#' already happened in `validate_join_unmatched()`): length 1 applies
#' identically to both sides (even though, per join type, only one side may
#' ever actually be consulted -- see `check_join_cardinality()`); length 2
#' (only ever passed for `inner_join()`) assigns positionally, `c(x, y)`.
#' @keywords internal
resolve_unmatched_sides <- function(unmatched) {
  if (length(unmatched) == 1) {
    list(x = unmatched, y = unmatched)
  } else {
    list(x = unmatched[[1]], y = unmatched[[2]])
  }
}

#' Raise dplyr's exact `multiple=`/`relationship=`/`unmatched=` conditions
#'
#' The Phase 7 J5 activation point: called from `lower_join()` (R/lower.R)
#' immediately after a mutating join (`left`/`inner`/`full`/`right`) returns
#' its `list(ptr, diag)` result, using the RAW (pre-`multiple=`-filter)
#' `diag` from `join_map_stats()` (`src/ops_join.cpp`). Never called for
#' `semi`/`anti`/`cross` -- those join types have no `multiple=`/
#' `unmatched=`/`relationship=` in dplyr's own signatures at all.
#'
#' Precedence (empirically verified against dplyr 1.2.1, all pairwise
#' combinations of "multiplicity violation present" x "unmatched violation
#' present"): multiplicity (`relationship=` or the auto many-to-many
#' warning) is ALWAYS evaluated before `unmatched=`, and within
#' `unmatched=`, the x-side check (`no_match`, "each row of x must have a
#' match") always fires before the y-side check (`remaining`, "each row of
#' y must be matched"). Per-join-type applicability of the unmatched checks
#' mirrors `dplyr:::standardise_join_no_match()`/
#' `standardise_join_remaining()`: x-side applies to `inner`/`right` (the
#' join types where a genuinely-unmatched x row can be silently dropped),
#' y-side applies to `inner`/`left` (where a genuinely-unmatched y row can
#' be silently dropped); `full` never reaches either branch in practice
#' since `full_join()` doesn't expose `unmatched=` at all (always passes
#' `"drop"`/`"drop"` from `full_join.tbl_gpu()`, R/join.R).
#'
#' The many-to-many auto-warning only fires when `relationship` is `NULL`
#' AND `multiple` is `"all"` (both defaults) -- explicitly verified against
#' dplyr 1.2.1: passing any non-default `multiple=` (the user is already
#' handling duplicates deliberately) or any explicit `relationship=`
#' suppresses the auto-warning entirely (an explicit `relationship=` runs
#' its own check instead, which may itself pass silently, e.g.
#' `relationship = "many-to-many"`).
#' @param diag `list(left_multi_first, right_multi_first,
#'   left_unmatched_first, right_unmatched_first)` from `join_map_stats()`
#' @param join_type One of `"left"`, `"inner"`, `"full"`, `"right"`
#' @param multiple,relationship,unmatched As validated by
#'   `validate_join_multiple()`/`validate_join_relationship()`/
#'   `validate_join_unmatched()`
#' @return `invisible(NULL)`; called for its error/warning side effects only
#' @keywords internal
check_join_cardinality <- function(diag, join_type, multiple, relationship, unmatched) {
  left_multi <- diag$left_multi_first
  right_multi <- diag$right_multi_first
  left_unmatched <- diag$left_unmatched_first
  right_unmatched <- diag$right_unmatched_first

  # ---- 1. multiplicity: relationship= (if set) or the auto many-to-many
  # warning (relationship = NULL, multiple = "all" only) -- always evaluated
  # on the RAW match set, ahead of any unmatched= check below.
  if (!is.null(relationship)) {
    switch(relationship,
      "one-to-one" = check_relationship_one_to_one(left_multi, right_multi),
      "one-to-many" = check_relationship_one_to_many(right_multi),
      "many-to-one" = check_relationship_many_to_one(left_multi),
      "many-to-many" = invisible(NULL)
    )
  } else if (identical(multiple, "all")) {
    warn_relationship_many_to_many(left_multi, right_multi)
  }

  # ---- 2. unmatched= (x-side before y-side, applicability per join type).
  sides <- resolve_unmatched_sides(unmatched)
  check_x <- join_type %in% c("inner", "right")
  check_y <- join_type %in% c("inner", "left")

  if (check_x && identical(sides$x, "error") && left_unmatched >= 0) {
    abort_join_matches_nothing(left_unmatched)
  }
  if (check_y && identical(sides$y, "error") && right_unmatched >= 0) {
    abort_join_matches_remaining(right_unmatched)
  }

  invisible(NULL)
}

#' Route a join call to the transparent CPU fallback (Phase 7 J6)
#'
#' Shared by every mutating/semi/anti join verb below for the two shapes the
#' native GPU path can't (yet, or ever) handle:
#'   1. A non-equi `join_by()` object (`join_by_is_non_equi()` is `TRUE`) --
#'      Phase 7 J7 replaces this with a native `mixed_join`/`conditional_join`
#'      lowering; until then this keeps user code working exactly like every
#'      other not-yet-native verb (mirrors `nest_join()`'s two-table fallback
#'      shape, `R/fallback.R`: collect the OTHER table, then run the real
#'      dplyr verb on `.data`'s own materialized/grouped tibble via
#'      `gpu_fallback()`).
#'   2. The `suffix = c("", ...)` degenerate name-collision case detected by
#'      `join_output_would_collide()` -- dplyr resolves this via a
#'      name-keyed column overwrite this package's fixed-column-count gather
#'      can't reproduce natively; falls back rather than emitting a wrong or
#'      duplicate-named result.
#' `args` is the full named-argument list the corresponding `dplyr::<verb>()`
#' call needs (using the ORIGINAL, not yet `resolve_join_keep()`-resolved,
#' user-facing values where relevant, e.g. `by`/`keep`, so dplyr's own
#' defaulting/validation runs exactly as it would for a bare call) --
#' `y_tbl` is collected once and reused across every retry-shaped caller.
#' @keywords internal
join_route_to_fallback <- function(verb_name, verb_fn, x, y, args) {
  y_tbl <- collect(y)
  gpu_fallback(verb_name, x, function(tbl) {
    rlang::inject(verb_fn(tbl, y_tbl, !!!args))
  })
}

#' Mutating joins for `tbl_gpu` (`left_join()`, `inner_join()`, `full_join()`,
#' `right_join()`)
#'
#' GPU-native equi and non-equi joins mirroring `dplyr::left_join()`'s own
#' `by=`/`keep=`/`na_matches=`/`multiple=`/`unmatched=`/`relationship=`
#' semantics (Phase 7). Row order, key coalescing, suffixing, and every
#' cardinality check reproduce dplyr's own contract exactly -- see "Details"
#' below for what runs natively on the GPU versus what transparently falls
#' back to CPU (`dplyr`) evaluation.
#'
#' @param x,y `tbl_gpu` objects to join. If `y` is not a `tbl_gpu`, set
#'   `copy = TRUE` to upload it automatically (`tbl_gpu(y)`).
#' @param by Join specification: `NULL` (natural join on every common column
#'   name), an unnamed character vector (`c("a", "b")`, matched by identical
#'   name on both sides), a named character vector (`c("lx" = "ry")`, for
#'   differently-named keys), or a [dplyr::join_by()] spec. `join_by()`
#'   supports both equi conditions (`a`, `a == b`) and non-equi conditions
#'   (`>=`, `>`, `<=`, `<`, and the `between()`/`within()`/`overlaps()`
#'   helpers that pre-expand to pairs of them) -- all GPU-native (Phase 7
#'   J6/J7). `join_by(closest(...))`/rolling joins, and any non-equi
#'   condition comparing `STRING` columns, are not GPU-native and
#'   transparently fall back to CPU evaluation (a `cuplyr.fallback`
#'   notification fires; see `options(cuplyr.fallback = )`), still with a
#'   result that matches `dplyr` exactly.
#' @param copy Whether to upload a non-`tbl_gpu` `y` via `tbl_gpu()`.
#' @param suffix Length-2 character vector used to disambiguate colliding
#'   non-key column names (default `c(".x", ".y")`). Ported from
#'   `dplyr:::add_suffixes()`: an already-taken suffixed name gets suffixed
#'   again (`val.y.y`) rather than silently colliding. `suffix = c("", "")`
#'   (or any element set to `""`) that still collides with a genuine
#'   duplicate name reproduces dplyr's name-keyed column-overwrite semantics
#'   via a CPU fallback -- this package's fixed-column-count gather has no
#'   native way to express "the right column's data silently replaces the
#'   left's, and the output has one fewer column than the naive merge".
#' @param ... Not used; dplyr itself requires these to be empty.
#' @param keep Whether to keep both sides' join key columns in the output.
#'   `NULL` (the default) resolves *per condition*, exactly like
#'   `dplyr:::join_cols()`: an equi condition's key columns coalesce to one
#'   (the right-side copy is dropped), while a non-equi condition's columns
#'   are always kept on both sides (there is no single "coalesced" value for
#'   an inequality). Explicit `keep = FALSE` is only legal when every
#'   condition is equi (errors with dplyr's own text otherwise); explicit
#'   `keep = TRUE` keeps and suffixes every column on both sides uniformly.
#' @param na_matches `"na"` (default): an `NA` key matches another `NA` key,
#'   like `base::merge()`. `"never"`: `NA` keys never match anything,
#'   including another `NA` (`cudf::null_equality::UNEQUAL` on the GPU).
#' @param multiple How to resolve a left/right row matching more than one row
#'   on the other side: `"all"` (default, every match kept), `"any"`,
#'   `"first"`, or `"last"`. Filtering runs device-side against the
#'   already-sorted join maps (Phase 7 J5) -- `"all"`, the default used by
#'   every performance benchmark, costs nothing extra.
#' @param unmatched How to handle rows with no match: `"drop"` (default) or
#'   `"error"` (raises dplyr's exact `dplyr_error_join_matches_nothing`/
#'   `dplyr_error_join_matches_remaining` conditions, including the
#'   offending row number). `full_join()` has no `unmatched=` parameter at
#'   all, matching dplyr's own signature (it never drops a row).
#' @param relationship Optional cardinality assertion: `NULL` (default --
#'   emits dplyr's own many-to-many advisory warning whenever *both* sides
#'   have a genuine duplicate), `"one-to-one"`, `"one-to-many"`,
#'   `"many-to-one"`, or `"many-to-many"` (silences the advisory warning).
#'   A violation raises dplyr's exact `dplyr_error_join_relationship_*`
#'   condition, including the class vector and first-offending-row bullet.
#' @return A `tbl_gpu` with dplyr's own row/column contract for the given
#'   join type. `right_join()`'s row order is x-matched rows (in `x`'s own
#'   order) followed by any unmatched `y` rows appended last.
#'
#' @details
#' ## GPU-native coverage (Phase 7)
#' - Equi joins (`join_by(a)`, `by = "col"`, `by = c(a = "b")`): a hash join
#'   (`cudf::inner_join()`/`left_join()`/`full_join()`), stable-sorted to
#'   reproduce dplyr's row order (Phase 7 J1/J4/J6).
#' - Non-equi/mixed `join_by()` conditions (`>=`, `>`, `<=`, `<`, and their
#'   `between()`/`within()`/`overlaps()` derivations): `cudf::mixed_join()`
#'   when at least one equi condition is present, `cudf::conditional_join()`
#'   otherwise (Phase 7 J7).
#' - `na_matches=`, `multiple=`, `unmatched=`, `relationship=` are fully
#'   GPU-native/checked, with dplyr's exact condition classes and message
#'   text (Phase 7 J1/J5).
#'
#' ## Falls back to CPU (transparent, notified)
#' - `join_by(closest(...))` / any rolling join.
#' - A non-equi condition comparing `STRING` columns (cudf's AST comparison
#'   evaluator has no string support; equi `STRING` joins are unaffected).
#' - `suffix = c("", ...)` when it would still produce a genuine duplicate
#'   output column name.
#' - `nest_join()` stays CPU-only entirely (needs list-column support,
#'   Phase 11) -- it is not one of the four verbs documented here.
#'
#' @name mutating-joins
#' @rdname mutating-joins
#' @export
#' @importFrom dplyr left_join
left_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), ..., keep = NULL,
                              na_matches = "na", multiple = "all",
                              unmatched = "drop", relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!join_by_natively_supported(by, x, y)) {
    return(join_route_to_fallback("left_join", dplyr::left_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  suffix <- validate_join_suffix(suffix)
  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  unmatched <- validate_join_unmatched(unmatched, max_length = 1L)
  relationship <- validate_join_relationship(relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  if (join_factor_key_levels_conflict(x$schema, y$schema, join_spec)) {
    return(join_route_to_fallback("left_join", dplyr::left_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  keep <- resolve_join_keep(keep, join_spec)

  if (join_output_would_collide(x$schema, y$schema, join_spec, suffix, keep)) {
    return(join_route_to_fallback("left_join", dplyr::left_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  push_join("left", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = unmatched,
           relationship = relationship)
}

#' @rdname mutating-joins
#' @export
#' @importFrom dplyr inner_join
inner_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), ..., keep = NULL,
                               na_matches = "na", multiple = "all",
                               unmatched = "drop", relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!join_by_natively_supported(by, x, y)) {
    return(join_route_to_fallback("inner_join", dplyr::inner_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  suffix <- validate_join_suffix(suffix)
  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  unmatched <- validate_join_unmatched(unmatched, max_length = 2L)
  relationship <- validate_join_relationship(relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  if (join_factor_key_levels_conflict(x$schema, y$schema, join_spec)) {
    return(join_route_to_fallback("inner_join", dplyr::inner_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  keep <- resolve_join_keep(keep, join_spec)

  if (join_output_would_collide(x$schema, y$schema, join_spec, suffix, keep)) {
    return(join_route_to_fallback("inner_join", dplyr::inner_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  push_join("inner", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = unmatched,
           relationship = relationship)
}

#' @rdname mutating-joins
#' @export
#' @importFrom dplyr full_join
full_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                              suffix = c(".x", ".y"), ..., keep = NULL,
                              na_matches = "na", multiple = "all",
                              relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!join_by_natively_supported(by, x, y)) {
    return(join_route_to_fallback("full_join", dplyr::full_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, relationship = relationship
    )))
  }

  # full_join() has no `unmatched=` in dplyr (every row, matched or not, is
  # always kept), so it isn't accepted here either -- passing it hits `...`
  # and is silently ignored, same as dplyr's own `...`-must-be-empty dots
  # (dplyr hard-errors on that; this is a narrower gap, not a behavior this
  # task needs to close).
  suffix <- validate_join_suffix(suffix)
  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  relationship <- validate_join_relationship(relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  if (join_factor_key_levels_conflict(x$schema, y$schema, join_spec)) {
    return(join_route_to_fallback("full_join", dplyr::full_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, relationship = relationship
    )))
  }

  keep <- resolve_join_keep(keep, join_spec)

  if (join_output_would_collide(x$schema, y$schema, join_spec, suffix, keep)) {
    return(join_route_to_fallback("full_join", dplyr::full_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, relationship = relationship
    )))
  }

  push_join("full", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = "drop",
           relationship = relationship)
}

#' @rdname mutating-joins
#' @export
#' @importFrom dplyr right_join
right_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE,
                               suffix = c(".x", ".y"), ..., keep = NULL,
                               na_matches = "na", multiple = "all",
                               unmatched = "drop", relationship = NULL) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!join_by_natively_supported(by, x, y)) {
    return(join_route_to_fallback("right_join", dplyr::right_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  suffix <- validate_join_suffix(suffix)
  na_matches <- validate_join_na_matches(na_matches)
  multiple <- validate_join_multiple(multiple)
  unmatched <- validate_join_unmatched(unmatched, max_length = 1L)
  relationship <- validate_join_relationship(relationship)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  if (join_factor_key_levels_conflict(x$schema, y$schema, join_spec)) {
    return(join_route_to_fallback("right_join", dplyr::right_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  keep <- resolve_join_keep(keep, join_spec)

  if (join_output_would_collide(x$schema, y$schema, join_spec, suffix, keep)) {
    return(join_route_to_fallback("right_join", dplyr::right_join, x, y, list(
      by = by, suffix = suffix, keep = keep, na_matches = na_matches,
      multiple = multiple, unmatched = unmatched, relationship = relationship
    )))
  }

  # right_join(x, y) is native (Phase 7 J4): gpu_right_join()
  # (src/ops_join.cpp) computes matched (x, y) pairs via cudf::inner_join()
  # plus unmatched-y rows via a filtered_join anti-join, assembled so the
  # shared stable-sort/gather/coalesce pipeline (build_join_result(), used
  # by every mutating join type) reproduces dplyr's row-order contract:
  # x-matched rows in x's order, then unmatched-y rows appended last. This
  # is just another "right"-typed ast_join node -- push_join()
  # (R/execute.R) dispatches it through the same unified eager/lazy path as
  # the other join types, and lower_join()'s "right" case (R/lower.R) calls
  # gpu_right_join() with exactly the same argument shape as
  # gpu_left_join()/gpu_inner_join()/gpu_full_join(), so both schedules
  # agree on the resulting schema.
  push_join("right", x, y, join_spec, suffix = suffix, keep = keep,
           na_matches = na_matches, multiple = multiple, unmatched = unmatched,
           relationship = relationship)
}

#' `semi_join()`/`anti_join()`: native via `cudf::filtered_join`
#'
#' Native GPU implementation via `cudf::filtered_join` (Phase 7 J2). Unlike
#' the mutating joins, dplyr's own `semi_join()`/`anti_join()` signature has
#' no `suffix`/`keep`/`multiple`/`unmatched`/`relationship` arguments at all
#' (verified via `args(dplyr:::semi_join.data.frame)`: `x, y, by, copy, ...,
#' na_matches`) -- there's nothing to merge, so no column-collision handling
#' is needed. The result is x's own schema verbatim (a row subset of x),
#' and -- verified empirically against dplyr 1.2.1 -- x's `group_by()`
#' grouping is PRESERVED (these are filters, not the column-merging
#' mutating joins that `push_join()`/`propagate_groups()`'s D3 rule always
#' clears groups for); `push_join()` special-cases `join_type %in%
#' c("semi", "anti")` to carry `x$groups` forward instead of dropping them.
#'
#' `by` accepts the same forms as the mutating joins (see `?mutating-joins`):
#' `NULL`, a character vector, or a [dplyr::join_by()] spec, including
#' non-equi conditions (`>=`, `>`, `<=`, `<`, `between()`/`within()`/
#' `overlaps()`) via `cudf::conditional_join`/`mixed_join` (Phase 7 J7).
#' `join_by(closest(...))`/rolling joins, and non-equi conditions over
#' `STRING` columns, fall back to CPU evaluation transparently (with a
#' `cuplyr.fallback` notification), same as the mutating joins.
#'
#' @param x,y `tbl_gpu` objects to join. If `y` is not a `tbl_gpu`, set
#'   `copy = TRUE` to upload it automatically.
#' @param by Join specification -- see "Details" above and `?mutating-joins`.
#' @param copy Whether to upload a non-`tbl_gpu` `y` via `tbl_gpu()`.
#' @param ... Not used; dplyr itself requires these to be empty.
#' @param na_matches `"na"` (default, `NA` keys match each other) or
#'   `"never"` (`NA` keys never match anything).
#' @return A `tbl_gpu` with `x`'s own schema, filtered to the rows that
#'   (`semi_join()`) or don't (`anti_join()`) have a match in `y`. `x`'s
#'   `group_by()` grouping is preserved (these are row filters, not
#'   column-merging joins).
#' @name semi-anti-joins
#' @rdname semi-anti-joins
#' @export
#' @importFrom dplyr semi_join
semi_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, ...,
                              na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!join_by_natively_supported(by, x, y)) {
    return(join_route_to_fallback("semi_join", dplyr::semi_join, x, y, list(
      by = by, na_matches = na_matches
    )))
  }

  na_matches <- validate_join_na_matches(na_matches)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  if (join_factor_key_levels_conflict(x$schema, y$schema, join_spec)) {
    return(join_route_to_fallback("semi_join", dplyr::semi_join, x, y, list(
      by = by, na_matches = na_matches
    )))
  }

  push_join("semi", x, y, join_spec, suffix = c(".x", ".y"), keep = FALSE,
           na_matches = na_matches)
}

#' `is_anti = TRUE` in the shared `gpu_semi_anti_join()` C++ entry point --
#' same native `filtered_join`-based implementation and
#' grouping-preservation rationale as `semi_join()`, see `?semi-anti-joins`.
#' @rdname semi-anti-joins
#' @export
#' @importFrom dplyr anti_join
anti_join.tbl_gpu <- function(x, y, by = NULL, copy = FALSE, ...,
                              na_matches = "na") {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (!join_by_natively_supported(by, x, y)) {
    return(join_route_to_fallback("anti_join", dplyr::anti_join, x, y, list(
      by = by, na_matches = na_matches
    )))
  }

  na_matches <- validate_join_na_matches(na_matches)

  join_spec <- parse_join_by(by, x, y)
  validate_join_cols(join_spec$left, x, "Left")
  validate_join_cols(join_spec$right, y, "Right")
  validate_key_types(x, y, join_spec)

  if (join_factor_key_levels_conflict(x$schema, y$schema, join_spec)) {
    return(join_route_to_fallback("anti_join", dplyr::anti_join, x, y, list(
      by = by, na_matches = na_matches
    )))
  }

  push_join("anti", x, y, join_spec, suffix = c(".x", ".y"), keep = FALSE,
           na_matches = na_matches)
}

#' `cross_join()`: native via `cudf::cross_join`
#'
#' Native GPU implementation (Phase 7 J3). Unlike every other join verb,
#' dplyr's own `cross_join()` signature has no `by`/`keep`/`na_matches`/
#' `multiple`/`unmatched`/`relationship` at all (verified via
#' `args(dplyr::cross_join)`: `x, y, ..., copy, suffix`) -- there are no keys
#' to match, every row of `x` is paired with every row of `y`. Verified
#' empirically against dplyr 1.2.1:
#' - Output column order is left-major: all of `x`'s columns (in their
#'   original order), then all of `y`'s (in their original order).
#' - Every COMMON column name (not just would-be join keys, since there are
#'   none) gets BOTH suffixes -- e.g. two tables both having `id`/`val`
#'   produces `id.x`/`val.x`/`id.y`/`val.y`, never a bare shared name. This
#'   is exactly `build_join_output_info()`'s own behavior for an all-common
#'   spec with `keep = TRUE` (its `drop_right` is always `character(0)` when
#'   `keep = TRUE`, regardless of what the spec's `left`/`right` contain), so
#'   an empty join spec (no keys at all) plus `keep = TRUE` reproduces it
#'   exactly -- see `ast_join()`'s `by`/`keep` fields below.
#' - Grouping is NOT unconditionally cleared, unlike `left_join()`/
#'   `inner_join()`/`full_join()`/`right_join()` (D3): dplyr's own
#'   `cross_join.data.frame()` ends with `dplyr_reconstruct(x_out, x)`,
#'   which keeps `x`'s grouping over exactly the group-var NAMES that
#'   survive, unchanged, in the output. A group column whose name is unique
#'   to `x` stays grouped; one whose name collides with `y`'s (and gets
#'   suffixed away, e.g. `"g"` -> `"g.x"`) is silently dropped from the
#'   grouping. `y`'s own grouping is always irrelevant. See `push_join()`'s
#'   (`R/execute.R`) `preserves_x_groups` handling, which reuses the exact
#'   same `intersect(x$groups, new_schema$names)` formula already used for
#'   `semi_join()`/`anti_join()` -- a colliding group column is renamed away
#'   from the new schema's names by `build_join_output_info()`'s own
#'   suffixing rule, so the intersection naturally excludes it with no
#'   cross-specific logic needed.
#' - 0-row `x` or `y` produces a 0-row result with the correct (merged)
#'   schema -- cudf's `cross_join()` handles this natively, no R-side
#'   special-casing needed.
#' - `copy = TRUE` with a plain `data.frame` `y` works like every other join.
#'
#' `gpu_cross_join()` (`src/ops_join.cpp`) throws if either side has 0
#' *columns* (verified in the `cudf/join/join.hpp` doc comment: "throw
#' cudf::logic_error if the number of columns in either left or right table
#' is 0") -- guarded here with a clearer, dplyr-flavored message before ever
#' reaching the GPU call, for both the eager and lazy schedules (the 0-column
#' case is knowable from the schemas alone, no execution needed).
#'
#' @param x,y `tbl_gpu` objects. If `y` is not a `tbl_gpu`, set
#'   `copy = TRUE` to upload it automatically.
#' @param ... Not used; dplyr itself requires these to be empty.
#' @param copy Whether to upload a non-`tbl_gpu` `y` via `tbl_gpu()`.
#' @param suffix Length-2 character vector disambiguating every common
#'   column name (every shared name gets both suffixes, since there are no
#'   join keys to coalesce). `suffix = c("", ...)` colliding with a genuine
#'   duplicate name falls back to CPU evaluation, same as the mutating
#'   joins' degenerate-suffix case.
#' @return A `tbl_gpu` with every row of `x` paired with every row of `y`
#'   (`nrow(x) * nrow(y)` rows), columns from `x` first then `y`. `x`'s
#'   `group_by()` grouping is preserved for any group column whose name
#'   survives suffixing unchanged.
#' @export
#' @importFrom dplyr cross_join
cross_join.tbl_gpu <- function(x, y, ..., copy = FALSE, suffix = c(".x", ".y")) {
  if (!is_tbl_gpu(y)) {
    if (isTRUE(copy)) {
      y <- tbl_gpu(y)
    } else {
      stop("`y` must be a tbl_gpu or set copy = TRUE.", call. = FALSE)
    }
  }

  if (length(x$schema$names) == 0 || length(y$schema$names) == 0) {
    stop("cross_join() requires both tables to have at least one column.",
         call. = FALSE)
  }

  suffix <- validate_join_suffix(suffix)

  # No keys at all -- an empty spec, `keep = TRUE` so build_join_output_info()
  # suffixes every common name (see roxygen above).
  join_spec <- list(left = character(0), right = character(0),
                    op = character(0), filter = "none")

  # Phase 7 J6: the same `suffix = c("", ...)` degenerate name-collision
  # case the mutating joins guard against (see `join_output_would_collide()`)
  # -- computable from schemas alone, routed to the CPU fallback rather than
  # emitting a wrong/duplicate-named result.
  if (join_output_would_collide(x$schema, y$schema, join_spec, suffix, TRUE)) {
    return(join_route_to_fallback("cross_join", dplyr::cross_join, x, y, list(
      suffix = suffix
    )))
  }

  push_join("cross", x, y, join_spec, suffix = suffix, keep = TRUE,
           na_matches = "na")
}
