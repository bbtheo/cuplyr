# Set operations: union(), union_all(), intersect(), setdiff(), setequal(),
# symdiff() (Phase 8).
#
# Composed entirely from existing GPU-native primitives -- no new C++:
#   union(x, y)      = distinct(bind_rows(x_aligned, y_aligned))
#   union_all(x, y)  = bind_rows(x_aligned, y_aligned)
#   intersect(x, y)  = distinct(semi_join(x_aligned, y_aligned, by = <all cols>))
#   setdiff(x, y)    = distinct(anti_join(x_aligned, y_aligned, by = <all cols>))
#   symdiff(x, y)    = bind_rows(setdiff(x_aligned, y_aligned), setdiff(y_aligned, x_aligned))
#   setequal(x, y)   = nrow(anti_join(x, y)) == 0 && nrow(anti_join(y, x)) == 0
# (`bind_rows()`/`bind_cols()` are this package's own top-level generics,
# R/bind.R; `distinct()`/`semi_join()`/`anti_join()` are dplyr generics this
# package registers `tbl_gpu` S3 methods for, R/distinct.R/R/join.R.)
#
# `semi_join()`/`anti_join()` were verified (Phase 7 J2) to keep only *x*'s
# duplicate rows that match *y* -- they do NOT collapse x's own internal
# duplicates, hence the `distinct()` after every one of them (verified
# empirically against dplyr 1.2.1: `intersect(tibble(a = c(2, 2, 3)),
# tibble(a = c(2, 4)))` returns ONE row, not two).
#
# Row-order contract (verified empirically against dplyr 1.2.1, see
# scratchpad/workflow_state.md Phase 8 kickoff note):
#   - union(): x's own distinct rows (in x's row order) first, then any
#     row from y not already present in x, in y's row order -- exactly what
#     `distinct(bind_rows(x, y))` (keep-first dedup) produces for free.
#   - union_all(): x's rows (as-is, including x's own duplicates) then y's.
#   - intersect()/setdiff(): x's row order (rows are always a subset of x).
#   - symdiff(): setdiff(x, y)'s rows, then setdiff(y, x)'s rows.
#   - setequal(): TRUE/FALSE, ignoring both row order and duplicates on
#     both sides (`vec_in()`-style membership check, not a literal
#     `identical()`).
#
# Column-compatibility contract (ported from `dplyr:::is_compatible()` /
# `check_compatible()`, verified via `body()` against dplyr 1.2.1 + vctrs
# 0.6+ `vec_ptype2()` common-type rules -- NOT the same, and deliberately
# STRICTER, than this package's own `bind_rows()` type promotion, see
# `check_set_op_compatible()` below for why):
#   - Same column count, same column NAME SET; column ORDER may differ
#     between x and y (y is silently reordered to x's order) but a
#     different column set (extra/missing names) is a hard error naming
#     the offending column(s) on each side.
#   - Per-column common type: identical types are always fine; the numeric
#     ladder (logical < integer < integer64 < double) and the timestamp
#     ladder (date < ... < datetime) promote to the wider member; anything
#     else (e.g. integer vs. character, or a numeric vs. a date) is a hard
#     error naming the column and both types. Factor columns get their own
#     handling (see `check_set_op_compatible()`).
#   - Output type/column order always follows x's (`vec_ptype2(x_col,
#     y_col)` order, `dplyr_reconstruct(out, x)`'s column order) --
#     verified empirically (swapping y's column order changes nothing
#     about the result).
#
# `...` must be empty for every one of these six verbs (verified via
# `body(dplyr:::union.data.frame)` et al: `check_dots_empty()` is the
# first statement in every one) -- `rlang::check_dots_empty()` produces a
# byte-identical error (same function dplyr itself calls).

#' Friendly type label for set-op incompatible-type error messages
#'
#' cuplyr's GPU type strings (`INT32`, `FLOAT64`, ...) don't correspond
#' 1:1 to R's/vctrs' own type vocabulary (`vec_ptype_full()`), so this is a
#' hand-picked "close enough" label used only in error text -- not an
#' attempt to byte-match `vec_ptype_full()`'s output (which for factors
#' includes a levels-derived hash we have no equivalent for).
#' @keywords internal
set_op_type_label <- function(type, is_factor = FALSE) {
  if (isTRUE(is_factor)) {
    return("factor")
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
    TIMESTAMP_NANOSECONDS = "datetime",
    type
  )
}

#' Is `name` a factor column in this schema?
#' @keywords internal
set_op_is_factor <- function(schema, name) {
  !is.null(schema$factor_levels) && !is.null(schema$factor_levels[[name]])
}

#' Raise dplyr's own "`x` and `y` are not compatible." error
#'
#' `bullets` is pre-formatted, final display text (already `"✖ "`-
#' prefixed where dplyr itself would show a bullet, plain otherwise -- see
#' the "`y` must be a data frame." case, which dplyr renders with NO
#' bullet marker at all). Joined into a single pre-formatted string rather
#' than handed to `rlang::abort()` as a named (`x = ...`) character vector:
#' verified empirically that `rlang::abort()`'s automatic cli bullet
#' decoration for an UNNAMED continuation line depends on the calling
#' package/frame context in ways this package can't reliably control, so
#' bullets are embedded literally instead of relying on that formatting --
#' this reproduces dplyr's exact `conditionMessage()` byte-for-byte
#' regardless of context.
#' @keywords internal
abort_set_op_incompatible <- function(bullets) {
  rlang::abort(paste(c("`x` and `y` are not compatible.", bullets), collapse = "\n"))
}

#' Resolve `y` for a set operation: must be a `tbl_gpu` or a `data.frame`
#'
#' Unlike the join verbs, dplyr's own set-op generics have no `copy=`
#' argument at all (verified via `args(dplyr::union)`: `x, y, ...`) -- a
#' plain `data.frame` `y` is uploaded automatically (mirrors
#' `collect_other_side()`'s inverse, R/fallback.R), and anything else
#' (e.g. a bare vector) gets dplyr's own exact error text (verified
#' empirically: `union(tibble(a = 1), 1:3)` errors with "`y` must be a
#' data frame.", no bullet marker).
#' @keywords internal
resolve_set_op_other_side <- function(y) {
  if (is_tbl_gpu(y)) {
    return(y)
  }
  if (is.data.frame(y)) {
    return(tbl_gpu(y, lazy = FALSE))
  }
  abort_set_op_incompatible("`y` must be a data frame.")
}

#' Check column-set/type compatibility for a set operation, dplyr-style
#'
#' Mirrors `dplyr:::is_compatible()` (verified empirically against dplyr
#' 1.2.1 + vctrs `vec_ptype2()`): same column count, same column name SET
#' (order tolerated), and a per-column common type. Deliberately NOT the
#' same promotion rule as this package's own `bind_rows()`
#' (`promote_types()`, R/bind.R): that function treats `STRING` as a
#' universal widest type (any type silently coerces to STRING), which is
#' exactly the coercion vctrs' `vec_ptype2()` REFUSES for a numeric/date
#' column vs. a character one (verified: `vec_ptype2(1L, "a")` errors,
#' unlike `bind_rows()`'s silent STRING promotion) -- dplyr's real set-op
#' contract is the STRICTER one, so this reimplements vctrs' common-type
#' ladder rather than reusing `promote_types()`.
#'
#' Factor columns get a four-way split (verified against real
#' `vec_ptype2()` behavior for factors, see inline comments): both-factor
#' with IDENTICAL levels is native (same underlying INT32 codes, safe to
#' compare/concatenate directly); both-factor with DIFFERING levels is ALSO
#' native as of Phase 11 L2 (`gpu_remap_codes()`, `src/ops_bind.cpp`,
#' remaps each side's codes onto the level UNION, mirroring `bind_rows()`'s
#' own Bug 2 fix); factor-vs-character is ALSO native as of Phase 11 L2
#' (`gpu_decode_factor()` decodes the factor side to its labels, mirroring
#' `bind_rows()`'s own Bug 3 fix) -- both used to route through
#' `$needs_fallback` before these two GPU primitives existed. Factor vs.
#' anything else (numeric, date, ...) is a genuine vctrs error, reproduced
#' here directly. `$needs_fallback` now only fires for the `Date`-vs-
#' `POSIXct` timestamp-promotion gap (see the timestamp ladder branch
#' below).
#'
#' @param x_schema,y_schema Schemas (`current_schema()`) of `x`/`y`.
#' @return A list with `$target_schema` (unified schema, x's names/order)
#'   and `$needs_fallback` (logical).
#' @keywords internal
check_set_op_compatible <- function(x_schema, y_schema) {
  nx <- length(x_schema$names)
  ny <- length(y_schema$names)
  if (nx != ny) {
    abort_set_op_incompatible(sprintf(
      "✖ Different number of columns: %d vs %d.", nx, ny
    ))
  }

  names_x <- x_schema$names
  names_y <- y_schema$names
  y_not_in_x <- setdiff(names_y, names_x)
  x_not_in_y <- setdiff(names_x, names_y)
  if (length(y_not_in_x) > 0 || length(x_not_in_y) > 0) {
    bullets <- character()
    if (length(y_not_in_x) > 0) {
      bullets <- c(bullets, sprintf(
        "✖ Cols in `y` but not `x`: %s.",
        paste(sprintf("`%s`", y_not_in_x), collapse = ", ")
      ))
    }
    if (length(x_not_in_y) > 0) {
      bullets <- c(bullets, sprintf(
        "✖ Cols in `x` but not `y`: %s.",
        paste(sprintf("`%s`", x_not_in_y), collapse = ", ")
      ))
    }
    abort_set_op_incompatible(bullets)
  }

  numeric_ladder <- c("BOOL8", "INT32", "INT64", "FLOAT64")
  timestamp_ladder <- c("TIMESTAMP_DAYS", "TIMESTAMP_SECONDS",
                         "TIMESTAMP_MILLISECONDS", "TIMESTAMP_MICROSECONDS",
                         "TIMESTAMP_NANOSECONDS")

  promoted_types <- character(nx)
  factor_levels <- list()
  needs_fallback <- FALSE

  for (i in seq_len(nx)) {
    nm <- names_x[i]
    tx <- unname(x_schema$types[match(nm, x_schema$names)])
    ty <- unname(y_schema$types[match(nm, y_schema$names)])
    fx <- set_op_is_factor(x_schema, nm)
    fy <- set_op_is_factor(y_schema, nm)

    if (fx && fy) {
      # Level UNION (Bug 2 fix, shared contract with bind_rows()'s own
      # compute_unified_schema()): identical levels is a no-op union;
      # differing levels is native too -- align_set_op_side() ->
      # align_to_schema() remaps each side's codes via gpu_remap_codes().
      lvls_x <- x_schema$factor_levels[[nm]]
      lvls_y <- y_schema$factor_levels[[nm]]
      promoted_types[i] <- tx
      factor_levels[[nm]] <- if (identical(lvls_x, lvls_y)) lvls_x else union(lvls_x, lvls_y)
      next
    }

    if (xor(fx, fy)) {
      other_type <- if (fx) ty else tx
      if (identical(other_type, "STRING")) {
        # Decode contract (Bug 3 fix): factor + character -> character.
        # align_to_schema()'s cast_column() decodes the factor side via
        # gpu_decode_factor() -- no factor_levels entry on the result (it's
        # STRING, not DICTIONARY32).
        promoted_types[i] <- "STRING"
      } else {
        abort_set_op_incompatible(sprintf(
          "✖ Incompatible types for column `%s`: %s vs %s.",
          nm, set_op_type_label(tx, fx), set_op_type_label(ty, fy)
        ))
      }
      next
    }

    if (identical(tx, ty)) {
      promoted_types[i] <- tx
    } else if (tx %in% numeric_ladder && ty %in% numeric_ladder) {
      promoted_types[i] <- numeric_ladder[max(match(tx, numeric_ladder), match(ty, numeric_ladder))]
    } else if (tx %in% timestamp_ladder && ty %in% timestamp_ladder) {
      # vctrs (and dplyr's set ops) happily promote e.g. Date + POSIXct to
      # POSIXct (verified empirically) -- but this package's own
      # `gpu_cast_column()` (src/ops_bind.cpp, shared with `bind_rows()`)
      # only implements numeric-family casts, not a real TIMESTAMP_DAYS ->
      # TIMESTAMP_MICROSECONDS cast (a pre-existing `bind_rows()` gap this
      # phase doesn't fix -- out of scope, no new cuDF primitive families
      # expected here). Since the two types already differ (this branch is
      # only reached when `identical(tx, ty)` above was FALSE), an actual
      # cast is always required here and always unsupported -- defer to
      # the CPU fallback rather than let `align_to_schema()` crash with a
      # raw C++ "Unsupported target type for casting" exception.
      needs_fallback <- TRUE
      promoted_types[i] <- timestamp_ladder[max(match(tx, timestamp_ladder), match(ty, timestamp_ladder))]
    } else {
      abort_set_op_incompatible(sprintf(
        "✖ Incompatible types for column `%s`: %s vs %s.",
        nm, set_op_type_label(tx, FALSE), set_op_type_label(ty, FALSE)
      ))
    }
  }

  list(
    target_schema = list(
      names = names_x,
      types = promoted_types,
      factor_levels = if (length(factor_levels) > 0) factor_levels else NULL
    ),
    needs_fallback = needs_fallback
  )
}

#' Align one side of a set operation to the unified target schema
#'
#' A thin, laziness-preserving wrapper over `align_to_schema()`
#' (R/bind.R): that function assumes an already-*materialized* table (it
#' reads/casts `tbl$ptr`/`tbl$schema` directly, with no AST awareness --
#' the same reason `bind_rows_gpu()` materializes every lazy input before
#' calling it), so this only materializes (and only then calls
#' `align_to_schema()`) when `current_schema()` (the lazy-AST-aware
#' schema, unlike `tbl$schema`) actually differs from the target -- the
#' common case (x/y already schema-identical) skips both entirely,
#' preserving whatever laziness `tbl` already had. Names/types alone don't
#' catch every case that needs a real align pass: two `DICTIONARY32`
#' columns can share the type string while their level VECTORS differ
#' (Bug 2's native remap path), so `factor_levels_match_target()`
#' (R/bind.R) is checked too. `align_to_schema()`'s own return also
#' resets `$groups` to `character()` -- restored here from the
#' caller-supplied `groups`; its `$schema$factor_levels` is reset to
#' whatever it computed on its own slow path (unset on the fast-path
#' passthrough), so it's overwritten unconditionally here with the
#' already-computed `target_schema$factor_levels` for consistency either way.
#' @keywords internal
align_set_op_side <- function(tbl, target_schema, groups = character()) {
  schema <- current_schema(tbl)
  needs_align <- !identical(schema$names, target_schema$names) ||
    !identical(unname(schema$types), unname(target_schema$types)) ||
    !factor_levels_match_target(schema$factor_levels, target_schema$factor_levels,
                                target_schema$names, unname(target_schema$types))

  if (needs_align) {
    if (identical(tbl$exec_mode, "lazy") && has_pending_ops(tbl)) {
      tbl <- compute(tbl)
    }
    tbl <- align_to_schema(tbl, target_schema)
  }

  tbl$groups <- groups
  tbl$schema$factor_levels <- target_schema$factor_levels
  # `align_to_schema()`'s slow path always returns an unnamed `$types`
  # (its own `unname()` calls), but a fast-path passthrough (schema
  # already matched, `tbl` untouched) keeps whatever naming the ORIGINAL
  # table's `$schema$types` had -- `tbl_gpu()` builds it via `vapply()`,
  # which names it by column by default. Left inconsistent, one side
  # ending up named and the other not makes `identical(lt, rt)` in
  # `validate_key_types()` (R/join.R) see a spurious mismatch even when
  # the actual type strings match (an attribute-only difference) --
  # unnamed unconditionally here so both sides are always comparably
  # unnamed downstream.
  tbl$schema$types <- unname(tbl$schema$types)
  tbl
}

#' Reattach factor levels after `bind_rows()`
#'
#' As of Phase 11 L2, `bind_rows()` (R/bind.R) already computes and attaches
#' the correct unioned `schema$factor_levels` on its own result (Bug 2
#' fix) -- since `x_aligned`/`y_aligned` are already aligned to the exact
#' same `target_schema`, `bind_rows_gpu()`'s own internal
#' `compute_unified_schema()` call trivially recomputes the identical
#' schema `check_set_op_compatible()` already computed here, so this is now
#' a belt-and-braces no-op re-assignment (kept for defensiveness/clarity
#' rather than trusting that invariant to hold silently at a distance).
#' `intersect()`/`setdiff()` (built from `semi_join()`/`anti_join()` +
#' `distinct()`, which already preserve `factor_levels` via
#' `propagate_factor_levels()`, R/execute.R) don't need this at all.
#' @keywords internal
restore_set_op_factor_levels <- function(result, target_schema) {
  result$schema$factor_levels <- target_schema$factor_levels
  result
}

#' Set operations on `tbl_gpu` tables
#'
#' GPU-native implementations of dplyr's row-set operations, composed
#' entirely from existing verbs (`bind_rows()`/`distinct()`/`semi_join()`/
#' `anti_join()`) -- see the file header comment in `R/sets.R` for the
#' exact composition and the empirically-verified row-order/column-
#' compatibility contracts (mirroring `dplyr:::is_compatible()` /
#' `check_compatible()`).
#'
#' Two factor columns with different level sets (level-union remap,
#' `gpu_remap_codes()`) and a factor column vs. a character column (decode
#' to labels, `gpu_decode_factor()`) are both GPU-native as of Phase 11 L2.
#' One column-type combination that dplyr itself accepts still isn't
#' implemented natively and transparently falls back to CPU evaluation
#' instead (a `cuplyr.fallback` notification fires, see
#' [cuplyr_fallback_notify()]): a `Date` column vs. a `POSIXct` column
#' (dplyr promotes to `POSIXct`; the underlying GPU cast this package's
#' `bind_rows()` machinery would need -- `TIMESTAMP_DAYS` ->
#' `TIMESTAMP_MICROSECONDS` -- isn't implemented, see
#' `check_set_op_compatible()`). Every other column-type mismatch that
#' would be a genuine error in real dplyr (e.g. integer vs. character, or
#' factor vs. integer) errors here too, naming the offending column and
#' both types.
#'
#' @param x,y `tbl_gpu` objects. `y` may also be a plain `data.frame`
#'   (uploaded automatically via [tbl_gpu()]) -- unlike the join verbs,
#'   these generics have no `copy=` argument.
#' @param ... Not used; must be empty (dplyr itself requires this too).
#' @return A `tbl_gpu` with `x`'s column order/grouping (see the file
#'   header comment for the exact row-order contract per verb), except
#'   `setequal()` which returns a plain `TRUE`/`FALSE`.
#' @name set-ops
#' @rdname set-ops
NULL

#' @rdname set-ops
#' @export
#' @importFrom dplyr union
union.tbl_gpu <- function(x, y, ...) {
  rlang::check_dots_empty()
  y <- resolve_set_op_other_side(y)
  chk <- check_set_op_compatible(current_schema(x), current_schema(y))

  if (chk$needs_fallback) {
    return(gpu_fallback("union", x, function(tbl) {
      dplyr::union(tbl, collect(y))
    }))
  }

  x_aligned <- align_set_op_side(x, chk$target_schema, groups = x$groups)
  y_aligned <- align_set_op_side(y, chk$target_schema, groups = character())

  combined <- bind_rows(x_aligned, y_aligned)
  result <- dplyr::distinct(combined)
  result$groups <- x$groups
  restore_set_op_factor_levels(result, chk$target_schema)
}

#' @rdname set-ops
#' @export
#' @importFrom dplyr union_all
union_all.tbl_gpu <- function(x, y, ...) {
  rlang::check_dots_empty()
  y <- resolve_set_op_other_side(y)
  chk <- check_set_op_compatible(current_schema(x), current_schema(y))

  if (chk$needs_fallback) {
    return(gpu_fallback("union_all", x, function(tbl) {
      dplyr::union_all(tbl, collect(y))
    }))
  }

  x_aligned <- align_set_op_side(x, chk$target_schema, groups = x$groups)
  y_aligned <- align_set_op_side(y, chk$target_schema, groups = character())

  result <- bind_rows(x_aligned, y_aligned)
  result$groups <- x$groups
  restore_set_op_factor_levels(result, chk$target_schema)
}

#' @rdname set-ops
#' @export
#' @importFrom dplyr intersect
intersect.tbl_gpu <- function(x, y, ...) {
  rlang::check_dots_empty()
  y <- resolve_set_op_other_side(y)
  chk <- check_set_op_compatible(current_schema(x), current_schema(y))

  if (chk$needs_fallback) {
    return(gpu_fallback("intersect", x, function(tbl) {
      dplyr::intersect(tbl, collect(y))
    }))
  }

  target_names <- chk$target_schema$names
  x_aligned <- align_set_op_side(x, chk$target_schema, groups = x$groups)
  y_aligned <- align_set_op_side(y, chk$target_schema, groups = character())

  matched <- dplyr::semi_join(x_aligned, y_aligned, by = target_names, na_matches = "na")
  dplyr::distinct(matched)
}

#' @rdname set-ops
#' @export
#' @importFrom dplyr setdiff
setdiff.tbl_gpu <- function(x, y, ...) {
  rlang::check_dots_empty()
  y <- resolve_set_op_other_side(y)
  chk <- check_set_op_compatible(current_schema(x), current_schema(y))

  if (chk$needs_fallback) {
    return(gpu_fallback("setdiff", x, function(tbl) {
      dplyr::setdiff(tbl, collect(y))
    }))
  }

  target_names <- chk$target_schema$names
  x_aligned <- align_set_op_side(x, chk$target_schema, groups = x$groups)
  y_aligned <- align_set_op_side(y, chk$target_schema, groups = character())

  unmatched <- dplyr::anti_join(x_aligned, y_aligned, by = target_names, na_matches = "na")
  dplyr::distinct(unmatched)
}

#' @rdname set-ops
#' @export
#' @importFrom dplyr symdiff
symdiff.tbl_gpu <- function(x, y, ...) {
  rlang::check_dots_empty()
  y <- resolve_set_op_other_side(y)
  chk <- check_set_op_compatible(current_schema(x), current_schema(y))

  if (chk$needs_fallback) {
    return(gpu_fallback("symdiff", x, function(tbl) {
      dplyr::symdiff(tbl, collect(y))
    }))
  }

  target_names <- chk$target_schema$names
  x_aligned <- align_set_op_side(x, chk$target_schema, groups = x$groups)
  y_aligned <- align_set_op_side(y, chk$target_schema, groups = character())

  # `y_aligned` is already reordered to `target_names` (x's column order)
  # by `align_set_op_side()`, so `anti_join(y_aligned, x_aligned, ...)`'s
  # verbatim-y_aligned-schema output needs no further reordering before
  # concatenating with the x-side setdiff -- both are already the same
  # (target) schema.
  x_minus_y <- dplyr::distinct(dplyr::anti_join(x_aligned, y_aligned, by = target_names, na_matches = "na"))
  y_minus_x <- dplyr::distinct(dplyr::anti_join(y_aligned, x_aligned, by = target_names, na_matches = "na"))

  result <- bind_rows(x_minus_y, y_minus_x)
  result$groups <- x$groups
  restore_set_op_factor_levels(result, chk$target_schema)
}

#' @rdname set-ops
#' @export
#' @importFrom dplyr setequal
setequal.tbl_gpu <- function(x, y, ...) {
  rlang::check_dots_empty()
  y <- resolve_set_op_other_side(y)
  chk <- check_set_op_compatible(current_schema(x), current_schema(y))

  if (chk$needs_fallback) {
    return(gpu_fallback("setequal", x, function(tbl) {
      dplyr::setequal(tbl, collect(y))
    }, as_is = TRUE))
  }

  target_names <- chk$target_schema$names
  x_aligned <- align_set_op_side(x, chk$target_schema, groups = character())
  y_aligned <- align_set_op_side(y, chk$target_schema, groups = character())

  # Mirrors `dplyr:::setequal.data.frame()`'s own
  # `all(vec_in(x, y)) && all(vec_in(y, x))`: every row of x (duplicates
  # included) must have a match somewhere in y, and vice versa -- exactly
  # "both anti-joins are empty" (short-circuits like dplyr's `&&`, no need
  # to compute the second anti-join if the first already proves FALSE).
  x_not_in_y <- compute(dplyr::anti_join(x_aligned, y_aligned, by = target_names, na_matches = "na"))
  if (nrow(x_not_in_y) > 0) {
    return(FALSE)
  }
  y_not_in_x <- compute(dplyr::anti_join(y_aligned, x_aligned, by = target_names, na_matches = "na"))
  nrow(y_not_in_x) == 0
}
