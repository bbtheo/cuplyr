# cuplyr (development version)

## The long tail: universal CPU fallback, factor fidelity, timestamps, base-R strings, datetime accessors, and exit-scope pins (Phase 11)

Phase 11 was scanned from dplyr 1.2.1's own test suite: string functions
barely appear and datetime accessors appear zero times, but UNKNOWN USER
FUNCTIONS inside `mutate()`/`summarise()`/`filter()` (closures, `identity()`,
`ifelse()`, `scale()`, `lengths()`, ...) hard-errored -- so this phase leads
with a universal fallback (L1) before the "long tail" surface area (L5/L6)
it also delivers.

* **L1 -- universal expression-level CPU fallback**: any `mutate()`/
  `filter()`/`summarise()` expression shape the GPU expression engine
  doesn't understand now routes the WHOLE verb call through the notified
  CPU fallback (all original dots re-injected, `.by=`/`.keep=`/`.before=`/
  `.after=` preserved), instead of hard-erroring. The trigger rule is
  precise: parser returns `NULL` (shape not understood) -> fall back;
  parser THROWS (a genuine user error, e.g. an undefined column) -> keep
  throwing. This single change fixes `top_n()`/`top_frac()` and the whole
  `mutate_at()`/`mutate_all()`/`mutate_if()`/`summarise_at()`/
  `summarise_all()`/`summarise_if()` colwise family end-to-end (none of
  these are `tbl_gpu` S3 methods -- they're plain functions that build and
  call `mutate()`/`filter()`/`summarise()` themselves, so L1 is exactly
  what they needed) and turns nine formerly-pinned cuplyr-only error
  messages into either successes or dplyr's own error text.
* **L2 -- factor fidelity**: three silent-wrongness bugs fixed. Mutating
  joins used to drop `factor_levels` entirely (now propagated through
  `build_join_schema()` keyed by post-suffix column name; a factor key with
  DIFFERENT levels on each side routes the whole join to the CPU fallback,
  since native joins compare raw codes). `bind_rows()` used to concatenate
  raw factor codes across differently-leveled factor columns (now computes
  the dplyr-matching level UNION and remaps each side via new
  `gpu_remap_codes()`). Factor-to-character `bind_rows()` used to refuse
  loudly (now decodes via new `gpu_decode_factor()`, matching dplyr's real
  "factor + character = character" contract). `gpu_physical_type()`
  documents the logical (`DICTIONARY32`)-vs-physical (`INT32` codes) type
  split that R schema types and actual GPU storage now honor consistently.
* **L3 -- timestamp literals & casts**: `Date`/`POSIXct` literals are
  first-class inside `filter()`/`mutate()` expressions (all 5 cuDF
  `TIMESTAMP_*` granularities), so `filter(d < as.Date("2010-01-05"))` --
  verbatim dplyr-test-style code -- now runs natively instead of
  hard-erroring. `as.Date()`/`as.POSIXct()` applied to an existing
  timestamp column lower to a single `cudf::cast()`, with R's exact
  default-UTC semantics (`as.Date.POSIXct()`'s default `tz` really is the
  literal string `"UTC"`, matching cuplyr's own always-UTC storage, so no
  adjustment is needed).
* **L4 -- `bind_rows()` vctrs strictness**: the shared type-promotion ladder
  (extracted from Phase 8's set-op checker) replaces the old, looser
  promotion table -- `STRING` is now always widest (never silently
  demoted), and numeric+character mixes error with vctrs' own message
  format. The long-parked `"4.0"`-from-floats divergence pin is retired
  (that code path is now unreachable).
* **L5 -- base-R string functions native in `mutate()`/`filter()`**:
  `toupper()`/`tolower()` (`cudf::strings::to_upper()`/`to_lower()`),
  `nchar()` (`count_characters()`, INT32), `substr()` (`slice_strings()`,
  exact R clamping semantics for out-of-range `start`/`stop`), `trimws()`
  (`strip()`, `which=`), `paste0()`/`paste()` (row-wise `concatenate()`,
  `sep=`, and R's own "a genuine `NA` argument becomes the literal string
  `\"NA\"`" rule), `grepl()` (`contains_re()`/`contains()` for `fixed=TRUE`,
  with R's "`NA` element -> `FALSE`" rule enforced explicitly since cuDF
  itself null-propagates), `sub()`/`gsub()` (`replace_re()`/`replace()`,
  first-match-only vs. all-matches), and `startsWith()`/`endsWith()`
  (`find.hpp`). Deliberately NOT implemented (falls back to CPU instead):
  `paste()`/`paste0()` with any non-STRING argument (no `as.character()`
  formatting replication -- the `paste0("x", 4)` is `"x4"`, not `"x4.0"`,
  trap), `collapse=`, `ignore.case=`/`perl=`/`useBytes=`, a non-literal
  pattern/position argument, `nchar(type != "chars")`. cuDF's regex dialect
  is not R's TRE/PCRE -- only simple ASCII patterns are oracle-tested.
  A real correctness risk this task found and fixed: a STRING-producing
  function nested inside a comparison/arithmetic expression (e.g.
  `toupper(x) == "ABC"`, `nchar(x) + 1L`) needs `resolve_static_type()`
  (`src/expr_eval.hpp`) to report the REAL result type of every new op, or
  libcudf's AST evaluator throws "non-matching operand types" -- fixed by
  extending both `resolve_static_type()` and the STRING-operand detection
  in `op_dispatches_to_ast()` for every new L5/L6 op.
* **L6 -- lubridate datetime accessors native in `mutate()`/`filter()`**:
  `year()`/`month()`/`day()`/`hour()`/`minute()`/`second()`/`quarter()`/
  `yday()`/`wday()` (`cudf::datetime::extract_datetime_component()`/
  `extract_quarter()`/`day_of_year()`, cast immediately from their native
  INT16 to the declared INT32 inside the C++ handler itself so nested usage
  sees the real type). `wday()`'s cuDF `WEEKDAY` component is ISO
  `Mon=1..Sun=7`; `lubridate::wday()`'s default (`week_start=7`, i.e.
  Sunday) is `Sun=1, Mon=2, ..., Sat=7` -- remapped via `(w %% 7) + 1`
  (verified empirically, and only applied when the global
  `lubridate.week.start` option is still at its own default). **MANDATORY
  namespace guard**: since none of these 9 names exist in base R, going
  native requires the call's function symbol to resolve -- in the
  quosure's own environment, or via an explicit `lubridate::` qualification
  -- to EXACTLY `lubridate`'s own function; a user's own same-named
  function, or lubridate not being installed/attached at all, falls back to
  the CPU path instead (reproducing dplyr's own behavior, including its own
  "could not find function" error when applicable) rather than silently
  misinterpreting the call. Any optional argument (`label=`, `abbr=`,
  `with_year=`, a non-default `week_start=`, ...) present at all falls
  back. Known, documented, and accepted divergence: `lubridate::second()`
  returns a fractional value for sub-second-precision `POSIXct` input,
  which the native path truncates to a whole INT32 (cuDF's `SECOND`
  component has no sub-second concept) -- whole-second values (the
  overwhelming majority of real usage) are unaffected. lubridate is a
  `Suggests`-only soft dependency (`requireNamespace()`-gated); every
  accessor falls back naturally when it isn't installed.
* **L7 -- exit**: `rows_insert()`/`rows_append()`/`rows_update()`/
  `rows_patch()`/`rows_upsert()`/`rows_delete()` were already
  fallback-registered (Phase 2) -- this pins oracle-equality, the fallback
  notification, and `in_place = TRUE` rejection for all six explicitly.
  `do()` is an ordinary S3 generic dplyr never gave a `tbl_gpu` method (it
  used to hard-error with "no applicable method for 'do'"); `do.tbl_gpu()`
  now routes through the same CPU-fallback primitive every other
  non-native verb uses. Verified (not rebuilt): `top_n()`/`top_frac()` and
  the `mutate_at()`/`mutate_all()`/`mutate_if()`/`summarise_at()`/
  `summarise_all()`/`summarise_if()` colwise family all work correctly
  post-L1 (`recode()` needed nothing -- it's a plain vector-level
  function).

Deliberately out of scope for the whole phase (documented, not deferred by
omission): native `rows_*()` (the CPU fallback already re-runs real dplyr,
so it's already 100% conformant -- native would be a large effort for zero
additional conformance), list-columns/`LIST`/`collect_list()`/`unnest()`
interop (`nest_by()`/`group_nest()`/`nest_join()`/`reframe()`/`rowwise()`
stay CPU-fallback-only), real `DICTIONARY32` factor storage (INT32 codes +
R-side levels stays the design), native INT64/`bit64` round-trip, `FLOAT32`
ingestion, `floor_date()`/`ceiling_date()`/`round_date()`, `difftime`, the
`stringr` surface, GPU `roll_median()`, and large-window rolling
aggregation (perf phase).

## `across()`/`if_any()`/`if_all()`/`pick()` and tidy-eval context (Phase 10)

* **`across(.cols, .fns, ..., .names=, .unpack=)`** is now GPU-native inside
  `mutate()`/`summarise()`: it's expanded at the quosure level into ordinary
  per-column dots (bare function, `~ .x` formula, `\(x)`/`function(x)`
  lambda, or a named/unnamed `list()` of any of those), reusing the exact
  same expression pipeline every other dot already goes through -- no new
  IR node, no new C++. `.cols`'s tidyselect (`everything()`, `where()`,
  `starts_with()`, ...) always excludes the current grouping columns,
  matching dplyr exactly. `across()` is rejected inside `filter()` with a
  message pointing at `if_any()`/`if_all()` (matching real dplyr 1.2.1's own
  behavior). `.unpack = TRUE` is deferred to a CPU fallback.
* **`if_any()`/`if_all()`** work inside `filter()`/`mutate()`/`summarise()`,
  expanding to a single OR/AND-folded predicate (reusing `filter()`'s
  existing three-valued-logic `&`/`|` handling).
* **`cur_group_id()`** is GPU-native inside `mutate()`/`filter()` (including
  nested in a comparison, e.g. `filter(cur_group_id() == 1)`, and with
  `.by=`): it reuses the already-GPU-native `group_data()` (Phase 9) plus an
  ordinary `left_join()` to broadcast each group's dense id back onto its
  rows. Inside `summarise()` it falls back to the CPU (a per-group, not
  per-row, shape).
* **`cur_column()`** works inside `across()`'s own function argument (its
  only valid context) via compile-time textual substitution.
* **`pick()`**, **`cur_group()`**, **`cur_group_rows()`**, **`cur_data()`**,
  **`cur_data_all()`**, and **`c_across()`** (as well as `across()`/
  `if_any()`/`if_all()` nested inside another call rather than being a
  dot's entire expression) are not GPU-native and route through a
  whole-call CPU fallback that re-runs the real dplyr call, notified the
  same way every other fallback is (`options(cuplyr.fallback=)`).
* **Bug fix**: `rowwise()`'s CPU fallback used to silently re-upload its
  result to a plain (non-rowwise) `tbl_gpu`, discarding its per-row
  grouping -- so any verb chained immediately afterward (in particular
  `mutate(sum(c_across(...)))`) silently computed the wrong (ordinary
  columnar) result instead of erroring. `rowwise()` now returns a real
  `rowwise_df` (never re-uploaded), so a chained call dispatches to real
  dplyr instead.

## Grouping metadata API & backend generics (Phase 9)

* **`group_data()`/`group_keys()`/`group_rows()`/`group_indices()`/
  `group_size()`/`n_groups()`** are now GPU-native (previously CPU
  fallbacks): one new C++ primitive, `gpu_group_info()`
  (`src/ops_groupby.cpp`), returns the distinct group keys, group-boundary
  offsets, and a row permutation from a single `cudf::groupby::get_groups()`
  call. Row order matches dplyr's own convention exactly: ascending key
  order, `NA` sorted last, each group's rows in original order.
* **`group_by()` computed columns**: a named argument (e.g.
  `group_by(df, g = x %/% 2)`, or even `group_by(df, g = x)`) now runs a
  real `mutate()` first, then groups by the new/overwritten column --
  matching dplyr exactly, including later dots referencing earlier
  computed ones and `.add=` combining with existing groups. `group_by()`
  with no arguments and `.add = FALSE` (the default) now correctly clears
  existing groups (previously incorrectly left them unchanged).
* **`group_by(.drop = FALSE)`**: unobserved factor-level combinations of
  the FACTOR grouping columns now appear in `group_data()` with
  `.rows = integer(0)` (non-factor grouping columns in that row are `NA`),
  matching dplyr's own factor-level-expansion algorithm. This is honored
  for the direct `group_by(.drop = FALSE) |> group_data()` call chain, not
  threaded through every other verb (a documented limitation, see
  `?group_by.tbl_gpu`).
* **`tbl_nongroup_vars()`/`group_by_drop_default()`** now work
  transparently (the former needed no new code at all -- it's plain dplyr
  composition over `tbl_vars()`/`group_vars()`, both pre-existing; the
  latter reads back the table's own most recent `.drop=`).
* **dplyr's backend extension contract** -- `dplyr_row_slice()`/
  `dplyr_col_modify()`/`dplyr_reconstruct()` -- and the 1-d, column-
  selecting form of `[` are now implemented for `tbl_gpu`, for third-party
  code that manipulates a dplyr backend generically rather than through
  its specific verb methods. `[.tbl_gpu` also fixes a pre-existing latent
  bug: without it, `gt[1]` silently returned `list(ptr = <pointer>)`
  (base R's *list* `[` method) instead of a 1-column `tbl_gpu`.
* The iteration family (`group_split()`/`group_map()`/`group_modify()`/
  `group_nest()`/`nest_by()`), `group_trim()`, and `rowwise()` remain CPU
  fallbacks -- they inherently produce R-side objects (or, for
  `group_trim()`, need a real GPU factor-code remap that's out of scope
  here).

## Set operations: `union()`/`union_all()`/`intersect()`/`setdiff()`/`symdiff()`/`setequal()` (Phase 8)

Every dplyr row-set operation is now GPU-native, composed entirely from
existing verbs -- no new C++ (see `?set-ops`):

* **`union()`**: `distinct(bind_rows(x, y))` -- x's own distinct rows (in
  x's row order), then any row from `y` not already present in `x`, in
  y's row order.
* **`union_all()`**: `bind_rows(x, y)` -- every row of `x`, then every row
  of `y`, duplicates included.
* **`intersect()`**: `distinct(semi_join(x, y, by = <every column>))`.
* **`setdiff()`**: `distinct(anti_join(x, y, by = <every column>))`.
* **`symdiff()`**: `setdiff(x, y)` concatenated with `setdiff(y, x)`.
* **`setequal()`**: two anti-join cardinality checks (`nrow() == 0` on
  both sides) -- a plain `TRUE`/`FALSE`, ignoring row order and
  duplicates on both sides, matching dplyr exactly.
* **Column compatibility**: same column count and name *set* (order is
  tolerated -- `y` is matched to `x` by name); a per-column common type
  follows vctrs' own (strict) common-type rules -- narrower than this
  package's `bind_rows()` (which treats `STRING` as universally
  coercible) -- so e.g. an integer column vs. a character column is a
  hard error here too, naming the offending column and both types, not a
  silent promotion to `STRING`.
* **`y` may be a plain `data.frame`** (uploaded automatically) -- unlike
  the join verbs, these generics have no `copy=` argument at all.
* Two column-type combinations real dplyr accepts fall back to CPU
  evaluation instead of erroring (a `cuplyr.fallback` notification
  fires): two factor columns with different level sets, or a factor
  column vs. a character column (both need recomputing/decoding factor
  codes, which this package's GPU column model doesn't support yet); and
  a `Date` column vs. a `POSIXct` column (dplyr promotes to `POSIXct`,
  but the underlying `TIMESTAMP_DAYS` -> `TIMESTAMP_MICROSECONDS` GPU
  cast isn't implemented). Two factor columns with *identical* levels
  are fully native.

## Joins completion: `join_by()`, non-equi conditions, cardinality checks, `semi_join()`/`anti_join()`/`cross_join()` (Phase 7)

Every mutating and filtering join verb now runs GPU-natively for the full
dplyr 1.2.1 surface, with row order, key coalescing, suffixing, and every
cardinality check reproducing dplyr's own contract exactly (see
`?mutating-joins`, `?semi-anti-joins`, `?dplyr::cross_join` for the full
per-argument documentation):

* **`join_by()` DSL**: equi conditions (`join_by(a)`, `join_by(a == b)`,
  `join_by(x$a == y$b)`) route through the existing native hash-join path.
  Non-equi and mixed conditions (`>=`, `>`, `<=`, `<`, and the
  `between()`/`within()`/`overlaps()` helpers that pre-expand to pairs of
  them) are GPU-native via `cudf::mixed_join()` (when at least one equi
  condition is present) or `cudf::conditional_join()` (none), including
  `left_join()`/`inner_join()`/`full_join()`/`right_join()`/`semi_join()`/
  `anti_join()`. `keep = NULL` resolves *per condition*, not per whole join
  (a mixed `join_by(c == d, a >= b)` drops `d` but keeps both `a` and `b`).
* **`na_matches = "never"`**: `NA` keys never match anything (including
  another `NA`), via `cudf::null_equality::UNEQUAL`, for every equi and
  non-equi join type. `"na"` (the default) additionally makes `NA` match
  `NA` under inclusive non-equi operators (`>=`/`<=`) in a `join_by()`
  condition, matching dplyr's own semantics exactly.
* **`multiple=`**: `"all"` (default), `"any"`, `"first"`, `"last"` — resolved
  device-side against the already-sorted join maps, at zero extra cost for
  the default `"all"`.
* **`unmatched=`**: `"drop"` (default) or `"error"`, raising dplyr's exact
  `dplyr_error_join_matches_nothing`/`dplyr_error_join_matches_remaining`
  conditions (class vector, message text, and offending row number all
  verified against dplyr 1.2.1 directly).
* **`relationship=`**: `"one-to-one"`, `"one-to-many"`, `"many-to-one"`,
  `"many-to-many"`, raising dplyr's exact `dplyr_error_join_relationship_*`
  conditions on violation. The default (`relationship = NULL`) also emits
  dplyr's own many-to-many advisory warning whenever both sides have a
  genuine duplicate.
* **`semi_join()`/`anti_join()`**: native via `cudf::filtered_join` (the
  non-deprecated OO replacement for `join.hpp`'s free `left_semi_join`/
  `left_anti_join`). `x`'s `group_by()` grouping is preserved (these are row
  filters, not column-merging joins).
* **`cross_join()`**: native via `cudf::cross_join`, left-major row/column
  order, every common name suffixed on both sides.
* **`right_join()` row order fix**: previously emitted y-order for unmatched
  rows; now matches dplyr's own x-matched-then-unmatched-y-tail contract.
  Implemented via a device-side join-map pipeline (sanitize → stable sort →
  optional diagnostics → `multiple=` filtering → gather/coalesce) shared by
  every mutating join type, replacing the old swapped-left-join workaround.
* **Suffix/keep edge cases**: ported `dplyr:::add_suffixes()`/
  `dplyr:::join_cols()` directly (an already-taken suffixed name now gets a
  second suffix, e.g. `val.y.y`, instead of silently colliding).

**Falls back to CPU (transparent, notified via `cuplyr.fallback`), by
deliberate design, not oversight:**

* `join_by(closest(...))` / rolling joins — needs a "keep only the extremal
  match per group" step this package's join kernels don't implement; a GPU
  route is scoped for Phase 12+.
* A non-equi condition comparing `STRING` columns — cudf's AST comparison
  evaluator has no string support (equi `STRING` joins are unaffected).
* `suffix = c("", ...)` when it would still produce a genuine duplicate
  output column name — dplyr resolves this via a name-keyed column
  *overwrite* that this package's fixed-column-count gather can't express
  natively.
* `nest_join()` — needs list-column support (Phase 11).

## New `summarise()` aggregations: `median()`/`quantile()`/`n_distinct()`/`first()`/`last()`/`nth()`/`any()`/`all()` (Phase 6, task 6.2)

`summarise()` (grouped, `.by=`, and ungrouped) gains eight new aggregation
functions, each verified against dplyr 1.2.1's own empirical behavior:

* **`median(x)`** matches R's own `na.rm = FALSE` default exactly like
  `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()` -- any `NA` in a group
  makes the whole group's result `NA`; `na.rm = TRUE` excludes `NA`s.
* **`quantile(x, probs)`** supports a single probability per call. cudf's
  `interpolation::LINEAR` is bit-for-bit R's own default `type = 7`
  (verified at several probabilities, no divergence to document). Unlike
  every other aggregation here, `na.rm = FALSE` (the default) with an
  actual `NA` present is a hard ERROR, matching base R's own
  `quantile.default()` exactly (not a `NA` result).
* **`n_distinct(x)`** counts `NA` as its own distinct value by default
  (matching dplyr's own `n_distinct()`); `na.rm = TRUE` excludes it.
* **`first(x)`/`last(x)`/`nth(x, n)`** use dplyr's own `na_rm=` (underscore)
  spelling: `na_rm = FALSE` (default) never skips a `NA` when picking the
  position (the result can itself be `NA`); `na_rm = TRUE` skips `NA`
  values. `nth()`'s out-of-range position returns `NA`, matching dplyr.
* **`any(x)`/`all(x)`** implement R's exact three-valued logic
  (`any(c(TRUE, NA))` is `TRUE`, `any(c(FALSE, NA))` is `NA`,
  `all(c(FALSE, NA))` is `FALSE`, `all(c(TRUE, NA))` is `NA`, with the
  default `na.rm = FALSE`; `na.rm = TRUE` drops `NA` first). cudf has no
  groupby ANY/ALL aggregation in this environment at all, so both are
  composed from MIN/MAX-of-`BOOL8` (cudf's own null-skipping) plus a
  "does this group have any null" indicator.
* All eight support the same aggregation sub-expression preprocessing as
  `sum()`/`mean()`/... (e.g. `median(x + 1)`, `any(x > 3)`).
* `median()`/`n_distinct()` additionally light up as window ("agg" family)
  functions in `mutate()`/`filter()`, since `get_groupby_agg()`
  (`src/ops_groupby.cpp`) is shared source between `summarise()` and the
  window-aggregate path (Phase 5, W4). `quantile()`/`any()`/`all()` are
  NOT available as window functions this wave (a probability parameter /
  the missing cudf groupby ANY/ALL instantiation, respectively, would need
  real new plumbing). `cor()`/`weighted.mean()` remain unsupported in
  `summarise()` (out of scope for this task).

## Rolling (moving-window) functions: `roll_mean()`/`roll_sum()`/`roll_min()`/`roll_max()`/`roll_sd()`/`roll_median()` (Theo-requested benchmark task)

GPU-native rolling windows, added as a sixth window-function kind ("rolling")
alongside the existing shift/rank/scan/agg families (`R/window.R`'s
decomposition machinery, `src/ops_window.cpp`'s `gpu_window()`) -- no new
verb wiring was needed in `mutate()`/`filter()` at all, both pick these up
automatically through the same shared window-decomposition pass every other
window function already goes through.

* `roll_mean(x, n, na.rm = FALSE)`, `roll_sum()`, `roll_min()`, `roll_max()`,
  `roll_sd()` (RcppRoll/slider-adjacent naming) are GPU-native via
  `cudf::rolling_window()`/`grouped_rolling_window()` (right-aligned only
  this wave; `align=` is not a parameter on this surface). Grouped rolling
  (`group_by()`/`.by=`) works via the same frame machinery every other
  window function uses, with no new C++ primitive beyond the rolling calls
  themselves.
* `data.table::frollmean()`/`frollsum()`/`frollmin()`/`frollmax()`/`frollsd()`/
  `frollmedian()` are accepted as GPU-native aliases for the above (plus
  `roll_median()`, see below), as long as `align="right"` (the default),
  `fill=NA` (the default), and `adaptive=FALSE`/`partial=FALSE`/
  `give.names=FALSE` (all defaults) hold -- any other value is a hard
  `mutate()` error (no silent behavior change), matching this task's
  explicit right-aligned-only scope.
* `roll_median()` (and its `data.table::frollmedian()` alias) is registered
  and parses/types identically to its five siblings, but is **not**
  GPU-native: cudf 25.12 has no rolling MEDIAN aggregation instantiation at
  all in this environment (verified via `nm -DC libcudf.so`), so it
  transparently falls back to a CPU `roll_median()` implementation
  (`R/roll.R`, also directly exported/usable standalone) via the existing
  `mutate_window_fallback()` path -- still produces exactly the same result,
  just off-GPU.
* `na.rm = FALSE` (the default): any window containing an `NA` is `NA`
  ("poisoning") -- falls out of `cudf::rolling_window()`'s own
  `min_periods = n` requirement on a fixed `n`-row window with NO extra
  masking pass needed (verified empirically, a pleasant simplification vs.
  the cum*/rank families' sticky-NA fix). `na.rm = TRUE`: leading
  physically-incomplete windows (fewer than `n` rows available yet) are
  still `NA`, but `NA` *values* inside an otherwise-complete window are
  skipped -- verified bit-for-bit against `data.table::frollmean(...,
  na.rm = TRUE)`'s own output, since this is NOT simply "cudf's default
  null-skipping behavior" (that alone would incorrectly compute a value for
  the leading incomplete windows too).
* See `benchmark/benchmark_rolling.R` for a locally-measured cuplyr-vs-
  `data.table::froll*()` head-to-head across three data-size tiers
  (1e6/1e7/1e8 rows) and the original benchmark-post workload grid (mean/
  median/min/max/sum/sd x window 11/101/1001).

## NA-semantics parity fixes: `summarise()` `na.rm=` and `arrange(desc())` NA placement (Phase 6, task 6.1)

Two live correctness divergences from dplyr, both found during Phase 5, are fixed:

* **`summarise()` aggregations now honor R's own `na.rm = FALSE` default.** `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()` previously always used cudf's native NA-excluding behavior regardless of the user's call, silently diverging from dplyr whenever a group contained an `NA` (e.g. `summarise(m = mean(x))` returned the NA-excluded mean instead of `NA`). `summarise()` now propagates `NA` to the WHOLE group's result when any input value is `NA`, matching dplyr exactly (`src/ops_groupby.cpp`'s `gpu_summarise()`, using the same MAX(is_null) + `copy_if_else()` technique the window-aggregate path already used, W4).
  * `na.rm = TRUE` is now also a supported, parsed argument (e.g. `sum(x, na.rm = TRUE)`, including on IR sub-expressions like `sum(sqrt(x), na.rm = TRUE)`), opting back into cudf's native NA-excluding behavior explicitly.
  * `n()` is unaffected (it counts rows regardless of nulls, matching dplyr already).
  * `count()`/`tally()`'s own `wt=` weighted sum (routed through `summarise()`) now passes `na.rm = TRUE` explicitly, since it can no longer rely on `summarise()`'s own (buggy) default to exclude `NA` weights.
  * Known follow-up gap, not fixed here: `na.rm = TRUE` on a group that is ENTIRELY `NA` returns `NA` for `sum()`/`mean()`/`min()`/`max()`, where base R's own empty-vector reductions give `0`/`NaN`/`Inf`/`-Inf` respectively (cudf's null-excluding aggregation returns `NULL` for a wholly-excluded group) — tracked in `scratchpad/workflow_state.md`'s "Parked / discovered" section, with a pinned regression test (`test-dplyr-summarise.R`) marking the current (divergent) behavior for a future fix to flip.
* **`arrange(desc(x))` now places `NA` LAST**, matching dplyr exactly (previously placed `NA` FIRST for any descending sort key — a pre-existing divergence found during Phase 5's W7 grouped-slice work, `src/ops_arrange.cpp`'s `gpu_arrange()` flips `null_order` with sort direction now, the same technique already used by the window rank kernel (W7) and `ops_slice.cpp`'s `gpu_slice_rank()`). Applies to every column type (numeric, string, ...) and every position in a multi-key sort, including mixed ascending/descending keys and `.by_group = TRUE`. `test-arrange.R`'s old tests, which had pinned the previous (wrong) behavior as intended, are corrected.

## Window functions, grouped mutate/filter/slice, and `.by=` (Phase 5)

`mutate()`, `filter()`, `slice()`/`slice_head()`/`slice_tail()`/`slice_min()`/`slice_max()`/`slice_sample()`, and `summarise()` all gained GPU-native support for window functions, per-group computation, and on-the-fly (`.by=`) grouping — all lowered through a new expression-decomposition layer (`R/window.R`) and a single new C++ kernel (`src/ops_window.cpp`, `gpu_window()`) shared by every window-bearing call, grouped or not.

* **Window functions in `mutate()`/`filter()`**, arbitrarily nested with ordinary arithmetic/comparisons and with each other (`mutate(y = x - lag(x))`, `mutate(z = lag(cumsum(x)))`, `filter(x > mean(x))`):
  * `row_number()` / `row_number(x)`, `min_rank()`/`dense_rank()`/`percent_rank()`/`cume_dist()`, `ntile()`.
  * `lag()`/`lead()` (`n=`, `default=`, `order_by=`).
  * `cumsum()`/`cummax()`/`cummin()`/`cumprod()`/`cummean()`/`cumall()`/`cumany()`, matching R's own `NA`-propagation (`cumsum(c(1, 2, NA, 4))` is `1, 3, NA, NA`).
  * `consecutive_id()` (single-column).
  * `mean()`/`sum()`/`min()`/`max()`/`n()`/`sd()`/`var()` broadcast to every row of their group (or the whole table, ungrouped) — matching R's own `na.rm = FALSE` default (any `NA` in a group makes the *whole* group's result `NA`); `sum(x, na.rm = TRUE)` is also understood, excluding `NA` per group.
  * `first()`/`last()`/`nth()`, `order_by()`/`with_order()`.
  * Every one of these is evaluated per `group_by()` group when `.data` is grouped, with `mutate()`'s row order always preserved; a grouped `filter()` predicate is likewise evaluated against every row of the ORIGINAL (pre-filter) group before any row is dropped, and the result preserves original row order (never reordered into group-key order).
* **`.by=`** on `mutate()`/`filter()`/`summarise()`: on-the-fly grouping without `group_by()`, e.g. `mutate(df, y = mean(x), .by = g)`. Behaves like `group_by(g) |> mutate(...) |> ungroup()` for `mutate()`/`filter()`; `summarise(.by=)` has one genuine difference from `group_by()`-driven `summarise()` — it emits one row per distinct `.by` combination in FIRST-APPEARANCE order (the order each combination first appears in the data), not sorted/hash-groupby order. The result is always ungrouped; supplying `.by=` on an already-`group_by()`-grouped table is an error, matching dplyr's own message exactly.
* **Grouped `slice()` family** (`slice()`, `slice_head()`/`slice_tail()`, `slice_min()`/`slice_max()` with `with_ties=`/`na_rm=`), both via `group_by()` and `.by=`/`by=`. Grouped output reorders into group order (ascending group-key for `group_by()`, first-appearance order for `by=`/`.by=`); `slice()` itself is native only for a strictly increasing, duplicate-free positive index vector or an all-negative one — anything requiring row duplication or reordering (e.g. `slice(c(3, 1))`) still falls back to CPU evaluation.
* **`slice_sample()`** (`n=`/`prop=`, `replace=`, ungrouped/grouped/`by=`) is now GPU-native with exact `set.seed()` RNG parity: index generation runs on the CPU (consuming R's RNG stream call-for-call the same way `dplyr::slice_sample()` does internally), the actual row gather runs on the GPU. `weight_by=` still falls back to CPU evaluation.
* Two real bugs found and fixed along the way:
  * A rank-family window frame with a *descending* order and `NA`s in the ranked column silently mis-ranked every non-null row (offset by the group's own null count) — `src/ops_window.cpp` now flips null placement with sort direction, matching real dplyr's "`NA`s sort last regardless of direction" rule.
  * `cuplyr`'s own `arrange(desc(x))` was already found to place `NA`s FIRST where real dplyr places them LAST regardless of direction — a pre-existing, unrelated divergence surfaced by this work; fixed in Phase 6, task 6.1 (see this file's own "NA-semantics parity fixes" section, above).
* The narrow `mutate(x, n = n())`/`mutate(x, s = sum(<expr>))` broadcast that only existed to support `add_count()`/`add_tally()` has been removed now that the general window/aggregate machinery covers the same shapes (and richer ones, e.g. `n() + 1`) natively, with no join and no CPU round-trip.

## Transparent CPU fallback layer (Phase 2)

Every dplyr generic that doesn't yet have a GPU-native `tbl_gpu` method now works anyway, via a transparent CPU fallback: `R/fallback.R`'s `gpu_fallback()` materializes any pending lazy operations, `collect()`s to a tibble (restoring `group_by()` structure from the table's own grouping so grouped semantics apply exactly as on a real `grouped_df`), runs the real dplyr verb on the CPU, and re-uploads the result via `tbl_gpu()` — restoring grouping (read back from the verb's actual result, since some verbs change it), execution mode (a lazy input stays lazy), and factor levels.

* Newly working on `tbl_gpu`: `distinct()`, `slice()`/`slice_head()`/`slice_tail()`/`slice_min()`/`slice_max()`/`slice_sample()`, `rename()`/`rename_with()`, `relocate()`, `pull()`, `count()`/`tally()`/`add_count()`, `transmute()`, `reframe()`, `glimpse()`, `rowwise()`, `semi_join()`/`anti_join()`/`nest_join()`/`cross_join()`, `union()`/`union_all()`/`intersect()`/`setdiff()`/`setequal()`/`symdiff()`, `group_data()`/`group_keys()`/`group_indices()`/`group_size()`/`n_groups()`/`group_split()`/`group_map()`/`group_modify()`/`group_nest()`/`group_trim()`/`nest_by()`, `rows_insert()`/`rows_append()`/`rows_update()`/`rows_patch()`/`rows_upsert()`/`rows_delete()`, and the superseded `sample_n()`/`sample_frac()`. These are wired one phase at a time to GPU-native implementations going forward (see `scratchpad/todo.md`).
* Verbs whose result isn't a data frame at all (`pull()`'s vector, `group_split()`'s list), whose result is a data frame containing list-columns with no GPU representation (`nest_by()`, `group_nest()`, `nest_join()`), or that only report metadata about the grouping/data (`group_data()`, `group_keys()`, `group_indices()`, `group_size()`, `n_groups()`) are returned as plain R objects rather than re-uploaded.
* `rows_*(in_place = TRUE)` is rejected with a clear error for `tbl_gpu`, since there is no mutable in-place table to update — assign the returned value instead.
* Every fallback calls the existing `cuplyr_fallback_notify()` hook, so `options(cuplyr.fallback = "warn")`/`"error"` (introduced for `filter()`/`summarise()`'s CPU-eval paths) now also gates these verbs.
* Known gaps, not fixed by this layer because they aren't reached via S3 dispatch: `add_tally()`, `top_n()`, and `top_frac()` are plain (non-generic) functions in dplyr 1.2 that call `mutate()`/`filter()` internally with constructs (`n()` inside `mutate()`, `top_n_rank()` inside `filter()`) the GPU-native/CPU-eval paths for those verbs don't yet understand; `group_rows()` and `group_walk()` are likewise non-generic but work transparently since they only call `group_data()`/`group_map()`, which are now implemented.

## Unified eager/lazy execution

* Every dplyr verb now builds a single AST node and routes through one shared execution path (`push_op()`/`push_join()` in `R/execute.R`, `lower_and_execute()` in `R/lower.R`) regardless of whether the table is eager or lazy. This collapsed a number of eager/lazy divergences that previously produced different results depending on execution mode; see the fixes below.

## Expression engine

`filter()`, `mutate()`, and the expressions inside `summarise()`'s aggregation calls now all parse through one shared rlang-based expression IR (`R/ir.R`) lowered to `cudf::compute_column()` (`src/expr_eval.hpp`), replacing three separate special-cased parsers. This is a large capability jump for all three verbs:

* `filter()` now understands `&`, `|`, `!`, `xor()`, arbitrarily nested and combined with comparisons; `%in%` (including a set containing `NA` and an empty set); `is.na()`; `between()`; string-column comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) against a literal or another string column; and math on columns inside a predicate (`+ - * / %% %/% ^`, `sqrt`, `log`, `abs`, etc.) — the same surface `mutate()` supports. `&&`/`||` are now rejected with a message pointing at `&`/`|`, matching dplyr 1.1+.

* `mutate()` now understands arbitrarily nested arithmetic (`(x + y) * z - 1`), scalar-on-either-side expressions (`10 - x`), `%%`/`%/%`, math functions (`sqrt`, `log`/`log(x, base)`, `log2`, `log10`, `exp`, `abs`, `floor`, `ceiling`, `sin`, `cos`, `tan`, `round()`), comparisons and logical operators, `is.na()`/`between()`/`%in%`, `NA` literals (typed and bare), and later dots referencing earlier dots' output columns.

* `summarise()`'s aggregation calls (e.g. `sum(...)`, `mean(...)`) now accept any sub-expression the IR understands, not just a single comparison or arithmetic operator — e.g. `sum(carb > 3 & wt < 4)`, `mean(sqrt(hp))`, `sum(carb %% 2 == 0)` all now work, in both eager and lazy modes, grouped and ungrouped.

* Each of `filter()`/`mutate()`'s multi-expression calls now lowers to exactly one fused GPU kernel (one `compute_column()` call), regardless of how many predicates/expressions or how deeply nested — previously each operator in a chain was a separate kernel launch.

## Bug fixes

* Fixed a silent data-corruption bug in the lazy-mode projection-pushdown optimizer: `push_down_projections()` could drop a *replaced* (not newly-created) column from the upstream projection when a lazy `mutate()` overwrote an existing column it didn't itself read as an input. This broke `gpu_mutate_expr()`'s replace-in-place name matching, silently appending the new value at the end instead — rotating column values across the table with no error and no warning. Found via the T6 `mutate()` dplyr-oracle test suite; only genuinely new output columns can be pruned from the upstream requirement now.

* `right_join()` now works at all; previously it errored on every call due to an internal argument-parsing mismatch.

* `full_join()` and `right_join()` now coalesce join key columns for unmatched rows instead of leaving them `NA`. For example, `full_join(x, y, by = "id")` now populates `id` from `y` for rows that only matched on the right side.

* `bind_rows()` now actually casts columns that get type-promoted to `STRING` (e.g. combining a numeric column in one table with a character column in another). Previously the promoted column kept its original GPU type while the schema claimed `STRING`, causing a type-mismatch failure.

* `mutate()` no longer silently promotes integer arithmetic to `FLOAT64`. `+`, `-`, `*`, `%%`, and `%/%` now preserve `INT32`/`INT64` when every input column and literal involved is integer or logical (matching R/dplyr, e.g. `TRUE + TRUE` is `2L`). Note that `INT32` results can now wrap around on overflow instead of silently promoting to a wider type, matching plain integer arithmetic semantics. `/` and `^` continue to always promote to `FLOAT64` (matching R: `1L / 2L` and `2L ^ 2L` are both doubles).

* `arrange(..., .by_group = TRUE)` now sorts prepended group columns in ascending order to match `dplyr::arrange.grouped_df()`, even when the user's own expression wraps a group column in `desc()` (e.g. `arrange(desc(g), x, .by_group = TRUE)` now sorts `g` ascending, `x` per the user's spec).

* `filter(TRUE)` is now a true no-op: it returns `.data` unchanged without doing GPU work, instead of round-tripping through a filter kernel.

* Lazy `summarise()` with `sd()`/`var()` no longer errors. Lazily evaluated aggregations now translate to the same cudf function names (`std`/`variance`) that the eager path always used.

* `summarise()` now accepts namespaced aggregation calls, e.g. `dplyr::n()` and `stats::sd(mpg)`, in both eager and lazy modes.

* `summarise(n = n())` now always returns an integer (`INT32`) column, matching `dplyr`. Previously, ungrouped `n()` silently returned a double due to a schema/GPU-column type mismatch.

* `summarise()` with an unnamed aggregation expression (e.g. `summarise(mean(mpg))`) now auto-names the output column from the expression text and warns, matching `mutate()`'s existing behavior for unnamed expressions. This had regressed to silently doing neither in either mode.

* Eager `select()` of a factor column now round-trips as a factor through `collect()`. Previously the eager path dropped factor level metadata, so `collect()` returned integer codes instead of factor labels.

## New features

* Added the `cuplyr.fallback` option to control whether `filter()`'s and `summarise()`'s CPU-evaluation fallback paths notify the caller. One of `"silent"` (default), `"warn"`, or `"error"`.

# cuplyr 0.1.1

## Bug fixes

* `collect()` now correctly restores factor columns with their original levels. Previously, factor columns were returned as integer codes instead of factors (#3).

* `names<-()` now validates the replacement value before assignment. It errors if the new names have wrong length, contain `NA` values, empty strings, or are not character (#4).

## Build system

* Fixed CCCL (CUDA Core Compute Libraries) header detection for RAPIDS 25.12+ in pixi/conda environments where headers are located in a `rapids/` subdirectory. The configure script now automatically detects `<cuda/stream_ref>` and related headers.

* Reordered include paths so CUDF/RMM headers take precedence over system CUDA headers, ensuring consistent CCCL versions.

## Documentation

* Added pkgdown documentation site with vignettes for getting started, complex analysis workflows, and query optimization.

# cuplyr 0.1.0

## Lazy evaluation

* `tbl_gpu()` gains a `lazy` argument to enable deferred execution. Operations build an AST (Abstract Syntax Tree) that is optimized and executed only when `collect()` or `compute()` is called. Set globally with `options(cuplyr.exec_mode = "lazy")` or `CUPLYR_EXEC_MODE=lazy` environment variable.

* Added an AST optimizer that applies multiple optimization passes before execution:
  - Projection pruning: push column selection close to data sources
  - Mutate fusion: combine consecutive mutate operations
  - Dead column pruning: remove unused intermediate columns
  - Filter pushdown: move filters earlier in the pipeline, including across joins
  - Filter reordering: execute cheaper filters first
  - Filter fusion: combine multiple filters into single GPU kernel

* `compute()` executes pending lazy operations and keeps the result on GPU.

* `collapse()` inserts an optimization barrier without executing.

* `as_lazy()` and `as_eager()` switch execution modes mid-pipeline.

* `is_lazy()` and `has_pending_ops()` check the current execution state.

* `show_query()` displays the pending operation tree for debugging.

## Join operations

* Added `inner_join()`, `left_join()`, `right_join()`, and `full_join()` for combining GPU tables (#2). Joins support automatic key detection (natural join), named vectors in `by` for different key names, `suffix` for column name conflicts, `keep` for retaining join keys, and `copy` to auto-transfer data frames to GPU.

## Bind operations

* Added `bind_rows()` for vertically combining GPU tables with automatic schema unification and type promotion.

* Added `bind_cols()` for horizontally combining GPU tables with `.name_repair` for duplicate column handling.

* Both bind functions automatically materialize lazy tables before binding.

# cuplyr 0.0.1

Initial release of cuplyr, a GPU-accelerated dplyr backend using NVIDIA's libcudf library.

## Core functionality

* `tbl_gpu()` transfers R data frames to GPU memory, returning a `tbl_gpu` object that works with dplyr verbs.

* `collect()` transfers GPU data back to R as a tibble.

* `as_tbl_gpu()` coerces data frames to GPU tables.

* `is_tbl_gpu()` tests if an object is a GPU table.

## dplyr verbs

* `filter()` supports scalar comparisons (`x > 5`, `x == "a"`) and column-to-column comparisons (`x > y`). Supports boolean vectors as filter masks.

* `select()` supports column selection by name, position, and tidyselect helpers.

* `mutate()` supports arithmetic operations (`+`, `-`, `*`, `/`, `^`) with scalars and between columns. Supports left-associative chains (e.g., `a + b + c`).

* `arrange()` sorts by one or more columns with `desc()` support for descending order. Supports `.by_group = TRUE` for grouped tables.

* `group_by()` sets grouping metadata for subsequent aggregation. `ungroup()` removes grouping.

* `summarise()` computes grouped aggregations with support for `sum()`, `mean()`, `min()`, `max()`, `n()`, `sd()`, and `var()`. Supports expressions inside aggregation functions (e.g., `sum(x > 0)`).

## Type support

* Supported R types: numeric (FLOAT64), integer (INT32), character (STRING), logical (BOOL8), Date (TIMESTAMP_DAYS), and POSIXct (TIMESTAMP_MICROSECONDS).

* factor columns are converted to INT32 codes.

* integer64 columns are converted to FLOAT64 with a warning about precision loss for values exceeding 2^53.

## GPU memory utilities

* `gpu_memory_usage()` estimates GPU memory footprint of a `tbl_gpu` object.

* `gpu_memory_state()` returns current GPU memory usage (total, free, used).

* `gpu_gc()` forces garbage collection to free GPU memory from unreferenced tables.

* `gpu_object_info()` returns detailed information about a GPU table.

* `verify_gpu_data()` confirms data resides on GPU, not in R memory.

* `gpu_size_comparison()` compares R object size vs GPU data size.

## GPU information

* `has_gpu()` checks if a compatible GPU is available.

* `gpu_details()` returns GPU device information (name, compute capability, memory).
