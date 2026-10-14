# cuplyr architecture reference

Deep reference for agents that read or change cuplyr's code (PM, coder,
reviewer). Process rules and commands live in `.claude/CLAUDE.md`; this file
is only about how the code works. When it disagrees with the code, the code
wins -- and fix this file in the same change.


## Quick Reference

### Core Data Structures
```r
# tbl_gpu structure (R/tbl-gpu.R)
list(
  ptr = <externalptr>,           # XPtr to cudf::table
  schema = list(
    names = c("col1", "col2"),   # column names
    types = c("FLOAT64", "INT32"), # GPU type strings
    # optional metadata: factor_levels (factor labels), tz (POSIXct zones)
  ),
  groups = c("col1"),            # group_by columns (can be empty)
  exec_mode = "eager",           # "eager" or "lazy"
  lazy_ops = list()              # AST for lazy evaluation
)
```

### Type Mappings
| R Type | GPU Type (schema/logical) | Notes |
|--------|----------|-------|
| logical | BOOL8 | |
| integer | INT32 | |
| double | FLOAT64 | |
| character | STRING | |
| Date | TIMESTAMP_DAYS | |
| POSIXct | TIMESTAMP_MICROSECONDS | |
| factor | DICTIONARY32 (physical: INT32) | codes only, 1-based, `schema$factor_levels` holds labels |
| integer64 | FLOAT64 | loses precision >2^53, warns |

#### Logical vs. physical type: factor columns (Phase 11 L2)

cuplyr deliberately does **not** adopt cudf's real `DICTIONARY32` column
type (a dictionary-encoded keys+indices structure). A factor is uploaded
and stored as a plain **INT32** column of 1-based R factor codes (R's own
`TYPEOF(x)` for a factor is `INTSXP`, so it flows through the exact same
`integer_to_gpu()` path as a real integer column, `src/transfer_io.cpp`);
the label strings live only in `schema$factor_levels[[col]]` on the R
side, never on the GPU. `"DICTIONARY32"` is therefore a **logical**
schema type only.

`gpu_physical_type()` (`R/utils.R`) is the single conversion point between
the two views: `"DICTIONARY32" -> "INT32"`, every other type unchanged.
Call sites split into two camps:
- **Physical-column sites** (talk to the actual `cudf::column`, which is
  always INT32 for a factor): must route the type string through
  `gpu_physical_type()` first -- e.g. `gpu_make_null_column()` (bind
  null-column padding), `gpu_cast_column()`'s target-type dispatch. These
  C++ primitives have no `DICTIONARY32` case at all and would error if
  handed the logical label directly.
- **Logical-semantics sites** (care whether a column IS a factor, not what
  bytes back it): must keep seeing `"DICTIONARY32"` verbatim, never routed
  through `gpu_physical_type()` -- the mutate arithmetic guard
  (`infer_mutate_output_type()`, R/ast.R), join key type resolution
  (`resolve_join_key_types()`, R/join.R), memory accounting
  (`R/gpu-memory.R`). Silently treating a factor as a plain integer here
  would let arithmetic on a factor slip through, or let two differently-
  leveled factor keys join as if they shared one integer domain.

Two GPU primitives (`src/ops_bind.cpp`) operate directly on factor codes
without ever needing a real dictionary type:
- `gpu_remap_codes(xptr, col_idx, map)`: remaps a factor column's codes
  onto a *different* level ordering (`bind_rows()`'s LEVEL-UNION contract
  for two factors with different level sets, and the native `union()`/
  `intersect()`/etc. set-op path, Phase 8/11).
- `gpu_decode_factor(xptr, col_idx, levels)`: decodes a factor column to
  its label strings (dplyr's factor + character -> character contract).

Both gather a small per-level lookup table (an uploaded INT32 map or
STRING levels column) using the physical codes as the index -- see
`src/ops_bind.cpp` for the null-sanitization details (codes are 1-based
and cudf's `gather()` needs a non-nullable, 0-based index).

#### POSIXct time zones (F32)

The GPU holds every POSIXct as UTC microseconds (cudf has no tz-aware type);
the column's `tzone` is R-side metadata in `schema$tz`, the sibling of
`schema$factor_levels` (helpers and propagation in `R/tz.R`):

- `schema$tz` is a named character vector, column name -> zone, only for
  TIMESTAMP_MICROSECONDS columns whose zone is not `"UTC"`. `""` is local time,
  `NA_character_` a column whose tzone attribute is absent. Like dplyr,
  absent becomes `""` wherever values are sliced or combined (filter, arrange,
  slice, distinct, summarise, joins, binds, set ops, `if_else()`/`coalesce()`/
  `lag()`/... and window outputs: `tz_localize()`); it stays absent through
  select/rename/relocate, a bare `mutate(t2 = t)` copy and `t +/- n`. No entry means `"UTC"`, which is also what every path
  that does not know the zone produces, so a missing propagation shows up as a
  wrong attribute, never a wrong instant. With duplicated column names the
  vector holds one entry per TIMESTAMP_MICROSECONDS column in column order,
  tagged `by_position` (`tz_by_position()` / `tz_from_positions()`, like
  `levels_by_position()`).
- Written at upload (`tbl_gpu.data.frame()`), restored by `collect_frame()`
  (`restore_tz()`, the C++ collect tags every column "UTC") and by
  `print()`/`glimpse()` previews. Carried by `propagate_schema_meta()`
  (`R/execute.R`: `push_op()` and `infer_schema_full()`, hence `compute()`)
  through `propagate_tz()`; joins (`build_join_output_info()`, a merged key
  takes `tz_fold()` of the two sides, `keep = TRUE` and other columns keep
  their own), `bind_rows()`/set ops (`compute_unified_schema()`,
  `check_set_op_compatible()`), `bind_cols()`, `names<-`, vctrs restore,
  `group_data` keys, the empty-group summarise. `infer_schema.ast_join()` reads
  its children through `infer_schema_full()`, so metadata survives a join over
  lazy ops.
- mutate/transmute: `update_schema_for_expr()` sets the output zone with
  `ir_infer_tz()` (`R/ir.R`, the `ir_infer_type()` sibling): column -> its
  zone, POSIXct literal -> its `tzone`, and every op that can return a
  POSIXct has an explicit rule in `ir_tz_rules`: `x` ops (`na_if()`,
  `replace_when()`/`replace_values()`, `lag()`/`lead()`, `first()`/`last()`/
  `nth()`, `min()`/`max()`, `roll_min()`/`roll_max()`) take `x`'s zone, since
  dplyr casts the other values to `x`; `common` ops (`if_else()` true/false/
  missing, `case_when()` values then `.default`, `coalesce()`) take
  `tz_fold()` of the value operands in order (vctrs' rule: the first zone,
  unless it is local, then the next); `as.POSIXct()` keeps a POSIXct's zone
  and makes a Date UTC. An op without a rule (unary `-`, `abs()`,
  `cummax()`/`cummin()`, which R rejects on a POSIXct) signals
  `cuplyr_timestamp_needs_cpu` from `ir_tz_of()`; `ir_parse_quo()` runs it
  over every POSIXct-valued call (`ir_check_tz_rules()`), so such an
  expression goes to the CPU instead of claiming a zone. Window ops and
  summarise `min/max/first/last/nth` keep the value column's zone. Set
  operations treat a zone that differs from the common one as needing
  alignment (`tz_matches_target()`, like the factor levels), so
  `align_to_schema()` hands back the target's zones for `intersect()`/
  `setdiff()`/`semi_join()` sides; a zone-only difference just rewrites
  `schema$tz` (no column copy).
- **Fallback rule**: the calendar accessors (`hour()`, `wday()`, ... and the
  user-callable `lubridate::as_date()`) on a POSIXct whose zone is not
  UTC-equivalent (`tz_is_utc()`: `""`/`NA` resolve to the session zone first)
  return `NULL` from the IR parser, so the verb's whole-call CPU fallback
  computes them with R's tz database (exact on DST days). `summarise()` falls
  back for an aggregate over such an accessor by the existing
  `ir_parse_quo()` -> `NULL` rule. `as.Date()` stays native (R 4.6: UTC). A
  Date unified with a non-UTC POSIXct (join keys, equi and non-equi:
  `join_date_posixct_needs_cpu()`; `bind_rows()` / set ops:
  `compute_unified_schema()$cpu_tz` / `check_set_op_compatible()`) goes to the
  CPU, since dplyr casts the Date to *local midnight of that zone*. UTC pairs
  keep the native UTC cast. `t + n` / `t - n` (one timestamp operand, the other
  numeric), `if_else()`/`case_when()`/`coalesce()` mixing a Date with a
  POSIXct, and a Date + POSIXct mix in `na_if()`/`replace_when()` (the GPU
  cast would reinterpret the ticks; a `lag()`/`lead()` default whose type
  differs from a timestamp `x` makes `ir_parse_shift_call()` return `NULL`
  for the same reason) signal `cuplyr_timestamp_needs_cpu` (a subclass of
  `cuplyr_factor_needs_cpu`: parse returns `NULL`, whole-call CPU fallback).
  `ir_parse_quo()` runs the type inference after the parse and re-raises only
  that subclass, so a factor expression still comes back as IR for
  `ir_factor_rule()`. `Date + Date` and `n - d` keep their plain
  errors (other operators on a timestamp go to the CPU, see F37 below).
- **Date arithmetic (F37)**: the `+`/`-` registry entries have a parse hook
  (`ir_parse_ts_arith()`, R/ir.R) because only the parsed args show whether a
  literal is whole. `Date +/- n` with `n` an INT32 expression or a finite
  whole literal in int32 range (`+` either order, `-` date minus number)
  becomes `date_add_days`/`date_sub_days` (typed TIMESTAMP_DAYS, handler in
  `apply_handler()`: INT32 bit_cast to DURATION_DAYS, then cudf timestamp
  +/- duration), so it is native in mutate, filter and grouped mutate; int32
  day overflow wraps. Any other Date shift (double column, fractional or NA
  literal) and POSIXct `t +/- n` take the CPU route. `ts - ts` (a difftime)
  a unary `-ts`/`+ts` also signal `cuplyr_timestamp_needs_cpu`, so dplyr computes
  them (`as.numeric(d2 - d)`, `filter(d2 - d > 3)` work; a difftime result
  column hits F18's `cuplyr_error_unsupported_class` at re-upload, hint
  `as.numeric()`; `-d` errors and `+d` returns `d`, as in base R). Making the shift native removed
  the CPU route that used to hide the GPU's timestamp gaps, so one rule in
  the type inference sends them back: `ir_check_ts_operands()` (R/ir.R), run
  by `ir_infer_type()` on every call node, so it covers parser-built nodes,
  desugars (`between()`) and any nesting. A timestamp argument is native only
  for the ops in `ir_ts_native_ops` (`+`/`-`, same-type comparisons,
  `is.na`, `if_else`/`case_when`/`coalesce`/`na_if`/`replace_when` with
  matching types, `lag`/`lead`/`first`/`last`/`nth`/`min`/`max`/
  `n_distinct`, the rank functions, the casts and the datetime accessors);
  anything else (`%in%`, a comparison with a number/string or another
  timestamp type, math, cumulative, `sum`/`mean`/`median`/`sd`/`var`,
  string functions) signals `cuplyr_timestamp_needs_cpu`. The rule goes by
  type, never by node kind: a bare Date column gets the same treatment, so
  `filter(d < 5)` and `filter(d %in% dates)` give dplyr's result through the
  CPU (`check_filter_comparison_types()` no longer sees timestamps). A
  timestamp mixed with another type in `ir_unify_types()` /
  `ir_check_cast_to_x()` goes to the CPU too (vctrs' error). summarise()
  aggregates aren't IR calls, so `summarise_dots_parseable()` keeps its own
  list: only `min`/`max`/`first`/`last`/`nth`/`n_distinct` take a timestamp.
  The `+`/`-` hook returns early when the schema and the literal args hold no
  timestamp (`ir_args_may_be_timestamp()`); that only costs native-ness,
  because the type inference repeats each of the hook's CPU decisions
  (`ir_check_ts_operands()`: unary `+`/`-`, a bare-NA shift;
  `infer_mutate_output_type()`: a shift, `ts - ts`). Literals: a Date literal
  is an int32 day count on the GPU (`build_scalar()` truncates), so
  `ir_check_ts_lit()` (run by `ir_infer_type()` on every literal, by
  `ir_parse_shift_call()` on a `lag()`/`lead()` `default`, and by
  `parse_mutate_dots()` on a constant dot) sends any Date literal with a
  non-whole value (`seq(a, b, length.out = k)` midpoints) to the CPU, in any
  slot; a fractional Date in the result is then floored at upload. The one
  exception is a whole column-free mutate()/transmute() dot, see
  "Evaluate-once" below.
- **Fractional Dates at upload**: a TIMESTAMP_DAYS column holds whole days, so
  `date_to_gpu()` (`src/transfer_io.cpp`) floors every double Date while it
  already loops over the column, and `df_to_gpu()` flags the columns it had to
  change in a `fractional_dates` attribute of the returned pointer;
  `tbl_gpu()` strips it and warns once (`cuplyr_warning_fractional_date`,
  naming the columns). No extra pass over whole Dates. Every upload goes
  through `tbl_gpu()`, so this is also what a CPU fallback producing `d + 0.5`
  or a fractional `mean(<Date>)` goes through.
- **Evaluate-once for column-free mutate()/transmute() dots**: dplyr evaluates
  each dot once (per group). `parse_mutate_dots()` evaluates a dot that reads
  no column itself (`mutate_constant_value()`, empty data mask) and builds the
  literal from that value, so `runif(1)`, a counter or `f()` runs once on the
  native path. When the call then takes the CPU route anyway (some other dot
  is unparseable), the values of directly written column-free dots of an
  ungrouped call are recorded by expanded position and
  `inline_evaluated_dots()` hands dplyr the value instead of the expression,
  under dplyr's label for an unnamed dot (the `"provenance"` attribute of
  `expand_across_dots()` maps expanded dots back to user dots). across() dots
  go to dplyr as written, since dplyr resolves `.cols` against the columns
  earlier dots create. Known limit: on the CPU route, column-free across()
  bodies, grouped/`.by` constants and column-free sub-calls folded inside
  other dots (`x + runif(1)`) run again.
- **Running schema for across()/if_any()/if_all() (F45)**: this qualifies the
  across() sentence above. `expand_across_dots()` resolves each such dot's
  `.cols` against `run`: the input schema updated by every earlier dot
  (created/overwritten columns with the type the native path infers via
  `ir_parse_quo()` + `make_mutate_expr()`, `x = NULL` removes, an across()'s
  outputs typed from its expansion on the pre-dot schema, if_any/if_all BOOL8).
  `run` is only advanced up to the last such dot, and expansion stops at the
  first dot that needs dplyr (the verbs rerun the original dots then). A type of
  NA is "exists, type unknown", which a column-free dot gets because it must
  not be evaluated a second time just to learn it. Every output whose type
  `across_infer_output()` can't give (column-free calls, mask-only calls like
  n(), unknown factor levels, every summarise output) is NA-typed and also goes
  in `run$maybe`, because its value may be NULL (`z = v`, `z = c()`) and then
  creates nothing (mutate also removes an existing column):
  `resolve_across_cols()` returns `NULL` (whole-call dplyr) unless the
  selection comes out identical with and without those columns and picks none
  of them. Setting a `run$maybe` column again when it isn't the last column
  sets `run$unknown_set`: dplyr's mask appends a column it re-creates after a
  NULL, so its position isn't known (`across_run_set()`). A literal `x = NULL`
  in mutate removes `x` from `run` and a later `x = ...` appends it, like the
  mask. `resolve_across_cols()` also returns `NULL` for a selection that is not
  name-only over an NA-typed column, for every selection once
  `run$unknown_set` is set (also by an unnamed dot that may return a data
  frame), and for a tidyselect error caused by a removed column. An if_any()/if_all()
  piece whose type is NA is dplyr's to check (`check_across_predicate_piece()`).
  summarise outputs are always NA-typed. A type is recorded only when it is the
  type R gives the column (`across_r_typed()`): every call must be in the
  allowlist `across_r_typed_fns` (a `pkg::` prefix must name the function's own
  package), minus `median()` of a non-double, `max()`/`min()`/`cummax()`/
  `cummin()`/unary `-` of a logical and environment values of other classes
  than Date/POSIXct. Anything else the IR parses (`ntile()`, lubridate's
  `year()`/`month()`/`second()`/`yday()`/`wday()`, `roll_*()`) is NA-typed but
  not `maybe`, so name-only selections over it stay native. An unnamed bare
  symbol naming a `run$maybe` column sets `run$unknown_set` (it may be an
  environment value, appended, unpacked or skipped).
- **Fractional Date constants (native floor)**: a column-free dot whose value
  is a fractional Date (finite, int32 range) is built from its floored value
  (`floor_fractional_date()`, -0.5 -> -1, like `date_to_gpu()`), grouped and
  `.by` calls included, and the verb raises the upload's warning
  (`warn_fractional_dates()`, `cuplyr_warning_fractional_date`) once it has
  committed to the native path, at verb time in eager and lazy mode, naming
  the columns whose final value is floored (not staged across() temps, not a
  column a later dot drops or overwrites). That gives the column the CPU
  route plus upload would give, with one evaluation, as long as nothing in the
  call sees the fraction: a later dot that may read the name sends the call
  to the CPU route. Readers are found syntactically: `expr_may_read_cols()`
  (the symbol anywhere, `.data`, or a call that reaches the mask without
  naming a column, `expr_reaches_mask()`: `pick()`, a nested `across()`,
  `cur_data()`, `get()`/`exists()`/`eval()`/`environment()` and similar; a
  user function calling `pick()` itself is not detected). A top-level
  `across()`/`if_any()`/`if_all()` is already expanded by then, against the
  running schema at its dot (F45), so a name-only selection does reach the floored
  column; every expansion of such a dot (provenance kind `"across"` or
  `"combined"`) after the user dot that floored a value counts as a reader;
  expansions of the across() that produced the value don't. A bare copy
  (`ee = dd`, also a staged across() assignment) isn't a reader: it copies
  the floored column and is floored itself. A fractional Date vector
  constant (uploaded by `ir_vec()`, floored and warned there) follows the
  same reader rule. A column-free dot that reaches the mask
  (`y = exists("x")`) is never inlined on the CPU route: dplyr evaluates it
  in its mask. Known limit: on a zero-row table the native path still warns
  although nothing is floored.

- **Timestamp mean()/median() (F41)**: a mean()/median() that is the whole value of a summarise() aggregate or mutate() dot over a Date/POSIXct is native (`src/ts_agg.hpp`: the integer rep as FLOAT64, aggregate, convert back; a Date floors and `lower_*()` warns `cuplyr_warning_fractional_date` from the `ts_fraction` attribute, a POSIXct rounds half away from zero with `cudf::round`). Inside any other expression it is CPU (`ir_check_ts_operands()`). In `parse_mutate_dots()` its output name follows the floored-constant reader rule above (tracked in the same list, but warned by the lowering, not by `"fractional_dates"`): a later dot that may read it, `d < m`, sends the call to the CPU, where dplyr compares the real double. A bare copy `m2 = m` stays native. `transmute()` has no window plan, so a window-bearing transmute (any `mean()`/`cumsum()`/`lag()`..., not only timestamps) takes the CPU route.

- **Integer semantics (F42)**: R's integer `+`, `-`, `*` give NA (warning "NAs produced by integer overflow") outside +-`.Machine$integer.max`, and `%/%`/`%%` by `0L` give NA. Double arithmetic is untouched.
  - *Checked nodes.* A binary node of those ops whose operands are all INT32/BOOL8 (`checked_int_kind()`, `src/expr_eval.hpp`; a bare NA literal adopts its sibling's type) is not an AST op itself: `op_dispatches_to_ast()` says no and `apply_checked_int()` evaluates it, but `ast_expressible()` says yes, so its ancestors still fuse into one AST that references the materialized result like a column. A chain of checked `+`/`-`/`*` (up to `kMaxFusedChecked` = 4 nodes, charged in pre-order on entry; a node past the cap is a leaf that `materialize()` evaluates as its own chain, so cost stays linear in chain length) is two `compute_column()` calls without a null literal (the nullable AST path is slow) plus a reduction: `value` (the chain in INT32, wrapping) and `ok` (BOOL8, the `NULL_LOGICAL_AND` of one `ABS(r) <= M` test per node over an unguarded INT64 tree `r`). `ok` is false only where a node's operands were valid and its result left the range, so `all(ok)` is exactly "no overflow" (a null in another branch can't hide one); on overflow the mask of `value` becomes `bools_to_mask(ok)` and `ctx.int_overflow` is set. A node with a null literal (the IR `na` flag is only set for a logical NA; `NA_integer_` is an INT32 literal with value NA, see `checked_lit_is_null()`) is NA on every row and never an overflow. `%/%`/`%%` (kind 2) stay a separate INT64 kernel plus a cast.
  - *Warning.* The flag travels as an `int_overflow` attribute of the pointer returned by `gpu_mutate_expr()`/`gpu_filter_expr()`/`gpu_compute_column()`; `warn_int_overflow()` (R/lower.R) warns once per call, never from C++: at `lower_and_execute()` time, i.e. at the verb when eager, at `collect()` when lazy; summarise's temp columns (`create_temp_column()`) are materialized while the verb runs, so theirs warns at the `summarise()` call even on a lazy table.
  - *cumsum().* `cumsum()` of INT32/BOOL8 scans in INT64 (cudf's SUM target type), one `minmax` says whether anything left the range, and only then are the rows from the first overflow of a group on nulled (second MAX scan) and `cumsum_overflow` flagged; `cummean()` desugars on `x + 0` (double).
  - *Window sum().* Window `sum()` of INT32/BOOL8 is declared INT32 (`type_window_int_sum()`); `gpu_window()` narrows like F34 and flags `sum_overflow` (FLOAT64 column): `lower_window()` then raises `cuplyr_error_sum_overflow` (`abort_sum_overflow()`) unless the node is flagged `ast_window(on_sum_overflow = "double")`. `gpu_window()`'s `sum_overflow` attribute has one flag per spec (out_pos), and a spec's own `on_sum_overflow` field, when present, overrides the node's. In dplyr only the overflowing groups' sums become doubles; the others stay integers and arithmetic on them still overflows to NA. One GPU column can't hold both, so every verb reruns exactly on the CPU where it can:
    - `mutate()`: eager `mutate_window()` catches the error and reruns the call through `gpu_fallback()`; a lazy plan lets it through (F34's lazy rule).
    - `filter()`: `plan_window_stages(sums = "lenient")` keeps every sum INT32 so arithmetic on it is checked. Eager pushes the window nodes with `"error"` and reruns any overflow through `filter_window_fallback()`. Lazy pushes them with `"double"`: a top-level sum only turns FLOAT64 when it really overflows (the C++ evaluator types `col` operands from the real column), which is inexact only when another group's in-range sum meets overflowing integer arithmetic (known residual); a sum nested in another window call's arguments (`max(x * sum(y))`, `ir_nested_window_sum_keys()`) gets spec-level `"error"`, because a later stage was typed from its INT32, and fails at `collect()`.
    - `summarise()`: the window-argument path (`resolve_agg_window_arg()`) plans with `"int"` and computes the stages before returning (even for a bare window column), in both modes. On overflow it raises `cuplyr_summarise_rerun`, and `summarise.tbl_gpu()` reruns the whole call with the original dots through `gpu_fallback()`, eager and lazy.
    - The retries go through `retry_on_sum_overflow()` (R/execute.R), which holds back the attempt's warnings and drops them when the rerun takes over.

## Source Map

### C++ Files
| File | Purpose |
|------|---------|
| `src/gpu_table.hpp` | `GpuTablePtr`, `make_gpu_table_xptr()`, `get_table_view()` |
| `src/cuda_utils.hpp` | `check_cuda()` error helper |
| `src/ops_common.hpp` | `get_compare_op()` (used by `expr_eval.hpp`'s string-comparison handler) |
| `src/transfer_io.cpp` | `df_to_gpu()`, `gpu_collect()`, `gpu_head()`, `gpu_dim()` |
| `src/ops_filter.cpp` | `gpu_filter_bool()`, `gpu_filter_mask()` — CPU-eval fallback mask application only; every IR-parseable predicate now goes through `gpu_filter_expr()` (`ops_expr.cpp`) |
| `src/expr_eval.hpp` | Expression-IR evaluator shared by filter/mutate/summarise temp columns (`eval_ctx`, `ast_expressible()`, `build_ast()`, `materialize()`, `apply_handler()`); `build_join_ast()` (Phase 7 J7) is a side-aware sibling for non-equi `join_by()` conditions — emits `column_reference(LEFT\|RIGHT)`, never retrofitted into `build_ast()` |
| `src/int_sum.hpp` | `narrow_int_sum()`/`is_int_sum()` (F34/F42): INT64 total of an integer/logical `sum()` -> INT32, or FLOAT64 plus a flag when it leaves R's integer range; shared by `gpu_summarise()` and the window "agg" family |
| `src/ops_expr.cpp` | `gpu_compute_column()`, `gpu_filter_expr()`, `gpu_mutate_expr()` — the IR entry points |
| `src/ops_select.cpp` | `gpu_select()` |
| `src/ops_groupby.cpp` | `gpu_summarise()`. All `quantile()` aggregations of one call that read the same value column share one groupby request (`QuantileRequests`), so cudf sorts it once however many probabilities there are (F22: reframe's k rewritten dots). `median()` of a float column is one of them, at 0.5 (keeping median's NA propagation for na.rm = FALSE, not quantile's error); an integer median stays cudf MEDIAN. The value is R's type 7, which cudf LINEAR is not exactly: it blends `(1-f)*lo + f*hi` even at f = 0 (`0 * Inf = NaN`, noise next to a far neighbour such as 8 next to 1e9), where R takes `x[lo]`, and R's index `1 + (n-1)*p` rounds in two steps, so it can be whole where cudf's position is not. So EVERY quantile request (FLOAT64 and INT32 columns, and the float median) carries COUNT_VALID (once per request) + LOWER + HIGHER, and `QuantileRequests::take()` recomputes R's index in separate (non-FMA) binary ops, with `n1 = max(count-1, 0)` once per request: a whole index takes that element, otherwise the lower value is used wherever LOWER == HIGHER. LOWER/HIGHER of an INT32 column are INT32 and are cast to LINEAR's FLOAT64 first. This costs about 2x the LINEAR-only time at k = 101 (2e6 rows, 1000 groups: 0.04 s to 0.08 s). GPU-computed NaN (`0 / 0`; an upload already turns R's NaN into null) is missing like in R: `inspect_float_column()` (one sum reduction as the gate, exact checks only when the sum isn't finite) finds NaN/Inf once per float column, and every na.rm = TRUE aggregation, plus min/max/median/quantile under na.rm = FALSE (`agg_reads_nan_as_null()`), reads the column with NaN made null, so quantile() errors and min/max/median give NA (R: NaN/NA). sum/mean/sd/var keep cudf's NaN; n_distinct/first/last/nth with na.rm = FALSE see the value |
| `src/ops_arrange.cpp` | `gpu_arrange()` |
| `src/ops_reframe.cpp` | `gpu_reframe_expand()` (F22): wide one-row-per-group table -> k rows per group on the device (`cudf::repeat` for keys/size-1 results, `cudf::interleave_columns` for a size-k result, a `sequence % m` gather map for literals) |
| `src/ops_join.cpp` | join logic (equi hash joins, native `semi`/`anti`/`cross`/`right`, `gpu_cond_join()` for non-equi `join_by()` via `mixed_join`/`conditional_join`) with the shared device-side sanitize→stable-sort→diag→`multiple=`-filter→gather/coalesce pipeline (`build_join_result()`) for dplyr row-order/cardinality parity |
| `src/ops_bind.cpp` | `gpu_bind_rows_aligned()`, `gpu_bind_cols_impl()`, `gpu_cast_column()`, `gpu_cast_to_string()`; factor fidelity (Phase 11 L2): `gpu_remap_codes()`, `gpu_decode_factor()` |
| `src/ops_gather.cpp` | vctrs support (F02): `gpu_handle_src()` (constant ALTREP source-id vector for row handles), `gpu_gather_rows()` (gather by (source,row) handles from one or several same-layout tables, NA -> null row) |
| `src/gpu_info.cpp` | device availability/info |

### R Files
| File | Purpose |
|------|---------|
| `R/tbl-gpu.R` | `tbl_gpu()`, `new_tbl_gpu()`, `is_tbl_gpu()`, `resolve_exec_mode()`. `dim()`/`nrow()` of a lazy table with pending ops run the plan on the GPU (`compute()`, no collect, caller untouched); `ncol()`/`NCOL()` are answered from the schema (`dim.tbl_gpu` checks that its caller is `base::ncol`/`NCOL`). Internal code that only needs a cheap size uses `base_dim()` (base table's dims, never executes). |
| `R/utils.R` | `gpu_type_from_r()`, `col_index()` |
| `R/tz.R` | POSIXct time zone metadata (F32): `tz_is_utc()`, `tz_fold()`, `tz_by_position()`/`tz_from_positions()`, `propagate_tz()`, `propagate_schema_meta()`, `restore_tz()`; see "POSIXct time zones" above |
| `R/execute.R` | unified execution helpers: `push_op()` (the sole exec_mode branch point; carries `groups` and `.drop` -- `.drop` only while a group column survives), `input_node()`, `current_schema()`, `propagate_groups()`, `propagate_factor_levels()`, `auto_name_dots()`, `cuplyr_fallback_notify()`, join analogues `join_input_node()`/`push_join()` |
| `R/ir.R` | Expression IR shared by filter/mutate: `ir_parse_quo()`, `ir_cols()`, `ir_cost()`, `ir_is_const()`, `ir_bind()`, `ir_infer_type()`, `ir_call_registry` |
| `R/filter.R` | filter verb — parses expressions via `ir_parse_quo()` into `make_predicate()` records, builds one `ast_filter` node via `push_op()`; constant expressions (TRUE/FALSE/vectors) are evaluated in R, not the GPU IR path; unparseable expressions fall back to a CPU-eval boolean mask (`filter_eval_mask()`) |
| `R/mutate.R` | mutate verb — parses expressions into `make_mutate_expr()` structs, builds one `ast_mutate` node via `push_op()`; type inference lives in `R/ast.R::infer_mutate_output_type()` |
| `R/select.R` | select verb — parses tidyselect, builds one `ast_select` node via `push_op()` |
| `R/arrange.R` | arrange verb — parses sort specs (incl. `.by_group`), builds one `ast_arrange` node via `push_op()` |
| `R/group-by.R` | `group_by()`, `ungroup()`, `group_vars()` — metadata-only, no AST node |
| `R/reframe.R` | native `reframe()` (F22): `reframe_native_plan()` screens the dots (`quantile(x, <const probs>, na.rm=)`, `range(x, na.rm=)`, column-free literals, scalar aggregations; anything else, and any sequential/`across()`/`pick()` dot, is the whole-call CPU fallback, decided before GPU work), rewrites each size-k dot into k scalar aggregations, runs `summarise_impl()` for the WIDE table, then pushes one `ast_reframe` node (barrier; `lower_reframe()` -> `gpu_reframe_expand()`). Call heads must resolve in the quosure env to the real function (a user's own `range` goes to the CPU), literals are plain vectors without `dim`. `reframe_wide_dots_ok()` is a POSITIVE allowlist, so reframe() doesn't inherit plain summarise()/IR/window divergences: each wide dot is `n()` or `sum/mean/min/max/sd/var/median/quantile(arg, na.rm = <literal>)`, `arg` a bare column or a column-reading expression walked by `reframe_arg_info()` over INT32/FLOAT64/BOOL8 columns, literals and `reframe_arg_fns` only (no window/aggregate call, no `^`/sqrt/log, no string operand; comparisons, `!`, `&`, `|`, is.na() only on NaN-free integer/logical operands, since a GPU NaN compares as a number), with per-aggregation types (`reframe_agg_arg_types()`: median() doubles only, quantile() INT32/FLOAT64, a bare timestamp, or a bare STRING under `session_collation_is_c()` (R/utils.R), only for min()/max()). The k dots of one expression share one summarise temp column (`preprocess_agg_expressions()` reuses a temp column, or a window argument's resolved column, for an identical parsed IR and `keep_bool`). A 0-group table with a size-k result raises vctrs' "Can't recycle" at lowering. `%%`/`%/%` of an integer by a double is native only for a finite, non-zero, whole literal divisor with |d| <= `.Machine$integer.max` (exact on the GPU); a fractional divisor (`1L %/% 0.1` is 9 in R, floor(x / y) gives 10) is the CPU's. A `sum()` of an INT32 argument can leave R's integer range (a double in dplyr), which a lazy plan can't type, so `reframe_wide_dots_ok()` returns the names of such dots (`plan$int32_sum`): on a lazy input the call computes the pending ops and runs the wide summarise eagerly (the reconcile in `summarise_impl()` widens the type), then the reframe node goes back to lazy; with pending ops this fires the `cuplyr_fallback_notify()` notice naming just those user-facing dots, mirroring `summarise_run_now()`. The wide step suppresses `notify_materialization()` notices (summarise's early compute of pending ops) |
| `R/summarise.R` | summarise/groupby verb — `parse_aggregations()` builds aggregation structs (cudf-accepted function names), builds one `ast_summarise` node via `push_op()`; temp-column preprocessing for expressions inside agg calls |
| `R/join.R` | join verbs — parse join spec (incl. `join_by()`, equi and non-equi) via `parse_join_by()`, dispatch through `push_join()` (join analogue of `push_op()`); `build_join_schema()`, `build_join_output_info()` (propagates `factor_levels` keyed by post-suffix output name, Phase 11 L2), `check_join_cardinality()` (`multiple=`/`unmatched=`/`relationship=`); routes `closest()`/rolling joins, STRING non-equi conditions, degenerate `suffix=c("",...)` collisions, and same-named equi keys that are factor-typed on both sides with different level sets (`join_factor_key_levels_conflict()`) to the notified CPU fallback (`join_route_to_fallback()`) |
| `R/bind.R` | `bind_rows()`, `bind_cols()` (plain functions, not S3 generics, so `!!!` splices work; errors cite the user's call; `bind_rows()` keeps the first input's groups/`.drop` and NA-pads zero-column inputs; `bind_cols()` follows vctrs' size rules: size-1 recycles, zero-column inputs give their size) with schema unification; `compute_unified_schema()` computes a factor level UNION (Phase 11 L2 Bug 2), `align_to_schema()`/`cast_column()` remap (`gpu_remap_codes()`) or decode (`gpu_decode_factor()`, Bug 3) factor columns as needed |
| `R/collect.R` | pulls data to R, warns on INT64 precision loss. `collect_frame()` is the UNGROUPED tibble (what every internal caller that needs rows uses: fallbacks, pull, sets, joins, vctrs, `$`); `collect()` = `collect_frame()` + `dplyr::grouped_df()` with the table's `.drop` (dplyr/dbplyr contract: a grouped tbl_gpu collects grouped). `as.data.frame()`/`as_tibble()` (ungrouped)/`as.list()`/`as.matrix()` also go through `collect_frame()` |
| `R/generics.R` | dplyr backend generics (`dplyr_col_modify()` etc.), `[`, and the base-R object contract (F04): `$`, `[[`, `$<-`, `[[<-`, `[<-`. For user code `gt$x` / `gt[["x"]]` fetch ONE column (ungroup + select + collect), `gt$x <- v` / `gt[[j]] <- v` are a GPU column modification (`gpu_assign_column()` -> `apply_col_changes()`: upload the new columns, `bind_cols()`, reorder; exec_mode and surviving groups kept), `[<-` and cell assignment are loud errors pointing at `mutate()`. Subscript checks, recycling and messages are tibble's own, run on a placeholder "stub" tibble; `[[`, `[[<-` and `$<-` take `(x, ...)` and forward the user's original promises to tibble (subscripts are evaluated once, in their own frame; labels in messages stay the user's expressions). A plain non-missing length-one `$<-`/`[[<-` value becomes a literal `mutate()` (a lazy plan is extended, not run); `x[[i, j]]` gathers one row on the GPU. **Internal fields**: a field name (`tbl_gpu_fields`) read/written by package code (caller's `topenv()` is the namespace, or its env chain below the global env holds the test marker `.cuplyr_internal_access` from `tests/testthat/helper-gpu.R`) is the raw list element; anything else (user code, other packages, global-env functions) gets the column. Code that must reach a field from elsewhere uses `unclass(x)$field` |
| `R/compute.R` | `compute()`, `collapse()`, `as_lazy()`, `as_eager()`, `show_query()` — `compute()` is the only caller of `optimize_ast()` |
| `R/ast.R` | AST node constructors (`ast_source`, `ast_filter`, etc.) and `infer_schema()` methods |
| `R/optimizer.R` | AST optimization passes (projection, filter pushdown, fusion) — runs only via `compute()`, never on the eager single-op path |
| `R/lower.R` | `lower_and_execute()` — the single execution path for every AST node in both eager and lazy modes |
| `R/gpu-memory.R` | memory reporting and GC helpers |
| `R/gpu.R` | `has_gpu()`, `gpu_details()` |
| `R/print.R` | `print.tbl_gpu()` method |
| `R/vctrs.R` | vctrs contract (F02): `vec_proxy()` returns a two-column frame of row handles (`.src` = integer source id, constant ALTREP `gpu_handle_src()`; `.row` = `seq_len(n)`), so `vec_size()` is O(1); `vec_restore()` decodes them into one GPU gather. Source registry holds sources weakly with ids that are never reused; `vec_proxy()` also pins the source strongly until the enclosing vctrs entry closure (innermost non-generic vctrs frame, `vctrs_entry_frame()`) returns, because vctrs copies handle values out of the proxy and may drop the table before restoring. A dead source is a loud `cuplyr_error_vctrs_dead_source`. `vec_proxy_equal/compare/order` and cross-layout casts collect (via `cuplyr_fallback_notify()`); `vec_ptype2`/`vec_cast` against tbl_gpu/tibble/data.frame (result stays a tbl_gpu); `length.tbl_gpu` = number of columns. `names()`/`names<-` answer vctrs (row names) with NULL / no-op, see `called_from_vctrs()` in `R/tbl-gpu.R`. `vec_rbind()`/`vec_cbind()` (so `dplyr::bind_rows()`/`bind_cols()`) on a tbl_gpu raise `cuplyr_error_vctrs_bind_unsupported`; use `cuplyr::bind_rows()`/`bind_cols()` (`R/bind.R`: `bind_rows_inputs()`/`bind_cols_inputs()` replicate dplyr 1.2.1's argument flattening -- single-bare-list unwrap, `vctrs::obj_is_list()` splicing, NULLs dropped -- then any tbl_gpu among the inputs routes to the GPU and everything else goes to dplyr unchanged). A lazy input's `compute()` is memoized per vctrs entry call (`vctrs_computed()`), released with the pins |

### Benchmark Files
| File | Purpose |
|------|---------|
| `benchmark/benchmark_memory.R` | lazy vs eager memory/time comparison |
| `benchmark/benchmark_filter_pushdown.R` | auto vs manual filter placement |

## Implementing a New dplyr Verb

### Step 1: C++ Implementation (`src/ops_<verb>.cpp`)

```cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include <cudf/...>  // operation-specific headers
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_<verb>(SEXP xptr, /* params */) {
    using namespace cuplyr;

    // 1. Get table view
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // 2. Validate inputs
    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds: %d", col_idx);
    }

    // 3. Perform cuDF operation
    auto result = cudf::some_operation(view, ...);

    // 4. Return new table
    return make_gpu_table_xptr(std::move(result));
}
```

### Step 2: R wiring (verb -> AST node -> lowering)

Verbs never call C++ directly. A verb parses its arguments, builds ONE AST
node on `input_node(.data)`, and hands it to `push_op()` (`R/execute.R`), the
only place that branches on eager vs lazy. Execution happens in
`R/lower.R::lower_and_execute()`. `select()` is the smallest real example:

```r
# R/ast.R -- constructor + schema inference
ast_<verb> <- function(input, ...) ast_node("<verb>", input = input, ...)
infer_schema.ast_<verb> <- function(node) { ... }   # output names/types

# R/<verb>.R -- the verb
#' @export
#' @importFrom dplyr <verb>
<verb>.tbl_gpu <- function(.data, ...) {
  schema <- current_schema(.data)          # schema after pending lazy ops
  # ... parse/validate args against `schema`
  push_op(.data, ast_<verb>(input_node(.data), ...))
}

# R/lower.R -- add a case to the switch(ast$type, ...) in lower_and_execute()
lower_<verb> <- function(ast, source_ptr) {
  input_ptr <- lower_and_execute(ast$input, source_ptr)
  input_schema <- infer_schema(ast$input)
  gpu_<verb>(input_ptr, ...)               # 0-based column indices
}
```

If the optimizer must reason about the new node (projection pruning, filter
pushdown), teach `R/optimizer.R` about it too, or make sure it is treated as a
barrier. Joins use `join_input_node()`/`push_join()` instead.

### Step 3: Update Exports

Run `tools/document` (it runs `Rcpp::compileAttributes()` and roxygen under
the GPU lock). Never hand-edit `NAMESPACE`, `src/RcppExports.cpp` or
`R/RcppExports.R`.

### Step 4: Tests (`tests/testthat/test-<verb>.R`)

```r
test_that("<verb>() basic case works", {
  skip_if_no_gpu()

  df <- data.frame(x = c(1, 2, 3))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::<verb>(...) |>
    collect()

  expect_equal(result$x, expected)
})
```

## AST & Lazy Evaluation

### One execution path
Every verb follows the same shape: parse its arguments, build **one** AST
node wrapping `input_node(.data)`, and hand it to `push_op()` (or, for
joins, `push_join()`) in `R/execute.R`. `push_op()` is the *only* place
that branches on `exec_mode`:
- **lazy**: the node is stored as the new `$lazy_ops`; no GPU work happens.
- **eager**: the node is lowered and executed immediately via
  `lower_and_execute()` (`R/lower.R`), with no optimizer pass — a
  single-op AST has nothing to optimize, and running the optimizer eagerly
  risks routing a multi-predicate filter through the fused kernel
  incorrectly (see design decision D2 in
  `scratchpad/unification_design.md` if present).

`optimize_ast()` (`R/optimizer.R`) only ever runs from `compute()`, which
is invoked by `collect()`/`show_query()`/explicit `compute()` calls on a
lazy table with pending ops. `R/lower.R::lower_and_execute()` is the single
execution path for every AST node in both schedules — there are no
separate eager/lazy implementations (`*_lazy()`/`*_one()` functions) per
verb any more.

### Lazy Mode Mechanics
- Lazy tables defer execution by building an AST in `$lazy_ops`
- Operations return new `tbl_gpu` with updated AST, no GPU work
- `collect()` triggers AST lowering and execution

### Optimizer Passes
The optimizer transforms the AST before execution. Pass order matters:

1. **Projection Pruning** (`push_down_projections`): Inserts `select` nodes to drop unused columns early
   - `build_join_output_info()` maps output columns back to left/right sources
   - Dead columns are dropped before expensive operations

2. **Mutate Fusion** (`fuse_mutates`): Combines consecutive mutate nodes
   - Guards: max 8 expressions, max 4 intermediates (lower outputs the upper reads)
   - Concatenates in original order (lowering is sequential, so overwrites need no reordering)

3. **Dead Column Pruning** (`prune_dead_columns`): Removes unused mutate outputs
   - Walks root to leaves tracking required columns
   - Empty mutate nodes are eliminated
   - Join pruning (here and in `push_down_projections`) shares
     `join_side_requirements()`: pruning must not change a required column's
     output name, so the collision partner of a suffixed (`v.x`) column is kept
     on the other side (verified by recomputing names on the pruned schemas)

4. **Filter Pushdown** (`push_down_filters`): Moves filters closer to data sources
   - Pushes across mutate when predicates don't depend on outputs
   - Join pushdown rules:
     - Inner join: left-only predicates -> left, right-only -> right
     - Left join: only left-only predicates pushed
     - Right join: only right-only predicates pushed
     - Full join: no side-only pushdown allowed
     - Nothing is pushed below a left/inner/right join with `unmatched = "error"`
       or `relationship` one-to-one/one-to-many/many-to-one: dplyr's checks read
       the full inputs, so a pushed filter could hide or create the error
     - No right-side pushdown when `multiple` is first/last/any (the pick is
       among all of an x row's y matches); left-side pushdown is unaffected
     - A pushed predicate is rewritten from output names (`v.x`) to the side's
       source names (`join_source_predicate()`)
     - Accepted limitation: with default `relationship = NULL` pushdown may
       suppress dplyr's advisory many-to-many warning (rows are identical)

5. **Filter Reordering** (`reorder_filters`): Executes cheaper filters first
   - Collects consecutive filter chains
   - Sorts by `estimated_cost` field

There is no separate filter-fusion pass: `lower_filter()` always folds every
predicate in a filter node together with `&` into one IR expression and
evaluates it as a single `gpu_filter_expr()`/`compute_column()` kernel, so a
filter node with N predicates is already exactly one fused kernel.

### Join-Specific Notes
- Lazy joins build `ast_join` with two inputs
- Source pointers attached via `set_ast_source_ptr()` before lowering
- Schema inference: `infer_schema.ast_join` uses `build_join_schema()` in `R/join.R`
- Right join (Phase 7 J4): native `gpu_right_join()` (`cudf::inner_join()` for
  matched pairs + a `filtered_join` anti-join for unmatched-y rows), NOT a
  swapped left join + column reorder (that helper was deleted in J4). Shares
  the same device-side join-map pipeline as every other mutating join type
  (see "Join Ordering & Unmatched Rows" below) to reproduce dplyr's
  x-matched-then-unmatched-y-tail row order.
- Non-equi/mixed `join_by()` conditions (Phase 7 J7): `ir_bind_join()`
  resolves each condition's columns by the spec's explicit side tag
  (`0`=left/`1`=right, not name-lookup order — required for self-joins with
  identical schemas), and `build_join_ast()` (`src/expr_eval.hpp`, a
  side-aware sibling of `build_ast()`, NOT a retrofit of it) emits
  `column_reference(LEFT|RIGHT)` nodes for `gpu_cond_join()`
  (`cudf::mixed_join()` when ≥1 equi condition, `cudf::conditional_join()`
  otherwise).
- Key type coercion (F06/A3): `resolve_join_key_types()` (verb time, schema
  only) walks every key pair (equi and non-equi) with
  `resolve_common_col_type()` (R/bind.R, shared with bind_rows/set ops) and
  raises `dplyr_error_join_incompatible_type` with dplyr's text (type labels
  from `vctrs::vec_ptype_full()`, so factor hashes match). Compatible-but-
  different equi pairs go in `join_spec$key_casts` (travels on `ast$by`);
  matching types leave it `NULL` and lowering is untouched (perf). Merged keys
  (`keep != TRUE`, mutating) are cast in place on both sides and the output
  key has the common type (`build_join_output_info()`, also `cast_keys` so
  the optimizer won't push a predicate on such a key below the join).
  `keep = TRUE`, semi/anti and non-equi columns keep their original types:
  `apply_join_key_casts()` (R/lower.R) matches on cast copies (only the side
  that needs a cast) appended as trailing temp columns and
  `drop_join_key_temps()` removes them afterwards (not via `right_drop`, which
  would also switch on right/full coalescing). Non-equi pairs are recorded in
  `key_casts` too, except numeric pairs (`INT32`/`INT64`/`FLOAT64`), which the
  AST evaluator widens itself (it can only cast to INT64/FLOAT64, so Date vs
  POSIXct or logical vs integer need the copies). The condition IR is bound
  against the schemas extended with the copies (`cond_by` names them), so the
  evaluator never sees mismatched operand types. `cast_join_key_column()`
  errors clearly on a factor key whose levels are unknown (NULL).
  Factor -> STRING uses `gpu_decode_factor()`, never `gpu_cast_column()`.
- `by = character()` (F06/A4): detected by `is_cross_by()` in every join verb;
  the lifecycle warning is raised once from the verb (`warn_join_cross_by()`,
  `env` = dplyr's namespace so it says "dplyr 1.1.0"). A join that ends on the
  CPU fallback (`join_route_to_fallback()`, `nest_join()`) warns from the verb
  too and mutes the copy dplyr raises inside
  (`mute_cross_by_warning()`). The node is a normal `join_type` join with an empty spec, `keep = TRUE`, `cross_by = TRUE`;
  `lower_cross_by_join()` decides at execution time from the two row counts
  (cartesian via `gpu_cross_join()`, NA-padded outer results for an empty
  side, semi/anti = "y has rows"), synthesizes the cardinality diag for
  `check_join_cardinality()` (default relationship many-to-many). Never a
  hash join on a constant key. A keyless side may not be pruned to zero
  columns (row count lost): `join_keep_a_column()` in the optimizer.
- Join validation (F36): `check_join_cols_unique()` (R/join.R) is dplyr's
  `check_join_vars()` ("Join columns in `x` must be unique.", plain
  `rlang_error`), run on the raw `by` in every join verb before any routing.
  The advisory many-to-many warning follows dplyr's `compute_join_relationship()`:
  `mutating_join()` decides at the verb call (`join_may_warn_many_to_many()`:
  all-`==` spec and `join_is_direct(user_env)`, a port of `is_direct()`); when it
  may not warn, a default `relationship` becomes `"many-to-many"` on the node
  (a no-op check, survives optimizer rebuilds). Known limitation (roxygen): a
  pushed-down filter can still change the warning's row numbers. Cast-key temp
  columns in `apply_join_key_casts()` get names absent from both schemas
  (`.cuplyr_key_x1`, then `_1`, ...). `vctrs::unspecified()` columns upload as
  BOOL8 and are named in `$unspecified` (tbl field, like `fixed_keys`); only
  `push_op()` row-only nodes (filter/arrange/slice/barrier), group_by/ungroup,
  compute and semi/anti joins (`push_join()`, x side) carry it, and
  `collect_frame()` restores the class for still-BOOL8 all-NA columns. Every
  other verb drops it (plain logical). The two places that bypass `push_op()`
  keep it honest: `names<-.tbl_gpu` remaps it by position (dropping a name
  that ends up duplicated), and the same-layout shortcut of
  `cast_gpu_to_gpu()` keeps only markers that `to` also has.
- Groups (F06): every join type keeps `intersect(x$groups, output names)`
  and `x$group_drop` (`push_join()`); there is no D3 "joins clear groups" rule
  any more. `gpu_fallback()` regroups with `.drop = x$group_drop` and reads
  `.drop` back from the result, so CPU-fallback joins keep it too.
- `nest_join()` is still a CPU fallback; its default `name` is
  `as_label(enexpr(y))` and it warns for `by = character()` from the verb.
- Ensure desired column names exist before calling `gpu_select`

### Optimizer Barriers
- Barrier reattachment uses a helper to avoid R copy-on-modify pitfalls
- Be careful when modifying AST nodes in place

## Common Patterns

### Expression Parsing (R side)
```r
# Capture unevaluated expressions
dots <- rlang::enquos(...)

# Get expression text
expr_text <- rlang::quo_text(expr)

# Get raw expression
raw_expr <- rlang::quo_get_expr(expr)

# Check if symbol (bare column name)
if (is.symbol(raw_expr)) col_name <- as.character(raw_expr)

# Check if call (function application)
if (is.call(raw_expr)) {
  fn_name <- as.character(raw_expr[[1]])  # e.g., "desc", "-"
  arg <- raw_expr[[2]]                     # first argument
}
```

### Column Index Conversion
```r
# R uses 1-based, C++ uses 0-based
col_idx_r <- match(col_name, .data$schema$names)  # 1-based, NA if not found
col_idx_cpp <- col_idx_r - 1L                      # 0-based for C++
```

### Validation Patterns (C++)
```cpp
// Bounds check
if (col_idx < 0 || col_idx >= view.num_columns()) {
    Rcpp::stop("Column index out of bounds: %d (table has %d columns)",
               col_idx, view.num_columns());
}

// NA check in LogicalVector
if (LogicalVector::is_na(value[i])) {
    Rcpp::stop("NA values not allowed in parameter");
}

// Row count limit (int32 for indices)
if (view.num_rows() > static_cast<cudf::size_type>(INT32_MAX)) {
    Rcpp::stop("Table too large (max ~2.1 billion rows)");
}
```

## Known Issues & Sharp Edges

### Environment-Specific cuDF Issues
_Installed version: cuDF 25.12.0 (see `cudf/version_config.hpp`; conda pkg `libcudf-25.12.00`)._
- **Header locations differ by version**: `bitmask_allocation_size_bytes` is in `cudf/null_mask.hpp`. There is no `cudf/bitmask.hpp` file in this version at all.
- **Join headers**: Use `<cudf/join/join.hpp>`, not `<cudf/join.hpp>`. That header declares `inner_join`, `left_join`, `full_join`, and `cross_join`. It also still declares free-function `left_semi_join`/`left_anti_join`, but both are marked `[[deprecated]]` in favor of the object-oriented API below.
  - Non-deprecated semi/anti joins: `<cudf/join/filtered_join.hpp>`, class `cudf::filtered_join` — construct once from a build table, then call `.semi_join(probe)` / `.anti_join(probe)` (each returns a `device_uvector<size_type>` of left-table indices).
  - Condition/mixed joins: `<cudf/join/conditional_join.hpp>` (AST-predicate joins) and `<cudf/join/mixed_join.hpp>` (equality keys + AST residual condition).
- **cuDF gather API**: `cudf::gather()` (in `cudf/copying.hpp`) takes an `out_of_bounds_policy bounds_policy` parameter (`NULLIFY` or `DONT_CHECK`, default `DONT_CHECK`) but has **no** `negative_index_policy` parameter — that enum only exists on the detail-namespace overload in `cudf/detail/gather.hpp`, not on the public API.
- **Avoid device-side Thrust** unless compiling with nvcc

### Type Consistency
R schema types MUST match actual GPU column types. If changing type handling:
- Update `R/utils.R::gpu_type_from_r()`
- Update `src/transfer_io.cpp::df_to_gpu()`
- Update tests for round-trip behavior

### String Column Operations
String columns use offset-based storage (Apache Arrow format):
- `col.child(0)` = offsets column (int32, n+1 elements)
- `col.data<char>()` = concatenated character data
- When slicing (e.g., head), must slice both offsets and chars correctly

### Memory Semantics
- Most ops allocate new tables (immutable design)
- Peak memory for sorting: ~2x table size
- Use `gpu_memory_state()` to monitor
- GC frees GPU memory via pointer release

### Grouping Behavior
- `group_by()` only sets metadata (`$groups`), no GPU work
- Groups are applied during `summarise()`
- `arrange(.by_group=TRUE)` prepends group columns to sort keys
- Preserve `$groups` when returning new tbl_gpu unless ungrouping

### summarise() edge contracts (F07)
- **Fixed group keys (`$fixed_keys`)**: `slice(.preserve = TRUE)` on a `group_by()`-grouped table stores the input's group-keys tibble (empty groups included) in `$fixed_keys` (`new_tbl_gpu(fixed_keys=)`, a `tbl_gpu_fields` member). Read only through `fixed_keys_of()` (R/group-by.R), which returns `NULL` unless its names equal `$groups`. It is honoured by `compute_group_data()` (group_data/keys/rows/indices/size/n_groups), `collect()` and `gpu_fallback()`'s regroup (`with_fixed_groups()`), and `summarise()` routes to the CPU fallback when it is set. `push_op()` carries it via `carry_fixed_keys()` only for select/rename/mutate/window nodes that leave the group columns untouched; `compute()` and `dplyr_reconstruct()` copy it; every other constructor omits it, which drops it. Default to dropping when adding a constructor. `filter()`/`filter_out()` (one engine, `filter_impl()` in R/filter.R) set it for `.preserve = TRUE` and for `.drop = FALSE` tables (dplyr keeps their whole group structure), and drop it on every other exit; `filter_out()` pushes one `ast_filter` with the predicate `is.na(P) | !P` for the ANDed dots, and sends opaque/vector/matrix dots, named dots and zero-column tables to `dplyr::filter_out()` on the CPU.
- **Zero-column tables**: cudf cannot hold rows without columns, so a zero-column `tbl_gpu` carries its row count in `$nrow` (`new_tbl_gpu(nrow=)`, stored only when the schema has no columns). Every constructor that copies a `tbl_gpu` must pass `nrow = x[["nrow"]]` (`push_op()`, `compute()`, `group_by()`, `ungroup()`). `dim()` gives `NA` and `collect()` errors when a zero-column table has no recorded count, so a dropped count can't become a silent 0. `nrow` is one of `tbl_gpu_fields` (R/generics.R), so internal `x[["nrow"]]` reads the field, never a user column of that name. Supported on such a table: `tbl_gpu()`, `dim()`/`base_dim()` (`zero_column_dim()`, never runs a plan), `collect()`/`collect_frame()` (so fallbacks, `pull()`, sets see the count), `compute()`/`collapse()`, `group_by()`/`ungroup()`, `summarise()` (zero dots: 1 x 0, built fresh without reading `$ptr` after computing any pending ops; with dots: CPU fallback, so `n()` is right), ungrouped `slice()`/`slice_head()`/`slice_tail()` (`slice_zero_column()`), `bind_rows()`/`bind_cols()` of zero-column tables only, a literal-mask `filter()` (`filter_eval_mask()`), `dplyr_row_slice()`, `$<-`/`[[<-` (`gpu_assign_column()` sizes by `dim()`, uploads the new columns at that count, errors on an unknown count). F14 adds the producers: `zero_column_result()` (R/execute.R, the one builder: ungrouped, same exec_mode, pending lazy ops computed first for the count) is used by `select()` (also on a zero-column input), `transmute()` with no output columns, `[` with an empty column subscript (rows applied first), `distinct()` of a zero-column table or with only NULL dots (`min(n, 1)` rows); `relocate()` of a zero-column table is the identity. Everything else errors through `check_zero_column_input()` (called by `push_op()`, `push_join()` for both sides, the binds and `vec_proxy.tbl_gpu()`) when the table has rows or an unknown count.
- **Zero dots**: ungrouped -> 1 x 0; grouped -> zero-aggregation summarise node (sorted by its keys like any grouped summarise); `.by` -> `distinct()`.
- **Group order and empty groups (F33)**: `gpu_summarise(sort_keys=)` sorts the assembled `[keys, aggs]` result by the keys (`cudf::sorted_order`, ascending, nulls after, then gather) *before* `summarise_output_index()` permutes it; `ast_summarise(sort_groups = TRUE)`. Only `summarise_by_desugar()` passes FALSE (it re-sorts by first appearance). On a `group_by(.drop = FALSE)` grouping with a factor key (`summarise_wants_empty_groups()`, schema-only check), `summarise_with_empty_groups()` (R/summarise.R) adds the empty groups of `compute_group_data()`: observed groups summarised as usual, an identity row from the same aggregations on a 0-row ungrouped slice, one `gpu_gather_rows(list(observed_aggs, identity_row), idx)` plus the uploaded `gd` keys. Aggregates never round-trip through R (an upload turns NaN into NA). Zero dots (`summarise_group_keys()`) uploads `compute_group_data()`'s keys instead. Lazy input is computed first on the GPU (a materialization point, no fallback notice). An aggregate named like a key goes to the CPU fallback. count()/tally() get their zero-count rows from this path. `carry_fixed_keys()` also keeps `$fixed_keys` through arrange/distinct nodes, and `push_join()` through semi/anti joins, when the input's `group_drop` is FALSE (dplyr's `dplyr_row_slice()`); with `.drop = TRUE` they are dropped.
- **Key redefinition**: an aggregate named like a grouping column replaces the key in place. `summarise_output_index()` (R/ast.R) is the single permutation of `gpu_summarise()`'s `[keys, aggs]` used by both `infer_schema.ast_summarise()` and `lower_summarise()`. Dots that reference an earlier dot's name (as a symbol, `.data$x` or `.data[["x"]]`), or repeat a name, go to the CPU fallback (`summarise_dots_are_sequential()`).
- **min()/max() identity**: `gpu_summarise()` takes a per-aggregation `identity` flag (set by `make_aggregation()` for min/max over INT32/FLOAT64 unless internal) and returns an `identity_used` attribute; `lower_summarise()` warns with base R's text. An INT32 result column becomes FLOAT64 iff some group is empty (or the input has 0 rows), so the type is data-dependent: such a summarise runs immediately even in lazy mode (`summarise_run_now()`, which notifies via `cuplyr_fallback_notify()` when it has to materialize pending ops) and the schema is corrected from `gpu_col_types()`.

### select()/relocate()/rename() resolution (F14)
- tidyselect in `select()`/`relocate()`/`rename()`/`rename_with()` goes through ONE resolver, `resolve_selection()` (R/select.R). A static walk, `selection_is_type_only()`, decides where it runs. Names, positions, ranges, `c()`/`-`/`!`/`&`/`|` and the name/pattern helpers (`starts_with()`, `all_of()`, `everything()`, `last_col()`, `group_cols()`, ...) carry no predicate. A `where()` (or a bare/spliced predicate function) passes only if `where_predicate_is_type_only()` proves it a base class test: `is.numeric/is.double/is.integer/is.character/is.logical/is.factor/is.ordered` or `inherits(x, "<constant>")` applied to the predicate's own argument, bare, `base::`, or inside a lambda/`function(x)`/closure value combining them with `!`, `&&`, `||`, `&`, `|`, `(`, a one-expression `{` (call heads are looked up in the lambda's or closure's environment). Such selections run on `tidyselect_proxy(schema, groups)`: a one-row frame with, per column, a value of the class `collect()` gives it (`tidyselect_proxy_value()`; a factor carries the column's `schema$factor_levels` and `ordered`), built with `vctrs::new_data_frame()` so duplicated names are allowed, and grouped like the table so `group_cols()` works. Everything else (a predicate that reads values, a type test the walk can't see through such as `lubridate::is.Date`, any other call) is a notified CPU fallback: the table is computed once, collected, grouped, and the selection resolved on the real data; the computed table then becomes the input of the verb, so the plan does not run twice. On a table with duplicated names that fallback is an error.
- Grouped `select()` follows dplyr's `ensure_group_vars()`: unselected grouping columns are prepended with "Adding missing grouping variables: ..." unless a rename already uses the name. Groups follow their column by position (`set_groups_by_position()`, also used by `relocate()`), overriding `propagate_groups()`'s by-name result. Internal code must not call `select()` on grouped data expecting a plain projection (use `ast_select`).
- Name builders (F40): `selection_is_type_only()` also accepts calls to the base functions in `name_builder_fns` (`names`, `sort`, `paste0`, `setdiff`, `[`, `c`, `:`, ...) and `.data$x`/`.data[["x"]]`, only when each is the base function (looked up in the env, compared by identity) and every argument is itself safe (`name_builder_is_safe()`); then the selection resolves on the proxy with no data. `resolve_by()` (`.by`/`by`) uses the same rule (proxy, else the real data with a notice). `across()`/`if_any()`/`if_all()` `.cols` that is not type-only makes `resolve_across_cols()` return `NULL`, which hands the whole call to dplyr (`across_fallback_result()`); type-only ones resolve on `tidyselect_proxy()` of the non-group columns.
- `ast_select(positions=)` (F40): like `ast_rename(positions=)`, set only when the input has duplicated names (`push_select_rename()`, shared by select/relocate; the rename on top is positional too); such a node is an optimizer barrier. `collect()` builds the tibble with `.name_repair = "minimal"` so a relocate result may keep duplicated names.
- `ast_rename(positions=)`: set only when the input has duplicated names (`rename_positions()`); the optimizer then passes no requirement through the node, and a source with duplicated names is never pruned.
- Duplicated names fail closed (F40): `push_op()` refuses any node on an input with duplicated names unless `node_is_position_safe()` allows it (`select`/`rename` with `positions`, `distinct` with `key_positions`, a `slice` with no `order_col`, `barrier`), raising dplyr's "Can't transform a data frame with duplicate names."; `push_join()` and every join verb raise "Input columns in `x`/`y` must be unique." (`check_unique_join_inputs()`). Verbs dplyr refuses also check at their entry (`check_unique_input_names()`: filter, mutate, transmute, summarise, arrange with dots, the slice verbs) so the message comes first. Position-correct paths: select/relocate/rename, `distinct()`/`unique()` (`distinct_key_positions()`; a bare duplicated name keys on its first column, as dplyr), `[`/`$`/`[[` (`ast_select_at()`), `head()`/`tail()` (a row-only slice node, though `slice_head()` refuses), set ops (whole-call fallback) and `bind_rows()` (vctrs' "unique" repair as a positional rename). `group_by()` and `pull()` raise dplyr's own errors. A new verb that builds a by-name node gets the error, not the first column.
- `count()`/`add_count()`/`distinct()` drop NULL dots; a dot the IR can't parse sends the whole call to `gpu_fallback()` with the original dots. `dimnames.tbl_gpu()` (so `colnames()`) and `unique.tbl_gpu()` (= `distinct()`) exist.

### Join Ordering & Unmatched Rows
- **cuDF join outputs are unordered**: We stable-sort join maps by left_map (then right_map) in `src/ops_join.cpp` to match dplyr
- **JoinNoMatch sentinel**: cuDF uses sentinel values; gather treats negatives as wraparound
- **Device-side join-map pipeline** (Phase 7 J4, `build_join_result()` in
  `src/ops_join.cpp`): sanitize join maps on device (replace both `< 0`
  *and* `>= nrows` sentinel values with `nrows` — the `>= nrows` half was a
  latent gap in the original CPU-side sanitize, fixed in J4) → stable sort
  → single device-to-host transfer only when diagnostics/`multiple=` are
  needed (`join_map_stats()`: `left_multi_first`/`right_multi_first`/
  `left_unmatched_first`/`right_unmatched_first`) → `multiple=` host
  filtering (Phase 7 J5, skipped entirely for the default `multiple =
  "all"`) → gather with `out_of_bounds_policy::NULLIFY` + coalesce. Every
  mutating join type (`left`/`inner`/`full`/`right`) shares this one
  pipeline; C++ exports return `list(ptr, diag)`, and `lower_join()`
  (`R/lower.R`) calls `check_join_cardinality()` (`R/join.R`) with the raw
  `diag` to raise dplyr's exact `multiple=`/`unmatched=`/`relationship=`
  conditions.
- **Right key dropping**: `keep = FALSE` drops right keys even when names differ (only legal for an all-equi spec — `keep = FALSE` on any genuinely non-equi `join_by()` condition errors with dplyr's own text; see `resolve_join_keep()`, `R/join.R`)
- **`keep = NULL`'s real resolution is per-condition, not per-join** (Phase 7
  J7 finding): a mixed `join_by(c == d, a >= b)` drops `d` (the equi pair)
  but keeps *both* `a` and `b` (the non-equi pair, suffixed) — ported
  directly from `dplyr:::join_cols()`'s own per-condition branch, see
  `build_join_output_info()`.

### Filter Parsing
- Expressions parse through `ir_parse_quo()` (`R/ir.R`) into `make_predicate()` records (`ir`, `cols`, `estimated_cost`, `is_deterministic`, `na_sensitive`); optimizer passes read `pred$cols`, never the IR tree directly
- Constant expressions (`ir_is_const()` — no column reference anywhere, e.g. `TRUE`, `FALSE`, `rep(TRUE, n)`) never reach the GPU IR path: they're evaluated in R (`eval_tidy` on the original quosure, not the constant-folded IR) so an n-row mask always has a real column to broadcast against; `TRUE` is a genuine no-op, `FALSE`/`NA` build an impossible predicate (`ir_lit_from_r(FALSE)`) that still flows through the normal lazy `push_op()`/`ast_filter` path
- Comparing a non-STRING column against a STRING literal (or vice versa) is caught by `check_filter_comparison_types()` before lowering, with a message naming the column and its type
- Expressions the IR doesn't recognize fall back to a CPU-eval boolean mask (`filter_eval_mask()`), with no data mask (only expressions that evaluate standalone, e.g. `rep(TRUE, n)`, can succeed there)

### Mutate Parsing
- Expressions parse through `ir_parse_quo()` (`R/ir.R`) into `make_mutate_expr()` records (`output_col`, `ir`, `input_cols`, `output_type`), lowered to a single `gpu_mutate_expr()` call per `mutate()` node; arbitrary nesting (`(x + y) * z - 1`), scalar-on-either-side, and later dots referencing earlier dots' outputs are all handled by the IR/evaluator, not by special-cased chain lowering
- An expression shape the IR doesn't recognize is a hard error (no CPU fallback for `mutate()`, unlike `filter()`)

### Summarise Aggregation Sub-Expressions
- `create_temp_column()` (`R/summarise.R`) is a single IR-based implementation: any sub-expression `ir_parse_quo()` understands is valid inside an aggregation call (`sum(x > 3 & y < 2)`, `mean(sqrt(x))`, `sum(x %% 2 == 0)`, etc.), built via one `gpu_mutate_expr()` call, same as `mutate()`
- A `BOOL8`-inferred sub-expression (comparisons, `&`/`|`/`!`, `is.na()`, ...) is declared `INT32` for the temp column so `sum()` promotes to `INT64` like R's `sum(logical)`; `gpu_mutate_expr()` performs the actual GPU-side cast
- `preprocess_agg_expressions()` decomposes the aggregation call's raw expression (not text) via `decompose_agg_call()`, so a bare column reference needs no temp column and any unparseable shape falls through to `parse_aggregations()`'s existing diagnosable error
- Dots whose argument parses to the same IR (and `keep_bool`) share one temp column, window arguments included (F22): the window stages run once, since their result depends only on the IR and the table's groups and row order

### Bind Operations
- `bind_rows()` computes unified schema via `compute_unified_schema()`
- Type promotion hierarchy: BOOL8 < INT32 < INT64 < FLOAT64; STRING is widest
- Missing columns filled with nulls via `gpu_make_null_column()`
- `bind_cols()` uses `vctrs::vec_as_names()` for name repair when available
- Both operations materialize lazy tables before binding

## Debugging Build Failures

- If a cudf header can't be found, check `$CONDA_PREFIX` (the shared pixi env; `tools/lib.sh` activates it) and `src/Makevars`
- Re-run `./configure` (via `tools/build`, which runs it when `src/Makevars` is missing) after updating CUDA/cudf libs
- Use `rg --files -g '*bitmask*' $CONDA_PREFIX/include/cudf` to locate moved headers
- Join build errors: confirm `#include <cudf/join/join.hpp>` for inner/left/full/cross join; use `#include <cudf/join/filtered_join.hpp>` (`cudf::filtered_join`) for semi/anti join since the free `left_semi_join`/`left_anti_join` functions are deprecated; avoid device-side Thrust unless compiling with nvcc
- GPU not detected is common in CI or local dev; tests use `skip_if_no_gpu()`
- Rcpp exports need regeneration after moving/adding functions: run `tools/document`

## Testing

### Available Helpers (`tests/testthat/helper-*.R`)
```r
skip_if_no_gpu()           # Skip test if no GPU available
expect_valid_tbl_gpu(x)    # Check tbl_gpu structure (allows exec_mode field)
expect_data_on_gpu(x)      # Verify data is on GPU
gc_gpu()                   # Force GPU garbage collection
gpu_memory_snapshot()      # Get memory state for comparison
```

### Test Pattern
```r
test_that("operation handles edge case", {
  skip_if_no_gpu()

  # Setup
  df <- data.frame(...)
  gpu_df <- tbl_gpu(df)

  # Execute
  result <- gpu_df |> some_operation() |> collect()

  # Assert
  expect_equal(result$col, expected)
})
```

### Lazy vs Eager Cross-Mode Testing
Use `tibble::as_tibble()` to avoid rownames/class mismatches when comparing:
```r
expect_equal(

  as_tibble(eager_result),
  as_tibble(lazy_result)
)
```

## Code Review Checklist

Before merging dev to master, verify:
- [ ] **Branch references**: Update Colab links back to `master` branch:
  - `README.md`: Change `blob/dev` to `blob/master` in Colab badge URLs
  - `notebooks/install_cuplyr.ipynb`: Change git clone from `-b dev` to remove the flag (or use `-b master`)
- [ ] **Type alignment**: `R/utils.R` and `src/transfer_io.cpp` agree on type mapping
- [ ] **Head/collect parity**: If a type is supported in `gpu_collect()`, ensure `gpu_head()` uses the same conversion path
- [ ] **Arrange semantics**: Stable sort support, NA ordering, group-prepend behavior
- [ ] **Rcpp exports**: If new C++ functions exist, `R/RcppExports.R` and `src/RcppExports.cpp` are updated
- [ ] **Tests**: New features have matching `tests/testthat/test-*.R` coverage with `skip_if_no_gpu()`
- [ ] **Joins**: Left-table order preserved, right-key dropping correct, pushdown rules followed
- [ ] **Binds**: Type promotion correct, null columns for missing, lazy tables materialized

## cuDF Header Quick Reference

| Operation | Header | Key Functions |
|-----------|--------|---------------|
| Sorting | `<cudf/sorting.hpp>` | `sorted_order`, `stable_sorted_order`, `sort` |
| Gathering/Scattering | `<cudf/copying.hpp>` | `gather`, `scatter`, `empty_like` |
| Filtering | `<cudf/stream_compaction.hpp>` | `apply_boolean_mask` |
| Binary ops | `<cudf/binaryop.hpp>` | `binary_operation` |
| Aggregation | `<cudf/aggregation.hpp>`, `<cudf/groupby.hpp>` | `groupby::aggregate` |
| Null handling | `<cudf/null_mask.hpp>` | `bitmask_allocation_size_bytes` |
| Scalars | `<cudf/scalar/scalar.hpp>`, `<cudf/scalar/scalar_factories.hpp>` | `make_numeric_scalar` |
| Joins | `<cudf/join/join.hpp>` | `inner_join`, `left_join`, `full_join`, `cross_join` |
| Semi/anti joins | `<cudf/join/filtered_join.hpp>` | `filtered_join` class: `.semi_join()`, `.anti_join()` (free `left_semi_join`/`left_anti_join` in `join.hpp` are `[[deprecated]]`) |
| Conditional/mixed joins | `<cudf/join/conditional_join.hpp>`, `<cudf/join/mixed_join.hpp>` | AST-predicate and equality+condition joins |
| Concatenation | `<cudf/concatenate.hpp>` | `concatenate` (for bind_rows/bind_cols) |
| Distinct/duplicates | `<cudf/stream_compaction.hpp>` | `distinct`, `distinct_indices`, `stable_distinct`, `unique_count`, `distinct_count` |
| Rank | `<cudf/sorting.hpp>` | `rank` |
| Scan (column) | `<cudf/reduction.hpp>` | `scan` |
| Scan/shift (grouped) | `<cudf/groupby.hpp>` | `groupby::scan`, `groupby::sort_scan`, `groupby::shift` |
| Conditional copy | `<cudf/copying.hpp>` | `copy_if_else` |
| Null/value replace | `<cudf/replace.hpp>` | `replace_nulls`, `find_and_replace_all`, `clamp` |
| AST expression eval | `<cudf/transform.hpp>`, `<cudf/ast/expressions.hpp>` | `compute_column` (+ `ast::literal`, `ast::column_reference`, `ast::operation`, `ast::tree`) |
| Strings | `<cudf/strings/*.hpp>` | e.g. `case.hpp`, `find.hpp`, `replace.hpp`, `split/`, `convert/` |
| Datetime | `<cudf/datetime.hpp>` | datetime component extraction/arithmetic |
| Sampling/slicing | `<cudf/copying.hpp>` | `slice`, `split`, `sample`, `shift` |
