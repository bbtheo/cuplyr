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
  # rowwise = TRUE               # ONLY on a rowwise table (F49); `groups` then holds
  #                              # the rowwise vars -- see "Rowwise tables" below
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

F50 runs label operations on a bare factor column natively by rewriting them
at parse time (`ir_parse_factor_cmp()`, `ir_parse_factor_in()`,
`ir_parse_factor_conversion()` in `R/ir.R`, only when the levels are known,
the schema names are unique and no level is NA): `f == "a"`, `%in%` and (for
ordered factors) `<`/`<=`/`>`/`>=` become comparisons of `factor_codes(f)`
(stripped to the column by `ir_bind()`) with the label's level position;
`as.character(f)` is the `factor_labels` handler (gathers the levels by code,
`src/factor_gather.hpp`); `as.integer`/`as.numeric` give the codes.
`ir_factor_rule()` allows `factor_codes`/`factor_labels` over a bare column.
Strings reach the GPU as their raw bytes, so `as.character(f)` decodes
natively only when every level is UTF-8 (`ir_strings_are_utf8()`) and no
non-UTF-8 string literal sits in the same expression, `meta` included
(`ir_decode_meets_non_utf8()`); otherwise it runs on the CPU. The names in
`ir_internal_ops` (`factor_codes`, `factor_labels`, `cast_int32`,
`cast_float64`) are registry entries only the parser builds: a user call by
those names is never dispatched to them.

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
- **Typed allowlist (F48)**: `ir_infer_type()` also runs `ir_check_type_operands()` (R/ir.R) beside `ir_check_ts_operands()`. `ir_native_arg_types` lists, per reducer/window op, the GPU types of its first argument that are native (numeric reducers and `cum*`/`roll_*`: BOOL8/INT32/INT64/FLOAT64; `median`: FLOAT64/INT64, since an integer/logical median's R type depends on the data; `cummax`/`cummin`: no BOOL8, R gives integers; `any`/`all`/`cumall`/`cumany`: BOOL8 only; `min`/`max`: STRING only when `session_collation_is_c()`). Any other STRING/BOOL8/INT32/INT64/FLOAT64 argument signals `cuplyr_type_needs_cpu` (`stop_type_needs_cpu()`, a subclass of `cuplyr_factor_needs_cpu` that `ir_parse_quo()` re-raises), so the dot takes the CPU fallback. Comparisons and `%in%` mixing a STRING with a number/logical, and `< <= > >=` on strings outside a C collation, take it too; `between()`'s desugared comparisons carry `meta$c_order` (vctrs order) and skip only the collation rule. `cumall()`/`cumany()`/`cummean()` desugar at parse time, so their parse hooks call `ir_type_gate()` themselves. A Period/Duration/Interval/difftime value (a literal, `lag()`'s `n`/`default`) is `cuplyr_type_needs_cpu` too (`ir_check_timespan_lit()`). `summarise_dots_parseable()` asks the same `ir_type_gate()` with the aggregate's argument type. A Period/Duration/difftime also goes to the CPU where a call reads it as a parameter (`ntile()`/`nth()`/`roll_*()` `n`, summarise's `nth()` `n` and `quantile()` `probs`, `slice()` indices; each tests `is_timespan_value()` and returns `NULL`/signals `cuplyr_type_needs_cpu` to the pre-pass), and a column-free mutate() constant catches the signal itself (`parse_mutate_dots()` builds that literal without `ir_parse_quo()`). The STRING collation rule lives only in `ir_type_gate()` (`ir_collated_ops`: `min`/`max` and the ordering comparisons, which `ir_check_type_operands()` asks). `ntile()` runs dplyr's FLOAT64 formula and ends in `cast_int32` (a `cudf::cast` handler, integer result; an INT32 chain paid F42's range checks per node, 33 ms vs 19 ms at 1e6 rows).
- **Logical operands that R types differently (F19-a2)**: window `max()`/`min()` and unary `-` of a logical are integers in R and `round()` of a logical is a double (cuDF's round rejects BOOL8). The parse hooks of those registry entries (`ir_parse_bool8_cast()`, the unary branch of `ir_parse_ts_arith("-")`; helper `ir_cast_bool8_first()`) wrap a BOOL8 first argument in `cast_int32` (`cast_float64` for `round()`), so the node's inferred type, the lazy schema and the evaluated column agree and the colwise/across() rewrites inherit it. `ir_native_arg_types` still lists BOOL8 for `min`/`max` (summarise's own max()/min() are INT32 through make_aggregation()'s cast). `summarise_dots_parseable()` treats any error of `extract_agg_shape_param()` (quantile `type != 7`, a `probs` or `nth()` shape the native path can't take) as "not parseable": the CPU gives dplyr's value or error. The CPU fallbacks of mutate/transmute/summarise pass the table as `.data =`, so a dot named `.` isn't partially matched to `.data`.
- **Named arguments in registry calls (F19-a2)**: `ir_parse_expr()` checks every registry call with `ir_args_bind_by_position()` before parsing its arguments by position. Formals come from the real R function (`ir_op_formals()`: base, then dplyr, stats, lubridate, via `formals(args(fn))`, cached). A call stays native only if every named argument has the exact name of the formal at its own position (before any `...`) and no positional argument would fall into `...`; anything else (reordered, partial, unknown or post-`...` names such as `.ptype`) returns NULL and the verb's CPU fallback runs dplyr. Base `max()`/`min()` ignore names in their `...` and are exempt. A new registry entry with no R function behind it takes unnamed arguments only. Hooks that read names themselves (`rlang::call_match()` against a proto: lag/lead, first/last/nth, ntile, string helpers, roll_*) keep doing so.
- **NA-flag spelling (F19-a2)**: each reducer takes only its own spelling (`na.rm` for base reducers and `n_distinct()`, `na_rm` for dplyr's first/last/nth; `summarise_na_flag` in R/summarise.R). The other spelling, a flag that isn't a single non-NA TRUE/FALSE evaluated outside the data mask (`ir_na_flag_value()`), or a duplicate makes the dot not parseable, so dplyr gives its value or error.
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
  CPU. A
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
| `src/gpu_table.hpp` | The `tbl_gpu$ptr` payload (F57): `GpuTable` = a vector of shared const columns, `GpuTablePtr` (what the XPtr holds), `make_gpu_table_xptr()` (from a fresh `cudf::table` or from shared columns), `get_table_view()`, `get_shared_column()`, `share_column()`; see "Shared columns" |
| `src/memory_resource.cpp` | The RMM memory resource installed at load (F57): `gpu_mr_install()`/`gpu_mr_exists()`/`gpu_mr_state()`/`gpu_mr_kind()`/`gpu_mr_release()`/`gpu_mr_release_threshold()`, plus `gpu_sync_device()` (`cudaDeviceSynchronize()` behind `gpu_sync()`); see "Memory resource" |
| `src/cuda_utils.hpp` | `check_cuda()` error helper |
| `src/r_strings.hpp` | R string <-> cudf boundary (F53): `r_string_utf8()`/`r_scalar_utf8()` translate every R string that becomes cudf data to UTF-8 (a "bytes" string is an error); `r_mk_utf8_char()` marks strings read back from cudf as UTF-8. Use them at any new site that sends R strings to cudf. |
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
| `src/ops_join.cpp` | join logic (equi hash joins, native `semi`/`anti`/`cross`/`right`, `gpu_cond_join()` for non-equi `join_by()` via `mixed_join`/`conditional_join`, `gpu_rolling_join()` for `closest()` (F24), `gpu_string_codes()` for non-equi STRING conditions) with the shared device-side sanitize→stable-sort→diag→`multiple=`-filter→gather/coalesce pipeline (`build_join_result()`) for dplyr row-order/cardinality parity |
| `src/ops_bind.cpp` | `gpu_bind_rows_aligned()`, `gpu_bind_cols_impl()`, `gpu_cast_column()`, `gpu_cast_to_string()`; factor fidelity (Phase 11 L2): `gpu_remap_codes()`, `gpu_decode_factor()` |
| `src/ops_gather.cpp` | vctrs support (F02): `gpu_handle_src()` (constant ALTREP source-id vector for row handles), `gpu_gather_rows()` (gather by (source,row) handles from one or several same-layout tables, NA -> null row) |
| `src/ops_rolling.cpp`, `src/ops_rolling.hpp` | F59: `roll_sum`/`roll_mean`/`roll_sd`/`roll_min`/`roll_max` for `gpu_window()`'s rolling family, in time independent of the window width: segments of w rows, a head scan plus a mirrored tail scan per segment (SUM, or MIN/MAX for van Herk min/max), sums of squares about a member of each window part, NA/Inf from prefix counts, sd overflow recomputed on a 2^-k-scaled column; exact INT64 integer sums; `roll_native()` holds the measured crossover widths below which `ops_window.cpp` keeps cudf's `grouped_rolling_window()`; the internal option `cuplyr.roll_path` (env `CUPLYR_ROLL_PATH`, read in `R/roll.R`, passed in each rolling spec) forces either path for tests and benchmarks. The algorithm is described at the top of the .cpp |
| `src/roll_median.cpp` | F59: CPU sliding median (`roll_median_cpp()`, a Fenwick tree over value ranks, O(n log n) whatever the width) behind `roll_median_sorted()` in `R/roll.R` |
| `src/gpu_info.cpp` | device availability/info |
| `src/ops_window.cpp` | `gpu_window()`: shift/scan/rank/agg windows; its rolling branch hands numeric `roll_*()` specs to `ops_rolling.cpp` and keeps `cudf::grouped_rolling_window()` only for what that file does not cover |
| `src/gpu_scope.cpp` | release scopes (F58): step and call scopes; free a user-level call's intermediate tables on return |
| `src/oom_inject.cpp` | `gpu_oom_inject()` test hook (F58) |

### R Files
| File | Purpose |
|------|---------|
| `R/tbl-gpu.R` | `tbl_gpu()`, `new_tbl_gpu()`, `is_tbl_gpu()`, `resolve_exec_mode()`. `dim()`/`nrow()` of a lazy table with pending ops run the plan on the GPU (`compute()`, no collect, caller untouched); `ncol()`/`NCOL()` are answered from the schema (`dim.tbl_gpu` checks that its caller is `base::ncol`/`NCOL`). Internal code that only needs a cheap size uses `base_dim()` (base table's dims, never executes). |
| `R/utils.R` | `gpu_type_from_r()`, `col_index()` |
| `R/tz.R` | POSIXct time zone metadata (F32): `tz_is_utc()`, `tz_fold()`, `tz_by_position()`/`tz_from_positions()`, `propagate_tz()`, `propagate_schema_meta()`, `restore_tz()`; see "POSIXct time zones" above |
| `R/execute.R` | unified execution helpers: `push_op()` (the sole exec_mode branch point; carries `groups` and `.drop` -- `.drop` only while a group column survives; `input_schema=` skips re-deriving a schema the caller has, `carry = FALSE` leaves the frame metadata to the caller), `push_chain()` (an eager verb's several nodes as one execution, see "Eager chains"), `input_node()`, `current_schema()`, `propagate_groups()`, `propagate_factor_levels()`, `auto_name_dots()`, `cuplyr_fallback_notify()`, join analogues `join_input_node()`/`push_join()` |
| `R/ir.R` | Expression IR shared by filter/mutate: `ir_parse_quo()`, `ir_cols()`, `ir_cost()`, `ir_is_const()`, `ir_bind()`, `ir_infer_type()`, `ir_call_registry` |
| `R/filter.R` | filter verb — parses expressions via `ir_parse_quo()` into `make_predicate()` records, builds one `ast_filter` node via `push_op()`; constant expressions (TRUE/FALSE/vectors) are evaluated in R, not the GPU IR path; unparseable expressions fall back to a CPU-eval boolean mask (`filter_eval_mask()`) |
| `R/mutate.R` | mutate verb — parses expressions into `make_mutate_expr()` structs, builds one `ast_mutate` node via `push_op()`; type inference lives in `R/ast.R::infer_mutate_output_type()` |
| `R/select.R` | select verb — parses tidyselect, builds one `ast_select` node via `push_op()` |
| `R/arrange.R` | arrange verb — parses sort specs (incl. `.by_group`), builds one `ast_arrange` node via `push_op()` |
| `R/group-by.R` | `group_by()`, `ungroup()`, `group_vars()` — metadata-only, no AST node |
| `R/reframe.R` | native `reframe()` (F22): `reframe_native_plan()` screens the dots (`quantile(x, <const probs>, na.rm=)`, `range(x, na.rm=)`, column-free literals, scalar aggregations; anything else, and any sequential/`across()`/`pick()` dot, is the whole-call CPU fallback, decided before GPU work), rewrites each size-k dot into k scalar aggregations, runs `summarise_impl()` for the WIDE table, then pushes one `ast_reframe` node (barrier; `lower_reframe()` -> `gpu_reframe_expand()`). Call heads must resolve in the quosure env to the real function (a user's own `range` goes to the CPU), literals are plain vectors without `dim`. `reframe_wide_dots_ok()` is a POSITIVE allowlist, so reframe() doesn't inherit plain summarise()/IR/window divergences: each wide dot is `n()` or `sum/mean/min/max/sd/var/median/quantile(arg, na.rm = <literal>)`, `arg` a bare column or a column-reading expression walked by `reframe_arg_info()` over INT32/FLOAT64/BOOL8 columns, literals and `reframe_arg_fns` only (no window/aggregate call, no `^`/sqrt/log, no string operand; comparisons, `!`, `&`, `|`, is.na() only on NaN-free integer/logical operands, since a GPU NaN compares as a number), with per-aggregation types (`reframe_agg_arg_types()`: median() doubles only, quantile() INT32/FLOAT64, a bare timestamp, or a bare STRING under `session_collation_is_c()` (R/utils.R), only for min()/max()). The k dots of one expression share one summarise temp column (`preprocess_agg_expressions()` reuses a temp column, or a window argument's resolved column, for an identical parsed IR and `keep_bool`). A 0-group table with a size-k result raises vctrs' "Can't recycle" at lowering. `%%`/`%/%` of an integer by a double is native only for a finite, non-zero, whole literal divisor with |d| <= `.Machine$integer.max` (exact on the GPU); a fractional divisor (`1L %/% 0.1` is 9 in R, floor(x / y) gives 10) is the CPU's. A `sum()` of an INT32 argument can leave R's integer range (a double in dplyr), which a lazy plan can't type, so `reframe_wide_dots_ok()` returns the names of such dots (`plan$int32_sum`): on a lazy input the call computes the pending ops and runs the wide summarise eagerly (the reconcile in `summarise_impl()` widens the type), then the reframe node goes back to lazy; with pending ops this fires the `cuplyr_fallback_notify()` notice naming just those user-facing dots, mirroring `summarise_run_now()`. The wide step suppresses `notify_materialization()` notices (summarise's early compute of pending ops) |
| `R/summarise.R` | summarise/groupby verb — `parse_aggregations()` builds aggregation structs (cudf-accepted function names), builds one `ast_summarise` node via `push_op()`; temp-column preprocessing for expressions inside agg calls |
| `R/join.R` | join verbs — parse join spec (incl. `join_by()`, equi and non-equi) via `parse_join_by()`, dispatch through `push_join()` (join analogue of `push_op()`); `build_join_schema()`, `build_join_output_info()` (propagates `factor_levels` keyed by post-suffix output name, Phase 11 L2), `check_join_cardinality()` (`multiple=`/`unmatched=`/`relationship=`); `join_by_natively_supported()`/`join_by_rolling_native()` is the positive allowlist for non-equi shapes (see Join-Specific Notes); routes two `closest()`s, `closest()` next to another inequality, factor/logical/Date-vs-POSIXct rolling keys, STRING-vs-non-STRING non-equi conditions, degenerate `suffix=c("",...)` collisions, and same-named equi keys that are factor-typed on both sides with different level sets (`join_factor_key_levels_conflict()`) to the notified CPU fallback (`join_route_to_fallback()`) |
| `R/rows.R` | `rows_insert/append/update/patch/upsert/delete` (F26): one driver `rows_run()`: dplyr's check order (`check_dots_empty`, in_place, `auto_copy`, so a non-GPU `y` needs `copy = TRUE`), then the real dplyr verb on 0-row `gpu_proto()` tables for every schema-level error/message (messages held back and replayed only if the native path completes; a CPU fallback makes dplyr emit them itself). `rows_native_ok()` is the positive allowlist (BOOL8/INT32/FLOAT64/STRING/Date/POSIXct, y column = x's type or INT32 to FLOAT64, BOOL8 to INT32/FLOAT64, equal named/UTC zones, no `fixed_keys`); anything else and every data-dependent error (conflict, unmatched or duplicated y keys, found with anti_join/distinct row counts) goes to `gpu_fallback()` so dplyr gives the result or the exact error. Native = `bind_rows_gpu`, `anti_join`, `left_join` of x with y's value columns under temp names plus a matched marker, then `if_else()` per column (`rows_patch_from()`): update takes y's value on a matched row; patch fills only a missing x value (FLOAT64: `is.na(v) | !(v == v)`, so a GPU-computed NaN counts) and takes a FLOAT64 y value only when it isn't missing itself (`if_else(t == t, t, NA_real_)`: x and y both missing gives NA like dplyr's `coalesce()`, never NaN); x's groups/frame meta/exec mode restored by `rows_finish()`. A `.drop = FALSE` `rows_delete()` that empties a group stores x's `slice_group_keys()` as `fixed_keys` (`rows_delete_keys()`, only when the distinct group-key count drops, like `result_fixed_keys()`); a repeated `by` column (`by = c("id", "id")`) goes to the fallback. A proto error with a logical y column against a character/Date/POSIXct x column falls back (an all-NA logical casts in dplyr, a 0-row proto can't tell) |
| `R/bind.R` | `bind_rows()`, `bind_cols()` (plain functions, not S3 generics, so `!!!` splices work; errors cite the user's call; `bind_rows()` keeps the first input's groups/`.drop` and NA-pads zero-column inputs; `bind_cols()` follows vctrs' size rules: size-1 recycles, zero-column inputs give their size) with schema unification; `compute_unified_schema()` computes a factor level UNION (Phase 11 L2 Bug 2), `align_to_schema()`/`cast_column()` remap (`gpu_remap_codes()`) or decode (`gpu_decode_factor()`, Bug 3) factor columns as needed |
| `R/collect.R` | pulls data to R, warns on INT64 precision loss. `collect_frame()` is the UNGROUPED tibble (what every internal caller that needs rows uses: fallbacks, pull, sets, joins, vctrs, `$`); `collect()` = `collect_frame()` + `dplyr::grouped_df()` with the table's `.drop` (dplyr/dbplyr contract: a grouped tbl_gpu collects grouped). `as.data.frame()`/`as_tibble()` (ungrouped)/`as.list()`/`as.matrix()` also go through `collect_frame()` |
| `R/generics.R` | dplyr backend generics (`dplyr_col_modify()` etc.), `[`, and the base-R object contract (F04): `$`, `[[`, `$<-`, `[[<-`, `[<-`. For user code `gt$x` / `gt[["x"]]` fetch ONE column (ungroup + select + collect), `gt$x <- v` / `gt[[j]] <- v` are a GPU column modification (`gpu_assign_column()` -> `apply_col_changes()`: upload the new columns, `bind_cols()`, reorder; exec_mode and surviving groups kept), `[<-` and cell assignment are loud errors pointing at `mutate()`. Subscript checks, recycling and messages are tibble's own, run on a placeholder "stub" tibble; `[[`, `[[<-` and `$<-` take `(x, ...)` and forward the user's original promises to tibble (subscripts are evaluated once, in their own frame; labels in messages stay the user's expressions). A plain non-missing length-one `$<-`/`[[<-` value becomes a literal `mutate()` (a lazy plan is extended, not run); `x[[i, j]]` gathers one row on the GPU. **Internal fields**: a field name (`tbl_gpu_fields`) read/written by package code (caller's `topenv()` is the namespace, or its env chain below the global env holds the test marker `.cuplyr_internal_access` from `tests/testthat/helper-gpu.R`) is the raw list element; anything else (user code, other packages, global-env functions) gets the column. Code that must reach a field from elsewhere uses `unclass(x)$field` |
| `R/compute.R` | `compute()`, `collapse()`, `as_lazy()`, `as_eager()`, `show_query()` — `compute()` is the only caller of `optimize_ast()` |
| `R/ast.R` | AST node constructors (`ast_source`, `ast_filter`, etc.) and `infer_schema()` methods |
| `R/optimizer.R` | AST optimization passes (projection, filter pushdown, fusion) — runs only via `compute()`, never on the eager single-op path |
| `R/lower.R` | `lower_and_execute()` — the single execution path for every AST node in both eager and lazy modes |
| `R/gpu-memory.R` | memory reporting and GC helpers: `gpu_memory_state()`, `gpu_gc()`, `gpu_sync()`, and the load-time resource switches (`memory_resource_config()`, `memory_resource_install()`, called by `.onLoad`) |
| `R/gpu.R` | `has_gpu()`, `gpu_details()` |
| `R/print.R` | `print.tbl_gpu()` method |
| `R/rowwise.R` | rowwise tables (F49): `is_rowwise_gpu()`, `rowwise.tbl_gpu()`, the CPU route `rowwise_cpu()`, the all-or-nothing native attempt `rowwise_attempt()`, the horizontal-aggregate rewrite `rowwise_rewrite_dots()`, rowwise `summarise()` -- see "Rowwise tables" |
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
     - No pushdown at all below an inner/left/right join that can raise dplyr's
       many-to-many warning (`relationship = NULL`, `multiple = "all"`): the join
       must see the full inputs (F53). `relationship = "many-to-many"` opts back in.
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
- Rolling joins (F24): `join_by(<== keys>, closest(x$t <op> y$s))` with exactly
  one `closest()` (`>=`, `>`, `<=`, `<`) is native when the rolling pair is one
  type in `join_rolling_types` (INT32, FLOAT64, Date, POSIXct, STRING) or an
  INT32/FLOAT64 mix (recorded in `key_casts`; the cast copy is searched).
  `parse_join_by()` carries `filter` as a vector parallel to `op`;
  `lower_join()` sends the `filter != "none"` condition to `gpu_rolling_join()`
  (`src/ops_join.cpp`), never to the AST join (an inequality join plus a
  per-row max is O(n*m) pairs). The kernel: stable-sort y by (equi keys,
  rolling key), nulls first; `lower_bound`/`upper_bound` give each x row's equi
  group and boundary candidate; validity checks (group range, null rules:
  non-null x needs a non-null candidate, null x matches only a null candidate
  and only for `>=`/`<=` with `na_matches = "na"`, `"never"` drops any x row
  with a null key); a FLOAT64 rolling value has three kinds (present, null,
  NaN from GPU arithmetic: cudf sorts NaN after +Inf), a candidate must have
  the x row's kind, a missing kind matches like null does, and under `"never"`
  a NaN in any FLOAT64 key of the x row matches nothing; a second search gives the tie block; scan/repeat/sequence
  expand blocks into gather maps (right/full append y rows nobody picked with
  the left sentinel), then `build_join_result()` runs unchanged, so
  `multiple=`/`unmatched=`/`relationship=` act on the post-filter matches like
  dplyr. semi/anti keep x rows with/without a block. The optimizer never
  pushes a right-only predicate below a `closest()` join (`right_push_ok`).
- Non-equi STRING pairs (F24): cudf's AST can't compare strings, so
  `resolve_join_key_types()` records a `string_codes` cast and
  `apply_join_key_casts()` appends one INT32 code column per side from
  `gpu_string_codes()` (dense rank of both columns concatenated, nulls stay
  null, bytewise UTF-8 = vctrs' C locale); `cond_by` names them. STRING against
  a non-STRING stays on the fallback.
- `multiple = "error"`/`"warning"` (undocumented, defunct in dplyr's NEWS but
  still working through vctrs) never reach the native code: `mutating_join()`
  sends them to the CPU fallback for every join shape, before validation, so
  `build_join_result()`/`apply_multiple_filter()` only see all/any/first/last
  (anything else is an internal error). `join_route_to_fallback()` runs dplyr
  from a frame that inherits the global env when the user's call was direct, so
  dplyr's own advisory many-to-many warning survives the fallback. Under
  right/full `multiple = "first"/"last"/"any"`, `apply_multiple_filter()` lists
  each y row no kept pair uses once as unmatched (not once per capped-out pair).
  `join_map_stats()` runs on the maps AFTER that filter: vctrs resolves
  `multiple` first, so `relationship=` and the "remaining y rows" of
  `unmatched = "error"` see only the kept matches.
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

### NaN vs NA (F55)
NaN is a value on the device: upload (`numeric_to_gpu()`, only `R_IsNA`
becomes null), arithmetic, collect, `is.nan()`, group keys and join keys keep
it. A fixed list of missingness-sensitive consumers reads a FLOAT64 operand
with NaN as null, via `nan_nulled_copy()` (`src/ops_common.hpp`) and
`eval_ctx::nan_as_null()` (`src/expr_eval.hpp`, cached per ctx index, a no-op
when the column holds no NaN so the AST kernel stays non-nullable): `is.na()`
and the six comparisons (`build_ast()`; a NaN literal there is a null scalar),
every `coalesce()` argument, `arrange()` keys, `cummax()`/`cummin()` input,
equi-join keys under `na_matches = "never"`. Don't add an in-AST NaN guard:
it forces the nullable kernel (+50-90% on comparisons). `is.nan()` is an
`apply_handler()` op (valid and NaN).

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
- Ops that build new rows (filter, arrange, slice, joins, summarise, bind_rows) allocate new columns; ops that pass columns through (select, relocate, rename, the untouched columns of mutate/transmute, a bare copy `a = x`, the input columns of a window mutate, bind_cols) **share** the input's columns, so they allocate nothing for them (see "Shared columns")
- Peak memory for sorting: ~2x table size
- `gpu_memory_state()` monitors it: `used_bytes` is the device (all processes, includes what the pool reserved), `pool_used_bytes` is what this session has allocated from its memory resource; see "Memory resource"
- GC frees GPU memory via pointer release: a column is freed when the last table sharing it is finalized. Under the pool this lowers `pool_used_bytes`, not the device's `used_bytes`
- **Release scopes (F58)** free a user-level call's intermediates when it returns instead of at the next GC (below). What a user-level call returns belongs to the user and is never freed early; a `tbl_gpu` the user made and dropped (the `group_by()` result of a one-line pipeline) still waits for GC, and so does the inner result of a verb dplyr builds from several cuplyr verb calls (`add_tally(sort = TRUE)` = `mutate()` then `arrange()`, each its own user-level call)
- Tests that ask how much a table occupies or that memory comes back measure the allocator (`gpu_alloc_used_bytes()`, `gpu_alloc_snapshot()` in `tests/testthat/helper-memory.R`, or F56's `gpu_alloc_tracking()`); a claim about the CUDA driver (device-level usage) runs in a pool-off subprocess (`run_with_pool_off()`)

#### Release scopes and the OOM retry (F58)

**Ownership rule.** A GPU table belongs to the *user-level call* that made it: the innermost cuplyr function on the stack whose caller is not cuplyr code (`gpu_entry_frame()`, `R/utils.R`). A verb called from the console, a user function, a data mask (a `filter()` predicate, a join's `y` promise), or another package is a user-level call. A cuplyr function called by cuplyr code, also through a dplyr generic (UseMethod gives the method its generic's caller), works for the user-level call around it. Base closures that call back (`lapply`, `vapply`, `Map`, `Reduce`) are looked through. "cuplyr code" means a frame whose function's environment is the namespace, or a closure made by a running cuplyr frame. A test file's environment descends from the namespace under `load_all()`, so `topenv()` can't decide it. A caller that is no running frame (a data mask, an `eval()` environment) counts as user code: when in doubt the table is the user's and is left to GC.

**Two kinds of scope** share one stack in `src/gpu_scope.cpp`:
- A *step scope*: `wrap_gpu_call()` opens one around every GPU step. While it is innermost, `make_gpu_table_xptr()` (the only place a `GpuTable` is made) records each new table in it. On close, it releases what it recorded that is not reachable from the step's value; the reachable tables move to the scope below.
- A *call scope*: the first step of a user-level call opens it, just below that step's scope (`gpu_call_scope_ensure()`). It stores its id in the call's frame (`.__cuplyr_call_scope__`, set from C++ together with the push) and registers `gpu_call_scope_exit(frame, returnValue(NULL))` on that frame with `do.call(on.exit, ..., envir = frame)`, before it opens. It records nothing. It collects what the call's steps kept, and on exit releases every one not reachable from the call's return value; a failed or interrupted call keeps nothing. What it keeps is **not** handed down: no enclosing scope ever sees a user-level call's result. This is what makes a `tbl_gpu()` evaluated inside a verb's argument, predicate or memoised helper safe.

So every multi-step verb (window filter stages, `summarise(.by)` row ids, the keys `cur_group_id()` joins on, `reframe(.by)`, a lazy input that `slice_sample()`/`bind_rows()`/`bind_cols()` computes, `group_trim()` loops) frees its step tables with no per-verb code. Nothing else opens scopes.

Rules for code in the package:
- `keep` is walked through lists, pairlists (dotted ones too), attributes (a `list_of`'s ptype) and ALTREP payloads (`gpu_handle_src()`, the `vec_proxy()` row-handle vector, holds its source table); it does not enter environments, closures or calls. A table returned only there, or parked in a cache that outlives the call, is released when the call returns. The walk runs only if the closing scope still holds a live table, and visits a shared list, pairlist or ALTREP once.
- A call that returns by a non-local return (`rlang::return_from()` in `vctrs_delegate_dplyr_binder()`) settles its scope with the value first (`gpu_call_scope_settle()`). An unwind would otherwise close it with nothing kept.
- A GPU op outside `wrap_gpu_call()` is not recorded (left to GC) and gets no OOM retry, so every op that creates a table goes through one. `gpu_head()` (the `print()`/`glimpse()` previews) returns host data and makes no table.
- A released table raises "GPU table was released" from `get_table_view()`/`get_shared_column()`; it never looks empty. Only tables a step created are ever released, so inputs, lazy sources, and the table `filter(TRUE)`/`rename()` hand back are safe. Closing an id also closes any scope left open above it (a call scope left open there is dropped without releasing anything, since its call may have returned what it holds); an unknown id is a no-op. Ids are doubles from a 64-bit counter.

**`wrap_gpu_call(op, expr, reword = TRUE)`** (`R/utils.R`) wraps every GPU step: `push_op()`/`push_chain()`/`push_join()`, `compute()`, `collect()`, `tbl_gpu()`'s upload, the casts/null columns/remaps of `bind_rows()` and set-operation alignment, `slice_sample()`'s gathers, the `.drop = FALSE` summarise assembly, vctrs null tables. In the outermost active wrapper only, it retries once on an OOM (`gc()` + `gpu_gc()`, then `expr` is evaluated again in the caller's frame, with the first attempt's warnings muffled in the retry); a second OOM raises `cuplyr_error_gpu_oom`. A multi-step verb retries the failed step, not the whole verb. So **`expr` must be safe to run twice**: it reads its inputs, assigns nothing it reads before assigning, and draws no random numbers (`slice_sample()` wraps only the gather after `sample.int()`). It rewords only what the GPU layer raised (a `C++Error`, or a plain `stop()`); a condition R code classed on purpose (`cuplyr_summarise_rerun`, `cuplyr_error_rowwise_retype`, `dplyr_error_join`, rlang errors) passes through unchanged. `compute()` passes `reword = FALSE`: a lazy plan's lowering errors read as they did before F58. `rowwise_attempt()`/`rowwise_rewrite_dots()` re-raise a `cuplyr_error_gpu_oom` instead of rerunning on the CPU.

To check a verb, measure what a full `gc()` releases with the result still referenced (`tests/testthat/test-parity-F58-free.R`). `gpu_oom_inject(n)` (`src/oom_inject.cpp`) is the test hook: the next `n` RMM allocations throw `rmm::out_of_memory`.

### Shared columns (F57)
`src/gpu_table.hpp`: the XPtr behind `tbl_gpu$ptr` holds a `GpuTable`, a vector of `std::shared_ptr<const cudf::column>`. `view()` builds a `cudf::table_view`, so every `get_table_view()` caller works unchanged. Rules for C++ ops:
- A column that belongs to a table is **immutable**. Nothing may `release()`/move a column out of an input table, take a `mutable_view()` of one, or `set_null_mask()` it; the `const` makes that a compile error. Ops mutate (`set_null_mask()`, ...) only columns they just created, while they still own them as `std::unique_ptr<cudf::column>`, then hand them over with `share_column()`/`make_gpu_table_xptr()`.
- To pass an input column through, put `get_shared_column(ptr, i)` into the result's column vector (`gpu_select`, `gpu_bind_cols_impl`, `gpu_window`'s input columns, `gpu_mutate_expr`'s untouched columns, the unreplaced columns of `gpu_remap_codes`/`gpu_decode_factor`/`gpu_cast_column`/`gpu_cast_to_string`), then `make_gpu_table_xptr(std::vector<SharedColumn>)`. `make_gpu_table_xptr(std::unique_ptr<cudf::table>)` still exists for freshly built tables.
- `gpu_mutate_expr()`: a bare column copy (`mutate(a = x)`) is an alias slot of the evaluation context (`eval_ctx::push_alias()`, `materialize_top_level(alias_ok = true)`), which the result resolves back to the input column; `gpu_compute_column()` keeps copying (its result must be an owned column). Two outputs on one computed slot share it.
- A column's device memory is freed when the last table (or, inside an op, view holder) sharing it goes; a result stays valid after its source table is garbage collected.
- Do not hold raw `cudf::column_view`s across calls: they are valid only while some table holds the column.

### Memory resource (F57)
`.onLoad` (`R/zzz.R`) calls `memory_resource_install()` (`R/gpu-memory.R`) when a GPU is present (there is no `.onUnload` step). `src/memory_resource.cpp` creates a cuplyr-owned resource and installs it with `rmm::mr::set_current_device_resource()`: `"async"` (default since F65) is a `cuda_async_memory_resource`, `"pool"` a `pool_memory_resource` over a cuplyr-owned `cuda_memory_resource`; both sit under a `statistics_resource_adaptor` (the installed object), so bytes in use are exact. Pool and async measured the same within noise (per-call 1e3/1e6 rows and a 1e7-row window mutate). Async is the default because it can give memory back; the pool keeps its high-water mark until exit. F65: async gets `initial_pool_size` = `cuplyr.pool_initial` (priming only; `nullopt` for 0) and, by default, `release_threshold` = `nullopt` (RMM: the device total, so no automatic release and freed memory stays cached like the pool). CUDA's threshold caps the pool's WHOLE reservation, tables in use included, and is applied at every sync (libcudf syncs inside many calls): a fixed default threshold made a session holding more than it run 23-34% slower than the pool at 1e7-2.5e7 rows (round-1 review), so the threshold is opt-in via `cuplyr.release_threshold` / `CUPLYR_RELEASE_THRESHOLD` (bytes; NA/negative into `gpu_mr_install()` = none; an explicit `optional(0)` for 0, which keeps nothing and costs plain-cudaMalloc speed past the threshold). `gpu_mr_release()` (`cudaDeviceSynchronize()` + `cudaMemPoolTrimTo(pool, 0)`, no-op for pool/cuda) is called at the end of `gpu_gc()` (which catches its CUDA error and warns), so memory returns on `gpu_gc()` and in the OOM retry; `gpu_gc()` reports `released_bytes` (drop in `pool_bytes`; 0 for pool, NA for cuda). Under async `pool_bytes` is `cudaMemPoolAttrReservedMemCurrent`. `gpu_mr_release_threshold()` (internal) reads the async pool's `cudaMemPoolAttrReleaseThreshold` (NA under pool/cuda; the device total by default), so tests can pin the default. An unknown kind warns (naming option or env var) and means async, or, with the off switch set, warns that it is ignored and the resource is "cuda"; a zero-length/NA `cuplyr.memory_pool` option, and a zero-length/NA/"" `cuplyr.release_threshold` option, count as unset. If creating the DEFAULT async resource fails (`cudaMallocAsync` unsupported: RMM's constructor throws), `memory_resource_install()` installs "pool" instead and warns; an async named in the option/env var (`memory_resource_config()$explicit`) or a failing pool warns and leaves plain cudaMalloc (the warning names every error). An error before `gpu_mr_install()` was called (an option `memory_resource_config()` can't read) still calls `gpu_mr_install(FALSE, ...)`, which creates nothing but pins this copy of the library. `R/gpu-memory.R`'s decision logic is unit-tested with `gpu_mr_install()`/`gpu_mr_exists()` mocked (`tests/testthat/test-F65-round3.R`). Against plain `cudaMalloc` at 1e6 rows (F57 measurement, ms): arrange 1.52 -> 0.92, filter 1.14 -> 0.70, plain mutate 0.90 -> 0.64, window mutate 1.80 -> 1.22, summarise 1.30 -> 0.88; a 1e7-row window mutate 3.5 -> 2.7.
- Switches, read by the first load that creates the resource (a later load in the same process reuses it whatever they say, and reports it; restart R to change them). `memory_resource_install()` asks `gpu_mr_exists()` (this copy's `g_inst`, or `find_in_other_copies()`) first and, when a resource exists, calls `gpu_mr_install(FALSE, ...)` (reuse only) without reading the switches, so a reload never warns about them; the option wins over the env var: `options(cuplyr.memory_pool = FALSE)` / `CUPLYR_MEMORY_POOL` ("0"/"false"/"no"/"off" = off); `cuplyr.pool_initial` / `CUPLYR_POOL_INITIAL` (bytes; default min(1 GiB, a quarter of the free device memory); a larger request is clamped to 90% of the free memory; growth is the pool's own, doubling); `cuplyr.memory_resource` / `CUPLYR_MEMORY_RESOURCE` ("async", "pool", or "cuda"/"none" = no resource; an unknown value warns); `cuplyr.release_threshold` / `CUPLYR_RELEASE_THRESHOLD` (async only, opt-in, see above); `cuplyr.pool_initial` only primes. Off, when no earlier load created a resource, never calls `set_current_device_resource()`: plain `cudaMalloc`, `resource == "cuda"`.
- `gpu_memory_state()` adds `resource` ("pool"/"async"/"cuda"), `pool_bytes` (what the resource holds from the device) and `pool_used_bytes` (allocated from it now); both NA when "cuda". `used_bytes` stays device-level (all processes) and includes the pages the resource holds, so it does not fall when a table is freed under the pool, and falls under async only after `gpu_gc()` or an opt-in release. `gpu_allocatable_bytes()` (`R/gpu-memory.R`) = device free + (`pool_bytes` - `pool_used_bytes`): what cuplyr can allocate; callers that ask "will it fit" (`warn_if_join_too_large()`, `skip_if_insufficient_gpu_memory()`, the attach message) use it, not `free_bytes`. `gpu_gc()$freed_bytes` is the drop in `pool_used_bytes` under a resource (device-level free memory doesn't move), the rise in free device memory otherwise. `gpu_details()`/`show_gpu()` stay device-level.
- Lifetime: one resource per process, never freed (not by any load or unload, not at exit). Every `pkgload::load_all()` `dyn.load()`s a fresh copy of `cuplyr.so`, and each copy has its own RMM per-device map (RMM's map statics are GLOBAL symbols in `cuplyr.so`), while libcudf stays bound to the map of the first copy loaded in the process; so a resource can be referenced from several copies' maps and from columns any copy created, and no copy can know it is unreferenced. `gpu_mr_install(enabled, kind, initial, release_threshold)` makes the process's resource current in this copy's map, creating it (`create()`) only when none exists and the switch is on. Later copies find it without any R state: each copy exports `extern "C" cuplyr_process_mr_v2()` (returns the copy's `g_inst`), and `find_in_other_copies()` walks the loaded objects with `dl_iterate_phdr()` for other `.../cuplyr.so` files, `dlsym()`s that function in each, and checks what it returns (magic, `kAbi`, `sizeof(Installed)`) before use. Nothing in the R session can forge, clobber or carry it over (round 2's `.cuplyr_mr_handle` option could be all three), and copies are never unmapped, so a pointer returned is always live. Bump `kAbi` and the `_v2` in the function name when `Installed` changes layout. `pin_this_library()` (`dladdr()` + `dlopen(RTLD_NOLOAD | RTLD_NODELETE)`) pins every copy at its first install call: the resource's vtables, the map libcudf is bound to and the finalizers of a copy's columns live in that copy, so `pkgload::unload()`'s `dyn.unload()` must not unmap it. libcudf's own scratch goes through the first copy's map, so after a reload an F56 tracking window (which wraps the current resource of the copy that opens it) does not see libcudf-internal scratch; `pool_used_bytes` still does once the first copy had a resource. F56's `gpu_alloc_tracking()` wraps whatever is current, so it sits on top of the pool and still counts every allocation cuplyr makes.
- Load-time behaviour is tested in `callr` subprocesses that `pkgload::load_all(compile = FALSE)` the worktree.

### `gpu_sync()` (F57)
Exported; waits until all GPU work queued on the device has finished (`cudaDeviceSynchronize()` in `gpu_sync_device()`), returns `invisible(NULL)`. Verbs return when their work is queued, so a timing needs it (`bench/rolling-lib.R::rw_gpu_finish()` calls it). `collect()` and `print()` already wait (they copy to the host).

### Eager chains (F57)
A few eager verbs push several nodes (a window `mutate()` is window -> mutate -> select; `mutate(.keep = / .before = / .after =)` and `transmute()` are mutate -> select). `push_chain(.data, build)` (`R/execute.R`) hands `build()` a lazy view of `.data`, so its `push_op(carry = FALSE, input_schema = <previous push's $schema>)` calls only do schema work, then lowers the resulting chain with ONE `lower_and_execute()` (no optimizer) into one result table, and carries the frame metadata once. A lazy `.data` just runs `build()`. The error wrapper names the chain (`"window+mutate+select"`). A plain `mutate()` that cannot need a select (`.keep = "all"`, no `.before`/`.after`, nothing dropped or staged: `mutate_needs_select()`) is a single `push_op()`. `retry_on_sum_overflow()` wraps the whole chain. Per-call R cost matters at small sizes (about 90% of a 1e3-row eager mutate is R): `set_union()`/`set_diff()`/`set_intersect()` (`R/utils.R`) replace base/dplyr `union`/`setdiff`/`intersect` on plain vectors (base R 4.5+ probes classes with `tryCatch` on every call); `ir_type_memo_set()`/`ir_infer_type_memo()` reuse the type `ir_parse_quo()` just inferred; `.subset2(x, "field")`, not `x$field`, in per-call code (`$.tbl_gpu` walks the caller's environments).

### Grouping Behavior
- `group_by()` only sets metadata (`$groups`), no GPU work
- Groups are applied during `summarise()`
- `arrange(.by_group=TRUE)` prepends group columns to sort keys
- Preserve `$groups` when returning new tbl_gpu unless ungrouping
- On a rowwise table `$groups` holds the rowwise vars, not groups: see "Rowwise tables"

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

### Rowwise tables (F49)

`rowwise(<tbl_gpu>)` is native and metadata-only (`R/rowwise.R`): it marks the
table, no GPU work, no download, a lazy table keeps its pending ops.

**Representation.** `$groups` holds the rowwise vars (possibly `character()`),
and the object carries `$rowwise = TRUE` -- a field that exists ONLY on rowwise
tables (`new_tbl_gpu(rowwise = TRUE)`; the conformance harness recognises an
uploaded fixture by its fields, so no other table may gain a field).
`is_rowwise_gpu(x)` tests it. `push_op()`/`push_join()` carry it (except through
summarise/reframe/distinct nodes), `compute()`/`as_lazy()`/`vec_restore()` keep
it, `ungrouped()` drops it and `with_surviving_groups()`/`set_groups_by_position()`
put it back. A rowwise table is never a bare data.frame. `tbl_gpu(<rowwise_df>)`
uploads one silently; `collect()` returns `dplyr::rowwise(out, all_of(vars))`
(identical to dplyr's); `as_tibble()`/`ungroup()` drop it; `group_by()` replaces
it (`.add = TRUE` appends to the vars); the `group_data()` family has one group
per row (`rowwise_group_data()`). `tbl_sum()`/`print()` show `Rowwise: <vars>`.
The conformance harness's `cf_collect()` (`tools/conformance-lib.R`) reads the
marker; that is its one rowwise-specific line.

**THE rule.** `$groups` is full of vars that are not groups, so any code that
reads it as groups would compute per rowwise-var group, silently wrong. A
`tbl_gpu` method therefore is EITHER rowwise-aware (table below) OR its first
statement routes the WHOLE call to dplyr on a CPU `rowwise_df`
(`rowwise_cpu(.data, "<verb>")`: rebuilds the call from the method's call,
evaluates it in the caller's environment, goes through `gpu_fallback()`).
`gpu_fallback()` itself builds `dplyr::rowwise(tbl, all_of(vars))` for a rowwise
input and re-uploads a `rowwise_df` result as a rowwise `tbl_gpu`. Adding a verb
that reads `$groups`? Add it to this table and guard it.

| verb | on a rowwise `tbl_gpu` |
|---|---|
| `rowwise`, `group_by`, `ungroup`, `group_vars`/`groups`/`tbl_vars`, the `group_data`/`group_keys`/`group_indices`/`group_size`/`group_rows`/`n_groups` family, `collect`/`compute`/`collapse`/`as_lazy`/`as_eager`, `as_tibble`, `print`/`tbl_sum`, `pull`, `glimpse`, `dim`/`names`, `group_trim` (returns the table: a rowwise table has no groups) | native |
| `arrange`, `rename`, `rename_with`, `select`, `relocate`, `head`/`tail`/`[`, `$<-`/`[[<-`/`[<-` (whole columns only, via `gpu_assign_column()`; a row subscript errors), `dplyr_row_slice`/`dplyr_col_modify`, the joins (`left/inner/full/right/semi/anti/cross_join`: `push_join()` keeps x's vars) | native, rowwise-ness kept (`select`/`relocate`/`[`/`head`/`tail`/`mutate`/`transmute` list the vars in column order like dplyr: `rowwise_column_order()`) |
| `mutate`, `transmute`, `filter`, `filter_out`, `summarise` | native for elementwise expressions and the rewritten row aggregates (below); the WHOLE call goes to dplyr for anything else |
| `slice*`, `sample_*`, `distinct`/`unique`, `count`/`tally`/`add_count`, `reframe`, `group_split`, the set operations, `rows_*`, `bind_rows`/`bind_cols`/`rbind`/`cbind` with a rowwise FIRST input, `group_map`/`group_modify`/`group_nest`/`nest_by`/`nest_join`/`do` | whole call on the CPU |

**Native mutate/transmute/filter/summarise.** Inside `rowwise()` every column
is one value per row, so an elementwise expression is the vectorised result: the
ordinary native path runs with `verb_groups = character()` (the vars are only
columns to keep: `mutate_run(rowwise = TRUE)`, `transmute_run(rowwise = TRUE)`,
`filter_run`). The attempt is all-or-nothing (`rowwise_attempt()`): `.by`,
unnamed rewritten dots, a window call (it would see one row), a vector literal, a
column-free call that is not pure (`runif(1)`), a CPU-fallback request inside the
native code (`gpu_fallback()` bails while an attempt runs), or ANY error, abandons
the attempt and reruns the whole call in dplyr with the user's original dots, so
dplyr gives its own result or its own error. `summarise()` is `transmute()` plus
dplyr's `.groups` rules (NULL/"keep" group by the vars, message only when
`summarise_should_inform()`; "drop"; "rowwise"; "drop_last" is dplyr's error).
`summarise(.groups = "rowwise")` on a plain table returns a rowwise `tbl_gpu`.
`cur_group_id()` is a row-number column (`rowwise_with_row_ids()`).

**Tier B: horizontal aggregates.** `rowwise_rewrite_dots()` rewrites, in the
user's expression, `sum`/`mean`/`min`/`max`/`prod` of `c_across(<tidyselect>)`
(resolved against the running schema minus the rowwise vars), of `c(<columns>)`,
of several column arguments or of one column, `n()` (1L) and
`length(c_across(..))`/`length(c(..))` (k, an IR literal whose `$reads` lists the
operand columns so `ir_cols()` and `.keep = "used"/"unused"` count them) into
`.cuplyr_rw(<IR node>, <cols>)` calls (`ir_parse_expr()` returns the node). The node is `rw_agg_i32`/`rw_agg_f64`
(`src/expr_eval.hpp`, a handler that folds the operand columns with `na.rm`
semantics: NA propagation, `sum`/`prod` neutral elements, `mean` over the non-NA
count with a real NaN for none, NULL_MIN/NULL_MAX for `min`/`max`; FLOAT64
operands: under `na.rm = TRUE` a NaN is missing too (`nans_to_nulls`), under
`na.rm = FALSE` `min`/`max` give NaN for a NaN operand unless an NA is present,
in either operand order). Only
logical/integer/double operands; `mean(a, b)` (trim) is not rewritten. Result
types are dplyr's: `mean`/`prod` and anything with a double operand give a
double; `sum`/`min`/`max` of logical/integer give an integer. Two are
data-dependent, and the handler flags them on the result pointer (`rw_flags`,
read by `check_rowwise_flags()` in `warn_int_overflow()`): an integer `sum()`
that leaves the integer range is a double (eager: probed on the input table
first and built as `rw_agg_f64`; lazy: the INT32 form, failing loudly with
`cuplyr_error_sum_overflow`, F42's rule), and an integer `min()`/`max(na.rm =
TRUE)` of an all-NA row is `+-Inf` (a double, with dplyr's warning; eager probes
and retypes, lazy has no native form: whole call on the CPU). A double
`min()`/`max(na.rm = TRUE)` of an all-NA row warns "no non-missing arguments to
min; returning Inf" at lowering.

### Join Ordering & Unmatched Rows
- **cuDF join outputs are unordered**: We stable-sort join maps by left_map (then right_map) in `src/ops_join.cpp` to match dplyr
- **JoinNoMatch sentinel**: cuDF uses sentinel values; gather treats negatives as wraparound
- **Device-side join-map pipeline** (Phase 7 J4, `build_join_result()` in
  `src/ops_join.cpp`): sanitize join maps on device (replace both `< 0`
  *and* `>= nrows` sentinel values with `nrows` — the `>= nrows` half was a
  latent gap in the original CPU-side sanitize, fixed in J4) → stable sort
  → `multiple=` device filtering (Phase 7 J5, skipped entirely for the
  default `multiple = "all"`) → diagnostics (`join_map_stats()`:
  `left_multi_first`/`right_multi_first`/`left_unmatched_first`/
  `right_unmatched_first`, scalars only, on the filtered maps) → gather with `out_of_bounds_policy::NULLIFY` + coalesce. Every
  mutating join type (`left`/`inner`/`full`/`right`) shares this one
  pipeline; C++ exports return `list(ptr, diag)`, and `lower_join()`
  (`R/lower.R`) calls `check_join_cardinality()` (`R/join.R`) with the
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
- Comparing a number/logical with a string (`filter(mpg > "a")`, `s == 2`) is dplyr's character comparison, not an error: `ir_check_type_operands()` (F48) sends it to the CPU (`check_filter_comparison_types()` is gone)
- `top_n()`/`top_frac()` reach `filter()` as a `top_n_rank(n, wt)` dot; `rewrite_top_n_rank_dots()` (`R/filter.R`) rewrites it to `min_rank(desc(wt)) <= n` / `min_rank(wt) <= abs(n)` for a constant `n` or `c * n()` (CPU fallbacks keep the original dots). Rank-family calls with one `desc(e)` argument parse natively (`meta$rank_desc` -> the window spec's `order_desc`)
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
gpu_memory_snapshot()      # Device-level memory state (a pool hides it; see "Memory resource")
gpu_alloc_snapshot()       # Allocator-level state (pool_used_bytes), what memory tests should use
run_with_pool_off(fn)      # Run fn in a subprocess with the memory pool off (device-level claims)
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
