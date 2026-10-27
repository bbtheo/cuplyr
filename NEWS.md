# cuplyr (development version)

## Lazy plans keep dplyr's integer-overflow warning (F70)

* In lazy mode the optimizer no longer changes which rows or columns an integer
  `+`, `-` or `*` (both operands integer or logical) is computed on, so the
  warning "NAs produced by integer overflow" now appears in the same pipelines
  as in eager mode. Before, a lazy plan could lose the warning (a filter moved
  below the `mutate()` holding the arithmetic, or the column was dropped before
  it was computed) or give one dplyr does not (a `filter()` merged with the one
  before it, or a predicate moved into a join input that holds rows the join
  drops). Values were always the same. Dead-column pruning keeps such an
  expression, filter pushdown does not move a filter below a `mutate()` holding
  one, a filter holding one is not merged with the filters below it, and such a
  predicate moves into a join only into the left input of a `left_join()` or the
  right input of a `right_join()`. When such a predicate cannot move, the whole
  `filter()` call stays above the join (dplyr computes all of a call's
  predicates on the same rows), including its other predicates. A node with a
  null literal operand (`a + NA`) never overflows and does not count. Plans for
  pipelines without integer arithmetic are unchanged.

## Docs: query-optimization vignette matches the optimizer (F69)

* `vignette("query-optimization", package = "cuplyr")` is rewritten against the
  optimizer's code, with `explain()` output you can reproduce. It describes the
  five passes as they are (no filter fusion pass; mutate fusion keeps written
  order and has two limits; a filter moves at most one node per optimization),
  the nine barrier node types, when a filter moves into a join (never for an
  `==` `inner_join()`, `left_join()` or `right_join()` called from your own
  code with `relationship` unset and `multiple = "all"`, the default; those
  joins with other conditions, `by = character()` or called from inside a
  package are recorded as many-to-many, and the pushdown rules for many-to-many
  joins apply to them), and when a pass is skipped because it cannot change the
  plan. It makes no speed claims; `vignette("performance", package = "cuplyr")`
  has the measurements.

## Faster eager verb calls (F68b)

* An eager verb call takes 36% (`arrange()`) to 82% (`select()`) less time,
  and recording a verb on a lazy table is cheaper by the same R-side work. At
  1e3 rows, where a verb is a few small kernels and nearly all of its time is
  R, an eager call takes (median of three runs, before -> after, ms)
  `filter()` 0.73 -> 0.36, `mutate()` 0.83 -> 0.37, `select()` 0.57 -> 0.10,
  `arrange()` 0.22 -> 0.14 and a grouped `summarise()` 0.81 -> 0.40. Adding a
  `filter()` or `mutate()` to a lazy table takes 0.16 to 0.19 ms (it was 0.5).
  Results, errors, warnings and messages, and the moment they are raised (at
  the verb, eager or lazy), are unchanged; nothing is cached across calls. The
  one exception: a dot named like an argument of `rlang::enquos()` that
  `rlang::quos()` lacks (`.homonyms`, `.check_assign`, `.ignore_null`) is now an
  ordinary named expression in `filter()`, `filter_out()`, `mutate()` and
  `summarise()`, as in dplyr, where it used to be taken as that argument.
* `select()` with plain column names skips tidyselect (anything else, and every
  error, warning and message, is still tidyselect's). Expression parsing and
  type checks skip the walks they cannot need (no timestamp, STRING or factor
  among the operands, no `across()`/`cur_group*()` in the dots), `summarise()`
  of `fn(column)` goes straight to its aggregation, and `filter()`,
  `filter_out()`, `mutate()` and `summarise()` capture their dots with `quos()`
  (the same quosures at a seventh of the cost).
* `collect()` and `compute()` of a lazy table: the optimizer recognises in one
  walk a plan its column passes would leave as it is (a chain of `filter()` and
  `mutate()` steps on the table itself, below any barrier such as `summarise()`
  or `arrange()`, whose columns are all used) and skips those passes; the plan
  it returns is the same. The five-verb pipeline of `vignette("performance", package =
  "cuplyr")` (median of three runs) takes 1.87 ms eager and 1.75 ms lazy at 1e3
  rows (it was 3.10 and 3.00) and 2.00 ms in both modes at 1e6 rows (3.38 and
  3.20). Recording a join on a deep lazy plan grows half as fast with the
  plan's depth (0.61 ms at 120 nodes against 0.30 at 3; it was 0.87).

## Faster lazy verb calls (F68a)

* Recording a verb on a lazy table no longer depends on how many verbs are
  already pending. It used to infer the schema of the whole pending plan, and
  more than once per verb; it now reads the schema the table stores. Adding a
  `filter()` or `mutate()` to a lazy table cost 0.6 ms on a 1-verb plan,
  16.5 ms on 50 and 64 ms on 100, and now costs 0.5 ms at every length.
  Recording a join on a very deep plan still grows with the plan's depth
  (0.9 ms at 120 nodes against 0.35 ms at 3).
* `collect()` and `compute()` take time in proportion to the plan's length: the
  optimizer passes and the lowering infer each node's schema once per call
  instead of walking the plan below every node, and the optimizer skips the
  passes that cannot apply to a plan. `collect()` of a 100-verb `mutate()` plan
  takes 9.5 ms instead of 53.
* Verb-time errors and warnings are unchanged: every check still runs when the
  verb is called, lazily as eagerly.
* The five-verb pipeline of `vignette("performance", package = "cuplyr")` (data
  on the GPU, result collected, median of three runs) now takes 3.00 ms lazy
  against 3.10 eager at 1e3 rows (it was 4.14 against 3.10) and 3.20 against
  3.38 ms at 1e6 rows (4.25 against 3.25). At 1e7 rows lazy is still slower, 6.20
  against 5.00 ms (it was 7.20 against 4.83), and at 5e7 rows 19.5 against 18.0
  ms (it was 20.5 against 17.5). In the vignette's optimizer case lazy is now
  faster than eager at 1e7 rows (6.40 against 7.20 ms). Recording the five verbs
  takes 2.07 ms (it was 2.7). The vignette's tables, text and dates are
  re-measured. The default execution mode is unchanged (eager).

## gibble: the user-facing name for GPU tables (F51)

* cuplyr's GPU tables are called gibbles. `print()` now starts with tibble's
  own header with the noun changed (`# A gibble:` and the dimensions),
  followed by `# Groups:` or `# Rowwise:` lines, instead of the
  `Rows:`/`Columns:`/`Groups:` lines. The header comes from pillar's own
  `tbl_format_header()`, so it wraps at the print width. The width follows
  pillar's order: `print(width =)`, then `options(pillar.width)` (then
  `tibble.width`, `dplyr.width`), then `getOption("width")`. The
  `$ col <type> values` body is unchanged. Its value truncation follows
  `glimpse()`'s rule: `print(width =)` as given, otherwise the width option
  only if it is finite, so `options(pillar.width = Inf)` or
  `options(tibble.width = Inf)` still cuts the body at `getOption("width")`.
* The group count in the header of a grouped gibble, and `n_groups()`, no
  longer build `group_data()` in most cases. A table with fixed group keys
  (e.g. `filter()` or `slice()` with `.preserve = TRUE`, or `filter()` of a
  `.drop = FALSE` table) already knows its groups, so for an eager table the
  count costs no GPU work (a lazy table with pending operations runs its plan
  first). Otherwise one GPU groupby returns just the count, without the row
  indices. That groupby still sorts every key, so its cost grows with the row
  count and the key width: about 4 ms for 2e6 integer groups, about 0.2 s for
  1e7 rows with 1e6 string groups (against about 1 ms for an ungrouped print).
  `group_data()` is still built for a `.drop = FALSE` table with a factor key
  and no fixed keys, since unused levels count as groups there.
* A lazy table with pending operations prints `# A gibble [lazy, not
  materialized]`, its result schema, `# Operations pending: <k>` and, for a
  grouped or rowwise table, a `# Groups:` or `# Rowwise:` line with the
  variables (no group count: that would need the plan). It never runs the plan.
  `k` is now the number of pending plan steps (AST nodes: a `collapse()`
  barrier or a multi-step verb counts each node); before, a lazy join printed
  the number of fields in its plan node (13) and a lazy table with a source
  printed the source's dimensions with `[preview unavailable]` columns.
* New exported aliases `gibble()`, `as_gibble()` and `is_gibble()` for
  `tbl_gpu()`, `as_tbl_gpu()` and `is_tbl_gpu()` (the same function objects).
  The class is still `c("tbl_gpu", "tbl")` and `pillar::tbl_sum()` still
  returns dplyr's `"A tibble"` values.
* The CRAN package gibble (geometry decomposition) also exports `gibble()`.
  When both packages are attached the one attached last masks the other;
  `cuplyr::gibble()` always works. cuplyr defines no `gibble.<class>` methods,
  so the two cannot dispatch into each other.

## Docs: eager vs lazy, performance vignette (F62)

* New `vignette("performance", package = "cuplyr")` with measured timings
  (RTX 5070, 1e3 to 5e7 rows) for a five-verb pipeline run eager and lazy, the
  per-verb call floor, transfer costs, and the memory each mode leaves
  allocated. It says plainly that with clean GPU memory lazy is slower than
  eager at every size measured: about 1 ms per pipeline up to 1e6 rows, 2.4 ms
  at 1e7 and 3 ms at 5e7. It explains why (each lazy verb still does its R-side
  checks, `collect()` adds about 1 ms to optimize and lower the plan, and lazy
  loses the overlap of one verb's GPU work with R preparing the next). Lazy's
  gain is that intermediates are freed when `collect()` or `compute()` returns
  (5 back-to-back runs at 5e7 rows: 68 to 155 ms each eager, 20 to 24 ms each
  lazy). It also lists where the GPU loses to dplyr and how to see CPU fallbacks
  (`options(cuplyr.fallback = "warn")`).
* The README leads with a lazy example that runs as written, and its "Lazy
  evaluation" section is now "Eager vs lazy". The default execution mode is
  unchanged (eager).
* The README lists the optimizer's five passes in the order they run.
  `vignette("getting-started")`, `vignette("query-optimization")` and
  `?tbl_gpu` link to the new vignette.

## async memory resource by default (F65)

* The default memory resource is now `"async"` (RMM's `cuda_async_memory_resource`,
  CUDA's stream-ordered allocator) instead of the RMM pool. By default it caches
  freed memory for reuse like the pool does and ran at the same speed in the
  measurements of `vignette("memory", package = "cuplyr")`, but unlike the pool
  it can give memory back to the device: `gpu_gc()` returns everything the pool
  holds unused. The pool kept its high-water mark until R exited, which starved
  other processes on the card.
* `cuplyr.release_threshold` / `CUPLYR_RELEASE_THRESHOLD` (bytes, async only,
  unset by default) makes CUDA return unused memory by itself at the next
  synchronisation while the pool's whole reservation, tables in use included,
  is above the threshold; `0` keeps nothing. The price is speed: a session that
  holds more than the threshold allocates from the driver again, at about plain
  `cudaMalloc` speed.
* To get the pool back: `options(cuplyr.memory_resource = "pool")` or the
  environment variable `CUPLYR_MEMORY_RESOURCE=pool`, set before loading cuplyr
  (one resource per R process). `cuplyr.memory_pool = FALSE` /
  `CUPLYR_MEMORY_POOL=0`, or `cuplyr.memory_resource = "cuda"` (alias `"none"`),
  still means no resource (plain `cudaMalloc`).
* An unknown value of `cuplyr.memory_resource` / `CUPLYR_MEMORY_RESOURCE` now
  warns, naming the option or variable and the value, and uses async (it used to
  mean the pool silently); with `cuplyr.memory_pool = FALSE` it warns that the
  value is ignored. A zero-length or `NA` `cuplyr.memory_pool` option counts as
  unset instead of breaking the load. Where `cudaMallocAsync` is unsupported,
  the default falls back to the pool with a warning.
* `cuplyr.pool_initial` only sets how much the resource reserves at load, under
  both kinds (`0` reserves nothing); it is not a limit. `gpu_gc()` gains
  `released_bytes`, the drop in `pool_bytes` across the call (0 under the pool,
  `NA` without a resource), and no longer fails on a CUDA error while returning
  memory (it warns).

## roll_median() is exact by default (F66)

* `roll_median()` now uses the built-in exact engine by default, which
  returns exactly `stats::median()` of each window converted to double
  (roughly 0.2 s per 1e6 rows at any width). Before, it used `data.table::frollmedian()`
  whenever data.table was installed, which can differ from R by one rounding
  step on windows with an even number of non-missing values although the docs
  said the result was exact.
* `options(cuplyr.roll_median = "frollmedian")` (or the environment variable
  `CUPLYR_ROLL_MEDIAN`) opts in to the faster `data.table::frollmedian()` path
  (data.table >= 1.18.4). Positions where data.table returns `Inf` or `-Inf`
  (including a midpoint of two huge finite values that overflows) take the
  exact engine's value. If it is requested but unavailable, cuplyr uses the
  exact engine and says so once per session. Other values are an error.

## Intermediates freed on return, OOM retry (F58)

* A cuplyr call you make (an eager verb, `compute()` or `collect()` of a lazy
  plan, `bind_rows()`) now frees the temporary GPU tables its own steps built
  (cast copies, temp columns, the tables its steps hand to each other, the
  upload of a `data.frame` argument, a lazy input it had to compute) when it
  returns, instead of leaving them to R's garbage collector. Ownership follows
  who called: what a call returns is yours and is never freed early, including
  a table your own code creates (or caches) while a verb evaluates its
  arguments, expressions or predicates. A table you create and drop yourself
  (`group_by(a, g = x %% 3)` passed straight into `summarise()`) still waits
  for R's collector, as does the inner result of a verb dplyr builds from
  several verb calls (`add_tally(sort = TRUE)` runs `mutate()`, then
  `arrange()`).
* A device out-of-memory failure in a GPU step (a verb's lowering, an upload,
  `compute()`, `collect()`, the casts/alignment of `bind_rows()` and set
  operations, `slice_sample()`'s gather) is retried once after `gc()` and
  `gpu_gc()`. A verb made of several steps retries the failed step, not the
  whole verb. If it fails again, cuplyr raises an error of class
  `cuplyr_error_gpu_oom` that says a retry after garbage collection was
  attempted and includes the original error text.
* Errors the GPU layer raised (C++ exceptions, plain `stop()`) in an eager verb
  are still worded "GPU operation 'x' failed", and `compute()`/`collect()` of
  a lazy plan raise them unworded, as before. A condition cuplyr's R code
  raised with its own class is no longer reworded by the wrappers.

## Perf foundations: memory pool, shared columns, `gpu_sync()` (F57)

* Loading cuplyr installs an RMM memory pool, so the many short-lived columns of
  a pipeline are carved out of memory reserved once instead of going through
  `cudaMalloc`/`cudaFree`. `options(cuplyr.memory_pool = FALSE)` or
  `CUPLYR_MEMORY_POOL=0` switches it off (the behaviour before this change);
  `cuplyr.pool_initial` / `CUPLYR_POOL_INITIAL` sets the initial size in bytes.
  Both are read once, at load; see `?gpu_memory_state`.
* `gpu_memory_state()` gains `resource`, `pool_bytes` and `pool_used_bytes`.
  `used_bytes` stays device-level and includes the pages the pool holds, so it
  does not fall when tables are freed; `pool_used_bytes` is what is allocated
  from the pool. `gpu_gc()$freed_bytes` is the drop in `pool_used_bytes` while a
  pool is installed.
* New `gpu_sync()` waits until all GPU work queued by cuplyr has finished, for
  timing code that leaves its result on the GPU.
* Verbs that pass a column through unchanged (`select()`, `relocate()`,
  `rename()`, the untouched columns of `mutate()`/`transmute()`, a bare copy
  `a = x`, `bind_cols()`, the input columns of a window `mutate()`) share it with
  the input table instead of deep-copying it. `mutate(r = f(x))` allocates only
  `r`.
* An eager window `mutate()` (and `mutate()` with `.keep`/`.before`/`.after`,
  `transmute()`) runs as one execution and builds one table instead of one per
  node. The per-call overhead of an eager `mutate()` on a small table is about
  25% lower.

## rowwise() on the GPU (F49)

* `rowwise()` of a `tbl_gpu` is now native and metadata-only: it returns a
  rowwise `tbl_gpu` (no download; a lazy table stays lazy). `collect()`
  returns dplyr's `rowwise_df`, `as_tibble()` and `ungroup()` drop the
  rowwise-ness, `group_by()` replaces it, and `tbl_gpu(<rowwise_df>)` uploads a
  rowwise table silently (it used to upload ungrouped, with a warning).
  `summarise(.groups = "rowwise")` returns a rowwise `tbl_gpu`.
* `mutate()`, `transmute()`, `filter()`, `filter_out()` and `summarise()` on a
  rowwise table run on the GPU for elementwise expressions, and for the row
  aggregates `sum()`/`mean()`/`min()`/`max()`/`prod()` of `c_across(...)`,
  of `c(a, b, ...)` or of several columns, `n()`, `cur_group_id()` and
  `length(c_across(...))`, with dplyr's NA rules and result types. A 1e5-row
  `rowwise() |> mutate(s = sum(c_across(a:e)))` takes milliseconds instead of
  about twenty seconds. Anything else on a rowwise table (an R function per
  row, list-columns, window functions, `.by`, the verbs listed in the
  architecture notes) runs the whole call in dplyr on the CPU and comes back
  as a rowwise `tbl_gpu`.
* An integer `sum()` of a row that leaves the integer range is a double, as in
  dplyr, on an eager table. A lazy plan fails with `cuplyr_error_sum_overflow`
  instead (its schema is fixed), and an integer `min()`/`max(na.rm = TRUE)` over
  columns in a lazy plan runs the call on the CPU.
* `nest_join()` with a grouped or rowwise `tbl_gpu` as `y` nests pieces of
  `y`'s own class, as dplyr does.
* Whole-column `[<-` now works on every `tbl_gpu`: `x[j] <- value` and
  `x[, j] <- value` add, replace or (with `NULL`) drop columns as on a tibble,
  keeping the groups (a rowwise table's vars in column order, as dplyr's
  `[<-.rowwise_df`). Cell assignment (`x[i, j] <- value`, `x[] <- value`, a
  matrix subscript) still errors, pointing at `mutate()`.
* `is.na()` and `anyNA()` of a whole `tbl_gpu` now error. They used to answer
  about the object's internal fields (all `FALSE`), so `x[is.na(x)] <- 0` did
  nothing; use `mutate(across(everything(), ~ coalesce(.x, 0)))`.
* When a rowwise verb's native attempt fails and the call reruns in dplyr, the
  fallback notification (`options(cuplyr.fallback = "warn")`) includes the
  error the attempt raised.

## group_nest() and nest_by() `.drop` (F28)

* `group_nest()` and `nest_by()` on a `tbl_gpu` now pass `...` through to
  `group_by()` unchanged, so `.drop = TRUE` drops empty factor levels as in
  dplyr (previously ignored).

## R integer arithmetic semantics (F42)

* Integer `+`, `-` and `*` (on integer or logical columns and literals) give
  `NA` with R's warning "NAs produced by integer overflow" when the result
  leaves +-`.Machine$integer.max`, in `mutate()`, `transmute()`, `filter()`
  and `summarise()` sub-expressions. They used to wrap around silently. The
  warning is raised once per verb call: at the verb on an eager table, at
  `collect()` on a lazy one (except `summarise()` temp columns, which are
  computed at `summarise()`).
* Integer `%/%` and `%%` by zero give `NA`, as in R.
* `cumsum()` of an integer or logical column no longer wraps around: from its
  first overflow on, every row of the group is `NA`, with R's "integer
  overflow in 'cumsum'" warning. `cummean()` of such a column accumulates in
  double, as in dplyr, instead of returning wrapped values.
* A window `sum()` of an integer or logical column (`mutate(s = sum(x))`,
  `filter(x * sum(x) > 0L)`, `summarise(m = max(x * sum(x)))`) is now an
  integer, as in dplyr, so arithmetic on it is checked integer arithmetic.
  It used to be a double. When a sum leaves R's integer range, dplyr makes that
  sum a double but keeps the other groups' sums integers. cuplyr gets the
  same result by rerunning the call on the CPU: always for an eager table,
  and for `summarise()` on a lazy table too.
* New lazy errors, of class `cuplyr_error_sum_overflow` (the F34 rule for a
  fixed lazy schema): a lazy `mutate()` window `sum()` that overflows, and a
  lazy `filter()` whose overflowing sum sits inside another window call
  (`filter(x * 1000L > mean(sum(x)))`), which used to return dplyr's rows.
  Use `sum(as.numeric(x))` or eager mode. A lazy `filter()` on a top-level
  sum that overflows still compares the double value, but if another group's
  in-range sum then meets overflowing integer arithmetic, that row is not
  dropped (known limitation).
* Checked integer arithmetic costs more than the old wrapping arithmetic:
  each group of up to four `+`/`-`/`*` nodes takes a second pass for the
  range test and a reduction. At 5e6 rows, `x + y` takes about 1.6 ms
  (1.2 ms before), a three-node expression 2.9 ms (1.6 ms) and a filter on
  one 2.1 ms (1.4 ms). Double arithmetic is unchanged.

## dplyr generic registrations (F05)

* `show_query()` is now a `tbl_gpu` method for dplyr's generic (it was a
  plain function masking it), so `dplyr::show_query(gt)` works. The argument
  is `x` rather than `.data`, and `x` is returned invisibly (it used to return
  the pending AST). cuplyr re-exports dplyr's generic.
* New `explain()`, `same_src()` and `auto_copy()` methods for `tbl_gpu`.
  `explain()` prints the pending operations and, for a lazy table with
  pending operations, the optimized plan `compute()` would run; nothing is
  executed.
* A GPU join (including `cross_join()`) with a non-GPU `y` and `copy = FALSE`
  now raises dplyr's "`x` and `y` must share the same src" error instead of
  a cuplyr-specific message. With `copy = TRUE`, `y` is uploaded after
  `as.data.frame()`, so a plain list works as in dplyr.

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

* `mutate()` no longer silently promotes integer arithmetic to `FLOAT64`. `+`, `-`, `*`, `%%`, and `%/%` now preserve `INT32`/`INT64` when every input column and literal involved is integer or logical (matching R/dplyr, e.g. `TRUE + TRUE` is `2L`). `INT32` results that leave R's integer range are `NA` with a warning, as in R (F42; they wrapped around before F42). `/` and `^` continue to always promote to `FLOAT64` (matching R: `1L / 2L` and `2L ^ 2L` are both doubles).

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

* `show_query()` displays the pending operation tree for debugging (since F05 it is a method for dplyr's generic and returns its input invisibly).

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
