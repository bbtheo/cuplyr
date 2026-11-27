# Run a dplyr verb on the CPU and re-upload the result

The core CPU fallback primitive. Materializes any pending lazy
operations, collects `.data` to a tibble (restoring
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
structure from `.data$groups` so grouped semantics apply on the CPU
exactly as they would on a real `grouped_df`; a rowwise table becomes
dplyr's `rowwise_df` of its rowwise vars), applies `expr_fn` (which
should call the real dplyr verb), and re-uploads the result via
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) (a
`rowwise_df` result goes back up as a rowwise table).

## Usage

``` r
gpu_fallback(
  verb_name,
  .data,
  expr_fn,
  ...,
  as_is = FALSE,
  detail = NULL,
  reason = detail,
  expr = NA_character_,
  also = NULL
)
```

## Arguments

- verb_name:

  Character, the verb name to use in the fallback notification (e.g.
  `"transmute"`, `"slice_min"`), gated by `options(cuplyr.fallback = )`
  (see
  [`cuplyr_fallback_notify()`](https://bbtheo.github.io/cuplyr/reference/cuplyr_fallback_notify.md)).

- .data:

  A `tbl_gpu` object.

- expr_fn:

  A function of one argument (a tibble, already grouped if `.data` was
  grouped) that calls the corresponding real dplyr verb and returns its
  result.

- as_is:

  Logical. If `TRUE`, `expr_fn`'s return value is always returned
  unchanged, never re-uploaded – for a verb that returns metadata
  *about* the grouping/data rather than a transformed dataset, even when
  that metadata happens to be a small data frame (dplyr's own
  grouping-metadata queries, e.g.
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)/[`group_keys()`](https://dplyr.tidyverse.org/reference/group_data.html),
  are GPU-native and don't use it; the parameter is for any fallback
  verb with the same "always a plain R object" shape).

- detail:

  Older spelling of `reason`.

- reason:

  Text saying why the call runs on the CPU: the `reason` of the notice
  ([`cuplyr_fallback_notify()`](https://bbtheo.github.io/cuplyr/reference/cuplyr_fallback_notify.md),
  [cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).
  Every caller gives one; a reason made by
  [`fallback_failed_reason()`](https://bbtheo.github.io/cuplyr/reference/fallback_failed_reason.md)
  or `shadow_screen_text()` carries the offending expression as its
  `"expr"` attribute. On a rowwise table it is prefixed with "rowwise
  table: " (`rowwise_rerun()` passes the reason its native attempt
  recorded).

- expr:

  The offending expression as text, when the whole call is not at fault
  (`NA`: the call as a whole goes).

- also:

  A second `tbl_gpu` (the `y` of a join, set operation or `rows_*()`; or
  a list of them) the fallback collects too: its size is added to the
  notice. It must not have pending operations (compute it first).

## Value

Either a new `tbl_gpu` (grouping, execution mode, and factor levels
restored/round-tripped), or `expr_fn`'s raw return value unchanged when:
`as_is = TRUE`; the result isn't a data frame at all (e.g.
[`group_map()`](https://dplyr.tidyverse.org/reference/group_map.html)'s
list); or the result is a data frame that contains list-columns (e.g.
[`nest_by()`](https://dplyr.tidyverse.org/reference/nest_by.html)/[`group_nest()`](https://dplyr.tidyverse.org/reference/group_nest.html)/
[`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)'s
nested-tibble columns), which have no GPU representation.

When the result is a grouped data frame whose
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
keeps groups with no row that a plain regroup of its data would not have
(dplyr's `dplyr_col_modify.grouped_df()` does this for a verb that
leaves the group columns alone, after
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)/[`slice()`](https://dplyr.tidyverse.org/reference/slice.html)
with `.preserve = TRUE`), the returned `tbl_gpu` carries those keys as
`$fixed_keys` (see `fixed_keys_of()`); verbs that regroup get none.

## Details

The lazy result of a lazy `.data` carries the step's history as its
`origin` field (`fallback_record_origin()`), which
[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) and
[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
print.
