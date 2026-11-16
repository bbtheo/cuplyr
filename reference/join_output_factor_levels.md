# Propagate `factor_levels` through a join's output-name mapping

`build_join_schema()`/[`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)
compute each output column's post-suffix NAME and which side's SOURCE
column it came from (`origin`/`source_names`, parallel to `names`) –
this reattaches `factor_levels`, keyed by that post-suffix output name,
by looking up each output column's own origin schema's `factor_levels`
under its PRE-suffix source name. This is the Bug 1 fix (Phase 11 L2):
before this, `build_join_schema()` returned only `names`/`types`, so
every mutating join
([`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
and
[`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html))
silently dropped factor metadata entirely –
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) would
then present a factor column as raw integer codes.

## Usage

``` r
join_output_factor_levels(
  left_schema,
  right_schema,
  out_names,
  origin,
  source_names
)
```

## Details

A coalesced equi-join KEY column (kept unsuffixed under `keep = FALSE`,
see
[`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)'s
`is_eq` branch) always has `origin == "left"` here (the right-side copy
of that key is dropped, never part of `out_names`) – correct, since the
join's native gather always keeps the LEFT key column's own physical
data for a coalesced key (see `src/ops_join.cpp`), so its levels are
always the left side's.
[`join_factor_key_levels_conflict()`](https://bbtheo.github.io/cuplyr/reference/join_factor_key_levels_conflict.md)
(below) is the correctness guard for the case this alone can't resolve:
an equi key that's factor-typed on BOTH sides with DIFFERENT level sets
would still coalesce to "the left side's levels" here, but the native
GPU join itself compares the two sides' raw INT32 codes directly – which
would silently mismatch two logically-equal labels encoded under
different codes. That case is caught and routed to the CPU fallback
before ever reaching this function (every join verb calls it right after
[`resolve_join_key_types()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_key_types.md)).
