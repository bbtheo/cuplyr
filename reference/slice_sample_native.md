# GPU-native `slice_sample()` (Phase 5, task W8)

Reproduces dplyr's own RNG stream bit-for-bit by generating row indices
on the CPU with the exact same
[`sample.int()`](https://rdrr.io/r/base/sample.html) call sequence real
dplyr makes, then gathering those indices on the GPU via the existing
`gpu_slice_indices()` primitive (Phase 3.2) – there is no AST node for
[`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html) at
all, unlike every other verb in this package: the index vector is the
whole point (it must be generated on the CPU to consume R's global RNG
state the same way real dplyr does), so there is nothing left for a lazy
AST/optimizer to defer.

## Usage

``` r
slice_sample_native(.data, gm, amount, is_prop, replace, fallback)
```

## Arguments

- .data:

  A `tbl_gpu` object (already materialized if it had pending lazy ops –
  the caller is responsible for calling
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)
  first, since CPU-side index generation needs a real row
  count/grouping, which a lazy AST doesn't have)

- gm:

  A `list(mode, group_cols)` as returned by
  [`slice_group_mode()`](https://bbtheo.github.io/cuplyr/reference/slice_group_mode.md)

- amount, is_prop:

  As resolved by `resolve_slice_amount()`

- replace:

  Logical scalar,
  [`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html)'s
  own `replace=`

- fallback:

  `function(tbl)` running dplyr's
  [`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html)
  on the CPU. Used when a group is empty and `replace = TRUE` still asks
  for rows: `sample.int(0, size, replace = TRUE)` fails, and dplyr
  raises that from slice(), after drawing the groups before it, so dplyr
  runs it.

## Value

A new (always eager) `tbl_gpu`, or `fallback()`'s result

## Details

Semantics verified empirically against dplyr 1.2.1 before implementing
(`dplyr:::slice_sample.data.frame`, `dplyr:::sample_int`,
`dplyr:::get_slice_size`, `dplyr:::compute_by_groups`, source read via
[`getAnywhere()`](https://rdrr.io/r/utils/getAnywhere.html)/[`print()`](https://rdrr.io/r/base/print.html)):

- Ungrouped: exactly one `sample.int(nrow, size, replace = replace)`
  call (`dplyr:::sample_int`'s own body:
  `if (size == 0L) integer(0) else sample.int(n, size, prob = wt, replace = replace)`,
  `wt` always `NULL` here since `weight_by=` never reaches this function
  – see below).

- [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped:
  dplyr iterates groups in ASCENDING GROUP-KEY order (NA-key group last)
  – the same order
  [`dplyr::group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  itself produces (confirmed by literally calling
  [`dplyr::group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)/[`dplyr::group_rows()`](https://dplyr.tidyverse.org/reference/group_data.html)
  on the collected group columns here, so this can never drift from real
  dplyr's own ordering). One
  [`sample.int()`](https://rdrr.io/r/base/sample.html) call per group,
  in that order, consuming the RNG stream in that same order; the
  resulting rows are concatenated in the SAME order (group-key order),
  matching real dplyr's own output row order exactly (confirmed
  empirically: this is genuinely a reordering relative to input row
  order, same as the other grouped `slice_*()` verbs' W7 finding).
  Output stays grouped by the same columns, matching real dplyr.

- `by=`: dplyr's own `compute_by_groups()` groups via
  [`vctrs::vec_group_loc()`](https://vctrs.r-lib.org/reference/vec_group.html),
  which orders groups by FIRST-APPEARANCE position in the input – NOT
  ascending group-key order (confirmed empirically with a case where
  ascending-key order and first-appearance order differ: the RNG draw
  order AND the output row order both follow first-appearance order, not
  ascending key, and NOT a final sort-by-original-row-position either).
  This CORRECTS `scratchpad/phase5_window_design.md` §7's own text
  ("`by=` =\> sort() final indices (original row order)"), which was
  written before this empirical check and turns out to describe neither
  the RNG order nor the output order dplyr actually uses – the real rule
  is exactly the grouped case's rule, just with a different (first-
  appearance, not ascending-key) group ordering source, i.e. exactly
  analogous to `summarise(.by=)`'s and W7's
  grouped-[`slice()`](https://dplyr.tidyverse.org/reference/slice.html)'s
  already-documented first-appearance rule. Implemented here via
  [`vctrs::vec_group_loc()`](https://vctrs.r-lib.org/reference/vec_group.html)
  directly (the identical function `compute_by_groups()` itself calls),
  rather than reimplementing first-appearance grouping by hand, to
  remove any group-order divergence risk. Output is always ungrouped,
  matching `by=`'s general contract.

- `n=`/`prop=` resolution: see `compute_slice_sample_size()`'s own docs
  for the exact (and slightly asymmetric) `replace=`-dependent clamp
  rule, verified against `dplyr:::get_slice_size()`'s source.
  Oversampling `n` without `replace` does NOT error – it silently clamps
  to the table's (or group's) own row count, same as
  [`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html).

- `weight_by=` always falls back to CPU (see
  [`slice_sample.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_sample.tbl_gpu.md)
  below) –
  [`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)
  re-runs the real
  [`dplyr::slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html)
  call, which already reproduces the identical
  `sample.int(..., prob = wt)` behavior, so there is nothing to gain
  (and real complexity to lose, since `wt` needs the ACTUAL data values,
  not just group columns) from a native path for this case.
