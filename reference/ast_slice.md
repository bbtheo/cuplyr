# Create a slice AST node

One node type covers the whole slice family
([`slice()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html)/
[`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)),
distinguished by `mode` – unlike most other verbs (which each get their
own `ast_*` constructor), these five share the same "select a subset of
rows, in some order, from a single input" shape and (mostly) the same
`n=`/`prop=` sizing rule, so one node with a mode-specific field set
avoids several near-identical constructors/lowering functions/optimizer
barrier entries.
[`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html)
isn't built on this node (see
[`slice_sample_native()`](https://bbtheo.github.io/cuplyr/reference/slice_sample_native.md)
in `R/slice.R`), and a slice the GPU path doesn't cover is run by the
CPU fallback (`R/fallback.R`) instead.

## Usage

``` r
ast_slice(
  input,
  mode,
  amount = NULL,
  is_prop = NULL,
  raw_indices = NULL,
  order_col = NULL,
  order_is_temp = FALSE,
  descending = NULL,
  with_ties = NULL,
  na_rm = NULL,
  bracket_arg = NULL,
  bracket_call = NULL,
  rows_resolved = FALSE,
  top_k = FALSE,
  group_cols = NULL,
  by_first = FALSE
)
```

## Arguments

- input:

  Input AST node

- mode:

  One of `"head"`, `"tail"`, `"index"`, `"rank"`, `"rank_grouped"`,
  `"bracket"` (tibble `x[i, ]` row subscript in `raw_indices`, NA /
  out-of-range give NA rows)

- amount:

  For `mode` `"head"`/`"tail"`/`"rank"`: the raw (unresolved) `n`/`prop`
  value (sign preserved; resolved against the actual row count only at
  lowering time, see `src/ops_common.hpp::compute_slice_size()`)

- is_prop:

  For `mode` `"head"`/`"tail"`/`"rank"`: `TRUE` if `amount` is a `prop`,
  `FALSE` if it's an `n`

- raw_indices:

  For `mode` `"index"`: the unresolved numeric index vector from
  [`slice()`](https://dplyr.tidyverse.org/reference/slice.html)'s dots
  (may contain 0/NA/duplicates/out-of-range/ negative values – fully
  resolved only at lowering time, since negative indices and
  out-of-range dropping both need the actual row count)

- order_col:

  For `mode` `"rank"`: the column name (in `input`'s output schema) to
  rank by – always a real column name, even when `order_by` was a
  computed expression (see `order_is_temp`)

- order_is_temp:

  For `mode` `"rank"`: `TRUE` if `order_col` is a synthetic column added
  by a wrapping `ast_mutate` (computed `order_by`, e.g.
  `slice_min(df, x + y)`) that must NOT appear in the final output –
  mirrors
  [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)'s
  same computed-key pattern (`R/distinct.R`)

- descending:

  For `mode` `"rank"`: `FALSE` for
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html),
  `TRUE` for
  [`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)

- with_ties, na_rm:

  For `mode` `"rank"`: as documented on
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)

- bracket_arg, bracket_call, rows_resolved:

  For `mode` `"bracket"`: `raw_indices` is either already resolved to
  1-based positions against the real row count (`rows_resolved = TRUE`,
  `NA` = all-`NA` row), or still the user's raw subscript, in which case
  `bracket_arg` (the user's subscript expression) and `bracket_call`
  (the `[` call) label any error raised when it is resolved at lowering

- top_k:

  For `mode` `"rank"`: `TRUE` on a node the optimizer made from
  `arrange(x) |> head(n)`
  ([`rewrite_slice_over_arrange()`](https://bbtheo.github.io/cuplyr/reference/rewrite_slice_over_arrange.md));
  the node is then `slice_min(x, n, with_ties = FALSE)` and nothing else

- group_cols, by_first:

  For `mode == "rank_grouped"` (a grouped
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)
  on a numeric or date key, run by `gpu_slice_rank_grouped()`): the
  grouping columns, and whether the groups come out in order of their
  first row (`by=`/`.by=`) rather than sorted by key
  ([`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)).
  The other rank fields mean what they do for `"rank"`.

## Value

An ast_slice node
