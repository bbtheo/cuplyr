# Check if an AST node is an optimization barrier

`distinct` is treated as a barrier like `summarise`/`arrange`: the
simplest safe choice, since a filter above a `distinct` node may
reference columns the `distinct` projected away (`.keep_all = FALSE`),
and pushing it below would change which columns are visible to it.
Optimization opportunity for later: a filter referencing only surviving
key columns (or any column when `.keep_all = TRUE`) could safely push
below distinct, the same way filter pushdown already works across
`select`.

## Usage

``` r
is_barrier(node)
```

## Arguments

- node:

  An AST node

## Value

TRUE if node is a barrier

## Details

`slice` (the whole family:
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html)/
[`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html))
is a barrier for the same reason `arrange` is: it's a row-selection op
whose result depends on the current row order/content, so reordering or
pushing other ops across it (e.g. a filter that would change which rows
[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html) sees)
would change the result.

`rename` is a barrier for the reasons documented on
[`ast_rename()`](https://bbtheo.github.io/cuplyr/reference/ast_rename.md)
above: the simplest safe choice for a schema-only rename, avoiding the
need to translate column references through its name map for anything
pushed across it.

`window` (`scratchpad/phase5_window_design.md` section 2.1) is a
barrier: every window spec's result depends on the FULL row set (which
rows exist) and the current row order (frames are built from a stable
sort of group+order columns) – nothing may be pushed across it, the same
reasoning as `arrange`/`slice`.

Phase 12 audit note: barrier classification only matters for node TYPES
that actually appear in an AST at all.
[`do()`](https://dplyr.tidyverse.org/reference/do.html)/[`rows_insert()`](https://dplyr.tidyverse.org/reference/rows.html)/
[`rows_append()`](https://dplyr.tidyverse.org/reference/rows.html)/[`rows_update()`](https://dplyr.tidyverse.org/reference/rows.html)/[`rows_patch()`](https://dplyr.tidyverse.org/reference/rows.html)/[`rows_upsert()`](https://dplyr.tidyverse.org/reference/rows.html)/
[`rows_delete()`](https://dplyr.tidyverse.org/reference/rows.html), and
every colwise/pick-bearing call
([`mutate_at()`](https://dplyr.tidyverse.org/reference/mutate_all.html),
[`pick()`](https://dplyr.tidyverse.org/reference/pick.html)-inside-an-expression,
...) are whole-call CPU fallbacks
([`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md),
R/fallback.R): they materialize any pending lazy ops via
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) FIRST,
then run the real dplyr verb on the collected data and re-upload – the
result always has `lazy_ops = NULL` (see
`tests/testthat/test-fallback.R`'s "fallback verbs materialize pending
lazy ops" test). None of these verbs ever construct an AST node of their
own, so "is this node type a barrier" is simply inapplicable to them –
there is nothing downstream of a fallback call for a barrier to block,
since nothing downstream can ever see a pending op that predates the
fallback.
