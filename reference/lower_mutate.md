# Lower mutate node

Task T5 (Phase 1 expression-engine cutover,
`scratchpad/phase1_expression_engine.md` section 5): each expression's
IR is bound
([`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md))
against a running index map that mirrors `gpu_mutate_expr()`'s contract
(`src/ops_expr.cpp`): expression `i`'s top-level result always lands at
ctx index `n_input + i`, so a `col` node referencing an *earlier*
expression's output (however many
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) calls it
originally came from –
[`fuse_mutates()`](https://bbtheo.github.io/cuplyr/reference/fuse_mutates.md)
may have merged several `ast_mutate` nodes into this one) must bind to
that *appended* index, never the original input column's position.

## Usage

``` r
lower_mutate(ast, source_ptr)
```

## Details

`index_names`/`index_types` track this: they start as the input schema,
and after each expression, any stale (superseded) occurrence of its
output name is blanked to `NA` – so a subsequent
[`match()`](https://rdrr.io/r/base/match.html) can't find the old
position – before the name is appended at the new slot. This keeps
`match(name, index_names)` always returning the single live occurrence,
whose 0-based position is exactly the ctx index `gpu_mutate_expr()` will
place that expression's result at.
