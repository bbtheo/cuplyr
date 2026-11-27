# Split a named list of per-dot expressions into window-decomposition batches

Pure index-partitioning: implements the chunking rule below in isolation
from
[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)'s
own stage-building loop, so a verb's own AST-emission wiring
(`R/mutate.R`'s `mutate_window()`, and the analogous wiring in
`R/filter.R`/`R/slice.R`) can drive `decompose_window_group()`
batch-by-batch itself – interleaving each batch's own materializing verb
node between batches, which
[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)'s
single flat call cannot do (see that function's docs: it folds each
batch's dot outputs into its *own* running schema bookkeeping only, it
never emits or knows about AST nodes) – while
[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)
keeps using this exact same split for its own (non-interleaved,
single-verb-node-per-call) callers, e.g. `test-window-plan.R`'s
hand-built-AST-free assertions.

## Usage

``` r
window_batches(exprs)
```

## Arguments

- exprs:

  A *named* list of IR nodes, one per dot, in verb-call order (see
  [`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)'s
  own `exprs` parameter docs for the naming requirement – this function
  assumes it, it does not itself validate)

## Value

A list of integer vectors, each a contiguous run of 1-based indices into
`exprs` forming one batch, in order;
[`list()`](https://rdrr.io/r/base/list.html) if `exprs` is empty

## Details

Dots are processed in order and batched together UNLESS a later dot's
window-call arguments reference an earlier dot's OWN output column name
(from the *current* batch) – see `window_arg_cols()`.
