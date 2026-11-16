# Plan the window-decomposition stages for a set of expressions

Drives
[`ir_extract_windows()`](https://bbtheo.github.io/cuplyr/reference/ir_extract_windows.md)
to full fixed point (every window call resolved) for each verb-level
"dot" in `exprs`, applying the multi-dot chunking rule from
`scratchpad/phase5_window_design.md` section 1.2: dots are processed in
order and batched together (decomposed jointly, sharing CSE and stage
passes) UNLESS a later dot's window-call arguments reference an earlier
dot's OWN output column name (from the *current* batch) – in which case
that later dot starts a new batch, after the finished batch's own dot
outputs are folded into the running schema as real columns (so the new
batch can reference them).

## Usage

``` r
plan_window_stages(
  exprs,
  schema,
  group_cols = character(),
  sums = c("int", "lenient")
)
```

## Arguments

- exprs:

  A *named* list of IR nodes, one per dot, in verb-call order;
  `names(exprs)` are the dots' intended output column names (needed for
  the chunking rule above)

- schema:

  The schema `exprs` were parsed against (list(names=, types=))

- group_cols:

  Character vector of group-by column names (passed through unchanged
  for a future
  [`ast_window()`](https://bbtheo.github.io/cuplyr/reference/ast_window.md)
  to consume – W1's own decomposition logic doesn't need to inspect
  these at all)

- sums:

  (F42) How an integer/logical window
  [`sum()`](https://rdrr.io/r/base/sum.html) is typed. `"int"`
  (default): INT32, as dplyr's integer sum. `"lenient"` (lazy
  [`filter()`](https://dplyr.tidyverse.org/reference/filter.html), whose
  window nodes are flagged `on_sum_overflow = "double"`): INT32 too, and
  a top-level sum may come back FLOAT64 when it overflows; one that
  feeds a later stage (inside another window call's arguments) has its
  spec flagged `on_sum_overflow = "error"` instead, because that stage
  is typed from the INT32.

## Value

`list(stages, post, schema, group_cols)`:

- `stages`: list of
  `list(pre = <list of make_mutate_expr() steps>, specs = <list of window_spec()s>)`,
  in the order they must execute

- `post`: named list (same names/order as `exprs`), the fully rewritten
  (window-free) IR for each dot – identical to the corresponding input
  node, unchanged, for any dot that had no window call at all

- `schema`: the schema after every stage's outputs, AND every dot's own
  output column, have been appended (in dot order)

- `group_cols`: passed through unchanged

## Details

This is a pure planning function: it builds the stage/spec/pre-mutate
structures a verb will later turn into actual `ast_mutate`/`ast_window`
nodes (section 1.3; not implemented until W2/W3) – no AST node is built
here, and nothing is pushed onto any `tbl_gpu`.
