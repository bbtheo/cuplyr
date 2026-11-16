# Build one side of a join's AST, always anchoring the source pointer

Like
[`input_node()`](https://bbtheo.github.io/cuplyr/reference/input_node.md),
but *unconditionally* attaches `tbl$ptr` to the leaf `ast_source` node,
even when `tbl` is lazy with no pending ops (the case where
[`input_node()`](https://bbtheo.github.io/cuplyr/reference/input_node.md)
deliberately leaves the pointer unset and relies on
[`lower_and_execute()`](https://bbtheo.github.io/cuplyr/reference/lower_and_execute.md)'s
single `source_ptr` argument as a fallback, filled in later by
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html)/[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)).
A join has *two* independent input trees, so that single-fallback
mechanism can't serve both leaves – instead, each side's own concrete
GPU pointer must be anchored explicitly at construction time. This is
safe because `tbl$ptr` is always a valid pointer to that side's own
(possibly stale, pre-pending-ops) base table:
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
carries it forward unchanged across lazy operations, only
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html)/[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
ever replace it.

## Usage

``` r
join_input_node(tbl)
```

## Arguments

- tbl:

  A `tbl_gpu` object (one side of a join)

## Value

An `ast_node` with its leaf `source_ptr` set to `tbl$ptr`
