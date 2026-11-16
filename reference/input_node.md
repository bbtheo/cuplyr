# Build the AST node that a new op should be layered on top of

- lazy + pending ops: returns `.data$lazy_ops` as-is.

- lazy + no pending ops: returns a fresh `ast_source(.data$schema)`,
  matching the historical lazy-verb convention (e.g. the now-deleted
  `filter_lazy()`, `select_lazy()`) of leaving the source pointer unset;
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)/
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
  supply `.data$ptr` as the lowering fallback `source_ptr`.

- eager: returns a fresh
  `ast_source(.data$schema, source_ptr = .data$ptr)` so
  [`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
  can call
  [`lower_and_execute()`](https://bbtheo.github.io/cuplyr/reference/lower_and_execute.md)
  immediately without a separate source table.

## Usage

``` r
input_node(.data)
```

## Arguments

- .data:

  A `tbl_gpu` object

## Value

An `ast_node` to use as the `input` of a new node
