# Create a source AST node (leaf node)

Create a source AST node (leaf node)

## Usage

``` r
ast_source(schema, source_ptr = NULL, scan = NULL, origin = NULL)
```

## Arguments

- schema:

  List with names and types vectors

- source_ptr:

  External pointer to the table the plan starts from

- scan:

  A parquet scan record
  ([`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)),
  or NULL

- origin:

  The CPU step that produced the table (`plan_origin()`, R/plan-text.R:
  plain text only), or NULL. Read only by
  [`explain()`](https://dplyr.tidyverse.org/reference/explain.html) and
  [`show_query()`](https://dplyr.tidyverse.org/reference/explain.html).

## Value

An ast_source node
