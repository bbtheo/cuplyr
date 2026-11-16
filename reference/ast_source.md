# Create a source AST node (leaf node)

Create a source AST node (leaf node)

## Usage

``` r
ast_source(schema, source_ptr = NULL, scan = NULL)
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

## Value

An ast_source node
