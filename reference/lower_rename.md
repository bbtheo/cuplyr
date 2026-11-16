# Lower rename node

Pure schema-only passthrough: column names live entirely in the R-side
schema (see
[`ast_rename()`](https://bbtheo.github.io/cuplyr/reference/ast_rename.md),
`R/ast.R`), never on the underlying GPU table, so there is no GPU call
to make here at all – this simply lowers `ast$input` and returns its
pointer unchanged.
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
(`R/execute.R`) is what actually applies the new (renamed) schema to the
resulting `tbl_gpu`, via
[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md).

## Usage

``` r
lower_rename(ast, source_ptr)
```
