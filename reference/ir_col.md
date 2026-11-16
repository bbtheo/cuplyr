# Create an IR column-reference node

Create an IR column-reference node

## Usage

``` r
ir_col(name)
```

## Arguments

- name:

  Character scalar, the column name

## Value

A `cuplyr_ir_col` node: `list(kind = "col", name = <chr1>)`. Gains an
`index` field (0-based) only after
[`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md)
runs.
