# Create a select AST node

Create a select AST node

## Usage

``` r
ast_select(input, columns, positions = NULL, pruned = FALSE)
```

## Arguments

- input:

  Input AST node

- columns:

  Character vector of column names to keep

- positions:

  `NULL` (select by `columns`), or the integer positions of the kept
  columns in `input`'s schema: used instead of the names when the input
  has duplicated column names, where a name is ambiguous. Such a node is
  an optimizer barrier and never pruned through.

- pruned:

  `TRUE` for a select the optimizer inserted to drop columns nothing
  reads
  ([`push_down_projections()`](https://bbtheo.github.io/cuplyr/reference/push_down_projections.md));
  [`push_down_filters()`](https://bbtheo.github.io/cuplyr/reference/push_down_filters.md)
  leaves a filter above it, as moving the filter below would undo the
  pruning. The field is only present when `TRUE`.

## Value

An ast_select node
