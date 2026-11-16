# Can a node run on a table with duplicated column names?

On such a table a column name is ambiguous, so any node that finds a
column by name (a predicate, an expression, a sort or join key, a
by-name select) would silently take the first of the same-named columns.
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
refuses every node on a duplicated-name input except the ones listed
here, which address columns by position or touch only rows:

- `select`/`rename` carrying `positions` (select(), relocate(),
  rename(), `[` and `$`/`[[` build them whenever names are duplicated);

- `distinct` carrying `key_positions` (distinct()/unique());

- `slice` without an ordering column: head()/tail(), `x[i, ]` and the
  row-index slices (the slice verbs themselves refuse duplicated names
  at their entry, as dplyr does);

- `barrier`. A new node type, or a by-name variant of these, fails
  closed.

## Usage

``` r
node_is_position_safe(node)
```

## Arguments

- node:

  An AST node about to be pushed

## Value

`TRUE` when `node` is position-correct
