# Apply an AST node to a tbl_gpu, dispatching on exec_mode

The only exec_mode branch point (D1). Lazy mode stores `node` as the new
`$lazy_ops` and does no GPU work; eager mode lowers and executes `node`
immediately. Both modes uniformly compute `schema` from `input_schema`
alone (via
[`infer_schema_from()`](https://bbtheo.github.io/cuplyr/reference/infer_schema_from.md),
with factor_levels reattached per D3) and `groups` (via
[`propagate_groups()`](https://bbtheo.github.io/cuplyr/reference/propagate_groups.md)).

## Usage

``` r
push_op(.data, node, input_schema = current_schema(.data), carry = TRUE)
```

## Arguments

- .data:

  A `tbl_gpu` object

- node:

  The new AST node (already wrapping `input_node(.data)`)

- input_schema:

  `current_schema(.data)`, when the caller already has it (a node chain,
  [`push_chain()`](https://bbtheo.github.io/cuplyr/reference/push_chain.md):
  the previous push's `$schema`)

- carry:

  `FALSE` leaves the frame metadata (attributes, bare class) off the
  result: a step of a
  [`push_chain()`](https://bbtheo.github.io/cuplyr/reference/push_chain.md),
  which carries it once at the end

## Value

A new `tbl_gpu` object
