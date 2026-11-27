# Create an IR function-call node

Create an IR function-call node

## Usage

``` r
ir_call(op, args, meta = list())
```

## Arguments

- op:

  Canonical cuplyr op name (not necessarily the R function name that
  produced it – see
  [ir_call_registry](https://bbtheo.github.io/cuplyr/reference/ir_call_registry.md)
  and `ir_op_alias()`, e.g. `ceiling` normalizes to `ceil`)

- args:

  List of IR nodes (the call's arguments, already parsed)

- meta:

  Slot for lowering metadata (e.g. a window call's `order_by=` override,
  or a `case_when` node's `n_when`)

## Value

A `cuplyr_ir_call` node
