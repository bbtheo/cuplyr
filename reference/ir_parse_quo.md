# Parse an rlang quosure into cuplyr IR

Columns shadow the environment; any subtree with no column reference is
constant-folded by evaluating it in the quosure's environment. Returns
`NULL` for anything the IR doesn't understand – callers (filter/mutate
parsers) fall back to the CPU-eval or error path.

## Usage

``` r
ir_parse_quo(quo, schema)
```

## Arguments

- quo:

  An `rlang` quosure

- schema:

  List with `names`/`types` (as from
  [`current_schema()`](https://bbtheo.github.io/cuplyr/reference/current_schema.md))

## Value

An IR node, or `NULL` if unsupported

## Details

A parse-time type check that fails only because a factor (DICTIONARY32)
operand is involved signals `cuplyr_factor_needs_cpu` (see
[`stop_factor_needs_cpu()`](https://bbtheo.github.io/cuplyr/reference/stop_factor_needs_cpu.md));
that class becomes `NULL` here, so every verb sends the dot to its CPU
fallback, where dplyr combines the factor itself.
