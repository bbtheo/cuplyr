# Infer the time zone of a POSIXct-valued IR node (F32)

The sibling of
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
for the `tzone` attribute the result would carry in R. Only meaningful
for a TIMESTAMP_MICROSECONDS result (it returns `"UTC"` for anything
else): a column has its schema zone, a POSIXct literal its own `tzone`,
and every op that can return a POSIXct has an explicit rule in
[ir_tz_rules](https://bbtheo.github.io/cuplyr/reference/ir_tz_rules.md).
An op without one raises `cuplyr_timestamp_needs_cpu` (the expression
goes to the CPU when the parse sees it,
[`ir_parse_quo()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_quo.md))
rather than guess a zone.

## Usage

``` r
ir_infer_tz(ir, schema)
```

## Arguments

- ir:

  An IR node

- schema:

  List with `names`/`types`/`tz`

## Value

A zone string as stored in `schema$tz`: a name, `""` (local), `NA` (no
tzone attribute) or `"UTC"`
