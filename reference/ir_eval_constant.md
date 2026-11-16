# Evaluate a raw (unparsed) expression as a constant in `env`

No data mask is supplied, so a genuine column reference (or anything
else that isn't resolvable from `env` alone) raises an ordinary R error,
caught here and reported as "not a constant" (`ok = FALSE`) rather than
propagating – the same "no data mask" trick `resolve_slice_dots()`
(`R/slice.R`) already relies on to distinguish plain R values from
column references. An integer64 value is `ok = FALSE` (F96), and so,
unevaluated, is a call that reaches the data mask by name (F79,
`expr_reaches_mask()`).

## Usage

``` r
ir_eval_constant(raw_expr, env)
```

## Arguments

- raw_expr:

  A language object (not a quosure)

- env:

  The environment to evaluate against

## Value

`list(ok = <logical>, value = <the evaluated R value, or NULL>)`
