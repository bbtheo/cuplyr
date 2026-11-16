# Parse `roll_mean(x, n, na.rm = FALSE)` and its five siblings

cuplyr's own rolling-window surface (RcppRoll/slider-adjacent naming –
see the file-level comment above this section for the full design
rationale). Right-aligned only: there is no `align=` parameter on this
surface at all (unlike the `frollmean()`-alias path below, which has to
reject a non-default `align=` explicitly since data.table's own
signature offers one).

## Usage

``` r
ir_parse_roll_call(expr, env, schema, fn_name)
```

## Arguments

- fn_name:

  One of `"roll_mean"`/`"roll_sum"`/`"roll_min"`/
  `"roll_max"`/`"roll_sd"`/`"roll_median"`
