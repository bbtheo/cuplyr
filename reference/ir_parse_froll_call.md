# Parse `data.table::frollmean(x, n, fill=, algo=, align=, na.rm=, hasNA=, adaptive=)` and its five siblings as an alias for `roll_mean()`/etc.

See the file-level comment above this section: only the default
`align = "right"`, `fill = NA`, `adaptive = FALSE`, `partial = FALSE`,
`give.names = FALSE` are supported (a non-default value returns `NULL`,
i.e. "not implemented, fall back" – mutate()'s usual contract for an
unrecognized shape: `partial = TRUE` changes windowing semantics
entirely – allows a genuinely incomplete window to produce a value
instead of `NA` – and `give.names = TRUE` changes the RETURN SHAPE to a
named `data.table`/list instead of a plain vector, so both must be
rejected, not silently ignored); `algo=`/`has.nf=`/`hasNA=` are accepted
and ignored (pure performance hints, verified against data.table's own
docs to never change the *result*). Proto signature verified against
this environment's actual installed
[`data.table::frollmean`](https://rdrr.io/pkg/data.table/man/froll.html)
(`args(data.table::frollmean)`), not assumed from memory/docs alone –
note the real signature has `has.nf=` (NOT `hasNA=`, which is a
separate, no-default legacy/back-compat argument that's ALSO present,
verified empirically).

## Usage

``` r
ir_parse_froll_call(expr, env, schema, dt_fn_name)
```

## Arguments

- dt_fn_name:

  One of `"frollmean"`/`"frollsum"`/`"frollmin"`/
  `"frollmax"`/`"frollsd"`/`"frollmedian"`
