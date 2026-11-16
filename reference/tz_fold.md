# vctrs' common tzone of two POSIXct operands

`vec_ptype2()` keeps `x`'s zone unless it is local (`""`/absent), then
takes `y`'s. `NULL` stands for "no POSIXct operand" (a Date or a
non-timestamp) and is skipped.

## Usage

``` r
tz_fold(x, y)
```

## Arguments

- x, y:

  Zones as stored in `schema$tz` (`""`, `NA`, a name), or `NULL`

## Value

The common zone, or `NULL` when both are `NULL`
