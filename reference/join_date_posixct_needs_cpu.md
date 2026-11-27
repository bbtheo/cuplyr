# Does a key pair unify a Date with a non-UTC POSIXct?

dplyr casts such a Date to local midnight of the POSIXct's zone
(`vec_cast(<Date>, <POSIXct tz>)`), which moves with DST. The native
cast is the UTC midnight, exact only for a UTC-equivalent zone; any
other pair (equi or non-equi) routes the whole join to the CPU.

## Usage

``` r
join_date_posixct_needs_cpu(x_schema, y_schema, join_spec)
```
