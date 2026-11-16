# Do a table's POSIXct zones already match a target schema's?

The metadata counterpart of
[`factor_levels_match_target()`](https://bbtheo.github.io/cuplyr/reference/factor_levels_match_target.md):
the GPU values are UTC instants either way, but a set operation has to
hand back the target's zones. An absent tzone and `""` count as equal
(slicing turns the former into the latter).

## Usage

``` r
tz_matches_target(schema, target_schema)
```

## Arguments

- schema, target_schema:

  Schemas with the same column names and types

## Value

`TRUE` when every column has the same zone
