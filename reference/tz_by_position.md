# Zone of every column of a schema, by position

Zone of every column of a schema, by position

## Usage

``` r
tz_by_position(schema)
```

## Arguments

- schema:

  A schema with `names`, `types` and (maybe) `tz`

## Value

A character vector the length of `schema$names`: the zone of a
TIMESTAMP_MICROSECONDS column (`"UTC"` when it has no entry), `"UTC"`
for every other column
