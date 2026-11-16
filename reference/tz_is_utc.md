# Is a tzone attribute value UTC-equivalent?

`""` and `NA` (attribute absent) both mean local time, so they resolve
to the session zone first. An undeterminable session zone is not UTC.

## Usage

``` r
tz_is_utc(tz)
```

## Arguments

- tz:

  Character scalar: a zone, `""` or `NA`

## Value

`TRUE` when instants and wall-clock times coincide in `tz`
