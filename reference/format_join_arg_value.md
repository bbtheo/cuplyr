# Format a scalar value for a "not one of ..." style error message

Strings render quoted (`"bogus"`); anything else falls back to
[`deparse()`](https://rdrr.io/r/base/deparse.html) so the message stays
readable without crashing on non-character input.

## Usage

``` r
format_join_arg_value(value)
```
