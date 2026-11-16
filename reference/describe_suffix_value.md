# Describe a `suffix=` value the way dplyr's own bad-value message would

A minimal, purpose-built port of just enough of dplyr's
`obj_type_friendly()`-driven text (walked via
`body(dplyr:::standardise_join_suffix)`, empirically verified against
dplyr 1.2.1's actual message for `character(0)`/a length-1 string/a
length-3 vector/a bare `NA`) to reproduce
[`validate_join_suffix()`](https://bbtheo.github.io/cuplyr/reference/validate_join_suffix.md)'s
messages exactly – not a general-purpose friendly-type formatter.

## Usage

``` r
describe_suffix_value(x)
```
