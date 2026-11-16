# The shared "Each row in X must match at most 1 row in Y" error

Mirrors `dplyr:::stop_join_matches_multiple()` exactly – the single
message-template function backing all three relationship-cardinality
violations below (one-to-one's two directions, one-to-many, many-to-one)
plus the (unreachable from any join verb this package implements –
dplyr's own `multiple=` never resolves to
[`vctrs::vec_locate_matches()`](https://vctrs.r-lib.org/reference/vec_locate_matches.html)'s
"error"/"warning" values from any join verb; confirmed empirically by
grepping dplyr's whole namespace source for a literal
`multiple = "error"`/`"warning"`, found only in `rows_*()`, out of scope
here) `dplyr_error_join_matches_multiple` class dplyr itself defines for
the same template.

## Usage

``` r
stop_join_matches_multiple(i, x_name, y_name, class)
```

## Arguments

- i:

  1-based row number to report

- x_name, y_name:

  Which table name each half of the message names

- class:

  The specific `dplyr_error_join_relationship_*` class
