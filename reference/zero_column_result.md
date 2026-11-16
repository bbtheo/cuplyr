# A zero-column result with a known number of rows

The one place verbs that end with no columns
([`select()`](https://dplyr.tidyverse.org/reference/select.html),
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html),
`[`,
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html))
build their result: an ungrouped zero-column `tbl_gpu` of `n` rows in
the input's exec_mode. `n` defaults to the input's row count; pending
lazy operations are computed first to get it (a materialization point,
as for
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
with no expressions).

## Usage

``` r
zero_column_result(.data, verb, n = NULL)
```

## Arguments

- .data:

  The input `tbl_gpu`

- verb:

  Character, for the plan-break notification

- n:

  Row count of the result, or `NULL` for `.data`'s own
