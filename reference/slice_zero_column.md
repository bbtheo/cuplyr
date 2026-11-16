# Slice a zero-column tbl_gpu by running dplyr on a stand-in

A zero-column table has no cudf rows to gather (its row count lives in
`$nrow`), so the kept positions are computed by dplyr itself on a
zero-column tibble of that height, and only the count is carried over. A
lazy zero-column table's pending ops are computed first. Returns `NULL`
when `.data` is not an ungrouped, known-height, zero-column table, so
the caller falls through to its usual path (which errors for a
zero-column table with rows).

## Usage

``` r
slice_zero_column(.data, fn)
```

## Arguments

- .data:

  A `tbl_gpu`

- fn:

  Function of one tibble, running the dplyr verb
