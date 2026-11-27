# First rows of a GPU table

[`utils::head()`](https://rdrr.io/r/utils/head.html) method for
`tbl_gpu`. Keeps the same rows as
[`utils::head.data.frame()`](https://rdrr.io/r/utils/head.html):
implemented as `slice_head(n = floor(n))` on the ungrouped table (see
[`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md)),
with the grouping put back on the result.

## Usage

``` r
# S3 method for class 'tbl_gpu'
head(x, n = 6L, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- n:

  Number of rows to keep (default `6`, matching
  [`utils::head()`](https://rdrr.io/r/utils/head.html)). Negative `n`
  keeps all but the last `abs(n)` rows. A fractional `n` is rounded
  down, as in base. Groups are ignored: the rows are the first `n` of
  the whole table, and the result keeps the grouping of `x`.

- ...:

  Not used.

## Value

A `tbl_gpu` object with the first `n` rows.
