# Last rows of a GPU table

[`utils::tail()`](https://rdrr.io/r/utils/head.html) method for
`tbl_gpu`. Keeps the same rows as
[`utils::tail.data.frame()`](https://rdrr.io/r/utils/head.html):
implemented as `slice_tail(n = ceiling(n))` on the ungrouped table (see
[`slice_tail.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_tail.tbl_gpu.md)
for the native-vs-fallback dispatch), with the grouping put back on the
result.

## Usage

``` r
# S3 method for class 'tbl_gpu'
tail(x, n = 6L, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- n:

  Number of rows to keep (default `6`, matching
  [`utils::tail()`](https://rdrr.io/r/utils/head.html)). Negative `n`
  keeps all but the first `abs(n)` rows. A fractional `n` is rounded up,
  as in base, so `-1 < n < 0` keeps every row. Groups are ignored: the
  rows are the last `n` of the whole table, and the result keeps the
  grouping of `x`.

- ...:

  Not used.

## Value

A `tbl_gpu` object with the last `n` rows.
