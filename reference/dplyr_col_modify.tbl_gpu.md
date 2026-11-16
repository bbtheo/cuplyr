# Add, remove, or replace columns of a GPU table with literal values (dplyr backend generic)

Implements dplyr's
[`dplyr_col_modify()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html)
extension-contract generic for `tbl_gpu`. Unlike
[mutate()](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md),
`cols` here is a named list of already-computed R VALUES (not
expressions) – a `NULL` entry removes that column, a non-`NULL` entry
adds or replaces it (recycled to `nrow(data)` via
[`vctrs::vec_recycle_common()`](https://vctrs.r-lib.org/reference/vec_recycle.html),
matching `dplyr:::dplyr_col_modify.data.frame()`'s own recycling
contract, including its error class on a size mismatch). Existing
columns being replaced stay at their original position; genuinely new
columns are appended at the end – matching dplyr's own contract exactly.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dplyr_col_modify(data, cols)
```

## Arguments

- data:

  A `tbl_gpu` object.

- cols:

  A named list: `NULL` entries remove that column, other entries are
  recycled to `nrow(data)` and added/replace that column.

## Value

A new `tbl_gpu`.

## Details

Implemented as: drop any column being removed/replaced from `data`,
upload the new/replacement columns as a small `tbl_gpu` (reusing the
ordinary
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
upload path – no expression IR needed, these are already concrete
values), and
[`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md)
the two back together in the right column order.
