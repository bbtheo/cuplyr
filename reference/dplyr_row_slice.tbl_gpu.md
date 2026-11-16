# Modify rows of a GPU table by position (dplyr backend generic)

Implements dplyr's
[`dplyr_row_slice()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html)
extension-contract generic for `tbl_gpu` – a thin wrapper over the same
`gpu_slice_indices()` primitive
[slice()](https://bbtheo.github.io/cuplyr/reference/slice.tbl_gpu.md)
itself uses for arbitrary positive-index gathers (duplicates/reordering
allowed). Per dplyr's own generic
([`dplyr_row_slice()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html),
not this method), `i` is always numeric (1-based positions) or logical
(a row mask); this method materializes any pending lazy operations
first, since there is no lazy AST node for an arbitrary literal index
gather.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dplyr_row_slice(data, i, ..., preserve = FALSE)
```

## Arguments

- data:

  A `tbl_gpu` object.

- i:

  A numeric (1-based positions) or logical (row mask) vector.

- ...:

  Ignored. Included for compatibility with the dplyr generic.

- preserve:

  Whether to keep the groups the slice emptied, as dplyr's
  [`dplyr_row_slice()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html).

## Value

A new `tbl_gpu`.

## Details

Matches dplyr's `dplyr_row_slice.grouped_df()` for a
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped
table: with `.drop = TRUE` (the default) the groups the slice emptied
vanish from
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html),
unless `preserve = TRUE`; a `.drop = FALSE` table keeps every group of
the input whatever `preserve` says (for any column type, not just
factors). Those keys are fixed on the result (`$fixed_keys`, see
[`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md))
only when a group was actually lost; otherwise a plain regroup gives the
same groups.
