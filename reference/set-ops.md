# Set operations on `tbl_gpu` tables

dplyr's row-set operations on GPU tables, run on the GPU:

- [`union()`](https://generics.r-lib.org/reference/setops.html): the
  distinct rows of `x`, in `x`'s order, then the rows of `y` not already
  in `x`, in `y`'s order.

- [`union_all()`](https://dplyr.tidyverse.org/reference/setops.html):
  the rows of `x` (duplicates kept), then those of `y`.

- [`intersect()`](https://generics.r-lib.org/reference/setops.html) and
  [`setdiff()`](https://generics.r-lib.org/reference/setops.html):
  distinct rows of `x`, in `x`'s order, that are (or are not) in `y`.

- [`symdiff()`](https://dplyr.tidyverse.org/reference/setops.html): the
  rows of `setdiff(x, y)`, then those of `setdiff(y, x)`.

- [`setequal()`](https://generics.r-lib.org/reference/setops.html):
  `TRUE` or `FALSE`, ignoring row order and duplicates on both sides.

As in dplyr, `x` and `y` must have the same column names (the order may
differ: `y` is reordered to `x`'s), and each column must have a common
type: logical, integer and double columns combine to the wider type, a
logical or integer column with an integer64 one gives integer64
(integer64 and double have no common type: an error, as in dplyr), a
`Date` and a `POSIXct` combine to `POSIXct`, two factors with different
levels combine to a factor with the union of the levels, and a factor
next to a character column gives a character column. A column-type mix
that dplyr refuses (integer vs. character, factor vs. integer, ...) is
an error naming the column and both types.

## Usage

``` r
# S3 method for class 'tbl_gpu'
union(x, y, ...)

# S3 method for class 'tbl_gpu'
union_all(x, y, ...)

# S3 method for class 'tbl_gpu'
intersect(x, y, ...)

# S3 method for class 'tbl_gpu'
setdiff(x, y, ...)

# S3 method for class 'tbl_gpu'
symdiff(x, y, ...)

# S3 method for class 'tbl_gpu'
setequal(x, y, ...)
```

## Arguments

- x, y:

  `tbl_gpu` objects. `y` may also be a plain `data.frame` (uploaded
  automatically via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)) –
  unlike the join verbs, these generics have no `copy=` argument.

- ...:

  Not used; must be empty (dplyr itself requires this too).

## Value

A `tbl_gpu` with `x`'s column order and grouping, with the rows in the
order described above, except
[`setequal()`](https://generics.r-lib.org/reference/setops.html), which
returns `TRUE` or `FALSE`.

## Details

These run in dplyr on the CPU: tables with duplicated column names; a
`Date` column next to a `POSIXct` column in a zone other than UTC (the
`Date` becomes local midnight of that zone); a column with element names
or a one-column matrix column, which dplyr combines by its own rules
(except in
[`setequal()`](https://generics.r-lib.org/reference/setops.html)); and a
rowwise `x`.
