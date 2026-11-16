# Get a glimpse of a GPU table's structure

Prints a compact summary of a `tbl_gpu`'s structure – row/column counts,
grouping (if any), and each column's type with a preview of its first
few values – similar to
[`dplyr::glimpse()`](https://pillar.r-lib.org/reference/glimpse.html).

## Usage

``` r
# S3 method for class 'tbl_gpu'
glimpse(x, width = NULL, ...)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- width:

  Output width (defaults to `getOption("width")`).

- ...:

  Ignored (present for generic compatibility with
  [`dplyr::glimpse()`](https://pillar.r-lib.org/reference/glimpse.html)).

## Value

`x`, invisibly. Like
[`dplyr::glimpse()`](https://pillar.r-lib.org/reference/glimpse.html),
this is a side-effecting print method: it never transforms `.data`, so
there is nothing to re-upload.

## Details

Unlike the (now-removed) Phase 2 CPU-fallback
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html), this
implementation never collects the whole table to R: any pending lazy
operations are materialized (via
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html)), but
the actual preview data comes from a small, capped row sample (like
`print.tbl_gpu()`), and row/column counts/types come from lightweight
metadata calls – never a full
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

A lazy table with pending operations runs its plan once for the preview
(the result is freed when
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) returns;
the table itself is unchanged), and one extra line starting `Lazy:`
follows `Columns:`. If the plan fails (an error, out of GPU memory),
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) prints
`Rows: ??`, the `Lazy:` line, the message and the stored schema instead
of raising.

Column value/type formatting does not attempt to reproduce dplyr's own
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) output
(which is formatted via `pillar`) byte-for-byte – only the structural
facts (row count, column count, grouping, column names, and column
types) are guaranteed to match.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  tbl_gpu(mtcars) |> glimpse()
}
#> Rows: 32
#> Columns: 11
#> $ mpg  <dbl> 21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.…
#> $ cyl  <dbl> 6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8,…
#> $ disp <dbl> 160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6, 27…
#> $ hp   <dbl> 110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180…
#> $ drat <dbl> 3.9, 3.9, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92, 3.92, 3…
#> $ wt   <dbl> 2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.4…
#> $ qsec <dbl> 16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20, 22.9, 18.3, …
#> $ vs   <dbl> 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,…
#> $ am   <dbl> 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,…
#> $ gear <dbl> 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 3, 3,…
#> $ carb <dbl> 4, 4, 1, 1, 2, 1, 4, 2, 2, 4, 4, 3, 3, 3, 4, 4, 4, 1, 2, 1, 1, 2,…
```
