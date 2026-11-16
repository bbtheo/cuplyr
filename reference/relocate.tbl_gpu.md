# Reorder columns of a GPU table

Moves columns of a GPU table to a new position, similar to
[`dplyr::relocate()`](https://dplyr.tidyverse.org/reference/relocate.html).
The reorder itself runs via
[`ast_select()`](https://bbtheo.github.io/cuplyr/reference/ast_select.md)
(schema

- a `gpu_select()` gather, no new C++ binding needed); a rename
  (`relocate(df, a = x)`) is schema-only, like
  [`rename.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/rename.tbl_gpu.md).

## Usage

``` r
# S3 method for class 'tbl_gpu'
relocate(.data, ..., .before = NULL, .after = NULL)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Columns to move (tidyselect syntax; may include `new = old` renames).

- .before, .after:

  Destination, as a tidyselect specification of an existing column; the
  selected columns are placed immediately before/ after it. At most one
  of `.before`/`.after` may be supplied; if neither is given, selected
  columns move to the front.

## Value

A `tbl_gpu` object with columns reordered (and any requested columns
renamed).

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  result <- gpu_mtcars |>
    relocate(hp, .after = mpg) |>
    collect()
}
```
