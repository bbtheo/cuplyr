# Rename columns of a GPU table with a function

Renames columns of a GPU table by applying a function to their names,
similar to
[`dplyr::rename_with()`](https://dplyr.tidyverse.org/reference/rename.html).
Runs entirely as a schema-only operation, like
[`rename.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/rename.tbl_gpu.md).

## Usage

``` r
# S3 method for class 'tbl_gpu'
rename_with(.data, .fn, .cols = dplyr::everything(), ...)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- .fn:

  A function (or one-sided formula, e.g. `~ paste0(.x, "_2")`) applied
  once to the character vector of selected (old) names; must return a
  character vector of the same length.

- .cols:

  Tidyselect specification of which columns to rename (default: all
  columns). Does not itself support renaming.

- ...:

  Additional arguments passed to `.fn`.

## Value

A `tbl_gpu` object with the selected columns renamed via `.fn`. Column
order, data, and types are unchanged.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  result <- gpu_mtcars |>
    rename_with(toupper) |>
    collect()
}
```
