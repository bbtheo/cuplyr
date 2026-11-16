# Rename columns of a GPU table

Renames columns of a GPU table, similar to
[`dplyr::rename()`](https://dplyr.tidyverse.org/reference/rename.html).
Runs entirely as a schema-only operation – no GPU work at all, since
column names are R-side metadata, never part of the underlying GPU table
data.

## Usage

``` r
# S3 method for class 'tbl_gpu'
rename(.data, ...)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Use `new_name = old_name` to rename columns. Supports tidyselect
  renaming helpers, e.g. `rename(df, all_of(c(a = "x")))`.

## Value

A `tbl_gpu` object with the requested columns renamed. Column order,
data, and types are unchanged. If `.data` is grouped and a group column
is renamed,
[`group_vars()`](https://dplyr.tidyverse.org/reference/group_data.html)
reflects the new name.

## Details

Renaming a column to a name already used by a different, un-renamed
column errors, matching
[`dplyr::rename()`](https://dplyr.tidyverse.org/reference/rename.html).
A same-call swap (e.g. `rename(df, x = y, y = x)`) is valid, since both
original names are replaced at once.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  result <- gpu_mtcars |>
    rename(miles_per_gallon = mpg) |>
    collect()
}
```
