# Keep only distinct/unique rows of a GPU table

Removes duplicate rows from a GPU table, similar to
[`dplyr::distinct()`](https://dplyr.tidyverse.org/reference/distinct.html).
Runs entirely on the GPU (`cudf::stable_distinct()`), keeping the first
occurrence of each distinct key and preserving original row order,
matching dplyr's contract exactly.

## Usage

``` r
# S3 method for class 'tbl_gpu'
distinct(.data, ..., .keep_all = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Optional variables to use when determining uniqueness. When every dot
  is an unnamed column reference (`distinct(df, x, .data$y)`) the keys
  are those columns, looked up by name without evaluating anything. When
  any dot is named or computed (`distinct(df, z = x + 1)`), every dot is
  parsed through the same expression IR
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) uses
  (see `R/ir.R`) and any computed expressions are evaluated once via an
  internal
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) step
  before the dedup runs. If omitted, uniqueness is determined using all
  columns.

- .keep_all:

  If `TRUE`, keeps all columns (in their original order, with any newly
  computed key columns appended at the end, matching
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)'s own
  append/replace rule). If `FALSE` (the default), only the columns used
  for the uniqueness check are kept, in the order given in `...`.

## Value

A `tbl_gpu` object with duplicate rows removed.

## Details

### Column selection and order

With no `...`, every column is the dedup key and every column is kept
(`.keep_all` has no effect in this case – it's already equivalent). With
`...`, the output's columns (when `.keep_all = FALSE`) are exactly the
named/deparsed `...` expressions, in the order given – e.g.
`distinct(df, y, x)` returns columns `y, x`, not `x, y`.

### Grouped tables

If `.data` is grouped, the group columns are always part of the dedup
key and always appear in the result, even if not named in `...` – dplyr
prepends any group columns missing from `...` ahead of the `...` columns
(a group column already present in `...` keeps its given position rather
than being duplicated at the front). Grouping metadata (`$groups`) is
preserved on the result.

### NA and NaN

Two `NA` key values are considered equal (one `NA` row survives), and
two (computed) `NaN` key values are likewise considered equal – both
match
[`dplyr::distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)'s
behavior, and are cudf's own defaults for `stable_distinct()`
(`null_equality::EQUAL`, `nan_equality::ALL_EQUAL`), so no
special-casing is needed here.

## See also

[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
for creating new columns,
[`group_by.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
for grouping,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # All columns as key, original row order preserved
  result <- gpu_mtcars |> distinct() |> collect()

  # Key on specific columns, only those columns in the result
  result <- gpu_mtcars |> distinct(cyl, gear) |> collect()

  # Key on specific columns, but keep every column (first row per key)
  result <- gpu_mtcars |> distinct(cyl, .keep_all = TRUE) |> collect()
}
```
