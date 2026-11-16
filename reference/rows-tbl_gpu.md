# Row-wise modification of a `tbl_gpu`

[`rows_insert()`](https://dplyr.tidyverse.org/reference/rows.html),
[`rows_append()`](https://dplyr.tidyverse.org/reference/rows.html),
[`rows_update()`](https://dplyr.tidyverse.org/reference/rows.html),
[`rows_patch()`](https://dplyr.tidyverse.org/reference/rows.html),
[`rows_upsert()`](https://dplyr.tidyverse.org/reference/rows.html) and
[`rows_delete()`](https://dplyr.tidyverse.org/reference/rows.html) for
`tbl_gpu`. Tables whose columns are logical, integer, double, character,
Date or POSIXct (a `y` column may also be a lossless widening of `x`'s:
integer to double, logical to integer or double) run on the GPU.
Anything else (factors, other casts, a zone that differs from `x`'s) and
every data-dependent error runs dplyr's own method on the CPU, which
gives dplyr's result or its exact error. Like the GPU joins, a `y` that
is not a `tbl_gpu` needs `copy = TRUE`. A lazy `x` is computed at call
time (the verbs are not nodes of the lazy plan), and the result stays
lazy.

## Usage

``` r
# S3 method for class 'tbl_gpu'
rows_insert(
  x,
  y,
  by = NULL,
  ...,
  conflict = c("error", "ignore"),
  copy = FALSE,
  in_place = FALSE
)

# S3 method for class 'tbl_gpu'
rows_append(x, y, ..., copy = FALSE, in_place = FALSE)

# S3 method for class 'tbl_gpu'
rows_update(
  x,
  y,
  by = NULL,
  ...,
  unmatched = c("error", "ignore"),
  copy = FALSE,
  in_place = FALSE
)

# S3 method for class 'tbl_gpu'
rows_patch(
  x,
  y,
  by = NULL,
  ...,
  unmatched = c("error", "ignore"),
  copy = FALSE,
  in_place = FALSE
)

# S3 method for class 'tbl_gpu'
rows_upsert(x, y, by = NULL, ..., copy = FALSE, in_place = FALSE)

# S3 method for class 'tbl_gpu'
rows_delete(
  x,
  y,
  by = NULL,
  ...,
  unmatched = c("error", "ignore"),
  copy = FALSE,
  in_place = FALSE
)
```

## Arguments

- x:

  A `tbl_gpu`.

- y:

  A `tbl_gpu`, or a data frame with `copy = TRUE`.

- by, conflict, unmatched, copy, in_place:

  As in
  [`dplyr::rows_insert()`](https://dplyr.tidyverse.org/reference/rows.html).

- ...:

  Must be empty.
