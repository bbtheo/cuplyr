# GPU bind_rows implementation

Keeps the first input's groups and `.drop` (a tbl_gpu's, or a grouped
data frame's), as
[`dplyr::bind_rows()`](https://dplyr.tidyverse.org/reference/bind_rows.html)
does. An input with no columns adds its rows as all-NA rows of the
unified schema.

## Usage

``` r
bind_rows_gpu(dots, .id = NULL, call = rlang::caller_env())
```

## Arguments

- dots:

  Inputs as flattened and checked by `bind_rows_route()`: tbl_gpu and
  data frames, named for every input when `.id` is given

- .id:

  Optional column name to identify source

- call:

  The user's call, for errors

## Value

A tbl_gpu
