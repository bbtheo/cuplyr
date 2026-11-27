# Grouping metadata: keys, row indices, per-row group id, sizes, and count

[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
(and the queries derived from it below) report the *grouping structure*
that dplyr's `grouped_df` carries as an attribute. A `tbl_gpu` has no
such attribute, so `cuplyr` computes it on demand with one GPU groupby.
All of them run on the GPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_data(.data)

# S3 method for class 'tbl_gpu'
group_keys(.tbl, ...)

# S3 method for class 'tbl_gpu'
group_indices(.data, ...)

# S3 method for class 'tbl_gpu'
group_size(x)

# S3 method for class 'tbl_gpu'
n_groups(x)
```

## Arguments

- .data, x, .tbl:

  A `tbl_gpu` object.

- ...:

  For
  [`group_keys()`](https://dplyr.tidyverse.org/reference/group_data.html)/[`group_indices()`](https://dplyr.tidyverse.org/reference/group_data.html):
  no longer accepted (matching dplyr 1.0.0+, where the `...` argument of
  both was deprecated then made defunct – passing anything here is an
  error).

## Details

[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
returns a tibble of the grouping columns (in ascending key order, `NA`
sorted last, as in dplyr) plus a `.rows` list-of column of 1-based row
indices; called on an ungrouped table it returns a single row whose
`.rows` holds every row, in original order.
[`group_keys()`](https://dplyr.tidyverse.org/reference/group_data.html)
is
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
minus `.rows`;
[`group_rows()`](https://dplyr.tidyverse.org/reference/group_data.html)
is just `.rows`;
[`group_indices()`](https://dplyr.tidyverse.org/reference/group_data.html)
is the per-ROW group id (1-based, matching
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)'s
row order);
[`group_size()`](https://dplyr.tidyverse.org/reference/group_data.html)
is the per-group row count
([`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
order);
[`n_groups()`](https://dplyr.tidyverse.org/reference/group_data.html) is
the group count.
