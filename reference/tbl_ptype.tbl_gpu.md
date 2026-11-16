# Prototype of a GPU table for dplyr's `_if` verbs

`dplyr::tbl_if_vars()` (behind
[`mutate_if()`](https://dplyr.tidyverse.org/reference/mutate_all.html),
[`select_if()`](https://dplyr.tidyverse.org/reference/select_all.html),
...) calls
[`tbl_ptype()`](https://dplyr.tidyverse.org/reference/tbl_ptype.html)
and then [`pull()`](https://dplyr.tidyverse.org/reference/pull.html)s
every column to test the predicate on. A type-only predicate is answered
by a one-row proxy of the column classes, with no GPU read; any other
predicate gets the data, downloaded once. Any other caller of the
generic gets `.data` back unchanged.

## Usage

``` r
# S3 method for class 'tbl_gpu'
tbl_ptype(.data)
```

## Arguments

- .data:

  A `tbl_gpu`.

## Value

A data frame, or `.data`.
