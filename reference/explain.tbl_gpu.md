# Explain a tbl_gpu query

`tbl_gpu` method for dplyr's
[`dplyr::explain()`](https://dplyr.tidyverse.org/reference/explain.html)
generic. cuplyr has no query language apart from the plan, so it prints
the same text as
[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html);
see there for the format. Nothing is executed.

## Usage

``` r
# S3 method for class 'tbl_gpu'
explain(x, ..., optimized = TRUE)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Must be empty.

- optimized:

  `TRUE` (the default) prints the plan the optimizer will run, `FALSE`
  the plan as written.

## Value

`x`, invisibly.
