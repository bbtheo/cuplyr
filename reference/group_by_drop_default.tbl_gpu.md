# Report whether `group_by()` drops unobserved factor levels by default

[`dplyr::group_by_drop_default()`](https://dplyr.tidyverse.org/reference/group_by_drop_default.html)
is a generic dplyr's own internals call to decide the default `.drop=`
value for verbs built on top of
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
(e.g.
[`group_trim()`](https://dplyr.tidyverse.org/reference/group_trim.html)'s
own default, see below). The `.default` method always returns `TRUE`;
this method returns `TRUE` for an ungrouped table and otherwise
whichever value the table's own most recent `group_by(..., .drop=)` call
recorded (`$group_drop`, see
[`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)'s
own "`.drop` semantics" docs for the scope of what's tracked).

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_by_drop_default(.tbl)
```

## Arguments

- .tbl:

  A `tbl_gpu` object.

## Value

`TRUE` or `FALSE`.
