# Refuse verbs on a zero-column table that has rows

A cudf table with no columns always has 0 rows, so a zero-column
`tbl_gpu` carries its row count on the R side (`x$nrow`, see
`new_tbl_gpu()`). Verbs lower to cudf calls that cannot see it, so
running one on such a table would silently return a table with the wrong
row count. Until zero-column tables are supported by every verb, fail
loudly instead. Called by
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md),
[`push_join()`](https://bbtheo.github.io/cuplyr/reference/push_join.md)
(both sides), the binds and `vec_proxy()`;
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
and the [`slice()`](https://dplyr.tidyverse.org/reference/slice.html)
family handle this input before it reaches
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md).

## Usage

``` r
check_zero_column_input(.data, input_schema, verb)
```

## Arguments

- .data:

  A `tbl_gpu`

- input_schema:

  `current_schema(.data)`

- verb:

  Character, the node type, for the message
