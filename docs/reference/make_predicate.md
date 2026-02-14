# Create a filter predicate structure

Create a filter predicate structure

## Usage

``` r
make_predicate(col_name, op, value, is_col_compare = FALSE)
```

## Arguments

- col_name:

  Column name for LHS

- op:

  Comparison operator

- value:

  Scalar value or column name for RHS

- is_col_compare:

  TRUE if RHS is a column name

## Value

A predicate list structure
