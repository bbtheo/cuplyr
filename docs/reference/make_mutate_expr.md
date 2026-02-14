# Create a mutate expression structure

Create a mutate expression structure

## Usage

``` r
make_mutate_expr(output_col, input_cols, op, scalar = NULL, input_types = NULL)
```

## Arguments

- output_col:

  Output column name

- input_cols:

  Character vector of input column names

- op:

  Operation: "+", "-", "\*", "/", "^", "copy", or function name

- scalar:

  Numeric scalar or NULL

- input_types:

  Character vector of input column types

## Value

An expression list structure
