# Create a filter AST node

Create a filter AST node

## Usage

``` r
ast_filter(input, predicates, grouped = FALSE)
```

## Arguments

- input:

  Input AST node

- predicates:

  List of predicate structures

- grouped:

  `TRUE` for a filter() on a grouped table, with `.by`, or on a rowwise
  table (`filter_run()` sets it; the optimizer carries it): nothing of
  such a filter is pushed into a parquet scan (R/pushdown.R)

## Value

An ast_filter node
