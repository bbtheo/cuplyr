# Create an arrange AST node

Create an arrange AST node

## Usage

``` r
ast_arrange(input, sort_specs, groups = character())
```

## Arguments

- input:

  Input AST node

- sort_specs:

  List of sort specifications (col_name, descending)

- groups:

  Character vector of group columns (for .by_group)

## Value

An ast_arrange node
