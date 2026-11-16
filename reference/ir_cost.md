# Estimate the relative evaluation cost of an IR subtree

Used by
[`reorder_filters()`](https://bbtheo.github.io/cuplyr/reference/reorder_filters.md)
to run cheaper predicates first. Literal = 1, column reference = 2, and
every call node adds 1 on top of the summed cost of its arguments – so a
scalar comparison (`x > 5`, cost 4) is cheaper than a column-column
comparison (`x > y`, cost 5), which in turn is cheaper than any deeper
nesting.

## Usage

``` r
ir_cost(ir)
```

## Arguments

- ir:

  An IR node (or `NULL`)

## Value

Integer cost
