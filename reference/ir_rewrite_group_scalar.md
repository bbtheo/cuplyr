# Rewrite window calls over one-value-per-group arguments

Bottom-up over a parsed expression; see the section comment above.

## Usage

``` r
ir_rewrite_group_scalar(ir, schema)
```

## Arguments

- ir:

  An IR node that contains a window call

- schema:

  The schema the expression was parsed against
