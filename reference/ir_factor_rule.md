# Where may a factor stay on the GPU? (one bottom-up pass)

The GPU holds only a factor's integer codes, while dplyr/base R resolve
a factor through its labels. A *factor operand* is a DICTIONARY32 column
reference, or a DICTIONARY32-typed `lit`/`vec` node (a factor value from
the environment). It may stay on the GPU only

## Usage

``` r
ir_factor_rule(ir, schema)
```

## Arguments

- ir:

  An IR node (or `NULL`)

- schema:

  List with `names`/`types` (and `factor_levels`)

## Value

`"none"` (no factor operand outside the allowed places), `"bare"` (the
root itself is a bare factor column), or `"cpu"`

## Details

1.  as a bare column reference (plain copy, key, or sort key on its
    own);

2.  directly inside [`is.na()`](https://rdrr.io/r/base/NA.html) /
    [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html);

3.  as the column of `factor_codes()`/`factor_labels()` (the parse hooks
    turn a label comparison or `as.character(f)` into these);

4.  as both operands of `==`/`!=` when both are factor *columns* with
    identical level vectors (labels, order and `ordered` flag);
    `<`/`<=`/ `>`/`>=` additionally need both ordered. (`%in%` is not
    here: the GPU's `%in%` only takes a literal set.)

Anything else that touches a factor operand (comparisons with
non-factors or other levels, arithmetic, helpers, windows, aggregates,
factor values from the environment) needs dplyr's labels, warnings and
errors, so callers send it to their CPU fallback.

Each node is visited once: a node's status is derived from its
children's statuses, never by re-walking a subtree, so the cost is
linear in IR size. IR nodes stored in a call's `meta` (e.g.
[`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)'s
`default`) count as children.
