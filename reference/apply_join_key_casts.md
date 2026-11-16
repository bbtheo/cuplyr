# Apply a join's key-cast plan (`by$key_casts`, see `resolve_join_key_types()`)

Merged equi keys (`keep != TRUE`, mutating joins): cast IN PLACE on both
sides – the single output key column has the common type anyway, and
right/full coalescing then fills unmatched-y rows from the cast y key.

## Usage

``` r
apply_join_key_casts(
  ast,
  left_ptr,
  right_ptr,
  left_schema,
  right_schema,
  left_pair_idx,
  right_pair_idx
)
```

## Value

`list(left_ptr, right_ptr, left_pair_idx, right_pair_idx, temps, cond_by, left_schema, right_schema)`;
`temps` is `NULL` when every cast was applied in place.

## Details

Every other cast pair – equi keys whose original column must survive
(`keep = TRUE`, and semi/anti, whose output is x verbatim) and non-equi
conditions (whose columns always keep their own types): match on cast
COPIES of just the side(s) that need a cast, appended as trailing
temporary columns. Equi pairs repoint the key indices at the copies; a
non-equi pair's condition is rewritten to name the copies (`cond_by`,
with the schemas extended by them) so the evaluator never sees
mismatched operand types.
[`drop_join_key_temps()`](https://bbtheo.github.io/cuplyr/reference/drop_join_key_temps.md)
removes the temporaries from the result. (They are dropped after the
join rather than via `right_drop` so the C++ side's right/full key
coalescing, which is gated on `right_drop`, never touches the surviving
originals.)
