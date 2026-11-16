# Friendly type label for set-op incompatible-type error messages

cuplyr's GPU type strings (`INT32`, `FLOAT64`, ...) don't correspond 1:1
to R's/vctrs' own type vocabulary (`vec_ptype_full()`), so this is a
hand-picked "close enough" label used only in error text – not an
attempt to byte-match `vec_ptype_full()`'s output (which for factors
includes a levels-derived hash we have no equivalent for).

## Usage

``` r
set_op_type_label(type, is_factor = FALSE)
```
