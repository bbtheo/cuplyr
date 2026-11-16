# Friendly, `vctrs`-bracket-style type label for a `bind_rows()` incompatible-type error (Phase 11 L4)

Mirrors real dplyr/vctrs' own `<double>`/`<character>`/`<date>`/
`<datetime<UTC>>` labels (verified empirically:
`bind_rows(data.frame(x = 1), data.frame(x = as.Date("2020-01-01")))`
-\> `` Can't combine `..1$x` <double> and `..2$x` <date>. ``). The one
label that can't be byte-matched is a factor's: real
[`vctrs::vec_ptype_full()`](https://vctrs.r-lib.org/reference/vec_ptype_full.html)
embeds a levels-derived hash (`<factor<4d52a>>`) this package has no
equivalent for –
[`set_op_type_label()`](https://bbtheo.github.io/cuplyr/reference/set_op_type_label.md)
(R/sets.R) documents the identical limitation for its own,
differently-formatted set-op error text.

## Usage

``` r
bind_rows_type_label(type, is_factor = FALSE)
```
