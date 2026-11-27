# vctrs-style type label for a join key, as in dplyr's incompatible-type error

Built from
[`vctrs::vec_ptype_full()`](https://vctrs.r-lib.org/reference/vec_ptype_full.html)
on a zero-length prototype so labels (including the levels hash in
`<factor<4d52a>>`) match dplyr's exactly. `tz` is a POSIXct key's zone
(`""` is local time, as vctrs labels it).

## Usage

``` r
join_key_type_label(type, levels = NULL, tz = "UTC")
```
