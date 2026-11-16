# Drop the temporary key copies `apply_join_key_casts()` appended

A mutating join's columns are
`[left cols, left temps, right cols kept, right temps]`; semi/anti
return `[left cols, left temps]`.

## Usage

``` r
drop_join_key_temps(ptr, temps)
```
