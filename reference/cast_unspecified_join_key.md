# Cast a checked all-NA logical join key to the other side's type (#70)

Replaces the column (see
[`check_unspecified_join_key()`](https://bbtheo.github.io/cuplyr/reference/check_unspecified_join_key.md))
in place with a null column of the other side's physical type.

## Usage

``` r
cast_unspecified_join_key(ptr, idx, cast)
```
