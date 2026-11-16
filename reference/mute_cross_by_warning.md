# Evaluate `expr`, muting dplyr's `by = character()` deprecation warning

Evaluate `expr`, muting dplyr's `by = character()` deprecation warning

## Usage

``` r
mute_cross_by_warning(expr, mute = TRUE)
```

## Arguments

- expr:

  Evaluated lazily, inside the handler.

- mute:

  When `FALSE`, `expr` is just evaluated.
