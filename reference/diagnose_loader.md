# Diagnose dynamic linker configuration for cuplyr

Prints diagnostic information about how the cuplyr shared library will
resolve its dependencies (libstdc++, libcudf, libcuda). Useful when
[`library(cuplyr)`](https://github.com/bbtheo/cuplyr) fails with missing
symbol errors.

## Usage

``` r
diagnose_loader(verbose = FALSE)
```

## Arguments

- verbose:

  If TRUE, print full ldd output.

## Value

Invisibly returns a list with diagnostic results.

## Examples

``` r
if (FALSE) { # \dontrun{
diagnose_loader()
} # }
```
