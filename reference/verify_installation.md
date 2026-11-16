# Verify cuplyr installation

Performs a post-install runtime verification by running a small GPU
round-trip: creates a GPU table, performs a filter, and collects back to
R. This confirms that the package loads, GPU is accessible, and basic
operations work.

## Usage

``` r
verify_installation()
```

## Value

Invisibly returns a list with:

- ok:

  Logical: `TRUE` if all checks pass

- package_loaded:

  Logical: package loads without error

- gpu_available:

  Logical: GPU detected and accessible

- gpu_name:

  Character: GPU model name, or `NA`

- roundtrip_ok:

  Logical: data round-trip succeeded

- time_seconds:

  Numeric: time for the round-trip test

## Details

This function runs three checks in sequence:

1.  **Package load**:
    [`library(cuplyr)`](https://github.com/bbtheo/cuplyr) succeeds

2.  **GPU detection**:
    [`gpu_details()`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
    shows an available GPU

3.  **Round-trip**: `tbl_gpu(mtcars) |> filter(mpg > 20) |> collect()`
    returns correct results

## See also

[`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
for pre-install dependency checks,
[`diagnostics()`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)
for full system diagnostics

## Examples

``` r
if (FALSE) { # \dontrun{
result <- verify_installation()
if (result$ok) {
  message("cuplyr is working!")
}
} # }
```
