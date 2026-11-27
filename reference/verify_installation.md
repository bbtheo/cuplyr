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

- libcudf_version:

  Character: the libcudf version cuplyr was built against (such as
  `"26.10.2"`), or `NA` when it is unknown

- roundtrip_ok:

  Logical: data round-trip succeeded

- time_seconds:

  Numeric: time for the round-trip test

## Details

This function runs four checks in sequence:

1.  **Package load**:
    [`library(cuplyr)`](https://github.com/bbtheo/cuplyr) succeeds

2.  **libcudf version**: the libcudf cuplyr was built against is 26.02
    or newer (an unknown version passes)

3.  **GPU detection**:
    [`gpu_details()`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
    shows an available GPU

4.  **Round-trip**: `tbl_gpu(mtcars) |> filter(mpg > 20) |> collect()`
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
