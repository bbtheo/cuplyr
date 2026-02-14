# Force GPU memory cleanup

Triggers R garbage collection to free GPU memory held by unreferenced
`tbl_gpu` objects. Use this between operations when GPU memory is
limited or before large allocations.

## Usage

``` r
gpu_gc(verbose = FALSE, aggressive = TRUE)
```

## Arguments

- verbose:

  Logical. If TRUE, prints memory freed. Default FALSE.

## Value

Invisibly returns a list with memory state before and after cleanup, and
the amount freed in bytes and gigabytes.

## Details

GPU memory is automatically freed when `tbl_gpu` objects are garbage
collected by R. However, R's garbage collector doesn't know about GPU
memory pressure and may not run immediately. This function forces
garbage collection and allows time for GPU cleanup.

Call this function:

- Between benchmark iterations

- After removing large GPU objects with
  [`rm()`](https://rdrr.io/r/base/rm.html)

- When you see out-of-memory errors

- Before allocating large new GPU tables

## See also

[`gpu_memory_state`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
for checking current memory usage

## Examples

``` r
if (has_gpu()) {
  # Create and discard a GPU table
  gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))
  rm(gpu_df)

  # Force cleanup
  gpu_gc(verbose = TRUE)
}
#> GPU memory freed: 8912896 bytes
```
