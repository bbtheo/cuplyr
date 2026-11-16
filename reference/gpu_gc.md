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

- aggressive:

  Logical. If TRUE (default), runs multiple GC passes with short delays
  to more aggressively trigger finalizers. If FALSE, runs a lighter
  cleanup pass.

## Value

Invisibly returns a list with memory state before and after cleanup, and
the amount freed in bytes and gigabytes. `freed_bytes` is the drop in
`pool_used_bytes` (allocator level) when cuplyr's memory resource is
installed, since the device-level free memory does not move while the
resource keeps its pages; with `resource == "cuda"` it is the rise in
device free memory. `released_bytes` is what the resource gave back to
the device: the drop in `pool_bytes` across the call. Under `"async"`
the call ends by trimming the async pool, so it is the memory the pool
held unused (0 or more); under `"pool"` it is 0 (RMM's pool never
shrinks); under `"cuda"` it is `NA`.

## Details

GPU memory is automatically freed when `tbl_gpu` objects are garbage
collected by R. However, R's garbage collector doesn't know about GPU
memory pressure and may not run immediately. This function forces
garbage collection and allows time for GPU cleanup.

A cuplyr call you make (an eager verb,
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html),
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html), a
bind) frees the temporary GPU tables its own steps built when it
returns. What a call returns is yours and is never freed early,
including a table that your code creates while a verb evaluates its
arguments or expressions. So this function is for tables you created and
dropped without R having collected them yet: the
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
result of a one-line pipeline lives until R's next collection. So does
the inner result of a dplyr verb that dplyr itself builds from several
verb calls, such as `add_tally(sort = TRUE)`. When a GPU step runs out
of device memory, cuplyr runs [`gc()`](https://rdrr.io/r/base/gc.html)
and `gpu_gc(aggressive = FALSE)` itself and retries that step once; a
second failure raises an error of class `cuplyr_error_gpu_oom`.

Under the default `"async"` resource, this function also synchronises
the device and returns all unused memory of cuplyr's async pool to the
device, so other processes can use it (see
[`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
and
[`vignette("memory", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/memory.md)).
The pool is then smaller than the session's recent peak, so the next
calls grow it again from the driver; that costs the first call after
`gpu_gc()` a few percent. If the device is in a CUDA error state, the
return step cannot run: `gpu_gc()` then gives a warning instead of an
error and leaves the pool as it is, so it stays safe in cleanup code.

Call this function:

- After removing large GPU objects with
  [`rm()`](https://rdrr.io/r/base/rm.html), when other processes need
  the memory (under `"async"` it returns what the pool holds unused)

- When you see out-of-memory errors

- Before allocating large new GPU tables

It is not meant for the loop of a benchmark: under `"async"` the trim
makes the next call grow the pool again.

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
#> GPU memory freed: 8002816 bytes (allocator level)
#> Returned to the device: 1.04 GB
```
