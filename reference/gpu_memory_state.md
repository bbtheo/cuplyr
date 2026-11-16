# Get GPU memory snapshot

Returns the current GPU memory state including total, free, and used
memory, and what cuplyr's memory resource holds. Useful for monitoring
GPU memory usage during operations.

## Usage

``` r
gpu_memory_state()
```

## Value

A list with:

- available:

  Logical: TRUE if GPU is available

- total_bytes:

  Total GPU memory in bytes

- free_bytes:

  Free GPU memory in bytes

- used_bytes:

  Used GPU memory in bytes: device level, `total - free`, for every
  process on the card. It includes everything the memory resource holds
  from the device, so freeing a table does not lower it while the
  resource keeps the memory for reuse (always under `"pool"`; under
  `"async"` until
  [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md) or
  an automatic release hands it back, see below). Use `pool_used_bytes`
  for what cuplyr's tables occupy.

- total_gb:

  Total GPU memory in gigabytes

- free_gb:

  Free GPU memory in gigabytes

- used_gb:

  Used GPU memory in gigabytes

- resource:

  `"async"` (the default), `"pool"` or `"cuda"`: the RMM memory resource
  cuplyr installed at load time (see below). `"cuda"` means none: every
  allocation is a plain `cudaMalloc`.

- pool_bytes:

  Bytes the resource currently holds from the device. Under `"pool"`
  that is the pool's size; under `"async"` it is the async pool's
  reserved memory (`cudaMemPoolAttrReservedMemCurrent`), which shrinks
  when memory is returned. `NA` when `resource` is `"cuda"`.

- pool_used_bytes:

  Bytes currently allocated from the resource to live columns and
  scratch space: what cuplyr is using, as opposed to what it reserved.
  It rises when a table is created and falls when the table is garbage
  collected. Exact under both resources. `NA` when `resource` is
  `"cuda"`.

## The memory resource

Loading cuplyr installs an RMM memory resource as the current device
resource, so the many short-lived columns a pipeline allocates are
carved out of memory reserved once instead of going through
`cudaMalloc`/`cudaFree` each time. There are three resources:

- `"async"` (default):

  RMM's `cuda_async_memory_resource`, CUDA's stream-ordered allocator
  (`cudaMallocAsync`). By default it keeps freed memory cached for
  reuse, as the pool does, and runs at the same speed (measured, see the
  vignette). Unlike the pool it can give the memory back:
  [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
  synchronises the device and returns everything the pool holds unused,
  so other processes on the card can use it. With the opt-in
  `cuplyr.release_threshold` (below) CUDA also returns unused memory by
  itself, at the next synchronisation, while the pool's whole
  reservation is above that threshold. The reservation counts tables in
  use, so a session that holds more than the threshold keeps nothing
  cached and allocates from the driver again, at about plain
  `cudaMalloc` speed.

- `"pool"`:

  RMM's `pool_memory_resource`. Same speed as async by default, but it
  keeps the high-water mark of its size until the R session ends, so the
  card's free memory (`free_bytes`) stays lower than the data you hold;
  use `pool_used_bytes` for the latter.
  [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
  cannot shrink it (`released_bytes` is 0).

- `"cuda"`:

  No resource (`cuplyr.memory_pool = FALSE`, or
  `cuplyr.memory_resource = "cuda"`): plain `cudaMalloc`/`cudaFree` for
  every allocation. Slowest per allocation; device free memory is always
  the truth, and `pool_bytes` and `pool_used_bytes` are `NA`.

Synchronisations happen at
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html),
[`print()`](https://rdrr.io/r/base/print.html) and
[`gpu_sync()`](https://bbtheo.github.io/cuplyr/reference/gpu_sync.md),
and also inside many libcudf calls (for example whenever one copies a
size or count back to the host), so an automatic release can happen in
the middle of a pipeline. See
[`vignette("memory", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/memory.md)
for measurements and advice on choosing.

The resource is chosen when the package loads, from options (which win)
or environment variables. There is one resource per R process: the
switches are read by the first load that creates it, and loading the
package again in the same session
([`pkgload::load_all()`](https://pkgload.r-lib.org/reference/load_all.html),
or a reload after unloading) reuses that resource, whatever the switches
say then, and reports it; such a load does not read the switches, so it
does not warn about them either. Restart R to change them.

Where the GPU or driver does not support `cudaMallocAsync`, the async
resource cannot be created. If async was the default (nothing chosen, or
an unknown kind), cuplyr then installs `"pool"` and gives a warning
saying so. If `"async"` was asked for by name, the warning says the
resource could not be installed and nothing is (`resource` is `"cuda"`).

- `cuplyr.memory_pool` / `CUPLYR_MEMORY_POOL`:

  `FALSE`, or `"0"`, `"false"`, `"no"`, `"off"`, installs nothing
  (`resource` is `"cuda"`, the behaviour before the pool existed) unless
  an earlier load in the session already created the resource. Anything
  else, or unset (a zero-length or `NA` option counts as unset),
  installs the resource.

- `cuplyr.memory_resource` / `CUPLYR_MEMORY_RESOURCE`:

  `"async"` (default) or `"pool"` to pick the kind of resource; `"cuda"`
  or `"none"` install nothing, like `cuplyr.memory_pool = FALSE`. Case
  and surrounding blanks do not matter. Unset or empty means async. Any
  other value gives a warning naming the option or variable and the
  value as typed, and async is used; when `cuplyr.memory_pool` switches
  the resource off, the warning says the value is ignored and no
  resource (`"cuda"`) is used.

- `cuplyr.pool_initial` / `CUPLYR_POOL_INITIAL`:

  How much the resource reserves at load, in bytes, so the first calls
  need no driver allocation. Default: a quarter of the free device
  memory, at most 1 GiB. A larger request is clamped to 90 percent of
  the free memory. `0` reserves nothing up front. Both `"pool"` and
  `"async"` grow by themselves when needed; this is not a limit and not
  a release threshold.

- `cuplyr.release_threshold` / `CUPLYR_RELEASE_THRESHOLD`:

  Async only, opt-in, in bytes; unset (the default) means no automatic
  release. It is CUDA's release threshold for the async pool: at a
  synchronisation CUDA returns unused memory to the device until the
  pool's whole reservation, memory in use included, is at or below it.
  `0` keeps nothing, the "share the card" setting. The price is speed:
  while the session holds more than the threshold, every allocation goes
  to the driver. A zero-length, `NA` or empty option counts as unset
  (the environment variable decides). Any other value that is not a
  non-negative number gives a warning naming the option or variable, and
  means unset. Ignored under `"pool"` and `"cuda"`.

All cuplyr options, and how an invalid value is handled, are listed on
[cuplyr-options](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md).

## See also

[`gpu_details`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
for device information (device level only),
[`gpu_memory_usage`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_usage.md)
for per-object memory estimates,
[cuplyr-options](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
for the options

## Examples

``` r
if (has_gpu()) {
  # Check memory before allocation
  before <- gpu_memory_state()

  # Allocate some GPU data
  gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))

  # Check memory after allocation
  after <- gpu_memory_state()

  cat("Memory used:", after$used_gb - before$used_gb, "GB\n")
  cat("Allocated from the resource:",
      after$pool_used_bytes - before$pool_used_bytes, "bytes\n")
}
#> Memory used: 0 GB
#> Allocated from the resource: 8e+06 bytes
```
