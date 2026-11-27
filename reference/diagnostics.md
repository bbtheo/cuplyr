# System diagnostics for cuplyr

Collects comprehensive system information for debugging installation
issues or filing bug reports. Includes cuplyr version, R/Rcpp versions,
CUDA/cuDF paths, libcudf versions, GPU info, the GPU memory resource,
the cuplyr options in effect and environment variables.

## Usage

``` r
diagnostics(redact = TRUE, format = c("text", "json"))
```

## Arguments

- redact:

  If `TRUE` (default), redact potentially sensitive paths (home
  directory, usernames). Set to `FALSE` for local debugging.

- format:

  `"text"` (default) prints a readable report. `"json"` prints one JSON
  object instead (and nothing else), for tools and scripts; its keys are
  the names of the returned list.

## Value

Invisibly returns a list with all diagnostic information, whichever
`format` is used. Also prints the report (or the JSON object) to the
console. Besides the package, system, CUDA and GPU fields, the list has:

- `libcudf_version` and `rmm_version`: the versions cuplyr was built
  against, as `"26.10.2"`.

- `libcudf_path`: the libcudf library loaded into this R session (`NA`
  if none), and `libcudf_loaded_version`, its version (`NA` when its
  headers can't be found). A difference from `libcudf_version` means
  cuplyr runs against another libcudf than it was built with. The
  printed report adds a `WARNING` line when either version is older than
  26.02, the oldest libcudf cuplyr supports.

- `memory`: a list with `available`, the memory `resource` (`"async"`,
  `"pool"` or `"cuda"`), `pool_bytes` and `pool_used_bytes`. See
  [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md).

- `options`: a data frame with one row per cuplyr option, user and
  internal: `option`, `value`, `default`, `source`, `env_var`,
  `internal` and `description`. `default` is `NA` both for a default
  computed at run time (the initial pool size) and for no value at all
  (no release threshold); the text report says "computed" or "unset".
  See
  [`cuplyr_options()`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md).

A lookup that fails gives `NA` for its field; `diagnostics()` itself
does not fail.

## Details

The output includes:

- **Package versions**: cuplyr, R, Rcpp, dplyr

- **CUDA info**: toolkit version, driver version, CUDA_HOME path

- **libcudf info**: CONDA_PREFIX, CUDF_HOME, the libcudf and RMM
  versions cuplyr was built against, the libcudf loaded and its version

- **GPU info**: device name, compute capability, memory

- **GPU memory resource**: the active resource and its pool size

- **Options**: every cuplyr option with its value, source and default

- **Library paths**: LD_LIBRARY_PATH, RPATH of cuplyr.so

- **Environment**: OS, distro, key environment variables

## See also

[`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
for pre-install checks,
[`verify_installation()`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)
for post-install verification,
[`cuplyr_options()`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
for the options

## Examples

``` r
# Full diagnostics
diagnostics()
#> cuplyr diagnostics
#> ================================================== 
#> 
#> Package Versions
#>   cuplyr:     0.2.0 
#>   R:         4.6.1 (x86_64-pc-linux-gnu)
#>   Rcpp:       1.1.2 
#>   dplyr:      1.2.1 
#> 
#> System
#>   OS:         Linux 
#>   Distro:     Omarchy 
#> 
#> CUDA
#>   CUDA_HOME:  /opt/cuda 
#>   CUDA ver:   13.4 
#>   Driver:     615.78.08 
#> 
#> RAPIDS / libcudf
#>   CONDA_PREFIX:  ~/cuplr/.pixi/envs/default 
#>   CUDF_HOME:     ~/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf 
#>   libcudf (built against): 26.10.2
#>   libcudf (loaded):        26.10.2  ~/cuplr/.pixi/envs/default/lib/libcudf.so
#>   RMM (built against):     26.10.0
#> 
#> GPU
#>   Available:  YES
#>   Name:       NVIDIA GeForce RTX 5070 
#>   Compute:    12.0 
#>   Memory:     11.6 GiB total,  9.7 GiB free
#> 
#> Memory resource
#>   Memory resource: async
#>   Pool held:       1 GiB
#>   Pool in use:     0 GiB
#> 
#> Options
#>   cuplyr.exec_mode = lazy (default; default lazy)
#>   cuplyr.fallback = silent (default; default silent)
#>   cuplyr.plan_break = silent (default; default silent)
#>   cuplyr.memory_resource = async (default; default async)
#>   cuplyr.memory_pool = TRUE (default; default TRUE)
#>   cuplyr.pool_initial = <unset> (default; default computed)
#>   cuplyr.release_threshold = <unset> (default; default unset)
#>   cuplyr.parquet_pushdown = TRUE (default; default TRUE)
#>   cuplyr.roll_median = exact (default; default exact)
#>   cuplyr.roll_path = auto (default; default auto) [internal]
#>   cuplyr.topk_path = auto (default; default auto) [internal]
#>   cuplyr.check_schema_cache = FALSE (default; default FALSE) [internal]
#> 
#> Library Paths
#>   LD_LIBRARY_PATH:  /usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server 
#>   cuplyr.so:        not found 
#> 
#> ================================================== 
#> Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues

# Redacted for sharing in bug reports
diagnostics(redact = TRUE)
#> cuplyr diagnostics
#> ================================================== 
#> 
#> Package Versions
#>   cuplyr:     0.2.0 
#>   R:         4.6.1 (x86_64-pc-linux-gnu)
#>   Rcpp:       1.1.2 
#>   dplyr:      1.2.1 
#> 
#> System
#>   OS:         Linux 
#>   Distro:     Omarchy 
#> 
#> CUDA
#>   CUDA_HOME:  /opt/cuda 
#>   CUDA ver:   13.4 
#>   Driver:     615.78.08 
#> 
#> RAPIDS / libcudf
#>   CONDA_PREFIX:  ~/cuplr/.pixi/envs/default 
#>   CUDF_HOME:     ~/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf 
#>   libcudf (built against): 26.10.2
#>   libcudf (loaded):        26.10.2  ~/cuplr/.pixi/envs/default/lib/libcudf.so
#>   RMM (built against):     26.10.0
#> 
#> GPU
#>   Available:  YES
#>   Name:       NVIDIA GeForce RTX 5070 
#>   Compute:    12.0 
#>   Memory:     11.6 GiB total,  9.7 GiB free
#> 
#> Memory resource
#>   Memory resource: async
#>   Pool held:       1 GiB
#>   Pool in use:     0 GiB
#> 
#> Options
#>   cuplyr.exec_mode = lazy (default; default lazy)
#>   cuplyr.fallback = silent (default; default silent)
#>   cuplyr.plan_break = silent (default; default silent)
#>   cuplyr.memory_resource = async (default; default async)
#>   cuplyr.memory_pool = TRUE (default; default TRUE)
#>   cuplyr.pool_initial = <unset> (default; default computed)
#>   cuplyr.release_threshold = <unset> (default; default unset)
#>   cuplyr.parquet_pushdown = TRUE (default; default TRUE)
#>   cuplyr.roll_median = exact (default; default exact)
#>   cuplyr.roll_path = auto (default; default auto) [internal]
#>   cuplyr.topk_path = auto (default; default auto) [internal]
#>   cuplyr.check_schema_cache = FALSE (default; default FALSE) [internal]
#> 
#> Library Paths
#>   LD_LIBRARY_PATH:  /usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server 
#>   cuplyr.so:        not found 
#> 
#> ================================================== 
#> Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues

# Unredacted for local debugging
diagnostics(redact = FALSE)
#> cuplyr diagnostics
#> ================================================== 
#> 
#> Package Versions
#>   cuplyr:     0.2.0 
#>   R:         4.6.1 (x86_64-pc-linux-gnu)
#>   Rcpp:       1.1.2 
#>   dplyr:      1.2.1 
#> 
#> System
#>   OS:         Linux 
#>   Distro:     Omarchy 
#> 
#> CUDA
#>   CUDA_HOME:  /opt/cuda 
#>   CUDA ver:   13.4 
#>   Driver:     615.78.08 
#> 
#> RAPIDS / libcudf
#>   CONDA_PREFIX:  /home/theo/cuplr/.pixi/envs/default 
#>   CUDF_HOME:     /home/theo/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf 
#>   libcudf (built against): 26.10.2
#>   libcudf (loaded):        26.10.2  /home/theo/cuplr/.pixi/envs/default/lib/libcudf.so
#>   RMM (built against):     26.10.0
#> 
#> GPU
#>   Available:  YES
#>   Name:       NVIDIA GeForce RTX 5070 
#>   Compute:    12.0 
#>   Memory:     11.6 GiB total,  9.7 GiB free
#> 
#> Memory resource
#>   Memory resource: async
#>   Pool held:       1 GiB
#>   Pool in use:     0 GiB
#> 
#> Options
#>   cuplyr.exec_mode = lazy (default; default lazy)
#>   cuplyr.fallback = silent (default; default silent)
#>   cuplyr.plan_break = silent (default; default silent)
#>   cuplyr.memory_resource = async (default; default async)
#>   cuplyr.memory_pool = TRUE (default; default TRUE)
#>   cuplyr.pool_initial = <unset> (default; default computed)
#>   cuplyr.release_threshold = <unset> (default; default unset)
#>   cuplyr.parquet_pushdown = TRUE (default; default TRUE)
#>   cuplyr.roll_median = exact (default; default exact)
#>   cuplyr.roll_path = auto (default; default auto) [internal]
#>   cuplyr.topk_path = auto (default; default auto) [internal]
#>   cuplyr.check_schema_cache = FALSE (default; default FALSE) [internal]
#> 
#> Library Paths
#>   LD_LIBRARY_PATH:  /usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server 
#>   cuplyr.so:        not found 
#> 
#> ================================================== 
#> Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues

# As JSON, for scripts
diagnostics(format = "json")
#> {"cuplyr_version":"0.2.0","r_version":"4.6.1","r_platform":"x86_64-pc-linux-gnu","rcpp_version":"1.1.2","dplyr_version":"1.2.1","os":"Linux","os_release":"Omarchy","cuda_home":"/opt/cuda","cuda_version":"13.4","driver_version":"615.78.08","conda_prefix":"~/cuplr/.pixi/envs/default","cudf_home":"~/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf","libcudf_version":"26.10.2","rmm_version":"26.10.0","libcudf_path":"~/cuplr/.pixi/envs/default/lib/libcudf.so","libcudf_loaded_version":"26.10.2","gpu":{"available":true,"device_count":1,"device_id":0,"name":"NVIDIA GeForce RTX 5070","compute_capability":"12.0","total_memory":12441747456,"free_memory":10438049792,"multiprocessors":48},"memory":{"available":true,"resource":"async","pool_bytes":1073741824,"pool_used_bytes":5632},"options":[{"option":"cuplyr.exec_mode","value":"lazy","default":"lazy","source":"default","env_var":"CUPLYR_EXEC_MODE","internal":false,"description":"Execution mode of new tables: lazy plans (default) or eager, one verb at a time."},{"option":"cuplyr.fallback","value":"silent","default":"silent","source":"default","env_var":null,"internal":false,"description":"What a CPU fallback does: stay silent, warn, or error."},{"option":"cuplyr.plan_break","value":"silent","default":"silent","source":"default","env_var":null,"internal":false,"description":"What a plan break does: stay silent or warn."},{"option":"cuplyr.memory_resource","value":"async","default":"async","source":"default","env_var":"CUPLYR_MEMORY_RESOURCE","internal":false,"description":"GPU memory resource: async (default), pool, or cuda (none). Read at load."},{"option":"cuplyr.memory_pool","value":"TRUE","default":"TRUE","source":"default","env_var":"CUPLYR_MEMORY_POOL","internal":false,"description":"Set FALSE (or 0, no, off) to use no memory resource. Read at load."},{"option":"cuplyr.pool_initial","value":null,"default":null,"source":"default","env_var":"CUPLYR_POOL_INITIAL","internal":false,"description":"Initial pool size in bytes (default: a quarter of free GPU memory, at most 1 GiB). Read at load."},{"option":"cuplyr.release_threshold","value":null,"default":null,"source":"default","env_var":"CUPLYR_RELEASE_THRESHOLD","internal":false,"description":"Bytes the async resource keeps before returning memory to the driver (unset: never). Read at load."},{"option":"cuplyr.parquet_pushdown","value":"TRUE","default":"TRUE","source":"default","env_var":null,"internal":false,"description":"Set FALSE to read Parquet files without projection and predicate pushdown."},{"option":"cuplyr.roll_median","value":"exact","default":"exact","source":"default","env_var":"CUPLYR_ROLL_MEDIAN","internal":false,"description":"Engine of roll_median(): exact (default) or frollmedian (data.table)."},{"option":"cuplyr.roll_path","value":"auto","default":"auto","source":"default","env_var":"CUPLYR_ROLL_PATH","internal":true,"description":"Internal: force the rolling-window path (auto, scans or cudf)."},{"option":"cuplyr.topk_path","value":"auto","default":"auto","source":"default","env_var":null,"internal":true,"description":"Internal: force the slice_min()/slice_max() route (auto, topk or sort)."},{"option":"cuplyr.check_schema_cache","value":"FALSE","default":"FALSE","source":"default","env_var":"CUPLYR_CHECK_SCHEMA_CACHE","internal":true,"description":"Internal: verify the cached schema of a lazy table on every use (debug)."}],"ld_library_path":"/usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server","cuplyr_so_path":"not found","cuplyr_runpath":null}
```
