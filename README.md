# cuplyr <img src="man/figures/logo.png" align="right" height="138" />

#### dplyr backend for GPU acceleration via RAPIDS cuDF

cuplyr implements a dplyr backend powered by [RAPIDS cuDF](https://github.com/rapidsai/cudf), NVIDIA's GPU DataFrame library. Write standard dplyr code, execute on GPU hardware.

```r
library(cuplyr)
library(dplyr)

# Example data, built here so the code runs as written
set.seed(1)
n <- 1e5
sales_data <- data.frame(
  year    = sample(2018:2024, n, replace = TRUE),
  region  = sample(c("north", "south", "east", "west"), n, replace = TRUE),
  quarter = sample(1:4, n, replace = TRUE),
  amount  = rnorm(n, mean = 50, sd = 30),
  price   = runif(n, 1, 10)
)
regions <- data.frame(
  region  = c("north", "south", "east", "west"),
  manager = c("Ana", "Ben", "Chi", "Dev")
)

tbl_gpu(sales_data) |>
  filter(year >= 2020, amount > 0) |>
  mutate(revenue = amount * price) |>
  group_by(region, quarter) |>
  summarise(total = sum(revenue), .groups = "drop") |>
  inner_join(tbl_gpu(regions), by = "region") |>
  arrange(desc(total)) |>
  collect()
```

## About

cuplyr translates dplyr operations into cuDF execution on NVIDIA GPUs. It follows the same backend pattern as dbplyr: write standard R code, execute on GPU hardware. This approach can provide significant speedups on larger datasets (typically >1M rows) without requiring major code changes.

`tbl_gpu()` makes cuplyr's GPU tables, called gibbles (they print as `# A gibble:` followed by the dimensions, `# A gibble [lazy]:` for a lazy table with pending steps). `gibble()`, `as_gibble()` and `is_gibble()` are aliases of `tbl_gpu()`, `as_tbl_gpu()` and `is_tbl_gpu()`; the class stays `tbl_gpu`. The CRAN package gibble also exports a `gibble()`: if both are attached, the last one attached masks the other, and `cuplyr::gibble()` always works.

**Built on [RAPIDS cuDF](https://rapids.ai/)**: cuDF is an open-source GPU DataFrame library developed by NVIDIA's RAPIDS team. It provides optimized CUDA kernels for data manipulation operations, backed by Apache Arrow's columnar memory format. cuplyr provides an R interface to this execution engine.

## Status

**v0.1.0**

This is experimental software under active development. Breaking changes should be expected.

### Supported operations

**Data manipulation**
- `filter()` – row filtering with comparison and logical operators
- `select()` – column selection and reordering
- `mutate()` – column transformations and arithmetic
- `arrange()` – row sorting with `desc()` support, NA handling follows dplyr conventions
- `group_by()` + `summarise()` – grouped aggregations (`sum`, `mean`, `min`, `max`, `n`)
- `left_join()`, `right_join()`, `inner_join()`, `full_join()` – GPU joins on key columns
- `collect()` – transfer results back to R
- `compute()` – execute pending operations, keep on GPU
- `tbl_gpu(..., lazy = FALSE)` – run each verb at once (eager) instead of the default lazy plan

### Lazy by default, eager on request

`tbl_gpu()` is lazy by default. Each verb checks its arguments, works out the new schema and records a step; nothing runs on the GPU until you need the data. `collect()` (result to R) or `compute()` (result stays on the GPU) optimizes the recorded plan, runs it, and frees every intermediate column. Errors in your code (a missing column, a type error) still appear at the verb, not at `collect()`.

Printing a lazy table with pending steps runs the plan and shows the first rows, under a `# A gibble [lazy]:` header. The table stays lazy and keeps nothing: printing it twice runs the plan twice, so `compute()` it first if you want to look at it repeatedly. At 1e7 rows, `print()` of a five-verb pipeline took about as long as `collect()` of it (6.6 against 5.9 ms median of 5, for a 100-row result).

Why lazy is the default: it holds no GPU memory between calls, and a single pipeline takes within a millisecond of eager. Each number below is the median of 15 runs (single) or 7 runs of five calls (back to back, ms per call) of a five-verb pipeline (`filter() |> mutate() |> group_by() |> summarise() |> arrange() |> collect()`) with the memory pool cleaned before each run; on this machine (RTX 5070):

| rows | eager ms | lazy ms | five back to back, eager ms each | five back to back, lazy ms each |
|---|---|---|---|---|
| 1e3 | 2.66 | 2.53 | 2.17 | 2.07 |
| 1e6 | 2.99 | 2.85 | 2.41 | 2.28 |
| 1e7 | 5.22 | 5.90 | 4.76 | 5.25 |
| 5e7 | 18.7 | 19.6 | 66.7 | 18.5 |

So with clean GPU memory lazy is level with eager, or a little faster, up to 1e6 rows, and 0.7 to 1.0 ms slower on a single pipeline at 1e7 and 5e7 rows (0.5 ms at 1e7 when five run back to back). The difference is that an eager pipeline leaves its intermediates on the GPU until R's next `gc()`. If you never call `gc()`, they pile up, and once they fill the card cuplyr's out-of-memory handler has to run: at 5e7 rows five eager pipelines run back to back, with no `gc()` between them, took 66.7 ms each against 18.5 ms lazy. The optimizer also drops columns the result never uses. (`vignette("performance")` has the detailed tables, timed with a plain `gc()` before each batch rather than a cleaned pool, and the cases where the GPU loses to dplyr.)

To run each verb at once instead, which can be easier to step through:

```r
# Per table
eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)

# Or for every table created afterwards
options(cuplyr.exec_mode = "eager")   # or the environment variable CUPLYR_EXEC_MODE=eager

# Pending steps, then the plan the optimizer will run
tbl_gpu(mtcars) |> filter(mpg > 20) |> mutate(kpl = mpg * 0.425) |> select(cyl, kpl) |> explain()
```

The optimizer applies, in this order:
- Projection pruning (drop unused columns early)
- Mutate fusion (combine consecutive `mutate()` calls, within limits)
- Dead-column pruning (drop mutate outputs nothing uses)
- Filter pushdown (move filters closer to the source: below a `mutate()` that doesn't compute their columns, below a `select()` that keeps them, and into some joins)
- Filter reordering (merge consecutive `filter()` calls into one filter that selects the rows once, in most cases, with cheaper predicates first)

The timings, the memory numbers and the cases where the GPU loses are in `vignette("performance", package = "cuplyr")`.

### Supported column types

| R Type | GPU Type |
|--------|----------|
| numeric (double) | FLOAT64 |
| integer | INT32 |
| character | STRING |
| logical | BOOL8 |
| Date | TIMESTAMP_DAYS |
| POSIXct | TIMESTAMP_MICROSECONDS |
| factor | INT32 (codes) |

### Not yet implemented

- Complex joins with `join_by()`
- Window functions
- String operations
- Multi-GPU support

Contributions and feedback are welcome.

## Architecture

- **R layer**: S3 methods implementing dplyr generics
- **AST optimizer**: Projection pruning, filter pushdown, operation fusion
- **Native bindings**: Rcpp interface to libcudf C++ API
- **Execution**: cuDF GPU kernels via libcudf
- **Memory**: GPU-resident data with automatic cleanup via R garbage collection

## Installation

### Which path is right for me?

| I want to... | Do this |
|--------------|---------|
| **Use it** (I have an NVIDIA GPU) | [Quick install](#quick-install) |
| **Contribute** (modify C++/R code) | [Developer setup](#developer-setup) |

### Requirements

| Component | Version |
|-----------|---------|
| NVIDIA GPU | Compute Capability >= 7.0 (Volta+) |
| CUDA Toolkit | >= 12.2 |
| RAPIDS libcudf | >= 25.12 |
| R | >= 4.3 |
| OS | Linux x86_64 only |

### Quick install

**Option A: One-liner** (auto-detects pixi, conda, or system CUDA)

```bash
git clone https://github.com/bbtheo/cuplyr.git && cd cuplyr && ./install.sh
```

**Option B: From R** (if you have CUDA + cuDF on your system)

```r
# Install R dependencies first
install.packages(c("Rcpp", "dplyr", "rlang", "vctrs", "pillar", "glue", "cli", "tidyselect", "tibble"))

# Then from the cuplyr directory:
cuplyr::install_cuplyr(method = "system")
```

**Option C: Using pixi** (reproducible, manages all CUDA/RAPIDS deps)

```bash
# Install pixi: curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/bbtheo/cuplyr.git
cd cuplyr
pixi run install
```

### Verify installation

```r
library(cuplyr)
verify_installation()
# Or check dependencies first:
check_deps()
```

### Troubleshooting

```r
# Full diagnostics for bug reports
diagnostics()
```

### Developer setup

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor workflow using `pixi shell`.
GitHub Actions trigger/run instructions are in [docs/github-actions-runbook.md](docs/github-actions-runbook.md).

## Performance

Benchmark code lives in `benchmark/benchmark.R`. For eager vs lazy timings, per-call and transfer costs, and where the GPU loses to dplyr, see `vignette("performance", package = "cuplyr")`.

Benchmarks on 25 million rows (synthetic taxi data, median of 10 iterations):

| Operation | dplyr | data.table | DuckDB | cuplyr | cuplyr vs dplyr | cuplyr vs data.table | cuplyr vs DuckDB |
|-----------|-------|------------|--------|--------|------------------|----------------------|------------------|
| Group & Summarise | 310.5 ms | 190.0 ms | 67.0 ms | 4.0 ms | **77.6x** | **47.5x** | **16.7x** |
| Filter | 444.0 ms | 479.0 ms | 585.0 ms | 11.0 ms | **40.4x** | **43.5x** | **53.2x** |
| Complete Workflow | 1237.0 ms | 574.5 ms | 126.5 ms | 20.0 ms | **61.9x** | **28.7x** | **6.3x** |

*Complete workflow: filter + mutate + group_by + summarise*

**Hardware**: Intel Core i9-12900K (16 cores), NVIDIA RTX 5070 (12 GB VRAM)

cuplyr allocates GPU memory through RMM's async resource by default. Like an RMM pool it caches freed memory for reuse, and unlike the pool `gpu_gc()` can return that memory to the device. `vignette("memory", package = "cuplyr")` compares async, the pool and plain `cudaMalloc`, with timings and held memory measured on an RTX 5070.

## Acknowledgments

This project is built on [RAPIDS cuDF](https://github.com/rapidsai/cudf) by NVIDIA and the RAPIDS AI team.

---

**License**: Apache 2.0

**Maintainer**: [@bbtheo](https://github.com/bbtheo)

**Documentation**: `DEVELOPER_GUIDE.md`
