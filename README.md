# cuplyr <img src="man/figures/logo.png" align="right" height="138" />

#### dplyr backend for GPU acceleration via RAPIDS cuDF

cuplyr implements a dplyr backend powered by [RAPIDS cuDF](https://github.com/rapidsai/cudf), NVIDIA's GPU DataFrame library. Write standard dplyr code, execute on GPU hardware.

```r
library(cuplyr)

tbl_gpu(sales_data, lazy = TRUE) |>
  filter(year >= 2020, amount > 0) |>
  mutate(revenue = amount * price) |>
  group_by(region, quarter) |>
  summarise(total = sum(revenue)) |>
  inner_join(regions, by = "region") |>
  arrange(desc(total)) |>
  collect()
```

## About

cuplyr translates dplyr operations into cuDF execution on NVIDIA GPUs. It follows the same backend pattern as dbplyr: write standard R code, execute on GPU hardware. This approach can provide significant speedups on larger datasets (typically >1M rows) without requiring major code changes.

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
- `compute()` – execute lazy operations, keep on GPU
- `tbl_gpu(..., lazy = TRUE)` – enable lazy evaluation with AST optimization

### Lazy evaluation

Lazy mode defers execution until `collect()` or `compute()`, enabling automatic optimizations:
- Projection pruning (drop unused columns early)
- Filter pushdown (move filters closer to data sources)
- Mutate fusion (combine consecutive transformations)

```r
# Enable globally
options(cuplyr.exec_mode = "lazy")

# Or per-table
tbl_gpu(data, lazy = TRUE)
```

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

Benchmark code lives in `benchmark/benchmark.R`.

Benchmarks on 25 million rows (synthetic taxi data, median of 10 iterations):

| Operation | dplyr | data.table | DuckDB | cuplyr | cuplyr vs dplyr | cuplyr vs data.table | cuplyr vs DuckDB |
|-----------|-------|------------|--------|--------|------------------|----------------------|------------------|
| Group & Summarise | 310.5 ms | 190.0 ms | 67.0 ms | 4.0 ms | **77.6x** | **47.5x** | **16.7x** |
| Filter | 444.0 ms | 479.0 ms | 585.0 ms | 11.0 ms | **40.4x** | **43.5x** | **53.2x** |
| Complete Workflow | 1237.0 ms | 574.5 ms | 126.5 ms | 20.0 ms | **61.9x** | **28.7x** | **6.3x** |

*Complete workflow: filter + mutate + group_by + summarise*

**Hardware**: Intel Core i9-12900K (16 cores), NVIDIA RTX 5070 (12 GB VRAM)

## Acknowledgments

This project is built on [RAPIDS cuDF](https://github.com/rapidsai/cudf) by NVIDIA and the RAPIDS AI team.

---

**License**: Apache 2.0

**Maintainer**: [@bbtheo](https://github.com/bbtheo)

**Documentation**: `DEVELOPER_GUIDE.md`
