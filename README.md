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
<<<<<<< HEAD
- `arrange()` – row sorting with `desc()` support
- `group_by()` / `ungroup()` – grouping for aggregation
- `summarise()` – grouped aggregations (`sum`, `mean`, `min`, `max`, `n`, `sd`, `var`)

**Joins**
- `inner_join()`, `left_join()`, `right_join()`, `full_join()`

**Combining tables**
- `bind_rows()` – vertical concatenation with schema unification
- `bind_cols()` – horizontal concatenation

**Execution control**
=======
- `arrange()` – row sorting with `desc()` support, NA handling follows dplyr conventions
- `group_by()` + `summarise()` – grouped aggregations (`sum`, `mean`, `min`, `max`, `n`)
- `left_join()`, `right_join()`, `inner_join()`, `full_join()` – GPU joins on key columns
>>>>>>> bc90422 (adding join functionality)
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

### Requirements

- NVIDIA GPU with Compute Capability >= 6.0
- CUDA Toolkit >= 12.0
- RAPIDS libcudf >= 25.12
- R >= 4.3

### Using pixi (recommended)

```bash
# Install pixi if not already installed (https://pixi.sh)
# curl -fsSL https://pixi.sh/install.sh | bash

git clone https://github.com/bbtheo/cuplyr.git
cd cuplyr
pixi run install
```

### From source

```bash
git clone https://github.com/bbtheo/cuplyr.git
cd cuplyr

# Ensure CUDA and cuDF are available, then:
R CMD INSTALL .
```

## Performance

Benchmarks on 50 million rows (synthetic taxi data):

| Operation | dplyr | data.table | cuplyr | vs dplyr | vs data.table |
|-----------|-------|------------|--------|----------|---------------|
| Group & Summarise | 625 ms | 389 ms | 35 ms | **18x** | **11x** |
| Filter | 930 ms | 962 ms | 247 ms | **4x** | **4x** |
| Complete Workflow | 2529 ms | 1158 ms | 42 ms | **60x** | **28x** |

*Complete workflow: filter + mutate + group_by + summarise*

**Hardware**: Intel Core i9-12900K (16 cores), NVIDIA RTX 5070 (12 GB VRAM)

GPU acceleration benefits grow with data size. For datasets under 1M rows, CPU-based solutions may be faster due to GPU transfer overhead.

## Acknowledgments

This project is built on [RAPIDS cuDF](https://github.com/rapidsai/cudf) by NVIDIA and the RAPIDS AI team.

---

**License**: Apache 2.0

**Maintainer**: [@bbtheo](https://github.com/bbtheo)

**Documentation**: `DEVELOPER_GUIDE.md`
