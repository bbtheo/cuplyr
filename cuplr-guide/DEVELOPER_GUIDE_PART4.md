## 18. Deliverables Checklist

### Required Deliverables

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| 1 | Developer Guide (this document) | ✓ | `DEVELOPER_GUIDE.md` |
| 2 | R Package Skeleton | ✓ | `cuplr/` directory |
| 3 | Unit Tests (6+) | ✓ | `cuplr/tests/testthat/` |
| 4 | Integration Tests (2+) | ✓ | `cuplr/tests/testthat/test-integration.R` |
| 5 | GitHub Actions CI | ✓ | `cuplr/.github/workflows/ci.yml` |
| 6 | Benchmark Scripts | ✓ | `cuplr/inst/benchmarks/` |
| 7 | Conda Recipe | ✓ | `cuplr/recipe/meta.yaml` |
| 8 | Dockerfile | ✓ | `cuplr/inst/docker/Dockerfile` |

### Package File Inventory

```
cuplr/
├── DESCRIPTION                 # Package metadata
├── NAMESPACE                   # Exports and imports
├── LICENSE                     # Apache 2.0
├── configure                   # Build configuration script
├── configure.win              # Windows stub (unsupported message)
├── README.md                   # Package README
├── NEWS.md                     # Changelog
├── .Rbuildignore              # Build exclusions
├── R/
│   ├── zzz.R                  # Package hooks
│   ├── tbl_gpu.R              # Core class definition
│   ├── dplyr-filter.R         # filter() implementation
│   ├── dplyr-select.R         # select() implementation
│   ├── dplyr-mutate.R         # mutate() implementation
│   ├── dplyr-arrange.R        # arrange() implementation
│   ├── dplyr-group.R          # group_by/ungroup implementation
│   ├── dplyr-summarise.R      # summarise() implementation
│   ├── dplyr-join.R           # join implementations
│   ├── collect.R              # collect/compute implementation
│   ├── ast.R                  # AST node definitions
│   ├── parse_expr.R           # Expression parser
│   ├── optimizer.R            # Query optimizer
│   ├── lower.R                # AST to libcudf lowering
│   ├── arrow_interop.R        # Arrow integration
│   ├── logging.R              # Verbose mode logging
│   ├── diagnostics.R          # GPU info and monitoring
│   ├── safety.R               # Memory limits and safety
│   ├── sanitize.R             # Expression validation
│   ├── compat.R               # Version compatibility
│   └── utils.R                # Utility functions
├── src/
│   ├── Makevars.in            # Build template
│   ├── init.cpp               # R registration
│   ├── gpu_table.hpp          # XPtr wrapper header
│   ├── transfer.cpp           # R <-> GPU data transfer
│   ├── filter.cpp             # Filter operations
│   ├── sort.cpp               # Sort operations
│   ├── groupby.cpp            # GroupBy operations
│   ├── join.cpp               # Join operations
│   ├── binary_ops.cpp         # Binary operations
│   ├── arrow_interop.cpp      # Arrow C Data Interface
│   ├── diagnostics.cpp        # GPU info
│   └── RcppExports.cpp        # Generated exports
├── inst/
│   ├── docker/
│   │   └── Dockerfile         # Development container
│   └── benchmarks/
│       ├── run_benchmarks.R   # Benchmark suite
│       └── results/           # Benchmark output
├── tests/
│   ├── testthat.R             # Test runner
│   └── testthat/
│       ├── helper-cuplr.R     # Test helpers
│       ├── test-basic.R       # Basic functionality
│       ├── test-filter.R      # Filter tests
│       ├── test-mutate.R      # Mutate tests
│       ├── test-arrange.R     # Arrange tests
│       ├── test-group.R       # Group by tests
│       ├── test-join.R        # Join tests
│       ├── test-integration.R # Integration tests
│       └── test-edge-cases.R  # Edge case tests
├── man/                        # Generated documentation
├── recipe/
│   ├── meta.yaml              # Conda recipe
│   └── build.sh               # Conda build script
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI
```

### Test Coverage Requirements

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test-basic.R | 4 | tbl_gpu creation, collect, print, dim |
| test-filter.R | 4 | >, <, ==, multiple conditions, NA handling |
| test-mutate.R | 3 | arithmetic, new columns, type preservation |
| test-arrange.R | 2 | ascending, descending |
| test-group.R | 3 | group_by, summarise, ungroup |
| test-join.R | 2 | left_join, inner_join |
| test-integration.R | 2 | complex pipelines, GPU vs CPU comparison |
| test-edge-cases.R | 4 | empty tables, all NA, large data, type edge cases |
| **Total** | **24** | |

---

## 19. Search Keywords & Primary Resources

### Search Keywords for Research

```
# Core Technology
libcudf API
RAPIDS libcudf C API
cudf cpp api
cudf::table
cudf::column
cudf::groupby::groupby
cudf::binary_operation
cudf::apply_boolean_mask
cudf::sort

# R Integration
Rcpp external pointer
Rcpp XPtr libcudf
Rcpp compiling with external libs
R package configure CUDA
R CMD INSTALL with CUDA
cpp11 CUDA integration

# dplyr Backend
dplyr backend implementation
dbplyr translation
dplyr S3 methods filter mutate
vctrs R package
pillar tbl_format

# Interoperability
Arrow C Data Interface
nanoarrow R package
reticulate RAPIDS
cudf arrow integration

# Build and Deploy
contrib build GPU packages
R package GitHub Actions GPU
nvidia container toolkit
RAPIDS docker images
conda-forge GPU packages

# Performance
cudf kernel fusion
RAPIDS memory management
RMM RAPIDS Memory Manager
GPU DataFrame performance
```

### Primary Documentation Resources

| Resource | URL | Use For |
|----------|-----|---------|
| libcudf API Docs | https://docs.rapids.ai/api/libcudf/stable/ | C++ API reference |
| libcudf Developer Guide | https://docs.rapids.ai/api/libcudf/stable/developer_guide | Design patterns |
| RAPIDS Installation | https://docs.rapids.ai/install/ | Version requirements |
| RAPIDS Support Notices | https://docs.rapids.ai/notices/rsn/ | Compatibility |
| cuDF GitHub | https://github.com/rapidsai/cudf | Source, examples |
| dbplyr New Backend | https://dbplyr.tidyverse.org/articles/new-backend.html | dplyr backend guide |
| dbplyr Translation | https://dbplyr.tidyverse.org/articles/translation-function.html | Expression translation |
| fstplyr Implementation | https://krlmlr.github.io/fstplyr/articles/implement.html | Non-DB backend example |
| Rcpp Documentation | https://dirk.eddelbuettel.com/code/rcpp/ | C++ integration |
| Rcpp XPtr Reference | https://dirk.eddelbuettel.com/code/rcpp/html/classRcpp_1_1XPtr.html | External pointers |
| nanoarrow R Package | https://arrow.apache.org/nanoarrow/latest/r/ | Arrow C interface |
| Arrow C Data Interface | https://arrow.apache.org/docs/format/CDataInterface.html | Zero-copy spec |
| vctrs Package | https://vctrs.r-lib.org/ | Type system |
| rlang Package | https://rlang.r-lib.org/ | Quosures, expressions |

### Example Projects to Study

| Project | URL | Relevance |
|---------|-----|-----------|
| cudf-python | https://github.com/rapidsai/cudf/tree/main/python | Python bindings pattern |
| dbplyr | https://github.com/tidyverse/dbplyr | SQL translation |
| dtplyr | https://github.com/tidyverse/dtplyr | data.table backend |
| arrow-r | https://github.com/apache/arrow/tree/main/r | Arrow R bindings |
| duckplyr | https://github.com/duckdb/duckplyr | DuckDB backend |

### RAPIDS C++ Headers to Study

```cpp
// Essential headers to understand
#include <cudf/table/table.hpp>          // cudf::table
#include <cudf/table/table_view.hpp>     // cudf::table_view
#include <cudf/column/column.hpp>        // cudf::column
#include <cudf/column/column_view.hpp>   // cudf::column_view
#include <cudf/types.hpp>                // type_id, data_type
#include <cudf/copying.hpp>              // slice, gather, scatter
#include <cudf/sorting.hpp>              // sort, sorted_order
#include <cudf/stream_compaction.hpp>    // apply_boolean_mask, distinct
#include <cudf/groupby.hpp>              // groupby::groupby
#include <cudf/aggregation.hpp>          // make_*_aggregation
#include <cudf/binaryop.hpp>             // binary_operation
#include <cudf/unary.hpp>                // unary_operation
#include <cudf/join.hpp>                 // left_join, inner_join
#include <cudf/interop.hpp>              // Arrow interop
#include <cudf/scalar/scalar.hpp>        // scalar types
#include <cudf/scalar/scalar_factories.hpp>
#include <rmm/device_buffer.hpp>         // GPU memory
#include <rmm/mr/device/per_device_resource.hpp>
```

---

## Appendix A: Quick Reference Card

### Creating GPU Tables

```r
# From data.frame
gpu_df <- tbl_gpu(df)

# From CSV (via Arrow for efficiency)
gpu_df <- tbl_gpu(arrow::read_csv_arrow("data.csv"))

# Check status
is_tbl_gpu(gpu_df)
dim(gpu_df)
names(gpu_df)
```

### dplyr Verbs

```r
# All standard verbs work
gpu_df %>%
  filter(x > 10) %>%
  select(x, y) %>%
  mutate(z = x + y) %>%
  arrange(desc(z)) %>%
  group_by(category) %>%
  summarise(
    total = sum(z),
    avg = mean(z),
    n = n()
  ) %>%
  collect()
```

### Joins

```r
left_join(gpu_df1, gpu_df2, by = "key")
inner_join(gpu_df1, gpu_df2, by = c("k1" = "k2"))
```

### Materialization

```r
# Execute and return to R
collect(gpu_df)

# Execute and keep on GPU
compute(gpu_df)
```

### Diagnostics

```r
# GPU info
gpu_info()

# Verbose mode
options(cuplr.verbose = TRUE)

# Memory monitoring
with_gpu_monitor({
  # operations
})

# Dump AST
dump_ast(lazy_gpu_df)
```

### Safety

```r
# Memory-safe execution
gpu_safe({
  tbl_gpu(big_df) %>% filter(x > 10) %>% collect()
}, max_memory_gb = 10, fallback_to_cpu = TRUE)
```

---

## Appendix B: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "GPU not found" | No CUDA driver | Install NVIDIA driver |
| "libcudf.so not found" | Missing library | Check LD_LIBRARY_PATH |
| "CUDA out of memory" | Data too large | Use chunking or filter early |
| "Unsupported type" | Non-standard R type | Convert to supported type first |
| "configure fails" | Missing CUDA/cudf | Install RAPIDS, set CUDA_HOME |

### Diagnostic Commands

```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check libcudf
ldconfig -p | grep libcudf
pkg-config --libs cudf

# Check R can find library
R -e ".Call('_cuplr_check_gpu')"

# Full diagnostics
R -e "cuplr::gpu_info()"
```

### Memory Debugging

```r
# Track allocations
options(cuplr.verbose = TRUE)

# Force garbage collection
gc()

# Check GPU memory
gpu_info()$memory_free

# Use smaller chunks
result <- gpu_chunked(big_df, chunk_size = 1e6, function(chunk) {
  chunk %>% filter(x > 10) %>% collect()
})
```

---

## Appendix C: Performance Tips

1. **Filter early**: Reduce data size before expensive operations
2. **Stay on GPU**: Chain operations without collecting
3. **Use lazy mode**: Let optimizer fuse operations
4. **Pre-sort if possible**: Tell groupby if keys are sorted
5. **Avoid strings when possible**: Numeric operations are faster
6. **Chunk large data**: Process in batches if memory-constrained
7. **Profile first**: Use verbose mode to identify bottlenecks

```r
# Example optimized pipeline
result <- tbl_gpu(df) %>%
  filter(year >= 2020) %>%          # Filter first
  select(year, region, amount) %>%   # Project only needed columns
  mutate(amount_adj = amount * 1.1) %>%
  group_by(year, region) %>%
  summarise(total = sum(amount_adj)) %>%
  collect()                          # Single collect at end
```

---

*End of Developer Guide*

**Document Version**: 1.0.0
**Last Updated**: 2025
**RAPIDS Target**: 25.12+
**Maintainer**: [Your Name]
