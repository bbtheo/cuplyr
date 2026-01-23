# cuplr

**dplyr backend for GPU acceleration via RAPIDS cuDF**

cuplr brings GPU computing to R's tidyverse by implementing a dplyr backend powered by [RAPIDS cuDF](https://github.com/rapidsai/cudf), NVIDIA's GPU DataFrame library. Use familiar dplyr syntax while leveraging GPU acceleration for data manipulation.

```r
library(cuplr)

tbl_gpu(sales_data) %>%
  filter(year >= 2020, amount > 0) %>%
  mutate(revenue = amount * price) %>%
  group_by(region, quarter) %>%
  summarise(total = sum(revenue)) %>%
  collect()
```

## About

cuplr translates dplyr operations into cuDF execution on NVIDIA GPUs. It follows the same backend pattern as dbplyr: write standard R code, execute on GPU hardware. This approach can significantly speed up operations on larger datasets (typically >10M rows) without requiring code rewrites.

**Built on [RAPIDS cuDF](https://rapids.ai/)**: cuDF is an open-source GPU DataFrame library developed by NVIDIA's RAPIDS team. It provides highly optimized CUDA kernels for data manipulation operations, backed by Apache Arrow's columnar memory format. cuplr acts as an R interface to this GPU-accelerated compute engine.

## Status

**v0.0.0.9000 – Early development**

This is experimental software under active development. Many features are incomplete or untested.

- ✅ `filter()`, `select()`, `mutate()` (basic operations)
- 🚧 `group_by()`, `arrange()`, `summarise()`
- 🚧 `left_join()`, `right_join()`, `inner_join()`, `full_join()`
- 🚧 Complex joins with `join_by()`
- 🚧 Expression optimization and lazy evaluation
- 🚧 Window functions, string operations
- 🚧 Comprehensive test coverage
- 🚧 Multi-GPU support, out-of-core computation

Contributions and feedback welcome.

## Architecture

- **R layer**: S3 methods implementing dplyr generics
- **Expression parser**: R quosures → internal AST
- **Query optimizer**: Operation fusion and predicate pushdown
- **Native bindings**: Rcpp interface to libcudf C++ API
- **Execution**: cuDF GPU kernels via libcudf
- **Memory**: Arrow C Data Interface for zero-copy transfer

See `DEVELOPER_GUIDE.md` for implementation details.

## Requirements

- NVIDIA GPU with CUDA support
- RAPIDS cuDF installation (see [RAPIDS installation guide](https://rapids.ai/start.html))
- R ≥ 4.0

## Acknowledgments

This project is built on [RAPIDS cuDF](https://github.com/rapidsai/cudf) by NVIDIA and the RAPIDS AI team. cuDF provides the core GPU execution engine and algorithms that make this possible.

---

**License**: Apache 2.0  
**Maintainer**: [@bbtheo](https://github.com/bbtheo)  
**Documentation**: `DEVELOPER_GUIDE.md`

Questions? Open an issue or start a discussion.