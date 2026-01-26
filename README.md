# cuplr

**dplyr backend for GPU acceleration via RAPIDS cuDF**

cuplr implements a dplyr backend powered by [RAPIDS cuDF](https://github.com/rapidsai/cudf), NVIDIA's GPU DataFrame library. It allows users to write standard dplyr code while executing operations on GPU hardware.

```r
library(cuplr)

tbl_gpu(sales_data) |>
  filter(year >= 2020, amount > 0) |>
  mutate(revenue = amount * price) |>
  group_by(region, quarter) |>
  summarise(total = sum(revenue)) |>
  arrange(desc(total)) |>
  collect()
```

## About

cuplr translates dplyr operations into cuDF execution on NVIDIA GPUs. It follows the same backend pattern as dbplyr: write standard R code, execute on GPU hardware. This approach can provide significant speedups on larger datasets (typically >10M rows) without requiring major code changes.

**Built on [RAPIDS cuDF](https://rapids.ai/)**: cuDF is an open-source GPU DataFrame library developed by NVIDIA's RAPIDS team. It provides optimized CUDA kernels for data manipulation operations, backed by Apache Arrow's columnar memory format. cuplr provides an R interface to this execution engine.


## Status

**v0.0.1 – Early development**

This is experimental software under active development.

### Supported operations

- `filter()` – row filtering with comparison and logical operators
- `select()` – column selection and reordering
- `mutate()` – column transformations and arithmetic
- `arrange()` – row sorting with `desc()` support, NA handling follows dplyr conventions
- `group_by()` + `summarise()` – grouped aggregations (`sum`, `mean`, `min`, `max`, `n`)
- `collect()` – transfer results back to R

### Supported column types

- `numeric` (double) -> FLOAT64
- `integer` -> INT32
- `character` -> STRING
- `logical` -> BOOL8
- `Date` -> TIMESTAMP_DAYS
- `POSIXct` -> TIMESTAMP_MICROSECONDS
- `factor` -> DICTIONARY32

### Not yet implemented

- `left_join()`, `right_join()`, `inner_join()`, `full_join()`
- Complex joins with `join_by()`
- Expression optimization and lazy evaluation
- Window functions, string operations
- Multi-GPU support, out-of-core computation

Contributions and feedback are welcome.

## Architecture

- **R layer**: S3 methods implementing dplyr generics
- **Expression parser**: R quosures to internal AST
- **Query optimizer**: Operation fusion and predicate pushdown
- **Native bindings**: Rcpp interface to libcudf C++ API
- **Execution**: cuDF GPU kernels via libcudf
- **Memory**: Arrow C Data Interface for zero-copy transfer

See `DEVELOPER_GUIDE.md` for implementation details.

## Requirements

- NVIDIA GPU with CUDA support
- RAPIDS cuDF installation (see [RAPIDS installation guide](https://rapids.ai/start.html))
- R >= 4.0

## Acknowledgments

This project is built on [RAPIDS cuDF](https://github.com/rapidsai/cudf) by NVIDIA and the RAPIDS AI team.

---

**License**: Apache 2.0
**Maintainer**: [@bbtheo](https://github.com/bbtheo)
**Documentation**: `DEVELOPER_GUIDE.md`

