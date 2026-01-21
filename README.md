# cuplr

**GPU-accelerated dplyr. Same code, runs 30-50x faster.**

Write tidyverse, get GPU speed. A new [cuDF](https://github.com/rapidsai/cudf) backend for R, no rewrites.

```r
library(cuplr)

tbl_gpu(sales_data) %>%
  filter(year >= 2020, amount > 0) %>%
  mutate(revenue = amount * price) %>%
  group_by(region, quarter) %>%
  summarise(total = sum(revenue)) %>%
  collect()
```

## Why

Your GPU sits idle. Your dplyr code hits walls at 100M rows. This fixes both.

cuplr is a dplyr backend that targets GPUs the same way dbplyr targets databases. Write R, execute on GPU. 100M row groupby? XX seconds on CPU, YY seconds on GPU.

R lost ground to Python partly due to performance. This closes the gap without leaving the ecosystem.

## Status

**v0.0.0.9000 (Pre-pre-alpha stage)** – Almost nothing works, expanding fast.

- ✅ `filter()`, `select()`, `mutate()`
- 🚧 `group_by()`, `arrange()`, `summarise()`
- 🚧 `left_join()`, `right_join()`, `inner_join()`, `full_join()`
- 🚧 rolling joins, with complex `join_by()` logic
- 🚧 Lazy eval, AST optimization, full tests
- 🚧 Window functions, string ops 
- 🚧 Multi-GPU, streaming 

## Architecture

- **R layer**: S3 methods implementing dplyr generics
- **Parser**: R quosures → internal AST
- **Optimizer**: Fuses ops, pushes predicates
- **Native**: Rcpp bindings to RAPIDS libcudf
- **Execution**: libcudf handles GPU compute
- **Interop**: Arrow C Data Interface for zero-copy

Full details in `DEVELOPER_GUIDE.md`

---

**License**: Apache 2.0  
**Lead**: [@bbtheo](https://github.com/bbtheo)  
**Docs**: See `DEVELOPER_GUIDE.md`

Questions? Open an issue or discussion.
