# cuplyr Agent Notes

This repo mixes R and C++ (Rcpp) for GPU-backed dplyr-like operations using libcudf. The trickiest parts are build tooling, GPU availability, and keeping R-level schemas aligned with GPU types.

## Local Dev (pixi)
### Fast edit/run loop
1) `pixi run configure` (only when CUDA/cudf paths change).
2) Edit code.
3) `pixi run install` (full rebuild) or `pixi run load-dev` (fast reload in R).
4) Spot-check in R (e.g., `gpu_details()`, `tbl_gpu(mtcars)`).

### When to use each task
- `pixi run load-dev`: tight inner loop for R changes or quick checks.
- `pixi run install`: after C++ changes or when exports changed.
- `pixi run dev`: when you suspect stale artifacts (clean + rebuild).
- `pixi run test`: after feature changes; requires GPU.

## Current C++ Layout (split from transfer.cpp)
- `src/transfer_io.cpp`: R <-> GPU conversion, collect/head/df_to_gpu.
- `src/ops_filter.cpp`: filter operations and mask handling.
- `src/ops_compare.cpp`: comparison ops used for summarise temp columns.
- `src/ops_mutate.cpp`: mutate and copy/replace ops.
- `src/ops_select.cpp`: select operations.
- `src/ops_groupby.cpp`: summarise/groupby logic.
- `src/gpu_info.cpp`: device availability/info.
- `src/cuda_utils.hpp`: `check_cuda()` helper.
- `src/ops_common.hpp`: shared operator mapping helpers.
- `src/gpu_table.hpp`: pointer ownership helpers for cudf::table.

## R Entry Points
- `R/tbl-gpu.R`: `tbl_gpu()` constructor + schema metadata.
- `R/mutate.R`, `R/filter.R`, `R/select.R`, `R/arrange.R`, `R/summarise.R`: dplyr verbs.
- `R/collect.R`: pulls data back to R and warns on INT64 precision.
- `R/gpu-memory.R`: memory reporting and GC helpers.
- `R/utils.R`: shared helpers (type mapping, `wrap_gpu_call()` for clearer GPU errors).

## Known Sharp Edges (things that were hard)
- **cudf header names differ by version.** `bitmask_allocation_size_bytes` lives in `cudf/null_mask.hpp` in this environment. Avoid `cudf/bitmask.hpp`.
- **R type vs GPU type mismatch** can silently break results. Keep `gpu_type_from_r()` and `df_to_gpu()` in sync (logical/Date/POSIXct especially).
- **GPU not detected** is common in CI or local dev. Tests use `skip_if_no_gpu()`; don’t remove it.
- **Rcpp exports need regeneration** after moving/adding functions: run `Rcpp::compileAttributes()` or `devtools::document()`.
- **INT64 precision**: `gpu_collect()` returns doubles; warn when values exceed 2^53.
- **Memory growth**: each GPU op tends to allocate new tables. Replacement mutate paths are optimized, but GC still matters.
- **Join memory warnings**: joins estimate output size and warn when close to available GPU memory; actual allocation can still fail.

## Debugging Local Build Failures
- If a cudf header can’t be found, check `pixi` environment paths and `src/Makevars`.
- Run `pixi run configure` after updating CUDA/cudf libs.
- Use `rg --files -g '*bitmask*' $CONDA_PREFIX/include/cudf` to locate moved headers.

## Adding New GPU Ops (local loop)
1) Implement in appropriate `src/ops_*.cpp` file.
2) Add an `// [[Rcpp::export]]` function.
3) Run `Rcpp::compileAttributes()` to regenerate `R/RcppExports.R` + `src/RcppExports.cpp`.
4) Add R wrapper in `R/*.R` and tests in `tests/testthat/`.

## When You Touch Types (local loop)
- Update schema: `R/utils.R` (`gpu_type_from_r`).
- Update collect/head conversion for new cudf types.
- Add tests for round-trip behavior.

## Code Review Quick Checklist (saves time)
- **Type alignment:** `R/utils.R` and `src/transfer_io.cpp` must agree on type mapping.
- **Head/collect parity:** If a type is supported in `gpu_collect()`, ensure `gpu_head()` uses the same conversion path.
- **Arrange semantics:** Confirm stable sort support in libcudf, NA ordering, and group-prepend behavior in `R/arrange.R`.
- **Rcpp exports:** If new C++ functions exist, verify `R/RcppExports.R` and `src/RcppExports.cpp` are updated.
- **Tests:** New verb features should have matching `tests/testthat/test-*.R` coverage and use `skip_if_no_gpu()`.
