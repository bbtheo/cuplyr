## 11. Debugging, Logging & Observability

### Verbose Mode Implementation

```r
# R/logging.R

#' Enable verbose logging
#'
#' @param enabled Logical, whether to enable verbose mode
#' @export
cuplr_verbose <- function(enabled = TRUE) {
  options(cuplr.verbose = enabled)
  if (enabled) {
    message("cuplr: Verbose mode enabled")
  }
  invisible(enabled)
}

# Internal logging function
log_cuplr <- function(..., level = "INFO") {
  if (!getOption("cuplr.verbose", FALSE)) return(invisible())

  timestamp <- format(Sys.time(), "%H:%M:%S.%OS3")
  msg <- paste0("[cuplr ", timestamp, " ", level, "] ", ...)
  message(msg)
}

log_op <- function(op_name, ...) {
  log_cuplr("OP: ", op_name, " - ", ...)
}

log_cudf_call <- function(fn_name, ...) {
  log_cuplr("CUDF: ", fn_name, "(", paste(..., sep = ", "), ")")
}
```

### AST Dump Function

```r
# R/debug.R

#' Print the AST for a lazy tbl_gpu
#'
#' @param x A tbl_gpu object
#' @param indent Current indentation level (internal)
#' @export
dump_ast <- function(x, indent = 0) {
  if (!is_tbl_gpu(x)) {
    cli::cli_abort("x must be a tbl_gpu object")
  }

  prefix <- strrep("  ", indent)

  cat(prefix, "tbl_gpu [\n", sep = "")
  cat(prefix, "  schema: ", paste(x$schema$names, collapse = ", "), "\n", sep = "")
  cat(prefix, "  types:  ", paste(x$schema$types, collapse = ", "), "\n", sep = "")

  if (length(x$groups) > 0) {
    cat(prefix, "  groups: ", paste(x$groups, collapse = ", "), "\n", sep = "")
  }

  cat(prefix, "  materialized: ", !is.null(x$ptr), "\n", sep = "")

  if (length(x$lazy_ops) > 0) {
    cat(prefix, "  lazy_ops:\n", sep = "")
    for (i in seq_along(x$lazy_ops)) {
      dump_ast_node(x$lazy_ops[[i]], indent + 2, i)
    }
  }

  cat(prefix, "]\n", sep = "")
  invisible(x)
}

dump_ast_node <- function(node, indent, index = NULL) {
  prefix <- strrep("  ", indent)
  idx_str <- if (!is.null(index)) paste0("[", index, "] ") else ""

  cat(prefix, idx_str, node$op, "\n", sep = "")

  if (node$op == "filter") {
    for (pred in node$args) {
      cat(prefix, "  predicate: ", deparse(rlang::quo_get_expr(pred)), "\n", sep = "")
    }
  } else if (node$op == "select") {
    cat(prefix, "  columns: ", paste(node$args, collapse = ", "), "\n", sep = "")
  } else if (node$op == "mutate") {
    for (nm in names(node$args)) {
      cat(prefix, "  ", nm, " = ", deparse(rlang::quo_get_expr(node$args[[nm]])), "\n", sep = "")
    }
  } else if (node$op == "arrange") {
    cat(prefix, "  columns: ", paste(node$args$columns, collapse = ", "), "\n", sep = "")
    cat(prefix, "  desc: ", paste(node$args$desc, collapse = ", "), "\n", sep = "")
  } else if (node$op == "group_by") {
    cat(prefix, "  groups: ", paste(node$args, collapse = ", "), "\n", sep = "")
  } else if (node$op == "summarise") {
    for (nm in names(node$args)) {
      cat(prefix, "  ", nm, " = ", deparse(rlang::quo_get_expr(node$args[[nm]])), "\n", sep = "")
    }
  }
}
```

### GPU Memory Diagnostics

```r
# R/diagnostics.R

#' Get GPU information and memory status
#'
#' @return A list with GPU device information
#' @export
gpu_info <- function() {
  info <- .Call(`_cuplr_gpu_info`)
  structure(info, class = "cuplr_gpu_info")
}

#' @export
print.cuplr_gpu_info <- function(x, ...) {
  cat("GPU Device Information\n")
  cat("======================\n")
  cat("Device:       ", x$name, "\n")
  cat("Compute Cap:  ", x$compute_capability, "\n")
  cat("Memory Total: ", format_bytes(x$memory_total), "\n")
  cat("Memory Free:  ", format_bytes(x$memory_free), "\n")
  cat("Memory Used:  ", format_bytes(x$memory_total - x$memory_free), "\n")
  cat("Utilization:  ", sprintf("%.1f%%", 100 * (1 - x$memory_free / x$memory_total)), "\n")
  invisible(x)
}

format_bytes <- function(bytes) {
  if (bytes >= 1e9) {
    sprintf("%.2f GB", bytes / 1e9)
  } else if (bytes >= 1e6) {
    sprintf("%.2f MB", bytes / 1e6)
  } else {
    sprintf("%.2f KB", bytes / 1e3)
  }
}

#' Monitor GPU memory during pipeline execution
#'
#' @param expr Expression to evaluate
#' @return Result of expression, with memory stats printed
#' @export
with_gpu_monitor <- function(expr) {
  before <- gpu_info()
  on.exit({
    after <- gpu_info()
    cat("\nGPU Memory Delta:\n")
    cat("  Before: ", format_bytes(before$memory_free), " free\n")
    cat("  After:  ", format_bytes(after$memory_free), " free\n")
    cat("  Change: ", format_bytes(before$memory_free - after$memory_free), "\n")
  })
  force(expr)
}
```

### C++ GPU Info Implementation

```cpp
// src/diagnostics.cpp
#include <Rcpp.h>
#include <cuda_runtime.h>

// [[Rcpp::export]]
Rcpp::List gpu_info_impl() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    return Rcpp::List::create(
        Rcpp::Named("name") = std::string(prop.name),
        Rcpp::Named("compute_capability") = std::to_string(prop.major) + "." + std::to_string(prop.minor),
        Rcpp::Named("memory_total") = static_cast<double>(total_mem),
        Rcpp::Named("memory_free") = static_cast<double>(free_mem),
        Rcpp::Named("multiprocessors") = prop.multiProcessorCount,
        Rcpp::Named("max_threads_per_block") = prop.maxThreadsPerBlock
    );
}
```

### Example Verbose Output

```
> options(cuplr.verbose = TRUE)
> df <- data.frame(x = 1:1e6, y = runif(1e6), g = sample(letters, 1e6, TRUE))
> result <- tbl_gpu(df) %>%
+   filter(x > 500000) %>%
+   group_by(g) %>%
+   summarise(mean_y = mean(y)) %>%
+   collect()

[cuplr 14:23:15.123 INFO] OP: tbl_gpu - Transferring 1000000 x 3 data.frame to GPU
[cuplr 14:23:15.456 INFO] CUDF: creating table with 3 columns
[cuplr 14:23:15.457 INFO] OP: filter - Adding lazy operation
[cuplr 14:23:15.457 INFO] OP: group_by - Adding lazy operation
[cuplr 14:23:15.457 INFO] OP: summarise - Adding lazy operation
[cuplr 14:23:15.458 INFO] OP: collect - Materializing lazy pipeline
[cuplr 14:23:15.458 INFO] Optimizing 3 operations...
[cuplr 14:23:15.458 INFO] CUDF: binary_operation(col[0], scalar(500000), GREATER)
[cuplr 14:23:15.512 INFO] CUDF: apply_boolean_mask(table, mask)
[cuplr 14:23:15.534 INFO] CUDF: groupby::groupby(keys=[2])
[cuplr 14:23:15.535 INFO] CUDF: groupby::aggregate(mean on col[1])
[cuplr 14:23:15.589 INFO] CUDF: Transferring result 26 x 2 to R
```

---

## 12. Performance & Optimization Guidance

### Kernel Fusion Strategy

Minimize GPU kernel launches by fusing operations:

```cpp
// src/fused_ops.cpp
// Example: Fused filter + mutate

#include "gpu_table.hpp"
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/transform.hpp>

// Instead of separate filter then mutate, compute both in one pass
// [[Rcpp::export]]
SEXP gpu_filter_mutate_fused(
    SEXP xptr,
    int filter_col,
    double filter_val,
    int mutate_col,
    double mutate_factor
) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Create filter mask
    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(filter_val);

    auto mask = cudf::binary_operation(
        view.column(filter_col),
        *scalar,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // Apply filter
    auto filtered = cudf::apply_boolean_mask(view, mask->view());

    // Now apply mutation on filtered data (smaller, more efficient)
    auto factor_scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(factor_scalar.get())->set_value(mutate_factor);

    auto new_col = cudf::binary_operation(
        filtered->view().column(mutate_col),
        *factor_scalar,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    // Build result table with new column appended
    std::vector<std::unique_ptr<cudf::column>> result_cols;
    for (cudf::size_type i = 0; i < filtered->num_columns(); ++i) {
        result_cols.push_back(std::make_unique<cudf::column>(filtered->view().column(i)));
    }
    result_cols.push_back(std::move(new_col));

    auto result = std::make_unique<cudf::table>(std::move(result_cols));
    return make_gpu_table_xptr(std::move(result));
}
```

### Memory Copy Minimization

```r
# R/performance.R

# BAD: Multiple round-trips
bad_example <- function(df) {
  gpu <- tbl_gpu(df)
  filtered <- gpu %>% filter(x > 10) %>% collect()  # GPU -> CPU
  gpu2 <- tbl_gpu(filtered)                          # CPU -> GPU
  result <- gpu2 %>% mutate(y = x * 2) %>% collect() # GPU -> CPU
  result
}

# GOOD: Stay on GPU, single collect at end
good_example <- function(df) {
  tbl_gpu(df) %>%
    filter(x > 10) %>%
    mutate(y = x * 2) %>%
    collect()  # Only one GPU -> CPU transfer
}
```

### Chunking Large Datasets

```r
#' Process large datasets in chunks
#'
#' @param df Large data frame
#' @param chunk_size Number of rows per chunk
#' @param fn Function to apply to each chunk (receives tbl_gpu)
#' @return Combined results
#' @export
gpu_chunked <- function(df, chunk_size = 1e7, fn) {
  n <- nrow(df)
  n_chunks <- ceiling(n / chunk_size)

  results <- vector("list", n_chunks)

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1) * chunk_size + 1
    end_row <- min(i * chunk_size, n)

    chunk <- df[start_row:end_row, , drop = FALSE]
    gpu_chunk <- tbl_gpu(chunk)

    results[[i]] <- fn(gpu_chunk) %>% collect()

    # Force cleanup
    rm(gpu_chunk)
    gc()
  }

  dplyr::bind_rows(results)
}

# Usage
# result <- gpu_chunked(huge_df, chunk_size = 5e6, function(chunk) {
#   chunk %>% filter(x > 10) %>% group_by(g) %>% summarise(n = n())
# })
```

### Multi-GPU Considerations

```r
# R/multi_gpu.R

#' Set active GPU device
#'
#' @param device_id Integer device ID (0-indexed)
#' @export
set_gpu_device <- function(device_id) {
  .Call(`_cuplr_set_device`, as.integer(device_id))
  invisible(device_id)
}

#' Get number of available GPUs
#'
#' @return Integer count of GPUs
#' @export
gpu_count <- function() {
  .Call(`_cuplr_device_count`)
}

#' Distribute work across multiple GPUs
#'
#' @param df Data frame to process
#' @param fn Processing function
#' @param n_gpus Number of GPUs to use (default: all available)
#' @return Combined results
#' @export
gpu_parallel <- function(df, fn, n_gpus = gpu_count()) {
  n <- nrow(df)
  chunk_size <- ceiling(n / n_gpus)

  # Use parallel package for multi-GPU
  results <- parallel::mclapply(seq_len(n_gpus), function(gpu_id) {
    set_gpu_device(gpu_id - 1)  # 0-indexed

    start_row <- (gpu_id - 1) * chunk_size + 1
    end_row <- min(gpu_id * chunk_size, n)

    chunk <- df[start_row:end_row, , drop = FALSE]
    gpu_chunk <- tbl_gpu(chunk)

    fn(gpu_chunk) %>% collect()
  }, mc.cores = n_gpus)

  dplyr::bind_rows(results)
}
```

### When to Fall Back to CPU

```r
# Decision heuristics for GPU vs CPU

should_use_gpu <- function(df, operation) {
  n <- nrow(df)

  # Small data: CPU is often faster due to transfer overhead
  if (n < 10000) {
    return(FALSE)
  }

  # String-heavy operations may not benefit as much
  string_cols <- sum(vapply(df, is.character, logical(1)))
  if (string_cols > ncol(df) / 2 && operation %in% c("mutate", "filter")) {
    # String operations have less GPU speedup
    if (n < 100000) return(FALSE)
  }

  # Check available GPU memory
  info <- gpu_info()
  estimated_size <- object.size(df) * 1.5  # Rough GPU overhead
  if (estimated_size > info$memory_free * 0.8) {
    warning("Data may exceed GPU memory, consider chunking")
  }

  TRUE
}
```

### Custom CUDA Kernels (Advanced)

For operations not in libcudf, you can write custom kernels:

```cpp
// src/custom_kernels.cu
// NOTE: This file requires nvcc compilation

#include <cuda_runtime.h>

// Example: Custom kernel for a specialized operation
__global__ void custom_transform_kernel(
    const double* input,
    double* output,
    int n,
    double param1,
    double param2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Custom computation
        output[idx] = input[idx] * param1 + param2 * sqrt(input[idx]);
    }
}

extern "C" void launch_custom_transform(
    const double* input,
    double* output,
    int n,
    double param1,
    double param2
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    custom_transform_kernel<<<num_blocks, block_size>>>(
        input, output, n, param1, param2
    );
    cudaDeviceSynchronize();
}
```

---

## 13. Interoperability

### Arrow C Data Interface

```r
# R/arrow_interop.R

#' Export tbl_gpu to Arrow format (zero-copy when possible)
#'
#' @param x A tbl_gpu object
#' @return A nanoarrow array stream
#' @export
as_nanoarrow_array_stream.tbl_gpu <- function(x, ...) {
  if (length(x$lazy_ops) > 0) {
    x <- compute(x)
  }

  # Get Arrow C Data Interface pointers from GPU table
  arrow_ptrs <- .Call(`_cuplr_export_to_arrow`, x$ptr)

  # Wrap in nanoarrow
  nanoarrow::nanoarrow_pointer_import(
    arrow_ptrs$schema_ptr,
    arrow_ptrs$array_ptr
  )
}

#' Import from Arrow to GPU
#'
#' @param stream A nanoarrow array stream or Arrow Table
#' @return A tbl_gpu object
#' @export
tbl_gpu.nanoarrow_array_stream <- function(data, ...) {
  # Export Arrow C Data Interface pointers
  schema_ptr <- nanoarrow::nanoarrow_pointer_export(data)

  # Import to GPU via cudf's Arrow integration
  gpu_ptr <- .Call(`_cuplr_import_from_arrow`, schema_ptr)

  # Build schema from Arrow metadata
  schema <- extract_schema_from_arrow(data)


  new_tbl_gpu(ptr = gpu_ptr, schema = schema)
}

#' Convert Arrow Table to GPU
#'
#' @param data An Arrow Table
#' @return A tbl_gpu object
#' @export
tbl_gpu.ArrowTabular <- function(data, ...) {
  # Use Arrow's C Data Interface
  stream <- arrow::as_record_batch_reader(data)
  tbl_gpu(nanoarrow::as_nanoarrow_array_stream(stream))
}
```

### C++ Arrow Integration

```cpp
// src/arrow_interop.cpp
#include "gpu_table.hpp"
#include <cudf/interop.hpp>
#include <nanoarrow/nanoarrow.h>

// Export cudf table to Arrow C Data Interface
// [[Rcpp::export]]
Rcpp::List export_to_arrow(SEXP xptr) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Allocate Arrow structures
    ArrowSchema* schema = new ArrowSchema;
    ArrowArray* array = new ArrowArray;

    // Use cudf's Arrow export (copies data to host)
    // Note: For true zero-copy, would need CUDA-aware Arrow
    auto arrow_table = cudf::to_arrow(view);

    // Export to C interface
    arrow::ExportRecordBatch(*arrow_table->ToRecordBatch(0).ValueOrDie(),
                             array, schema);

    return Rcpp::List::create(
        Rcpp::Named("schema_ptr") = Rcpp::XPtr<ArrowSchema>(schema),
        Rcpp::Named("array_ptr") = Rcpp::XPtr<ArrowArray>(array)
    );
}

// Import from Arrow C Data Interface
// [[Rcpp::export]]
SEXP import_from_arrow(SEXP schema_xptr, SEXP array_xptr) {
    Rcpp::XPtr<ArrowSchema> schema(schema_xptr);
    Rcpp::XPtr<ArrowArray> array(array_xptr);

    // Import to Arrow C++ then to cudf
    auto result = arrow::ImportRecordBatch(array.get(), schema.get());
    if (!result.ok()) {
        Rcpp::stop("Failed to import Arrow data: %s", result.status().message());
    }

    auto arrow_table = arrow::Table::FromRecordBatches({result.ValueOrDie()});
    auto cudf_table = cudf::from_arrow(*arrow_table.ValueOrDie());

    return cuplr::make_gpu_table_xptr(std::move(cudf_table));
}
```

### Reticulate Bridge to Python RAPIDS

```r
# R/python_bridge.R

#' Call Python cudf when C++ API is insufficient
#'
#' @param gpu_tbl A tbl_gpu object
#' @param py_code Python code to execute (has access to 'df' variable)
#' @return A tbl_gpu object
#' @export
gpu_python <- function(gpu_tbl, py_code) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    cli::cli_abort("Package 'reticulate' is required for Python bridge")
  }

  # Ensure Python cudf is available
  if (!reticulate::py_module_available("cudf")) {
    cli::cli_abort("Python cudf module not found. Install with: pip install cudf-cu12")
  }

  # Export to Arrow, import in Python
  arrow_stream <- as_nanoarrow_array_stream(gpu_tbl)

  reticulate::py_run_string("
import cudf
import pyarrow as pa
")

  # Transfer via Arrow IPC (temporary - real impl would use shared memory)
  temp_file <- tempfile(fileext = ".arrow")
  arrow::write_ipc_file(
    arrow::as_arrow_table(arrow_stream),
    temp_file
  )

  reticulate::py$df <- reticulate::py_eval(
    sprintf("cudf.read_feather('%s')", temp_file)
  )

  # Execute user code
  reticulate::py_run_string(py_code)

  # Get result back
  result_py <- reticulate::py$df

  # Convert back to R via Arrow
  result_arrow <- reticulate::py_eval("df.to_arrow()")

  # Clean up
  unlink(temp_file)

  # Return as tbl_gpu
  tbl_gpu(result_arrow)
}

# Example usage:
# result <- gpu_python(my_tbl, "
#   df = df.drop_duplicates(subset=['col1', 'col2'])
#   df['new_col'] = df['x'].rolling(window=10).mean()
# ")
```

### dtplyr / data.table Interop

```r
# R/dtplyr_interop.R

#' Convert between tbl_gpu and dtplyr lazy tables
#'
#' @param x A lazy_dt or tbl_gpu
#' @return Converted object
#' @export
as_lazy_dt.tbl_gpu <- function(x, ...) {
  if (!requireNamespace("dtplyr", quietly = TRUE)) {
    cli::cli_abort("Package 'dtplyr' is required")
  }

  # Materialize and convert
  df <- collect(x)
  dtplyr::lazy_dt(data.table::as.data.table(df))
}

#' @export
tbl_gpu.dtplyr_step <- function(data, ...) {
  # Collect dtplyr result then transfer to GPU
  df <- dplyr::collect(data)
  tbl_gpu(as.data.frame(df))
}
```

---

## 14. Packaging, Distribution & Licensing

### Recommended License

```
# LICENSE file
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full Apache 2.0 license text]
```

**Rationale**: Apache 2.0 matches RAPIDS/libcudf licensing, ensuring compatibility and allowing commercial use while requiring attribution.

### Why Not CRAN

CRAN distribution is **not recommended** for cuplr because:

1. **Binary dependencies**: libcudf, CUDA runtime not available on CRAN build servers
2. **GPU requirement**: CRAN check servers don't have GPUs
3. **Large binary size**: libcudf is 100+ MB
4. **Version coupling**: Tight dependency on CUDA/driver versions

### Recommended Distribution Channels

| Channel | Audience | Pros | Cons |
|---------|----------|------|------|
| GitHub Releases | Developers | Easy updates, source available | Manual install |
| conda-forge | Data scientists | Dependency resolution, binary | Recipe maintenance |
| Docker Hub | DevOps/CI | Reproducible, complete env | Larger size |
| Internal registry | Enterprise | Controlled, secure | Setup overhead |

### Conda Recipe

```yaml
# recipe/meta.yaml
{% set version = "0.1.0" %}

package:
  name: r-cuplr
  version: {{ version }}

source:
  git_url: https://github.com/yourorg/cuplr
  git_rev: v{{ version }}

build:
  number: 0
  skip: true  # [not linux]
  rpaths:
    - lib/R/lib/
    - lib/

requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - cmake
    - make
  host:
    - r-base >=4.3
    - r-rcpp >=1.0.12
    - r-dplyr >=1.1.0
    - r-rlang >=1.1.0
    - r-vctrs >=0.6.0
    - r-pillar >=1.9.0
    - r-glue >=1.6.0
    - r-cli >=3.6.0
    - libcudf >=25.12
    - cudatoolkit >=12.0
  run:
    - r-base >=4.3
    - r-rcpp >=1.0.12
    - r-dplyr >=1.1.0
    - r-rlang >=1.1.0
    - r-vctrs >=0.6.0
    - r-pillar >=1.9.0
    - r-glue >=1.6.0
    - r-cli >=3.6.0
    - libcudf >=25.12
    - cudatoolkit >=12.0
    - __cuda  # Virtual package for CUDA runtime

test:
  commands:
    - $R -e "library(cuplr)"
    - $R -e "cuplr::gpu_info()"  # [gpu]

about:
  home: https://github.com/yourorg/cuplr
  license: Apache-2.0
  license_family: Apache
  license_file: LICENSE
  summary: GPU-accelerated dplyr backend using RAPIDS libcudf
  description: |
    cuplr provides a dplyr-compatible interface for GPU-accelerated
    data manipulation using NVIDIA's RAPIDS libcudf library.

extra:
  recipe-maintainers:
    - your-github-handle
```

### Build Script for Conda

```bash
# recipe/build.sh
#!/bin/bash
set -ex

export CUDA_HOME="${PREFIX}"
export CUDF_HOME="${PREFIX}"

# Run configure
chmod +x configure
./configure

# Build and install
R CMD INSTALL --build .
```

### Release Checklist

```markdown
## Release Checklist for cuplr v{VERSION}

### Pre-release
- [ ] All tests pass locally with GPU
- [ ] Update version in DESCRIPTION
- [ ] Update NEWS.md with changes
- [ ] Update README.md if needed
- [ ] Verify compatibility with latest RAPIDS version
- [ ] Run benchmarks, update benchmark results

### Release
- [ ] Create git tag: `git tag -a v{VERSION} -m "Release v{VERSION}"`
- [ ] Push tag: `git push origin v{VERSION}`
- [ ] Create GitHub Release with changelog
- [ ] Build source tarball: `R CMD build .`
- [ ] Upload tarball to GitHub Release

### Post-release
- [ ] Build and push Docker image
- [ ] Update conda-forge recipe (PR to feedstock)
- [ ] Announce on relevant channels
- [ ] Update documentation site
```

---

## 15. Example User Workflows

### Workflow 1: Basic Data Analysis

```r
library(cuplr)
library(dplyr)

# Check GPU is available
gpu_info()
#> GPU Device Information
#> ======================
#> Device:       NVIDIA A100-SXM4-40GB
#> Compute Cap:  8.0
#> Memory Total: 42.50 GB
#> Memory Free:  41.23 GB
#> Memory Used:  1.27 GB
#> Utilization:  3.0%

# Load data to GPU
sales <- read.csv("sales_100M.csv")
sales_gpu <- tbl_gpu(sales)

# Familiar dplyr pipeline
result <- sales_gpu %>%
  filter(year >= 2020, amount > 0) %>%
  mutate(
    revenue = amount * price,
    quarter = ceiling(month / 3)
  ) %>%
  group_by(region, quarter) %>%
  summarise(
    total_revenue = sum(revenue),
    avg_order = mean(amount),
    n_orders = n()
  ) %>%
  arrange(desc(total_revenue)) %>%
  collect()

print(result)
#> # A tibble: 80 x 5
#>    region  quarter total_revenue avg_order n_orders
#>    <chr>     <dbl>         <dbl>     <dbl>    <int>
#>  1 West          4    1234567890      45.2  2345678
#>  2 East          4    1198765432      42.1  2234567
#> # ... with 78 more rows
```

**Lowered libcudf calls** (with verbose mode):

```
[cuplr] CUDF: binary_operation(col[year], scalar(2020), GREATER_EQUAL) -> mask1
[cuplr] CUDF: binary_operation(col[amount], scalar(0), GREATER) -> mask2
[cuplr] CUDF: binary_operation(mask1, mask2, BITWISE_AND) -> mask_combined
[cuplr] CUDF: apply_boolean_mask(table, mask_combined)
[cuplr] CUDF: binary_operation(col[amount], col[price], MUL) -> revenue_col
[cuplr] CUDF: binary_operation(col[month], scalar(3), DIV) -> temp
[cuplr] CUDF: unary_operation(temp, CEIL) -> quarter_col
[cuplr] CUDF: groupby::groupby(keys=[region, quarter])
[cuplr] CUDF: groupby::aggregate([sum(revenue), mean(amount), count(*)])
[cuplr] CUDF: sort(by=[total_revenue], order=[DESC])
```

### Workflow 2: Joining Large Tables

```r
# Two large tables
customers_gpu <- tbl_gpu(customers_df)   # 50M rows
orders_gpu <- tbl_gpu(orders_df)         # 200M rows

# Join and aggregate
customer_summary <- orders_gpu %>%
  inner_join(customers_gpu, by = "customer_id") %>%
  group_by(customer_segment, region) %>%
  summarise(
    total_orders = n(),
    total_value = sum(order_value),
    avg_value = mean(order_value)
  ) %>%
  collect()

# Timing comparison
library(bench)

bench::mark(
  cpu = orders_df %>%
    inner_join(customers_df, by = "customer_id") %>%
    group_by(customer_segment, region) %>%
    summarise(
      total_orders = n(),
      total_value = sum(order_value),
      avg_value = mean(order_value),
      .groups = "drop"
    ),
  gpu = orders_gpu %>%
    inner_join(customers_gpu, by = "customer_id") %>%
    group_by(customer_segment, region) %>%
    summarise(
      total_orders = n(),
      total_value = sum(order_value),
      avg_value = mean(order_value)
    ) %>%
    collect(),
  check = FALSE,
  min_iterations = 3
)
#> # A tibble: 2 x 6
#>   expression      min   median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr> <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
#> 1 cpu           45.2s    47.8s    0.0209    12.4GB     1.23
#> 2 gpu           1.23s    1.45s    0.689     2.1GB      0.12
```

### Workflow 3: Time Series with Window Functions

```r
# Stock price data
prices_gpu <- tbl_gpu(stock_prices)  # 100M rows

# Calculate moving averages (when supported)
with_indicators <- prices_gpu %>%
  group_by(ticker) %>%
  arrange(date) %>%
  mutate(
    ma_5 = mean(close, .window = 5),
    ma_20 = mean(close, .window = 20),
    returns = (close - lag(close, 1)) / lag(close, 1)
  ) %>%
  ungroup() %>%
  collect()

# For unsupported operations, use Python bridge
advanced_indicators <- gpu_python(prices_gpu, "
import cudf

# cuDF supports more window functions
df['ema_12'] = df.groupby('ticker')['close'].transform(
    lambda x: x.ewm(span=12).mean()
)
df['rsi'] = df.groupby('ticker')['close'].transform(
    lambda x: 100 - (100 / (1 + x.diff().clip(lower=0).rolling(14).mean() /
                                 (-x.diff().clip(upper=0)).rolling(14).mean()))
)
")
```

### Workflow 4: Memory Profiling

```r
# Monitor GPU memory during processing
with_gpu_monitor({
  big_df <- data.frame(
    x = runif(1e8),
    y = runif(1e8),
    g = sample(letters, 1e8, replace = TRUE)
  )

  result <- tbl_gpu(big_df) %>%
    filter(x > 0.5) %>%
    group_by(g) %>%
    summarise(mean_y = mean(y)) %>%
    collect()
})
#>
#> GPU Memory Delta:
#>   Before: 38.50 GB free
#>   After:  36.89 GB free
#>   Change: 1.61 GB
```

---

## 16. Security & Safety Considerations

### GPU Memory Exhaustion

```r
# R/safety.R

#' Safely execute GPU operations with memory limits
#'
#' @param expr Expression to evaluate
#' @param max_memory_gb Maximum GPU memory to use (GB)
#' @param fallback_to_cpu Fall back to CPU if memory exceeded
#' @export
gpu_safe <- function(expr, max_memory_gb = NULL, fallback_to_cpu = TRUE) {
  info <- gpu_info()

  if (!is.null(max_memory_gb)) {
    max_bytes <- max_memory_gb * 1e9
    if (info$memory_free < max_bytes * 0.1) {
      if (fallback_to_cpu) {
        cli::cli_warn("GPU memory low, falling back to CPU")
        return(eval(substitute(expr), envir = parent.frame()))
      } else {
        cli::cli_abort("Insufficient GPU memory: {format_bytes(info$memory_free)} available")
      }
    }
  }

  tryCatch(
    expr,
    error = function(e) {
      if (grepl("out of memory|CUDA_ERROR_OUT_OF_MEMORY", e$message, ignore.case = TRUE)) {
        if (fallback_to_cpu) {
          cli::cli_warn("GPU out of memory, falling back to CPU")
          # Re-evaluate without GPU
          # This requires detecting tbl_gpu and converting to df
          eval(substitute(expr), envir = parent.frame())
        } else {
          cli::cli_abort("GPU out of memory: {e$message}")
        }
      } else {
        stop(e)
      }
    }
  )
}
```

### Input Validation

```cpp
// src/validation.cpp
#include <Rcpp.h>
#include "gpu_table.hpp"

// Validate column index is in bounds
void validate_column_index(int idx, int ncol) {
    if (idx < 0 || idx >= ncol) {
        Rcpp::stop("Column index %d out of bounds [0, %d)", idx, ncol);
    }
}

// Validate numeric value is finite
void validate_finite(double val, const char* param_name) {
    if (!std::isfinite(val)) {
        Rcpp::stop("Parameter '%s' must be finite, got %f", param_name, val);
    }
}

// Validate string doesn't contain injection patterns
bool is_safe_identifier(const std::string& s) {
    // Only allow alphanumeric and underscore
    for (char c : s) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }
    return !s.empty() && !std::isdigit(s[0]);
}

void validate_column_name(const std::string& name) {
    if (!is_safe_identifier(name)) {
        Rcpp::stop("Invalid column name: '%s'. Names must be alphanumeric with underscores.",
                   name.c_str());
    }
}
```

### Expression Sanitization

```r
# R/sanitize.R

# Allowlist of safe functions for GPU execution
SAFE_FUNCTIONS <- c(
  # Arithmetic
  "+", "-", "*", "/", "^", "%%", "%/%",
  # Comparison
  ">", ">=", "<", "<=", "==", "!=",
  # Logical
  "&", "|", "!", "xor",
  # Math
 "abs", "sqrt", "exp", "log", "log10", "log2",
  "sin", "cos", "tan", "asin", "acos", "atan",
  "ceiling", "floor", "round", "trunc",
  # Aggregation
  "sum", "mean", "min", "max", "sd", "var", "n", "median",
  "first", "last",
  # String (subset)
  "nchar", "substr", "toupper", "tolower",
  # Special
  "is.na", "is.null", "ifelse", "case_when", "coalesce",
  # dplyr
  "desc", "lag", "lead", "row_number", "between"
)

validate_expression <- function(expr) {
  if (is.symbol(expr) || is.atomic(expr)) {
    return(TRUE)
  }

  if (is.call(expr)) {
    fn_name <- as.character(expr[[1]])

    # Check for forbidden patterns
    if (fn_name %in% c("system", "system2", "shell", "eval", "parse",
                       "source", "readLines", "writeLines", "file",
                       ".Call", ".External", ".C", ".Fortran")) {
      cli::cli_abort("Function '{fn_name}' is not allowed in GPU expressions")
    }

    # Warn for unknown functions
    if (!fn_name %in% SAFE_FUNCTIONS && !fn_name %in% c("(", "c")) {
      cli::cli_warn("Function '{fn_name}' may not be supported on GPU")
    }

    # Recursively validate arguments
    for (i in seq_along(expr)[-1]) {
      validate_expression(expr[[i]])
    }
  }

  TRUE
}
```

---

## 17. Maintenance & Migration Notes

### RAPIDS Version Upgrade Process

```markdown
## Upgrading RAPIDS libcudf Version

### 1. Check Release Notes
- Visit https://docs.rapids.ai/notices/rsn/
- Note API changes, deprecations, new features

### 2. Update Build Configuration
```bash
# Update Dockerfile base image
sed -i 's/rapidsai\/base:25.12/rapidsai\/base:26.02/g' Dockerfile

# Update conda recipe
# Edit recipe/meta.yaml: libcudf >=26.02
```

### 3. Test Compilation
```bash
# In Docker environment
./configure
R CMD build .
R CMD check cuplr_*.tar.gz
```

### 4. Run Full Test Suite
```bash
R -e "testthat::test_package('cuplr')"
```

### 5. Update Feature Detection
If new APIs are available:
```r
# R/compat.R
has_cudf_feature <- function(feature) {
  switch(feature,
    "distinct_count" = .Call(`_cuplr_has_distinct_count`),
    "regex_replace" = packageVersion("cuplr") >= "0.2.0",
    FALSE
  )
}
```

### 6. Document Changes
- Update NEWS.md
- Update README with new requirements
- Update DESCRIPTION SystemRequirements
```

### CUDA Toolkit Support Matrix

```r
# R/compat.R

CUDA_SUPPORT_MATRIX <- list(
  "25.12" = list(cuda_min = "12.0", cuda_max = "12.5", driver_min = "525.60.13"),
  "26.02" = list(cuda_min = "12.0", cuda_max = "12.6", driver_min = "535.54.03")
)

check_cuda_compat <- function(rapids_version = NULL) {
  # Detect RAPIDS version if not specified
  if (is.null(rapids_version)) {
    rapids_version <- .Call(`_cuplr_rapids_version`)
  }

  compat <- CUDA_SUPPORT_MATRIX[[rapids_version]]
  if (is.null(compat)) {
    cli::cli_warn("Unknown RAPIDS version: {rapids_version}")
    return(invisible(FALSE))
  }

  cuda_version <- .Call(`_cuplr_cuda_version`)
  driver_version <- .Call(`_cuplr_driver_version`)

  issues <- character()

  if (compareVersion(cuda_version, compat$cuda_min) < 0) {
    issues <- c(issues, glue::glue(
      "CUDA {cuda_version} is below minimum {compat$cuda_min}"
    ))
  }

  if (compareVersion(cuda_version, compat$cuda_max) > 0) {
    issues <- c(issues, glue::glue(
      "CUDA {cuda_version} is above maximum {compat$cuda_max}"
    ))
  }

  if (length(issues) > 0) {
    cli::cli_warn(c("Compatibility issues detected:", issues))
    return(invisible(FALSE))
  }

  cli::cli_alert_success("CUDA/RAPIDS compatibility OK")
  invisible(TRUE)
}
```

### Deprecation Policy

```r
# R/deprecated.R

#' @name cuplr-deprecated
#' @title Deprecated functions in cuplr
#'
#' These functions are deprecated and will be removed in future versions.
NULL

# Example deprecation wrapper
gpu_table <- function(...) {
  lifecycle::deprecate_warn(
    when = "0.2.0",
    what = "gpu_table()",
    with = "tbl_gpu()"
  )
  tbl_gpu(...)
}
```

### Migration Guide Template

```markdown
## Migrating from cuplr 0.x to 1.0

### Breaking Changes

1. **Function renamed**: `gpu_table()` → `tbl_gpu()`
   ```r
   # Old
   gpu_table(df)
   # New
   tbl_gpu(df)
   ```

2. **Lazy evaluation default**: Operations are now lazy by default
   ```r
   # Old (eager)
   result <- tbl_gpu(df) %>% filter(x > 10)
   # result is immediately computed

   # New (lazy)
   result <- tbl_gpu(df) %>% filter(x > 10)
   # result is lazy, call collect() or compute()
   result <- result %>% collect()
   ```

3. **NA handling**: Now follows R semantics more closely
   - `filter(x > 10)` excludes NA values (like R)
   - Use `filter(x > 10 | is.na(x))` to include NAs

### New Features

- Window functions: `lag()`, `lead()`, `row_number()`
- String operations: Full stringr compatibility
- Arrow interop: Zero-copy data exchange

### Deprecated

- `as.gpu.data.frame()` - use `tbl_gpu()` instead
- `gpu_collect()` - use `collect()` instead
```

