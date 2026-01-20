## 6. Binding Strategy & C++ Glue

### Recommended Approach: Rcpp with XPtr

We recommend **Rcpp** over cpp11 for this project because:
1. Better documented patterns for external pointers
2. Existing examples with CUDA integration
3. Mature finalizer support for GPU memory cleanup

### External Pointer Wrapper Pattern

```cpp
// src/gpu_table.hpp
#ifndef CUPLR_GPU_TABLE_HPP
#define CUPLR_GPU_TABLE_HPP

#include <Rcpp.h>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>

namespace cuplr {

// Wrap cudf::table in a shared_ptr for R interop
using GpuTablePtr = std::shared_ptr<cudf::table>;

// Custom destructor that ensures GPU cleanup
inline void release_gpu_table(GpuTablePtr* ptr) {
    if (ptr != nullptr) {
        // Reset triggers cudf::table destructor, freeing GPU memory
        ptr->reset();
        delete ptr;
    }
}

// Create XPtr with custom destructor
inline Rcpp::XPtr<GpuTablePtr> make_gpu_table_xptr(std::unique_ptr<cudf::table> tbl) {
    // Convert unique_ptr to shared_ptr for R ownership
    auto* sptr = new GpuTablePtr(std::move(tbl));
    return Rcpp::XPtr<GpuTablePtr>(sptr, true);  // true = register destructor
}

// Extract table_view from XPtr (non-owning view)
inline cudf::table_view get_table_view(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    return (*xptr)->view();
}

// Get mutable table reference
inline cudf::table& get_table_ref(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    return **xptr;
}

} // namespace cuplr

#endif // CUPLR_GPU_TABLE_HPP
```

### Data Transfer: R data.frame → GPU

```cpp
// src/transfer.cpp
#include "gpu_table.hpp"
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplr {

// Create GPU column from R numeric vector
std::unique_ptr<column> numeric_to_gpu(NumericVector x) {
    size_type n = x.size();

    // Allocate device memory
    rmm::device_buffer data(n * sizeof(double),
                            rmm::cuda_stream_default,
                            rmm::mr::get_current_device_resource());

    // Copy from host to device
    cudaMemcpy(data.data(), &x[0], n * sizeof(double), cudaMemcpyHostToDevice);

    // Handle NAs by creating validity mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;

    // Check for NAs
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    for (size_type i = 0; i < n; ++i) {
        if (NumericVector::is_na(x[i])) {
            // Clear bit i
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_default,
                                       rmm::mr::get_current_device_resource());
    }

    return std::make_unique<column>(
        data_type{type_id::FLOAT64},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R integer vector
std::unique_ptr<column> integer_to_gpu(IntegerVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int32_t),
                            rmm::cuda_stream_default,
                            rmm::mr::get_current_device_resource());

    cudaMemcpy(data.data(), &x[0], n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Handle NAs
    rmm::device_buffer null_mask;
    size_type null_count = 0;
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);

    for (size_type i = 0; i < n; ++i) {
        if (IntegerVector::is_na(x[i])) {
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_default,
                                       rmm::mr::get_current_device_resource());
    }

    return std::make_unique<column>(
        data_type{type_id::INT32},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R character vector
std::unique_ptr<column> character_to_gpu(CharacterVector x) {
    size_type n = x.size();

    // Convert to std::vector<std::string> for cudf factory
    std::vector<std::string> strings(n);
    std::vector<bool> valids(n, true);

    for (size_type i = 0; i < n; ++i) {
        if (CharacterVector::is_na(x[i])) {
            valids[i] = false;
            strings[i] = "";
        } else {
            strings[i] = as<std::string>(x[i]);
        }
    }

    // Use cudf factory function
    auto host_span = cudf::host_span<std::string const>(strings.data(), strings.size());
    auto valid_span = cudf::host_span<bool const>(valids.data(), valids.size());

    return cudf::make_strings_column(host_span, valid_span);
}

} // namespace cuplr

// [[Rcpp::export]]
SEXP df_to_gpu(DataFrame df) {
    using namespace cuplr;

    int ncol = df.size();
    CharacterVector names = df.names();

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(ncol);

    for (int i = 0; i < ncol; ++i) {
        SEXP col = df[i];

        switch (TYPEOF(col)) {
            case REALSXP:
                columns.push_back(numeric_to_gpu(col));
                break;
            case INTSXP:
                columns.push_back(integer_to_gpu(col));
                break;
            case STRSXP:
                columns.push_back(character_to_gpu(col));
                break;
            case LGLSXP:
                // Convert logical to integer then to BOOL8
                columns.push_back(integer_to_gpu(as<IntegerVector>(col)));
                break;
            default:
                Rcpp::stop("Unsupported column type at index %d", i);
        }
    }

    auto tbl = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(tbl));
}
```

### GPU → R data.frame Transfer

```cpp
// src/transfer.cpp (continued)

// [[Rcpp::export]]
DataFrame gpu_to_df(SEXP xptr, CharacterVector names) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int ncol = view.num_columns();
    size_type nrow = view.num_rows();

    List result(ncol);

    for (int i = 0; i < ncol; ++i) {
        cudf::column_view col = view.column(i);
        cudf::type_id type = col.type().id();

        switch (type) {
            case cudf::type_id::FLOAT64: {
                NumericVector out(nrow);
                cudaMemcpy(&out[0], col.data<double>(),
                          nrow * sizeof(double), cudaMemcpyDeviceToHost);
                // Handle nulls
                if (col.has_nulls()) {
                    // Download validity mask and set NAs
                    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(nrow));
                    cudaMemcpy(validity.data(), col.null_mask(),
                              validity.size(), cudaMemcpyDeviceToHost);
                    for (size_type j = 0; j < nrow; ++j) {
                        if (!((validity[j/8] >> (j%8)) & 1)) {
                            out[j] = NA_REAL;
                        }
                    }
                }
                result[i] = out;
                break;
            }
            case cudf::type_id::INT32: {
                IntegerVector out(nrow);
                cudaMemcpy(&out[0], col.data<int32_t>(),
                          nrow * sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (col.has_nulls()) {
                    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(nrow));
                    cudaMemcpy(validity.data(), col.null_mask(),
                              validity.size(), cudaMemcpyDeviceToHost);
                    for (size_type j = 0; j < nrow; ++j) {
                        if (!((validity[j/8] >> (j%8)) & 1)) {
                            out[j] = NA_INTEGER;
                        }
                    }
                }
                result[i] = out;
                break;
            }
            case cudf::type_id::STRING: {
                // String columns require special handling
                auto str_col = cudf::strings_column_view(col);
                CharacterVector out(nrow);
                // Use cudf utility to get strings as host vector
                // This is simplified - real implementation would use
                // cudf::strings::detail functions
                result[i] = out;
                break;
            }
            default:
                Rcpp::stop("Unsupported column type for GPU->R transfer");
        }
    }

    result.names() = names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -nrow);

    return result;
}
```

### Implementing filter() in C++

```cpp
// src/filter.cpp
#include "gpu_table.hpp"
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>

// [[Rcpp::export]]
SEXP gpu_filter_gt(SEXP xptr, int col_idx, double value) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Get the column to filter on (0-indexed)
    cudf::column_view filter_col = view.column(col_idx);

    // Create scalar for comparison
    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    scalar->set_valid_async(true);
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    // Create boolean mask: col > value
    auto mask = cudf::binary_operation(
        filter_col,
        *scalar,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // Apply boolean mask to filter table
    auto result = cudf::apply_boolean_mask(view, mask->view());

    return make_gpu_table_xptr(std::move(result));
}

// More flexible filter with expression support
// [[Rcpp::export]]
SEXP gpu_filter_mask(SEXP tbl_xptr, SEXP mask_xptr, int mask_col_idx) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> tbl_ptr(tbl_xptr);
    Rcpp::XPtr<GpuTablePtr> mask_ptr(mask_xptr);

    cudf::table_view tbl_view = get_table_view(tbl_ptr);
    cudf::table_view mask_view = get_table_view(mask_ptr);

    // Get boolean column from mask table
    cudf::column_view bool_mask = mask_view.column(mask_col_idx);

    auto result = cudf::apply_boolean_mask(tbl_view, bool_mask);
    return make_gpu_table_xptr(std::move(result));
}
```

### Compilation Setup

```make
# src/Makevars.in (template - configure will generate Makevars)

# Paths set by configure
CUDA_HOME = @CUDA_HOME@
CUDF_INCLUDE = @CUDF_INCLUDE@
CUDF_LIB = @CUDF_LIB@
RMM_INCLUDE = @RMM_INCLUDE@

# Compiler settings
CXX_STD = CXX17
PKG_CXXFLAGS = -I$(CUDF_INCLUDE) -I$(RMM_INCLUDE) -I$(CUDA_HOME)/include \
               -DFMT_HEADER_ONLY $(SHLIB_OPENMP_CXXFLAGS)

PKG_LIBS = -L$(CUDF_LIB) -lcudf -L$(CUDA_HOME)/lib64 -lcudart \
           $(SHLIB_OPENMP_CXXFLAGS) -Wl,-rpath,$(CUDF_LIB)

# Ensure nvcc is not used for R package compilation
# All CUDA code is in libcudf; we only link against it
```

---

## 7. Build System & Packaging

### DESCRIPTION File

```
Package: cuplr
Title: GPU-Accelerated Data Manipulation with dplyr Syntax
Version: 0.1.0
Authors@R: c(
    person("Your", "Name", email = "you@example.com", role = c("aut", "cre")),
    person("RAPIDS Team", role = "cph", comment = "libcudf library")
  )
Description: Provides a dplyr backend that executes operations on NVIDIA GPUs
    using the RAPIDS libcudf library. Supports filter, select, mutate, arrange,
    group_by, summarise, and join operations with familiar tidyverse syntax
    while achieving significant speedups on large datasets.
License: Apache License (>= 2.0)
URL: https://github.com/yourorg/cuplr
BugReports: https://github.com/yourorg/cuplr/issues
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.3.1
SystemRequirements:
    NVIDIA GPU with Compute Capability >= 6.0,
    CUDA Toolkit >= 12.0,
    RAPIDS libcudf >= 25.12
Depends:
    R (>= 4.3.0)
Imports:
    Rcpp (>= 1.0.12),
    dplyr (>= 1.1.0),
    rlang (>= 1.1.0),
    vctrs (>= 0.6.0),
    pillar (>= 1.9.0),
    glue (>= 1.6.0),
    cli (>= 3.6.0)
Suggests:
    testthat (>= 3.0.0),
    bench,
    arrow,
    nanoarrow,
    bit64,
    reticulate
LinkingTo:
    Rcpp
Config/testthat/edition: 3
NeedsCompilation: yes
```

### NAMESPACE File

```
# Generated by roxygen2: do not edit by hand

# Imports
import(dplyr)
importFrom(Rcpp, sourceCpp)
importFrom(rlang, enquo, enquos, eval_tidy, quo_get_expr, is_quosure)
importFrom(vctrs, vec_ptype2, vec_cast)
importFrom(pillar, tbl_sum, tbl_format_header)
importFrom(glue, glue)
importFrom(cli, cli_abort, cli_warn)

# Exports
export(tbl_gpu)
export(as_tbl_gpu)
export(is_tbl_gpu)
export(collect)
export(compute)
export(gpu_info)

# S3 Methods
S3method(print, tbl_gpu)
S3method(dim, tbl_gpu)
S3method(names, tbl_gpu)
S3method(as.data.frame, tbl_gpu)
S3method(collect, tbl_gpu)
S3method(compute, tbl_gpu)

# dplyr verb methods
S3method(filter, tbl_gpu)
S3method(select, tbl_gpu)
S3method(mutate, tbl_gpu)
S3method(arrange, tbl_gpu)
S3method(group_by, tbl_gpu)
S3method(ungroup, tbl_gpu)
S3method(summarise, tbl_gpu)
S3method(summarize, tbl_gpu)
S3method(left_join, tbl_gpu)
S3method(inner_join, tbl_gpu)
S3method(right_join, tbl_gpu)
S3method(distinct, tbl_gpu)
S3method(slice, tbl_gpu)
S3method(head, tbl_gpu)
S3method(tail, tbl_gpu)
S3method(rename, tbl_gpu)

# Internal C++ functions
useDynLib(cuplr, .registration = TRUE)
```

### Configure Script

```bash
#!/bin/bash
# configure - Detect CUDA and libcudf, generate src/Makevars

echo "Configuring cuplr..."

# Default paths
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDF_HOME="${CUDF_HOME:-}"
CONDA_PREFIX="${CONDA_PREFIX:-}"

# Function to find libcudf
find_cudf() {
    # Try conda environment first
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/include/cudf/cudf.hpp" ]; then
        echo "$CONDA_PREFIX"
        return 0
    fi

    # Try common installation paths
    for path in /usr/local /opt/rapids /usr; do
        if [ -f "$path/include/cudf/cudf.hpp" ]; then
            echo "$path"
            return 0
        fi
    done

    # Check if CUDF_HOME is set
    if [ -n "$CUDF_HOME" ] && [ -f "$CUDF_HOME/include/cudf/cudf.hpp" ]; then
        echo "$CUDF_HOME"
        return 0
    fi

    return 1
}

# Detect CUDA
if [ ! -d "$CUDA_HOME" ]; then
    # Try to find CUDA
    for cuda_path in /usr/local/cuda /opt/cuda /usr/lib/cuda; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/include/cuda.h" ]; then
            CUDA_HOME="$cuda_path"
            break
        fi
    done
fi

if [ ! -f "$CUDA_HOME/include/cuda.h" ]; then
    echo "ERROR: CUDA not found. Please set CUDA_HOME environment variable."
    echo "       Example: export CUDA_HOME=/usr/local/cuda"
    exit 1
fi

echo "Found CUDA at: $CUDA_HOME"
echo "CUDA version: $($CUDA_HOME/bin/nvcc --version | grep release | awk '{print $6}')"

# Detect libcudf
CUDF_PREFIX=$(find_cudf)
if [ -z "$CUDF_PREFIX" ]; then
    echo "ERROR: libcudf not found. Please install RAPIDS or set CUDF_HOME."
    echo ""
    echo "Installation options:"
    echo "  1. Conda: conda install -c rapidsai -c conda-forge -c nvidia libcudf"
    echo "  2. Pip:   pip install libcudf-cu12"
    echo "  3. Set:   export CUDF_HOME=/path/to/cudf"
    exit 1
fi

CUDF_INCLUDE="$CUDF_PREFIX/include"
CUDF_LIB="$CUDF_PREFIX/lib"

echo "Found libcudf at: $CUDF_PREFIX"

# Check for RMM (RAPIDS Memory Manager)
if [ -f "$CUDF_PREFIX/include/rmm/rmm.hpp" ]; then
    RMM_INCLUDE="$CUDF_PREFIX/include"
else
    RMM_INCLUDE="$CUDF_INCLUDE"
fi

# Verify libcudf shared library exists
if [ ! -f "$CUDF_LIB/libcudf.so" ]; then
    echo "WARNING: libcudf.so not found in $CUDF_LIB"
    echo "         Package may fail to load at runtime."
fi

# Generate Makevars from template
sed -e "s|@CUDA_HOME@|$CUDA_HOME|g" \
    -e "s|@CUDF_INCLUDE@|$CUDF_INCLUDE|g" \
    -e "s|@CUDF_LIB@|$CUDF_LIB|g" \
    -e "s|@RMM_INCLUDE@|$RMM_INCLUDE|g" \
    src/Makevars.in > src/Makevars

echo ""
echo "Configuration complete. Generated src/Makevars:"
cat src/Makevars
echo ""
echo "Run 'R CMD INSTALL .' to build the package."
```

### Dockerfile

```dockerfile
# Dockerfile for cuplr development and CI
# Based on RAPIDS CUDA 12 developer image

ARG RAPIDS_VERSION=25.12
ARG CUDA_VERSION=12.5
ARG UBUNTU_VERSION=22.04

FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py3.11-amd64

LABEL maintainer="your@email.com"
LABEL description="cuplr development environment with RAPIDS libcudf"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    gnupg \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Add R repository and install R
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
    gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > \
    /etc/apt/sources.list.d/r-project.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R package dependencies
RUN R -e "install.packages(c( \
    'Rcpp', 'dplyr', 'rlang', 'vctrs', 'pillar', 'glue', 'cli', \
    'testthat', 'bench', 'arrow', 'nanoarrow', 'bit64', 'reticulate', \
    'devtools', 'roxygen2', 'pkgdown' \
), repos='https://cloud.r-project.org')"

# Set environment variables for cuplr build
ENV CUDA_HOME=/usr/local/cuda
ENV CUDF_HOME=/opt/conda
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

# Create working directory
WORKDIR /cuplr

# Copy package source
COPY . /cuplr

# Configure and build
RUN chmod +x configure && \
    ./configure && \
    R CMD INSTALL .

# Run tests by default
CMD ["R", "-e", "testthat::test_package('cuplr')"]
```

### GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvcr.io/nvidia/rapidsai/base:25.12-cuda12.5-py3.11-amd64
      options: --gpus all

    steps:
      - uses: actions/checkout@v4

      - name: Install R
        run: |
          apt-get update
          apt-get install -y software-properties-common
          wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
            gpg --dearmor -o /usr/share/keyrings/r-project.gpg
          echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > \
            /etc/apt/sources.list.d/r-project.list
          apt-get update
          apt-get install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev

      - name: Install R dependencies
        run: |
          R -e "install.packages(c('Rcpp', 'dplyr', 'rlang', 'vctrs', 'pillar', 'glue', 'cli', 'testthat', 'bench'), repos='https://cloud.r-project.org')"

      - name: Configure
        run: |
          export CUDA_HOME=/usr/local/cuda
          export CUDF_HOME=/opt/conda
          chmod +x configure
          ./configure

      - name: Build package
        run: R CMD build .

      - name: Check package
        run: R CMD check cuplr_*.tar.gz --no-manual

      - name: Install package
        run: R CMD INSTALL cuplr_*.tar.gz

      - name: Run tests
        run: R -e "testthat::test_package('cuplr')"

  benchmark:
    needs: build
    runs-on: ubuntu-latest
    container:
      image: nvcr.io/nvidia/rapidsai/base:25.12-cuda12.5-py3.11-amd64
      options: --gpus all

    steps:
      - uses: actions/checkout@v4

      - name: Setup (same as build job)
        run: |
          # ... same setup steps ...
          echo "Setup complete"

      - name: Run benchmarks
        run: |
          R -e "source('inst/benchmarks/run_benchmarks.R')"

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: inst/benchmarks/results/
```

---

## 8. Minimal Working Prototype

### Directory Structure

```
cuplr/
├── DESCRIPTION
├── NAMESPACE
├── LICENSE
├── configure
├── R/
│   ├── zzz.R
│   ├── tbl_gpu.R
│   ├── dplyr-filter.R
│   ├── dplyr-select.R
│   ├── dplyr-mutate.R
│   ├── dplyr-arrange.R
│   ├── dplyr-group.R
│   ├── dplyr-summarise.R
│   ├── dplyr-join.R
│   ├── collect.R
│   └── utils.R
├── src/
│   ├── Makevars.in
│   ├── init.cpp
│   ├── gpu_table.hpp
│   ├── transfer.cpp
│   ├── filter.cpp
│   ├── sort.cpp
│   ├── groupby.cpp
│   └── RcppExports.cpp
├── inst/
│   ├── docker/
│   │   └── Dockerfile
│   └── benchmarks/
│       └── run_benchmarks.R
├── tests/
│   └── testthat/
│       ├── test-basic.R
│       ├── test-filter.R
│       ├── test-mutate.R
│       └── helper-cuplr.R
└── man/
    └── (generated by roxygen2)
```

### R/zzz.R - Package Load Hook

```r
#' @useDynLib cuplr, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

.onLoad <- function(libname, pkgname) {
  # Check GPU availability
  gpu_ok <- tryCatch(
    {
      .Call(`_cuplr_check_gpu`)
      TRUE
    },
    error = function(e) {
      packageStartupMessage(
        "cuplr: No GPU detected or CUDA unavailable. ",
        "GPU operations will fail."
      )
      FALSE
    }
  )

  # Set package options
  op <- options()
  op.cuplr <- list(
    cuplr.verbose = FALSE,
    cuplr.lazy = TRUE,
    cuplr.gpu_available = gpu_ok
  )
  toset <- !(names(op.cuplr) %in% names(op))
  if (any(toset)) options(op.cuplr[toset])

  invisible()
}

.onAttach <- function(libname, pkgname) {
  if (getOption("cuplr.gpu_available", FALSE)) {
    info <- gpu_info()
    packageStartupMessage(
      "cuplr: Using GPU '", info$name, "' with ",
      round(info$memory_total / 1e9, 1), " GB memory"
    )
  }
}
```

### R/tbl_gpu.R - Core Class

```r
#' Create a GPU-backed tibble
#'
#' @param data A data frame to transfer to GPU memory
#' @param ... Additional arguments (unused)
#' @return A `tbl_gpu` object
#' @export
#' @examples
#' if (interactive()) {
#'   df <- data.frame(x = 1:1000, y = rnorm(1000))
#'   gpu_df <- tbl_gpu(df)
#'   gpu_df
#' }
tbl_gpu <- function(data, ...) {
  UseMethod("tbl_gpu")
}

#' @export
tbl_gpu.data.frame <- function(data, ...) {
  # Transfer to GPU
  ptr <- .Call(`_cuplr_df_to_gpu`, data)

  schema <- list(
    names = names(data),
    types = vapply(data, gpu_type_from_r, character(1))
  )

  new_tbl_gpu(ptr = ptr, schema = schema)
}

#' @export
tbl_gpu.tbl_gpu <- function(data, ...) {
  data
}

# Internal constructor
new_tbl_gpu <- function(ptr = NULL,
                        schema = list(names = character(), types = character()),
                        lazy_ops = list(),
                        groups = character()) {
  structure(
    list(
      ptr = ptr,
      schema = schema,
      lazy_ops = lazy_ops,
      groups = groups
    ),
    class = c("tbl_gpu", "tbl_lazy", "tbl")
  )
}

#' @export
is_tbl_gpu <- function(x) {
  inherits(x, "tbl_gpu")
}

#' @export
as_tbl_gpu <- function(x, ...) {
  tbl_gpu(x, ...)
}

# Print method
#' @export
print.tbl_gpu <- function(x, ..., n = 10) {
  cat("# A GPU tibble: ")

  if (is.null(x$ptr)) {
    cat("[lazy, not materialized]\n")
    cat("# Schema: ", paste(x$schema$names, collapse = ", "), "\n")
    cat("# Operations pending: ", length(x$lazy_ops), "\n")
  } else {
    dims <- dim(x)
    cat(format(dims[1], big.mark = ","), " x ", dims[2], "\n", sep = "")

    if (length(x$groups) > 0) {
      cat("# Groups: ", paste(x$groups, collapse = ", "), "\n")
    }

    # Show first n rows
    preview <- head(collect(x), n)
    print(tibble::as_tibble(preview))

    if (dims[1] > n) {
      cat("# ... with ", format(dims[1] - n, big.mark = ","),
          " more rows\n", sep = "")
    }
  }

  invisible(x)
}

#' @export
dim.tbl_gpu <- function(x) {
  if (is.null(x$ptr)) {
    c(NA_integer_, length(x$schema$names))
  } else {
    .Call(`_cuplr_gpu_dim`, x$ptr)
  }
}

#' @export
names.tbl_gpu <- function(x) {
  x$schema$names
}

#' @export
`names<-.tbl_gpu` <- function(x, value) {
  x$schema$names <- value
  x
}

# Type helper
gpu_type_from_r <- function(x) {
  if (is.logical(x)) return("BOOL8")
  if (is.integer(x)) return("INT32")
  if (is.double(x)) {
    if (inherits(x, "Date")) return("TIMESTAMP_DAYS")
    if (inherits(x, "POSIXct")) return("TIMESTAMP_MICROSECONDS")
    return("FLOAT64")
  }
  if (is.character(x)) return("STRING")
  if (is.factor(x)) return("DICTIONARY32")
  if (inherits(x, "integer64")) return("INT64")
  "UNKNOWN"
}
```

### R/dplyr-filter.R

```r
#' @importFrom dplyr filter
#' @importFrom rlang enquos quo_get_expr eval_tidy
#' @export
filter.tbl_gpu <- function(.data, ..., .preserve = FALSE) {
  dots <- enquos(...)

  if (length(dots) == 0) {
    return(.data)
  }

  # For lazy mode, store operation
 if (getOption("cuplr.lazy", TRUE) && length(.data$lazy_ops) > 0) {
    .data$lazy_ops <- c(.data$lazy_ops, list(
      list(op = "filter", args = dots)
    ))
    return(.data)
  }

  # Eager execution
  for (quo in dots) {
    .data <- execute_filter(.data, quo)
  }

  .data
}

execute_filter <- function(.data, quo) {
  expr <- quo_get_expr(quo)

  # Parse simple comparison: col > value
  if (is.call(expr) && length(expr) == 3) {
    op <- as.character(expr[[1]])
    lhs <- expr[[2]]
    rhs <- expr[[3]]

    # Check if LHS is column name
    if (is.symbol(lhs)) {
      col_name <- as.character(lhs)
      col_idx <- match(col_name, .data$schema$names) - 1L  # 0-indexed

      if (is.na(col_idx)) {
        cli::cli_abort("Column '{col_name}' not found in GPU table")
      }

      # Evaluate RHS
      value <- eval_tidy(rhs)

      # Map R operator to C++ function
      new_ptr <- switch(op,
        ">"  = .Call(`_cuplr_gpu_filter_gt`, .data$ptr, col_idx, value),
        ">=" = .Call(`_cuplr_gpu_filter_gte`, .data$ptr, col_idx, value),
        "<"  = .Call(`_cuplr_gpu_filter_lt`, .data$ptr, col_idx, value),
        "<=" = .Call(`_cuplr_gpu_filter_lte`, .data$ptr, col_idx, value),
        "==" = .Call(`_cuplr_gpu_filter_eq`, .data$ptr, col_idx, value),
        "!=" = .Call(`_cuplr_gpu_filter_neq`, .data$ptr, col_idx, value),
        cli::cli_abort("Unsupported filter operator: {op}")
      )

      new_tbl_gpu(
        ptr = new_ptr,
        schema = .data$schema,
        groups = .data$groups
      )
    } else {
      cli::cli_abort("Complex filter expressions not yet supported")
    }
  } else {
    cli::cli_abort("Unsupported filter expression type")
  }
}
```

### R/collect.R

```r
#' Materialize GPU table to R data frame
#'
#' @param x A `tbl_gpu` object
#' @param ... Additional arguments (unused)
#' @return A data frame
#' @export
collect.tbl_gpu <- function(x, ...) {
  # Execute any pending lazy operations first
  if (length(x$lazy_ops) > 0) {
    x <- compute(x)
  }

  if (is.null(x$ptr)) {
    cli::cli_abort("Cannot collect: GPU table has no data pointer")
  }

  # Transfer from GPU to R
  df <- .Call(`_cuplr_gpu_to_df`, x$ptr, x$schema$names)

  # Apply type conversions
  for (i in seq_along(df)) {
    gpu_type <- x$schema$types[i]
    if (gpu_type == "TIMESTAMP_DAYS") {
      df[[i]] <- as.Date(df[[i]], origin = "1970-01-01")
    } else if (gpu_type == "TIMESTAMP_MICROSECONDS") {
      df[[i]] <- as.POSIXct(df[[i]] / 1e6, origin = "1970-01-01")
    }
  }

  tibble::as_tibble(df)
}

#' Execute lazy operations and keep result on GPU
#'
#' @param x A `tbl_gpu` object
#' @param ... Additional arguments (unused)
#' @return A `tbl_gpu` object with operations materialized
#' @export
compute.tbl_gpu <- function(x, ...) {
  if (length(x$lazy_ops) == 0) {
    return(x)
  }

  if (getOption("cuplr.verbose", FALSE)) {
    message("cuplr: Executing ", length(x$lazy_ops), " lazy operations")
  }

  # Execute operations in order
  result <- x
  result$lazy_ops <- list()  # Clear lazy ops for execution

  for (op in x$lazy_ops) {
    result <- execute_lazy_op(result, op)
  }

  result
}

execute_lazy_op <- function(.data, op) {
  switch(op$op,
    "filter" = {
      for (quo in op$args) {
        .data <- execute_filter(.data, quo)
      }
      .data
    },
    "select" = execute_select(.data, op$args),
    "mutate" = execute_mutate(.data, op$args),
    "arrange" = execute_arrange(.data, op$args),
    "group_by" = execute_group_by(.data, op$args),
    "summarise" = execute_summarise(.data, op$args),
    cli::cli_abort("Unknown lazy operation: {op$op}")
  )
}
```

### src/init.cpp - Registration

```cpp
// src/init.cpp
#include <Rcpp.h>
#include <R_ext/Rdynload.h>

// Forward declarations
SEXP _cuplr_check_gpu();
SEXP _cuplr_df_to_gpu(SEXP df);
SEXP _cuplr_gpu_to_df(SEXP xptr, SEXP names);
SEXP _cuplr_gpu_dim(SEXP xptr);
SEXP _cuplr_gpu_filter_gt(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_gte(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_lt(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_lte(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_eq(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_neq(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_info();

static const R_CallMethodDef CallEntries[] = {
    {"_cuplr_check_gpu", (DL_FUNC) &_cuplr_check_gpu, 0},
    {"_cuplr_df_to_gpu", (DL_FUNC) &_cuplr_df_to_gpu, 1},
    {"_cuplr_gpu_to_df", (DL_FUNC) &_cuplr_gpu_to_df, 2},
    {"_cuplr_gpu_dim", (DL_FUNC) &_cuplr_gpu_dim, 1},
    {"_cuplr_gpu_filter_gt", (DL_FUNC) &_cuplr_gpu_filter_gt, 3},
    {"_cuplr_gpu_filter_gte", (DL_FUNC) &_cuplr_gpu_filter_gte, 3},
    {"_cuplr_gpu_filter_lt", (DL_FUNC) &_cuplr_gpu_filter_lt, 3},
    {"_cuplr_gpu_filter_lte", (DL_FUNC) &_cuplr_gpu_filter_lte, 3},
    {"_cuplr_gpu_filter_eq", (DL_FUNC) &_cuplr_gpu_filter_eq, 3},
    {"_cuplr_gpu_filter_neq", (DL_FUNC) &_cuplr_gpu_filter_neq, 3},
    {"_cuplr_gpu_info", (DL_FUNC) &_cuplr_gpu_info, 0},
    {NULL, NULL, 0}
};

extern "C" void R_init_cuplr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
```

### tests/testthat/test-basic.R

```r
test_that("tbl_gpu can be created from data.frame", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(
    x = c(1, 2, 3, 4, 5),
    y = c(10.5, 20.5, 30.5, 40.5, 50.5)
  )

  gpu_df <- tbl_gpu(df)

  expect_s3_class(gpu_df, "tbl_gpu")
  expect_equal(dim(gpu_df), c(5L, 2L))
  expect_equal(names(gpu_df), c("x", "y"))
})

test_that("collect returns data to R", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(x = 1:5, y = 6:10)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_s3_class(result, "tbl_df")
  expect_equal(result$x, 1:5)
  expect_equal(result$y, 6:10)
})

test_that("filter works with simple comparisons", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(x = 1:10, y = 11:20)
  gpu_df <- tbl_gpu(df)

  result <- gpu_df %>%
    filter(x > 5) %>%
    collect()

  expected <- df %>%
    dplyr::filter(x > 5)

  expect_equal(result$x, expected$x)
  expect_equal(result$y, expected$y)
})

test_that("NA values are preserved through round-trip", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(
    x = c(1, NA, 3, NA, 5),
    y = c(NA, 2.5, NA, 4.5, 5.5)
  )

  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(is.na(result$x), is.na(df$x))
  expect_equal(is.na(result$y), is.na(df$y))
})
```

---

## 9. Lazy Translation & AST Approach

### AST Node Definitions

```r
# R/ast.R - Internal AST representation

# Base AST node
ast_node <- function(type, ...) {
  structure(
    list(type = type, ...),
    class = c(paste0("ast_", type), "ast_node")
  )
}

# Operation nodes
ast_filter <- function(predicates) {
  ast_node("filter", predicates = predicates)
}

ast_select <- function(columns) {
  ast_node("select", columns = columns)
}

ast_mutate <- function(expressions) {
  ast_node("mutate", expressions = expressions)
}

ast_arrange <- function(columns, descending) {
  ast_node("arrange", columns = columns, descending = descending)
}

ast_group_by <- function(columns) {
  ast_node("group_by", columns = columns)
}

ast_summarise <- function(aggregations) {
  ast_node("summarise", aggregations = aggregations)
}

ast_join <- function(type, right_table, by) {
  ast_node("join", join_type = type, right = right_table, by = by)
}

# Expression nodes (for predicates and computations)
ast_binary_op <- function(op, left, right) {
  ast_node("binary_op", operator = op, left = left, right = right)
}

ast_column_ref <- function(name) {
  ast_node("column_ref", name = name)
}

ast_literal <- function(value) {
  ast_node("literal", value = value)
}

ast_function_call <- function(fn, args) {
  ast_node("function_call", fn = fn, args = args)
}
```

### Expression Parser

```r
# R/parse_expr.R - Parse R expressions to AST

parse_expr_to_ast <- function(expr, env = parent.frame()) {
  if (is.symbol(expr)) {
    # Column reference
    return(ast_column_ref(as.character(expr)))
  }

  if (is.atomic(expr) && length(expr) == 1) {
    # Literal value
    return(ast_literal(expr))
  }

  if (is.call(expr)) {
    fn <- as.character(expr[[1]])

    # Binary operators
    if (fn %in% c("+", "-", "*", "/", "^", "%%", "%/%",
                  ">", ">=", "<", "<=", "==", "!=",
                  "&", "|")) {
      left <- parse_expr_to_ast(expr[[2]], env)
      right <- parse_expr_to_ast(expr[[3]], env)
      return(ast_binary_op(fn, left, right))
    }

    # Unary operators
    if (fn == "!" && length(expr) == 2) {
      return(ast_node("unary_op", operator = "!", operand = parse_expr_to_ast(expr[[2]], env)))
    }

    # Function calls
    args <- lapply(expr[-1], parse_expr_to_ast, env = env)
    return(ast_function_call(fn, args))
  }

  cli::cli_abort("Cannot parse expression: {deparse(expr)}")
}

# Convert quosure to AST
quosure_to_ast <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)
  parse_expr_to_ast(expr, env)
}
```

### Query Optimizer

```r
# R/optimizer.R - AST optimization passes

optimize_ast <- function(ops) {
  ops <- push_down_filters(ops)
  ops <- push_down_projections(ops)
  ops <- fuse_consecutive_filters(ops)
  ops
}

# Predicate pushdown: move filters earlier in pipeline
push_down_filters <- function(ops) {
  if (length(ops) < 2) return(ops)

  result <- list()
  pending_filters <- list()

  for (op in ops) {
    if (op$type == "filter") {
      # Collect filter predicates
      pending_filters <- c(pending_filters, op$predicates)
    } else if (op$type %in% c("select", "mutate")) {
      # Can push filters before select/mutate if columns exist
      if (length(pending_filters) > 0) {
        # Check which predicates can be pushed
        pushable <- vapply(pending_filters, function(pred) {
          cols <- extract_column_refs(pred)
          # All referenced columns must exist before this op
          TRUE  # Simplified - real implementation checks column existence
        }, logical(1))

        if (any(pushable)) {
          result <- c(result, list(ast_filter(pending_filters[pushable])))
          pending_filters <- pending_filters[!pushable]
        }
      }
      result <- c(result, list(op))
    } else {
      # Flush pending filters
      if (length(pending_filters) > 0) {
        result <- c(result, list(ast_filter(pending_filters)))
        pending_filters <- list()
      }
      result <- c(result, list(op))
    }
  }

  # Flush remaining filters
  if (length(pending_filters) > 0) {
    result <- c(result, list(ast_filter(pending_filters)))
  }

  result
}

# Projection pushdown: only read needed columns
push_down_projections <- function(ops) {
  # Analyze which columns are used by each operation
  # Remove unused columns early
  ops  # Placeholder - full implementation tracks column usage
}

# Fuse consecutive filters into single operation
fuse_consecutive_filters <- function(ops) {
  if (length(ops) < 2) return(ops)

  result <- list()
  i <- 1

  while (i <= length(ops)) {
    if (ops[[i]]$type == "filter") {
      # Collect consecutive filters
      predicates <- ops[[i]]$predicates
      while (i < length(ops) && ops[[i + 1]]$type == "filter") {
        i <- i + 1
        predicates <- c(predicates, ops[[i]]$predicates)
      }
      result <- c(result, list(ast_filter(predicates)))
    } else {
      result <- c(result, list(ops[[i]]))
    }
    i <- i + 1
  }

  result
}

# Extract column references from AST node
extract_column_refs <- function(node) {
  if (inherits(node, "ast_column_ref")) {
    return(node$name)
  }
  if (is.list(node)) {
    unlist(lapply(node, extract_column_refs))
  } else {
    character(0)
  }
}
```

### Lowering AST to libcudf Calls

```r
# R/lower.R - Convert AST to libcudf operations

lower_to_cudf <- function(tbl, ops) {
  if (getOption("cuplr.verbose", FALSE)) {
    message("cuplr: Lowering ", length(ops), " operations to libcudf")
  }

  for (op in ops) {
    tbl <- lower_op(tbl, op)
  }

  tbl
}

lower_op <- function(tbl, op) {
  if (getOption("cuplr.verbose", FALSE)) {
    message("  -> ", op$type)
  }

  switch(op$type,
    "filter" = lower_filter(tbl, op),
    "select" = lower_select(tbl, op),
    "mutate" = lower_mutate(tbl, op),
    "arrange" = lower_arrange(tbl, op),
    "group_by" = lower_group_by(tbl, op),
    "summarise" = lower_summarise(tbl, op),
    "join" = lower_join(tbl, op),
    cli::cli_abort("Cannot lower operation: {op$type}")
  )
}

lower_filter <- function(tbl, op) {
  # Build combined boolean mask from all predicates
  mask_ptr <- NULL

  for (pred in op$predicates) {
    pred_mask <- evaluate_predicate(tbl, pred)
    if (is.null(mask_ptr)) {
      mask_ptr <- pred_mask
    } else {
      # AND masks together
      mask_ptr <- .Call(`_cuplr_gpu_and_masks`, mask_ptr, pred_mask)
    }
  }

  new_ptr <- .Call(`_cuplr_gpu_apply_mask`, tbl$ptr, mask_ptr)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = tbl$schema,
    groups = tbl$groups
  )
}

evaluate_predicate <- function(tbl, pred) {
  if (inherits(pred, "ast_binary_op")) {
    left <- evaluate_expr(tbl, pred$left)
    right <- evaluate_expr(tbl, pred$right)

    op_code <- switch(pred$operator,
      ">" = 1L, ">=" = 2L, "<" = 3L, "<=" = 4L,
      "==" = 5L, "!=" = 6L,
      "&" = 7L, "|" = 8L,
      cli::cli_abort("Unsupported predicate operator: {pred$operator}")
    )

    .Call(`_cuplr_gpu_binary_op`, left, right, op_code)
  } else {
    cli::cli_abort("Cannot evaluate predicate: {class(pred)[1]}")
  }
}

evaluate_expr <- function(tbl, expr) {
  if (inherits(expr, "ast_column_ref")) {
    col_idx <- match(expr$name, tbl$schema$names) - 1L
    .Call(`_cuplr_gpu_get_column`, tbl$ptr, col_idx)
  } else if (inherits(expr, "ast_literal")) {
    .Call(`_cuplr_gpu_scalar`, expr$value)
  } else if (inherits(expr, "ast_binary_op")) {
    left <- evaluate_expr(tbl, expr$left)
    right <- evaluate_expr(tbl, expr$right)
    op_code <- match(expr$operator, c("+", "-", "*", "/", "^", "%%", "%/%"))
    .Call(`_cuplr_gpu_binary_op`, left, right, op_code)
  } else {
    cli::cli_abort("Cannot evaluate expression: {class(expr)[1]}")
  }
}
```

### Verbose Mode Output Example

```
> options(cuplr.verbose = TRUE)
> tbl_gpu(df) %>% filter(x > 10, y < 50) %>% mutate(z = x + y) %>% collect()

cuplr: Executing 2 lazy operations
cuplr: Optimizing AST...
  -> Fused 2 filter predicates
cuplr: Lowering 2 operations to libcudf
  -> filter
     cudf::binary_operation(col[0], scalar(10), GREATER) -> mask1
     cudf::binary_operation(col[1], scalar(50), LESS) -> mask2
     cudf::binary_operation(mask1, mask2, BITWISE_AND) -> mask_final
     cudf::apply_boolean_mask(table, mask_final)
  -> mutate
     cudf::binary_operation(col[0], col[1], ADD) -> new_col
     append column to table
# A tibble: 3 x 3
      x     y     z
  <dbl> <dbl> <dbl>
1    15    30    45
2    20    40    60
3    25    45    70
```

---

## 10. Testing & Validation

### Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit | Test individual R functions | `tests/testthat/test-*.R` |
| Integration | Compare GPU vs CPU results | `tests/testthat/test-integration.R` |
| Performance | Benchmark against dplyr | `inst/benchmarks/` |
| Edge cases | NA handling, empty tables, types | `tests/testthat/test-edge-cases.R` |

### tests/testthat/helper-cuplr.R

```r
# Test helper functions

skip_if_no_gpu <- function() {
  skip_if_not(
    getOption("cuplr.gpu_available", FALSE),
    "No GPU available for testing"
  )
}

# Compare GPU and CPU results with tolerance
expect_gpu_cpu_equal <- function(gpu_result, cpu_result, tolerance = 1e-10) {
  gpu_df <- if (is_tbl_gpu(gpu_result)) collect(gpu_result) else gpu_result
  cpu_df <- if (inherits(cpu_result, "data.frame")) cpu_result else as.data.frame(cpu_result)

  expect_equal(nrow(gpu_df), nrow(cpu_df))
  expect_equal(ncol(gpu_df), ncol(cpu_df))
  expect_equal(names(gpu_df), names(cpu_df))

  for (col in names(gpu_df)) {
    if (is.numeric(gpu_df[[col]])) {
      expect_equal(gpu_df[[col]], cpu_df[[col]], tolerance = tolerance,
                   label = paste("Column", col))
    } else {
      expect_equal(gpu_df[[col]], cpu_df[[col]], label = paste("Column", col))
    }
  }
}

# Generate test data
make_test_df <- function(n = 1000, seed = 42) {
  set.seed(seed)
  data.frame(
    int_col = sample(1:100, n, replace = TRUE),
    dbl_col = rnorm(n, mean = 50, sd = 10),
    chr_col = sample(letters[1:10], n, replace = TRUE),
    grp_col = sample(LETTERS[1:5], n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}
```

### tests/testthat/test-filter.R

```r
test_that("filter with > works correctly", {
  skip_if_no_gpu()

  df <- make_test_df(1000)
  gpu_df <- tbl_gpu(df)

  gpu_result <- gpu_df %>% filter(int_col > 50)
  cpu_result <- df %>% dplyr::filter(int_col > 50)

  expect_gpu_cpu_equal(gpu_result, cpu_result)
})

test_that("filter with multiple conditions works", {
  skip_if_no_gpu()

  df <- make_test_df(1000)
  gpu_df <- tbl_gpu(df)

  gpu_result <- gpu_df %>% filter(int_col > 25, int_col < 75)
  cpu_result <- df %>% dplyr::filter(int_col > 25, int_col < 75)

  expect_gpu_cpu_equal(gpu_result, cpu_result)
})

test_that("filter handles NA values correctly", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, NA, 3, 4, NA, 6),
    y = c(10, 20, NA, 40, 50, NA)
  )
  gpu_df <- tbl_gpu(df)

  # Filter should exclude NA comparisons (like R)
  gpu_result <- gpu_df %>% filter(x > 2) %>% collect()
  cpu_result <- df %>% dplyr::filter(x > 2)

  expect_equal(nrow(gpu_result), nrow(cpu_result))
  expect_equal(gpu_result$x, cpu_result$x)
})

test_that("filter on empty result returns empty table", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10)
  gpu_df <- tbl_gpu(df)

  result <- gpu_df %>% filter(x > 100) %>% collect()

  expect_equal(nrow(result), 0)
  expect_equal(names(result), "x")
})
```

### tests/testthat/test-integration.R

```r
test_that("complex pipeline matches dplyr", {
  skip_if_no_gpu()

  df <- make_test_df(10000)
  gpu_df <- tbl_gpu(df)

  gpu_result <- gpu_df %>%
    filter(int_col > 20) %>%
    mutate(computed = dbl_col * 2) %>%
    group_by(grp_col) %>%
    summarise(
      mean_val = mean(computed),
      count = n()
    ) %>%
    arrange(grp_col) %>%
    collect()

  cpu_result <- df %>%
    dplyr::filter(int_col > 20) %>%
    dplyr::mutate(computed = dbl_col * 2) %>%
    dplyr::group_by(grp_col) %>%
    dplyr::summarise(
      mean_val = mean(computed),
      count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::arrange(grp_col)

  expect_gpu_cpu_equal(gpu_result, cpu_result, tolerance = 1e-6)
})

test_that("join operations match dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(
    key = c(1, 2, 3, 4, 5),
    val_left = c("a", "b", "c", "d", "e")
  )
  right_df <- data.frame(
    key = c(2, 3, 4, 6, 7),
    val_right = c("x", "y", "z", "w", "v")
  )

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  # Left join
  gpu_result <- gpu_left %>%
    left_join(gpu_right, by = "key") %>%
    collect()

  cpu_result <- left_df %>%
    dplyr::left_join(right_df, by = "key")

  expect_gpu_cpu_equal(gpu_result, cpu_result)

  # Inner join
  gpu_result <- gpu_left %>%
    inner_join(gpu_right, by = "key") %>%
    collect()

  cpu_result <- left_df %>%
    dplyr::inner_join(right_df, by = "key")

  expect_gpu_cpu_equal(gpu_result, cpu_result)
})
```

### Benchmark Script

```r
# inst/benchmarks/run_benchmarks.R

library(cuplr)
library(dplyr)
library(bench)

# Configuration
sizes <- c(1e5, 1e6, 1e7, 1e8)
results <- list()

cat("cuplr Benchmark Suite\n")
cat("=====================\n\n")

for (n in sizes) {
  cat(sprintf("Dataset size: %s rows\n", format(n, big.mark = ",")))

  # Generate data
  set.seed(42)
  df <- data.frame(
    x = runif(n),
    y = runif(n),
    g = sample(letters[1:26], n, replace = TRUE)
  )

  gpu_df <- tbl_gpu(df)

  # Benchmark filter
  bm_filter <- bench::mark(
    cpu = df %>% filter(x > 0.5),
    gpu = gpu_df %>% filter(x > 0.5) %>% collect(),
    check = FALSE,
    min_iterations = 5
  )

  # Benchmark group_by + summarise
  bm_group <- bench::mark(
    cpu = df %>% group_by(g) %>% summarise(mean_x = mean(x), .groups = "drop"),
    gpu = gpu_df %>% group_by(g) %>% summarise(mean_x = mean(x)) %>% collect(),
    check = FALSE,
    min_iterations = 5
  )

  # Benchmark arrange
  bm_sort <- bench::mark(
    cpu = df %>% arrange(x),
    gpu = gpu_df %>% arrange(x) %>% collect(),
    check = FALSE,
    min_iterations = 5
  )

  results[[as.character(n)]] <- list(
    filter = bm_filter,
    group_by = bm_group,
    arrange = bm_sort
  )

  cat(sprintf("  filter:   CPU %.2fs, GPU %.2fs (%.1fx speedup)\n",
              as.numeric(bm_filter$median[1]),
              as.numeric(bm_filter$median[2]),
              as.numeric(bm_filter$median[1]) / as.numeric(bm_filter$median[2])))

  cat(sprintf("  group_by: CPU %.2fs, GPU %.2fs (%.1fx speedup)\n",
              as.numeric(bm_group$median[1]),
              as.numeric(bm_group$median[2]),
              as.numeric(bm_group$median[1]) / as.numeric(bm_group$median[2])))

  cat(sprintf("  arrange:  CPU %.2fs, GPU %.2fs (%.1fx speedup)\n",
              as.numeric(bm_sort$median[1]),
              as.numeric(bm_sort$median[2]),
              as.numeric(bm_sort$median[1]) / as.numeric(bm_sort$median[2])))

  cat("\n")

  # Clean up GPU memory
  rm(gpu_df)
  gc()
}

# Save results
saveRDS(results, "inst/benchmarks/results/benchmark_results.rds")
cat("Results saved to inst/benchmarks/results/benchmark_results.rds\n")
```

