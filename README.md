# cuplyr <img src="man/figures/logo.png" align="right" height="138" />

#### dplyr backend for GPU acceleration via RAPIDS cuDF

cuplyr implements a dplyr backend powered by [RAPIDS cuDF](https://github.com/rapidsai/cudf), NVIDIA's GPU DataFrame library. It allows users to write standard dplyr code while executing operations on GPU hardware.

```r
library(cuplyr)

tbl_gpu(sales_data) |>
  filter(year >= 2020, amount > 0) |>
  mutate(revenue = amount * price) |>
  group_by(region, quarter) |>
  summarise(total = sum(revenue)) |>
  arrange(desc(total)) |>
  collect()
```

## About

cuplyr translates dplyr operations into cuDF execution on NVIDIA GPUs. It follows the same backend pattern as dbplyr: write standard R code, execute on GPU hardware. This approach can provide significant speedups on larger datasets (typically >10M rows) without requiring major code changes.

**Built on [RAPIDS cuDF](https://rapids.ai/)**: cuDF is an open-source GPU DataFrame library developed by NVIDIA's RAPIDS team. It provides optimized CUDA kernels for data manipulation operations, backed by Apache Arrow's columnar memory format. cuplyr provides an R interface to this execution engine.


## Status

**v0.0.1 – Early development**

This is experimental software under active development. Breaking changes should be expected.

### Supported operations

- `filter()` – row filtering with comparison and logical operators
- `select()` – column selection and reordering
- `mutate()` – column transformations and arithmetic
- `arrange()` – row sorting with `desc()` support, NA handling follows dplyr conventions
- `group_by()` + `summarise()` – grouped aggregations (`sum`, `mean`, `min`, `max`, `n`)
- `left_join()`, `right_join()`, `inner_join()`, `full_join()` – GPU joins on key columns
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

- **GPU**: NVIDIA with Compute Capability 6.0+ (Pascal generation, 2016 or newer)
- **Driver**: NVIDIA driver 525.60.13 or newer
- **CUDA**: 12.0 or newer
- **OS**: Linux x86_64 (native), Windows/macOS (via Docker)
- **R**: 4.3.0 or newer
- **RAPIDS**: libcudf 25.12 or newer

> **Note**: cuplyr requires an NVIDIA GPU with CUDA support. AMD and Intel GPUs are not supported. Windows and macOS users can run cuplyr via Docker containers.

## Installation

### Quick Start (Docker)

The fastest way to try cuplyr on any platform with an NVIDIA GPU:

```bash
# Pull RAPIDS base image and start container
docker run --gpus all -it rapidsai/base:25.12-cuda12-py3.11 bash

# Inside the container, install R and dependencies
apt-get update && apt-get install -y r-base r-base-dev

# Install R package dependencies
R -e 'install.packages(c("Rcpp", "dplyr", "rlang", "vctrs", "pillar", "glue", "cli", "tidyselect", "tibble"))'

# Install cuplyr from GitHub
R -e 'install.packages("remotes"); remotes::install_github("bbtheo/cuplyr")'
```

### Linux: Native Installation

#### Option 1: Pixi (Recommended)

[Pixi](https://pixi.sh) manages all CUDA and RAPIDS dependencies automatically:

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and enter the repository
git clone https://github.com/bbtheo/cuplyr.git
cd cuplyr

# Configure and install (handles all dependencies)
pixi run configure
pixi run install

# Verify installation
pixi run r -e 'library(cuplyr); gpu_info()'
```

#### Option 2: Conda

For users with an existing conda environment:

```bash
# Create environment with RAPIDS
conda create -n cuplyr-env -c rapidsai -c conda-forge -c nvidia \
  libcudf=25.12 librmm=25.12 cuda-toolkit=12.8 r-base=4.3

conda activate cuplyr-env

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export CUDF_HOME=$CONDA_PREFIX

# Clone and install
git clone https://github.com/bbtheo/cuplyr.git
cd cuplyr
./configure
R CMD INSTALL .
```

### Windows & macOS: Docker

cuplyr requires NVIDIA CUDA, which is only natively available on Linux. Windows and macOS users should use Docker:

**Windows**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend and enable GPU support in settings.

**macOS**: Docker GPU passthrough is limited. Consider using a cloud GPU instance (see below).

Once Docker is configured, follow the Quick Start instructions above.

### Cloud Platforms

cuplyr works on any cloud platform with NVIDIA GPUs:

- **AWS**: Use P3/P4 instances with the RAPIDS AMI
- **GCP**: Use A2/G2 instances with Deep Learning VM
- **Google Colab**: Free GPU access (limited memory)

See the [RAPIDS installation guide](https://docs.rapids.ai/install) for cloud-specific instructions.

## Verifying Installation

After installation, verify cuplyr is working:

```r
library(cuplyr)

# Check GPU detection
gpu_info()

# Test basic operations
tbl_gpu(mtcars) |>
  filter(mpg > 20) |>
  select(mpg, cyl, hp) |>
  collect()
```

## Troubleshooting

### Driver & GPU Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| "No GPU detected" | `nvidia-smi` fails | Install/update NVIDIA driver (525+) |
| "CUDA version mismatch" | Driver too old for CUDA 12 | Update driver to 525.60.13 or newer |
| "Insufficient compute capability" | GPU too old | Need Pascal+ GPU (GTX 1000 series, 2016+) |

### Build & Configuration Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| "CUDA not found" | `echo $CUDA_HOME` empty | `export CUDA_HOME=/usr/local/cuda` |
| "cudf/types.hpp not found" | libcudf missing | Install via conda/pixi or set `CUDF_HOME` |
| "configure script fails" | Missing RAPIDS libs | Ensure rmm, kvikio in same prefix as cudf |
| "undefined reference to cudf::" | Link error | Verify `CUDF_LIB` path in Makevars |
| "C++20 required" | Old compiler | Need GCC 11+ with C++20 support |

### Runtime Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| "libcudf.so not found" | Library path issue | Check `LD_LIBRARY_PATH` includes cudf lib dir |
| "CUDA out of memory" | GPU memory full | Use smaller data, call `gc()`, check `gpu_memory_state()` |
| "illegal memory access" | CUDA kernel crash | Often driver/toolkit mismatch; reinstall CUDA |

### Diagnostic Commands

```bash
# Check GPU and driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check library linking
ldd $(R RHOME)/library/cuplyr/libs/cuplyr.so | grep cudf
```

```r
# Inside R
library(cuplyr)
gpu_info()
gpu_memory_state()
```

## Acknowledgments

This project is built on [RAPIDS cuDF](https://github.com/rapidsai/cudf) by NVIDIA and the RAPIDS AI team.

---

**License**: Apache 2.0

**Maintainer**: [@bbtheo](https://github.com/bbtheo)

**Documentation**: `DEVELOPER_GUIDE.md`
