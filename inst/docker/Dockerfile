FROM rapidsai/base:26.10-cuda13-py3.11

LABEL maintainer="your@email.com"
LABEL description="cuplyr development environment with RAPIDS libcudf"

# Switch to root for installing packages
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    gnupg \
    wget \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Add R repository for Ubuntu 24.04 (Noble) and install R
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
    gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" > \
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

# Set correct paths for RAPIDS conda environment
ENV CUDA_HOME=/opt/conda/targets/x86_64-linux
ENV CUDF_HOME=/opt/conda
ENV PATH=/opt/conda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /cuplyr

COPY . /cuplyr

#RUN chmod +x configure && \
#    ./configure && \
#    R CMD INSTALL .

#CMD ["R", "-e", "testthat::test_package('cuplyr')"]