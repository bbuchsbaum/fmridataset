# fmridataset <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->
[![R-CMD-check](https://github.com/bbuchsbaum/fmridataset/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bbuchsbaum/fmridataset/actions/workflows/R-CMD-check.yaml)
[![Codecov test coverage](https://codecov.io/gh/bbuchsbaum/fmridataset/branch/main/graph/badge.svg)](https://app.codecov.io/gh/bbuchsbaum/fmridataset?branch=main)
[![lint](https://github.com/bbuchsbaum/fmridataset/actions/workflows/lint.yaml/badge.svg)](https://github.com/bbuchsbaum/fmridataset/actions/workflows/lint.yaml)
[![pkgcheck](https://github.com/bbuchsbaum/fmridataset/workflows/pkgcheck/badge.svg)](https://github.com/bbuchsbaum/fmridataset/actions?query=workflow%3Apkgcheck)
[![CRAN status](https://www.r-pkg.org/badges/version/fmridataset)](https://CRAN.R-project.org/package=fmridataset)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

## Overview

`fmridataset` provides a unified S3 class for representing functional magnetic resonance imaging (fMRI) data from various sources. The package supports multiple data backends and offers a consistent interface for working with fMRI datasets regardless of their underlying storage format.

## Features

- **Unified Interface**: Work with fMRI data from NIfTI files, BIDS projects, pre-loaded NeuroVec objects, and in-memory matrices through a single API
- **Lazy Loading**: Efficient memory management with on-demand data loading
- **Flexible Backends**: Pluggable storage backends for different data formats
- **Data Chunking**: Built-in support for processing large datasets in chunks
- **Temporal Structure**: Rich sampling frame representation for run lengths, TR, and temporal organization
- **Integration Ready**: Seamlessly integrates with neuroimaging analysis workflows

## Installation

You can install the development version of fmridataset from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("bbuchsbaum/fmridataset")
```

## Quick Start

### Creating Datasets

```r
library(fmridataset)

# From NIfTI files
dataset <- fmri_dataset(
  scans = c("run1.nii", "run2.nii"),
  mask = "mask.nii", 
  TR = 2.0,
  run_length = c(240, 240)
)

# From in-memory matrix
mat_data <- matrix(rnorm(1000), nrow = 100, ncol = 10)
dataset <- matrix_dataset(
  datamat = mat_data,
  TR = 1.5, 
  run_length = 100
)

# From pre-loaded NeuroVec objects  
dataset <- fmri_mem_dataset(
  scans = list(neurovec1, neurovec2),
  mask = mask_vol,
  TR = 2.0
)
```

### Data Access

```r
# Get full data matrix
data_matrix <- get_data_matrix(dataset)

# Get spatial mask
mask <- get_mask(dataset)

# Access temporal properties
n_timepoints(dataset$sampling_frame)
n_runs(dataset$sampling_frame)
get_TR(dataset$sampling_frame)
```

### Data Chunking

```r
# Process data in chunks
chunks <- data_chunks(dataset, nchunks = 5)
for (i in 1:5) {
  chunk <- chunks$nextElem()
  # Process chunk$data, chunk$voxel_ind, etc.
}

# Run-wise processing
run_chunks <- data_chunks(dataset, runwise = TRUE)
run1_data <- run_chunks$nextElem()
```

### Type Conversions

```r
# Convert to matrix format
mat_dataset <- as.matrix_dataset(dataset)

# All dataset types support the same interface
print(dataset)
summary(dataset$sampling_frame)
```

## Architecture

The package uses a modular architecture with the following key components:

- **Storage Backends**: Pluggable data access layer (`matrix_backend`, `nifti_backend`)
- **Dataset Constructors**: High-level dataset creation functions
- **Sampling Frames**: Temporal structure representation
- **Data Access Methods**: Consistent interface for data retrieval
- **Chunking System**: Efficient processing of large datasets

## Related Packages

- [`neuroim2`](https://github.com/bbuchsbaum/neuroim2): Neuroimaging data structures
- [`fmristore`](https://github.com/bbuchsbaum/fmristore): Advanced fMRI data storage
- [`bidser`](https://github.com/bbuchsbaum/bidser): BIDS dataset utilities

## Getting Help

- Check the [package documentation](https://bbuchsbaum.github.io/fmridataset/) for detailed guides
- Report bugs or request features on [GitHub Issues](https://github.com/bbuchsbaum/fmridataset/issues)
- See the [vignettes](https://bbuchsbaum.github.io/fmridataset/articles/) for detailed examples

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

GPL (>= 3) 