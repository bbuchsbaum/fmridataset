# fmridataset (Python)

Python port of the [fmridataset](https://github.com/bbuchsbaum/fmridataset) R package.

Unified container for fMRI datasets with pluggable storage backends,
temporal structure handling, chunked iteration, and lazy data access.

## Features

- **Pluggable backends** — MatrixBackend, NiftiBackend, H5Backend, ZarrBackend, LatentBackend, StudyBackend
- **Temporal structure** — `SamplingFrame` tracks TR, run lengths, block IDs, and run boundaries
- **Chunked iteration** — memory-efficient voxel-wise or run-wise chunking via `data_chunks()`
- **Lazy data access** — backends load data on demand; `FmriSeries` wraps results with metadata
- **Multi-subject support** — `StudyDataset` combines per-subject datasets with mask validation
- **Latent reconstruction** — `LatentDataset` reconstructs `basis @ loadings.T + offset` from HDF5
- **Backend registry** — register/create backends by name via `BackendRegistry`

## Installation

```bash
# Core (numpy + pandas only)
pip install -e .

# With specific backends
pip install -e ".[h5]"       # HDF5 support (h5py)
pip install -e ".[zarr]"     # Zarr support (zarr-python)
pip install -e ".[cache]"    # LRU caching (cachetools)

# Everything
pip install -e ".[all]"

# Development (tests + type checking)
pip install -e ".[dev,all]"
```

## Quick Start

```python
import numpy as np
from fmridataset import matrix_dataset, data_chunks

# Create a dataset from an in-memory matrix
mat = np.random.randn(200, 1000)  # 200 timepoints, 1000 voxels
ds = matrix_dataset(mat, TR=2.0, run_length=[100, 100])

print(ds)  # <MatrixDataset shape=(200, 1000) runs=2 TR=2.0>

# Iterate in chunks
for chunk in data_chunks(ds, nchunks=4):
    print(chunk.data.shape, chunk.voxel_ind.shape)

# Run-wise iteration
for chunk in data_chunks(ds, runwise=True):
    print(f"Run {chunk.chunk_num}: {chunk.data.shape}")
```

## Available Backends

| Backend | Format | Optional Dependency | Description |
|---------|--------|-------------------|-------------|
| `MatrixBackend` | In-memory | — | Wraps a NumPy matrix directly |
| `NiftiBackend` | NIfTI | `nibabel` | Reads `.nii` / `.nii.gz` files |
| `H5Backend` | HDF5 | `h5py` | Reads fmristore-schema HDF5 files |
| `ZarrBackend` | Zarr | `zarr` | Reads 4D Zarr arrays (cloud-ready) |
| `LatentBackend` | HDF5 | `h5py` | Latent decomposition (basis/loadings/offset) |
| `StudyBackend` | Composite | — | Combines multiple subject backends |

## API Overview

### Core Classes

- **`SamplingFrame`** — immutable temporal structure (TR, run lengths, block IDs)
- **`FmriDataset`** — wraps a backend + sampling frame + event table
- **`MatrixDataset`** — convenience subclass with direct `.datamat` access
- **`StudyDataset`** — multi-subject dataset with `.subject_ids`
- **`LatentDataset`** — latent-space dataset with `.get_latent_scores()` / `.get_spatial_loadings()`

### Constructors

- `matrix_dataset(data, TR, run_length)` — create from a NumPy matrix
- `fmri_dataset(backend, TR, run_length)` — create from any backend
- `study_dataset(datasets, subject_ids)` — combine per-subject datasets
- `latent_dataset(source, TR, run_length)` — create from latent HDF5 files

### Data Access

- `ds.get_data(rows, cols)` — read timepoints x voxels from the backend
- `ds.get_mask()` — boolean mask of valid voxels
- `data_chunks(ds, nchunks)` — chunked iteration over voxels or runs
- `to_matrix_dataset(ds)` — materialise any dataset as a MatrixDataset

## Testing

```bash
cd python

# Run all tests (excluding cross-language parity tests)
pytest tests/ -k "not parity"

# Run with coverage
pytest tests/ --cov=fmridataset

# Type checking
mypy --strict src/fmridataset/
```

## Relationship to R Package

This is a faithful port of the R `fmridataset` package. The Python version mirrors
the R architecture — pluggable storage backends, `SamplingFrame` for temporal
structure, chunked iteration, and lazy data access — while using idiomatic Python
patterns (ABC classes, dataclasses, generators). Both packages live in the same
repository and share cross-language parity tests via `rpy2`.
