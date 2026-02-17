"""fmridataset — Unified container for fMRI datasets (Python port)."""

from ._version import __version__
from .backend_protocol import BackendDims, StorageBackend
from .backend_registry import BackendRegistry, _register_builtins
from .backends.matrix_backend import MatrixBackend
from .cache import lru_cache
from .config import read_fmri_config, write_fmri_config
from .conversions import to_matrix_dataset
from .data_access import get_data, get_data_matrix, get_mask
from .data_chunks import (
    ChunkIterator,
    DataChunk,
    collect_chunks,
    data_chunks,
)
from .dataset import FmriDataset, MatrixDataset
from .dataset_constructors import (
    fmri_dataset,
    fmri_zarr_dataset,
    matrix_dataset,
    zarr_dataset,
)
from .errors import BackendIOError, ConfigError, FmriDatasetError
from .fmri_series import (
    FmriSeries,
    fmri_series,
    resolve_selector,
    resolve_timepoints,
)
from .latent_dataset import LatentDataset, latent_dataset
from .mask_utils import mask_to_logical, mask_to_volume
from .sampling_frame import SamplingFrame
from .fmri_group import FmriGroup, fmri_group
from .selectors import (
    AllSelector,
    IndexSelector,
    MaskSelector,
    ROISelector,
    SeriesSelector,
    SphereSelector,
    VoxelSelector,
)
from .study_dataset import StudyDataset, study_dataset

# Auto-register built-in backends on import
_register_builtins()

__all__ = [
    # version
    "__version__",
    # errors
    "FmriDatasetError",
    "BackendIOError",
    "ConfigError",
    # sampling frame
    "SamplingFrame",
    # backend protocol
    "StorageBackend",
    "BackendDims",
    # backend registry
    "BackendRegistry",
    # backends
    "MatrixBackend",
    # dataset
    "FmriDataset",
    "MatrixDataset",
    "LatentDataset",
    "StudyDataset",
    # constructors
    "matrix_dataset",
    "fmri_dataset",
    "fmri_zarr_dataset",
    "latent_dataset",
    "study_dataset",
    "zarr_dataset",
    # data access
    "get_data",
    "get_data_matrix",
    "get_mask",
    # mask utilities
    "mask_to_logical",
    "mask_to_volume",
    # series
    "FmriSeries",
    "fmri_series",
    "resolve_selector",
    "resolve_timepoints",
    # group
    "FmriGroup",
    "fmri_group",
    # selectors
    "SeriesSelector",
    "IndexSelector",
    "AllSelector",
    "ROISelector",
    "VoxelSelector",
    "SphereSelector",
    "MaskSelector",
    # chunks
    "DataChunk",
    "ChunkIterator",
    "data_chunks",
    "collect_chunks",
    # conversions
    "to_matrix_dataset",
    # config
    "read_fmri_config",
    "write_fmri_config",
    # cache
    "lru_cache",
]
