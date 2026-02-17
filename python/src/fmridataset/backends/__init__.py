"""Storage backend implementations for fmridataset."""

from .matrix_backend import MatrixBackend

__all__ = [
    "MatrixBackend",
    # Lazy imports for optional-dep backends:
    # from .nifti_backend import NiftiBackend
    # from .h5_backend import H5Backend
    # from .zarr_backend import ZarrBackend
    # from .latent_backend import LatentBackend
    # from .study_backend import StudyBackend
]
