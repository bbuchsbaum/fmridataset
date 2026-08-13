# Latent Storage Backend

A storage backend implementation for latent space representations of
fMRI data. This backend works with data that has been decomposed into
temporal components (basis functions) and spatial loadings.

## Details

Unlike traditional voxel-based backends, latent backends store:

- Temporal basis functions (time × components)

- Spatial loadings (voxels × components)

- Optional per-voxel offsets

The backend maintains compatibility with the storage_backend contract
while providing specialized methods for latent data access.

Supports LatentNeuroVec objects from both the fmristore and fmrilatent
packages. fmristore is only required when source contains file paths
(.lv.h5). fmrilatent objects with lazy BasisHandle/LoadingsHandle slots
are materialized automatically on access.
