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
