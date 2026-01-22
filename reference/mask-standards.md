# Mask Representation Standards

This package enforces consistent mask representation across all
components:

## Details

### Backend Level (Internal)

- [`backend_get_mask()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_get_mask.md)
  always returns a **logical vector**

- Length equals the product of spatial dimensions

- TRUE indicates valid voxels, FALSE indicates excluded voxels

- No NA values allowed

- Must contain at least one TRUE value

### User Level (Public API)

- [`get_mask()`](https://bbuchsbaum.github.io/fmridataset/reference/get_mask.md)
  returns format appropriate to dataset type:

  - For volumetric datasets: 3D array or NeuroVol object

  - For matrix datasets: logical vector

  - For latent datasets: logical vector (components, not voxels)

### Conversion Helpers

- [`mask_to_logical()`](https://bbuchsbaum.github.io/fmridataset/reference/mask_to_logical.md):
  Convert any mask representation to logical vector

- [`mask_to_volume()`](https://bbuchsbaum.github.io/fmridataset/reference/mask_to_volume.md):
  Convert logical vector to 3D array given dimensions
