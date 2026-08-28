# fmridataset <img src="man/figures/logo.png" align="right" height="139" />

[Changelog](NEWS.md) ·
[Canonical data model](inst/architecture/ADR-001-canonical-data-model.md) ·
[Issues](https://github.com/bbuchsbaum/fmridataset/issues) ·
[Contributing](CONTRIBUTING.md)

`fmridataset` is an R package for keeping fMRI arrays aligned with what their
rows and columns mean. It represents each assay as observations by features and
carries stable IDs, annotations, entities, relations, and explicit spatial
identity through views, transformations, and storage round trips.

Use it when raw time series, beta estimates, parcel values, surface data, or
latent representations must retain their meaning as they move between analysis
steps and storage systems.

> **Status:** The `0.10.0` development line is an active migration toward 1.0
> and requires R 4.3 or newer. `fmri_frame()` is the canonical entry point;
> older dataset constructors remain available as transitional 0.x adapters.
> APIs may still change before 1.0.

## Installation

The current frame API is available from GitHub:

```r
install.packages("remotes")
remotes::install_github("bbuchsbaum/fmridataset")
```

`fmridataset` is not currently on CRAN. The published
[R-universe build](https://bbuchsbaum.r-universe.dev/fmridataset) is version
0.8.9 and documents the legacy dataset API.

## Quick start

Create a small volume-backed frame, then select observations and voxels by
their stable IDs:

```r
library(fmridataset)

signal <- matrix(seq_len(24), nrow = 6, ncol = 4)
voxel_space <- volume_space(
  dim = c(2, 2, 1),
  affine = diag(4),
  template = "toy"
)

frame <- fmri_frame(
  assays = list(signal = signal),
  observations = data.frame(
    .obs_id = paste0("volume-", seq_len(6)),
    run = rep(c("run-1", "run-2"), each = 3)
  ),
  space = voxel_space
)

view <- frame[c("volume-6", "volume-1"), c("voxel-4", "voxel-2")]

collect_assay(view)
#>      [,1] [,2]
#> [1,]   24   12
#> [2,]   19    7

observation_ids(view)
#> [1] "volume-6" "volume-1"

feature_ids(view)
#> [1] "voxel-4" "voxel-2"
```

The numerical view, observation metadata, feature metadata, and restricted
volume space all retain the requested order. For large assays,
`source_realization_cost()` reports storage bytes, realized output bytes, and a
conservative peak that includes selection, dtype-conversion, and
compressed-input buffers.
`collect_assay()`, `collect_chunks()`, spatial collection, and a finite
`as_delarr(memory_budget = ...)` ceiling enforce that peak estimate before
reading; block execution and lazy array sources avoid requiring full
materialization.

### Load one BIDS subject

With `bidser` 0.5.0 or newer, a subject's fMRIPrep BOLD runs can be opened as
one lazy frame:

```r
bold <- read_bids_bold(
  "/data/my-study",
  subject = "01",
  task = "memory",
  space = "MNI152NLin2009cAsym"
)

run_1 <- filter_obs(bold, run_id == "run-1")
map <- spatial_map(bold, observation = 1)
```

Construction reads BOLD headers and the matching run masks, but not BOLD
values. By default the frame uses the intersection of the run masks. No
resampling or cross-space alignment is performed implicitly; ambiguous spaces,
masks, or multi-echo selections produce an error requiring an explicit choice.
Events remain a keyed auxiliary table rather than being copied onto volumes.

## What it covers

- **Aligned assays:** keep one or more numerical assays tied to the same
  observation and feature axes, with explicit roles, units, and provenance.
- **Annotated domains:** attach scalar metadata, multivariate blocks,
  experimental entities, and validated relations without copying assay data.
- **Spatial identity:** represent volume, surface, parcel, basis, and composite
  feature spaces; compatibility is checked by identity rather than dimensions.
- **Explicit transformations:** map between feature spaces with validated,
  serializable operators and derivation provenance.
- **Bounded execution:** read lazy in-memory, NIfTI, sharded, HDF5-backed, and
  experimental Zarr sources through observation-by-feature selections.
- **Portable semantics:** serialize the logical frame contract with FDS v1 and
  bind it to physical storage without changing axis or spatial identity.

## Fit and boundaries

`fmridataset` owns semantic containers, alignment, views, sources, spaces, and
the logical FDS schema. Companion packages own adjacent responsibilities:

- [`neuroim2`](https://github.com/bbuchsbaum/neuroim2) provides native
  neuroimaging objects.
- [`delarr`](https://github.com/bbuchsbaum/delarr) provides lazy numerical
  plans and bounded execution.
- [`fmristore`](https://github.com/bbuchsbaum/fmristore) provides certified
  HDF5 layouts, atomic writes, append, and recovery.
- [`multidesign`](https://github.com/bbuchsbaum/multidesign) and
  [`fmrigds`](https://github.com/bbuchsbaum/fmrigds) own design compilation and
  statistical execution.
- [`bidser`](https://github.com/bbuchsbaum/bidser) provides BIDS discovery used
  by `read_bids_bold()` and the optional BIDS-to-HDF5 workflow.

HDF5 is the certified persistence direction for 1.0. Zarr support remains
experimental, and `DelayedArray` is optional interoperability rather than the
internal execution model. Legacy `fmri_dataset()`, `matrix_dataset()`, backend,
and sampling-frame workflows remain available during the 0.x migration.

## Documentation

- Run `help(package = "fmridataset")` and `?fmri_frame` for documentation that
  matches the installed package.
- Read the [canonical data model](inst/architecture/ADR-001-canonical-data-model.md)
  for ownership, compatibility, and migration decisions.
- Read the [FDS v1 decision](inst/architecture/ADR-002-fds-v1-logical-schema.md)
  for the backend-neutral persistence contract.
- See the [changelog](NEWS.md) for the current development surface.
- The [hosted package site](https://bbuchsbaum.github.io/fmridataset/) currently
  describes the published 0.8.9 release.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks. Changes
to public behavior should include behavioral tests, updated roxygen
documentation, and a `NEWS.md` entry.

## License

GPL (>= 3)
