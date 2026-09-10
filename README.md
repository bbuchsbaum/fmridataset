# fmridataset

<!-- badges: start -->
[![R-CMD-check](https://github.com/bbuchsbaum/fmridataset/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bbuchsbaum/fmridataset/actions/workflows/R-CMD-check.yaml)
[![pkgdown](https://github.com/bbuchsbaum/fmridataset/actions/workflows/pkgdown.yaml/badge.svg)](https://bbuchsbaum.github.io/fmridataset/)
[![test-coverage](https://github.com/bbuchsbaum/fmridataset/actions/workflows/test-coverage.yaml/badge.svg)](https://github.com/bbuchsbaum/fmridataset/actions/workflows/test-coverage.yaml)
[![Codecov test coverage](https://codecov.io/gh/bbuchsbaum/fmridataset/branch/main/graph/badge.svg)](https://app.codecov.io/gh/bbuchsbaum/fmridataset?branch=main)
<!-- badges: end -->

[Changelog](NEWS.md) ·
[Canonical data model](inst/architecture/ADR-001-canonical-data-model.md) ·
[API audiences](inst/architecture/API-AUDIENCES.md) ·
[Issues](https://github.com/bbuchsbaum/fmridataset/issues) ·
[Contributing](CONTRIBUTING.md)

`fmridataset` is an R package for keeping fMRI arrays aligned with what their
rows and columns mean. It represents each assay as observations by features and
carries stable IDs, annotations, entities, relations, and explicit spatial
identity through views, transformations, and storage round trips.

Use it when raw time series, beta estimates, parcel values, surface data, or
latent representations must retain their meaning as they move between analysis
steps and storage systems.

> **Status:** The `0.10.0` development line is the road to 1.0 and requires R
> 4.3 or newer. `fmri_frame()` is the only data container. The pre-frame 0.x
> dataset and backend API has been removed; the last commit carrying it is
> `3ae565e`. APIs may still change before 1.0.

## Installation

```r
install.packages("remotes")
remotes::install_github("bbuchsbaum/fmridataset")
```

`fmridataset` is not on CRAN. The published
[R-universe build](https://bbuchsbaum.r-universe.dev/fmridataset) is version
0.8.9 and documents the old dataset API, not this one.

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
    run_id = rep(c("run-1", "run-2"), each = 3),
    TR = 2
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
volume space all retain the requested order. Nothing is read until
`collect_assay()` asks for values, and `explain(frame)` describes the frame,
its sources, and the cost of realizing it without reading any.

Axis IDs are required. Importers derive reproducible IDs from declared keys
with `axis_frame(id_policy = "deterministic", ...)`. Exploratory session-only
IDs require `id_policy = "ephemeral"`; they are visibly marked and must be
replaced before FDS persistence or semantic certification.

### Runs and timing

Run structure, TR, and censoring are observation metadata under a validated
schema rather than separate state:

```r
has_temporal_schema(frame)
#> [1] TRUE

schema <- temporal_schema(frame)
schema$run_lengths
#> run-1 run-2
#>     3     3

as_sampling_frame(frame)   # an fmrihrf::sampling_frame, when runs are contiguous
```

The run column is discovered from a run-typed relation, `scan_id`, or
`run_id`, or named explicitly with `run_col`. Frames without acquisition
structure, such as beta estimates, simply report `has_temporal_schema()` as
`FALSE`.

### Load one BIDS subject

With `bidser` 0.5.0 or newer, a subject's fMRIPrep BOLD runs open as one lazy
frame:

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

### Persist and reopen

```r
path <- write_frame(frame, "frame.h5")   # atomic, manifest-backed HDF5 via fmristore
again <- open_frame(path)                # assays reopen as lazy sources
```

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
  experimental Zarr sources through observation-by-feature selections, with
  realization budgets enforced before any read.
- **Collections and studies:** group native-space frames that cannot share a
  feature axis, and link heterogeneous frames through shared entities,
  relations, and typed maps.
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
  by `read_bids_bold()`.

HDF5 is the certified persistence direction for 1.0. Zarr support remains
experimental.

## Documentation

- Run `help(package = "fmridataset")` and `?fmri_frame` for documentation that
  matches the installed package.
- Read the [canonical data model](inst/architecture/ADR-001-canonical-data-model.md)
  for ownership and compatibility decisions, and the
  [API audiences](inst/architecture/API-AUDIENCES.md) for which exports are
  user, extension, or developer surface.
- Read the [FDS v1 decision](inst/architecture/ADR-002-fds-v1-logical-schema.md)
  for the backend-neutral persistence contract.
- See the [changelog](NEWS.md) for the current development surface.
- The [hosted package site](https://bbuchsbaum.github.io/fmridataset/)
  currently describes the published 0.8.9 release.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks. Changes
to public behavior should include behavioral tests, updated roxygen
documentation, and a `NEWS.md` entry.

## License

GPL (>= 3)
