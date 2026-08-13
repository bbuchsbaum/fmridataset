# Compress a BIDS Study into a Single HDF5 Archive

Converts a BIDS directory (or
[`bidser::bids_project`](https://bbuchsbaum.github.io/bidser/reference/bids_project.html))
into a single compressed HDF5 file containing compressed fMRI data,
events, confounds, and study metadata. The output file can be opened
with
[`bids_h5_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/bids_h5_dataset.md).

## Usage

``` r
compress_bids_study(
  x,
  file,
  mode = c("parcellated", "latent"),
  clusters = NULL,
  summary_fun = mean,
  encoding = NULL,
  n_components = NULL,
  template = NULL,
  mask = NULL,
  space = "MNI152NLin2009cAsym",
  tasks = NULL,
  subjects = NULL,
  sessions = NULL,
  confounds = NULL,
  compression = 4L,
  verbose = TRUE
)
```

## Arguments

- x:

  A
  [`bidser::bids_project`](https://bbuchsbaum.github.io/bidser/reference/bids_project.html)
  object **or** a character path to a BIDS directory (automatically
  opened with
  [`bidser::bids_project()`](https://bbuchsbaum.github.io/bidser/reference/bids_project.html)).

- file:

  Character. Path for the output `.h5` file. Parent directory must
  exist. Existing files are overwritten.

- mode:

  Character. Compression strategy: `"parcellated"` (default) or
  `"latent"`.

- clusters:

  A
  [`neuroim2::ClusteredNeuroVol`](https://bbuchsbaum.github.io/neuroim2/reference/ClusteredNeuroVol-class.html)
  defining the parcellation atlas in study space. Required for
  `mode = "parcellated"`; ignored for `"latent"`.

- summary_fun:

  Function applied to voxel time-series within each parcel to produce a
  scalar summary (default: `mean`). Only used for
  `mode = "parcellated"`.

- encoding:

  A `fmrilatent` encoding specification object (e.g.
  `fmrilatent::spec_time_dct(k = 15)`). Required for `mode = "latent"`
  unless `n_components` is provided.

- n_components:

  Integer. Shorthand for latent PCA with K components. If `encoding` is
  `NULL` and `n_components` is provided,
  `fmrilatent::spec_space_pca(k = n_components)` is used. Only used for
  `mode = "latent"`.

- template:

  Optional `fmrilatent` template object (e.g. from
  [`fmrilatent::parcel_basis_template()`](https://rdrr.io/pkg/fmrilatent/man/parcel_basis_template.html)
  or
  [`fmrilatent::build_hierarchical_template()`](https://rdrr.io/pkg/fmrilatent/man/build_hierarchical_template.html)).
  When provided, the template's spatial loadings are stored once in
  `/latent_meta/template/` and per-scan data is reduced to `[T, K]`
  projection coefficients (no per-scan loadings). This significantly
  reduces file size for multi-subject studies. Only used for
  `mode = "latent"`.

- mask:

  A
  [`neuroim2::LogicalNeuroVol`](https://bbuchsbaum.github.io/neuroim2/reference/LogicalNeuroVol-class.html)
  brain mask. For `mode = "parcellated"`, derived from `clusters` when
  `NULL`. For `mode = "latent"`, `mask` is required (cannot be derived
  without clusters).

- space:

  Character. Template space name stored as metadata (default:
  `"MNI152NLin2009cAsym"`).

- tasks:

  Character vector. Task filter; `NULL` means all tasks.

- subjects:

  Character vector. Subject filter; `NULL` means all subjects.

- sessions:

  Character vector. Session filter; `NULL` means all sessions (including
  session-less datasets).

- confounds:

  A confound specification passed to
  [`bidser::read_confounds()`](https://bbuchsbaum.github.io/bidser/reference/read_confounds.html),
  e.g. a character vector of column names, a
  [`bidser::confound_set()`](https://bbuchsbaum.github.io/bidser/reference/confound_set.html),
  or `NULL` to skip confound writing.

- compression:

  Integer 0–9. HDF5 gzip compression level (default 4).

- verbose:

  Logical. If `TRUE` (default) print progress messages.

## Value

A `bids_h5_dataset` object (reader for the newly created file). If the
reader is not yet available the file path is returned invisibly.

## Details

The writer streams scans one at a time — only one NIfTI image is held in
memory at a time. For each scan it:

1.  Reads the NIfTI via
    [`neuroim2::read_vec()`](https://bbuchsbaum.github.io/neuroim2/reference/read_vec.html).

2.  For **parcellated** mode: computes parcel averages via
    [`fmristore::summarize_by_clusters()`](https://bbuchsbaum.github.io/fmristore/reference/summarize_by_clusters.html)
    and writes `[T, K]` to `/scans/<name>/data/summary_data`.

3.  For **latent** mode: encodes via
    [`fmrilatent::encode()`](https://rdrr.io/pkg/fmrilatent/man/encode.html)
    and writes basis `[T, K]`, loadings `[V, K]`, and (optionally)
    offset `[V]` to `/scans/<name>/data/`.

4.  Writes events, confounds, censor, and metadata sub-groups.

5.  Releases the NIfTI from memory.

After all scans are written the `/scan_index/` lookup table is populated
and the function returns a `bids_h5_dataset` reader object.

## HDF5 schema

See `bids_plan.md` in the package source for the full v1.0 schema. The
root `compression_mode` attribute reflects the chosen `mode`.

## Examples

``` r
if (FALSE) { # \dontrun{
library(bidser)
library(neuroim2)
library(fmristore)

bids_dir  <- system.file("extdata", "ds001", package = "bidser")
atlas     <- fmristore::get_schaefer_atlas(100)   # example atlas

# Parcellated mode
study <- compress_bids_study(
  x          = bids_dir,
  file       = tempfile(fileext = ".h5"),
  clusters   = atlas,
  tasks      = "nback",
  verbose    = TRUE
)

# Latent mode (PCA with 50 components)
study_lat <- compress_bids_study(
  x            = bids_dir,
  file         = tempfile(fileext = ".h5"),
  mode         = "latent",
  n_components = 50L,
  mask         = brain_mask,
  tasks        = "nback",
  verbose      = TRUE
)
} # }
```
