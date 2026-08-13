# BIDS H5 Archive: Compressing a Study into a Single File

An fMRI study lives as thousands of files: NIfTI images, events.tsv
files, confound regressors, JSON sidecars. The BIDS H5 archive captures
an entire study in **one compressed HDF5 file** that is queryable by
subject, task, session, and run. You download one file and start
analyzing immediately.

``` r

library(fmridataset)
```

## Which compression mode should you use?

The archive supports two modes. Pick the one that matches your analysis:

| Mode | What it stores | Reconstruction | Best for |
|:---|:---|:---|:---|
| `parcellated` | Cluster averages `[T, K]` | Parcel-level only | ROI analyses, connectivity |
| `latent` | Basis `[T, K]` + loadings `[V, K]` | Full voxel resolution | Searchlight, fine-grained spatial |

Both modes store events, confounds, censor vectors, and BIDS metadata
alongside the data. The standard `fmridataset` API works on either.

## How do you create an archive?

[`compress_bids_study()`](https://bbuchsbaum.github.io/fmridataset/reference/compress_bids_study.md)
reads a BIDS directory and streams scans one at a time into the archive.
Only one NIfTI is held in memory at once.

### Parcellated mode

``` r

library(bidser)

bids <- bids_project("/path/to/my_study")
atlas <- neuroim2::read_vol("schaefer_400.nii.gz")

study <- compress_bids_study(
  bids,
  file = "my_study.h5",
  mode = "parcellated",
  clusters = atlas
)
```

The result is a `bids_h5_study_dataset` you can use immediately.

### Latent mode

Latent mode uses
[`fmrilatent::encode()`](https://rdrr.io/pkg/fmrilatent/man/encode.html)
to compress each scan into a low-rank basis + loadings representation
that can be reconstructed back to voxel resolution.

``` r

library(fmrilatent)

study <- compress_bids_study(
  bids,
  file = "my_study_latent.h5",
  mode = "latent",
  encoding = spec_time_dct(k = 30),
  mask = brain_mask
)
```

For multi-subject studies that share a spatial atlas, a **shared
template** stores the loadings once and only keeps per-scan
coefficients, significantly reducing file size:

``` r

tpl <- parcel_basis_template(parcellation, basis_spec = basis_slepian(k = 10))

study <- compress_bids_study(
  bids,
  file = "my_study_template.h5",
  mode = "latent",
  template = tpl,
  mask = brain_mask
)
```

## How do you open an archive?

One function, one file:

``` r

study <- bids_h5_dataset("my_study.h5")
study
```

    #> <bids_h5_study_dataset>
    #>   File: my_study.h5
    #>   Mode: parcellated (400 parcels)
    #>   Subjects: 20 | Tasks: nback, rest | Sessions: pre, post
    #>   Scans: 80 | Total timepoints: 24000 | TR: 2s

## Exploring the study

BIDS metadata is directly accessible:

``` r

participants(study)
tasks(study)
sessions(study)
scan_manifest(study)
```

[`scan_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/scan_manifest.md)
returns a tibble with one row per scan:

    #>   scan_name                       subject task  session run n_time
    #>   sub-01_ses-pre_task-nback_run-01 01     nback pre     01  300
    #>   sub-01_ses-pre_task-rest_run-01  01     rest  pre     01  200
    #>   ...

## Subsetting by task, subject, or session

Use
[`subset_bids_h5()`](https://bbuchsbaum.github.io/fmridataset/reference/subset_bids_h5.md)
to carve out the data you need. This returns a new
`bids_h5_study_dataset` backed by the same file:

``` r

nback <- subset_bids_h5(study, task = "nback")
sub01 <- subset_bids_h5(study, subject = "01")
pre_nback <- subset_bids_h5(study, task = "nback", session = "pre")
```

Subsetting is cheap — it selects scan backends from the shared H5
connection, no data is copied.

## Accessing data

The standard `fmridataset` API works on the result:

``` r

mat <- get_data_matrix(nback)
dim(mat)
```

    #> [1] 6000  400

That is `[total_timepoints, K]` where K is the number of parcels (or
components in latent mode). Per-subject data:

``` r

mat01 <- get_data_matrix(nback, subject_id = "01")
dim(mat01)
```

    #> [1] 600 400

### Events

The event table combines all scans with `task`, `session`, `subject_id`,
and both the BIDS `run` label and an internal `run_id`:

``` r

head(nback$event_table)
```

    #>   onset duration trial_type run run_id subject_id task  session
    #>   0.0   2.0      face       01  1      01         nback pre
    #>   4.0   2.0      house      01  1      01         nback pre
    #>   ...

### Confounds

``` r

conf <- get_confounds(study, scan_name = "sub-01_ses-pre_task-nback_run-01")
head(conf)
```

### Chunked iteration

Memory-efficient processing via
[`data_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.md):

``` r

chunks <- data_chunks(nback, nchunks = 10)
while (!is.null(chunk <- iterators::nextElem(chunks))) {
  # Process chunk$data [T, K_chunk]
}
```

### Group operations

Convert to `fmri_group` for per-subject analyses:

``` r

group <- study_to_group(nback)
```

## Latent-mode extras

When working with a latent-mode archive, three additional accessors are
available:

``` r

info <- encoding_info(study)
info$encoding_family
info$n_components
info$has_shared_template

loadings <- get_loadings(study, scan_name = "sub-01_task-nback_run-01")
dim(loadings)

recon <- reconstruct_voxels(study,
  scan_name = "sub-01_task-nback_run-01",
  rows = 1:10,
  voxels = roi_indices
)
dim(recon)
```

[`reconstruct_voxels()`](https://bbuchsbaum.github.io/fmridataset/reference/reconstruct_voxels.md)
computes `basis %*% t(loadings) + offset` on the fly, so you only
materialize the slice you need.

## Parcellation metadata

For parcellated archives,
[`parcellation_info()`](https://bbuchsbaum.github.io/fmridataset/reference/parcellation_info.md)
gives you the cluster mapping:

``` r

pinfo <- parcellation_info(study)
pinfo$n_parcels
pinfo$cluster_ids
```

This returns `NULL` for latent-mode archives.

## Next steps

- [`vignette("fmridataset-intro")`](https://bbuchsbaum.github.io/fmridataset/articles/fmridataset-intro.md)
  for the core dataset API
- [`vignette("study-level-analysis")`](https://bbuchsbaum.github.io/fmridataset/articles/study-level-analysis.md)
  for multi-subject workflows without BIDS
- [`vignette("backend-development-basics")`](https://bbuchsbaum.github.io/fmridataset/articles/backend-development-basics.md)
  if you want to write a custom backend
- [`?compress_bids_study`](https://bbuchsbaum.github.io/fmridataset/reference/compress_bids_study.md),
  [`?bids_h5_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/bids_h5_dataset.md),
  [`?subset_bids_h5`](https://bbuchsbaum.github.io/fmridataset/reference/subset_bids_h5.md)
  for full parameter docs
