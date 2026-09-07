# Persistence, identity, and import

A frame that only lives in one R session is of limited use. This
vignette covers the three ways a frame meets the outside world: its
backend-neutral manifest, the HDF5 round trip that stores it, and the
BIDS on-ramp that creates one from an fMRIPrep derivative tree. Running
through all three is the question of identity: what it means for two
frames, two spaces, or two arrays to be *the same*.

``` r

library(fmridataset)
```

The running example is a small volume-backed frame with two runs.

``` r

set.seed(3)
voxels <- volume_space(dim = c(3, 3, 2), affine = diag(4), template = "toy")
n_obs <- 6L
signal <- matrix(round(rnorm(n_obs * 18), 2), n_obs, 18)
observations <- data.frame(
  .obs_id = sprintf("vol-%02d", seq_len(n_obs)),
  run_id = rep(c("run-1", "run-2"), each = 3),
  TR = 2
)
frame <- fmri_frame(
  assays = list(bold = signal),
  observations = observations,
  space = voxels,
  metadata = list(study = "toy")
)
```

## The FDS manifest

[`fds_frame_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_frame_manifest.md)
produces the FDS version 1 manifest: everything a frame means, and
nothing about where its numbers are stored. It holds the axis IDs and
tables, the feature space, the assay declarations with their shapes and
dtypes, entities, relations, typed tables, metadata, and provenance. It
performs no I/O and reads no assay values.

``` r

manifest <- fds_frame_manifest(frame)
names(manifest)
#>  [1] "schema"       "object_type"  "shape"        "axes"         "arrays"      
#>  [6] "assays"       "entities"     "relations"    "tables"       "active_assay"
#> [11] "metadata"     "provenance"   "extensions"
manifest$schema$id
#> [1] "org.fmridataset.fds/v1"
manifest$assays$bold[c("name", "array", "dtype", "shape")]
#> $name
#> [1] "bold"
#> 
#> $array
#> [1] "assays/bold"
#> 
#> $dtype
#> [1] "float64"
#> 
#> $shape
#> [1]  6 18
```

[`fds_manifest_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_manifest_digest.md)
canonicalizes the manifest and hashes it. This is the frame’s *semantic*
identity. Because physical sources are not in the manifest, the same
frame has the same digest however it is stored.

``` r

semantic <- fds_manifest_digest(manifest)
substr(semantic, 1, 16)
#> [1] "a5c65f49523f5752"

chunked <- fmri_frame(
  assays = list(bold = memory_source(signal, chunks = c(2L, 9L))),
  observations = observations,
  space = voxels,
  metadata = list(study = "toy")
)
identical(fds_manifest_digest(fds_frame_manifest(chunked)), semantic)
#> [1] TRUE
```

The digest is also unchanged when the *values* change, because values
are not semantics. Rename one observation, and it changes.

``` r

doubled <- fmri_frame(
  assays = list(bold = signal * 2),
  observations = observations,
  space = voxels,
  metadata = list(study = "toy")
)
identical(fds_manifest_digest(fds_frame_manifest(doubled)), semantic)
#> [1] TRUE

renamed <- fmri_frame(
  assays = list(bold = signal),
  observations = transform(observations, .obs_id = sub("vol", "v", .obs_id)),
  space = voxels,
  metadata = list(study = "toy")
)
identical(fds_manifest_digest(fds_frame_manifest(renamed)), semantic)
#> [1] FALSE
```

Frames with ephemeral IDs have no manifest at all:
[`fds_frame_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_frame_manifest.md)
refuses them, so nothing random can reach disk.

## Identity domains

Identity is typed.
[`identity_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/identity_descriptor.md)
returns a digest together with the domain it belongs to and the
canonicalization contract that produced it, so two hexadecimal strings
from different domains are never confused.

| Domain | Answers | Operation |
|----|----|----|
| semantic | Is this the same frame, ignoring storage? | [`fds_manifest_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_manifest_digest.md) |
| schema | Do these frames have the same column and space contracts? | [`frame_schema_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-schema-validation.md) |
| space | Is this exactly the same feature space? | [`space_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md), [`same_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md) |
| source | Is this the same descriptor of the same revision of the same physical thing? | [`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) |
| provenance | Is this the same derivation history? | [`provenance_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md) |
| content | Do these arrays hold the same values? | [`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md), passed in explicitly |

The domain is inferred from the object when it can be.

``` r

identity_descriptor(frame)$domain
#> [1] "semantic"
identity_descriptor(space(frame))$domain
#> [1] "space"
identity_descriptor(assay(frame)$source)$domain
#> [1] "source"
identity_descriptor(frame)$canonicalization$id
#> [1] "org.fmridataset.r-canonical/v1"
```

[`explain()`](https://bbuchsbaum.github.io/fmridataset/reference/explain.md)
reports the schema, semantic, and source identities of a frame side by
side, and none of them costs a read.

## Fingerprints are not content hashes

A source has two identities, computed separately, and the package never
substitutes one for the other.

[`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
is a cheap *revision* fingerprint of the descriptor: its type, shape,
dtype, chunk grid, selectors, and whatever physical evidence the backend
can observe without reading values, such as a file’s size and
modification time. It is computed once at construction and cached. For
an in-memory source there is no file to observe, so the fingerprint
includes a per-object identity token. Two memory sources built from the
same matrix are two objects, and they have two fingerprints.

``` r

a <- memory_source(signal)
b <- memory_source(signal)
identical(source_fingerprint(a), source_fingerprint(b))
#> [1] FALSE
```

[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
is the one operation that identifies an array by its values. It streams
the source in bounded blocks and is proportional to the number of
values, so nothing calls it implicitly. It agrees across copies, storage
dtypes, chunk grids, and compositions of the same values.

``` r

identical(content_hash(a), content_hash(b))
#> [1] TRUE

as_float32 <- memory_source(signal, dtype = "float32", chunks = c(1L, 18L))
identical(content_hash(a), content_hash(as_float32))
#> [1] TRUE

stacked <- row_bound_source(list(memory_source(signal[1:2, ]), memory_source(signal[3:6, ])))
identical(content_hash(a), content_hash(stacked))
#> [1] TRUE
```

A fingerprint says *same descriptor, same revision*. A content hash says
*same numbers*. Neither implies the other. If you want equal-valued
memory sources to share a fingerprint, opt in with
`identity = "content"` and pay the hashing cost once, visibly, at
construction. A `revision` label replaces the per-object token, so two
memory sources built independently under the same revision share a
fingerprint without any hashing: the caller is asserting that they are
the same source in the same revision.

``` r

identical(
  source_fingerprint(memory_source(signal, identity = "content")),
  source_fingerprint(memory_source(signal, identity = "content"))
)
#> [1] TRUE
```

A content hash becomes a typed identity by being handed to
[`identity_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/identity_descriptor.md)
explicitly. That is the only way the content domain is ever populated.

``` r

receipt <- content_hash(assay(frame)$source)
identity_descriptor(assay(frame)$source, domain = "content", content_digest = receipt)$digest ==
  receipt
#> [1] TRUE
```

File-backed sources use their fingerprint evidence to notice change:
every open and read re-observes the file and raises
`fmridataset_error_source_stale` when it no longer matches, distinct
from the `fmridataset_error_backend_io` a genuine I/O failure raises.
The policy is recorded in
`inst/architecture/ADR-009-source-fingerprints-and-content-hashes.md`.

## HDF5 round trip

[`write_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/write_frame.md)
persists a frame as an HDF5 file through the `fmristore` package, which
owns the certified layout and the atomic commit.
[`open_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/write_frame.md)
reads it back. Reopened assays are lazy HDF5 sources; opening a frame
reads its manifest, not its values. Neither function computes a content
hash.

``` r

path <- file.path(tempdir(), "toy-frame.h5")
committed <- write_frame(frame, path)
reopened <- open_frame(committed)
reopened
#> <fmri_frame> 6 observations x 18 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space 045f2e10b586
class(assay(reopened)$source)[1]
#> [1] "h5_array_source"
```

The reopened frame has the same semantic identity, the same space, the
same IDs, the same run structure, and, when you ask for it, the same
values.

``` r

identical(fds_manifest_digest(fds_frame_manifest(reopened)), semantic)
#> [1] TRUE
same_space(space(reopened), space(frame))$same
#> [1] TRUE
identical(observation_ids(reopened), observation_ids(frame))
#> [1] TRUE
temporal_schema(reopened)$run_lengths
#> run-1 run-2 
#>     3     3
identical(content_hash(assay(reopened)$source), receipt)
#> [1] TRUE
```

What it does not share is a source fingerprint, because an HDF5 dataset
and an in-memory matrix are different physical things.

``` r

identical(
  source_fingerprint(assay(reopened)$source),
  source_fingerprint(assay(frame)$source)
)
#> [1] FALSE
```

Views over the reopened frame behave exactly as they do in memory.

``` r

collect_assay(reopened[c("vol-06", "vol-01"), c("voxel-2", "voxel-1")])
#>       [,1]  [,2]
#> [1,] -1.13  0.03
#> [2,]  0.09 -0.96
```

## Import from BIDS

[`read_bids_bold()`](https://bbuchsbaum.github.io/fmridataset/reference/read_bids_bold.md)
opens one subject’s preprocessed BOLD runs from an fMRIPrep derivative
tree as a single lazy frame. It uses the `bidser` package for discovery.
Construction reads the BOLD headers and the run masks, resolves one
common volume space, and mints deterministic observation IDs from the
scan path and volume index. It does not read BOLD values.

``` r

bold <- read_bids_bold(
  "/data/my-study",
  subject = "01",
  task = "memory",
  space = "MNI152NLin2009cAsym"
)
```

To show the result without a real dataset, the chunk below writes a
minimal fMRIPrep-style tree with two three-volume runs into a temporary
directory. The helper is the same fixture the package’s tests use.

``` r

make_fixture <- function(root = tempfile("bids-")) {
  func_dir <- file.path(root, "derivatives", "fmriprep", "sub-01", "func")
  dir.create(func_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root, "sub-01", "func"), recursive = TRUE, showWarnings = FALSE)
  writeLines('{"Name":"toy","BIDSVersion":"1.10.0"}', file.path(root, "dataset_description.json"))
  writeLines(
    '{"Name":"fMRIPrep","BIDSVersion":"1.10.0","DatasetType":"derivative","GeneratedBy":[{"Name":"fMRIPrep"}]}',
    file.path(root, "derivatives", "fmriprep", "dataset_description.json")
  )
  write.table(
    data.frame(participant_id = "sub-01"), file.path(root, "participants.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE
  )
  grid <- c(2L, 2L, 2L)
  masks <- list(
    array(c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE), grid),
    array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE), grid)
  )
  for (run in 1:2) {
    stem <- sprintf("sub-01_task-memory_run-%02d_space-MNI152NLin6Asym", run)
    bold <- array(seq_len(prod(grid) * 3L) + 100L * (run - 1L), c(grid, 3L))
    neuroim2::write_vec(
      neuroim2::NeuroVec(bold, neuroim2::NeuroSpace(c(grid, 3L))),
      file.path(func_dir, paste0(stem, "_desc-preproc_bold.nii"))
    )
    writeLines('{"RepetitionTime":2}', file.path(func_dir, paste0(stem, "_desc-preproc_bold.json")))
    neuroim2::write_vol(
      neuroim2::LogicalNeuroVol(masks[[run]], neuroim2::NeuroSpace(grid)),
      file.path(func_dir, paste0(stem, "_desc-brain_mask.nii"))
    )
    write.table(
      data.frame(onset = 2 * (run - 1), duration = 1, trial_type = c("old", "new")[run]),
      file.path(root, "sub-01", "func", sprintf("sub-01_task-memory_run-%02d_events.tsv", run)),
      sep = "\t", row.names = FALSE, quote = FALSE
    )
  }
  root
}
root <- make_fixture()
```

``` r

bold <- read_bids_bold(root, subject = "01", task = "memory", space = "MNI152NLin6Asym")
bold
#> <fmri_frame> 6 observations x 3 features
#>   assays: signal 
#>   active: signal 
#>   space: volume_space 197a0e8bc3da
```

The two run masks differ, and by default the frame uses their
intersection: three voxels. No resampling and no alignment happen
implicitly; ambiguous spaces, masks, or multi-echo selections are errors
that ask for a choice.

``` r

feature_ids(bold)
#> [1] "voxel-2" "voxel-3" "voxel-4"
```

The observation table carries the BIDS entities, the within-run volume
index, the run time, and the `TR`, so the temporal contract holds
without further work. A `run` entity keyed on `scan_id` and a `subject`
entity are attached, and `scan_id` is what the temporal schema uses as
the run column, because a BIDS `run` label is only unique within a
session.

``` r

observations(bold)[, c("run_id", "volume_index", "run_time", "TR")]
#> # A tibble: 6 × 4
#>   run_id volume_index run_time    TR
#>   <chr>         <int>    <dbl> <dbl>
#> 1 run-1             1        0     2
#> 2 run-1             2        2     2
#> 3 run-1             3        4     2
#> 4 run-2             1        0     2
#> 5 run-2             2        2     2
#> 6 run-2             3        4     2
entities(bold)
#> <entity_registry> 2 entity types
#>   subject, run
temporal_schema(bold)$columns$run
#> [1] "scan_id"
unname(temporal_schema(bold)$run_lengths)
#> [1] 3 3
```

Observation IDs are durable and deterministic: the same tree imported
twice yields the same IDs.

``` r

ids_are_durable(bold)
#> [1] TRUE
basename(observation_ids(bold)[1:2])
#> [1] "sub-01_task-memory_run-01_space-MNI152NLin6Asym_desc-preproc_bold::volume-000000"
#> [2] "sub-01_task-memory_run-01_space-MNI152NLin6Asym_desc-preproc_bold::volume-000001"
```

Reading is lazy and goes through the NIfTI source. A run filter reads
only that run’s volumes, and because the frame keeps the complete
feature domain of its volume space, spatial maps can take the native
read path.

``` r

collect_assay(filter_obs(bold, run_id == "run-2"))
#>      [,1] [,2] [,3]
#> [1,]  102  103  104
#> [2,]  110  111  112
#> [3,]  118  119  120
execution_path(bold, "spatial")
#> [1] "native"
```

Matching `events.tsv` files are attached as a keyed event table.

``` r

table_data(bold$tables$events)[, c("onset", "duration", "trial_type")]
#> # A tibble: 2 × 3
#>   onset duration trial_type
#>   <dbl>    <dbl> <chr>     
#> 1     0        1 old       
#> 2     2        1 new
```

## Choosing the right identity

Use the semantic digest to ask whether two frames mean the same thing,
and the schema digest to ask whether they could be bound or collected
together. Use
[`same_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
before any operation that assumes columns line up. Key caches and plans
on the source fingerprint. Reach for
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
only when the question is about the numbers, accept its cost knowingly,
and record the receipt where you need it, because the package will not
do it for you.
