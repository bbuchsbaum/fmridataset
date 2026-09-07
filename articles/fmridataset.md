# Frames: aligned fMRI data with explicit identity

An fMRI analysis moves the same numbers through many hands: a
preprocessing pipeline, a mask, a model, a parcellation, a file on disk.
At each step the rows and columns of the array still mean something, and
that meaning is easy to lose. `fmridataset` keeps it. Its one data
container, the `fmri_frame`, holds an observation-by-feature array
together with what each observation and each feature *is*, and refuses
operations that would silently break that link.

This vignette builds a frame from a small synthetic matrix and walks
through the everyday operations: selecting by ID and by metadata,
reading values lazily, inspecting a frame without reading it, describing
its run structure, and binding frames together. It ends with the short
list of promises every frame keeps.

``` r

library(fmridataset)
```

## Build a frame

A frame needs three things: at least one assay (the numbers),
observation metadata with a stable ID per row, and a feature space that
says what the columns are. Here the columns are the voxels of a tiny 3
by 3 by 2 volume, and the rows are eight volumes acquired in two runs.

``` r

set.seed(1)
voxels <- volume_space(dim = c(3, 3, 2), affine = diag(4), template = "toy")
n_obs <- 8L
signal <- matrix(
  round(rnorm(n_obs * n_features(voxels)), 2),
  nrow = n_obs, ncol = n_features(voxels)
)

frame <- fmri_frame(
  assays = list(bold = signal),
  observations = data.frame(
    .obs_id = sprintf("vol-%02d", seq_len(n_obs)),
    run_id = rep(c("run-1", "run-2"), each = 4),
    TR = 2,
    motion = round(runif(n_obs), 2)
  ),
  space = voxels
)
frame
#> <fmri_frame> 8 observations x 18 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space 045f2e10b586
```

The `.obs_id` column supplies the observation IDs. The volume space
supplies the feature IDs, one per voxel in its support, and derives a
feature table from them.

``` r

observations(frame)
#> # A tibble: 8 × 4
#>   .obs_id run_id    TR motion
#>   <chr>   <chr>  <dbl>  <dbl>
#> 1 vol-01  run-1      2   0.13
#> 2 vol-02  run-1      2   0.22
#> 3 vol-03  run-1      2   0.23
#> 4 vol-04  run-1      2   0.13
#> 5 vol-05  run-2      2   0.98
#> 6 vol-06  run-2      2   0.33
#> 7 vol-07  run-2      2   0.51
#> 8 vol-08  run-2      2   0.68
head(features(frame), 3)
#> # A tibble: 3 × 5
#>   .feature_id .linear_index     i     j     k
#>   <chr>               <int> <int> <int> <int>
#> 1 voxel-1                 1     1     1     1
#> 2 voxel-2                 2     2     1     1
#> 3 voxel-3                 3     3     1     1
```

Nothing about the matrix itself was trusted for identity. If `signal`
had carried row names that disagreed with `.obs_id`, construction would
have failed rather than picking one.

## Select by ID or by metadata

Square-bracket selection takes IDs on either axis and returns a *view*:
a lightweight object that remembers which rows and columns of the parent
frame it stands for. The requested order is kept on every axis.

``` r

view <- frame[c("vol-08", "vol-01"), c("voxel-5", "voxel-2")]
view
#> <fmri_view> 2 observations x 2 features
#>   base: 8 x 18 
#>   assays: bold
observation_ids(view)
#> [1] "vol-08" "vol-01"
feature_ids(view)
#> [1] "voxel-5" "voxel-2"
collect_assay(view)
#>      [,1]  [,2]
#> [1,] 0.76 -0.04
#> [2,] 0.39  0.58
```

Metadata predicates do the same job when you know a property rather than
an ID.
[`filter_obs()`](https://bbuchsbaum.github.io/fmridataset/reference/filter_obs.md)
evaluates an expression against the observation table and
[`select_features()`](https://bbuchsbaum.github.io/fmridataset/reference/select_features.md)
against the feature table.

``` r

run_2 <- filter_obs(frame, run_id == "run-2")
observation_ids(run_2)
#> [1] "vol-05" "vol-06" "vol-07" "vol-08"

still <- filter_obs(frame, motion < 0.5)
observation_ids(still)
#> [1] "vol-01" "vol-02" "vol-03" "vol-04" "vol-06"

top_slice <- select_features(frame, k == 2)
feature_ids(top_slice)
#> [1] "voxel-10" "voxel-11" "voxel-12" "voxel-13" "voxel-14" "voxel-15" "voxel-16"
#> [8] "voxel-17" "voxel-18"
```

Views compose. Selecting from a view produces another view over the same
base frame, and the feature space follows along: the space of
`top_slice` is the same volume restricted to the nine voxels of its
upper slice.

``` r

corner <- top_slice[, c("voxel-18", "voxel-10")]
space(corner)$support
#> [1] 18 10
```

Selectors that could mean two things are rejected. Duplicated IDs,
unknown IDs, and mixed positive and negative positions all raise a
structured error instead of a guess.

``` r

frame[c("vol-01", "vol-01"), ]
#> Error:
#> ! observation ID selectors must be unique.
frame[, "voxel-99"]
#> Error:
#> ! Unknown feature ID in selector.
```

## Views are lazy

A view stores a selection, not values. To see that nothing is read until
you ask, wrap the matrix in a
[`counting_source()`](https://bbuchsbaum.github.io/fmridataset/reference/counting_source.md),
the package’s read instrumentation for conformance tests, and build a
frame over it.

``` r

counted <- counting_source(signal)
lazy <- fmri_frame(
  assays = list(bold = counted),
  observations = observations(frame),
  space = voxels
)

subset <- filter_obs(lazy, run_id == "run-1")[, c("voxel-1", "voxel-2")]
source_counts(counted)$reads
#> [1] 0

values <- collect_assay(subset)
dim(values)
#> [1] 4 2
source_counts(counted)[c("reads", "values")]
#> $reads
#> [1] 1
#> 
#> $values
#> [1] 8
```

Filtering and subsetting issued no reads.
[`collect_assay()`](https://bbuchsbaum.github.io/fmridataset/reference/collect_assay.md)
issued one, and it read exactly the eight values the view stands for,
not the full matrix. The same holds for file-backed sources, where the
difference is a small selection against a large volume on disk.

[`collect_assay()`](https://bbuchsbaum.github.io/fmridataset/reference/collect_assay.md)
also checks an estimated memory cost against a budget before it reads
anything, so a request that would not fit fails early and cheaply.

``` r

collect_assay(frame, memory_budget = 10)
#> Error:
#> ! collect_assay() is estimated to retain 1152 output bytes and peak at 1152 bytes (float64 source values realized as R double), above memory_budget of 10 bytes.
```

## Inspect without reading

[`explain()`](https://bbuchsbaum.github.io/fmridataset/reference/explain.md)
returns a bounded, serializable summary of a frame or view. It reports
the shape, the identities of the axes and space, the sources behind each
assay, and what realizing them would cost. It never reads assay values.

``` r

report <- explain(frame)
report$shape
#> observation     feature 
#>           8          18
report$space
#> $type
#> [1] "volume_space"
#> 
#> $features
#> [1] 18
#> 
#> $digest
#> [1] "045f2e10b586d76f3d69bfabed58970ebfc0f85a7b988a2e7e8db04e9b714d99"
report$assays$bold[c("source_type", "dtype", "chunks", "realization_bytes")]
#> $source_type
#> [1] "memory_source"
#> 
#> $dtype
#> [1] "float64"
#> 
#> $chunks
#> [1]  8 18
#> 
#> $realization_bytes
#> [1] 1152
```

The digests are typed. The schema digest covers the column and space
contracts, the semantic digest covers the whole source-free manifest,
and the assay fingerprint identifies the physical source. None of them
hashes the numbers;
[`vignette("persistence-and-import")`](https://bbuchsbaum.github.io/fmridataset/articles/persistence-and-import.md)
explains when you want one that does.

``` r

substr(unlist(report$digests), 1, 12)
#>         schema       semantic    observation        feature 
#> "2d4b2168731a" "7869b1292a8f" "af8ede984ac5" "2af08b81f2f2"
```

## Run structure is metadata

A frame carries no acquisition timing of its own. When its observations
are volumes acquired in runs, that is a fact recorded in the observation
table, under a small validated contract: a run column, an optional `TR`
in seconds that is constant within each run, and an optional logical
`censor` column.
[`temporal_schema()`](https://bbuchsbaum.github.io/fmridataset/reference/temporal-schema.md)
derives the description from those columns each time it is asked, so it
can never go stale.

``` r

has_temporal_schema(frame)
#> [1] TRUE
schema <- temporal_schema(frame)
schema
#> <frame_temporal_schema> 8 observations in 2 runs
#>   run-1               4 observations  TR 2 s
#>   run-2               4 observations  TR 2 s
schema$run_lengths
#> run-1 run-2 
#>     4     4
```

The run column is discovered rather than assumed: a relation to a run
entity wins, then `scan_id`, then `run_id`, and an explicit `run_col`
overrides all three. When the `fmrihrf` package is installed,
[`as_sampling_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/temporal-schema.md)
builds the run-length encoded `sampling_frame` that design and modelling
code consumes.

``` r

as_sampling_frame(frame)
#> Sampling Frame
#> ==============
#> 
#> Structure:
#>   2 blocks
#>   Total scans: 8
#> 
#> Timing:
#>   TR: 2 s
#>   Precision: 0.1 s
#> 
#> Duration:
#>   Total time: 16.0 s
```

Because the schema follows the observation order, a filtered view still
describes itself correctly.

``` r

temporal_schema(still)
#> <frame_temporal_schema> 5 observations in 2 runs
#>   run-1               4 observations  TR 2 s
#>   run-2               1 observations  TR 2 s
```

A view whose runs are interleaved is a legal frame, but a sampling frame
is a run-length encoding and cannot represent it. The conversion refuses
rather than reordering behind your back.

``` r

interleaved <- frame[c("vol-01", "vol-05", "vol-02"), ]
temporal_schema(interleaved)$contiguous
#> [1] FALSE
as_sampling_frame(interleaved)
#> Error:
#> ! A sampling frame is a run-length encoding and cannot describe a frame whose runs are interleaved or reordered. Restore acquisition order before converting, or work from temporal_schema() directly.
```

Frames without run structure, such as a matrix of beta estimates, are
first-class; for them
[`has_temporal_schema()`](https://bbuchsbaum.github.io/fmridataset/reference/temporal-schema.md)
is simply `FALSE`.

## Durable and ephemeral IDs

Every axis has a declared ID policy. The default, `require`, means the
caller supplied the IDs, as `.obs_id` did above. Both axes of `frame`
are durable.

``` r

ids_are_durable(frame)
#> [1] TRUE
axis_id_policy(observation_axis(frame))$policy
#> [1] "require"
```

Importers that mint IDs from declared keys use the `deterministic`
policy, which derives a SHA-256 ID from a namespace and the typed key
values. The same keys always produce the same IDs, so two independent
imports of the same data agree.

``` r

scans <- axis_frame(
  data.frame(subject = "01", run = rep(1:2, each = 4), volume = rep(1:4, 2)),
  axis = "observation",
  id_policy = "deterministic",
  id_namespace = "study:v1",
  id_keys = c("subject", "run", "volume")
)
substr(axis_ids(scans)[1:2], 1, 24)
#> [1] "obs-d5d0f69943ca66ed9f95" "obs-e846dd51a2559acd9cd7"
axis_id_policy(scans)[c("policy", "namespace", "keys")]
#> $policy
#> [1] "deterministic"
#> 
#> $namespace
#> [1] "study:v1"
#> 
#> $keys
#> [1] "subject" "run"     "volume"
```

For quick exploratory work you can ask for `ephemeral` IDs. They are
UUIDs with a visible `ephemeral-` prefix, and a frame that carries them
cannot be persisted or given a certified semantic identity until a
durable axis replaces them. Omitting `space` gives the feature axis the
same treatment.

``` r

scratch <- fmri_frame(
  assays = list(bold = signal),
  observations = axis_frame(data.frame(x = seq_len(n_obs)), id_policy = "ephemeral")
)
substr(observation_ids(scratch)[1], 1, 14)
#> [1] "ephemeral-obs-"
ids_are_durable(scratch)
#> [1] FALSE
fds_frame_manifest(scratch)
#> Error:
#> ! FDS persistence and semantic certification reject ephemeral axis IDs; the observation and feature axis carry ephemeral IDs. To fix: build the observation axis with supplied IDs or id_policy = "deterministic"; pass a feature space (space = ...) with supplied or deterministic IDs instead of letting fmri_frame() mint an ephemeral index_space.
```

## Bind frames along observations

[`bind_observations()`](https://bbuchsbaum.github.io/fmridataset/reference/bind_observations.md)
stacks frames that share a feature space. The result is lazy: assay
values stay in their original sources and are routed by row when read.
Observation order follows the operand order, IDs must not collide, and
the spaces must be the same space, not merely the same size.

``` r

run_1 <- filter_obs(frame, run_id == "run-1")
both <- bind_observations(run_2, run_1)
both
#> <fmri_frame> 8 observations x 18 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space 045f2e10b586
observation_ids(both)
#> [1] "vol-05" "vol-06" "vol-07" "vol-08" "vol-01" "vol-02" "vol-03" "vol-04"
temporal_schema(both)$run_lengths
#> run-2 run-1 
#>     4     4
```

Frame-level metadata is reconciled by an explicit policy. The default
demands equality; `"merge"` combines records recursively and fails only
on genuine conflict.

``` r

site_a <- fmri_frame(
  assays = list(bold = signal[1:4, ]),
  observations = observations(frame)[1:4, ],
  space = voxels,
  metadata = list(site = "A", scanner = list(vendor = "X"))
)
site_b <- fmri_frame(
  assays = list(bold = signal[5:8, ]),
  observations = observations(frame)[5:8, ],
  space = voxels,
  metadata = list(site = "A", scanner = list(field = "3T"))
)

bind_observations(site_a, site_b)
#> Error:
#> ! Bound frame metadata differ; use metadata_policy = 'merge' for a conflict-free record merge.

merged <- bind_observations(site_a, site_b, metadata_policy = "merge")
str(unclass(merged$metadata$scanner))
#> List of 2
#>  $ vendor: chr "X"
#>  $ field : chr "3T"
```

Every bind appends a provenance record naming its inputs, so a bound
frame can say where its rows came from.

``` r

merged$provenance
#> <provenance_graph> 1 records
#>   tips: 17ecefd5599efa57bd7605839621f489efab8716fe2eac16ddd985eac8e406d5
```

A frame over a different volume, even one with identical dimensions, is
refused.

``` r

other_volume <- volume_space(dim = c(3, 3, 2), affine = diag(4), template = "other")
elsewhere <- fmri_frame(
  assays = list(bold = signal[1:4, ]),
  observations = observations(frame)[1:4, ],
  space = other_volume
)
bind_observations(site_a, elsewhere)
#> Error:
#> ! Feature spaces differ in type, digest, or IDs.
```

## What a frame promises

These are the contracts from the package’s canonical data model
(`inst/architecture/ADR-001-canonical-data-model.md`), stated as the
behavior you can rely on.

- **Two axes, always.** An assay is observations by features. Factors
  and continuous variables are annotations on an axis, never a third
  dimension.
- **Every assay shares the axes.** All assays in a frame have the same
  observation IDs and the same feature IDs, in the same order.
- **IDs are stable and explicit.** Subsetting, reordering, and binding
  carry IDs through unchanged. Nothing regenerates them, and nothing
  aligns by position.
- **Spatial identity is semantic.** Two spaces are the same when their
  class, digest, and ordered feature IDs agree. Matching dimensions are
  not evidence.
- **Ambiguity fails early.** Duplicated selectors, conflicting dimnames,
  disagreeing metadata, and colliding IDs raise structured errors rather
  than being resolved by a rule you did not choose.
- **Execution is lazy and bounded.** Views read nothing, and a bind
  never reads assay values; it reads a feature or entity block backed by
  an array source only when the operands’ block fingerprints differ and
  the values must be compared. Realization goes through an explicit
  budget, and
  [`explain()`](https://bbuchsbaum.github.io/fmridataset/reference/explain.md)
  tells you the cost first.
- **Descriptors, not handles.** A frame contains serializable
  descriptors of its sources, never open files, environments, or
  closures, so it can be serialized and rebuilt elsewhere.

[`vignette("feature-spaces")`](https://bbuchsbaum.github.io/fmridataset/articles/feature-spaces.md)
covers the feature-space types and how to move a frame between them.
[`vignette("persistence-and-import")`](https://bbuchsbaum.github.io/fmridataset/articles/persistence-and-import.md)
covers the FDS manifest, HDF5 round trips, identity, and BIDS import.
