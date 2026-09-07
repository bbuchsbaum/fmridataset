# Feature spaces and maps

The columns of an fMRI array are voxels, or vertices, or parcels, or
latent components. A frame’s *feature space* says which. It names every
column with a stable ID and knows how to turn a row of values back into
a spatial object. This vignette builds the space types the package ships
with and shows how views restrict them. It explains what spatial
identity means, then moves a frame from one space to another through an
explicit, lazy map.

``` r

library(fmridataset)
library(Matrix)
```

## A masked volume

[`volume_space()`](https://bbuchsbaum.github.io/fmridataset/reference/volume_space.md)
describes a packed volume: a 3-D grid, a voxel-to-world affine, and a
*support*, the linear indices of the voxels that are in play. Feature
IDs are derived from the support, so they survive masking and reordering
unchanged.

``` r

grid <- c(4L, 4L, 2L)
mask <- array(FALSE, grid)
mask[2:3, 2:3, ] <- TRUE

brain <- volume_space(dim = grid, affine = diag(4), support = mask, template = "toy")
n_features(brain)
#> [1] 8
feature_ids(brain)
#> [1] "voxel-6"  "voxel-7"  "voxel-10" "voxel-11" "voxel-22" "voxel-23" "voxel-26"
#> [8] "voxel-27"
head(feature_data(brain), 3)
#> # A tibble: 3 × 5
#>   .feature_id .linear_index     i     j     k
#>   <chr>               <int> <int> <int> <int>
#> 1 voxel-6                 6     2     2     1
#> 2 voxel-7                 7     3     2     1
#> 3 voxel-10               10     2     3     1
```

The feature table gives each voxel its linear index and grid
coordinates. Predicates in
[`select_features()`](https://bbuchsbaum.github.io/fmridataset/reference/select_features.md)
can use any of these columns.

The frame used for the rest of the vignette has six observations over
this masked volume.

``` r

set.seed(2)
n_obs <- 6L
signal <- matrix(
  round(rnorm(n_obs * n_features(brain)), 2),
  nrow = n_obs, ncol = n_features(brain)
)
frame <- fmri_frame(
  assays = list(bold = signal),
  observations = data.frame(
    .obs_id = sprintf("vol-%02d", seq_len(n_obs)),
    run_id = "run-1",
    TR = 2
  ),
  space = brain
)
frame
#> <fmri_frame> 6 observations x 8 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space af6c4ea8564d
```

## Views restrict the space

A view’s feature space is the base space restricted to the selected
features, in the selected order. Restriction is an operation on the
space itself, not on a copy of its table. The restricted space still
knows its grid, its affine, and where its voxels sit.

``` r

pair <- frame[, c("voxel-11", "voxel-6")]
space(pair)$support
#> [1] 11  6
feature_ids(pair)
#> [1] "voxel-11" "voxel-6"
same_space(space(pair), restrict_space(brain, c(4L, 1L)))$same
#> [1] TRUE
```

Restricting to nothing is legal and gives a space with zero features,
which is what an empty
[`select_features()`](https://bbuchsbaum.github.io/fmridataset/reference/select_features.md)
returns.

``` r

n_features(restrict_space(brain, integer()))
#> [1] 0
```

A row of the frame can always be reconstructed into the native spatial
object. For a volume that is a
[`neuroim2::NeuroVol`](https://bbuchsbaum.github.io/neuroim2/reference/NeuroVol.html)
with `NA` outside the support.

``` r

map <- spatial_map(frame, "vol-01")
dim(map)
#> [1] 4 4 2
sum(!is.na(map))
#> [1] 8
```

The same works from a view, where the unselected voxels are also `NA`.

``` r

sum(!is.na(spatial_map(pair, "vol-01")))
#> [1] 2
```

## Same space, not merely same size

[`same_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
reports exact semantic identity: the two spaces must agree on class,
digest, and ordered feature IDs.
[`compatible_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
is an alias for the same check, kept for the 1.0 transition. Neither
word means a map exists.

An independently constructed space with the same grid, affine, support,
and template is the same space.

``` r

again <- volume_space(dim = grid, affine = diag(4), support = mask, template = "toy")
same_space(brain, again)$same
#> [1] TRUE
```

Change the affine and the voxels have the same IDs but sit somewhere
else in the world. The IDs still agree; the digest does not; the spaces
are different.

``` r

rescaled <- volume_space(
  dim = grid, affine = diag(c(2, 2, 2, 1)), support = mask, template = "toy"
)
report <- same_space(brain, rescaled)
report[c("same", "same_class", "same_digest", "same_feature_ids")]
#> $same
#> [1] FALSE
#> 
#> $same_class
#> [1] TRUE
#> 
#> $same_digest
#> [1] FALSE
#> 
#> $same_feature_ids
#> [1] TRUE
```

[`assert_compatible_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
turns that report into a structured error, and it is what
[`bind_observations()`](https://bbuchsbaum.github.io/fmridataset/reference/bind_observations.md)
and
[`map_features()`](https://bbuchsbaum.github.io/fmridataset/reference/map_features.md)
call before doing anything else.

``` r

assert_compatible_space(brain, rescaled)
#> Error:
#> ! Feature spaces differ in type, digest, or IDs.
```

## Parcels over a volume

A
[`parcel_space()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space.md)
is linked to a parent space through a membership matrix. It has one row
per parent feature and one column per parcel, holding non-negative
weights. Here the two slices of the masked volume become two parcels.
The atlas identity is part of every parcel’s feature ID.

``` r

slice <- feature_data(brain)$k
membership <- sparseMatrix(
  i = seq_len(n_features(brain)), j = slice, x = 1,
  dims = c(n_features(brain), 2L)
)
slabs <- parcel_space(
  parent = brain,
  parcel_ids = c("lower", "upper"),
  membership = membership,
  atlas = "slab",
  data = data.frame(id = c("lower", "upper"), label = c("Lower slab", "Upper slab"))
)
feature_ids(slabs)
#> [1] "slab:lower" "slab:upper"
parcel_aggregation(slabs)
#> 2 x 8 sparse Matrix of class "dgCMatrix"
#>                                             
#> [1,] 0.25 0.25 0.25 0.25 .    .    .    .   
#> [2,] .    .    .    .    0.25 0.25 0.25 0.25
```

The aggregation operator is derived from the membership: weighted means
by default, weighted sums on request.
[`parcel_space_from_atlas()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space_from_atlas.md)
builds the same object from a `neuroatlas` atlas aligned to a parent
space.

## Moving a frame between spaces

A parcel space owns the canonical map from its parent, so
[`map_features()`](https://bbuchsbaum.github.io/fmridataset/reference/map_features.md)
needs only the target. The result is a new frame in the parcel space.
Its assay is a *lazy* transformed source: nothing is aggregated until it
is read.

``` r

by_parcel <- map_features(frame, target = slabs)
by_parcel
#> <fmri_frame> 6 observations x 2 features
#>   assays: bold 
#>   active: bold 
#>   space: parcel_space 789147486788
collect_assay(by_parcel)
#>         [,1]    [,2]
#> [1,]  0.1075 -0.4575
#> [2,] -0.1675 -0.2825
#> [3,]  1.8600  0.4050
#> [4,] -1.1950  0.2150
#> [5,]  0.7025 -0.1700
#> [6,]  0.7750 -0.5900
```

Those are the per-slice means of `signal`, checked here against a direct
computation.

``` r

direct <- cbind(rowMeans(signal[, slice == 1]), rowMeans(signal[, slice == 2]))
all.equal(collect_assay(by_parcel), direct)
#> [1] TRUE
```

The transformation is recorded in the new frame’s provenance, and a
parcel frame reconstructs back into the parent volume by spreading each
parcel value over its member voxels.

``` r

by_parcel$provenance
#> <provenance_graph> 1 records
#>   tips: af035af7e5cacad5e1914b22ffac34438055155ecee144b462e5db798b26967e
range(spatial_map(by_parcel, "vol-01"), na.rm = TRUE)
#> [1] -0.4575  0.1075
```

Any other linear transformation is an explicit
[`feature_map()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_map.md):
a target-by-source operator plus both spaces. The operator’s dimensions
are checked against both. The source space must be the frame’s space
exactly.

``` r

pairs <- index_space(2L, ids = c("pair-a", "pair-b"))
operator <- sparseMatrix(
  i = c(1, 1, 2, 2), j = c(1, 2, 3, 4), x = 0.5,
  dims = c(2L, n_features(brain))
)
pair_means <- feature_map(from = brain, to = pairs, operator = operator, map_type = "pair_mean")
pair_means
#> <feature_map> 8 source features -> 2 target features
#>   type: pair_mean 
#>   digest: d487b5cd0a11
collect_assay(map_features(frame, map = pair_means))[1:2, ]
#>        [,1]   [,2]
#> [1,] -0.095  0.310
#> [2,] -0.030 -0.305
```

A map declared from a different space is refused even though the
operator would fit.

``` r

wrong <- feature_map(from = rescaled, to = pairs, operator = operator)
map_features(frame, map = wrong)
#> Error:
#> ! Feature spaces differ in type, digest, or IDs.
```

## Bases and synthesis-only bases

A
[`basis_space()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space.md)
is a representational axis over a parent space. It holds an encoder that
projects parent values onto components, a decoder that synthesizes
parent values from component scores, or both. When you hold only a
decoder,
[`basis_space_from_decoder()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space_from_decoder.md)
derives the exact least-squares encoder and validates that it is a left
inverse.

``` r

decoder <- matrix(0, n_features(brain), 3L)
decoder[1:4, 1L] <- 1
decoder[5:8, 2L] <- 1
decoder[, 3L] <- seq_len(n_features(brain)) / 8

components <- basis_space_from_decoder(
  parent = brain,
  component_ids = c("c1", "c2", "c3"),
  decoder = decoder,
  basis_type = "toy_basis"
)
basis_projection_info(components)[c("left_inverse_validated", "left_inverse_error")]
#> $left_inverse_validated
#> [1] TRUE
#> 
#> $left_inverse_error
#> [1] 5.551115e-16
dim(basis_analysis(components))
#> [1] 3 8
```

With an encoder available,
[`map_features()`](https://bbuchsbaum.github.io/fmridataset/reference/map_features.md)
projects the frame onto the components lazily, and
[`spatial_map()`](https://bbuchsbaum.github.io/fmridataset/reference/spatial_map.md)
synthesizes a volume back from scores.

``` r

scores <- map_features(frame, target = components)
round(collect_assay(scores)[1:2, ], 3)
#>        [,1]   [,2]   [,3]
#> [1,]  0.153 -0.340 -0.144
#> [2,] -1.640 -4.111  4.712
range(spatial_map(scores, "vol-01"), na.rm = TRUE)
#> [1] -0.4845  0.1345
```

Fitted dictionaries from ICA or dictionary learning are often
rank-deficient, so no exact left inverse exists. Ask for
`encoder = "none"` and you get a synthesis-only basis. It can
reconstruct parent values from scores but has no canonical map from the
parent. A frame of scores must therefore be constructed directly in that
space.

``` r

synthesis_only <- basis_space_from_decoder(
  parent = brain,
  component_ids = c("c1", "c2", "c3"),
  decoder = decoder,
  encoder = "none"
)
is.null(basis_analysis(synthesis_only))
#> [1] TRUE
basis_projection_info(synthesis_only)$left_inverse_validated
#> [1] FALSE

score_frame <- fmri_frame(
  assays = list(scores = matrix(round(rnorm(n_obs * 3), 2), n_obs, 3L)),
  observations = observations(frame),
  space = synthesis_only
)
range(spatial_map(score_frame, "vol-01"), na.rm = TRUE)
#> [1] -2.84000 -0.40375
```

Because the encoder is absent,
[`vectorize_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
on this basis raises an error that says so, and
`map_features(frame, target = synthesis_only)` has no operator to build
a map from.

## Composite spaces

A
[`composite_space()`](https://bbuchsbaum.github.io/fmridataset/reference/composite_space.md)
concatenates named child spaces into one feature axis, for example left
cortex, right cortex, and subcortical volume. Feature IDs are
part-qualified and the routing order is authoritative. Each child stays
the authority for its own identity and reconstruction.

``` r

mixed_space <- composite_space(
  list(cortex = slabs, latent = components),
  composite_type = "toy_composite"
)
feature_ids(mixed_space)
#> [1] "cortex::slab:lower" "cortex::slab:upper" "latent::c1"        
#> [4] "latent::c2"         "latent::c3"
feature_data(mixed_space)[, c(".feature_id", ".part", ".part_index")]
#> # A tibble: 5 × 3
#>   .feature_id        .part  .part_index
#>   <chr>              <chr>        <int>
#> 1 cortex::slab:lower cortex           1
#> 2 cortex::slab:upper cortex           2
#> 3 latent::c1         latent           1
#> 4 latent::c2         latent           2
#> 5 latent::c3         latent           3
```

A frame over a composite space is selected and reconstructed like any
other. Selecting one part yields a composite space that holds only that
part, and a spatial map returns one native object per part.

``` r

mixed <- fmri_frame(
  assays = list(mixed = cbind(collect_assay(by_parcel), collect_assay(scores))),
  observations = observations(frame),
  space = mixed_space
)
latent_only <- select_features(mixed, .part == "latent")
feature_ids(latent_only)
#> [1] "latent::c1" "latent::c2" "latent::c3"
composite_part_names(space(latent_only))
#> [1] "latent"

parts <- spatial_map(mixed, "vol-01")
names(parts$parts)
#> [1] "cortex" "latent"
```

## Where the package stops

`fmridataset` owns spatial identity, restriction, the linear map
descriptor, and lazy application of that map. It does not fit bases,
choose atlases, or run models. `neuroatlas` supplies parcellations.
`fmrilatent` supplies fitted bases through
[`basis_space_from_fmrilatent()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space_from_fmrilatent.md).
`neurosurf` supplies surface geometry for
[`surface_space_from_neurosurf()`](https://bbuchsbaum.github.io/fmridataset/reference/surface_space_from_neurosurf.md).
Each hands the package a serializable space; the package hands back
frames whose columns still mean what they meant.
