# Validate and compare canonical frame schemas

Validate and compare canonical frame schemas

## Usage

``` r
validate_frame_schema(schema)

compare_frame_schema(x, reference, mode = c("same", "collection", "bind"))

validate_against_schema(x, reference, mode = c("same", "collection", "bind"))

frame_schema_digest(x)
```

## Arguments

- schema:

  A canonical frame schema.

- x:

  A frame, view, or schema to validate.

- reference:

  A frame, view, or schema used as the reference contract.

- mode:

  Comparison mode. `same` compares the complete schema; `collection`
  permits different observation counts and feature identities; `bind`
  permits different observation counts but requires one feature
  identity.

## Value

Validators return their input invisibly. `compare_frame_schema()`
returns a structured `frame_schema_compatibility` report.
`frame_schema_digest()` returns a single hex digest string.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
schema <- frame_schema(frame)
validate_frame_schema(schema)
compare_frame_schema(frame, frame)$compatible
#> [1] TRUE
frame_schema_digest(frame)
#> [1] "e01aae5cbc76cbb9b0a7ac43a4977a7188b69f355103849f0c1556eaa0761def"
```
