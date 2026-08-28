# ADR-005: A basis space may be analysis-only, synthesis-only, or both

Status: accepted
Date: 2026-08-27

## Context

`basis_space()` required an *encoder* — the analysis operator taking parent
features to components — and treated the *decoder* as optional. A basis could
therefore only exist if parent data could be projected into its components.

That excluded the case the latent path actually serves. Spatial loadings from
PCA, ICA, or dictionary learning are a **synthesis** dictionary: they take
component scores back to voxels. `basis_space_from_decoder()` recovered an
encoder by exact least squares, but only for a full-column-rank dictionary, and
ICA and dictionary-learning fits are routinely rank-deficient or non-orthogonal.
For those, no exact left inverse exists, and the constructor refused a basis
that was perfectly well-defined in the direction anyone needed it.

This was a real narrowing relative to `latent_dataset`, which stores loadings
and offers `get_spatial_loadings()` and `reconstruct_voxels()` — synthesis only.
It never projected new voxel data into the latent space.

Separately, `fmrireg` reaches into `dataset$lvec %||% dataset$backend$data[[1]]`
and reads an S4 `@loadings` slot in four places because neither model exported an
accessor for loadings.

## Decision

A `basis_space` carries an analysis operator, a synthesis operator, or both, and
is refused only when it has neither.

- `basis_analysis()` returns the encoder or `NULL`.
- `basis_synthesis()` returns the decoder or `NULL`. This is the loadings
  accessor that was missing from both models.
- `vectorize_space()` requires an encoder; `reconstruct_space()` requires a
  decoder. Each refuses with a message naming which direction is unavailable and
  which one still works.
- `projection$left_inverse_validated` is `TRUE` only when both operators are
  present and the left-inverse check actually ran.

`basis_space_from_decoder(..., encoder = c("least_squares", "none"))` makes the
choice explicit. The default is unchanged: derive the exact least-squares
encoder, and refuse a rank-deficient dictionary. The refusal now names
`encoder = "none"` as the route to a synthesis-only basis.

Restriction preserves directionality. A two-way basis recomputes its encoder
from the restricted decoder, because narrowing the component axis changes the
least-squares solution; a synthesis-only basis stays synthesis-only rather than
having an encoder manufactured for it.

## Consequences

The rank restriction is no longer a silent exclusion. It applies only to the
exact-inverse path, where it is a genuine mathematical requirement, and callers
who do not need projection have a documented way through.

`basis_synthesis()` gives the 0.11 shim a real accessor for
`get_spatial_loadings()`, and gives `fmrireg` something to migrate its four
hand-rolled `@loadings` fallbacks onto.

Pseudo-inverses were deliberately not adopted. A Moore-Penrose encoder would
satisfy the shape requirement while failing the left-inverse property the
structure claims to validate, which would make `left_inverse_validated` a lie
rather than a fact. A basis that cannot project says so.
