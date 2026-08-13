# ADR-005: Durable axis ID policy

## Status

Accepted for the 1.0 development line.

## Decision

Every observation, feature, component, and entity has a stable character ID.
Construction uses one of three explicit policies:

- `require`: the caller supplies IDs. This is the default.
- `deterministic`: IDs are SHA-256 values derived from a declared namespace,
  axis role, ordered key names, and typed key values.
- `ephemeral`: UUID-backed IDs are allowed for exploratory in-memory work but
  are visibly prefixed with `ephemeral-` and cannot enter an FDS manifest or a
  certified semantic identity.

Subsetting preserves the source policy descriptor and existing IDs; it never
regenerates them. Binding retains a shared descriptor when all inputs agree and
otherwise records the result as supplied durable IDs. Entity keys are always
caller-supplied and therefore use the `require` policy.

For BIDS imports, the importer must declare the relevant entity columns and a
within-run index. For example:

```r
axis_frame(
  scans,
  axis = "observation",
  id_policy = "deterministic",
  id_namespace = "bids:ds000001:bold-volume:v1",
  id_keys = c("subject", "session", "task", "run", "volume_index")
)
```

Missing entity values must first be normalized by the importer to an explicit
BIDS sentinel such as `"none"`; missing key values are rejected. Repeated key
tuples are collisions and fail construction.

## Feature-ID representation

Version 1 retains ordinary character vectors as the public and persisted ID
representation. The benchmark in
`inst/benchmarks/benchmark-id-representation.R` compares this with compact
integer positions at 50,000, 100,000, and 1,000,000 features. On the reference
arm64 R 4.5.1 run, one million formatted IDs occupied approximately 88 MB
(about 88 bytes per feature), while integer positions occupied approximately
4 MB. This overhead is material but bounded and buys direct, portable,
self-describing lookup. Assay values remain the dominant storage cost, and
volume/surface spaces may generate structured IDs from compact support indices
instead of storing repeated strings internally.

A future schema may add dictionary encoding as a physical codec optimization.
It must decode to exactly the same ordered character IDs and cannot alter
semantic or spatial identity.

## Consequences

- Persisted objects cannot silently contain random IDs.
- Equal dimensions remain insufficient evidence of identity.
- Importers own normalization of external entity keys.
- Ephemeral frames can be filtered and analyzed locally, but must be assigned a
  durable axis and feature space before persistence or certification.
