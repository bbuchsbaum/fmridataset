# Legacy migration fixtures

These immutable fixtures exercise the supported 0.x-to-frame migration
boundary. They are not golden masters for the retired dataset/backend API.

- `matrix_dataset.rds` is a self-contained legacy matrix dataset.
- `fmri_series.rds` is the historical serialized series envelope.
- `sampling_frame.rds` verifies that temporal metadata without an assay is
  rejected explicitly.
- `mock_neurvec.rds` verifies the explicit `NeuroVec` migration path.
- `frame-migration-contracts.md` records the required semantic outcome.

The fixtures are consumed by `test-frame-coercion.R`. Do not regenerate them
with the 1.0 code: their value is that they remain byte-for-byte examples of
the final 0.x serialization formats. Any replacement requires an explicit
migration-contract review.
