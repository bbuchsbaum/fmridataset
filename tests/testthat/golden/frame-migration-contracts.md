# Legacy-to-frame migration contracts

The existing RDS fixtures remain byte-stable inputs. Migration tests must map
them as follows without changing the fixture files.

| Fixture | Observation axis | Feature axis | Assay | Required preservation |
|---|---|---|---|---|
| `matrix_dataset.rds` | Existing rows/time points | Existing matrix columns in a namespaced `index_space` | `signal` | Numeric values and row/column order |
| `fmri_series.rds` | Generated IDs for the serialized envelope rows | Generated feature IDs for the serialized envelope columns | `signal` | Numeric values, dimensions, and serialized class provenance |
| `sampling_frame.rds` | Not applicable | Not an assay-bearing object | None | Reject with an explicit instruction to attach assay data |
| `mock_neurvec.rds` | Acquired volumes with explicit TR supplied at migration | Packed voxel support in `volume_space` | `signal` | Stored geometry, support, order, and reconstruction |

All generated IDs are deterministic and persist in the migrated object.
Re-running migration on an already migrated frame is an identity operation.
Matrix and series namespaces incorporate source identity, so migration never
infers equal space from equal dimensions alone. File/backend datasets require a
backend-specific source and feature-space adapter and are not realized by the
generic migration path.
