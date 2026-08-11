# Legacy-to-frame migration contracts

The existing RDS fixtures remain byte-stable inputs. Migration tests must map
them as follows without changing the fixture files.

| Fixture | Observation axis | Feature axis | Assay | Required preservation |
|---|---|---|---|---|
| `matrix_dataset.rds` | Existing rows/time points | Existing matrix columns in a namespaced `index_space` | `signal` | Numeric values and row/column order |
| `fmri_series.rds` | `temporal_info` with generated `.obs_id` | `voxel_info` with feature-space IDs | `signal` | Temporal/voxel metadata and selection provenance |
| `sampling_frame.rds` | Derived observation metadata | Not an assay-bearing object | None | TR, run boundaries, events, censoring |
| `mock_neurvec.rds` | Acquired volumes | Packed voxel support in `volume_space` | `signal` | NeuroSpace geometry, support, and reconstruction |

All generated IDs are persisted in the migrated object. Re-running migration
on an already migrated frame is an identity operation. Migration never infers
equal space from equal dimensions alone.
