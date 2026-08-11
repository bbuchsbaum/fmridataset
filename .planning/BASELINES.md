# Implementation baselines

Captured: 2026-08-11

These results qualify the starting state; they are not release certification.
Companion repositories contain user changes and are read-only until their
dedicated integration tickets begin in isolated worktrees.

| Package | Revision | Test result | Qualification |
|---|---|---|---|
| `fmridataset` | `0d3a0a595e77a5d796691d3b5216a05ab0b8e3d2` | Pass; 70 skips, 5 warnings | DelayedArray, real HDF5, study-memory, and Zarr paths remain uncertified |
| `delarr` | `0bc42504d46a936ee41245e938e82504b6ffa8cc` | Pass | Checkout has extensive unrelated tracked and untracked changes |
| `fmristore` | `0b834048f20f6a9cfc3c0bc36fa171ca71558359` | Four errors, two skips | Latent concat/NeuroVecSeq compatibility and HDF5 chunk tests must be resolved before certified writer work |
| `multidesign` | `a0b4fbfdc67722f87a57647cae17939ea948bb7b` | Pass with one warning | Checkout has untracked local artifacts |
| `fmrigds` | `6c8062b118f3bc7d5f27b993fe2df0972fecf340` | Pass | Tested on the `agent/group-examination` branch with an untracked issue note |
| `neuroim2` | `a474e6dab261aae93095dd0534aa7dd0840e2367` | One snapshot failure and two parallel-searchlight errors | Snapshot write is sandbox-limited; missing worker-visible helpers require resolution before native parallel certification |

## Isolation policy

- New semantic work begins in `fmridataset` without editing companion trees.
- A companion package is edited only after its dedicated Beads ticket is ready,
  a clean isolated worktree is created from the recorded revision, and the
  intended paths are reserved.
- Current companion failures are not reclassified as product successes. They
  remain explicit gates for FDS-031, FDS-038, FDS-067, and FDS-071.
- PR tests may use small and virtual sources, but persistent-store and
  full-scale claims require their separately listed gates.
