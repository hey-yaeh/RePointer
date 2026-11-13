# Provenance and modifications

The frozen source is preserved with its original experiment-stage filenames.
The M1 worker is copied from the audited source snapshot, which removes **three
workstation-specific model search paths**. That single portability adjustment means
the public M1 script is not byte-identical to the worker in the historic run.
Original and adapted checksums can be found in `docs/SOURCE_HASHES.json`.

The November 2026 cleanup leaves all eight public frozen files byte-identical to
`SOURCE_INVENTORY.json`. New implementations live only under `src/repointer/`;
they do not replace historical source bindings. Synthetic fixtures are invented
from NumPy and small word templates, not derived from HSI. Selected function-level
comparisons are documented in `SOURCE_AUDIT.md`; full numerical parity is unverified.
Private companion presence recorded below describes incoming archive provenance,
not assets inspected in this cleanup environment.

On 13 November 2026 the owner provided the external M1 archive location. Its run
`20260810T200141Z` was audited read-only: the copied worker matches the original
`d5be82...` digest, and the copied aggregator matches the public frozen digest.
Scientific artifact hashes and all 7,575 selected-motion schemas passed. The external
run remains private; only a sanitized reference is added under `protocol/m1/`.
This new observation does not revalidate other private companion assets.

A later follow-up located the remaining final-stage archives and original research
environment. Real CPU training and inference replay produced new private outputs,
while preserving original directories and all eight frozen source hashes. External
acceptance/manifest hashes are summarized under `protocol/final/`; measured stage
results are in `docs/validation/real_reproduction_20261113.json`. This evidence is
separate from organizer-server score verification and full from-scratch training.

P0C0 artifacts received from the archive `p0c0_shared_generation_protocol.zip`
(run `20260810T184546Z`). They include:

- 2,525 labeled samples from five group-disjoint folds.
- Private apex detector weights v1.0 and v1.1.
- Frozen comparison and motion schema contracts.
- P0C0 acceptance (`P0C0_SHARED_PROTOCOL_FROZEN`).

Original `shared_generation_manifest.csv` includes released labels and absolute workstation
source paths. It is intentionally not present in this public archive.
A public reference copy of the spatial and temporal proxy contracts removes local
provenance path fields while preserving scientific definitions.
