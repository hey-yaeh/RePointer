# Documentation guide

| Task | Start here |
|---|---|
| Install and run the asset-free example | [Getting Started](GETTING_STARTED.md) |
| Repeat the real pretrained inference chain or timing retraining | [Real Reproduction](REAL_REPRODUCTION.md), [Measured Results](validation/real_reproduction_20261113.json) |
| Connect an existing private M1 archive | [External Assets](EXTERNAL_ASSETS.md) |
| Understand what belongs in the public repository | [Repository Layout](REPOSITORY_LAYOUT.md) |
| Assess paper-reproduction readiness | [Reproducibility Status](REPRODUCIBILITY_STATUS.md) |
| Review what changed and what was tested | [Progress](PROGRESS.md), [Local Validation](LOCAL_VALIDATION.json) |
| Understand module contracts and stage lineage | [Architecture](ARCHITECTURE.md), [Source Stage Map](SOURCE_STAGE_MAP.md) |
| Inspect dependencies, checkpoint contracts and missing evidence | [Source Audit](SOURCE_AUDIT.md) |
| Review the external archive checks actually performed | [M1 Audit Snapshot](validation/m1_external_audit_20261113.json), [Asset Status Snapshot](validation/external_asset_status_20261113.json) |
| Verify historical source provenance | [Provenance](PROVENANCE.md), [Source Inventory](SOURCE_INVENTORY.json), [Source Hashes](SOURCE_HASHES.json) |
| Work on the code or prepare a release | [Contributing](CONTRIBUTING.md), [Data and Licensing](DATA_AND_LICENSE.md), [Publish Checklist](PUBLISH_CHECKLIST.md) |

Status terms have distinct meanings:

- **Bundled:** included in the source repository. This does not imply that a full
  experiment has been executed.
- **Located externally:** a private artifact directory has been identified. It is
  configured by the user, not redistributed with source.
- **Integrity-checked:** the stated file, hash, schema and coverage checks passed.
  Check each report's scope; motion arrays are optional in the default M1 audit.
- **Reproduced:** a new controlled experiment has been run and validated against
  the intended evidence. Official-score reproduction remains unverified.

The historical M1 run is located externally and integrity-checked. Synthetic CLI
validation and real historical-script replay are distinct. The 100-test real replay
is now locally validated; full from-scratch training and organizer-server scores
remain unverified. Read the real experiment report before interpreting a PASS.
