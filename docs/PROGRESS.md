# Cleanup progress — 13 November 2026

The cleaned package, asset-free workflow and real historical replay are implemented.
Real WordPhaseNet/router retraining and 100-test A/B/selector inference have executed
on CPU. Full from-scratch paper reproduction and organizer-server scores remain
unverified. Original historical assets remain in the owner's private workspace.

Completed:

- Audited all eight frozen scripts, imports, dependencies, checkpoints and contracts;
  added [Source Audit](SOURCE_AUDIT.md) and [Architecture](ARCHITECTURE.md).
- Preserved the eight frozen files and their archived SHA-256 hashes. Tests prevent
  reference imports from writing bytecode into that tree.
- Added validated motion/query/donor/candidate contracts, portable JSON/TOML config,
  timed-word preprocessing, optional BiGRU WordPhaseNet, retrieval, quaternion warp,
  shape-aware IK adapters, tapered hold, A/B construction, weighted/cross-fitted ridge
  ranking, robust whole-candidate selection and local challenge packaging.
- Added synthetic training/inference/smoke CLIs, input/source/model hash bindings,
  exclusive output creation, packaging exclusions and local CI configuration.
- Updated README, installation/reproduction instructions, provenance and publishing
  checklist. The owner subsequently approved MIT licensing and public source publication.
- Added a documentation index and explicit public/external repository layout.
- Added external-asset configuration, path-free availability and a read-only M1
  artifact auditor. A sanitized reference binds the known run without redistributing
  its motion files, per-sample tables, logs or weights.
- Located the remaining historical stages, data, licensed models and research
  interpreter. Added a safe `replay` command and real-data command templates.
- Retrained P1A over five real folds (60 epochs each), retrained P2A with all 2,525
  choices matching history, and regenerated/selected all 100 test clips with local
  official validator success. Detailed scope is in [Real Reproduction](REAL_REPRODUCTION.md).

Validation performed:

| Check | Actual result |
|---|---|
| Full pytest suite | **286 passed, 1 skipped** on Python 3.12.3 / NumPy 1.26.4 / pytest 7.4.4 |
| Source integrity audit | Eight files match archived SHA-256, sizes and syntax |
| Installed CLI synthetic smoke | Passed: eight training queries, three folds, two selected float32 outputs and local ZIP |
| Candidate output identity | Selected files equal complete A/B bytes; repeated inference arrays match |
| Frozen numerical comparisons | Timed-word features, warp/hold, router/ranker transforms, hypothesis policy, selector and quality scale tested on invented inputs |
| Installation/package | Editable install and wheel build passed without downloading dependencies |
| Learned timing / licensed models | Host optional test skipped; real historical CPU training and licensed FK/IK executed using external Python 3.9 / Torch 2.5.1 environment |
| External M1 archive | Reference-bound audit passed: all 7,692 manifest files checked and all 7,575 selected motion hashes/schemas passed; six append-only logs reported separately |
| Real inference replay | 100/100 final A/B decisions match; max final pose component difference `1.73e-6` radians; official local validators pass; server scores unverified |

Local synthetic verification artifacts are under ignored
`outputs/validation-smoke/`. Earlier fixture runs were preserved separately under
`outputs/initial-validation/`. Default example paths remain available for a fresh run.
The organization follow-up uses `outputs/organization-validation-current/` and records
the external audit in [M1 audit](validation/m1_external_audit_20261113.json) and
[Asset status](validation/external_asset_status_20261113.json). These snapshots contain
only aggregate status/counts and no private paths, labels or model parameters.
See [LOCAL_VALIDATION.json](LOCAL_VALIDATION.json) for the machine-readable record.

Commands to run from this checkout with the existing Python dependencies:

```bash
PYTHONPATH=src python3 -m repointer audit-source
python3 -m pytest -q
PYTHONPATH=src python3 -m repointer smoke --config configs/synthetic.json
```

After setting `REPOINTER_M1_FULL_DIR` in your shell to the original accepted run:

```bash
PYTHONPATH=src python3 -m repointer assets-status --config configs/external_assets.example.json
PYTHONPATH=src python3 -m repointer audit-m1 --config configs/external_assets.example.json \
  --reference protocol/m1/external_run_reference.json --verify-hashes --verify-motions
```

The external audit can take several minutes and never executes generation or model
loading. See [External Assets](EXTERNAL_ASSETS.md) for exact check scopes and the
remaining reconstruction requirements. The [Documentation Index](README.md)
and [Repository Layout](REPOSITORY_LAYOUT.md) explain where files belong.

For installation, individual training/inference steps, optional WordPhaseNet and
fresh-run configuration, follow [Getting Started](GETTING_STARTED.md). Reusing an
existing output directory fails intentionally; choose new config paths.

Remaining work:

- Audit the final manuscript and organizer result evidence before claiming full
  paper/official-score reproduction. All previously unlocated stage archives and
  the P1_FULL/P2C0 launchers are now found externally.
- Rebuild all M1/P1_FULL/MH motion-training artifacts for a full from-scratch claim;
  final inference replay currently reuses verified historical training evidence.
- Independently validate cleaned native backends and checkpoint migration. The real
  replay uses preserved historical implementations, with CPU versus CUDA differences
  measured explicitly. Newly retrained timing weights are a separate experiment.
- Run official validation/evaluation before claiming paper-score reproduction.
  MIT licensing and public source publication are owner-approved. Third-party asset
  redistribution still needs separate permission. The supplied `.git` remains
  read-only; an independent local commit and Git bundle preserve the source snapshot.
  Remote repository creation/upload and CI verification are still pending.
