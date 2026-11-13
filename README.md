# RePointer

**Retrieval-Anchored Candidate Generation and Detector-Robust Selection for Referential Gestures**  
Yuan He and Rachel McDonnell · HSI @ ECCV 2026

RePointer adapts a natural full-body motion donor to a timed transcript and pointing
target. This repository contains the immutable experimental source, a cleaned Python
package, an executable synthetic workflow, and an explicit real-data historical replay tool.

**Real 100-test inference has been rerun with licensed SMPL-X and passed the official
local validator. Organizer-server scores and full from-scratch paper reproduction
remain unverified.** Required historical assets were located externally and remain
private. See [Real Reproduction](docs/REAL_REPRODUCTION.md) for measured CPU/CUDA
differences, stage retraining evidence and exact scope. The synthetic workflow remains
available without private data or model assets.

```text
P0C0 frozen shared generation protocol
  Timed words + hand + target geometry + body shape
    → P1A WordPhaseNet: bidirectional GRU, 64-bin apex-time distribution
    → M1: natural same-hand retrieval → apex-preserving warp → shape-aware IK → hold
    → P2A Candidate A: learned/domain-hand median timing + phase router
    → P2A-MH Candidate B: four timing hypotheses × multiple donors + cross-fitted ranker
    → P2C1: complete candidate materialization
    → P2C2: detector-robust selection of complete A or B unchanged
```

WordPhaseNet uses timed transcript features, **not raw audio**. The final method uses
retrieval and geometric adaptation; the earlier learned residual/diffusion generator
is excluded.

## Run the asset-free workflow

Python 3.10 or newer; NumPy is the only core runtime dependency.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m repointer audit-source
python -m pytest -q
python -m repointer smoke --config configs/synthetic.json
```

The smoke command creates eight invented training queries, eight donor clips and two
inference queries. It fits a NumPy timing prior and ridge router/ranker, builds A/B,
selects complete motions, and validates a local ZIP. Outputs are under ignored
`outputs/`; existing directories are never overwritten. For another run, copy the
configuration and choose three fresh paths.

The fixture configuration reduces retrieval/donor/IK budgets for speed. It uses a
synthetic skeleton and motion-only heuristics, not licensed SMPL-X or frozen detector
weights. An optional PyTorch WordPhaseNet training path is documented in
[Getting Started](docs/GETTING_STARTED.md); its local execution depends on PyTorch.

## Use an external historical run

Keep the M1 archive in its original private directory. Set `REPOINTER_M1_FULL_DIR`
in your shell to the accepted **run directory**, containing `aggregate/` and `folds/`.
The archive's parent directory and its `runs/` directory are not the run itself.

```bash
python -m repointer assets-status --config configs/external_assets.example.json
python -m repointer audit-m1 --config configs/external_assets.example.json \
  --reference protocol/m1/external_run_reference.json
```

The audit reads artifacts and reports aggregate status without copying files, loading
weights or executing archived code. Add `--verify-hashes --verify-motions` to check
every selected motion. This audits historical evidence; it does not rerun generation.
See [External Assets](docs/EXTERNAL_ASSETS.md) for configuration and reconstruction gaps.

`python -m repointer replay` runs the real historical stages using an explicitly
configured external research interpreter and a fresh private output root. It supports
the pretrained inference chain and separate WordPhaseNet retraining. Setup and commands
are in [Real Reproduction](docs/REAL_REPRODUCTION.md).

## Repository guide

| Location | Purpose |
|---|---|
| `frozen/` | Eight immutable historical stage scripts, checked against archived hashes |
| `src/repointer/` | Validated contracts, timing, retrieval, warp, IK, candidates, ranking, selector and challenge utilities |
| `configs/` | Separate synthetic-run and external-asset configuration templates |
| `protocol/` | P0C0 contracts and a sanitized M1 reference with no data or weights |
| `tests/` | Stage tests, frozen-code comparisons and asset-free CLI integration |
| `docs/` | Audit, architecture, provenance, reproduction gaps and progress report |

Start with [Getting Started](docs/GETTING_STARTED.md), [Source Audit](docs/SOURCE_AUDIT.md),
[Architecture](docs/ARCHITECTURE.md), [Reproducibility Status](docs/REPRODUCIBILITY_STATUS.md)
and [Progress](docs/PROGRESS.md). The complete historical stage map is in
[Source Stage Map](docs/SOURCE_STAGE_MAP.md). [Documentation Index](docs/README.md)
maps the files and distinguishes public source from external research artifacts.

## Reported results and evidence limits

The incoming prerelease documentation reports these organizer scores for the revised
100-clip submission. Organizer reports and final submission files are not bundled;
these numbers have **not been reproduced or independently verified here**.

| Metric | MM-Conv Flow | RePointer |
|---|---:|---:|
| Temporal alignment (Score 1a) | 0.6070 | 0.8637 |
| Spatial pointing (Score 1b) | 0.6905 | 0.9049 |
| Combined grounding (STG) | 0.4411 | 0.7892 |

These are reported whole-system scores, not controlled ablations or proof of motion
naturalness. Local spatial/temporal proxies are separate from the organizer scorer.

## Assets and publication

No challenge data, derived labels, pretrained checkpoints, private assets or SMPL-X
models are included. Keep learned weights outside the public source tree and acquire
restricted assets through their owners. See [Data and Licensing](docs/DATA_AND_LICENSE.md).
RePointer-owned source code is available under the [MIT License](LICENSE).
Third-party dependencies, HSI data, model checkpoints and SMPL-X assets retain
their own licenses and permissions; the code license does not grant rights to them.
